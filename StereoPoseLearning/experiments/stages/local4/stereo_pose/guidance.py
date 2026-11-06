"""Predicted-mask/landmark feature routing; no ground truth enters this module."""
import torch
from torch import nn
import torch.nn.functional as F


def landmark_locations(logits, power=4.0):
    """Subpixel expectation of sharpened sigmoid maps, in align_corners=False coords.

    Sigmoid maps are supervised Gaussian maps, not classification logits. Raising
    probabilities to a power suppresses diffuse low-probability background.
    An empty map falls back to a uniform distribution, never a NaN.
    """
    p = logits.float().sigmoid()
    h, w = p.shape[-2:]
    yy, xx = torch.meshgrid((torch.arange(h, device=p.device)+.5)*2/h-1,
                            (torch.arange(w, device=p.device)+.5)*2/w-1, indexing='ij')
    weights = p.pow(power) + 1e-12
    weights = weights / weights.sum((-2, -1), keepdim=True)
    centres = torch.stack(((weights*xx).sum((-2, -1)), (weights*yy).sum((-2, -1))), -1)
    return centres, p.amax((-2, -1))


class GuidedRegions(nn.Module):
    """One foreground descriptor + five ordered C/H1/H2/J1/J2 patch descriptors."""
    def __init__(self, width, config):
        super().__init__()
        self.patch_size = int(config.get('roi_patch_size', 8))
        self.pool_size = int(config.get('roi_pool_size', 2))
        if not 1 <= self.pool_size <= self.patch_size:
            raise ValueError('ROI pool size must be between 1 and the sampled patch size')
        self.radius_px = float(config.get('roi_radius_px', 24.0))
        self.power = float(config.get('landmark_power', 4.0))
        self.foreground_floor = float(config.get('foreground_context_floor', .01))
        self.patch_project = nn.Sequential(nn.Linear(self.pool_size**2*width, width), nn.LayerNorm(width), nn.GELU())
        self.metadata = nn.Sequential(nn.Linear(5, width), nn.GELU(), nn.Linear(width, width))
        self.region_embedding = nn.Parameter(torch.randn(6, width)*.01)
        # Runs separately on each predicted local patch, without mixing cameras,
        # timestamps or keypoints. Legacy checkpoints keep the identity path.
        self.local_encoder = None
        if config.get('roi_local_encoder', False):
            self.local_encoder = nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1, groups=width),
                nn.GroupNorm(8, width), nn.GELU(), nn.Conv2d(width, width, 1))

    def forward(self, spatial, mask_logits, heat_logits, image_size):
        n, channels, h, w = spatial.shape
        ih, iw = image_size
        xy, confidence = landmark_locations(heat_logits, self.power)
        # Fixed physical image-pixel neighbourhoods preserve RGB texture around
        # amodal points, including points under tissue. No visibility filtering.
        radius = xy.new_tensor([2*self.radius_px/iw, 2*self.radius_px/ih])
        t = (torch.arange(self.patch_size, device=xy.device)+.5)*2/self.patch_size-1
        gy, gx = torch.meshgrid(t, t, indexing='ij')
        offsets = torch.stack((gx, gy), -1)*radius
        grid = xy[:, :, None, None, :] + offsets[None, None]
        # Sample the same 8x8 patch in both variants; only local processing and
        # retained pooling resolution change, not the radius or camera model.
        patches = F.grid_sample(spatial.float(), grid.reshape(n, 5*self.patch_size, self.patch_size, 2),
                                mode='bilinear', padding_mode='border', align_corners=False)
        patches = patches.reshape(n, channels, 5, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).reshape(n*5, channels, self.patch_size, self.patch_size)
        if self.local_encoder is not None:
            patches = patches + self.local_encoder(patches.to(spatial.dtype)).float()
        patches = F.adaptive_avg_pool2d(patches, (self.pool_size, self.pool_size))
        patches = patches.flatten(1).reshape(n, 5, channels*self.pool_size**2)
        local = self.patch_project(patches.to(spatial.dtype))
        mask = mask_logits.float().sigmoid()
        weights = mask.square() + self.foreground_floor
        weights = weights / weights.sum((-2, -1), keepdim=True)
        foreground = (spatial.float()*weights).sum((-2, -1))
        yy, xx = torch.meshgrid((torch.arange(h, device=xy.device)+.5)*2/h-1,
                                (torch.arange(w, device=xy.device)+.5)*2/w-1, indexing='ij')
        centre = torch.stack(((weights[:, 0]*xx).sum((-2,-1)),
                              (weights[:, 0]*yy).sum((-2,-1))), -1)
        extent = torch.stack((((xx-centre[:,0,None,None]).square()*weights[:,0]).sum((-2,-1)).sqrt(),
                              ((yy-centre[:,1,None,None]).square()*weights[:,0]).sum((-2,-1)).sqrt()), -1)
        whole_meta = torch.cat((centre, extent, mask.mean((-2,-1))), -1)
        local_meta = torch.cat((xy, radius.expand(n,5,2), confidence[...,None]), -1)
        meta = torch.cat((whole_meta[:,None], local_meta), 1)
        tokens = torch.cat((foreground[:,None].to(local.dtype), local), 1)
        tokens = tokens + self.metadata(meta.to(tokens.dtype)) + self.region_embedding[None]
        # Exact inverse of normalized coordinates for image pixel centres.
        uv = (xy+1)*xy.new_tensor([iw, ih])/2-.5
        return tokens, uv, confidence
