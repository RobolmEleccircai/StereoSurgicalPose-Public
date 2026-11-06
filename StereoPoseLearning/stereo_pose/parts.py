"""Foreground-only raw-image wrist and paired-jaw recognition.

The local encoders never receive FPN features or full-image tokens. Predicted
locations/masks are detached; only segmentation/landmark supervision can train
those routing decisions. Missing visible foreground remains an observation with
low confidence, never a reason to discard a frame or fabricate visible pixels.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


def part_regions(uv, image_size, config):
    """Return wrist/jaw boxes in network-image pixel-centre coordinates.

    H1/H2 nearly coincide in the fixed CAD; use their mean as the common hinge.
    Wrist includes distal shaft C and both jaws. Jaw includes both blades and H.
    Adaptive boxes are square in original image pixels; padding may leave the
    image and is explicitly neutral rather than repeating border image content.
    """
    points = uv.detach().float()
    hinge = points[:, 1:3].mean(1, keepdim=True)
    point_sets = (torch.cat((points[:, :1], hinge, points[:, 3:5]), 1),
                  torch.cat((hinge, points[:, 3:5]), 1))
    scale = min(image_size) / 360.0
    centres, extents = [], []
    for name, selected in zip(('wrist', 'jaw'), point_sets):
        lo, hi = selected.amin(1), selected.amax(1)
        centre = (lo + hi) * .5
        margin = float(config.get(name+'_roi_margin_px', 10.0)) * scale
        radius = ((hi-lo)*.5).amax(-1, keepdim=True) + margin
        minimum = float(config.get(name+'_roi_min_half_px', 24.0 if name == 'wrist' else 20.0)) * scale
        maximum = float(config.get(name+'_roi_max_half_px', 128.0 if name == 'wrist' else 96.0)) * scale
        if not 0 < minimum <= maximum:
            raise ValueError('Invalid '+name+' ROI extent bounds')
        centres.append(centre)
        extents.append(radius.clamp(minimum, maximum).expand(-1, 2))
    return torch.stack(centres, 1), torch.stack(extents, 1)


def jitter_part_regions(centres, half_extents, image_size, batch, time,
                        centre_px=0.0, scale_fraction=0.0, training=False):
    """Train-only, time-consistent tolerance to predicted ROI placement errors.

    Uniform centre offsets are bounded by ``centre_px`` at a 360-pixel input
    height. Each window, eye and part receives its own offset and isotropic
    scale, reused at every timestamp. Geometry labels and source RGB stay
    unchanged; the returned transforms also drive local heatmap supervision.
    No labels select a crop, and disabled/evaluation calls consume no RNG.
    """
    if (not math.isfinite(centre_px) or centre_px < 0
            or not math.isfinite(scale_fraction) or not 0 <= scale_fraction < 1):
        raise ValueError('ROI jitter requires a nonnegative centre bound and scale fraction in [0, 1)')
    if not training or (centre_px == 0 and scale_fraction == 0):
        return centres, half_extents
    if (centres.shape != half_extents.shape or centres.ndim != 3
            or centres.shape != (batch*time*2, 2, 2) or batch < 1 or time < 1):
        raise ValueError('ROI jitter requires flattened [batch,time,eye,part,xy] boxes')
    centres = centres.detach().float().reshape(batch, time, 2, 2, 2)
    half_extents = half_extents.detach().float().reshape(batch, time, 2, 2, 2)
    # Shared across time, independent between eyes/parts; never introduce a
    # synthetic per-frame jump into a causal temporal training window.
    shape = (batch, 1, 2, 2)
    offset = torch.zeros((*shape, 2), device=centres.device, dtype=centres.dtype)
    scale = torch.ones((*shape, 1), device=centres.device, dtype=centres.dtype)
    if centre_px:
        offset.uniform_(-centre_px, centre_px).mul_(min(image_size)/360.0)
    if scale_fraction:
        scale.uniform_(1-scale_fraction, 1+scale_fraction)
    return (centres+offset).reshape(-1, 2, 2), (half_extents*scale).reshape(-1, 2, 2)


def foreground_crops(rgb, mask_probability, centres, half_extents, image_size,
                     crop_size=160, cutoff=.2, dilation_px=1, chunk_size=8):
    """Mask native-resolution RGB BEFORE interpolation and local encoding.

    rgb: uint8 [N,3,Hraw,Wraw]; mask_probability: [N,1,h,w]. All boxes are
    [N,P,2] in image_size=(Hnetwork,Wnetwork) pixel-centre coordinates.
    Returned [N,P,4,crop_size,crop_size] contains normalized foreground RGB plus
    its visible mask. Background RGB is exactly zero after normalization.
    The detached routing contract prevents pose gradients from making the mask
    encode background. Chunking avoids allocating a full float 720p video batch.
    """
    if rgb.dtype != torch.uint8 or rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError('Local recognition requires native RGB uint8 [N,3,H,W]')
    if centres.ndim != 3 or centres.shape != half_extents.shape or centres.shape[0] != rgb.shape[0] or centres.shape[-1] != 2:
        raise ValueError('Invalid local ROI coordinates')
    if mask_probability.shape[0] != rgb.shape[0] or mask_probability.shape[1] != 1:
        raise ValueError('Predicted visible masks must match the RGB batch')
    if not 0 <= cutoff < 1 or dilation_px not in (0, 1, 2) or crop_size < 16 or chunk_size < 1:
        raise ValueError('Invalid foreground crop settings')
    centres, half_extents = centres.detach().float(), half_extents.detach().float()
    n, parts = centres.shape[:2]
    ih, iw = image_size
    axis = (torch.arange(crop_size, device=rgb.device, dtype=torch.float32)+.5)*2/crop_size-1
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    offsets = torch.stack((xx, yy), -1)
    pixel_grid = centres[:, :, None, None] + half_extents[:, :, None, None] * offsets
    grid = (pixel_grid+.5)*pixel_grid.new_tensor([2/iw, 2/ih])-1
    mean = grid.new_tensor([.485,.456,.406])[None,:,None,None]
    std = grid.new_tensor([.229,.224,.225])[None,:,None,None]
    outputs = []
    for start in range(0, n, chunk_size):
        end = min(start+chunk_size, n)
        mask = F.interpolate(mask_probability[start:end].detach().float(),
                             size=rgb.shape[-2:], mode='bilinear', align_corners=False)
        # Below-cutoff probabilities have NO appearance path into the branch.
        mask = ((mask-cutoff)/(1-cutoff)).clamp(0, 1)
        if dilation_px:
            mask = F.max_pool2d(mask, 2*dilation_px+1, stride=1, padding=dilation_px)
        image = (rgb[start:end].float()/255-mean)/std
        masked = torch.cat((image*mask, mask), 1)
        crop = F.grid_sample(masked, grid[start:end].reshape(end-start, parts*crop_size, crop_size, 2),
                             mode='bilinear', padding_mode='zeros', align_corners=False)
        crop = crop.reshape(end-start,4,parts,crop_size,crop_size).permute(0,2,1,3,4)
        outputs.append(crop)
    return torch.cat(outputs, 0)


class LocalPartEncoder(nn.Module):
    """Retain spatial detail with a small RGB+mask encoder, not an upsampled FPN."""
    def __init__(self, width):
        super().__init__()
        def block(cin, cout, stride):
            return nn.Sequential(nn.Conv2d(cin,cout,3,stride=stride,padding=1,bias=False),
                                 nn.GroupNorm(8,cout), nn.GELU())
        self.stem = nn.Sequential(block(4,32,2), block(32,48,2), block(48,64,1))
        self.refine = nn.Sequential(nn.Conv2d(64,64,3,padding=1,groups=64,bias=False),
                                    nn.GroupNorm(8,64),nn.GELU(),nn.Conv2d(64,64,1))
        self.heatmap = nn.Conv2d(64,3,1)
        self.project = nn.Sequential(nn.Linear(64*4*4,width),nn.LayerNorm(width),nn.GELU())
        nn.init.constant_(self.heatmap.bias, -4.)

    def forward(self, crop):
        feature = self.stem(crop)
        feature = feature + self.refine(feature)
        descriptor = self.project(F.adaptive_avg_pool2d(feature,(4,4)).flatten(1))
        return descriptor, self.heatmap(feature)


class PartBranch(nn.Module):
    """Independent local encoder, stereo feature fusion and causal GRU."""
    def __init__(self, width, hidden, output_dim):
        super().__init__()
        self.encoder = LocalPartEncoder(width)
        # K/D (9), box centre/extent (4), visible fraction (1), keypoint confidence (1).
        self.metadata = nn.Sequential(nn.Linear(15,width),nn.GELU(),nn.Linear(width,width))
        self.cross_view = nn.Sequential(nn.LayerNorm(2*width),nn.Linear(2*width,width),nn.GELU(),
                                        nn.Linear(width,width))
        self.temporal = nn.GRU(width,hidden,batch_first=True)
        self.head = nn.Sequential(nn.Linear(2*hidden,hidden),nn.GELU(),nn.Linear(hidden,output_dim))

    def forward(self, crops, metadata, batch, time, hidden=None):
        descriptor, heat = self.encoder(crops)
        descriptor = descriptor + self.metadata(metadata.to(descriptor.dtype))
        paired = descriptor.reshape(batch*time,2,-1)
        # The other eye contributes local instrument evidence, never global FPN tokens.
        fused = paired + self.cross_view(torch.cat((paired,paired.flip(1)),-1))
        sequence = fused.reshape(batch,time,2,-1).permute(0,2,1,3).reshape(batch*2,time,-1)
        state, next_hidden = self.temporal(sequence,hidden)
        state = state.reshape(batch,2,time,-1).permute(0,2,1,3)
        return self.head(state.flatten(-2)).float(), heat, next_hidden


class ForegroundParts(nn.Module):
    """Wrist and paired-jaw recognition with disjoint learned temporal states."""
    def __init__(self, config, width, hidden):
        super().__init__()
        self.config = dict(config)
        self.roi_jitter_centre_px = float(config.get('part_roi_jitter_center_px', 0.0))
        self.roi_jitter_scale_fraction = float(config.get('part_roi_jitter_scale_fraction', 0.0))
        if (not math.isfinite(self.roi_jitter_centre_px) or self.roi_jitter_centre_px < 0
                or not math.isfinite(self.roi_jitter_scale_fraction)
                or not 0 <= self.roi_jitter_scale_fraction < 1):
            raise ValueError('Invalid training ROI jitter bounds')
        self.wrist_branch = PartBranch(width,hidden,2)
        self.jaw_branch = PartBranch(width,hidden,1)

    def forward(self, hires_images, mask_logits, landmark_uv, landmark_confidence,
                camera, image_size, batch, time, hidden=None):
        centres, extents = part_regions(landmark_uv, image_size, self.config)
        centres, extents = jitter_part_regions(centres, extents, image_size, batch, time,
            self.roi_jitter_centre_px, self.roi_jitter_scale_fraction, self.training)
        crops = foreground_crops(hires_images,mask_logits.detach().sigmoid(),centres,extents,image_size,
            crop_size=int(self.config.get('part_crop_size',160)),
            cutoff=float(self.config.get('part_mask_cutoff',.2)),
            dilation_px=int(self.config.get('part_mask_dilation_px',1)),
            chunk_size=int(self.config.get('part_crop_chunk_size',8)))
        observed = crops[:,:,3].mean((-2,-1)).detach()
        norm = centres.new_tensor([image_size[1],image_size[0]])
        boxmeta = torch.cat(((centres+.5)/norm, extents/norm),-1)
        confidence = landmark_confidence.detach().mean(-1)
        common = camera.reshape(batch*time*2,9)
        logits, heatmaps, states = [],[],[]
        for i,branch in enumerate((self.wrist_branch,self.jaw_branch)):
            meta = torch.cat((common,boxmeta[:,i],observed[:,i,None],confidence[:,None]),-1)
            previous = None if hidden is None else hidden[i:i+1]
            q,heat,state = branch(crops[:,i].contiguous(),meta,batch,time,previous)
            logits.append(q);heatmaps.append(heat);states.append(state)
        hh,ww = heatmaps[0].shape[-2:]
        result = {
            'part_heatmap_logits':torch.stack(heatmaps,1).reshape(batch,time,2,2,3,hh,ww),
            'part_roi_centres':centres.reshape(batch,time,2,2,2),
            'part_roi_half_extents':extents.reshape(batch,time,2,2,2),
            'part_observed_fraction':observed.reshape(batch,time,2,2),
        }
        return torch.cat(logits,-1),result,torch.cat(states,0)
