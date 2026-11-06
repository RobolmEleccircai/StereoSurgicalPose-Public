"""Geometry-supervised, unrectified stereo, causal articulated pose network.

This is a task-specific architecture, not an implementation of a named paper.
Only RGB, each camera's K/D and recurrent features enter forward().
"""
import math

import torch
from torch import nn
import torch.nn.functional as F
import timm

from .geometry import rotation_6d_to_matrix, values17
from .guidance import GuidedRegions
from .parts import ForegroundParts


class CrossViewBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2),
                                 nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, tokens):
        left, right = tokens.unbind(1)
        ln, rn = self.norm(left), self.norm(right)
        # Both directions use the original pair, not a sequentially changed view.
        left = left + self.attention(ln, rn, rn, need_weights=False)[0]
        right = right + self.attention(rn, ln, ln, need_weights=False)[0]
        return torch.stack((left + self.ffn(left), right + self.ffn(right)), 1)


class StereoPoseNet(nn.Module):
    def __init__(self, config, joint_bounds, pretrained=None):
        super().__init__()
        self.config = dict(config)
        width = config.get("feature_dim", 128)
        hidden = config.get("hidden_dim", 256)
        self.grid = tuple(config.get("token_grid", [6, 10]))
        if pretrained is None:
            pretrained = config.get("pretrained", True)
        self.backbone = timm.create_model(config["backbone"], pretrained=pretrained,
                                          features_only=True, out_indices=(0, 1, 2, 3))
        channels = self.backbone.feature_info.channels()
        self.laterals = nn.ModuleList(nn.Conv2d(c, width, 1) for c in channels)
        self.spatial = nn.Sequential(nn.Conv2d(width, width, 3, padding=1),
                                     nn.GroupNorm(8, width), nn.GELU())
        self.mask_head = nn.Conv2d(width, 1, 1)
        self.heatmap_head = nn.Conv2d(width, 5, 1)
        self.position = nn.Linear(2, width)
        self.camera = nn.Sequential(nn.Linear(9, width), nn.GELU(), nn.Linear(width, width))
        self.view_embedding = nn.Parameter(torch.randn(2, width) * .01)
        self.fusion = nn.Sequential(*[CrossViewBlock(width, config.get("attention_heads", 4))
                                     for _ in range(config.get("fusion_layers", 2))])
        self.pool_score = nn.Linear(width, 1)
        self.temporal = nn.GRU(width, hidden, batch_first=True)
        self.pose_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 9))
            for _ in range(2)])
        self.joint_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))
        self.guided = config.get('feature_guidance', False)
        self.foreground_parts = bool(config.get('foreground_parts', False))
        if self.foreground_parts and not self.guided:
            raise ValueError('Foreground parts requires the guided localization model')
        if self.guided:
            region_config = dict(config)
            if self.foreground_parts:
                region_config['foreground_context_floor'] = 0.0
            self.guided_regions = GuidedRegions(width, region_config)
            self.guided_pose_pool = nn.Sequential(nn.Linear(6*width, width), nn.LayerNorm(width), nn.GELU())
            self.guided_joint_pool = nn.Sequential(nn.Linear(5*width, width), nn.LayerNorm(width), nn.GELU())
            self.guided_joint_temporal = nn.GRU(width, hidden, batch_first=True)
            # Retain these legacy parameters for checkpoint compatibility, but
            # never route pose or angles through unconstrained full-image pooling.
            self.position.requires_grad_(False)
            self.pool_score.requires_grad_(False)
        if self.foreground_parts:
            self.foreground_parts_net = ForegroundParts(config,width,hidden)
            # Old modules remain loadable for auditable checkpoint transfer,
            # but cannot provide a global-image bypass into the new angle heads.
            for module in (self.joint_head,self.guided_joint_pool,self.guided_joint_temporal):
                module.requires_grad_(False)
        bounds = torch.as_tensor(joint_bounds, dtype=torch.float32)
        if bounds.shape != (3, 2) or not torch.all(bounds[:, 1] > bounds[:, 0]):
            raise ValueError("Expected three physical lower/upper joint bounds")
        self.register_buffer("joint_lower", bounds[:, 0])
        self.register_buffer("joint_upper", bounds[:, 1])
        for head in self.pose_heads:
            nn.init.normal_(head[-1].weight, std=.001)
            with torch.no_grad():
                head[-1].bias.copy_(torch.tensor([0., 0., math.log(math.expm1(1.)),
                                                1., 0., 0., 0., 1., 0.]))
        nn.init.constant_(self.heatmap_head.bias, -4.)

    def forward(self, images, K, D, hidden=None, hires_images=None):
        if images.ndim != 6 or images.shape[2:4] != (2, 3):
            raise ValueError("images must be [B,T,2,3,H,W]")
        batch, time, _, _, height, width = images.shape
        if K.shape != (batch, time, 2, 3, 3) or D.shape != (batch, time, 2, 5):
            raise ValueError("K/D must match both views and all timestamps")
        if self.foreground_parts:
            if (hires_images is None or hires_images.dtype != torch.uint8 or hires_images.ndim != 6
                    or hires_images.shape[:4] != (batch,time,2,3)):
                raise ValueError('Foreground parts requires hires_images uint8 [B,T,2,3,Hraw,Wraw]')
            if hires_images.shape[-2] < height or hires_images.shape[-1] < width:
                raise ValueError('Local RGB source must not be smaller than the localization image')
            if hidden is not None and hidden.shape != (3,batch*2,self.temporal.hidden_size):
                raise ValueError('Foreground state must contain body, wrist and jaw GRU states')
        input_hidden = hidden
        flattened=images.reshape(batch * time * 2, 3, height, width)
        if self.config.get("channels_last",False):flattened=flattened.contiguous(memory_format=torch.channels_last)
        maps = self.backbone(flattened)
        pyramid = None
        for feature, lateral in reversed(list(zip(maps, self.laterals))):
            projected = lateral(feature)
            # torch 2.0 CUDA has no BF16 bilinear kernel; interpolate in FP32
            # and return to the current feature dtype without losing gradients.
            pyramid = projected if pyramid is None else projected + F.interpolate(
                pyramid.float(), size=projected.shape[-2:], mode="bilinear", align_corners=False).to(projected.dtype)
        spatial = self.spatial(pyramid)
        mask = self.mask_head(spatial)
        heat = self.heatmap_head(spatial)
        if self.guided:
            tokens, landmark_uv, landmark_confidence = self.guided_regions(spatial, mask, heat, (height, width))
        else:
            tokens = F.adaptive_avg_pool2d(spatial, self.grid).flatten(2).transpose(1, 2)
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, self.grid[0], device=images.device),
                                    torch.linspace(-1, 1, self.grid[1], device=images.device), indexing="ij")
            coordinates = torch.stack((xx, yy), -1).reshape(-1, 2)
            tokens = tokens + self.position(coordinates).to(tokens.dtype)[None]
        camera = torch.cat((torch.stack((K[..., 0, 0] / width, K[..., 1, 1] / height,
                                        K[..., 0, 2] / width, K[..., 1, 2] / height), -1), D), -1)
        tokens = tokens.reshape(batch * time, 2, -1, tokens.shape[-1])
        tokens = tokens + self.camera(camera.reshape(batch * time, 2, 9))[:, :, None]
        tokens = tokens + self.view_embedding[None, :, None]
        tokens = self.fusion(tokens)
        if self.guided:
            pooled = self.guided_pose_pool(tokens.flatten(-2)).reshape(batch, time, 2, -1)
            if not self.foreground_parts:
                local_pooled = self.guided_joint_pool(tokens[:, :, 1:].flatten(-2)).reshape(batch, time, 2, -1)
                local_input = local_pooled.permute(0, 2, 1, 3).reshape(batch*2, time, -1)
                local_hidden = None if hidden is None else hidden[1:2]
                local_states, next_local_hidden = self.guided_joint_temporal(local_input, local_hidden)
                local_states = local_states.reshape(batch, 2, time, -1).permute(0, 2, 1, 3)
            hidden = None if hidden is None else hidden[:1]
        else:
            weights = self.pool_score(tokens).softmax(dim=-2)
            pooled = (tokens * weights).sum(-2).reshape(batch, time, 2, -1)
        recurrent_input = pooled.permute(0, 2, 1, 3).reshape(batch * 2, time, -1)
        states, next_hidden = self.temporal(recurrent_input, hidden)
        states = states.reshape(batch, 2, time, -1).permute(0, 2, 1, 3)
        raw_pose = torch.stack([self.pose_heads[e](states[:, :, e]) for e in range(2)], 2).float()
        # ray (x,y) is UNDISTORTED normalized optical coordinate; Z is mm.
        z = F.softplus(raw_pose[..., 2:3]) * self.config.get("depth_scale_mm", 100.) + .001
        C = torch.cat((raw_pose[..., :2] * z, z), -1)
        R = rotation_6d_to_matrix(raw_pose[..., 3:])
        part_results = {}
        if self.foreground_parts:
            joint_logits,part_results,next_local_hidden = self.foreground_parts_net(
                hires_images.reshape(batch*time*2,3,*hires_images.shape[-2:]),mask,
                landmark_uv,landmark_confidence,camera,(height,width),batch,time,
                None if input_hidden is None else input_hidden[1:])
        else:
            joint_logits = self.joint_head((local_states if self.guided else states).flatten(-2)).float()
        joints = self.joint_lower + (self.joint_upper - self.joint_lower) * joint_logits.sigmoid()
        h, w = mask.shape[-2:]
        result = {"C": C, "R": R, "joints": joints,
                "mask_logits": mask.reshape(batch, time, 2, 1, h, w),
                "heatmap_logits": heat.reshape(batch, time, 2, 5, h, w)}
        result.update(part_results)
        if self.guided:
            result['landmark_uv'] = landmark_uv.reshape(batch, time, 2, 5, 2)
            result['landmark_confidence'] = landmark_confidence.reshape(batch, time, 2, 5)
            next_hidden = torch.cat((next_hidden, next_local_hidden), 0)
        return result, next_hidden

    @staticmethod
    def export_values(prediction):
        return values17(prediction["C"], prediction["R"], prediction["joints"])


def joint_bounds_from_model(model):
    by_name = {j["name"]: j for j in model["joints"]}
    names = ["psm_tool_pitch_joint", "psm_tool_yaw_joint", "psm_tool_gripper2_joint"]
    return [[by_name[n]["lower_rad"], by_name[n]["upper_rad"]] for n in names]
