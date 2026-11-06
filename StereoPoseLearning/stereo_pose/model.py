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
from .translation import C_TRANSLATION_HEAD, landmark_ray_depth_position


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
        if config.get("c_translation_head") != C_TRANSLATION_HEAD:
            raise ValueError("This model requires c_translation_head='landmark_ray_depth_v1'; legacy translation heads are incompatible")
        self.guided = bool(config.get("feature_guidance", False))
        self.foreground_parts = bool(config.get("foreground_parts", False))
        if not self.guided or not self.foreground_parts:
            raise ValueError("Unified C translation requires guided localization and foreground parts")
        width = config.get("feature_dim", 128)
        hidden = config.get("hidden_dim", 256)
        self.depth_scale_mm = float(config.get("depth_scale_mm", 100.0))
        if not math.isfinite(self.depth_scale_mm) or self.depth_scale_mm <= 0:
            raise ValueError("Optical depth scale must be finite and positive")
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
        self.camera = nn.Sequential(nn.Linear(9, width), nn.GELU(), nn.Linear(width, width))
        self.view_embedding = nn.Parameter(torch.randn(2, width) * .01)
        self.fusion = nn.Sequential(*[CrossViewBlock(width, config.get("attention_heads", 4))
                                     for _ in range(config.get("fusion_layers", 2))])
        self.temporal = nn.GRU(width, hidden, batch_first=True)
        self.rotation_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 6))
            for _ in range(2)])
        # The C location has exactly one source: landmark channel 0. These
        # heads predict only Z from [body recurrent state, fused C token].
        self.depth_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden + width, hidden), nn.GELU(), nn.Linear(hidden, 1))
            for _ in range(2)])
        region_config = dict(config)
        region_config['foreground_context_floor'] = 0.0
        self.guided_regions = GuidedRegions(width, region_config)
        self.guided_pose_pool = nn.Sequential(nn.Linear(6*width, width), nn.LayerNorm(width), nn.GELU())
        self.foreground_parts_net = ForegroundParts(config, width, hidden)
        bounds = torch.as_tensor(joint_bounds, dtype=torch.float32)
        if bounds.shape != (3, 2) or not torch.all(bounds[:, 1] > bounds[:, 0]):
            raise ValueError("Expected three physical lower/upper joint bounds")
        self.register_buffer("joint_lower", bounds[:, 0])
        self.register_buffer("joint_upper", bounds[:, 1])
        for head in self.rotation_heads:
            nn.init.zeros_(head[-1].weight)
            with torch.no_grad():
                head[-1].bias.copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))
        for head in self.depth_heads:
            nn.init.normal_(head[-1].weight, std=.001)
            nn.init.constant_(head[-1].bias, math.log(math.expm1(1.)))
        nn.init.constant_(self.heatmap_head.bias, -4.)

    def forward(self, images, K, D, hidden=None, hires_images=None):
        if images.ndim != 6 or images.shape[2:4] != (2, 3):
            raise ValueError("images must be [B,T,2,3,H,W]")
        batch, time, _, _, height, width = images.shape
        if K.shape != (batch, time, 2, 3, 3) or D.shape != (batch, time, 2, 5):
            raise ValueError("K/D must match both views and all timestamps")
        if (hires_images is None or hires_images.dtype != torch.uint8 or hires_images.ndim != 6
                or hires_images.shape[:4] != (batch,time,2,3)):
            raise ValueError('Foreground parts requires hires_images uint8 [B,T,2,3,Hraw,Wraw]')
        if hires_images.shape[-2] < height or hires_images.shape[-1] < width:
            raise ValueError('Local RGB source must not be smaller than the localization image')
        if hidden is not None and hidden.shape != (3,batch*2,self.temporal.hidden_size):
            raise ValueError('Foreground state must contain body, wrist and jaw GRU states')
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
        tokens, landmark_uv, landmark_confidence = self.guided_regions(spatial, mask, heat, (height, width))
        camera = torch.cat((torch.stack((K[..., 0, 0] / width, K[..., 1, 1] / height,
                                        K[..., 0, 2] / width, K[..., 1, 2] / height), -1), D), -1)
        tokens = tokens.reshape(batch * time, 2, -1, tokens.shape[-1])
        tokens = tokens + self.camera(camera.reshape(batch * time, 2, 9))[:, :, None]
        tokens = tokens + self.view_embedding[None, :, None]
        tokens = self.fusion(tokens)
        pooled = self.guided_pose_pool(tokens.flatten(-2)).reshape(batch, time, 2, -1)
        # Token 0 is the whole instrument; token 1 is C after cross-view fusion.
        C_token = tokens[:, :, 1].reshape(batch, time, 2, -1)
        recurrent_input = pooled.permute(0, 2, 1, 3).reshape(batch * 2, time, -1)
        states, next_hidden = self.temporal(recurrent_input, None if hidden is None else hidden[:1])
        states = states.reshape(batch, 2, time, -1).permute(0, 2, 1, 3)
        raw_rotation = torch.stack([self.rotation_heads[e](states[:, :, e]) for e in range(2)], 2).float()
        depth_features = torch.cat((states, C_token), dim=-1)
        raw_depth = torch.stack([self.depth_heads[e](depth_features[:, :, e]) for e in range(2)], 2).float()
        z = F.softplus(raw_depth) * self.depth_scale_mm + .001
        all_landmark_uv = landmark_uv.reshape(batch, time, 2, 5, 2)
        C_uv = all_landmark_uv[..., 0, :]
        C = landmark_ray_depth_position(C_uv, z, K, D)
        R = rotation_6d_to_matrix(raw_rotation)
        joint_logits,part_results,next_local_hidden = self.foreground_parts_net(
            hires_images.reshape(batch*time*2,3,*hires_images.shape[-2:]),mask,
            landmark_uv,landmark_confidence,camera,(height,width),batch,time,
            None if hidden is None else hidden[1:])
        joints = self.joint_lower + (self.joint_upper - self.joint_lower) * joint_logits.sigmoid()
        h, w = mask.shape[-2:]
        result = {"C": C, "R": R, "joints": joints, "c_uv": C_uv, "c_depth": z,
                "mask_logits": mask.reshape(batch, time, 2, 1, h, w),
                "heatmap_logits": heat.reshape(batch, time, 2, 5, h, w)}
        result.update(part_results)
        result['landmark_uv'] = all_landmark_uv
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
