"""Structured state supervision with fixed-identity articulated geometry losses."""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .geometry import FixedKinematics, project_brown


def rotation_error_degrees(prediction: Tensor, truth: Tensor) -> Tensor:
    relative = prediction.transpose(-1, -2) @ truth
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2
    return torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))


def _rotation_loss(prediction: Tensor, truth: Tensor) -> Tensor:
    # Chordal SO(3) loss has finite derivatives at zero error and is invariant
    # to q/-q because the inputs are rotation matrices.
    return (prediction-truth).square().sum((-2, -1)).mean()/8


def local_heatmap_targets(points: Tensor, K: Tensor, D: Tensor, centres: Tensor,
                          half_extents: Tensor, heatmap_size: tuple[int, int],
                          image_size: Tensor, sigma: float = 1.5) -> tuple[Tensor, Tensor]:
    """Amodal H/J targets in predicted wrist/jaw crops, without visibility filtering.

    ``points`` is [..., eye, N, 3]. Crop centres/half extents are
    [..., eye, part, 2] in LOW-RES network image pixel-centre coordinates.
    Output maps are [..., eye, part, N, H, W]. A target must be in front of
    the camera, inside the source image and inside this crop. A hidden surface
    point remains supervised when those geometric conditions hold.

    This is exactly the inverse of align_corners=False crop sampling:
    u = centre_x + (2 * (j + .5) / W - 1) * half_extent_x.
    Crop placement is detached so this auxiliary cannot move the ROI to avoid
    its target. No ground truth selects a crop or enters the visual network.
    """
    h, w = map(int, heatmap_size)
    if min(h, w) <= 0 or sigma <= 0:
        raise ValueError('Local heatmap dimensions and sigma must be positive')
    if centres.shape != half_extents.shape or centres.shape[-1] != 2:
        raise ValueError('Local crop centres and half extents must have matching xy shapes')
    if centres.shape[:-2] != points.shape[:-2]:
        raise ValueError('Local crops must match point batch/time/eye dimensions')
    if not torch.isfinite(half_extents).all() or (half_extents <= 0).any():
        raise ValueError('Local crop half extents must be finite and positive')
    uv = project_brown(points, K, D)
    sizes = image_size.to(uv)
    while sizes.ndim < uv.ndim:
        sizes = sizes.unsqueeze(-2)
    # Image size stores (height, width), whereas uv stores (x, y).
    in_image = ((uv[..., 0] >= -.5) & (uv[..., 0] < sizes[..., 1] - .5)
                & (uv[..., 1] >= -.5) & (uv[..., 1] < sizes[..., 0] - .5))
    valid_point = (points[..., 2] > 1e-3) & torch.isfinite(uv).all(-1) & in_image
    normalized = ((uv.unsqueeze(-3) - centres.detach().unsqueeze(-2))
                  / half_extents.detach().unsqueeze(-2))
    valid = valid_point.unsqueeze(-2) & (normalized >= -1).all(-1) & (normalized < 1).all(-1)
    location = (normalized + 1) * uv.new_tensor([w, h]) / 2 - .5
    location = torch.where(valid[..., None], location, torch.zeros_like(location))
    yy, xx = torch.meshgrid(torch.arange(h, dtype=uv.dtype, device=uv.device),
                            torch.arange(w, dtype=uv.dtype, device=uv.device), indexing='ij')
    distance2 = ((xx - location[..., 0, None, None]).square()
                 + (yy - location[..., 1, None, None]).square())
    maps = torch.exp(-distance2 / (2 * sigma * sigma)) * valid[..., None, None]
    return maps, valid


class PoseLoss(nn.Module):
    DEFAULT_WEIGHTS = {"position": 1.0, "rotation": 1.0, "joints": 1.0,
                       "geometry": 1.0, "part_rotation": 0.25, "projection": 0.2,
                       "depth": 0.1, "mask": 0.1, "heatmap": 1.0, "landmark_consistency": 0.0,
                       "local_heatmap": 0.0}

    def __init__(self, kinematics: FixedKinematics, weights: dict | None = None,
                 position_scale_mm: float = 10.0, projection_scale_px: float = 20.0, joint_scale_deg=None,
                 decouple_geometry: bool = False, local_heatmap_sigma: float = 1.5):
        super().__init__()
        self.kinematics = kinematics
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            if set(weights) - set(self.weights):
                raise ValueError("Unknown loss weight: " + str(set(weights)-set(self.weights)))
            self.weights.update(weights)
        if position_scale_mm <= 0 or projection_scale_px <= 0:
            raise ValueError("Loss normalization scales must be positive")
        self.position_scale_mm = float(position_scale_mm)
        self.projection_scale_px = float(projection_scale_px)
        self.decouple_geometry = bool(decouple_geometry)
        self.local_heatmap_sigma = float(local_heatmap_sigma)
        if self.local_heatmap_sigma <= 0:
            raise ValueError('Local heatmap sigma must be positive')
        scale=torch.ones(3) if joint_scale_deg is None else torch.deg2rad(torch.tensor(joint_scale_deg,dtype=torch.float32))
        if scale.shape!=(3,) or not torch.isfinite(scale).all() or not (scale>0).all():raise ValueError('Invalid joint scales')
        self.register_buffer('joint_scale',scale)

    def forward(self, pred: dict[str, Tensor], target: dict[str, Tensor]):
        zero = pred["C"].sum()*0
        losses = {name: zero for name in self.weights}
        geom = self.kinematics(pred["C"], pred["R"], pred["joints"])
        if "points" not in target or "R_jaws" not in target:
            truth_geom = self.kinematics(target["C"], target["R"], target["joints"])
        else:
            truth_geom = target
        true_points = target.get("points", truth_geom["points"])
        true_jaws = target.get("R_jaws", truth_geom["R_jaws"])
        # Optional ablation only. Each branch is supervised with the other
        # branch anchored to truth; gradients cannot ask joint angles to
        # compensate for shaft position/rotation errors. Equal averaging keeps
        # the loss scale comparable. The actual complete prediction remains
        # used for metrics and the positive-depth safety penalty.
        supervised_geometries = [geom]
        if self.decouple_geometry:
            supervised_geometries = [
                self.kinematics(target['C'], target['R'], pred['joints']),
                self.kinematics(pred['C'], pred['R'], target['joints']),
            ]
        losses["position"] = F.smooth_l1_loss(pred["C"]/self.position_scale_mm, target["C"]/self.position_scale_mm)
        losses["rotation"] = _rotation_loss(pred["R"], target["R"])
        losses["joints"] = F.smooth_l1_loss((pred["joints"]-target["joints"])/self.joint_scale, torch.zeros_like(pred["joints"]))
        losses["geometry"] = sum(F.smooth_l1_loss(g["points"]/self.position_scale_mm,
                                                 true_points/self.position_scale_mm)
                                  for g in supervised_geometries) / len(supervised_geometries)
        losses["part_rotation"] = sum(_rotation_loss(g["R_jaws"], true_jaws)
                                       for g in supervised_geometries) / len(supervised_geometries)
        # A surface behind the optical centre cannot be silently ignored. The
        # barrier supplies a corrective gradient, while 1 mm clipping avoids
        # exploding distortion powers during early random initialization.
        losses["depth"] = F.relu(1.0-geom["points"][..., 2]).mean()/self.position_scale_mm
        projection_error = None
        if "K" in target and "D" in target:
            safe_points = torch.cat((geom["points"][..., :2], geom["points"][..., 2:].clamp_min(1.0)), -1)
            uv = project_brown(safe_points, target["K"], target["D"])
            gt_uv = project_brown(true_points, target["K"], target["D"])
            valid = (true_points[..., 2] > 1e-3) & torch.isfinite(gt_uv).all(-1)
            per_point = F.smooth_l1_loss(uv/self.projection_scale_px, gt_uv/self.projection_scale_px,
                                       reduction="none").mean(-1)
            losses["projection"] = torch.where(valid, per_point, 0).sum()/valid.sum().clamp_min(1)
            if self.decouple_geometry:
                branch_losses = []
                for g in supervised_geometries:
                    branch_safe = torch.cat((g['points'][..., :2], g['points'][..., 2:].clamp_min(1.0)), -1)
                    branch_uv = project_brown(branch_safe, target['K'], target['D'])
                    branch_error = F.smooth_l1_loss(branch_uv/self.projection_scale_px,
                        gt_uv/self.projection_scale_px, reduction='none').mean(-1)
                    branch_losses.append(torch.where(valid, branch_error, 0).sum()/valid.sum().clamp_min(1))
                losses['projection'] = sum(branch_losses)/len(branch_losses)
            projection_error = torch.where(valid, (uv-gt_uv).norm(dim=-1), 0).sum()/valid.sum().clamp_min(1)
            if self.weights['landmark_consistency'] and 'landmark_uv' in pred:
                # Amodal landmarks: an occluded but in-frame reference remains
                # supervised. Detector coordinates are a detached teacher here;
                # the original heatmap loss anchors them to actual labels.
                confidence = pred['landmark_confidence'].detach()
                applicable = valid[..., :5] & target['heatmap_valid'].bool() & (confidence >= .3)
                residual = F.smooth_l1_loss(uv[..., :5, :]/self.projection_scale_px,
                    pred['landmark_uv'].detach()/self.projection_scale_px, reduction='none').mean(-1)
                reliability = applicable*confidence
                losses['landmark_consistency'] = (residual*reliability).sum()/reliability.sum().clamp_min(1)
        if "mask_logits" in pred and "masks" in target:
            mask = target["masks"].to(pred["mask_logits"])
            logits = pred["mask_logits"]
            if logits.shape != mask.shape:
                raise ValueError(f"Mask shape mismatch: {logits.shape} vs {mask.shape}")
            probability = logits.sigmoid()
            bce = F.binary_cross_entropy_with_logits(logits, mask)
            dims = (-3, -2, -1)
            dice = 1-(2*(probability*mask).sum(dims)+1)/(probability.sum(dims)+mask.sum(dims)+1)
            losses["mask"] = bce+dice.mean()
        if "heatmap_logits" in pred and "heatmaps" in target:
            logits = pred["heatmap_logits"]
            maps = target["heatmaps"].to(logits)
            if logits.shape != maps.shape:
                raise ValueError(f"Heatmap shape mismatch: {logits.shape} vs {maps.shape}")
            valid = target.get("heatmap_valid", torch.ones_like(maps[..., 0, 0])).to(logits)
            error = ((logits.sigmoid()-maps).square()*(1+20*maps)).mean((-2, -1))
            losses["heatmap"] = (error*valid).sum()/valid.sum().clamp_min(1)
        local_valid_count = None
        if self.weights['local_heatmap']:
            required_pred = ('part_heatmap_logits', 'part_roi_centres', 'part_roi_half_extents')
            if any(name not in pred for name in required_pred):
                raise ValueError('Local heatmap supervision requires predicted part maps and crop transforms')
            if any(name not in target for name in ('K', 'D', 'image_size')):
                raise ValueError('Local heatmap supervision requires imaging calibration and source image size')
            logits = pred['part_heatmap_logits']
            if logits.shape[-4:-2] != (2, 3):
                raise ValueError('Expected wrist/jaw local maps, each with H/J1/J2 channels')
            local_points = torch.stack((true_points[..., 1:3, :].mean(-2),
                                        true_points[..., 3, :], true_points[..., 4, :]), -2)
            maps, local_valid = local_heatmap_targets(local_points, target['K'], target['D'],
                pred['part_roi_centres'], pred['part_roi_half_extents'], tuple(logits.shape[-2:]),
                target['image_size'], self.local_heatmap_sigma)
            if maps.shape != logits.shape:
                raise ValueError(f'Local heatmap shape mismatch: {logits.shape} vs {maps.shape}')
            error = ((logits.float().sigmoid()-maps).square()*(1+20*maps)).mean((-2, -1))
            losses['local_heatmap'] = (error*local_valid).sum()/local_valid.sum().clamp_min(1)
            local_valid_count = local_valid.sum()
        total = sum(self.weights[name]*value for name, value in losses.items())
        with torch.no_grad():
            metrics = {"loss/"+name: value.detach() for name, value in losses.items()}
            metrics.update({'weighted_loss/'+name:(self.weights[name]*value).detach() for name,value in losses.items()})
            errors=torch.rad2deg((pred['joints']-target['joints']).abs()).mean(tuple(range(pred['joints'].ndim-1)))
            for i,name in enumerate(('wrist_pitch_deg','wrist_yaw_deg','jaw_half_deg')):metrics[name]=errors[i]
            metrics.update({
                "loss/total": total.detach(),
                "position_mm": (pred["C"]-target["C"]).norm(dim=-1).mean(),
                "rotation_deg": rotation_error_degrees(pred["R"], target["R"]).mean(),
                "joint_mae_deg": torch.rad2deg((pred["joints"]-target["joints"]).abs()).mean(),
                "tip_mm": (geom["points"][..., 3:5, :]-true_points[..., 3:5, :]).norm(dim=-1).mean(),
                "part_rotation_deg": rotation_error_degrees(geom["R_jaws"], true_jaws).mean(),
            })
            if projection_error is not None:
                metrics["projection_px"] = projection_error.detach()
            if local_valid_count is not None:
                metrics['local_heatmap_valid_points'] = local_valid_count.detach().to(total)
        return total, metrics
