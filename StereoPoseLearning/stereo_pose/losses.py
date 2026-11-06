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


class PoseLoss(nn.Module):
    DEFAULT_WEIGHTS = {"position": 1.0, "rotation": 1.0, "joints": 1.0,
                       "geometry": 1.0, "part_rotation": 0.25, "projection": 0.2,
                       "depth": 0.1, "mask": 0.1, "heatmap": 1.0}

    def __init__(self, kinematics: FixedKinematics, weights: dict | None = None,
                 position_scale_mm: float = 10.0, projection_scale_px: float = 20.0):
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
        losses["position"] = F.smooth_l1_loss(pred["C"]/self.position_scale_mm, target["C"]/self.position_scale_mm)
        losses["rotation"] = _rotation_loss(pred["R"], target["R"])
        losses["joints"] = F.smooth_l1_loss(pred["joints"], target["joints"])
        losses["geometry"] = F.smooth_l1_loss(geom["points"]/self.position_scale_mm, true_points/self.position_scale_mm)
        losses["part_rotation"] = _rotation_loss(geom["R_jaws"], true_jaws)
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
            projection_error = torch.where(valid, (uv-gt_uv).norm(dim=-1), 0).sum()/valid.sum().clamp_min(1)
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
        total = sum(self.weights[name]*value for name, value in losses.items())
        with torch.no_grad():
            metrics = {"loss/"+name: value.detach() for name, value in losses.items()}
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
        return total, metrics
