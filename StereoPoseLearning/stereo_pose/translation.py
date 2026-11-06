"""Differentiable C translation from one landmark and its optical depth.

The landmark uses distorted image-pixel coordinates at the resolution of K.
Depth is the optical Z of the internal shaft reference, not a surface depth map.
No independent lateral translation or post-decoding correction is available.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .geometry import _distortion_terms


C_TRANSLATION_HEAD = "landmark_ray_depth_v1"


def landmark_ray_depth_position(
    uv: Tensor,
    depth_mm: Tensor,
    K: Tensor,
    D: Tensor,
    *,
    iterations: int = 10,
) -> Tensor:
    """Return camera-optical C [...,3] in mm from uv [...,2] and Z [...,1].

    K and Brown coefficients D must have the same leading dimensions as uv.
    The caller supplies positive depth; StereoPoseNet guarantees it by softplus.
    Shape/device checks use tensor metadata, with no per-step GPU synchronization.
    Float64 geometry avoids autocast/TF32 rounding without changing global flags.
    The float32 result remains differentiable with respect to uv and depth.
    """
    if uv.ndim < 1 or uv.shape[-1] != 2:
        raise ValueError("C landmark must end in two image-pixel coordinates")
    leading = uv.shape[:-1]
    if (depth_mm.shape != leading + (1,) or K.shape != leading + (3, 3)
            or D.shape != leading + (5,)):
        raise ValueError("C landmark, optical depth and K/D must have matching leading dimensions")
    if any(not value.is_floating_point() for value in (uv, depth_mm, K, D)):
        raise TypeError("C landmark, depth and calibration must use floating-point tensors")
    if any(value.device != uv.device for value in (depth_mm, K, D)):
        raise ValueError("C landmark, depth and calibration must be on the same device")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
        raise ValueError("Brown inversion needs a positive integer iteration count")
    with torch.autocast(device_type=uv.device.type, enabled=False):
        pixels, calibration, distortion = uv.to(torch.float64), K.to(torch.float64), D.to(torch.float64)
        # inv_ex with check_errors=False avoids the host synchronization in
        # linalg.inv. Valid, nonsingular calibration is an input precondition;
        # no per-batch inspection of the device-side info tensor is needed.
        inverse, _ = torch.linalg.inv_ex(calibration, check_errors=False)
        homogeneous = torch.cat((pixels, torch.ones_like(pixels[..., :1])), dim=-1)
        distorted = (inverse @ homogeneous[..., None]).squeeze(-1)
        xd, yd = (distorted[..., :2] / distorted[..., 2:]).unbind(-1)
        x, y = xd, yd
        # Same differentiable Brown-Conrady Newton inversion as geometry.py,
        # with its shared distortion polynomial and the asynchronous K inverse.
        for _ in range(iterations):
            radial, tx, ty, (k1, k2, p1, p2, k3), r2 = _distortion_terms(x, y, distortion)
            g = k1 + 2*k2*r2 + 3*k3*r2*r2
            rx, ry = 2*x*g, 2*y*g
            j11 = radial + x*rx + 2*p1*y + 6*p2*x
            j12 = x*ry + 2*p1*x + 2*p2*y
            j21 = y*rx + 2*p1*x + 2*p2*y
            j22 = radial + y*ry + 6*p1*y + 2*p2*x
            residual_x, residual_y = x*radial+tx-xd, y*radial+ty-yd
            determinant = j11*j22 - j12*j21
            eps = torch.finfo(determinant.dtype).eps * 8
            determinant = torch.where(determinant.abs() >= eps, determinant,
                                      torch.where(determinant >= 0, eps, -eps))
            step_x = (j22*residual_x-j12*residual_y)/determinant
            step_y = (-j21*residual_x+j11*residual_y)/determinant
            x, y = x-step_x, y-step_y
        ray_xy = torch.stack((x, y), dim=-1)
        ray = torch.cat((ray_xy, torch.ones_like(ray_xy[..., :1])), dim=-1)
        return (ray * depth_mm.to(torch.float64)).to(torch.float32)
