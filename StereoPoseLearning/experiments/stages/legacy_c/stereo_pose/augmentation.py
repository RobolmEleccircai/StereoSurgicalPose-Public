"""Online, training-only photometric augmentation after shared-cache retrieval.

These ranges are simulation-training design values, not camera measurements.
No spatial transform is applied: K, D, masks, keypoints, poses and timestamps
remain unchanged. A sequence shares brightness/contrast/gamma/color parameters
between frames and eyes; sensor noise has independent pixel realizations.
Raw uint8 high-resolution images are augmented first, quantized to uint8, then
resized to regenerate the normalized detector image with matching appearance.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


DEFAULTS = {
    "probability": 0.8,
    "brightness": [0.8, 1.2],
    "contrast": [0.85, 1.15],
    "channel_gain": [0.95, 1.05],
    "gamma": [0.9, 1.1],
    "noise_probability": 0.3,
    "noise_std": [0.0, 0.01],
    "blur_probability": 0.2,
    "blur_sigma": [0.5, 1.0],
}


def _validate(config):
    unknown = set(config) - set(DEFAULTS) - {"enabled"}
    if unknown:
        raise ValueError("Unknown augmentation options: " + str(sorted(unknown)))
    values = {**DEFAULTS, **config}
    for key in ("probability", "noise_probability", "blur_probability"):
        x = float(values[key])
        if not math.isfinite(x) or not 0 <= x <= 1:
            raise ValueError("Invalid augmentation probability: " + key)
    for key in ("brightness", "contrast", "channel_gain", "gamma", "noise_std", "blur_sigma"):
        x = values[key]
        if len(x) != 2 or not all(math.isfinite(float(v)) for v in x) or x[1] < x[0]:
            raise ValueError("Invalid augmentation range: " + key)
        if x[0] < 0 or (key in ("gamma", "blur_sigma") and x[0] <= 0):
            raise ValueError("Invalid nonpositive augmentation range: " + key)
    if values["blur_sigma"][1] > 3:
        raise ValueError("Blur sigma must remain <= 3 input pixels")
    return values


def _draw(bounds, shape, device):
    return torch.empty(shape, device=device, dtype=torch.float32).uniform_(float(bounds[0]), float(bounds[1]))


def _blur(rgb, sigma):
    """Separable Gaussian; sigma is shared across a complete sequence."""
    radius = max(1, math.ceil(3 * sigma))
    grid = torch.arange(-radius, radius + 1, device=rgb.device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (grid / sigma).square())
    kernel = kernel / kernel.sum()
    images = rgb.reshape(-1, 3, *rgb.shape[-2:])
    horizontal = kernel[None, None, None, :].expand(3, 1, 1, -1)
    vertical = kernel[None, None, :, None].expand(3, 1, -1, 1)
    images = F.conv2d(F.pad(images, (radius, radius, 0, 0), mode="replicate"), horizontal, groups=3)
    images = F.conv2d(F.pad(images, (0, 0, radius, radius), mode="replicate"), vertical, groups=3)
    return images.reshape_as(rgb)


def augment_batch(batch: dict, config: dict | None) -> dict:
    """Return a new augmented batch; never mutate cached images or targets.

    Call only during training, after transferring a collated batch to its device.
    Empty/disabled configurations are exact no-ops, including object identity.
    The model receives predicted-mask-gated RGB later; this operation never uses
    ground-truth masks to suppress input pixels.
    """
    if not config or not config.get("enabled", True):
        return batch
    settings = _validate(config)
    for sequence in batch.get("metadata", []):
        for frame in sequence:
            if frame.get("split", "train") != "train":
                raise ValueError("Photometric augmentation is training-only")
    images = batch["images"]
    if images.ndim != 6 or images.shape[2:4] != (2, 3):
        raise ValueError("images must be [B,T,2,3,H,W]")
    highres = batch.get("hires_images")
    if highres is not None and (highres.dtype != torch.uint8 or highres.ndim != 6
                               or highres.shape[:4] != images.shape[:4]):
        raise ValueError("hires_images must be uint8 [B,T,2,3,Hraw,Wraw]")
    device = images.device
    if highres is not None and highres.device != device:
        raise ValueError("Transfer both image resolutions to the same device first")
    count = images.shape[0]
    active = torch.rand(count, device=device) < settings["probability"]
    if not bool(active.any()):
        return batch
    # One random parameter set per sequence. Iteration limits peak full-size
    # float image memory instead of materializing all B*T*2 raw images at once.
    brightness = _draw(settings["brightness"], (count,), device)
    contrast = _draw(settings["contrast"], (count,), device)
    gamma = _draw(settings["gamma"], (count,), device)
    gain = _draw(settings["channel_gain"], (count, 3), device)
    noisy = torch.rand(count, device=device) < settings["noise_probability"]
    noise_std = _draw(settings["noise_std"], (count,), device)
    blurred = torch.rand(count, device=device) < settings["blur_probability"]
    blur_sigma = _draw(settings["blur_sigma"], (count,), device)
    mean = images.new_tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1)
    std = images.new_tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1)
    output = dict(batch)
    output["images"] = images.clone()
    if highres is not None:
        output["hires_images"] = highres.clone()
    for b in range(count):
        if not bool(active[b]):
            continue
        rgb = highres[b].float().div(255) if highres is not None else images[b].float() * std + mean
        rgb = (rgb - 0.5) * contrast[b] + 0.5
        rgb = (rgb * brightness[b] * gain[b].view(1, 1, 3, 1, 1)).clamp(0, 1)
        rgb = rgb.pow(gamma[b])
        if bool(blurred[b]):
            rgb = _blur(rgb, float(blur_sigma[b]))
        if bool(noisy[b]):
            rgb = rgb + torch.randn_like(rgb) * noise_std[b]
        rgb = rgb.clamp(0, 1)
        if highres is not None:
            raw = rgb.mul(255).round().to(torch.uint8)
            output["hires_images"][b].copy_(raw)
            # Use the same quantized image content as the local raw RGB branch.
            rgb = F.interpolate(raw.reshape(-1, 3, *raw.shape[-2:]).float().div(255),
                                size=images.shape[-2:], mode="bilinear", align_corners=False,
                                antialias=True).reshape(*images[b].shape)
        output["images"][b].copy_(((rgb - mean) / std).to(images.dtype))
    return output
