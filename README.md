# StereoSurgicalPose

Stereo vision for surgical instrument localization and pose estimation, with reproducible camera configurations and simulation experiment records.

## Modules

| Directory | Contents |
|---|---|
| `CameraImagingConfig/` | Approved camera-imaging parameters, reference calibration and a preview |
| `StereoCalibration/` | Module scaffold |
| `SurgicalPoseEstimation/` | Module scaffold |

## Confirmed imaging baseline

Accepted on **2026-11-07**:

- Minimum vignette gain: **3.5%** in linear RGB.
- Global brightness multiplier: **0.75** (25% reduction in linear RGB).
- Ellipse center: **(959.5, 539.5) px**; semiaxes: **(840, 470) px**.
- Smoothstep transition: **0.30–1.10** in normalized elliptical radius.
- Preview resolution: **1920 × 1080 per eye**.

[Parameter record](CameraImagingConfig/docs/IMAGING_BASELINE.md) · [Machine-readable baseline](CameraImagingConfig/config/confirmed_baseline.json) · [Sources](CameraImagingConfig/docs/SOURCES.md)

![Approved stereo preview](CameraImagingConfig/examples/approved_stereo_preview.jpg)

The current preview uses cached path-traced rendering from Isaac Sim 5.1. Visual acceptance is recorded; a calibrated match to the target physical camera has not been established.
