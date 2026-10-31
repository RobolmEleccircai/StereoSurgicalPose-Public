# Approved imaging baseline

Accepted: **2026-11-07**. Source experiment: `run14_border_3p5_global075`.

| Setting | Applied value |
|---|---:|
| Resolution, per eye | 1920 × 1080 |
| Minimum vignette gain | 0.035 |
| Global linear brightness multiplier | 0.75 |
| Ellipse center | (959.5, 539.5) px |
| Ellipse semiaxes | (840, 470) px |
| Transition start / end | 0.30 / 1.10 |
| Transition function | Smoothstep |
| Effective focal length | 4 mm |
| f-number | 5.6 |
| Focus distance | 100 mm |
| Source renderer | PathTracing |
| Source sampling settings | 64 samples/frame; total setting 256 |

```text
r = sqrt(((u - 959.5)/840)^2 + ((v - 539.5)/470)^2)
t = clip((r - 0.30)/0.80, 0, 1)
mask_gain = 1 - 0.965*t*t*(3 - 2*t)
RGB_linear_final = RGB_linear_source_response * mask_gain * 0.75
```

The global reduction is applied in linear RGB before sRGB encoding and noise. Relative to the source response, the center has gain 0.75 and the outer region has gain **0.02625 = 0.035 × 0.75**. The displayed 8-bit values are not multiplied directly by 0.75.

The original response multiplier of 16 becomes **12**, equivalent to **+3.5849625007 EV**. `applied_parameters.json` stores effective RGB scales that already include the global multiplier; do not apply it a second time.

The final processing order is: cached native-depth-of-field HDR and distortion mapping → existing field blur → fixed source RGB response → vignette → global ×0.75 → sRGB → existing signal-dependent noise with fixed seeds → PNG.

These optics, blur, noise and vignette settings are recorded simulation choices. The user accepted the visual result; they are not claimed to be manufacturer specifications or a calibration of the target real camera. Intrinsics/distortion and reference stereo geometry come from the DRENDS ExVivo profile, which is not verified as the calibration of the target recording.

Optical coordinates use X right, Y down, Z forward. The stored reference relation is `X_right = R_right_from_left @ X_left + t_right_from_left_m`, with translations in meters. The reference baseline is approximately **5.5300843135 mm**.

The example is a static synchronized pair. It does not establish continuous capture FPS, pose-estimation accuracy, real-time inference performance or calibrated ECM-to-camera mounting.

[Complete applied parameters](../config/applied_parameters.json) · [Intrinsics and distortion](../config/camera_intrinsics.json) · [Sources](SOURCES.md)
