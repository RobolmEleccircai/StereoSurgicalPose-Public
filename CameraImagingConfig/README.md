# Camera imaging and temporal generation

This module keeps sourced intrinsic/distortion parameters and the continuous scene generator. [Sources](docs/SOURCES.md) distinguish published camera calibration from simulation design values. DRENDS ExVivo parameters describe that experimental setup, not universal dVRK factory calibration.

- `config/camera_intrinsics.json`: original 1920×1080 K/D profile; the generator scales pixel centers to 1280×720.
- `config/applied_parameters.json`: recorded appearance settings.
- `config/confirmed_baseline.json`: vignette contour, minimum linear gain 0.035 and global multiplier 0.75; do not apply the effective gain twice.
- [TemporalStereo_v2](src/TemporalStereo_v2/README.md): planning, bounded validation, generation and overlays.

USD, textures, raw recordings and generated image datasets are external assets. The released code alone is not a downloadable preassembled anatomical scene. No photometric calibration to a physical endoscope is claimed.
