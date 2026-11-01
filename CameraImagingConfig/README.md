# CameraImagingConfig

Tested imaging configuration and capture code for the current **single-instrument** scene: right PSM active, left PSM inactive, ECM and stereo cameras retained.

- [Imaging baseline](docs/IMAGING_BASELINE.md)
- [Parameter sources](docs/SOURCES.md)
- [Confirmed baseline JSON](config/confirmed_baseline.json)
- [Complete applied imaging configuration](config/applied_parameters.json)
- [Intrinsics and distortion](config/camera_intrinsics.json)
- [Reference stereo transform](reference/stereo_extrinsics.json)

Reference extrinsics are simulation/evaluation data, not an estimated result. The current preview consists of static synchronized stereo pairs. Large assets, original recordings and render caches are stored separately.

## Code index

| Directory under `src/` | Role |
|---|---|
| `ShaftTipStereo_2026-11-07/` | Current capture, C/d/T definitions, sample poses, projection and dataset packaging |
| `SingleInstrument_2026-11-07/` | Earlier single-instrument scene conversion and preview |
| `CameraRefinement_2026-11-06/` | Sensor effects, approved DarkBorder_v9 processing and parameter records |
| `StereoPreview_2026-11-06/` | Inverse distortion-map builder |
| `calibration/` | Reference calibration, derived profile and provenance |

Start with `ShaftTipStereo_2026-11-07/capture.py`. It explicitly calls `SetActive(False)` for `/SurgicalTools/PSM_Left`. `landmarks.py` defines the reference points; `package_dataset.py` verifies and exports labels and overlays. The three actual render overlays and their manifest are included under `render/native/`.

These are byte-preserved runtime snapshots, not a relocatable installed package. Their absolute Ubuntu paths and external-asset dependencies are documented in [runtime notes](../docs/RUNNING.md). `SHA256SUMS` covers this module's tracked files; the repository-wide source inventory additionally records their original locations.

Current samples use PathTracing at 64 samples/frame (total setting 256), native depth of field, the existing distortion mapping, field blur, response, vignette and noise. The ECM optical mount remains an explicit preview convention, not a measured official hand-eye transform. Parameter records distinguish published calibration from simulation starting values and visually accepted settings.
