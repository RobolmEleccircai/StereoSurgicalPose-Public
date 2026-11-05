# CameraImagingConfig

Imaging configuration and capture code for the **single-instrument** scene: right PSM active, left PSM inactive, ECM and stereo cameras retained. The current entry point is [TemporalStereo_v2](src/TemporalStereo_v2/README.md): 21 independent trajectories, 2,100 stereo pairs, 1280×720 per eye and 60 Hz sampling timestamps.

- [Current dataset allocation and configuration](src/TemporalStereo_v2/DATASET_SPLITS.md)
- [Server migration, dependencies and launch commands](src/TemporalStereo_v2/SERVER_MIGRATION.md)
- [Model input/output and numerical decoding](src/TemporalStereo_v2/MODEL_INPUT_OUTPUT.md)
- [Shared pose calculation and overlays](src/TemporalStereo_v2/pose_display/README.md)
- [Completed bounded validation](src/TemporalStereo_v2/validation_evidence/balanced_2100/README.md)

Occluded samples retain their images, 3D truth and separate fixed-point/region visibility labels. Directly visible fixed endpoints are not an additional acceptance requirement.

## Imaging parameters and sources

- [Imaging baseline](docs/IMAGING_BASELINE.md)
- [Parameter sources](docs/SOURCES.md)
- [Confirmed baseline JSON](config/confirmed_baseline.json)
- [Complete applied imaging configuration](config/applied_parameters.json)
- [Intrinsics and distortion](config/camera_intrinsics.json)
- [Reference stereo transform](reference/stereo_extrinsics.json)

Reference extrinsics are simulation/evaluation data, not an estimated result. The original static previews and camera-reference records below remain available. Large assets, original recordings and render caches are stored separately.

## Code index

| Directory under `src/` | Role |
|---|---|
| `TemporalStereo_v2/` | Current trajectory planning, batch capture, labels, calculation/display tools, validation and portable deployment |
| `ShaftTipStereo_2026-11-07/` | Historical three-pair static capture, C/d/T definitions and original projection checks |
| `SingleInstrument_2026-11-07/` | Earlier single-instrument scene conversion and preview |
| `CameraRefinement_2026-11-06/` | Sensor effects, approved DarkBorder_v9 processing and parameter records |
| `StereoPreview_2026-11-06/` | Inverse distortion-map builder |
| `calibration/` | Reference calibration, derived profile and provenance |

For current work start with `TemporalStereo_v2/generate.py`; its README documents `plan`, `validate` and `generate`. For the earlier static capture, `ShaftTipStereo_2026-11-07/capture.py` disables `/SurgicalTools/PSM_Left`; its original render overlays and manifest remain under `render/native/`.

The dated legacy directories are byte-preserved runtime snapshots. Their original absolute paths are recorded in [historical runtime notes](../docs/RUNNING.md#historical-2026-11-07-snapshot). The current v2 entry point uses a portable machine configuration and asset bundle. `SHA256SUMS` covers the listed camera-reference and historical snapshot files; v2 runs and transfer bundles have their own code, configuration and asset manifests.

Current samples use PathTracing at 64 samples/frame (total setting 256), native depth of field, the existing distortion mapping, field blur, response, vignette and noise. The ECM optical mount remains an explicit preview convention, not a measured official hand-eye transform. Parameter records distinguish published calibration from simulation starting values and visually accepted settings.
