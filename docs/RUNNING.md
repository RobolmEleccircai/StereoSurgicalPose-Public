# Runtime and backup notes

This backup records the code that produced the three included stereo examples on 2026-11-07. Source files retain their original bytes and absolute runtime paths; cloning alone does not install Isaac Sim or download the external scene assets.

## Existing Ubuntu runtime

- Runtime root: `/home/leorobot/PoseProject/CameraImagingConfig/`.
- Simulator: `/home/leorobot/isaacsim-5.1.0/`, Isaac Sim 5.1.0; Isaac Lab v2.3.2 is also installed.
- Current capture: `ShaftTipStereo_2026-11-07/capture.py`.
- Current post-processing: `ShaftTipStereo_2026-11-07/postprocess.sh` and `package_dataset.py`.
- CPU post-processing Python: `/home/leorobot/anaconda3/envs/grasp5_pixel_gpu/bin/python` with NumPy, SciPy, Pillow and OpenCV.
- Source scene: `/home/leorobot/sufia-bc/scenes/OR_scene_CTLiver-Prostate-Bladder/organs_psm_ecm.usda`, plus its referenced organ textures and ORBIT-Surgical robot assets.
- Required local map cache: `CameraRefinement_2026-11-06/raw_pixel_to_ideal_maps.npz`, with its accompanying `map_config.json`.

`CameraImagingConfig/src/` mirrors selected paths relative to the runtime root. Scene overlays reference original Ubuntu asset locations and previous single-instrument scene overlays. Large USD/texture assets, HDR/depth arrays, distortion-map arrays and recordings are intentionally outside Git. Source checksums are recorded in the capture manifest and source inventory; asset sources are listed in [SOURCES.md](../CameraImagingConfig/docs/SOURCES.md).

## Capture interface

With the original runtime and dependencies present, the capture script accepts:

```bash
/home/leorobot/isaacsim-5.1.0/python.sh \
  /home/leorobot/PoseProject/CameraImagingConfig/ShaftTipStereo_2026-11-07/capture.py \
  --out /path/to/a/new/native-output-directory \
  --count 3 --renderer PathTracing --spp 64
```

The output directory must not already exist. This is the script interface, not a claim that arbitrary new output directories work with the unchanged downstream packaging scripts. Those currently target the original `ShaftTipStereo_2026-11-07/render/native`, `render`, `final` and `dataset` paths. `postprocess.sh` takes the running capture PID, waits for completion, then runs sensor effects, DarkBorder_v9 and dataset packaging. Existing successful outputs should be preserved before a future rerun or path refactor.

The current script contains three explicit instrument placements. It creates both eye render products from a frozen stage and saves per-eye truth. Resolution is fixed at 1920 × 1080, with a 2240 × 1472 intermediate render. The relative camera pose is fixed by the reference profile. Neither the 720p discussion nor multiple camera-rig configurations has been implemented here.

## Verification without rendering

From this repository root:

```bash
python3 scripts/verify_snapshot.py
git diff --check
```

The verifier checks source/sample SHA-256 values, JSON and Python syntax, shell syntax, single-instrument manifest state, six image hashes and dimensions, and C/d/T shape and unit directions. The saved `checks.json` additionally records the independent OpenCV projection checks performed during dataset packaging. The backup verification does not launch the simulator or certify physical-camera accuracy.

## Historical calibration dependency

The calibration experiment imports the earlier local algorithm file `/home/leorobot/capstone-1/Grasp5-Stereo-SurgPose/pixel_extraction/geometry.py` (recorded SHA-256 `7c35957eb7784dd5307819625591645aff7260835339beb8f45aa59943adef8c`). This earlier project and the original `PoseProject/StereoCalibration` and `PoseProject/SurgicalPoseEstimation` folders are left untouched. The diagnostics are archived separately from the current capture entry point.
