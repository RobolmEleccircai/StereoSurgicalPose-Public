# Continuous stereo generator

Single PSM, articulated ECM, tissue scene; 1280×720 per eye, 60 Hz timestamps, PathTracing. Default `settings.balanced21.json` plans 21 independent trajectories / 2,100 pairs. See [split protocol](../../../docs/DATA_PROTOCOL.md).

Planning and frame timestamps are independent of rendering speed. Source images are raw distorted RGB; per-eye truth is in optical camera coordinates. Contact is a numerical surface-gap approximation without tissue deformation. Occlusion and exposure flags are retained.

## Required external assets

Prepare an asset directory containing `scene.usdc`, `calibration.json`, `appearance.json` and all referenced textures/materials. Calibration follows the supplied profile schema; appearance follows `CameraImagingConfig/config/applied_parameters.json`. A portable bundle can be exported from an already assembled scene using `generate.py --pack-assets`; this command does not obtain or construct the scene for you.

The composed scene must contain `/models_topo_blender`, `/SurgicalTools/PSM_Right/Robot`, `/Endoscope/ECM/Robot` and `/Endoscope/ECM/Robot/ecm_tip_link/StereoPreviewRig/{Left,Right}`. `PSM_Left` must be inactive. Geometry units, joint names/frames and material dependencies are checked by the generator. A newly assembled or changed scene requires re-exporting its fixed instrument model and checking source hashes before model training.

## Commands

Create `machine.local.json` from `machine.example.json` and set your Isaac installation, asset and output paths. For server generation set `role` to `server`; `generate` is refused for `laptop`.

```bash
export ISAAC_ROOT=/opt/isaacsim-5.1.0
"$ISAAC_ROOT/python.sh" -m pip install --no-deps --target .deps -r requirements-collision.txt
python3 generate.py --mode plan --machine machine.local.json --settings settings.balanced21.json --run-id plan_check
python3 generate.py --mode validate --machine machine.local.json --settings settings.balanced21.json --run-id sample_check
# Explicit full generation, only after checking configuration and assets:
python3 generate.py --mode generate --machine machine.local.json --settings settings.balanced21.json --run-id dataset_2100
```

Validation is bounded at 32 attempts. This source release has not run these commands. Resume requires matching code, configuration, assets and trajectory fingerprints. Never change a completed frame's split to fit a training experiment.

`dataset_policy.py` implements the sequence manifest and split checks; `paths.py`/`planner.py` handle continuous motion; `safety.py` checks geometry and swept motion; `capture.py`/`pose_truth.py`/`quality_labels.py` write synchronized data and truth. `dispatch.py` partitions whole trajectories between workers. Dataset generation requires the separately installed Isaac/geometry dependencies; model training uses a separate PyTorch environment.
