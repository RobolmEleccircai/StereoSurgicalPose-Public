# Single-instrument temporal stereo capture

One command renders the complete **2,100 stereo pairs**: 14 continuous sequences of 150 frames, at **60 Hz**, with **1280 × 720 pixels per eye**. Every category includes instrument motion and jaw articulation. There is no separate closed-jaw phase.

## Timing

- Each left/right pair uses exactly the same frozen scene state.
- Frame `i` has timestamp `i / 60` seconds. Adjacent states are 16.666667 ms apart.
- Each sequence contains 150 frames, with timestamps 0 through 2.483333 seconds; its video plays for 2.5 seconds at 60 fps.
- Motion and jaw velocity are evaluated using this time interval. Frames follow a smooth continuous trajectory, rather than independent random poses.
- Offline path-tracing duration is independent of the represented video frame rate. The separate model throughput requirement remains at least 30 predictions per second.
- Motion blur and shutter integration are disabled. These are instantaneous rendered states; 60 Hz does not specify a physical exposure time. The copied historical `appearance_source.json` is provenance only: its old 30 Hz timing is not used by this capture.

## Run

On the configured Ubuntu computer:

```bash
cd /home/leorobot/PoseProject/CameraImagingConfig/TemporalStereo_2026-11-07
bash run_all.sh
```

`prepare.py` checks all trajectory states before rendering. `capture.py` renders every pair, preserving verified completed frames when resumed. `finalize.py` checks all outputs and creates 60 fps stereo preview videos. Only a successful final validation sets `dataset/status.json` to `complete`.

The external scene/assets and Isaac Sim 5.1 installation must already exist at the paths in `config.json`. Collision checking additionally uses `python-fcl==0.7.0.11`, installed in this new folder's `.deps/`. Original scenes, assets and the older project folders are not rewritten.

## Variation in one complete run

Each row contains two sequences (300 stereo pairs). All rows also vary the PSM's position, orientation and jaw angle.

| Category | Additional variation | Purpose |
|---|---|---|
| `pose_motion` | Continuous PSM joint motion | Position, direction and viewpoint coverage |
| `lighting` | Shared light intensity multiplier 0.7–1.3, smoothly varying | Brightness and highlight variation |
| `sensor_quality` | Noise multiplier 0.5–1.5; blur multiplier 0.75–1.25 | Sensitivity to image degradation |
| `background_edge` | Relative camera/tool movement against existing anatomy | Background and anatomical edge variation; no guaranteed occlusion fraction |
| `ecm_motion` | Rigid movement of the existing ECM/camera assembly | Continuous camera viewpoint variation |
| `stereo_rig` | Baseline multiplier 0.95–1.05, relative yaw offset ±0.5° | Camera relationship variation between sequences |
| `combined_pose_jaw` | Tool motion, camera movement, illumination and opening/closing together | Combined conditions, including fully closed/open endpoints |

Within each sequence the stereo relationship is fixed; moving the ECM moves both eyes together. ECM movement here is a rigid transform of the assembly, not a validated ECM joint trajectory.

PSM trajectories use the official USD joint frames and limits. Sampled offsets around the existing pose are yaw/pitch ±3°, insertion ±10 mm, roll ±25°, wrist pitch ±15° and wrist yaw ±8°; endpoint differences are at most half these spans. Camera movement endpoints use initial translations ±4 mm and rotations ±2°, followed by changes up to ±3 mm and ±1°. These sampling ranges are dataset design choices, not published hardware tolerances.

All 150 states in each candidate sequence must retain C, T and both jaw endpoints in each image, with a projected C–T distance of at least 70 pixels. Distal tool meshes are checked for triangle intersection against nearby anatomy. This does not validate contact forces, deformable tissue or every robot self-collision. The renderer also requires a visible instrument mask. Train/validation/test split is by sequence (10/2/2); do not randomly split neighboring frames.

## Jaw constraints and sources

The official asset has joint limits **−30° to 0°** and **0° to 30°**. Targets are symmetric `(-alpha, +alpha)`, with each jaw's angle from the centerline `alpha` between **0° and 30°** (total opening 0°–60°). Jaw speed is limited to **0.2 rad/s**. A single sequence changes its half-angle by at most 15°, with smooth endpoint velocities.

- [Manufacturer instrument catalog, page 27](https://dvrk.lcsr.jhu.edu/downloads/manuals/davinci-classic-s-si-instrument-accessory-catalog-08-2011.pdf#page=27): Large Needle Driver 400006/420006, jaw open angle 0°–30°, defined by the diagram as angle from the centerline.
- [ORBIT-Surgical source](https://github.com/orbit-surgical/orbit-surgical): the installed `source/extensions/orbit.surgical.assets/data/Robots/dVRK/PSM/psm_col.usd` supplies joint frames/limits; `source/extensions/orbit.surgical.assets/orbit/surgical/assets/psm.py` specifies gripper actuator `velocity_limit=0.2`. This is a simulation control setting, not a measured hardware speed.
- [dVRK 400006 configuration](https://github.com/jhu-dvrk/sawIntuitiveResearchKit/blob/main/share/tool/LARGE_NEEDLE_DRIVER_400006.json): wider controller command limits are not used as visible mesh opening limits.

`dataset/joint_audit.json` records the actual loaded limits and source scene hash. Motion is applied through the joint graph with dynamics stopped; it is a kinematic dataset, not a physical manipulation trial.

## Images and labels

Each pair includes RGB images, visible instrument masks, and C/d/T in **each eye's optical coordinates** (X right, Y down, Z forward):

- C: fixed shaft reference point at the roll-link origin, in mm.
- d: unit shaft-axis direction along the roll link's +Z axis.
- T: midpoint of the two articulated jaw endpoints, in mm.

These are **18 main scalar values per pair**. With an open jaw, T lies between the tips and is not a physical surface point. Both individual jaw endpoint positions, their projections, jaw angle, joint positions, timestamps and K/D are also saved. No roll angle is implied by the shaft direction alone.

The camera transform and baseline truth are saved separately under `camera_truth/` for evaluation; they are not required model inputs. Yellow overlay lines show the shaft, cyan marks C, magenta marks T, and green/orange mark the two physical jaw endpoints.

## Imaging settings and provenance

- Intrinsics and Brown–Conrady distortion come from the existing `calibration/profiles/drends_exvivo_1080.json` and its recorded sources. For 720p, focal lengths scale by 2/3 and principal points use `c_new=(c_old+0.5)*2/3-0.5`; distortion coefficients remain unchanged. This retains the original field of view.
- The intermediate ideal camera is 1536 × 960, with focal scale 1066.666667 pixels. An inverse distortion map generates the final 1280 × 720 image and mask. Optical starting settings remain 4 mm focal length, f/5.6 and 100 mm focus distance. These optical values are the existing simulation starting values, not measured official dVRK optics.
- PathTracing uses 64 samples per frame, four rendering subframes, total sample setting 256, native depth of field and OptiX denoising. Rendering subframes do not advance the nominal trajectory.
- The accepted elliptical vignette is preserved in normalized image coordinates: center `(639.5,359.5)`, semiaxes `(560,313.333333)` pixels, normalized smooth transition 0.3–1.1, **minimum linear gain 0.035**, followed by the accepted **global linear multiplier 0.75**. Ellipse shape and transition width are unchanged.
- Existing per-eye color response and noise formulas are retained from `CameraRefinement_2026-11-06/InitialImaging_v1/initial_parameters.json`. Noise and field blur are heuristic effects, not a calibrated physical sensor noise model. The shared point light's baseline intensity is 1,000,000 renderer units, with the sequence multiplier applied to it.
- The ECM optical mounting transform remains the existing preview convention, not an official measured hand–eye calibration. This run does not claim to reproduce the actual camera in `grasp5.bag` quantitatively.

`config.json`, `camera_profile.json`, `map_config.json`, `appearance_source.json`, `schedule.json`, per-frame labels and the separate camera truth record the values actually used. `validation.json` checks hashes, frame continuity, jaw limits/speed and projection arithmetic; these checks do not establish real-world pose accuracy.
