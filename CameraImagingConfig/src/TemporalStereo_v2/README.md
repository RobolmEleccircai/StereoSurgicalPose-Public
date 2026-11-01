# Temporal stereo capture v2

Single official PSM instrument with an articulated ECM stereo rig. The default dataset is 14 sequences × 150 synchronized pairs at 60 Hz, 1280 × 720 per eye. The laptop is a validation machine; it must not generate the complete dataset.

## Modes and setup

Copy `machine.example.json` to an ignored `machine.local.json` and set the installation, asset, dependency and output directories. Paths may be absolute, home-relative, or relative to the machine file. No username is embedded in the implementation. Use Isaac Sim **5.1.0**; the launcher discovers its USD extension and checks its version.

Install the extra collision binding into an isolated directory using Isaac Sim Python:

```bash
"$ISAAC_ROOT/python.sh" -m pip install --no-deps --target .deps -r requirements-collision.txt
```

The entry point is run with ordinary Python; it starts the correct Isaac Sim Python worker:

```bash
# Geometry and failure-injection tests; zero renders.
python3 generate.py --machine machine.local.json --tests-only --run-id unit_tests

# All trajectory and swept-motion checks; zero renders.
python3 generate.py --mode plan --machine machine.local.json --run-id geometry_check

# Default mode: at most 32 render attempts on the laptop.
python3 generate.py --machine machine.local.json --run-id validation_v2
```

`generate` is rejected when the machine role is `laptop`. Validation's budget is stored outside individual run directories and persists across crashes and new run IDs. Do not change the budget scope to bypass the user-approved cap. Four consecutive frames from each of seven categories plus four jaw/contact keyframes form the 32-pair validation selection. Both jaw-limit keyframes are rendered first, using one safe wrist view. The open keyframe supplies one directly visible physical endpoint per eye; the closed keyframe retains its actual self-occlusion labels. Those keyframes are explicitly marked; they must not be concatenated into a continuous training video.

## Server transfer

Create a new portable asset bundle; original files are never rewritten:

```bash
python3 generate.py --machine machine.local.json --pack-assets /new/path/stereo_assets
```

The bundle flattens the composed USD scene into `scene.usdc`, copies active material/texture dependencies, preserves UDIM groups and custom MDL texture directories, and exports settings/calibration/appearance. Editor thumbnails and unused authoring history are excluded. NVIDIA SDK MDL modules remain provided by the pinned Isaac Sim installation. The bundle contains source hashes and an export dependency audit.

On the server, copy this source directory and the bundle, install the isolated dependency, create a machine configuration with `role: server`, and set `asset_root` to the bundle. Then run:

```bash
python3 generate.py --mode generate --machine machine.server.json \
  --settings /path/to/stereo_assets/settings.json --run-id dataset_2100
```

One invocation plans, checks, renders and validates all categories. It never silently reduces resolution, rendering quality, frame count or category coverage. Rerun the same command to resume committed pairs. Changed source, parameters, dependencies or trajectories require a new run ID; incompatible results cannot be mixed. A failed run remains explicitly failed. This documentation does not authorize starting a server run.

## Trajectories and safety

Trajectories use official USD joint frames, bounded inverse kinematics and C2 quintic interpolation. PSM and ECM are articulated through their own joint graphs; the ECM root is not moved as a substitute for joint motion. Jaw targets are symmetric, with each half-angle in 0–30 degrees and speed at most 0.2 rad/s. Other angular joints are limited to 1 rad/s; dataset insertion speed is conservatively limited to 10 mm/s. Direct bounded sampling replaces clipped insertion samples.

| Category | Paths |
|---|---|
| Pose | Smooth return and arc motions |
| Lighting | Approach/hover/withdraw and surface contact |
| Sensor quality | Free motion and surface sliding |
| Occlusion | Tissue-clear → tissue-occluded → tissue-clear jaw-tip regions; actual surface/self-occlusion is labeled separately |
| ECM motion | Moving ECM with free motion and sliding |
| Stereo rig | Different fixed relationships between sequences |
| Combined | Contact/sliding and occlusion with articulation and appearance variation |

Contact paths use approximately 20% approach, 20% hold, 40% slide and 20% withdrawal. The approach is 2 mm and the slide is 2–4 mm. Sliding follows a closed, consistently oriented anatomical mesh. These distances, category choices and perturbation ranges are dataset design settings, not manufacturer specifications.

All active anatomy is collected from `/models_topo_blender`. Missing/empty geometry fails. Actual visible PSM surfaces and the existing ECM housing geometry are checked against tissue. Hidden PSM duplicate proxies are not treated as additional anatomy. Surface intersection, component containment and closest distance are checked separately. Explicit structural exceptions cover adjacent/fixed bodies, the two jaws' shared hinge, the telescoping shaft/housing pair and wrist hinge cheeks; they never exclude instrument–organ intersections.

Free movement requires 2 mm tissue clearance. Geometric contact uses a target gap of 0.05 mm and a 0–0.1 mm classification band. This is an approximately touching, undeformed surface configuration, **not validated contact forces or soft-tissue dynamics**. The physical jaw mesh determines clearance; the virtual midpoint T does not. Vessels and back muscles with open topology remain surface obstacles but cannot be selected as contact targets.

Frame intervals are checked using conservative articulated point-motion bounds and mesh separation. Uncertified intervals are subdivided; unresolved intervals or any intersection are rejected. Closed-organ containment catches a tool entirely inside tissue even when surfaces are not intersecting. There is no skipped-frame repair that would create discontinuous motion.

## Image formation and labels

Each output includes RGB PNGs, visible tool masks, per-eye C/d/T (18 main numbers), both physical jaw endpoints, K/D, joint positions and velocities, timestamps, contact state and physical endpoint visibility. C is the internal roll-link origin; d follows the shaft +Z axis; T is the midpoint of the two actual jaw endpoints. C and T are not assigned fictitious surface-visibility labels. Direction alone does not specify axial roll or a complete 6D rotation.

Pose coordinates use optical X right/Y down/Z forward, positions in millimeters and unit directions. Joint positions use radians for revolute joints and meters for prismatic insertion; joint velocities use rad/s and m/s respectively. The separate jaw half-angle field is in degrees. Camera transforms and baseline are separate evaluation records, not required model inputs. Each left/right pair is rendered from one frozen state. Timestamps advance by 1/60 second, irrespective of offline rendering time. There is no shutter integration or motion blur.

The existing DRENDS ExVivo K/D calibration is retained. At 720p, focal lengths scale by 2/3, principal points by `(c+0.5)*2/3-0.5`, and distortion coefficients remain unchanged. Native ideal rendering is inverse-mapped into the distorted output. PathTracing remains 64 samples/frame and four rendering subframes (total setting 256), with native depth of field. Existing 4 mm, f/5.6, 100 mm focus settings are simulation starting values, not measured dVRK optics.

The accepted vignette is unchanged in normalized image coordinates: center `(639.5,359.5)`, semiaxes `(560,313.333333)` output pixels, smooth transition 0.3–1.1, minimum linear gain 0.035, global linear multiplier 0.75. Existing color response, field blur and heuristic sensor noise are retained. Light intensity variation is 0.7–1.3, noise multiplier 0.5–1.5, blur multiplier 0.75–1.25. Stereo perturbations are baseline ×0.95–1.05 and relative yaw ±0.5 degrees, fixed within a sequence. These are documented augmentation choices.

The fixed postprocessing values are also explicit: field-blur sigma is `(width/1920) * blur_multiplier * sqrt(0.3² + (0.8*r²)²)` pixels, with `r` normalized from each principal point to the farthest image corner. Seventeen Gaussian levels approximate that spatially varying blur. Color gains are exactly `illumination.combined_rgb_scales[left/right]` from the recorded appearance source. Linear RGB is converted with the standard sRGB piecewise transfer function, then clipped to 0–1. In 8-bit pixel units, independent channel noise has standard deviation `noise_multiplier * sqrt(1 + 6*luminance)` and the shared row term has standard deviation `0.3 * noise_multiplier`. These retained processing values are simulation design choices, not a measured sensor-noise calibration.

Source calibration and effective capture settings are separate. The inherited ECM optical mounting convention remains an explicit preview assumption; it is not relabeled as an official measured hand–eye calibration. No quantitative match to `grasp5.bag` or real-world pose accuracy is claimed by numerical consistency tests.

## Outputs and acceptance

Each run contains `config/`, `trajectories/`, `sequences/`, `previews/` and `validation/`. A pair is committed only after both images, masks and camera truth exist and a separate label-hash receipt has been written. Every trajectory and the job index also has a sealed hash record. Resume verifies all those files. Train/validation/test split is by complete sequence (10/2/2 for the default 14 sequences).

`validation/report.json` checks source preservation, output hashes, image sizes, independent distorted projection arithmetic (<1e-6 px), stereo-coordinate consistency, visible endpoint alignment with masks (≤2 px), tissue-occlusion consistency against the visible mask, frame continuity, joint speeds and MP4 frame count/rate. Projection arithmetic is not an image-estimation accuracy measurement. A successful laptop run is `validated` with `dataset_complete: false`; only a fully verified server run can have `dataset_complete: true`.

Occlusion search seeds and bounded excursions are explicit scene-specific sampling design values in `settings.json`. They are revalidated against the current meshes and joint limits; they are not manufacturer parameters. A fixed endpoint may remain self-occluded by its own jaw even when its tip region is clear of tissue. The per-point surface visibility field preserves that distinction.

Native USD camera attributes and endoscope light intensity are included in effective settings. Apertures are derived and recorded in `config/render_map.json`. The uncalibrated light-intensity setting is not a measured lumen/candela claim.

The [completed laptop validation report](validation_evidence/README.md) records 20 passing tests, all 14 checked trajectories, exactly 32 rendered pairs, and an actual resume with zero new render attempts. It includes source/effective camera settings, fingerprints, joint coverage and rendered examples. No full dataset was generated. Near-field tissue saturation in the occlusion sample remains an explicitly documented appearance finding.

## Offline named annotations

`tools/reannotate.py` rebuilds previews using only saved RGB PNGs and label JSON files; it never starts Isaac Sim. It requires Pillow and a Chinese-capable font. Saved images and labels remain unchanged and are checked before/after processing. Output names C, T, J1 and J2 explicitly, uses a positive-d shaft arrow starting exactly at C and ending at C+15d (millimeters), and distinguishes visible endpoints, self-occlusion and tissue occlusion. The 15 mm length is solely a display choice. T is only named as the midpoint of the two physical tips, with no direction arrow. C/T use reference-point diamonds without a surface-visibility claim. The arrow endpoint is not a physical landmark.

```bash
python3 tools/reannotate.py --run /path/to/existing/run \
  --output /path/to/existing/run/previews_labeled \
  --font /path/to/chinese-capable-font.ttf
```

The utility lives outside the capture modules so it does not change the validated simulation-source fingerprint or require rendering the saved dataset again. `annotation_report.json` records the utility hash, unchanged source-file check and generated preview hashes.

## Sources

- [ORBIT-Surgical](https://github.com/orbit-surgical/orbit-surgical): official PSM/ECM asset joint frames and limits; local `psm.py` gripper control speed 0.2 rad/s.
- [Manufacturer instrument catalog, page 27](https://dvrk.lcsr.jhu.edu/downloads/manuals/davinci-classic-s-si-instrument-accessory-catalog-08-2011.pdf#page=27): Large Needle Driver opening angle from centerline 0–30 degrees.
- [DRENDS calibration dataset](https://doi.org/10.5281/zenodo.17598453): published camera calibration; original provenance and hashes remain in the copied source profile.
- [Isaac Sim 5.1 camera documentation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_camera.html): camera and rendering interfaces.
- [FCL](https://github.com/flexible-collision-library/fcl) and [python-fcl](https://github.com/BerkeleyAutomation/python-fcl): geometric distance and collision queries.

- [OpenUSD composition inspection](https://openusd.org/dev/api/class_pcp_prim_index.html): local composition errors used to reject missing USD references and payloads.
