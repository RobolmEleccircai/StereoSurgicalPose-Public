# Temporal stereo capture v2

Single official PSM instrument with an articulated ECM stereo rig. The default `settings.balanced21.json` specifies 21 independent, variable-length sequences: **1,400 train + 350 validation + 350 test = 2,100 stereo pairs**, at 60 Hz and 1280 × 720 per eye. See [DATASET_SPLITS.md](DATASET_SPLITS.md) for exact lengths, coverage gates, original-time clipping and evaluation indexes. The legacy `settings.json` and historical samples remain unchanged. The laptop is a validation machine; it must not generate the complete dataset.

The [balanced validation report](validation_evidence/balanced_2100/README.md) records 36 passing tests, all 21 checked trajectories, all 2,100 planned-state decoding checks, 32 accepted rendered stereo pairs, and an actual resume with zero new capture attempts. Full rendered dataset coverage and target-server performance remain untested.

## Modes and setup

For reusable pose decoding, the approved arrows/XYZ overlays, and a video prediction callback, see [pose_display/README.md](pose_display/README.md). Saved-data replay and video display run without starting Isaac Sim. Batch previews and prediction overlays share the same renderer; existing capture outputs remain unchanged.

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

`generate` is rejected when the machine role is `laptop`. Validation's budget is stored outside individual run directories and persists across crashes and new run IDs. Do not change the budget scope to bypass the user-approved cap. The balanced policy selects one pair from each of 21 sequences, seven adjacent pairs (one per category), and four actual planned states: closed, 30° half-opening, contact and withdrawal. These 32 samples are diagnostics, not a completed training dataset. Jaw key states come from independent safe trajectories, so their mask difference is not a controlled opening-only experiment. Visibility remains the actual geometric visibility; no fixed endpoint is forced visible.

## Server transfer

See [SERVER_MIGRATION.md](SERVER_MIGRATION.md) for the prepared transfer layout, exact runtime/dependency versions, offline FCL installation, no-render preflight, configuration template and full-generation command. The server preflight does not start SimulationApp or dataset generation.

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

The [model input/output and kinematic decoding specification](MODEL_INPUT_OUTPUT.md) defines the proposed 17-value network output, its existing training-label sources, and the fixed USD geometry required to derive d, H1/H2, J1/J2, T and complete jaw frames. A separate no-render regression reconstructed the 14 historical quality-label stereo pairs without giving the decoder camera extrinsics or saved jaw poses. Current balanced-run evidence is recorded separately in [balanced validation](validation_evidence/balanced_2100/README.md). These are numerical decoding checks, not trained-network accuracy claims.

Each output includes RGB PNGs, visible tool masks, per-eye C/d/T (18 main numbers), both physical jaw endpoints, K/D, joint positions and velocities, timestamps, contact state and physical endpoint visibility. C is the internal roll-link origin; d follows the shaft +Z axis; T is the midpoint of the two actual jaw endpoints. C and T are not assigned fictitious surface-visibility labels. Direction alone does not specify axial roll or a complete 6D rotation.

Balanced captures use label schema 5 (legacy captures retain schema 4), retaining the schema-3 `body_pose_truth` (full PSM body transforms, authored jaw anchors, source scene hash) and `eyes.left/right.local_frames.shaft/jaw1/jaw2.T_optical_from_local_mm`. These are 4×4 transforms with physical origins C/H1/H2. Their rotation columns are the part's local +X/+Y/+Z directions expressed in the corresponding camera's optical coordinates. They are obtained from the same frozen joint state used for RGB, checked against authored USD transforms, and independently checked between eyes. They are not inferred from silhouette pixels or from d alone. Earlier samples remain unchanged; their saved joint truth and asset version permit offline recovery.

### Separate fixed-point, distal-region and exposure labels

Each eye now has three independent fields:

| Field | Meaning |
|---|---|
| `fixed_tip_visibility` | Original J1/J2 material-point visibility, occluder identity and in-frame state. `jaw_visibility` remains an identical compatibility alias. |
| `jaw_tip_region_visibility` | Per-jaw distal surface visibility, sample-state counts and fractions, and visible-mask support. |
| `image_quality` | Raw RGB near-white fraction, separate instrument-mask fraction and explicit overexposure flag. |

The distal region is the last **1.0 mm** along the native jaw body's +Y axis, clipped from the actual visual triangle mesh. **128 deterministic equal-area-stratified samples** lie on this fixed surface patch and move with the jaw. They are not selected from the camera-facing silhouette. `region_visible` means at least one in-frame sampled surface point is geometrically visible; the count/fraction records how many. The denominator covers the entire 3D patch, including its naturally hidden back surface, and is not a fraction of image pixels. Every geometrically visible patch must have visible-instrument mask support; sample support uses a 2 px tolerance. Ray hits within 0.05 mm of their target surface point are excluded, consistently with the existing endpoint test. The exact body-local samples and hashes are saved in `config/tip_regions.json` and referenced by every new committed label. This patch size, sampling count and tolerances are recorded dataset-label design settings, not manufacturer parameters or a claim of continuous exact surface visibility.

The `overexposed` flag is a documented RGB heuristic: **at least 1% of saved image pixels have R, G and B all ≥250** on an 8-bit scale. This is not a measured sensor saturation threshold. `quality_flags.overexposed_any_eye` summarizes the pair. All flagged images are retained; `validation/overexposed_frames.json` is a separate index. Exposure labels do not change fixed-point visibility, region visibility, rendering parameters or training targets. Approved XYZ panels keep their positions and transparency; preview footers display the region and exposure fields separately.

`tools/annotate_saved_quality.py` derives sidecars and previews from existing RGB, masks, joint states and USD geometry without starting SimulationApp. It preserves all original images and labels:

```bash
python3 tools/annotate_saved_quality.py --machine machine.local.json \
  --settings /path/to/existing/run/config/effective_settings.json \
  --run /path/to/existing/run --output /path/to/new/quality_review
```

The 32-pair regression retained **2/128 visible fixed points**, found **112/128 visible distal regions**, and separately flagged the four seq_006 overexposed pairs. All 224 original data files remained unchanged. The 26 no-render tests include actual surface sampling, exposure thresholds, full pose truth, safety and the finite render-authorization extension.

The historical `validation_selection: "categories_14"` profile selects **seven categories × two adjacent frames**, with a 14-pair invocation cap. It is retained for legacy regression. The current balanced policy requires `balanced_32`: 21 sequence samples, seven adjacent samples and four jaw/contact key states, all read directly from the checked paths without isolated pose overrides. Full `generate` remains forbidden on laptops.

The laptop budget ledger is never reset. A later explicit user request for additional renders is recorded as an `additional_authorizations` entry bound to one run ID and a finite attempt count; attempts retain their authorization ID. The November 8 quality-label validation uses 14 additional attempts after the original 32. The extension cannot be reused by another run or replenish itself after a crash. No code path automatically grants more attempts.

The [completed quality-label validation](validation_evidence/quality_labels/README.md) records the existing-data regression and a successful new 14-pair capture covering all seven categories. The new run has 48/56 visible distal-region observations, 0/56 visible fixed landmarks, and two independently flagged overexposed pairs. Its full-pose, temporal, image/mask, exposure and projection checks passed. The small selection does not repeat the original isolated jaw-limit keyframes or claim a new positive fixed-landmark visibility check. The laptop rendered exactly 14 additional pairs and did not generate the full dataset.

Pose coordinates use optical X right/Y down/Z forward, positions in millimeters and unit directions. Joint positions use radians for revolute joints and meters for prismatic insertion; joint velocities use rad/s and m/s respectively. The separate jaw half-angle field is in degrees. Camera transforms and baseline are separate evaluation records, not required model inputs. Each left/right pair is rendered from one frozen state. Timestamps advance by 1/60 second, irrespective of offline rendering time. There is no shutter integration or motion blur.

The existing DRENDS ExVivo K/D calibration is retained. At 720p, focal lengths scale by 2/3, principal points by `(c+0.5)*2/3-0.5`, and distortion coefficients remain unchanged. Native ideal rendering is inverse-mapped into the distorted output. PathTracing remains 64 samples/frame and four rendering subframes (total setting 256), with native depth of field. Existing 4 mm, f/5.6, 100 mm focus settings are simulation starting values, not measured dVRK optics.

The accepted vignette is unchanged in normalized image coordinates: center `(639.5,359.5)`, semiaxes `(560,313.333333)` output pixels, smooth transition 0.3–1.1, minimum linear gain 0.035, global linear multiplier 0.75. Existing color response, field blur and heuristic sensor noise are retained. Light intensity variation is 0.7–1.3, noise multiplier 0.5–1.5, blur multiplier 0.75–1.25. Stereo perturbations are baseline ×0.95–1.05 and relative yaw ±0.5 degrees, fixed within a sequence. These are documented augmentation choices.

The fixed postprocessing values are also explicit: field-blur sigma is `(width/1920) * blur_multiplier * sqrt(0.3² + (0.8*r²)²)` pixels, with `r` normalized from each principal point to the farthest image corner. Seventeen Gaussian levels approximate that spatially varying blur. Color gains are exactly `illumination.combined_rgb_scales[left/right]` from the recorded appearance source. Linear RGB is converted with the standard sRGB piecewise transfer function, then clipped to 0–1. In 8-bit pixel units, independent channel noise has standard deviation `noise_multiplier * sqrt(1 + 6*luminance)` and the shared row term has standard deviation `0.3 * noise_multiplier`. These retained processing values are simulation design choices, not a measured sensor-noise calibration.

Source calibration and effective capture settings are separate. The inherited ECM optical mounting convention remains an explicit preview assumption; it is not relabeled as an official measured hand–eye calibration. No quantitative match to `grasp5.bag` or real-world pose accuracy is claimed by numerical consistency tests.

## Outputs and acceptance

Each run contains `config/`, `trajectories/`, `sequences/`, `previews/` and `validation/`. A pair is committed only after both images, masks and camera truth exist and a separate label-hash receipt has been written. Every trajectory and the job index also has a sealed hash record. Resume verifies all those files. Train/validation/test use seven independent motion groups each, with 1,400/350/350 frames; the three full planned indexes explicitly mark whether each pair has been rendered and verified.

`validation/report.json` checks source preservation, output hashes, image sizes, independent distorted projection arithmetic (<1e-6 px), stereo-coordinate consistency, visible endpoint alignment with masks (≤2 px), tissue-occlusion consistency against the visible mask, frame continuity, joint speeds and MP4 frame count/rate. Projection arithmetic is not an image-estimation accuracy measurement. A successful laptop run is `validated` with `dataset_complete: false`; only a fully verified server run can have `dataset_complete: true`.

Occlusion search seeds and bounded excursions are explicit scene-specific sampling design values in the effective settings. They are revalidated against the current meshes and joint limits; they are not manufacturer parameters. A fixed endpoint may remain self-occluded by its own jaw even when its tip region is clear of tissue. The per-point surface visibility field preserves that distinction.

Native USD camera attributes and endoscope light intensity are included in effective settings. Apertures are derived and recorded in `config/render_map.json`. The uncalibrated light-intensity setting is not a measured lumen/candela claim.

The [historical laptop validation report](validation_evidence/README.md) records the original 20 passing tests, 14 checked trajectories, 32 rendered pairs, and a resume with zero new render attempts. It includes source/effective camera settings, fingerprints, joint coverage and rendered examples. It is not evidence that the new 21-trajectory policy passed; use the separate [balanced-run report](validation_evidence/balanced_2100/README.md). No full dataset was generated. Near-field tissue saturation in the old occlusion sample remains an explicitly documented appearance finding.

## Offline named annotations

The default display is `--style three-arrows`: yellow from C to C+15d, blue from the physical jaw-1 hinge H1 to tip J1, and orange from the physical jaw-2 hinge H2 to tip J2. Jaw arrows no longer start at C. J1/J2 remain assigned to their original jaw bodies, never reordered by image position. Their camera-frame positions and the saved jaw half-angle are shown independently; the virtual midpoint T is not substituted for either tip. The opening readout gives per-jaw half-angle θ and symmetric opening 2θ. Arrows to occluded tips are dashed and labeled self-occluded or tissue-occluded; arrows to visible tips are solid. This describes tip visibility, not visibility of every point along a virtual arrow. All arrows are sampled in 3D and projected with each eye's K/D. No CAD contours or crosshairs are added. Nearly coincident endpoints or hinges are not moved apart for display. The older midpoint-arrow and point/coordinate diagnostics remain available explicitly with `--style two-arrows` and `--style points`.

The three-arrow preview draws the **instrument's own XYZ frame** in each image’s lower-left orientation panel, with physical origin C: +X red, +Y green and +Z blue-violet. The frame is the official `psm_tool_roll_link` frame. Its +Z coincides with shaft direction d; X/Y follow the body's full rotation, including axial roll. These are not camera-aligned axes, nor axes inferred from d alone. The tool reads existing full body transforms from `RUN/pose_geometry/geometry.json` (or `--pose-geometry`) and computes `T_optical_from_body = inverse(T_world_from_optical) * T_world_from_body`. It checks the saved label hash, proper rotation, origin C and +Z/d agreement, and fails if full pose truth is missing rather than substituting a camera frame. This is the shaft frame at C, not a jaw/wrist-local frame; J1/J2 still show the articulated jaw endpoints.

Each jaw has its own physical XYZ frame at H1/H2, labeled X1/Y1/Z1 and X2/Y2/Z2. For readability, its orientation diagram is shown separately in each eye's **top-left corner for jaw 1 and top-right corner for jaw 2**; the shaft XYZ diagram is displayed in the lower-left corner. The physical yellow shaft-direction arrow remains at C on the instrument. Origins come from the actual USD `psm_tool_gripper1_joint` and `psm_tool_gripper2_joint` anchors, using `physics:body1` and `physics:localPos1` with the body's authored scale converted to millimeters. Parent-side `body0/localPos0` anchors independently verify closure. Frame orientation uses the physical jaw link's native XYZ rotation, translated to the true hinge, so it follows wrist motion and that jaw's opening. It does not use the parent-side joint frame's orientation for both jaws. H1/H2 are internal joint references, not surface landmarks with invented visibility labels.

Corner diagrams apply a **2D translation only** to the already projected shaft/jaw-axis polylines; scale is exactly 1. Their projected directions, relative lengths, foreshortening and distortion are preserved. Each panel is explicitly marked as a displaced display diagram: its screen origin is not a physical landmark or pixel-position ground truth. The true H1/H2 markers and H1→J1 / H2→J2 tip arrows stay at their original image locations. The report keeps physical 3D/pixel geometry in `coordinate_axis_arrows` and records presentation-only translations and paths separately in `jaw_axis_insets` and `shaft_axis_inset`. All three diagrams retain their physical rotations; none represents a new landmark location. The three blue panel backgrounds use color `#111d2a` at 40% opacity (alpha 102/255, 60% transparency); borders, arrows and text remain fully opaque. This blends only the preview overlay, leaving saved RGB unchanged.

`tools/export_jaw_hinges.py` reads the original USD and the existing body-pose file to create `RUN/pose_geometry/jaw_hinges.json` (override with `--jaw-hinges`). It records authored joint paths, local transforms, source hashes and parent/child anchor agreement across all saved frames. It uses only USD reading and matrix arithmetic; it never creates a SimulationApp or renders images. The current model has coincident jaw hinges: maximum numerical H1/H2 separation across the 32 saved states is about 0.00000637 mm. Both physical starts are retained exactly. Closed-jaw frames can have coincident orientations, but their corner diagrams remain separately labeled; opening separates their orientations.

Each axis is a true K/D projection of the segment from C to C + L times the corresponding instrument rotation column expressed in the camera frame. One common display length L is used for all axes in both eyes of a pair. L starts at half the shortest original 3D pose-arrow length, then is reduced until every projected coordinate arrow is at most 55% of the shortest original pose arrow in its eye. Perspective foreshortening is preserved. This changes only the annotation overlay, not the original pose arrows, RGB or labels. The report records full pose-file provenance, body-to-camera transforms, 3D axis directions/endpoints, projected paths and display lengths. All of these are saved simulation truth, not model predictions or a reconstruction of full roll from the landmark outputs.

Jaw triads similarly project H1/H2 plus their own rotation columns. Their shared display length is at most half the shortest hinge-to-tip length and is reduced until all six projected axes are at most 55% of their corresponding H-to-J arrow in both eyes. This preserves real perspective before translating the complete diagrams into their separate corners. Tip positions and the shaft-direction arrow are unchanged when the jaw-arrow starts change from C to H1/H2.

XYZ values also remain below each eye: C, J1, J2, H1 and H2 are positions in that eye's optical frame (millimeters); d is a dimensionless unit direction. The numeric camera frame has X right, Y down and Z forward, with position origin at that eye's optical center. These numeric coordinates are distinct from the local axis arrows at C/H1/H2. Separate color-matched rows show the displayed jaw-arrow displacement vectors J1-H1 and J2-H2 in millimeters. Position/displacement display uses two decimal places and direction uses four; projection and saved numeric reports retain full precision.

For landmark supervision, per-eye C, d, J1 and J2 are 12 scalars per eye (24 across both eyes), plus shared jaw half-angle θ. T is an auxiliary derived midpoint. These landmarks alone do not guarantee unique axial roll or full articulation in every configuration. For a model intended to reproduce shaft roll, wrist motion and symmetric jaw opening on this known instrument, the proposed nonredundant physical target is per-eye C and full shaft rotation R, plus shared `psm_tool_pitch_joint`, `psm_tool_yaw_joint` and jaw half-angle. R must represent a proper rotation, not nine unconstrained independent directions. All underlying supervision is now saved; this is a target contract, not a trained model or a claim of solved visual observability.

With the versioned official kinematic chain, C/R and those articulation angles determine d (R's third column), physical H1/H2, both jaw rotations and J1/J2; T is `(J1+J2)/2`. These redundant quantities remain saved for losses, projection checks and evaluation, without requiring independent prediction heads for every derived field. Symmetric current jaw commands are q1=−θ, q2=+θ; independent asymmetric jaw motion is not part of this capture profile. Position errors are measured in mm, orientation/shaft/wrist/opening errors in degrees, and reprojection errors in pixels, with visible/self-occluded/tissue-occluded results separated. Temporal jitter and end-to-end inference throughput must be measured separately; none is demonstrated by ground-truth previews.

Model inputs are synchronized unannotated images (or consecutive clips) and the known per-eye K/D as needed. Relative camera transform/baseline and global poses remain evaluation truth, not mandatory model inputs. Outputs are explicitly camera-relative; driving another robot later additionally needs a mapping to that robot's base/control frame and its inverse kinematics. The historical validation report's 18-scalar C/d/T summary is retained as evidence from that version.

`tools/reannotate.py` rebuilds previews using saved RGB PNGs, label JSON files, saved full body transforms and the exported USD hinge definitions; it never starts Isaac Sim. It requires Pillow, NumPy and a Chinese-capable font. Saved images, labels, pose geometry and hinge definitions remain unchanged and are checked before/after processing. Missing or mismatched hinge provenance is an error; the script does not substitute C or guessed jaw origins. The shaft arrow starts exactly at C and ends at C+15d (millimeters); its length is solely a display choice and its endpoint is not a physical landmark. In the optional `--style points` diagnostic, C/T are reference markers, without a surface-visibility claim; J1/J2 retain their actual saved visibility labels.

The optional point diagnostic lists camera-frame C/T coordinates in millimeters, their image pixels and unit direction d. Every style recomputes distorted projections from 3D truth and saved K/D, checking agreement with existing labels below 1e-6 pixels. Pixel coordinates refer to the original per-eye image, excluding added title/footer margins. All previews identify simulation-ground-truth projections rather than learned model predictions. The annotation report records the numerical projections and maximum discrepancy.

```bash
# Once per saved pose-geometry file, export the physical joint anchors offline.
python3 tools/export_jaw_hinges.py --machine /path/to/machine.json \
  --settings /path/to/existing/run/config/effective_settings.json \
  --pose-geometry /path/to/existing/run/pose_geometry/geometry.json \
  --output /path/to/existing/run/pose_geometry/jaw_hinges.json

python3 tools/reannotate.py --run /path/to/existing/run \
  --output /path/to/existing/run/previews_labeled \
  --font /path/to/chinese-capable-font.ttf
```

The capture pipeline now imports the same `vector_panel` and geometry functions rather than maintaining a second drawing implementation. The annotation source is included in new run fingerprints. Changed capture source requires a new run ID; do not rewrite an old manifest or reset the laptop render budget. Offline reannotation still requires no new RGB rendering. `annotation_report.json` records the utility hash, unchanged source-file check and generated preview hashes.

## Capture-time preview

After each complete stereo pair is committed, `live_preview.py` saves the approved translucent XYZ overlay to `previews/SEQ_FRAME.jpg`, its projection/provenance JSON, and atomically replaces `previews/latest.jpg` and `latest.json`. `previews/live.html` reloads the latest completed pair every two seconds. Sequence/frame identifiers and numeric values are embedded in the same stereo preview. This is progress monitoring at completed-render speed, not 60 Hz rendering or inference. The raw training PNGs never contain these overlays. Interrupted/missing previews are rebuilt from committed pairs during resume without another RGB render.

Set optional `machine.preview_font` to a Chinese-capable font. If omitted, the publisher checks installed Noto CJK, WenQuanYi and macOS Arial Unicode locations; missing fonts fail before capture starts. Font and annotation hashes are recorded in preview metadata. The shaft XYZ remains at the lower left, jaw1/jaw2 at upper left/right, and all three backgrounds remain 60% transparent.

The page can be opened locally, or served from the run directory using `python3 -m http.server --bind 127.0.0.1 8765 --directory /path/to/run` and viewed at `/previews/live.html` through an SSH tunnel. No server job or HTTP listener is started automatically.

Replay existing pairs through exactly the same publisher without opening Isaac Sim:

```bash
python3 tools/replay_previews.py --run /path/to/existing/run \
  --output /path/to/new/preview_review --font /path/to/chinese-font.ttf
```

The integration replay used all 32 saved pairs and produced byte-identical JPGs to the approved offline overlays with the same font. It also rejected corrupt rotation truth and repaired a missing latest preview; source data remained unchanged. This does not claim a new GPU capture or server full-dataset acceptance. The earlier laptop verification and its coverage limitations remain applicable: only 2 of 128 selected physical endpoint observations were directly visible, the stereo-configuration variation was narrow, and the existing optics/mount remain documented simulation assumptions. Broader training coverage and real-camera validation remain separate from correct truth export and preview integration.

## CAD pose overlays from saved frames

For direct visual comparison with the tool's shape, `tools/export_pose_geometry.py` reads the original USD visual meshes and the saved joint states. It uses USD and forward kinematics only, never `SimulationApp`, physics stepping or a new RGB render. It verifies C and T against both saved camera-coordinate labels before exporting millimeter geometry and body transforms.

```bash
python3 tools/export_pose_geometry.py --machine machine.local.json \
  --settings /path/to/assets/settings.json --run /path/to/existing/run \
  --output /path/to/existing/run/pose_geometry

python3 tools/pose_geometry_overlay.py --run /path/to/existing/run \
  --geometry /path/to/existing/run/pose_geometry \
  --output /path/to/existing/run/previews_pose_geometry \
  --font /path/to/chinese-capable-font.ttf
```

The second command requires NumPy and Pillow. It projects component silhouette/boundary edges of the actual CAD with each camera's distortion. Cyan lines show the CAD overlay; internal lines are translucent to retain the original appearance. Magenta dashed lines show projected geometry outside the saved visible-instrument mask. The mask sets opacity and display-region classification only; it does not generate, move or fit the CAD edges. This is not a new per-facet visibility annotation. C anchors the yellow positive-d arrow, while T is identified by a named leader without a circle or crosshair.

These diagnostic images use the **full saved joint truth**, in addition to camera calibration. They are not a reconstruction of all articulation from C/d/T alone, and are not model predictions. The learning-target definitions are unchanged. The original RGB, masks and labels remain byte-identical; the overlay report records this check and zero new RGB renders.

## Sources

- [ORBIT-Surgical](https://github.com/orbit-surgical/orbit-surgical): official PSM/ECM asset joint frames and limits; local `psm.py` gripper control speed 0.2 rad/s.
- [Manufacturer instrument catalog, page 27](https://dvrk.lcsr.jhu.edu/downloads/manuals/davinci-classic-s-si-instrument-accessory-catalog-08-2011.pdf#page=27): Large Needle Driver opening angle from centerline 0–30 degrees.
- [DRENDS calibration dataset](https://doi.org/10.5281/zenodo.17598453): published camera calibration; original provenance and hashes remain in the copied source profile.
- [Isaac Sim 5.1 camera documentation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_camera.html): camera and rendering interfaces.
- [FCL](https://github.com/flexible-collision-library/fcl) and [python-fcl](https://github.com/BerkeleyAutomation/python-fcl): geometric distance and collision queries.

- [OpenUSD composition inspection](https://openusd.org/dev/api/class_pcp_prim_index.html): local composition errors used to reject missing USD references and payloads.
