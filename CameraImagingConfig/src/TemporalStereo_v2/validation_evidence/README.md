# Laptop validation evidence

The checked implementation's source digest is recorded in `environment_and_readback.json`. Execution uses the portable asset bundle and Isaac Sim 5.1.0 on an RTX 4060 Laptop GPU with 8 GB memory.

## Completed geometry and portability checks

- `unit_tests.json`: 20 passing tests, including missing anatomy/references, surface penetration, fully contained geometry, invalid joint limits, unsafe motion between frames, preserved geometric contact/sliding, corrupt records, and the cumulative laptop render cap. These tests render no images.
- `trajectory_coverage.json`: all 14 complete trajectories passed, covering 2,100 planned frame states and 9,220 total checked states after interval subdivision. There are 490 frame states in the geometric-contact band. Failed candidates were replanned before acceptance; no intermediate frames were removed.
- `portable_runtime_check.json`: portable asset export preserved all checked PSM/ECM joint values and transforms exactly. MP4 encoding/decoding preserved a four-frame 60 fps test stream.
- `environment_and_readback.json`: source and run fingerprints, software versions, dependency audit and independent USD/FK readback checks. The portable dependency audit found no unresolved active assets or absolute authored asset paths.

The earlier implementation and samples were independently checked against their 98 recorded hashes and remained unchanged.

## Render acceptance

Render acceptance is recorded separately in `report.json` when the bounded 32-pair run finishes. Planning success is not a claim that 2,100 image pairs were rendered. The laptop run must retain `dataset_complete: false`; no server generation is started by this validation.

The two jaw-limit keyframes are isolated validation poses, not adjacent temporal training frames. The other clips use consecutive frame indices with a 1/60-second interval. RGB training images contain no overlaid labels; annotated previews are separate files.

The numerical projection tolerance measures agreement of coordinate/projection implementations. It does not measure learned image-based pose accuracy or physical camera calibration accuracy. Surface contact is an undeformed, near-zero-gap geometric approximation and does not claim force validation.
