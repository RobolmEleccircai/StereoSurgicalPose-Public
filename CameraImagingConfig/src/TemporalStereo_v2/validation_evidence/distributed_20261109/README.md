# Distributed execution validation — 2026-11-09

This update adds immutable whole-sequence assignment, shard execution, resumable worker preparation, and non-rendering output merging. It does not change trajectories, optical parameters, pose calculations, or occlusion policy.

- **23 unit tests passed** on the Ubuntu Isaac Sim 5.1.0 Python runtime: 13 distributed execution tests and 10 existing dataset-policy tests. The 13 distributed tests also passed with local system Python, without Isaac Sim or NumPy.
- **26 real-asset geometry/calculation checks passed**, including joint limits, contact, interval safety, projection, full local rotations and corruption handling; see [geometry_tests.json](geometry_tests.json).
- **Asset/runtime preflight passed** for the current 21-sequence settings: 76 packaged files, 78 active dependencies, 17 anatomy meshes and 12 moving meshes; see [preflight.json](preflight.json).
- Repository historical source/sample integrity, Python syntax, whitespace and changed documentation links passed.

The dispatch tests verify disjoint 352 / 1,748 jobs, total 1,400 / 350 / 350 split counts, role binding, changed-plan rejection, local exclusive locking, duplicate-run rejection, interrupted preparation recovery, same-job retries, missing/duplicate ownership rejection, and a synthetic 2,100-pair file merge that preserves label bytes. Partial worker outputs cannot claim a complete dataset. Synthetic files are temporary fixtures, not rendered images and not visual acceptance evidence.

**New rendered stereo pairs: 0. Full generation started: no.** No production dispatch was prepared in this change. Actual two-GPU rendering, 4080 throughput and final full-dataset acceptance remain pending.

The active Ubuntu code is in the new directory `/home/leorobot/PoseProject/CameraImagingConfig/TemporalStereo_v2_distributed_20261109`. Old code, images and runs are preserved. Its `machine.local.json` and `settings.distributed.local.json` select the existing assets and current 21-trajectory configuration without modifying the older 14-trajectory asset settings.

Machine commands and transfer details are in [DISTRIBUTED_GENERATION.md](../../DISTRIBUTED_GENERATION.md). The machine-readable summary and exact tested source hash are in [report.json](report.json).
