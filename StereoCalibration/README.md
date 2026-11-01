# StereoCalibration

Historical calibration diagnostics from the single-instrument jaw/ring experiment. These scripts are retained for traceability; they are not a prerequisite for the current per-camera C/d/T capture.

`experiments/JawRingBaseline_2026-11-07/` preserves endpoint extraction, CAD measurement, relative-pose estimation, error separation and solver audits with their JSON results.

The reference baseline was **5.5300843135 mm**. The image-derived pipeline returned **11.6185009637 mm** (approximately **110.096% error**), so this experiment did **not** validate reliable image-based calibration.

| Inputs used in the diagnostic | Baseline (mm) | Approximate relative error |
|---|---:|---:|
| True endpoints + true R/translation direction | 5.530084334 | 0% |
| True endpoints + estimated R/translation direction | 10.386500586 | 87.818% |
| Detected endpoints + true R/translation direction | 6.671252336 | 20.636% |
| Detected endpoints + estimated R/translation direction | 11.618500964 | 110.096% |

The audits isolate both endpoint error and instability in relative-pose estimation. Reports with oracle geometry are diagnostic controls, not claims that the estimated camera relationship was correct. See `separation_check/result.json` and `relative_pose_audit/conclusion.json`.

The copied scripts retain dependencies on the earlier local algorithm at `/home/leorobot/capstone-1/Grasp5-Stereo-SurgPose/pixel_extraction/geometry.py` and external source-run assets/caches. That earlier project is not modified or copied into Isaac Sim by this backup. These experimental scripts are not standalone after cloning; consult [runtime notes](../docs/RUNNING.md).
