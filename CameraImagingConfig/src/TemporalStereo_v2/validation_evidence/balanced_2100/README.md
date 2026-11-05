# Balanced 2,100-pair validation evidence

This directory records the implementation checks for `balanced_2100_v1`. It does not contain, or claim completion of, the full 2,100-pair rendered dataset.

## Completed without new rendering

- `policy_tests.txt`: 10 policy tests, including exact allocation, training coverage, uncompressed source time, independent trajectories, duplicate rejection, incomplete indexes and exposure coverage.
- `geometry_unit_tests.json`: 26 real-asset checks, including missing anatomy, penetrations, containment, inter-frame crossings, legal contact/sliding, official joint limits, transforms and damaged receipts.
- `planned_split_coverage.json`: all 21 independent trajectories passed; 1,400 training, 350 validation and 350 test frames. Includes per-sequence phase/opening/parameter statistics and cross-split coverage.
- `dataset_split_policy.json`, `effective_settings.json`, `run_manifest.json`: actual policy, configuration and source/asset fingerprints for this run.

`planned_safety_summary.json` aggregates the 21 sealed path checks: 3,500 parent samples and 14,131 checked states including adaptive interval subdivision, from which exactly 2,100 output states are selected. The planned insertion joint ranges from 0.05854722061546687 to 0.15629742932459917 metres; no output sample is at the lower limit. These are trajectory validation results, not image or model-accuracy results.

## Calculation modules: existing images

`existing_calculations/decoder_report.json` and `display_report.json` passed on 14 previously generated stereo pairs (28 camera views), without rendering or changing the source data.

The decoder receives 17 values: each eye's shaft-reference position and shaft quaternion (7 per eye), plus shared wrist pitch, wrist yaw and symmetric jaw half-angle. Fixed constants come from the official USD instrument model. Per-frame jaw poses, inter-camera extrinsics and baseline are **not** decoder inputs. Projection additionally uses the corresponding eye's K and distortion coefficients.

The reconstructed quantities checked against saved simulation truth are C, shaft direction d, H1/H2, J1/J2, midpoint T, all three body rotations, projected landmarks, shaft/jaw arrows and body XYZ axes. Maximum differences over those 14 pairs:

| Quantity | Maximum difference |
|---|---:|
| 3D landmark position | 1.6287101518e-6 mm |
| Shaft direction, vector L2 | 1.5144411e-8 |
| Rotation matrix element | 1.1007347154e-7 |
| Landmark/arrow/axis projection | 1.5129281792e-5 pixels |

The approved saved overlays were pixel-identical for all 14 pairs. Nine invalid/stale input cases were rejected. The two-frame, 60 fps recorded-prediction video adapter check also passed. Its source is explicitly marked as ground-truth replay.

`new_2pair_decoder_report.json` separately checks the first two newly rendered balanced samples. It passed with maximum position difference 7.8721501166e-7 mm and projection difference 5.1207061915e-6 pixels. This frozen snapshot was made outside the active run so later capture writes cannot contaminate the check.

These checks demonstrate numerical decoding and display consistency with truth inputs, **not learned model accuracy**. The decoder's finite-precision comparison tolerances (1e-4 mm, 1e-6 rotation/direction, 1e-3 pixels) are distinct from the generator's independent raw-label projection check of less than 1e-6 pixels. They must not be conflated.

## Calculation modules: all planned states

`all_planned_decoder_report.json` extends the check to all **2,100 states / 4,200 eye views** across the 21 checked trajectories, including exact 0° and 30° jaw half-angle limits. `tools/verify_planned_decoding.py` compares the fixed-model decoder with the full USD kinematic chain and the existing label projector. It reads sealed trajectories, does not start SimulationApp, and produces no images.

| Quantity | Maximum difference over all planned states |
|---|---:|
| C, H1/H2, J1/J2, T | 1.8215313257e-6 mm |
| Shaft direction, vector L2 | 2.2155079979e-8 |
| Rotation matrix element | 1.2453587514e-7 |
| Projected landmark | 1.7535644588e-5 pixels |
| Left/right 3D consistency | 5.7903528317e-13 mm |

The stereo consistency check uses known camera transforms **after** each eye is decoded. These transforms do not enter the decoder. This check covers the planned kinematic range; actual rendered image/mask alignment is checked separately by the bounded capture acceptance.

## Subsequent fixed-tip coverage finding (2026-11-09)

The [independent visibility audit](../balanced_fixed_tip_audit/README.md) found that all 8,400 planned fixed-tip observations lack direct visibility, not just the selected 128 observations. The selected 128 classifications were reproduced independently. The balanced path/acceptance logic does not require positive fixed-tip visibility coverage; it checks visible distal regions instead. The checks below remain valid within their stated scope, but this missing coverage must be addressed before claiming that directly visible fixed endpoints are represented. No production trajectories or images were changed by the audit.

## Bounded rendering acceptance

Run: `balanced_2100_20261108_acceptance`. The authorized selection is 21 sequence samples, seven adjacent samples and four key states, exactly 32 stereo pairs. Full rendered distribution coverage, training performance and target-server rendering are separate checks; this laptop run cannot establish them.

`adjacent_review.json` checks the seven actual adjacent stereo-frame pairs. All preserve a 1/60-second source-time interval. Both masks change in every category (67–1,970 changed pixels per eye), including the camera-motion and contact-path cases. No images were created by this check.

The bounded capture **completed and passed**. `capture_acceptance/report.json` records 32 stereo pairs, all 21 source trajectories, 60 Hz and 1280×720 per eye. The independent raw-label projection error is at most `1.1368683772e-13 px`; left/right position consistency error is at most `3.8016111490e-13 mm`. All seven adjacent-frame checks passed.

Closed and maximum-opening examples use the checked path states at exactly 0° and 30° half-angle. The combined contact keyframe has a 0.0495122086 mm surface gap; the withdrawal keyframe has a 2.0494397373 mm gap and is no longer labelled contact. No pose overrides, quality reductions or additional capture attempts were used.

`capture_acceptance/resume_check.json` records an actual repeated invocation: 32 attempts before and after, zero new attempts, all 224 committed sequence files unchanged, final state `validated`, `dataset_complete: false`.

### Visibility and coverage limits

There are 124 visible distal-region observations out of 128, and **zero directly visible fixed endpoints** in these selected 32 pairs. Therefore this sample does not provide a new positive fixed-endpoint-to-mask alignment test. Region/mask checks, numerical point projection, self/tissue occlusion labels and visual review remain separate evidence. Two pairs have independently flagged overexposure and were retained.

`capture_acceptance/rendered_split_coverage.json` is explicitly a **partial diagnostic sample**. The selected images cannot establish full training/held-out rendered visibility and exposure coverage; full acceptance remains gated on all 2,100 stereo pairs. Missing conditions in this partial report must not be reported as full-dataset coverage success. The complete planned feature/phase/bin checks passed separately before rendering.

## Calculation and display: all 32 new pairs

`new_calculations/report.json` verifies all 64 camera views, with maximum position difference `1.6611531119e-6 mm` and arrow/landmark/axis projection difference `1.5716527741e-5 px`. `new_calculations/display_report.json` checks all 32 approved previews pixel-for-pixel, rejects nine invalid/stale input cases, and passes the two-frame 60 fps video adapter check. The source data remains unchanged and these checks render no new RGB images.

`new_calculations/instrument_model.json` is the fixed model exported from the portable scene asset; `targets_17.json` contains the corresponding 17-value examples. Use this model with these new samples. Its scene hash is checked before comparisons, so constants from another asset version cannot be silently mixed in.

`visual_review.json` records manual checks of representative category and jaw-limit previews. The overall result is bounded script/calculation validation, not a trained-model evaluation or completed full dataset. No server generation was started.

Representative unchanged previews: [closed](examples/seq_015_validation_028.jpg), [maximum opening](examples/seq_000_validation_029.jpg), [contact](examples/seq_018_validation_030.jpg), [tissue occlusion and overexposure](examples/seq_010_validation_010.jpg).
