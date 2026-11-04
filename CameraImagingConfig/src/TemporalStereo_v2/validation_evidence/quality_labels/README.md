# Independent visibility and exposure labels

This change separates a fixed physical jaw landmark from the visible distal surface around that landmark. It also records overexposure independently of geometric visibility. It does not change the approved image formation settings or replace the original landmark definitions.

## Definitions used in this validation

- **Fixed point:** the existing J1/J2 material point on each jaw, with the original in-frame and occluder classification. `fixed_tip_visibility` and the compatibility field `jaw_visibility` must agree.
- **Distal region:** the last 1.0 mm of the actual jaw visual mesh along its native +Y axis. The region has 128 deterministic, equal-area-stratified surface samples, saved in body-local coordinates. It is visible when at least one in-frame sample is geometrically visible. Counts and fractions are also saved. The denominator is the entire sampled 3D surface patch, including its back surface; it is not an image-area fraction.
- **Mask agreement:** visible surface samples are checked against the rendered visible-tool mask with a 2 px support tolerance. Ray intersections within 0.05 mm of the sample surface are excluded.
- **Overexposure:** at least 1% of saved RGB pixels have all three 8-bit channels at or above 250. This is a documented image-quality heuristic, not a calibrated sensor saturation threshold. Flagged images remain in the dataset and appear in a separate index.

These sampling sizes and thresholds are dataset-label design settings. They are not manufacturer specifications.

## Existing-data regression

[Machine-readable result](saved_data_check.json): 32 existing stereo pairs, 128 per-jaw/per-eye observations.

| Check | Result |
|---|---:|
| Visible fixed points | 2 / 128 |
| Visible distal regions | 112 / 128 |
| Overexposed stereo pairs | 4 / 32 |
| Original files verified unchanged | 224 |
| New renders during this regression | 0 |

The four overexposed pairs are `seq_006/validation_014` through `validation_017`. All 16 distal-region observations in those pairs are tissue-occluded. The other 112 distal regions have visible samples, even though most of their fixed landmarks are self-occluded.

[No-render tests](unit_tests.json): all 26 tests passed, including deterministic sampling on actual jaw triangles, exposure thresholds, geometry and pose checks, and the finite run-specific render allowance.

## Small new capture

The `categories_14` selection checks all 14 planned trajectories and captures only two consecutive frames from each of the seven categories. It preserves 1280×720 per eye, 60 Hz timestamps, PathTracing and the accepted appearance settings. The new run is capped at 14 stereo render attempts. The original 32 attempts remain in the laptop ledger. Full dataset generation remains disabled on the laptop.

Run `quality_smoke_20261108` completed successfully with runtime code commit `66fe102`. The generator produced 14 new PathTracing stereo pairs; the independent verifier then marked the run `validated`, with `dataset_complete: false`. The renderer shut down after completion. The full 2,100-pair dataset was not generated.

| Check | Result |
|---|---:|
| Categories / new pairs | 7 / 14, exactly 2 consecutive frames per category |
| Planned trajectories checked | 14, with 9,220 checked states |
| Visible fixed points | 0 / 56 observations |
| Visible distal regions | 48 / 56 observations |
| Overexposed pairs | 2 / 14 |
| Maximum independent projection arithmetic difference | 1.137 × 10⁻¹³ px |
| Maximum cross-camera position consistency difference | 3.909 × 10⁻¹³ mm |
| Complete local-copy hash checks | 98 |
| Additional render attempts / unique jobs | 14 / 14, no retries |
| Laptop ledger attempts retained | 32 prior + 14 new = 46 |

An observation here means one jaw in one camera image. The two tissue-occluded pairs account for all eight non-visible distal-region observations and are also the two overexposed pairs. All other 48 distal regions have geometrically visible samples supported by the rendered tool masks.

The small selection uses frames 75 and 76 at 1.25 s and 1.266666667 s. It does not include the isolated maximum-opening keyframe that supplied the two visible fixed-point observations in the original 32-pair regression. Therefore this new run does **not** claim a new positive fixed-point/mask alignment test: its report correctly records `visible_tip_checks: 0`. It does verify regional mask support, full body/camera transformations, image sizes and hashes, exposure statistics, and nonzero mask changes in every two-frame clip. The numerical projection results are consistency checks, not visual pose-estimation accuracy.

Both final-category frames are `contact_slide`, with nearest surface gaps of 0.0502428 mm and 0.0502403 mm. Their per-jaw half-opening angles are 4.0503° and 4.1510°. All four per-eye jaw regions are visible in each pair; neither pair is overexposed. Contact remains the configured nonpenetrating geometric approximation, without tissue deformation or force simulation.

Evidence:

- [Independent capture verification](new_capture_report.json)
- [Visibility and exposure by frame](new_visibility_and_exposure.json)
- [Overexposed frame index](new_overexposed_frames.json)
- [Capture and local-copy summary](new_capture_summary.json)
- [Code, settings and region hashes](new_capture_provenance.json)
- [Preserved finite render budget](render_budget_evidence.json)

New capture previews:

- [Fixed points self-occluded, distal regions visible](examples/visible_region_self_occluded_point.jpg)
- [Tissue-occluded regions and independently flagged overexposure](examples/tissue_occluded_overexposed.jpg)
- [Contact sliding with visible distal regions](examples/contact_slide_visible_regions.jpg)

The preview captions about no re-rendering refer to the annotation step: overlays are drawn from the saved raw images and truth. All 14 raw stereo pairs in this run were newly captured by the modified generator. No image brightness or rendering-quality setting was adjusted to change the exposure classification.
