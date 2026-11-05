# Balanced temporal dataset: 2,100 stereo pairs

`settings.balanced21.json` is the current versioned sampling policy (`balanced_2100_v1`). It replaces sequence-number modulo assignment with an explicit manifest. It does not change camera calibration, optical appearance, image quality, physical limits or the 2,100-pair total. The old `settings.json` and old runs remain available for regression.

## Exact allocation

| Category | Train | Validation | Test | Parent path / held-out clip |
|---|---:|---:|---:|---|
| Position and orientation (`pose_motion`) | 180 | 32 | 32 | Smooth free-space return; moving opening-limit segment |
| Lighting (`lighting`) | 180 | 32 | 32 | Approach, hover, withdraw; includes hover |
| Image quality (`sensor_quality`) | 180 | 32 | 32 | Approach/contact/slide/withdraw; approach and contact |
| Occlusion (`occlusion`) | 240 | 140 | 140 | Tissue-clear → tissue-hidden → tissue-clear, with small ECM motion |
| ECM motion (`ecm_motion`) | 180 | 32 | 32 | PSM arc plus articulated ECM motion |
| Stereo relationship (`stereo_rig`) | 180 | 32 | 32 | Independent free-space return under different fixed rigs |
| Combined (`combined_pose_jaw`) | 260 | 50 | 50 | Articulation, lighting, image quality, ECM, surface slide and withdrawal |
| **Total** | **1,400** | **350** | **350** | **21 independent trajectories; 4,200 raw RGB images** |

Every category also changes position, articulation, jaw opening, light and image conditions. Category names identify the principal evaluation direction, not disjoint single-factor datasets. At 60 Hz, 180 samples represent a 3-second sampled clip, whose first-to-last timestamp span is 179/60 seconds. The shortest clip has 32 samples, so a common temporal input window can contain at most 32 consecutive frames. Never join different sequences into one input window.

## Covered features and independent held-out motion

Training, validation and test have the same per-category feature profile and intensity bins. They use independent random seeds and independently planned PSM paths, poses and within-bin appearance/rig parameters. Training trajectories are planned first. A held-out clip is accepted only if each `(motion phase, jaw-opening class, tissue-occlusion state)` occurs in the corresponding training category. All three splits must contain closed, small, medium and large jaw openings. Closed means half-angle within 1e-6 degrees of zero; other bins are (0,10), [10,20), [20,30] degrees.

Sampling ranges below are **dataset design values authorized for this run, not manufacturer specifications**. Calibration provenance remains in `config/camera_source.json`; actual effective K/D remain in `config/camera_effective.json`.

| Factor | Full range | Low / medium / high bins |
|---|---|---|
| Light multiplier | 0.7–1.3 | Equal thirds: 0.7–0.9 / 0.9–1.1 / 1.1–1.3 |
| Noise multiplier | 0.5–1.5 | Equal thirds |
| Blur multiplier | 0.75–1.25 | Equal thirds |
| Baseline multiplier | 0.95–1.05 | Equal thirds |
| Relative yaw increment | −0.5°–+0.5° | Equal thirds |

Each factor has two low, three medium and two high sequences **in each split**. Different factors use different fixed permutations, recorded in the sequence manifest. The same category keeps the same bin across all three splits. A rig's relative transform is constant within its entire parent trajectory, including when ECM joints move. Distinct sequences vary rig parameters. Light changes smoothly in the selected bin; noise and blur multipliers are per-sequence parameters, with independently seeded pixel noise.

Jaw half-angle remains 0–30°, symmetric, at ≤0.2 rad/s. Per-category opening intervals are 30→24°, 4→9°, 11→17°, 12→16°, 19→13°, 0→6°, and 22→28°; final angles independently vary ±0.5°. Free-space closed/open-limit trajectories hold their initial half-angle for 0.25 seconds, then follow the same speed-checked quintic curve. This supplies actual limit examples in every split without changing the tool's position or creating isolated jumps. All paths pass the existing exact joint bounds, swept-surface safety and real-mesh contact checks.

`trajectory_group_id` hashes the sampled physical PSM motion, ignoring lights, cameras and noise, and canonicalizes time reversal. `feature_profile_id` identifies the category's feature combination. Cross-split near-copy checking additionally compares 33 normalized-time world-space shaft poses and articulated joint states using a reversal-aware discrete Fréchet comparison (1 mm translation, 2° orientation/articulation thresholds). These are recorded duplicate-detection tolerances, not sensor accuracy. Recoloring, adding noise or reversing a trajectory does not make an independent test example.

## Short clips keep their physical time

Training uses its entire planned path. Each validation/test candidate independently plans and safety-checks a 150-frame parent. The selected 32/50/140-frame window is contiguous; unused parent states are stored but not rendered. Rejected candidates are replanned up to the configured finite limit; failure never increases the image count or removes interior frames.

The 32-frame image-quality clips retain at least 10 approach and 10 contact-hold samples. The 50-frame combined clips retain at least 16 sliding and 16 withdrawal samples. These window-selection design thresholds prevent a single phase-boundary frame from being counted as meaningful action coverage; they do not change speeds or image counts.

For every output frame:

- `frame_index` starts at zero, `time_s = frame_index/60`, `timestamp_ns = round(frame_index*1e9/60)`.
- `source_frame_index` and `source_time_s` preserve the parent sample and time. Sequence metadata records `clip_start_frame`, `clip_start_time_s`, output and parent lengths.
- PSM/ECM positions, velocities and image-effect values are copied from the checked parent. Capture checks them against the parent polynomial at **source time**, not shortened output time. No motion or illumination interpolation is redone over the short clip.
- Occlusion clips require a target physical endpoint with at least five tissue-clear frames, fifteen tissue-hidden frames and five tissue-clear recovery frames. Fixed endpoint self-occlusion and distal surface visibility remain separate labels; “tissue-clear” does not mean the fixed point must face the camera.

## Output and evaluation

All existing RGB, masks, C, d, H1/H2, J1/J2, T, full part orientations, physical joint states and K/D are retained. Camera transforms/baseline remain separate evaluation truth. Schema 5 adds split/group/profile identifiers, source time and original clip metadata. Fixed-point visibility, distal-region visibility and overexposure labels remain independent and do not filter images out.

`indices/train.jsonl`, `validation.jsonl` and `test.jsonl` describe every expected full-dataset frame. During planning or a 32-pair diagnostic run they have `rendered_and_verified: false`; `indices/manifest.json` reports verified counts and cannot claim the dataset is complete. Only actual `dataset_frame` receipts populate these indexes as verified. `index.jsonl` in the run root lists the files actually produced by that run, including explicitly identified diagnostic samples.

Each full-dataset row includes `category_equal_weight = 1/(7 × frames_in_that_category_and_split)`. Multiply per-frame errors by these weights and sum to obtain the seven-category equally weighted mean. Also report each category separately, so the 140-frame occlusion clips do not dominate the aggregate.

`validation/planned_split_coverage.json` checks all 21 paths, frame counts, bins, opening states, required feature/state combinations, original-time consistency and cross-split duplicates before rendering. `validation/rendered_split_coverage.json` checks fixed-point visibility, distal-region visibility and overexposure against corresponding training categories/phases. Partial diagnostic coverage is explicitly marked partial; complete acceptance is performed only after all 2,100 pairs exist. A missing training condition in the final rendered set fails acceptance rather than being silently filtered.

## Commands and bounded validation

```bash
# Independent policy tests (Isaac's Python includes NumPy/SciPy).
"$ISAAC_ROOT/python.sh" -m unittest discover -s tests -v

# Real-asset failure injection; no renders.
python3 generate.py --mode plan --tests-only --machine machine.local.json \
  --settings settings.balanced21.json --run-id balanced_unit_checks

# Plan all 21 trajectories; no renders.
python3 generate.py --mode plan --machine machine.local.json \
  --settings settings.balanced21.json --run-id balanced_2100

# At most 32 stereo render attempts, subject to the persistent laptop ledger.
python3 generate.py --mode validate --machine machine.local.json \
  --settings settings.balanced21.json --run-id balanced_2100
```

For relocated assets use the package's `assets/settings.json` explicitly. Validation selects 21 sequence examples, seven adjacent examples (one per category), and closed/maximum-open/contact/withdrawal states: exactly 32 pairs. It checks timestamps, original source states, masks, independent projections below 1e-6 pixels and shared arrow overlays. It does **not** establish full rendered dataset coverage or trained model accuracy.

The completed bounded run and calculation checks are documented in [validation evidence](validation_evidence/balanced_2100/README.md), including the explicitly partial visibility/exposure coverage report and the zero-new-render resume check.

Server-only future command: `python3 generate.py --mode generate --machine machine.server.local.json --settings /path/to/assets/settings.json --run-id balanced_2100`. One invocation produces all 2,100 pairs and final checks. The laptop rejects this mode. Neither this documentation nor packaging starts a server task. The earlier 60–90-hour server budget remains an estimate, pending an actual server benchmark; trajectory count does not increase the image budget.
