# Moving unfinished work to RTX 4080 Super + RTX 5090

This operation changes worker ownership of whole trajectories. It does not replan motion, change camera perturbations, alter image quality, or increase the 2,100-pair total. A single trajectory has one active owner. Previously completed pairs retain their original images, labels, receipts and rendering provenance.

## Execution order

1. Install and validate the target runtime. Run a small isolated render and record its adapter hash and measured steady pair times.
2. Finish the current pair on each old worker, stop both named processes, and copy only committed pairs into stopped snapshots. Keep the original outputs available for rollback.
3. Use `tools/redistribute.py create` to create a new immutable dispatch. It checks the original plan, hashes all imports, and balances remaining frame counts using measured seconds per pair. Pin the active 4080 trajectory to that worker and the former laptop trajectories to the 5090.
4. Prepare each new worker with `generate.py --mode shard --prepare-only`. Import the completed frames assigned to it with `tools/redistribute.py import`. A transferred frame keeps its old renderer identity, validated against the frozen import ledger.
5. Stop the superseded processes before launching `deployment/runtime6/run_assigned_worker.py`. That launcher checks the local runtime, assets, adapter validation report, run lock and assignment claim. A later restart uses the same command and verifies existing receipts.
6. Run the ordinary shard verification and merge only after both new assignments finish. A partial migrated run is not a completed dataset.

## Commands

Use absolute paths appropriate to the machine. `SDK` is the pinned Isaac Sim runtime; `CODE`, `OLD_DISPATCH`, `NEW_DISPATCH`, `SNAP4060`, `SNAP4080`, `MACHINE` and `RUN` are explicit deployment paths. Configure the USD extension and isolated FCL dependency on `PYTHONPATH` as in the runtime deployment instructions.

```bash
"$SDK/python.sh" "$CODE/tools/redistribute.py" create \
  --dispatch "$OLD_DISPATCH" --output "$NEW_DISPATCH" \
  --source-code "$ORIGINAL_CODE" --dependencies "$GEOMETRY_AUDIT" \
  --sources "$SNAP4060" "$SNAP4080" \
  --seconds-4080 33.33 --seconds-5090 19.92 \
  --pin-4080 seq_000 --pin-5090 seq_003 seq_011 seq_013

python3 "$CODE/generate.py" --mode shard --prepare-only \
  --machine "$MACHINE" --settings "$NEW_DISPATCH/config/effective_settings.json" \
  --dispatch "$NEW_DISPATCH" --worker "$WORKER" --run-id "$RUN_ID"

"$SDK/python.sh" "$CODE/tools/redistribute.py" import \
  --sources "$SNAP4060" "$SNAP4080" --run "$RUN" --worker "$WORKER"

python3 "$CODE/deployment/runtime6/run_assigned_worker.py" \
  --code "$CODE" --machine "$MACHINE" --dispatch "$NEW_DISPATCH" \
  --run "$RUN" --validation-report "$PROBE_REPORT"
```

The timing values above are measurements from this deployment, not GPU specifications or a guarantee for every scene. The scheduler excludes already committed pairs from its remaining-time estimate. No synthetic frame is rendered by the redistribution command.

## Exact metadata across runtimes

The original 5.1 and 6.0 numeric stacks serialize some jaw surface sample coordinates with differences at floating-point roundoff scale. Their JSON byte hashes consequently differ. Replacing one shared `tip_regions.json` would invalidate the older frame receipts even though the original images and numerical labels remain intact.

`capture.py` now archives each exact definition in `config/definition_versions/<sha256>.json`. `io_utils.resolve_tip_definition(label_path, record)` first checks the label’s original relative reference and, if necessary, resolves its exact archived hash. Readers loading this shared metadata should use that helper. There is no approximate-hash acceptance: a missing or corrupted definition still fails. Imports and final merging preserve these files, and all original frame labels and receipts remain byte-identical. Regression tests cover replacement of the current definition, repeated redistribution and archive corruption.

## Provenance and failures

The original plan fingerprint and trajectory bytes are retained. The new dispatch has a distinct execution-code hash and dispatch ID. Its `origin_manifests` preserve the SDK, assets and adapter information of imported data; `imported_frames` binds each label hash to its original worker. Old labels are not rewritten to imply that the new GPU rendered them. A missing or corrupt imported frame fails explicitly rather than silently rerendering it under the old identity.

The two servers use the verified Isaac Sim 6 adapter because the original Isaac Sim 5.1 renderer failed on these cloud drivers. Existing laptop images remain identified as Isaac Sim 5.1 output. The 63-state geometry comparison had a maximum absolute difference of zero over 27,421 numerical values; this checks numerical labels, not pixel equality or real-world accuracy. The adapter revision that prints an exception before Kit shutdown changes error reporting only and is separately hashed.

Neither the split nor the optical settings change: 1,400 / 350 / 350 pairs, 60 Hz timestamps, 1280×720 per eye, path tracing at the existing quality, vignette floor 0.035 and overall linear gain 0.75. Baseline and relative yaw retain the frozen variation; pitch and roll are not newly added during this migration.

## Active deployment (2026-11-09)

- Final dispatch ID: `89cf1b1ed1f1c37bff9020d3cb55b4b747590b45dfe3b35bd325488f40ed1a17`.
- Execution core SHA-256: `901c7c32a2dac337fb3abaab0fe92f402ece28caf2ad09a159d731e6826ec25e`.
- 4080 Super: `seq_000, seq_006, seq_012, seq_018, seq_019`, **850 pairs**.
- 5090: the other 16 whole trajectories, **1,250 pairs**, including all former laptop assignments.
- The final handover imports **122 + 31 = 153** already committed pairs. This includes the original 20 laptop pairs, 114 original 4080 pairs, and 19 cloud pairs completed during handover. Original output directories remain preserved; their superseded processes are stopped.
- Both servers use `/root/autodl-tmp/stereo_surgical/project/code_redistributed_v2_20261109` and `project/dispatch_4080_5090_v2_20261109` below the deployment root.
- Active output directories: `project/outputs/worker_4080_redistributed_v2_20261109` and `project/outputs/worker_5090_redistributed_v2_20261109`.
- Active logs: `operations/worker_4080_redistributed_v2.log` and `operations/worker_5090_redistributed_v2.log`.
- Machine files: `project/config/machine.4080.runtime6.local.json` and `project/config/machine.5090.local.json`.
- Validation reports: `project/checks/runtime6_probe_validation.json` (4080), `project/checks/probe_5090_validation.json` (5090), and each run’s `validation/handover.json`.

The 4080 retains the validated adapter from the first runtime deployment (`65bedfd7…`); the 5090 adapter adds exception reporting before Kit shutdown (`b42bc7d1…`). Neither adapter changes rendering parameters. Exact full hashes are recorded in each runtime manifest. The shared core is identical.

During resume, `status.json.completed_pairs` counts the jobs processed in traversal order; imported frames in a later sequence are counted when that sequence is reached. For an immediate total, count committed labels across the entire run, including those imported later sequences. Do not add the old source-directory count again.
