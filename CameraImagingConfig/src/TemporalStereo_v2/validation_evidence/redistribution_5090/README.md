# 4080 Super + 5090 handover evidence — 2026-11-09

The 4060 task was stopped after 20 committed pairs. Whole trajectories now belong to one of two cloud workers: 4080 Super (850 pairs) or 5090 (1,250 pairs). The final handover preserved 153 existing pairs, including pairs completed during the initial transfer. All imported image and label bytes are unchanged. See [deployment instructions](../../REDISTRIBUTION.md).

Validation performed:

- 28 unit tests passed under the actual Isaac Sim 6 Python environment, including disjoint assignment, repeated import provenance, metadata-version lookup and corruption rejection.
- The isolated 5090 probe rendered four pairs. Steady pair intervals were approximately 19.92, 21.38 and 19.88 seconds. Preview inspection confirmed organs, the instrument and the approved overlays.
- The geometry comparison checked 63 frozen states and 27,421 numerical values, with zero maximum absolute difference. This comparison did not include the separately versioned tip-region surface-sample serialization.
- Region definitions differed by at most 1.7763568394002505e-15 in numeric fields between the original stacks. Both exact JSON files are retained; no hash check was weakened and no image or frame label was rewritten.
- Final running-output checks: **4080 124 pairs** (122 imported + 2 newly rendered); **5090 35 pairs** (31 imported + 4 newly rendered). Their reports verify image/mask/label integrity, original import identity, projection arithmetic and stereo coordinate consistency.
- Both workers continued producing new pairs after these snapshots. The superseded renderer processes were terminated, and the temporary inter-server transfer key was removed.

This is deployment and partial-output evidence, **not completed-dataset acceptance**. Each worker runs full shard verification after rendering; final merging and dataset acceptance remain outstanding until both verified shards are collected. The speed measurements concern these scenes and settings, not a general GPU benchmark.
