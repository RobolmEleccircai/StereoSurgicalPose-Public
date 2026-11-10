# Native stereo inference optimization — step-by-step record

## Scope and reference

Work takes place on the RTX 4060 Laptop GPU, using fixed-checkpoint checkpoint SHA256
`639fc8c7e56263f4ab889aa53cc5132e3610eea4c8b2d8b0db671c41d8a30325`.
The reference is the accepted TensorRT visual frontend plus eager C++ LibTorch,
GPU preprocessing and the three-vector C++ display. Original eager and TensorRT
libraries are retained. No training, new rendering, resolution reduction or
quantization-aware training is part of this work.

Every stage records its purpose, implementation, commands, failures, numerical
checks and complete stereo latency before the next stage begins. Measurements
include preprocessing, inference, output transfer, FK, projection and completed
RGB drawing, excluding disk decoding, camera acquisition, encoding and refresh.
The existing 56 pairs are seven separate eight-frame sequences, not a full-test
accuracy assessment. All three recurrent states persist within each sequence.

## A — Fused C reconstruction and reusable constants

### Why

`position()` reconstructs two C points from predicted image coordinates and depth.
The old eager implementation recomputed K inverse and launched many small tensor
operations for ten Newton iterations every frame. Fixed ROI grids, normalization
constants and image-size vectors were also rebuilt. Local auxiliary heatmaps were
computed despite not contributing to the 17 inference values or recurrent states.

### What changes

- A separate `STEREO_OPTIMIZED` build produces `stereo_libtorch_opt.so`.
- `position_cuda.cu` performs the same inverse-camera multiplication, ten Newton
  steps, determinant epsilon protection and depth scaling in FP64, then stores
  FP32. `--fmad=false` is used; fast math is not enabled.
- `set_calibration` updates FP64 K inverse and D only when the existing camera
  configuration key changes. Existing buffers are updated in place.
- Per-model lazy caches hold guided-region grids/radii, crop offsets, size vectors
  and normalization constants. Reference builds still execute the old expressions.
- Normal optimized inference skips only the wrist/jaw auxiliary heatmaps. Debug
  execution retains them; main landmark heatmaps and segmentation are unchanged.
- `execution_mode=optimized_eager` selects the new library; absent configuration
  selects `reference`. The 17-value public output and state reset rules are unchanged.

### Build attempts and corrections

1. The first launch incorrectly assumed CMake was in the inference Conda environment.
   It was not. The verified system executable is `/usr/bin/cmake`; Python remains the
   existing inference environment's Python 3.11.18. No environment was replaced.
2. NVCC rejected `std::numeric_limits<double>::epsilon()` in a device function.
   It was replaced by the numerically identical compile-time `DBL_EPSILON`, retaining
   multiplication by eight. No relaxed constexpr or reduced precision was introduced.

### Verification protocol

`verify_position_cuda.py` tests source-image edges/interior, each stored camera,
zero and actual distortion, and depths 10/30/100/300/1000 mm against the original
ATen formula. Maximum permitted position disagreement is 0.0001 mm.

`verify_trt_visual.py` now accepts baseline/candidate predictor configurations,
executes the real public predictor and independently checks diagnostic forward.
It reports physical outputs, projected points, internal tensors and ground-truth
errors, with per-category mean/P95 truth error increases limited to 0.005 mm/degree.
A separate comparison against original eager LibTorch retains the established
0.1 mm / 0.1 degree / 0.25 pixel numerical conversion limits.

Status: A completed and accepted; numerical and state checks passed.

### Reproduction commands

Run from `StereoPoseLearning` with the existing inference environment active.
The pinned TensorRT headers and reference engine are prepared by the existing
TensorRT setup described in `TENSORRT_VISUAL.md`.

```bash
python tools/build_optimized_runtime.py --cuda-root /usr/local/cuda-12.8
python tools/verify_position_cuda.py --inputs inputs --output runs/optimization/A_position.json
python tools/verify_runtime_state.py --inputs inputs --baseline-config inputs/predictor_tensorrt.json --candidate-config configs/inference/optimized_eager.example.json --output runs/optimization/A_state.json
python tools/verify_trt_visual.py --inputs inputs --baseline-config inputs/predictor_tensorrt.json --candidate-config configs/inference/optimized_eager.example.json --truth inputs/accuracy_truth.json --output runs/optimization/A_accuracy
python tools/benchmark_realtime.py --inputs inputs --inference-backend libtorch --modes fp32 --predictor-config configs/inference/optimized_eager.example.json --rounds 10 --output runs/optimization/A_performance
```

The predictor paths resolve relative to their JSON file. The example expects the
original checkpoint in `inputs/best_val.pt`. Use a new output directory for each run.

### Correctness results

- 720 point cases across both cameras, zero/actual distortion and five depths:
  maximum FP32 output difference **0 mm**; 0.0001 mm requirement passed.
- All 56 real predictor outputs match the current TensorRT reference exactly;
  C, hinges, tips, shaft rotation, joint angles and projections have zero difference.
- Ground-truth mean errors are unchanged: C **0.158457824 mm**, tips
  **0.197426894 mm**, pitch **0.186402860°**, yaw **0.285754152°**, jaw half-angle
  **0.126455538°**. Per-category mean/P95 requirements passed.
- Continuous state, explicit reset, duplicate timestamps, time gaps, sequence
  changes, changed K/D, invalid image and recovery checks passed. All three hidden
  states are compared, not just the final 17 values.
- The first performance pair was superseded because the state-check process
  might have overlapped its ending. The reported results use the subsequent
  sequential, isolated rerun; the superseded logs are not evidence of speed.

### Isolated performance result and decision

Both runs warmed 56 pairs and measured 560 pairs, with original drawing unchanged.

| Runtime | Mean total | P95 total | Hz |
|---|---:|---:|---:|
| Accepted TensorRT reference | 26.5411 ms | 27.8082 ms | 37.6775 |
| A: fused C + caches + auxiliary pruning | 20.6431 ms | 21.1289 ms | 48.4423 |

Mean complete-pair latency decreased **22.22%**. This stage passes the 10% goal
on this isolated comparison; final acceptance still requires three alternating
baseline/final runs. A separate comparison against original eager LibTorch also
passed the existing conversion tolerances. This result is for the combined A
changes, not a claim that the CUDA kernel alone saved all 5.898 ms.

Machine-readable evidence: `evidence/runtime_optimization_A_4060_20261111.json`.

## B — Fixed buffers and CUDA Graph replay

### Why and implementation

After A, the eager downstream still dispatches many individual ATen operations.
This stage replays the fixed-shape native network with CUDA Graph, leaving GPU
preprocessing and final CPU export outside capture. Python only controls the
existing API and capture; all network operations remain C++ LibTorch/TensorRT.

- The optional optimized TensorRT context keeps three output buffers. Original
  builds retain independently allocated outputs. Debug consumers of optimized
  buffers must finish before the next call.
- `optimized_graph` runs preprocessing and inference on one non-default stream.
  This is for capture ordering, not a claim that changing stream alone is faster.
- Fixed buffers hold images, K, D, history and high-resolution RGB. Before replay,
  the actual current inputs and previous state are copied into these buffers.
- Three warmup passes use zero scratch history. Capture does not advance the
  external predictor history. Each real input causes exactly one replay.
- The existing predictor still owns discontinuity checks and state reset. Camera
  calibration caches update in place, retaining captured addresses.
- A capture exception emits a warning and records `actual_execution=optimized_eager`
  plus the error; it cannot silently count as a successful graph result.

Status: B completed and accepted; actual execution was `optimized_graph`, with no fallback.

### B verification, performance and decision

The 56-pair comparison, all per-category truth checks and all 15 state/reset cases
passed. Output values and the three hidden states matched the reference exactly
in these cases, including changed calibration and recovery after an invalid input.
No failure workaround was needed to capture the network.

With 56 warmup pairs and 560 measured pairs, total mean was **19.7608 ms**,
P95 **20.3987 ms**, or **50.6054 Hz**. The preceding A measurement was
20.6431 ms / 48.4423 Hz. This single-stage measurement supports retaining graph
execution for final repeated testing; it is not an isolated causal attribution
between output-buffer reuse and graph launch savings.

Use the same commands as A with `configs/inference/optimized_graph.example.json`.
Machine-readable evidence: `evidence/runtime_optimization_B_4060_20261111.json`.

## C — Cached panels and two persistent painting workers

### Why and implementation

The original complete-pair drawing costs approximately 5.4 ms and paints the two
eyes serially. The three translucent rounded panels per eye repeat identical
mask/border work on every frame.

A separate `STEREO_OPTIMIZED_OVERLAY` build retains the C ABI and original layout:

- Two persistent workers paint separate halves of the current RGB output. Each
  has its own FreeType/text cache; both finish before `render` returns.
- Rounded fill masks and per-byte fill transforms are cached. Borders are cached
  as exact per-channel input-byte lookup tables derived from the original ordered
  OpenCV border rasterization. This preserves antialias overlap rounding instead
  of approximating the border with a different alpha mask.
- Caches are thread-local and bounded to 16 styles per worker; additional styles
  use the original renderer. Worker destruction releases its caches and font data.
- Existing glyph caching, dynamic numeric formatting, 61-point distorted curves,
  arrow widths/colors and XYZ panel geometry remain unchanged.
- Worker exceptions are collected only after both jobs finish; output is not
  returned as valid after failure. No frame skipping or asynchronous stale output.

The renderer build command is:

```bash
cmake -S reference/pose_display/native -B reference/pose_display/native/build/optimized -DSTEREO_OPTIMIZED_OVERLAY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build reference/pose_display/native/build/optimized -j2
```

The benchmark accepts `--overlay-library reference/pose_display/native/build/optimized/libsurgical_overlay.so`.
CPU process utilization and peak resident memory are now recorded in addition to
GPU telemetry. CPU utilization can exceed 100% when multiple cores are active.

Status: C completed and accepted; cached/parallel drawing preserves all tested pixels.

### C verification, performance and decision

77 comparisons (56 actual pairs plus seven examples each with random, black and
white backgrounds) had **zero changed pixel channels** and identical detailed
geometry. Seven before/after previews were saved outside Git. The compiled
reference renderer remains untouched.

With 56 warmup pairs / 560 measured pairs, drawing mean is **3.8993 ms**;
complete-pair mean **18.4561 ms**, P95 **19.2374 ms**, or **54.1825 Hz**.
Process CPU utilization is **114.7%** (approximately 1.15 logical cores on average)
and peak process RSS **2367 MiB**, including Python/PyTorch, weights, all preloaded
images and renderer caches. This is not GPU memory. Detailed diagnostic paint
CPU times sum the two worker contributions; they must not be added as serial
latency because the workers overlap.

The result supports retaining the cache/worker implementation for final testing.
Machine-readable evidence: `evidence/runtime_optimization_C_4060_20261111.json`.

## D — Two bounded TensorRT candidates

### Why and exact candidate definitions

The accepted engine uses builder optimization level 3 with 53 Conv/MatMul layers
in FP16 and other floating arithmetic constrained to FP32. Two candidates are
built, rather than sweeping arbitrary precision combinations:

1. `candidate_level5`: builder level 5, otherwise identical precision constraints.
2. `candidate_gelu`: level 5 plus FP16 for Conv-adjacent GELU arithmetic only.

The second candidate checks ONNX dependencies: a GELU group must contain Erf and
only Constant/Div/Erf/Add/Mul operations, with exactly one nonconstant external
input produced by Conv. LayerNorm, GRN, reductions and geometric math are excluded.
The builder fails if no eligible activation exists, rather than claiming an
unchanged engine is a new precision experiment. FP32 output boundaries and disabled
TF32 are retained. Requested precision per layer and TensorRT engine-inspector
output are recorded next to each candidate; the original engine is not overwritten.

```bash
python tools/build_trt_visual.py --checkpoint inputs/best_val.pt --output native/libtorch/build/tensorrt/candidate_level5.engine --builder-level 5
python tools/build_trt_visual.py --checkpoint inputs/best_val.pt --output native/libtorch/build/tensorrt/candidate_gelu.engine --builder-level 5 --activation-policy conv_gelu
```

Each candidate receives the same 56-pair truth comparison and complete 560-pair
benchmark as C. Any truth-accuracy violation rejects the candidate; a speed change
within repeat variability is not considered sufficient evidence to replace the
accepted engine. Status: both candidates passed sample accuracy; level 5 with original precision constraints is selected for final repeated testing.

### Final state-regression extension

The final configuration additionally replays 128 consecutive calls using the
existing eight images cyclically, with continuous frame identifiers/timestamps.
This checks long-running state-buffer reuse; it is explicitly synthetic runtime
input, not 128 additional accuracy samples. A deterministic capture rejection
also verifies the warning, recorded fallback mode and correct first prediction.

### D results and selection

| Candidate | FP16 layers | Build time | Mean complete pair | P95 | Hz |
|---|---:|---:|---:|---:|---:|
| Level 5, existing constraints | 53 | 96.34 s | 15.3075 ms | 16.1567 ms | 65.3273 |
| Level 5 + eligible GELU | 123 | 109.25 s | 15.6916 ms | 16.4194 ms | 63.7285 |

Both passed the 56-pair conversion and per-category truth-accuracy checks. Neither
is numerically identical to the previous TensorRT engine: maximum C disagreement
is 0.006277 mm / 0.005205 mm respectively. Actual truth errors, not this numerical
drift alone, determine acceptance; every per-class mean/P95 increase stays within
the specified 0.005 mm/degree limits.

The GELU candidate is slower and brings no reason to relax more precision, so it
is **not selected**. The level-5 conservative engine is retained as
`visual_level5.engine` (same bytes/hash, renamed from `candidate_level5.engine`).
This result reflects builder tactic/fusion choices; no retraining or image-size
change occurred. The final repeated benchmark will compare this complete runtime
against the original TensorRT baseline, not against a different-resolution model.

Layer requests and engine-inspector reports are retained with the build evidence;
only the reference and selected engine binaries remain after final verification.
Machine-readable evidence: `evidence/runtime_optimization_D_4060_20261111.json`.

