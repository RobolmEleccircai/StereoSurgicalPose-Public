# Continuous stereo prediction video on the 4060

## Purpose and actual inputs

The 2026-11-11 review runs the deployed model on consecutive existing stereo RGB images and draws the approved three vector panels with the optimized C++ overlay. It does not render new simulation images. Saved pose, masks, visibility, baseline and camera extrinsics are not passed to the predictor. Only RGB, each camera's K/D and the causal recurrent states are supplied, with the checkpoint's instrument geometry used for prediction decoding.

The runtime uses the validated fixed-checkpoint checkpoint and its matching TensorRT level-5 engine, CUDA Graph, GPU preprocessing and the accepted `preprocess`/`hires_crop` fusions. The checkpoint SHA256 is `639fc8c7e56263f4ab889aa53cc5132e3610eea4c8b2d8b0db671c41d8a30325`. The optimized overlay library SHA256 is `c7ce83788a68d67ccb6eee47751d5331a077bfe092459151557f147b1b1fb59e`.

## Temporal behavior

The tool checks contiguous source frame indices and 60 Hz timestamps. It warms up on scratch history, resets before each independent sequence, then carries all three recurrent states through the sequence. No interpolation, smoothing, frame dropping or extra state resets are applied. RGB frames remain 1280×720 per eye; the side-by-side output is 2560×720.

| Sequence | Category | Source frames | Video duration |
| --- | --- | --- | --- |
| seq_000 | Position and orientation | 0–179 | 3.0 s |
| seq_003 | Lighting | 0–179 | 3.0 s |
| seq_006 | Imaging quality | 0–89 | 1.5 s |

These are existing local training sequences used for visual continuity inspection, not a held-out accuracy evaluation. Each has its own video. The combined 450-frame, 60 fps video has hard cuts at 3.0 and 6.0 seconds. A cut between independent scenes is not a model discontinuity. The player also loops the combined video.

## Reproduction

Run from the deployed inference project, using its existing inference environment. Choose a new empty output directory. The script rejects overwriting a populated directory.

```bash
python tools/render_prediction_video.py \
  --manifest /path/to/data/manifest.json \
  --sequences seq_000 seq_003 seq_006 \
  --predictor-config configs/inference_cuda_fused.json \
  --overlay-library reference/pose_display/native/build/optimized/libsurgical_overlay.so \
  --output runs/prediction_video_20261111
```

`predictions.jsonl` retains the actual 17 model outputs, sequence/frame identity, timestamp and K/D for each image pair. `report.json` records the model/library hashes, actual execution mode, frame counts, reset checks, source ranges and playback metadata. Each sequence also has a representative JPEG. FFmpeg encodes H.264 at CRF 18 and 60 fps; this encoding affects the review video, not the model input or saved predictions.

## Observed result

All 450 pairs completed with finite predictions, valid overlays, no unexpected mid-sequence resets and no missing output frames. FFprobe confirms 450 frames at 60/1 fps. The actual execution mode was `optimized_graph`. The three approved vector annotations, instrument-local XYZ axes, component values and translucent panels are drawn in each eye.

Observed prediction-plus-drawing mean was 13.96 ms per stereo pair while backup and video encoding were active. This is not an isolated performance benchmark or a claim that recording/encoding achieved that latency. Video playback is at the source 60 fps rate; camera acquisition, display refresh and encoding are outside that timing. The purpose here is to expose the model's unfiltered temporal predictions visually.

The output was saved on the laptop in `/home/leorobot/PoseProject/StereoPoseInference_20261111/runs/prediction_video_20261111/` and opened in a looping desktop player.

## Longer playback from the complete archived dataset

The same tool supports `--playback-fps 30`. This changes the output video's time base only: every source frame is still inferred once with its original 60 Hz timestamp and recurrent history. No frames are removed or interpolated. Thus 2,520 source pairs occupy 84 seconds of playback, representing 42 seconds of combined source motion at half speed. Independent trajectories remain separated by hard cuts and recurrent-state resets. This does not create a continuous 84-second physical trajectory or measure performance on native 30 Hz camera input.

Omit `--sequences` to process every sequence in the manifest, longest first. For the cloud archive, use `--path-prefix` to resolve the original absolute paths to the local archive; the original manifest, labels, images and split assignments remain unchanged. This all-sequence video is a qualitative review across the saved data, not a held-out accuracy evaluation.

```bash
python tools/render_prediction_video.py \
  --manifest /data/prepared/manifest.json \
  --path-prefix /root/autodl-tmp /path/to/archive/cloud_autodl_tmp \
  --predictor-config configs/inference_cuda_fused.json \
  --overlay-library reference/pose_display/native/build/optimized/libsurgical_overlay.so \
  --playback-fps 30 \
  --output runs/prediction_video_all_30fps_20261111
```

The report records source fps, playback fps, playback speed ratio, per-sequence source/playback durations, frame counts, cut positions, manifest hash, path remapping and execution mode. The combined file is `stereo_pose_predictions_30fps.mp4`; per-sequence videos permit inspecting motion without a scene cut.

Completed review: all 23 trajectories across seven categories produced 2,520 prediction records and 2,520 video frames. FFprobe confirmed 2560×720, 30/1 fps and exactly 84.000000 seconds. There were no unexpected state resets within trajectories; the runtime remained `optimized_graph`. The longer video replaced the earlier short video in the laptop's looping desktop player. Results are under `/home/leorobot/PoseProject/StereoPoseInference_20261111/runs/prediction_video_all_30fps_20261111/`. No source archive files or model weights were changed.
