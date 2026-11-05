# Shared pose display verification — 2026-11-08

Executed on the Ubuntu laptop, Intel Core Ultra 5 125H, using the existing Python 3.11.13 interpreter and dependencies listed in `report.json`. No Isaac Sim/Omniverse/USD Python modules were imported; no GPU rendering or trained model inference was started.

The input was the immutable `quality_smoke_20261108` run: 14 stereo pairs, seven categories, two consecutive frames per sequence. The model constants were the previously exported `model_io_decoding/instrument_model.json` (SHA-256 recorded in the report).

| Check | Observed result |
|---|---|
| Original complete annotation JPEGs | 14/14 byte-identical after renderer extraction |
| Source run files | All 185 files unchanged, including RGB, labels, masks, camera truth and existing previews |
| 17-value decoding | Maximum position difference 1.629e-6 mm; projected landmark difference 1.513e-5 px; drawn XYZ rotation-column difference 1.101e-7 |
| Invalid inputs | Nonfinite/zero-quaternion/limits/negative-opening/behind-camera/wrong-frame/wrong-time/wrong-size/wrong-image-geometry rejected |
| Unknown visibility | Retained as unknown, never assumed visible |
| Invalid prediction | Current raw images displayed without stale arrows |
| Video callback and encoding | Two synchronized consecutive frames, 60 fps, explicit ground-truth replay source |
| Compact CPU postprocessing | 50 calls cycling through all 14 poses with changing frame IDs/timestamps: median 21.27 ms, p95 22.76 ms |
| New RGB renders | 0 |

The compact benchmark includes 17-value decoding, distorted projection, arrows, Chinese labels, displaced XYZ panels and stereo canvas assembly at 2560×770. It excludes model inference, image reading, video encoding and GUI refresh. Text caches were warmed during the preceding checks; the separate two-frame video subprocess averaged 49.87 ms for callback plus drawing, including its cold startup frames. These measurements do not prove sustained end-to-end 30 Hz prediction or 60 Hz computation.

The 60 fps replay uses the real two-frame source sequences without duplicating or interpolating frames. This checks temporal wiring, not long-video quality or a trained model's accuracy. `report.json` includes the tested annotation code and model hashes. Reproduce with `tools/verify_pose_display.py`; provide a new output directory outside the source run.
