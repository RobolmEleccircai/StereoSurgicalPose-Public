# 器械位姿计算、投影与视频显示

实时视频入口现默认使用 **C++ 后端**，先按 [C++ 编译与验证说明](../../docs/NATIVE_OVERLAY.md) 编译。原 Python/Pillow 接口和完整报告仍保留用于对照；视频加 `--backend python` 可显式使用旧版。下文原接口示例对应旧后端，新实时接口为 `CppStereoOverlay`。

本目录把“模型输出 → 运动学解码 → 每目投影 → 箭头和 XYZ 面板”整理为独立模块。批量生成器的真值预览与未来推理程序共用 `renderer.py`。这里不训练视觉网络，也不启动 Isaac Sim。

## 1. 入口和文件职责

| 文件 | 职责 |
|---|---|
| `decoding.py` | 17 个预测值 + 固定器械模型 → C、d、H1/H2、J1/J2、T、杆体/双钳口旋转 |
| `projection.py` | 使用每目的 K 和 OpenCV 五项畸变系数投影 |
| `geometry.py` | C→C+15d、H1→J1、H2→J2；三组随器械转动的 XYZ 轴 |
| `renderer.py` | 统一绘制物理箭头、标签、XYZ 面板；真值和预测分别标明 |
| `text_cache.py` | 视频模式的有界文字缓存；不改变箭头几何或透明度 |
| `pipeline.py` | 对外接口 `StereoFrame`、`PosePrediction`、`StereoOverlay.render()` |
| `video.py` | 已同步的左右视频逐帧调用预测函数，保存叠加视频与逐帧日志，可打开显示窗口 |
| `adapters.py`、`replay.py` | 从已有数据提取 17 个真值并回放，明确标注“非模型预测” |
| `recorded.py` | 视频接口测试用的真值记录读取器；不是视觉模型 |

旧入口 `tools/kinematic_decoder.py`、`tools/reannotate.py` 保留兼容转发。生成器 `live_preview.py` 已使用统一绘图逻辑。工具不读取旧标注图片当输入；绘图始终使用原始 RGB。

## 2. 模型交给显示模块的内容

每对图像预测 17 个数，固定顺序如下：

```text
[C_left_x, C_left_y, C_left_z,
 q_left_x, q_left_y, q_left_z, q_left_w,
 C_right_x, C_right_y, C_right_z,
 q_right_x, q_right_y, q_right_z, q_right_w,
 wrist_pitch, wrist_yaw, jaw_half_angle]
```

- C：毫米；q：杆体自身坐标系到对应相机光学坐标系的四元数，`xyzw`，进入解码时归一化。
- 三个角：弧度。两目共享器械内部状态；当前两片钳口对称联动，关节值分别为 −θ、+θ，θ 在 0～30° 内。
- 左、右输出分别属于各自相机光学坐标系：X 向右、Y 向下、Z 向前。无需先估计双目外参或 baseline。
- 固定模型来自同一版官方器械 USD：[instrument_model.json](../validation_evidence/model_io_decoding/instrument_model.json)。换器械/模型版本必须重新导出，不能沿用旧铰接中心或尺寸。

公式和训练输入/监督字段见 [MODEL_INPUT_OUTPUT.md](../MODEL_INPUT_OUTPUT.md)。显示模块根据 17 个数计算其余几何；H/J/XYZ 不是再次猜测，也不从帧真值读取。

**这 17 个数不能判断组织遮挡或图像过曝。** 未提供独立可见性结果时，尖端标“可见性未知”，箭头虚线。可选 `visibility` 需声明来源；固定点可见性、尖端区域可见性和过曝仍分别处理。不会把没有预测出来的可见性冒充仿真真值。

## 3. 在推理循环中调用

运行时依赖见 [requirements-display.txt](../requirements-display.txt)。从 `TemporalStereo_v2` 目录执行，或把该目录加入 Python 模块搜索路径。需要支持中文的字体，可显式传入 `font=`。

```python
import json
from pose_display import StereoFrame, PosePrediction, StereoOverlay

model_geometry = json.load(open('instrument_model.json'))
calibration = json.load(open('calibration.json'))
display = StereoOverlay(model_geometry, calibration, compact=True)

# 每帧：左右 RGB uint8 图像必须同步，尺寸与 calibration 完全一致。
frame = StereoFrame(frame_id, timestamp_ns, left_rgb, right_rgb)

# 你的视觉网络/时序网络返回 17 个物理量。
# 若为 GPU tensor，先 values.detach().cpu().numpy()；单位和输出顺序按上述约定转换。
prediction = PosePrediction(frame_id, timestamp_ns, values17)
stereo_rgb, metadata = display.render(frame, prediction)
# stereo_rgb 可交给窗口、视频编码器或实时流；metadata 保存解码/投影结果。
```

相机配置格式：

```text
{
  "image": {"width": 1280, "height": 720, "space": "raw_distorted"},
  "left":  {"K": 3×3 矩阵, "D": [k1,k2,p1,p2,k3]},
  "right": {"K": 3×3 矩阵, "D": [k1,k2,p1,p2,k3]}
}
```

以上是结构说明，不是可以直接使用的相机参数文件。K/D 必须来自实际输入视频对应的配置。显示用原始带畸变图像，不能把去畸变图像配上旧 D；裁剪/缩放后须更新内参。网络内部的缩放/归一化由模型适配器负责，不改变物理三维输出坐标系。当前批准的布局要求每目至少 1280×720。

模型预测与图像的 `frame_id`、`timestamp_ns` 必须一致。异步推理时，缓存对应图像后再绘制；不能把前一帧预测贴到当前图像上。时间戳来自同步采集端，不能用推理完成时刻代替。时序模型可在回调对象内维护历史帧。

无效四元数、非有限数、关节越限、相机后方投影、尺寸不匹配、错帧/错时间戳均拒绝。视频程序显示当前原图和错误提示，不沿用上一帧箭头；日志记为失败。单独使用 API 时可捕获异常并调用 `display.unavailable(frame, reason)`。

## 4. 视频预测接口

模型适配器提供 `build_predictor(config)`，返回一个 `predict(frame, calibration)` 函数/可调用对象。它读取 `frame.left_rgb/right_rgb`，运行真实模型，并返回上述 `PosePrediction`。模型初始化和权重加载放在工厂函数中，不要每帧加载。

```bash
python -m pose_display.video \
  --left /path/to/left.mp4 --right /path/to/right.mp4 \
  --calibration /path/to/calibration.json \
  --instrument-model /path/to/instrument_model.json \
  --predictor your_model_adapter:build_predictor \
  --predictor-config /path/to/model_config.json \
  --output /path/to/new/prediction_review --show
```

`your_model_adapter` 是后续真实模型的接入模块，不是本项目已经训练好的网络。当前交付的是可调用接口与真值回放验证。

两段输入视频须事先同步、固定且相同帧率；程序按帧号配对，检查帧数/帧率，不负责硬件同步或恢复变帧率时间戳。60 fps 输入保存为 60 fps 视频，但输出文件帧率不代表计算速度。`--show` 打开窗口，Q/Esc 结束会报告中断；无桌面服务器省略它。真实相机流可直接调用第 3 节 API，不经过文件入口。

输出目录必须是新目录，包含：

- `overlay.mp4`：每侧 1280×720 原图上叠加箭头，加 50 px 标题；双目画布 2560×770。
- `frames.jsonl`：17 个输出、帧号/时间戳、来源、有效状态和耗时；`--geometry-log` 额外保存完整投影曲线。
- `calibration.json`、`instrument_model.json`：此次实际使用参数。
- `report.json`：完整/失败状态、帧数和耗时。失败不会标为通过。

`compact=False` 仍保留完整数字说明页，供离线检查。视频模式省去长数字页，保留器械图上的箭头和三个 XYZ 面板。

## 5. 保留的图示定义

- 黄箭头从 C 出发，沿 d，三维绘图长度 15 mm。
- 蓝箭头 H1→J1、橙箭头 H2→J2，起点为各自物理铰接中心；T 仍计算并保存在投影结果中，但不冒充实际尖端。
- 钳口 1 XYZ 在每目左上角、钳口 2 在右上角、杆体在左下角；方向随对应部件转动。
- XYZ 仅在屏幕中移位显示；原点的三维定义仍分别为 H1、H2、C。小轴投影长度最多为参考箭头的 55%；接近视线方向退化时使用 12 px 显示参考下限，避免小轴完全消失。
- 面板底色 RGBA `(17,29,42,102)`：60% 透明；文字、箭头和边框保持不透明。
- 共用铰接位置、闭合尖端接近重合时保持真实重合，不人为错开物理箭头。
- 标题明确区分仿真真值、真值输入回放、模型预测。可见性未知不会伪标成可见。

## 6. 用已有图片验证，不重新渲染

```bash
python -m pose_display.replay \
  --run /path/to/existing/run \
  --instrument-model /path/to/instrument_model.json \
  --output /path/to/new/replay

python tools/verify_pose_display.py \
  --run /path/to/existing/run \
  --instrument-model /path/to/instrument_model.json \
  --output /path/to/new/checks
```

回放使用已保存的图像和真值，标题明确“非模型预测”。每序列要求连续帧；当前 14 对验证数据每类只有 2 帧，生成的 60 fps 片段很短，不复制或插帧冒充长视频。

验证内容包括：旧版完整预览逐像素回归、17 维解码误差、真实箭头起点和面板位置、未知可见性、错误输入拒绝、帧号/时间戳对齐、双目视频回调读写、原文件哈希保持不变。结果见 [验证报告](../validation_evidence/pose_display/report.json)。测试数值来自真值输入，不是网络预测精度；绘图耗时不包含未来视觉模型、视频编解码和窗口刷新，不能据此宣称完整系统已达到 30 Hz。

本次笔记本测得缓存预热后双目计算与绘图中位数 **21.27 ms**、p95 **22.76 ms**（50 次，轮换 14 个姿态）；独立两帧视频进程含冷启动的平均值为 49.87 ms。完整条件与限制见 [验收说明](../validation_evidence/pose_display/README.md)。

源码指纹现已覆盖共享模块，防止修改绘图/计算后误用旧缓存。已有数据及其旧指纹保持原样；未来生成使用新 run_id，不混入原 run。此次没有启动全量生成、训练或重新渲染。
