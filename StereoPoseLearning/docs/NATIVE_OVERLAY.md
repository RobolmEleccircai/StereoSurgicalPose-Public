# C++ 实时双目绘图

> 公开版范围：保留历史执行性能、实现步骤及数值等价性记录，不发布该权重的独立预测精度结论。固定权重和图像未随源码分发；以下是历史实测，整理时没有重新测速。JSON 报告的原字段位于 `report` 中。

版本范围：下面的 `vectors` 模式、`canvas_shape` 和C ABI 2已提供对应源码并在4060验证。基础提交 `fce52c9` 使用原完整标注和C ABI 1；更新时应同时更新包装层与本地绘图库源码并重新编译，不能混用旧二进制。

当前实时模式为 `vectors`：每目只显示杆轴、两片钳口的物理箭头，三个自身XYZ面板，以及三个向量的数值。杆轴显示单位方向 `d`；钳口显示 `J1-H1`、`J2-H2`（毫米），均在对应相机光学坐标系中。它们分别从C、H1、H2出发。面板中的XYZ箭头仍随部件转动，仅移位显示。无帧号、时间戳、解释文字、关键点标签或顶部标题栏。`full` 保留原离线审阅显示，用于历史对照。

本次更新C ABI为2，Python包装与本地库需要一起更新并重新编译，防止不同画布尺寸的版本混用。视频入口默认为 `--display-mode vectors`；历史完整标注可显式选择 `--display-mode full`。

实时视频入口 `reference/pose_display/video.py` 默认使用 C++ 后端。训练、模型结构、权重和17维输出定义不变。旧 Pillow 后端保留，用 `--backend python` 选择；完整数字报告仍使用原 `StereoOverlay(compact=False)`。

## 实现与接口

- `reference/pose_display/native/overlay.cpp`：双目运动学、五参数畸变投影、物理箭头、中文标签和三组 XYZ 面板，全部在 C++ 中计算和绘制。
- `native/overlay.h`：独立 C ABI；纯 C++ 程序可以直接链接，不需要 Python。
- `native/render_pair.cpp`：读取两张原图和保存的17维预测值，独立生成标注图的 C++ 示例。
- `reference/pose_display/cpp_backend.py`：现有 PyTorch 推理循环的薄适配层，每对图像调用一次本地绘图函数。正常绘图不调用 Python 的运动学、投影或逐箭头绘制；错误提示页保留原 Pillow 实现。
- `tools/verify_cpp_overlay.py`：复现几何、图像所有权、可见性、视频入口及耗时对照。

正常绘图直接使用 RGB 输入，一次复制左右原图到输出画布，避免 PIL 面板拼接及转回 NumPy。输出可由调用方预分配复用。固定 CAD 变换在初始化时缓存；曲线保留61个畸变投影采样点，计算结果复用；动态标题按字符缓存，避免每帧重新栅格化整个标题。三维计算使用 double，没有用降低精度换取速度。

保持黄箭头 C→C+15d、蓝箭头 H1→J1、橙箭头 H2→J2；钳口 XYZ 在左右上角，杆体 XYZ 在左下角。XYZ 物理长度仍按两目共同约束选取，面板底色 `(17,29,42,102)` 不变。未知可见性仍用虚线并标“可见性未知”，不会由17维姿态伪造遮挡判断。

OpenCV 抗锯齿和 FreeType 字体栅格化与 Pillow 有差异，文字边界导致标签位置可能略有差别；不要求整张标注图逐像素相同。三维点、方向、物理箭头和 XYZ 投影曲线必须数值一致。动态帧号显示使用 ASCII 字符；原始帧号完整保存在元数据中。

## Linux 编译

依赖：C++17 编译器、CMake≥3.16、OpenCV core/imgproc/imgcodecs/freetype、中文字体。本次服务器已有依赖，未改训练环境。其他 Ubuntu 机器可安装 `build-essential cmake libopencv-dev libopencv-contrib-dev fonts-noto-cjk`。

在 `StereoPoseLearning` 根目录执行：

```bash
cmake -S reference/pose_display/native \
  -B reference/pose_display/native/build -DCMAKE_BUILD_TYPE=Release
cmake --build reference/pose_display/native/build -j 2
```

产物为 `libsurgical_overlay.so` 和 `overlay_pair`。编译产物不进入 Git；迁移服务器后重编译。不要启用 `-ffast-math`。可用 `SURGICAL_OVERLAY_LIBRARY` 指定库的绝对路径。缺少库会明确报错，不静默回退导致误报性能。

## 接入现有推理

```python
from pose_display import CppStereoOverlay, StereoFrame, PosePrediction
import numpy as np

display = CppStereoOverlay(instrument_model, calibration, font=font_path)
canvas = np.empty(display.canvas_shape, dtype=np.uint8)
frame = StereoFrame(frame_id, timestamp_ns, left_rgb, right_rgb)
prediction = PosePrediction(frame_id, timestamp_ns, values17)
image, metadata = display.render(frame, prediction, out=canvas)
# 完成当前帧的编码/显示或复制后，才可复用 canvas。
display.close()
```

未传 `out` 时，每次返回独立图像，不会被下一次绘制覆盖。图像尺寸必须与 K/D 对应、至少每目1280×720。Python默认 `display_mode='vectors'`，720p双目输出2560×720；`display_mode='full'` 输出2560×770，包含50像素标题栏。输出不能与输入共享内存；错帧、错时间戳、关节越限、零四元数、非有限值和相机后方投影会拒绝。一个句柄用于一个流，不并发操作；Python 适配层提供锁。

`geometry_log=False` 默认仅保存17值、来源、时间戳、后端及耗时，不构建详细曲线 JSON。开启 `geometry_log=True` 或视频参数 `--geometry-log` 时，输出 `geometry_schema: native_v1`：每目包含 C/d/H1/H2/J1/J2/T、三个旋转矩阵、像素位置、可见性编码、三个物理箭头和三组 XYZ 曲线。此调试结构与旧 Pillow 详细日志不同，读取器应检查 schema。`compute_geometry(values17)` 可独立返回这些结果，不绘图。

现有视频命令增加 `--backend cpp` 即可；默认也是 cpp。保存和显示复用同一次 RGB→BGR 转换。报告同时记录“推理＋绘图”和“读取＋推理＋绘图＋日志＋编码提交”的时间；后者仍不包括限速等待、编码器最终刷新和显示器呈现延迟。

## 纯 C++ 单对回放

`configuration.json` 包含 `model`（固定器械配置）和 `calibration`（image/left/right）。`prediction.json` 包含 `values17` 数组。输入是已有原始左右 RGB 图片，而非已有标注图。

```bash
reference/pose_display/native/build/overlay_pair \
  configuration.json /path/to/NotoSansCJK-Regular.ttc prediction.json \
  left.png right.png overlay.png
```

该示例只重放保存的模型输出，不运行网络，不启动 Isaac Sim。直接使用 C ABI 时，调用方负责缓冲区尺寸、步幅、生命周期与帧同步。

## 验证方法和计时边界

```bash
python tools/verify_cpp_overlay.py \
  --project "$PWD" \
  --manifest /path/to/training_run/manifest.json \
  --samples /path/to/saved_prediction_samples.json \
  --font /path/to/NotoSansCJK-Regular.ttc \
  --output /path/to/new/native_check
```

样本文件格式为 `samples[{sequence_id, frame_index, prediction_values17}]`。测试读取现有图片，140组保存/扰动姿态分别通过旧 NumPy 与新 C++ 实现计算，检查全部点、旋转、物理箭头、XYZ 长度和61点投影曲线，误差要求小于10⁻⁶像素。另检查无效输入、输出内存所有权、可见性语义、纯 C++ 示例、四帧视频回调以及错误帧报告。

两种后端在同一进程交替计时，每种140次、预热35次，轮换七组已有图像和预测，并改变帧号/时间戳。单帧指一对1280×720图像，包含解码、投影和精简标注，排除图片读取、网络推理、视频编码和窗口显示。训练继续运行，诊断仅用 CPU、低优先级。实际报告应与代码一起保存；不能把绘图倒数当作完整预测系统帧率。

2026-11-11 在4090服务器 CPU（Xeon Platinum 8470Q，训练同时运行）上的最终结果：

| 后端 | 平均 ms/双目帧 | 中位数 | P95 |
|---|---:|---:|---:|
| 原 Python/Pillow 精简版 | 26.05 | 26.22 | 32.86 |
| C++，复用输出画布 | **8.18** | **8.03** | **8.50** |
| C++，每次返回独立图像 | 8.68 | 8.05 | 10.56 |

同场平均加速约 **3.18倍**。140组几何检查最大投影差约4.55×10⁻¹³像素，仅说明数值实现一致，不代表模型估计精度。纯 C++ 示例、视频读写和错帧报告均通过。详细结果及源文件哈希见 [验证记录](evidence/native_overlay_20261111.json)。
