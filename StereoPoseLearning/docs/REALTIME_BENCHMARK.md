# 单对双目推理与 C++ 绘图计时

> 公开版范围：保留历史执行性能、实现步骤及数值等价性记录，不发布该权重的独立预测精度结论。固定权重和图像未随源码分发；以下是历史实测，整理时没有重新测速。JSON 报告的原字段位于 `report` 中。

入口：`tools/benchmark_realtime.py`。直接调用现有 `StereoPredictor` 和 `CppStereoOverlay`，不改变模型结构、输入分辨率、权重或运动学定义，也不启动训练。

## 本次4060部署

Ubuntu 笔记本项目目录为 `~/PoseProject/StereoPoseInference_20261111/`，独立 Conda 环境为 `stereo_pose_infer311`。原训练和 Isaac Sim 环境保留。

```bash
conda create -n stereo_pose_infer311 python=3.11 pip -y
conda activate stereo_pose_infer311
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install timm==1.0.22 numpy==1.26.4 scipy==1.15.3 \
  Pillow==11.3.0 PyYAML==6.0.3 opencv-python-headless==4.11.0.86
cmake -S reference/pose_display/native -B reference/pose_display/native/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build reference/pose_display/native/build -j 2
```

C++ 依赖及纯 C++ 接口见 [NATIVE_OVERLAY.md](NATIVE_OVERLAY.md)。不需要安装 Isaac Sim、CUDA toolkit 或训练数据生成环境；CUDA运行库由 PyTorch wheel 提供，宿主机需要兼容的 NVIDIA 驱动。

## 数据与命令

`inputs/best_val.pt` 为固定最佳验证权重；`inputs/samples.json` 保存权重 SHA256、场景版本及样本清单。每条样本只有图片路径、序列/帧号/时间戳、类别及每目 K/D。网络权重和图片不提交到 Git。

```bash
conda activate stereo_pose_infer311
cd ~/PoseProject/StereoPoseInference_20261111
python tools/benchmark_realtime.py --inputs inputs \
  --output runs/benchmark_new --expected-gpu 4060 --rounds 3
```

输出目录必须为空。`--modes fp32` 仅测现有默认精度；默认另做 `bf16` 对照。BF16只在该推理进程中启用 autocast，不改权重、不覆盖当前推理模块的默认设置。独立进程退出即释放GPU资源。

## 测量口径

- 一帧是一对1280×720原始 RGB；全局定位输入640×360，局部分支仍取原始720p图像。
- `batch=1`，`time=1`，每次预测当前一对图像；三个GRU状态在同一条连续序列内保留。
- 七类各8帧连续输入，共56对；每种精度预热56对，再测三轮共168对。序列之间重置状态，序列内部禁止意外重置。
- 主计时从内存中的原始左右 RGB 开始，包含缩放、归一化、全部CPU→GPU传输、完整模型、17维结果GPU→CPU传输、C++运动学与畸变投影、中文标注及XYZ面板，直到完整双目画布生成。
- 模型返回 CPU NumPy 数组已等待GPU结果；同时用 CUDA event 检查网络区间，不把异步提交耗时冒充GPU完成耗时。
- 不包括磁盘读图/图片解码、相机采集、编码、GUI显示或显示器刷新。模型及字体初始化不进入稳态每帧计时，首次预热耗时另记。
- `Hz=1000/平均总耗时(ms)`；报告另给P95/P99和33.33ms内完成的比例，不用多帧批量吞吐冒充实时单帧速度。

输出 `report.json`、逐帧 `frames.jsonl`、GPU功耗/温度/频率记录及每种精度一张真实推理标注图。保留权重和源码哈希、软件版本、原始及网络分辨率。

## 2026-11-11 实测结果

硬件为 RTX 4060 Laptop GPU 8 GB，驱动580.95.05，Ubuntu22.04，接通电源、系统性能模式。软件为 Python3.11.18、PyTorch2.9.1+cu128、torchvision0.24.1+cu128、timm1.0.22；本机编译的 C++ 后端链接系统 OpenCV4.5.4及FreeType。CPU线程数为4，保留原推理设置（cuDNN benchmark关闭、matmul TF32关闭、cuDNN TF32开启）。完整Python包版本见 [requirements-inference-4060.lock.txt](../requirements-inference-4060.lock.txt)。

使用固定的固定权重最佳验证权重，SHA256为 `639fc8c7e56263f4ab889aa53cc5132e3610eea4c8b2d8b0db671c41d8a30325`。七类各8对连续样本；每种精度预热56对，测量168对。4090训练进程继续运行，测速全部在4060执行。

| 精度 | 预处理、全部传输及模型推理 | C++运动学、投影及绘图 | 总耗时均值 | 总耗时P95 | 双目对/秒 |
|---|---:|---:|---:|---:|---:|
| 原默认FP32 | 44.23 ms | 6.46 ms | **50.68 ms** | 52.19 ms | **19.73 Hz** |
| BF16 autocast对照 | 45.19 ms | 6.42 ms | 51.60 ms | 53.34 ms | 19.38 Hz |

FP32的网络CUDA计时间隔均值为25.76 ms；定位图像缩放、归一化和上传的CPU壁钟时间为14.24 ms。后者不包含原始720p局部ROI输入的上传，不能把它单独视为全部预处理耗时。C++绘图约占完整链路的12.7%；目前完整链路未达到30 Hz。网络CUDA区间不等于端到端时延，不能用其倒数替代上述19.73 Hz。

BF16在本次运行中没有提速，未修改生产推理的默认FP32设置。相对FP32，BF16的C平均差异0.0728 mm（最大0.2210 mm），实际尖端平均差异0.0785 mm（最大0.2271 mm），腕部俯仰/偏航/钳口半角平均差异分别0.0184°/0.0133°/0.0095°。

首次预热一对的耗时约1.45秒（FP32）和1.30秒（切入BF16）；它们单独记录，不计入稳态结果。上述帧率包含完整中文标注、杆轴箭头、两片钳口箭头和三个XYZ面板，不包括相机采集等待、磁盘解码、视频编码、GUI展示与显示器刷新。

完整统计见 [实测报告](evidence/realtime_4060_20261111.json)。笔记本运行记录位于 `~/PoseProject/StereoPoseInference_20261111/runs/benchmark_20261111/`，包含逐帧计时、GPU监测及两种精度的输出图。模型与图片未改写，测试结束后推理进程已退出并释放GPU。
