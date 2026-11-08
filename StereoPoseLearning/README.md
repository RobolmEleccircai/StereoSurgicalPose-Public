# StereoPoseLearning

双目时序手术器械位姿模型。当前实现为 **前景引导定位 + 原图腕部/钳口双分支 + 单一 C 像素射线与深度预测**，输出左右相机坐标系下的器械状态，并复用固定运动学与箭头绘图接口。

- [逐步架构探索记录](docs/ARCHITECTURE_EXPLORATION.md)：问题、假设、改动、受控实验、结果与未解决项。
- [当前结构、17 个输出及损失](docs/ARCHITECTURE.md)。
- [环境、数据和推理接口](docs/REPRODUCING.md)。
- [C++ 实时绘图、编译和性能验证](docs/NATIVE_OVERLAY.md)。
- [4060 推理环境与单对双目端到端计时](docs/REALTIME_BENCHMARK.md)。
- [LibTorch直接推理、GPU预处理与原版对照](docs/LIBTORCH_INFERENCE.md)：4060完整双目流程实测均值31.02Hz（含C++投影绘图）。
- [验证方法及统计口径](docs/VALIDATION_PROTOCOL.md)。
- [论文与实现来源](docs/SOURCES.md)。
- [历史结构源码和复现实验](experiments/README.md)。

## 输入与输出

输入同步左右原始畸变 RGB、每目的 K/D 和过去帧的隐状态。全局定位输入为 640×360，局部分支从原始 1280×720 图像裁取。网络不接收左右外参、baseline、关节真值或真值分割。

输出 `C_L(3), quaternion_L_xyzw(4), C_R(3), quaternion_R_xyzw(4), wrist_pitch, wrist_yaw, jaw_half` 共 17 个数值。位置为毫米，角度为弧度，坐标轴为相机光学坐标系 x 向右、y 向下、z 向前。固定几何计算杆轴 d、真实铰链 H1/H2、实际尖端 J1/J2、中点 T 和部件 XYZ 姿态。两侧分别用自身 K/D 投影。

```text
stereo_pose/              当前网络、数据、损失、训练、推理与运动学
assets/                  固定器械尺寸、关节链与尖端定义
reference/pose_display/  C++ 实时绘图、NumPy 对照和箭头/XYZ 显示接口
configs/default.json     当前模型与训练配置模板
experiments/stages/      五个历史结构的源码、配置模板及文件哈希
requirements*.txt        依赖与已验证 Linux CUDA 环境锁定
tools/                   路径迁移、固定权重评估、诊断与指标绘图
tests/                  几何、时序、分支路由、采样及恢复测试
```

Git 仅保存源码、配置、必要的小型证据和文档。数据图片、权重、缓存、视频和运行日志不上传。完整复现既有数值需要额外提供对应的只读数据清单、数据文件、checkpoint 和运行配置；仅克隆仓库可以运行不依赖私有数据的测试，并在准备数据后重新训练。

## 快速入口

以下命令在本目录运行，环境安装步骤见[复现说明](docs/REPRODUCING.md)。先准备已有清单，再显式启动训练：
