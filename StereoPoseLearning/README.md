# StereoPoseLearning

前景引导的双目时序器械状态估计，含杆体、腕部和钳口三个分支。

- [结构图与尺寸](docs/MODEL_DIAGRAMS.md)
- [架构探索方法](docs/ARCHITECTURE_EXPLORATION.md)
- [17 值、坐标与损失](docs/ARCHITECTURE.md)
- [环境、数据和推理接口](docs/REPRODUCING.md)
- [评估口径](docs/VALIDATION_PROTOCOL.md)
- [执行性能记录](docs/PERFORMANCE_INDEX.md)

输入同步左右 RGB、各自 K/D 和历史状态；输出左右 C／杆体四元数，以及共享 wrist_pitch、wrist_yaw、jaw_half。位置为 mm，角度为 rad，四元数为 xyzw；相机光学坐标 X 向右、Y 向下、Z 向前。
