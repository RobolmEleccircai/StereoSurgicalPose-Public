# 4080 Super：现有 R595 驱动下的运行适配

2026-11-09，本实例的驱动为 595.91.07。Isaac Sim 5.1 的无界面渲染启动在 `librtx.scenedb.plugin.so` 崩溃；补齐系统动态库后仍可复现。**未更改宿主驱动**，在数据盘并列安装官方 Isaac Sim 6.0 standalone，完成以下兼容验证后用于 4080 分片。4060 保持原来的 5.1 环境。

这是本次已冻结工程的部署入口，不是声称所有 Isaac 5.1 项目都兼容 6.0。

## 实际运行版本与不变项

| 项目 | 4060 / 冻结规划环境 | 4080 渲染环境 |
|---|---|---|
| Isaac Sim | 5.1.0-rc.19+release.26219.9c81211b.gl | 6.0.0-rc.59+release.41464.5f2772bc.gl |
| Python | 3.11.13 | 3.12.13 |
| NumPy / SciPy | 1.26.0 / 1.15.3 | 2.3.1 / 1.17.0 |
| trimesh / Pillow | 4.5.1 / 11.3.0 | 4.11.1 / 12.1.1 |
| OpenCV | 4.11.0 | 4.13.0 |
| python-fcl | 0.7.0.11，cp311 | 0.7.0.11，cp312 |

4080 的 cp312 FCL wheel 单独安装到 `project/deps6`；没有覆盖 SDK 自带的数值包。额外系统库包括 `libxt6`、`libglu1-mesa` 以及原迁移说明的 Vulkan/X11/EGL 依赖。

共同使用提交 `7d4a31c` 的采集核心，核心 SHA-256 为 `5a725ef053469c1af6fceba5a26d2e127fb638216e7e0d006872a05ac8428113`。冻结分工 ID 为 `54d7b2dce7817266efb8c592c50b1e873a106737c5a5ecf136ff03c88ee1e7d7`。

- 4060：`seq_003, seq_013, seq_011`，352 对。
- 4080：其余 18 条完整轨迹，1,748 对。不存在同一轨迹由两台机器同时生成的情况。
- 轨迹、相机内参与畸变、1280×720、60 Hz 时间戳、PathTracing 64 spp × 4 subframes、暗角最低线性亮度 0.035、整体亮度系数 0.75 保持冻结配置。
- 自遮挡和组织遮挡继续保留并标记，没有新增可见性筛选。

## 验证结果与边界

1. 6.0 的无界面路径追踪启动和退出成功，现有 595.91.07 驱动没有再次触发原来的启动崩溃。
2. 两个 SDK 对相同 63 个冻结状态的 27,421 个几何及标签数值进行比较，最大绝对差为 0。
3. 实际相机读回的焦距、孔径、对焦距离、光圈和场景单位一致。
4. 隔离输出目录完成两对双目图片；用 5.1 的原验证代码检查 6.0 输出，图片、掩码、真值、文件完整性通过。最大投影算术误差约 `1.14e-13 px`，最大左右坐标变换误差约 `3.32e-13 mm`。这是数值一致性，不是模型预测精度。
5. 已检查两张双目预览：器官、器械和姿态箭头正常显示。固定端点自遮挡标签仍然保留。

SDK 的三个标准 MDL 文件及渲染器版本不同，**上述验证不声明两个版本的像素完全相同**。运行版本与实际依赖均保存，便于后续追踪。两对样本验证也不等于全量验收。

## 文件与入口

- `audit_runtime.py`：对同一资产和冻结轨迹采集两个 SDK 的几何、标签和依赖记录，不渲染。
- `compare_runtime.py`：比较两份记录。
- `capture_runtime6.py`：为移除的 `add_update_semantics` 提供到原生 `add_labels` 的进程内映射，再调用原 `capture.py`；不改 SDK 文件或采集核心。
- `validate_runtime6_probe.py`：验证本次隔离目录的两对兼容测试图片。其目视检查文字是本次人工检查记录，不是自动图像审美判断器。
- `run_worker6.py`：检查冻结分工、实际依赖和适配器哈希，锁定 4080 分片，渲染后运行原 5.1 验收器。适配器和启动器分别记录自己的哈希，不冒用核心哈希。

本次部署根目录为 `/root/autodl-tmp/stereo_surgical`，其下有：

```text
isaacsim-5.1.0/                 # 冻结规划与独立验收
isaacsim-6.0.0/                 # 4080 实际渲染
operations/                    # 本目录五个脚本的运行副本
operations/4080_20261109/       # workflow6.log、workflow6.pid、state.json
project/code_distributed_7d4a31c/
project/assets/
project/deps/                  # cp311 FCL
project/deps6/                 # cp312 FCL
project/config/machine.4080.local.json
project/config/machine.4080.runtime6.local.json
project/dispatch_2100_20261109/
project/checks/                 # 两个运行时审计及样本验证报告
project/outputs/worker_4080_20261109/
```

这两个机器配置只有 SDK、依赖目录及期望版本不同；资产、输出根目录和 worker ID 相同。分片已由原 5.1 入口完成 prepare-only，运行以下命令不会重新规划或领取 4060 的轨迹：

```bash
python3 /root/autodl-tmp/stereo_surgical/operations/run_worker6.py \
  --root /root/autodl-tmp/stereo_surgical
```

**当前任务已启动，不能另开一份。** 只有进程已停止且需要恢复时才重用该入口；它检查锁和已提交帧，不允许更换渲染器混入已有结果。当前生产任务由脱离 SSH 会话的进程运行，退出 SSH 不会结束任务。

`manifest.json.dependencies` 明确标为冻结规划准备环境；`render_dependencies` 和 `render_execution` 记录真实的 6.0 渲染环境、驱动、适配器/启动器哈希及验证报告哈希。每帧 `render_worker` 可关联对应 manifest。合并时两个 worker manifest 都会保留。

状态查看：

```bash
cat /root/autodl-tmp/stereo_surgical/project/outputs/worker_4080_20261109/status.json
tail /root/autodl-tmp/stereo_surgical/operations/4080_20261109/workflow6.log
```

## 官方依据

- [NVIDIA Isaac Sim 的 R595 / 5.1 启动问题](https://github.com/isaac-sim/IsaacSim/issues/706)
- [相关问题及 6.0 修复记录](https://github.com/isaac-sim/IsaacSim/issues/677)
- [Isaac Sim 6.0 下载](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/download.html)
- [6.0 Python 安装说明](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_python.html)

官方 Linux standalone 压缩包 MD5：`40ec5248271a0c2e7bc03f1ae725ca4c`，下载后已核对。SSH 凭据不写入代码、文档或 Git。
