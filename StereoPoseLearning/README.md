# StereoPoseLearning

双目时序 PSM 位姿学习工程。Ubuntu 4060 路径：`~/PoseProject/StereoPoseLearning/`。原有生成工程、数据划分和场景不改动。本目录包含训练、validation、test 和 prediction；本次只做有限的运行验证，没有启动正式训练。

输入为同步左右 RGB、各目的 K/D。输出保持 17 值：左右各自 C(3) + 杆体四元数 xyzw(4)，再加共享腕 pitch、腕 yaw、钳口半开合角(3)。不把真实左右外参、baseline、关节读数或真值掩码传入网络。

模型：共享 ConvNeXtV2-Nano → 128 通道 FPN → 2 层双向注意力 → 因果 GRU(256) → 每目位姿头及共享关节头。mask/关键点热图为辅助监督；固定可微 CAD 运动学用于几何监督和解码。主干加载预训练初始化后，全模型联合训练。

- [架构、字段和训练设置](docs/ARCHITECTURE.md)
- [论文、官方实现及设计来源](docs/SOURCES.md)
- [4060 运行验证记录](docs/VALIDATION.md)
- [默认配置](configs/default.json)

```text
StereoPoseLearning/
├── stereo_pose/       # 数据读取、模型、可微几何、损失、训练评估、预测
├── assets/            # 与采集一致的固定 CAD 关节链与尖端定义
├── reference/         # 已确认的旧解码/箭头/XYZ 显示脚本原样快照
├── configs/           # 版本化工程配置
├── docs/              # 设计、来源、运行证据
├── tests/             # 几何、数据接口、因果时序验证
├── tools/             # 已存真值数值核对
├── data/              # 本地数据清单，不加入 Git
├── runs/train/        # checkpoint、日志、实际配置
├── runs/validation/   # validation 统计
├── runs/test/         # 独立 test 统计
└── runs/prediction/   # 预测 JSON 与旧版箭头/XYZ 叠加图
```

## 环境

4060 使用新建的独立 Conda 环境 `stereo_pose_learning`。设置 `PYTHONNOUSERSITE=1`，禁止加载 `~/.local` 中其他工程安装的 Python 包。旧 `.venv` 保留用于历史记录，已经弃用；不再借用其他环境的包，也不修改原数据或原 Conda 环境。

本环境使用 Python 3.10.22、PyTorch 2.0.0、torchvision 0.15.1、timm 1.0.22、NumPy 1.26.4、SciPy 1.15.3、Pillow 11.3.0、opencv-python-headless 4.11.0.86、PyYAML 6.0.3、pytest 8.4.2。新环境已通过依赖检查、14项测试、GPU反向传播及图片/视频预测复核，见 [验证记录](docs/VALIDATION.md)。跨机器创建独立环境使用 `environment.yml`；完整已安装版本见 `requirements-4060.lock.txt`。

```bash
cd ~/PoseProject/StereoPoseLearning
conda activate stereo_pose_learning
export PYTHONNOUSERSITE=1
python -m pip check
python -m pytest tests -q
./run.sh --help
```

下文所有命令均在该环境中执行。`run.sh` 使用激活环境的 Python，或本机 `.python-path` 中记录的新环境解释器；不再以 `.venv/bin/python` 启动。`.python-path` 属于机器配置，迁移时应写入目标机器的实际解释器路径。

## 数据索引

合并机器生成结果后，索引真实存在且已提交的图像/掩码/标签。可重复提供 `--run-root`，原始数据只读。输出清单保存文件哈希、split、连续源帧和资产身份；不删除遮挡、过曝或角度外推样本。

```bash
./run.sh index \
  --run-root /path/to/merged_run \
  --output data/full_manifest.json
```

```bash
./run.sh smoke --manifest data/smoke_manifest.json \
  --output runs/train/my_smoke_check
```

## 正式训练、验证与测试

以下为后续手动执行命令，本次没有启动。窗口8、batch1、梯度累积4；验证使用流式历史状态，遇轨迹切换/断帧重置。验证选择七类等权尖端误差最低的 checkpoint；测试不参与选择。

./run.sh validate --manifest data/full_manifest.json \
  --checkpoint runs/train/run_001/best.pt \
  --output runs/validation/run_001.json

./run.sh test --manifest data/full_manifest.json \
  --checkpoint runs/train/run_001/best.pt \
  --output runs/test/run_001.json
```

## 图像与视频 prediction

`calibration.json` 只需要下面的成像字段（K/D 填实际相机参数，不是示例假值）：

```text
image: {width: 1280, height: 720, space: "raw_distorted"}
left:  {K: 3×3 matrix, D: [k1,k2,p1,p2,k3]}
right: {K: 3×3 matrix, D: [k1,k2,p1,p2,k3]}
```

```bash
./run.sh prediction --checkpoint runs/train/run_001/best.pt \
  --left /path/to/left.png --right /path/to/right.png \
  --calibration /path/to/calibration.json \
  --output runs/prediction/pair_001
```

预测内部缩放到640×360并同步修正K，绘图保留原始1280×720。输出 `prediction.json`、`overlay.png`，沿用已确认的杆体/两片钳口箭头和 XYZ 布局。没有独立预测可见性时标为 `unknown`，不偷用真值遮挡标签。

同步视频使用 `configs/predictor.local.json`，包含实际 checkpoint 路径、`device: "cuda"`、`fps: 60`；实拍30fps视频应写30。通过已确认视频接口运行：

```bash
./run.sh video --left /path/to/left.mp4 --right /path/to/right.mp4 \
  --calibration /path/to/calibration.json \
  --instrument-model assets/instrument_model.json \
  --predictor-config configs/predictor.local.json \
  --output runs/prediction/video_001
```

模型文件与 checkpoint 的固定几何必须一致。因果推理只读当前和过去帧。诊断权重默认拒绝用于预测，只有显式 `--allow-diagnostic` 才可测试输出接口；这种输出不代表已训练效果。

## 加载优化与指标图

证据：[加载性能](docs/evidence/loading_benchmark.json)、[恢复状态核对](docs/evidence/resume_validation.json)、[训练曲线快照](docs/evidence/training_curves.png)。新增并行读取一致性和异步拷贝顺序测试后，16项测试通过。

当前双目模型为约1613万参数的可运行候选。能输出与反向传播17值，不等于已经达到目标三维精度或30Hz端到端速度；这些由后续真实训练和完整链路计时回答。
