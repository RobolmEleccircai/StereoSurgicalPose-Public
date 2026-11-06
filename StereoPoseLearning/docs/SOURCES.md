# 模型设计依据与来源

核查日期：2026-11-09。优先读取 arXiv 方法、实验和限制段落，并核对作者代码。下表区分文献已经验证的内容与本工程采用的设计；本项目不声称原样复现其中任何一个完整系统。

| 来源 | 已核查的内容 | 本工程采用与边界 |
|---|---|---|
| [GDRNPP: A Geometry-guided and Fully Learning-based Object Pose Estimator](https://arxiv.org/html/2102.12145v5)，TPAMI 2025 扩展稿，§III-B、§IV；[官方代码](https://github.com/shanice-l/gdrnpp_bop2022) | 从 RGB 学习几何中间表示，并通过可学习的位姿模块端到端训练。使用额外深度的 refinement 属于另一部分。 | 采用几何辅助监督与联合训练原则。原版处理刚体，本工程的关节链、双目和时序属于改造；不引入必需深度输入。 |
| [Monocular pose estimation of articulated open surgery tools—in the wild](https://arxiv.org/html/2407.12138v3)，MedIA 2025，作者接受稿于 2026-01 更新；§3.3、§4.2.2 | 同时学习位姿、开合角、掩码和稠密对应。训练使用可微 Patch-PnP；其消融支持把几何与最终位姿联合监督。 | 支持关节式器械的多任务学习。它只验证一个开合角，真实数据评估主要是投影掩码，不能当作本项目完整三维 PSM 精度证据。本工程没有照搬其稠密对应标签需求。 |
| [EfficientPose 6D: Scalable and Efficient 6D Object Pose Estimation](https://arxiv.org/html/2502.14061v1)，2025，§4.1、表1/2 | 比较 GDRNPP 的轻量主干和几何头，候选包含 ConvNeXtV2-Nano、FastViT-S12。 | 为 Nano 主干和轻量几何头提供选择依据。其 ROI 设置、硬件和计时条件不能直接换算为本项目双目 30 Hz；本次未找到可核实的独立作者实现仓库。 |
| [ConvNeXt V2 官方实现与预训练权重](https://github.com/facebookresearch/ConvNeXt-V2) | 包含 Nano 等模型定义和预训练权重；作者仓库已归档但代码仍可读取。 | 本工程通过 `timm` 加载对应主干。预训练仅是初始化，之后整个模型联合微调；实际权重标识与软件版本应随运行保存。 |
| [Instrument-Splatting](https://arxiv.org/html/2503.04082v1)，MICCAI 2025，§2.1–2.3；[官方代码](https://github.com/jinlab-imvr/Instrument-Splatting) | LND 以杆体相机位姿和三个内部角度表示，再用 FK 得到部件。原版采用逐帧优化和初始化，不是轻量直接回归模型。 | 采用状态参数化与 FK 约束思想。固定尺寸、锚点和零位取当前 USD 导出模型，不采用论文另一模型的数值；不把其 GS 渲染速度当成位姿预测速度。 |
| [Real-time Rendering-based Surgical Instrument Tracking via Evolutionary Optimization](https://arxiv.org/html/2603.11404v3)，2026，§III-A、表I/II；[官方代码](https://github.com/hanyang-hu/online_dvrk_tracking) | 支持无关节读数的位姿和三关节优化，讨论 LND 180°对称性。部分实验允许交换两尖端匹配。报告的完整在线高帧率配置使用了关节读数。 | 支持视觉估计相似物理状态的可行性，作为几何对照依据。不能推导本网络精度、固定 J1/J2 身份恢复或纯视觉双目速度。当前工程不采用交换尖端后取最小值冒充身份正确。 |
| [Extending 6D Object Pose Estimators for Stereo Vision](https://arxiv.org/html/2402.05610v2)，2024，§3.1–3.5 | 比较左右特征在不同阶段融合，也包括依赖水平对齐的视差分支。 | 采用联合视觉特征的思路，不使用需已校正图像的视差路径。两侧独立位姿头、共享关节头和未知外参条件是本项目设计，不是原文已验证的全部条件。 |
| [TinyDETR-Pose](https://arxiv.org/html/2608.15297v1)，2026，§3、§5 | 轻量直接位姿回归；作者指出对细微旋转、遮挡及部分非对称对象存在精度局限，统一 ADD-S 可能弱化旋转翻转的惩罚。 | 保留为速度导向的比较思路。不会仅凭单图延迟选择它作主模型，也不把对称最近邻距离作为固定身份状态的唯一监督。 |

## 本项目设计值，不是论文原值

以下项目由当前数据接口、4060 的计算预算和实时目标共同确定，属于可验证的工程初值：

- `640×360` 网络输入，原始 `1280×720` 图像保留。
- 128 通道 FPN，2 层、4 头跨视图注意力，256 维因果 GRU。
- 连续 8 对图像的训练窗口、单窗口批量、4 次梯度累积。
- 掩码与 C/H1/H2/J1/J2 投影热图辅助监督。
- 左右各自杆体状态、三个共享关节角、对外 17 值接口。
- 学习率、损失权重、训练轮数和检查点选择规则。

这些配置详见 [ARCHITECTURE.md](ARCHITECTURE.md) 与实际运行配置。整个组合尚需用本项目数据训练评估；不将各篇论文在不同数据和硬件上的数字拼接成预计准确率或 FPS。

## 来自当前工程的几何与数据依据

- [模型输入输出与已执行解码检查](../../CameraImagingConfig/src/TemporalStereo_v2/MODEL_INPUT_OUTPUT.md)：位置 mm、四元数 `xyzw`、关节角 rad、模型点定义、标签字段及计算验证。
- [现有显示接口](../../CameraImagingConfig/src/TemporalStereo_v2/pose_display/README.md)：17 值转 H/J、杆轴和 XYZ，再分别投影到左右图像。
- [固定器械模型来源](../../CameraImagingConfig/src/TemporalStereo_v2/validation_evidence/balanced_2100/new_calculations/instrument_model.json)：本次采集资产导出的关节轴、固定变换、铰链和尖端。

本目录部署后可直接使用 [固定模型快照](../assets/instrument_model.json) 和 [显示脚本快照](../reference/pose_display/README.md)，无需依赖旧工程目录结构。显示快照来源与文件哈希保存在 [PROVENANCE.json](../reference/PROVENANCE.json)。

预训练权重使用 [timm官方模型卡](https://huggingface.co/timm/convnextv2_nano.fcmae_ft_in22k_in1k) 所列权重；其模型卡标记CC-BY-NC-4.0。模型卡中的参数量包含分类头，本项目移除分类头后增加自定义模块，实际参数量以运行报告为准。

文献中的几何尺寸、相机标定或机器人读数不会替代这些现有数据。相机的 K/D 属于已知输入；真实左右外参和 baseline 仍只用于独立评估，不进入视觉模型。
