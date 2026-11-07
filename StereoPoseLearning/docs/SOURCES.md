# 模型设计来源

核查日期：2026-11-11。下表说明参考内容与本项目自行实现的边界；本项目不是这些论文的完整复现。

| 来源 | 参考内容 | 本工程的边界 |
|---|---|---|
| [Monocular pose estimation of articulated open surgery tools — in the wild](https://arxiv.org/html/2407.12138v2) | 关节器械位姿、局部图像、mask/几何监督和关节角预测 | 本项目的双目融合、因果时序、PSM固定链和两个相机输出是自定义实现，不能移用论文数值当作本项目效果 |
| [SurgPose: Generalisable Surgical Instrument Pose Estimation using Zero-Shot Learning and Stereo Vision](https://arxiv.org/html/2505.11439v1) | 分割与位姿估计的组合；RAFT-Stereo深度、Mask R-CNN、SAM-6D路径 | 原方法深度依赖校准几何。本工程不采用这条必需baseline的深度路径，而分别输出两相机pose |
| [ConvNeXt V2](https://arxiv.org/abs/2301.00808) 与 [官方代码](https://github.com/facebookresearch/ConvNeXt-V2) | 视觉主干结构和预训练方式 | 本项目通过timm加载Nano特征主干，增加自定义局部分支、时序、输出头 |
| [timm Nano 权重模型卡](https://huggingface.co/timm/convnextv2_nano.fcmae_ft_in22k_in1k) | FCMAE预训练、ImageNet-22k/1k微调权重的标识与来源 | 预训练是初始化，任务训练联合更新模型；模型卡标注CC-BY-NC-4.0，权重不随本仓库再分发 |

128通道FPN、4×4局部特征、160×160原图ROI、两个角度分支、射线×深度C表示、loss解耦、训练窗口及超参数是本项目经实验确定的工程选择，不写成论文规定的数值。

几何与显示来源：

- [固定器械模型](../assets/instrument_model.json)：从实际USD导出的关节轴、绑定变换、真实铰链和尖端局部位置，记录源场景哈希。
- [显示代码及来源哈希](../reference/PROVENANCE.json)：独立NumPy解码与已确认的箭头/XYZ布局。
- [数据生成模型接口](../../CameraImagingConfig/src/TemporalStereo_v2/MODEL_INPUT_OUTPUT.md)：真值、各自相机坐标、单位和字段约定。

论文来源用于说明方法与实现依据；不同数据集、图像尺寸和硬件上的准确率与 FPS 不能直接拼接比较。
