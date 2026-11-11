# 性能、资源与精度统计索引

本页集中导航既有实测结果，不替换原报告，也不把不同测试条件的数字混成一次实验。2026-11-12 项目清理保留了所有已提交性能证据、精度对照、资源记录及测速工具；未重新运行 GPU 性能测试。

## 已完成的推理优化

| 阶段 | 已记录的完整双目耗时 / Hz | 过程说明 | 原始汇总 |
|---|---|---|---|
| LibTorch + GPU 预处理 + C++ 三向量显示 | 约 31.96 Hz；以对应报告的条件为准 | [LibTorch](LIBTORCH_INFERENCE.md) | [资源审计](evidence/inference_resource_4060_20261111.json)、[GPU预处理](evidence/gpu_preprocess_4060_20261111.json) |
| TensorRT 视觉前端 | 约 37.76 Hz | [TensorRT](TENSORRT_VISUAL.md) | [精度和速度](evidence/tensorrt_visual_4060_20261111.json) |
| C 点融合、Graph、绘图、引擎 level 5 | 三轮约 15.23 ms / 65.68 Hz | [完整过程](RUNTIME_OPTIMIZATION.md) | [最终对照](evidence/runtime_optimization_final_4060_20261111.json) |
| CUDA 预处理 + 原图局部裁剪融合 | 同轮对照 15.277 → 12.981 ms；65.46 → 77.04 Hz | [CUDA融合](CUDA_FUSION.md) | [三轮最终对照](evidence/cuda_fusion_final_4060_20261111.json) |

最后一项每个程序先预热 56 对、测 560 对，交替重复三轮，每侧 1,680 次测量；候选完整流程 P95 13.605 ms、P99 14.588 ms，超过 16.67 ms 的比例约 0.119%。Hz 按 `1000 / mean_ms` 计算，不把绘图模块单独的 Hz 当成端到端 Hz。

计时范围是**内存中的双目 RGB → 预处理/上传 → 完整模型 → 结果回传 → 运动学与投影 → 两目三个向量及面板绘制完成**。读盘、解码、相机采集、视频编码和屏幕刷新不在该范围。结果来自同一固定权重的对应回归环境，不是任何机器、输入或热状态下的保证值。

## 分步骤和未采用候选

- 运行调度 A/B/C/D：[C 点与缓存](evidence/runtime_optimization_A_4060_20261111.json)、[CUDA Graph](evidence/runtime_optimization_B_4060_20261111.json)、[并行绘图](evidence/runtime_optimization_C_4060_20261111.json)、[TensorRT 构建候选](evidence/runtime_optimization_D_4060_20261111.json)。
- 算子融合 A/B/C：[定点缩放与归一化](evidence/cuda_fusion_A_4060_20261111.json)、[前景处理与原图 ROI](evidence/cuda_fusion_B_4060_20261111.json)、[关键点归约候选](evidence/cuda_fusion_C_4060_20261111.json)。关键点归约没有稳定的完整流程收益，默认关闭，结果仍保留。
- [初始端到端统计](evidence/realtime_4060_20261111.json)、[LibTorch对照](evidence/libtorch_4060_20261111.json)、[C++绘图统计](evidence/native_overlay_20261111.json)、[数据加载测试](evidence/loading_benchmark.json)。
- [训练曲线](evidence/training_curves.png)、[架构探索](ARCHITECTURE_EXPLORATION.md)和[对应证据](evidence/architecture_exploration.json)。

56 对七类样本用于转换精度和数值回归。相对基线输出差异与相对真值误差在原报告中分开记录，不能当成完整测试集的新模型精度结论。视频录制期间的耗时只是带后台任务的观察值，不替代隔离测速。

## 重现与工具

- [环境和模型复现](REPRODUCING.md)、[当前部署配置](../configs/inference_cuda_fused.json)。
- `tools/benchmark_realtime.py`：统一完整双目计时。
- `tools/benchmark_comparison.py`：相同条件下的交替比较。
- `tools/profile_realtime.py`：单独的分阶段分析，避免干扰正式计时。
- `tools/verify_*`：CUDA、TensorRT、时序状态、数值与绘图回归。
- `tools/benchmark_loading.py`：训练数据加载对照。

机器上的 `runs/*benchmark*`、`runs/*optimization*`、`runs/*fusion*`、`runs/*audit*` 等原始实验目录不上传大型文件，但清理时完整保留。仓库中的 JSON 汇总和方法文档是长期索引，不删失败候选、不只保留最后一个好看的数字。
