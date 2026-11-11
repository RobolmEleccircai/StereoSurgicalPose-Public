# 执行性能与数值等价性记录

这些是固定模型的历史执行测量，不是独立测试精度结果。保留全部 16 份性能／资源报告，包括被拒绝候选。公开报告记录原文件 SHA，去掉标签误差统计后，原始耗时、资源与数值实现差异不变；数据在 JSON 的 `report` 字段中。

| 阶段 | 完整双目耗时／速率 | 过程 |
|---|---|---|
| C++ LibTorch、GPU 预处理 | 约 31.96 Hz | [LibTorch](LIBTORCH_INFERENCE.md) |
| TensorRT 视觉前端 | 约 37.76 Hz | [TensorRT](TENSORRT_VISUAL.md) |
| C 点融合、Graph、并行绘图、level5 | 约 15.23 ms / 65.68 Hz | [运行优化](RUNTIME_OPTIMIZATION.md) |
| CUDA 预处理＋原图局部裁剪融合 | 同轮对照 15.277→12.981 ms / 77.04 Hz | [融合过程](CUDA_FUSION.md) |

完整范围：内存双目 RGB→预处理与上传→完整网络→输出回传→运动学／投影→两目三个向量与面板绘制完成。不含图片读取、解码、相机采集、编码、屏幕刷新。每次一对双目，不用批量吞吐率替代延迟。

最终测量每版本预热 56 对、测 560 对、交替三轮；各 1,680 次。最终 P95 约 13.605 ms，P99 约 14.588 ms，超过 16.67 ms 的比例约 0.119%。77 Hz 是平均处理速率，不是所有输入均达到该速率的保证。

## 保留的报告
- [cuda_fusion_A_4060_20261111](evidence/cuda_fusion_A_4060_20261111.json)
- [cuda_fusion_B_4060_20261111](evidence/cuda_fusion_B_4060_20261111.json)
- [cuda_fusion_C_4060_20261111](evidence/cuda_fusion_C_4060_20261111.json)
- [cuda_fusion_final_4060_20261111](evidence/cuda_fusion_final_4060_20261111.json)
- [gpu_preprocess_4060_20261111](evidence/gpu_preprocess_4060_20261111.json)
- [inference_resource_4060_20261111](evidence/inference_resource_4060_20261111.json)
- [libtorch_4060_20261111](evidence/libtorch_4060_20261111.json)
- [loading_benchmark](evidence/loading_benchmark.json)
- [native_overlay_20261111](evidence/native_overlay_20261111.json)
- [realtime_4060_20261111](evidence/realtime_4060_20261111.json)
- [runtime_optimization_A_4060_20261111](evidence/runtime_optimization_A_4060_20261111.json)
- [runtime_optimization_B_4060_20261111](evidence/runtime_optimization_B_4060_20261111.json)
- [runtime_optimization_C_4060_20261111](evidence/runtime_optimization_C_4060_20261111.json)
- [runtime_optimization_D_4060_20261111](evidence/runtime_optimization_D_4060_20261111.json)
- [runtime_optimization_final_4060_20261111](evidence/runtime_optimization_final_4060_20261111.json)
- [tensorrt_visual_4060_20261111](evidence/tensorrt_visual_4060_20261111.json)

`tools/benchmark_comparison.py` 用相同权重、引擎、样本和绘图库交替比较；`tools/profile_realtime.py` 的侵入式分阶段分析另跑，不把追踪开销混入正式结果。实现转换比较原程序与优化程序的输出，不能把二者接近等同于对真实标签准确。
