# 4060 CUDA融合实施记录

> 公开版范围：保留历史执行性能、实现步骤及数值等价性记录，不发布该权重的独立预测精度结论。固定权重和图像未随源码分发；以下是历史实测，整理时没有重新测速。JSON 报告的原字段位于 `report` 中。

## 固定对照和计时口径

2026-11-11开始。固定固定权重（SHA256 `639fc8c7e56263f4ab889aa53cc5132e3610eea4c8b2d8b0db671c41d8a30325`）、当前TensorRT level5引擎、CUDA Graph和优化C++绘图。历史15.2257 ms仅作参考，本次重新测量。原动态库、引擎和权重保持不变；新库单独构建。4090训练不参与此次操作。

端到端计时：内存双目RGB→预处理和传输→完整模型和结果回传→运动学、投影及三个向量的数值/XYZ轴/透明面板绘制完成。排除读盘、解码、相机采集、视频编码及屏幕刷新。每次56对预热、560对测量；最终交替三轮。算子追踪另行执行。

## A：预处理

现状：`pass()`每个方向分别调用四次索引、转换、乘法和累加；归一化又建立int64索引和中间图。目标是减少启动及中间数据读写，不改变缩放滤波器。

实现：水平四抽头CUDA核先输出uint8；纵向四抽头、第二次uint8舍入及原FP32归一化查表合并成第二个核。复用原22位系数、边界索引和归一化表，保留NHWC底层存储及对外NCHW视图。两个缩放方向不能合并舍入。水平/小图/归一化缓冲区在初始化时分配；高分辨率原图布局路径保持原计算。使用当前CUDA流，禁止快速数学和隐式全设备同步。

验收：56对真实图片及黑白、棋盘格、随机、边界和非连续输入的缩小RGB、归一化、K/D和高分辨率图必须逐值一致。

实测：56对真实图＋6组边界图，缩小RGB、归一化、高分辨率输入、17值、隐藏状态、派生点及绘图最大差均为0；原Pillow路径也逐像素一致。

| 单轮560对，均关闭详细追踪 | 完整均值 | P95 | Hz |
|---|---:|---:|---:|
| 当前程序 | 15.2692 ms | 15.9858 ms | 65.4912 |
| A预处理融合 | 14.0605 ms | 14.7401 ms | 71.1214 |

阶段加速7.92%，待最终三轮交替确认。保留A。

实施问题：初次源码替换误匹配`host_input`导致成员重复，编译明确失败；移除重复声明后构建成功。验证工具最初未接入新增配置参数，参数解析拒绝运行；补齐入口后重新完整验证。没有放宽任何数值条件。

检查证据保存在运行目录`A_preprocess.json`、`A_accuracy/report.json`、`baseline_performance/report.json`、`A_performance/report.json`。

## B、C及最终验收

三个步骤均已完成，逐步结果见下文。最终选择A+B，C默认关闭。

## B：局部裁剪实现过程

实现说明：掩码sigmoid继续使用原ATen计算；后续上采样、阈值和膨胀在一个核内完成，每个输出查对应邻域并保留原顺序。裁剪核直接读取原始uint8 RGB与掩码，在四个源像素上逐个完成归一化和掩码相乘，然后按原grid_sample权重相加。消除整幅FP32 RGB和四通道拼接图。只保留720p单通道掩码中间结果；边界填零、坐标转换与160×160大小保持不变。单算子和全流程验证结果见本节末尾。


B结果：48组空/满/随机/阈值附近掩码、膨胀0/1/2、正常/出界/边缘/非连续网格均通过。裁剪最大绝对差9.536743×10⁻⁷（FP32），使用atol/rtol各2×10⁻⁶的单算子检查；未启用快速数学。隐藏状态最大差0.0000999346，属于局部输入微小数值差的传播，不声称逐值相同。状态推进另以相同融合核的普通执行对照Graph验证。

累计A+B：均值12.9991 ms、P95 13.6952 ms、76.9284 Hz。相对A再降7.55%，相对本轮初测基线下降14.87%；保留B，最终仍做三轮交替。

实现中未改插值/阈值顺序，也未调整权重或引擎。原图标签/几何只在评估中读取。

## C：关键点归约实现过程

实现说明：每个关键点使用一个CUDA线程块，先归约权重和与最大置信度，再重新计算标量权重并先归一化、后乘坐标，避免将浮点语义改为“先求分子、最后统一相除”。保留FP32、指数4、1e-12稳定项和原像素中心坐标。两次遍历都在同一核内，不生成完整概率/权重图。


C结果：12组随机/零/极低/极高及尖峰/边界热图检查通过，归一化坐标最大差1.1920929×10⁻⁷。初次测试脚本将参数变量覆盖为张量，导致结果写入失败；改正变量名后重新运行全部算子用例，保留失败日志，不将失败运行当作完成。

为判断C本身是否有收益，用相同A+B、相同优化绘图，按AB→ABC、ABC→AB、AB→ABC交替三轮，每轮56对预热＋560对测量：

| 组合 | 三轮完整均值 | P95 | Hz |
|---|---:|---:|---:|
| A+B | 12.945854 ms | 13.643986 ms | 77.244807 |
| A+B+C | 12.970714 ms | 13.685816 ms | 77.096759 |

C平均慢0.192%，且没有超过重复测量波动的稳定优势，P95也未通过。**最终不启用C**；代码作为显式实验开关保留，默认配置只有A+B。精度通过不能替代性能通过，也不通过减少图片/计算内容制造收益。不能仅凭此对照断言具体硬件瓶颈，最终另行记录核启动及分阶段信息。

## 最终三轮验收（A+B）

每个程序三轮各560对，合计各1680次完整双目处理；每轮先预热56对。顺序为当前→融合、融合→当前、当前→融合。正式测量关闭分阶段CUDA事件/钩子及算子追踪，使用相同优化绘图库。

| 指标 | 当前程序 | A+B融合 |
|---|---:|---:|
| 完整均值(ms) | 15.277357 | 12.980717 |
| P50(ms) | 15.239088 | 12.895873 |
| P95(ms) | 16.030006 | 13.604779 |
| P99(ms) | 16.699261 | 14.587519 |
| 最大单对(ms) | 19.158265 | 17.031961 |
| 完整Hz | 65.456348 | 77.037349 |
| 预处理＋完整推理＋回传(ms) | 11.773906 | 9.695662 |
| 运动学＋投影＋绘图(ms) | 3.503451 | 3.285054 |

完整均值下降15.033%，通过10%工程目标。三轮融合均值为12.991855、12.959813、12.990482 ms，全部低于所有对照轮次；P95也通过。绘图库完全相同，测得绘图段时间变化不能宣称为绘图算法修改的收益。

超过16.67 ms：当前18/1680，融合2/1680；超过33.33 ms均为0/1680。因此77 Hz是平均处理速率，不是所有帧均满足60 Hz时限的保证。

## 实现数值一致性

56对来自七段独立8帧片段；两侧J1/J2都统计。相对当前程序，C、杆体旋转、杆轴方向最大差为0；尖端最大差0.0000940245 mm，钳口部件旋转最大差0.0003557604°，左右投影最大差0.0006233726像素。

逐类别/整体的均值、P95和最大值、17个字段差、三个GRU状态差保存在`final_accuracy_detailed/report.json`。此结果是56对实现回归，不代表完整测试集。

### 状态和绘图

连续128次调用、主动重置、重复时间戳、时间间隔过大、序列切换、K/D变化、坏图拒绝及恢复全部通过。使用相同融合核的普通执行对照Graph，避免把浮点算子差异与状态推进错误混淆；故意制造捕获失败也正确告警并回退。正式三轮全部实际运行optimized_graph，无回退。

77次同预测绘图检查（56对实际图＋七类各黑/白/噪声）几何完全相同，最大通道差0。没有修改绘图库或文字/箭头/XYZ轴/透明面板内容。七类模型输出对照图上排当前程序、下排A+B，见`final_accuracy_detailed/*.jpg`。

### 资源与单独追踪

| 项目 | 当前程序 | A+B融合 |
|---|---:|---:|
| 整卡遥测峰值显存(MiB) | 769.000 | 595.000 |
| GPU平均利用率(%) | 59.833 | 58.481 |
| 平均功率(W) | 46.145 | 45.193 |
| 平均温度(°C) | 56.100 | 56.074 |
| 平均SM频率(MHz) | 2283.000 | 2308.333 |
| PyTorch分配峰值(MiB) | 153.709 | 132.657 |
| 进程CPU约占一个逻辑核(%) | 114.59 | 116.16 |

整卡遥测覆盖预热和测量并包含桌面/上下文，不能当作模型独占显存或仅正式帧的功耗；PyTorch统计不含TensorRT直接分配。

独立侵入式追踪14对显示：CUDA核数640.5→592.5／对（平均减少48个）；GPU核活动时间9.654→7.840 ms；预处理CUDA事件区间2.612→1.561 ms（包含上传）。融合核活动时间：水平缩放0.0550 ms、纵向缩放＋归一化0.0352 ms、掩码0.1957 ms、局部采样0.0196 ms。上述追踪有额外开销，不能相加替代正式端到端延迟，也不能用它推算完整Hz。

### 对照公平性与兼容修正

预处理回归和完整推理分别读取各自配置、执行各自预处理，诊断前向不再错误共用基线输入。测速工具增加双方绘图库参数及哈希一致检查；正式模式跳过CUDA事件、前向钩子和预处理分段计时。默认空`cuda_fusions`继续走当前程序，未覆盖原动态库。请求融合却缺失库会显式导入失败，未知开关拒绝。

## 复现命令

在4060现有推理工程根目录，使用现有Python3.11/PyTorch2.9.1+cu128/TensorRT10.13.3.9环境。CUDA12.8、SM89；不新增Python依赖。已有`inputs`内应有冻结的固定权重、56对图像清单及`accuracy_truth.json`，保留当前level5引擎及优化绘图库。配置均以自身目录解析相对路径。

```bash
python tools/build_cuda_fusions.py --cuda-root /usr/local/cuda-12.8

python tools/verify_gpu_preprocess.py --inputs inputs --baseline-config configs/inference_cuda_baseline.json --candidate-config configs/inference_cuda_preprocess.json --output runs/fusion_reproduce/preprocess.json
```

上述命令使用仅启用preprocess的专项配置，要求输入、17值和状态逐值一致。完整A+B有已记录的FP32局部插值微差，使用下面的完整数值阈值检查。

```bash
python tools/verify_cuda_fusions.py --output runs/fusion_reproduce/operators.json
python tools/verify_trt_visual.py --inputs inputs --baseline-config configs/inference_cuda_baseline.json --candidate-config configs/inference_cuda_fused.json --truth inputs/accuracy_truth.json --overlay-library reference/pose_display/native/build/optimized/libsurgical_overlay.so --output runs/fusion_reproduce/accuracy
python tools/verify_runtime_state.py --inputs inputs --baseline-config configs/inference_cuda_fused_eager.json --candidate-config configs/inference_cuda_fused.json --output runs/fusion_reproduce/state.json
python tools/benchmark_comparison.py --inputs inputs --baseline-config configs/inference_cuda_baseline.json --candidate-config configs/inference_cuda_fused.json --baseline-overlay reference/pose_display/native/build/optimized/libsurgical_overlay.so --candidate-overlay reference/pose_display/native/build/optimized/libsurgical_overlay.so --timing-detail minimal --output runs/fusion_reproduce/performance
python tools/profile_realtime.py --inputs inputs --mode operators --predictor-config configs/inference_cuda_fused.json --overlay-library reference/pose_display/native/build/optimized/libsurgical_overlay.so --output runs/fusion_reproduce/profile
```

每次使用新的输出目录；验证失败不覆盖旧报告。`cuda_fusions`支持preprocess、hires_crop和实验性landmark_reduce；最终只启用前两项，执行模式保持optimized_graph。构建脚本从原CMake缓存或显式参数读取依赖，输出到build/fused，原build/optimized和build/tensorrt不重新构建。

## 文件与证据

- 远端运行目录：`runs/cuda_fusion_20261111/`；A/B/C分别有编译日志、验证日志、数值报告与测速报告。
- 本地完整报告与七类图片：工作区`outputs/cuda_fusion_4060_20261111/`。
- 仓库小型证据：`docs/evidence/cuda_fusion_{A,B,C,final}_4060_20261111.json`。
- 不提交权重、引擎、图片、编译目标或大型原始追踪；追踪临时文件已由工具删除。保留当前运行库和一个融合库，C仅是同一融合库内默认关闭的开关。
- 实现提交：28f36d3（预处理）、c00db4e（局部裁剪）、b4a8910（关键点实验）。1908bc9为验证与复现入口，文档单独提交。

复现构建命令实际执行成功，融合库SHA256仍与正式测速记录一致。驱动580.95.05。完整源文件、配置及库哈希保存在`final_provenance.json`，摘要收入最终验收证据。
