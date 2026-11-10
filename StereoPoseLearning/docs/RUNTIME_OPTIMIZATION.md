# 4060 双目推理优化：逐步实现、验证与结果

## 固定条件与测量口径

本记录按 A → B → C → D 的顺序展开。每步先完成实现、正确性检查和独立测速，再进入下一步；每步单独提交，保留机器可读证据。只在4060笔记本运行，没有改动服务器训练，也没有重新渲染或训练模型。

- 硬件：RTX 4060 Laptop 8GB；驱动580.95.05，CUDA工具链12.8.93。
- 环境：Ubuntu22.04、Python3.11.18、PyTorch2.9.1+cu128、TensorRT10.13.3.9。
- 权重：固定权重；SHA256 `639fc8c7e56263f4ab889aa53cc5132e3610eea4c8b2d8b0db671c41d8a30325`。
- 每目输入1280×720，主干640×360，局部裁剪160×160；单对双目、单时刻输入，三个GRU状态在序列内连续保留。
- 输出17值的定义、坐标系、运动学和三个向量的数值／箭头／自身XYZ轴／透明面板不变。
- 基线是先前通过验收的TensorRT视觉前端＋普通C++ LibTorch＋GPU预处理＋原C++绘图；基线源码提交为 `dbe9ffb`，原库和引擎保留，没有覆盖。

**完整耗时**：RAM中的左右RGB → GPU预处理 → 完整模型 → 17值回传 → 运动学与投影 → 标注画布完成。不包含读盘、图片解码、相机采集等待、编码、窗口提交和显示器刷新。

它是样本回归检查，**不是完整测试集精度**。

## A：合并 C 点计算，缓存固定内容

### 为什么这样修改

原 `position()` 每帧重新求K的逆矩阵，并用大量零碎ATen操作对两个C点执行10次牛顿迭代。固定ROI网格、归一化常量和尺寸向量也反复构造；腕部／钳口辅助热图即使不用于17值输出或时序状态，仍在计算。

### 如何实现

1. 新增独立 `STEREO_OPTIMIZED` 构建，生成 `stereo_libtorch_opt.so`，不替换原库。
2. `position_cuda.cu` 将逆相机变换、10次去畸变迭代、行列式保护及深度缩放合为一个CUDA核。中间量保持FP64，最终存为FP32；使用 `--fmad=false`，不开启快速数学。
3. 相机配置键变化时才更新左右K逆矩阵和D，更新现有缓存内容，避免改变以后被图捕获的地址。
4. 按模型缓存固定网格、ROI半径、归一化和尺寸常量；参考构建仍执行原表达式。
5. 普通优化推理跳过仅用于调试的局部辅助热图；调试路径保留。主关键点热图和整器械分割仍参与推理。

### 遇到的问题及处理

- 首次构建错误地假定Conda环境中有CMake。核查后使用系统 `/usr/bin/cmake`，Python仍使用现有推理环境，没有更换环境。
- NVCC不接受在设备函数中调用 `std::numeric_limits<double>::epsilon()`。改用数值相同的编译期 `DBL_EPSILON`，仍乘以8；没有通过放宽精度解决。
- 第一组测速末尾可能与状态检查重叠，因此不采用它作为正式结果，重新顺序执行独占测速。

### 检查与结果

- `verify_position_cuda.py` 检查左右相机、图像边缘与内部位置、零／实际畸变，以及10/30/100/300/1000 mm深度，共720点。最大差 **0 mm**，通过0.0001 mm阈值。
- 56对真实预测的17值与当前TensorRT基线逐值一致；C、铰链、尖端、旋转、角度及投影差均为0。

- 连续状态、主动重置、重复时间戳、时间间隔过大、序列切换、K/D变化、坏图拒绝及恢复均通过。

| 独占测速，预热56对、测量560对 | 总均值 | P95 | Hz |
|---|---:|---:|---:|
| TensorRT基线 | 26.5411 ms | 27.8082 ms | 37.6775 |
| A完成后 | 20.6431 ms | 21.1289 ms | 48.4423 |

A组合修改使均值下降22.22%；不能将全部收益归因于CUDA核单项。另与原始纯LibTorch版比较，也通过原转换阈值。证据：[A记录](evidence/runtime_optimization_A_4060_20261111.json)。

## B：固定缓冲区与 CUDA Graph

### 为什么与如何实现

A完成后，剩余网络仍逐个提交ATen操作。B捕获固定形状的完整原生网络，预处理及最终CPU回传留在捕获外。Python负责已有接口和捕获调度，网络计算仍由C++ LibTorch／TensorRT执行。

- 优化版TensorRT复用三个输出缓冲区；原构建仍返回独立张量。优化版调试结果应在下一次调用前消费。
- 预处理和推理在同一非默认流执行，建立固定图像、K、D、历史状态和高分辨率RGB缓冲区。
- 三次预热用零临时历史；捕获不推进外部历史。每帧写入真实输入和上一帧状态后，只重放一次。
- 原有时间／序列／相机变化重置规则继续生效；相机缓存原地更新。
- 捕获失败会警告，并记录 `actual_execution=optimized_eager` 和原因，不将回退冒充CUDA Graph成功。

### 检查与结果

实际运行模式为 `optimized_graph`，没有发生回退。总均值 **19.7608 ms**，P95 **20.3987 ms**，**50.6054 Hz**。这是固定缓冲区与图执行的组合收益，不把单独换CUDA流宣传为主要加速来源。

最终还增加了128次连续调用检查：循环使用已有8帧，保持连续时间和编号，确认状态缓冲区长期复用正确；这不是新增准确率样本。故意制造一次捕获拒绝，验证明确回退及预测正确。使用同一最终引擎的普通执行作为状态对照，避免把引擎数值变化混入状态检查。

证据：[B记录](evidence/runtime_optimization_B_4060_20261111.json)。

## C：缓存面板，左右目并行绘图

### 为什么与如何实现

原绘图约5.4 ms，左右串行，每目三个透明面板重复计算。新增独立 `STEREO_OPTIMIZED_OVERLAY` 构建，C接口保持兼容。

- 两个常驻工作线程分别绘制左右半幅，各持有独立FreeType／文字缓存；两侧都结束才返回当前帧。
- 缓存圆角填充掩码和逐字节混合表。边框缓存由原OpenCV绘制顺序生成的逐通道查找表，保留抗锯齿交叠处的整数舍入，而非换用近似透明遮罩。
- 每线程最多缓存16种样式，超出后调用原逻辑；线程退出释放缓存。正常三向量显示只有三种边框样式。
- 保留原字符缓存、动态数值格式、61点带畸变曲线、箭头尺寸／颜色和面板布局。没有隔帧绘制。
- 收集工作线程异常前先等待两侧结束，失败时不返回有效结果。诊断版两线程CPU绘制时间可以相加表示CPU工作量，但不能当作串行墙钟耗时。

### 检查与结果

56对已有图片，另对七类各测试随机噪声、纯黑、纯白背景，共77次对比：**像素通道最大差0，几何完全相同**。保存七类上下对照图，上排原绘图，下排优化绘图。

预热56对、测量560对：绘图均值 **3.8993 ms**；完整均值 **18.4561 ms**，P95 **19.2374 ms**，**54.1825 Hz**。进程CPU占用约114.7%，相当于平均使用1.15个逻辑核；峰值RSS约2367 MiB，包含PyTorch、权重、预载图片和缓存，不是显存。

证据：[C记录](evidence/runtime_optimization_C_4060_20261111.json)。

## D：仅比较两个 TensorRT 候选

### 候选定义与过程

1. **level5**：构建优化等级由3升至5，保持53个Conv/MatMul层FP16及其余精度约束。
2. **gelu**：在level5基础上，仅允许紧邻Conv的GELU运算使用FP16，共123层FP16。

第二项按ONNX依赖判断：激活组必须含Erf、只由Constant/Div/Erf/Add/Mul构成，而且唯一非常量外部输入来自Conv。LayerNorm、GRN、归约和几何计算不包括在内。没有合格层时构建直接报错，避免把未变化的引擎当作新实验。两者边界仍为FP32，TF32关闭。逐层请求精度和引擎检查报告保存在构建证据中。

| 候选 | 构建耗时 | 完整均值 | P95 | Hz |
|---|---:|---:|---:|---:|
| level5，原精度约束 | 96.34 s | 15.3075 ms | 16.1567 ms | 65.3273 |
| level5＋GELU FP16 | 109.25 s | 15.6916 ms | 16.4194 ms | 63.7285 |

**采用level5，拒绝额外GELU FP16候选**：后者更慢，没有理由扩大低精度范围。采用的引擎从 `candidate_level5.engine` 改名为 `visual_level5.engine`，内容和哈希不变；原 `visual_fp16.engine` 保留。

证据：[D记录](evidence/runtime_optimization_D_4060_20261111.json)。旧报告的 `original` / `tensorrt` 是比较槽位名称：A/B/D中前者指当前TensorRT基线；原LibTorch转换检查中前者才是纯LibTorch。新版报告增加显式槽位说明。

## 构建、选择与复现

在 `StereoPoseLearning` 目录、现有推理环境中执行。TensorRT头文件按 [TensorRT说明](TENSORRT_VISUAL.md)准备；原库用于基线和预处理，不能删除。

```bash
python tools/build_optimized_runtime.py --cuda-root /usr/local/cuda-12.8
cmake -S reference/pose_display/native -B reference/pose_display/native/build/optimized -DSTEREO_OPTIMIZED_OVERLAY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build reference/pose_display/native/build/optimized -j2
python tools/build_trt_visual.py --checkpoint inputs/best_val.pt --output native/libtorch/build/tensorrt/visual_level5.engine --builder-level 5
```

引擎存在时构建器拒绝覆盖。换机器应在目标GPU重新构建并重新验收；TensorRT计算方案选择可能导致重建引擎的字节哈希不同，不能承诺跨构建逐字节复现。

推理配置使用 `execution_mode`：缺省或 `reference` 保持旧行为；`optimized_eager` 选择A；`optimized_graph` 选择图执行。`configs/inference/optimized_final.example.json` 选择最终引擎。路径相对配置文件，17值接口不变。绘图库独立选择，保留原版默认值。

```bash
python tools/verify_position_cuda.py --inputs inputs --output runs/new_check/position.json
python tools/verify_trt_visual.py --inputs inputs --baseline-config inputs/predictor_tensorrt.json --candidate-config configs/inference/optimized_final.example.json --truth inputs/accuracy_truth.json --output runs/new_check/accuracy
python tools/benchmark_comparison.py --inputs inputs --baseline-config inputs/predictor_tensorrt.json --candidate-config configs/inference/optimized_final.example.json --candidate-overlay reference/pose_display/native/build/optimized/libsurgical_overlay.so --output runs/new_comparison
```

`benchmark_comparison.py` 顺序执行基线／最终版各三次；每次预热56对，正式560对。每组进程结束后才启动下一组。输出逐次报告、GPU遥测和汇总，正式测速不启用详细性能追踪。GPU遥测覆盖预热及测量，CPU使用率统计正式循环；PyTorch显存统计不含TensorRT直接分配。

这些是实现优化验收阈值，不是模型任务精度目标。


## 最终验收结果

基线／最终版交替三轮，各1680次正式双目测量。每轮最终版均快于所有基线轮次，P95也更低，实际执行模式全部为 `optimized_graph`，没有回退。

| 完整流程指标 | 当前TensorRT基线 | 最终优化版 |
|---|---:|---:|
| 总耗时均值 | 26.5158 ms | **15.2257 ms** |
| P50 | 26.4290 ms | 15.1962 ms |
| P95 | 27.5560 ms | 15.8574 ms |
| P99 | 28.3415 ms | 16.3335 ms |
| 最大值 | 36.6681 ms | 19.5506 ms |
| 双目处理速率 | 37.7133 Hz | **65.6786 Hz** |
| 超过33.33 ms | 1/1680 | 0/1680 |
| 预处理＋完整推理＋回传均值 | 21.1107 ms | 11.7157 ms |
| 运动学＋投影＋绘图均值 | 5.4051 ms | 3.5100 ms |

平均完整耗时降低 **42.58%**，达到至少10%的目标。最终版三轮均值为15.2542、15.2075、15.2152 ms；这不是只测前端引擎的结果。平均超过60 Hz不代表任何时刻都低于16.67 ms，最大单对耗时仍有19.55 ms；此次没有超过30 Hz所对应的33.33 ms，但不能保证未来永不超时。

左右目合并，尖端包括J1/J2；仅56对样本。下表为真实标签误差，**不是优化前后数值差**。

最终存在微小变化，不声称逐值完全相同。七类及整体的均值/P95增量均满足0.005 mm／0.005°要求。相比原始纯LibTorch，最大C差0.004911 mm、尖端差0.006031 mm、杆体旋转差0.012681°、投影差0.044234像素，也通过原转换上限。

128次连续状态复用及全部重置／异常恢复检查通过；使用最终预测的77次绘图比较仍为逐像素一致。捕获拒绝的故障注入正确回退，并明确记录模式，没有被当作图执行成功。

### 资源与保留内容

- 正式循环进程CPU占用：基线约100%，最终约115.5%；最终峰值RSS约2161～2163 MiB。
- PyTorch分配峰值：基线210.66 MiB、最终153.71 MiB；**不能据此宣称整卡显存降低**。
- 含预热的整卡遥测峰值：701 → 769 MiB；平均GPU占用约54.0% → 59.8%，平均功率45.29 → 46.18 W。此遥测包括桌面及TensorRT分配，采样区间与正式帧计时不同。
- 原权重、原LibTorch库、原TensorRT库／引擎、原绘图库的哈希均复核一致。
- 未采用的GELU引擎已删除，释放34,831,132字节；构建参数、逐层精度与结果仍保留。Git不包含权重、引擎、原始图片或构建缓存。

最终证据：[完整验收JSON](evidence/runtime_optimization_final_4060_20261111.json)。4060运行记录位于 `runs/runtime_optimization_20261111/`，七类绘图对照位于其中 `final_pixels/`；当前TensorRT与最终预测对照位于 `D_level5_accuracy/`。本地完整报告和预览同步到工作区的 `outputs/runtime_optimization_4060_20261111/`。

