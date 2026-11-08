# LibTorch 直接推理

按现有网络结构用 C++ LibTorch 实现 eager 前向计算，加载原 `best_val.pt` 中的同一份参数。不导出计算图，不使用 TorchScript、ONNX、TensorRT、AOTInductor、量化、算子融合或自定义 CUDA kernel。FP32、原有FP64去畸变计算、输入分辨率及算子语义保持不变。

现有 Python 包装层负责一次性的 checkpoint 读取、输入交接及流状态管理；权重张量直接交给 C++ 类，不生成第二份权重文件。每帧网络前向与17值计算在 C++ 执行。默认预处理也在 C++ LibTorch 中使用 GPU 完成；原 CPU 预处理保留为对照。投影绘图继续使用已验证的 C++ 模块。这不是“整个应用已改为纯C++”的声明。

## 文件与编译

- `native/libtorch/model.cpp` / `model.h`：当前ConvNeXtV2 Nano、FPN、双目注意力、三组GRU、腕部/钳口分支、C射线深度与17值计算。
- `native/libtorch/bindings.cpp`：张量接口，只传递参数与输入输出；前向期间释放Python GIL。
- `native/libtorch/preprocess.cpp` / `preprocess.h`：固定1280×720到640×360的GPU预处理，原始双目RGB只上传一次。
- `stereo_pose/libtorch_predict.py`：预分配锁页内存、相机参数缓存；复用原 `StereoPredictor` 的时间戳检查、状态重置和结果传回。
- `tools/verify_libtorch.py`：逐层、连续帧、运动学及标注图一致性检查。
- `tools/verify_gpu_preprocess.py`：原CPU预处理与GPU预处理的像素、输出、连续帧状态和标注图对照。

当前定义对应 `convnextv2_nano.fcmae_ft_in22k_in1k`、特征128、隐状态256、每次一对图像和一个时刻；全局输入640×360，原始局部输入1280×720。更换网络结构后需同步修改C++定义并重新验证，不能把新结构权重直接当作兼容模型。

在4060的 `~/PoseProject/StereoPoseInference_20261111/` 运行：

```bash
conda activate stereo_pose_infer311
cmake -S native/libtorch -B native/libtorch/build \
  -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$(which python)"
cmake --build native/libtorch/build -j 2
python tools/verify_libtorch.py --inputs inputs \
  --output runs/new_libtorch_check/verification.json
python tools/verify_gpu_preprocess.py --inputs inputs \
  --output runs/new_gpu_preprocess_check/verification.json
python tools/benchmark_realtime.py --inputs inputs \
  --output runs/new_libtorch_benchmark --modes fp32 --rounds 3 \
  --inference-backend libtorch --preprocessing gpu
```

链接环境中PyTorch2.9.1+cu128自带的LibTorch，使用相同C++ ABI。没有安装另一套LibTorch或CUDA工具链。视频入口可使用工厂 `stereo_pose.libtorch_predict:build_predictor`，配置为原checkpoint路径及fps，可选 `preprocessing: "gpu"`（默认）或 `"cpu"`，绘图后端继续为cpp。CPU预处理对照使用 `--preprocessing cpu` 和独立输出目录。

## 2026-11-11 验证与测速

固定权重最佳验证权重，SHA256 `639fc8c7e56263f4ab889aa53cc5132e3610eea4c8b2d8b0db671c41d8a30325`。七类各8对连续图像共56对，没有使用姿态真值或双目外参作为输入。

56对样本中，四级主干、FPN输出、关键点、局部热图、C/旋转/关节、17值及GRU状态的最大绝对差均为0。由输出计算的C/H1/H2/J1/J2/T、关节角和最终C++标注画布也一致；重复时间戳仍正确触发状态重置。这说明本次样本上的实现一致，不代表模型相对真值的误差为0。

在同一台RTX4060 Laptop GPU上先重新测Python基线，再测LibTorch；每个版本预热56对、计时168对，batch=1、连续帧保留状态、FP32、相同预处理与绘图。

| 指标 | Python前向 | LibTorch C++前向 |
|---|---:|---:|
| 预处理＋传输＋推理均值 | 44.24 ms | 41.41 ms |
| C++运动学、投影及绘图均值 | 6.40 ms | 6.42 ms |
| 完整流程均值 | **50.64 ms** | **47.84 ms** |
| 完整流程P95 | 52.37 ms | 49.41 ms |
| 双目对/秒 | **19.75 Hz** | **20.90 Hz** |
| 网络CUDA计时间隔均值 | 25.81 ms | 23.25 ms |

仅替换C++前向后，完整流程耗时减少约5.5%，尚未达到30Hz。CUDA计时间隔不等于完整链路时延，不能用其倒数替代20.90Hz。总计时包含720p双目输入的预处理、传输、模型、17值回传、物理箭头和XYZ面板；不含读盘解码、相机采集等待、编码、窗口展示或显示器刷新。

汇总证据见 [LibTorch对照报告](evidence/libtorch_4060_20261111.json)。笔记本逐帧记录在 `runs/libtorch_20261111/`，一致性检查在 `runs/libtorch_check/`。4090训练未修改、未中断。

## GPU预处理：2026-11-11更新

本节是当前默认路径；上节记录首次只替换模型前向时的结果。

1. 在初始化时分配可复用的锁页内存和GPU原图缓冲区。每帧CPU仅把两目RGB复制到锁页缓冲区，不再缩放、转FP32、归一化或转CHW。
2. 一次异步H2D传输上传两目原始uint8 RGB，共5,529,600字节；原路径还上传缩小后的FP32图像，图像传输总量是11,059,200字节。
3. GPU执行可分离双线性抗混叠缩放、归一化和布局转换；同一份GPU原图也供高分辨率局部分支使用。没有改变网络计算、权重、精度、裁剪方式和输入分辨率。
4. 缩放保持Pillow的22位整数权重、边界权重归一化和每个方向取整。归一化使用GPU上的256×3常量查找表，准确保留原NumPy FP32三步运算的舍入结果。参照[Pillow 11.3重采样实现](https://github.com/python-pillow/Pillow/blob/11.3.0/src/libImaging/Resample.c)，没有直接替换成舍入语义不同的浮点插值。
5. K/D在数值发生变化时更新，通常不用逐帧上传。相机变更仍触发原有时序状态重置。
6. 原图传输和后续GPU工作在同一CUDA流中按序执行；CPU不在这些阶段逐步等待。读取17值用于CPU绘图时仍等待模型完成。事件保护锁页缓冲区的复用，包括失败后的重试。当前是一对图像在途，不声明已经实现多帧流水线重叠。锁页内存的复用约束参照[PyTorch异步传输说明](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)。

**一致性验证：**七类共56对连续图像，缩放uint8、归一化FP32、K/D、高分辨率输入、17值、GRU状态、C/H1/H2/J1/J2/T、关节角及C++绘图像素的最大差均为0。另验证6对合成图像（黑、白、随机、棋盘、梯度、边界），非连续内存输入、标定变化、重复时间戳和错误恢复。这是实现一致性，不是对姿态真值的精度结论。

**新一轮同机对照：**两种路径均使用LibTorch C++前向和C++绘图，各预热56对、计时168对，FP32、单对双目输入。计时包括异步任务真正完成后的17值回传及最终绘图，而非仅计CUDA任务提交。

| 指标 | CPU预处理 | GPU预处理（当前默认） |
|---|---:|---:|
| 预处理＋传输＋推理＋结果回传均值 | 45.41 ms | 26.00 ms |
| C++运动学、投影及绘图均值 | 6.47 ms | 6.23 ms |
| 完整流程均值 | 51.88 ms | **32.24 ms** |
| 完整流程P95 | 53.57 ms | **33.35 ms** |
| 双目对/秒 | 19.27 Hz | **31.02 Hz** |

完整流程均值减少约37.9%；当前168次计时中94.64%在33.33ms以内。平均速度超过30Hz，但不能声称所有帧均满足30Hz时限。冷启动、读盘解码、相机采集、视频编码及窗口刷新不在此计时范围内。CPU/GPU预处理对照以本节同轮测量为准，不能把不同时刻的CPU基线差异归因于代码收益。

`input_preparation_host_wall_ms`在GPU路径中仅表示CPU交接与提交耗时，不能误当作GPU预处理完成耗时。速度结论使用`end_to_end_ms`。CPU阶段没有为了计时插入额外的逐步GPU等待。

证据：[GPU预处理对照报告](evidence/gpu_preprocess_4060_20261111.json)。4060详细记录为`runs/cpu_preprocess_control_20261111/`、`runs/gpu_preprocess_20261111/`和`runs/gpu_preprocess_check/`。本次只在4060上验证推理，未操作4090训练。

## 当前流程与资源审计：仅三个向量显示

版本范围：本节记录在仓库基础提交 `fce52c9` 上修改并验证的4060工作区版本。本次上传范围为文档与测量报告；新增三向量显示和分析工具代码尚未随本次文档提交上传。因此，本节新增的 `--display-mode`、分析工具与诊断编译选项应在对应工作区执行，不能直接用于尚未同步这些实现的旧版代码。前文保留原已提交版本的对照结果。

用户明确的实时显示范围是：杆轴、两片钳口三个向量的数值、物理箭头和三个自身XYZ面板。当前默认`vectors`模式不绘制帧号、时间戳、说明、关键点文字或标题栏。每目面板数值分别为`d`（单位方向）、`J1-H1`与`J2-H2`（毫米），表达在对应相机光学坐标系中；这些位移不是相对于相机原点的尖端位置。全部C/H/J等结果仍可从17值计算，但不额外叠加到实时画面。

### 测量口径与总体结果

- 设备：RTX4060 Laptop 8GB，18个可用逻辑CPU；performance模式，原有FP32权重与FP64射线计算，cuDNN允许TF32的设置保持不变。
- 权重、56对原始720p双目图像、相机K/D与前述验证相同；七类各8对连续帧，保留因果GRU状态。
- 正式延迟与资源统计：预热56对，计时560对；未开启侵入式PyTorch Profiler。测量覆盖RAM RGB输入到最终2560×720双目画布。
- 网络细分：独立CUDA事件测量168对，记录阶段边界，不在阶段间插入同步。此轮网络合计23.04ms，总流程比无细分计时的旧显示对照多约0.10ms；阶段数字按近似值解读。
- 算子统计：另用14对图像开启Profiler记录形状、FLOPs估计及CUDA活动；这轮的CPU耗时有明显采样开销，不用于宣称部署速度。网络统计使用显示调整前的同一个未改动网络；最新整体结果使用三个向量显示。
- CPU绘图细分：七组姿态，预热70次，计时210次。与完整推理相分离，因此绘图细分均值与整链路5.44ms不必完全相等。
- 不含相机采集等待、读盘解码、视频编码、窗口提交与显示器刷新。

| 当前完整流程 | 结果 |
|---|---:|
| 平均耗时 | **31.285 ms/双目对** |
| 中位数 | 31.215 ms |
| P95 | **32.483 ms** |
| 最大值（560次中） | 33.833 ms |
| 平均处理速度 | **31.965 双目对/秒** |
| 33.33ms以内的比例 | 99.46% |

### 实际数据流

1. CPU接收两张1280×720 RGB uint8图像及各自K/D，检查图像、时间戳、流身份和标定变化。
2. CPU复制到可复用锁页缓冲区，整对原图一次异步上传GPU，共5,529,600字节。K/D仅改变时更新。
3. GPU缩放到640×360，保持原Pillow抗混叠和取整规则，进行归一化与布局转换；720p原图留在GPU给局部分支。
4. ConvNeXtV2 Nano的stem和四级主干提取特征，FPN生成融合特征、器械掩码和五个关键点热图。
5. 基于关键点抽取局部特征，融合相机内参与左右目特征，杆体GRU预测杆体旋转与深度。
6. 用预测C像素位置、深度及各自K/D，执行原有10次FP64去畸变迭代，计算左右C。
7. 用预测掩码与关键点，从GPU上的原始720p图像构建腕部和钳口160×160局部输入。两个局部分支分别完成图像编码、双目融合和GRU，预测腕部俯仰/偏航与钳口半开合角。
8. 拼成17值：`C_L(3), q_L(4), C_R(3), q_R(4), pitch, yaw, jaw_half`。仅68字节结果传回CPU；三组时序状态共6,144字节继续保留GPU。
9. C++根据固定器械几何解算H1/H2、J1/J2、T和各部件姿态，用对应相机K/D投影物理箭头与XYZ轴，复制原图并绘制面板、数值，输出双目画布。

### 各阶段时间

| 阶段 | 时间 ms/对 | 计时口径 |
|---|---:|---|
| CPU检查、准备锁页输入，至GPU提交前 | 0.578 | CPU墙钟 |
| 上传原图 | 1.032 | GPU事件区间 |
| GPU水平缩放 | 0.831 | GPU事件区间 |
| GPU垂直缩放 | 0.246 | GPU事件区间 |
| GPU归一化、布局转换 | 0.130 | GPU事件区间 |
| 网络整体 | **22.875** | 最新无细分记录中的GPU区间 |
| 返回等待和输出检查 | 0.094 | CPU墙钟，包含未完成GPU工作的等待 |
| C++完整后处理绘图 | **5.441** | CPU墙钟 |

GPU预处理总区间约2.241ms。CPU提交和GPU执行可以重叠，不能把CPU前向调用时间24.628ms再加到22.875ms网络GPU区间上。17值实际GPU到CPU传输在算子记录中约0.0016ms，不是0.094ms全部，也不是瓶颈。

| 网络模块 | GPU区间 ms/对 | Profiler支持算子的运算量估计 GFLOPs/对 |
|---|---:|---:|
| ConvNeXtV2主干（stem及四级） | **12.587** | 44.315 |
| FPN、掩码与关键点头 | 1.554 | 9.644 |
| 关键点引导局部特征 | 0.619 | 0.0324 |
| 左右目注意力融合 | 0.571 | 0.00637 |
| 杆体GRU与旋转/深度头 | 0.641 | 0.00106 |
| **C点射线去畸变与三维恢复** | **4.151** | 0.00000164（未覆盖全部几何运算） |
| 720p掩码处理与局部裁剪 | 1.474 | 0.00594 |
| 腕部分支 | 0.557 | 0.3273 |
| 钳口分支 | 0.491 | 0.3273 |
| 四元数转换、17值拼装 | 0.371 | 极小 |

主干四级细分：stem0.272ms，第0级5.431ms，第1级2.274ms，第2级3.758ms，第3级0.852ms。运算量最多的模块不必耗时最多：高分辨率的第0级有更多特征内存访问和逐元素处理。

Profiler总计约**54.675 GFLOPs/双目对**，包含其可统计的卷积、矩阵乘、部分加乘；归一化、插值、部分GRU/融合内核等没有被全部计算，不能把它称为完整硬件运算计数。可统计卷积约54.520 GFLOPs。参考[PyTorch Profiler说明](https://docs.pytorch.org/docs/2.9/profiler.html)。

### 绘图到底花在哪里

最新整链路中：运动学及畸变投影0.138ms，画布复制0.604ms，左/右目绘制2.357/2.270ms，包装与检查约0.071ms。

单独CPU绘图采样进一步拆解：

| 两目合计 | 平均 ms |
|---|---:|
| 箭头、抗锯齿折线及描边 | **2.905** |
| 六个半透明圆角面板及边框 | **1.519** |
| 三个向量数值及简短标识（每目） | **0.074** |
| 该轮完整绘图 | 5.133 |

因此，保留向量数值的成本很小。当前CPU绘图的大头是箭头光栅化和面板混合，三维运动学与数值格式化不是重点。

### 资源消耗

| 项目 | 当前测量 |
|---|---:|
| 模型参数 | **17,945,453（17.95M）** |
| FP32参数大小 | **68.46 MiB** |
| CUDA当前张量分配 | 93.18 MiB |
| CUDA张量峰值 | **251.91 MiB** |
| CUDA分配器保留池 | 330 MiB |
| nvidia-smi整卡显存占用 | 583 MiB（含上下文和桌面等） |
| GPU时间忙碌率 | 64.07% |
| 显存控制器忙碌率 | 42.10% |
| GPU平均功耗 | **49.01 W** |
| GPU平均SM频率 | 2521.9 MHz |
| GPU平均温度 | 57.1°C |
| 进程CPU占用 | 约1.00个核心，18逻辑CPU的5.56% |
| 进程RSS | 2057 MiB，约2.01 GiB |
| 其中预加载56对原图 | 295.31 MiB |

进程RSS含Python/LibTorch库、CPU权重加载分配、图片预加载及其它缓存，不是网络参数大小。CUDA张量、保留池、整卡占用是不同口径，不能相加。GPU忙碌率不是峰值FLOPs利用率，也不是SM占用率测量。没有因显存容量不足而需要减少模型的证据。

### 还能优化的明确方向

1. **C点几何计算的执行方式。** 仅左右两条射线，却以大量逐元素LibTorch操作执行10轮FP64迭代，正常事件区间4.15ms；Profiler所记实际内核活动约1.11ms，存在明显提交间隙。可以比较等价的C++ CPU两点计算与单个GPU内核，保留原公式、迭代、坐标定义和最终17值接口。必须做数值及端到端对照；本次没有替换此计算。
2. **预置模型内部常量。** 原图已只上传一次，但网络内仍有6次小常量H2D，总共56字节。对应逐帧小张量构造会产生同步；Profiler记录每对7次`cudaStreamSynchronize`（含末尾结果回传）。这些常量可以初始化时放到GPU复用，不改变计算定义。Profiler约11.7ms的CPU同步等待与已有GPU工作重叠，绝不代表缓存常量就能直接减去11.7ms。
3. **CPU绘图。** 优先研究左右目独立缓存后的并行绘制、圆角面板遮罩复用和箭头抗锯齿绘制。分别核对输出形状、数值、箭头曲线和视觉效果；当前只按用户要求减少显示内容，没有做这类优化。
4. **主干计算。** 主干占网络约55%的区间。以后可用LibTorch混合精度做单独对照，同时保留FP64几何，测C、腕部、开合及投影差异。收益需实测，不能仅根据硬件规格推断。
5. **推理与绘图流水线。** 可以让上一帧CPU绘图与下一帧GPU推理重叠，但这主要改善连续处理吞吐，不能冒充同一对图像从输入到画布的时延缩短；需保留帧身份、结果顺序与因果时序状态。

约1,740个GPU内核/双目对。仅靠增加batch会改变当前单对实时口径；目前应先处理以上开销。优化收益不可简单相加。

### 复测与证据

常规绘图编译使用前述命令。诊断库放在已有忽略目录`build/profile`，不会覆盖正式LibTorch库；`STEREO_PROFILE_RANGES`默认关闭：

```bash
cmake -S native/libtorch -B native/libtorch/build/profile \
  -DCMAKE_BUILD_TYPE=Release -DSTEREO_PROFILE_RANGES=ON \
  -DPython3_EXECUTABLE="$(which python)"
cmake --build native/libtorch/build/profile -j 2
cmake -S reference/pose_display/native -B reference/pose_display/native/build/profile \
  -DCMAKE_BUILD_TYPE=Release -DSTEREO_PROFILE_RANGES=ON
cmake --build reference/pose_display/native/build/profile -j 2
python tools/profile_realtime.py --inputs inputs --output runs/new_audit/measure \
  --mode measure --rounds 10 --display-mode vectors
python tools/profile_realtime.py --inputs inputs --output runs/new_audit/stages \
  --mode stages --rounds 3 --display-mode vectors
python tools/profile_realtime.py --inputs inputs --output runs/new_audit/operators \
  --mode operators --display-mode vectors
python tools/verify_vector_display.py --inputs inputs --output runs/new_audit/display_check
```

`measure`使用正式模型库；`stages`使用轻量事件边界；`operators`使用侵入式Profiler，三种结果分开保存。诊断阶段回调按单线程单推理流使用。大体积Chrome trace在提取统计后立即删除，不保存重复模型和数据。绘图验证覆盖七类：新旧显示模式的几何相同，与独立NumPy解码的最大差约1.42×10⁻¹⁴；改变帧号/时间戳不会改变向量模式像素；正式与诊断绘图库逐像素一致。

完整数值见[资源与计算审计证据](evidence/inference_resource_4060_20261111.json)。4060原始记录在`runs/resource_audit_20261111/`。本次未运行训练或Isaac Sim。
