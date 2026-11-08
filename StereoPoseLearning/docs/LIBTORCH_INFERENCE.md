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
