# TensorRT视觉前端：4060实测

本页保留首次TensorRT接入的基线结果。后续逐步优化、当前配置和最终复测见[运行时优化记录](RUNTIME_OPTIMIZATION.md)。

2026-11-11，在4060笔记本实现并验证。只替换ConvNeXt V2 Nano、FPN、整器械分割头与五关键点热图头。后续局部特征、双目融合、三组GRU、C点FP64去畸变、腕部/钳口分支及C++绘图保留原实现。未重新训练，未改变输入尺寸或17值接口。

## 实现与精度

- 使用原固定权重，SHA256：`639fc8c7e56263f4ab889aa53cc5132e3610eea4c8b2d8b0db671c41d8a30325`。
- 引擎输入`images`为FP32 `[2,3,360,640]`；2表示左右目，不是两对图像。原始图像每目1280×720。
- 引擎输出FP32 `spatial [2,128,90,160]`、`mask [2,1,90,160]`、`heat [2,5,90,160]`，继续在同一CUDA流上交给LibTorch。输入布局转换也计入推理时间。
- 53个卷积/矩阵乘层指定FP16，其余敏感计算保留FP32；TensorRT引擎关闭TF32。下游原有FP32/FP64设置不变。不是整个模型转成FP16。
- 默认不传`visual_engine`时仍使用原版。新库位于`native/libtorch/build/tensorrt/stereo_libtorch_trt.so`，原`build/stereo_libtorch.so`没有覆盖。
- 训练权重、原版库和原C++绘图库的前后SHA256均一致。切换时校验引擎与检查点哈希，禁止误用另一检查点的引擎。
- 引擎由已有Python结构导出临时ONNX并编译，部署前向由C++ TensorRT API调用，不通过Python执行网络层。成功后删除临时ONNX；约33MiB引擎只保留在4060构建目录。

## 同机速度对照

七类各8帧，共56对独立保存的连续回归图片；每种方式预热56对、正式测量560对。序列内保留三组时序状态，每条独立序列重新初始化。测速期间未执行训练或渲染。

| 指标 | 原版LibTorch（复核轮） | TensorRT视觉前端 |
|---|---:|---:|
| 预处理、传输、完整推理与17值返回 | 26.076 ms | 21.110 ms |
| C++运动学、投影和三个向量绘图 | 5.446 ms | 5.372 ms |
| 总耗时均值 | **31.522 ms** | **26.482 ms** |
| 总耗时P95 | 32.690 ms | 27.504 ms |
| 双目处理速度 | **31.724 Hz** | **37.762 Hz** |
| 网络CUDA区间 | 23.153 ms | 18.214 ms |

总耗时减少约16.0%，吞吐提升约19.0%。TensorRT这轮560次中有1次超过33.33ms，最大34.794ms，因此不能保证每一帧都满足30Hz时限。原版先测一次均值31.703ms；安装编译结束后复核31.522ms，表中使用后者。绘图代码没有变化，小幅时间差属于测量波动。

计时从RAM中的RGB开始，包含GPU预处理、全部传输、完整模型、CPU结果读取、三个向量的数值/箭头/自身XYZ面板及最终画布。不含相机采集等待、图片解码、视频编码、窗口提交和显示器刷新。不是只测TensorRT引擎或ConvNeXt。

PyTorch显存峰值不含TensorRT直接分配的显存；不能仅凭PyTorch分配统计宣称显存下降。本轮TensorRT结束时整卡占用约701MiB。

本页记录的基线引擎与LibTorch共用原有默认CUDA流。TensorRT会提示默认流可能带来额外同步；本次测量包含该开销，未宣称已完成非默认流优化。

## 实现数值一致性

仅对应这56对测试图片，不代表完整测试集。

两种实现之间：C位置最大差0.00558mm，尖端最大差0.00729mm，杆体旋转最大差0.01614°，关节角最大差0.01087°，C/H/J投影最大差0.05576像素，分割二值掩码平均IoU为0.999869。两种实现各自维护独立GRU状态；不是只比较单帧特征。

转换检查阈值为位置最大差0.1mm、杆体旋转与关节最大差0.1°、投影最大差0.25像素，均通过。这些是实现转换的工程阈值，不是任务精度指标。检查同时输出特征/热图误差，没有要求FP16与FP32逐位相同。

## 依赖与构建

验证环境：Ubuntu22.04、RTX4060 Laptop 8GB、驱动580.95.05、Python3.11.18、PyTorch2.9.1+cu128、TensorRT10.13.3.9、NumPy1.26.4。在现有推理环境和项目根目录执行，增加依赖使用[requirements-tensorrt.txt](../requirements-tensorrt.txt)，不升级已有PyTorch。

```bash
python -m pip install -r requirements-tensorrt.txt
```

C++头文件来自NVIDIA官方`v10.13.3`提交`94e2b9ef6d2cce74c76cdad499cca36cc4949197`，保留许可证。下载到忽略的构建目录：

```python
from pathlib import Path
import urllib.request, re
root = Path('native/libtorch/build/tensorrt_include')
root.mkdir(parents=True, exist_ok=True)
base = 'https://raw.githubusercontent.com/NVIDIA/TensorRT/94e2b9ef6d2cce74c76cdad499cca36cc4949197/include/'
queue, done = ['NvInfer.h'], set()
while queue:
    name = queue.pop()
    if name in done:
        continue
    data = urllib.request.urlopen(base + name, timeout=30).read()
    (root / name).write_bytes(data)
    done.add(name)
    queue.extend(re.findall(r'#include [<"](Nv[^>"]+\.h)[>"]', data.decode()))
```

```bash
TRT_SITE=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
cmake -S native/libtorch -B native/libtorch/build/tensorrt \
  -DCMAKE_BUILD_TYPE=Release -DSTEREO_WITH_TENSORRT=ON \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DTENSORRT_INCLUDE="$PWD/native/libtorch/build/tensorrt_include" \
  -DTENSORRT_LIBRARY="$TRT_SITE/tensorrt_libs/libnvinfer.so.10" \
  -DCUDA_RUNTIME_INCLUDE="$TRT_SITE/nvidia/cuda_runtime/include"
cmake --build native/libtorch/build/tensorrt -j 2
python tools/build_trt_visual.py --checkpoint inputs/best_val.pt \
  --output native/libtorch/build/tensorrt/visual_fp16.engine --precision fp16
```

首次仍需按[LibTorch说明](LIBTORCH_INFERENCE.md)编译原版库（GPU预处理使用它）和[C++绘图库](NATIVE_OVERLAY.md)。引擎应在目标机器构建；更换检查点必须重新构建。构建器同时写`.json`版本/哈希清单，原版库保持独立。

## 验证与运行

```bash
python tools/verify_trt_visual.py --inputs inputs \
  --engine native/libtorch/build/tensorrt/visual_fp16.engine \
  --truth inputs/accuracy_truth.json --output runs/new_trt_check
python tools/benchmark_realtime.py --inputs inputs --output runs/new_trt_benchmark \
  --modes fp32 --inference-backend libtorch --preprocessing gpu \
  --display-mode vectors --rounds 10 \
  --visual-engine native/libtorch/build/tensorrt/visual_fp16.engine
```

`--modes fp32`表示未替换的LibTorch部分仍为FP32；视觉前端混合精度由`visual_fp16.engine`及其清单决定。原版对照使用相同命令并删除`--visual-engine`，输出到另一目录。

视频入口仍使用`stereo_pose.libtorch_predict:build_predictor`，新增可选配置`visual_engine`。4060已提供`inputs/predictor_tensorrt.json`，以配置文件所在目录解析路径：

```json
{
  "checkpoint": "best_val.pt",
  "fps": 60,
  "preprocessing": "gpu",
  "visual_engine": "../native/libtorch/build/tensorrt/visual_fp16.engine"
}
```

删除`visual_engine`即可选择原版，无需重新训练。参数`fps:60`是输入时序约定，不是实测处理速度。

结果见[对照证据](evidence/tensorrt_visual_4060_20261111.json)。4060详细结果在`runs/tensorrt_visual_20261111/`，预览上排原版、下排TensorRT，各排左右目。

技术来源：[NVIDIA C++推理接口](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/c-api-docs.html)、[精度注意事项](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html)。本实现固定使用10.13.3接口，不混用最新主版本的构建API。

