# 环境、数据和推理接口

公开版保留可执行的模型及训练／评估源代码，但不描述实际训练轮次、续训过程或具体训练运行方案。数据统计见[数据说明](../../docs/DATA_PROTOCOL.md)，结构和输出见[模型说明](ARCHITECTURE.md)。不附带已训练权重和运行日志。

## 软件与外部输入

代码使用 Python 3.11、PyTorch 2.9.1、torchvision 0.24.1、timm 1.0.22；版本清单在 `requirements.txt`、`requirements-linux-cu128.lock.txt` 和 `environment.yml`。CUDA 部署另用 `requirements-tensorrt.txt`。训练／推理不需要导入 Isaac Sim；图像生成使用独立仿真环境。

外部文件包括左右 RGB、K/D、图像标签清单，以及与代码匹配的 checkpoint。固定器械几何随 `assets/instrument_model.json` 提供；更换场景／器械资产后必须重新核对几何和来源身份。

## 数学和数据接口检查

```bash
python -m unittest discover -s tests -p test_public_split_policy.py -v
# 安装对应依赖后，可执行整个源码测试套件：
python -m pytest tests -q
```

来源验证拒绝同轨迹跨集合、图片复制改名、原始标签与清单归属不一致。它不重分配数据，也不根据模型误差删除图像。测试计算契约不代表实际预测准确率。

## 单对双目与视频预测

```bash
./run.sh prediction --checkpoint /data/model/checkpoint.pt \
  --left /data/example/left.png --right /data/example/right.png \
  --calibration /data/example/calibration.json --output runs/prediction/pair
./run.sh video --help
```

calibration JSON 包含 `image:{width:1280,height:720,space:"raw_distorted"}` 和 `left/right:{K:3×3,D:[k1,k2,p1,p2,k3]}`。模型不需要相机间外参。视频维持三个因果状态，序列／相机配置变化或异常后重置。

C++ 绘图见 [NATIVE_OVERLAY](NATIVE_OVERLAY.md)；LibTorch、TensorRT 和 CUDA 构建见[性能索引](PERFORMANCE_INDEX.md)。公开目录没有重新运行这些历史 GPU 基准；用新权重、输入或设备时应重新测量。

## 源码范围

`data.py` 读取连续帧，`model.py`/`parts.py` 实现网络，`losses.py` 定义目标，`runtime.py` 提供训练和评价入口，`predict.py`/`libtorch_predict.py` 提供推理接口。保留这些实现用于代码审查和后续研究，不附加某次实际训练的步骤说明。
