# 架构阶段与受控实验

五个 `stages/` 子目录保留真实阶段的 `stereo_pose/*.py`。`source.json` 记录源码哈希；配置是去掉机器路径的结构/训练模板，精确恢复使用原运行包的实际配置。

| 名称 | 主要用途 |
|---|---|
| baseline | 全局网格 token、共享关节头 |
| guided | 预测前景/关键点引导，2×2 局部表示 |
| local4 | 4×4 池化与局部编码 |
| foreground_parts | 原图腕部/钳口双分支 |
| legacy_c | 最终独立 X/Y 回归旧结构，用于转换对照 |

不要只把旧配置传给当前模型，结构和参数名可能不同。建立隔离工作目录：

```bash
python tools/materialize_stage.py --stage local4 --output /workspace/experiment_local4
```

## 诊断工具对应关系

历史脚本保留原协议，不能对不匹配的模型/数据强行运行并解释结果。当前整理没有重新执行这些耗时实验；报告中的历史证据保留来源文件哈希，新增发布测试另列，不混写为同一次实验。
