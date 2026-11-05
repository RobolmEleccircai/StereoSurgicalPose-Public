# 4080 / Isaac Sim 6.0 部署验证记录

见 [部署说明与实际启动入口](../../deployment/runtime6/README.md)。本目录记录 2026-11-09 的兼容验证，不是 2,100 对完整数据集验收。

- `runtime5_dependencies.json` / `runtime6_dependencies.json`：两个运行时的实际版本、资产依赖和几何统计；完整 63 状态数值原件保存在部署机的 `project/checks/runtime{5,6}_geometry.json`。
- `runtime6_geometry_comparison.json`：63 个同状态比较，27,421 个数值最大差为 0；不声明渲染像素完全一致。
- `runtime6_probe_validation.json`：两对隔离样本的图像/掩码/真值完整性和数值检查；目视检查为本次记录。
- `formal_4080_first_pair.json`：首对正式数据的独立检查和所属 worker。其通过不代表其余 1,747 对已生成或验收。

`visible_tip_mask_distance_px` 为空表示该帧没有“固定端点直接可见”的样本可用于这项检查，并非渲染或标签失败。器械区域掩码及尖端区域可见性另有记录；遮挡样本按设计保留。

实际渲染使用 Isaac Sim 6.0、原冻结规划使用 5.1。两个身份分别保存，不把 6.0 输出标为 5.1 渲染。适配器和启动器有独立哈希；共享核心、路径与分工身份保持不变。
