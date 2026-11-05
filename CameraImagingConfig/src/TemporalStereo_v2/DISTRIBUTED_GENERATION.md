# 4060 + 4080 Super 按完整轨迹分工

本版本新增分工执行与无渲染合并入口。**本次修改和测试不启动图片生成。** 总数、轨迹类别、训练划分、60 Hz 时间轴和成像质量不变；不增加可见性筛选，遮挡图片及完整真值继续保留。

## 固定分配

| worker_id | 机器角色 | 序列 | 对数 |
|---|---|---|---:|
| `rtx4060` | `laptop` | `seq_003`（光照训练 180）、`seq_013`（ECM 验证 32）、`seq_011`（遮挡测试 140） | 352 |
| `rtx4080super` | `server` | 其余 18 条 | 1,748 |

两份清单必须互斥且覆盖全部 21 条。合并后仍是训练 1,400、验证 350、测试 350 对。此比例是工作量安排，不是已经测得的显卡速度比。

一条轨迹的所有帧、左右目、掩码、真值和预览由同一个 worker 生成。`machine.worker_id` 与任务清单和 `role` 必须一致，不能把笔记本设置为 server 绕过限制。

## 新入口和文件

- `dispatch.py prepare`：冻结一份已检查的完整轨迹，生成不可变分配清单；不渲染。
- `generate.py --mode shard`：只渲染当前 worker 的完整轨迹。
- `generate.py --mode shard --prepare-only`：检查本机依赖并安装冻结轨迹，不加载渲染器。
- `generate.py --mode merge`：合并两个已经完成的 worker 输出，执行原有全量验收；不渲染。
- `deployment/machine.4060.example.json`、`deployment/machine.4080super.example.json`：机器配置模板。

旧的 `plan`、`validate`、`generate` 入口保留。`validate` 仍有原 32 对限制；`generate` 仍拒绝笔记本。笔记本的正式分工通过 `shard` 明确允许 352 个指定图像对，失败/损坏帧可重试，不会扩展成其他序列。已完成且文件校验正确的帧直接跳过。

## 准备同一个版本

以下命令以迁移包根目录为工作目录，包含 `code/`、`assets/`、`config/`、`deps/` 和 `fonts/`。两台机器都使用本版本完整 `TemporalStereo_v2/` 作为 `code/`；旧 `server_transfer_balanced_20261108.tar.gz` 中的代码不包含本功能，需要替换其 `code/`，资产可继续使用原包。

复制相应机器模板到 `config/machine.4060.local.json` 和 `config/machine.4080super.local.json`，设置各自真实 `isaac_root`。模板的相对目录按 `config/` 位置设计。保持 Isaac Sim 5.1.0 及已记录的 Python 依赖；SDK 安装、python-fcl 和字体说明仍见 [服务器迁移](SERVER_MIGRATION.md)。

这里的 `assets/settings.json` 指 **balanced 迁移包的 21 条配置**，不能替换为旧 `server_assets_20261108/settings.json`（14 条）。分工清单会拒绝旧配置。Ubuntu 新代码目录内已单独准备 `settings.distributed.local.json` 和 `machine.local.json`，对应当前 21 条配置及已有资产，可用于本机后续规划。

### 只规划一次并冻结

在 4060 上使用新代码生成一份完整规划。此步不渲染；4080 不重复规划。

```bash
python3 code/generate.py --mode plan \
  --machine config/machine.4060.local.json \
  --settings assets/settings.json --run-id distributed_plan_v1

ISAAC_ROOT="$HOME/isaacsim-5.1.0"
"$ISAAC_ROOT/python.sh" code/dispatch.py prepare \
  --plan-run outputs/distributed_plan_v1 --output dispatch_2100_v1
```

如果输出根目录另有配置，把上面的 `--plan-run` 改为真实路径。冻结包包含全部轨迹、相机/关节配置、代码和资产内容身份及任务分配，不包含 RGB。将同一个 `dispatch_2100_v1/` 完整复制到 4080。

旧代码生成的计划和旧 32 对诊断图片保留作为历史证据，不冒充本次正式任务。冻结计划要求当前代码版本，运行中不要编辑代码、轨迹或分工文件。

### 可先只检查两台机器的任务准备

在下列两条正式命令末尾加 `--prepare-only`，会核对本机依赖、复制共享轨迹、绑定 worker 输出目录，然后退出，不渲染。准备完成后去掉该参数，使用同一命令开始或恢复。

### 4060：352 对

```bash
python3 code/generate.py --mode shard \
  --machine config/machine.4060.local.json \
  --settings assets/settings.json --dispatch dispatch_2100_v1 \
  --worker rtx4060 --run-id worker_4060_v1
```

### 4080 Super：1,748 对

```bash
python3 code/generate.py --mode shard \
  --machine config/machine.4080super.local.json \
  --settings assets/settings.json --dispatch dispatch_2100_v1 \
  --worker rtx4080super --run-id worker_4080super_v1
```

两台可以同时运行。各自输出到本机 `output_root`；不要共同写一个 run 目录。每个 worker 完成后写入 `state: shard_complete`、`dataset_complete: false`。部分 worker 没有其他机器的训练图片，因此跨集合的最终覆盖验收在合并后执行；已有单帧真值、掩码、连续性和文件检查仍执行。

## 防重与恢复

- 冻结清单检查序列归属无重叠、无遗漏；采集循环再次限制到本机清单。
- 机器配置绑定 worker 身份；每帧保存 `render_worker`，合并时复核其所属序列。
- 同一输出根目录持有 worker 独占锁，并永久记录该 worker 对应的 run 路径。误换 run-id 不能重复领到同一任务。
- 进程意外退出后重跑相同命令，使用原目录、代码、配置、资产及冻结计划。合法文件直接跳过；损坏/缺失帧重试同一任务并记录尝试次数。
- 当前版本**不自动转移任务**。不能在运行中修改清单让另一台接手，避免两台同时生成同一轨迹。此版本固定执行 352 / 1,748；需要重新分配时先停止相关任务，再设计明确的所有权交接。
- 锁不跨两台不相连的电脑共享；互斥来自冻结的不同 worker 清单。不要把同一个 worker 的配置及输出目录克隆到第三台机器重复运行。

资产身份比较使用实际文件内容、材质引用对应的文件内容及运行库版本；允许两台机器安装/输出路径不同。渲染结果不承诺跨 GPU 逐像素相同，原始生成机器信息和参数来源分别保留。

## 合并与最终索引

将 4060 的整个 `outputs/worker_4060_v1/`（包括 `.commits`、`config`、`trajectories` 和 `previews`）复制到 4080 的 `incoming/worker_4060_v1/`。不要只传 PNG，也不要覆盖另一台的输出。

```bash
python3 code/generate.py --mode merge \
  --machine config/machine.4080super.local.json \
  --settings assets/settings.json --run-id dataset_2100_merged_v1 \
  --shard-runs incoming/worker_4060_v1 outputs/worker_4080super_v1
```

合并前检查两个不同 worker、同一冻结计划、全部指定帧及文件哈希；重复、缺失、损坏或不同版本均明确报错。合并复制原始图像和标签，不改真值、不重新渲染。逐机器的来源清单存于 `config/workers/`。

`merge` 需要新的输出 run；合并中断时保留原始 worker 数据，换一个新的合并 run-id 重试。额外预留约一份完整数据集的磁盘空间。

只有原有完整验收通过后才写 `state: complete`、`dataset_complete: true`、`rendered_pairs: 2100`，并输出统一的 train/validation/test 索引及七类等权统计。两台分别完成不等于已经完成合并验收。

## 测试范围

`tests/test_dispatch.py` 使用小文件夹具检查分配、冻结、跨目录部署、错误拒绝、恢复和 2,100 对文件合并；夹具不是渲染图像，不作为成像效果或服务器性能证据。现有数据集策略测试继续覆盖原 1,400 / 350 / 350 划分、连续时间、开合、遮挡和原有覆盖规则。目标 4080 的 GPU 渲染实测仍待执行。

2026-11-09 已通过 23 项单元/策略测试、26 项实际场景几何/计算检查及当前 21 条配置的资产运行库检查，新增渲染为 0。详见 [测试记录](validation_evidence/distributed_20261109/README.md)。
