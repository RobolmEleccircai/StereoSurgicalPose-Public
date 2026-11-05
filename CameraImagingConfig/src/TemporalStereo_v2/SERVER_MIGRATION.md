# 服务器迁移与启动说明

**4080 实际部署更新（2026-11-09）**：当前云主机的 R595 驱动与 5.1 渲染启动不兼容。已并列安装并验证 6.0，保留 5.1 用于冻结规划和独立验收；未要求平台改驱动。此实例请使用 [6.0 兼容部署入口](deployment/runtime6/README.md)，下文原 5.1 命令仍作为基础环境说明。4060 保持 5.1。

**两台机器分工方案（2026-11-09）**：4060 生成 352 对，4080 Super 生成其余 1,748 对；使用新增 `shard` / `merge` 入口，详见 [分工生成说明](DISTRIBUTED_GENERATION.md)。下文的笔记本仅验证限制是原单服务器方案；本次明确授权的 352 对通过专门的分工入口执行，普通 `generate` 仍不允许笔记本全量生成。旧迁移包的代码需更新到本版本，两台共用同一冻结计划。本文其余资产和依赖安装步骤继续适用。

2026-11-09 明确：**自遮挡和组织遮挡是本数据集的设计内容。** 固定尖端不可见时保留图像、完整三维真值和遮挡标签，不要求增加直接可见样本，不因此修改轨迹或阻止生成。[固定尖端核查](validation_evidence/balanced_fixed_tip_audit/README.md)解释已有统计，其结论不是新增筛选门槛。此前将该统计描述为必须修复的覆盖缺陷不符合用户要求，本说明已纠正。生产生成逻辑、路径及旧迁移包未因本次文档整理而改变。

当前默认配置为 21 条独立轨迹：训练 1,400、验证 350、测试 350 对，总计仍为 2,100 对。具体分配、短片段时间与覆盖要求见 [DATASET_SPLITS.md](DATASET_SPLITS.md)。笔记本只检查全部路径并验证最多 32 对；本交付不启动服务器全量任务，服务器运行能力仍需在目标机器验证。旧的 14 段配置、样本和迁移包保留。

## 1. 迁移包与目录

Ubuntu 上的交付目录：`/home/leorobot/PoseProject/CameraImagingConfig/server_transfer_balanced_20261108/`。旁边的同名 `.tar.gz` 与 `.tar.gz.sha256` 用于整体迁移。代码也保存在私有 Git 仓库；**只克隆 Git 不包含约 633 MiB 的场景资产**。

解压后保持以下结构，可以放在任意有读写权限的服务器目录：

```text
server_transfer_balanced_20261108/
├── code/                  # 整个 TemporalStereo_v2：生成、碰撞检查、标签、绘图、验收
│   ├── generate.py
│   ├── planner.py / capture.py / verify.py
│   ├── tools/             # 包含不渲染的 server_preflight.py
│   ├── pose_display/      # 计算、投影、箭头/XYZ、视频接口
│   ├── deployment/        # 机器模板、已测依赖版本
│   └── validation_evidence/
├── assets/
│   ├── scene.usdc         # 已组合的器官 + 单 PSM + ECM + 双目相机
│   ├── assets/            # 本地纹理、UDIM、材质依赖
│   ├── network/           # 已下载的自定义 MDL 与贴图
│   ├── calibration.json   # 成像内参与畸变的来源配置
│   ├── appearance.json    # 已确认的成像外观配置
│   ├── settings.json      # 此迁移包应使用的生成参数
│   └── bundle_manifest.json
├── fonts/                 # Noto 中文字体及许可证
├── wheelhouse/            # python-fcl 0.7.0.11 的 Linux x86_64 / CPython 3.11 wheel
├── config/machine.server.example.json
├── release.json           # 代码提交、资产/依赖来源、打包范围
├── MANIFEST.sha256        # 解压后的静态文件校验
└── SERVER_MIGRATION.md
```

`deps/`、`outputs/`、`checks/` 在服务器上创建。笔记本的 `.deps`、Conda 环境、旧生成结果、旧路径缓存、SSH 凭据和 Isaac Sim 安装本体不放入迁移包。

源资产来自既有 `server_assets_20261108`：76 个资产清单文件，共约 663,412,380 bytes（含清单约 632.7 MiB）。器官/机器人 USD 已组合为 `scene.usdc`，所以本生成入口不需要另外安装 ORBIT-Surgical、SuFIA-BC 或 Isaac Lab 的 Python 包。原始工程可保留用于今后编辑模型；这次运行不依赖它们的原路径。清单里保留的旧路径是来源记录，运行时重新解析交付包中的路径。

## 2. 服务器运行环境

原交付针对 **Ubuntu 22.04/24.04、Linux x86_64、Isaac Sim standalone 5.1.0、其自带 Python 3.11**。本次 4080 实例经过单独验证，使用上方链接的 6.0 运行适配；其他机器不能仅凭版本号假定兼容。

GPU 必须支持 RTX 渲染/RT Cores。NVIDIA 的 5.1 要求页列出最低 16 GB 显存、32 GB RAM、50 GB SSD，并明确不支持没有 RT Cores 的 A100/H100；**训练服务器有 CUDA GPU，不代表能运行本项目的路径追踪**。页面所列 Linux 已测驱动为 580.65.06，选用与目标 GPU 兼容的官方驱动。[NVIDIA 系统要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)

Isaac Sim 5.1.0 的二进制安装包由服务器单独安装；本项目通过 `isaac_root/python.sh` 使用其运行库。NVIDIA SDK 自带的标准 MDL 仍由该安装提供。部署方法可参考 [官方 standalone 安装说明](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_workstation.html)。网页目前标注该旧版本已不再维护；这里锁定 5.1.0 是为保持现有验证条件，换版本需另做兼容性验证。

当前生成器是 **headless 单进程、单 GPU**，无需打开 Ubuntu 桌面或远程 GUI，可在 SSH 中运行；不会自动使用服务器所有 GPU。不要同时启动多个进程写同一个 run。

依赖分为两部分：

| 依赖 | 已验证版本 | 处理方式 |
|---|---|---|
| Isaac Sim / USD / RTX / Replicator / Camera API | 5.1.0 | standalone 安装提供 |
| NumPy / SciPy | 1.26.0 / 1.15.3 | 使用 Isaac 自带版本 |
| trimesh / Pillow | 4.5.1 / 11.3.0 | 使用 Isaac 自带版本 |
| OpenCV 模块 | 4.11.0 | 使用 Isaac 自带 cv2，不根据 pip 包名判断缺失 |
| python-fcl | 0.7.0.11 | 从随包 wheel 安装到独立 `deps/` |
| 中文字体 | NotoSansCJK-Regular.ttc | 已随包提供，模板指向 `fonts/` |

不要在模拟器环境安装 `requirements-display.txt`，那是独立推理/绘图环境的依赖清单。生成器只补装 `requirements-collision.txt` 中的 python-fcl，避免覆盖模拟器已测依赖。无需复制或激活笔记本的 `isaaclab51` Conda 环境。

按当前 14 对小样本的原图、标签和预览占用线性估算，2,100 对约 9.1 GB，另有轨迹、视频与缓存；建议为输出单独预留至少 30 GB。该值是存储预算，不是大小上限，也不包含 Isaac Sim 安装空间。

## 3. 传输、校验与机器配置

在笔记本上把下面命令中的 `SERVER` 改为服务器用户名和地址：

```bash
rsync -avP \
  /home/leorobot/PoseProject/CameraImagingConfig/server_transfer_balanced_20261108.tar.gz \
  /home/leorobot/PoseProject/CameraImagingConfig/server_transfer_balanced_20261108.tar.gz.sha256 \
  SERVER:~/
```

在服务器接收目录运行：

```bash
sha256sum -c server_transfer_balanced_20261108.tar.gz.sha256
tar -xzf server_transfer_balanced_20261108.tar.gz
cd server_transfer_balanced_20261108
sha256sum -c MANIFEST.sha256
mkdir -p deps outputs checks
cp config/machine.server.example.json config/machine.server.local.json
```

修改 `config/machine.server.local.json` 的 `isaac_root` 为服务器真实安装路径。其余相对路径以该 JSON 文件所在的 `config/` 为基准，可直接保持：

```json
{
  "role": "server",
  "isaac_root": "/opt/isaacsim-5.1.0",
  "asset_root": "../assets",
  "output_root": "../outputs",
  "deps_root": "../deps",
  "expected_isaac_version": "5.1.0",
  "preview_font": "../fonts/NotoSansCJK-Regular.ttc"
}
```

如果数据要写到大容量盘，只修改 `output_root` 为该盘上的绝对目录。不要移动正在生成的 run，或混用另一套机器/资产路径继续旧缓存。不要把笔记本改成 `role: server` 来绕过渲染限制。

## 4. 安装唯一额外 Python 依赖

仍在交付根目录，`ISAAC_ROOT` 与机器 JSON 中的路径保持一致：

```bash
ISAAC_ROOT=/opt/isaacsim-5.1.0
"$ISAAC_ROOT/python.sh" -m pip install --no-index --no-deps \
  --find-links ./wheelhouse --target ./deps \
  -r code/requirements-collision.txt
```

随包 wheel 适用于 Linux x86_64 / CPython 3.11。其他架构不属于本次已核对迁移配置。不要把 `pip install` 安装目标换成系统 Python 或 Conda。

## 5. 只检查，不生成图像

下面入口不启动 SimulationApp、不使用 GPU 渲染，也不触发全量生成：

```bash
python3 code/tools/server_preflight.py \
  --machine config/machine.server.local.json \
  --settings assets/settings.json \
  --output checks/server_preflight.json
```

检查项目：Isaac/Python/数值依赖版本、全部资产哈希、材质引用是否只依赖交付包和 SDK、器官/PSM/ECM 几何、FCL 实际查询、字体和视频编码器、输出目录可写性，并记录 GPU 和驱动。报告的 `passed` **仅表示这些无渲染检查通过**；`server_render_validation_pending: true` 明确表示服务器 GPU 渲染尚待检查。重复检查需使用新的报告文件名。

进一步做失败注入和轨迹检查，也不会渲染：

```bash
python3 code/generate.py --tests-only \
  --machine config/machine.server.local.json \
  --settings assets/settings.json --run-id server_unit_checks

python3 code/generate.py --mode plan \
  --machine config/machine.server.local.json \
  --settings assets/settings.json --run-id dataset_2100
```

`plan` 会规划全部路径、检查碰撞和帧间安全，然后停止，不输出训练图片。它可能耗时较长。规划首先验收全部 1,400/350/350 分配、训练覆盖和跨集合近似重复。未来全量入口本身也会执行规划，因此并非必须手动重复运行一次 `plan`。

实际服务器首次启用路径追踪之前，还需验证 NVIDIA/Vulkan/RTX 能工作。可以使用官方兼容性工具，或在明确开始服务器渲染验证时运行 `--mode validate --run-id server_render_check`（最多 32 对）；**这属于会生成少量图像的单独命令，本次没有执行**。资产离线齐备不保证 Isaac 扩展在一台全新机器上完全不访问网络，首次运行应按其安装要求准备依赖及缓存。

## 6. 未来一次启动全量生成

**下面仅交付命令，本次不执行。** 服务器环境确认后，在 SSH 的 tmux/screen 等持久终端会话中运行，避免会话断开影响任务：

```bash
python3 code/generate.py --mode generate \
  --machine config/machine.server.local.json \
  --settings assets/settings.json --run-id dataset_2100
```

一个命令依次规划、采集、验收七类、每类训练/验证/测试各一条独立轨迹，共 21 条不同长度的片段＝**2,100 对、4,200 张原始 RGB PNG**。每目 1280×720，同步 60 Hz 时间戳；60 Hz 是采样时间轴，不承诺路径追踪每秒渲染 60 帧。七类路径、允许范围内的连续开合、接触/滑动/遮挡统一生成，不用按钳口状态手工分开执行。

必须显式传 `--settings assets/settings.json`。`code/settings.json` 保留原工程资产路径，不能在交付目录中直接省略该参数。现有包保持 PathTracing、64 spp × 4 subframes、最低线性亮度 0.035、整体系数 0.75，保留当前内参、畸变、固定点/尖端区域可见性、过曝标记及完整部件姿态标签。

中断后，在代码、资产、配置、依赖及路径均未变化的情况下，重新运行同一命令恢复；已提交且哈希正确的图像对会跳过，缺失/损坏项会重新处理。改变上述条件须换 `run-id`，不能手动改旧 manifest 来绕过指纹检查。

## 7. 输出和完成判据

```text
outputs/dataset_2100/
├── config/             # 实际使用配置、来源和运行依赖
├── trajectories/       # 全路径、关节和安全检查
├── sequences/seq_xxx/  # images、masks、labels、camera_truth
├── previews/           # 每帧箭头/XYZ 标注图、最新完成帧网页、60 fps 双目视频
├── indices/            # train / validation / test 索引、七类等权权重及实际完成状态
├── validation/         # 验收、覆盖、过曝与可见性记录
└── status.json
```

批量生成期间 `previews/live.html` 展示最近完成的一对标注图。这里的刷新频率取决于实际渲染速度。现有 `*_stereo_60fps.mp4` 是原始双目图像视频；若要箭头叠加视频，可在采集完成后调用 `pose_display.replay`，不必重新渲染。

真正完成必须同时满足 `status.json` 的 `state: complete`、`dataset_complete: true`、`validation_complete: true`，以及 `validation/report.json` 的 `passed: true`、`rendered_pairs: 2100`。只有 `planned`、进程退出、生成了部分 PNG，均不代表全量完成。规划不可达、资产缺失或验收失败会明确报错，不静默减少数量或降低画质。

## 8. 当前已做与未做

- 历史回归：原资产包哈希核对、旧版轨迹检查、32 对原验证和 14 对质量标签验证。新平衡划分的独立证据位于 `code/validation_evidence/balanced_2100/`；以其中实际报告为准，不把旧报告当作新划分已通过的证明。
- 本次新划分：36 项测试、21 条完整路径、全部 2,100 个规划状态的计算复核、32 对实际采集验收均通过；重复运行新增渲染次数为 0。固定尖端直接可见为 0/128，尖端区域可见为 124/128，过曝样本为两对。少量图片的特征分布覆盖报告明确标为部分采样，完整图像分布仍需全量生成后验收。
- 未做：目标服务器 GPU/驱动/显存实测；目标服务器渲染验证；全量生成、完整耗时测量及最终 2,100 对验收。
- 此次服务器迁移不增加组织形变/受力模拟，不训练位姿模型；接触仍为现有无穿透表面接近/滑动定义。
