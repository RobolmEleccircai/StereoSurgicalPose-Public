# 模型输入、预测值与器械运动学解码

本文规定当前单个 PSM、对称联动钳口的数据接口，并记录实际数值验证。它是模型接口设计及解码验证，**尚未实现或训练视觉预测网络**。

**遮挡是既定训练数据状态。** 自遮挡或组织遮挡时，器械的三维位置、姿态、关节角和派生点真值仍然保存，不因不可见而删除图像、将三维值清零或替换尖端身份。固定点可见性、尖端区域可见性分别作为标签；不要求固定尖端必须直接可见。

结论：左右相机坐标系下的杆体位置与完整旋转，加上共享的腕部俯仰、腕部偏转和钳口半开合角，可以通过已知器械模型计算 d、H1/H2、J1/J2、T 和两片钳口的完整姿态。这里的必要条件是：**器械型号、几何尺寸、关节定义与固定变换已知，并与当前 USD 模型一致**。17 个数值本身不包含这些固定几何尺寸。

## 1. 模型输入

| 输入 | 定义与来源 |
|---|---|
| `rgb_left`、`rgb_right` | 同一时刻的左右 RGB 图像。数据文件为每目 1280×720、带已记录畸变的原始 PNG，不使用带箭头的预览图训练。 |
| `K_left`、`K_right` | 对应当前输入图像尺寸的已知相机内参。作为相机条件输入，并供投影计算使用。 |
| `D_left`、`D_right` | 已知畸变，顺序为 `[k1,k2,p1,p2,k3]`。原始图像与投影采用同一畸变定义。 |
| 连续帧与时间信息 | 时序模型使用同一序列内连续 N 对图像及其时间信息，间隔 1/60 s。单帧模式为 N=1；时序窗口 N 尚未在网络中确定。 |
| 固定器械模型 | 下文的关节轴、固定变换、铰链锚点和尖端局部坐标。它们是解码器的模型常量，不是每帧预测值。 |

单帧图像张量可组织为 `(B,2,3,H,W)`，时序版本为 `(B,N,2,3,H,W)`，视图顺序固定为左、右。RGB 数值可归一化为 `[0,1]`。这里规定的是输入含义，不要求未来网络必须在 720p 原尺寸上运行。

如果缩放图像，应同步变换 K。对于宽高缩放比例 `sx,sy`，使用当前工程的像素中心约定：

\[
A=\begin{bmatrix}s_x&0&(s_x-1)/2\\0&s_y&(s_y-1)/2\\0&0&1\end{bmatrix},\quad K'=AK.
\]

单纯缩放时 D 不变。若改为去畸变图像，必须使用对应的新 K 和去畸变后的投影定义，不能再套原始图像的 D。

**不把以下真值作为视觉模型输入：**左右相机外参、baseline、器械关节角真值、H/J/C 的三维真值、仿真分割掩码。掩码可用于辅助监督，但实际推理不能依赖仿真真值掩码。ECM 的关节真值也不是当前相机坐标系下位姿预测的必需输入。

## 2. 网络直接预测的 17 个数值

采用单位四元数表示杆体完整旋转，分量顺序固定为 **`[x,y,z,w]`**：

| 数组索引 | 字段 | 个数 | 单位 / 含义 | 当前标签中的训练真值来源 |
|---|---|---:|---|---|
| 0–2 | `C_left_mm` | 3 | 左相机坐标系下的 C，mm | `eyes.left.tool.shaft_reference_position_mm` |
| 3–6 | `shaft_quaternion_left_xyzw` | 4 | 杆体局部坐标系到左相机坐标系的旋转 | `eyes.left.local_frames.shaft.T_optical_from_local_mm` 的左上 3×3，转换为四元数 |
| 7–9 | `C_right_mm` | 3 | 右相机坐标系下的 C，mm | `eyes.right.tool.shaft_reference_position_mm` |
| 10–13 | `shaft_quaternion_right_xyzw` | 4 | 杆体局部坐标系到右相机坐标系的旋转 | 右目对应杆体矩阵的左上 3×3 |
| 14 | `wrist_pitch_rad` | 1 | 腕部俯仰，rad | `joint_positions.psm.psm_tool_pitch_joint` |
| 15 | `wrist_yaw_rad` | 1 | 腕部偏转，rad | `joint_positions.psm.psm_tool_yaw_joint` |
| 16 | `jaw_half_angle_rad` | 1 | 每片钳口相对中心的半开合角 θ，rad | `joint_positions.psm.psm_tool_gripper2_joint` |

这是 **17 个输出数值，不是 17 个独立自由度**。四元数需要归一化；零四元数无效。`q` 与 `−q` 表示同一个旋转，旋转损失应按旋转本身或符号不变方式计算，不能把两者误认为不同姿态。

位置和旋转在左右相机坐标系下分别表达；三个内部关节角属于同一件器械，**只预测一组共享值**。当前生成器满足：

\[
q_{g1}=-\theta,\qquad q_{g2}=+\theta.
\]

因此对称开合总角度为 `2θ`，不需要重复预测两个自由开合角。该约束已在本次全部 14 对标签中检查。如果将来换成独立驱动钳口，应调整输出字段和对应模型，不能沿用该对称假设。

当前 USD 记录的关节范围为：腕部俯仰约 ±1.396299887 rad，腕部偏转 ±π/6，半开合角 0～π/6。解码器按导出的模型范围校验，不允许越限值被当作合法姿态。

原始数据保存的是旋转矩阵和关节状态；上述四元数及 17 维训练目标由数据加载阶段转换得到，不需要重新采集图像。[已有 14 对数据的目标转换示例](validation_evidence/model_io_decoding/targets_17.json)。

## 3. 固定模型参数从哪里来

模型常量直接从当前 USD 源资产读取，**没有使用每帧 H/J 坐标拟合参数**。导出结果见 [instrument_model.json](validation_evidence/model_io_decoding/instrument_model.json)，其中记录源场景 SHA-256。

从杆体 `psm_tool_roll_link` 开始的关节链是：

```text
杆体 S → wrist_pitch → wrist_yaw → jaw1
                              └→ jaw2
```

每个关节保存：

- 父、子部件身份与关节旋转轴。
- `A_parent_joint_mm`：父部件坐标系到其关节框架的固定变换 A。
- `B_parent_child_bind_mm`：源资产绑定状态下，子部件到父部件的固定变换 B。
- `bind_angle_rad`：绑定状态的关节角 q⁰。
- USD 关节限位。

每片钳口另保存：

- `hinge_local_mm`：该钳口部件坐标系下的真实关节锚点 h，由 USD 的关节 `body1/localPos1` 按实际尺度转换得到。
- `tip_local_mm`：该钳口表面固定尖端点 j，沿用数据集现有 CAD 表面尖端定义。

所有导出平移统一为 mm。关节旋转轴在关节框架内定义，因此必须使用 A；不能把关节角直接当作相机欧拉角，也不能忽略关节框架与钳口部件框架的固定旋转。

## 4. 从预测值计算三维量

记 `e` 为 left 或 right，相机坐标使用 X 向右、Y 向下、Z 向前。以下矩阵把右侧坐标系中的点变换到左侧坐标系。

### 4.1 杆体位置、完整旋转和方向 d

将四元数归一化并转成旋转矩阵 R，建立：

\[
{}^eT_S=\begin{bmatrix}{}^eR_S&C_e\\0&1\end{bmatrix},
\qquad d_e={}^eR_S\begin{bmatrix}0\\0\\1\end{bmatrix}.
\]

当前模型把杆体局部 +Z 定义为 d。R 同时包含杆轴朝向和绕轴自转；仅有 d 无法保留完整的轴向自转信息。

对于归一化四元数 `(x,y,z,w)`，旋转矩阵为：

\[
R=\begin{bmatrix}
1-2(y^2+z^2)&2(xy-zw)&2(xz+yw)\\
2(xy+zw)&1-2(x^2+z^2)&2(yz-xw)\\
2(xz-yw)&2(yz+xw)&1-2(x^2+y^2)
\end{bmatrix}.
\]

### 4.2 沿关节链计算两片钳口

每个关节 k 在角度 q 下的父子变换为：

\[
F_k(q)=A_k\,\operatorname{Rot}(a_k,q-q_k^0)\,A_k^{-1}B_k.
\]

其中 `Rot` 是围绕 USD 指定关节轴的齐次旋转，A、B、q⁰ 都是上节导出的固定模型参数。两片钳口部件的相机坐标系位姿为：

\[
{}^eT_{G1}={}^eT_S F_p(q_p)F_y(q_y)F_{g1}(-\theta),
\]
\[
{}^eT_{G2}={}^eT_S F_p(q_p)F_y(q_y)F_{g2}(+\theta).
\]

这个计算不读取相机世界位姿、左右相机外参、baseline 或保存的钳口位姿真值。

### 4.3 H1/H2、J1/J2、T 和钳口 XYZ

将 `{}^eT_{Gi}` 写成旋转 `R_i` 和平移 `t_i`：

\[
H_i^e=R_i^e h_i+t_i^e,\qquad J_i^e=R_i^e j_i+t_i^e,
\qquad T_e=\frac{J_1^e+J_2^e}{2}.
\]

- H1/H2 使用模型的真实关节锚点，不由尖端或图像轮廓猜测。
- J1/J2 是各自钳口上的固定表面点，身份不按图像中的上下位置交换。
- T 只是两尖端中点，不能替代两片钳口的独立状态。
- `R_i` 的三列分别是该钳口自身 +X、+Y、+Z 在对应相机坐标系中的方向。
- 画钳口 XYZ 时原点取 H_i，方向取 R_i 的三列；杆体 XYZ 原点取 C，方向取 R_S 的三列。

当前模型两片钳口的铰接位置基本重合，但计算中仍使用各自的 USD 锚点和部件变换。

## 5. 投影到左右图像

对每目独立使用自己的三维结果、K 和 D。三维点 P=(X,Y,Z) 必须满足 Z>0。先计算：

\[
x=X/Z,\quad y=Y/Z,\quad r^2=x^2+y^2,
\quad f=1+k_1r^2+k_2r^4+k_3r^6,
\]
\[
x_d=xf+2p_1xy+p_2(r^2+2x^2),
\quad y_d=yf+p_1(r^2+2y^2)+2p_2xy,
\]
\[
u=f_xx_d+c_x,\qquad v=f_yy_d+c_y.
\]

当前 K 的 skew 为零，使用上述五参数畸变投影。箭头对应的三维线段为：

| 箭头 | 三维起点 → 终点 |
|---|---|
| 杆轴 | `C → C + Ld`；现有主箭头显示长度 L=15 mm |
| 钳口1 | `H1 → J1` |
| 钳口2 | `H2 → J2` |
| 杆体 XYZ | `C → C + l R_S[:,i]` |
| 钳口 XYZ | `H_i → H_i + l R_i[:,j]` |

应沿线段采样再投影，因为畸变后的轨迹可能不是直线。XYZ 的显示长度 l、角落面板的位置与透明度是绘图参数，不是网络需要预测的物理量；原图中真实箭头的物理起点仍为 C/H1/H2。当前角落坐标轴面板只是平移后的姿态示意。

只要分别得到两目坐标系中的姿态，就可以分别投影；此步骤不需要先估计两个相机之间的关系。投影位置本身不决定可见性，实线/虚线仍应使用对应的可见性信息。完整场景遮挡和过曝也不能仅由这 17 个器械状态数值推出。

## 6. 哪些数据用于监督，而不是新增输出头

网络直接监督 C、杆体旋转、腕部角度和半开合角。通过上述可微或数值运动学得到的 d、H1/H2、J1/J2、T、钳口姿态，还可以与数据集真值比较，构成几何辅助损失；不必把这些关联量全部重复设为独立预测头。

当前提供的解码脚本是 NumPy/SciPy 数值核对版，不提供神经网络反向传播。训练时若把运动学结果纳入损失，需要在支持自动微分的训练框架中实现相同公式。

- H1/H2 真值来自每目 `local_frames.jaw1/jaw2.T_optical_from_local_mm` 的最后一列前三项。
- 钳口姿态真值来自相同矩阵的左上 3×3。
- d、J1/J2、T 真值来自每目的 `tool` 字段。
- 分割掩码可用于辅助分割监督或检查投影。
- 固定点可见性、尖端区域可见性与过曝标签可用于分层评估、采样和损失权重设计；三者不能互相替代。
- 被遮挡点仍有有效的仿真三维真值。C、H1/H2 等内部参考点不应伪造“表面可见”标签。

损失权重、视觉网络、时序窗口及推理速度尚未通过训练确定。图像中部分器械、反光和对称性可能造成姿态歧义；**能从给定状态值解码出几何，不代表已经证明可以从任意图像唯一、准确地估计这些状态值**。

## 7. 已执行的无渲染验证

使用本次 `quality_smoke_20261108` 的 14 对图像标签，共 28 个相机视图：

1. 先仅从 USD 导出并冻结固定模型参数。
2. 从每帧训练真值提取上述 17 个数。
3. 调用只接收 `decode(values17, fixed_model)` 的独立解码函数。
4. 将输出与已保存的 d、H/J/T 和钳口旋转真值比较。
5. 仅额外使用每目的 K/D，比较关键点及箭头、XYZ 轴采样线的投影。
6. 检查源数据文件未改变；没有启动 SimulationApp 或生成新 RGB。

实际最大差异：

| 量 | 最大数值差异 |
|---|---:|
| C | 0 mm |
| H1/H2 | 6.951 × 10⁻⁷ mm |
| J1/J2 | 1.629 × 10⁻⁶ mm |
| T | 1.184 × 10⁻⁶ mm |
| d | L2 差 1.515 × 10⁻⁸ |
| 杆体/钳口旋转矩阵 | 元素最大绝对差 1.101 × 10⁻⁷ |
| 关键点、箭头及轴线投影 | 坐标最大绝对差 1.513 × 10⁻⁵ px |

比较对象包括 `C→C+15d`、`H1→J1`、`H2→J2` 和三组 XYZ 轴，每条线采样 61 点。XYZ 轴的验证长度取 3 mm，仅用于数值比较，不是新设的器械结构尺寸或渲染参数。

本测试使用的位置容差为 10⁻⁴ mm，方向和旋转元素差容差为 10⁻⁶，投影差容差为 10⁻³ px；全部通过。它与已有生成器的原始投影公式一致性检查是不同的测试，不能把这里的差异冒充训练后模型的预测误差。

98 个源数据文件保持原样，新渲染次数为 0。详见 [完整验证报告](validation_evidence/model_io_decoding/report.json)。

## 8. 代码与复核入口

本次平衡划分不改变上述 17 维输出合同。训练/验证/测试分别为 1,400/350/350 对，按独立轨迹分组；时序输入只能在同一序列内取连续帧，共同窗口上限为 32 帧。具体索引、七类等权评估和来源时间见 [DATASET_SPLITS.md](DATASET_SPLITS.md)。

新增的已有数据复核独立保存在 [解码计算报告](validation_evidence/balanced_2100/existing_calculations/decoder_report.json) 和 [绘图/视频接口报告](validation_evidence/balanced_2100/existing_calculations/display_report.json)。这次重新执行了计算及绘图检查，未重渲染旧图片；新划分的样本检查另行保存，不与旧数据的结果混记。

另已对新版全部 2,100 个已规划状态（4,200 个相机视图）检查同一计算模块，包含每片钳口 0°、30° 边界。由 17 个数值和固定 USD 模型重算 C、d、H1/H2、J1/J2、T、杆体及钳口旋转，与完整 USD 运动链对照：最大位置差 `1.8215313257e-6 mm`，最大投影差 `1.7535644588e-5 px`，左右坐标一致性差 `5.7903528317e-13 mm`。[全部规划状态计算报告](validation_evidence/balanced_2100/all_planned_decoder_report.json)

对应入口为 `tools/verify_planned_decoding.py`，参数为 `--machine`、`--settings`、`--run`、`--instrument-model` 和 `--output`。其中固定器械模型使用 `tools/verify_model_io.py` 导出的 `instrument_model.json`；输出报告放在源 run 之外。此入口不启动渲染器，也不生成图片。相机外参仅用于最后独立核对左右计算结果，没有传入解码函数。这一结果证明计算代码在当前规划范围内数值一致，不代表图像可见性或训练模型精度。

新版 32 对实际采集样本也已全部复核：最大位置差 `1.6611531119e-6 mm`，箭头/坐标轴投影差 `1.5716527741e-5 px`，32 张已确认预览逐像素一致，视频接口与九项错误输入拒绝检查通过。[新样本计算报告](validation_evidence/balanced_2100/new_calculations/report.json) · [绘图报告](validation_evidence/balanced_2100/new_calculations/display_report.json)。

这批可迁移场景对应的固定模型为 [instrument_model.json](validation_evidence/balanced_2100/new_calculations/instrument_model.json)，对应 [17 维目标示例](validation_evidence/balanced_2100/new_calculations/targets_17.json)。后续接入这些样本时使用这一版本；它保留场景哈希，不能与另一版资产的固定变换混用。

- [独立解码函数](pose_display/decoding.py)和[投影函数](pose_display/projection.py)：NumPy/SciPy 实现，不依赖 Isaac Sim、图像标签或相机外参。旧 `tools/kinematic_decoder.py` 保留兼容入口。
- [预测视频与统一绘图接口](pose_display/README.md)：接收 17 个预测值，计算杆体/双钳口箭头和 XYZ 面板，与批量采集预览共用绘图逻辑；含已有数据回放和无效预测处理。
- [模型导出及对照验证脚本](tools/verify_model_io.py)：使用 USD 运行库读模型，不启动渲染器。
- [固定模型参数](validation_evidence/model_io_decoding/instrument_model.json)
- [17 维训练目标示例](validation_evidence/model_io_decoding/targets_17.json)

在已有机器配置的工程目录中复核：

```bash
python3 tools/verify_model_io.py \
  --machine machine.local.json \
  --settings /path/to/run/config/effective_settings.json \
  --run /path/to/run \
  --output /path/to/new/model_io_check
```

输出目录必须位于原始 run 之外。最后一个输出报告的 `passed: true` 只表示运动学及投影解码验证通过，不表示已经训练了视觉模型，也不代表已经完成另一台实机的基座配准、逆运动学控制或实时性验证。
