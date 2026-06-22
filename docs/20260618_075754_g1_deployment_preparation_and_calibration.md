# HumanEgo 到 G1 的准备与标定清单

目标：在不绕晕坐标系的前提下，先把真机部署前必须准备的东西讲清楚，尤其是哪些量已经能直接从 G1 机器人拿到，哪些量还需要你补齐或确认。

当前前提：
- 你已经有 HumanEgo 的训练权重
- 当前权重是 `serve_bread` 任务
- 该权重是单右手模型，`single_hand=true`
- 该权重的推理参考系是 `camera_frame`
- 该权重启用了 `object_centric` ICT 和 `region_attn`

仓库里的相关参考：
- [HumanEgo 推理模板](../inference/run_inference.py)
- [HumanEgo 推理接口](../inference/interfaces.py)
- [HumanEgo policy 加载逻辑](../inference/policy.py)
- [G1 手眼标定说明](../../RoboClaw/docs/agibot_g1_hand_eye_and_eef_control_guide.md)

## 1. 先用人话理解这件事

HumanEgo 真机部署本质上只做三件事：
1. 拿到当前相机看到的图像
2. 把图像和机器人状态变成模型能理解的输入
3. 把模型预测的“手的未来姿态”转成 G1 真正能执行的末端目标

所以你真正需要弄清楚的只有三类坐标系：
- 相机坐标系：相机“看见”的世界
- 机器人基座坐标系：机器人“站着”的世界
- 末端执行器坐标系：夹爪/手腕真正要控制的点

另外还有两个“模型内部坐标系”：
- 物体坐标系：用于 object-centric ICT
- HumanEgo 的“手”坐标系：模型训练时定义的手姿态习惯

## 2. 两种部署形态

### 2.1 外置固定相机

如果你用外置 RGB-D 相机：
- 相机位置固定
- 只要做一次 `T_base_in_cam` 手眼标定
- 之后每一帧都能把相机里的目标换算到机器人基座里

这是最简单的形态。

### 2.2 G1 头部相机

如果你用 G1 头部相机：
- 相机跟着头动
- 不能只存一个静态 `T_base_in_cam`
- 你需要：
  - 头/腰关节状态
  - 机器人 FK
  - 一个固定的 `T_head_pitch_camera`

运行时用下面这个公式算相机在 base 里的位姿：

```text
T_base_camera(q) = T_base_head_pitch(q) * T_head_pitch_camera
```

这个形态更适合 G1 真机，但标定和坐标理解要多一层。

补一句和你当前部署强相关的结论：
- 如果 HumanEgo 推理阶段只输出右臂末端轨迹和夹爪动作
- 并且 G1 的头和腰在执行时保持固定不动

那就不需要在线计算上面的动态 `T_base_camera(q)`，直接用一次标定得到的静态相机外参即可。  
也就是说，对你现在这个版本，头部相机不一定要求“头和腰关节状态参与推理”，只有在你未来要让头/腰跟着视野或任务动态运动时，才需要把 FK 链接进来。

## 3. 你需要先直接拿到什么

下面这些最好是“直接拿到”的，不要自己先猜。

| 项目 | 最好直接来源 | 为什么需要 |
|---|---|---|
| RGB 图像 | 相机 SDK | 给 policy 和感知用 |
| 深度图 `depth_m` | RGB-D 相机 SDK | 估计物体 3D 位姿时要用 |
| 相机内参 `K` | SDK / 标定结果 | 像素点转 3D 射线 |
| 图像分辨率 | SDK | 内参和图像必须一致 |
| 时间戳 | 相机/机器人 SDK | 图像和关节状态要对齐 |
| 机器人关节状态 | G1 SDK | 用 FK 算末端位姿 |
| 机器人末端位姿 | FK 或 SDK | 作为控制和验证基准 |
| 夹爪开合状态 | SDK | policy 要预测 grasp |
| home 位姿 | 机器人配置 | 每次任务开始先回零 |
| 机器人 base 定义 | 机器人 SDK / 文档 | 统一基座坐标系 |
| 任务对象 prompt | 你手动填写 | 让检测器知道看什么 |
| `anchor_key` | 你手动指定 | 哪个物体当参考原点 |
| 安全限位 | 你手动配置 | 防止撞桌/撞人 |

你现在这个项目里，有一批东西其实已经准备好了，不需要再做一次手眼标定去“重新求”：

| 已有项 | 来源 | 备注 |
|---|---|---|
| 头部相机内参 | `G1/parameters/head_intrinsic_params.json` | 可直接用 |
| 头部相机外参 | `G1/parameters/head_extrinsic_params.json` | 可直接用 |
| 右手相机内参 | `G1/parameters/hand_right_intrinsic_params.json` | 如果后面要用手腕相机，可直接用 |
| 右手相机外参 | `G1/parameters/hand_right_extrinsic_params.json` | 同上 |
| 相机设备信息 | `G1/parameters/rs_camera_info.json`、`G1/parameters/fisheye_camera_info.json` | 可直接对设备名和分辨率 |
| 机器人几何链路 | `G1/G1_URDF_Omnipicker.zip` | 可用于 FK 和链路名对齐 |
| 关节范围/平滑参数 | `G1/parameters/threshold_parameters.yaml`、`G1/parameters/smooth_parameters.yaml` | 可直接用于安全约束 |
| DH 参数 | `G1/parameters/dh_parameters.yaml` | 可作为 FK 参考或校验 |

所以，对你当前这版部署来说，**内参和外参不需要重新手眼标定来“从零求”**，更像是：
- 直接读取机器人已经给出的参数
- 再确认这些参数对应的 frame 名和方向约定
- 最后把它们接到 HumanEgo 的推理链路里

## 4. 你必须标定什么

下面这些不是“可有可无”，而是部署能不能动起来的关键。

### 4.1 相机内参标定

输出：
- `fx, fy, cx, cy`
- 分辨率对应关系
- 可选畸变参数

作用：
- 把 2D 像素换成 3D 方向
- 把深度值还原成相机坐标系下的点
- 物体 6D 位姿估计必须依赖它

如果错了：
- 物体位置会飘
- 手眼标定再准也没用

直接拿到的情况：
- 许多相机 SDK 会直接给
- RealSense 类相机通常能直接读出

对你当前项目：
- `head_intrinsic_params.json` 已经可直接用
- `hand_left_intrinsic_params.json` / `hand_right_intrinsic_params.json` 也已经可直接用
- 这一步更准确地说是“读取和校验”，不是重新标定

### 4.2 RGB-D 对齐

输出：
- RGB 像素和深度像素一一对应

作用：
- 你在 RGB 上检测到的物体区域，才能去深度图里取同一批像素的深度

如果错了：
- 物体点云会错位
- 3D 物体姿态会偏

直接拿到的情况：
- 如果 SDK 已经输出对齐好的 depth，那最好
- 如果没对齐，要自己做 image registration

对你当前项目：
- 如果 G1 的头相机 SDK 本身已经提供对齐后的 color/depth，那这一步就是确认接口
- 如果后面用的是单目 RGB 头相机，那这一项可以不进主链路

### 4.3 手眼标定

这个是最重要的。

#### 外置相机时

输出：
- `T_base_in_cam`

意思：
- 把“相机坐标系”放到“机器人 base_link”里

作用：
- 你在相机里看到的物体，能换算成机器人能抓的地方

对你当前项目：
- 如果你已经拿到了机器人直接给出的 `head_extrinsic_params.json`
- 而且头和腰固定不动

那这里更像“读取固定外参”，不是再做一次新的手眼标定

#### G1 头部相机时

输出：
- `T_head_pitch_camera`

再加上：
- 头/腰关节状态
- 机器人 FK

运行时算：

```text
T_base_camera(q) = T_base_head_pitch(q) * T_head_pitch_camera
```

意思：
- 相机不是固定在桌上的，而是挂在头上
- 所以它的位置是随头部姿态变化的

如果错了：
- 机器人会去错位置
- 常见表现是“看起来目标对了，但手总偏一截”

对你当前项目：
- 头相机外参已经在参数包里
- 只有当你想让头/腰动态运动时，才需要把 `T_base_head_pitch(q)` 在线算进去
- 你现在这版如果头和腰固定，主链路里可以直接用静态外参

### 4.4 末端工具 / TCP 标定

输出：
- `T_flange_tool` 或 TCP 偏移

意思：
- 机器人法兰不等于夹爪真正接触物体的点
- 需要知道“真实控制点”相对法兰在哪里

作用：
- 让模型预测的末端姿态，能落到夹爪真正可用的位置

如果错了：
- 抓取点会前后偏
- 姿态看着对，实际接触点不对

### 4.5 HumanEgo 的 `T_align`

输出：
- 一个固定的 4x4 变换矩阵

意思：
- 把“模型里训练出来的手坐标系”对齐到你 G1 真正控制的末端坐标系

作用：
- HumanEgo 的 policy 预测的是“手”的姿态
- G1 需要的是“末端执行器”的姿态
- `T_align` 就是这两个约定之间的桥

如果错了：
- 位置可能还行
- 但朝向会系统性错一截

这不是相机标定，是“坐标约定对齐”。

### 4.6 时间同步

输出：
- 图像时间戳和机器人状态时间戳尽量对齐

作用：
- policy 看到的是“同一时刻”的世界

如果错了：
- 机器人在追一个已经过时的画面
- 闭环会抖

### 4.7 物体初始化位姿

输出：
- 每个对象在相机里的 6D 位姿
- 至少要有 anchor object 的位姿

作用：
- HumanEgo 的 ICT 需要 object-centric 参考
- region attention 还要算 `anchor_uv`

如果错了：
- 模型输入的 ICT 就不对

## 5. 直接拿到 vs 间接求

### 5.1 最好直接拿到

这些优先从 SDK、厂商文档、设备标定文件里直接拿：
- 相机内参 `K`
- 相机与深度对齐关系
- 机器人关节状态
- 机器人 FK
- 夹爪状态
- home 位姿
- 机器人 base 定义
- 已知的 TCP 偏移
- 任务对象 prompt

### 5.2 需要标定或间接求

这些通常不是现成给你的：
- `T_base_in_cam`
- `T_head_pitch_camera`
- `T_align`
- anchor object 的初始位姿
- 任务运行时每帧的 object pose

### 5.3 需要在线算出来

这些是部署时每帧推出来的：
- `T_base_camera(q)`
- `T_ee_in_cam`
- `anchor_uv`
- `x_ict`
- policy 输出的未来轨迹

## 6. 第一版最小可运行清单

如果你只想先让右臂动起来，最少要有：
- 一个 RGB-D 相机
- 相机内参
- 图像和深度对齐
- G1 关节状态
- G1 FK
- 一个可用的手眼外参
- 一个可用的 TCP 偏移
- `T_align`
- `obj1 / obj2` 的任务 prompt
- anchor object 的初始位姿
- 安全限位和急停

## 7. 当前这个 HumanEgo 权重的默认任务信息

这份 checkpoint 对应的 released 任务是 `serve_bread`，默认对象 prompt 是：

```text
obj1: "piece of bread ."
obj2: "a plate ."
```

当前模型是单右手，所以第一版不要把左臂也接进来。
