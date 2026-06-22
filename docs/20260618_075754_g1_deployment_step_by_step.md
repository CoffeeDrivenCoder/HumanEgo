# HumanEgo 部署到 Agibot G1 的步骤

目标：把已经训练好的 HumanEgo 权重，按步骤接到 G1 真机上，先跑通单臂闭环，再逐步增强到稳定部署。

当前前提：
- 你已经有训练好的 HumanEgo checkpoint
- 当前 checkpoint 是单右手 `serve_bread`
- 目标机器人是 Agibot G1
- 目标先是“能推理、能闭环、能安全执行”

参考代码：
- [推理入口](../inference/run_inference.py)
- [policy 封装](../inference/policy.py)
- [控制器](../inference/controller.py)
- [接口定义](../inference/interfaces.py)
- [G1 手眼说明](../../RoboClaw/docs/agibot_g1_hand_eye_and_eef_control_guide.md)

## 1. 总体架构

推荐做成三层：

### 1.1 Server

职责：
- 加载 HumanEgo policy
- 读取相机图像和机器人状态
- 运行感知和推理
- 输出控制命令
- 维护任务状态和安全状态

### 1.2 Client

职责：
- 开始任务
- 结束任务
- 人工接管
- 恢复自动控制
- 显示状态

### 1.3 Robot / Sensor Driver

职责：
- 读相机
- 读机器人关节和末端状态
- 执行末端控制
- 控制夹爪

高频控制不要走 HTTP，HTTP 只管状态切换。

## 2. 部署顺序

### Step 0: 先确认权重和配置

输入：
- `latest.pt`
- `config.json`
- `dataset_stats.json`

输出：
- 一个可运行的推理配置文件

要做什么：
- 确认 checkpoint 是单右手
- 确认 `frame_mode=camera_frame`
- 确认 `use_region_attn=True`
- 确认 `T_align` 使用的是这个权重对应的约定

验证点：
- `policy.py` 能成功加载
- `strict=True` 不报错

### Step 1: 先把 G1 机器人本体接口跑通

输入：
- G1 SDK
- 机器人连接信息

输出：
- 能读关节状态
- 能读末端位姿
- 能回 home
- 能开合夹爪

要做什么：
- 先不接 HumanEgo
- 只确认机器人 SDK 本身稳定

最少要测：
- `head_joint_states` 或 arm joint states
- FK
- gripper open/close
- home

验证点：
- 机器人不抖
- 状态读取稳定
- home 不报错

### Step 2: 先把相机接口跑通

输入：
- RGB-D 相机
- 相机配置

输出：
- 一帧同步 RGB
- 一帧同步 depth
- 相机内参 `K`

要做什么：
- 验证图像分辨率
- 验证 depth 和 RGB 对齐
- 验证时间戳

验证点：
- RGB 图和深度图能对上
- 目标物体在 RGB 和 depth 上位置一致

### Step 3: 做手眼标定

输入：
- 相机
- 机器人
- 标定板或 AprilTag
- 关节状态

输出：
- 外置相机：`T_base_in_cam`
- G1 头相机：`T_head_pitch_camera`

要做什么：
- 让相机看到固定标定板
- 改变机器人或头部姿态
- 解出相机和机器人基座之间的关系

补充：
- 如果你当前只让 G1 的头和腰固定
- 且 HumanEgo 只输出右臂末端和夹爪动作

那这个步骤的产物最终可以收敛成一个静态相机外参，不必把头/腰的在线 FK 放进部署主链路里。

验证点：
- 同一个固定标签，机器人姿态变化后，换算到 base 下的位置基本不漂

### Step 4: 做 TCP / 末端工具标定

输入：
- 机器人末端几何
- 已知工具长度或实测数据

输出：
- `T_flange_tool`
- 或等价的 TCP 偏移

要做什么：
- 确认你控制的点，真的是 HumanEgo 需要的抓取点

验证点：
- 末端朝向对时，夹爪实际接触位置也对

### Step 5: 做 HumanEgo 的 `T_align`

输入：
- checkpoint 所用手部约定
- 你 G1 末端的控制约定

输出：
- 一个固定 `4x4` 变换矩阵

要做什么：
- 把模型预测的“手”对到 G1 的“末端”

验证点：
- 姿态不再系统性偏 90 度或 180 度

### Step 6: 先把 perception 跑通

输入：
- RGB-D
- 物体 prompt
- anchor key

输出：
- `ObjectState` 字典
- `anchor_uv`
- clean image

要做什么：
- 用 DINO-SAM / 深度 / PCA 得到物体位姿
- 先只做静态场景，不闭环控制

验证点：
- 物体框和位姿稳定
- anchor 物体的位置靠谱

### Step 7: 接 HumanEgo policy

输入：
- clean image
- ICT
- `anchor_uv`
- checkpoint

输出：
- 未来 50 步右臂轨迹
- grasp 概率
- done 概率

要做什么：
- 直接复用 [policy.py](../inference/policy.py)
- `build_ict()` 必须和训练时一致

验证点：
- 能稳定返回轨迹
- 轨迹 shape 正确

### Step 8: 接控制器

输入：
- policy 轨迹
- grasp 概率
- 控制频率

输出：
- 末端目标
- 夹爪开合命令

要做什么：
- 复用 [controller.py](../inference/controller.py)
- 先小速度、小步长

验证点：
- 末端移动平滑
- 不会一下子冲太远

### Step 9: 先做离线 dry-run

输入：
- 录制数据
- 或 live 图像但禁用运动

输出：
- 预测轨迹
- 叠图
- 误差和延迟日志

要做什么：
- 不让机器人真正动
- 只验证整条链路

验证点：
- 轨迹方向和目标视觉一致
- 没有明显坐标翻转

### Step 10: 低速单臂真机闭环

输入：
- 相机
- G1
- policy
- perception
- 标定结果

输出：
- 真实右臂动作
- 真机 grasp
- episode 日志

要做什么：
- 先只跑右臂
- 先低 `control_hz`
- 先小 `max_pos_step`

验证点：
- 机器人动作和图像目标一致
- 不撞桌面
- 不离开工作空间

### Step 11: 增加任务管理和人工接管

输入：
- start / stop / intervention 事件

输出：
- 状态机
- 任务会话
- 干预日志

要做什么：
- 加一个本地 server
- Client 只发状态变化

验证点：
- 人工接管后模型停止发动作
- 恢复时重新从当前状态起步

### Step 12: 再考虑双臂或其他任务

输入：
- 新任务权重或新配置

输出：
- 新的推理约定

要做什么：
- 双臂只能在相应 checkpoint 支持时再加

验证点：
- 不拿单右手 checkpoint 硬驱双臂

## 3. 每一步的输入输出总表

| 步骤 | 输入 | 输出 |
|---|---|---|
| 0 | checkpoint + config + stats | 可运行推理配置 |
| 1 | G1 SDK | 机器人状态、FK、home、夹爪 |
| 2 | 相机 SDK | RGB、depth、K、timestamp |
| 3 | 标定板 + 关节状态 | `T_base_in_cam` / `T_head_pitch_camera` |
| 4 | 末端几何 | TCP / tool offset |
| 5 | 训练约定 | `T_align` |
| 6 | RGB-D + prompt | object pose + clean image |
| 7 | clean image + ICT | 未来轨迹 + done |
| 8 | 轨迹 + grasp | 可执行控制命令 |
| 9 | 录制数据 | dry-run 验证结果 |
| 10 | 全链路在线输入 | 真机闭环动作 |
| 11 | 任务事件 | server/client 状态机 |

## 4. 最推荐的第一版落地策略

第一版不要做太大：
- 只接右臂
- 只接一个相机
- 只做一个任务 `serve_bread`
- 只做低速闭环
- 先不做人工旁路和多任务切换

这样最容易先把“能跑”变成“跑稳”。
