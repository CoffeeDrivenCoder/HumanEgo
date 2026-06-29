# 2026-06-22 至 2026-06-28 HumanEgo/G1 复现周总结

## 总体目标

本周目标是把 HumanEgo `serve_bread` 从离线推理推进到 Agibot G1 真机闭环复现，并用尽可能少的实验定位问题来源：视觉定位、模型输出、坐标转换、G1 末端控制/IK，或夹爪接口适配。

## 主要进度

### 1. 打通 G1 到 HumanEgo 的真机推理链路

已形成完整 server-client 架构：

```text
G1 RGB-D + 当前机器人状态
-> 服务器端 RGB-D 物体定位 + HumanEgo policy 推理
-> HumanEgo hand target 转 G1 TCP/link7 target
-> 机器人端执行末端位姿和夹爪控制
```

完成内容：

- 机器人端 client 支持采集 RGB-D、相机内参、当前 TCP/base-camera 状态。
- 服务器端支持 HumanEgo 推理、RGB-D 物体位姿估计和目标位姿转换。
- 图像上传从原始 `1280x800` 优化到模型输入尺寸 `320x240`，端到端请求延迟从约 `1.1s` 降到约 `0.29s`。
- 增加 dry-run、interactive one-step、full-step、full-auto 等分阶段运行脚本。

### 2. 验证 G1 坐标链和末端控制语义

已确认 G1 末端控制接口 `set_end_effector_pose_control()` 的使用方式：

```text
目标坐标系：base_link
控制 frame：arm_right_link7
pose 格式：{x, y, z, qx, qy, qz, qw}
control_group：["right_arm"]
```

真机 `+Z 1cm` 验证结果：

```text
commanded_delta_m = [0.0, 0.0, 0.0100]
observed_delta_m  = [0.00036, -0.00004, 0.00931]
```

结论：G1 基础末端 pose 控制链路可用，HumanEgo 输出经过坐标转换后可以送入 G1 执行。

### 3. 建立 RGB-D 物体位姿验证流程

明确 `object_source=fixed` 只能用于链路调试，真实抓取必须使用 `object_source=rgbd`。

已完成：

- 服务器端接入 DINO-SAM + depth lifting + PCA，估计真实 `obj1` 面包和 `obj2` 盘子的 `T_obj_in_cam`。
- 增加 `allow_fixed_object_fallback=false`，避免 RGB-D 失败时静默退回假物体位姿。
- 增加 RGB、depth colormap、object/TCP/target 投影图和 `vision_summary.json` 等中间产物。
- 建立 pose gate：先验证 object/TCP 投影和三维数值，再允许机器人运动。

### 4. 规范化实验产物管理

统一使用：

```text
artifacts/g1_humanego/<session>/<role>/<run>/
```

其中：

```text
server       服务器端请求、响应、可视化
client       机器人端 dry-run
interactive 机器人端交互/执行日志
diagnostics 低层诊断
```

关键输出包括：

- `interactive_step_report.json`
- `step_summaries.json/jsonl`
- `server_response.json`
- `step_record.json`
- split-layer 可视化图

这样后续每次实验都能直接追踪数值、图像和执行结果。

### 5. 推进到 raw/full 真机执行验证

单步 raw/full 运行表现正常：

```text
target translation 约 10cm，实际执行约 9.8cm
target rotation 约 15.6deg，实际执行约 15.8deg
link7 到 obj1 距离明显减小
```

说明：

- 模型输出、坐标转换、G1 单步控制不是整体错误。
- full pose 比 position-only 更接近最终复现路径。
- position-only 会破坏模型原始策略闭环，不适合作为最终判断依据。

## 遇到的问题与解决方案

### 问题 1：上传接口返回 502，误判为收发失败

现象：

```text
HTTP Error 502: Bad Gateway
```

分析：

- 客户端本地 zip 和 run_dir 已生成。
- 服务器端仍能收到部分请求并产生 server artifact。
- 502 更多是 public upload/反向代理层问题，不等同于 HumanEgo 推理失败。

解决：

- 后续以本地 artifact 和服务器 `artifacts/g1_humanego/...` 为主要依据。
- 脚本保留 `upload_result.json`，但不再把 upload 失败直接当作推理失败。

### 问题 2：早期可视化图层太杂乱

现象：

- 物体、TCP、target、坐标轴全部画在一张图上，肉眼难以判断。

解决：

- 增加 split-layer 输出：
  - `response_projection_clean.jpg`
  - `response_projection_objects.jpg`
  - `response_projection_tcp.jpg`
  - `response_projection_axes.jpg`
- 后续判断视觉和目标方向时分别看不同图层。

### 问题 3：盘子分割不稳定，容易受桌面颜色和机械臂遮挡影响

现象：

- plate 与桌面/桌垫颜色接近时，分割容易吃进背景或手臂。
- 机械臂靠近后遮挡物体，导致实时 RGB-D 物体 pose 漂移。

解决：

- 给 plate 分割增加条件筛选。
- 增加 `object_lock=base_after_first`：第一帧用 RGB-D 定位物体，之后锁定 base-frame 物体位置并回传服务器。

结果：

```text
step 1+ 使用 object_source_used=payload
object_lock_active=true
vision_warnings={}
```

因此后续异常不再优先归因于物体分割漂移。

### 问题 4：夹爪命令域和状态域混用

现象：

- G1 gripper state 可能返回 `0-120` raw 值。
- `move_gripper()` 控制命令使用归一化 `0-1`。
- 旧逻辑把 raw state 直接作为保持值塞回 `move_gripper([left,right])`，导致非目标侧夹爪也可能运动。

已解决：

- 确认夹爪语义：`0=open`，`1=closed`。
- 增加 gripper before/after/delta 日志。
- 将 gripper payload 尽量归一化到命令域。

剩余风险：

- 最新日志仍显示左手保持逻辑有边界问题。
- 后续应优先改成单侧 scalar payload，或明确 state/command 域和左右 index，不再用模糊阈值判断。

### 问题 5：连续 full-auto 后期出现 target/actual mismatch

现象：

在 full + gripper tracking gate 运行中，模型目标仍然要求靠近物体，但 G1 实际末端运动明显变小或方向不一致：

```text
step 5: target 约 3.43cm，actual 约 0.98cm，ratio 0.284，cos 0.394
step 6: target 约 3.48cm，actual 约 0.66cm，ratio 0.189，cos 0.147
SDK control call 仍 ok=true
```

分析：

- 单步 full pose 可以正常跟踪，说明链路不是全局错误。
- 后续连续运行中，命令被 SDK 接受但实际末端没有跟上。
- 当前更像 G1 full pose 控制/IK/底层跟踪在特定姿态附近进入难执行区域。

已做措施：

- 增加 tracking gate：当 `actual_delta / target_delta < 0.30` 或 `cos_to_target < 0.50` 连续 2 步时自动停止并记录现场。
- 增加 `post_ee_delta`、`post_gripper_arm_delta`、`cos_to_target` 等指标。

待验证：

- 先修干净夹爪污染，再复跑 full-auto tracking gate。
- 增加 orientation-limited 对照，判断问题是否主要来自 full orientation。
- 评估是否切换到 G1 官方 `trajectory_tracking_control + DELTA_POSE` 轨迹跟踪接口。

## 当前结论

1. G1 RGB-D、相机参数、base-camera 外参、TCP/link7 坐标链已经完成第一轮验证。
2. HumanEgo server-client 推理链路已打通，模型输出可以转换成 G1 link7 目标。
3. RGB-D 物体定位经过 object lock 后已基本排除为当前主因。
4. raw/full 单步执行结果良好，说明模型输出和坐标转换不是整体错误。
5. 连续 full-auto 后期的主要风险集中在 G1 full pose 控制/IK/底层跟踪，以及夹爪 payload 污染。

## 下周建议

1. 彻底修复夹爪控制：
   - 优先只发送目标侧夹爪命令；
   - 或明确左右 index 与 state/command 域，避免非目标侧误动。

2. 复跑 full-auto tracking gate：
   - 禁止或隔离夹爪影响；
   - 检查 target/actual mismatch 是否仍在 step 5/6 附近出现。

3. 做控制接口对照：
   - 当前 HumanEgo 使用 `set_end_effector_pose_control()`，即绝对末端位姿控制。
   - G1 PDF 中还记录了 `trajectory_tracking_control + DELTA_POSE`，即笛卡尔末端相对位姿轨迹跟踪。
   - 建议新增一个 `DELTA_POSE` 对照脚本，用同一组 HumanEgo target 比较两种接口的实际跟踪能力。

4. 在 full-auto 稳定前继续保留最小验证原则：
   - 先看数值和可视化；
   - 再单步执行；
   - 最后连续闭环。
