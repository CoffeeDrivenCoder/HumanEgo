# 2026-06-26 G1 HumanEgo Debug Summary

## Goal

复现 HumanEgo serve_bread 闭环，并尽量用少量实验定位失败来自视觉、模型输出、G1 控制/IK，还是夹爪接口适配。

## 已完成

1. 规范化 G1 实验产物目录。
   - 当前主 session: `artifacts/g1_humanego/20260626_pose_gate`
   - 每次运行保存 `interactive_step_report.json`、`step_summaries.json/jsonl`、逐步 `server_response.json`、`step_record.json`。

2. 增加视觉中间产物。
   - server 每次推理保存 RGB、depth colormap、object/TCP/target 投影图和 `vision_summary.json`。
   - 用于确认 bread/plate/两个夹爪的视觉定位是否稳定。

3. 修复/规避物体分割不稳定。
   - plate 分割加入筛选条件。
   - interactive 端加入 `object_lock=base_after_first`：第一帧 RGB-D 定位物体，之后用锁定的 base-frame 物体位置回传 server，避免机械臂遮挡导致物体漂移。
   - 后续实验确认 step 1+ 使用 `object_source_used=payload`，物体位置不再由实时分割决定。

4. 验证 G1 控制基础。
   - EE frame 验证：`set_end_effector_pose_control` 接收 base frame 下的 `arm_right_link7` pose。
   - 手动 `z=-1cm` probe：实际 z 约 `-8.9mm`，说明当前位置 G1 可以向下运动。
   - gripper 目标语义确认：`0=open`，`1=closed`，模型 gripper 输出可驱动夹爪。

5. 增加完整 full 模型运行脚本。
   - `scripts/run_g1_humanego_full_step_to_public_server.sh`
   - `scripts/run_g1_humanego_full_auto_to_public_server.sh`
   - 固定使用 `target_source=raw`、`target_adapter=full`、`execute_gripper=true`，避免误跑 position-only 调试版本。

6. 增加 tracking gate。
   - full auto 中默认启用。
   - 当 `actual_delta / target_delta < 0.30` 或 `cos_to_target < 0.50` 连续 2 步时自动停止并记录 `stopped_by`。
   - 目的是捕获“模型仍给出明显目标，但 G1 实际执行跟不上”的首个位置。

## 关键观察

### 视觉和物体定位

在 object lock 后，物体定位不是当前主要问题：

- step 1+ 使用 `object_source_used=payload`
- `object_lock_active=true`
- `vision_warnings={}`

因此后续异常不应优先归因于 bread/plate 分割漂移。

### 模型输出

不同 adapter 下结论不同：

- `position_only + z_bias` 连续运行时，后半段模型 target 自身开始侧向绕走/远离物体，说明 position-only 破坏了完整模型策略闭环，不适合作为最终复现判断。
- `full + gripper` tracking gate 运行时，模型 target 在停止前仍然持续让 link7 靠近 obj1；例如 step 5/6 的 `distance_target_delta_m` 仍为负。

因此当前更可靠的复现路径必须使用 full pose，而不是 position-only。

### G1 full pose 执行

full 模型单步运行表现正常：

- 约 10cm target translation 可执行到约 9.8cm
- 约 15.6deg target rotation 可执行到约 15.8deg
- link7->obj1 距离明显减小

但 full auto 后续出现执行跟踪下降：

- step 5: target 约 `3.43cm`，actual 约 `0.98cm`，ratio `0.284`，cos `0.394`
- step 6: target 约 `3.48cm`，actual 约 `0.66cm`，ratio `0.189`，cos `0.147`
- SDK control call 仍 `ok=true`，说明命令被接受，但实际末端未跟上。

这支持“full pose 后续进入 G1 IK/底层控制难跟踪区域”的判断，但还需要在夹爪适配干净后复跑确认。

### 夹爪接口

已修复一个 gripper payload 问题：

- 旧逻辑把 `gripper_states()` 的 raw state 直接放回 `move_gripper([left,right])`。
- 新逻辑会把 raw state 转成 `0-1` 命令域，并在 summary 记录左右夹爪 before/after/delta。

但最新 tracking gate 日志显示仍有边界问题：当左手 raw state 为小于 1 的值时，代码可能把它当作 command 值保持，导致左手被错误闭合。下一步需要进一步修正 gripper state/command 域判断，或改成只发单侧 scalar payload。

## 当前结论

1. 视觉定位已基本排除为主因。
2. G1 基础 EE 控制和 z 方向运动已验证可用。
3. full 单步模型行为可执行，说明整条链路不是完全错误。
4. full 连续运行后期出现明显 target/actual mismatch，疑似 full pose IK/控制跟踪问题。
5. 夹爪 payload 仍可能污染 full auto 实验，必须先完全修干净，再最终判断 IK。

## 下一步

1. 修复 gripper 保持逻辑：
   - 优先尝试单侧 scalar payload；
   - 或明确 gripper index 与 state/command 域，不再用模糊阈值判断。

2. 复跑 full auto tracking gate：
   - 确认左右夹爪 delta 中非目标侧接近 0；
   - 检查是否仍在 step 5/6 附近触发 target/actual mismatch。

3. 做 limited orientation 对照：
   - `target_adapter=position_orientation_limited`
   - `max_orientation_deg=3~5`
   - 若 limited 不触发 tracking gate，而 full 触发，可更强地定位为 full pose/IK/控制跟踪问题。

