# 2026-06-30 G1 左右臂末端位姿到 7 关节 IK 验证计划

## 目标

验证是否可以把末端目标位姿直接解算成 G1 左/右单臂各自的 7 个关节值，并用关节控制替代当前 `DELTA_POSE` 末端增量控制。

核心问题：

```text
side in {left, right}
target T_arm_{side}_link7_in_base
-> IK(q_{side}_7)
-> FK(q_{side}_7)
-> 真机执行 q_{side}_7
```

是否能稳定满足：

```text
FK(q_{side}_7) 与对应 side 的 target 位姿一致
真机实际 arm_{side}_link7 位姿与对应 target 位姿一致
```

## 成功标准

### 离线成功标准

```text
FK position error <= 2-5 mm
FK rotation error <= 1-2 deg
q_solution 无明显跳变
q_solution 不接近关节限位
左右臂分别满足标准，不能只验证右臂
```

### 真机成功标准

```text
实际 link7 position error <= 5-10 mm
实际 link7 rotation error <= 2-3 deg
执行过程无明显抖动、反向、跳变
左右臂分别满足标准；第一阶段不做双臂同时控制
```

## 明日执行顺序

### Step 1. 确认机器人模型和关节顺序

输入资源：

```text
G1/G1_URDF_Omnipicker.zip
RobotDds.arm_joint_states()
RobotController.get_motion_status()
arm_left_link7
arm_right_link7
```

需要确认：

```text
1. arm_joint_states 中左臂 7 个关节的 index。
2. arm_joint_states 中右臂 7 个关节的 index。
3. URDF 中左臂 7 个关节的顺序。
4. URDF 中右臂 7 个关节的顺序。
5. SDK 的 arm_left_link7 frame 是否和 URDF FK 的 left link7 frame 对齐。
6. SDK 的 arm_right_link7 frame 是否和 URDF FK 的 right link7 frame 对齐。
```

输出记录：

```text
left_arm_joint_names
left_arm_joint_indices_in_sdk_state
left_arm_joint_values_current
left_sdk_T_link7_in_base_current

right_arm_joint_names
right_arm_joint_indices_in_sdk_state
right_arm_joint_values_current
right_sdk_T_link7_in_base_current
```

若任意一侧关节顺序不清楚，先不做该侧 IK。

### Step 2. FK 对齐验证

目的：

```text
先证明 URDF/FK 模型能复现 SDK 当前 link7 位姿。
```

流程：

```text
对 side=left 和 side=right 分别执行：

1. 读取当前 q_{side}_7。
2. 读取 SDK motion_status 中对应 arm_{side}_link7 的 T_link7_in_base。
3. 用 URDF FK 计算 FK(q_{side}_7)。
4. 比较 FK(q_{side}_7) 与 SDK T_arm_{side}_link7_in_base。
```

判断：

```text
如果 FK 对不上，问题在 robot model / joint order / frame convention。
如果 FK 对得上，才进入 IK。
```

需要保存：

```text
fk_validation_left_report.json
fk_validation_right_report.json
fk_validation_summary.json
```

### Step 3. IK 当前位姿自洽测试

目的：

```text
验证 IK solver 对当前真实位姿能解回当前关节附近。
```

流程：

```text
对 side=left 和 side=right 分别执行：

target = 当前 SDK T_arm_{side}_link7_in_base
q_init = 当前 q_{side}_7
q_solution = IK(target, q_init)
FK(q_solution) -> 与 target 比较
```

判断：

```text
q_solution 应接近 q_init
FK(q_solution) 应几乎等于 target
```

如果当前位姿都解不准，不进入真机控制。

需要保存：

```text
ik_current_pose_self_consistency_left.json
ik_current_pose_self_consistency_right.json
ik_current_pose_self_consistency_summary.json
```

### Step 4. 离线小目标 IK 测试

目的：

```text
验证当前姿态附近的小目标可以被稳定反解。
```

目标集合：

```text
对 side=left 和 side=right 分别执行同一批局部目标：

+x 1 cm
-x 1 cm
+y 1 cm
-y 1 cm
+z 1 cm
-z 1 cm
单轴小旋转 2 deg
组合小动作：1 cm 平移 + 2 deg 旋转
```

每个目标记录：

```text
side
target_T_link7_in_base
q_init
q_solution
dq
FK(q_solution)
position_error_m
rotation_error_deg
joint_limit_margin
solver_success
```

判断：

```text
如果小目标离线 IK 稳定，再进入真机小步验证。
如果组合动作失败但纯平移成功，说明姿态约束/冗余处理需要单独调。
```

需要保存：

```text
ik_small_target_batch_left_report.json
ik_small_target_batch_right_report.json
ik_small_target_batch_summary.json
```

### Step 5. 真机小步关节执行验证

目的：

```text
验证 IK 得到的 q_solution 通过关节控制后，实际 link7 能到目标位姿。
```

执行原则：

```text
每次只执行一个小目标。
先 prompt 单步确认，再运动。
每次执行后读取 SDK motion_status。
动作异常立刻停止。
```

推荐第一批目标：

```text
对 side=left 和 side=right 分别执行；不要左右臂同时执行：

+x 1 cm
-x 1 cm
-z 1 cm
单轴旋转 2 deg
```

每步比较：

```text
side
target_T_link7_in_base
FK(q_solution)
actual_T_link7_after
actual_position_error_m
actual_rotation_error_deg
```

判断：

```text
如果 FK 准但真机不准，问题在关节控制接口或执行层。
如果 FK 和真机都准，IK 控制链路可用。
```

需要保存：

```text
ik_joint_control_probe_left_report.json
ik_joint_control_probe_right_report.json
ik_joint_control_probe_summary.json
```

### Step 6. HumanEgo 单步 target IK 验证

目的：

```text
把真实 HumanEgo 输出的 T_link7_target_in_base 接入 IK，但先不连续执行。
当前 serve_bread checkpoint 主要是右手；IK 基础设施必须支持 left/right。
如果模型只输出 right target，则 HumanEgo target IK 先验证 right；
left 侧先用 synthetic/safe target 完成 FK/IK/关节控制验证。
```

流程：

```text
1. 跑 HumanEgo dry-run / single-step，得到 side 对应的 T_link7_target_in_base。
2. 用当前 q_{side}_7 作为 q_init 做 IK。
3. 检查 FK(q_solution) 与 HumanEgo target 的误差。
4. 人工确认 dq 和 target 位姿合理。
5. 再考虑 prompt 单步执行。
```

判断：

```text
如果 HumanEgo target 离线 IK 解不出或 dq 很大，说明目标对当前 G1 不友好。
如果能解且 FK 准，再做真机单步执行。
```

需要保存：

```text
humanego_target_ik_right_report.json
humanego_target_ik_left_report.json  # 如果有 left target
humanego_target_ik_summary.json
```

## 风险点

1. **左右臂关节顺序错误**

最危险，必须先用 FK 对齐验证排除。
左右臂不能假设同序，必须分别确认 SDK state index 和 URDF joint order。

2. **frame 不一致**

要明确 IK 解的是：

```text
T_arm_left_link7_target_in_base
T_arm_right_link7_target_in_base
```

不是 HumanEgo hand frame，也不是 TCP frame。
如果控制 TCP，需要先用固定 `T_tcp_in_link7` 转回对应 side 的 link7 target。

3. **7DoF 冗余导致跳解**

IK 必须以当前 side 的当前关节为 seed，并加入最小 `dq` 或 posture regularization。

4. **关节控制接口语义未知**

即使 IK 正确，也必须分别验证 left/right joint command 的单位、顺序、阻塞/非阻塞语义。

5. **HumanEgo target 可能本身对 G1 姿态不友好**

如果 IK 解靠近关节限位或需要大幅姿态变化，应先做 orientation limit 或 substep。

6. **左右臂镜像/符号差异**

左臂和右臂的关节正方向、TCP 固定变换、link7 frame 可能不是简单镜像。
不要把右臂验证结果直接套到左臂。

## 建议目录结构

每次验证按 side 分开保存，避免左右结果混在一起：

```text
artifacts/g1_humanego/<session>/diagnostics/ik_validation/
  left/
    fk_validation_report.json
    ik_current_pose_self_consistency.json
    ik_small_target_batch_report.json
    ik_joint_control_probe_report.json
  right/
    fk_validation_report.json
    ik_current_pose_self_consistency.json
    ik_small_target_batch_report.json
    ik_joint_control_probe_report.json
  summary.json
```

## 推荐结论格式

明天每一步结束后，只记录三类结论：

```text
1. 通过：可以进入下一步。
2. 未通过：失败指标是什么。
3. 定位：失败更像 joint order / FK frame / IK solver / joint control / target 本身。
```
