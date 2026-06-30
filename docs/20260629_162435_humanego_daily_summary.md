# 2026-06-29 HumanEgo/G1 有效实验总结

## 今日目标

把问题从“HumanEgo 复现整体行为不稳定”拆开，重点判断：

1. G1 SDK `DELTA_POSE` 控制本身是否能准确跟踪目标轨迹。
2. HumanEgo 模型输出的轨迹是否合理。
3. 之前在线复现中 target/actual mismatch 更像控制问题，还是模型/状态输入问题。

## 有效实验

### 1. 纯 SDK delta-pose 轨迹验证

使用手写安全轨迹 `cfg/g1_delta_pose_sequences/delta_pose_safe_20step.json`，绕当前位置执行 20 步小幅平移/旋转。

关键结果：

```text
平移实际/目标比例平均约 1.007
最大平移误差约 1.5mm
旋转平均误差约 0.44deg
最大旋转误差约 1.35deg
方向 cos 平均约 0.984
```

有效结论：

```text
在 1cm 级平移、几度旋转、可达且连续平滑的 delta-pose 轨迹上，
G1 SDK 控制误差很小，底层 delta-pose 执行能力基本可信。
```

### 2. 首帧 HumanEgo 自回归 rollout

新增并运行 rollout preview：

```text
第一帧 RGB-D + 当前机械臂状态
-> RGB-D 定位 obj1/obj2
-> 固定物体位姿
-> 假设每一步模型目标都已达到
-> 自回归生成 20 步 HumanEgo 轨迹
```

该流程不控制机器人，只生成：

```text
autoregressive_rollout.json
server_response.json
vision_contact_sheet / object projection / mask debug
```

轨迹量级：

```text
每步平移：0.43cm ~ 4.04cm，平均 1.36cm
每步旋转：1.08deg ~ 7.33deg，平均 2.99deg
大旋转 >10deg：0 次
```

有效结论：

```text
这条 HumanEgo 生成的轨迹数值不离谱，作为 SDK replay 控制测试是可行的。
```

### 3. 轨迹目标趋势分析

对 rollout 中 TCP 到两个物体的三维距离做统计：

```text
TCP -> obj1 面包：16.70cm -> 28.15cm
整体远离约 11.45cm

TCP -> obj2 盘子：32.91cm -> 25.05cm
整体靠近约 7.86cm
前 15/20 步都在靠近盘子
```

有效结论：

```text
这条轨迹不是朝面包抓取，而是整体朝盘子方向移动。
```

结合本次输入：

```text
input gripper = 1.0
HumanEgo 约定：0=open, 1=closed
```

当前最合理解释：

```text
模型可能把当前状态理解成“夹爪已闭合/已经抓住面包”，
因此策略阶段切到“往盘子方向移动/放置”。
```

## 今日有效结论

1. **G1 SDK delta-pose 控制本身目前表现很好。**
   在已验证的小步连续轨迹上，误差是毫米级和约 1 度量级。

2. **本次 HumanEgo rollout 轨迹数值可执行，但任务语义不对。**
   它更像在靠近盘子，而不是靠近面包。

3. **之前 HumanEgo 在线复现中的 target/actual mismatch 还不能直接归因于 SDK。**
   纯 SDK 小步轨迹已经证明底层控制可以很准；完整在线链路里的偏差可能来自目标生成、任务阶段判断、状态输入、控制节奏或局部可达性。

4. **当前最值得关注的模型输入问题是 gripper 状态。**
   本次 rollout 的输入夹爪为 `1.0`，可能导致模型认为已经完成抓取，提前进入放置阶段。

## 还不能下的结论

暂时不能直接说：

```text
HumanEgo 完整复现架构经常导致机械臂实际位姿和目标位姿差很远。
```

目前只能说：

```text
之前在线实验观察到过明显 target/actual mismatch；
但还需要用固定 HumanEgo 轨迹 replay 排除模型重推理、视觉、状态更新和控制节奏的干扰。
```

## 下一步最小验证

### A. 打开夹爪后重新跑 rollout preview

目的：

```text
验证 gripper=0/open 时，模型是否重新生成靠近面包的抓取轨迹。
```

判断标准：

```text
TCP -> obj1 距离应整体下降；
如果仍远离面包，则问题更可能是 hand/TCP frame、训练分布或模型适配。
```

### B. 固定 HumanEgo rollout 轨迹做 SDK 单步 replay

目的：

```text
只验证 HumanEgo 生成轨迹 -> SDK delta-pose -> 机器人实际运动 是否一致。
```

每步记录：

```text
target translation / actual translation / translation error
target rotation / actual rotation / rotation error
cos_to_target
```

如果 replay 误差仍然很小，则可以更强地排除 SDK 控制能力问题，把重点转回模型输入和任务阶段判断。

### C. 最终控制一致性实验：10-step HumanEgo action replay

实验设计：

```text
1. 在同一个初始机器人位姿和同一帧 RGB-D 下，生成 10 个 HumanEgo action。
2. 将这 10 个 action 保存成 JSON，包括每步目标平移、目标旋转、action_data、当前/目标 link7 位姿。
3. 不重新推理、不重新分割、不在线更新目标，直接用该 JSON 通过 SDK DELTA_POSE 单步 replay。
4. 每一步执行后记录真实 link7 delta，并和 JSON 中的目标 action 对齐比较。
```

这个实验只回答一个问题：

```text
同一条 HumanEgo 生成的 delta-pose 轨迹，G1 SDK 实际执行是否一致？
```

比较指标：

```text
target_translation_m
observed_translation_m
translation_error_m
translation_ratio
cos_to_target
target_rotation_deg
observed_rotation_deg
rotation_error_deg
```

预期结论分支：

```text
如果 10 步 replay 误差仍是毫米级/约 1-2deg：
  说明 SDK DELTA_POSE 控制链路基本可信。
  之前在线 HumanEgo 复现的大误差更可能来自在线目标更新、时序打断、闭环停止、状态输入或模型阶段判断。

如果 10 步 replay 也出现明显 target/actual mismatch：
  说明 HumanEgo 真实生成的位姿序列虽然数值看似正常，
  但对当前 G1 SDK/IK/姿态控制并不稳定，需要继续定位姿态、路径或关节配置问题。
```

注意事项：

```text
生成 action JSON 和 replay 之间尽量不要移动机械臂；
如果初始位姿改变，应优先 replay 存储的 delta/action_data，而不是绝对 target pose。
```
