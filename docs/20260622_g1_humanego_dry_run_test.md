# G1 HumanEgo 真机 Dry-Run 测试

目的：在不发任何控制命令的前提下，验证：

```text
G1 RGB-D
-> 当前 TCP pose
-> HumanEgo policy 推理
-> T_hand_target_in_cam
-> T_link7_target_in_base
-> right_pose flat dict
```

## 机器人端运行

先拉最新代码：

```bash
cd ~/桌面/HumanEgo
git pull origin main
```

运行 dry-run：

```bash
cd ~/桌面/HumanEgo
bash scripts/run_g1_humanego_dry_run_to_public_server.sh
```

默认配置：

- 只跑 1 次推理。
- 使用固定物体位姿占位，不跑 DINO-SAM。
- 保存 RGB、depth、当前 pose、policy 输出和转换后的 G1 目标。
- 不调用 `set_end_effector_pose_control`。
- 不调用 `move_gripper`。

如果要强制 CPU：

```bash
G1_HUMANEGO_DEVICE=cpu \
bash scripts/run_g1_humanego_dry_run_to_public_server.sh
```

如果要连续 dry-run 3 次：

```bash
G1_HUMANEGO_STEPS=3 \
bash scripts/run_g1_humanego_dry_run_to_public_server.sh
```

## 输出检查

上传包里主要看：

```text
humanego_dry_run_report.json
iter_000/iteration_report.json
iter_000/rgb_bgr.png
iter_000/depth_m.npy
```

关键字段：

- `control_sent` 必须是 `false`。
- `policy_preview.done_prob`：模型 done 概率。
- `policy_preview.sides.right[0].T_tcp_target_in_cam`：模型第一步 TCP target。
- `policy_preview.sides.right[0].T_link7_target_in_base`：转换后的 G1 link7 target。
- `policy_preview.sides.right[0].right_pose_flat_limited`：后续控制接口可用的安全限幅目标。
- `gripper_g1_raw_0_open_120_closed`：夹爪目标，0 张开，120 闭合。

第一轮只判断数值是否合理，不让机器人动。
