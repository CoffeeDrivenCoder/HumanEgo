# G1 末端控制 Frame 验证

目的：确认 `RobotController.set_end_effector_pose_control()` 接收的目标是否为 `base_link` 下的 `arm_right_link7` pose。

## 机器人端运行

先只读观察，不会发控制：

```bash
cd ~/桌面/HumanEgo
bash scripts/run_g1_verify_ee_control_to_public_server.sh
```

保持当前位置测试，会发送当前 `arm_right_link7` pose：

```bash
cd ~/桌面/HumanEgo
G1_EE_MODE=hold \
G1_EE_CONFIRM=RUN_CONTROL \
G1_EE_TAG=ee_hold_right \
bash scripts/run_g1_verify_ee_control_to_public_server.sh
```

小位移测试，默认右臂 `base_link +Z 0.01m`，执行后会回到移动前 pose：

```bash
cd ~/桌面/HumanEgo
G1_EE_MODE=move \
G1_EE_CONFIRM=RUN_CONTROL \
G1_EE_TAG=ee_move_z_1cm_right \
G1_EE_DELTA_AXIS=z \
G1_EE_DELTA_M=0.01 \
bash scripts/run_g1_verify_ee_control_to_public_server.sh
```

## 判断标准

看上传报告 `ee_control_frame_report.json` 里的：

- `hold_delta_m`：应接近 `[0, 0, 0]`。
- `move_delta_analysis.commanded_delta_m`：脚本发出的目标位移。
- `move_delta_analysis.observed_delta_m`：`motion_status` 读回的实际位移。

如果 `observed_delta_m` 和 `commanded_delta_m` 的主轴、方向、量级一致，就可以认为控制目标是 `base_link` 下的 `arm_right_link7` pose。

注意：`move` 模式会真实发控制命令。首次测试建议 `0.01m`，手放急停附近。

## 2026-06-22 真机结果

三步验证均已上传并检查：

- observe：成功读取 `arm_right_link7`。
- hold：`flat_dict` pose 格式调用成功，保持当前位置误差约 `0.1mm` 量级。
- move：发送 `base_link +Z 0.01m`，读回位移：

```text
commanded_delta_m = [0.0, 0.0, 0.0100]
observed_delta_m  = [0.00036, -0.00004, 0.00931]
```

结论：

```text
set_end_effector_pose_control 接收 base_link 坐标系下的 arm_right_link7 pose。
right_pose 格式为 flat dict: {x, y, z, qx, qy, qz, qw}。
control_group 使用 ["right_arm"]。
```
