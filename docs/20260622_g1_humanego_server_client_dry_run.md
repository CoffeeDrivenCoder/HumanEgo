# G1 HumanEgo Server-Client Dry-Run

目标架构：

```text
G1 机器人端：RGB-D / 当前 TCP 状态 / 执行控制
服务器端：HumanEgo checkpoint / 视觉模型 / policy 推理 / 目标转换
```

本轮先做 dry-run：机器人端不执行控制，只把数据发给服务器；服务器返回转换后的 G1 目标 pose。

## 服务器端

在服务器上启动 HumanEgo inference server：

```bash
cd /home/ubuntu/projects/wangk/HumanEgo
bash scripts/start_g1_humanego_inference_server.sh
```

默认监听：

```text
server local: 0.0.0.0:50051
public:       http://111.0.22.33:30003/infer
```

如果要强制 CPU：

```bash
G1_HUMANEGO_DEVICE=cpu \
bash scripts/start_g1_humanego_inference_server.sh
```

## 机器人端

机器人端 pull 后运行：

```bash
cd ~/桌面/HumanEgo
git pull origin main
bash scripts/run_g1_humanego_client_dry_run_to_public_server.sh
```

机器人端会发送：

- `head` RGB JPEG。
- 相机内参 `K`。
- `T_base_camera`。
- 当前 `T_link7_in_base`。
- 当前 `T_tcp_in_link7`。
- 当前 `T_tcp_in_cam`。
- 当前 gripper 状态。

服务器会返回：

- HumanEgo `done_prob`。
- `T_tcp_target_in_cam`。
- `T_link7_target_in_base`。
- `right_pose_flat_limited`。
- `gripper_g1_raw_0_open_120_closed`。

## 安全说明

这个 server-client dry-run 不调用：

```text
set_end_effector_pose_control
move_gripper
```

也就是说不会让机器人运动。它只验证网络、模型推理、坐标转换和目标数值。

旧的 `run_g1_humanego_dry_run_to_public_server.sh` 是机器人端本地加载模型的备用调试脚本；最终部署优先使用本文件描述的 server-client 方式。
