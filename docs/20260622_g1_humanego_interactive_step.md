# G1 HumanEgo 交互式单步执行

用途：每次从服务器获取一次 HumanEgo 动作，人工确认后只执行一步。

## 服务器端

保持 HumanEgo inference server 运行：

```bash
cd /home/ubuntu/projects/wangk/HumanEgo
bash scripts/start_g1_humanego_inference_server.sh
```

## 机器人端

先确认机器人周围安全、手在急停附近，然后运行：

```bash
cd ~/桌面/HumanEgo
git pull origin main
G1_HUMANEGO_CONFIRM=RUN_CONTROL \
bash scripts/run_g1_humanego_interactive_step_to_public_server.sh
```

每轮会打印：

- done probability
- 模型目标位移
- gripper 目标值
- `right_pose`

提示符：

```text
[Enter]=execute one step, s=skip/replan, q=quit >
```

含义：

- 直接回车：执行当前一步。
- `s`：跳过当前目标，重新采集并请求下一步。
- `q`：退出。

默认使用服务器返回的 `right_pose_flat_limited`，即配置里的 `max_pos_step=0.03m` 限幅目标。默认不执行 gripper，只执行右臂末端 pose。

如果想完全使用 raw target：

```bash
G1_HUMANEGO_CONFIRM=RUN_CONTROL \
G1_HUMANEGO_TARGET_SOURCE=raw \
bash scripts/run_g1_humanego_interactive_step_to_public_server.sh
```

第一轮建议先用默认 `limited`。
