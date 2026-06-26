# 2026-06-22 G1 HumanEgo 调试总结

## 今天完成了什么

### 1. 确认 G1 末端控制接口语义

完成了 `set_end_effector_pose_control()` 真机验证：

```text
控制目标：base_link 坐标系下的 arm_right_link7 pose
pose 格式：{x, y, z, qx, qy, qz, qw}
control_group：["right_arm"]
```

小位移测试结果：

```text
发送：base_link +Z 0.0100m
读回：[+0.00036, -0.00004, +0.00931]m
```

结论：G1 末端 pose 控制链路可用。

### 2. 打通 server-client 推理架构

明确最终架构：

```text
机器人端：采集 RGB-D / 机器人状态 / 执行控制
服务器端：视觉算法 / HumanEgo 模型 / 坐标转换
```

今天已实现：

- 服务器端 HumanEgo inference server。
- 机器人端 client dry-run。
- 机器人端只上传 RGB、K、当前 TCP/base-camera 状态。
- 服务器返回 `T_link7_target_in_base` 和 `right_pose_flat_limited`。

验证结果：

```text
server response ok
server 推理耗时约 0.10s
端到端请求优化后约 0.29s
```

### 3. 优化图像传输

将机器人端上传图像从原始 `1280x800` 优化到模型输入尺寸：

```text
模型输入：320x240
机器人上传：320x240
K 已同步缩放
```

传输从约 `1.1s` 降到约 `0.29s`。

### 4. 实现交互式单步控制脚本

新增交互式控制方式：

```text
每次请求一次模型输出
打印目标动作
人工回车确认
只执行一步
执行后读回实际运动
```

默认：

- 使用 `3cm` 限幅目标。
- 不执行 gripper。
- 不连续闭环。

### 5. 补上服务器端 RGB-D 物体位姿估计入口

发现之前 `obj1/面包` 是固定假位姿，只能用于链路测试，不能用于真实抓取。

今天已补：

```text
object_source=fixed：链路调试
object_source=rgbd：服务器用机器人 RGB-D 估计真实面包/盘子 pose
```

服务器端现在可以使用：

```text
DINO-SAM + depth lifting + PCA
```

来得到真实：

```text
T_obj_in_cam
```

## 当前关键结论

现在已经完成：

- G1 RGB-D 获取。
- G1 相机内参/外参获取。
- corobot FK + `T_base_camera` 验证。
- G1 TCP/link7 坐标链验证。
- G1 末端控制 frame 验证。
- server-client HumanEgo 推理链路。
- 模型输出到 G1 link7 目标的转换。
- 交互式单步控制脚本。
- 服务器端真实 RGB-D 物体位姿估计入口。

还没完成的是：

```text
真实 RGB-D 下 bread/plate 的物体位姿稳定性验证
```

这一步完成后，才能进入真正的面包抓取测试。

## 明天怎么做

### Step 1：启动服务器真实物体位姿模式

服务器端：

```bash
cd /home/ubuntu/projects/wangk/HumanEgo
git pull origin main

G1_HUMANEGO_OBJECT_SOURCE=rgbd \
bash scripts/start_g1_humanego_inference_server.sh
```

### Step 2：机器人端发送 RGB-D dry-run

机器人端：

```bash
cd ~/桌面/HumanEgo
git pull origin main

G1_HUMANEGO_SEND_DEPTH=true \
bash scripts/run_g1_humanego_client_dry_run_to_public_server.sh
```

先检查：

```text
input_summary.object_source_used 是否为 rgbd
objects.obj1.T_in_cam 是否像真实面包位置
objects.obj2.T_in_cam 是否像真实盘子位置
policy 输出目标是否合理
```

### Step 3：如果物体 pose 合理，再做交互式单步

机器人端：

```bash
G1_HUMANEGO_CONFIRM=RUN_CONTROL \
G1_HUMANEGO_SEND_DEPTH=true \
bash scripts/run_g1_humanego_interactive_step_to_public_server.sh
```

每一步：

```text
看目标
回车执行一步
观察机器人运动方向
s 跳过异常目标
q 退出
```

### Step 4：逐步打开完整能力

顺序建议：

1. 只执行右臂 pose，不动 gripper。
2. 运动方向稳定后，再执行 gripper。
3. 单步稳定后，再低频连续闭环。
4. 最后再尝试完整 serve_bread。

## 注意

明天不要一开始就连续闭环，也不要一开始就执行 gripper。先确认真实面包/盘子的 `T_obj_in_cam` 是对的，再让机器人动。
