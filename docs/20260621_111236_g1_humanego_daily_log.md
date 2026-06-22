# G1 HumanEgo 今日进度日志

时间：`20260621_111236`

## 今天做了什么

1. 验证 G1 头部相机 RGB-D 可用
   - `head` RGB：`800 x 1280 x 3`
   - `head_depth`：`800 x 1280 x 1`
   - depth 是 `uint16 mm`
   - `get_image_nearest("head_depth", rgb_ts)` 可用

2. 验证相机参数服务可用
   - `parameters.py` 可在 G1 客户端读取：
     - `head_intrinsic_params.json`
     - `head_extrinsic_params.json`
   - 参数来源：
     - `http://10.42.0.101:8849/camera_parameters/...`
   - 返回的 `T` 和 G1 文档里的 `T_head_pitch_camera` 很接近，基本按固定头部相机外参处理。

3. 验证 `corobot` 当前不可用
   - G1 端报告：
     - `No module named 'corobot'`
   - 所以今天还不能用文档里的官方 `compute_head_fk(...)`。

4. 验证当前手写 URDF FK 还没对齐
   - 夹爪在图像里可见。
   - 但用当前 URDF FK + 参数服务 `T` 投影时，右手候选点都在相机后方，`z < 0`。
   - 结论：相机没问题，问题在 `T_base_head_pitch(q)` / FK frame 对齐。

5. 完成 G1 TCP 和第一版 `T_align`
   - 新增：`/home/ubuntu/projects/wangk/HumanEgo/inference/G1Geometry.py`
   - 从 URDF 写出：
     - `link7 -> gripper_center/TCP`
     - TCP 偏移：`0.14308 m`
   - 新增 G1 单右手配置：
     - `/home/ubuntu/projects/wangk/HumanEgo/cfg/inference/g1_serve_bread_right.yaml`
   - `serve_bread` 已确认是 Aria/ego 数据训练：
     - `single_hand=true`
     - `single_hand_side=right`
     - `frame_mode=camera_frame`
     - `action_mode=absolute`
   - 因此第一版 `T_align` 使用 G1 TCP -> HumanEgo/Aria right-hand midpoint frame，不用单位阵。

6. 新增不开控制的检查脚本
   - `/home/ubuntu/projects/wangk/HumanEgo/scripts/g1_dry_run_tcp_align.py`
   - `/home/ubuntu/projects/wangk/HumanEgo/scripts/g1_print_target_pose.py`
   - 作用：只打印 TCP、`T_align`、目标 pose 转换，不运动机器人。

## 现在还差什么

距离“直接运行 HumanEgo 推理并驱动 G1 运动”还差：

1. `corobot` / 官方 FK
   - 需要算准：
     - `T_base_head_pitch(q)`
     - `T_base_camera = T_base_head_pitch(q) * T_head_pitch_camera`

2. 相机到 base 的方向验证
   - 补 FK 后重新跑投影验证。
   - 目标：左右夹爪 TCP 投影落在图像中，且相机坐标 `z > 0`。

3. G1 末端控制接口验证
   - 确认 `set_end_effector_pose_control()` 控制的是：
     - link7
     - gripper base
     - 还是 gripper center/TCP
   - 确认 `control_group`、四元数顺序、目标坐标系。

4. `G1RobotArm` adapter
   - 读取当前 TCP pose。
   - 把 HumanEgo 输出的 camera-frame TCP target 转成 G1 控制目标。
   - gripper 开合归一化。

5. Object pose / perception 接入
   - 用 G1 RGB-D 测 `object_pose_rgbd.py` 能否稳定输出 bread/plate 的 `T_obj_in_cam`。
   - 确认 DINO-SAM 配置和权重路径。

## 明天建议步骤

1. 在 G1 端补齐 `corobot`
   - 确保可以 import：
     - `corobot.utils.kinematics.Kinematics`
   - 确保可以调用：
     - `compute_head_fk(head_yaw, head_pitch, waist_pitch, waist_height)`

2. 重新跑相机外参验证

```bash
cd ~/桌面/HumanEgo
bash scripts/run_g1_verify_transform_to_public_server.sh
```

3. 我检查新上传数据
   - 看 `corobot_head_fk.ok`
   - 看 `T_base_camera` 候选投影
   - 确认 `T_head_pitch_camera` 是否需要取逆

4. 如果 FK 验证通过
   - 写 `G1RobotArm` adapter 的只读版本。
   - 只打印：
     - 当前 TCP in base
     - 当前 TCP in camera
     - HumanEgo hand target in camera
     - G1 TCP/link7 target

5. 测 object pose
   - 跑 `object_pose_rgbd.py` 路线。
   - 确认 bread/plate 的 `T_obj_in_cam` 合理。

6. 最后再做小范围控制
   - 先不要接 HumanEgo policy。
   - 手写一个很小的 TCP 位移目标。
   - 成功后再接 policy 输出。

## 20260622 补充：corobot FK 已验证

上传数据：
- `g1_T_verify_20260622_064259_corobot_fk_check`

结论：
- `corobot_head_fk.ok=true`
- `corobot` 使用的 URDF：
  - `/home/ke/miniconda3/envs/a2d/corobot/urdf_solver/A2D_viz.urdf`
- 参数服务返回的 `T` 可以按 `T_head_pitch_camera` 使用，不需要取逆。
- 运行时使用：

```text
T_base_camera = T_base_head_pitch(q) @ T_head_pitch_camera
```

投影结果：
- `I_corobot_assume_T_param_is_cam_in_head_pitch` 成立。
- 右手 TCP 投影：
  - `u=624.7`
  - `v=683.4`
  - `z=0.453m`
- 同像素 depth 中位数约 `0.468m`，差约 `1.5cm`，合理。

下一步更新：
- FK / camera-base 外参这关基本通过。
- 下一步写 `G1RobotArm` 只读版，先打印当前 TCP in base / camera。
- 仍然不要直接开 HumanEgo 控制。

## 20260622 补充：G1RobotArm 只读版已写

新增代码：
- `/home/ubuntu/projects/wangk/HumanEgo/inference/G1RobotArm.py`
- `/home/ubuntu/projects/wangk/HumanEgo/scripts/g1_robotarm_readonly_check.py`
- `/home/ubuntu/projects/wangk/HumanEgo/scripts/run_g1_robotarm_readonly_to_public_server.sh`

能力：
- 读取 `RobotDds` 头/腰状态。
- 用 `corobot` 算 `T_base_head_pitch(q)`。
- 读取参数服务的 `T_head_pitch_camera`。
- 计算：
  - `T_base_camera`
  - `T_tcp_in_base`
  - `T_tcp_in_cam`
  - `T_hand_in_cam`
- 把一个示例 HumanEgo hand target 转成 G1 TCP/link7 target。
- 不发送任何控制命令。

下一步需要在 G1 机器人端运行只读检查：

```bash
cd ~/桌面/HumanEgo
bash scripts/run_g1_robotarm_readonly_to_public_server.sh
```
