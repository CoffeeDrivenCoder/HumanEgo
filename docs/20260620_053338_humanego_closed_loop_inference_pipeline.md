# HumanEgo 在 G1 上复现：当前补齐状态

创建时间：`20260620_053338`

这份文档只记录两件事：
- 我们已经补齐/写了什么。
- 接下来还缺什么，需要你在 G1 真机或配置中确认。

## 1. 已经补齐的信息

### 1.1 模型路径

已补齐：
- checkpoint 目录：`/data/wangk/checkpoints/humanego/serve_bread/HumanEgo`
- policy 权重：`/data/wangk/checkpoints/humanego/serve_bread/HumanEgo/latest.pt`
- 训练配置：`/data/wangk/checkpoints/humanego/serve_bread/HumanEgo/config.json`
- 数据统计：`/data/wangk/checkpoints/humanego/serve_bread/HumanEgo/dataset_stats.json`

还没确认：
- `dataset_stats.json` 里 action/object/gripper 的字段和单位。

已确认：
- `serve_bread` 是 Aria/ego 数据训练，不是机器人数据训练。
- `single_hand=true`，`single_hand_side=right`。
- `frame_mode=camera_frame`，`action_mode=absolute`。
- `use_region_attn=true`，后续推理需要 anchor object 的 UV。

### 1.2 G1 相机

已补齐：
- 相机配置文件：`/home/ubuntu/projects/wangk/HumanEgo/cfg/inference/g1_head_rgbd.yaml`
- 默认使用 G1 `head` RGB 和 `head_depth` depth。
- G1 客户端运行 `G1/parameter.py` 读取机器人内外参。
- 参数读取优先级：`/data/parameters` -> `http://10.42.0.101:8849/camera_parameters` -> 仓库 `G1/parameters` -> 仓库 `G1/parameters.zip`。
- 默认 RGB 从 SDK 的 RGB 转成推理侧 BGR。
- 默认 depth 从 `uint16 mm` 转成 `float32 m`。

已在真机确认：
- `head` RGB 可获取，shape 为 `800 x 1280 x 3`，dtype 为 `uint8`。
- `head_depth` depth 可获取，shape 为 `800 x 1280 x 1`，dtype 为 `uint16`。
- packet 显示 depth 格式为 `RS2_FORMAT_Z16`，按毫米原始深度处理。
- RGB 和 depth 都约 `30 FPS`。
- depth 有效像素比例约 `0.71`。
- `get_image_nearest("head_depth", rgb_ts)` 可用。
- G1 客户端已通过 `http://10.42.0.101:8849/camera_parameters` 读取到 `head` 内参 `K` 和外参 `T`。

注意：
- 当前 SDK 的 `get_image_shape("head")` / `get_image_shape("head_depth")` 会报 `KeyError`，后续代码不要依赖它，直接用 `get_latest_image()` 返回 ndarray 的 shape。
- `parameters.py` 必须在 G1 客户端/机器人端运行，用于从机器人本机 `/data/parameters` 或内部 HTTP 参数服务拿真实相机内外参。当前真机验证来源是 HTTP 参数服务。

### 1.3 G1 控制接口结论

根据 G1 PDF，目前结论是：
- 当前关节状态可从 `RobotDds.arm_joint_states()`、`gripper_states()` 等接口读。
- 当前末端状态优先从 `RobotController.get_motion_status()` 读。
- 末端控制优先用 `RobotController.set_end_effector_pose_control()`。
- 第一版不自己写 IK，IK 交给 G1 底层控制器。
- FK 只作为兜底：如果 `get_motion_status()` 读不到末端 pose，再考虑 URDF/corobot FK。

还没确认：
- 真机 `get_motion_status()["frames"]` 的真实结构。
- 左右手末端 frame 名称。
- 返回 pose 是 wrist/link7、gripper base，还是 gripper center。
- `set_end_effector_pose_control()` 的真机 `control_group` 格式和安全调用方式。

### 1.4 Object pose / perception 路线

已确认主路线：
- 不需要手动标定每个物体 pose。
- 第一版使用项目已有模块：`/home/ubuntu/projects/wangk/HumanEgo/inference/object_pose_rgbd.py`
- 流程：`RGB-D + K -> DINO-SAM 分割 -> depth 提升 3D 点 -> PCA 拟合 6D pose`。
- 输出：`ObjectState(T_in_cam, kpts_local)`，其中 `T_in_cam` 是物体在相机坐标系下的 6D pose。

serve_bread 当前物体 prompt 参考：
- `obj1: "piece of bread ."`
- `obj2: "a plate ."`

还没接入：
- `run_inference.py` 当前默认 `ReferencePerception` 还是占位实现。
- 后面需要把推理入口切到 `RGBDObjectPosePerception`。

还没确认：
- G1 真机 RGB-D 和 `K` 是否能让 DINO-SAM + depth lifting 稳定输出物体点云。
- DINO-SAM 权重和配置路径是否在真机环境可用。
- prompt 是否需要根据真实场景微调。

## 2. 已经写了哪些推理代码

### 2.1 已写

`/home/ubuntu/projects/wangk/HumanEgo/inference/G1Camera.py`

已实现：
- `G1HeadRGBDCamera`
- 加载 G1 相机配置。
- 加载 `head` 相机内参 `K`。
- 调用 `a2d_sdk.robot.CosineCamera` 获取 RGB-D。
- RGB 转 BGR。
- depth 毫米转米。
- 可选用 `get_image_nearest()` 做 RGB-D 时间对齐。
- 返回 HumanEgo 推理接口需要的 `Frame(rgb, depth_m, K)`。

`/home/ubuntu/projects/wangk/HumanEgo/cfg/inference/g1_head_rgbd.yaml`

已实现：
- G1 head RGB-D 相机配置。
- RGB/depth 名称。
- 参数路径。
- RGB/depth 格式约定。

`/home/ubuntu/projects/wangk/HumanEgo/inference/G1Geometry.py`

已实现：
- 从 G1 URDF 固定关系写出 `link7 -> gripper_center/TCP`。
- 右手 `T_tcp_in_link7`：TCP 相对 link7 沿 `+Z` 偏 `0.14308m`，旋转等价于 `Rz(pi)`。
- 第一版 `T_align = T_hand_in_tcp`，用于把 G1 TCP frame 对齐到 HumanEgo/Aria 右手 midpoint frame。

`/home/ubuntu/projects/wangk/HumanEgo/cfg/inference/g1_serve_bread_right.yaml`

已实现：
- G1 单右手 `serve_bread` 推理配置。
- 指向 `/data/wangk/checkpoints/humanego/serve_bread/HumanEgo/latest.pt`。
- 使用 G1 head RGB-D 相机配置。
- 写入第一版 `T_align`。

`/home/ubuntu/projects/wangk/HumanEgo/scripts/g1_dry_run_tcp_align.py`

已实现：
- 不依赖 FK、不调用控制，只打印 checkpoint 模式、G1 TCP、`T_align` 和示例目标转换。

`/home/ubuntu/projects/wangk/HumanEgo/scripts/g1_print_target_pose.py`

已实现：
- 给定一个 HumanEgo hand pose，只打印对应的 G1 TCP/link7 camera-frame target，不运动机器人。

`/home/ubuntu/projects/wangk/HumanEgo/inference/run_inference.py`

已实现：
- 增加 camera factory。
- 当相机配置 `type` 是 `g1_cosine_rgbd` / `g1_head_rgbd` / `g1` 时，自动使用 `G1HeadRGBDCamera`。
- 原来的 RealSense 路径仍保留。

已做基础验证：
- `G1Camera.py` 和 `run_inference.py` 通过 Python 语法编译。
- 用假的 `CosineCamera` 模拟过 RGB/depth 输入，验证了 BGR 转换、depth 转米、K 读取。

### 2.2 还没写

还没写的推理/控制代码：
- `G1RobotArm` adapter。
- 从 `get_motion_status()` 解析当前 EE pose 的代码。
- `T_cam_in_base` / `T_base_in_cam` 配置和转换代码。
- `move_ee_in_cam()` 调用 `set_end_effector_pose_control()` 的代码。
- G1 gripper adapter。
- `T_align` 已有第一版配置，但还没接入 `G1RobotArm`。
- G1 上的安全限幅、workspace 限制、急停保护逻辑。
- 真机 debug 脚本：保存 RGB-D、打印 motion status、单步测试末端控制。

还没接通的完整闭环：
- checkpoint 实际加载。
- policy 输入 token 构造检查。
- `RGBDObjectPosePerception` 接入 `run_inference.py`。
- clean image。
- 模型输出到 G1 末端控制的完整执行链路。

## 3. 需要你补齐或真机确认的信息

优先级最高：

1. G1 SDK 环境  
   确认真机上能 import `a2d_sdk.robot.CosineCamera`、`RobotDds`、`RobotController`。

2. 相机真机输出  
   保存一帧 `head` RGB、`head_depth` depth，确认 shape、单位、对齐情况。

3. 当前末端 pose  
   打印 `RobotController.get_motion_status()`，尤其是 `frames` 字段。

4. 末端控制接口  
   小范围测试 `set_end_effector_pose_control()`，确认 `control_group`、坐标系、四元数顺序和安全模式。

5. camera-base 外参  
   需要等 `corobot`/FK 补齐后验证 `T_base_camera = T_base_head_pitch(q) * T_head_pitch_camera`。

6. `T_align`  
   已有第一版：G1 gripper center/TCP -> HumanEgo/Aria 右手 midpoint frame。后续需要用真机投影和小动作验证方向。

7. gripper 方向和值域  
   确认 G1 `move_gripper()` 中 `0/1` 分别对应张开还是闭合。

8. object pose / perception  
   不需要重新选方案。确认 `object_pose_rgbd.py` 依赖的 DINO-SAM 配置/权重可用，并用 G1 RGB-D 测试是否能输出 `T_obj_in_cam`。

## 4. 下一步最小实施顺序

1. 先在 G1 真机跑相机，保存 RGB-D 和 `K`。
2. 用 `object_pose_rgbd.py` 的路线测试 DINO-SAM + depth lifting + PCA，确认能输出 bread/plate 的 `T_obj_in_cam`。
3. 明天补 `corobot`，验证 `T_base_camera`。
4. 打印 `get_motion_status()`，确认末端 pose frame 和控制 frame。
5. 写 `G1RobotArm` adapter，只做读取当前 pose，不执行。
6. 加 `T_base_camera`，打通 `T_tcp_in_base -> T_tcp_in_cam`。
7. 只打印 HumanEgo policy 输出和 G1 TCP/link7 目标 pose。
8. 数值合理后，再小范围测试 `set_end_effector_pose_control()`。

## 5. 真机接口验证脚本

已新增：
- G1 端采集脚本：`/home/ubuntu/projects/wangk/HumanEgo/scripts/g1_collect_diagnostics.py`
- G1 端相机 T 语义验证脚本：`/home/ubuntu/projects/wangk/HumanEgo/scripts/g1_verify_camera_transform.py`
- 接收端 HTTP 脚本：`/home/ubuntu/projects/wangk/HumanEgo/scripts/g1_diagnostics_receiver.py`
- 服务器端启动脚本：`/home/ubuntu/projects/wangk/HumanEgo/scripts/start_g1_diagnostics_receiver_server.sh`
- G1 端上传脚本：`/home/ubuntu/projects/wangk/HumanEgo/scripts/run_g1_diagnostics_to_public_server.sh`
- G1 端 T 验证上传脚本：`/home/ubuntu/projects/wangk/HumanEgo/scripts/run_g1_verify_transform_to_public_server.sh`

运行位置：
- `g1_collect_diagnostics.py` 跑在 G1 机器人端，必须在能 import `a2d_sdk.robot` 的 G1 SDK Python 环境里运行。
- `G1/parameter.py` 也跑在 G1 机器人端，用于读取机器人本机相机参数。
- `g1_diagnostics_receiver.py` 跑在服务器/接收端，不需要 `a2d_sdk`。

当前服务器端口映射：
- SSH：`ubuntu@111.0.22.33 -p 51001`
- HTTP 诊断接收：公网 `111.0.22.33:30002` -> 服务器本地 `8000`
- 所以接收端监听服务器本地 `8000`，G1 端上传到 `http://111.0.22.33:30002/upload`。

默认采集内容：
- `a2d_sdk` import 是否成功。
- `CosineCamera` 的 `head` / `head_depth` 图像、shape、timestamp。
- G1 相机内参/外参读取结果。
- `RobotDds` 关节状态、gripper 状态、nearest 状态接口。
- `RobotController.get_motion_status()` 原始返回。
- SDK 对象公开方法和关键方法签名。
- 所有结果打包为 `g1_diag_*.zip`。

默认不会运动机器人。

接收端电脑先运行：

```bash
cd /home/ubuntu/projects/wangk/HumanEgo
bash scripts/start_g1_diagnostics_receiver_server.sh
```

G1 机器人端运行：

```bash
cd /home/ubuntu/projects/wangk/HumanEgo
python3 G1/parameter.py
bash scripts/run_g1_diagnostics_to_public_server.sh
```

如果只想重测相机数据，先不采机器人状态：

```bash
cd /home/ubuntu/projects/wangk/HumanEgo
G1_DIAG_TAG=camera_retry bash scripts/run_g1_diagnostics_to_public_server.sh \
  --skip-robot \
  --camera-warmup-s 2.0 \
  --camera-tries 30 \
  --camera-sleep-s 0.1
```

如果要验证参数里的 `T` 到底是不是完整 `T_cam_in_base`，在 G1 端运行：

```bash
cd /home/ubuntu/projects/wangk/HumanEgo
bash scripts/run_g1_verify_transform_to_public_server.sh
```

这个脚本会：
- 抓一帧 RGB-D。
- 读取 `parameters.py` 中的 `K` 和 `T`。
- 读取 `get_motion_status()` 中的 `arm_right_link7`。
- 使用 URDF 固定关系计算 `link7 -> gripper_center`。
- 分别测试 `T` 是 `T_cam_in_base` 和 `T_base_in_cam` 两种解释。
- 输出 `projection_overlay.png` 和 `transform_report.json`。

如果先不上传，也可以只在 G1 本地保存：

```bash
python3 scripts/g1_collect_diagnostics.py --samples 3 --tag first_g1_check
```

可选 object pose 测试：

```bash
python3 scripts/g1_collect_diagnostics.py \
  --samples 1 \
  --tag object_pose_check \
  --run-object-pose \
  --upload-url http://111.0.22.33:30002/upload
```

控制接口测试暂时不要第一轮运行。等先看完 `get_motion_status()` 里的 frame 名称后，再决定是否运行：

```bash
python3 scripts/g1_collect_diagnostics.py \
  --samples 1 \
  --tag ee_hold_check \
  --enable-control \
  --confirm-control RUN_CONTROL \
  --test-ee-hold \
  --control-side right \
  --ee-frame-hint 真实末端frame名称
```
