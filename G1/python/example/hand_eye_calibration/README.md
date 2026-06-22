# Agibot G1 / G01 头部相机手眼标定脚本

本目录脚本针对智元 Agibot G1 / G01 的 GDK 头部相机流程，不是 Unitree G1。

你现在使用的是 calib.io 棋盘纸：

```text
9x12 inner corners
Checker Size: 20 mm
```

对应脚本参数：

```text
--pattern-cols 9
--pattern-rows 12
--square-size-m 0.02
```

如果单图检测失败，并且画面里棋盘是旋转 90 度的，再尝试：

```text
--pattern-cols 12 --pattern-rows 9
```

标定前请把纸张贴平，最好贴在硬板或平整桌面上。纸张翘曲、皱褶、反光会直接影响内参和后续手眼结果。

## 0. 依赖检查

在 GDK Python 环境中运行：

```bash
cd /home/ubuntu/projects/wangk/RoboClaw

python3 - <<'PY'
import numpy
import scipy
import cv2
from PIL import Image
from a2d_sdk.robot import RobotDds, CosineCamera
from corobot.utils.kinematics import Kinematics
print("ok")
PY
```

如果缺包：

```bash
python3 -m pip install numpy scipy opencv-python pillow
```

AprilTag 流程才需要：

```bash
python3 -m pip install pupil-apriltags
```

## 1. 先做相机内参标定

内参标定只需要头部相机和棋盘格，不需要 URDF，也不需要 IK 或机械臂运动。

### 1.1 采集内参图片

把棋盘固定/拿稳，让头部相机能看到完整棋盘。建议采 25 到 40 张，最少不要少于 10 张。

采集时覆盖这些画面：

- 棋盘在画面中心、左上、右上、左下、右下。
- 棋盘离相机近一些、远一些。
- 棋盘有轻微倾斜，不要所有图片都正对相机。
- 每张必须完整看到所有内角点。

命令：

```bash
cd /home/ubuntu/projects/wangk/RoboClaw

python3 docs/python/example/hand_eye_calibration/collect_checkerboard_intrinsics.py \
  --output-dir /tmp/agibot_g1_checkerboard_intrinsics/run_001 \
  --camera-name head \
  --pattern-cols 9 \
  --pattern-rows 12 \
  --square-size-m 0.02 \
  --num-samples 30 \
  --save-annotated
```

脚本会每次提示按 Enter。检测成功才保存样本；如果一直检测不到，先看第 1.3 节。

输出：

```text
/tmp/agibot_g1_checkerboard_intrinsics/run_001/
  metadata.json
  intrinsics_samples.jsonl
  images/0001_head.jpg ...
  annotated/0001_head.jpg ...
```

### 1.2 求解内参

```bash
cd /home/ubuntu/projects/wangk/RoboClaw

python3 docs/python/example/hand_eye_calibration/solve_checkerboard_intrinsics.py \
  --data-dir /tmp/agibot_g1_checkerboard_intrinsics/run_001 \
  --save-annotated
```

输出核心文件：

```text
/tmp/agibot_g1_checkerboard_intrinsics/run_001/intrinsics.json
```

重点看终端输出：

```text
RMS reprojection error
Mean reprojection error
Max reprojection error
```

建议标准：

```text
RMS < 0.5 px   比较好
RMS < 1.0 px   通常可用
RMS > 1.0 px   建议重采，检查纸是否平整、图像是否模糊、角点是否覆盖画面边缘
```

### 1.3 单张图片调试棋盘检测

如果采集脚本提示检测不到，可以拿任意一张保存的图片或你手动保存的 head 相机图片测试：

```bash
cd /home/ubuntu/projects/wangk/RoboClaw

python3 docs/python/example/hand_eye_calibration/detect_checkerboard_image.py \
  --image /tmp/agibot_g1_checkerboard_intrinsics/run_001/images/0001_head.jpg \
  --pattern-cols 9 \
  --pattern-rows 12 \
  --square-size-m 0.02 \
  --annotated-output /tmp/agibot_g1_checkerboard_intrinsics/debug_checkerboard.jpg
```

如果失败，尝试交换行列：

```bash
python3 docs/python/example/hand_eye_calibration/detect_checkerboard_image.py \
  --image /tmp/agibot_g1_checkerboard_intrinsics/run_001/images/0001_head.jpg \
  --pattern-cols 12 \
  --pattern-rows 9 \
  --square-size-m 0.02 \
  --annotated-output /tmp/agibot_g1_checkerboard_intrinsics/debug_checkerboard_12x9.jpg
```

## 2. 再做棋盘格手眼外参标定

完成内参后，才能用棋盘格估计 `T_camera_board`，再求：

```text
T_head_pitch_camera
```

运行时 RoboClaw 使用：

```text
T_base_camera(q) = T_base_head_pitch(q) * T_head_pitch_camera
```

其中：

- `T_base_head_pitch(q)` 来自 G1/G01 的 GDK FK，输入头部和腰部关节状态。
- `T_head_pitch_camera` 是本流程要求出的固定外参。
- 棋盘在手眼采集期间必须固定不动。

### 2.1 采集手眼数据

把棋盘固定在桌面上，整个采集过程不要移动棋盘。移动头部和腰部，让棋盘在相机画面中出现在不同位置和角度。

```bash
cd /home/ubuntu/projects/wangk/RoboClaw

python3 docs/python/example/hand_eye_calibration/collect_checkerboard_hand_eye.py \
  --output-dir /tmp/agibot_g1_checkerboard_hand_eye/run_001 \
  --intrinsics-json /tmp/agibot_g1_checkerboard_intrinsics/run_001/intrinsics.json \
  --camera-name head \
  --pattern-cols 9 \
  --pattern-rows 12 \
  --square-size-m 0.02 \
  --num-samples 30 \
  --save-annotated
```

推荐姿态覆盖：

- 头 yaw 左 / 中 / 右。
- 头 pitch 上 / 中 / 下。
- 腰部有不同姿态。
- 棋盘在画面中心和四周都出现过。
- 每次采集前确认棋盘完整可见。

### 2.2 求解手眼外参

如果 GDK 环境可以自动找到 `A2D_viz.urdf`：

```bash
cd /home/ubuntu/projects/wangk/RoboClaw

python3 docs/python/example/hand_eye_calibration/solve_checkerboard_hand_eye.py \
  --data-dir /tmp/agibot_g1_checkerboard_hand_eye/run_001
```

如果自动查找失败，显式传入 URDF：

```bash
python3 docs/python/example/hand_eye_calibration/solve_checkerboard_hand_eye.py \
  --data-dir /tmp/agibot_g1_checkerboard_hand_eye/run_001 \
  --urdf-path /path/to/A2D_viz.urdf
```

输出：

```text
/tmp/agibot_g1_checkerboard_hand_eye/run_001/t_head_pitch_camera.yaml
/tmp/agibot_g1_checkerboard_hand_eye/run_001/t_head_pitch_camera.json
```

### 2.3 验证手眼结果

```bash
cd /home/ubuntu/projects/wangk/RoboClaw

python3 docs/python/example/hand_eye_calibration/validate_checkerboard_hand_eye.py \
  --data-dir /tmp/agibot_g1_checkerboard_hand_eye/run_001 \
  --calibration /tmp/agibot_g1_checkerboard_hand_eye/run_001/t_head_pitch_camera.yaml
```

重点看：

```text
Position error mean
Position error rmse
Position error max
```

初期目标：

```text
max < 0.02 m
```

较好结果：

```text
max < 0.005 到 0.01 m
```

## 3. AprilTag 备用流程

本目录仍保留 AprilTag 脚本：

```text
collect_hand_eye_data.py
solve_hand_eye.py
validate_hand_eye.py
detect_apriltag_image.py
```

如果后续你换成 AprilTag 标定板，可以使用这些脚本。当前这张 calib.io 棋盘纸请优先使用 `checkerboard_*` 脚本。

## 4. 常见问题

如果棋盘检测失败：

1. 确认参数是内角点数量，不是黑白格数量。你的纸用 `9 x 12`。
2. 确认 `--square-size-m 0.02`，单位是米。
3. 棋盘必须完整进入画面，不要裁掉边缘。
4. 纸面要平，不能明显弯曲或起皱。
5. 避免强反光、阴影和运动模糊。
6. 检测失败时尝试 `--pattern-cols 12 --pattern-rows 9`。

如果内参误差大：

1. 删除模糊、过曝、角点靠近边缘但不清晰的图片。
2. 增加不同距离和不同画面位置的样本。
3. 重新贴平棋盘纸。
4. 确认采集和使用的是同一个相机分辨率。

如果手眼误差大：

1. 手眼采集期间棋盘是否移动过。
2. 内参 `intrinsics.json` 是否来自同一个相机、同一个分辨率。
3. 头部和腰部关节状态是否正常读取。
4. URDF 是否是 G1/G01 GDK 对应的 `A2D_viz.urdf`。
5. 采集姿态变化是否足够大。
