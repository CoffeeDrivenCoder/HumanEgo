# 1. 代码位置

这部分代码主要分布在下面几个文件里。

`src/mcp_control_demo/perception/apriltag.py`

这里负责从头部相机图像中检测 AprilTag，并得到物体或标签在相机坐标系下的位置。输出字段主要是 `position_camera_m` 或 `translation_m`，单位是米。

`src/mcp_control_demo/calibration/config.py`

这里负责读取标定配置，并提供 `camera_to_exec_point()` 和 `exec_to_camera_point()` 两个坐标转换接口。当前项目中 `exec_frame` 默认是 `base_link`，所以这里的 `camera_to_exec_point()` 实际就是把相机坐标转换到底盘坐标系。

`src/mcp_control_demo/calibration/transforms.py`

这里是底层数学工具函数，真正执行齐次变换矩阵乘法的是 `transform_point()`。

`.a2d_pkg/corobot/policy_tasks/rule_control_task.py`

这里是真实机器人任务入口。它会从当前 observation 里读取头部关节和腰部关节，通过机器人运动学计算当前 `head_pitch_link` 到 `base_link` 的位姿，再和固定相机外参 `T_head_pitch_camera` 相乘，动态得到当前相机到 `base_link` 的变换矩阵 `T_exec_camera(q)`。

`src/mcp_control_demo/control/action_builder.py`

这里负责把相机坐标系中的目标点转成 `base_link` 下的手臂执行目标，并构造 CoRobot 底层可执行的 `Action`。

# 2. 头部相机感知：从图像得到相机坐标系下的物体位置

## 2.1 从 tag 结果中取出相机坐标

位置：`src/mcp_control_demo/perception/apriltag.py`

```python
def extract_tag_position_camera(tag_pose: dict[str, Any]) -> list[float]:
    for key in ("position_camera_m", "translation_m"):
        if tag_pose.get(key) is not None:
            return [float(v) for v in tag_pose[key]]

    camera_pose = tag_pose.get("camera_pose") or {}
    if camera_pose.get("position_m") is not None:
        return [float(v) for v in camera_pose["position_m"]]

    raise ValueError("tag pose does not contain a camera-frame position")
```

这个函数的作用是：从 AprilTag 检测结果中取出标签在相机坐标系下的位置。项目里兼容了几个字段名：

```text
position_camera_m
translation_m
camera_pose.position_m
```

只要检测结果里有其中一种表示，就会返回 `[x, y, z]`。这里返回的坐标还没有转到底盘坐标系，它仍然是在 `head_camera_optical` 相机坐标系下的三维位置。

## 2.2 AprilTagPerceptionService：检测并缓存 tag 位姿

位置：`src/mcp_control_demo/perception/apriltag.py`

```python
class AprilTagPerceptionService:
    """Detect AprilTags from CoRobot observations and cache results by tag_id."""

    def __init__(
        self,
        *,
        camera_name: str = "head",
        camera_frame: str = "head_camera_optical",
        tag_family: str = "tag25h9",
        tag_size_m: float = 0.018,
        stale_after_s: float = 1.0,
        fallback_camera_params: dict[str, Any] | None = None,
    ):
        self.camera_name = camera_name
        self.camera_frame = camera_frame
        self.tag_family = tag_family
        self.tag_size_m = float(tag_size_m)
        self.stale_after_s = float(stale_after_s)
        self.fallback_camera_params = fallback_camera_params
        self._cache: dict[int, dict[str, Any]] = {}
```

这个类封装了 AprilTag 感知服务。默认使用的相机是 `head`，坐标系是 `head_camera_optical`。也就是说，机器人头部摄像头看到的目标，最开始都是在 `head_camera_optical` 坐标系下表达的。

```python
def detect_from_observation(self, observation: Any) -> dict[str, Any]:
    image = _get_camera_image(observation, self.camera_name)
    if image is None:
        return {"ok": False, "message": f"observation image not found: {self.camera_name}"}

    camera_params = _get_camera_params(observation, self.camera_name) or self.fallback_camera_params
    detections = detect_tags_from_image(
        image,
        camera_params,
        camera_frame=self.camera_frame,
        camera_name=self.camera_name,
        tag_family=self.tag_family,
        tag_size_m=self.tag_size_m,
    )

    now = time.time()
    for item in detections:
        item["timestamp_s"] = now
        self._cache[int(item["tag_id"])] = item

    return {
        "ok": True,
        "camera_name": self.camera_name,
        "camera_frame": self.camera_frame,
        "tag_family": self.tag_family,
        "tag_size_m": self.tag_size_m,
        "detections": detections,
    }
```

这个函数的作用是：从当前机器人 observation 里取出头部相机图像，然后调用 `detect_tags_from_image()` 做 AprilTag 检测。检测到的结果会按 `tag_id` 缓存在 `_cache` 里，后续可以通过 `get_tag_pose()` 直接取。

```python
def get_tag_pose(self, tag_id: int, *, allow_stale: bool = False) -> dict[str, Any]:
    tag_id = int(tag_id)
    item = self._cache.get(tag_id)
    if item is None:
        return {"ok": False, "message": f"tag_id {tag_id} not found", "tag_id": tag_id}

    age_s = time.time() - float(item.get("timestamp_s", 0.0))
    if age_s > self.stale_after_s and not allow_stale:
        return {
            "ok": False,
            "message": f"tag_id {tag_id} is stale",
            "tag_id": tag_id,
            "age_s": age_s,
            "stale_after_s": self.stale_after_s,
            "last_detection": item,
        }

    return {"ok": True, "age_s": age_s, "stale": age_s > self.stale_after_s, **item}
```

这个函数的作用是：根据 `tag_id` 返回对应 AprilTag 的位姿信息。如果缓存结果太旧，并且没有设置 `allow_stale=True`，就会返回 stale 错误，避免机械臂使用过期视觉结果。

## 2.3 真正的 AprilTag 检测函数

位置：`src/mcp_control_demo/perception/apriltag.py`

```python
def detect_tags_from_image(
    image: Any,
    camera_params: dict[str, Any] | None,
    *,
    camera_frame: str,
    camera_name: str,
    tag_family: str,
    tag_size_m: float,
) -> list[dict[str, Any]]:
    cv2, Detector = _load_runtime_deps()
    bgr = np.asarray(image)
    detection_image, pose_matrix, undistorted = _prepare_image_for_pose(cv2, bgr, camera_params)
    gray = cv2.cvtColor(detection_image, cv2.COLOR_RGB2GRAY) if detection_image.ndim == 3 else detection_image
    fx, fy, cx, cy = _camera_params_from_matrix(pose_matrix)

    detector = Detector(
        families=tag_family,
        nthreads=2,
        quad_decimate=1.5,
        quad_sigma=0.8,
        refine_edges=True,
        decode_sharpening=0.25,
    )

    raw = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(fx, fy, cx, cy),
        tag_size=float(tag_size_m),
    )

    results: list[dict[str, Any]] = []
    for det in raw:
        if det.pose_t is None or det.pose_R is None:
            continue

        position = np.asarray(det.pose_t, dtype=np.float64).reshape(3)
        rotation = np.asarray(det.pose_R, dtype=np.float64).reshape(3, 3)
        results.append(
            {
                "tag_id": int(det.tag_id),
                "tag_family": det.tag_family.decode() if hasattr(det.tag_family, "decode") else str(det.tag_family),
                "camera_name": camera_name,
                "camera_frame": camera_frame,
                "tag_size_m": float(tag_size_m),
                "position_camera_m": [float(v) for v in position.tolist()],
                "translation_m": [float(v) for v in position.tolist()],
                "distance_m": float(np.linalg.norm(position)),
                "rotation_matrix": rotation.tolist(),
                "euler_rpy_deg": _rotation_matrix_to_euler_deg(rotation),
                "center_px": [float(v) for v in np.asarray(det.center).reshape(2).tolist()],
                "corners_px": np.asarray(det.corners, dtype=np.float64).reshape(4, 2).tolist(),
                "camera_params": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "undistorted": undistorted},
            }
        )

    return results
```

这个函数的作用是：

1. 从图像和相机内参中读取 `fx, fy, cx, cy`。
2. 调用 `pupil_apriltags.Detector` 检测 AprilTag。
3. 使用 `estimate_tag_pose=True` 直接估计 tag 在相机坐标系下的 3D 位姿。
4. 把 `det.pose_t` 保存成 `position_camera_m` 和 `translation_m`。

所以，从视觉检测部分看，物体位置的来源就是：

```text
头部相机图像 + 相机内参 + AprilTag 尺寸
    -> AprilTag 位姿估计
    -> position_camera_m / translation_m
    -> head_camera_optical 坐标系下的 [x, y, z]
```

# 3. 坐标标定与转换：相机坐标系到 base_link 坐标系

## 3.1 配置文件中的坐标系约定和固定外参

位置：`.a2d_pkg/corobot/config/rule_control_task_config.yml`

```yaml
# Runtime control uses dynamic FK only:
# T_exec_camera(q) = T_base_head_pitch(q) * T_head_pitch_camera

mcp_control:
  transform_mode: dynamic_fk
  camera_frame: head_camera_optical
  exec_frame: base_link

  intrinsics:
    camera_name: head
    camera_model: Realsense-D455
    image_size:
      width: 1280
      height: 720
    fx: 645.2637329101562
    fy: 644.3807373046875
    cx: 642.1536865234375
    cy: 362.27099609375

  extrinsics:
    parent_frame: head_pitch_link
    child_frame: head_camera_optical
    T_head_pitch_camera:
      - [0.01154905419417851, 0.03633581308553096, -0.9992728996798792, -0.09309730346114839]
      - [-0.010873071465479051, -0.999275902324378, -0.03646158733116647, 0.03977041211949885]
      - [-0.9998741899179753, 0.01128626249982923, -0.011145609658446354, -0.01592936593693035]
      - [0.0, 0.0, 0.0, 1.0]

  camera_approach_axis: [0.0, 0.0, -1.0]
```

这里有两个关键点。

第一，外部传入和感知输出默认都是相机坐标系：

```text
camera_frame: head_camera_optical
```

第二，执行坐标系默认是底盘坐标系：

```text
exec_frame: base_link
```

因此代码里的 `exec_frame` 在当前配置下就是 `base_link`。也就是说：

```text
camera_to_exec_point() = camera_to_base_link_point()
```

配置文件中保存的 `T_head_pitch_camera` 是固定外参，表示相机相对于 `head_pitch_link` 的安装关系。因为机器人头和腰会动，所以不能只用一个固定的 `camera -> base_link` 外参，而是每次根据当前头部/腰部关节状态动态计算：

```text
T_exec_camera(q) = T_base_head_pitch(q) * T_head_pitch_camera
```

其中：

```text
T_head_pitch_camera：固定标定外参
T_base_head_pitch(q)：由当前头部/腰部关节状态通过 FK 算出来
T_exec_camera(q)：当前时刻 camera_frame 到 base_link 的变换
```

## 3.2 CalibrationConfig：保存标定参数并提供转换接口

位置：`src/mcp_control_demo/calibration/config.py`

```python
@dataclass(frozen=True)
class CalibrationConfig:
    """Runtime calibration used to plan in camera frame and execute in CoRobot frame."""

    camera_frame: str = "head_camera_optical"
    exec_frame: str = "base_link"
    t_exec_camera: np.ndarray | None = None
    t_head_pitch_camera: np.ndarray | None = None
    transform_mode: str = "dynamic_fk"
    urdf_path: str | None = None
    intrinsics: dict[str, Any] | None = None
    camera_approach_axis: np.ndarray = field(default_factory=lambda: np.asarray([0.0, 0.0, -1.0]))
```

这个类保存相机坐标系、执行坐标系、固定外参和运行时动态外参。

其中最重要的是两个矩阵：

```text
t_head_pitch_camera：固定外参，来自配置文件 T_head_pitch_camera
t_exec_camera：运行时动态外参，表示当前 camera_frame -> exec_frame/base_link
```

## 3.3 从配置文件读取 T_head_pitch_camera

位置：`src/mcp_control_demo/calibration/config.py`

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "CalibrationConfig":
    source = _calibration_section(data)
    extrinsics = source.get("extrinsics") or {}
    if not isinstance(extrinsics, dict):
        extrinsics = {}

    source_transform = str(extrinsics.get("source_transform") or "T_head_pitch_camera")
    t_head_pitch_camera = _first_present(
        source,
        "T_head_pitch_camera",
        (extrinsics, source_transform),
        (extrinsics, "T_head_pitch_camera"),
    )

    urdf_path = _first_present(source, "urdf_path", (extrinsics, "urdf_source"))
    transform_mode = str(source.get("transform_mode") or "dynamic_fk")

    if transform_mode != "dynamic_fk":
        raise ValueError(f"mcp_control only supports transform_mode='dynamic_fk', got {transform_mode!r}")

    if t_head_pitch_camera is None:
        raise ValueError("dynamic_fk calibration requires fixed T_head_pitch_camera")

    return cls(
        camera_frame=str(source.get("camera_frame") or "head_camera_optical"),
        exec_frame=str(source.get("exec_frame") or "base_link"),
        t_head_pitch_camera=None if t_head_pitch_camera is None else as_matrix4(t_head_pitch_camera),
        transform_mode=transform_mode,
        urdf_path=None if urdf_path is None else str(urdf_path),
        intrinsics=source.get("intrinsics"),
        camera_approach_axis=normalize_vector(source.get("camera_approach_axis", [0.0, 0.0, -1.0])),
    )
```

这个函数的作用是：从 `mcp_control` 配置节中读取标定信息。它会强制要求 `transform_mode` 是 `dynamic_fk`，并且必须存在固定相机外参 `T_head_pitch_camera`。

注意，这里读出来的只是固定外参 `t_head_pitch_camera`，还不是最终用于执行的 `t_exec_camera`。最终的 `t_exec_camera` 要在 `RuleControlTask` 里结合当前机器人关节状态动态计算。

## 3.4 给 CalibrationConfig 写入运行时动态外参

位置：`src/mcp_control_demo/calibration/config.py`

```python
def require_camera_frame(self, camera_frame: str | None) -> None:
    requested = camera_frame or self.camera_frame
    if requested != self.camera_frame:
        raise ValueError(f"unsupported camera_frame={requested!r}; configured frame is {self.camera_frame!r}")


def require_transform(self) -> np.ndarray:
    if self.t_exec_camera is None:
        raise ValueError("missing runtime T_exec_camera; dynamic_fk transform must be computed from observation")
    return self.t_exec_camera


def with_t_exec_camera(self, t_exec_camera: np.ndarray) -> "CalibrationConfig":
    return replace(self, t_exec_camera=as_matrix4(t_exec_camera))
```

这几个函数的作用是：

`require_camera_frame()` 用来检查传入坐标是不是当前配置支持的相机坐标系。当前默认只支持 `head_camera_optical`。

`require_transform()` 用来确保运行时已经有 `T_exec_camera`。如果没有这个矩阵，就不能把相机坐标转换到底盘坐标。

`with_t_exec_camera()` 用来把动态计算出来的 `T_exec_camera(q)` 写回 `CalibrationConfig`，后续 `camera_to_exec_point()` 就会使用这个矩阵。

## 3.5 相机坐标点和 base_link 坐标点的转换接口

位置：`src/mcp_control_demo/calibration/config.py`

```python
def camera_to_exec_point(
    self,
    point_camera_m: list[float] | np.ndarray,
    camera_frame: str | None = None,
) -> np.ndarray:
    self.require_camera_frame(camera_frame)
    return transform_point(self.require_transform(), point_camera_m)


def exec_to_camera_point(
    self,
    point_exec_m: list[float] | np.ndarray,
    camera_frame: str | None = None,
) -> np.ndarray:
    self.require_camera_frame(camera_frame)
    return transform_point(invert_transform(self.require_transform()), point_exec_m)
```

这两个函数是坐标转换的直接入口。

在当前配置中：

```text
exec_frame = base_link
```

所以：

```text
camera_to_exec_point(point_camera_m) = 把 head_camera_optical 下的点转到 base_link 下
exec_to_camera_point(point_exec_m) = 把 base_link 下的点反算回 head_camera_optical 下
```

## 3.6 transform_point：真正执行齐次矩阵乘法的地方

位置：`src/mcp_control_demo/calibration/transforms.py`

```python
def as_matrix4(values: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"expected 4x4 transform matrix, got shape {matrix.shape}")
    return matrix


def invert_transform(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    return np.linalg.inv(as_matrix4(matrix))


def transform_point(matrix: Sequence[Sequence[float]], point: Sequence[float]) -> np.ndarray:
    transform = as_matrix4(matrix)
    homogeneous = np.ones(4, dtype=np.float64)
    homogeneous[:3] = np.asarray(point, dtype=np.float64).reshape(3)
    return (transform @ homogeneous)[:3]
```

`transform_point()` 就是最底层的坐标点转换。它把三维点 `[x, y, z]` 扩展成齐次坐标 `[x, y, z, 1]`，然后左乘 4×4 变换矩阵：

```text
p_exec = T_exec_camera @ [x_camera, y_camera, z_camera, 1]^T
```

因为当前 `exec_frame` 是 `base_link`，所以这里也可以理解为：

```text
p_base = T_base_camera @ p_camera
```

## 3.7 RuleControlTask：根据当前头部/腰部状态动态计算 T_exec_camera

位置：`.a2d_pkg/corobot/policy_tasks/rule_control_task.py`

```python
def _calibration_for_observation(self, observation: Any) -> CalibrationConfig:
    calibration = self._calibration_config()

    if calibration.t_exec_camera is not None:
        return calibration

    if not calibration.has_dynamic_fk:
        return calibration

    return calibration.with_t_exec_camera(self._dynamic_t_exec_camera(observation, calibration))
```

这个函数的作用是：每次执行动作前，根据当前 observation 生成一个带有 `t_exec_camera` 的 `CalibrationConfig`。

如果 `t_exec_camera` 已经存在，就直接用；如果当前是 `dynamic_fk` 模式，就调用 `_dynamic_t_exec_camera()` 动态计算。

```python
def _dynamic_t_exec_camera(self, observation: Any, calibration: CalibrationConfig) -> np.ndarray:
    if calibration.exec_frame != "base_link":
        raise ValueError("dynamic_fk transform currently supports exec_frame='base_link' only")

    if calibration.t_head_pitch_camera is None:
        raise ValueError("dynamic_fk requires fixed T_head_pitch_camera calibration")

    states = _observation_states(observation)
    head = _float_list(_get(states, "head_joint_states"), 2)
    waist = _float_list(_get(states, "waist_joint_states"), 2)

    if head is None or waist is None:
        raise ValueError("dynamic_fk requires current head_joint_states and waist_joint_states in observation")

    head = normalize_head_joint_states_rad(head)

    xyzquat = self._kinematics_for_calibration(calibration).compute_head_fk(
        float(head[0]),
        float(head[1]),
        float(waist[0]),
        float(waist[1]),
    )

    t_base_head_pitch = np.eye(4, dtype=np.float64)
    t_base_head_pitch[:3, :3] = R.from_quat(xyzquat[3:]).as_matrix()
    t_base_head_pitch[:3, 3] = np.asarray(xyzquat[:3], dtype=np.float64)

    return t_base_head_pitch @ calibration.t_head_pitch_camera
```

这个函数是“头眼坐标转底盘坐标”的核心。

它做了几件事：

1. 确认执行坐标系是 `base_link`。
2. 从 observation 里读取当前头部关节 `head_joint_states` 和腰部关节 `waist_joint_states`。
3. 调用 `compute_head_fk()` 计算当前 `head_pitch_link` 在 `base_link` 下的位置和姿态。
4. 把 FK 结果转成 4×4 矩阵 `t_base_head_pitch`。
5. 再乘上固定相机外参 `T_head_pitch_camera`。
6. 得到最终的相机到 `base_link` 的运行时变换：

```text
T_exec_camera(q) = T_base_head_pitch(q) * T_head_pitch_camera
```

也就是：

```text
T_base_camera(q) = T_base_head_pitch(q) * T_head_pitch_camera
```

```python
def _kinematics_for_calibration(self, calibration: CalibrationConfig):
    urdf_path = str(self._dynamic_fk_urdf_path(calibration))
    if self._kinematics is None or self._kinematics_urdf_path != urdf_path:
        from corobot.utils.kinematics import Kinematics
        with redirect_stdout(StringIO()):
            self._kinematics = Kinematics(urdf_path)
        self._kinematics_urdf_path = urdf_path
    return self._kinematics


def _dynamic_fk_urdf_path(self, calibration: CalibrationConfig) -> Path:
    if calibration.urdf_path is not None:
        return self._resolve_config_path(calibration.urdf_path)

    from corobot.utils.fk_solver import _find_urdf_solver_dir
    return (_find_urdf_solver_dir() / "A2D_viz.urdf").resolve()
```

这两个函数负责加载 FK 所需要的 URDF。`compute_head_fk()` 依赖这里加载的机器人运动学模型。

## 3.8 头部关节单位归一化

位置：`src/mcp_control_demo/control/joint_units.py`

```python
HEAD_YAW_RAD_ABS_LIMIT = 1.5708
HEAD_PITCH_RAD_ABS_LIMIT = 0.5233
HEAD_UNIT_LIMIT_MARGIN_RAD = 0.05


def normalize_head_joint_states_rad(head_joint_states: Sequence[float]) -> list[float]:
    """Return head joint states in radians.

    CoRobot FK expects radians, but some RobotDds fallback observations expose
    head joints in degrees. If any head joint is outside the URDF radian range,
    treat the full head pair as degrees so yaw/pitch stay in the same unit.
    """
    values = [float(value) for value in head_joint_states]
    if len(values) < 2:
        return values

    yaw, pitch = values[:2]
    looks_like_degrees = (
        abs(yaw) > HEAD_YAW_RAD_ABS_LIMIT + HEAD_UNIT_LIMIT_MARGIN_RAD
        or abs(pitch) > HEAD_PITCH_RAD_ABS_LIMIT + HEAD_UNIT_LIMIT_MARGIN_RAD
    )

    if not looks_like_degrees:
        return values

    return [math.radians(value) for value in values]
```

这个函数的作用是：保证传给 FK 的头部关节角是弧度。如果 observation 里头部关节看起来像角度制，就统一转成弧度制。这个处理很重要，因为 `_dynamic_t_exec_camera()` 里调用 `compute_head_fk()` 前会先执行：

```python
head = normalize_head_joint_states_rad(head)
```

如果单位不统一，最终算出来的 `T_base_head_pitch(q)` 就会错，进而导致相机坐标转底盘坐标也错。

# 4. 执行阶段：用 base_link 坐标指导机械臂动作

## 4.1 move_eef 接口：外部传入的是相机坐标

位置：`.a2d_pkg/corobot/policy_tasks/rule_control_task.py`

```python
@expose_api(method="POST", path="/skill/move_eef")
def move_eef(
    self,
    arm: str,
    target_position_camera_m: list[float],
    target_orientation_camera_xyzw: list[float] | None = None,
    duration_s: float = 1.0,
    gripper_value: float | None = None,
) -> dict[str, Any]:
    obs = self._observation()
    calibration = self._calibration_for_observation(obs)
    camera_frame = calibration.camera_frame or DEFAULT_CAMERA_FRAME

    action, meta = build_move_eef_action(
        obs,
        calibration,
        arm=arm,
        camera_frame=camera_frame,
        target_position_camera_m=target_position_camera_m,
        target_orientation_camera_xyzw=target_orientation_camera_xyzw,
        duration_s=duration_s,
        gripper_value=gripper_value,
    )

    self._execute(action, meta["actual_duration_s"])
    return {"action": action, "meta": meta}
```

这个接口说明，外部调用 `/skill/move_eef` 时传入的是：

```text
target_position_camera_m
```

也就是目标点在相机坐标系下的位置。进入函数后，代码先拿当前 observation，再调用 `_calibration_for_observation(obs)` 动态生成当前时刻的 `T_exec_camera(q)`，然后交给 `build_move_eef_action()` 构造真正的机械臂动作。

## 4.2 build_move_eef_action：确定当前机械臂起点

位置：`src/mcp_control_demo/control/action_builder.py`

```python
def build_move_eef_action(
    observation: Any,
    calibration: CalibrationConfig,
    *,
    arm: str,
    target_position_camera_m: list[float],
    camera_frame: str | None = None,
    target_orientation_camera_xyzw: list[float] | None = None,
    duration_s: float | None = 1.0,
    gripper_value: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    current_pose = _current_eef_pose(observation, arm, calibration.exec_frame)
    current_position = _pose_position(current_pose)
    current_orientation = _pose_orientation(current_pose)

    if current_position is not None and current_orientation is not None:
        current_center = wrist_to_gripper_center_exec(current_position, current_orientation)
    else:
        target_exec = calibration.camera_to_exec_point(target_position_camera_m, camera_frame)
        current_center = target_exec.copy()

    current_camera = calibration.exec_to_camera_point(current_center, camera_frame)

    return build_move_eef_between_camera_points(
        observation,
        calibration,
        arm=arm,
        start_position_camera_m=current_camera.tolist(),
        target_position_camera_m=target_position_camera_m,
        camera_frame=camera_frame,
        target_orientation_camera_xyzw=target_orientation_camera_xyzw,
        duration_s=duration_s,
        gripper_value=gripper_value,
    )
```

这个函数的作用是：先确定机械臂当前末端位置，再把当前末端位置和目标位置一起交给 `build_move_eef_between_camera_points()`。

这里有一个细节：底层 A2D 控制的是 wrist/link7，但对外接口更希望使用夹爪中心 TCP。因此代码会用：

```python
current_center = wrist_to_gripper_center_exec(current_position, current_orientation)
```

把当前 wrist/link7 坐标换成夹爪中心坐标。

如果当前 observation 里拿不到机械臂末端位姿，就把目标点先转到底盘坐标，临时把当前点设成目标点：

```python
target_exec = calibration.camera_to_exec_point(target_position_camera_m, camera_frame)
current_center = target_exec.copy()
```

## 4.3 build_move_eef_between_camera_points：相机坐标转 base_link 坐标并生成轨迹

位置：`src/mcp_control_demo/control/action_builder.py`

```python
def build_move_eef_between_camera_points(
    observation: Any,
    calibration: CalibrationConfig,
    *,
    arm: str,
    start_position_camera_m: list[float],
    target_position_camera_m: list[float],
    camera_frame: str | None = None,
    target_orientation_camera_xyzw: list[float] | None = None,
    duration_s: float | None = 1.0,
    gripper_value: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    arm = _validate_arm(arm)
    timing = make_timing(duration_s)

    start_exec = calibration.camera_to_exec_point(start_position_camera_m, camera_frame)
    target_exec = calibration.camera_to_exec_point(target_position_camera_m, camera_frame)

    orientation_exec_xyzw = _target_orientation_exec_xyzw(
        observation,
        calibration,
        arm,
        camera_frame,
        target_orientation_camera_xyzw,
    )
    orientation_exec_rpy = _quat_xyzw_to_rpy(orientation_exec_xyzw)

    start_wrist_exec = gripper_center_to_wrist_exec(start_exec, orientation_exec_xyzw)
    target_wrist_exec = gripper_center_to_wrist_exec(target_exec, orientation_exec_xyzw)

    rows = _interpolate_pose_rows(start_wrist_exec, target_wrist_exec, orientation_exec_rpy, timing.num_steps)
    action = _eef_action(calibration.exec_frame, arm, rows, timing)

    if gripper_value is not None:
        _attach_gripper_rows(action, observation, arm, float(gripper_value), timing.num_steps)

    return action, _timing_meta(timing) | {
        "arm": arm,
        "camera_frame": camera_frame or calibration.camera_frame,
        "exec_frame": calibration.exec_frame,
        "start_position_camera_m": _round_list(start_position_camera_m),
        "target_position_camera_m": _round_list(target_position_camera_m),
        "target_position_exec_m": _round_list(target_exec),
        "start_wrist_position_exec_m": _round_list(start_wrist_exec),
        "target_wrist_position_exec_m": _round_list(target_wrist_exec),
        "gripper_center_offset_link7_m": _round_list(GRIPPER_CENTER_OFFSET_LINK7_M),
        "a2d_target_frame": f"arm_{arm}_link7",
    }
```

这个函数是机械臂动作构造里最关键的一段。

它真正完成了：

```python
start_exec = calibration.camera_to_exec_point(start_position_camera_m, camera_frame)
target_exec = calibration.camera_to_exec_point(target_position_camera_m, camera_frame)
```

由于当前 `exec_frame` 是 `base_link`，所以这里的 `start_exec` 和 `target_exec` 就是底盘坐标系下的起点和目标点。

然后代码会把“夹爪中心目标点”转换成底层控制需要的 “wrist/link7 目标点”：

```python
start_wrist_exec = gripper_center_to_wrist_exec(start_exec, orientation_exec_xyzw)
target_wrist_exec = gripper_center_to_wrist_exec(target_exec, orientation_exec_xyzw)
```

最后调用 `_interpolate_pose_rows()` 插值出一段轨迹，再调用 `_eef_action()` 包装成 CoRobot 可执行的动作。

## 4.4 姿态转换：相机坐标系姿态转 base_link 姿态

位置：`src/mcp_control_demo/control/action_builder.py`

```python
def _target_orientation_exec_xyzw(
    observation: Any,
    calibration: CalibrationConfig,
    arm: str,
    camera_frame: str | None,
    target_orientation_camera_xyzw: list[float] | None,
) -> list[float]:
    if target_orientation_camera_xyzw is not None:
        calibration.require_camera_frame(camera_frame)
        return transform_orientation_xyzw(calibration.require_transform(), target_orientation_camera_xyzw)

    current = _current_eef_orientation(observation, arm, calibration.exec_frame)
    return current if current is not None else [0.0, 0.0, 0.0, 1.0]
```

这个函数负责处理目标姿态。

如果外部传入了 `target_orientation_camera_xyzw`，说明目标姿态也是相机坐标系下的四元数。代码会用当前的 `T_exec_camera` 把它转到 `base_link` 下。

如果外部没有传目标姿态，就沿用当前机械臂末端姿态。如果当前姿态也拿不到，就使用单位四元数 `[0, 0, 0, 1]`。

位置：`src/mcp_control_demo/calibration/transforms.py`

```python
def transform_orientation_xyzw(matrix: Sequence[Sequence[float]], quat_xyzw: Sequence[float]) -> list[float]:
    transform = as_matrix4(matrix)
    rotation = transform[:3, :3] @ quat_xyzw_to_matrix(quat_xyzw)
    return matrix_to_quat_xyzw(rotation)
```

这个函数只使用 4×4 变换矩阵里的旋转部分，不使用平移部分，因为姿态转换只需要旋转。

## 4.5 夹爪中心 TCP 和 wrist/link7 的偏移转换

位置：`src/mcp_control_demo/control/action_builder.py`

```python
GRIPPER_CENTER_OFFSET_LINK7_M = np.asarray([0.0, 0.0, 0.14308], dtype=np.float64)
```

这个常量表示：夹爪中心 TCP 相对于 wrist/link7 的固定偏移。对外输入的目标点是夹爪中心，但底层 A2D 实际控制的是 wrist/link7，所以必须扣掉这个偏移。

```python
def wrist_to_gripper_center_exec(
    wrist_position_exec_m: list[float] | np.ndarray,
    wrist_orientation_exec_xyzw: list[float] | np.ndarray,
) -> np.ndarray:
    return np.asarray(wrist_position_exec_m, dtype=np.float64).reshape(3) + _tcp_offset_exec(
        wrist_orientation_exec_xyzw
    )


def gripper_center_to_wrist_exec(
    gripper_center_exec_m: list[float] | np.ndarray,
    wrist_orientation_exec_xyzw: list[float] | np.ndarray,
) -> np.ndarray:
    return np.asarray(gripper_center_exec_m, dtype=np.float64).reshape(3) - _tcp_offset_exec(
        wrist_orientation_exec_xyzw
    )


def _tcp_offset_exec(wrist_orientation_exec_xyzw: list[float] | np.ndarray) -> np.ndarray:
    return R.from_quat(np.asarray(wrist_orientation_exec_xyzw, dtype=np.float64).reshape(4)).apply(
        GRIPPER_CENTER_OFFSET_LINK7_M
    )
```

这几个函数的作用是：在 `base_link` 坐标系下，根据当前 wrist 姿态，把固定 TCP 偏移旋转到当前方向，然后进行加减。

`wrist_to_gripper_center_exec()` 用于把当前 wrist 位置换算成夹爪中心位置。

`gripper_center_to_wrist_exec()` 用于把外部给的夹爪中心目标点换算成底层真正要控制的 wrist/link7 目标点。

## 4.6 生成 EEF_ABS 动作

位置：`src/mcp_control_demo/control/action_builder.py`

```python
def _interpolate_pose_rows(
    start_position: np.ndarray,
    target_position: np.ndarray,
    orientation_rpy: list[float],
    steps: int,
) -> list[list[float]]:
    rows: list[list[float]] = []
    for index in range(1, steps + 1):
        alpha = float(index) / float(steps)
        pos = start_position + (target_position - start_position) * alpha
        rows.append([*map(float, pos.tolist()), *map(float, orientation_rpy)])
    return rows
```

这个函数把起点和终点之间插值成多帧轨迹。每一行的格式是：

```text
[x, y, z, roll, pitch, yaw]
```

这些值已经是在 `base_link` 坐标系下的 wrist/link7 目标位姿。

```python
def _eef_action(exec_frame: str, arm: str, rows: list[list[float]], timing: TrajectoryTiming) -> dict[str, Any]:
    field = f"{arm}_arm"
    return {
        "timestamps": int(time.time() * 1e9),
        "trajectory_reference_time": timing.actual_duration_s,
        "base_link": exec_frame,
        field: {"kind": "EEF_ABS", "values": rows},
    }
```

这个函数把插值轨迹包装成 CoRobot 底层可以执行的动作。因为 `exec_frame` 当前是 `base_link`，所以生成的 action 里：

```python
"base_link": exec_frame
```

实际就是：

```python
"base_link": "base_link"
```

`field` 会根据左右手变成：

```text
left_arm
right_arm
```

`kind` 是：

```text
EEF_ABS
```

表示这是末端执行器在执行坐标系下的绝对位姿控制。

## 4.7 完整流程说明

完整链路可以按下面的顺序理解。

第一步，机器人通过头部相机获得图像：

```text
observation.images["head"]
```

第二步，`AprilTagPerceptionService.detect_from_observation()` 从 observation 中读取头部相机图像和相机内参，然后调用 `detect_tags_from_image()` 进行 AprilTag 检测。

第三步，`detect_tags_from_image()` 使用相机内参和 tag 尺寸估计 tag 位姿，输出：

```text
position_camera_m / translation_m
```

此时坐标仍然是在：

```text
head_camera_optical
```

第四步，执行动作前，`RuleControlTask.move_eef()` 会读取当前 observation，并调用：

```python
calibration = self._calibration_for_observation(obs)
```

第五步，`_calibration_for_observation()` 会继续调用 `_dynamic_t_exec_camera()`，根据当前 `head_joint_states` 和 `waist_joint_states` 计算：

```text
T_base_head_pitch(q)
```

然后和固定外参相乘：

```text
T_exec_camera(q) = T_base_head_pitch(q) * T_head_pitch_camera
```

在当前配置中：

```text
exec_frame = base_link
```

所以也就是：

```text
T_base_camera(q) = T_base_head_pitch(q) * T_head_pitch_camera
```

第六步，`CalibrationConfig.camera_to_exec_point()` 使用这个动态矩阵把相机坐标点转到底盘坐标系：

```python
target_exec = calibration.camera_to_exec_point(target_position_camera_m, camera_frame)
```

等价于：

```text
p_base = T_base_camera(q) @ p_camera
```

第七步，`build_move_eef_between_camera_points()` 把夹爪中心目标点转换成底层实际控制的 wrist/link7 目标点：

```python
target_wrist_exec = gripper_center_to_wrist_exec(target_exec, orientation_exec_xyzw)
```

第八步，`_interpolate_pose_rows()` 把当前 wrist 位姿和目标 wrist 位姿插值成多帧轨迹。

第九步，`_eef_action()` 把轨迹包装成 CoRobot 动作：

```python
{
    "base_link": "base_link",
    "right_arm": {
        "kind": "EEF_ABS",
        "values": rows,
    },
}
```

第十步，`RuleControlTask.move_eef()` 调用：

```python
self._execute(action, meta["actual_duration_s"])
```

最终由底层运动控制器执行机械臂动作。

因此，这段代码的核心逻辑可以概括为：

```text
头部相机检测目标
    -> 得到 head_camera_optical 下的目标位置 p_camera
    -> 读取当前头部/腰部关节状态
    -> FK 得到 T_base_head_pitch(q)
    -> 乘固定相机外参 T_head_pitch_camera
    -> 得到 T_base_camera(q)
    -> p_base = T_base_camera(q) @ p_camera
    -> 将夹爪中心目标点修正为 wrist/link7 目标点
    -> 生成 base_link 下的 EEF_ABS 轨迹
    -> 控制机械臂执行
```
