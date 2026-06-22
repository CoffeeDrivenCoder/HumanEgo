#!/usr/bin/env python3
"""Shared helpers for Agibot G1 head-camera hand-eye calibration."""

from __future__ import annotations

import ast
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DOC_EXAMPLE_T_HEAD_PITCH_CAMERA = np.asarray(
    [
        [0.01154905419417851, 0.03633581308553096, -0.9992728996798792, -0.09309730346114839],
        [-0.010873071465479051, -0.999275902324378, -0.03646158733116647, 0.03977041211949885],
        [-0.9998741899179753, 0.01128626249982923, -0.011145609658446354, -0.01592936593693035],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

HEAD_YAW_RAD_ABS_LIMIT = 1.5708
HEAD_PITCH_RAD_ABS_LIMIT = 0.5233
HEAD_UNIT_LIMIT_MARGIN_RAD = 0.05


def now_ns() -> int:
    return int(time.time() * 1e9)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def append_jsonl(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write("\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def as_matrix4(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"expected 4x4 matrix, got shape {matrix.shape}")
    return matrix


def normalize_head_joint_states_rad(head_joint_states: Iterable[float]) -> list[float]:
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


def quat_xyzw_to_matrix(quat_xyzw: Iterable[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat_xyzw]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise ValueError("zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def xyzquat_xyzw_to_matrix(xyzquat: Iterable[float]) -> np.ndarray:
    vals = [float(v) for v in xyzquat]
    if len(vals) != 7:
        raise ValueError(f"expected xyzquat length 7, got {len(vals)}")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quat_xyzw_to_matrix(vals[3:])
    matrix[:3, 3] = vals[:3]
    return matrix


def make_transform(rotation_matrix: Any, translation: Any) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return matrix


def transform_point(matrix: Any, point: Any) -> np.ndarray:
    matrix = as_matrix4(matrix)
    homogeneous = np.ones(4, dtype=np.float64)
    homogeneous[:3] = np.asarray(point, dtype=np.float64).reshape(3)
    return (matrix @ homogeneous)[:3]


def matrix_to_list(matrix: Any, digits: int = 12) -> list[list[float]]:
    matrix = as_matrix4(matrix)
    return [[round(float(v), digits) for v in row] for row in matrix.tolist()]


def camera_params_from_args(args: Any) -> dict[str, Any]:
    if args.intrinsics_json:
        data = load_json(args.intrinsics_json)
        source = data.get("intrinsics", data)
        return {
            "camera_name": source.get("camera_name", args.camera_name),
            "camera_model": source.get("camera_model", ""),
            "image_size": source.get("image_size", {}),
            "fx": float(source["fx"]),
            "fy": float(source["fy"]),
            "cx": float(source["cx"]),
            "cy": float(source["cy"]),
        }
    missing = [name for name in ("fx", "fy", "cx", "cy") if getattr(args, name) is None]
    if missing:
        raise ValueError(f"missing intrinsics fields: {missing}; pass --intrinsics-json or --fx/--fy/--cx/--cy")
    return {
        "camera_name": args.camera_name,
        "camera_model": args.camera_model,
        "image_size": {"width": args.image_width, "height": args.image_height},
        "fx": float(args.fx),
        "fy": float(args.fy),
        "cx": float(args.cx),
        "cy": float(args.cy),
    }


def detect_apriltag(
    image: Any,
    *,
    camera_params: dict[str, Any],
    tag_size_m: float,
    tag_family: str,
    tag_id: int | None = None,
) -> dict[str, Any] | None:
    import cv2
    from pupil_apriltags import Detector

    img = np.asarray(image)
    if img.ndim == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
    elif img.ndim == 2:
        gray = img
    else:
        raise ValueError(f"unsupported image shape for AprilTag detection: {img.shape}")

    fx = float(camera_params["fx"])
    fy = float(camera_params["fy"])
    cx = float(camera_params["cx"])
    cy = float(camera_params["cy"])

    detector = Detector(
        families=str(tag_family),
        nthreads=2,
        quad_decimate=1.5,
        quad_sigma=0.8,
        refine_edges=True,
        decode_sharpening=0.25,
    )
    detections = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(fx, fy, cx, cy),
        tag_size=float(tag_size_m),
    )
    usable = []
    for det in detections:
        if tag_id is not None and int(det.tag_id) != int(tag_id):
            continue
        if det.pose_t is None or det.pose_R is None:
            continue
        usable.append(det)
    if not usable:
        return None

    usable.sort(key=lambda det: float(getattr(det, "decision_margin", 0.0)), reverse=True)
    det = usable[0]
    position = np.asarray(det.pose_t, dtype=np.float64).reshape(3)
    rotation = np.asarray(det.pose_R, dtype=np.float64).reshape(3, 3)
    transform = make_transform(rotation, position)
    return {
        "tag_id": int(det.tag_id),
        "tag_family": det.tag_family.decode() if hasattr(det.tag_family, "decode") else str(det.tag_family),
        "tag_size_m": float(tag_size_m),
        "position_camera_m": [float(v) for v in position.tolist()],
        "translation_m": [float(v) for v in position.tolist()],
        "rotation_matrix": rotation.tolist(),
        "T_camera_tag": transform.tolist(),
        "center_px": [float(v) for v in np.asarray(det.center).reshape(2).tolist()],
        "corners_px": np.asarray(det.corners, dtype=np.float64).reshape(4, 2).tolist(),
        "decision_margin": float(getattr(det, "decision_margin", 0.0)),
        "camera_params": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        },
    }


def save_rgb_image(path: str | Path, image: Any) -> None:
    from PIL import Image

    path = Path(path)
    ensure_dir(path.parent)
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def poll_numeric_state(getter: Any, length: int, name: str, timeout_s: float = 2.0, interval_s: float = 0.05) -> list[float]:
    deadline = time.time() + timeout_s
    last_vals = None
    while time.time() < deadline:
        vals, _ = getter()
        last_vals = vals
        try:
            values = [float(v) for v in vals]
        except Exception:
            time.sleep(interval_s)
            continue
        if len(values) == length and all(np.isfinite(values)):
            return values
        time.sleep(interval_s)
    raise RuntimeError(f"{name} not ready within {timeout_s}s; last value={last_vals!r}")


def load_transform_file(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = load_json(path)
        if isinstance(data, dict):
            for key in ("T_head_pitch_camera", "t_head_pitch_camera", "matrix"):
                if key in data:
                    return as_matrix4(data[key])
        return as_matrix4(data)

    rows: list[list[float]] = []
    in_matrix = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("T_head_pitch_camera:"):
            in_matrix = True
            continue
        if in_matrix:
            if not line:
                continue
            if not line.startswith("-"):
                if rows:
                    break
                continue
            _, value = line.split("-", 1)
            rows.append([float(v) for v in ast.literal_eval(value.strip())])
            if len(rows) == 4:
                break
    if len(rows) != 4:
        raise ValueError(f"could not find T_head_pitch_camera in {path}")
    return as_matrix4(rows)


def load_kinematics(urdf_path: str | None):
    from contextlib import redirect_stdout
    from io import StringIO

    from corobot.utils.kinematics import Kinematics

    resolved_urdf = urdf_path
    if resolved_urdf is None:
        try:
            from corobot.utils.fk_solver import _find_urdf_solver_dir

            resolved_urdf = str((_find_urdf_solver_dir() / "A2D_viz.urdf").resolve())
        except Exception as exc:
            raise RuntimeError(
                "Could not auto-locate A2D_viz.urdf. Pass --urdf-path explicitly."
            ) from exc

    with redirect_stdout(StringIO()):
        kinematics = Kinematics(str(resolved_urdf))
    return kinematics, str(resolved_urdf)


def compute_t_base_head_pitch(kinematics: Any, head_joint_states: Iterable[float], waist_joint_states: Iterable[float]) -> np.ndarray:
    head = normalize_head_joint_states_rad(head_joint_states)
    waist = [float(v) for v in waist_joint_states]
    if len(head) < 2 or len(waist) < 2:
        raise ValueError(f"expected head length 2 and waist length 2, got head={head}, waist={waist}")
    xyzquat = kinematics.compute_head_fk(float(head[0]), float(head[1]), float(waist[0]), float(waist[1]))
    return xyzquat_xyzw_to_matrix(xyzquat)


def sample_t_camera_tag(sample: dict[str, Any]) -> np.ndarray:
    tag = sample.get("tag") or {}
    if tag.get("T_camera_tag") is not None:
        return as_matrix4(tag["T_camera_tag"])
    if tag.get("rotation_matrix") is not None and tag.get("position_camera_m") is not None:
        return make_transform(tag["rotation_matrix"], tag["position_camera_m"])
    raise ValueError(f"sample {sample.get('sample_id')} does not contain a tag pose")


def write_calibration_outputs(
    *,
    output_yaml: str | Path,
    output_json: str | Path,
    t_head_pitch_camera: np.ndarray,
    camera_params: dict[str, Any] | None,
    tag_size_m: float | None,
    stats: dict[str, Any],
    urdf_path: str | None,
) -> None:
    output_yaml = Path(output_yaml)
    output_json = Path(output_json)
    ensure_dir(output_yaml.parent)
    ensure_dir(output_json.parent)

    data = {
        "T_head_pitch_camera": matrix_to_list(t_head_pitch_camera),
        "camera_params": camera_params or {},
        "tag_size_m": tag_size_m,
        "stats": stats,
        "urdf_path": urdf_path,
    }
    save_json(output_json, data)

    intr = camera_params or {}
    image_size = intr.get("image_size") or {}
    rows = matrix_to_list(t_head_pitch_camera)
    with output_yaml.open("w", encoding="utf-8") as f:
        f.write("# Generated Agibot G1 head-camera hand-eye calibration.\n")
        f.write("mcp_control:\n")
        f.write("  transform_mode: dynamic_fk\n")
        f.write("  camera_frame: head_camera_optical\n")
        f.write("  exec_frame: base_link\n\n")
        f.write("  intrinsics:\n")
        f.write(f"    camera_name: {intr.get('camera_name', 'head')}\n")
        if intr.get("camera_model"):
            f.write(f"    camera_model: {intr.get('camera_model')}\n")
        if image_size:
            f.write("    image_size:\n")
            f.write(f"      width: {image_size.get('width')}\n")
            f.write(f"      height: {image_size.get('height')}\n")
        for key in ("fx", "fy", "cx", "cy"):
            if key in intr:
                f.write(f"    {key}: {float(intr[key])}\n")
        f.write("\n")
        f.write("  extrinsics:\n")
        f.write("    parent_frame: head_pitch_link\n")
        f.write("    child_frame: head_camera_optical\n")
        f.write("    T_head_pitch_camera:\n")
        for row in rows:
            f.write(f"      - {row}\n")
        f.write("\n")
        f.write("  camera_approach_axis: [0.0, 0.0, -1.0]\n")
        f.write("\n")
        f.write("validation:\n")
        for key, value in stats.items():
            f.write(f"  {key}: {value}\n")

