#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify the meaning of G1 camera extrinsic T by projecting robot frames."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import os
import sys
import time
import traceback
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMERA_CFG = PROJECT_ROOT / "cfg" / "inference" / "g1_head_rgbd.yaml"
DEFAULT_PARAMETER_PY = PROJECT_ROOT / "G1" / "parameter.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quat_xyzw_to_R(q) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def Rz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def T_from_R_t(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def T_origin(xyz, rpy) -> np.ndarray:
    return T_from_R_t(rpy_to_R(float(rpy[0]), float(rpy[1]), float(rpy[2])), xyz)


def T_prismatic_z(q: float) -> np.ndarray:
    return T_from_R_t(np.eye(3), [0.0, 0.0, float(q)])


def T_revolute_z(q: float) -> np.ndarray:
    return T_from_R_t(Rz(float(q)), [0.0, 0.0, 0.0])


def normalize_angle_maybe_degrees(value: float) -> float:
    value = float(value)
    if abs(value) > 2.0 * math.pi:
        return math.radians(value)
    return value


HEAD_YAW_RAD_ABS_LIMIT = 1.5708
HEAD_PITCH_RAD_ABS_LIMIT = 0.5233
HEAD_UNIT_LIMIT_MARGIN_RAD = 0.05


def normalize_head_joint_states_rad(head_states: Any):
    values = coerce_state_list(head_states)
    if len(values) < 2:
        return values
    yaw, pitch = values[:2]
    looks_like_degrees = (
        abs(yaw) > HEAD_YAW_RAD_ABS_LIMIT + HEAD_UNIT_LIMIT_MARGIN_RAD
        or abs(pitch) > HEAD_PITCH_RAD_ABS_LIMIT + HEAD_UNIT_LIMIT_MARGIN_RAD
    )
    if looks_like_degrees:
        return [math.radians(value) for value in values]
    return values


def coerce_state_list(value: Any):
    if isinstance(value, tuple) and len(value) == 2:
        value = value[0]
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return [float(v) for v in value]


def compute_g1_head_fk(head_states, waist_states) -> Dict[str, Any]:
    """Compute base_link -> head_link1/head_link2 using G1 URDF constants.

    SDK observations suggest:
      waist_states = [body_joint2_pitch_rad, body_joint1_height_m]
      head_states may be degrees; convert yaw/pitch as one unit-consistent pair.
    """
    raw_head = coerce_state_list(head_states)
    head = normalize_head_joint_states_rad(raw_head)
    waist = coerce_state_list(waist_states)
    head_yaw = float(head[0])
    head_pitch = float(head[1])
    waist_pitch = normalize_angle_maybe_degrees(waist[0])
    waist_height = float(waist[1])

    T_body1_in_base = T_origin([0.0, 0.0, 0.6485], [0.0, 0.0, 0.0]) @ T_prismatic_z(waist_height)
    T_body2_in_body1 = T_origin([0.131, 0.0, 0.0], [-math.pi / 2.0, 0.0, 0.0]) @ T_revolute_z(waist_pitch)
    T_head1_in_body2 = T_origin([0.0, -0.441, 0.0], [math.pi / 2.0, 0.0, 0.0]) @ T_revolute_z(head_yaw)
    T_head2_in_head1 = T_origin([0.050238, 0.0, 0.060065], [-math.pi / 2.0, 0.0, 0.0]) @ T_revolute_z(head_pitch)

    T_body2_in_base = T_body1_in_base @ T_body2_in_body1
    T_head1_in_base = T_body2_in_base @ T_head1_in_body2
    T_head2_in_base = T_head1_in_base @ T_head2_in_head1
    return {
        "raw_head_states": raw_head,
        "raw_waist_states": waist,
        "used": {
            "head_yaw_rad": head_yaw,
            "head_pitch_rad": head_pitch,
            "waist_pitch_rad": waist_pitch,
            "waist_height_m": waist_height,
        },
        "T_body2_in_base": T_body2_in_base,
        "T_head1_in_base": T_head1_in_base,
        "T_head2_in_base": T_head2_in_base,
    }


def xyzquat_xyzw_to_T(values) -> np.ndarray:
    vals = [float(v) for v in values]
    if len(vals) != 7:
        raise ValueError(f"expected xyzquat length 7, got {len(vals)}")
    return T_from_R_t(quat_xyzw_to_R(vals[3:]), vals[:3])


def compute_corobot_head_fk(head_states, waist_states, urdf_path: str | None = None) -> Dict[str, Any]:
    """Compute base_link -> head_pitch with the G1 SDK/corobot kinematics if present."""
    from contextlib import redirect_stdout
    from io import StringIO

    from corobot.utils.kinematics import Kinematics

    resolved_urdf = urdf_path
    if not resolved_urdf:
        from corobot.utils.fk_solver import _find_urdf_solver_dir

        resolved_urdf = str((_find_urdf_solver_dir() / "A2D_viz.urdf").resolve())

    raw_head = coerce_state_list(head_states)
    head = normalize_head_joint_states_rad(raw_head)
    waist = coerce_state_list(waist_states)
    head_yaw = float(head[0])
    head_pitch = float(head[1])
    waist_pitch = normalize_angle_maybe_degrees(waist[0])
    waist_height = float(waist[1])

    with redirect_stdout(StringIO()):
        kinematics = Kinematics(str(resolved_urdf))
    xyzquat = kinematics.compute_head_fk(head_yaw, head_pitch, waist_pitch, waist_height)
    return {
        "ok": True,
        "urdf_path": str(resolved_urdf),
        "raw_head_states": raw_head,
        "raw_waist_states": waist,
        "used": {
            "head_yaw_rad": head_yaw,
            "head_pitch_rad": head_pitch,
            "waist_pitch_rad": waist_pitch,
            "waist_height_m": waist_height,
        },
        "xyzquat_xyzw": [float(v) for v in xyzquat],
        "T_head_pitch_in_base": xyzquat_xyzw_to_T(xyzquat),
    }


def parse_motion_pose(frame: dict) -> np.ndarray:
    p = frame["position"]
    q = frame["orientation"]["quaternion"]
    R = quat_xyzw_to_R([q["x"], q["y"], q["z"], q["w"]])
    return T_from_R_t(R, [p["x"], p["y"], p["z"]])


def motion_frame_names(status: dict) -> list[str]:
    frames = status.get("frames")
    if isinstance(frames, dict):
        return sorted(str(name) for name in frames.keys())
    if isinstance(frames, list):
        names = []
        for idx, item in enumerate(frames):
            if isinstance(item, dict):
                names.append(str(item.get("name") or item.get("frame") or item.get("frame_name") or idx))
            else:
                names.append(str(idx))
        return names
    return []


def project_point(K: np.ndarray, p_cam: np.ndarray) -> Tuple[float, float, float]:
    x, y, z = [float(v) for v in p_cam[:3]]
    if z <= 1e-6:
        return float("nan"), float("nan"), z
    u = float(K[0, 0]) * x / z + float(K[0, 2])
    v = float(K[1, 1]) * y / z + float(K[1, 2])
    return u, v, z


def wait_latest_image(camera_group: Any, camera_name: str, tries: int = 30, sleep_s: float = 0.1):
    last_img = None
    last_ts = None
    for _ in range(tries):
        img, ts = camera_group.get_latest_image(camera_name)
        last_img, last_ts = img, ts
        if img is not None:
            return img, ts
        time.sleep(sleep_s)
    return last_img, last_ts


def wait_motion_status(controller: Any, tries: int = 30, sleep_s: float = 0.1):
    last_status = None
    attempts = []
    for idx in range(tries):
        try:
            status = controller.get_motion_status()
            last_status = status
            attempts.append({"idx": idx, "is_none": status is None, "keys": list(status.keys()) if isinstance(status, dict) else None})
            if isinstance(status, dict) and status.get("frames"):
                return status, attempts
        except Exception as exc:
            attempts.append({"idx": idx, "error_type": type(exc).__name__, "error": str(exc)})
        time.sleep(sleep_s)
    return last_status, attempts


def draw_cross(img, uv, color, label):
    u, v, z = uv
    if not np.isfinite(u) or not np.isfinite(v):
        return
    x, y = int(round(u)), int(round(v))
    h, w = img.shape[:2]
    if -200 <= x < w + 200 and -200 <= y < h + 200:
        cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, 24, 2)
        cv2.putText(img, f"{label} z={z:.2f}", (max(0, min(w - 1, x + 8)), max(20, min(h - 1, y - 8))), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def upload_zip(zip_path: Path, upload_url: str) -> Dict[str, Any]:
    data = zip_path.read_bytes()
    req = urllib.request.Request(
        upload_url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/zip",
            "Content-Length": str(len(data)),
            "X-G1-Diagnostics-Filename": zip_path.name,
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return {"ok": True, "status": resp.status, "response": resp.read().decode("utf-8", errors="replace")}


def make_zip(src_dir: Path) -> Path:
    zip_path = src_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir.parent))
    return zip_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "g1_transform_verify_runs"))
    parser.add_argument("--tag", default="transform_verify")
    parser.add_argument("--camera-cfg", default=str(DEFAULT_CAMERA_CFG))
    parser.add_argument("--parameter-py", default=str(DEFAULT_PARAMETER_PY))
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument("--upload-url", default="")
    parser.add_argument("--rgb-name", default="head")
    parser.add_argument("--depth-name", default="head_depth")
    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument("--motion-tries", type=int, default=30)
    parser.add_argument("--motion-sleep-s", type=float, default=0.1)
    parser.add_argument("--urdf-path", default="", help="Optional URDF path for corobot Kinematics.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir).expanduser().resolve() / f"g1_T_verify_{stamp}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {"ok": False, "args": vars(args)}
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from a2d_sdk.robot import CosineCamera, RobotController, RobotDds

        param_path = Path(args.parameter_py).expanduser()
        if not param_path.exists() and (PROJECT_ROOT / "scripts" / "parameters.py").exists():
            param_path = PROJECT_ROOT / "scripts" / "parameters.py"
        params_module = load_module(param_path, "g1_parameter_runtime")
        params = params_module.load_all_parameters("head")
        K = np.asarray(params["intrinsics"]["K"], dtype=np.float64).reshape(3, 3)
        T_param = np.asarray(params["extrinsics"]["T"], dtype=np.float64).reshape(4, 4)

        camera_group = CosineCamera([args.rgb_name, args.depth_name])
        time.sleep(args.warmup_s)
        try:
            rgb, rgb_ts = wait_latest_image(camera_group, args.rgb_name)
            depth, depth_ts = wait_latest_image(camera_group, args.depth_name)
        finally:
            if hasattr(camera_group, "close"):
                camera_group.close()

        if rgb is None:
            raise RuntimeError(f"RGB frame is None: {args.rgb_name}")
        rgb_arr = np.asarray(rgb)
        if rgb_arr.ndim != 3 or rgb_arr.shape[2] not in (3, 4):
            raise RuntimeError(f"unexpected RGB shape: {rgb_arr.shape}")
        rgb_bgr = cv2.cvtColor(rgb_arr[:, :, :3], cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(run_dir / "rgb_bgr.png"), rgb_bgr)
        np.save(run_dir / "rgb_bgr.npy", rgb_bgr)
        if depth is not None:
            depth_arr = np.asarray(depth)
            if depth_arr.ndim == 3 and depth_arr.shape[-1] == 1:
                depth_arr = depth_arr[:, :, 0]
            np.save(run_dir / "depth_raw.npy", depth_arr)
        np.save(run_dir / "K.npy", K)
        np.save(run_dir / "T_param.npy", T_param)

        robot = RobotDds()
        head_states = robot.head_joint_states()
        waist_states = robot.waist_joint_states()
        head_fk = compute_g1_head_fk(head_states, waist_states)
        try:
            corobot_head_fk = compute_corobot_head_fk(head_states, waist_states, args.urdf_path or None)
        except Exception as exc:
            corobot_head_fk = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

        controller = RobotController()
        motion_status, motion_attempts = wait_motion_status(controller, args.motion_tries, args.motion_sleep_s)
        report["motion_status_attempts"] = motion_attempts
        if not isinstance(motion_status, dict):
            raise RuntimeError(f"get_motion_status did not return a dict, last={motion_status!r}")
        all_frame_names = motion_frame_names(motion_status)
        camera_related_frame_names = [
            name
            for name in all_frame_names
            if any(token in name.lower() for token in ("head", "camera", "cam", "eye", "pitch"))
        ]
        sides = ["left", "right"] if args.side == "both" else [args.side]
        side_frames: Dict[str, Dict[str, Any]] = {}
        for side in sides:
            frame_name = f"arm_{side}_link7"
            if frame_name not in motion_status.get("frames", {}):
                side_frames[side] = {"ok": False, "frame_name": frame_name, "error": f"{frame_name} not in motion_status"}
                continue
            T_link7_in_base = parse_motion_pose(motion_status["frames"][frame_name])

            # URDF fixed transform for G1 omnipicker center relative to link7.
            # arm_*_link7 in motion_status corresponds to URDF arm_*_end_link.
            ee_yaw = math.pi / 2.0 if side == "left" else -math.pi / 2.0
            T_gripper_base_in_link7 = T_from_R_t(rpy_to_R(0.0, 0.0, ee_yaw), [0.0, 0.0, 0.0])
            T_center_in_gripper_base = T_from_R_t(rpy_to_R(0.0, 0.0, -math.pi / 2.0), [0.0, 0.0, 0.14308])
            T_center_in_link7 = T_gripper_base_in_link7 @ T_center_in_gripper_base
            T_center_in_base = T_link7_in_base @ T_center_in_link7
            side_frames[side] = {
                "ok": True,
                "frame_name": frame_name,
                "T_link7_in_base": T_link7_in_base,
                "T_center_in_link7": T_center_in_link7,
                "T_center_in_base": T_center_in_base,
            }

        candidates = {
            "A_assume_T_param_is_cam_in_base": T_param,
            "B_assume_T_param_is_base_in_cam": np.linalg.inv(T_param),
            "C_head2_assume_T_param_is_cam_in_head2": head_fk["T_head2_in_base"] @ T_param,
            "D_head2_assume_T_param_is_head2_in_cam": head_fk["T_head2_in_base"] @ np.linalg.inv(T_param),
            "E_head1_assume_T_param_is_cam_in_head1": head_fk["T_head1_in_base"] @ T_param,
            "F_head1_assume_T_param_is_head1_in_cam": head_fk["T_head1_in_base"] @ np.linalg.inv(T_param),
            "G_body2_assume_T_param_is_cam_in_body2": head_fk["T_body2_in_base"] @ T_param,
            "H_body2_assume_T_param_is_body2_in_cam": head_fk["T_body2_in_base"] @ np.linalg.inv(T_param),
        }
        if corobot_head_fk.get("ok"):
            T_head_pitch_in_base = corobot_head_fk["T_head_pitch_in_base"]
            candidates.update(
                {
                    "I_corobot_assume_T_param_is_cam_in_head_pitch": T_head_pitch_in_base @ T_param,
                    "J_corobot_assume_T_param_is_head_pitch_in_cam": T_head_pitch_in_base @ np.linalg.inv(T_param),
                }
            )

        results = {}
        overlay = rgb_bgr.copy()
        color_palette = [
            ((0, 0, 255), (0, 128, 255)),
            ((255, 0, 0), (255, 128, 0)),
            ((0, 255, 0), (0, 200, 120)),
            ((255, 0, 255), (180, 0, 180)),
            ((0, 255, 255), (0, 180, 180)),
            ((180, 180, 0), (120, 120, 0)),
            ((255, 255, 255), (160, 160, 160)),
            ((80, 80, 255), (80, 160, 255)),
        ]
        for idx, (cand_name, T_cam_in_base) in enumerate(candidates.items()):
            T_base_in_cam = np.linalg.inv(T_cam_in_base)
            label = cand_name.split("_", 1)[0]
            color_link, color_center = color_palette[idx % len(color_palette)]
            cand_result: Dict[str, Any] = {"T_cam_in_base": T_cam_in_base.tolist(), "sides": {}}
            for side, side_data in side_frames.items():
                if not side_data.get("ok"):
                    cand_result["sides"][side] = side_data
                    continue
                p_link7_cam = (T_base_in_cam @ side_data["T_link7_in_base"] @ np.array([0, 0, 0, 1.0]))[:3]
                p_center_cam = (T_base_in_cam @ side_data["T_center_in_base"] @ np.array([0, 0, 0, 1.0]))[:3]
                uv_link7 = project_point(K, p_link7_cam)
                uv_center = project_point(K, p_center_cam)
                draw_cross(overlay, uv_link7, color_link, f"{label} {side[0]}L7")
                draw_cross(overlay, uv_center, color_center, f"{label} {side[0]}TCP")
                cand_result["sides"][side] = {
                    "frame_name": side_data["frame_name"],
                    "link7_p_cam": p_link7_cam.tolist(),
                    "link7_uvz": list(uv_link7),
                    "center_p_cam": p_center_cam.tolist(),
                    "center_uvz": list(uv_center),
                }
            results[cand_name] = cand_result

        cv2.imwrite(str(run_dir / "projection_overlay.png"), overlay)
        report.update(
            {
                "ok": True,
                "parameter_source": {
                    "intrinsics": params["intrinsics"].get("source"),
                    "extrinsics": params["extrinsics"].get("source"),
                },
                "camera": {
                    "rgb_name": args.rgb_name,
                    "rgb_ts": int(rgb_ts) if rgb_ts is not None else None,
                    "rgb_shape": list(rgb_arr.shape),
                    "depth_name": args.depth_name,
                    "depth_ts": int(depth_ts) if depth_ts is not None else None,
                    "depth_shape": list(np.asarray(depth).shape) if depth is not None else None,
                },
                "K": K.tolist(),
                "T_param": T_param.tolist(),
                "head_waist_fk": {
                    "raw_head_states": head_fk["raw_head_states"],
                    "raw_waist_states": head_fk["raw_waist_states"],
                    "used": head_fk["used"],
                    "T_body2_in_base": head_fk["T_body2_in_base"].tolist(),
                    "T_head1_in_base": head_fk["T_head1_in_base"].tolist(),
                    "T_head2_in_base": head_fk["T_head2_in_base"].tolist(),
                },
                "corobot_head_fk": {
                    **{k: v for k, v in corobot_head_fk.items() if k != "T_head_pitch_in_base"},
                    "T_head_pitch_in_base": corobot_head_fk["T_head_pitch_in_base"].tolist() if corobot_head_fk.get("ok") else None,
                },
                "motion_status_frame_names": all_frame_names,
                "motion_status_camera_related_frame_names": camera_related_frame_names,
                "motion_status_frames": {
                    side: {
                        "ok": data.get("ok"),
                        "frame_name": data.get("frame_name"),
                        "T_link7_in_base": data["T_link7_in_base"].tolist() if data.get("ok") else None,
                        "T_center_in_link7_urdf": data["T_center_in_link7"].tolist() if data.get("ok") else None,
                        "T_center_in_base": data["T_center_in_base"].tolist() if data.get("ok") else None,
                    }
                    for side, data in side_frames.items()
                },
                "projection_results": results,
                "note": "If projected points are behind camera or far outside image, that candidate T direction is wrong or T_param is not full camera-in-base.",
            }
        )
    except Exception as exc:
        report.update({"ok": False, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})

    (run_dir / "transform_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = make_zip(run_dir)
    upload = None
    if args.upload_url:
        try:
            upload = upload_zip(zip_path, args.upload_url)
        except Exception as exc:
            upload = {"ok": False, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        (run_dir / "upload_result.json").write_text(json.dumps(upload, ensure_ascii=False, indent=2), encoding="utf-8")
        zip_path = make_zip(run_dir)

    print(json.dumps({"run_dir": str(run_dir), "zip_path": str(zip_path), "upload": upload}, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
