#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive one-step HumanEgo control client for G1.

Each iteration:
  1. read current G1 RGB-D and link7/TCP state
  2. ask the HumanEgo server for one target
  3. print the proposed target
  4. wait for operator input
  5. execute exactly one set_end_effector_pose_control command

This script is intentionally step-by-step. It never loops into autonomous
continuous control.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = PROJECT_ROOT / "cfg" / "inference" / "g1_serve_bread_right.yaml"

for path in (PROJECT_ROOT, PROJECT_ROOT / "inference", PROJECT_ROOT / "scripts"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from g1_humanego_client_dry_run import (  # noqa: E402
    encode_jpeg_b64,
    encode_depth_npz_b64,
    json_safe,
    log,
    post_json,
    resize_depth_to_shape,
    resize_image_and_K,
    resolve_project_path,
    upload_zip,
)
from g1_humanego_dry_run import pose_dict_from_T  # noqa: E402


EPS = 1e-12


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def load_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_zip(src_dir: Path) -> Path:
    zip_path = src_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir.parent))
    return zip_path


def compact_pose(pose: dict[str, Any]) -> str:
    return (
        f"x={pose['x']:+.4f}, y={pose['y']:+.4f}, z={pose['z']:+.4f}, "
        f"qx={pose['qx']:+.4f}, qy={pose['qy']:+.4f}, "
        f"qz={pose['qz']:+.4f}, qw={pose['qw']:+.4f}"
    )


def build_payload(frame: Any, state: dict[str, Any], args: argparse.Namespace, request_id: str) -> dict[str, Any]:
    rgb_send, K_send, image_send_info = resize_image_and_K(
        frame.rgb,
        frame.K,
        args.send_width,
        args.send_height,
    )
    jpeg_b64 = encode_jpeg_b64(rgb_send, args.jpeg_quality)
    log(f"request {request_id}: sending RGB {image_send_info['sent_shape']} jpeg_b64_bytes={len(jpeg_b64)}")
    depth_send_info = {"sent": False}
    depth_b64 = None
    if args.send_depth:
        depth_send = resize_depth_to_shape(frame.depth_m, rgb_send.shape[:2])
        depth_b64, depth_send_info = encode_depth_npz_b64(depth_send, args.depth_encoding)
        depth_send_info["sent"] = True
        depth_send_info["base64_bytes"] = len(depth_b64)
        log(
            f"request {request_id}: sending depth {depth_send_info['shape']} "
            f"encoding={args.depth_encoding} base64_bytes={len(depth_b64)}"
        )
    payload = {
        "request_id": request_id,
        "client_time_utc": datetime.now(timezone.utc).isoformat(),
        "preview_steps": 1,
        "K": np.asarray(K_send, dtype=np.float64).tolist(),
        "rgb_jpeg_b64": jpeg_b64,
        "frame_summary": {
            "rgb_shape": list(rgb_send.shape),
            "source_rgb_shape": list(frame.rgb.shape),
            "image_send": image_send_info,
            "depth_shape": list(frame.depth_m.shape),
            "depth_send": depth_send_info,
            "depth_valid_ratio": float(np.isfinite(frame.depth_m).mean()),
        },
        "current": {
            "T_head_pitch_camera": np.asarray(state["T_head_pitch_camera"], dtype=np.float64).tolist(),
            "T_base_camera": np.asarray(state["T_base_camera"], dtype=np.float64).tolist(),
            "T_base_in_cam": np.asarray(state["T_base_in_cam"], dtype=np.float64).tolist(),
            "T_link7_in_base": np.asarray(state["T_link7_in_base"], dtype=np.float64).tolist(),
            "T_tcp_in_link7": np.asarray(state["T_tcp_in_link7"], dtype=np.float64).tolist(),
            "T_tcp_in_base": np.asarray(state["T_tcp_in_base"], dtype=np.float64).tolist(),
            "T_tcp_in_cam": np.asarray(state["T_tcp_in_cam"], dtype=np.float64).tolist(),
            "gripper": float(state["gripper"]),
            "gripper_state": json_safe(state["gripper_state"]),
            "corobot_fk": json_safe(state["corobot_fk"]),
        },
    }
    if depth_b64 is not None:
        payload["depth_m_npz_b64"] = depth_b64
        payload["depth_encoding"] = depth_send_info
    return payload


def T_from_pose_dict(pose: dict[str, float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [pose["x"], pose["y"], pose["z"]]
    x, y, z, w = [float(pose[k]) for k in ("qx", "qy", "qz", "qw")]
    n = max(float(np.linalg.norm([x, y, z, w])), 1e-12)
    x, y, z, w = x / n, y / n, z / n, w / n
    T[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return T


def select_target(
    step_preview: dict[str, Any],
    source: str,
    side: str,
    before_T_link7: np.ndarray,
) -> dict[str, float]:
    prefix = f"{side}_pose_flat"
    if source == "raw":
        return {k: float(v) for k, v in step_preview[f"{prefix}_raw"].items()}
    if source == "limited":
        return {k: float(v) for k, v in step_preview[f"{prefix}_limited"].items()}
    if source == "position_keep_orientation":
        T = T_from_pose_dict({k: float(v) for k, v in step_preview[f"{prefix}_limited"].items()})
        T[:3, :3] = np.asarray(before_T_link7, dtype=np.float64)[:3, :3]
        return pose_dict_from_T(T)
    raise ValueError(f"unknown target source: {source}")


def position_from_pose_dict(pose: dict[str, float]) -> np.ndarray:
    return np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float64)


def rotation_angle_deg(R_delta: np.ndarray) -> float:
    R_delta = np.asarray(R_delta, dtype=np.float64).reshape(3, 3)
    value = (float(np.trace(R_delta)) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def R_axis(axis: str, angle_deg: float) -> np.ndarray:
    axis = axis.lower()
    angle = np.radians(float(angle_deg))
    c, s = float(np.cos(angle)), float(np.sin(angle))
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    raise ValueError(f"unknown rotation axis {axis!r}; expected x/y/z")


def rotation_vector_from_delta(R_delta: np.ndarray) -> np.ndarray:
    R_delta = np.asarray(R_delta, dtype=np.float64).reshape(3, 3)
    angle = np.radians(rotation_angle_deg(R_delta))
    if abs(angle) <= EPS:
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [
            R_delta[2, 1] - R_delta[1, 2],
            R_delta[0, 2] - R_delta[2, 0],
            R_delta[1, 0] - R_delta[0, 1],
        ],
        dtype=np.float64,
    )
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= EPS:
        return np.zeros(3, dtype=np.float64)
    return axis / axis_norm * np.degrees(angle)


def limited_rotation(R_start: np.ndarray, R_target: np.ndarray, max_angle_deg: float) -> tuple[np.ndarray, dict[str, Any]]:
    R_start = np.asarray(R_start, dtype=np.float64).reshape(3, 3)
    R_target = np.asarray(R_target, dtype=np.float64).reshape(3, 3)
    R_delta = R_target @ R_start.T
    angle_deg = rotation_angle_deg(R_delta)
    max_angle_deg = abs(float(max_angle_deg))
    if max_angle_deg <= 0.0 or angle_deg <= max_angle_deg:
        return R_target, {
            "raw_angle_deg": angle_deg,
            "max_angle_deg": max_angle_deg,
            "clipped": False,
            "applied_angle_deg": angle_deg,
        }

    angle_rad = np.radians(angle_deg)
    axis = np.array(
        [
            R_delta[2, 1] - R_delta[1, 2],
            R_delta[0, 2] - R_delta[2, 0],
            R_delta[1, 0] - R_delta[0, 1],
        ],
        dtype=np.float64,
    )
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= EPS or abs(np.sin(angle_rad)) <= EPS:
        return R_start, {
            "raw_angle_deg": angle_deg,
            "max_angle_deg": max_angle_deg,
            "clipped": True,
            "applied_angle_deg": 0.0,
            "degenerate_axis": True,
        }
    axis /= axis_norm
    step_rad = np.radians(max_angle_deg)
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    R_step = np.eye(3, dtype=np.float64) + np.sin(step_rad) * K + (1.0 - np.cos(step_rad)) * (K @ K)
    return R_step @ R_start, {
        "raw_angle_deg": angle_deg,
        "max_angle_deg": max_angle_deg,
        "clipped": True,
        "applied_angle_deg": max_angle_deg,
        "axis": axis.tolist(),
    }


def object_position_in_base(response: dict[str, Any], current_state: dict[str, Any], object_key: str) -> np.ndarray | None:
    objects = response.get("input_summary", {}).get("objects", {})
    obj = objects.get(object_key)
    if not obj:
        return None
    T_obj_cam = np.asarray(obj["T_in_cam"], dtype=np.float64)
    T_base_camera = np.asarray(current_state["T_base_camera"], dtype=np.float64)
    return (T_base_camera @ T_obj_cam)[:3, 3]


def adapt_target_pose(
    target_pose: dict[str, float],
    before_T_link7: np.ndarray,
    mode: str,
    axis_step_m: float,
    max_orientation_deg: float,
    probe_axis: str,
    probe_deg: float,
    probe_frame: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    adapted = dict(target_pose)
    current_pose = pose_dict_from_T(before_T_link7)
    if mode == "full":
        return adapted, {"mode": mode}

    raw_delta = position_from_pose_dict(target_pose) - before_T_link7[:3, 3]
    if mode == "position_only":
        for key in ("qx", "qy", "qz", "qw"):
            adapted[key] = current_pose[key]
        return adapted, {"mode": mode, "raw_delta_m": raw_delta.tolist()}

    if mode == "axis_only":
        axis = int(np.argmax(np.abs(raw_delta)))
        step = float(np.clip(raw_delta[axis], -abs(axis_step_m), abs(axis_step_m)))
        target_xyz = before_T_link7[:3, 3].copy()
        target_xyz[axis] += step
        adapted.update({"x": float(target_xyz[0]), "y": float(target_xyz[1]), "z": float(target_xyz[2])})
        for key in ("qx", "qy", "qz", "qw"):
            adapted[key] = current_pose[key]
        return adapted, {
            "mode": mode,
            "raw_delta_m": raw_delta.tolist(),
            "selected_axis": ["x", "y", "z"][axis],
            "axis_step_m": step,
            "axis_step_limit_m": float(abs(axis_step_m)),
        }

    if mode in {"orientation_only", "position_orientation_limited"}:
        T_target = T_from_pose_dict(target_pose)
        T_adapted = T_target.copy()
        R_limited, rotation_info = limited_rotation(
            before_T_link7[:3, :3],
            T_target[:3, :3],
            max_orientation_deg,
        )
        T_adapted[:3, :3] = R_limited
        if mode == "orientation_only":
            T_adapted[:3, 3] = before_T_link7[:3, 3]
        adapted = pose_dict_from_T(T_adapted)
        return adapted, {
            "mode": mode,
            "raw_delta_m": raw_delta.tolist(),
            "orientation_limit": rotation_info,
        }

    if mode == "orientation_probe":
        T_adapted = np.asarray(before_T_link7, dtype=np.float64).copy()
        R_probe = R_axis(probe_axis, probe_deg)
        if probe_frame == "local":
            T_adapted[:3, :3] = before_T_link7[:3, :3] @ R_probe
        elif probe_frame == "base":
            T_adapted[:3, :3] = R_probe @ before_T_link7[:3, :3]
        else:
            raise ValueError(f"unknown probe frame {probe_frame!r}; expected local/base")
        adapted = pose_dict_from_T(T_adapted)
        R_delta_base = T_adapted[:3, :3] @ before_T_link7[:3, :3].T
        return adapted, {
            "mode": mode,
            "raw_delta_m": raw_delta.tolist(),
            "probe_axis": probe_axis,
            "probe_deg": float(probe_deg),
            "probe_frame": probe_frame,
            "target_rotation_vector_base_deg": rotation_vector_from_delta(R_delta_base).tolist(),
        }

    raise ValueError(f"unknown target adapter: {mode}")


def call_ee_control_once(controller: Any, pose: dict[str, float], side: str, lifetime: float) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "lifetime": float(lifetime),
        "control_group": [f"{side}_arm"],
    }
    if side == "right":
        kwargs["right_pose"] = pose
    else:
        kwargs["left_pose"] = pose

    started = time.time()
    try:
        result = controller.set_end_effector_pose_control(**kwargs)
        return {
            "ok": True,
            "duration_s": time.time() - started,
            "kwargs": json_safe(kwargs),
            "result": json_safe(result),
        }
    except Exception as exc:
        return {
            "ok": False,
            "duration_s": time.time() - started,
            "kwargs": json_safe(kwargs),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def call_ee_control(
    controller: Any,
    pose: dict[str, float],
    side: str,
    lifetime: float,
    send_hz: float,
    execute_s: float,
) -> dict[str, Any]:
    send_hz = max(0.0, float(send_hz))
    execute_s = max(0.0, float(execute_s))
    if send_hz <= 0.0 or execute_s <= 0.0:
        result = call_ee_control_once(controller, pose, side, lifetime)
        result["send_mode"] = "single"
        result["num_sends"] = 1
        return result

    interval_s = 1.0 / send_hz
    deadline = time.time() + execute_s
    attempts = []
    idx = 0
    while True:
        attempt = call_ee_control_once(controller, pose, side, lifetime)
        attempt["idx"] = idx
        attempts.append(attempt)
        idx += 1
        if not attempt.get("ok"):
            break
        remaining = deadline - time.time()
        if remaining <= 0.0:
            break
        time.sleep(min(interval_s, remaining))

    return {
        "ok": bool(attempts and all(item.get("ok") for item in attempts)),
        "send_mode": "repeat",
        "num_sends": len(attempts),
        "send_hz": send_hz,
        "execute_s": execute_s,
        "command_lifetime_s": float(lifetime),
        "duration_s": sum(float(item.get("duration_s", 0.0)) for item in attempts),
        "attempts": attempts,
    }


def prompt_operator() -> str:
    try:
        text = input("[Enter]=execute one step, s=skip/replan, q=quit > ")
    except EOFError:
        return "q"
    return text.strip().lower()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default=str(DEFAULT_CFG))
    parser.add_argument("--server-url", default="http://111.0.22.33:30003/infer")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "g1_humanego_interactive_runs"))
    parser.add_argument("--tag", default="interactive_step")
    parser.add_argument("--side", choices=["right", "left"], default="right")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--target-source", choices=["position_keep_orientation", "limited", "raw"], default="position_keep_orientation")
    parser.add_argument("--approach-object-key", default="obj1")
    parser.add_argument(
        "--target-adapter",
        choices=[
            "full",
            "position_only",
            "axis_only",
            "orientation_only",
            "position_orientation_limited",
            "orientation_probe",
        ],
        default="full",
    )
    parser.add_argument("--axis-step-m", type=float, default=0.01)
    parser.add_argument("--max-orientation-deg", type=float, default=10.0)
    parser.add_argument("--probe-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--probe-deg", type=float, default=10.0)
    parser.add_argument("--probe-frame", choices=["local", "base"], default="local")
    parser.add_argument("--confirm-control", default="")
    parser.add_argument("--lifetime", type=float, default=0.5)
    parser.add_argument("--send-hz", type=float, default=10.0)
    parser.add_argument("--execute-s", type=float, default=1.0)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--send-width", type=int, default=320)
    parser.add_argument("--send-height", type=int, default=240)
    parser.add_argument("--send-depth", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--depth-encoding", choices=["z16", "float16", "float32"], default="z16")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--upload-url", default="")
    parser.add_argument("--upload-timeout-s", type=float, default=20.0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    cfg_path = resolve_project_path(args.cfg)
    cfg = load_cfg(cfg_path)
    run_dir = Path(args.out_dir).expanduser().resolve() / f"g1_humanego_interactive_{utc_stamp()}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "ok": False,
        "control_sent": False,
        "args": vars(args),
        "cfg_path": str(cfg_path),
        "steps": [],
    }

    cam = None
    arm = None
    try:
        from G1Camera import G1HeadRGBDCamera
        from G1RobotArm import G1RobotArmReadOnly

        log("initializing G1 camera")
        cam = G1HeadRGBDCamera(resolve_project_path(cfg["camera"]["cfg_path"]))
        log("initializing read-only G1 arm state adapter")
        arm = G1RobotArmReadOnly(side=args.side)

        if args.confirm_control != "RUN_CONTROL":
            log("WARNING: --confirm-control RUN_CONTROL not set; this script will preview but refuse to execute.")

        for idx in range(max(1, int(args.max_steps))):
            step_dir = run_dir / f"step_{idx:03d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            request_id = f"{utc_stamp()}_{args.tag}_{idx:03d}"

            log(f"step {idx}: reading RGB-D frame")
            frame = cam.get_frame()
            log(f"step {idx}: reading current robot state")
            state = arm.get_debug_state()
            before_T_link7 = np.asarray(state["T_link7_in_base"], dtype=np.float64)
            payload = build_payload(frame, state, args, request_id)

            cv2.imwrite(str(step_dir / "rgb_sent_bgr.jpg"), frame.rgb if args.send_width <= 0 else cv2.resize(frame.rgb, (args.send_width, args.send_height)))
            request_summary = dict(payload)
            request_summary.pop("rgb_jpeg_b64", None)
            (step_dir / "request_summary.json").write_text(
                json.dumps(json_safe(request_summary), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            started = time.time()
            server_result = post_json(args.server_url, payload, args.timeout_s)
            response = server_result["json"]
            log(
                f"step {idx}: server response status={server_result['status']} "
                f"bytes={server_result['num_bytes']} duration={time.time() - started:.3f}s"
            )
            (step_dir / "server_response.json").write_text(
                json.dumps(json_safe(response), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            side_previews = response.get("policy_preview", {}).get("sides", {}).get(args.side) or []
            if not side_previews:
                raise RuntimeError(f"server response missing policy_preview.sides.{args.side}[0]")
            step_preview = side_previews[0]
            raw_target_pose = select_target(step_preview, args.target_source, args.side, before_T_link7)
            target_pose, adapter_info = adapt_target_pose(
                raw_target_pose,
                before_T_link7,
                args.target_adapter,
                args.axis_step_m,
                args.max_orientation_deg,
                args.probe_axis,
                args.probe_deg,
                args.probe_frame,
            )
            target_delta = position_from_pose_dict(target_pose) - before_T_link7[:3, 3]
            target_delta_norm = float(np.linalg.norm(target_delta))
            target_T_link7 = T_from_pose_dict(target_pose)
            target_rotation_delta_deg = rotation_angle_deg(target_T_link7[:3, :3] @ before_T_link7[:3, :3].T)
            object_base = object_position_in_base(response, state, args.approach_object_key)
            approach_metrics = None
            if object_base is not None:
                before_dist = float(np.linalg.norm(before_T_link7[:3, 3] - object_base))
                target_dist = float(np.linalg.norm(position_from_pose_dict(target_pose) - object_base))
                approach_metrics = {
                    "object_key": args.approach_object_key,
                    "object_position_in_base": object_base.tolist(),
                    "before_link7_to_object_m": before_dist,
                    "target_link7_to_object_m": target_dist,
                    "target_minus_before_m": target_dist - before_dist,
                    "closer": bool(target_dist < before_dist),
                }

            print("\n=== HumanEgo proposed step ===")
            print(f"step: {idx}")
            print(f"done_prob: {response['policy_preview']['done_prob']:.3f}")
            print(f"object_source: {response.get('input_summary', {}).get('object_source_used')}")
            object_error = response.get("input_summary", {}).get("object_error")
            if object_error:
                print(f"object_error: {object_error.get('error_type')}: {object_error.get('error')}")
            print(f"target_source: {args.target_source}")
            print(f"target_adapter: {args.target_adapter}")
            print(f"target_delta_m: {target_delta.tolist()}  norm={target_delta_norm:.4f}")
            print(f"target_rotation_delta_deg: {target_rotation_delta_deg:.2f}")
            if "orientation_limit" in adapter_info:
                orientation_limit = adapter_info["orientation_limit"]
                print(
                    "orientation_limit: "
                    f"raw={orientation_limit['raw_angle_deg']:.2f}deg, "
                    f"applied={orientation_limit['applied_angle_deg']:.2f}deg, "
                    f"clipped={orientation_limit['clipped']}"
                )
            if adapter_info.get("mode") == "orientation_probe":
                print(
                    "orientation_probe: "
                    f"axis={adapter_info['probe_axis']}, "
                    f"deg={adapter_info['probe_deg']:+.2f}, "
                    f"frame={adapter_info['probe_frame']}, "
                    f"target_rotvec_base_deg={adapter_info['target_rotation_vector_base_deg']}"
                )
            print(f"server raw_delta_norm_m: {step_preview['safety_translation_limit']['raw_delta_norm_m']:.4f}")
            print(f"server clipped: {step_preview['safety_translation_limit']['clipped']}")
            print(f"gripper target raw: {step_preview['gripper_g1_raw_0_open_120_closed']:.2f} / 120 (not executed)")
            if approach_metrics:
                print(
                    f"distance link7->{args.approach_object_key}: "
                    f"before={approach_metrics['before_link7_to_object_m']:.4f}m, "
                    f"target={approach_metrics['target_link7_to_object_m']:.4f}m, "
                    f"delta={approach_metrics['target_minus_before_m']:+.4f}m"
                )
            print(f"{args.side}_pose: {compact_pose(target_pose)}")

            operator = prompt_operator()
            step_record: Dict[str, Any] = {
                "idx": idx,
                "request_id": request_id,
                "server_ok": bool(response.get("ok")),
                "target_source": args.target_source,
                "target_adapter": args.target_adapter,
                "raw_target_pose": raw_target_pose,
                "target_pose": target_pose,
                "target_adapter_info": adapter_info,
                "target_delta_m": target_delta.tolist(),
                "target_delta_norm_m": target_delta_norm,
                "target_rotation_delta_deg": target_rotation_delta_deg,
                "approach_metrics": approach_metrics,
                "operator_input": operator,
                "server_response": response,
            }

            if operator == "q":
                log("operator requested quit")
                step_record["executed"] = False
                report["steps"].append(step_record)
                break
            if operator == "s":
                log("operator skipped this target")
                step_record["executed"] = False
                report["steps"].append(step_record)
                continue
            if args.confirm_control != "RUN_CONTROL":
                log("refusing to execute because --confirm-control RUN_CONTROL is missing")
                step_record["executed"] = False
                step_record["blocked_reason"] = "missing RUN_CONTROL confirmation"
                report["steps"].append(step_record)
                break

            log(f"step {idx}: executing one EE target")
            control_result = call_ee_control(
                arm.controller,
                target_pose,
                args.side,
                args.lifetime,
                args.send_hz,
                args.execute_s,
            )
            report["control_sent"] = True
            step_record["executed"] = bool(control_result.get("ok"))
            step_record["control_result"] = control_result

            immediate_T_link7 = arm.get_T_link7_in_base()
            immediate_delta = immediate_T_link7[:3, 3] - before_T_link7[:3, 3]
            step_record["immediate_T_link7_in_base"] = immediate_T_link7.tolist()
            step_record["immediate_delta_m"] = immediate_delta.tolist()
            step_record["immediate_delta_norm_m"] = float(np.linalg.norm(immediate_delta))
            log(
                f"step {idx}: immediate_delta={immediate_delta.tolist()} "
                f"norm={np.linalg.norm(immediate_delta):.4f}"
            )
            time.sleep(float(args.settle_s))
            after_T_link7 = arm.get_T_link7_in_base()
            observed_delta = after_T_link7[:3, 3] - before_T_link7[:3, 3]
            step_record["after_T_link7_in_base"] = after_T_link7.tolist()
            step_record["settled_delta_m"] = observed_delta.tolist()
            step_record["settled_delta_norm_m"] = float(np.linalg.norm(observed_delta))
            step_record["observed_delta_m"] = observed_delta.tolist()
            step_record["observed_delta_norm_m"] = float(np.linalg.norm(observed_delta))
            step_record["observed_rotation_delta_deg"] = rotation_angle_deg(
                after_T_link7[:3, :3] @ before_T_link7[:3, :3].T
            )
            step_record["observed_rotation_vector_base_deg"] = rotation_vector_from_delta(
                after_T_link7[:3, :3] @ before_T_link7[:3, :3].T
            ).tolist()
            if object_base is not None:
                after_dist = float(np.linalg.norm(after_T_link7[:3, 3] - object_base))
                step_record["observed_approach_metrics"] = {
                    "object_key": args.approach_object_key,
                    "before_link7_to_object_m": approach_metrics["before_link7_to_object_m"] if approach_metrics else None,
                    "after_link7_to_object_m": after_dist,
                    "after_minus_before_m": after_dist - (approach_metrics["before_link7_to_object_m"] if approach_metrics else after_dist),
                    "closer": bool(approach_metrics and after_dist < approach_metrics["before_link7_to_object_m"]),
                }
            log(
                f"step {idx}: settled_delta={observed_delta.tolist()} "
                f"norm={np.linalg.norm(observed_delta):.4f} "
                f"rot_deg={step_record['observed_rotation_delta_deg']:.2f}"
            )
            if object_base is not None:
                log(
                    f"step {idx}: observed link7->{args.approach_object_key} "
                    f"distance {step_record['observed_approach_metrics']['after_link7_to_object_m']:.4f}m "
                    f"delta={step_record['observed_approach_metrics']['after_minus_before_m']:+.4f}m"
                )
            report["steps"].append(step_record)
            (step_dir / "step_record.json").write_text(
                json.dumps(json_safe(step_record), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        report["ok"] = True
    except Exception as exc:
        report.update(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if cam is not None:
            report["camera_close_skipped"] = True
            log("skipping G1 camera close for interactive test to avoid SDK shutdown blocking")
        if arm is not None:
            report["arm_close_skipped"] = True
            log("skipping read-only arm adapter close for interactive test")

    (run_dir / "interactive_step_report.json").write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log("building local result zip")
    zip_path = make_zip(run_dir)
    upload = None
    if args.upload_url:
        try:
            log(f"uploading result zip to {args.upload_url}")
            upload = upload_zip(zip_path, args.upload_url, args.upload_timeout_s)
            log(f"upload complete status={upload.get('status')}")
        except Exception as exc:
            upload = {"ok": False, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        (run_dir / "upload_result.json").write_text(json.dumps(upload, ensure_ascii=False, indent=2), encoding="utf-8")
        zip_path = make_zip(run_dir)

    print(json.dumps({"run_dir": str(run_dir), "zip_path": str(zip_path), "upload": upload}, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
