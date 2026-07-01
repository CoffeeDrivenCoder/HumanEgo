#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive one-step HumanEgo control client for G1.

Each iteration:
  1. read current G1 RGB-D and link7/TCP state
  2. ask the HumanEgo server for one target
  3. print the proposed target
  4. wait for operator input
  5. execute exactly one EE control command

This script is intentionally step-by-step unless --control-mode auto is set.
"""

from __future__ import annotations

import argparse
import ast
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
from g1_artifacts import artifact_dir, run_dir as artifact_run_dir  # noqa: E402
from g1_urdf_ik import DEFAULT_G1_ZIP, G1UrdfKinematics, pose_error  # noqa: E402


EPS = 1e-12
IK_RELAXED_POSITION_TOLERANCE_M = 0.010
IK_RELAXED_ROTATION_TOLERANCE_DEG = 5.0


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


def build_payload(
    frame: Any,
    state: dict[str, Any],
    args: argparse.Namespace,
    request_id: str,
    locked_objects: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    if locked_objects:
        T_base_in_cam = np.asarray(state["T_base_in_cam"], dtype=np.float64).reshape(4, 4)
        payload_objects: dict[str, Any] = {}
        for key, item in locked_objects.items():
            T_obj_in_base = np.asarray(item["T_in_base"], dtype=np.float64).reshape(4, 4)
            T_obj_in_cam = T_base_in_cam @ T_obj_in_base
            payload_objects[key] = {
                "T_in_cam": T_obj_in_cam.tolist(),
                "kpts_local": item.get("kpts_local", []),
                "lock_source": item.get("lock_source"),
            }
        payload["objects"] = payload_objects
        payload["object_lock"] = {
            "mode": "base_static",
            "object_keys": sorted(payload_objects.keys()),
        }
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


def object_quality_warnings(response: dict[str, Any]) -> dict[str, list[str]]:
    vision = response.get("vision_summary") or {}
    out: dict[str, list[str]] = {}
    for key, item in (vision.get("objects") or {}).items():
        warnings = item.get("warnings") or []
        if warnings:
            out[str(key)] = [str(v) for v in warnings]
    return out


def lock_objects_in_base(response: dict[str, Any], state: dict[str, Any], request_id: str) -> dict[str, Any]:
    objects = (response.get("input_summary") or {}).get("objects") or {}
    T_base_camera = np.asarray(state["T_base_camera"], dtype=np.float64).reshape(4, 4)
    locked: dict[str, Any] = {}
    for key, item in objects.items():
        T_obj_in_cam = np.asarray(item["T_in_cam"], dtype=np.float64).reshape(4, 4)
        locked[str(key)] = {
            "T_in_base": (T_base_camera @ T_obj_in_cam).tolist(),
            "kpts_local": item.get("kpts_local", []),
            "kpts_local_count": item.get("kpts_local_count"),
            "lock_source": request_id,
        }
    return locked


def locked_object_summary(locked_objects: dict[str, Any] | None) -> dict[str, Any] | None:
    if not locked_objects:
        return None
    return {
        key: {
            "T_in_base_xyz_m": [float(v) for v in np.asarray(item["T_in_base"], dtype=np.float64)[:3, 3]],
            "kpts_local_count": item.get("kpts_local_count"),
            "lock_source": item.get("lock_source"),
        }
        for key, item in locked_objects.items()
    }


def compact_step_summary(step_record: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    gripper_result = step_record.get("gripper_result") or {}
    gripper_before = (gripper_result.get("before") or {}).get("selected_raw")
    gripper_after = (gripper_result.get("after") or {}).get("selected_raw")
    gripper_before_values = (gripper_result.get("before") or {}).get("values")
    gripper_after_values = (gripper_result.get("after") or {}).get("values")
    post_ee_tracking = step_record.get("post_ee_translation_tracking") or {}
    settled_tracking = step_record.get("settled_translation_tracking") or {}
    approach = step_record.get("approach_metrics") or {}
    post_ee_approach = step_record.get("post_ee_approach_metrics") or {}
    observed_approach = step_record.get("observed_approach_metrics") or {}
    axis = step_record.get("axis_alignment") or {}
    tracking_gate = step_record.get("tracking_gate") or {}
    control_result = step_record.get("control_result") or {}
    control_delta_pose = control_result.get("delta_pose") or {}
    control_ik = control_result.get("ik") or {}
    control_ik_error = control_result.get("ik_fk_vs_target_error") or {}
    post_ee_target_error = step_record.get("post_ee_target_pose_error") or {}
    settled_target_error = step_record.get("settled_target_pose_error") or {}
    closed_loop = step_record.get("closed_loop_result") or {}
    return {
        "idx": step_record.get("idx"),
        "request_id": step_record.get("request_id"),
        "operator_input": step_record.get("operator_input"),
        "executed": step_record.get("executed"),
        "server_ok": step_record.get("server_ok"),
        "done_prob": (response.get("policy_preview") or {}).get("done_prob"),
        "object_source_used": (response.get("input_summary") or {}).get("object_source_used"),
        "object_lock_active": step_record.get("object_lock_active"),
        "vision_warnings": step_record.get("vision_warnings"),
        "target_delta_m": step_record.get("target_delta_m"),
        "target_delta_norm_m": step_record.get("target_delta_norm_m"),
        "target_rotation_delta_deg": step_record.get("target_rotation_delta_deg"),
        "ee_control_mode": step_record.get("ee_control_mode"),
        "delta_pose_action_data": control_delta_pose.get("action_data"),
        "delta_pose_rotation_vector_deg": control_delta_pose.get("rotation_vector_deg"),
        "delta_pose_rotation_frame": control_delta_pose.get("rotation_frame"),
        "ik_abs_joint_success": control_ik.get("success"),
        "ik_abs_joint_q_delta_abs_max_rad": control_result.get("q_delta_abs_max_rad"),
        "ik_abs_joint_q_delta_norm_rad": control_result.get("q_delta_norm_rad"),
        "ik_abs_joint_position_error_m": control_ik_error.get("position_error_m"),
        "ik_abs_joint_rotation_error_deg": control_ik_error.get("rotation_error_deg"),
        "ik_abs_joint_blocked_reason": control_result.get("blocked_reason"),
        "ik_abs_joint_selected_candidate": control_result.get("ik_selected_candidate"),
        "ik_abs_joint_selected_reason": control_result.get("ik_selected_reason"),
        "ik_abs_joint_pose_acceptance": control_result.get("ik_pose_acceptance"),
        "ik_abs_joint_candidate_count": control_result.get("ik_candidate_count"),
        "ik_abs_joint_safe_candidate_count": control_result.get("ik_safe_candidate_count"),
        "ik_abs_joint_strict_safe_candidate_count": control_result.get("ik_strict_safe_candidate_count"),
        "ik_abs_joint_relaxed_safe_candidate_count": control_result.get("ik_relaxed_safe_candidate_count"),
        "closed_loop_enabled": closed_loop.get("enabled"),
        "closed_loop_reached": closed_loop.get("reached"),
        "closed_loop_attempts": closed_loop.get("num_attempts"),
        "closed_loop_final_position_error_m": closed_loop.get("final_position_error_m"),
        "closed_loop_final_rotation_error_deg": closed_loop.get("final_rotation_error_deg"),
        "closed_loop_orientation_mode": closed_loop.get("orientation_mode"),
        "closed_loop_reason": closed_loop.get("reason"),
        "server_raw_delta_norm_m": (((response.get("policy_preview") or {}).get("sides") or {}).get("right") or [{}])[0].get("safety_translation_limit", {}).get("raw_delta_norm_m"),
        "server_clipped": (((response.get("policy_preview") or {}).get("sides") or {}).get("right") or [{}])[0].get("safety_translation_limit", {}).get("clipped"),
        "post_ee_delta_m": step_record.get("post_ee_delta_m"),
        "post_ee_delta_norm_m": step_record.get("post_ee_delta_norm_m"),
        "post_ee_rotation_delta_deg": step_record.get("post_ee_rotation_delta_deg"),
        "post_ee_target_position_error_m": post_ee_target_error.get("position_error_m"),
        "post_ee_target_rotation_error_deg": post_ee_target_error.get("rotation_error_deg"),
        "post_ee_cos_to_target": post_ee_tracking.get("cosine_to_target_delta"),
        "settled_delta_m": step_record.get("settled_delta_m"),
        "settled_delta_norm_m": step_record.get("settled_delta_norm_m"),
        "settled_rotation_delta_deg": step_record.get("observed_rotation_delta_deg"),
        "settled_target_position_error_m": settled_target_error.get("position_error_m"),
        "settled_target_rotation_error_deg": settled_target_error.get("rotation_error_deg"),
        "settled_cos_to_target": settled_tracking.get("cosine_to_target_delta"),
        "tracking_ratio": tracking_gate.get("ratio"),
        "tracking_gate_bad": tracking_gate.get("bad"),
        "tracking_gate_bad_streak": tracking_gate.get("bad_streak"),
        "tracking_gate_reason": tracking_gate.get("reason"),
        "distance_before_m": approach.get("before_link7_to_object_m"),
        "distance_target_m": approach.get("target_link7_to_object_m"),
        "distance_target_delta_m": approach.get("target_minus_before_m"),
        "distance_post_ee_m": post_ee_approach.get("post_ee_link7_to_object_m"),
        "distance_post_ee_delta_m": post_ee_approach.get("post_ee_minus_before_m"),
        "distance_after_m": observed_approach.get("after_link7_to_object_m"),
        "distance_after_delta_m": observed_approach.get("after_minus_before_m"),
        "gripper_target_0_open_1_closed": (step_record.get("gripper_command") or {}).get("command_0_open_1_closed"),
        "gripper_before_raw": gripper_before,
        "gripper_after_raw": gripper_after,
        "gripper_delta_raw": gripper_result.get("observed_delta_raw"),
        "gripper_before_values_raw": gripper_before_values,
        "gripper_after_values_raw": gripper_after_values,
        "gripper_delta_values_raw": gripper_result.get("observed_delta_values_raw"),
        "tcp_current_best_axis": (axis.get("current") or {}).get("best_axis"),
        "tcp_current_angle_deg": (axis.get("current") or {}).get("best_angle_to_object_deg"),
        "tcp_target_best_axis": (axis.get("target") or {}).get("best_axis"),
        "tcp_target_angle_deg": (axis.get("target") or {}).get("best_angle_to_object_deg"),
    }


def append_step_outputs(
    step_record: dict[str, Any],
    response: dict[str, Any],
    report: dict[str, Any],
    step_summaries: list[dict[str, Any]],
    step_summaries_path: Path,
    step_summaries_jsonl_path: Path,
    step_dir: Path | None = None,
) -> dict[str, Any]:
    summary = compact_step_summary(step_record, response)
    step_summaries.append(summary)
    step_summaries_path.write_text(json.dumps(json_safe(step_summaries), ensure_ascii=False, indent=2), encoding="utf-8")
    with step_summaries_jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(summary), ensure_ascii=False) + "\n")
    report["steps"].append(step_record)
    if step_dir is not None:
        (step_dir / "step_record.json").write_text(
            json.dumps(json_safe(step_record), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return summary


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


def rotation_vector_rad_from_delta(R_delta: np.ndarray) -> np.ndarray:
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
    return axis / axis_norm * angle


def link7_delta_pose_command(
    before_T_link7: np.ndarray,
    target_T_link7: np.ndarray,
    rotation_frame: str = "base",
) -> dict[str, Any]:
    before_T_link7 = np.asarray(before_T_link7, dtype=np.float64).reshape(4, 4)
    target_T_link7 = np.asarray(target_T_link7, dtype=np.float64).reshape(4, 4)
    translation_delta = target_T_link7[:3, 3] - before_T_link7[:3, 3]
    if rotation_frame == "base":
        R_delta = target_T_link7[:3, :3] @ before_T_link7[:3, :3].T
    elif rotation_frame == "local":
        R_delta = before_T_link7[:3, :3].T @ target_T_link7[:3, :3]
    else:
        raise ValueError(f"unknown delta pose rotation frame {rotation_frame!r}; expected base/local")
    rotvec_rad = rotation_vector_rad_from_delta(R_delta)
    action_data = [
        float(translation_delta[0]),
        float(translation_delta[1]),
        float(translation_delta[2]),
        float(rotvec_rad[0]),
        float(rotvec_rad[1]),
        float(rotvec_rad[2]),
    ]
    return {
        "translation_delta_m": translation_delta.tolist(),
        "rotation_frame": rotation_frame,
        "rotation_delta_matrix": R_delta.tolist(),
        "rotation_vector_rad": rotvec_rad.tolist(),
        "rotation_vector_deg": np.degrees(rotvec_rad).tolist(),
        "rotation_angle_deg": rotation_angle_deg(R_delta),
        "action_data": action_data,
    }


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


def axis_alignment_to_object(
    T_link7_in_base: np.ndarray,
    T_tcp_in_link7: np.ndarray,
    object_base: np.ndarray,
) -> dict[str, Any]:
    T_tcp_in_base = np.asarray(T_link7_in_base, dtype=np.float64) @ np.asarray(T_tcp_in_link7, dtype=np.float64)
    tcp_pos = T_tcp_in_base[:3, 3]
    to_object = np.asarray(object_base, dtype=np.float64).reshape(3) - tcp_pos
    dist = float(np.linalg.norm(to_object))
    if dist <= EPS:
        return {
            "ok": False,
            "reason": "tcp is at object position",
            "tcp_position_in_base": tcp_pos.tolist(),
            "tcp_to_object_dist_m": dist,
        }

    direction = to_object / dist
    axes: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(("x", "y", "z")):
        axis_vec = T_tcp_in_base[:3, idx]
        axis_vec = axis_vec / max(float(np.linalg.norm(axis_vec)), EPS)
        for sign, prefix in ((1.0, "+"), (-1.0, "-")):
            signed_axis = sign * axis_vec
            dot = float(np.clip(np.dot(signed_axis, direction), -1.0, 1.0))
            axes[f"{prefix}{name}"] = {
                "dot_to_object": dot,
                "angle_to_object_deg": float(np.degrees(np.arccos(dot))),
            }

    best_axis = min(axes, key=lambda key: axes[key]["angle_to_object_deg"])
    return {
        "ok": True,
        "tcp_position_in_base": tcp_pos.tolist(),
        "object_position_in_base": np.asarray(object_base, dtype=np.float64).reshape(3).tolist(),
        "object_vector_from_tcp_m": to_object.tolist(),
        "tcp_to_object_dist_m": dist,
        "axes": axes,
        "best_axis": best_axis,
        "best_angle_to_object_deg": axes[best_axis]["angle_to_object_deg"],
        "best_dot_to_object": axes[best_axis]["dot_to_object"],
    }


def alignment_improvement(current: dict[str, Any], target: dict[str, Any]) -> dict[str, Any] | None:
    if not current or not target or not current.get("ok") or not target.get("ok"):
        return None
    target_axis = target["best_axis"]
    current_angle = current["axes"][target_axis]["angle_to_object_deg"]
    target_angle = target["axes"][target_axis]["angle_to_object_deg"]
    return {
        "target_best_axis": target_axis,
        "current_angle_for_target_axis_deg": current_angle,
        "target_angle_for_target_axis_deg": target_angle,
        "angle_reduction_deg": current_angle - target_angle,
    }


def compact_alignment(label: str, alignment: dict[str, Any], improvement: dict[str, Any] | None = None) -> str:
    if not alignment or not alignment.get("ok"):
        return f"{label}: unavailable"
    text = (
        f"{label}: best_axis={alignment['best_axis']} "
        f"angle={alignment['best_angle_to_object_deg']:.1f}deg "
        f"dist={alignment['tcp_to_object_dist_m']:.3f}m"
    )
    if improvement:
        text += f" improve={improvement['angle_reduction_deg']:+.1f}deg"
    return text


def translation_tracking_report(target_delta: np.ndarray, observed_delta: np.ndarray) -> dict[str, Any]:
    target_delta = np.asarray(target_delta, dtype=np.float64).reshape(3)
    observed_delta = np.asarray(observed_delta, dtype=np.float64).reshape(3)
    target_norm = float(np.linalg.norm(target_delta))
    observed_norm = float(np.linalg.norm(observed_delta))
    out: dict[str, Any] = {
        "target_delta_m": target_delta.tolist(),
        "observed_delta_m": observed_delta.tolist(),
        "target_norm_m": target_norm,
        "observed_norm_m": observed_norm,
        "error_m": (observed_delta - target_delta).tolist(),
        "error_norm_m": float(np.linalg.norm(observed_delta - target_delta)),
    }
    if target_norm > EPS and observed_norm > EPS:
        cosine = float(np.dot(target_delta, observed_delta) / (target_norm * observed_norm))
        out["cosine_to_target_delta"] = float(np.clip(cosine, -1.0, 1.0))
    return out


def adapt_target_pose(
    target_pose: dict[str, float],
    before_T_link7: np.ndarray,
    mode: str,
    axis_step_m: float,
    target_z_bias_m: float,
    max_orientation_deg: float,
    probe_axis: str,
    probe_deg: float,
    probe_frame: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    adapted = dict(target_pose)
    current_pose = pose_dict_from_T(before_T_link7)
    target_z_bias_m = float(target_z_bias_m)
    if abs(target_z_bias_m) > EPS:
        adapted["z"] = float(adapted["z"] + target_z_bias_m)
    if mode == "full":
        return adapted, {"mode": mode, "target_z_bias_m": target_z_bias_m}

    raw_delta = position_from_pose_dict(target_pose) - before_T_link7[:3, 3]
    biased_delta = position_from_pose_dict(adapted) - before_T_link7[:3, 3]
    if mode == "position_only":
        for key in ("qx", "qy", "qz", "qw"):
            adapted[key] = current_pose[key]
        return adapted, {
            "mode": mode,
            "raw_delta_m": raw_delta.tolist(),
            "biased_delta_m": biased_delta.tolist(),
            "target_z_bias_m": target_z_bias_m,
        }

    if mode == "axis_only":
        axis = int(np.argmax(np.abs(biased_delta)))
        step = float(np.clip(biased_delta[axis], -abs(axis_step_m), abs(axis_step_m)))
        target_xyz = before_T_link7[:3, 3].copy()
        target_xyz[axis] += step
        adapted.update({"x": float(target_xyz[0]), "y": float(target_xyz[1]), "z": float(target_xyz[2])})
        for key in ("qx", "qy", "qz", "qw"):
            adapted[key] = current_pose[key]
        return adapted, {
            "mode": mode,
            "raw_delta_m": raw_delta.tolist(),
            "biased_delta_m": biased_delta.tolist(),
            "target_z_bias_m": target_z_bias_m,
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
            "biased_delta_m": biased_delta.tolist(),
            "target_z_bias_m": target_z_bias_m,
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
            "biased_delta_m": biased_delta.tolist(),
            "target_z_bias_m": target_z_bias_m,
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


def read_joint_state_with_retry(
    getter: Any,
    name: str,
    min_len: int,
    tries: int = 20,
    sleep_s: float = 0.05,
) -> dict[str, Any]:
    last_raw = None
    last_error = None
    for _ in range(max(1, int(tries))):
        raw = getter()
        last_raw = raw
        try:
            values = coerce_float_list(raw)
            if len(values) >= int(min_len):
                data, timestamp = split_state_result(raw)
                return {
                    "values": values,
                    "raw": json_safe(raw),
                    "data": json_safe(data),
                    "timestamp": json_safe(timestamp),
                    "length": len(values),
                }
            last_error = f"expected at least {min_len} values, got {len(values)}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(max(0.0, float(sleep_s)))
    raise RuntimeError(f"{name} not ready after {tries} tries: {last_error}; last_raw={last_raw!r}")


def read_robot_joint_states_for_trajectory(robot: Any) -> dict[str, Any]:
    head_state = read_joint_state_with_retry(robot.head_joint_states, "head_joint_states", min_len=2)
    waist_state = read_joint_state_with_retry(robot.waist_joint_states, "waist_joint_states", min_len=2)
    arm_state = read_joint_state_with_retry(robot.arm_joint_states, "arm_joint_states", min_len=14)
    head = head_state["values"]
    waist = waist_state["values"]
    arm = arm_state["values"]
    return {
        "head": head,
        "waist": waist,
        "arm": arm,
        "raw": {
            "head": head_state["raw"],
            "waist": waist_state["raw"],
            "arm": arm_state["raw"],
        },
        "timestamps": {
            "head": head_state["timestamp"],
            "waist": waist_state["timestamp"],
            "arm": arm_state["timestamp"],
        },
        "lengths": {
            "head": head_state["length"],
            "waist": waist_state["length"],
            "arm": arm_state["length"],
        },
    }


def side_q_from_arm_state(arm_values: list[float], side: str, mapping: str) -> tuple[np.ndarray, list[int]]:
    values = [float(v) for v in arm_values]
    if len(values) < 14:
        raise ValueError(f"arm_joint_states must contain at least 14 values, got {len(values)}")
    side = side.lower()
    mapping = mapping.lower()
    if mapping == "left_first":
        indices = list(range(0, 7)) if side == "left" else list(range(7, 14))
    elif mapping == "right_first":
        indices = list(range(7, 14)) if side == "left" else list(range(0, 7))
    else:
        raise ValueError(f"unknown arm state mapping {mapping!r}")
    return np.asarray([values[i] for i in indices], dtype=np.float64), indices


def arm_state_with_side_q(arm_values: list[float], side: str, mapping: str, q_side: np.ndarray) -> list[float]:
    out = [float(v) for v in arm_values]
    q = np.asarray(q_side, dtype=np.float64).reshape(7)
    _, indices = side_q_from_arm_state(out, side, mapping)
    for idx, value in zip(indices, q):
        out[idx] = float(value)
    return out


def split_left_right_from_full_arm(arm_values: list[float], mapping: str) -> tuple[list[float], list[float]]:
    left_q, _ = side_q_from_arm_state(arm_values, "left", mapping)
    right_q, _ = side_q_from_arm_state(arm_values, "right", mapping)
    return left_q.tolist(), right_q.tolist()


def waist_values_with_height_offset(waist_values: list[float], height_offset_m: float) -> list[float]:
    values = [float(v) for v in waist_values]
    if len(values) >= 2:
        values[1] += float(height_offset_m)
    return values


def interpolate_q(q_start: np.ndarray, q_target: np.ndarray, num_points: int) -> list[np.ndarray]:
    q_start = np.asarray(q_start, dtype=np.float64).reshape(7)
    q_target = np.asarray(q_target, dtype=np.float64).reshape(7)
    count = max(1, int(num_points))
    return [q_start + (q_target - q_start) * (idx / count) for idx in range(1, count + 1)]


def ik_seed_offsets() -> list[tuple[str, np.ndarray]]:
    zeros = np.zeros(7, dtype=np.float64)
    specs: list[tuple[str, np.ndarray]] = [("current", zeros)]
    for joint_idx, magnitude in [(3, 0.10), (3, 0.20), (5, 0.12), (6, 0.12), (1, 0.08)]:
        for sign in (-1.0, 1.0):
            offset = zeros.copy()
            offset[joint_idx] = sign * magnitude
            specs.append((f"j{joint_idx + 1}_{sign * magnitude:+.2f}", offset))
    for name, pairs in [
        ("elbow_wrist_a", [(3, -0.16), (5, -0.10), (6, -0.08)]),
        ("elbow_wrist_b", [(3, 0.16), (5, 0.10), (6, 0.08)]),
        ("elbow_wrist_c", [(3, -0.16), (5, 0.10), (6, -0.08)]),
        ("elbow_wrist_d", [(3, 0.16), (5, -0.10), (6, 0.08)]),
    ]:
        offset = zeros.copy()
        for joint_idx, value in pairs:
            offset[joint_idx] = value
        specs.append((name, offset))
    return specs


def ik_weight_profiles() -> list[dict[str, float | str]]:
    return [
        {"name": "default", "position_weight": 20.0, "rotation_weight": 1.0, "smooth_weight": 0.035, "home_weight": 0.005},
        {"name": "rot2_low_smooth", "position_weight": 20.0, "rotation_weight": 2.5, "smooth_weight": 0.010, "home_weight": 0.002},
        {"name": "rot5_low_smooth", "position_weight": 24.0, "rotation_weight": 5.0, "smooth_weight": 0.003, "home_weight": 0.000},
        {"name": "rot8_no_home", "position_weight": 30.0, "rotation_weight": 8.0, "smooth_weight": 0.001, "home_weight": 0.000},
        {"name": "balanced_pose", "position_weight": 80.0, "rotation_weight": 12.0, "smooth_weight": 0.0005, "home_weight": 0.000},
        {"name": "position_first", "position_weight": 200.0, "rotation_weight": 1.0, "smooth_weight": 0.010, "home_weight": 0.000},
        {"name": "rotation_first", "position_weight": 20.0, "rotation_weight": 50.0, "smooth_weight": 0.003, "home_weight": 0.000},
        {"name": "pose_no_smooth", "position_weight": 35.0, "rotation_weight": 10.0, "smooth_weight": 0.000, "home_weight": 0.000},
    ]


def candidate_summary_from_ik(
    *,
    name: str,
    seed_name: str,
    weight_name: str,
    ik: Any,
    q_before: np.ndarray,
    max_joint_delta_rad: float,
) -> dict[str, Any]:
    q_delta = np.asarray(ik.q_solution, dtype=np.float64).reshape(7) - np.asarray(q_before, dtype=np.float64).reshape(7)
    q_delta_abs_max = float(np.max(np.abs(q_delta)))
    safe_by_delta = bool(q_delta_abs_max <= float(max_joint_delta_rad))
    relaxed_pose_ok = bool(
        float(ik.position_error_m) <= IK_RELAXED_POSITION_TOLERANCE_M
        and float(ik.rotation_error_deg) <= IK_RELAXED_ROTATION_TOLERANCE_DEG
    )
    return {
        "name": name,
        "seed_name": seed_name,
        "weight_name": weight_name,
        "success": bool(ik.success),
        "relaxed_pose_ok": relaxed_pose_ok,
        "relaxed_position_tolerance_m": IK_RELAXED_POSITION_TOLERANCE_M,
        "relaxed_rotation_tolerance_deg": IK_RELAXED_ROTATION_TOLERANCE_DEG,
        "safe_by_joint_delta": safe_by_delta,
        "position_error_m": float(ik.position_error_m),
        "rotation_error_deg": float(ik.rotation_error_deg),
        "q_delta_abs_max_rad": q_delta_abs_max,
        "q_delta_norm_rad": float(np.linalg.norm(q_delta)),
        "min_limit_margin_rad": ik.min_limit_margin_rad,
        "num_function_evals": int(ik.num_function_evals),
        "cost": float(ik.cost),
        "q_solution": np.asarray(ik.q_solution, dtype=np.float64).tolist(),
        "q_delta_from_current": q_delta.tolist(),
        "ik": ik.to_json(),
    }


def select_ik_abs_joint_candidate(
    kin: G1UrdfKinematics,
    side: str,
    target_T: np.ndarray,
    q_before: np.ndarray,
    waist_for_urdf: list[float],
    max_nfev: int,
    max_joint_delta_rad: float,
) -> dict[str, Any]:
    side = kin.side_from_name(side)
    target_T = np.asarray(target_T, dtype=np.float64).reshape(4, 4)
    q_before = np.asarray(q_before, dtype=np.float64).reshape(7)
    lows, highs = kin.joint_limits(side)
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for seed_name, offset in ik_seed_offsets():
        q_seed = np.clip(q_before + offset, lows, highs)
        for weights in ik_weight_profiles():
            weight_name = str(weights["name"])
            if (seed_name, weight_name) in seen:
                continue
            seen.add((seed_name, weight_name))
            ik = kin.solve_link7_ik(
                side,
                target_T,
                q_seed,
                waist_states=waist_for_urdf,
                max_nfev=max_nfev,
                position_weight=float(weights["position_weight"]),
                rotation_weight=float(weights["rotation_weight"]),
                smooth_weight=float(weights["smooth_weight"]),
                home_weight=float(weights["home_weight"]),
            )
            candidates.append(
                candidate_summary_from_ik(
                    name=f"{seed_name}/{weight_name}",
                    seed_name=seed_name,
                    weight_name=weight_name,
                    ik=ik,
                    q_before=q_before,
                    max_joint_delta_rad=max_joint_delta_rad,
                )
            )

    safe_candidates = [item for item in candidates if item["safe_by_joint_delta"]]
    strict_safe = [item for item in safe_candidates if item["success"]]
    relaxed_safe = [item for item in safe_candidates if item["relaxed_pose_ok"]]
    if strict_safe:
        selected = min(
            strict_safe,
            key=lambda item: (
                float(item["rotation_error_deg"]),
                float(item["position_error_m"]),
                float(item["q_delta_abs_max_rad"]),
                float(item["cost"]),
            ),
        )
        reason = "strict_success_safe"
        pose_acceptance = "strict"
    elif relaxed_safe:
        selected = min(
            relaxed_safe,
            key=lambda item: (
                max(
                    float(item["position_error_m"]) / IK_RELAXED_POSITION_TOLERANCE_M,
                    float(item["rotation_error_deg"]) / IK_RELAXED_ROTATION_TOLERANCE_DEG,
                ),
                float(item["position_error_m"]) / IK_RELAXED_POSITION_TOLERANCE_M
                + float(item["rotation_error_deg"]) / IK_RELAXED_ROTATION_TOLERANCE_DEG,
                float(item["q_delta_abs_max_rad"]),
                float(item["cost"]),
            ),
        )
        reason = "relaxed_pose_safe"
        pose_acceptance = "relaxed"
    elif safe_candidates:
        selected = min(
            safe_candidates,
            key=lambda item: (
                float(item["rotation_error_deg"]),
                float(item["position_error_m"]),
                float(item["q_delta_abs_max_rad"]),
                float(item["cost"]),
            ),
        )
        reason = "best_safe_but_not_strict_success"
        pose_acceptance = "none"
    else:
        selected = min(
            candidates,
            key=lambda item: (
                float(item["q_delta_abs_max_rad"]),
                float(item["rotation_error_deg"]),
                float(item["position_error_m"]),
                float(item["cost"]),
            ),
        )
        reason = "no_safe_candidate"
        pose_acceptance = "none"

    best_overall = min(
        candidates,
        key=lambda item: (
            float(item["rotation_error_deg"]),
            float(item["position_error_m"]),
            float(item["q_delta_abs_max_rad"]),
            float(item["cost"]),
        ),
    )
    return {
        "selected": selected,
        "selected_reason": reason,
        "pose_acceptance": pose_acceptance,
        "ok": bool(selected["safe_by_joint_delta"] and (selected["success"] or selected["relaxed_pose_ok"])),
        "candidate_count": len(candidates),
        "safe_candidate_count": len(safe_candidates),
        "strict_safe_candidate_count": len(strict_safe),
        "relaxed_safe_candidate_count": len(relaxed_safe),
        "relaxed_position_tolerance_m": IK_RELAXED_POSITION_TOLERANCE_M,
        "relaxed_rotation_tolerance_deg": IK_RELAXED_ROTATION_TOLERANCE_DEG,
        "best_overall": best_overall,
        "candidates": candidates,
    }


def call_ee_ik_abs_joint_trajectory_once(
    controller: Any,
    robot: Any,
    target_T_link7: np.ndarray,
    side: str,
    trajectory_reference_time: float,
    waist_height_offset_m: float,
    arm_state_mapping: str,
    num_points: int,
    max_nfev: int,
    max_joint_delta_rad: float,
    urdf_zip: str,
    execute_control: bool = True,
) -> dict[str, Any]:
    started = time.time()
    target_T = np.asarray(target_T_link7, dtype=np.float64).reshape(4, 4)
    joint_states = read_robot_joint_states_for_trajectory(robot)
    waist_for_urdf = waist_values_with_height_offset(joint_states["waist"], waist_height_offset_m)
    q_before, arm_indices = side_q_from_arm_state(joint_states["arm"], side, arm_state_mapping)
    kin = G1UrdfKinematics(urdf_zip)
    before_fk_T = kin.link7_fk(side, q_before, waist_states=waist_for_urdf)
    ik_selection = select_ik_abs_joint_candidate(
        kin,
        side,
        target_T,
        q_before,
        waist_for_urdf,
        max_nfev,
        max_joint_delta_rad,
    )
    selected_ik = ik_selection["selected"]
    q_target = np.asarray(selected_ik["q_solution"], dtype=np.float64).reshape(7)
    q_delta = q_target - q_before
    q_delta_norm = float(np.linalg.norm(q_delta))
    q_delta_abs_max = float(np.max(np.abs(q_delta)))
    target_fk_T = kin.link7_fk(side, q_target, waist_states=waist_for_urdf)
    ik_fk_error = pose_error(target_fk_T, target_T)
    ik_selected_brief = {
        "name": selected_ik.get("name"),
        "seed_name": selected_ik.get("seed_name"),
        "weight_name": selected_ik.get("weight_name"),
        "success": selected_ik.get("success"),
        "relaxed_pose_ok": selected_ik.get("relaxed_pose_ok"),
        "safe_by_joint_delta": selected_ik.get("safe_by_joint_delta"),
        "position_error_m": selected_ik.get("position_error_m"),
        "rotation_error_deg": selected_ik.get("rotation_error_deg"),
        "q_delta_abs_max_rad": selected_ik.get("q_delta_abs_max_rad"),
        "q_delta_norm_rad": selected_ik.get("q_delta_norm_rad"),
        "cost": selected_ik.get("cost"),
    }
    ik_candidate_fields = {
        "ik_selected_candidate": ik_selected_brief,
        "ik_selected_reason": ik_selection["selected_reason"],
        "ik_pose_acceptance": ik_selection["pose_acceptance"],
        "ik_candidate_count": ik_selection["candidate_count"],
        "ik_safe_candidate_count": ik_selection["safe_candidate_count"],
        "ik_strict_safe_candidate_count": ik_selection["strict_safe_candidate_count"],
        "ik_relaxed_safe_candidate_count": ik_selection["relaxed_safe_candidate_count"],
        "ik_relaxed_position_tolerance_m": ik_selection["relaxed_position_tolerance_m"],
        "ik_relaxed_rotation_tolerance_deg": ik_selection["relaxed_rotation_tolerance_deg"],
        "ik_best_overall_candidate": ik_selection["best_overall"],
        "ik_candidates": ik_selection["candidates"],
    }

    def base_result() -> dict[str, Any]:
        return {
            "duration_s": time.time() - started,
            "mode": "ik_abs_joint",
            "side": side,
            "joint_states": joint_states,
            "waist_values_for_urdf": waist_for_urdf,
            "arm_state_mapping": arm_state_mapping,
            "arm_state_indices": arm_indices,
            "q_before": q_before.tolist(),
            "q_target": q_target.tolist(),
            "q_delta": q_delta.tolist(),
            "q_delta_norm_rad": q_delta_norm,
            "q_delta_abs_max_rad": q_delta_abs_max,
            "before_fk_T_link7_in_base": before_fk_T.tolist(),
            "target_T_link7_in_base": target_T.tolist(),
            "target_fk_T_link7_in_base": target_fk_T.tolist(),
            "ik": selected_ik["ik"],
            "ik_fk_vs_target_error": ik_fk_error,
            **ik_candidate_fields,
        }

    if not ik_selection["ok"]:
        if not selected_ik.get("safe_by_joint_delta"):
            blocked_reason = "joint_delta_too_large"
        elif not selected_ik.get("success") and not selected_ik.get("relaxed_pose_ok"):
            blocked_reason = "ik_failed_all_candidates"
        else:
            blocked_reason = "ik_selection_failed"
        result = {
            **base_result(),
            "ok": False,
            "blocked_reason": blocked_reason,
            "max_joint_delta_rad": float(max_joint_delta_rad),
        }
        return {
            key: json_safe(value)
            for key, value in result.items()
        }

    q_rows = interpolate_q(q_before, q_target, num_points)
    full_arm_rows = [arm_state_with_side_q(joint_states["arm"], side, arm_state_mapping, row) for row in q_rows]
    robot_actions = []
    for full_arm in full_arm_rows:
        left_q, right_q = split_left_right_from_full_arm(full_arm, arm_state_mapping)
        robot_actions.append(
            {
                "left_arm": {"action_data": left_q, "control_type": "ABS_JOINT"},
                "right_arm": {"action_data": right_q, "control_type": "ABS_JOINT"},
            }
        )
    kwargs = {
        "infer_timestamp": int(time.time() * 1e9),
        "robot_states": {
            "head": joint_states["head"],
            "waist": joint_states["waist"],
            "arm": joint_states["arm"],
        },
        "robot_actions": robot_actions,
        "robot_link": "base_link",
        "trajectory_reference_time": float(trajectory_reference_time),
    }
    if not execute_control:
        result = {
            **base_result(),
            "ok": True,
            "preview_only": True,
            "num_points": int(num_points),
            "trajectory_reference_time": float(trajectory_reference_time),
            "full_arm_target_rows": full_arm_rows,
            "kwargs": json_safe(kwargs),
        }
        return {
            key: json_safe(value)
            for key, value in result.items()
        }
    try:
        result = controller.trajectory_tracking_control(**kwargs)
        output = {
            **base_result(),
            "ok": True,
            "num_points": int(num_points),
            "trajectory_reference_time": float(trajectory_reference_time),
            "full_arm_target_rows": full_arm_rows,
            "kwargs": json_safe(kwargs),
            "result": json_safe(result),
        }
        return {
            key: json_safe(value)
            for key, value in output.items()
        }
    except Exception as exc:
        output = {
            **base_result(),
            "ok": False,
            "num_points": int(num_points),
            "trajectory_reference_time": float(trajectory_reference_time),
            "kwargs": json_safe(kwargs),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        return {
            key: json_safe(value)
            for key, value in output.items()
        }


def call_ee_delta_pose_trajectory_once(
    controller: Any,
    robot: Any,
    before_T_link7: np.ndarray,
    target_T_link7: np.ndarray,
    side: str,
    trajectory_reference_time: float,
    rotation_frame: str,
) -> dict[str, Any]:
    delta_pose = link7_delta_pose_command(before_T_link7, target_T_link7, rotation_frame=rotation_frame)
    joint_states = read_robot_joint_states_for_trajectory(robot)
    zero = [0.0] * 6
    action_data = [float(v) for v in delta_pose["action_data"]]
    robot_action = {
        "left_arm": {
            "action_data": action_data if side == "left" else zero,
            "control_type": "DELTA_POSE",
        },
        "right_arm": {
            "action_data": action_data if side == "right" else zero,
            "control_type": "DELTA_POSE",
        },
    }
    infer_timestamp = int(time.time() * 1e9)
    robot_states = {
        "head": joint_states["head"],
        "waist": joint_states["waist"],
        "arm": joint_states["arm"],
    }
    kwargs = {
        "infer_timestamp": infer_timestamp,
        "robot_states": robot_states,
        "robot_actions": [robot_action],
        "robot_link": "base_link",
        "trajectory_reference_time": float(trajectory_reference_time),
    }
    started = time.time()
    try:
        result = controller.trajectory_tracking_control(**kwargs)
        return {
            "ok": True,
            "duration_s": time.time() - started,
            "mode": "delta_pose",
            "side": side,
            "delta_pose": delta_pose,
            "joint_states": joint_states,
            "kwargs": json_safe(kwargs),
            "result": json_safe(result),
        }
    except Exception as exc:
        return {
            "ok": False,
            "duration_s": time.time() - started,
            "mode": "delta_pose",
            "side": side,
            "delta_pose": delta_pose,
            "joint_states": joint_states,
            "kwargs": json_safe(kwargs),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def call_ee_control_mode(
    controller: Any,
    robot: Any,
    pose: dict[str, float],
    before_T_link7: np.ndarray,
    target_T_link7: np.ndarray,
    side: str,
    lifetime: float,
    send_hz: float,
    execute_s: float,
    mode: str,
    delta_pose_rotation_frame: str,
    delta_pose_reference_time: float | None,
    ik_abs_joint_reference_time: float | None = None,
    ik_abs_joint_waist_height_offset_m: float = -0.3,
    ik_abs_joint_arm_state_mapping: str = "left_first",
    ik_abs_joint_num_points: int = 20,
    ik_abs_joint_max_nfev: int = 300,
    ik_abs_joint_max_delta_rad: float = 0.35,
    ik_abs_joint_urdf_zip: str = str(DEFAULT_G1_ZIP),
) -> dict[str, Any]:
    if mode == "absolute_pose":
        result = call_ee_control(controller, pose, side, lifetime, send_hz, execute_s)
        result["ee_control_mode"] = mode
        return result
    if mode == "delta_pose":
        reference_time = float(delta_pose_reference_time) if delta_pose_reference_time is not None else float(execute_s)
        reference_time = max(reference_time, 0.05)
        started = time.time()
        result = call_ee_delta_pose_trajectory_once(
            controller,
            robot,
            before_T_link7,
            target_T_link7,
            side,
            reference_time,
            delta_pose_rotation_frame,
        )
        command_duration_s = float(result.get("duration_s", 0.0))
        if result.get("ok"):
            wait_s = max(0.0, float(execute_s) - (time.time() - started))
            if wait_s > 0.0:
                time.sleep(wait_s)
        result["ee_control_mode"] = mode
        result["send_mode"] = "single_delta_pose"
        result["num_sends"] = 1
        result["command_duration_s"] = command_duration_s
        result["duration_s"] = time.time() - started
        result["execute_s"] = float(execute_s)
        result["trajectory_reference_time"] = reference_time
        return result
    if mode == "ik_abs_joint":
        reference_time = (
            float(ik_abs_joint_reference_time)
            if ik_abs_joint_reference_time is not None
            else float(execute_s)
        )
        reference_time = max(reference_time, 0.05)
        started = time.time()
        result = call_ee_ik_abs_joint_trajectory_once(
            controller,
            robot,
            target_T_link7,
            side,
            reference_time,
            ik_abs_joint_waist_height_offset_m,
            ik_abs_joint_arm_state_mapping,
            ik_abs_joint_num_points,
            ik_abs_joint_max_nfev,
            ik_abs_joint_max_delta_rad,
            ik_abs_joint_urdf_zip,
        )
        command_duration_s = float(result.get("duration_s", 0.0))
        if result.get("ok"):
            wait_s = max(0.0, float(execute_s) - (time.time() - started))
            if wait_s > 0.0:
                time.sleep(wait_s)
        result["ee_control_mode"] = mode
        result["send_mode"] = "single_ik_abs_joint_trajectory"
        result["num_sends"] = 1
        result["command_duration_s"] = command_duration_s
        result["duration_s"] = time.time() - started
        result["execute_s"] = float(execute_s)
        result["trajectory_reference_time"] = reference_time
        return result
    raise ValueError(f"unknown EE control mode {mode!r}")


def pose_error_report(current_T: np.ndarray, target_T: np.ndarray) -> dict[str, Any]:
    current_T = np.asarray(current_T, dtype=np.float64).reshape(4, 4)
    target_T = np.asarray(target_T, dtype=np.float64).reshape(4, 4)
    translation_error = target_T[:3, 3] - current_T[:3, 3]
    R_error_base = target_T[:3, :3] @ current_T[:3, :3].T
    return {
        "translation_error_m": translation_error.tolist(),
        "position_error_m": float(np.linalg.norm(translation_error)),
        "rotation_error_deg": rotation_angle_deg(R_error_base),
        "rotation_vector_base_deg": rotation_vector_from_delta(R_error_base).tolist(),
    }


def reached_pose_tolerance(
    error: dict[str, Any],
    position_tolerance_m: float,
    rotation_tolerance_deg: float,
) -> bool:
    position_ok = float(error["position_error_m"]) <= float(position_tolerance_m)
    rotation_tolerance_deg = float(rotation_tolerance_deg)
    if rotation_tolerance_deg < 0.0:
        return position_ok
    return position_ok and float(error["rotation_error_deg"]) <= rotation_tolerance_deg


def call_ee_closed_loop_control(
    controller: Any,
    robot: Any,
    state_adapter: Any,
    target_pose: dict[str, float],
    target_T_link7: np.ndarray,
    side: str,
    lifetime: float,
    send_hz: float,
    execute_s: float,
    mode: str,
    delta_pose_rotation_frame: str,
    delta_pose_reference_time: float | None,
    ik_abs_joint_reference_time: float | None,
    ik_abs_joint_waist_height_offset_m: float,
    ik_abs_joint_arm_state_mapping: str,
    ik_abs_joint_num_points: int,
    ik_abs_joint_max_nfev: int,
    ik_abs_joint_max_delta_rad: float,
    ik_abs_joint_urdf_zip: str,
    max_attempts: int,
    position_tolerance_m: float,
    rotation_tolerance_deg: float,
    settle_s: float,
    min_residual_m: float,
    stop_on_regress: bool,
    regress_patience: int,
    orientation_mode: str,
) -> dict[str, Any]:
    target_T_link7 = np.asarray(target_T_link7, dtype=np.float64).reshape(4, 4)
    attempts: list[dict[str, Any]] = []
    started = time.time()
    best_position_error = float("inf")
    regress_streak = 0
    reason = None
    reached = False
    max_attempts = max(1, int(max_attempts))

    for attempt_idx in range(max_attempts):
        before_T = state_adapter.get_T_link7_in_base()
        before_error = pose_error_report(before_T, target_T_link7)
        best_position_error = min(best_position_error, float(before_error["position_error_m"]))
        if reached_pose_tolerance(before_error, position_tolerance_m, rotation_tolerance_deg):
            reached = True
            reason = "already_within_tolerance"
            attempts.append({
                "idx": attempt_idx,
                "command_sent": False,
                "before_error": before_error,
                "after_error": before_error,
                "reason": reason,
            })
            break
        if float(before_error["position_error_m"]) < float(min_residual_m):
            reason = "residual_below_min_command"
            attempts.append({
                "idx": attempt_idx,
                "command_sent": False,
                "before_error": before_error,
                "after_error": before_error,
                "reason": reason,
            })
            break

        command_T_link7 = target_T_link7.copy()
        if orientation_mode == "current":
            command_T_link7[:3, :3] = before_T[:3, :3]
        elif orientation_mode != "target":
            raise ValueError(f"unknown closed-loop orientation mode {orientation_mode!r}; expected target/current")
        command_pose = pose_dict_from_T(command_T_link7)

        control_result = call_ee_control_mode(
            controller,
            robot,
            command_pose,
            before_T,
            command_T_link7,
            side,
            lifetime,
            send_hz,
            execute_s,
            mode,
            delta_pose_rotation_frame,
            delta_pose_reference_time,
            ik_abs_joint_reference_time,
            ik_abs_joint_waist_height_offset_m,
            ik_abs_joint_arm_state_mapping,
            ik_abs_joint_num_points,
            ik_abs_joint_max_nfev,
            ik_abs_joint_max_delta_rad,
            ik_abs_joint_urdf_zip,
        )
        if settle_s > 0.0:
            time.sleep(float(settle_s))
        after_T = state_adapter.get_T_link7_in_base()
        after_error = pose_error_report(after_T, target_T_link7)
        observed_delta = after_T[:3, 3] - before_T[:3, 3]
        target_residual = np.asarray(before_error["translation_error_m"], dtype=np.float64)
        tracking = translation_tracking_report(target_residual, observed_delta)
        attempts.append({
            "idx": attempt_idx,
            "command_sent": True,
            "before_T_link7_in_base": before_T.tolist(),
            "command_T_link7_in_base": command_T_link7.tolist(),
            "command_pose": command_pose,
            "after_T_link7_in_base": after_T.tolist(),
            "before_error": before_error,
            "after_error": after_error,
            "observed_delta_m": observed_delta.tolist(),
            "observed_delta_norm_m": float(np.linalg.norm(observed_delta)),
            "residual_translation_tracking": tracking,
            "control_result": control_result,
        })
        if not control_result.get("ok"):
            reason = "control_error"
            break
        if reached_pose_tolerance(after_error, position_tolerance_m, rotation_tolerance_deg):
            reached = True
            reason = "reached_tolerance"
            break

        current_position_error = float(after_error["position_error_m"])
        if current_position_error + EPS < best_position_error:
            best_position_error = current_position_error
            regress_streak = 0
        else:
            regress_streak += 1
        if stop_on_regress and regress_streak >= max(1, int(regress_patience)):
            reason = "position_error_not_improving"
            break

    final_T = state_adapter.get_T_link7_in_base()
    final_error = pose_error_report(final_T, target_T_link7)
    if reached_pose_tolerance(final_error, position_tolerance_m, rotation_tolerance_deg):
        reached = True
        reason = reason or "reached_tolerance"
    reason = reason or "max_attempts_exhausted"
    return {
        "enabled": True,
        "ok": bool(attempts and all((not item.get("command_sent")) or (item.get("control_result") or {}).get("ok") for item in attempts)),
        "reached": reached,
        "reason": reason,
        "num_attempts": len(attempts),
        "num_commands": sum(1 for item in attempts if item.get("command_sent")),
        "max_attempts": max_attempts,
        "position_tolerance_m": float(position_tolerance_m),
        "rotation_tolerance_deg": float(rotation_tolerance_deg),
        "min_residual_m": float(min_residual_m),
        "orientation_mode": orientation_mode,
        "stop_on_regress": bool(stop_on_regress),
        "regress_patience": int(regress_patience),
        "duration_s": time.time() - started,
        "final_T_link7_in_base": final_T.tolist(),
        "final_position_error_m": final_error["position_error_m"],
        "final_rotation_error_deg": final_error["rotation_error_deg"],
        "final_error": final_error,
        "attempts": attempts,
    }


def split_state_result(value: Any) -> tuple[Any, Any]:
    if isinstance(value, tuple) and len(value) == 2:
        return value[0], value[1]
    return value, None


def coerce_float_list(value: Any) -> list[float]:
    data, _timestamp = split_state_result(value)
    if isinstance(data, str):
        data = ast.literal_eval(data)
    if isinstance(data, (int, float)):
        return [float(data)]
    if data is None:
        raise ValueError("gripper state is None")
    values = list(data)
    if not values:
        raise ValueError("gripper state is empty")
    if any(v is None for v in values):
        raise ValueError(f"gripper state contains None values: {values!r}")
    return [float(v) for v in values]


def gripper_index(side: str, num_values: int) -> int:
    if num_values <= 0:
        raise ValueError("cannot select gripper index from empty values")
    if side == "left":
        return 0
    return min(1, num_values - 1)


def state_to_gripper_command_value(value: float, state_max_raw: float = 120.0) -> float:
    value = float(value)
    state_max_raw = float(state_max_raw)
    if state_max_raw > 1.0 and abs(value) > 1.0:
        return value / state_max_raw
    return value


def gripper_command_to_state_estimate(value: float, state_max_raw: float = 120.0) -> float:
    state_max_raw = float(state_max_raw)
    if state_max_raw > 1.0:
        return float(value) * state_max_raw
    return float(value)


def read_gripper_state(robot: Any, side: str) -> dict[str, Any]:
    raw_result = robot.gripper_states()
    data, timestamp = split_state_result(raw_result)
    values = coerce_float_list(raw_result)
    idx = gripper_index(side, len(values))
    return {
        "raw_result": json_safe(raw_result),
        "data": json_safe(data),
        "timestamp": json_safe(timestamp),
        "values": values,
        "side": side,
        "selected_index": idx,
        "selected_raw": values[idx],
    }


def gripper_model_command(step_preview: dict[str, Any]) -> float:
    if "gripper_humanego_0_open_1_closed" in step_preview:
        return float(step_preview["gripper_humanego_0_open_1_closed"])
    return float(step_preview["gripper_g1_raw_0_open_120_closed"]) / 120.0


def select_gripper_command(step_preview: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    source = args.gripper_source
    if source == "model":
        value = gripper_model_command(step_preview)
    elif source == "hold":
        value = None
    elif source == "open":
        value = 0.0
    elif source == "closed":
        value = 1.0
    elif source == "manual":
        if args.gripper_target is None:
            raise ValueError("--gripper-target is required when --gripper-source manual")
        value = float(args.gripper_target)
    else:
        raise ValueError(f"unknown gripper source: {source}")

    clipped = False
    raw_value = value
    if value is not None:
        clipped_value = float(np.clip(value, args.gripper_min, args.gripper_max))
        clipped = abs(clipped_value - value) > EPS
        value = clipped_value

    return {
        "source": source,
        "raw_command_0_open_1_closed": raw_value,
        "command_0_open_1_closed": value,
        "clipped": clipped,
        "min": float(args.gripper_min),
        "max": float(args.gripper_max),
        "state_estimate_0_open_120_closed": None if value is None else float(value) * 120.0,
    }


def call_gripper_control_once(robot: Any, side: str, command: float | None) -> dict[str, Any]:
    before = read_gripper_state(robot, side)
    before_values = [float(v) for v in before["values"]]
    command_values = [state_to_gripper_command_value(v) for v in before_values]
    idx = gripper_index(side, len(command_values))
    if command is None:
        command_values[idx] = state_to_gripper_command_value(before["selected_raw"])
    else:
        command_values[idx] = float(command)
    payload: Any = command_values[0] if len(command_values) == 1 else command_values
    started = time.time()
    try:
        result = robot.move_gripper(payload)
        return {
            "ok": True,
            "duration_s": time.time() - started,
            "before": before,
            "target_command_values_0_open_1_closed": command_values,
            "target_state_estimate_values_raw": [
                gripper_command_to_state_estimate(v) for v in command_values
            ],
            "selected_index": idx,
            "payload": json_safe(payload),
            "result": json_safe(result),
        }
    except Exception as exc:
        return {
            "ok": False,
            "duration_s": time.time() - started,
            "before": before,
            "payload": json_safe(payload),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def prompt_operator() -> str:
    try:
        text = input("[Enter]=execute one step, s=skip/replan, q=quit > ")
    except EOFError:
        return "q"
    return text.strip().lower()


def choose_operator(control_mode: str) -> str:
    if control_mode == "auto":
        return "auto"
    return prompt_operator()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default=str(DEFAULT_CFG))
    parser.add_argument("--server-url", default="http://111.0.22.33:30003/infer")
    parser.add_argument("--out-dir", default=str(artifact_dir("interactive")))
    parser.add_argument("--tag", default="interactive_step")
    parser.add_argument("--side", choices=["right", "left"], default="right")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--control-mode", choices=["prompt", "auto"], default="prompt")
    parser.add_argument("--tracking-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tracking-min-ratio", type=float, default=0.30)
    parser.add_argument("--tracking-min-cos", type=float, default=0.50)
    parser.add_argument("--tracking-min-target-m", type=float, default=0.01)
    parser.add_argument("--tracking-bad-steps", type=int, default=2)
    parser.add_argument("--target-source", choices=["position_keep_orientation", "limited", "raw"], default="position_keep_orientation")
    parser.add_argument("--approach-object-key", default="obj1")
    parser.add_argument("--object-lock", choices=["none", "base_after_first"], default="none")
    parser.add_argument("--object-lock-require-clean", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--target-z-bias-m", type=float, default=0.0)
    parser.add_argument("--max-orientation-deg", type=float, default=10.0)
    parser.add_argument("--probe-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--probe-deg", type=float, default=10.0)
    parser.add_argument("--probe-frame", choices=["local", "base"], default="local")
    parser.add_argument("--confirm-control", default="")
    parser.add_argument("--lifetime", type=float, default=0.5)
    parser.add_argument("--send-hz", type=float, default=10.0)
    parser.add_argument("--execute-s", type=float, default=1.0)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--ee-control-mode", choices=["absolute_pose", "delta_pose", "ik_abs_joint"], default="absolute_pose")
    parser.add_argument("--delta-pose-rotation-frame", choices=["base", "local"], default="base")
    parser.add_argument("--delta-pose-reference-time", type=float, default=None)
    parser.add_argument("--ik-abs-joint-reference-time", type=float, default=None)
    parser.add_argument("--ik-abs-joint-waist-height-offset-m", type=float, default=-0.3)
    parser.add_argument("--ik-abs-joint-arm-state-mapping", choices=["left_first", "right_first"], default="left_first")
    parser.add_argument("--ik-abs-joint-num-points", type=int, default=20)
    parser.add_argument("--ik-abs-joint-max-nfev", type=int, default=300)
    parser.add_argument("--ik-abs-joint-max-delta-rad", type=float, default=0.35)
    parser.add_argument("--ik-abs-joint-urdf-zip", default=str(DEFAULT_G1_ZIP))
    parser.add_argument("--closed-loop-ee", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--closed-loop-max-attempts", type=int, default=5)
    parser.add_argument("--closed-loop-position-tolerance-m", type=float, default=0.01)
    parser.add_argument("--closed-loop-rotation-tolerance-deg", type=float, default=-1.0)
    parser.add_argument("--closed-loop-execute-s", type=float, default=None)
    parser.add_argument("--closed-loop-settle-s", type=float, default=0.10)
    parser.add_argument("--closed-loop-min-residual-m", type=float, default=0.002)
    parser.add_argument("--closed-loop-orientation-mode", choices=["target", "current"], default="target")
    parser.add_argument("--closed-loop-stop-on-regress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--closed-loop-regress-patience", type=int, default=2)
    parser.add_argument("--execute-gripper", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gripper-source", choices=["model", "hold", "open", "closed", "manual"], default="model")
    parser.add_argument("--gripper-target", type=float, default=None)
    parser.add_argument("--gripper-min", type=float, default=0.0)
    parser.add_argument("--gripper-max", type=float, default=1.0)
    parser.add_argument("--gripper-settle-s", type=float, default=0.5)
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
    out_base = Path(args.out_dir).expanduser().resolve()
    default_base = artifact_dir("interactive")
    if out_base == default_base:
        run_dir = artifact_run_dir("interactive", args.tag, prefix="interactive")
    else:
        run_dir = out_base / f"g1_humanego_interactive_{utc_stamp()}_{args.tag}"
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
    locked_objects: dict[str, Any] | None = None
    object_lock_blocked_reason = None
    step_summaries: list[dict[str, Any]] = []
    step_summaries_path = run_dir / "step_summaries.json"
    step_summaries_jsonl_path = run_dir / "step_summaries.jsonl"
    tracking_bad_streak = 0
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
            payload_locked_objects = locked_objects if args.object_lock == "base_after_first" and locked_objects else None
            payload = build_payload(frame, state, args, request_id, locked_objects=payload_locked_objects)
            if payload_locked_objects:
                log(
                    f"step {idx}: using locked base-frame objects "
                    f"{sorted(payload_locked_objects.keys())}; server RGB-D segmentation bypassed"
                )

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
            warnings = object_quality_warnings(response)
            if args.object_lock == "base_after_first" and locked_objects is None:
                if warnings and args.object_lock_require_clean:
                    object_lock_blocked_reason = {
                        "request_id": request_id,
                        "reason": "vision warnings in first RGB-D detection",
                        "warnings": warnings,
                    }
                    log(f"step {idx}: object lock blocked by vision warnings: {warnings}")
                else:
                    locked_objects = lock_objects_in_base(response, state, request_id)
                    log(f"step {idx}: locked static base-frame objects: {sorted(locked_objects.keys())}")

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
                args.target_z_bias_m,
                args.max_orientation_deg,
                args.probe_axis,
                args.probe_deg,
                args.probe_frame,
            )
            target_delta = position_from_pose_dict(target_pose) - before_T_link7[:3, 3]
            target_delta_norm = float(np.linalg.norm(target_delta))
            raw_target_T_link7 = T_from_pose_dict(raw_target_pose)
            target_T_link7 = T_from_pose_dict(target_pose)
            target_rotation_delta_deg = rotation_angle_deg(target_T_link7[:3, :3] @ before_T_link7[:3, :3].T)
            object_base = object_position_in_base(response, state, args.approach_object_key)
            approach_metrics = None
            current_alignment = None
            raw_alignment = None
            target_alignment = None
            raw_alignment_improvement = None
            target_alignment_improvement = None
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
                T_tcp_in_link7 = np.asarray(state["T_tcp_in_link7"], dtype=np.float64)
                current_alignment = axis_alignment_to_object(before_T_link7, T_tcp_in_link7, object_base)
                raw_alignment = axis_alignment_to_object(raw_target_T_link7, T_tcp_in_link7, object_base)
                target_alignment = axis_alignment_to_object(target_T_link7, T_tcp_in_link7, object_base)
                raw_alignment_improvement = alignment_improvement(current_alignment, raw_alignment)
                target_alignment_improvement = alignment_improvement(current_alignment, target_alignment)
            gripper_command = select_gripper_command(step_preview, args)

            print("\n=== HumanEgo proposed step ===")
            print(f"step: {idx}")
            print(f"done_prob: {response['policy_preview']['done_prob']:.3f}")
            print(f"object_source: {response.get('input_summary', {}).get('object_source_used')}")
            if args.object_lock != "none":
                lock_state = "active" if locked_objects else "pending"
                if object_lock_blocked_reason:
                    lock_state = f"blocked: {object_lock_blocked_reason.get('warnings')}"
                print(f"object_lock: {args.object_lock} ({lock_state})")
                if warnings:
                    print(f"vision_warnings: {warnings}")
            object_error = response.get("input_summary", {}).get("object_error")
            if object_error:
                print(f"object_error: {object_error.get('error_type')}: {object_error.get('error')}")
            print(f"target_source: {args.target_source}")
            print(f"target_adapter: {args.target_adapter}")
            print(f"ee_control_mode: {args.ee_control_mode}")
            ik_preview = None
            if args.ee_control_mode == "delta_pose":
                delta_preview = link7_delta_pose_command(
                    before_T_link7,
                    target_T_link7,
                    rotation_frame=args.delta_pose_rotation_frame,
                )
                print(
                    "delta_pose command: "
                    f"action_data={delta_preview['action_data']} "
                    f"rotvec_deg={delta_preview['rotation_vector_deg']} "
                    f"frame={delta_preview['rotation_frame']}"
                )
            if args.ee_control_mode == "ik_abs_joint":
                ik_preview = call_ee_ik_abs_joint_trajectory_once(
                    arm.controller,
                    arm.robot,
                    target_T_link7,
                    args.side,
                    args.ik_abs_joint_reference_time if args.ik_abs_joint_reference_time is not None else args.execute_s,
                    args.ik_abs_joint_waist_height_offset_m,
                    args.ik_abs_joint_arm_state_mapping,
                    args.ik_abs_joint_num_points,
                    args.ik_abs_joint_max_nfev,
                    args.ik_abs_joint_max_delta_rad,
                    args.ik_abs_joint_urdf_zip,
                    execute_control=False,
                )
                print(
                    "ik_abs_joint preview: "
                    f"ok={ik_preview.get('ok')} "
                    f"blocked={ik_preview.get('blocked_reason')} "
                    f"selected={((ik_preview.get('ik_selected_candidate') or {}).get('name'))} "
                    f"reason={ik_preview.get('ik_selected_reason')} "
                    f"acceptance={ik_preview.get('ik_pose_acceptance')} "
                    f"candidates={ik_preview.get('ik_candidate_count')} "
                    f"safe={ik_preview.get('ik_safe_candidate_count')} "
                    f"strict_safe={ik_preview.get('ik_strict_safe_candidate_count')} "
                    f"relaxed_safe={ik_preview.get('ik_relaxed_safe_candidate_count')} "
                    f"q_delta_abs_max_rad={ik_preview.get('q_delta_abs_max_rad')} "
                    f"q_delta_norm_rad={ik_preview.get('q_delta_norm_rad')} "
                    f"ik_pos_err={((ik_preview.get('ik_fk_vs_target_error') or {}).get('position_error_m'))} "
                    f"ik_rot_err={((ik_preview.get('ik_fk_vs_target_error') or {}).get('rotation_error_deg'))}"
                )
                if not ik_preview.get("ok"):
                    print(
                        "ik_abs_joint safety stop: "
                        f"{ik_preview.get('blocked_reason') or ik_preview.get('error_type') or 'unknown'}"
                    )
            if abs(float(args.target_z_bias_m)) > EPS:
                print(f"target_z_bias_m: {args.target_z_bias_m:+.4f}")
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
            if gripper_command["command_0_open_1_closed"] is None:
                gripper_text = "hold current"
            else:
                gripper_text = (
                    f"{gripper_command['command_0_open_1_closed']:.3f} "
                    f"(~{gripper_command['state_estimate_0_open_120_closed']:.1f}/120)"
                )
            print(
                "gripper target: "
                f"{gripper_text}, source={gripper_command['source']}, "
                f"execute={args.execute_gripper}"
            )
            if approach_metrics:
                print(
                    f"distance link7->{args.approach_object_key}: "
                    f"before={approach_metrics['before_link7_to_object_m']:.4f}m, "
                    f"target={approach_metrics['target_link7_to_object_m']:.4f}m, "
                    f"delta={approach_metrics['target_minus_before_m']:+.4f}m"
                )
                print(compact_alignment("tcp axes current", current_alignment))
                print(compact_alignment("tcp axes raw_model", raw_alignment, raw_alignment_improvement))
                print(compact_alignment("tcp axes target", target_alignment, target_alignment_improvement))
            print(f"{args.side}_pose: {compact_pose(target_pose)}")

            step_record: Dict[str, Any] = {
                "idx": idx,
                "request_id": request_id,
                "server_ok": bool(response.get("ok")),
                "target_source": args.target_source,
                "target_adapter": args.target_adapter,
                "ee_control_mode": args.ee_control_mode,
                "raw_target_pose": raw_target_pose,
                "target_pose": target_pose,
                "before_T_link7_in_base": before_T_link7.tolist(),
                "raw_target_T_link7_in_base": raw_target_T_link7.tolist(),
                "target_T_link7_in_base": target_T_link7.tolist(),
                "target_adapter_info": adapter_info,
                "target_delta_m": target_delta.tolist(),
                "target_delta_norm_m": target_delta_norm,
                "target_rotation_delta_deg": target_rotation_delta_deg,
                "gripper_command": gripper_command,
                "object_lock_mode": args.object_lock,
                "object_lock_active": bool(payload_locked_objects),
                "object_lock_summary": locked_object_summary(locked_objects),
                "object_lock_blocked_reason": object_lock_blocked_reason,
                "vision_warnings": warnings,
                "approach_metrics": approach_metrics,
                "axis_alignment": {
                    "current": current_alignment,
                    "raw_model": raw_alignment,
                    "target": target_alignment,
                    "raw_model_improvement": raw_alignment_improvement,
                    "target_improvement": target_alignment_improvement,
                },
                "operator_input": None,
                "ik_abs_joint_preview": ik_preview,
                "server_response": response,
            }

            if ik_preview is not None and not ik_preview.get("ok"):
                reason = ik_preview.get("blocked_reason") or ik_preview.get("error_type") or "ik_abs_joint_preview_failed"
                log(f"step {idx}: IK ABS_JOINT safety stop before prompt: {reason}")
                step_record["executed"] = False
                step_record["blocked_reason"] = f"ik_abs_joint_preview_{reason}"
                step_record["control_result"] = ik_preview
                report["stopped_by"] = {
                    "type": "ik_abs_joint_safety_preview",
                    "step": idx,
                    "reason": reason,
                    "q_delta_abs_max_rad": ik_preview.get("q_delta_abs_max_rad"),
                    "max_joint_delta_rad": args.ik_abs_joint_max_delta_rad,
                }
                append_step_outputs(step_record, response, report, step_summaries, step_summaries_path, step_summaries_jsonl_path, step_dir)
                break

            operator = choose_operator(args.control_mode)
            step_record["operator_input"] = operator
            if operator == "auto":
                log(f"step {idx}: auto control mode executing without prompt")

            if operator == "q":
                log("operator requested quit")
                step_record["executed"] = False
                append_step_outputs(step_record, response, report, step_summaries, step_summaries_path, step_summaries_jsonl_path, step_dir)
                break
            if operator == "s":
                log("operator skipped this target")
                step_record["executed"] = False
                append_step_outputs(step_record, response, report, step_summaries, step_summaries_path, step_summaries_jsonl_path, step_dir)
                continue
            if args.confirm_control != "RUN_CONTROL":
                log("refusing to execute because --confirm-control RUN_CONTROL is missing")
                step_record["executed"] = False
                step_record["blocked_reason"] = "missing RUN_CONTROL confirmation"
                append_step_outputs(step_record, response, report, step_summaries, step_summaries_path, step_summaries_jsonl_path, step_dir)
                break

            if args.closed_loop_ee:
                closed_loop_execute_s = (
                    float(args.closed_loop_execute_s)
                    if args.closed_loop_execute_s is not None
                    else float(args.execute_s)
                )
                log(
                    f"step {idx}: executing EE target with closed-loop residual tracking "
                    f"max_attempts={args.closed_loop_max_attempts} "
                    f"pos_tol={args.closed_loop_position_tolerance_m:.4f}m "
                    f"rot_tol={args.closed_loop_rotation_tolerance_deg:.2f}deg"
                )
                closed_loop_result = call_ee_closed_loop_control(
                    arm.controller,
                    arm.robot,
                    arm,
                    target_pose,
                    target_T_link7,
                    args.side,
                    args.lifetime,
                    args.send_hz,
                    closed_loop_execute_s,
                    args.ee_control_mode,
                    args.delta_pose_rotation_frame,
                    args.delta_pose_reference_time,
                    args.ik_abs_joint_reference_time,
                    args.ik_abs_joint_waist_height_offset_m,
                    args.ik_abs_joint_arm_state_mapping,
                    args.ik_abs_joint_num_points,
                    args.ik_abs_joint_max_nfev,
                    args.ik_abs_joint_max_delta_rad,
                    args.ik_abs_joint_urdf_zip,
                    args.closed_loop_max_attempts,
                    args.closed_loop_position_tolerance_m,
                    args.closed_loop_rotation_tolerance_deg,
                    args.closed_loop_settle_s,
                    args.closed_loop_min_residual_m,
                    args.closed_loop_stop_on_regress,
                    args.closed_loop_regress_patience,
                    args.closed_loop_orientation_mode,
                )
                control_attempts = [
                    item for item in closed_loop_result.get("attempts", []) if item.get("command_sent")
                ]
                if control_attempts:
                    control_result = control_attempts[-1].get("control_result") or {}
                else:
                    control_result = {
                        "ok": bool(closed_loop_result.get("reached")),
                        "mode": args.ee_control_mode,
                        "send_mode": "closed_loop_no_command",
                    }
                control_result = dict(control_result)
                control_result["closed_loop"] = {
                    key: value
                    for key, value in closed_loop_result.items()
                    if key != "attempts"
                }
                step_record["closed_loop_result"] = closed_loop_result
                log(
                    f"step {idx}: closed-loop reached={closed_loop_result['reached']} "
                    f"reason={closed_loop_result['reason']} "
                    f"attempts={closed_loop_result['num_attempts']} "
                    f"commands={closed_loop_result['num_commands']} "
                    f"final_pos_err={closed_loop_result['final_position_error_m']:.4f}m "
                    f"final_rot_err={closed_loop_result['final_rotation_error_deg']:.2f}deg"
                )
            else:
                log(f"step {idx}: executing one EE target")
                control_result = call_ee_control_mode(
                    arm.controller,
                    arm.robot,
                    target_pose,
                    before_T_link7,
                    target_T_link7,
                    args.side,
                    args.lifetime,
                    args.send_hz,
                    args.execute_s,
                    args.ee_control_mode,
                    args.delta_pose_rotation_frame,
                    args.delta_pose_reference_time,
                    args.ik_abs_joint_reference_time,
                    args.ik_abs_joint_waist_height_offset_m,
                    args.ik_abs_joint_arm_state_mapping,
                    args.ik_abs_joint_num_points,
                    args.ik_abs_joint_max_nfev,
                    args.ik_abs_joint_max_delta_rad,
                    args.ik_abs_joint_urdf_zip,
                )
            report["control_sent"] = bool(report.get("control_sent")) or bool(control_result.get("ok"))
            step_record["executed"] = bool(control_result.get("ok"))
            step_record["control_result"] = control_result
            if not control_result.get("ok"):
                reason = control_result.get("blocked_reason") or control_result.get("error_type") or "control_failed"
                log(f"step {idx}: control safety stop before motion bookkeeping: {reason}")
                step_record["blocked_reason"] = reason
                report["stopped_by"] = {
                    "type": "control_safety",
                    "step": idx,
                    "mode": args.ee_control_mode,
                    "reason": reason,
                    "q_delta_abs_max_rad": control_result.get("q_delta_abs_max_rad"),
                    "max_joint_delta_rad": args.ik_abs_joint_max_delta_rad if args.ee_control_mode == "ik_abs_joint" else None,
                }
                append_step_outputs(step_record, response, report, step_summaries, step_summaries_path, step_summaries_jsonl_path, step_dir)
                break

            post_ee_T_link7 = arm.get_T_link7_in_base()
            post_ee_delta = post_ee_T_link7[:3, 3] - before_T_link7[:3, 3]
            step_record["post_ee_T_link7_in_base"] = post_ee_T_link7.tolist()
            step_record["post_ee_delta_m"] = post_ee_delta.tolist()
            step_record["post_ee_delta_norm_m"] = float(np.linalg.norm(post_ee_delta))
            step_record["post_ee_rotation_delta_deg"] = rotation_angle_deg(
                post_ee_T_link7[:3, :3] @ before_T_link7[:3, :3].T
            )
            step_record["post_ee_target_pose_error"] = pose_error_report(post_ee_T_link7, target_T_link7)
            step_record["post_ee_translation_tracking"] = translation_tracking_report(target_delta, post_ee_delta)
            if object_base is not None:
                post_ee_dist = float(np.linalg.norm(post_ee_T_link7[:3, 3] - object_base))
                step_record["post_ee_approach_metrics"] = {
                    "object_key": args.approach_object_key,
                    "before_link7_to_object_m": approach_metrics["before_link7_to_object_m"] if approach_metrics else None,
                    "post_ee_link7_to_object_m": post_ee_dist,
                    "post_ee_minus_before_m": post_ee_dist - (approach_metrics["before_link7_to_object_m"] if approach_metrics else post_ee_dist),
                    "closer": bool(approach_metrics and post_ee_dist < approach_metrics["before_link7_to_object_m"]),
                }
            log(
                f"step {idx}: post_ee_delta={post_ee_delta.tolist()} "
                f"norm={np.linalg.norm(post_ee_delta):.4f} "
                f"rot_deg={step_record['post_ee_rotation_delta_deg']:.2f} "
                f"cos_to_target={step_record['post_ee_translation_tracking'].get('cosine_to_target_delta')} "
                f"target_err={step_record['post_ee_target_pose_error']['position_error_m']:.4f}m/"
                f"{step_record['post_ee_target_pose_error']['rotation_error_deg']:.2f}deg"
            )

            gripper_result = None
            if args.execute_gripper:
                log(
                    f"step {idx}: executing gripper target "
                    f"{gripper_command['command_0_open_1_closed']}"
                )
                gripper_result = call_gripper_control_once(
                    arm.robot,
                    args.side,
                    gripper_command["command_0_open_1_closed"],
                )
                time.sleep(max(0.0, float(args.gripper_settle_s)))
                gripper_result["after"] = read_gripper_state(arm.robot, args.side)
                before_raw = float(gripper_result["before"]["selected_raw"])
                after_raw = float(gripper_result["after"]["selected_raw"])
                gripper_result["observed_delta_raw"] = after_raw - before_raw
                before_values = [float(v) for v in gripper_result["before"]["values"]]
                after_values = [float(v) for v in gripper_result["after"]["values"]]
                gripper_result["observed_delta_values_raw"] = [
                    after_v - before_v for before_v, after_v in zip(before_values, after_values)
                ]
                gripper_result["after_command_values_0_open_1_closed"] = [
                    state_to_gripper_command_value(v) for v in after_values
                ]
                post_gripper_T_link7 = arm.get_T_link7_in_base()
                post_gripper_delta = post_gripper_T_link7[:3, 3] - before_T_link7[:3, 3]
                gripper_result["post_gripper_T_link7_in_base"] = post_gripper_T_link7.tolist()
                gripper_result["post_gripper_arm_delta_m"] = post_gripper_delta.tolist()
                gripper_result["post_gripper_arm_delta_norm_m"] = float(np.linalg.norm(post_gripper_delta))
                gripper_result["post_gripper_arm_rotation_delta_deg"] = rotation_angle_deg(
                    post_gripper_T_link7[:3, :3] @ before_T_link7[:3, :3].T
                )
                gripper_result["post_gripper_translation_tracking"] = translation_tracking_report(
                    target_delta,
                    post_gripper_delta,
                )
                log(
                    f"step {idx}: gripper before={before_raw:.4f} "
                    f"after={after_raw:.4f} delta={after_raw - before_raw:+.4f}"
                )
                log(
                    f"step {idx}: gripper all_before={before_values} "
                    f"all_after={after_values} "
                    f"all_delta={gripper_result['observed_delta_values_raw']}"
                )
                log(
                    f"step {idx}: post_gripper_arm_delta={post_gripper_delta.tolist()} "
                    f"norm={np.linalg.norm(post_gripper_delta):.4f} "
                    f"rot_deg={gripper_result['post_gripper_arm_rotation_delta_deg']:.2f} "
                    f"cos_to_target={gripper_result['post_gripper_translation_tracking'].get('cosine_to_target_delta')}"
                )
            else:
                gripper_result = {
                    "executed": False,
                    "reason": "--execute-gripper not set",
                    "before": read_gripper_state(arm.robot, args.side),
                }
            step_record["gripper_result"] = gripper_result

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
            step_record["settled_translation_tracking"] = translation_tracking_report(target_delta, observed_delta)
            step_record["observed_rotation_delta_deg"] = rotation_angle_deg(
                after_T_link7[:3, :3] @ before_T_link7[:3, :3].T
            )
            step_record["settled_target_pose_error"] = pose_error_report(after_T_link7, target_T_link7)
            target_norm_for_gate = float(step_record["settled_translation_tracking"].get("target_norm_m") or 0.0)
            observed_norm_for_gate = float(step_record["settled_translation_tracking"].get("observed_norm_m") or 0.0)
            tracking_ratio = observed_norm_for_gate / target_norm_for_gate if target_norm_for_gate > EPS else None
            tracking_cos = step_record["settled_translation_tracking"].get("cosine_to_target_delta")
            tracking_bad = False
            tracking_reason = None
            if args.tracking_gate and target_norm_for_gate >= float(args.tracking_min_target_m):
                ratio_bad = tracking_ratio is not None and tracking_ratio < float(args.tracking_min_ratio)
                cos_bad = tracking_cos is not None and tracking_cos < float(args.tracking_min_cos)
                tracking_bad = bool(ratio_bad or cos_bad)
                if tracking_bad:
                    reasons = []
                    if ratio_bad:
                        reasons.append(f"ratio {tracking_ratio:.3f} < {args.tracking_min_ratio:.3f}")
                    if cos_bad:
                        reasons.append(f"cos {tracking_cos:.3f} < {args.tracking_min_cos:.3f}")
                    tracking_reason = "; ".join(reasons)
                    tracking_bad_streak += 1
                else:
                    tracking_bad_streak = 0
            step_record["tracking_gate"] = {
                "enabled": bool(args.tracking_gate),
                "ratio": tracking_ratio,
                "cosine": tracking_cos,
                "bad": tracking_bad,
                "bad_streak": tracking_bad_streak,
                "reason": tracking_reason,
                "min_ratio": float(args.tracking_min_ratio),
                "min_cos": float(args.tracking_min_cos),
                "min_target_m": float(args.tracking_min_target_m),
                "bad_steps_to_stop": int(args.tracking_bad_steps),
            }
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
                f"rot_deg={step_record['observed_rotation_delta_deg']:.2f} "
                f"target_err={step_record['settled_target_pose_error']['position_error_m']:.4f}m/"
                f"{step_record['settled_target_pose_error']['rotation_error_deg']:.2f}deg"
            )
            if object_base is not None:
                log(
                    f"step {idx}: observed link7->{args.approach_object_key} "
                    f"distance {step_record['observed_approach_metrics']['after_link7_to_object_m']:.4f}m "
                    f"delta={step_record['observed_approach_metrics']['after_minus_before_m']:+.4f}m"
                )
            report["steps"].append(step_record)
            summary = compact_step_summary(step_record, response)
            step_summaries.append(summary)
            step_summaries_path.write_text(
                json.dumps(json_safe(step_summaries), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            with step_summaries_jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(json_safe(summary), ensure_ascii=False) + "\n")
            (step_dir / "step_record.json").write_text(
                json.dumps(json_safe(step_record), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if args.tracking_gate and tracking_bad_streak >= max(1, int(args.tracking_bad_steps)):
                stop_reason = {
                    "type": "tracking_gate",
                    "step": idx,
                    "bad_streak": tracking_bad_streak,
                    "reason": tracking_reason,
                    "ratio": tracking_ratio,
                    "cosine": tracking_cos,
                    "target_delta_norm_m": target_norm_for_gate,
                    "settled_delta_norm_m": observed_norm_for_gate,
                }
                report["stopped_by"] = stop_reason
                log(f"tracking gate stopping run: {stop_reason}")
                break

        report["ok"] = True
    except KeyboardInterrupt:
        report.update(
            {
                "ok": False,
                "interrupted": True,
                "error_type": "KeyboardInterrupt",
                "error": "Interrupted by operator",
                "traceback": traceback.format_exc(),
            }
        )
    except Exception as exc:
        log(f"ERROR: {type(exc).__name__}: {exc}")
        traceback.print_exc()
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
    step_summaries_path.write_text(
        json.dumps(json_safe(step_summaries), ensure_ascii=False, indent=2),
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

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "zip_path": str(zip_path),
                "step_summaries_path": str(step_summaries_path),
                "step_summaries_jsonl_path": str(step_summaries_jsonl_path),
                "upload": upload,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
