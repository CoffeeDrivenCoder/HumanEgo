#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate URDF IK targets through G1 SDK ABS_JOINT control.

This probe closes the chain:
  desired link7 pose -> URDF IK -> 7 arm joints -> SDK ABS_JOINT -> SDK link7 pose.
It supports left and right arms and requires RUN_CONTROL before sending motion.
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
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "inference", PROJECT_ROOT / "scripts"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from G1RobotArm import parse_motion_pose, wait_motion_status  # noqa: E402
from g1_artifacts import artifact_dir, run_dir as artifact_run_dir  # noqa: E402
from g1_humanego_client_dry_run import json_safe, log, upload_zip  # noqa: E402
from g1_humanego_interactive_step_client import read_robot_joint_states_for_trajectory  # noqa: E402
from g1_urdf_ik import DEFAULT_G1_ZIP, G1UrdfKinematics, pose_error, rotation_angle_deg  # noqa: E402


EPS = 1e-12


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def make_zip(src_dir: Path) -> Path:
    zip_path = src_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir.parent))
    return zip_path


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


def motion_T_for_side(status: dict[str, Any], side: str) -> np.ndarray:
    frames = status.get("frames") or {}
    frame_name = "arm_left_link7" if side == "left" else "arm_right_link7"
    if frame_name not in frames:
        raise KeyError(f"{frame_name} missing from motion_status frames: {sorted(frames)}")
    return parse_motion_pose(frames[frame_name])


def current_motion_T(controller: Any, side: str, tries: int, sleep_s: float) -> np.ndarray:
    status = wait_motion_status(controller, tries=tries, sleep_s=sleep_s)
    if not isinstance(status, dict):
        raise RuntimeError(f"get_motion_status did not return a dict: {status!r}")
    return motion_T_for_side(status, side)


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
    raise ValueError(f"unknown axis {axis!r}")


def target_from_delta(before_T: np.ndarray, delta_axis: str, delta_m: float, rotation_axis: str, rotation_deg: float) -> np.ndarray:
    target_T = np.asarray(before_T, dtype=np.float64).reshape(4, 4).copy()
    target_T[{"x": 0, "y": 1, "z": 2}[delta_axis.lower()], 3] += float(delta_m)
    if abs(float(rotation_deg)) > EPS:
        target_T[:3, :3] = R_axis(rotation_axis, rotation_deg) @ target_T[:3, :3]
    return target_T


def interpolate_q(q_start: np.ndarray, q_target: np.ndarray, num_points: int) -> list[np.ndarray]:
    q_start = np.asarray(q_start, dtype=np.float64).reshape(7)
    q_target = np.asarray(q_target, dtype=np.float64).reshape(7)
    count = max(1, int(num_points))
    return [q_start + (q_target - q_start) * (idx / count) for idx in range(1, count + 1)]


def call_abs_joint_trajectory(
    controller: Any,
    robot_states: dict[str, Any],
    full_arm_rows: list[list[float]],
    mapping: str,
    reference_time: float,
) -> dict[str, Any]:
    robot_actions = []
    for full_arm in full_arm_rows:
        left_q, right_q = split_left_right_from_full_arm(full_arm, mapping)
        robot_actions.append(
            {
                "left_arm": {"action_data": left_q, "control_type": "ABS_JOINT"},
                "right_arm": {"action_data": right_q, "control_type": "ABS_JOINT"},
            }
        )

    kwargs = {
        "infer_timestamp": int(time.time() * 1e9),
        "robot_states": {
            "head": robot_states["head"],
            "waist": robot_states["waist"],
            "arm": robot_states["arm"],
        },
        "robot_actions": robot_actions,
        "robot_link": "base_link",
        "trajectory_reference_time": float(reference_time),
    }
    started = time.time()
    try:
        result = controller.trajectory_tracking_control(**kwargs)
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


def vector_norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=np.float64)))


def validate_one_side(
    *,
    args: argparse.Namespace,
    kin: G1UrdfKinematics,
    robot: Any,
    controller: Any,
    side: str,
) -> dict[str, Any]:
    joint_states_before = read_robot_joint_states_for_trajectory(robot)
    waist_for_urdf = waist_values_with_height_offset(joint_states_before["waist"], args.waist_height_offset_m)
    q_before, arm_indices = side_q_from_arm_state(joint_states_before["arm"], side, args.arm_state_mapping)
    sdk_before_T = current_motion_T(controller, side, args.motion_tries, args.motion_sleep_s)
    urdf_before_T = kin.link7_fk(side, q_before, waist_states=waist_for_urdf)
    target_T = target_from_delta(
        sdk_before_T,
        args.delta_axis,
        args.delta_m,
        args.rotation_axis,
        args.rotation_deg,
    )
    ik = kin.solve_link7_ik(side, target_T, q_before, waist_states=waist_for_urdf, max_nfev=args.max_nfev)
    q_target = ik.q_solution
    q_delta = q_target - q_before
    q_delta_abs_max = float(np.max(np.abs(q_delta)))
    target_fk_T = kin.link7_fk(side, q_target, waist_states=waist_for_urdf)
    target_fk_error = pose_error(target_fk_T, target_T)
    commanded_translation = target_T[:3, 3] - sdk_before_T[:3, 3]
    commanded_rotation_deg = rotation_angle_deg(target_T[:3, :3] @ sdk_before_T[:3, :3].T)

    item: dict[str, Any] = {
        "side": side,
        "control_sent": False,
        "arm_state_mapping": args.arm_state_mapping,
        "arm_state_indices": arm_indices,
        "joint_states_before": json_safe(joint_states_before),
        "waist_values_for_urdf": waist_for_urdf,
        "q_before": q_before.tolist(),
        "sdk_before_T_link7_in_base": sdk_before_T.tolist(),
        "urdf_before_T_link7_in_base": urdf_before_T.tolist(),
        "urdf_before_vs_sdk_error": pose_error(urdf_before_T, sdk_before_T),
        "target_T_link7_in_base": target_T.tolist(),
        "commanded_delta_m": commanded_translation.tolist(),
        "commanded_delta_norm_m": vector_norm(commanded_translation),
        "commanded_rotation_delta_deg": commanded_rotation_deg,
        "ik": ik.to_json(),
        "q_target": q_target.tolist(),
        "q_delta": q_delta.tolist(),
        "q_delta_norm_rad": vector_norm(q_delta),
        "q_delta_abs_max_rad": q_delta_abs_max,
        "urdf_target_fk_T_link7_in_base": target_fk_T.tolist(),
        "urdf_target_fk_vs_target_error": target_fk_error,
    }

    print(
        f"\n=== G1 URDF IK -> ABS_JOINT probe: {side} ===\n"
        f"target_delta_m: {commanded_translation.tolist()} norm={item['commanded_delta_norm_m']:.4f}\n"
        f"target_rotation_delta_deg: {commanded_rotation_deg:.3f}\n"
        f"ik_success: {ik.success} ik_pos_err={ik.position_error_m:.6f}m ik_rot_err={ik.rotation_error_deg:.3f}deg\n"
        f"q_delta_abs_max_rad: {q_delta_abs_max:.4f}\n"
    )

    if not ik.success:
        item["blocked_reason"] = "ik_failed"
        return item
    if q_delta_abs_max > float(args.max_joint_delta_rad):
        item["blocked_reason"] = "joint_delta_too_large"
        print(
            f"[g1_urdf_ik_joint_control] skip {side}: max joint delta {q_delta_abs_max:.4f} rad "
            f"> limit {args.max_joint_delta_rad:.4f} rad"
        )
        return item
    if args.control_mode == "prompt":
        try:
            operator = input("[Enter]=execute IK ABS_JOINT target, s=skip, q=quit > ").strip().lower()
        except EOFError:
            operator = "q"
        item["operator_input"] = operator
        if operator == "q":
            item["blocked_reason"] = "operator_quit"
            item["quit_requested"] = True
            return item
        if operator == "s":
            item["blocked_reason"] = "operator_skip"
            return item
    else:
        item["operator_input"] = args.control_mode

    if args.control_mode == "dry-run":
        item["blocked_reason"] = "dry_run_only"
        return item
    if args.confirm_control != "RUN_CONTROL":
        item["blocked_reason"] = "missing_RUN_CONTROL_confirmation"
        return item

    q_rows = interpolate_q(q_before, q_target, args.num_points)
    full_arm_rows = [arm_state_with_side_q(joint_states_before["arm"], side, args.arm_state_mapping, row) for row in q_rows]
    control = call_abs_joint_trajectory(
        controller,
        joint_states_before,
        full_arm_rows,
        args.arm_state_mapping,
        args.reference_time,
    )
    item["control_sent"] = True
    item["control_result"] = control
    item["full_arm_target_rows"] = full_arm_rows
    if control.get("ok"):
        time.sleep(max(0.0, float(args.execute_s)))
    if args.settle_s > 0.0:
        time.sleep(max(0.0, float(args.settle_s)))

    joint_states_after = read_robot_joint_states_for_trajectory(robot)
    waist_after_for_urdf = waist_values_with_height_offset(joint_states_after["waist"], args.waist_height_offset_m)
    q_after, _ = side_q_from_arm_state(joint_states_after["arm"], side, args.arm_state_mapping)
    sdk_after_T = current_motion_T(controller, side, args.motion_tries, args.motion_sleep_s)
    urdf_after_T = kin.link7_fk(side, q_after, waist_states=waist_after_for_urdf)
    final_pose_error = pose_error(sdk_after_T, target_T)
    observed_translation = sdk_after_T[:3, 3] - sdk_before_T[:3, 3]
    observed_rotation_deg = rotation_angle_deg(sdk_after_T[:3, :3] @ sdk_before_T[:3, :3].T)
    q_error = q_after - q_target
    item.update(
        {
            "joint_states_after": json_safe(joint_states_after),
            "waist_after_values_for_urdf": waist_after_for_urdf,
            "q_after": q_after.tolist(),
            "q_error_after_minus_target": q_error.tolist(),
            "q_error_norm_rad": vector_norm(q_error),
            "q_error_abs_max_rad": float(np.max(np.abs(q_error))),
            "sdk_after_T_link7_in_base": sdk_after_T.tolist(),
            "urdf_after_T_link7_in_base": urdf_after_T.tolist(),
            "urdf_after_vs_sdk_error": pose_error(urdf_after_T, sdk_after_T),
            "final_sdk_after_vs_target_error": final_pose_error,
            "observed_delta_m": observed_translation.tolist(),
            "observed_delta_norm_m": vector_norm(observed_translation),
            "observed_rotation_delta_deg": observed_rotation_deg,
            "translation_norm_ratio": (
                vector_norm(observed_translation) / vector_norm(commanded_translation)
                if vector_norm(commanded_translation) > EPS
                else None
            ),
        }
    )
    print(
        "RESULT "
        f"side={side} "
        f"target_trans={item['commanded_delta_norm_m']:.4f}m "
        f"actual_trans={item['observed_delta_norm_m']:.4f}m "
        f"target_rot={commanded_rotation_deg:.2f}deg "
        f"actual_rot={observed_rotation_deg:.2f}deg "
        f"final_pos_err={final_pose_error['position_error_m']:.5f}m "
        f"final_rot_err={final_pose_error['rotation_error_deg']:.3f}deg "
        f"q_abs_max_err={item['q_error_abs_max_rad']:.5f}rad"
    )
    return item


def summarize_steps(steps: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"steps": len(steps), "control_sent": sum(1 for s in steps if s.get("control_sent"))}
    for step in steps:
        side = str(step.get("side"))
        final = step.get("final_sdk_after_vs_target_error") or {}
        summary[side] = {
            "control_sent": bool(step.get("control_sent")),
            "blocked_reason": step.get("blocked_reason"),
            "ik_success": ((step.get("ik") or {}).get("success")),
            "ik_position_error_m": ((step.get("ik") or {}).get("position_error_m")),
            "ik_rotation_error_deg": ((step.get("ik") or {}).get("rotation_error_deg")),
            "commanded_delta_norm_m": step.get("commanded_delta_norm_m"),
            "observed_delta_norm_m": step.get("observed_delta_norm_m"),
            "commanded_rotation_delta_deg": step.get("commanded_rotation_delta_deg"),
            "observed_rotation_delta_deg": step.get("observed_rotation_delta_deg"),
            "final_position_error_m": final.get("position_error_m"),
            "final_rotation_error_deg": final.get("rotation_error_deg"),
            "q_error_abs_max_rad": step.get("q_error_abs_max_rad"),
            "urdf_before_vs_sdk_position_error_m": ((step.get("urdf_before_vs_sdk_error") or {}).get("position_error_m")),
            "urdf_after_vs_sdk_position_error_m": ((step.get("urdf_after_vs_sdk_error") or {}).get("position_error_m")),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf-zip", default=str(DEFAULT_G1_ZIP))
    parser.add_argument("--out-dir", default=str(artifact_dir("diagnostics")))
    parser.add_argument("--tag", default="g1_urdf_ik_joint_control")
    parser.add_argument("--side", choices=["left", "right", "both"], default="right")
    parser.add_argument("--arm-state-mapping", choices=["left_first", "right_first"], default="left_first")
    parser.add_argument("--waist-height-offset-m", type=float, default=-0.3)
    parser.add_argument("--control-mode", choices=["prompt", "auto", "dry-run"], default="prompt")
    parser.add_argument("--confirm-control", default="")
    parser.add_argument("--delta-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--delta-m", type=float, default=0.01)
    parser.add_argument("--rotation-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--rotation-deg", type=float, default=2.0)
    parser.add_argument("--num-points", type=int, default=20)
    parser.add_argument("--reference-time", type=float, default=1.0)
    parser.add_argument("--execute-s", type=float, default=1.0)
    parser.add_argument("--settle-s", type=float, default=0.5)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument("--max-joint-delta-rad", type=float, default=0.35)
    parser.add_argument("--motion-tries", type=int, default=30)
    parser.add_argument("--motion-sleep-s", type=float, default=0.1)
    parser.add_argument("--upload-url", default="")
    parser.add_argument("--upload-timeout-s", type=float, default=20.0)
    args = parser.parse_args()

    out_base = Path(args.out_dir).expanduser().resolve()
    default_base = artifact_dir("diagnostics")
    if out_base == default_base:
        run_dir = artifact_run_dir("diagnostics", args.tag, prefix="g1_urdf_ik_joint_control")
    else:
        run_dir = out_base / f"g1_urdf_ik_joint_control_{utc_stamp()}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "ok": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "control_sent": False,
        "note": "Requires RUN_CONTROL to send SDK ABS_JOINT. Uses URDF IK with waist height offset.",
    }
    try:
        from a2d_sdk.robot import RobotController, RobotDds

        robot = RobotDds()
        controller = RobotController()
        kin = G1UrdfKinematics(args.urdf_zip)
        sides = ["right", "left"] if args.side == "both" else [args.side]
        steps = []
        for side in sides:
            step = validate_one_side(args=args, kin=kin, robot=robot, controller=controller, side=side)
            steps.append(step)
            if step.get("quit_requested"):
                break
        summary = summarize_steps(steps)
        report.update(
            {
                "ok": all((step.get("control_sent") and (step.get("control_result") or {}).get("ok")) for step in steps)
                if args.confirm_control == "RUN_CONTROL" and args.control_mode != "dry-run"
                else True,
                "control_sent": any(step.get("control_sent") for step in steps),
                "steps": steps,
                "summary": summary,
            }
        )
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
        report.update(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )

    report_path = run_dir / "g1_urdf_ik_joint_control_report.json"
    report_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = run_dir / "g1_urdf_ik_joint_control_summary.json"
    summary_path.write_text(json.dumps(json_safe(report.get("summary") or {}), ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = make_zip(run_dir)
    upload = None
    if args.upload_url:
        try:
            upload = upload_zip(zip_path, args.upload_url, args.upload_timeout_s)
        except Exception as exc:
            upload = {"ok": False, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        (run_dir / "upload_result.json").write_text(json.dumps(json_safe(upload), ensure_ascii=False, indent=2), encoding="utf-8")
        zip_path = make_zip(run_dir)

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "zip_path": str(zip_path),
                "report_path": str(report_path),
                "summary_path": str(summary_path),
                "summary": report.get("summary"),
                "upload": upload,
                "ok": report.get("ok"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if report.get("ok") else 2)


if __name__ == "__main__":
    main()
