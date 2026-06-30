#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe G1 trajectory_tracking_control ABS_POSE consistency."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "inference", PROJECT_ROOT / "scripts"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from g1_artifacts import artifact_dir, run_dir as artifact_run_dir  # noqa: E402
from g1_humanego_client_dry_run import json_safe, log, upload_zip  # noqa: E402
from g1_humanego_interactive_step_client import (  # noqa: E402
    read_robot_joint_states_for_trajectory,
    rotation_angle_deg,
    translation_tracking_report,
)
from g1_replay_humanego_eef_abs_corobot import row_from_T  # noqa: E402


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


def R_axis(axis: str, angle_deg: float) -> np.ndarray:
    axis = axis.lower()
    angle = math.radians(float(angle_deg))
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    raise ValueError(f"unknown axis {axis!r}")


def target_from_delta(start_T: np.ndarray, axis: str, delta_m: float, rotation_axis: str, rotation_deg: float) -> np.ndarray:
    target_T = np.asarray(start_T, dtype=np.float64).reshape(4, 4).copy()
    target_T[{"x": 0, "y": 1, "z": 2}[axis.lower()], 3] += float(delta_m)
    if abs(float(rotation_deg)) > EPS:
        target_T[:3, :3] = R_axis(rotation_axis, rotation_deg) @ target_T[:3, :3]
    return target_T


def interpolate_Ts(start_T: np.ndarray, target_T: np.ndarray, num_points: int) -> list[np.ndarray]:
    start_T = np.asarray(start_T, dtype=np.float64).reshape(4, 4)
    target_T = np.asarray(target_T, dtype=np.float64).reshape(4, 4)
    num_points = max(1, int(num_points))
    out: list[np.ndarray] = []
    for idx in range(1, num_points + 1):
        alpha = float(idx) / float(num_points)
        T = start_T.copy()
        T[:3, 3] = start_T[:3, 3] + alpha * (target_T[:3, 3] - start_T[:3, 3])
        if idx == num_points:
            T[:3, :3] = target_T[:3, :3]
        out.append(T)
    return out


def opposite_side(side: str) -> str:
    if side == "right":
        return "left"
    if side == "left":
        return "right"
    raise ValueError(f"unknown side {side!r}")


def read_link7_T_from_motion_status(
    controller: Any,
    side: str,
    tries: int,
    sleep_s: float,
    wait_motion_status_fn: Any,
    parse_motion_pose_fn: Any,
) -> np.ndarray:
    status = wait_motion_status_fn(controller, tries, sleep_s)
    if not isinstance(status, dict):
        raise RuntimeError(f"get_motion_status did not return a dict: {status!r}")
    frames = status.get("frames") or {}
    frame_name = f"arm_{side}_link7"
    if frame_name not in frames:
        raise RuntimeError(f"{frame_name} not in motion_status frames: {sorted(frames.keys())}")
    return parse_motion_pose_fn(frames[frame_name])


def call_abs_pose_trajectory_once(
    controller: Any,
    robot: Any,
    side: str,
    active_rows: list[list[float]],
    inactive_rows: list[list[float]],
    reference_time: float,
) -> dict[str, Any]:
    if len(active_rows) != len(inactive_rows):
        raise ValueError(f"active_rows and inactive_rows length mismatch: {len(active_rows)} != {len(inactive_rows)}")
    if not active_rows:
        raise ValueError("active_rows is empty")

    joint_states = read_robot_joint_states_for_trajectory(robot)
    robot_actions = []
    for idx, row in enumerate(active_rows):
        active_action_data = [float(v) for v in row]
        inactive_action_data = [float(v) for v in inactive_rows[idx]]
        left_action_data = active_action_data if side == "left" else inactive_action_data
        right_action_data = active_action_data if side == "right" else inactive_action_data
        robot_actions.append(
            {
                "left_arm": {
                    "action_data": left_action_data,
                    "control_type": "ABS_POSE",
                },
                "right_arm": {
                    "action_data": right_action_data,
                    "control_type": "ABS_POSE",
                },
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
        "trajectory_reference_time": float(reference_time),
    }
    started = time.time()
    try:
        result = controller.trajectory_tracking_control(**kwargs)
        return {
            "ok": True,
            "duration_s": time.time() - started,
            "joint_states": joint_states,
            "kwargs": json_safe(kwargs),
            "result": json_safe(result),
        }
    except Exception as exc:
        return {
            "ok": False,
            "duration_s": time.time() - started,
            "joint_states": joint_states,
            "kwargs": json_safe(kwargs),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def evaluate_motion(before_T: np.ndarray, target_T: np.ndarray, after_T: np.ndarray) -> dict[str, Any]:
    commanded_delta = target_T[:3, 3] - before_T[:3, 3]
    observed_delta = after_T[:3, 3] - before_T[:3, 3]
    tracking = translation_tracking_report(commanded_delta, observed_delta)
    target_norm = float(tracking.get("target_norm_m") or 0.0)
    observed_norm = float(tracking.get("observed_norm_m") or 0.0)
    tracking["norm_ratio"] = observed_norm / target_norm if target_norm > EPS else None
    target_rot = rotation_angle_deg(target_T[:3, :3] @ before_T[:3, :3].T)
    observed_rot = rotation_angle_deg(after_T[:3, :3] @ before_T[:3, :3].T)
    return {
        "commanded_delta_m": commanded_delta.tolist(),
        "commanded_delta_norm_m": float(np.linalg.norm(commanded_delta)),
        "observed_delta_m": observed_delta.tolist(),
        "observed_delta_norm_m": float(np.linalg.norm(observed_delta)),
        "translation_tracking": tracking,
        "target_rotation_delta_deg": target_rot,
        "observed_rotation_delta_deg": observed_rot,
        "rotation_error_deg_abs": abs(observed_rot - target_rot),
        "final_pose_position_error_m": float(np.linalg.norm(target_T[:3, 3] - after_T[:3, 3])),
        "final_pose_rotation_error_deg": rotation_angle_deg(target_T[:3, :3] @ after_T[:3, :3].T),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(artifact_dir("diagnostics")))
    parser.add_argument("--tag", default="abs_pose_sequence")
    parser.add_argument("--side", choices=["right", "left"], default="right")
    parser.add_argument("--mode", choices=["hold", "move"], default="hold")
    parser.add_argument("--control-mode", choices=["prompt", "auto", "dry-run"], default="prompt")
    parser.add_argument("--confirm-control", default="")
    parser.add_argument("--delta-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--delta-m", type=float, default=-0.01)
    parser.add_argument("--rotation-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--rotation-deg", type=float, default=0.0)
    parser.add_argument("--num-points", type=int, default=30)
    parser.add_argument("--reference-time", type=float, default=2.0)
    parser.add_argument("--execute-s", type=float, default=2.0)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--upload-url", default="")
    parser.add_argument("--upload-timeout-s", type=float, default=20.0)
    args = parser.parse_args()

    out_base = Path(args.out_dir).expanduser().resolve()
    default_base = artifact_dir("diagnostics")
    if out_base == default_base:
        run_dir = artifact_run_dir("diagnostics", args.tag, prefix="abs_pose_sequence")
    else:
        run_dir = out_base / f"g1_abs_pose_sequence_{utc_stamp()}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "ok": False,
        "args": vars(args),
        "control_sent": False,
        "note": "Direct G1 SDK trajectory_tracking_control ABS_POSE probe. Requires RUN_CONTROL to move.",
    }
    arm = None
    try:
        from G1RobotArm import G1RobotArmReadOnly, parse_motion_pose, wait_motion_status

        arm = G1RobotArmReadOnly(side=args.side)
        inactive_side = opposite_side(args.side)
        before_T = arm.get_T_link7_in_base()
        inactive_before_T = read_link7_T_from_motion_status(
            arm.controller,
            inactive_side,
            arm.motion_tries,
            arm.motion_sleep_s,
            wait_motion_status,
            parse_motion_pose,
        )
        if args.mode == "hold":
            target_T = before_T.copy()
        else:
            target_T = target_from_delta(before_T, args.delta_axis, args.delta_m, args.rotation_axis, args.rotation_deg)
        target_Ts = interpolate_Ts(before_T, target_T, args.num_points)
        rows = [row_from_T(T) for T in target_Ts]
        inactive_target_Ts = [inactive_before_T.copy() for _ in target_Ts]
        inactive_rows = [row_from_T(T) for T in inactive_target_Ts]
        report.update(
            {
                "active_side": args.side,
                "inactive_side": inactive_side,
                "before_T_link7_in_base": before_T.tolist(),
                "target_T_link7_in_base": target_T.tolist(),
                "rows": rows,
                "target_Ts_link7_in_base": [T.tolist() for T in target_Ts],
                "inactive_before_T_link7_in_base": inactive_before_T.tolist(),
                "inactive_hold_T_link7_in_base": inactive_before_T.tolist(),
                "inactive_rows": inactive_rows,
                "inactive_target_Ts_link7_in_base": [T.tolist() for T in inactive_target_Ts],
                "inactive_note": "Inactive arm receives repeated current ABS_POSE hold rows; ABS_POSE zero rows are never used.",
            }
        )
        rows_path = run_dir / "abs_pose_rows.json"
        rows_path.write_text(
            json.dumps(
                json_safe(
                    {
                        "active_side": args.side,
                        "inactive_side": inactive_side,
                        "active_rows": rows,
                        "inactive_hold_rows": inactive_rows,
                    }
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            "\n=== G1 ABS_POSE trajectory_tracking_control probe ===\n"
            f"mode: {args.mode}\n"
            f"side: {args.side}\n"
            f"inactive_side_hold: {inactive_side}\n"
            f"rows: {len(rows)}\n"
            f"reference_time: {args.reference_time:.3f}\n"
            f"target_delta_m: {(target_T[:3, 3] - before_T[:3, 3]).tolist()}\n"
            f"target_rotation_delta_deg: {rotation_angle_deg(target_T[:3, :3] @ before_T[:3, :3].T):.3f}\n"
            f"rows_json: {rows_path}\n"
        )
        if args.control_mode == "prompt":
            operator = input("[Enter]=execute ABS_POSE trajectory, q=quit > ").strip().lower()
            report["operator_input"] = operator
            if operator == "q":
                report["blocked_reason"] = "operator_quit"
                raise SystemExit(0)
        else:
            report["operator_input"] = args.control_mode

        if args.control_mode == "dry-run":
            report["ok"] = True
            report["blocked_reason"] = "dry_run_only"
        elif args.confirm_control != "RUN_CONTROL":
            report["blocked_reason"] = "missing RUN_CONTROL confirmation"
        else:
            control = call_abs_pose_trajectory_once(
                arm.controller,
                arm.robot,
                args.side,
                rows,
                inactive_rows,
                args.reference_time,
            )
            report["control_sent"] = True
            report["control_result"] = control
            if control.get("ok"):
                time.sleep(max(0.0, float(args.execute_s)))
            if args.settle_s > 0.0:
                time.sleep(max(0.0, float(args.settle_s)))
            after_T = arm.get_T_link7_in_base()
            inactive_after_T = read_link7_T_from_motion_status(
                arm.controller,
                inactive_side,
                arm.motion_tries,
                arm.motion_sleep_s,
                wait_motion_status,
                parse_motion_pose,
            )
            report["after_T_link7_in_base"] = after_T.tolist()
            report["inactive_after_T_link7_in_base"] = inactive_after_T.tolist()
            report["tracking"] = evaluate_motion(before_T, target_T, after_T)
            report["inactive_tracking"] = evaluate_motion(inactive_before_T, inactive_before_T, inactive_after_T)
            tracking = report["tracking"]["translation_tracking"]
            inactive_tracking = report["inactive_tracking"]
            print(
                "RESULT "
                f"target_trans={report['tracking']['commanded_delta_norm_m']:.4f}m "
                f"actual_trans={report['tracking']['observed_delta_norm_m']:.4f}m "
                f"ratio={tracking.get('norm_ratio')} "
                f"cos={tracking.get('cosine_to_target_delta')} "
                f"target_rot={report['tracking']['target_rotation_delta_deg']:.2f}deg "
                f"actual_rot={report['tracking']['observed_rotation_delta_deg']:.2f}deg "
                f"final_pos_err={report['tracking']['final_pose_position_error_m']:.4f}m "
                f"final_rot_err={report['tracking']['final_pose_rotation_error_deg']:.2f}deg "
                f"inactive_{inactive_side}_drift={inactive_tracking['observed_delta_norm_m']:.4f}m "
                f"inactive_{inactive_side}_rot={inactive_tracking['observed_rotation_delta_deg']:.2f}deg"
            )
            report["ok"] = bool(control.get("ok"))
    except SystemExit:
        pass
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
    finally:
        if arm is not None:
            log("skipping read-only arm adapter close for ABS_POSE probe")

    report_path = run_dir / "abs_pose_sequence_report.json"
    report_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = make_zip(run_dir)
    upload = None
    if args.upload_url:
        try:
            upload = upload_zip(zip_path, args.upload_url, args.upload_timeout_s)
        except Exception as exc:
            upload = {"ok": False, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        (run_dir / "upload_result.json").write_text(json.dumps(json_safe(upload), ensure_ascii=False, indent=2), encoding="utf-8")
        zip_path = make_zip(run_dir)

    print(json.dumps({"run_dir": str(run_dir), "zip_path": str(zip_path), "report_path": str(report_path), "upload": upload}, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
