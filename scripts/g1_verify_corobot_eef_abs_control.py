#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify direct CoRobot G01Env EEF_ABS control around the current G1 pose."""

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
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    target_T[axis_idx, 3] += float(delta_m)
    if abs(float(rotation_deg)) > EPS:
        target_T[:3, :3] = R_axis(rotation_axis, rotation_deg) @ target_T[:3, :3]
    return target_T


def interpolate_rows(start_T: np.ndarray, target_T: np.ndarray, num_points: int) -> list[list[float]]:
    start_T = np.asarray(start_T, dtype=np.float64).reshape(4, 4)
    target_T = np.asarray(target_T, dtype=np.float64).reshape(4, 4)
    num_points = max(1, int(num_points))
    rows: list[list[float]] = []
    for idx in range(1, num_points + 1):
        alpha = float(idx) / float(num_points)
        T = start_T.copy()
        T[:3, 3] = start_T[:3, 3] + alpha * (target_T[:3, 3] - start_T[:3, 3])
        if idx == num_points:
            T[:3, :3] = target_T[:3, :3]
        rows.append(row_from_T(T))
    return rows


def build_action(rows: list[list[float]], side: str, duration_s: float) -> dict[str, Any]:
    return {
        "timestamps": int(time.time() * 1e9),
        "trajectory_reference_time": float(duration_s),
        "base_link": "base_link",
        f"{side}_arm": {
            "kind": "EEF_ABS",
            "values": rows,
        },
    }


def execute_corobot_env(action: dict[str, Any], wait_action_time: float) -> dict[str, Any]:
    started = time.time()
    env = None
    try:
        from corobot.envs.g01_env import G01Env

        env = G01Env()
        setup_result = env.setup()
        execute_result = env.execute_action(action, wait_action_time=float(wait_action_time))
        return {
            "ok": True,
            "executor": "corobot_env",
            "duration_s": time.time() - started,
            "setup_result": json_safe(setup_result),
            "execute_result": json_safe(execute_result),
        }
    except Exception as exc:
        return {
            "ok": False,
            "executor": "corobot_env",
            "duration_s": time.time() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


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
    parser.add_argument("--tag", default="corobot_eef_abs_verify")
    parser.add_argument("--side", choices=["right", "left"], default="right")
    parser.add_argument("--mode", choices=["hold", "move"], default="hold")
    parser.add_argument("--control-mode", choices=["prompt", "auto", "dry-run"], default="prompt")
    parser.add_argument("--confirm-control", default="")
    parser.add_argument("--delta-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--delta-m", type=float, default=-0.01)
    parser.add_argument("--rotation-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--rotation-deg", type=float, default=0.0)
    parser.add_argument("--num-points", type=int, default=30)
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--upload-url", default="")
    parser.add_argument("--upload-timeout-s", type=float, default=20.0)
    args = parser.parse_args()

    out_base = Path(args.out_dir).expanduser().resolve()
    default_base = artifact_dir("diagnostics")
    if out_base == default_base:
        run_dir = artifact_run_dir("diagnostics", args.tag, prefix="corobot_eef_abs_verify")
    else:
        run_dir = out_base / f"corobot_eef_abs_verify_{utc_stamp()}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "ok": False,
        "args": vars(args),
        "control_sent": False,
        "note": "Direct CoRobot G01Env EEF_ABS control probe. Requires RUN_CONTROL to move.",
    }
    arm = None
    try:
        from G1RobotArm import G1RobotArmReadOnly

        arm = G1RobotArmReadOnly(side=args.side)
        before_T = arm.get_T_link7_in_base()
        if args.mode == "hold":
            target_T = before_T.copy()
            rows = [row_from_T(before_T.copy()) for _ in range(max(1, int(args.num_points)))]
        else:
            target_T = target_from_delta(before_T, args.delta_axis, args.delta_m, args.rotation_axis, args.rotation_deg)
            rows = interpolate_rows(before_T, target_T, args.num_points)
        action = build_action(rows, args.side, args.duration_s)
        report.update(
            {
                "before_T_link7_in_base": before_T.tolist(),
                "target_T_link7_in_base": target_T.tolist(),
                "rows": rows,
                "corobot_action": action,
            }
        )
        action_path = run_dir / "corobot_eef_abs_action.json"
        action_path.write_text(json.dumps(json_safe(action), ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            "\n=== CoRobot EEF_ABS control probe ===\n"
            f"mode: {args.mode}\n"
            f"side: {args.side}\n"
            f"rows: {len(rows)}\n"
            f"duration_s: {args.duration_s:.3f}\n"
            f"target_delta_m: {(target_T[:3, 3] - before_T[:3, 3]).tolist()}\n"
            f"target_rotation_delta_deg: {rotation_angle_deg(target_T[:3, :3] @ before_T[:3, :3].T):.3f}\n"
            f"action_json: {action_path}\n"
        )

        if args.control_mode == "prompt":
            operator = input("[Enter]=execute CoRobot EEF_ABS probe, q=quit > ").strip().lower()
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
            control_result = execute_corobot_env(action, args.duration_s)
            report["control_sent"] = True
            report["control_result"] = control_result
            if control_result.get("ok"):
                time.sleep(max(0.0, float(args.settle_s)))
            after_T = arm.get_T_link7_in_base()
            report["after_T_link7_in_base"] = after_T.tolist()
            report["tracking"] = evaluate_motion(before_T, target_T, after_T)
            tracking = report["tracking"]["translation_tracking"]
            print(
                "RESULT "
                f"target_trans={report['tracking']['commanded_delta_norm_m']:.4f}m "
                f"actual_trans={report['tracking']['observed_delta_norm_m']:.4f}m "
                f"ratio={tracking.get('norm_ratio')} "
                f"cos={tracking.get('cosine_to_target_delta')} "
                f"target_rot={report['tracking']['target_rotation_delta_deg']:.2f}deg "
                f"actual_rot={report['tracking']['observed_rotation_delta_deg']:.2f}deg "
                f"final_pos_err={report['tracking']['final_pose_position_error_m']:.4f}m "
                f"final_rot_err={report['tracking']['final_pose_rotation_error_deg']:.2f}deg"
            )
            report["ok"] = bool(control_result.get("ok"))
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
            log("skipping read-only arm adapter close for CoRobot EEF_ABS verify")

    report_path = run_dir / "corobot_eef_abs_verify_report.json"
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

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "zip_path": str(zip_path),
                "report_path": str(report_path),
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
