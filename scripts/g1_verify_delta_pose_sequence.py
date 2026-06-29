#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe G1 trajectory_tracking_control DELTA_POSE consistency.

This is a pure SDK/control diagnostic. It does not call HumanEgo inference or
vision. It sends a small predefined sequence around the current link7 pose and
logs commanded DELTA_POSE versus the settled link7 motion.
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from g1_artifacts import artifact_dir, run_dir as artifact_run_dir  # noqa: E402
from g1_humanego_client_dry_run import json_safe, log, upload_zip  # noqa: E402
from g1_humanego_interactive_step_client import (  # noqa: E402
    link7_delta_pose_command,
    read_robot_joint_states_for_trajectory,
    rotation_angle_deg,
    rotation_vector_from_delta,
    translation_tracking_report,
)


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
    angle = np.radians(float(angle_deg))
    c, s = float(np.cos(angle)), float(np.sin(angle))
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    raise ValueError(f"unknown axis {axis!r}")


def make_step_transform(step: dict[str, Any], rotation_frame: str) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(step.get("translation_m") or [0.0, 0.0, 0.0], dtype=np.float64).reshape(3)
    rot_axis = step.get("rotation_axis")
    rot_deg = float(step.get("rotation_deg", 0.0))
    if rot_axis and abs(rot_deg) > EPS:
        T[:3, :3] = R_axis(str(rot_axis), rot_deg)
    return T


def apply_delta(start_T: np.ndarray, delta_T: np.ndarray, rotation_frame: str) -> np.ndarray:
    start_T = np.asarray(start_T, dtype=np.float64).reshape(4, 4)
    delta_T = np.asarray(delta_T, dtype=np.float64).reshape(4, 4)
    target_T = start_T.copy()
    target_T[:3, 3] = start_T[:3, 3] + delta_T[:3, 3]
    if rotation_frame == "base":
        target_T[:3, :3] = delta_T[:3, :3] @ start_T[:3, :3]
    elif rotation_frame == "local":
        target_T[:3, :3] = start_T[:3, :3] @ delta_T[:3, :3]
    else:
        raise ValueError(f"unknown rotation_frame {rotation_frame!r}")
    return target_T


def default_sequence(step_m: float, rot_deg: float) -> list[dict[str, Any]]:
    return [
        {"name": "+x", "translation_m": [step_m, 0.0, 0.0]},
        {"name": "-x", "translation_m": [-step_m, 0.0, 0.0]},
        {"name": "+y", "translation_m": [0.0, step_m, 0.0]},
        {"name": "-y", "translation_m": [0.0, -step_m, 0.0]},
        {"name": "+z", "translation_m": [0.0, 0.0, step_m]},
        {"name": "-z", "translation_m": [0.0, 0.0, -step_m]},
        {"name": "+rx", "translation_m": [0.0, 0.0, 0.0], "rotation_axis": "x", "rotation_deg": rot_deg},
        {"name": "-rx", "translation_m": [0.0, 0.0, 0.0], "rotation_axis": "x", "rotation_deg": -rot_deg},
        {"name": "+rz", "translation_m": [0.0, 0.0, 0.0], "rotation_axis": "z", "rotation_deg": rot_deg},
        {"name": "-rz", "translation_m": [0.0, 0.0, 0.0], "rotation_axis": "z", "rotation_deg": -rot_deg},
    ]


def load_sequence(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.sequence_json:
        data = json.loads(args.sequence_json)
        if not isinstance(data, list):
            raise ValueError("--sequence-json must decode to a list")
        return data
    if args.sequence_file:
        data = json.loads(Path(args.sequence_file).expanduser().read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("--sequence-file must contain a JSON list")
        return data
    return default_sequence(args.step_m, args.rot_deg)


def call_delta_pose_once(
    controller: Any,
    robot: Any,
    side: str,
    before_T: np.ndarray,
    target_T: np.ndarray,
    rotation_frame: str,
    reference_time: float,
) -> dict[str, Any]:
    delta_pose = link7_delta_pose_command(before_T, target_T, rotation_frame=rotation_frame)
    joint_states = read_robot_joint_states_for_trajectory(robot)
    action_data = [float(v) for v in delta_pose["action_data"]]
    zero = [0.0] * 6
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
    kwargs = {
        "infer_timestamp": int(time.time() * 1e9),
        "robot_states": {
            "head": joint_states["head"],
            "waist": joint_states["waist"],
            "arm": joint_states["arm"],
        },
        "robot_actions": [robot_action],
        "robot_link": "base_link",
        "trajectory_reference_time": float(reference_time),
    }
    started = time.time()
    try:
        result = controller.trajectory_tracking_control(**kwargs)
        return {
            "ok": True,
            "duration_s": time.time() - started,
            "delta_pose": delta_pose,
            "joint_states": joint_states,
            "kwargs": json_safe(kwargs),
            "result": json_safe(result),
        }
    except Exception as exc:
        return {
            "ok": False,
            "duration_s": time.time() - started,
            "delta_pose": delta_pose,
            "joint_states": joint_states,
            "kwargs": json_safe(kwargs),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def execute_sequence(args: argparse.Namespace, arm: Any, sequence: list[dict[str, Any]]) -> dict[str, Any]:
    steps = []
    start_T = arm.get_T_link7_in_base()
    for idx, spec in enumerate(sequence):
        before_T = arm.get_T_link7_in_base()
        delta_T = make_step_transform(spec, args.rotation_frame)
        target_T = apply_delta(before_T, delta_T, args.rotation_frame)
        commanded_delta = target_T[:3, 3] - before_T[:3, 3]
        target_rotation_delta_deg = rotation_angle_deg(target_T[:3, :3] @ before_T[:3, :3].T)
        item: dict[str, Any] = {
            "idx": idx,
            "name": str(spec.get("name", f"step_{idx:03d}")),
            "spec": spec,
            "before_T_link7_in_base": before_T.tolist(),
            "target_T_link7_in_base": target_T.tolist(),
            "commanded_delta_m": commanded_delta.tolist(),
            "commanded_delta_norm_m": float(np.linalg.norm(commanded_delta)),
            "target_rotation_delta_deg": target_rotation_delta_deg,
            "target_rotation_vector_base_deg": rotation_vector_from_delta(
                target_T[:3, :3] @ before_T[:3, :3].T
            ).tolist(),
        }
        log(
            f"step {idx} {item['name']}: command trans={commanded_delta.tolist()} "
            f"norm={item['commanded_delta_norm_m']:.4f}m rot={target_rotation_delta_deg:.2f}deg"
        )
        if args.confirm_control != "RUN_CONTROL":
            item["control_sent"] = False
            item["blocked_reason"] = "missing RUN_CONTROL confirmation"
            steps.append(item)
            continue

        control = call_delta_pose_once(
            arm.controller,
            arm.robot,
            args.side,
            before_T,
            target_T,
            args.rotation_frame,
            args.reference_time,
        )
        item["control_sent"] = True
        item["control_result"] = control
        if control.get("ok"):
            time.sleep(max(0.0, float(args.execute_s)))
        if args.settle_s > 0.0:
            time.sleep(max(0.0, float(args.settle_s)))
        after_T = arm.get_T_link7_in_base()
        observed_delta = after_T[:3, 3] - before_T[:3, 3]
        item["after_T_link7_in_base"] = after_T.tolist()
        item["observed_delta_m"] = observed_delta.tolist()
        item["observed_delta_norm_m"] = float(np.linalg.norm(observed_delta))
        item["translation_tracking"] = translation_tracking_report(commanded_delta, observed_delta)
        item["observed_rotation_delta_deg"] = rotation_angle_deg(after_T[:3, :3] @ before_T[:3, :3].T)
        item["observed_rotation_vector_base_deg"] = rotation_vector_from_delta(
            after_T[:3, :3] @ before_T[:3, :3].T
        ).tolist()
        item["rotation_error_deg_abs"] = abs(item["observed_rotation_delta_deg"] - target_rotation_delta_deg)
        item["return_to_start_delta_m"] = (after_T[:3, 3] - start_T[:3, 3]).tolist()
        log(
            f"step {idx} {item['name']}: observed trans={observed_delta.tolist()} "
            f"norm={item['observed_delta_norm_m']:.4f}m "
            f"cos={item['translation_tracking'].get('cosine_to_target_delta')} "
            f"rot={item['observed_rotation_delta_deg']:.2f}deg"
        )
        steps.append(item)
        if not control.get("ok"):
            break
    return {
        "start_T_link7_in_base": start_T.tolist(),
        "end_T_link7_in_base": arm.get_T_link7_in_base().tolist(),
        "steps": steps,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(artifact_dir("diagnostics")))
    parser.add_argument("--tag", default="delta_pose_sequence")
    parser.add_argument("--side", choices=["right", "left"], default="right")
    parser.add_argument("--confirm-control", default="")
    parser.add_argument("--step-m", type=float, default=0.01)
    parser.add_argument("--rot-deg", type=float, default=5.0)
    parser.add_argument("--rotation-frame", choices=["base", "local"], default="base")
    parser.add_argument("--reference-time", type=float, default=0.5)
    parser.add_argument("--execute-s", type=float, default=0.5)
    parser.add_argument("--settle-s", type=float, default=0.5)
    parser.add_argument("--sequence-json", default="")
    parser.add_argument("--sequence-file", default="")
    parser.add_argument("--upload-url", default="")
    parser.add_argument("--upload-timeout-s", type=float, default=20.0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    out_base = Path(args.out_dir).expanduser().resolve()
    default_base = artifact_dir("diagnostics")
    if out_base == default_base:
        run_dir = artifact_run_dir("diagnostics", args.tag, prefix="delta_pose_sequence")
    else:
        run_dir = out_base / f"g1_delta_pose_sequence_{utc_stamp()}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "ok": False,
        "args": vars(args),
        "control_sent": False,
        "note": "Pure SDK DELTA_POSE probe. Requires --confirm-control RUN_CONTROL to move.",
    }
    arm = None
    try:
        from G1RobotArm import G1RobotArmReadOnly

        sequence = load_sequence(args)
        report["sequence"] = sequence
        arm = G1RobotArmReadOnly(side=args.side)
        result = execute_sequence(args, arm, sequence)
        report.update(result)
        report["control_sent"] = any(item.get("control_sent") for item in result["steps"])
        report["ok"] = all((not item.get("control_sent")) or (item.get("control_result") or {}).get("ok") for item in result["steps"])
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
            log("skipping read-only arm adapter close for delta pose sequence probe")

    report_path = run_dir / "delta_pose_sequence_report.json"
    report_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    summary = []
    for item in report.get("steps") or []:
        tracking = item.get("translation_tracking") or {}
        summary.append(
            {
                "idx": item.get("idx"),
                "name": item.get("name"),
                "control_sent": item.get("control_sent"),
                "commanded_delta_m": item.get("commanded_delta_m"),
                "commanded_delta_norm_m": item.get("commanded_delta_norm_m"),
                "observed_delta_m": item.get("observed_delta_m"),
                "observed_delta_norm_m": item.get("observed_delta_norm_m"),
                "translation_ratio": (
                    item.get("observed_delta_norm_m") / item.get("commanded_delta_norm_m")
                    if item.get("commanded_delta_norm_m")
                    else None
                ),
                "translation_cosine": tracking.get("cosine_to_target_delta"),
                "target_rotation_delta_deg": item.get("target_rotation_delta_deg"),
                "observed_rotation_delta_deg": item.get("observed_rotation_delta_deg"),
                "rotation_error_deg_abs": item.get("rotation_error_deg_abs"),
            }
        )
    (run_dir / "delta_pose_sequence_summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    zip_path = make_zip(run_dir)
    upload = None
    if args.upload_url:
        try:
            upload = upload_zip(zip_path, args.upload_url, args.upload_timeout_s)
        except Exception as exc:
            upload = {"ok": False, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        (run_dir / "upload_result.json").write_text(json.dumps(json_safe(upload), ensure_ascii=False, indent=2), encoding="utf-8")
        zip_path = make_zip(run_dir)

    print(json.dumps(
        {
            "run_dir": str(run_dir),
            "zip_path": str(zip_path),
            "report_path": str(report_path),
            "summary_path": str(run_dir / "delta_pose_sequence_summary.json"),
            "upload": upload,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
