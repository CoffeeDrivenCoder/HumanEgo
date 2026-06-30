#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay fixed HumanEgo SDK DELTA_POSE action_data and compare tracking."""

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


def load_sequence(path: str) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    data = json.loads(p.read_text(encoding="utf-8"))
    if "actions" not in data:
        raise ValueError(f"sequence file missing actions: {p}")
    data["_path"] = str(p)
    return data


def action_rotation_angle_deg(action_data: list[float]) -> float:
    rotvec = np.asarray(action_data[3:6], dtype=np.float64)
    return float(np.degrees(np.linalg.norm(rotvec)))


def call_fixed_action_once(
    controller: Any,
    robot: Any,
    *,
    side: str,
    action_data: list[float],
    reference_time: float,
) -> dict[str, Any]:
    action_data = [float(v) for v in action_data]
    zero = [0.0] * 6
    joint_states = read_robot_joint_states_for_trajectory(robot)
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


def execute_replay(args: argparse.Namespace, arm: Any, sequence: dict[str, Any]) -> dict[str, Any]:
    actions = list(sequence.get("actions") or [])
    if args.max_actions > 0:
        actions = actions[: args.max_actions]
    steps: list[dict[str, Any]] = []
    start_T = arm.get_T_link7_in_base()
    default_reference_time = (
        float(args.reference_time)
        if args.reference_time is not None
        else None
    )

    for idx, action in enumerate(actions):
        action_data = [float(v) for v in action["action_data"]]
        side = args.side or str(action.get("side") or sequence.get("side") or "right")
        reference_time = default_reference_time
        if reference_time is None:
            reference_time = float(action.get("trajectory_reference_time") or args.fallback_reference_time)
        before_T = arm.get_T_link7_in_base()
        target_translation = np.asarray(action_data[:3], dtype=np.float64)
        target_rotation_deg = action_rotation_angle_deg(action_data)
        online_observed = action.get("online_observed") or {}
        item: dict[str, Any] = {
            "idx": idx,
            "online_step_idx": action.get("online_step_idx"),
            "request_id": action.get("request_id"),
            "side": side,
            "action_data": action_data,
            "trajectory_reference_time": reference_time,
            "before_T_link7_in_base": before_T.tolist(),
            "target_translation_m": target_translation.tolist(),
            "target_translation_norm_m": float(np.linalg.norm(target_translation)),
            "target_rotation_deg_from_action": target_rotation_deg,
            "source_action": action,
        }
        log(
            f"replay step {idx} online_step={action.get('online_step_idx')}: "
            f"target_trans={target_translation.tolist()} "
            f"norm={item['target_translation_norm_m']:.4f}m "
            f"rot={target_rotation_deg:.2f}deg ref_time={reference_time:.3f}s"
        )
        if args.control_mode == "prompt":
            try:
                operator = input("[Enter]=replay action, s=skip, q=quit > ").strip().lower()
            except EOFError:
                operator = "q"
            item["operator_input"] = operator
            if operator == "q":
                item["control_sent"] = False
                item["blocked_reason"] = "operator_quit"
                steps.append(item)
                break
            if operator == "s":
                item["control_sent"] = False
                item["blocked_reason"] = "operator_skip"
                steps.append(item)
                continue
        else:
            item["operator_input"] = "auto"

        if args.confirm_control != "RUN_CONTROL":
            item["control_sent"] = False
            item["blocked_reason"] = "missing RUN_CONTROL confirmation"
            steps.append(item)
            continue

        control = call_fixed_action_once(
            arm.controller,
            arm.robot,
            side=side,
            action_data=action_data,
            reference_time=reference_time,
        )
        item["control_sent"] = True
        item["control_result"] = control
        if control.get("ok"):
            time.sleep(max(0.0, float(args.execute_s)))
        if args.settle_s > 0.0:
            time.sleep(max(0.0, float(args.settle_s)))

        after_T = arm.get_T_link7_in_base()
        observed_delta = after_T[:3, 3] - before_T[:3, 3]
        observed_rot_deg = rotation_angle_deg(after_T[:3, :3] @ before_T[:3, :3].T)
        tracking = translation_tracking_report(target_translation, observed_delta)
        target_norm = float(tracking.get("target_norm_m") or 0.0)
        observed_norm = float(tracking.get("observed_norm_m") or 0.0)
        tracking["norm_ratio"] = observed_norm / target_norm if target_norm > EPS else None
        item.update(
            {
                "after_T_link7_in_base": after_T.tolist(),
                "observed_delta_m": observed_delta.tolist(),
                "observed_delta_norm_m": float(np.linalg.norm(observed_delta)),
                "observed_rotation_delta_deg": observed_rot_deg,
                "observed_rotation_vector_base_deg": rotation_vector_from_delta(
                    after_T[:3, :3] @ before_T[:3, :3].T
                ).tolist(),
                "translation_tracking": tracking,
                "rotation_error_deg_abs": abs(observed_rot_deg - target_rotation_deg),
                "return_to_start_delta_m": (after_T[:3, 3] - start_T[:3, 3]).tolist(),
            }
        )
        online_delta = online_observed.get("settled_delta_m")
        if online_delta is not None:
            online_delta_arr = np.asarray(online_delta, dtype=np.float64)
            item["online_vs_replay"] = {
                "online_settled_delta_m": online_delta_arr.tolist(),
                "replay_observed_delta_m": observed_delta.tolist(),
                "delta_difference_m": (observed_delta - online_delta_arr).tolist(),
                "delta_difference_norm_m": float(np.linalg.norm(observed_delta - online_delta_arr)),
                "online_settled_delta_norm_m": online_observed.get("settled_delta_norm_m"),
                "replay_observed_delta_norm_m": float(np.linalg.norm(observed_delta)),
                "online_settled_rotation_delta_deg": online_observed.get("settled_rotation_delta_deg"),
                "replay_observed_rotation_delta_deg": observed_rot_deg,
                "rotation_difference_deg": (
                    None
                    if online_observed.get("settled_rotation_delta_deg") is None
                    else observed_rot_deg - float(online_observed.get("settled_rotation_delta_deg"))
                ),
            }
        ratio = tracking.get("norm_ratio")
        cos = tracking.get("cosine_to_target_delta")
        log(
            f"replay step {idx}: observed={observed_delta.tolist()} "
            f"norm={item['observed_delta_norm_m']:.4f}m "
            f"ratio={ratio} cos={cos} rot={observed_rot_deg:.2f}deg"
        )
        print(
            "RESULT "
            f"step={idx} online_step={action.get('online_step_idx')} "
            f"target_trans={item['target_translation_norm_m']:.4f}m "
            f"actual_trans={item['observed_delta_norm_m']:.4f}m "
            f"ratio={ratio if ratio is not None else 'n/a'} "
            f"cos={cos if cos is not None else 'n/a'} "
            f"target_rot={target_rotation_deg:.2f}deg "
            f"actual_rot={observed_rot_deg:.2f}deg "
            f"rot_abs_err={item['rotation_error_deg_abs']:.2f}deg"
        )
        steps.append(item)
        if not control.get("ok"):
            break
    return {
        "start_T_link7_in_base": start_T.tolist(),
        "end_T_link7_in_base": arm.get_T_link7_in_base().tolist(),
        "steps": steps,
    }


def build_summary(report: dict[str, Any]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for item in report.get("steps") or []:
        tracking = item.get("translation_tracking") or {}
        online_vs = item.get("online_vs_replay") or {}
        summary.append(
            {
                "idx": item.get("idx"),
                "online_step_idx": item.get("online_step_idx"),
                "control_sent": item.get("control_sent"),
                "target_translation_m": item.get("target_translation_m"),
                "target_translation_norm_m": item.get("target_translation_norm_m"),
                "observed_delta_m": item.get("observed_delta_m"),
                "observed_delta_norm_m": item.get("observed_delta_norm_m"),
                "translation_ratio": tracking.get("norm_ratio"),
                "translation_cosine": tracking.get("cosine_to_target_delta"),
                "target_rotation_deg_from_action": item.get("target_rotation_deg_from_action"),
                "observed_rotation_delta_deg": item.get("observed_rotation_delta_deg"),
                "rotation_error_deg_abs": item.get("rotation_error_deg_abs"),
                "online_replay_delta_difference_norm_m": online_vs.get("delta_difference_norm_m"),
                "online_settled_delta_norm_m": online_vs.get("online_settled_delta_norm_m"),
                "rotation_difference_deg": online_vs.get("rotation_difference_deg"),
            }
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence_json")
    parser.add_argument("--out-dir", default=str(artifact_dir("diagnostics")))
    parser.add_argument("--tag", default="humanego_action_replay")
    parser.add_argument("--side", choices=["right", "left"], default="")
    parser.add_argument("--control-mode", choices=["prompt", "auto"], default="prompt")
    parser.add_argument("--confirm-control", default="")
    parser.add_argument("--max-actions", type=int, default=0)
    parser.add_argument("--reference-time", type=float, default=None)
    parser.add_argument("--fallback-reference-time", type=float, default=1.0)
    parser.add_argument("--execute-s", type=float, default=1.0)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--upload-url", default="")
    parser.add_argument("--upload-timeout-s", type=float, default=20.0)
    args = parser.parse_args()

    sequence = load_sequence(args.sequence_json)
    out_base = Path(args.out_dir).expanduser().resolve()
    default_base = artifact_dir("diagnostics")
    if out_base == default_base:
        run_dir = artifact_run_dir("diagnostics", args.tag, prefix="humanego_action_replay")
    else:
        run_dir = out_base / f"g1_humanego_action_replay_{utc_stamp()}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "ok": False,
        "args": vars(args),
        "sequence_path": sequence.get("_path"),
        "sequence_meta": {k: v for k, v in sequence.items() if k not in {"actions", "_path"}},
        "control_sent": False,
        "note": "Replays fixed HumanEgo SDK DELTA_POSE action_data. Requires --confirm-control RUN_CONTROL to move.",
    }
    arm = None
    try:
        from G1RobotArm import G1RobotArmReadOnly

        arm = G1RobotArmReadOnly(side=args.side or sequence.get("side", "right"))
        result = execute_replay(args, arm, sequence)
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
            log("skipping read-only arm adapter close for HumanEgo action replay")

    report_path = run_dir / "humanego_action_replay_report.json"
    summary_path = run_dir / "humanego_action_replay_summary.json"
    report_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    summary = build_summary(report)
    summary_path.write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
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
