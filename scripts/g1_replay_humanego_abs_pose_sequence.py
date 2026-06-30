#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay recorded HumanEgo link7 targets through direct G1 SDK ABS_POSE.

This is the direct-SDK counterpart of the CoRobot EEF_ABS replay probe. It
consumes a prior interactive report, action replay sequence, rollout JSON, or a
run directory containing one of those files. For each recorded
target_T_link7_in_base it sends an ABS_POSE trajectory interpolated from the
current measured link7 pose to the recorded absolute target.
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
for path in (PROJECT_ROOT, PROJECT_ROOT / "inference", PROJECT_ROOT / "scripts"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from g1_artifacts import artifact_dir, run_dir as artifact_run_dir  # noqa: E402
from g1_humanego_client_dry_run import json_safe, log, upload_zip  # noqa: E402
from g1_humanego_interactive_step_client import rotation_angle_deg, translation_tracking_report  # noqa: E402
from g1_replay_humanego_eef_abs_corobot import (  # noqa: E402
    extract_targets,
    matrix_from_any,
    row_from_T,
)
from g1_verify_abs_pose_sequence import (  # noqa: E402
    call_abs_pose_trajectory_once,
    evaluate_motion,
    interpolate_Ts,
    opposite_side,
    read_link7_T_from_motion_status,
)


EPS = 1e-12
TARGET_MODES = ("full", "position_only", "orientation_only")


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


def load_replay_json(path: str) -> Any:
    p = Path(path).expanduser().resolve()
    if p.is_dir():
        candidates = [
            p / "humanego_abs_pose_replay_targets.json",
            p / "humanego_abs_pose_replay_report.json",
            p / "humanego_action_replay_sequence.json",
            p / "interactive_step_report.json",
            p / "autoregressive_rollout.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                p = candidate
                break
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data["_path"] = str(p)
    elif isinstance(data, list):
        data = {"targets": data, "_path": str(p)}
    else:
        raise ValueError(f"expected JSON object or list in {p}, got {type(data).__name__}")
    return data


def compare_to_recorded_before(target: dict[str, Any], replay_before_T: np.ndarray, target_T: np.ndarray) -> dict[str, Any]:
    recorded_before = target.get("before_T_link7_in_base")
    out: dict[str, Any] = {"available": recorded_before is not None}
    if recorded_before is None:
        return out
    recorded_before_T = matrix_from_any(recorded_before, "recorded_before_T_link7_in_base")
    recorded_delta = target_T[:3, 3] - recorded_before_T[:3, 3]
    replay_delta = target_T[:3, 3] - replay_before_T[:3, 3]
    out.update(
        {
            "recorded_before_T_link7_in_base": recorded_before_T.tolist(),
            "recorded_target_delta_m": recorded_delta.tolist(),
            "recorded_target_delta_norm_m": float(np.linalg.norm(recorded_delta)),
            "replay_target_delta_m": replay_delta.tolist(),
            "replay_target_delta_norm_m": float(np.linalg.norm(replay_delta)),
            "before_pose_position_difference_m": float(np.linalg.norm(replay_before_T[:3, 3] - recorded_before_T[:3, 3])),
            "before_pose_rotation_difference_deg": rotation_angle_deg(
                replay_before_T[:3, :3] @ recorded_before_T[:3, :3].T
            ),
            "recorded_target_rotation_delta_deg": rotation_angle_deg(
                target_T[:3, :3] @ recorded_before_T[:3, :3].T
            ),
            "replay_target_rotation_delta_deg": rotation_angle_deg(
                target_T[:3, :3] @ replay_before_T[:3, :3].T
            ),
        }
    )
    return out


def command_target_from_model(before_T: np.ndarray, model_target_T: np.ndarray, target_mode: str) -> np.ndarray:
    before_T = np.asarray(before_T, dtype=np.float64).reshape(4, 4)
    model_target_T = np.asarray(model_target_T, dtype=np.float64).reshape(4, 4)
    if target_mode == "full":
        return model_target_T.copy()
    command_T = before_T.copy()
    if target_mode == "position_only":
        command_T[:3, 3] = model_target_T[:3, 3]
        return command_T
    if target_mode == "orientation_only":
        command_T[:3, :3] = model_target_T[:3, :3]
        return command_T
    raise ValueError(f"unknown target_mode {target_mode!r}; expected one of {TARGET_MODES}")


def normalize_target_item(item: dict[str, Any], *, side: str, seq_idx: int) -> dict[str, Any] | None:
    T_raw = item.get("target_T_link7_in_base") or item.get("model_target_T_link7_in_base")
    if T_raw is None:
        return None
    out = dict(item)
    out["seq_idx"] = out.get("seq_idx", seq_idx)
    out["side"] = out.get("side") or side
    out["target_T_link7_in_base"] = matrix_from_any(T_raw, "target_T_link7_in_base").tolist()
    before_raw = out.get("before_T_link7_in_base")
    if before_raw is not None:
        out["before_T_link7_in_base"] = matrix_from_any(before_raw, "before_T_link7_in_base").tolist()
    return out


def extract_replay_targets(data: Any, *, side: str, max_actions: int) -> list[dict[str, Any]]:
    if isinstance(data, list):
        candidates = data
    elif isinstance(data, dict) and isinstance(data.get("targets"), list):
        candidates = data["targets"]
    else:
        return extract_targets(data, side=side, max_actions=max_actions)

    targets: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        normalized = normalize_target_item(item, side=side, seq_idx=len(targets))
        if normalized is None:
            continue
        targets.append(normalized)
        if max_actions > 0 and len(targets) >= max_actions:
            break
    if not targets:
        raise RuntimeError("no target_T_link7_in_base entries found in replay target list")
    return targets


def build_step_summary(step: dict[str, Any]) -> dict[str, Any]:
    tracking = ((step.get("tracking") or {}).get("translation_tracking") or {})
    inactive_tracking = step.get("inactive_tracking") or {}
    recorded_compare = step.get("recorded_before_compare") or {}
    online_observed = step.get("online_observed") or {}
    return {
        "idx": step.get("idx"),
        "online_step_idx": step.get("online_step_idx"),
        "request_id": step.get("request_id"),
        "target_mode": step.get("target_mode"),
        "control_sent": step.get("control_sent"),
        "control_ok": (step.get("control_result") or {}).get("ok"),
        "target_delta_norm_m": (step.get("tracking") or {}).get("commanded_delta_norm_m"),
        "model_target_delta_norm_m": step.get("model_target_delta_from_current_norm_m"),
        "command_target_delta_norm_m": step.get("command_target_delta_from_current_norm_m"),
        "actual_delta_norm_m": (step.get("tracking") or {}).get("observed_delta_norm_m"),
        "translation_ratio": tracking.get("norm_ratio"),
        "translation_cosine": tracking.get("cosine_to_target_delta"),
        "final_pose_position_error_m": (step.get("tracking") or {}).get("final_pose_position_error_m"),
        "target_rotation_delta_deg": (step.get("tracking") or {}).get("target_rotation_delta_deg"),
        "model_target_rotation_delta_deg": step.get("model_target_rotation_from_current_deg"),
        "command_target_rotation_delta_deg": step.get("command_target_rotation_from_current_deg"),
        "actual_rotation_delta_deg": (step.get("tracking") or {}).get("observed_rotation_delta_deg"),
        "final_pose_rotation_error_deg": (step.get("tracking") or {}).get("final_pose_rotation_error_deg"),
        "final_model_pose_position_error_m": step.get("final_model_pose_position_error_m"),
        "final_model_pose_rotation_error_deg": step.get("final_model_pose_rotation_error_deg"),
        "inactive_drift_m": inactive_tracking.get("observed_delta_norm_m"),
        "inactive_rotation_deg": inactive_tracking.get("observed_rotation_delta_deg"),
        "recorded_before_position_difference_m": recorded_compare.get("before_pose_position_difference_m"),
        "recorded_target_delta_norm_m": recorded_compare.get("recorded_target_delta_norm_m"),
        "online_settled_delta_norm_m": online_observed.get("settled_delta_norm_m"),
        "online_settled_rotation_delta_deg": online_observed.get("settled_rotation_delta_deg"),
    }


def execute_replay(args: argparse.Namespace, arm: Any, targets: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    from G1RobotArm import parse_motion_pose, wait_motion_status

    inactive_side = opposite_side(args.side)
    steps: list[dict[str, Any]] = []
    start_T = arm.get_T_link7_in_base()
    inactive_start_T = read_link7_T_from_motion_status(
        arm.controller,
        inactive_side,
        arm.motion_tries,
        arm.motion_sleep_s,
        wait_motion_status,
        parse_motion_pose,
    )
    summaries_jsonl_path = run_dir / "step_summaries.jsonl"

    for idx, target in enumerate(targets):
        model_target_T = matrix_from_any(target["target_T_link7_in_base"], "target_T_link7_in_base")
        before_T = arm.get_T_link7_in_base()
        inactive_before_T = read_link7_T_from_motion_status(
            arm.controller,
            inactive_side,
            arm.motion_tries,
            arm.motion_sleep_s,
            wait_motion_status,
            parse_motion_pose,
        )
        command_target_T = command_target_from_model(before_T, model_target_T, args.target_mode)
        target_Ts = interpolate_Ts(before_T, command_target_T, args.interp_points)
        active_rows = [row_from_T(T) for T in target_Ts]
        inactive_target_Ts = [inactive_before_T.copy() for _ in target_Ts]
        inactive_rows = [row_from_T(T) for T in inactive_target_Ts]
        model_target_delta = model_target_T[:3, 3] - before_T[:3, 3]
        command_target_delta = command_target_T[:3, 3] - before_T[:3, 3]
        model_target_rot = rotation_angle_deg(model_target_T[:3, :3] @ before_T[:3, :3].T)
        command_target_rot = rotation_angle_deg(command_target_T[:3, :3] @ before_T[:3, :3].T)
        item: dict[str, Any] = {
            "idx": idx,
            "online_step_idx": target.get("online_step_idx"),
            "request_id": target.get("request_id"),
            "target_mode": args.target_mode,
            "side": args.side,
            "inactive_side": inactive_side,
            "source_target": target,
            "before_T_link7_in_base": before_T.tolist(),
            "model_target_T_link7_in_base": model_target_T.tolist(),
            "command_target_T_link7_in_base": command_target_T.tolist(),
            "target_T_link7_in_base": command_target_T.tolist(),
            "inactive_before_T_link7_in_base": inactive_before_T.tolist(),
            "num_interpolation_points": len(active_rows),
            "trajectory_reference_time": args.reference_time,
            "model_target_delta_from_current_m": model_target_delta.tolist(),
            "model_target_delta_from_current_norm_m": float(np.linalg.norm(model_target_delta)),
            "model_target_rotation_from_current_deg": model_target_rot,
            "command_target_delta_from_current_m": command_target_delta.tolist(),
            "command_target_delta_from_current_norm_m": float(np.linalg.norm(command_target_delta)),
            "command_target_rotation_from_current_deg": command_target_rot,
            "target_delta_from_current_m": command_target_delta.tolist(),
            "target_delta_from_current_norm_m": float(np.linalg.norm(command_target_delta)),
            "target_rotation_from_current_deg": command_target_rot,
            "recorded_before_compare": compare_to_recorded_before(target, before_T, model_target_T),
            "online_observed": target.get("online_observed") or {},
        }
        log(
            f"ABS_POSE replay step {idx} online_step={target.get('online_step_idx')}: "
            f"mode={args.target_mode} "
            f"command_delta_norm={item['command_target_delta_from_current_norm_m']:.4f}m "
            f"command_rot={command_target_rot:.2f}deg rows={len(active_rows)}"
        )
        print(
            "\n=== ABS_POSE replay step "
            f"{idx}/{len(targets) - 1} ===\n"
            f"online_step: {target.get('online_step_idx')}\n"
            f"target_mode: {args.target_mode}\n"
            f"model_delta_m: {model_target_delta.tolist()} norm={item['model_target_delta_from_current_norm_m']:.4f}\n"
            f"model_rotation_delta_deg: {model_target_rot:.2f}\n"
            f"command_delta_m: {command_target_delta.tolist()} norm={item['command_target_delta_from_current_norm_m']:.4f}\n"
            f"command_rotation_delta_deg: {command_target_rot:.2f}\n"
            f"interp_points: {len(active_rows)} reference_time={args.reference_time:.3f}s\n"
        )

        if args.control_mode == "prompt":
            try:
                operator = input("[Enter]=execute ABS_POSE target, s=skip, q=quit > ").strip().lower()
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
                summaries_jsonl_path.open("a", encoding="utf-8").write(
                    json.dumps(json_safe(build_step_summary(item)), ensure_ascii=False) + "\n"
                )
                continue
        else:
            item["operator_input"] = args.control_mode

        if args.confirm_control != "RUN_CONTROL":
            item["control_sent"] = False
            item["blocked_reason"] = "missing RUN_CONTROL confirmation"
            steps.append(item)
            break

        control = call_abs_pose_trajectory_once(
            arm.controller,
            arm.robot,
            args.side,
            active_rows,
            inactive_rows,
            args.reference_time,
        )
        item["control_sent"] = True
        item["control_result"] = control
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
        item["after_T_link7_in_base"] = after_T.tolist()
        item["inactive_after_T_link7_in_base"] = inactive_after_T.tolist()
        item["tracking"] = evaluate_motion(before_T, command_target_T, after_T)
        item["inactive_tracking"] = evaluate_motion(inactive_before_T, inactive_before_T, inactive_after_T)
        item["final_model_pose_position_error_m"] = float(np.linalg.norm(model_target_T[:3, 3] - after_T[:3, 3]))
        item["final_model_pose_rotation_error_deg"] = rotation_angle_deg(model_target_T[:3, :3] @ after_T[:3, :3].T)
        tracking = item["tracking"]["translation_tracking"]
        inactive_tracking = item["inactive_tracking"]
        summary = build_step_summary(item)
        summaries_jsonl_path.open("a", encoding="utf-8").write(
            json.dumps(json_safe(summary), ensure_ascii=False) + "\n"
        )
        print(
            "RESULT "
            f"step={idx} online_step={target.get('online_step_idx')} "
            f"mode={args.target_mode} "
            f"target_trans={item['tracking']['commanded_delta_norm_m']:.4f}m "
            f"actual_trans={item['tracking']['observed_delta_norm_m']:.4f}m "
            f"ratio={tracking.get('norm_ratio')} "
            f"cos={tracking.get('cosine_to_target_delta')} "
            f"final_pos_err={item['tracking']['final_pose_position_error_m']:.4f}m "
            f"target_rot={item['tracking']['target_rotation_delta_deg']:.2f}deg "
            f"actual_rot={item['tracking']['observed_rotation_delta_deg']:.2f}deg "
            f"final_rot_err={item['tracking']['final_pose_rotation_error_deg']:.2f}deg "
            f"inactive_{inactive_side}_drift={inactive_tracking['observed_delta_norm_m']:.4f}m"
        )
        steps.append(item)
        if not control.get("ok"):
            break

    return {
        "start_T_link7_in_base": start_T.tolist(),
        "inactive_start_T_link7_in_base": inactive_start_T.tolist(),
        "end_T_link7_in_base": arm.get_T_link7_in_base().tolist(),
        "inactive_end_T_link7_in_base": read_link7_T_from_motion_status(
            arm.controller,
            inactive_side,
            arm.motion_tries,
            arm.motion_sleep_s,
            wait_motion_status,
            parse_motion_pose,
        ).tolist(),
        "steps": steps,
        "step_summaries_jsonl_path": str(summaries_jsonl_path),
    }


def build_summary(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [build_step_summary(step) for step in report.get("steps") or []]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory_json", help="interactive report, action sequence, rollout JSON, or run directory")
    parser.add_argument("--out-dir", default=str(artifact_dir("diagnostics")))
    parser.add_argument("--tag", default="humanego_abs_pose_replay")
    parser.add_argument("--side", choices=["right", "left"], default="right")
    parser.add_argument("--control-mode", choices=["prompt", "auto", "dry-run"], default="prompt")
    parser.add_argument("--confirm-control", default="")
    parser.add_argument("--max-actions", type=int, default=10)
    parser.add_argument("--target-mode", choices=TARGET_MODES, default="full")
    parser.add_argument("--interp-points", type=int, default=30)
    parser.add_argument("--reference-time", type=float, default=2.0)
    parser.add_argument("--execute-s", type=float, default=2.0)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--upload-url", default="")
    parser.add_argument("--upload-timeout-s", type=float, default=20.0)
    args = parser.parse_args()

    source = load_replay_json(args.trajectory_json)
    targets = extract_replay_targets(source, side=args.side, max_actions=args.max_actions)
    out_base = Path(args.out_dir).expanduser().resolve()
    default_base = artifact_dir("diagnostics")
    if out_base == default_base:
        run_dir = artifact_run_dir("diagnostics", args.tag, prefix="humanego_abs_pose_replay")
    else:
        run_dir = out_base / f"g1_humanego_abs_pose_replay_{utc_stamp()}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "ok": False,
        "args": vars(args),
        "source_path": source.get("_path"),
        "num_targets": len(targets),
        "targets": targets,
        "control_sent": False,
        "note": (
            "Replays recorded HumanEgo target_T_link7_in_base values through direct "
            "G1 SDK trajectory_tracking_control ABS_POSE. Each model target is "
            "converted according to --target-mode, then interpolated from current "
            "measured link7 pose; inactive arm is held at its current ABS_POSE target."
        ),
    }
    targets_path = run_dir / "humanego_abs_pose_replay_targets.json"
    targets_path.write_text(json.dumps(json_safe(targets), ensure_ascii=False, indent=2), encoding="utf-8")
    report["targets_path"] = str(targets_path)

    arm = None
    try:
        print(
            "\n=== HumanEgo ABS_POSE SDK replay preview ===\n"
            f"source: {source.get('_path')}\n"
            f"side: {args.side}\n"
            f"target_mode: {args.target_mode}\n"
            f"targets: {len(targets)}\n"
            f"interp_points_per_target: {args.interp_points}\n"
            f"reference_time_per_target: {args.reference_time:.3f}s\n"
            f"targets_json: {targets_path}\n"
        )
        if args.control_mode == "dry-run":
            report["ok"] = True
            report["blocked_reason"] = "dry_run_only"
        else:
            from G1RobotArm import G1RobotArmReadOnly

            arm = G1RobotArmReadOnly(side=args.side)
            result = execute_replay(args, arm, targets, run_dir)
            report.update(result)
            report["control_sent"] = any(step.get("control_sent") for step in result["steps"])
            sent_steps = [step for step in result["steps"] if step.get("control_sent")]
            report["ok"] = bool(sent_steps) and all((step.get("control_result") or {}).get("ok") for step in sent_steps)
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
            log("skipping read-only arm adapter close for HumanEgo ABS_POSE replay")

    report_path = run_dir / "humanego_abs_pose_replay_report.json"
    summary_path = run_dir / "humanego_abs_pose_replay_summary.json"
    step_summaries_path = run_dir / "step_summaries.json"
    report_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    summary = build_summary(report)
    summary_path.write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    step_summaries_path.write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
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
                "step_summaries_path": str(step_summaries_path),
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
