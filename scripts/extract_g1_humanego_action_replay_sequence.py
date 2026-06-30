#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract replayable SDK DELTA_POSE actions from a HumanEgo interactive report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def resolve_report_path(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if p.is_dir():
        p = p / "interactive_step_report.json"
    if not p.exists():
        raise FileNotFoundError(f"interactive report not found: {p}")
    return p


def get_nested(mapping: dict[str, Any], keys: list[Any], default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        elif isinstance(value, list) and isinstance(key, int) and 0 <= key < len(value):
            value = value[key]
        else:
            return default
        if value is None:
            return default
    return value


def sdk_action_data(step: dict[str, Any], side: str) -> list[float] | None:
    control = step.get("control_result") or {}
    arm_key = f"{side}_arm"
    from_kwargs = get_nested(control, ["kwargs", "robot_actions", 0, arm_key, "action_data"])
    if from_kwargs is not None:
        return [float(v) for v in from_kwargs]
    delta_pose = control.get("delta_pose") or {}
    action_data = delta_pose.get("action_data")
    if action_data is not None:
        return [float(v) for v in action_data]
    return None


def extract_actions(report: dict[str, Any], *, side: str, max_actions: int) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for step in report.get("steps") or []:
        if not step.get("executed"):
            continue
        control = step.get("control_result") or {}
        if control.get("mode") != "delta_pose":
            continue
        action_data = sdk_action_data(step, side)
        if not action_data or len(action_data) != 6:
            continue
        delta_pose = control.get("delta_pose") or {}
        kwargs = control.get("kwargs") or {}
        online_tracking = step.get("settled_translation_tracking") or {}
        item = {
            "seq_idx": len(actions),
            "online_step_idx": step.get("idx"),
            "request_id": step.get("request_id"),
            "side": side,
            "action_data": action_data,
            "rotation_frame": delta_pose.get("rotation_frame") or "base",
            "trajectory_reference_time": control.get("trajectory_reference_time")
            or kwargs.get("trajectory_reference_time"),
            "online_target": {
                "target_delta_m": step.get("target_delta_m"),
                "target_delta_norm_m": step.get("target_delta_norm_m"),
                "target_rotation_delta_deg": step.get("target_rotation_delta_deg"),
                "before_T_link7_in_base": step.get("before_T_link7_in_base"),
                "target_T_link7_in_base": step.get("target_T_link7_in_base"),
                "raw_target_T_link7_in_base": step.get("raw_target_T_link7_in_base"),
            },
            "online_observed": {
                "post_ee_delta_m": step.get("post_ee_delta_m"),
                "post_ee_delta_norm_m": step.get("post_ee_delta_norm_m"),
                "post_ee_rotation_delta_deg": step.get("post_ee_rotation_delta_deg"),
                "settled_delta_m": step.get("settled_delta_m"),
                "settled_delta_norm_m": step.get("settled_delta_norm_m"),
                "settled_rotation_delta_deg": step.get("observed_rotation_delta_deg"),
                "settled_cos_to_target": online_tracking.get("cosine_to_target_delta"),
                "settled_ratio": online_tracking.get("norm_ratio"),
                "after_T_link7_in_base": step.get("after_T_link7_in_base"),
            },
            "online_control": {
                "ok": control.get("ok"),
                "send_mode": control.get("send_mode"),
                "execute_s": control.get("execute_s"),
                "duration_s": control.get("duration_s"),
                "command_duration_s": control.get("command_duration_s"),
                "delta_pose": delta_pose,
                "trajectory_reference_time": kwargs.get("trajectory_reference_time"),
            },
        }
        actions.append(item)
        if max_actions > 0 and len(actions) >= max_actions:
            break
    return actions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("interactive_report", help="interactive_step_report.json or its run directory")
    parser.add_argument("--side", choices=["right", "left"], default="right")
    parser.add_argument("--max-actions", type=int, default=10)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    report_path = resolve_report_path(args.interactive_report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    actions = extract_actions(report, side=args.side, max_actions=args.max_actions)
    if not actions:
        raise RuntimeError(f"no executed delta_pose actions found in {report_path}")

    out_path = Path(args.out).expanduser().resolve() if args.out else report_path.parent / "humanego_action_replay_sequence.json"
    payload = {
        "format": "g1_humanego_action_replay_sequence.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_report": str(report_path),
        "source_run_dir": str(report_path.parent),
        "side": args.side,
        "num_actions": len(actions),
        "source_args": report.get("args"),
        "note": (
            "Replay uses action_data as the fixed control value. The replay script "
            "will read fresh robot_states at execution time."
        ),
        "actions": actions,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out_path), "num_actions": len(actions)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

