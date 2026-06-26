#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify G1 gripper control with one small, logged command.

Default mode is read-only. Control commands are sent only when both conditions
are met:
  --mode hold|delta|target
  --confirm-control RUN_CONTROL
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import sys
import time
import traceback
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from g1_artifacts import artifact_dir, run_dir as artifact_run_dir  # noqa: E402


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": type(value).__name__, "num_bytes": len(value)}
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_zip(src_dir: Path) -> Path:
    zip_path = src_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir.parent))
    return zip_path


def upload_zip(zip_path: Path, upload_url: str, timeout_s: float = 20.0) -> Dict[str, Any]:
    data = zip_path.read_bytes()
    req = urllib.request.Request(
        upload_url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/zip",
            "Content-Length": str(len(data)),
            "X-G1-Diagnostics-Filename": zip_path.name,
            "Connection": "close",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return {
            "ok": True,
            "status": resp.status,
            "response": resp.read().decode("utf-8", errors="replace"),
        }


def call_and_capture(name: str, fn, *args, **kwargs) -> Dict[str, Any]:
    started = time.time()
    item: Dict[str, Any] = {"ok": False, "name": name, "started_unix": started}
    try:
        value = fn(*args, **kwargs)
        item.update(
            {
                "ok": True,
                "duration_s": time.time() - started,
                "value": json_safe(value),
            }
        )
    except Exception as exc:
        item.update(
            {
                "ok": False,
                "duration_s": time.time() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    return item


def split_state_result(value: Any) -> tuple[Any, Any]:
    if isinstance(value, tuple) and len(value) == 2:
        return value[0], value[1]
    return value, None


def coerce_float_list(value: Any) -> List[float]:
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


def read_gripper_state(robot: Any, side: str) -> Dict[str, Any]:
    raw_result = robot.gripper_states()
    data, timestamp = split_state_result(raw_result)
    values = coerce_float_list(raw_result)
    selected_indices = selected_gripper_indices(side, len(values))
    selected = {str(idx): values[idx] for idx in selected_indices}
    return {
        "raw_result": json_safe(raw_result),
        "data": json_safe(data),
        "timestamp": json_safe(timestamp),
        "values": values,
        "side": side,
        "selected_indices": selected_indices,
        "selected_values": selected,
    }


def selected_gripper_indices(side: str, num_values: int) -> List[int]:
    if num_values <= 0:
        raise ValueError("cannot select gripper index from empty values")
    if side == "both":
        return list(range(num_values))
    if side == "left":
        return [0]
    return [min(1, num_values - 1)]


def clip_value(value: float, low: float, high: float) -> float:
    return min(max(float(value), float(low)), float(high))


def state_to_command_value(value: float, state_max_raw: float) -> float:
    value = float(value)
    state_max_raw = float(state_max_raw)
    if state_max_raw > 1.0 and abs(value) > 1.0:
        return value / state_max_raw
    return value


def command_to_state_estimate(value: float, state_max_raw: float) -> float:
    state_max_raw = float(state_max_raw)
    if state_max_raw > 1.0:
        return float(value) * state_max_raw
    return float(value)


def build_command_values(before_values: List[float], args: argparse.Namespace) -> tuple[List[float], Dict[str, Any]]:
    base_command_values = [state_to_command_value(v, args.state_max_raw) for v in before_values]
    command_values = list(base_command_values)
    indices = selected_gripper_indices(args.side, len(command_values))
    info: Dict[str, Any] = {
        "mode": args.mode,
        "side": args.side,
        "selected_indices": indices,
        "before_state_values": list(before_values),
        "base_command_values": list(base_command_values),
        "state_max_raw": float(args.state_max_raw),
        "clip": bool(args.clip),
        "min_raw": args.min_raw,
        "max_raw": args.max_raw,
    }

    if args.mode == "observe":
        info["target_values"] = list(command_values)
        return command_values, info
    if args.mode == "hold":
        info["target_values"] = list(command_values)
        return command_values, info
    if args.mode == "target":
        if args.target_raw is None:
            raise ValueError("--target-raw is required when --mode target")
        for idx in indices:
            command_values[idx] = float(args.target_raw)
        info["target_raw"] = float(args.target_raw)
    elif args.mode == "delta":
        for idx in indices:
            command_values[idx] = command_values[idx] + float(args.delta_raw)
        info["delta_raw"] = float(args.delta_raw)
    else:
        raise ValueError(f"unknown mode: {args.mode}")

    unclipped = list(command_values)
    if args.clip:
        command_values = [clip_value(v, args.min_raw, args.max_raw) for v in command_values]
    info["unclipped_target_values"] = unclipped
    info["target_values"] = list(command_values)
    return command_values, info


def command_payload(values: List[float], before_values: List[float], args: argparse.Namespace) -> Any:
    if args.payload_format == "list":
        return list(values)
    if args.payload_format == "scalar":
        idx = selected_gripper_indices(args.side, len(values))[-1]
        return float(values[idx])
    if len(before_values) == 1:
        return float(values[0])
    return list(values)


def sample_states(robot: Any, side: str, samples: int, interval_s: float) -> List[Dict[str, Any]]:
    out = []
    for idx in range(max(1, int(samples))):
        item = read_gripper_state(robot, side)
        item["sample_idx"] = idx
        item["sample_time_unix"] = time.time()
        out.append(item)
        if idx + 1 < max(1, int(samples)):
            time.sleep(max(0.0, float(interval_s)))
    return out


def analyze_delta(
    before: Dict[str, Any],
    before_command_values: List[float],
    target_values: List[float],
    after: Dict[str, Any],
    side: str,
    state_max_raw: float,
    change_threshold: float,
) -> Dict[str, Any]:
    before_values = before["values"]
    after_values = after["values"]
    indices = selected_gripper_indices(side, min(len(before_values), len(after_values), len(target_values)))
    per_index: Dict[str, Any] = {}
    changed = False
    moved_toward_target = False
    for idx in indices:
        before_v = float(before_values[idx])
        target_v = float(target_values[idx])
        after_v = float(after_values[idx])
        before_command_v = float(before_command_values[idx])
        after_command_est = state_to_command_value(after_v, state_max_raw)
        target_state_est = command_to_state_estimate(target_v, state_max_raw)
        observed_delta = after_v - before_v
        observed_command_delta_est = after_command_est - before_command_v
        commanded_delta = target_v - before_command_v
        expected_state_delta = target_state_est - before_v
        idx_changed = abs(observed_delta) >= float(change_threshold)
        idx_toward = (
            abs(after_v - target_state_est) < abs(before_v - target_state_est)
            if abs(commanded_delta) > 1e-12
            else True
        )
        changed = changed or idx_changed
        moved_toward_target = moved_toward_target or idx_toward
        per_index[str(idx)] = {
            "before_state": before_v,
            "before_command_estimate": before_command_v,
            "target_command": target_v,
            "target_state_estimate": target_state_est,
            "after_state": after_v,
            "after_command_estimate": after_command_est,
            "commanded_delta_command": commanded_delta,
            "expected_delta_state": expected_state_delta,
            "observed_delta_state": observed_delta,
            "observed_delta_command_estimate": observed_command_delta_est,
            "target_state_error": after_v - target_state_est,
            "changed": idx_changed,
            "moved_toward_target": idx_toward,
        }
    return {
        "selected_indices": indices,
        "change_threshold": float(change_threshold),
        "per_index": per_index,
        "any_selected_changed": changed,
        "any_selected_moved_toward_target": moved_toward_target,
    }


def prompt_before_control(args: argparse.Namespace, payload: Any) -> str:
    if not args.prompt or args.mode == "observe":
        return ""
    print("\n=== G1 gripper control probe ===")
    print(f"mode: {args.mode}")
    print(f"side: {args.side}")
    print(f"payload_format: {args.payload_format}")
    print(f"move_gripper payload: {payload}")
    try:
        text = input("[Enter]=send gripper command, q=quit > ")
    except EOFError:
        return "q"
    return text.strip().lower()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(artifact_dir("diagnostics", "gripper_control")))
    parser.add_argument("--tag", default="gripper_control")
    parser.add_argument("--side", choices=["left", "right", "both"], default="right")
    parser.add_argument("--mode", choices=["observe", "hold", "delta", "target"], default="observe")
    parser.add_argument("--confirm-control", default="")
    parser.add_argument("--delta-raw", type=float, default=0.05)
    parser.add_argument("--target-raw", type=float, default=None)
    parser.add_argument("--payload-format", choices=["auto", "list", "scalar"], default="auto")
    parser.add_argument("--clip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-raw", type=float, default=0.0)
    parser.add_argument("--max-raw", type=float, default=1.0)
    parser.add_argument("--state-max-raw", type=float, default=120.0)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--sample-interval-s", type=float, default=0.1)
    parser.add_argument("--change-threshold", type=float, default=0.005)
    parser.add_argument("--prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--close-robot", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force-exit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--upload-url", default="")
    parser.add_argument("--upload-timeout-s", type=float, default=20.0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    out_base = Path(args.out_dir).expanduser().resolve()
    default_base = artifact_dir("diagnostics", "gripper_control")
    if out_base == default_base:
        run_dir = artifact_run_dir("diagnostics", args.tag, prefix="gripper_control")
    else:
        run_dir = out_base / f"g1_gripper_control_{utc_stamp()}_{args.tag}"
    ensure_dir(run_dir)

    report: Dict[str, Any] = {
        "ok": False,
        "control_sent": False,
        "args": vars(args),
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "cwd": os.getcwd(),
            "project_root": str(PROJECT_ROOT),
            "argv": sys.argv,
        },
        "note": "Observe mode is read-only. Hold/delta/target modes require --confirm-control RUN_CONTROL.",
    }
    robot = None
    exit_code = 1
    try:
        from a2d_sdk.robot import RobotDds

        robot = RobotDds()
        time.sleep(0.2)
        report["robot"] = {
            "class": repr(type(robot)),
            "gripper_states_signature": str(inspect.signature(robot.gripper_states)),
            "move_gripper_signature": str(inspect.signature(robot.move_gripper)),
        }

        before = read_gripper_state(robot, args.side)
        report["before"] = before
        report["pre_samples"] = sample_states(robot, args.side, args.samples, args.sample_interval_s)
        command_values, command_info = build_command_values(before["values"], args)
        payload = command_payload(command_values, before["values"], args)
        report["command"] = {
            **command_info,
            "payload_format": args.payload_format,
            "payload": json_safe(payload),
        }

        if args.mode == "observe":
            report.update({"ok": True, "result": "read_only_observe"})
            exit_code = 0
        elif args.confirm_control != "RUN_CONTROL":
            report.update(
                {
                    "ok": False,
                    "control_sent": False,
                    "error": "Refusing to send gripper control without --confirm-control RUN_CONTROL.",
                }
            )
        else:
            operator = prompt_before_control(args, payload)
            report["operator_input"] = operator
            if operator == "q":
                report.update({"ok": False, "control_sent": False, "result": "operator_quit"})
            else:
                report["control_sent"] = True
                report["move_gripper_call"] = call_and_capture("move_gripper", robot.move_gripper, payload)
                time.sleep(max(0.0, float(args.settle_s)))
                report["after_samples"] = sample_states(robot, args.side, args.samples, args.sample_interval_s)
                after = report["after_samples"][-1]
                report["after"] = after
                report["delta_analysis"] = analyze_delta(
                    before,
                    command_info["base_command_values"],
                    command_values,
                    after,
                    args.side,
                    args.state_max_raw,
                    args.change_threshold,
                )
                call_ok = bool(report["move_gripper_call"].get("ok"))
                if args.mode == "hold":
                    report.update({"ok": call_ok, "result": "hold_command_complete"})
                else:
                    report.update({"ok": call_ok, "result": "gripper_command_complete"})
                exit_code = 0 if call_ok else 1

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
        if robot is not None and args.close_robot:
            report["cleanup"] = {"robot_shutdown": call_and_capture("robot.shutdown", robot.shutdown)}
        elif robot is not None:
            report["cleanup"] = {"robot_shutdown": {"skipped": True, "reason": "--close-robot not set"}}

        report_path = run_dir / "gripper_control_report.json"
        report_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
        zip_path = make_zip(run_dir)
        upload = None
        if args.upload_url:
            try:
                upload = upload_zip(zip_path, args.upload_url, args.upload_timeout_s)
            except Exception as exc:
                upload = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            report["upload"] = upload
            report_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
            zip_path = make_zip(run_dir)

        summary = {
            "ok": report.get("ok"),
            "control_sent": report.get("control_sent"),
            "result": report.get("result"),
            "run_dir": str(run_dir),
            "zip_path": str(zip_path),
            "before_selected": report.get("before", {}).get("selected_values"),
            "command": report.get("command"),
            "after_selected": report.get("after", {}).get("selected_values"),
            "delta_analysis": report.get("delta_analysis"),
            "upload": upload,
            "error": report.get("error"),
        }
        print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
        sys.stdout.flush()
        sys.stderr.flush()

    return exit_code


if __name__ == "__main__":
    exit_code = main()
    if "--no-force-exit" not in sys.argv:
        os._exit(exit_code)
    raise SystemExit(exit_code)
