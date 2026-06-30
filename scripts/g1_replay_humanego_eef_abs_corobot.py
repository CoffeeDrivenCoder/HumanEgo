#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay recorded HumanEgo link7 targets as a CoRobot-style EEF_ABS trajectory.

This script intentionally does not call HumanEgo inference. It consumes a prior
interactive report or extracted replay sequence, builds a raw CoRobot action:

    {"base_link": "base_link", "right_arm": {"kind": "EEF_ABS", "values": rows}}

where each row is [x, y, z, roll, pitch, yaw] in base_link/link7 frame.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
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


EPS = 1e-12
DEFAULT_COROBOT_URLS = [
    "http://localhost:8765/skill/execute_action",
    "http://localhost:8765/execute_action",
    "http://localhost:8765/action",
]


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


def load_json(path: str) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if p.is_dir():
        candidates = [
            p / "humanego_action_replay_sequence.json",
            p / "interactive_step_report.json",
            p / "autoregressive_rollout.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                p = candidate
                break
    data = json.loads(p.read_text(encoding="utf-8"))
    data["_path"] = str(p)
    return data


def matrix_from_any(value: Any, label: str) -> np.ndarray:
    T = np.asarray(value, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{label} must be 4x4, got shape={T.shape}")
    return T


def matrix_to_rpy_zyx(R: np.ndarray) -> list[float]:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    sy = float(math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]))
    singular = sy < 1e-9
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return [float(roll), float(pitch), float(yaw)]


def row_from_T(T: np.ndarray) -> list[float]:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    rpy = matrix_to_rpy_zyx(T[:3, :3])
    return [float(T[0, 3]), float(T[1, 3]), float(T[2, 3]), *rpy]


def extract_targets_from_interactive_report(data: dict[str, Any], *, side: str, max_actions: int) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for step in data.get("steps") or []:
        if not step.get("executed"):
            continue
        T_raw = step.get("target_T_link7_in_base")
        if T_raw is None:
            T_raw = (step.get("target_pose") or {}).get("T_link7_in_base")
        if T_raw is None:
            continue
        before_raw = step.get("before_T_link7_in_base")
        targets.append(
            {
                "seq_idx": len(targets),
                "source": "interactive_step_report",
                "online_step_idx": step.get("idx"),
                "request_id": step.get("request_id"),
                "side": side,
                "target_T_link7_in_base": matrix_from_any(T_raw, "target_T_link7_in_base").tolist(),
                "before_T_link7_in_base": None if before_raw is None else matrix_from_any(before_raw, "before_T_link7_in_base").tolist(),
                "target_delta_m": step.get("target_delta_m"),
                "target_delta_norm_m": step.get("target_delta_norm_m"),
                "target_rotation_delta_deg": step.get("target_rotation_delta_deg"),
                "online_observed": {
                    "settled_delta_m": step.get("settled_delta_m"),
                    "settled_delta_norm_m": step.get("settled_delta_norm_m"),
                    "settled_rotation_delta_deg": step.get("observed_rotation_delta_deg"),
                    "after_T_link7_in_base": step.get("after_T_link7_in_base"),
                },
            }
        )
        if max_actions > 0 and len(targets) >= max_actions:
            break
    return targets


def extract_targets_from_action_sequence(data: dict[str, Any], *, side: str, max_actions: int) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for action in data.get("actions") or []:
        online_target = action.get("online_target") or {}
        T_raw = online_target.get("target_T_link7_in_base") or online_target.get("raw_target_T_link7_in_base")
        if T_raw is None:
            continue
        before_raw = online_target.get("before_T_link7_in_base")
        targets.append(
            {
                "seq_idx": len(targets),
                "source": "humanego_action_replay_sequence",
                "online_step_idx": action.get("online_step_idx"),
                "request_id": action.get("request_id"),
                "side": action.get("side") or side,
                "target_T_link7_in_base": matrix_from_any(T_raw, "target_T_link7_in_base").tolist(),
                "before_T_link7_in_base": None if before_raw is None else matrix_from_any(before_raw, "before_T_link7_in_base").tolist(),
                "target_delta_m": online_target.get("target_delta_m"),
                "target_delta_norm_m": online_target.get("target_delta_norm_m"),
                "target_rotation_delta_deg": online_target.get("target_rotation_delta_deg"),
                "source_action": action,
                "online_observed": action.get("online_observed") or {},
            }
        )
        if max_actions > 0 and len(targets) >= max_actions:
            break
    return targets


def extract_targets_from_rollout(data: dict[str, Any], *, side: str, max_actions: int) -> list[dict[str, Any]]:
    rollout = data.get("autoregressive_rollout") or data
    steps = rollout.get("steps") or rollout.get("targets") or []
    targets: list[dict[str, Any]] = []
    for step in steps:
        T_raw = (
            step.get("target_T_link7_in_base")
            or step.get("T_link7_target_in_base")
            or step.get(f"{side}_target_T_link7_in_base")
        )
        if T_raw is None:
            continue
        targets.append(
            {
                "seq_idx": len(targets),
                "source": "autoregressive_rollout",
                "online_step_idx": step.get("idx") or step.get("step"),
                "request_id": step.get("request_id"),
                "side": side,
                "target_T_link7_in_base": matrix_from_any(T_raw, "target_T_link7_in_base").tolist(),
                "source_step": step,
            }
        )
        if max_actions > 0 and len(targets) >= max_actions:
            break
    return targets


def extract_targets(data: dict[str, Any], *, side: str, max_actions: int) -> list[dict[str, Any]]:
    if "actions" in data:
        targets = extract_targets_from_action_sequence(data, side=side, max_actions=max_actions)
    elif "steps" in data:
        targets = extract_targets_from_interactive_report(data, side=side, max_actions=max_actions)
    else:
        targets = extract_targets_from_rollout(data, side=side, max_actions=max_actions)
    if not targets:
        raise RuntimeError(f"no target_T_link7_in_base entries found in {data.get('_path')}")
    return targets


def build_eef_abs_action(targets: list[dict[str, Any]], *, side: str, duration_s: float) -> dict[str, Any]:
    rows = [row_from_T(matrix_from_any(item["target_T_link7_in_base"], "target_T_link7_in_base")) for item in targets]
    return {
        "timestamps": int(time.time() * 1e9),
        "trajectory_reference_time": float(duration_s),
        "base_link": "base_link",
        f"{side}_arm": {
            "kind": "EEF_ABS",
            "values": rows,
        },
    }


def post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    data = json.dumps(json_safe(payload), ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(data)),
            "Connection": "close",
        },
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            try:
                parsed: Any = json.loads(body)
            except Exception:
                parsed = body
            return {
                "ok": 200 <= int(resp.status) < 300,
                "status": int(resp.status),
                "duration_s": time.time() - started,
                "url": url,
                "response": parsed,
            }
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "status": int(exc.code),
            "duration_s": time.time() - started,
            "url": url,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "response": exc.read().decode("utf-8", errors="replace"),
        }
    except Exception as exc:
        return {
            "ok": False,
            "duration_s": time.time() - started,
            "url": url,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def call_corobot_action(urls: list[str], action: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    payload_variants = [
        {"action": action, "duration_s": action.get("trajectory_reference_time")},
        action,
    ]
    attempts: list[dict[str, Any]] = []
    for url in urls:
        for payload_idx, payload in enumerate(payload_variants):
            result = post_json(url, payload, timeout_s)
            result["payload_variant"] = payload_idx
            attempts.append(result)
            if result.get("ok"):
                return {
                    "ok": True,
                    "selected_url": url,
                    "selected_payload_variant": payload_idx,
                    "attempts": attempts,
                }
    return {"ok": False, "attempts": attempts}


def call_corobot_env_action(action: dict[str, Any], wait_action_time: float) -> dict[str, Any]:
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


def evaluate_final_tracking(
    start_T: np.ndarray | None,
    final_T: np.ndarray | None,
    expected_T: np.ndarray,
) -> dict[str, Any]:
    out: dict[str, Any] = {"available": start_T is not None and final_T is not None}
    if start_T is None or final_T is None:
        return out
    start_T = np.asarray(start_T, dtype=np.float64).reshape(4, 4)
    final_T = np.asarray(final_T, dtype=np.float64).reshape(4, 4)
    expected_T = np.asarray(expected_T, dtype=np.float64).reshape(4, 4)
    expected_delta = expected_T[:3, 3] - start_T[:3, 3]
    observed_delta = final_T[:3, 3] - start_T[:3, 3]
    tracking = translation_tracking_report(expected_delta, observed_delta)
    target_norm = float(tracking.get("target_norm_m") or 0.0)
    observed_norm = float(tracking.get("observed_norm_m") or 0.0)
    tracking["norm_ratio"] = observed_norm / target_norm if target_norm > EPS else None
    return {
        "available": True,
        "expected_final_T_link7_in_base": expected_T.tolist(),
        "observed_final_T_link7_in_base": final_T.tolist(),
        "translation_tracking": tracking,
        "expected_rotation_delta_deg": rotation_angle_deg(expected_T[:3, :3] @ start_T[:3, :3].T),
        "observed_rotation_delta_deg": rotation_angle_deg(final_T[:3, :3] @ start_T[:3, :3].T),
        "final_pose_position_error_m": float(np.linalg.norm(expected_T[:3, 3] - final_T[:3, 3])),
        "final_pose_rotation_error_deg": rotation_angle_deg(expected_T[:3, :3] @ final_T[:3, :3].T),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory_json", help="interactive report, action sequence, rollout JSON, or run directory")
    parser.add_argument("--out-dir", default=str(artifact_dir("diagnostics")))
    parser.add_argument("--tag", default="humanego_eef_abs_corobot_replay")
    parser.add_argument("--side", choices=["right", "left"], default="right")
    parser.add_argument("--control-mode", choices=["dry-run", "prompt", "auto"], default="prompt")
    parser.add_argument("--confirm-control", default="")
    parser.add_argument("--executor", choices=["corobot_env", "http"], default="corobot_env")
    parser.add_argument("--max-actions", type=int, default=10)
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--corobot-action-url", default="")
    parser.add_argument("--corobot-timeout-s", type=float, default=10.0)
    parser.add_argument("--upload-url", default="")
    parser.add_argument("--upload-timeout-s", type=float, default=20.0)
    args = parser.parse_args()

    source = load_json(args.trajectory_json)
    out_base = Path(args.out_dir).expanduser().resolve()
    default_base = artifact_dir("diagnostics")
    if out_base == default_base:
        run_dir = artifact_run_dir("diagnostics", args.tag, prefix="humanego_eef_abs_corobot")
    else:
        run_dir = out_base / f"g1_humanego_eef_abs_corobot_{utc_stamp()}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "ok": False,
        "args": vars(args),
        "source_path": source.get("_path"),
        "control_sent": False,
        "note": (
            "Builds a RoboClaw/CoRobot-style raw EEF_ABS action from recorded "
            "HumanEgo target_T_link7_in_base rows. Execution requires a robot-side "
            "CoRobot endpoint that accepts raw action JSON."
        ),
    }
    arm = None
    try:
        targets = extract_targets(source, side=args.side, max_actions=args.max_actions)
        action = build_eef_abs_action(targets, side=args.side, duration_s=args.duration_s)
        action_path = run_dir / "corobot_eef_abs_action.json"
        targets_path = run_dir / "corobot_eef_abs_targets.json"
        action_path.write_text(json.dumps(json_safe(action), ensure_ascii=False, indent=2), encoding="utf-8")
        targets_path.write_text(json.dumps(json_safe(targets), ensure_ascii=False, indent=2), encoding="utf-8")

        report.update(
            {
                "targets": targets,
                "num_targets": len(targets),
                "corobot_action": action,
                "corobot_action_path": str(action_path),
                "targets_path": str(targets_path),
            }
        )
        print(
            "\n=== CoRobot EEF_ABS trajectory preview ===\n"
            f"source: {source.get('_path')}\n"
            f"side: {args.side}\n"
            f"rows: {len(targets)}\n"
            f"duration_s: {args.duration_s:.3f}\n"
            f"first row: {action[f'{args.side}_arm']['values'][0]}\n"
            f"last row:  {action[f'{args.side}_arm']['values'][-1]}\n"
            f"action_json: {action_path}\n"
        )

        if args.control_mode != "dry-run":
            from G1RobotArm import G1RobotArmReadOnly

            arm = G1RobotArmReadOnly(side=args.side)
            start_T = arm.get_T_link7_in_base()
            report["start_T_link7_in_base"] = start_T.tolist()
        else:
            start_T = None

        if args.control_mode == "prompt":
            operator = input("[Enter]=execute CoRobot EEF_ABS action, q=quit > ").strip().lower()
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
            if args.executor == "corobot_env":
                log("executing CoRobot EEF_ABS action via corobot.envs.g01_env.G01Env.execute_action")
                control_result = call_corobot_env_action(action, args.duration_s)
            else:
                urls = [args.corobot_action_url] if args.corobot_action_url else list(DEFAULT_COROBOT_URLS)
                log(f"posting CoRobot EEF_ABS action to candidate urls: {urls}")
                control_result = call_corobot_action(urls, action, args.corobot_timeout_s)
            report["control_sent"] = bool(control_result.get("ok"))
            report["control_result"] = control_result
            if control_result.get("ok"):
                time.sleep(max(0.0, float(args.duration_s) + float(args.settle_s)))
            final_T = arm.get_T_link7_in_base() if arm is not None else None
            if final_T is not None:
                report["final_T_link7_in_base"] = final_T.tolist()
            expected_final_T = matrix_from_any(targets[-1]["target_T_link7_in_base"], "expected_final_T")
            report["final_tracking"] = evaluate_final_tracking(start_T, final_T, expected_final_T)
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
            log("skipping read-only arm adapter close for CoRobot EEF_ABS replay")

    report_path = run_dir / "corobot_eef_abs_replay_report.json"
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
                "action_path": report.get("corobot_action_path"),
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
