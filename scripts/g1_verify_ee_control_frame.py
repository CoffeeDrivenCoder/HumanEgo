#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify G1 set_end_effector_pose_control target-frame semantics.

Default mode is read-only. Control commands are sent only when both conditions
are met:
  --mode hold|move
  --confirm-control RUN_CONTROL
"""

from __future__ import annotations

import argparse
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
from typing import Any, Dict, Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
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


def parse_pose_value(value: Any) -> Optional[Dict[str, float]]:
    if value is None:
        return None
    if isinstance(value, dict):
        if all(k in value for k in ("x", "y", "z", "qx", "qy", "qz", "qw")):
            return {k: float(value[k]) for k in ("x", "y", "z", "qx", "qy", "qz", "qw")}
        pos = value.get("position") or value.get("pos")
        ori = value.get("orientation") or value.get("quat") or value.get("quaternion")
        if isinstance(pos, dict) and isinstance(ori, dict):
            if "quaternion" in ori and isinstance(ori["quaternion"], dict):
                ori = ori["quaternion"]
            if all(k in pos for k in ("x", "y", "z")) and all(k in ori for k in ("x", "y", "z", "w")):
                return {
                    "x": float(pos["x"]),
                    "y": float(pos["y"]),
                    "z": float(pos["z"]),
                    "qx": float(ori["x"]),
                    "qy": float(ori["y"]),
                    "qz": float(ori["z"]),
                    "qw": float(ori["w"]),
                }
        for nested_key in ("pose", "frame_pose", "transform"):
            parsed = parse_pose_value(value.get(nested_key))
            if parsed is not None:
                return parsed
    if isinstance(value, (list, tuple)) and len(value) >= 7:
        vals = [float(v) for v in value[:7]]
        return {
            "x": vals[0],
            "y": vals[1],
            "z": vals[2],
            "qx": vals[3],
            "qy": vals[4],
            "qz": vals[5],
            "qw": vals[6],
        }
    return None


def pose_position(pose: Dict[str, float]) -> np.ndarray:
    return np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float64)


def pose_with_offset(pose: Dict[str, float], axis: str, delta_m: float) -> Dict[str, float]:
    out = dict(pose)
    out[axis] = float(out[axis] + delta_m)
    return out


def pose_payload(pose: Dict[str, float], fmt: str) -> Any:
    if fmt == "flat_dict":
        return {
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "z": float(pose["z"]),
            "qx": float(pose["qx"]),
            "qy": float(pose["qy"]),
            "qz": float(pose["qz"]),
            "qw": float(pose["qw"]),
        }
    if fmt == "frame_dict":
        return {
            "position": {
                "x": float(pose["x"]),
                "y": float(pose["y"]),
                "z": float(pose["z"]),
            },
            "orientation": {
                "quaternion": {
                    "x": float(pose["qx"]),
                    "y": float(pose["qy"]),
                    "z": float(pose["qz"]),
                    "w": float(pose["qw"]),
                }
            },
        }
    if fmt == "xyzquat_list":
        return [
            float(pose["x"]),
            float(pose["y"]),
            float(pose["z"]),
            float(pose["qx"]),
            float(pose["qy"]),
            float(pose["qz"]),
            float(pose["qw"]),
        ]
    raise ValueError(f"unknown pose format: {fmt}")


def frame_name_for_side(side: str) -> str:
    return f"arm_{side}_link7"


def get_frame_pose(controller: Any, frame_name: str) -> tuple[Dict[str, Any], Dict[str, float]]:
    status = controller.get_motion_status()
    if not isinstance(status, dict):
        raise RuntimeError(f"get_motion_status returned non-dict: {status!r}")
    frames = status.get("frames") or {}
    if frame_name not in frames:
        raise RuntimeError(f"{frame_name} not in motion_status frames: {sorted(frames.keys())}")
    pose = parse_pose_value(frames[frame_name])
    if pose is None:
        raise RuntimeError(f"cannot parse pose from frame {frame_name}: {frames[frame_name]!r}")
    return status, pose


def wait_frame_pose(controller: Any, frame_name: str, tries: int, sleep_s: float) -> tuple[Dict[str, Any], Dict[str, float]]:
    last_error = None
    for _ in range(tries):
        try:
            return get_frame_pose(controller, frame_name)
        except Exception as exc:
            last_error = exc
            time.sleep(sleep_s)
    raise RuntimeError(f"failed to read {frame_name}: {last_error}")


def sample_poses(controller: Any, frame_name: str, count: int, interval_s: float) -> list[Dict[str, Any]]:
    samples = []
    for idx in range(count):
        status, pose = get_frame_pose(controller, frame_name)
        samples.append(
            {
                "idx": idx,
                "time_s": time.time(),
                "pose": pose,
                "motion_status_mode": status.get("mode"),
                "motion_status_error": status.get("error"),
            }
        )
        if idx + 1 < count:
            time.sleep(interval_s)
    return samples


def call_control(
    controller: Any,
    *,
    side: str,
    pose: Dict[str, float],
    pose_format: str,
    lifetime: float,
    control_group: str,
) -> Dict[str, Any]:
    payload = pose_payload(pose, pose_format)
    kwargs: Dict[str, Any] = {
        "lifetime": float(lifetime),
        "control_group": [control_group],
    }
    if side == "left":
        kwargs["left_pose"] = payload
    else:
        kwargs["right_pose"] = payload

    started = time.time()
    item: Dict[str, Any] = {
        "ok": False,
        "side": side,
        "pose_format": pose_format,
        "control_group": [control_group],
        "lifetime": lifetime,
        "target_pose_flat": pose,
        "target_payload": json_safe(payload),
        "kwargs_keys": sorted(kwargs.keys()),
    }
    try:
        result = controller.set_end_effector_pose_control(**kwargs)
        item.update(
            {
                "ok": True,
                "duration_s": time.time() - started,
                "result": json_safe(result),
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


def choose_pose_format(
    controller: Any,
    *,
    side: str,
    pose: Dict[str, float],
    requested_format: str,
    lifetime: float,
    control_group: str,
    settle_s: float,
) -> tuple[Optional[str], list[Dict[str, Any]]]:
    formats = ["frame_dict", "flat_dict", "xyzquat_list"] if requested_format == "auto" else [requested_format]
    attempts = []
    for fmt in formats:
        call = call_control(
            controller,
            side=side,
            pose=pose,
            pose_format=fmt,
            lifetime=lifetime,
            control_group=control_group,
        )
        attempts.append(call)
        time.sleep(settle_s)
        if call["ok"]:
            return fmt, attempts
    return None, attempts


def vector_report(commanded_delta: np.ndarray, observed_delta: np.ndarray) -> Dict[str, Any]:
    commanded_norm = float(np.linalg.norm(commanded_delta))
    observed_norm = float(np.linalg.norm(observed_delta))
    out: Dict[str, Any] = {
        "commanded_delta_m": commanded_delta.tolist(),
        "observed_delta_m": observed_delta.tolist(),
        "commanded_norm_m": commanded_norm,
        "observed_norm_m": observed_norm,
        "position_error_m": float(np.linalg.norm(observed_delta - commanded_delta)),
    }
    if commanded_norm > 1e-9:
        unit = commanded_delta / commanded_norm
        along = float(np.dot(observed_delta, unit))
        perpendicular = observed_delta - along * unit
        out.update(
            {
                "observed_along_command_axis_m": along,
                "perpendicular_error_m": float(np.linalg.norm(perpendicular)),
                "same_sign": bool(along * commanded_norm > 0),
                "dominant_axis_matches": bool(np.argmax(np.abs(observed_delta)) == np.argmax(np.abs(commanded_delta))),
            }
        )
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "g1_ee_control_verify_runs"))
    parser.add_argument("--tag", default="ee_control_frame")
    parser.add_argument("--side", choices=["left", "right"], default="right")
    parser.add_argument("--mode", choices=["observe", "hold", "move"], default="observe")
    parser.add_argument("--confirm-control", default="")
    parser.add_argument("--pose-format", choices=["auto", "frame_dict", "flat_dict", "xyzquat_list"], default="auto")
    parser.add_argument("--control-group", default="", help="Default: '<side>_arm'.")
    parser.add_argument("--delta-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--delta-m", type=float, default=0.01)
    parser.add_argument("--lifetime", type=float, default=0.5)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--return-to-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--sample-interval-s", type=float, default=0.1)
    parser.add_argument("--status-tries", type=int, default=30)
    parser.add_argument("--status-sleep-s", type=float, default=0.1)
    parser.add_argument("--upload-url", default="")
    parser.add_argument("--upload-timeout-s", type=float, default=20.0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    run_dir = ensure_dir(Path(args.out_dir).expanduser().resolve() / f"g1_ee_control_{utc_stamp()}_{args.tag}")
    report: Dict[str, Any] = {
        "ok": False,
        "args": vars(args),
        "note": "Observe mode is read-only. Hold/move modes require --confirm-control RUN_CONTROL.",
    }
    controller = None

    try:
        from a2d_sdk.robot import RobotController

        controller = RobotController()
        frame_name = frame_name_for_side(args.side)
        control_group = args.control_group or f"{args.side}_arm"

        report["controller"] = {
            "class": repr(type(controller)),
            "set_end_effector_pose_control_signature": str(inspect.signature(controller.set_end_effector_pose_control)),
            "target_frame_hypothesis": f"{frame_name} in base_link coordinates",
        }
        report["frame_name"] = frame_name
        report["control_group"] = control_group

        _status, start_pose = wait_frame_pose(controller, frame_name, args.status_tries, args.status_sleep_s)
        report["start_pose"] = start_pose
        report["pre_samples"] = sample_poses(controller, frame_name, args.samples, args.sample_interval_s)

        if args.mode == "observe":
            report.update({"ok": True, "control_sent": False, "result": "read_only_observe"})
        elif args.confirm_control != "RUN_CONTROL":
            report.update(
                {
                    "ok": False,
                    "control_sent": False,
                    "error": "Refusing to send control without --confirm-control RUN_CONTROL.",
                }
            )
        else:
            chosen_format, hold_attempts = choose_pose_format(
                controller,
                side=args.side,
                pose=start_pose,
                requested_format=args.pose_format,
                lifetime=args.lifetime,
                control_group=control_group,
                settle_s=args.settle_s,
            )
            report["control_sent"] = True
            report["hold_attempts"] = hold_attempts
            report["chosen_pose_format"] = chosen_format
            report["hold_after_samples"] = sample_poses(controller, frame_name, args.samples, args.sample_interval_s)
            hold_after_pose = report["hold_after_samples"][-1]["pose"]
            report["hold_delta_m"] = (pose_position(hold_after_pose) - pose_position(start_pose)).tolist()

            if chosen_format is None:
                report.update({"ok": False, "error": "No pose payload format was accepted by set_end_effector_pose_control."})
            elif args.mode == "hold":
                report.update({"ok": True, "result": "hold_current_complete"})
            else:
                move_start_pose = hold_after_pose
                target_pose = pose_with_offset(move_start_pose, args.delta_axis, args.delta_m)
                commanded_delta = pose_position(target_pose) - pose_position(move_start_pose)
                move_call = call_control(
                    controller,
                    side=args.side,
                    pose=target_pose,
                    pose_format=chosen_format,
                    lifetime=args.lifetime,
                    control_group=control_group,
                )
                report["move_call"] = move_call
                time.sleep(args.settle_s)
                report["move_after_samples"] = sample_poses(controller, frame_name, args.samples, args.sample_interval_s)
                move_after_pose = report["move_after_samples"][-1]["pose"]
                observed_delta = pose_position(move_after_pose) - pose_position(move_start_pose)
                report["move_delta_analysis"] = vector_report(commanded_delta, observed_delta)
                report["interpretation_hint"] = (
                    "If observed_delta_m has the same dominant axis/sign as commanded_delta_m, "
                    "the control target is likely base_link coordinates for arm_<side>_link7."
                )

                if args.return_to_start and move_call["ok"]:
                    return_call = call_control(
                        controller,
                        side=args.side,
                        pose=move_start_pose,
                        pose_format=chosen_format,
                        lifetime=args.lifetime,
                        control_group=control_group,
                    )
                    report["return_call"] = return_call
                    time.sleep(args.settle_s)
                    report["return_after_samples"] = sample_poses(
                        controller, frame_name, args.samples, args.sample_interval_s
                    )
                    return_after_pose = report["return_after_samples"][-1]["pose"]
                    report["return_delta_from_move_start_m"] = (
                        pose_position(return_after_pose) - pose_position(move_start_pose)
                    ).tolist()

                report.update({"ok": bool(move_call["ok"]), "result": "move_probe_complete"})

    except Exception as exc:
        report.update({"ok": False, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
    finally:
        cleanup: Dict[str, Any] = {}
        if controller is not None:
            for method_name in ("shutdown", "close"):
                if not hasattr(controller, method_name):
                    continue
                try:
                    getattr(controller, method_name)()
                    cleanup["controller"] = {"ok": True, "method": method_name}
                    break
                except Exception as exc:
                    cleanup["controller"] = {
                        "ok": False,
                        "method": method_name,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    break
        if cleanup:
            report["cleanup"] = cleanup

    (run_dir / "ee_control_frame_report.json").write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    zip_path = make_zip(run_dir)
    upload = None
    if args.upload_url:
        try:
            upload = upload_zip(zip_path, args.upload_url, args.upload_timeout_s)
        except Exception as exc:
            upload = {"ok": False, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        (run_dir / "upload_result.json").write_text(json.dumps(upload, ensure_ascii=False, indent=2), encoding="utf-8")
        zip_path = make_zip(run_dir)

    print(json.dumps({"run_dir": str(run_dir), "zip_path": str(zip_path), "upload": upload}, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
