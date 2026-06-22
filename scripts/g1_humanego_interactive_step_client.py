#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive one-step HumanEgo control client for G1.

Each iteration:
  1. read current G1 RGB-D and link7/TCP state
  2. ask the HumanEgo server for one target
  3. print the proposed target
  4. wait for operator input
  5. execute exactly one set_end_effector_pose_control command

This script is intentionally step-by-step. It never loops into autonomous
continuous control.
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
from typing import Any, Dict

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = PROJECT_ROOT / "cfg" / "inference" / "g1_serve_bread_right.yaml"

for path in (PROJECT_ROOT, PROJECT_ROOT / "inference", PROJECT_ROOT / "scripts"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from g1_humanego_client_dry_run import (  # noqa: E402
    encode_jpeg_b64,
    json_safe,
    log,
    post_json,
    resize_image_and_K,
    resolve_project_path,
    upload_zip,
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def load_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_zip(src_dir: Path) -> Path:
    zip_path = src_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir.parent))
    return zip_path


def compact_pose(pose: dict[str, Any]) -> str:
    return (
        f"x={pose['x']:+.4f}, y={pose['y']:+.4f}, z={pose['z']:+.4f}, "
        f"qx={pose['qx']:+.4f}, qy={pose['qy']:+.4f}, "
        f"qz={pose['qz']:+.4f}, qw={pose['qw']:+.4f}"
    )


def build_payload(frame: Any, state: dict[str, Any], args: argparse.Namespace, request_id: str) -> dict[str, Any]:
    rgb_send, K_send, image_send_info = resize_image_and_K(
        frame.rgb,
        frame.K,
        args.send_width,
        args.send_height,
    )
    jpeg_b64 = encode_jpeg_b64(rgb_send, args.jpeg_quality)
    log(f"request {request_id}: sending RGB {image_send_info['sent_shape']} jpeg_b64_bytes={len(jpeg_b64)}")
    return {
        "request_id": request_id,
        "client_time_utc": datetime.now(timezone.utc).isoformat(),
        "preview_steps": 1,
        "K": np.asarray(K_send, dtype=np.float64).tolist(),
        "rgb_jpeg_b64": jpeg_b64,
        "frame_summary": {
            "rgb_shape": list(rgb_send.shape),
            "source_rgb_shape": list(frame.rgb.shape),
            "image_send": image_send_info,
            "depth_shape": list(frame.depth_m.shape),
            "depth_valid_ratio": float(np.isfinite(frame.depth_m).mean()),
        },
        "current": {
            "T_head_pitch_camera": np.asarray(state["T_head_pitch_camera"], dtype=np.float64).tolist(),
            "T_base_camera": np.asarray(state["T_base_camera"], dtype=np.float64).tolist(),
            "T_base_in_cam": np.asarray(state["T_base_in_cam"], dtype=np.float64).tolist(),
            "T_link7_in_base": np.asarray(state["T_link7_in_base"], dtype=np.float64).tolist(),
            "T_tcp_in_link7": np.asarray(state["T_tcp_in_link7"], dtype=np.float64).tolist(),
            "T_tcp_in_base": np.asarray(state["T_tcp_in_base"], dtype=np.float64).tolist(),
            "T_tcp_in_cam": np.asarray(state["T_tcp_in_cam"], dtype=np.float64).tolist(),
            "gripper": float(state["gripper"]),
            "gripper_state": json_safe(state["gripper_state"]),
            "corobot_fk": json_safe(state["corobot_fk"]),
        },
    }


def select_target(step_preview: dict[str, Any], source: str) -> dict[str, float]:
    if source == "raw":
        return {k: float(v) for k, v in step_preview["right_pose_flat_raw"].items()}
    if source == "limited":
        return {k: float(v) for k, v in step_preview["right_pose_flat_limited"].items()}
    raise ValueError(f"unknown target source: {source}")


def position_from_pose_dict(pose: dict[str, float]) -> np.ndarray:
    return np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float64)


def call_ee_control(controller: Any, pose: dict[str, float], side: str, lifetime: float) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "lifetime": float(lifetime),
        "control_group": [f"{side}_arm"],
    }
    if side == "right":
        kwargs["right_pose"] = pose
    else:
        kwargs["left_pose"] = pose

    started = time.time()
    try:
        result = controller.set_end_effector_pose_control(**kwargs)
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


def prompt_operator() -> str:
    try:
        text = input("[Enter]=execute one step, s=skip/replan, q=quit > ")
    except EOFError:
        return "q"
    return text.strip().lower()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default=str(DEFAULT_CFG))
    parser.add_argument("--server-url", default="http://111.0.22.33:30003/infer")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "g1_humanego_interactive_runs"))
    parser.add_argument("--tag", default="interactive_step")
    parser.add_argument("--side", choices=["right", "left"], default="right")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--target-source", choices=["limited", "raw"], default="limited")
    parser.add_argument("--confirm-control", default="")
    parser.add_argument("--lifetime", type=float, default=0.5)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--send-width", type=int, default=320)
    parser.add_argument("--send-height", type=int, default=240)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--upload-url", default="")
    parser.add_argument("--upload-timeout-s", type=float, default=60.0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    cfg_path = resolve_project_path(args.cfg)
    cfg = load_cfg(cfg_path)
    run_dir = Path(args.out_dir).expanduser().resolve() / f"g1_humanego_interactive_{utc_stamp()}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "ok": False,
        "control_sent": False,
        "args": vars(args),
        "cfg_path": str(cfg_path),
        "steps": [],
    }

    cam = None
    arm = None
    try:
        from G1Camera import G1HeadRGBDCamera
        from G1RobotArm import G1RobotArmReadOnly

        log("initializing G1 camera")
        cam = G1HeadRGBDCamera(resolve_project_path(cfg["camera"]["cfg_path"]))
        log("initializing read-only G1 arm state adapter")
        arm = G1RobotArmReadOnly(side=args.side)

        if args.confirm_control != "RUN_CONTROL":
            log("WARNING: --confirm-control RUN_CONTROL not set; this script will preview but refuse to execute.")

        for idx in range(max(1, int(args.max_steps))):
            step_dir = run_dir / f"step_{idx:03d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            request_id = f"{utc_stamp()}_{args.tag}_{idx:03d}"

            log(f"step {idx}: reading RGB-D frame")
            frame = cam.get_frame()
            log(f"step {idx}: reading current robot state")
            state = arm.get_debug_state()
            before_T_link7 = np.asarray(state["T_link7_in_base"], dtype=np.float64)
            payload = build_payload(frame, state, args, request_id)

            cv2.imwrite(str(step_dir / "rgb_sent_bgr.jpg"), frame.rgb if args.send_width <= 0 else cv2.resize(frame.rgb, (args.send_width, args.send_height)))
            request_summary = dict(payload)
            request_summary.pop("rgb_jpeg_b64", None)
            (step_dir / "request_summary.json").write_text(
                json.dumps(json_safe(request_summary), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            started = time.time()
            server_result = post_json(args.server_url, payload, args.timeout_s)
            response = server_result["json"]
            log(
                f"step {idx}: server response status={server_result['status']} "
                f"bytes={server_result['num_bytes']} duration={time.time() - started:.3f}s"
            )
            (step_dir / "server_response.json").write_text(
                json.dumps(json_safe(response), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            step_preview = response["policy_preview"]["sides"][args.side][0]
            target_pose = select_target(step_preview, args.target_source)
            target_delta = position_from_pose_dict(target_pose) - before_T_link7[:3, 3]
            target_delta_norm = float(np.linalg.norm(target_delta))

            print("\n=== HumanEgo proposed step ===")
            print(f"step: {idx}")
            print(f"done_prob: {response['policy_preview']['done_prob']:.3f}")
            print(f"target_source: {args.target_source}")
            print(f"target_delta_m: {target_delta.tolist()}  norm={target_delta_norm:.4f}")
            print(f"server raw_delta_norm_m: {step_preview['safety_translation_limit']['raw_delta_norm_m']:.4f}")
            print(f"server clipped: {step_preview['safety_translation_limit']['clipped']}")
            print(f"gripper target raw: {step_preview['gripper_g1_raw_0_open_120_closed']:.2f} / 120 (not executed)")
            print(f"right_pose: {compact_pose(target_pose)}")

            operator = prompt_operator()
            step_record: Dict[str, Any] = {
                "idx": idx,
                "request_id": request_id,
                "server_ok": bool(response.get("ok")),
                "target_source": args.target_source,
                "target_pose": target_pose,
                "target_delta_m": target_delta.tolist(),
                "target_delta_norm_m": target_delta_norm,
                "operator_input": operator,
                "server_response": response,
            }

            if operator == "q":
                log("operator requested quit")
                step_record["executed"] = False
                report["steps"].append(step_record)
                break
            if operator == "s":
                log("operator skipped this target")
                step_record["executed"] = False
                report["steps"].append(step_record)
                continue
            if args.confirm_control != "RUN_CONTROL":
                log("refusing to execute because --confirm-control RUN_CONTROL is missing")
                step_record["executed"] = False
                step_record["blocked_reason"] = "missing RUN_CONTROL confirmation"
                report["steps"].append(step_record)
                break

            log(f"step {idx}: executing one EE target")
            control_result = call_ee_control(arm.controller, target_pose, args.side, args.lifetime)
            report["control_sent"] = True
            step_record["executed"] = bool(control_result.get("ok"))
            step_record["control_result"] = control_result

            time.sleep(float(args.settle_s))
            after_T_link7 = arm.get_T_link7_in_base()
            observed_delta = after_T_link7[:3, 3] - before_T_link7[:3, 3]
            step_record["after_T_link7_in_base"] = after_T_link7.tolist()
            step_record["observed_delta_m"] = observed_delta.tolist()
            step_record["observed_delta_norm_m"] = float(np.linalg.norm(observed_delta))
            log(
                f"step {idx}: observed_delta={observed_delta.tolist()} "
                f"norm={np.linalg.norm(observed_delta):.4f}"
            )
            report["steps"].append(step_record)
            (step_dir / "step_record.json").write_text(
                json.dumps(json_safe(step_record), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        report["ok"] = True
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
        if cam is not None:
            report["camera_close_skipped"] = True
            log("skipping G1 camera close for interactive test to avoid SDK shutdown blocking")
        if arm is not None:
            report["arm_close_skipped"] = True
            log("skipping read-only arm adapter close for interactive test")

    (run_dir / "interactive_step_report.json").write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log("building local result zip")
    zip_path = make_zip(run_dir)
    upload = None
    if args.upload_url:
        try:
            log(f"uploading result zip to {args.upload_url}")
            upload = upload_zip(zip_path, args.upload_url, args.upload_timeout_s)
            log(f"upload complete status={upload.get('status')}")
        except Exception as exc:
            upload = {"ok": False, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        (run_dir / "upload_result.json").write_text(json.dumps(upload, ensure_ascii=False, indent=2), encoding="utf-8")
        zip_path = make_zip(run_dir)

    print(json.dumps({"run_dir": str(run_dir), "zip_path": str(zip_path), "upload": upload}, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
