#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only robot-side validation of G1 URDF FK/IK against SDK link7 poses."""

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
from g1_humanego_client_dry_run import json_safe, upload_zip  # noqa: E402
from g1_humanego_interactive_step_client import read_robot_joint_states_for_trajectory  # noqa: E402
from G1RobotArm import parse_motion_pose, wait_motion_status  # noqa: E402
from g1_urdf_ik import G1UrdfKinematics, DEFAULT_G1_ZIP, pose_error  # noqa: E402


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


def side_q_from_arm_state(arm_values: list[float], side: str, mapping: str) -> tuple[np.ndarray, list[int]]:
    values = [float(v) for v in arm_values]
    if len(values) < 14:
        raise ValueError(f"arm_joint_states must contain at least 14 values, got {len(values)}")
    side = side.lower()
    mapping = mapping.lower()
    if mapping == "left_first":
        indices = list(range(0, 7)) if side == "left" else list(range(7, 14))
    elif mapping == "right_first":
        indices = list(range(7, 14)) if side == "left" else list(range(0, 7))
    else:
        raise ValueError(f"unknown arm state mapping {mapping!r}")
    return np.asarray([values[i] for i in indices], dtype=np.float64), indices


def motion_T_for_side(status: dict[str, Any], side: str) -> np.ndarray:
    frames = status.get("frames") or {}
    frame_name = "arm_left_link7" if side == "left" else "arm_right_link7"
    if frame_name not in frames:
        raise KeyError(f"{frame_name} missing from motion_status frames: {sorted(frames)}")
    return parse_motion_pose(frames[frame_name])


def waist_values_with_height_offset(waist_values: list[float], height_offset_m: float) -> list[float]:
    values = [float(v) for v in waist_values]
    if len(values) >= 2:
        values[1] += float(height_offset_m)
    return values


def validate_side(
    kin: G1UrdfKinematics,
    side: str,
    arm_values: list[float],
    waist_values: list[float],
    motion_status: dict[str, Any],
    mapping: str,
    max_nfev: int,
) -> dict[str, Any]:
    q, indices = side_q_from_arm_state(arm_values, side, mapping)
    sdk_T = motion_T_for_side(motion_status, side)
    fk_T = kin.link7_fk(side, q, waist_states=waist_values)
    fk_error = pose_error(fk_T, sdk_T)
    fk_to_sdk_translation_delta = sdk_T[:3, 3] - fk_T[:3, 3]
    ik_result = kin.solve_link7_ik(side, sdk_T, q, waist_states=waist_values, max_nfev=max_nfev)
    ik_fk_T = kin.link7_fk(side, ik_result.q_solution, waist_states=waist_values)
    ik_error = pose_error(ik_fk_T, sdk_T)
    return {
        "side": side,
        "arm_state_mapping": mapping,
        "arm_state_indices": indices,
        "q_from_sdk": q.tolist(),
        "waist_values_for_urdf": [float(v) for v in waist_values],
        "sdk_frame_name": kin.sdk_frame_names[side],
        "sdk_T_link7_in_base": sdk_T.tolist(),
        "urdf_fk_T_link7_in_base": fk_T.tolist(),
        "urdf_fk_vs_sdk_error": fk_error,
        "urdf_fk_to_sdk_translation_delta_m": fk_to_sdk_translation_delta.tolist(),
        "ik_current_pose_self_consistency": {
            "ik": ik_result.to_json(),
            "fk_T_link7_in_base": ik_fk_T.tolist(),
            "fk_vs_sdk_error": ik_error,
        },
        "model": kin.describe_side(side),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf-zip", default=str(DEFAULT_G1_ZIP))
    parser.add_argument("--out-dir", default=str(artifact_dir("diagnostics")))
    parser.add_argument("--tag", default="g1_urdf_fk_ik_robot_validate")
    parser.add_argument("--side", action="append", choices=["left", "right"], default=[])
    parser.add_argument("--arm-state-mapping", choices=["left_first", "right_first"], default="left_first")
    parser.add_argument("--try-both-mappings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument(
        "--waist-height-offset-m",
        type=float,
        default=0.0,
        help="Add this offset to SDK waist[1] before URDF FK/IK. Use to test SDK/URDF base-height convention.",
    )
    parser.add_argument("--upload-url", default="")
    parser.add_argument("--upload-timeout-s", type=float, default=20.0)
    args = parser.parse_args()

    sides = args.side or ["left", "right"]
    out_base = Path(args.out_dir).expanduser().resolve()
    default_base = artifact_dir("diagnostics")
    if out_base == default_base:
        run_dir = artifact_run_dir("diagnostics", args.tag, prefix="g1_urdf_fk_ik_robot")
    else:
        run_dir = out_base / f"g1_urdf_fk_ik_robot_{utc_stamp()}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "ok": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "note": "Read-only validation. This script does not send any robot control command.",
    }
    try:
        from a2d_sdk.robot import RobotController, RobotDds

        robot = RobotDds()
        controller = RobotController()
        joint_states = read_robot_joint_states_for_trajectory(robot)
        status = wait_motion_status(controller)
        kin = G1UrdfKinematics(args.urdf_zip)
        waist_values_for_urdf = waist_values_with_height_offset(joint_states["waist"], args.waist_height_offset_m)
        mappings = ["left_first", "right_first"] if args.try_both_mappings else [args.arm_state_mapping]
        validation: dict[str, Any] = {}
        for mapping in mappings:
            validation[mapping] = {
                side: validate_side(
                    kin,
                    side,
                    joint_states["arm"],
                    waist_values_for_urdf,
                    status,
                    mapping,
                    args.max_nfev,
                )
                for side in sides
            }
        chosen = validation[args.arm_state_mapping]
        report.update(
            {
                "ok": True,
                "control_sent": False,
                "joint_states": joint_states,
                "waist_values_for_urdf": waist_values_for_urdf,
                "waist_height_offset_m": float(args.waist_height_offset_m),
                "motion_status": json_safe(status),
                "validation": validation,
                "chosen_mapping": args.arm_state_mapping,
                "chosen_summary": {
                    side: {
                        "fk_position_error_m": chosen[side]["urdf_fk_vs_sdk_error"]["position_error_m"],
                        "fk_position_error_vector_m": chosen[side]["urdf_fk_vs_sdk_error"]["position_error_vector_m"],
                        "fk_to_sdk_translation_delta_m": chosen[side]["urdf_fk_to_sdk_translation_delta_m"],
                        "fk_rotation_error_deg": chosen[side]["urdf_fk_vs_sdk_error"]["rotation_error_deg"],
                        "ik_position_error_m": chosen[side]["ik_current_pose_self_consistency"]["fk_vs_sdk_error"]["position_error_m"],
                        "ik_rotation_error_deg": chosen[side]["ik_current_pose_self_consistency"]["fk_vs_sdk_error"]["rotation_error_deg"],
                        "ik_success": chosen[side]["ik_current_pose_self_consistency"]["ik"]["success"],
                    }
                    for side in chosen
                },
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

    report_path = run_dir / "g1_urdf_fk_ik_robot_validation_report.json"
    report_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = run_dir / "g1_urdf_fk_ik_robot_validation_summary.json"
    summary_path.write_text(json.dumps(json_safe(report.get("chosen_summary") or {}), ensure_ascii=False, indent=2), encoding="utf-8")
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
                "summary": report.get("chosen_summary"),
                "upload": upload,
                "ok": report.get("ok"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if report.get("ok") else 2)


if __name__ == "__main__":
    main()
