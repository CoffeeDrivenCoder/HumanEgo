#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manual G1 FK/IK sampling validation.

The operator manually moves the robot arms. Each sample reads SDK joint states
and SDK link7 poses, then validates URDF FK and IK against those real states.
This script is read-only and sends no control commands.
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

from G1RobotArm import parse_motion_pose, wait_motion_status  # noqa: E402
from g1_artifacts import artifact_dir, run_dir as artifact_run_dir  # noqa: E402
from g1_humanego_client_dry_run import json_safe, upload_zip  # noqa: E402
from g1_humanego_interactive_step_client import read_robot_joint_states_for_trajectory  # noqa: E402
from g1_urdf_ik import DEFAULT_G1_ZIP, G1UrdfKinematics, pose_error  # noqa: E402


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


def waist_values_with_height_offset(waist_values: list[float], height_offset_m: float) -> list[float]:
    values = [float(v) for v in waist_values]
    if len(values) >= 2:
        values[1] += float(height_offset_m)
    return values


def motion_T_for_side(status: dict[str, Any], side: str) -> np.ndarray:
    frames = status.get("frames") or {}
    frame_name = "arm_left_link7" if side == "left" else "arm_right_link7"
    if frame_name not in frames:
        raise KeyError(f"{frame_name} missing from motion_status frames: {sorted(frames)}")
    return parse_motion_pose(frames[frame_name])


def max_abs(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.max(np.abs(arr))) if arr.size else 0.0


def threshold_pass(error: dict[str, Any], pos_tol_m: float, rot_tol_deg: float) -> bool:
    return (
        float(error.get("position_error_m", float("inf"))) <= float(pos_tol_m)
        and float(error.get("rotation_error_deg", float("inf"))) <= float(rot_tol_deg)
    )


def validate_side_sample(
    kin: G1UrdfKinematics,
    *,
    side: str,
    joint_states: dict[str, Any],
    motion_status: dict[str, Any],
    mapping: str,
    waist_height_offset_m: float,
    max_nfev: int,
    fk_pos_tol_m: float,
    fk_rot_tol_deg: float,
    ik_pos_tol_m: float,
    ik_rot_tol_deg: float,
    q_abs_tol_rad: float,
) -> dict[str, Any]:
    q_real, indices = side_q_from_arm_state(joint_states["arm"], side, mapping)
    waist_for_urdf = waist_values_with_height_offset(joint_states["waist"], waist_height_offset_m)
    sdk_T = motion_T_for_side(motion_status, side)
    fk_T = kin.link7_fk(side, q_real, waist_states=waist_for_urdf)
    fk_error = pose_error(fk_T, sdk_T)

    # Use the real measured joint vector as IK seed to avoid judging an arbitrary
    # alternate IK branch as a failure.
    ik = kin.solve_link7_ik(side, sdk_T, q_real, waist_states=waist_for_urdf, max_nfev=max_nfev)
    ik_fk_T = kin.link7_fk(side, ik.q_solution, waist_states=waist_for_urdf)
    ik_fk_error = pose_error(ik_fk_T, sdk_T)
    q_delta = ik.q_solution - q_real
    q_delta_abs_max = max_abs(q_delta)

    return {
        "side": side,
        "arm_state_mapping": mapping,
        "arm_state_indices": indices,
        "q_real_from_sdk": q_real.tolist(),
        "waist_values_raw": [float(v) for v in joint_states["waist"]],
        "waist_values_for_urdf": waist_for_urdf,
        "sdk_T_link7_in_base": sdk_T.tolist(),
        "urdf_fk_T_link7_in_base": fk_T.tolist(),
        "urdf_fk_vs_sdk_error": fk_error,
        "ik_from_sdk_pose_seeded_by_real_q": ik.to_json(),
        "ik_fk_T_link7_in_base": ik_fk_T.tolist(),
        "ik_fk_vs_sdk_error": ik_fk_error,
        "q_ik_minus_q_real": q_delta.tolist(),
        "q_delta_norm_rad": float(np.linalg.norm(q_delta)),
        "q_delta_abs_max_rad": q_delta_abs_max,
        "checks": {
            "fk_pose_ok": threshold_pass(fk_error, fk_pos_tol_m, fk_rot_tol_deg),
            "ik_success": bool(ik.success),
            "ik_fk_pose_ok": threshold_pass(ik_fk_error, ik_pos_tol_m, ik_rot_tol_deg),
            "ik_q_close_to_real": q_delta_abs_max <= float(q_abs_tol_rad),
        },
    }


def sample_once(
    kin: G1UrdfKinematics,
    robot: Any,
    controller: Any,
    args: argparse.Namespace,
    sample_idx: int,
    sides: list[str],
) -> dict[str, Any]:
    joint_states = read_robot_joint_states_for_trajectory(robot)
    status = wait_motion_status(controller, tries=args.motion_tries, sleep_s=args.motion_sleep_s)
    if not isinstance(status, dict):
        raise RuntimeError(f"get_motion_status did not return a dict: {status!r}")
    side_reports = {
        side: validate_side_sample(
            kin,
            side=side,
            joint_states=joint_states,
            motion_status=status,
            mapping=args.arm_state_mapping,
            waist_height_offset_m=args.waist_height_offset_m,
            max_nfev=args.max_nfev,
            fk_pos_tol_m=args.fk_position_tolerance_m,
            fk_rot_tol_deg=args.fk_rotation_tolerance_deg,
            ik_pos_tol_m=args.ik_position_tolerance_m,
            ik_rot_tol_deg=args.ik_rotation_tolerance_deg,
            q_abs_tol_rad=args.ik_q_tolerance_rad,
        )
        for side in sides
    }
    return {
        "sample_idx": int(sample_idx),
        "sample_utc": datetime.now(timezone.utc).isoformat(),
        "joint_states": json_safe(joint_states),
        "motion_status": json_safe(status),
        "sides": side_reports,
    }


def summarize_samples(samples: list[dict[str, Any]], sides: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_samples": len(samples),
        "sides": {},
    }
    for side in sides:
        reps = [(sample.get("sides") or {}).get(side) for sample in samples]
        reps = [rep for rep in reps if isinstance(rep, dict)]
        fk_pos = [float((rep.get("urdf_fk_vs_sdk_error") or {}).get("position_error_m", float("nan"))) for rep in reps]
        fk_rot = [float((rep.get("urdf_fk_vs_sdk_error") or {}).get("rotation_error_deg", float("nan"))) for rep in reps]
        ik_pos = [float((rep.get("ik_fk_vs_sdk_error") or {}).get("position_error_m", float("nan"))) for rep in reps]
        ik_rot = [float((rep.get("ik_fk_vs_sdk_error") or {}).get("rotation_error_deg", float("nan"))) for rep in reps]
        q_abs = [float(rep.get("q_delta_abs_max_rad", float("nan"))) for rep in reps]
        checks = [rep.get("checks") or {} for rep in reps]
        summary["sides"][side] = {
            "samples": len(reps),
            "fk_position_error_m_max": float(np.nanmax(fk_pos)) if fk_pos else None,
            "fk_position_error_m_mean": float(np.nanmean(fk_pos)) if fk_pos else None,
            "fk_rotation_error_deg_max": float(np.nanmax(fk_rot)) if fk_rot else None,
            "ik_fk_position_error_m_max": float(np.nanmax(ik_pos)) if ik_pos else None,
            "ik_fk_position_error_m_mean": float(np.nanmean(ik_pos)) if ik_pos else None,
            "ik_fk_rotation_error_deg_max": float(np.nanmax(ik_rot)) if ik_rot else None,
            "q_delta_abs_max_rad_max": float(np.nanmax(q_abs)) if q_abs else None,
            "q_delta_abs_max_rad_mean": float(np.nanmean(q_abs)) if q_abs else None,
            "all_fk_pose_ok": all(bool(c.get("fk_pose_ok")) for c in checks) if checks else False,
            "all_ik_success": all(bool(c.get("ik_success")) for c in checks) if checks else False,
            "all_ik_fk_pose_ok": all(bool(c.get("ik_fk_pose_ok")) for c in checks) if checks else False,
            "all_ik_q_close_to_real": all(bool(c.get("ik_q_close_to_real")) for c in checks) if checks else False,
        }
    summary["overall_ok"] = all(
        side_summary.get("all_fk_pose_ok")
        and side_summary.get("all_ik_success")
        and side_summary.get("all_ik_fk_pose_ok")
        for side_summary in summary["sides"].values()
    )
    return summary


def print_sample_summary(sample: dict[str, Any], sides: list[str]) -> None:
    idx = sample["sample_idx"]
    parts = [f"sample {idx}"]
    for side in sides:
        rep = sample["sides"][side]
        fk = rep["urdf_fk_vs_sdk_error"]
        ik_fk = rep["ik_fk_vs_sdk_error"]
        ik = rep["ik_from_sdk_pose_seeded_by_real_q"]
        parts.append(
            f"{side}: fk={fk['position_error_m']:.6f}m/{fk['rotation_error_deg']:.3f}deg "
            f"ik={ik_fk['position_error_m']:.6f}m/{ik_fk['rotation_error_deg']:.3f}deg "
            f"qmax={rep['q_delta_abs_max_rad']:.5f}rad success={ik['success']}"
        )
    print(" | ".join(parts))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf-zip", default=str(DEFAULT_G1_ZIP))
    parser.add_argument("--out-dir", default=str(artifact_dir("diagnostics")))
    parser.add_argument("--tag", default="g1_manual_fk_ik_sampling")
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--arm-state-mapping", choices=["left_first", "right_first"], default="left_first")
    parser.add_argument("--waist-height-offset-m", type=float, default=-0.3)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument("--fk-position-tolerance-m", type=float, default=0.001)
    parser.add_argument("--fk-rotation-tolerance-deg", type=float, default=0.1)
    parser.add_argument("--ik-position-tolerance-m", type=float, default=0.001)
    parser.add_argument("--ik-rotation-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--ik-q-tolerance-rad", type=float, default=0.05)
    parser.add_argument("--motion-tries", type=int, default=30)
    parser.add_argument("--motion-sleep-s", type=float, default=0.1)
    parser.add_argument("--no-prompt", action="store_true", help="Sample immediately instead of waiting for Enter.")
    parser.add_argument("--sample-interval-s", type=float, default=0.0)
    parser.add_argument("--upload-url", default="")
    parser.add_argument("--upload-timeout-s", type=float, default=20.0)
    args = parser.parse_args()

    sides = ["left", "right"] if args.side == "both" else [args.side]
    out_base = Path(args.out_dir).expanduser().resolve()
    default_base = artifact_dir("diagnostics")
    if out_base == default_base:
        run_dir = artifact_run_dir("diagnostics", args.tag, prefix="g1_manual_fk_ik_sampling")
    else:
        run_dir = out_base / f"g1_manual_fk_ik_sampling_{utc_stamp()}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "ok": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "control_sent": False,
        "note": "Read-only manual sampling. Move the arm by hand between samples; this script sends no control command.",
    }
    samples: list[dict[str, Any]] = []
    try:
        from a2d_sdk.robot import RobotController, RobotDds

        robot = RobotDds()
        controller = RobotController()
        kin = G1UrdfKinematics(args.urdf_zip)
        print(
            "\n=== G1 manual FK/IK sampling ===\n"
            f"sides: {sides}\n"
            f"samples: {args.samples}\n"
            f"waist_height_offset_m: {args.waist_height_offset_m}\n"
            "Move the arm by hand before each sample. This script is read-only.\n"
        )
        for idx in range(max(1, int(args.samples))):
            if not args.no_prompt:
                try:
                    operator = input(f"[sample {idx + 1}/{args.samples}] Move arm, then Enter=record, q=finish > ").strip().lower()
                except EOFError:
                    operator = "q"
                if operator == "q":
                    break
            elif args.sample_interval_s > 0.0 and idx > 0:
                time.sleep(float(args.sample_interval_s))
            sample = sample_once(kin, robot, controller, args, idx, sides)
            samples.append(sample)
            print_sample_summary(sample, sides)
        summary = summarize_samples(samples, sides)
        report.update(
            {
                "ok": bool(samples),
                "samples": samples,
                "summary": summary,
            }
        )
    except KeyboardInterrupt:
        report.update(
            {
                "ok": bool(samples),
                "interrupted": True,
                "error_type": "KeyboardInterrupt",
                "error": "Interrupted by operator",
                "samples": samples,
                "summary": summarize_samples(samples, sides) if samples else {},
                "traceback": traceback.format_exc(),
            }
        )
    except Exception as exc:
        report.update(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "samples": samples,
                "summary": summarize_samples(samples, sides) if samples else {},
                "traceback": traceback.format_exc(),
            }
        )

    report_path = run_dir / "g1_manual_fk_ik_sampling_report.json"
    report_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = run_dir / "g1_manual_fk_ik_sampling_summary.json"
    summary_path.write_text(json.dumps(json_safe(report.get("summary") or {}), ensure_ascii=False, indent=2), encoding="utf-8")
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
                "summary": report.get("summary"),
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
