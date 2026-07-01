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
from scipy.spatial.transform import Rotation


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


def make_base_translation_target(T: np.ndarray, name: str, delta: list[float]) -> dict[str, Any]:
    target = np.asarray(T, dtype=np.float64).reshape(4, 4).copy()
    delta_arr = np.asarray(delta, dtype=np.float64).reshape(3)
    target[:3, 3] += delta_arr
    return {
        "name": name,
        "type": "base_translation",
        "delta_m": delta_arr.tolist(),
        "delta_rotation_deg": 0.0,
        "target_T_link7_in_base": target,
    }


def make_base_rotation_target(T: np.ndarray, name: str, axis: str, deg: float) -> dict[str, Any]:
    target = np.asarray(T, dtype=np.float64).reshape(4, 4).copy()
    rotvec = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}[axis]
    R_delta = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64) * np.deg2rad(float(deg))).as_matrix()
    target[:3, :3] = R_delta @ target[:3, :3]
    return {
        "name": name,
        "type": "base_rotation",
        "delta_m": [0.0, 0.0, 0.0],
        "axis": axis,
        "delta_rotation_deg": float(deg),
        "target_T_link7_in_base": target,
    }


def make_ik_probe_targets(T: np.ndarray, translation_m: float, rotation_deg: float) -> list[dict[str, Any]]:
    step = float(translation_m)
    deg = float(rotation_deg)
    targets: list[dict[str, Any]] = []
    for axis_name, delta in [
        ("x_pos", [step, 0.0, 0.0]),
        ("x_neg", [-step, 0.0, 0.0]),
        ("y_pos", [0.0, step, 0.0]),
        ("y_neg", [0.0, -step, 0.0]),
        ("z_pos", [0.0, 0.0, step]),
        ("z_neg", [0.0, 0.0, -step]),
    ]:
        targets.append(make_base_translation_target(T, f"translate_{axis_name}", delta))
    for axis in ["x", "y", "z"]:
        targets.append(make_base_rotation_target(T, f"rotate_{axis}_pos", axis, deg))
        targets.append(make_base_rotation_target(T, f"rotate_{axis}_neg", axis, -deg))
    return targets


def run_ik_case(
    kin: G1UrdfKinematics,
    *,
    side: str,
    target_T: np.ndarray,
    q_seed: np.ndarray,
    q_reference: np.ndarray,
    waist_for_urdf: list[float],
    label: str,
    seed_label: str,
    max_nfev: int,
    pos_tol_m: float,
    rot_tol_deg: float,
    q_abs_tol_rad: float | None = None,
) -> dict[str, Any]:
    started = time.time()
    ik = kin.solve_link7_ik(side, target_T, q_seed, waist_states=waist_for_urdf, max_nfev=max_nfev)
    duration_s = time.time() - started
    fk_T = kin.link7_fk(side, ik.q_solution, waist_states=waist_for_urdf)
    fk_error = pose_error(fk_T, target_T)
    q_delta_seed = ik.q_solution - q_seed
    q_delta_reference = ik.q_solution - q_reference
    q_delta_reference_abs_max = max_abs(q_delta_reference)
    return {
        "label": label,
        "seed_label": seed_label,
        "duration_s": duration_s,
        "ik": ik.to_json(),
        "ik_fk_T_link7_in_base": fk_T.tolist(),
        "ik_fk_vs_target_error": fk_error,
        "q_solution": ik.q_solution.tolist(),
        "q_delta_from_seed": q_delta_seed.tolist(),
        "q_delta_from_seed_norm_rad": float(np.linalg.norm(q_delta_seed)),
        "q_delta_from_seed_abs_max_rad": max_abs(q_delta_seed),
        "q_delta_from_reference": q_delta_reference.tolist(),
        "q_delta_from_reference_norm_rad": float(np.linalg.norm(q_delta_reference)),
        "q_delta_from_reference_abs_max_rad": q_delta_reference_abs_max,
        "checks": {
            "ik_success": bool(ik.success),
            "pose_within_tolerance": threshold_pass(fk_error, pos_tol_m, rot_tol_deg),
            "q_close_to_reference": None if q_abs_tol_rad is None else q_delta_reference_abs_max <= float(q_abs_tol_rad),
        },
    }


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
    probe_targets: bool,
    probe_translation_m: float,
    probe_rotation_deg: float,
    probe_max_joint_delta_rad: float,
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
    actual_pose_cases = {
        "real_seed": run_ik_case(
            kin,
            side=side,
            target_T=sdk_T,
            q_seed=q_real,
            q_reference=q_real,
            waist_for_urdf=waist_for_urdf,
            label="actual_pose",
            seed_label="real_seed",
            max_nfev=max_nfev,
            pos_tol_m=ik_pos_tol_m,
            rot_tol_deg=ik_rot_tol_deg,
            q_abs_tol_rad=q_abs_tol_rad,
        ),
        "home_seed": run_ik_case(
            kin,
            side=side,
            target_T=sdk_T,
            q_seed=kin.home_q(side),
            q_reference=q_real,
            waist_for_urdf=waist_for_urdf,
            label="actual_pose",
            seed_label="home_seed",
            max_nfev=max_nfev,
            pos_tol_m=ik_pos_tol_m,
            rot_tol_deg=ik_rot_tol_deg,
            q_abs_tol_rad=None,
        ),
    }

    probe_reports = []
    if probe_targets:
        for target in make_ik_probe_targets(sdk_T, probe_translation_m, probe_rotation_deg):
            case = run_ik_case(
                kin,
                side=side,
                target_T=target["target_T_link7_in_base"],
                q_seed=q_real,
                q_reference=q_real,
                waist_for_urdf=waist_for_urdf,
                label=target["name"],
                seed_label="real_seed",
                max_nfev=max_nfev,
                pos_tol_m=ik_pos_tol_m,
                rot_tol_deg=ik_rot_tol_deg,
                q_abs_tol_rad=None,
            )
            case["target_spec"] = {
                key: value
                for key, value in target.items()
                if key != "target_T_link7_in_base"
            }
            case["checks"]["joint_delta_within_probe_limit"] = (
                float(case["q_delta_from_reference_abs_max_rad"]) <= float(probe_max_joint_delta_rad)
            )
            probe_reports.append(case)

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
        "ik_performance": {
            "actual_pose_cases": actual_pose_cases,
            "probe_targets_enabled": bool(probe_targets),
            "probe_translation_m": float(probe_translation_m),
            "probe_rotation_deg": float(probe_rotation_deg),
            "probe_max_joint_delta_rad": float(probe_max_joint_delta_rad),
            "probe_cases": probe_reports,
        },
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
            probe_targets=args.ik_probe_targets,
            probe_translation_m=args.ik_probe_translation_m,
            probe_rotation_deg=args.ik_probe_rotation_deg,
            probe_max_joint_delta_rad=args.ik_probe_max_joint_delta_rad,
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
        perf = [rep.get("ik_performance") or {} for rep in reps]
        real_seed_cases = [
            ((item.get("actual_pose_cases") or {}).get("real_seed") or {})
            for item in perf
        ]
        home_seed_cases = [
            ((item.get("actual_pose_cases") or {}).get("home_seed") or {})
            for item in perf
        ]
        probe_cases = [
            case
            for item in perf
            for case in (item.get("probe_cases") or [])
            if isinstance(case, dict)
        ]

        def case_values(cases: list[dict[str, Any]], path: tuple[str, ...]) -> list[float]:
            values = []
            for case in cases:
                cur: Any = case
                for key in path:
                    cur = (cur or {}).get(key)
                if cur is not None:
                    values.append(float(cur))
            return values

        def case_bool_rate(cases: list[dict[str, Any]], path: tuple[str, ...]) -> float | None:
            if not cases:
                return None
            passed = 0
            for case in cases:
                cur: Any = case
                for key in path:
                    cur = (cur or {}).get(key)
                passed += 1 if bool(cur) else 0
            return float(passed / len(cases))

        real_seed_durations = case_values(real_seed_cases, ("duration_s",))
        home_seed_durations = case_values(home_seed_cases, ("duration_s",))
        probe_durations = case_values(probe_cases, ("duration_s",))
        probe_pos = case_values(probe_cases, ("ik_fk_vs_target_error", "position_error_m"))
        probe_rot = case_values(probe_cases, ("ik_fk_vs_target_error", "rotation_error_deg"))
        probe_q_abs = case_values(probe_cases, ("q_delta_from_reference_abs_max_rad",))
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
            "ik_performance": {
                "actual_pose_real_seed_success_rate": case_bool_rate(real_seed_cases, ("checks", "ik_success")),
                "actual_pose_real_seed_pose_ok_rate": case_bool_rate(real_seed_cases, ("checks", "pose_within_tolerance")),
                "actual_pose_real_seed_q_close_rate": case_bool_rate(real_seed_cases, ("checks", "q_close_to_reference")),
                "actual_pose_real_seed_duration_s_mean": float(np.nanmean(real_seed_durations)) if real_seed_durations else None,
                "actual_pose_real_seed_duration_s_max": float(np.nanmax(real_seed_durations)) if real_seed_durations else None,
                "actual_pose_home_seed_success_rate": case_bool_rate(home_seed_cases, ("checks", "ik_success")),
                "actual_pose_home_seed_pose_ok_rate": case_bool_rate(home_seed_cases, ("checks", "pose_within_tolerance")),
                "actual_pose_home_seed_duration_s_mean": float(np.nanmean(home_seed_durations)) if home_seed_durations else None,
                "actual_pose_home_seed_duration_s_max": float(np.nanmax(home_seed_durations)) if home_seed_durations else None,
                "probe_case_count": len(probe_cases),
                "probe_success_rate": case_bool_rate(probe_cases, ("checks", "ik_success")),
                "probe_pose_ok_rate": case_bool_rate(probe_cases, ("checks", "pose_within_tolerance")),
                "probe_joint_delta_ok_rate": case_bool_rate(probe_cases, ("checks", "joint_delta_within_probe_limit")),
                "probe_position_error_m_max": float(np.nanmax(probe_pos)) if probe_pos else None,
                "probe_rotation_error_deg_max": float(np.nanmax(probe_rot)) if probe_rot else None,
                "probe_q_delta_abs_max_rad_max": float(np.nanmax(probe_q_abs)) if probe_q_abs else None,
                "probe_duration_s_mean": float(np.nanmean(probe_durations)) if probe_durations else None,
                "probe_duration_s_max": float(np.nanmax(probe_durations)) if probe_durations else None,
                "probe_failures": [
                    {
                        "sample_idx": sample.get("sample_idx"),
                        "label": case.get("label"),
                        "position_error_m": (case.get("ik_fk_vs_target_error") or {}).get("position_error_m"),
                        "rotation_error_deg": (case.get("ik_fk_vs_target_error") or {}).get("rotation_error_deg"),
                        "q_delta_abs_max_rad": case.get("q_delta_from_reference_abs_max_rad"),
                        "ik_success": (case.get("checks") or {}).get("ik_success"),
                        "pose_within_tolerance": (case.get("checks") or {}).get("pose_within_tolerance"),
                        "joint_delta_within_probe_limit": (case.get("checks") or {}).get("joint_delta_within_probe_limit"),
                    }
                    for sample in samples
                    for case in ((((sample.get("sides") or {}).get(side) or {}).get("ik_performance") or {}).get("probe_cases") or [])
                    if not (
                        ((case.get("checks") or {}).get("ik_success"))
                        and ((case.get("checks") or {}).get("pose_within_tolerance"))
                        and ((case.get("checks") or {}).get("joint_delta_within_probe_limit"))
                    )
                ][:20],
            },
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
        perf = rep.get("ik_performance") or {}
        probe_cases = perf.get("probe_cases") or []
        probe_ok = [
            bool((case.get("checks") or {}).get("ik_success"))
            and bool((case.get("checks") or {}).get("pose_within_tolerance"))
            and bool((case.get("checks") or {}).get("joint_delta_within_probe_limit"))
            for case in probe_cases
        ]
        probe_part = "probe=off"
        if probe_cases:
            probe_pos_max = max(float((case.get("ik_fk_vs_target_error") or {}).get("position_error_m", 0.0)) for case in probe_cases)
            probe_rot_max = max(float((case.get("ik_fk_vs_target_error") or {}).get("rotation_error_deg", 0.0)) for case in probe_cases)
            probe_q_max = max(float(case.get("q_delta_from_reference_abs_max_rad", 0.0)) for case in probe_cases)
            probe_part = (
                f"probe_ok={sum(probe_ok)}/{len(probe_ok)} "
                f"probe_max={probe_pos_max:.4f}m/{probe_rot_max:.2f}deg "
                f"probe_qmax={probe_q_max:.3f}rad"
            )
        parts.append(
            f"{side}: fk={fk['position_error_m']:.6f}m/{fk['rotation_error_deg']:.3f}deg "
            f"ik={ik_fk['position_error_m']:.6f}m/{ik_fk['rotation_error_deg']:.3f}deg "
            f"qmax={rep['q_delta_abs_max_rad']:.5f}rad success={ik['success']} "
            f"{probe_part}"
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
    parser.add_argument("--ik-probe-targets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ik-probe-translation-m", type=float, default=0.01)
    parser.add_argument("--ik-probe-rotation-deg", type=float, default=5.0)
    parser.add_argument("--ik-probe-max-joint-delta-rad", type=float, default=0.5)
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
