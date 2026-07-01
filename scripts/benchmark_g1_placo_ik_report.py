#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark G1 placo IK on manual FK/IK sampling reports."""

from __future__ import annotations

import argparse
import json
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
for path in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from g1_artifacts import artifact_dir, run_dir as artifact_run_dir  # noqa: E402
from g1_humanego_client_dry_run import json_safe  # noqa: E402
from g1_placo_ik import G1PlacoKinematics  # noqa: E402
from g1_urdf_ik import DEFAULT_G1_ZIP  # noqa: E402


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


def make_probe_targets(T: np.ndarray, translation_m: float, rotation_deg: float) -> list[dict[str, Any]]:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    step = float(translation_m)
    deg = float(rotation_deg)
    targets: list[dict[str, Any]] = []
    for name, delta in [
        ("translate_x_pos", [step, 0.0, 0.0]),
        ("translate_x_neg", [-step, 0.0, 0.0]),
        ("translate_y_pos", [0.0, step, 0.0]),
        ("translate_y_neg", [0.0, -step, 0.0]),
        ("translate_z_pos", [0.0, 0.0, step]),
        ("translate_z_neg", [0.0, 0.0, -step]),
    ]:
        target = T.copy()
        target[:3, 3] += np.asarray(delta, dtype=np.float64)
        targets.append({"label": name, "target_T": target, "target_spec": {"type": "base_translation", "delta_m": delta}})
    for axis in ["x", "y", "z"]:
        axis_vec = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}[axis]
        for sign, suffix in [(1.0, "pos"), (-1.0, "neg")]:
            target = T.copy()
            R_delta = Rotation.from_rotvec(np.asarray(axis_vec, dtype=np.float64) * np.deg2rad(sign * deg)).as_matrix()
            target[:3, :3] = R_delta @ target[:3, :3]
            targets.append(
                {
                    "label": f"rotate_{axis}_{suffix}",
                    "target_T": target,
                    "target_spec": {"type": "base_rotation", "axis": axis, "delta_rotation_deg": sign * deg},
                }
            )
    return targets


def reconstruct_probe_targets(side_report: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    base_T = np.asarray(side_report["sdk_T_link7_in_base"], dtype=np.float64).reshape(4, 4)
    perf = side_report.get("ik_performance") or {}
    old_cases = perf.get("probe_cases") or []
    if not old_cases:
        return make_probe_targets(base_T, args.probe_translation_m, args.probe_rotation_deg)

    targets = []
    for case in old_cases:
        spec = dict(case.get("target_spec") or {})
        label = str(case.get("label") or spec.get("name") or "probe")
        target = base_T.copy()
        if spec.get("type") == "base_translation":
            target[:3, 3] += np.asarray(spec.get("delta_m") or [0.0, 0.0, 0.0], dtype=np.float64)
        elif spec.get("type") == "base_rotation":
            axis = str(spec.get("axis") or "z")
            axis_vec = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}[axis]
            deg = float(spec.get("delta_rotation_deg") or 0.0)
            R_delta = Rotation.from_rotvec(np.asarray(axis_vec, dtype=np.float64) * np.deg2rad(deg)).as_matrix()
            target[:3, :3] = R_delta @ target[:3, :3]
        else:
            continue
        targets.append({"label": label, "target_T": target, "target_spec": spec})
    return targets


def run_case(
    kin: G1PlacoKinematics,
    *,
    side: str,
    label: str,
    q_seed: np.ndarray,
    waist: list[float],
    target_T: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    result = kin.solve_link7_ik(
        side,
        target_T,
        q_seed,
        waist_states=waist,
        position_weight=args.position_weight,
        orientation_weight=args.orientation_weight,
        position_tolerance_m=args.position_tolerance_m,
        rotation_tolerance_deg=args.rotation_tolerance_deg,
        max_joint_delta_rad=args.max_joint_delta_rad,
    )
    return {
        "label": label,
        "side": side,
        "target_T_link7_in_base": np.asarray(target_T, dtype=np.float64).tolist(),
        "placo": result.to_json(),
        "checks": {
            "success": bool(result.success),
            "pose_ok": result.position_error_m <= args.position_tolerance_m
            and result.rotation_error_deg <= args.rotation_tolerance_deg,
            "joint_delta_ok": result.q_delta_abs_max_rad <= args.max_joint_delta_rad,
            "within_joint_limits": bool(result.within_joint_limits),
        },
    }


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    def values(path: tuple[str, ...]) -> list[float]:
        result = []
        for case in cases:
            cur: Any = case
            for key in path:
                cur = (cur or {}).get(key)
            if cur is not None:
                result.append(float(cur))
        return result

    def rate(path: tuple[str, ...]) -> float | None:
        if not cases:
            return None
        passed = 0
        for case in cases:
            cur: Any = case
            for key in path:
                cur = (cur or {}).get(key)
            passed += 1 if bool(cur) else 0
        return float(passed / len(cases))

    pos = values(("placo", "position_error_m"))
    rot = values(("placo", "rotation_error_deg"))
    qmax = values(("placo", "q_delta_abs_max_rad"))
    durations = values(("placo", "duration_s"))
    return {
        "case_count": len(cases),
        "success_rate": rate(("checks", "success")),
        "pose_ok_rate": rate(("checks", "pose_ok")),
        "joint_delta_ok_rate": rate(("checks", "joint_delta_ok")),
        "within_joint_limits_rate": rate(("checks", "within_joint_limits")),
        "position_error_m_mean": float(np.nanmean(pos)) if pos else None,
        "position_error_m_max": float(np.nanmax(pos)) if pos else None,
        "rotation_error_deg_mean": float(np.nanmean(rot)) if rot else None,
        "rotation_error_deg_max": float(np.nanmax(rot)) if rot else None,
        "q_delta_abs_max_rad_mean": float(np.nanmean(qmax)) if qmax else None,
        "q_delta_abs_max_rad_max": float(np.nanmax(qmax)) if qmax else None,
        "duration_s_mean": float(np.nanmean(durations)) if durations else None,
        "duration_s_max": float(np.nanmax(durations)) if durations else None,
        "failures": [
            {
                "sample_idx": case.get("sample_idx"),
                "side": case.get("side"),
                "label": case.get("label"),
                "position_error_m": (case.get("placo") or {}).get("position_error_m"),
                "rotation_error_deg": (case.get("placo") or {}).get("rotation_error_deg"),
                "q_delta_abs_max_rad": (case.get("placo") or {}).get("q_delta_abs_max_rad"),
                "within_joint_limits": (case.get("placo") or {}).get("within_joint_limits"),
                "error": (case.get("placo") or {}).get("error"),
            }
            for case in cases
            if not ((case.get("checks") or {}).get("success"))
        ][:50],
    }


def benchmark_report(data: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    kin = G1PlacoKinematics(
        args.urdf_zip,
        max_iterations=args.max_iterations,
        dt=args.dt,
        regularization_weight=args.regularization_weight,
        manipulability_weight=args.manipulability_weight,
        enable_joint_limits=args.enable_placo_joint_limits,
    )
    samples = data.get("samples") or []
    all_cases: list[dict[str, Any]] = []
    actual_cases: list[dict[str, Any]] = []
    probe_cases: list[dict[str, Any]] = []
    for sample in samples[: args.max_samples if args.max_samples > 0 else None]:
        sample_idx = int(sample.get("sample_idx", len(actual_cases)))
        for side, side_report in (sample.get("sides") or {}).items():
            if side not in {"left", "right"}:
                continue
            q_seed = np.asarray(side_report["q_real_from_sdk"], dtype=np.float64).reshape(7)
            waist = side_report.get("waist_values_for_urdf") or [0.0, -0.3]
            actual_T = np.asarray(side_report["sdk_T_link7_in_base"], dtype=np.float64).reshape(4, 4)
            actual = run_case(
                kin,
                side=side,
                label="actual_pose",
                q_seed=q_seed,
                waist=waist,
                target_T=actual_T,
                args=args,
            )
            actual["sample_idx"] = sample_idx
            actual_cases.append(actual)
            all_cases.append(actual)
            if args.include_probes:
                for target in reconstruct_probe_targets(side_report, args):
                    case = run_case(
                        kin,
                        side=side,
                        label=target["label"],
                        q_seed=q_seed,
                        waist=waist,
                        target_T=target["target_T"],
                        args=args,
                    )
                    case["sample_idx"] = sample_idx
                    case["target_spec"] = target["target_spec"]
                    probe_cases.append(case)
                    all_cases.append(case)
    side_summaries: dict[str, Any] = {}
    for side in ["left", "right"]:
        side_summaries[side] = {
            "actual": summarize([case for case in actual_cases if case.get("side") == side]),
            "probe": summarize([case for case in probe_cases if case.get("side") == side]),
            "all": summarize([case for case in all_cases if case.get("side") == side]),
        }
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_report_args": data.get("args"),
        "benchmark_args": vars(args),
        "source_num_samples": len(samples),
        "tested_num_samples": len({case.get("sample_idx") for case in actual_cases}),
        "actual_cases": actual_cases,
        "probe_cases": probe_cases,
        "summary": {
            "actual": summarize(actual_cases),
            "probe": summarize(probe_cases),
            "all": summarize(all_cases),
            "sides": side_summaries,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_json")
    parser.add_argument("--out-dir", default=str(artifact_dir("diagnostics")))
    parser.add_argument("--tag", default="g1_placo_ik_benchmark")
    parser.add_argument("--urdf-zip", default=str(DEFAULT_G1_ZIP))
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples")
    parser.add_argument("--include-probes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--probe-translation-m", type=float, default=0.01)
    parser.add_argument("--probe-rotation-deg", type=float, default=5.0)
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--orientation-weight", type=float, default=0.2)
    parser.add_argument("--regularization-weight", type=float, default=1e-3)
    parser.add_argument("--manipulability-weight", type=float, default=0.0)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--dt", type=float, default=5e-2)
    parser.add_argument("--position-tolerance-m", type=float, default=0.001)
    parser.add_argument("--rotation-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--max-joint-delta-rad", type=float, default=0.5)
    parser.add_argument("--enable-placo-joint-limits", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    out_base = Path(args.out_dir).expanduser().resolve()
    default_base = artifact_dir("diagnostics")
    if out_base == default_base:
        run_dir = artifact_run_dir("diagnostics", args.tag, prefix="g1_placo_ik_benchmark")
    else:
        run_dir = out_base / f"g1_placo_ik_benchmark_{utc_stamp()}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = Path(args.report_json).expanduser().resolve()
    output: dict[str, Any] = {"ok": False, "report_json": str(report_path), "args": vars(args)}
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        started = time.time()
        output.update(benchmark_report(data, args))
        output["duration_s"] = time.time() - started
        output["ok"] = True
    except Exception as exc:
        output.update({
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })

    full_report_path = run_dir / "g1_placo_ik_benchmark_report.json"
    summary_path = run_dir / "g1_placo_ik_benchmark_summary.json"
    full_report_path.write_text(json.dumps(json_safe(output), ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(
        json.dumps(json_safe(output.get("summary") or {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    zip_path = make_zip(run_dir)
    print(json.dumps({
        "run_dir": str(run_dir),
        "zip_path": str(zip_path),
        "report_path": str(full_report_path),
        "summary_path": str(summary_path),
        "summary": output.get("summary"),
        "ok": output.get("ok"),
        "error": output.get("error"),
    }, ensure_ascii=False, indent=2))
    return 0 if output.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
