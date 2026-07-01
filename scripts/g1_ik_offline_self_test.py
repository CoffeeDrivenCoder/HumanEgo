#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline self-test for G1 left/right URDF FK/IK without robot hardware."""

from __future__ import annotations

import argparse
import json
import math
import sys
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
from g1_urdf_ik import G1UrdfKinematics, DEFAULT_G1_ZIP, pose_error  # noqa: E402


def T_delta_translation(axis: str, delta_m: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[{"x": 0, "y": 1, "z": 2}[axis], 3] = float(delta_m)
    return T


def T_delta_rotation(axis: str, deg: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    vec = np.zeros(3, dtype=np.float64)
    vec[{"x": 0, "y": 1, "z": 2}[axis]] = math.radians(float(deg))
    T[:3, :3] = Rotation.from_rotvec(vec).as_matrix()
    return T


def compose_target(start_T: np.ndarray, delta_T: np.ndarray, rotation_frame: str = "base") -> np.ndarray:
    start_T = np.asarray(start_T, dtype=np.float64).reshape(4, 4)
    delta_T = np.asarray(delta_T, dtype=np.float64).reshape(4, 4)
    out = start_T.copy()
    out[:3, 3] = start_T[:3, 3] + delta_T[:3, 3]
    if rotation_frame == "base":
        out[:3, :3] = delta_T[:3, :3] @ start_T[:3, :3]
    elif rotation_frame == "local":
        out[:3, :3] = start_T[:3, :3] @ delta_T[:3, :3]
    else:
        raise ValueError(rotation_frame)
    return out


def small_targets() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for axis in ["x", "y", "z"]:
        specs.append({"name": f"+{axis}_1cm", "delta_T": T_delta_translation(axis, 0.01)})
        specs.append({"name": f"-{axis}_1cm", "delta_T": T_delta_translation(axis, -0.01)})
    for axis in ["x", "y", "z"]:
        specs.append({"name": f"+r{axis}_2deg", "delta_T": T_delta_rotation(axis, 2.0)})
        specs.append({"name": f"-r{axis}_2deg", "delta_T": T_delta_rotation(axis, -2.0)})
    combo = T_delta_translation("x", 0.01)
    combo[:3, :3] = T_delta_rotation("z", 2.0)[:3, :3]
    specs.append({"name": "+x_1cm_+rz_2deg", "delta_T": combo})
    return specs


def test_side(kin: G1UrdfKinematics, side: str, max_nfev: int) -> dict[str, Any]:
    q_home = kin.home_q(side)
    home_T = kin.link7_fk(side, q_home)
    current_ik = kin.solve_link7_ik(side, home_T, q_home, max_nfev=max_nfev)
    current_fk = kin.link7_fk(side, current_ik.q_solution)
    current_err = pose_error(current_fk, home_T)
    rows: list[dict[str, Any]] = []
    q_seed = q_home.copy()
    for spec in small_targets():
        target_T = compose_target(home_T, spec["delta_T"], rotation_frame="base")
        result = kin.solve_link7_ik(side, target_T, q_seed, max_nfev=max_nfev)
        fk_T = kin.link7_fk(side, result.q_solution)
        err = pose_error(fk_T, target_T)
        rows.append(
            {
                "name": spec["name"],
                "ik": result.to_json(),
                "target_T_link7_in_base": target_T.tolist(),
                "fk_T_link7_in_base": fk_T.tolist(),
                "pose_error": err,
            }
        )
        if result.success:
            q_seed = result.q_solution
    return {
        "side": side,
        "model": kin.describe_side(side),
        "home_q": q_home.tolist(),
        "home_T_link7_in_base": home_T.tolist(),
        "current_pose_self_consistency": {
            "ik": current_ik.to_json(),
            "fk_T_link7_in_base": current_fk.tolist(),
            "pose_error": current_err,
        },
        "small_target_batch": rows,
        "summary": {
            "current_success": bool(current_ik.success),
            "small_targets": len(rows),
            "small_successes": sum(1 for row in rows if (row["ik"] or {}).get("success")),
            "max_position_error_m": max(float(row["pose_error"]["position_error_m"]) for row in rows) if rows else None,
            "max_rotation_error_deg": max(float(row["pose_error"]["rotation_error_deg"]) for row in rows) if rows else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf-zip", default=str(DEFAULT_G1_ZIP))
    parser.add_argument("--out-dir", default=str(artifact_dir("diagnostics")))
    parser.add_argument("--tag", default="g1_ik_offline_self_test")
    parser.add_argument("--side", action="append", choices=["left", "right"], default=[])
    parser.add_argument("--max-nfev", type=int, default=300)
    args = parser.parse_args()

    sides = args.side or ["left", "right"]
    out_base = Path(args.out_dir).expanduser().resolve()
    default_base = artifact_dir("diagnostics")
    if out_base == default_base:
        run_dir = artifact_run_dir("diagnostics", args.tag, prefix="g1_ik_offline")
    else:
        run_dir = out_base / f"g1_ik_offline_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    kin = G1UrdfKinematics(args.urdf_zip)
    side_reports = {side: test_side(kin, side, args.max_nfev) for side in sides}
    report = {
        "ok": all(rep["summary"]["current_success"] for rep in side_reports.values()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "urdf_zip": str(Path(args.urdf_zip).expanduser().resolve()),
        "sides": side_reports,
    }
    report_path = run_dir / "g1_ik_offline_self_test_report.json"
    report_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        side: rep["summary"]
        for side, rep in side_reports.items()
    }
    summary_path = run_dir / "g1_ik_offline_self_test_summary.json"
    summary_path.write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "report_path": str(report_path), "summary_path": str(summary_path), "summary": summary}, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

