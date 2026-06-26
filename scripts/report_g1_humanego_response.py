#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print numeric checks for one G1 HumanEgo response."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import Any

import numpy as np

from g1_artifacts import artifact_dir, legacy_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER_RUNS = artifact_dir("server")
LEGACY_SERVER_RUNS = legacy_dir("g1_humanego_server_runs")
EPS = 1e-12


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def matrix_from(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.size != 16:
        return None
    return arr.reshape(4, 4)


def project_point(K: np.ndarray, point_cam: np.ndarray) -> tuple[float, float] | None:
    x, y, z = [float(v) for v in point_cam[:3]]
    if not np.isfinite([x, y, z]).all() or z <= EPS:
        return None
    return (float(K[0, 0] * x / z + K[0, 2]), float(K[1, 1] * y / z + K[1, 2]))


def rotation_angle_deg(R_delta: np.ndarray) -> float:
    value = (float(np.trace(R_delta)) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def rotation_vector_deg(R_delta: np.ndarray) -> np.ndarray:
    angle_deg = rotation_angle_deg(R_delta)
    angle_rad = np.radians(angle_deg)
    if abs(angle_rad) <= EPS:
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [
            R_delta[2, 1] - R_delta[1, 2],
            R_delta[0, 2] - R_delta[2, 0],
            R_delta[1, 0] - R_delta[0, 1],
        ],
        dtype=np.float64,
    )
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= EPS:
        return np.zeros(3, dtype=np.float64)
    return axis / axis_norm * angle_deg


def first_right_step(response: dict[str, Any]) -> dict[str, Any]:
    steps = (((response.get("policy_preview") or {}).get("sides") or {}).get("right") or [])
    return steps[0] if steps else {}


def latest_response_run(root: Path) -> Path:
    roots = [root]
    if root == DEFAULT_SERVER_RUNS and LEGACY_SERVER_RUNS.exists():
        roots.append(LEGACY_SERVER_RUNS)
    candidates = []
    for item in roots:
        candidates.extend(path.parent for path in sorted(item.glob("*/response.json")))
    if not candidates:
        raise RuntimeError(f"No response.json files under {', '.join(str(p) for p in roots)}")
    return max(candidates, key=lambda path: (path / "response.json").stat().st_mtime)


def fmt_vec(vec: np.ndarray | list[float] | tuple[float, ...], digits: int = 4) -> str:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{v:+.{digits}f}" for v in arr) + "]"


def fmt_uv(uv: tuple[float, float] | None) -> str:
    if uv is None:
        return "-"
    return f"({uv[0]:.1f}, {uv[1]:.1f})"


def response_path_from_arg(path: Path) -> Path:
    path = path.expanduser()
    if path.is_dir():
        return path / "response.json"
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_or_response", nargs="?", type=Path)
    parser.add_argument("--server-runs", type=Path, default=DEFAULT_SERVER_RUNS)
    parser.add_argument("--session", default=None, help="Artifact session name, e.g. 20260626_pose_gate")
    parser.add_argument("--latest", action="store_true")
    args = parser.parse_args()

    if args.session is not None:
        args.server_runs = artifact_dir("server").parents[1] / args.session / "server"

    if args.latest:
        run_dir = latest_response_run(args.server_runs)
        response_path = run_dir / "response.json"
    elif args.run_or_response is not None:
        response_path = response_path_from_arg(args.run_or_response)
        run_dir = response_path.parent
    else:
        parser.error("provide a run/response path or use --latest")

    response = load_json(response_path)
    input_summary = response.get("input_summary") or {}
    K = np.asarray(input_summary.get("K"), dtype=np.float64).reshape(3, 3)
    step0 = first_right_step(response)

    T_current_tcp_cam = matrix_from(input_summary.get("current_T_tcp_in_cam"))
    T_target_tcp_cam = matrix_from(step0.get("T_tcp_target_in_cam"))
    T_current_link7_base = None
    request_path = run_dir / "request_summary.json"
    if request_path.exists():
        request = load_json(request_path)
        current = request.get("current") or {}
        T_current_link7_base = matrix_from(current.get("T_link7_in_base"))

    T_target_link7_base = matrix_from(step0.get("T_link7_target_in_base"))
    T_target_link7_base_limited = matrix_from(step0.get("T_link7_target_in_base_limited"))

    print(f"run: {run_dir.name}")
    print(f"response: {response_path}")
    print(f"ok: {response.get('ok')}")
    print(f"object_source_used: {input_summary.get('object_source_used')}")
    object_error = input_summary.get("object_error")
    print(f"object_error: {object_error if object_error else '-'}")
    print(f"done_prob: {(response.get('policy_preview') or {}).get('done_prob')}")
    print(f"latency_s: {response.get('latency_s')}")
    print("")

    print("camera-frame poses:")
    if T_current_tcp_cam is not None:
        print(f"  current_tcp xyz_m={fmt_vec(T_current_tcp_cam[:3, 3])} uv={fmt_uv(project_point(K, T_current_tcp_cam[:3, 3]))}")
    if T_target_tcp_cam is not None:
        print(f"  target_tcp  xyz_m={fmt_vec(T_target_tcp_cam[:3, 3])} uv={fmt_uv(project_point(K, T_target_tcp_cam[:3, 3]))}")
    if T_current_tcp_cam is not None and T_target_tcp_cam is not None:
        d = T_target_tcp_cam[:3, 3] - T_current_tcp_cam[:3, 3]
        R_delta = T_target_tcp_cam[:3, :3] @ T_current_tcp_cam[:3, :3].T
        print(f"  target-current tcp delta_cam_m={fmt_vec(d)} norm={np.linalg.norm(d):.4f}")
        print(
            "  target-current tcp rot_delta_deg="
            f"{rotation_angle_deg(R_delta):.2f} rotvec_deg={fmt_vec(rotation_vector_deg(R_delta), 2)}"
        )

    for key, item in (input_summary.get("objects") or {}).items():
        T_obj_cam = matrix_from((item or {}).get("T_in_cam"))
        if T_obj_cam is None:
            continue
        print(
            f"  {key:<10} xyz_m={fmt_vec(T_obj_cam[:3, 3])} "
            f"uv={fmt_uv(project_point(K, T_obj_cam[:3, 3]))} "
            f"kpts={item.get('kpts_local_count')}"
        )

    print("")
    print("base-frame link7 target:")
    if T_current_link7_base is not None:
        print(f"  current_link7 xyz_m={fmt_vec(T_current_link7_base[:3, 3])}")
    if T_target_link7_base is not None:
        print(f"  raw_target    xyz_m={fmt_vec(T_target_link7_base[:3, 3])}")
    if T_target_link7_base_limited is not None:
        print(f"  limited_target xyz_m={fmt_vec(T_target_link7_base_limited[:3, 3])}")
    if T_current_link7_base is not None and T_target_link7_base is not None:
        d_base = T_target_link7_base[:3, 3] - T_current_link7_base[:3, 3]
        R_delta_base = T_target_link7_base[:3, :3] @ T_current_link7_base[:3, :3].T
        print(f"  raw delta_base_m={fmt_vec(d_base)} norm={np.linalg.norm(d_base):.4f}")
        print(
            "  raw rot_delta_base_deg="
            f"{rotation_angle_deg(R_delta_base):.2f} rotvec_deg={fmt_vec(rotation_vector_deg(R_delta_base), 2)}"
        )

    safety = step0.get("safety_translation_limit") or {}
    if safety:
        print("")
        print("server safety preview:")
        print(f"  raw_delta_m={fmt_vec(safety.get('raw_delta_m', []))}")
        print(f"  raw_delta_norm_m={safety.get('raw_delta_norm_m')}")
        print(f"  clipped={safety.get('clipped')}")
        print(f"  clipped_delta_m={fmt_vec(safety.get('clipped_delta_m', []))}")
        print(f"  clipped_delta_norm_m={safety.get('clipped_delta_norm_m')}")
    if "gripper_g1_raw_0_open_120_closed" in step0:
        print(f"  gripper_g1_raw_0_open_120_closed={step0['gripper_g1_raw_0_open_120_closed']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
