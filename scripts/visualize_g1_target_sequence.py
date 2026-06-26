#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize recent HumanEgo TCP targets as a short projected sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from g1_artifacts import artifact_dir, legacy_dir, utc_stamp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER_RUNS = artifact_dir("server")
LEGACY_SERVER_RUNS = legacy_dir("g1_humanego_server_runs")
TARGET_COLORS = [
    (255, 70, 230),
    (210, 40, 255),
    (150, 30, 255),
    (90, 60, 255),
    (40, 110, 255),
]
CURRENT_COLOR = (255, 220, 40)
OBJ_COLORS = {
    "obj1": (30, 220, 30),
    "obj2": (40, 170, 255),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def matrix_from(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.size != 16:
        return None
    return arr.reshape(4, 4)


def first_right_step(response: dict[str, Any]) -> dict[str, Any]:
    steps = (((response.get("policy_preview") or {}).get("sides") or {}).get("right") or [])
    return steps[0] if steps else {}


def project_point(K: np.ndarray, point_cam: np.ndarray) -> tuple[float, float] | None:
    x, y, z = [float(v) for v in point_cam[:3]]
    if not np.isfinite([x, y, z]).all() or z <= 1e-9:
        return None
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    if not np.isfinite([u, v]).all():
        return None
    return float(u), float(v)


def clamp_uv(image: np.ndarray, uv: tuple[float, float]) -> tuple[int, int]:
    h, w = image.shape[:2]
    u, v = uv
    return int(np.clip(round(u), 0, w - 1)), int(np.clip(round(v), 0, h - 1))


def draw_text(image: np.ndarray, text: str, xy: tuple[int, int], color: tuple[int, int, int], scale: float = 0.48) -> None:
    x, y = xy
    h, w = image.shape[:2]
    x = int(np.clip(x, 4, w - 160))
    y = int(np.clip(y, 18, h - 8))
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, lineType=cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, lineType=cv2.LINE_AA)


def draw_point(
    image: np.ndarray,
    uv: tuple[float, float],
    label: str,
    color: tuple[int, int, int],
    *,
    radius: int = 7,
    filled: bool = False,
) -> None:
    xy = clamp_uv(image, uv)
    thickness = -1 if filled else 2
    cv2.circle(image, xy, radius, color, thickness, lineType=cv2.LINE_AA)
    cv2.drawMarker(image, xy, color, markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
    draw_text(image, label, (xy[0] + 9, xy[1] - 9), color)


def latest_response_runs(root: Path, count: int) -> list[Path]:
    roots = [root]
    if root == DEFAULT_SERVER_RUNS and LEGACY_SERVER_RUNS.exists():
        roots.append(LEGACY_SERVER_RUNS)
    candidates = [
        path.parent
        for item in roots
        for path in sorted(item.glob("*/response.json"))
        if (path.parent / "rgb_bgr.jpg").exists()
    ]
    if not candidates:
        raise RuntimeError(f"No server run with response.json and rgb_bgr.jpg under {', '.join(str(p) for p in roots)}")
    ordered = sorted(candidates, key=lambda path: (path / "response.json").stat().st_mtime)
    return ordered[-max(1, int(count)) :]


def response_row(run_dir: Path) -> dict[str, Any]:
    response = load_json(run_dir / "response.json")
    input_summary = response.get("input_summary") or {}
    K = np.asarray(input_summary.get("K"), dtype=np.float64).reshape(3, 3)
    step = first_right_step(response)
    current_T = matrix_from(input_summary.get("current_T_tcp_in_cam"))
    target_T = matrix_from(step.get("T_tcp_target_in_cam"))
    if current_T is None:
        raise RuntimeError(f"{run_dir} missing input_summary.current_T_tcp_in_cam")
    if target_T is None:
        raise RuntimeError(f"{run_dir} missing policy_preview target TCP")

    current_xyz = current_T[:3, 3]
    target_xyz = target_T[:3, 3]
    current_uv = project_point(K, current_xyz)
    target_uv = project_point(K, target_xyz)
    safety = step.get("safety_translation_limit") or {}
    objects = {}
    for key, item in (input_summary.get("objects") or {}).items():
        obj_T = matrix_from((item or {}).get("T_in_cam"))
        if obj_T is None:
            continue
        obj_xyz = obj_T[:3, 3]
        objects[str(key)] = {
            "cam_xyz_m": np.round(obj_xyz, 6).tolist(),
            "uv": None if project_point(K, obj_xyz) is None else np.round(project_point(K, obj_xyz), 2).tolist(),
        }
    return {
        "run": run_dir.name,
        "path": str(run_dir),
        "response_mtime": (run_dir / "response.json").stat().st_mtime,
        "object_source_used": input_summary.get("object_source_used"),
        "done_prob": (response.get("policy_preview") or {}).get("done_prob"),
        "current_tcp_cam_xyz_m": np.round(current_xyz, 6).tolist(),
        "target_tcp_cam_xyz_m": np.round(target_xyz, 6).tolist(),
        "current_tcp_uv": None if current_uv is None else np.round(current_uv, 2).tolist(),
        "target_tcp_uv": None if target_uv is None else np.round(target_uv, 2).tolist(),
        "target_minus_current_cam_m": np.round(target_xyz - current_xyz, 6).tolist(),
        "target_minus_current_norm_m": float(np.linalg.norm(target_xyz - current_xyz)),
        "raw_delta_norm_m": safety.get("raw_delta_norm_m"),
        "clipped": safety.get("clipped"),
        "clipped_delta_norm_m": safety.get("clipped_delta_norm_m"),
        "gripper_g1_raw_0_open_120_closed": step.get("gripper_g1_raw_0_open_120_closed"),
        "objects": objects,
    }


def visualize_sequence(
    run_dirs: list[Path],
    out_path: Path,
    json_out: Path,
    draw_objects: bool,
    show_current: bool,
) -> dict[str, Any]:
    rows = [response_row(path) for path in run_dirs]
    latest_run = run_dirs[-1]
    image = cv2.imread(str(latest_run / "rgb_bgr.jpg"), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read image: {latest_run / 'rgb_bgr.jpg'}")

    if draw_objects:
        latest_objects = rows[-1]["objects"]
        for key in sorted(latest_objects):
            uv = latest_objects[key].get("uv")
            if uv is None:
                continue
            draw_point(image, tuple(uv), key, OBJ_COLORS.get(key, (255, 255, 255)), radius=6)

    target_points: list[tuple[int, int]] = []
    current_points: list[tuple[int, int]] = []
    for idx, row in enumerate(rows, start=1):
        target_uv = row.get("target_tcp_uv")
        current_uv = row.get("current_tcp_uv")
        color = TARGET_COLORS[(idx - 1) % len(TARGET_COLORS)]
        if show_current and current_uv is not None:
            current_xy = clamp_uv(image, tuple(current_uv))
            current_points.append(current_xy)
            cv2.circle(image, current_xy, 4, CURRENT_COLOR, 1, lineType=cv2.LINE_AA)
            draw_text(image, f"tcp{idx}", (current_xy[0] + 7, current_xy[1] + 15), CURRENT_COLOR, scale=0.38)
        if target_uv is None:
            continue
        target_xy = clamp_uv(image, tuple(target_uv))
        target_points.append(target_xy)
        draw_point(image, tuple(target_uv), f"target{idx}", color, radius=8, filled=False)
        if show_current and current_uv is not None:
            cv2.arrowedLine(image, clamp_uv(image, tuple(current_uv)), target_xy, color, 1, line_type=cv2.LINE_AA, tipLength=0.12)

    for prev_xy, next_xy in zip(target_points, target_points[1:]):
        cv2.arrowedLine(image, prev_xy, next_xy, (255, 255, 255), 4, line_type=cv2.LINE_AA, tipLength=0.18)
        cv2.arrowedLine(image, prev_xy, next_xy, (255, 40, 220), 2, line_type=cv2.LINE_AA, tipLength=0.18)

    title = f"target sequence last {len(rows)} | background={latest_run.name}"
    draw_text(image, title[:120], (12, 24), (255, 255, 255), scale=0.55)

    for prev, row in zip(rows, rows[1:]):
        prev_xyz = np.asarray(prev["target_tcp_cam_xyz_m"], dtype=np.float64)
        xyz = np.asarray(row["target_tcp_cam_xyz_m"], dtype=np.float64)
        row["target_step_from_prev_cam_m"] = np.round(xyz - prev_xyz, 6).tolist()
        row["target_step_from_prev_norm_m"] = float(np.linalg.norm(xyz - prev_xyz))
        prev_uv = prev.get("target_tcp_uv")
        uv = row.get("target_tcp_uv")
        if prev_uv is not None and uv is not None:
            uv_delta = np.asarray(uv, dtype=np.float64) - np.asarray(prev_uv, dtype=np.float64)
            row["target_step_from_prev_uv_px"] = np.round(uv_delta, 2).tolist()
            row["target_step_from_prev_uv_norm_px"] = float(np.linalg.norm(uv_delta))

    payload = {
        "generated_utc": utc_stamp(),
        "background_run": latest_run.name,
        "runs": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), image):
        raise RuntimeError(f"failed to write {out_path}")
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path, help="Run directories containing response.json and rgb_bgr.jpg")
    parser.add_argument("--server-runs", type=Path, default=DEFAULT_SERVER_RUNS)
    parser.add_argument("--session", default=None, help="Artifact session name, e.g. 20260626_pose_gate")
    parser.add_argument("--latest", type=int, default=3, help="Use latest N server runs when no explicit runs are provided")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--no-objects", action="store_true", help="Do not draw latest obj1/obj2 positions")
    parser.add_argument("--no-current", action="store_true", help="Do not draw current TCP points or current-to-target arrows")
    args = parser.parse_args()

    if args.session is not None:
        args.server_runs = artifact_dir("server").parents[1] / args.session / "server"

    if args.runs:
        run_dirs = [path.expanduser().resolve() for path in args.runs]
    else:
        run_dirs = latest_response_runs(args.server_runs.expanduser().resolve(), args.latest)
    if len(run_dirs) < 2:
        parser.error("target sequence visualization needs at least two runs")

    latest_run = run_dirs[-1]
    out_path = args.out or latest_run / f"target_sequence_last{len(run_dirs)}.jpg"
    json_out = args.json_out or latest_run / f"target_sequence_last{len(run_dirs)}.json"
    payload = visualize_sequence(
        run_dirs=run_dirs,
        out_path=out_path.expanduser().resolve(),
        json_out=json_out.expanduser().resolve(),
        draw_objects=not args.no_objects,
        show_current=not args.no_current,
    )
    print(f"wrote image: {out_path.expanduser().resolve()}")
    print(f"wrote json:  {json_out.expanduser().resolve()}")
    for row in payload["runs"]:
        step = row.get("target_step_from_prev_norm_m")
        step_text = "-" if step is None else f"{step:.4f}m"
        print(
            f"{row['run']} source={row.get('object_source_used')} "
            f"done={float(row.get('done_prob') or 0.0):.4f} "
            f"target_uv={row.get('target_tcp_uv')} "
            f"target_step_from_prev={step_text}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
