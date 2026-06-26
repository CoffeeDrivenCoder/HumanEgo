#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project G1 HumanEgo response poses back onto the saved RGB image."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import Any

import cv2
import numpy as np

from g1_artifacts import artifact_dir, legacy_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER_RUNS = artifact_dir("server")
LEGACY_SERVER_RUNS = legacy_dir("g1_humanego_server_runs")


COLORS = {
    "obj1": (30, 220, 30),
    "obj2": (40, 170, 255),
    "current_tcp": (255, 220, 40),
    "target_tcp": (255, 40, 220),
}
SHORT_LABELS = {
    "obj1": "obj1",
    "obj2": "obj2",
    "current_tcp": "tcp",
    "target_tcp": "target",
}
VIEW_LABELS = {
    "clean": "all clean",
    "objects": "objects",
    "tcp": "tcp",
    "axes": "pose axes",
    "all": "all + axes",
}
AXIS_COLORS = {
    "x": (0, 0, 255),
    "y": (0, 180, 0),
    "z": (255, 0, 0),
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


def project_point(K: np.ndarray, point_cam: np.ndarray) -> tuple[int, int] | None:
    x, y, z = [float(v) for v in point_cam[:3]]
    if not np.isfinite([x, y, z]).all() or z <= 1e-6:
        return None
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    if not np.isfinite([u, v]).all():
        return None
    return int(round(u)), int(round(v))


def draw_marker(
    image: np.ndarray,
    uv: tuple[int, int],
    label: str,
    color: tuple[int, int, int],
    radius: int = 8,
    show_uv: bool = False,
) -> None:
    h, w = image.shape[:2]
    u, v = uv
    in_view = 0 <= u < w and 0 <= v < h
    u_clip = int(np.clip(u, 0, w - 1))
    v_clip = int(np.clip(v, 0, h - 1))
    cv2.circle(image, (u_clip, v_clip), radius, color, 2, lineType=cv2.LINE_AA)
    cv2.drawMarker(image, (u_clip, v_clip), color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    short = SHORT_LABELS.get(label, label)
    if show_uv:
        text = f"{short} ({u},{v})" if in_view else f"{short} off ({u},{v})"
    else:
        text = short if in_view else f"{short} off"
    cv2.putText(
        image,
        text,
        (min(max(u_clip + 10, 4), w - 260), min(max(v_clip - 10, 18), h - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        3,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (min(max(u_clip + 10, 4), w - 260), min(max(v_clip - 10, 18), h - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        lineType=cv2.LINE_AA,
    )


def draw_pose_axes(
    image: np.ndarray,
    K: np.ndarray,
    T_in_cam: np.ndarray,
    label: str,
    axis_length_m: float,
) -> dict[str, Any]:
    origin = T_in_cam[:3, 3]
    origin_uv = project_point(K, origin)
    result: dict[str, Any] = {
        "origin_uv": None if origin_uv is None else list(origin_uv),
        "axis_length_m": float(axis_length_m),
        "axes": {},
    }
    if origin_uv is None:
        return result

    h, w = image.shape[:2]
    origin_xy = (
        int(np.clip(origin_uv[0], 0, w - 1)),
        int(np.clip(origin_uv[1], 0, h - 1)),
    )
    for idx, axis_name in enumerate(("x", "y", "z")):
        endpoint = origin + T_in_cam[:3, idx] * float(axis_length_m)
        endpoint_uv = project_point(K, endpoint)
        result["axes"][axis_name] = None if endpoint_uv is None else list(endpoint_uv)
        if endpoint_uv is None:
            continue
        endpoint_xy = (
            int(np.clip(endpoint_uv[0], 0, w - 1)),
            int(np.clip(endpoint_uv[1], 0, h - 1)),
        )
        cv2.arrowedLine(
            image,
            origin_xy,
            endpoint_xy,
            AXIS_COLORS[axis_name],
            2,
            line_type=cv2.LINE_AA,
            tipLength=0.22,
        )
        cv2.putText(
            image,
            f"{label}.{axis_name}",
            endpoint_xy,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 0),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"{label}.{axis_name}",
            endpoint_xy,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            AXIS_COLORS[axis_name],
            1,
            lineType=cv2.LINE_AA,
        )
    return result


def first_right_step(response: dict[str, Any]) -> dict[str, Any]:
    steps = (((response.get("policy_preview") or {}).get("sides") or {}).get("right") or [])
    return steps[0] if steps else {}


def collect_pose_matrices(response: dict[str, Any]) -> dict[str, np.ndarray]:
    poses: dict[str, np.ndarray] = {}
    input_summary = response.get("input_summary") or {}
    for key, item in (input_summary.get("objects") or {}).items():
        T = matrix_from((item or {}).get("T_in_cam"))
        if T is not None:
            poses[str(key)] = T

    T_current = matrix_from(input_summary.get("current_T_tcp_in_cam"))
    if T_current is not None:
        poses["current_tcp"] = T_current

    T_target = matrix_from(first_right_step(response).get("T_tcp_target_in_cam"))
    if T_target is not None:
        poses["target_tcp"] = T_target
    return poses


def labels_for_view(view: str) -> set[str]:
    if view in {"clean", "all"}:
        return {"obj1", "obj2", "current_tcp", "target_tcp"}
    if view == "objects":
        return {"obj1", "obj2"}
    if view == "tcp":
        return {"current_tcp", "target_tcp"}
    if view == "axes":
        return {"obj1", "obj2", "current_tcp", "target_tcp"}
    raise ValueError(f"unknown view: {view}")


def output_path_for_view(run_dir: Path, view: str, out_path: Path | None) -> Path:
    if out_path is not None:
        return out_path
    if view == "clean":
        return run_dir / "response_projection_clean.jpg"
    return run_dir / f"response_projection_{view}.jpg"


def visualize_run(
    run_dir: Path,
    out_path: Path | None,
    view: str,
    axis_length_m: float,
    show_uv: bool,
) -> dict[str, Any]:
    response_path = run_dir / "response.json"
    rgb_path = run_dir / "rgb_bgr.jpg"
    if not response_path.exists():
        raise FileNotFoundError(f"missing response.json: {response_path}")
    if not rgb_path.exists():
        raise FileNotFoundError(f"missing rgb_bgr.jpg: {rgb_path}")

    response = load_json(response_path)
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read image: {rgb_path}")

    K = np.asarray((response.get("input_summary") or {}).get("K"), dtype=np.float64).reshape(3, 3)
    poses = collect_pose_matrices(response)
    selected = labels_for_view(view)
    draw_axes = view in {"axes", "all"}
    projected: dict[str, Any] = {}
    for label, T_in_cam in poses.items():
        if label not in selected:
            continue
        point_cam = T_in_cam[:3, 3]
        uv = project_point(K, point_cam)
        projected[label] = None if uv is None else {
            "uv": list(uv),
            "cam_xyz_m": np.round(point_cam, 4).tolist(),
        }
        if uv is not None:
            draw_marker(image, uv, label, COLORS.get(label, (255, 255, 255)), show_uv=show_uv)
        if draw_axes:
            axis_projection = draw_pose_axes(image, K, T_in_cam, label, axis_length_m)
            if isinstance(projected.get(label), dict):
                projected[label]["axis_projection"] = axis_projection

    current_uv = projected.get("current_tcp", {}).get("uv") if isinstance(projected.get("current_tcp"), dict) else None
    target_uv = projected.get("target_tcp", {}).get("uv") if isinstance(projected.get("target_tcp"), dict) else None
    if view in {"clean", "tcp", "all", "axes"} and current_uv and target_uv:
        cv2.arrowedLine(
            image,
            tuple(current_uv),
            tuple(target_uv),
            COLORS["target_tcp"],
            2,
            line_type=cv2.LINE_AA,
            tipLength=0.12,
        )

    title = (
        f"{VIEW_LABELS[view]} | {run_dir.name} | "
        f"source={(response.get('input_summary') or {}).get('object_source_used')}"
    )
    cv2.putText(image, title[:130], (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, lineType=cv2.LINE_AA)
    cv2.putText(image, title[:130], (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    out_path = output_path_for_view(run_dir, view, out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), image)
    if not ok:
        raise RuntimeError(f"failed to write {out_path}")

    return {
        "run": run_dir.name,
        "out": str(out_path),
        "view": view,
        "object_source_used": (response.get("input_summary") or {}).get("object_source_used"),
        "projected": projected,
    }


def latest_run_with_rgb(root: Path) -> Path:
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
    return max(candidates, key=lambda path: (path / "response.json").stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path, help="Run directories containing response.json and rgb_bgr.jpg")
    parser.add_argument("--server-runs", type=Path, default=DEFAULT_SERVER_RUNS)
    parser.add_argument("--session", default=None, help="Artifact session name, e.g. 20260626_pose_gate")
    parser.add_argument("--out", type=Path, default=None, help="Output path; only valid with one input run")
    parser.add_argument("--latest", action="store_true", help="Use the latest server run with saved RGB")
    parser.add_argument("--view", choices=["clean", "objects", "tcp", "axes", "all"], default="clean")
    parser.add_argument("--split-layers", action="store_true", help="Write clean, objects, tcp, and axes views")
    parser.add_argument("--show-uv", action="store_true", help="Include pixel coordinates in image labels")
    parser.add_argument("--axis-length-m", type=float, default=0.06)
    args = parser.parse_args()

    if args.session is not None:
        args.server_runs = artifact_dir("server").parents[1] / args.session / "server"

    if args.latest:
        runs = [latest_run_with_rgb(args.server_runs)]
    else:
        runs = args.runs
    if not runs:
        parser.error("provide at least one run directory or use --latest")
    if args.out is not None and (len(runs) != 1 or args.split_layers):
        parser.error("--out can only be used with exactly one run and without --split-layers")

    summaries = []
    for run in runs:
        run_dir = run.expanduser().resolve()
        views = ["clean", "objects", "tcp", "axes"] if args.split_layers else [args.view]
        for view in views:
            summaries.append(
                visualize_run(
                    run_dir,
                    args.out,
                    view=view,
                    axis_length_m=float(args.axis_length_m),
                    show_uv=bool(args.show_uv),
                )
            )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
