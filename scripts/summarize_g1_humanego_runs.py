#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize G1 HumanEgo server/client run artifacts.

The main question this answers is whether recent RGB-D object poses are stable
enough to continue from dry-run into interactive one-step robot control.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import json
import math
from typing import Any

import numpy as np

from g1_artifacts import artifact_dir, legacy_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER_RUNS = artifact_dir("server")
LEGACY_SERVER_RUNS = legacy_dir("g1_humanego_server_runs")


@dataclass
class RunSummary:
    run: str
    path: Path
    ok: bool
    source: str | None
    error_type: str | None
    obj_pos_cam: dict[str, np.ndarray]
    done_prob: float | None
    latency_s: float | None
    raw_delta_norm_m: float | None
    clipped_delta_norm_m: float | None
    clipped: bool | None
    max_step_m: float | None
    gripper_g1: float | None


def _matrix_pos(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.size != 16:
        return None
    return arr.reshape(4, 4)[:3, 3]


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def load_response(path: Path) -> RunSummary:
    data = json.loads(path.read_text(encoding="utf-8"))
    input_summary = data.get("input_summary") or {}
    objects = input_summary.get("objects") or {}
    obj_pos_cam: dict[str, np.ndarray] = {}
    for key, item in objects.items():
        pos = _matrix_pos((item or {}).get("T_in_cam"))
        if pos is not None:
            obj_pos_cam[str(key)] = pos

    preview = data.get("policy_preview") or {}
    right_steps = ((preview.get("sides") or {}).get("right") or [])
    step0 = right_steps[0] if right_steps else {}
    safety = step0.get("safety_translation_limit") or {}
    object_error = input_summary.get("object_error") or {}

    return RunSummary(
        run=path.parent.name,
        path=path,
        ok=bool(data.get("ok")),
        source=input_summary.get("object_source_used"),
        error_type=object_error.get("error_type"),
        obj_pos_cam=obj_pos_cam,
        done_prob=_float_or_none(preview.get("done_prob")),
        latency_s=_float_or_none(data.get("latency_s")),
        raw_delta_norm_m=_float_or_none(safety.get("raw_delta_norm_m")),
        clipped_delta_norm_m=_float_or_none(safety.get("clipped_delta_norm_m")),
        clipped=safety.get("clipped") if isinstance(safety.get("clipped"), bool) else None,
        max_step_m=_float_or_none(safety.get("max_step_m")),
        gripper_g1=_float_or_none(step0.get("gripper_g1_raw_0_open_120_closed")),
    )


def describe_values(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def describe_positions(rows: list[RunSummary], obj_key: str) -> dict[str, Any]:
    pts = [row.obj_pos_cam[obj_key] for row in rows if obj_key in row.obj_pos_cam]
    if not pts:
        return {"n": 0}
    arr = np.stack(pts, axis=0)
    diffs = np.linalg.norm(np.diff(arr, axis=0), axis=1) if len(arr) > 1 else np.asarray([], dtype=np.float64)
    return {
        "n": int(len(arr)),
        "mean_cam_m": np.round(arr.mean(axis=0), 4).tolist(),
        "std_cam_m": np.round(arr.std(axis=0), 4).tolist(),
        "min_cam_m": np.round(arr.min(axis=0), 4).tolist(),
        "max_cam_m": np.round(arr.max(axis=0), 4).tolist(),
        "step_delta_norm_m": {
            "mean": None if len(diffs) == 0 else float(np.mean(diffs)),
            "max": None if len(diffs) == 0 else float(np.max(diffs)),
        },
    }


def print_counter(title: str, counter: Counter) -> None:
    print(title)
    if not counter:
        print("  none")
        return
    for key, count in counter.most_common():
        print(f"  {key if key is not None else '<missing>'}: {count}")


def print_position_block(title: str, stats: dict[str, Any]) -> None:
    print(title)
    if stats.get("n", 0) == 0:
        print("  no samples")
        return
    delta = stats["step_delta_norm_m"]
    print(f"  n: {stats['n']}")
    print(f"  mean cam xyz m: {stats['mean_cam_m']}")
    print(f"  std  cam xyz m: {stats['std_cam_m']}")
    print(f"  min  cam xyz m: {stats['min_cam_m']}")
    print(f"  max  cam xyz m: {stats['max_cam_m']}")
    if delta["mean"] is not None:
        print(f"  step delta norm m: mean={delta['mean']:.4f}, max={delta['max']:.4f}")


def format_optional(value: float | None, ndigits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{ndigits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-runs", type=Path, default=DEFAULT_SERVER_RUNS)
    parser.add_argument("--session", default=None, help="Artifact session name, e.g. 20260626_pose_gate")
    parser.add_argument("--last", type=int, default=30, help="How many latest responses to include in the detailed tail")
    parser.add_argument("--recent-rgbd", type=int, default=10, help="How many latest RGB-D responses to summarize separately")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if args.session is not None:
        args.server_runs = artifact_dir("server").parents[1] / args.session / "server"

    server_runs = Path(args.server_runs).expanduser().resolve()
    roots = [server_runs]
    if server_runs == DEFAULT_SERVER_RUNS and LEGACY_SERVER_RUNS.exists():
        roots.append(LEGACY_SERVER_RUNS)
    response_paths = sorted(path for root in roots for path in root.glob("*/response.json"))
    rows: list[RunSummary] = []
    failures: list[tuple[Path, str]] = []
    for path in response_paths:
        try:
            rows.append(load_response(path))
        except Exception as exc:
            failures.append((path, f"{type(exc).__name__}: {exc}"))

    if not rows:
        print(f"No response.json files found under {args.server_runs}")
        return 1

    source_counts = Counter(row.source for row in rows)
    error_counts = Counter(row.error_type for row in rows)
    clipped_counts = Counter(row.clipped for row in rows)
    rgbd_rows = [row for row in rows if row.source == "rgbd"]
    recent_rgbd_rows = rgbd_rows[-max(1, int(args.recent_rgbd)) :]
    tail_rows = rows[-max(1, int(args.last)) :]

    print(f"responses: {len(rows)}")
    if failures:
        print(f"parse failures: {len(failures)}")
    print_counter("object_source_used:", source_counts)
    print_counter("object_error_type:", error_counts)
    print_counter("translation clipped:", clipped_counts)

    print("\nAll RGB-D object pose stability:")
    print_position_block("obj1:", describe_positions(rgbd_rows, "obj1"))
    print_position_block("obj2:", describe_positions(rgbd_rows, "obj2"))

    print(f"\nLatest {len(recent_rgbd_rows)} RGB-D object pose stability:")
    print_position_block("obj1:", describe_positions(recent_rgbd_rows, "obj1"))
    print_position_block("obj2:", describe_positions(recent_rgbd_rows, "obj2"))

    print("\nPolicy/control scalar stats:")
    for label, getter in [
        ("done_prob", lambda row: row.done_prob),
        ("latency_s", lambda row: row.latency_s),
        ("raw_delta_norm_m", lambda row: row.raw_delta_norm_m),
        ("clipped_delta_norm_m", lambda row: row.clipped_delta_norm_m),
        ("gripper_g1", lambda row: row.gripper_g1),
    ]:
        vals = [getter(row) for row in rows if getter(row) is not None]
        stats = describe_values(vals)
        print(
            f"  {label}: n={stats['n']} mean={format_optional(stats['mean'])} "
            f"std={format_optional(stats['std'])} min={format_optional(stats['min'])} "
            f"max={format_optional(stats['max'])}"
        )

    print(f"\nLatest {len(tail_rows)} responses:")
    print("  run source err done latency raw_step clipped obj1_cam obj2_cam")
    for row in tail_rows:
        obj1 = np.round(row.obj_pos_cam["obj1"], 4).tolist() if "obj1" in row.obj_pos_cam else "-"
        obj2 = np.round(row.obj_pos_cam["obj2"], 4).tolist() if "obj2" in row.obj_pos_cam else "-"
        print(
            "  "
            f"{row.run} {row.source or '-'} {row.error_type or '-'} "
            f"{format_optional(row.done_prob)} {format_optional(row.latency_s, 3)} "
            f"{format_optional(row.raw_delta_norm_m)} {row.clipped if row.clipped is not None else '-'} "
            f"{obj1} {obj2}"
        )

    if args.json_out is not None:
        payload = {
            "responses": len(rows),
            "parse_failures": [{"path": str(path), "error": error} for path, error in failures],
            "object_source_used": dict(source_counts),
            "object_error_type": dict(error_counts),
            "translation_clipped": {str(key): value for key, value in clipped_counts.items()},
            "all_rgbd": {
                "obj1": describe_positions(rgbd_rows, "obj1"),
                "obj2": describe_positions(rgbd_rows, "obj2"),
            },
            "latest_rgbd": {
                "obj1": describe_positions(recent_rgbd_rows, "obj1"),
                "obj2": describe_positions(recent_rgbd_rows, "obj2"),
            },
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
