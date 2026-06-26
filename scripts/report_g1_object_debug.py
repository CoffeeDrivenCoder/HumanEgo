#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Report RGB-D object pose debug files for one HumanEgo server run."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

from g1_artifacts import artifact_dir


def latest_run(server_runs: Path) -> Path:
    candidates = [p.parent for p in server_runs.expanduser().resolve().glob("*/response.json")]
    if not candidates:
        raise RuntimeError(f"No server responses under {server_runs}")
    return max(candidates, key=lambda p: (p / "response.json").stat().st_mtime)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path)
    parser.add_argument("--session", default=None)
    parser.add_argument("--server-runs", type=Path, default=artifact_dir("server"))
    parser.add_argument("--latest", action="store_true")
    args = parser.parse_args()

    if args.session is not None:
        args.server_runs = artifact_dir("server").parents[1] / args.session / "server"
    if args.latest:
        run_dir = latest_run(args.server_runs)
    elif args.run is not None:
        run_dir = args.run.expanduser().resolve()
    else:
        parser.error("provide a run path or use --latest")

    response = load_json(run_dir / "response.json")
    object_debug = ((response.get("input_summary") or {}).get("object_debug") or {})
    print(f"run: {run_dir}")
    print(f"object_source_used: {(response.get('input_summary') or {}).get('object_source_used')}")
    if not object_debug:
        print("object_debug: missing. Restart server with the updated code and run a new dry-run.")
        return 1

    for key, item in (object_debug.get("objects") or {}).items():
        points = item.get("points") or {}
        segmentation = item.get("segmentation") or {}
        print(f"\n{key}: prompt={item.get('prompt')!r}")
        print(f"  T xyz: {[round(row[3], 4) for row in item.get('T_in_cam', [[0,0,0,0]] * 3)[:3]]}")
        print(f"  kpts_local_count: {item.get('kpts_local_count')}")
        if segmentation:
            print(f"  segmentation mode: {segmentation.get('mode')}")
            if segmentation.get("mode") == "dinosam_candidates":
                print(f"  candidates: {segmentation.get('candidate_count')}")
                print(f"  selected_idx: {segmentation.get('selected_idx')}")
                print(f"  selected_score: {segmentation.get('selected_score')}")
                selected_stats = segmentation.get("selected_stats") or {}
                if selected_stats:
                    print(
                        "  selected 2d: "
                        f"area={selected_stats.get('area_px')} "
                        f"bbox={selected_stats.get('bbox_xyxy')} "
                        f"bbox_wh={selected_stats.get('bbox_wh')} "
                        f"aspect={selected_stats.get('bbox_aspect')} "
                        f"fill={selected_stats.get('fill_ratio')} "
                        f"circ={selected_stats.get('circularity')}"
                    )
                for cand in (segmentation.get("candidates") or [])[:8]:
                    status = "ok" if cand.get("accepted") else "reject"
                    print(
                        "    cand "
                        f"{cand.get('idx')}: {status} "
                        f"area={cand.get('area_px')} "
                        f"bbox={cand.get('bbox_xyxy')} "
                        f"score={cand.get('score')} "
                        f"reasons={cand.get('reject_reasons')}"
                    )
            elif segmentation.get("mode") == "plate_circle_fallback":
                selected = segmentation.get("selected_circle") or {}
                print(
                    "  selected circle: "
                    f"center={selected.get('center_xy')} "
                    f"radius={selected.get('radius_px')} "
                    f"score={selected.get('score')} "
                    f"edge={selected.get('edge_support')} "
                    f"overlap={selected.get('disk_overlap')}"
                )
                for cand in (segmentation.get("candidates") or [])[:8]:
                    print(
                        "    circle "
                        f"{cand.get('idx')}: "
                        f"center={cand.get('center_xy')} "
                        f"r={cand.get('radius_px')} "
                        f"score={cand.get('score')} "
                        f"edge={cand.get('edge_support')} "
                        f"overlap={cand.get('disk_overlap')}"
                    )
            else:
                print(f"  segmentation raw_mask_pixels: {segmentation.get('raw_mask_pixels')}")
        print(f"  raw_mask_pixels: {points.get('raw_mask_pixels')}")
        print(f"  used_mask_pixels: {points.get('used_mask_pixels')}")
        print(f"  valid_depth_pixels: {points.get('valid_depth_pixels')}")
        print(f"  valid_depth_ratio_raw_mask: {points.get('valid_depth_ratio_raw_mask')}")
        print(f"  valid_uv_bbox: {points.get('valid_uv_bbox')}")
        print(f"  depth median/min/max: {points.get('valid_depth_median_m')} / {points.get('valid_depth_min_m')} / {points.get('valid_depth_max_m')}")
        print(f"  points_center_m: {points.get('points_center_m')}")
        print(f"  points_extent_m: {points.get('points_extent_m')}")
        saved = item.get("saved_files") or {}
        for label in ("mask_overlay", "valid_depth_overlay", "circle_candidates", "mask", "point_debug"):
            if label in saved:
                print(f"  {label}: {saved[label]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
