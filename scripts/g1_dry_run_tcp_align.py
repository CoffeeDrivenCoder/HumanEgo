#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dry-run G1 TCP and HumanEgo T_align without FK or robot control."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = PROJECT_ROOT / "cfg" / "inference" / "g1_serve_bread_right.yaml"


def resolve_project_path(path: str | os.PathLike[str]) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    for base in (Path.cwd(), PROJECT_ROOT):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (PROJECT_ROOT / path).resolve()


def load_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def matrix_to_jsonable(T: np.ndarray) -> list[list[float]]:
    return [[round(float(v), 9) for v in row] for row in np.asarray(T).reshape(4, 4)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default=str(DEFAULT_CFG))
    parser.add_argument("--side", default="right", choices=["right", "left"])
    parser.add_argument("--sample-hand-pos", nargs=3, type=float, default=[0.05, -0.01, 0.62])
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT / "inference"))
    sys.path.insert(0, str(PROJECT_ROOT))

    from G1Geometry import fixed_T_hand_in_tcp, fixed_T_tcp_in_link7, geometry_summary

    cfg_path = resolve_project_path(args.cfg)
    cfg = load_cfg(cfg_path)
    ckpt_path = resolve_project_path(cfg["policy"]["ckpt"])
    ckpt_dir = ckpt_path.parent
    train_cfg = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
    stats = json.loads((ckpt_dir / "dataset_stats.json").read_text(encoding="utf-8"))

    T_tcp_in_link7 = fixed_T_tcp_in_link7(args.side)
    T_hand_in_tcp_from_code = fixed_T_hand_in_tcp(args.side)
    T_hand_in_tcp_from_cfg = np.asarray(cfg["robot"]["T_align"], dtype=np.float64).reshape(4, 4)

    T_sample_hand_in_cam = np.eye(4, dtype=np.float64)
    T_sample_hand_in_cam[:3, 3] = np.asarray(args.sample_hand_pos, dtype=np.float64)
    T_sample_tcp_target_in_cam = T_sample_hand_in_cam @ np.linalg.inv(T_hand_in_tcp_from_cfg)
    T_sample_link7_target_in_cam = T_sample_tcp_target_in_cam @ np.linalg.inv(T_tcp_in_link7)

    report: Dict[str, Any] = {
        "cfg_path": str(cfg_path),
        "checkpoint": {
            "ckpt": str(ckpt_path),
            "config_json": str(ckpt_dir / "config.json"),
            "dataset_stats_json": str(ckpt_dir / "dataset_stats.json"),
            "task": train_cfg.get("task"),
            "data_sources": train_cfg.get("data_sources"),
            "train_sessions": len(train_cfg.get("MPS_PATHS_TRAIN") or []),
            "eval_sessions": len(train_cfg.get("MPS_PATHS_EVAL") or []),
            "single_hand": train_cfg.get("single_hand"),
            "single_hand_side": train_cfg.get("single_hand_side"),
            "frame_mode": train_cfg.get("frame_mode"),
            "action_mode": train_cfg.get("action_mode"),
            "use_region_attn": train_cfg.get("use_region_attn"),
            "img_name": train_cfg.get("img_name"),
            "pos_stats": stats.get("pos"),
        },
        "g1_geometry": geometry_summary(args.side),
        "cfg_T_align_matches_code": bool(np.allclose(T_hand_in_tcp_from_cfg, T_hand_in_tcp_from_code)),
        "T_hand_in_tcp_from_cfg": matrix_to_jsonable(T_hand_in_tcp_from_cfg),
        "dry_run_sample": {
            "meaning": "sample hand target in camera -> TCP target in camera -> link7 target in camera",
            "T_sample_hand_in_cam": matrix_to_jsonable(T_sample_hand_in_cam),
            "T_sample_tcp_target_in_cam": matrix_to_jsonable(T_sample_tcp_target_in_cam),
            "T_sample_link7_target_in_cam": matrix_to_jsonable(T_sample_link7_target_in_cam),
        },
        "blocked_until_fk": [
            "Cannot convert camera-frame TCP target to base-frame control target until T_base_camera is validated.",
            "Cannot do real control until set_end_effector_pose_control frame semantics are verified.",
        ],
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
