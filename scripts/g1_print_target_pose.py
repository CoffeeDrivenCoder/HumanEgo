#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print HumanEgo hand target -> G1 TCP/link7 targets without sending control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = PROJECT_ROOT / "cfg" / "inference" / "g1_serve_bread_right.yaml"


def load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def mat(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"expected 4x4, got {arr.shape}")
    return arr


def clean(T: np.ndarray) -> list[list[float]]:
    return [[round(float(v), 9) for v in row] for row in T.tolist()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default=str(DEFAULT_CFG))
    parser.add_argument("--side", default="right", choices=["right", "left"])
    parser.add_argument("--hand-pos", nargs=3, type=float, default=[0.05, -0.01, 0.62])
    parser.add_argument(
        "--hand-rot",
        nargs=9,
        type=float,
        default=[1, 0, 0, 0, 1, 0, 0, 0, 1],
        help="Row-major 3x3 T_hand_in_cam rotation.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT / "inference"))
    from G1Geometry import fixed_T_tcp_in_link7

    cfg = load_cfg(Path(args.cfg).expanduser().resolve())
    T_hand_in_tcp = mat(cfg["robot"]["T_align"])
    T_tcp_in_link7 = fixed_T_tcp_in_link7(args.side)

    T_hand_in_cam = np.eye(4, dtype=np.float64)
    T_hand_in_cam[:3, :3] = np.asarray(args.hand_rot, dtype=np.float64).reshape(3, 3)
    T_hand_in_cam[:3, 3] = np.asarray(args.hand_pos, dtype=np.float64)

    T_tcp_target_in_cam = T_hand_in_cam @ np.linalg.inv(T_hand_in_tcp)
    T_link7_target_in_cam = T_tcp_target_in_cam @ np.linalg.inv(T_tcp_in_link7)

    report = {
        "side": args.side,
        "note": "No FK and no control. These are camera-frame targets only.",
        "T_hand_target_in_cam": clean(T_hand_in_cam),
        "T_tcp_target_in_cam": clean(T_tcp_target_in_cam),
        "T_link7_target_in_cam": clean(T_link7_target_in_cam),
        "next_after_fk": "Convert camera-frame target to base-frame target with T_base_camera, then verify set_end_effector_pose_control frame semantics.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
