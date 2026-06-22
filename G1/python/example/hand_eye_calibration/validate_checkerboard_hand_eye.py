#!/usr/bin/env python3
"""Validate a solved Agibot G1 checkerboard hand-eye calibration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from checkerboard_common import sample_t_camera_board
from hand_eye_common import (
    compute_t_base_head_pitch,
    load_jsonl,
    load_kinematics,
    load_transform_file,
    transform_point,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--urdf-path", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    samples = [s for s in load_jsonl(data_dir / "samples.jsonl") if s.get("board_detected", True) and s.get("board")]
    if not samples:
        raise RuntimeError("No detected-checkerboard samples found")

    t_head_pitch_camera = load_transform_file(args.calibration)
    kinematics, resolved_urdf = load_kinematics(args.urdf_path)

    positions = []
    for sample in samples:
        t_base_head = compute_t_base_head_pitch(
            kinematics,
            sample["head_joint_states"],
            sample["waist_joint_states"],
        )
        t_camera_board = sample_t_camera_board(sample)
        t_base_camera = t_base_head @ t_head_pitch_camera
        positions.append(transform_point(t_base_camera, t_camera_board[:3, 3]))

    positions_np = np.asarray(positions, dtype=np.float64)
    mean_pos = np.mean(positions_np, axis=0)
    errs = np.linalg.norm(positions_np - mean_pos.reshape(1, 3), axis=1)

    print(f"URDF: {resolved_urdf}")
    print(f"Samples: {len(samples)}")
    print(f"Mean checkerboard origin in base_link: {mean_pos.tolist()}")
    print(f"Position error mean: {float(np.mean(errs)):.6f} m")
    print(f"Position error rmse: {float(np.sqrt(np.mean(errs * errs))):.6f} m")
    print(f"Position error max:  {float(np.max(errs)):.6f} m")
    print("\nPer-sample errors:")
    for sample, err, pos in zip(samples, errs, positions_np):
        print(
            f"  sample {int(sample['sample_id']):04d}: "
            f"err={float(err):.6f} m pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
