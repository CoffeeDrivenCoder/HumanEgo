#!/usr/bin/env python3
"""Solve T_head_pitch_camera from collected checkerboard hand-eye samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from checkerboard_common import sample_t_camera_board
from hand_eye_common import (
    DOC_EXAMPLE_T_HEAD_PITCH_CAMERA,
    compute_t_base_head_pitch,
    load_json,
    load_jsonl,
    load_kinematics,
    load_transform_file,
    matrix_to_list,
    write_calibration_outputs,
)
from solve_hand_eye import build_stats, mat_to_rtvec, mean_pose, rtvec_to_mat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--urdf-path", default=None)
    parser.add_argument("--initial-transform", default=None)
    parser.add_argument("--identity-initial", action="store_true")
    parser.add_argument("--rotation-weight-m", type=float, default=0.05)
    parser.add_argument("--max-nfev", type=int, default=2000)
    parser.add_argument("--output-yaml", default=None)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    samples = [s for s in load_jsonl(data_dir / "samples.jsonl") if s.get("board_detected", True) and s.get("board")]
    if len(samples) < 6:
        raise RuntimeError(f"Need at least 6 detected checkerboard samples, got {len(samples)}")

    kinematics, resolved_urdf = load_kinematics(args.urdf_path)

    t_base_head = []
    t_camera_board = []
    for sample in samples:
        t_base_head.append(
            compute_t_base_head_pitch(
                kinematics,
                sample["head_joint_states"],
                sample["waist_joint_states"],
            )
        )
        t_camera_board.append(sample_t_camera_board(sample))

    if args.identity_initial:
        x0_mat = np.eye(4, dtype=np.float64)
        print("Initial T_head_pitch_camera: identity")
    elif args.initial_transform:
        x0_mat = load_transform_file(args.initial_transform)
        print(f"Initial T_head_pitch_camera: {args.initial_transform}")
    else:
        x0_mat = DOC_EXAMPLE_T_HEAD_PITCH_CAMERA.copy()
        print("Initial T_head_pitch_camera: doc example from roboclaw_camera_to_base_transform_notes.md")

    initial_base_board = mean_pose([a @ x0_mat @ c for a, c in zip(t_base_head, t_camera_board)])
    p0 = np.concatenate([mat_to_rtvec(x0_mat), mat_to_rtvec(initial_base_board)])

    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as R

    def residual(params: np.ndarray) -> np.ndarray:
        x = rtvec_to_mat(params[:6])
        b = rtvec_to_mat(params[6:12])
        b_rot = R.from_matrix(b[:3, :3])
        errors = []
        for a, c in zip(t_base_head, t_camera_board):
            pred = a @ x @ c
            errors.extend((pred[:3, 3] - b[:3, 3]).tolist())
            rot_err = (b_rot.inv() * R.from_matrix(pred[:3, :3])).as_rotvec()
            errors.extend((args.rotation_weight_m * rot_err).tolist())
        return np.asarray(errors, dtype=np.float64)

    result = least_squares(residual, p0, method="trf", max_nfev=args.max_nfev)
    t_head_pitch_camera = rtvec_to_mat(result.x[:6])
    t_base_board = rtvec_to_mat(result.x[6:12])
    predicted = [a @ t_head_pitch_camera @ c for a, c in zip(t_base_head, t_camera_board)]
    stats = build_stats(predicted, t_base_board)
    stats["optimizer_cost"] = float(result.cost)
    stats["optimizer_success"] = bool(result.success)

    metadata_path = data_dir / "metadata.json"
    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    camera_params = metadata.get("camera_params") or samples[0].get("camera_params")

    output_yaml = Path(args.output_yaml) if args.output_yaml else data_dir / "t_head_pitch_camera.yaml"
    output_json = Path(args.output_json) if args.output_json else data_dir / "t_head_pitch_camera.json"
    write_calibration_outputs(
        output_yaml=output_yaml,
        output_json=output_json,
        t_head_pitch_camera=t_head_pitch_camera,
        camera_params=camera_params,
        tag_size_m=None,
        stats=stats,
        urdf_path=resolved_urdf,
    )

    print("\nSolved T_head_pitch_camera:")
    for row in matrix_to_list(t_head_pitch_camera):
        print(f"  {row}")
    print("\nValidation on calibration samples:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"\nWrote YAML: {output_yaml}")
    print(f"Wrote JSON: {output_json}")
    return 0 if result.success else 2


if __name__ == "__main__":
    sys.exit(main())
