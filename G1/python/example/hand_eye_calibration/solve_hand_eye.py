#!/usr/bin/env python3
"""Solve T_head_pitch_camera from collected Agibot G1 hand-eye samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from hand_eye_common import (
    DOC_EXAMPLE_T_HEAD_PITCH_CAMERA,
    as_matrix4,
    compute_t_base_head_pitch,
    load_json,
    load_jsonl,
    load_kinematics,
    load_transform_file,
    matrix_to_list,
    sample_t_camera_tag,
    write_calibration_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory containing samples.jsonl from collect_hand_eye_data.py.")
    parser.add_argument("--urdf-path", default=None, help="Path to A2D_viz.urdf. If omitted, tries GDK auto-discovery.")
    parser.add_argument("--initial-transform", default=None, help="Optional JSON/YAML file containing initial T_head_pitch_camera.")
    parser.add_argument("--identity-initial", action="store_true", help="Use identity as initial T_head_pitch_camera.")
    parser.add_argument("--rotation-weight-m", type=float, default=0.05, help="Scale rotation residuals into meter-like units.")
    parser.add_argument("--max-nfev", type=int, default=2000)
    parser.add_argument("--output-yaml", default=None)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def mat_to_rtvec(matrix: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    matrix = as_matrix4(matrix)
    out = np.zeros(6, dtype=np.float64)
    out[:3] = R.from_matrix(matrix[:3, :3]).as_rotvec()
    out[3:] = matrix[:3, 3]
    return out


def rtvec_to_mat(values: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    values = np.asarray(values, dtype=np.float64).reshape(6)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = R.from_rotvec(values[:3]).as_matrix()
    matrix[:3, 3] = values[3:]
    return matrix


def mean_pose(mats: list[np.ndarray]) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.mean([m[:3, 3] for m in mats], axis=0)
    rotations = R.from_matrix([m[:3, :3] for m in mats])
    pose[:3, :3] = rotations.mean().as_matrix()
    return pose


def build_stats(predicted_base_tag: list[np.ndarray], t_base_tag: np.ndarray) -> dict[str, float]:
    from scipy.spatial.transform import Rotation as R

    pos_errs = []
    rot_errs_deg = []
    target_rot = R.from_matrix(t_base_tag[:3, :3])
    for pred in predicted_base_tag:
        pos_errs.append(float(np.linalg.norm(pred[:3, 3] - t_base_tag[:3, 3])))
        rot_err = (target_rot.inv() * R.from_matrix(pred[:3, :3])).magnitude()
        rot_errs_deg.append(float(np.rad2deg(rot_err)))
    pos = np.asarray(pos_errs, dtype=np.float64)
    rot = np.asarray(rot_errs_deg, dtype=np.float64)
    return {
        "sample_count": int(len(predicted_base_tag)),
        "position_rmse_m": float(np.sqrt(np.mean(pos * pos))),
        "position_mean_m": float(np.mean(pos)),
        "position_max_m": float(np.max(pos)),
        "rotation_mean_deg": float(np.mean(rot)),
        "rotation_max_deg": float(np.max(rot)),
    }


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    samples_path = data_dir / "samples.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError(samples_path)

    samples = [s for s in load_jsonl(samples_path) if s.get("tag_detected", True) and s.get("tag")]
    if len(samples) < 6:
        raise RuntimeError(f"Need at least 6 detected-tag samples, got {len(samples)}")

    kinematics, resolved_urdf = load_kinematics(args.urdf_path)

    t_base_head = []
    t_camera_tag = []
    for sample in samples:
        t_base_head.append(
            compute_t_base_head_pitch(
                kinematics,
                sample["head_joint_states"],
                sample["waist_joint_states"],
            )
        )
        t_camera_tag.append(sample_t_camera_tag(sample))

    if args.identity_initial:
        x0_mat = np.eye(4, dtype=np.float64)
        print("Initial T_head_pitch_camera: identity")
    elif args.initial_transform:
        x0_mat = load_transform_file(args.initial_transform)
        print(f"Initial T_head_pitch_camera: {args.initial_transform}")
    else:
        x0_mat = DOC_EXAMPLE_T_HEAD_PITCH_CAMERA.copy()
        print("Initial T_head_pitch_camera: doc example from roboclaw_camera_to_base_transform_notes.md")

    initial_base_tag = mean_pose([a @ x0_mat @ c for a, c in zip(t_base_head, t_camera_tag)])
    p0 = np.concatenate([mat_to_rtvec(x0_mat), mat_to_rtvec(initial_base_tag)])

    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as R

    def residual(params: np.ndarray) -> np.ndarray:
        x = rtvec_to_mat(params[:6])
        b = rtvec_to_mat(params[6:12])
        b_rot = R.from_matrix(b[:3, :3])
        errors = []
        for a, c in zip(t_base_head, t_camera_tag):
            pred = a @ x @ c
            errors.extend((pred[:3, 3] - b[:3, 3]).tolist())
            rot_err = (b_rot.inv() * R.from_matrix(pred[:3, :3])).as_rotvec()
            errors.extend((args.rotation_weight_m * rot_err).tolist())
        return np.asarray(errors, dtype=np.float64)

    result = least_squares(residual, p0, method="trf", max_nfev=args.max_nfev)
    t_head_pitch_camera = rtvec_to_mat(result.x[:6])
    t_base_tag = rtvec_to_mat(result.x[6:12])
    predicted = [a @ t_head_pitch_camera @ c for a, c in zip(t_base_head, t_camera_tag)]
    stats = build_stats(predicted, t_base_tag)
    stats["optimizer_cost"] = float(result.cost)
    stats["optimizer_success"] = bool(result.success)

    metadata_path = data_dir / "metadata.json"
    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    camera_params = metadata.get("camera_params") or samples[0].get("camera_params")
    tag_size_m = metadata.get("tag_size_m") or (samples[0].get("tag") or {}).get("tag_size_m")

    output_yaml = Path(args.output_yaml) if args.output_yaml else data_dir / "t_head_pitch_camera.yaml"
    output_json = Path(args.output_json) if args.output_json else data_dir / "t_head_pitch_camera.json"
    write_calibration_outputs(
        output_yaml=output_yaml,
        output_json=output_json,
        t_head_pitch_camera=t_head_pitch_camera,
        camera_params=camera_params,
        tag_size_m=tag_size_m,
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

