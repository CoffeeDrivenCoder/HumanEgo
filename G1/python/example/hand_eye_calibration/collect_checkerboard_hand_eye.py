#!/usr/bin/env python3
"""Collect Agibot G1 checkerboard hand-eye calibration samples."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from checkerboard_common import (
    draw_checkerboard_detection,
    intrinsics_from_json,
    solve_checkerboard_pose,
)
from hand_eye_common import append_jsonl, ensure_dir, now_ns, poll_numeric_state, save_json, save_rgb_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--intrinsics-json", required=True)
    parser.add_argument("--camera-name", default="head")
    parser.add_argument("--pattern-cols", type=int, default=9)
    parser.add_argument("--pattern-rows", type=int, default=12)
    parser.add_argument("--square-size-m", type=float, default=0.02)
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument("--auto-interval-s", type=float, default=0.0)
    parser.add_argument("--classic-detector", action="store_true")
    parser.add_argument("--save-undetected", action="store_true")
    parser.add_argument("--save-annotated", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    image_dir = ensure_dir(out_dir / "images")
    annotated_dir = ensure_dir(out_dir / "annotated") if args.save_annotated else None
    samples_path = out_dir / "samples.jsonl"
    camera_matrix, dist_coeffs, camera_params = intrinsics_from_json(args.intrinsics_json)

    metadata = {
        "calibration_type": "checkerboard_hand_eye",
        "camera_name": args.camera_name,
        "camera_params": camera_params,
        "intrinsics_json": args.intrinsics_json,
        "pattern_cols": args.pattern_cols,
        "pattern_rows": args.pattern_rows,
        "square_size_m": args.square_size_m,
        "created_time_ns": now_ns(),
    }
    save_json(out_dir / "metadata.json", metadata)

    print(f"Output directory: {out_dir}")
    print("Keep the checkerboard fixed on the table for the whole hand-eye run.")
    print("Move head/waist to varied viewpoints, then press Enter to capture each sample.")

    from a2d_sdk.robot import CosineCamera, RobotDds

    robot = RobotDds()
    camera_group = CosineCamera([args.camera_name])
    time.sleep(args.warmup_s)

    collected = 0
    attempts = 0
    try:
        while collected < args.num_samples:
            if args.auto_interval_s > 0.0:
                time.sleep(args.auto_interval_s)
            else:
                user_input = input(f"[{collected + 1}/{args.num_samples}] Press Enter to capture, q to stop: ")
                if user_input.strip().lower() in {"q", "quit", "exit"}:
                    break

            attempts += 1
            head = poll_numeric_state(robot.head_joint_states, 2, "head_joint_states")
            waist = poll_numeric_state(robot.waist_joint_states, 2, "waist_joint_states")
            arm = poll_numeric_state(robot.arm_joint_states, 14, "arm_joint_states")
            image, image_ts = camera_group.get_latest_image(args.camera_name)
            if image is None:
                print("No camera image received; retry this sample.")
                continue

            bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            board = solve_checkerboard_pose(
                bgr,
                pattern_cols=args.pattern_cols,
                pattern_rows=args.pattern_rows,
                square_size_m=args.square_size_m,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                use_sb=not args.classic_detector,
            )
            detected = board is not None
            if not detected and not args.save_undetected:
                print("No checkerboard detected; adjust viewpoint/lighting and retry.")
                continue

            sample_id = collected + 1
            image_rel = Path("images") / f"{sample_id:04d}_{args.camera_name}.jpg"
            save_rgb_image(out_dir / image_rel, image)

            if args.save_annotated:
                annotated_path = annotated_dir / f"{sample_id:04d}_{args.camera_name}.jpg"
                annotated = draw_checkerboard_detection(
                    bgr,
                    board,
                    pattern_cols=args.pattern_cols,
                    pattern_rows=args.pattern_rows,
                )
                cv2.imwrite(str(annotated_path), annotated)

            sample = {
                "sample_id": sample_id,
                "attempt": attempts,
                "timestamp_ns": int(image_ts) if image_ts is not None else now_ns(),
                "image_path": str(image_rel),
                "camera_name": args.camera_name,
                "camera_params": camera_params,
                "board_detected": detected,
                "board": board,
                "head_joint_states": head,
                "waist_joint_states": waist,
                "arm_joint_states": arm,
            }
            append_jsonl(samples_path, sample)
            save_json(out_dir / f"sample_{sample_id:04d}.json", sample)

            collected += 1
            if detected:
                p = board["position_camera_m"]
                print(f"Saved sample {sample_id:04d}: p_camera=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) m")
            else:
                print(f"Saved sample {sample_id:04d}: no checkerboard detected")
    finally:
        try:
            robot.shutdown()
        except Exception:
            pass

    print(f"Collected {collected} samples in {out_dir}")
    return 0 if collected > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
