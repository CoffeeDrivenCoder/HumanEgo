#!/usr/bin/env python3
"""Collect Agibot G1 head-camera hand-eye calibration samples."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from hand_eye_common import (
    append_jsonl,
    camera_params_from_args,
    detect_apriltag,
    ensure_dir,
    now_ns,
    poll_numeric_state,
    save_json,
    save_rgb_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Directory for samples.jsonl and captured images.")
    parser.add_argument("--camera-name", default="head", help="GDK camera name. Default: head.")
    parser.add_argument("--camera-model", default="Realsense-D455", help="Camera model label written to metadata.")
    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-height", type=int, default=720)
    parser.add_argument("--intrinsics-json", default=None, help="Optional JSON file containing fx/fy/cx/cy.")
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--tag-size-m", type=float, required=True, help="Printed AprilTag edge size in meters.")
    parser.add_argument("--tag-family", default="tag25h9")
    parser.add_argument("--tag-id", type=int, default=None, help="Optional expected tag id.")
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument("--auto-interval-s", type=float, default=0.0, help="If >0, collect automatically at this interval.")
    parser.add_argument("--save-undetected", action="store_true", help="Save images even when no tag is detected.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    image_dir = ensure_dir(out_dir / "images")
    samples_path = out_dir / "samples.jsonl"
    camera_params = camera_params_from_args(args)

    metadata = {
        "camera_name": args.camera_name,
        "camera_params": camera_params,
        "tag_size_m": args.tag_size_m,
        "tag_family": args.tag_family,
        "tag_id": args.tag_id,
        "created_time_ns": now_ns(),
    }
    save_json(out_dir / "metadata.json", metadata)

    print(f"Output directory: {out_dir}")
    print(f"Samples file: {samples_path}")
    print("Move the head/waist so the fixed AprilTag is visible from varied viewpoints.")
    print("Press Enter to capture each sample, or type q then Enter to stop.")

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

            tag = detect_apriltag(
                image,
                camera_params=camera_params,
                tag_size_m=args.tag_size_m,
                tag_family=args.tag_family,
                tag_id=args.tag_id,
            )
            detected = tag is not None
            if not detected and not args.save_undetected:
                print("No AprilTag detected; adjust viewpoint/lighting and retry.")
                continue

            sample_id = collected + 1
            image_rel = Path("images") / f"{sample_id:04d}_{args.camera_name}.jpg"
            save_rgb_image(out_dir / image_rel, image)

            sample = {
                "sample_id": sample_id,
                "attempt": attempts,
                "timestamp_ns": int(image_ts) if image_ts is not None else now_ns(),
                "image_path": str(image_rel),
                "camera_name": args.camera_name,
                "camera_params": camera_params,
                "tag_detected": detected,
                "tag": tag,
                "head_joint_states": head,
                "waist_joint_states": waist,
                "arm_joint_states": arm,
            }
            append_jsonl(samples_path, sample)
            save_json(out_dir / f"sample_{sample_id:04d}.json", sample)

            collected += 1
            if detected:
                p = tag["position_camera_m"]
                print(
                    f"Saved sample {sample_id:04d}: tag_id={tag['tag_id']} "
                    f"p_camera=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) m"
                )
            else:
                print(f"Saved sample {sample_id:04d}: no tag detected")

    finally:
        try:
            robot.shutdown()
        except Exception:
            pass

    print(f"Collected {collected} samples in {out_dir}")
    return 0 if collected > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

