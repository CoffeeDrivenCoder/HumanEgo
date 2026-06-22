#!/usr/bin/env python3
"""Collect Agibot G1 head-camera images for checkerboard intrinsic calibration."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from checkerboard_common import detect_checkerboard, draw_checkerboard_detection
from hand_eye_common import append_jsonl, ensure_dir, now_ns, save_json, save_rgb_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--camera-name", default="head")
    parser.add_argument("--camera-model", default="Realsense-D455")
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
    samples_path = out_dir / "intrinsics_samples.jsonl"

    metadata = {
        "calibration_type": "checkerboard_intrinsics",
        "camera_name": args.camera_name,
        "camera_model": args.camera_model,
        "pattern_cols": args.pattern_cols,
        "pattern_rows": args.pattern_rows,
        "square_size_m": args.square_size_m,
        "created_time_ns": now_ns(),
    }
    save_json(out_dir / "metadata.json", metadata)

    print(f"Output directory: {out_dir}")
    print("Keep the checkerboard flat. Move the head/waist or board to cover center, corners, near, and far views.")
    print("Press Enter to capture each accepted image, or type q then Enter to stop.")

    from a2d_sdk.robot import CosineCamera

    camera_group = CosineCamera([args.camera_name])
    time.sleep(args.warmup_s)

    collected = 0
    attempts = 0
    while collected < args.num_samples:
        if args.auto_interval_s > 0.0:
            time.sleep(args.auto_interval_s)
        else:
            user_input = input(f"[{collected + 1}/{args.num_samples}] Press Enter to capture, q to stop: ")
            if user_input.strip().lower() in {"q", "quit", "exit"}:
                break

        attempts += 1
        image, image_ts = camera_group.get_latest_image(args.camera_name)
        if image is None:
            print("No camera image received; retry this sample.")
            continue

        detection = detect_checkerboard(
            image,
            pattern_cols=args.pattern_cols,
            pattern_rows=args.pattern_rows,
            use_sb=not args.classic_detector,
        )
        detected = detection is not None
        if not detected and not args.save_undetected:
            print("No checkerboard detected; flatten the paper, improve lighting, or change viewpoint.")
            continue

        sample_id = collected + 1
        image_rel = Path("images") / f"{sample_id:04d}_{args.camera_name}.jpg"
        save_rgb_image(out_dir / image_rel, image)

        if args.save_annotated:
            annotated_path = annotated_dir / f"{sample_id:04d}_{args.camera_name}.jpg"
            bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            annotated = draw_checkerboard_detection(
                bgr,
                detection,
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
            "checkerboard_detected": detected,
            "checkerboard": detection,
        }
        append_jsonl(samples_path, sample)
        save_json(out_dir / f"intrinsics_sample_{sample_id:04d}.json", sample)

        collected += 1
        if detected:
            center = detection["center_px"]
            print(f"Saved sample {sample_id:04d}: center_px=({center[0]:.1f}, {center[1]:.1f})")
        else:
            print(f"Saved sample {sample_id:04d}: no checkerboard detected")

    print(f"Collected {collected} images in {out_dir}")
    return 0 if collected > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
