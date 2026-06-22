#!/usr/bin/env python3
"""Detect one checkerboard image and optionally estimate its camera-frame pose."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from checkerboard_common import (
    detect_checkerboard,
    draw_checkerboard_detection,
    intrinsics_from_json,
    solve_checkerboard_pose,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--pattern-cols", type=int, default=9, help="Inner corners per row. Your calib.io board says 9.")
    parser.add_argument("--pattern-rows", type=int, default=12, help="Inner corners per column. Your calib.io board says 12.")
    parser.add_argument("--square-size-m", type=float, default=0.02, help="Checker size in meters. Your board says 20 mm.")
    parser.add_argument("--intrinsics-json", default=None, help="Optional intrinsics JSON; enables pose output.")
    parser.add_argument("--classic-detector", action="store_true", help="Use cv2.findChessboardCorners instead of SB.")
    parser.add_argument("--annotated-output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(args.image)

    if args.intrinsics_json:
        camera_matrix, dist_coeffs, _ = intrinsics_from_json(args.intrinsics_json)
        detection = solve_checkerboard_pose(
            image_bgr,
            pattern_cols=args.pattern_cols,
            pattern_rows=args.pattern_rows,
            square_size_m=args.square_size_m,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            use_sb=not args.classic_detector,
        )
    else:
        detection = detect_checkerboard(
            image_bgr,
            pattern_cols=args.pattern_cols,
            pattern_rows=args.pattern_rows,
            use_sb=not args.classic_detector,
        )

    if detection is None:
        print("No checkerboard detected")
        print("If the board is rotated 90 degrees, retry with --pattern-cols 12 --pattern-rows 9.")
        return 1

    print(f"detected: {args.pattern_cols} x {args.pattern_rows} inner corners")
    print(f"method: {detection['method']}")
    print(f"center_px: {detection['center_px']}")
    if "position_camera_m" in detection:
        print(f"position_camera_m: {detection['position_camera_m']}")
        print("T_camera_board:")
        for row in detection["T_camera_board"]:
            print(f"  {row}")

    if args.annotated_output:
        out_path = Path(args.annotated_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        annotated = draw_checkerboard_detection(
            image_bgr,
            detection,
            pattern_cols=args.pattern_cols,
            pattern_rows=args.pattern_rows,
        )
        cv2.imwrite(str(out_path), annotated)
        print(f"annotated_output: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
