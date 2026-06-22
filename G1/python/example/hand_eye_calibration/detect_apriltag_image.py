#!/usr/bin/env python3
"""Detect one AprilTag image and print its camera-frame pose."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from hand_eye_common import camera_params_from_args, detect_apriltag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--camera-name", default="head")
    parser.add_argument("--camera-model", default="")
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--intrinsics-json", default=None)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--tag-size-m", type=float, required=True)
    parser.add_argument("--tag-family", default="tag25h9")
    parser.add_argument("--tag-id", type=int, default=None)
    parser.add_argument("--annotated-output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(args.image)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if args.image_width is None:
        args.image_width = int(rgb.shape[1])
    if args.image_height is None:
        args.image_height = int(rgb.shape[0])

    camera_params = camera_params_from_args(args)
    tag = detect_apriltag(
        rgb,
        camera_params=camera_params,
        tag_size_m=args.tag_size_m,
        tag_family=args.tag_family,
        tag_id=args.tag_id,
    )
    if tag is None:
        print("No AprilTag detected")
        return 1

    print(f"tag_id: {tag['tag_id']}")
    print(f"position_camera_m: {tag['position_camera_m']}")
    print("T_camera_tag:")
    for row in tag["T_camera_tag"]:
        print(f"  {row}")

    if args.annotated_output:
        import numpy as np

        corners = tag["corners_px"]
        poly = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
        annotated = bgr.copy()
        cv2.polylines(annotated, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
        center = tuple(int(v) for v in tag["center_px"])
        cv2.circle(annotated, center, 4, (0, 0, 255), -1)
        out_path = Path(args.annotated_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated)
        print(f"annotated_output: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
