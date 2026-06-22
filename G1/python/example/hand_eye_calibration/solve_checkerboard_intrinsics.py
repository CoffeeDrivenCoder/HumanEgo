#!/usr/bin/env python3
"""Solve camera intrinsics from collected checkerboard images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from checkerboard_common import checkerboard_object_points, detect_checkerboard, draw_checkerboard_detection
from hand_eye_common import ensure_dir, load_json, load_jsonl, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--pattern-cols", type=int, default=None)
    parser.add_argument("--pattern-rows", type=int, default=None)
    parser.add_argument("--square-size-m", type=float, default=None)
    parser.add_argument("--camera-name", default=None)
    parser.add_argument("--camera-model", default=None)
    parser.add_argument("--classic-detector", action="store_true")
    parser.add_argument("--fix-aspect-ratio", action="store_true")
    parser.add_argument("--save-annotated", action="store_true")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def reprojection_errors(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs) -> list[float]:
    errors = []
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        projected = projected.reshape(-1, 2)
        observed = imgp.reshape(-1, 2)
        err = np.sqrt(np.mean(np.sum((observed - projected) ** 2, axis=1)))
        errors.append(float(err))
    return errors


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    metadata_path = data_dir / "metadata.json"
    metadata = load_json(metadata_path) if metadata_path.exists() else {}

    pattern_cols = int(args.pattern_cols or metadata.get("pattern_cols", 9))
    pattern_rows = int(args.pattern_rows or metadata.get("pattern_rows", 12))
    square_size_m = float(args.square_size_m or metadata.get("square_size_m", 0.02))
    camera_name = args.camera_name or metadata.get("camera_name", "head")
    camera_model = args.camera_model or metadata.get("camera_model", "")

    samples_path = data_dir / "intrinsics_samples.jsonl"
    if samples_path.exists():
        samples = load_jsonl(samples_path)
        image_paths = [data_dir / s["image_path"] for s in samples if s.get("image_path")]
    else:
        image_paths = sorted((data_dir / "images").glob("*.jpg"))
    if not image_paths:
        raise RuntimeError(f"No images found in {data_dir}")

    objp = checkerboard_object_points(pattern_cols, pattern_rows, square_size_m)
    objpoints = []
    imgpoints = []
    used_images = []
    image_size = None
    annotated_dir = ensure_dir(data_dir / "intrinsics_annotated") if args.save_annotated else None

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"skip unreadable image: {image_path}")
            continue
        detection = detect_checkerboard(
            image_bgr,
            pattern_cols=pattern_cols,
            pattern_rows=pattern_rows,
            use_sb=not args.classic_detector,
        )
        if args.save_annotated:
            annotated = draw_checkerboard_detection(
                image_bgr,
                detection,
                pattern_cols=pattern_cols,
                pattern_rows=pattern_rows,
            )
            cv2.imwrite(str(annotated_dir / image_path.name), annotated)
        if detection is None:
            print(f"not detected: {image_path.name}")
            continue

        h, w = image_bgr.shape[:2]
        if image_size is None:
            image_size = (w, h)
        elif image_size != (w, h):
            raise RuntimeError(f"Image size changed: {image_path} is {(w, h)}, expected {image_size}")

        objpoints.append(objp.copy())
        imgpoints.append(np.asarray(detection["corners_px"], dtype=np.float32).reshape(-1, 1, 2))
        used_images.append(str(image_path.relative_to(data_dir)))

    if len(objpoints) < 10:
        raise RuntimeError(f"Need at least 10 detected checkerboard images, got {len(objpoints)}")
    if image_size is None:
        raise RuntimeError("No usable images")

    flags = 0
    if args.fix_aspect_ratio:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
        flags=flags,
    )
    per_view_errors = reprojection_errors(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs)

    intrinsics = {
        "camera_name": camera_name,
        "camera_model": camera_model,
        "image_size": {"width": int(image_size[0]), "height": int(image_size[1])},
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": [float(v) for v in dist_coeffs.reshape(-1).tolist()],
        "rms_reprojection_error_px": float(rms),
        "mean_reprojection_error_px": float(np.mean(per_view_errors)),
        "max_reprojection_error_px": float(np.max(per_view_errors)),
    }
    output = {
        "calibration_type": "checkerboard_intrinsics",
        "pattern": {
            "pattern_cols": pattern_cols,
            "pattern_rows": pattern_rows,
            "square_size_m": square_size_m,
        },
        "intrinsics": intrinsics,
        "used_images": used_images,
        "per_view_reprojection_error_px": per_view_errors,
    }

    output_json = Path(args.output_json) if args.output_json else data_dir / "intrinsics.json"
    save_json(output_json, output)

    print(f"Used images: {len(used_images)} / {len(image_paths)}")
    print(f"RMS reprojection error: {float(rms):.4f} px")
    print(f"Mean reprojection error: {float(np.mean(per_view_errors)):.4f} px")
    print(f"Max reprojection error: {float(np.max(per_view_errors)):.4f} px")
    print("Camera matrix:")
    for row in camera_matrix.tolist():
        print(f"  {row}")
    print(f"Distortion coeffs: {[float(v) for v in dist_coeffs.reshape(-1).tolist()]}")
    print(f"Wrote: {output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
