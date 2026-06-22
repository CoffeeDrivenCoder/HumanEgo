#!/usr/bin/env python3
"""Checkerboard helpers for Agibot G1 camera calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hand_eye_common import as_matrix4, load_json, make_transform


def checkerboard_object_points(pattern_cols: int, pattern_rows: int, square_size_m: float) -> np.ndarray:
    """Return OpenCV object points for inner checkerboard corners."""
    if pattern_cols <= 1 or pattern_rows <= 1:
        raise ValueError("pattern_cols and pattern_rows must both be > 1")
    if square_size_m <= 0.0:
        raise ValueError("square_size_m must be positive")

    objp = np.zeros((pattern_cols * pattern_rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2)
    objp *= float(square_size_m)
    return objp


def image_to_gray(image: Any) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] >= 3:
        return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    raise ValueError(f"unsupported image shape: {img.shape}")


def detect_checkerboard(
    image: Any,
    *,
    pattern_cols: int,
    pattern_rows: int,
    use_sb: bool = True,
) -> dict[str, Any] | None:
    gray = image_to_gray(image)
    pattern_size = (int(pattern_cols), int(pattern_rows))

    if use_sb and hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags)
        method = "findChessboardCornersSB"
    else:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        method = "findChessboardCorners"
        if ok:
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                40,
                0.001,
            )
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    if not ok or corners is None:
        return None

    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    expected = int(pattern_cols) * int(pattern_rows)
    if corners.shape[0] != expected:
        return None

    return {
        "pattern_cols": int(pattern_cols),
        "pattern_rows": int(pattern_rows),
        "method": method,
        "corners_px": corners.tolist(),
        "center_px": np.mean(corners, axis=0).tolist(),
        "image_size": {"width": int(gray.shape[1]), "height": int(gray.shape[0])},
    }


def draw_checkerboard_detection(
    image_bgr: np.ndarray,
    detection: dict[str, Any] | None,
    *,
    pattern_cols: int,
    pattern_rows: int,
) -> np.ndarray:
    annotated = image_bgr.copy()
    if detection is None:
        cv2.putText(
            annotated,
            "checkerboard not detected",
            (24, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    corners = np.asarray(detection["corners_px"], dtype=np.float32).reshape(-1, 1, 2)
    cv2.drawChessboardCorners(annotated, (int(pattern_cols), int(pattern_rows)), corners, True)
    center = tuple(int(v) for v in detection["center_px"])
    cv2.circle(annotated, center, 5, (0, 0, 255), -1)
    return annotated


def intrinsics_from_json(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data = load_json(path)
    source = data.get("intrinsics", data)

    if "camera_matrix" in source:
        camera_matrix = np.asarray(source["camera_matrix"], dtype=np.float64).reshape(3, 3)
    else:
        camera_matrix = np.asarray(
            [
                [float(source["fx"]), 0.0, float(source["cx"])],
                [0.0, float(source["fy"]), float(source["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    dist_values = source.get("dist_coeffs", source.get("distortion_coefficients", []))
    dist_coeffs = np.asarray(dist_values, dtype=np.float64).reshape(-1, 1)
    if dist_coeffs.size == 0:
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    camera_params = {
        "camera_name": source.get("camera_name", data.get("camera_name", "head")),
        "camera_model": source.get("camera_model", data.get("camera_model", "")),
        "image_size": source.get("image_size", data.get("image_size", {})),
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": [float(v) for v in dist_coeffs.reshape(-1).tolist()],
    }
    return camera_matrix, dist_coeffs, camera_params


def solve_checkerboard_pose(
    image: Any,
    *,
    pattern_cols: int,
    pattern_rows: int,
    square_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    use_sb: bool = True,
) -> dict[str, Any] | None:
    detection = detect_checkerboard(
        image,
        pattern_cols=pattern_cols,
        pattern_rows=pattern_rows,
        use_sb=use_sb,
    )
    if detection is None:
        return None

    objp = checkerboard_object_points(pattern_cols, pattern_rows, square_size_m).astype(np.float64)
    corners = np.asarray(detection["corners_px"], dtype=np.float64).reshape(-1, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(
        objp,
        corners,
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(dist_coeffs, dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    rotation, _ = cv2.Rodrigues(rvec)
    transform = make_transform(rotation, tvec.reshape(3))
    detection.update(
        {
            "square_size_m": float(square_size_m),
            "rvec": [float(v) for v in rvec.reshape(-1).tolist()],
            "tvec_m": [float(v) for v in tvec.reshape(-1).tolist()],
            "position_camera_m": [float(v) for v in tvec.reshape(-1).tolist()],
            "rotation_matrix": rotation.tolist(),
            "T_camera_board": transform.tolist(),
        }
    )
    return detection


def sample_t_camera_board(sample: dict[str, Any]) -> np.ndarray:
    board = sample.get("board") or {}
    if board.get("T_camera_board") is not None:
        return as_matrix4(board["T_camera_board"])
    if board.get("rotation_matrix") is not None and board.get("position_camera_m") is not None:
        return make_transform(board["rotation_matrix"], board["position_camera_m"])
    raise ValueError(f"sample {sample.get('sample_id')} does not contain a checkerboard pose")
