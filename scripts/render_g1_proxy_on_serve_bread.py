#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render a lightweight single-arm G1-style proxy on HumanEgo serve_bread frames.

This is the first robotization pass for the released HumanEgo data. It does not
require a MuJoCo G1 model. Instead it uses the accurate HumanEgo hand trajectory
and hand keypoints to draw a stable 2D/3D proxy arm and gripper over an inpainted
background. The data loading and pose conversion are intentionally separated from
the proxy renderer so a real G1 MJCF renderer can replace it later.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MPS_PATH = Path("/data/wangk/data/serve_bread/aria/mps_serve_bread_006_vrs")
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "render_g1_proxy" / "mps_serve_bread_006_vrs_proxy.mp4"


@dataclass
class FrameInputs:
    frame_dir: Path
    training: dict
    aria_hands: dict | None
    rgb_bgr: np.ndarray
    background_bgr: np.ndarray
    mask_arm: np.ndarray | None


@dataclass
class RenderGeometry:
    root_uv: np.ndarray
    elbow_uv: np.ndarray
    wrist_uv: np.ndarray
    gripper_center_uv: np.ndarray
    gripper_forward_uv: np.ndarray | None
    thumb_uv: np.ndarray | None
    index_uv: np.ndarray | None
    tcp_uv: np.ndarray | None
    tcp_forward_uv: np.ndarray | None
    tcp_z: float | None
    grasp: float


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def imread_required(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def load_g1_hand_alignment() -> np.ndarray:
    sys.path.insert(0, str(PROJECT_ROOT / "inference"))
    try:
        from G1Geometry import fixed_T_hand_in_tcp

        return fixed_T_hand_in_tcp("right")
    except Exception:
        # Same matrix as cfg/inference/g1_serve_bread_right.yaml.
        return np.asarray(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


def find_frame_dirs(mps_path: Path) -> list[Path]:
    all_data = mps_path / "preprocess" / "all_data"
    if not all_data.is_dir():
        raise FileNotFoundError(f"Missing all_data directory: {all_data}")
    frame_dirs = [
        p
        for p in all_data.iterdir()
        if p.is_dir() and p.name.isdigit() and (p / "training_data.json").exists()
    ]
    return sorted(frame_dirs, key=lambda p: int(p.name))


def find_rgb_frame_dirs(mps_path: Path) -> list[Path]:
    all_data = mps_path / "preprocess" / "all_data"
    if not all_data.is_dir():
        raise FileNotFoundError(f"Missing all_data directory: {all_data}")
    frame_dirs = [
        p
        for p in all_data.iterdir()
        if p.is_dir() and p.name.isdigit() and (p / "rgb.png").exists()
    ]
    return sorted(frame_dirs, key=lambda p: int(p.name))


def select_frame_dirs(
    frame_dirs: list[Path],
    start_index: int,
    start_frame_id: int | None,
    num_frames: int | None,
    stride: int,
) -> list[Path]:
    if start_frame_id is not None:
        start_index = 0
        for i, p in enumerate(frame_dirs):
            if int(p.name) >= start_frame_id:
                start_index = i
                break
    selected = frame_dirs[start_index::stride]
    if num_frames is not None:
        selected = selected[:num_frames]
    return selected


def read_camera_config(mps_path: Path) -> dict | None:
    return read_json(mps_path / "preprocess" / "aria_cam_rgb_config.json")


def load_background(frame_dir: Path, mode: str, inpaint_radius: float, dilate_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    rgb_bgr = imread_required(frame_dir / "rgb.png")
    mask_path = frame_dir / "mask_arm.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None

    if mode == "rgb":
        return rgb_bgr, rgb_bgr.copy(), mask

    if mode == "woarm-kpts":
        bg_path = frame_dir / "rgb_WoArm_WArmObjKpts.png"
        return rgb_bgr, imread_required(bg_path), mask

    if mode == "woarm":
        bg_path = frame_dir / "rgb_WoArm.png"
        if bg_path.exists():
            return rgb_bgr, imread_required(bg_path), mask
        mode = "inpaint"

    if mode != "inpaint":
        raise ValueError(f"Unknown background mode: {mode}")

    if mask is None:
        return rgb_bgr, rgb_bgr.copy(), None
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = np.ones((k, k), np.uint8)
        mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
    background = cv2.inpaint(rgb_bgr, mask_u8, inpaint_radius, cv2.INPAINT_TELEA)
    return rgb_bgr, background, mask_u8


def project_points(points_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    z = pts[:, 2].copy()
    uv = np.full((len(pts), 2), np.nan, dtype=np.float64)
    valid = z > 1e-6
    if np.any(valid):
        proj = (K @ pts[valid].T).T
        uv[valid] = proj[:, :2] / proj[:, 2:3]
    return uv, z


def project_world_point(training: dict, point_world: Iterable[float]) -> tuple[np.ndarray | None, float | None]:
    try:
        K = np.asarray(training["metadata"]["k"], dtype=np.float64).reshape(3, 3)
        c2w = np.asarray(training["metadata"]["c2w"], dtype=np.float64).reshape(4, 4)
        point_world_h = np.ones(4, dtype=np.float64)
        point_world_h[:3] = np.asarray(point_world, dtype=np.float64).reshape(3)
        point_cam = np.linalg.inv(c2w) @ point_world_h
        uv, z = project_points(point_cam[:3], K)
        if not np.all(np.isfinite(uv[0])):
            return None, None
        return uv[0], float(z[0])
    except Exception:
        return None, None


def safe_uv(value: Iterable[float], width: int, height: int) -> np.ndarray:
    uv = np.asarray(value, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(uv)):
        uv = np.asarray([width * 0.7, height * 0.55], dtype=np.float64)
    return uv


def hand_keypoints_2d(aria_hands: dict | None) -> np.ndarray | None:
    if not aria_hands:
        return None
    hand_r = aria_hands.get("hand_r")
    if not hand_r:
        return None
    kpts = np.asarray(hand_r.get("kpts_2d", []), dtype=np.float64)
    if kpts.shape != (21, 2) or not np.all(np.isfinite(kpts)):
        return None
    return kpts


def pose_translation(value: object) -> np.ndarray | None:
    try:
        T = np.asarray(value, dtype=np.float64).reshape(4, 4)
        if not np.all(np.isfinite(T)):
            return None
        return T[:3, 3]
    except Exception:
        return None


def world_point_uv(training: dict, hand_r: dict, key: str) -> np.ndarray | None:
    value = hand_r.get(key)
    if value is None:
        return None
    uv, _ = project_world_point(training, value)
    return uv


def world_pose_uv(training: dict, hand_r: dict, key: str) -> np.ndarray | None:
    point = pose_translation(hand_r.get(key))
    if point is None:
        return None
    uv, _ = project_world_point(training, point)
    return uv


def visual_hand_points(training: dict, aria_hands: dict | None) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Return wrist, gripper midpoint, thumb tip, index tip in image pixels.

    HumanEgo stores Project Aria keypoints, not MediaPipe ordering:
      0 thumb tip, 1 index tip, 5 wrist, 6 thumb base, 8 index base.
    Prefer optimized world-space points because they are the same source used to
    build T_hand_to_world; fall back to 2D keypoints for older/preliminary data.
    """
    hand_r = aria_hands.get("hand_r") if aria_hands else None
    if not hand_r:
        return None, None, None, None

    kpts = hand_keypoints_2d(aria_hands)
    wrist_uv = world_pose_uv(training, hand_r, "wrist_pose_opt_world")
    center_uv = world_point_uv(training, hand_r, "midpoint_translation_opt_world")
    thumb_uv = world_point_uv(training, hand_r, "thumb_translation_opt_world")
    index_uv = world_point_uv(training, hand_r, "index_translation_opt_world")

    if kpts is not None:
        if wrist_uv is None:
            wrist_uv = kpts[5]
        if thumb_uv is None:
            thumb_uv = kpts[0]
        if index_uv is None:
            index_uv = kpts[1]
        if center_uv is None:
            center_uv = 0.5 * (thumb_uv + index_uv)

    return wrist_uv, center_uv, thumb_uv, index_uv


def estimate_root_and_elbow(mask: np.ndarray | None, wrist_uv: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    if mask is None or not np.any(mask > 0):
        root = np.asarray([width + 45.0, np.clip(wrist_uv[1] + 75.0, 0, height - 1)], dtype=np.float64)
        elbow = 0.55 * root + 0.45 * wrist_uv
        return root, elbow

    ys, xs = np.where(mask > 0)
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)

    x_hi = np.percentile(pts[:, 0], 96)
    root_band = pts[pts[:, 0] >= x_hi]
    if len(root_band) == 0:
        root_band = pts
    root = np.asarray([width + 45.0, np.median(root_band[:, 1])], dtype=np.float64)

    # The mask centroid gives a surprisingly useful visual elbow for egocentric
    # forearms, and keeps the proxy arm sitting on top of the removed human arm.
    mid_band = pts[pts[:, 0] < root[0] - 12]
    if len(mid_band) < 20:
        mid_band = pts
    centroid = np.mean(mid_band, axis=0)
    line_mid = 0.52 * root + 0.48 * wrist_uv
    elbow = 0.55 * centroid + 0.45 * line_mid
    elbow[0] = np.clip(elbow[0], -50, width + 50)
    elbow[1] = np.clip(elbow[1], -50, height + 50)
    return root, elbow


def parse_axis(axis: str) -> tuple[int, float]:
    axis = axis.lower()
    if len(axis) == 1:
        sign = 1.0
        name = axis
    elif len(axis) == 2 and axis[0] in "+-":
        sign = -1.0 if axis[0] == "-" else 1.0
        name = axis[1]
    else:
        raise ValueError(f"Invalid axis {axis!r}; expected x/y/z or +x/-x/+y/-y/+z/-z")
    if name not in {"x", "y", "z"}:
        raise ValueError(f"Invalid axis {axis!r}; expected x/y/z or +x/-x/+y/-y/+z/-z")
    return {"x": 0, "y": 1, "z": 2}[name], sign


def compute_tcp_projection(
    training: dict,
    T_hand_in_tcp: np.ndarray,
    forward_axis: str,
) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
    try:
        K = np.asarray(training["metadata"]["k"], dtype=np.float64).reshape(3, 3)
        c2w = np.asarray(training["metadata"]["c2w"], dtype=np.float64).reshape(4, 4)
        hand = training["entities"]["hands"]["right"]
        T_hand_w = np.asarray(hand["T_hand_to_world"], dtype=np.float64).reshape(4, 4)
        T_tcp_w = T_hand_w @ np.linalg.inv(T_hand_in_tcp)
        T_tcp_c = np.linalg.inv(c2w) @ T_tcp_w
        axis_idx, axis_sign = parse_axis(forward_axis)
        origin_uv, z = project_points(T_tcp_c[:3, 3], K)
        forward_uv, _ = project_points(T_tcp_c[:3, 3] + axis_sign * 0.08 * T_tcp_c[:3, axis_idx], K)
        if not np.all(np.isfinite(origin_uv[0])):
            return None, None, None
        if not np.all(np.isfinite(forward_uv[0])):
            forward_uv = np.full_like(origin_uv, np.nan)
        return origin_uv[0], forward_uv[0], float(z[0])
    except Exception:
        return None, None, None


def reconstruct_training_from_aria(frame_dir: Path, aria_hands: dict | None, camera_config: dict | None) -> dict | None:
    if not aria_hands or camera_config is None:
        return None
    hand_r = aria_hands.get("hand_r")
    if not hand_r:
        return None
    c2w = hand_r.get("c2w")
    midpoint = hand_r.get("midpoint_translation_opt_world")
    orientation = hand_r.get("midpoint_orientation_opt_world")
    if c2w is None or midpoint is None or orientation is None:
        return None
    try:
        T_hand_to_world = np.eye(4, dtype=np.float64)
        T_hand_to_world[:3, :3] = np.asarray(orientation, dtype=np.float64).reshape(3, 3)
        T_hand_to_world[:3, 3] = np.asarray(midpoint, dtype=np.float64).reshape(3)
        return {
            "metadata": {
                "k": camera_config["k"],
                "c2w": c2w,
            },
            "entities": {
                "hands": {
                    "right": {
                        "T_hand_to_world": T_hand_to_world.tolist(),
                        "grasp": 1.0 if hand_r.get("grasp_state", False) else 0.0,
                    }
                }
            },
            "_source": f"reconstructed_from_aria_hands:{frame_dir.name}",
        }
    except Exception:
        return None


def make_geometry(
    frame: FrameInputs,
    T_hand_in_tcp: np.ndarray,
    forward_axis: str,
    orientation_source: str,
) -> RenderGeometry | None:
    training = frame.training
    height, width = frame.background_bgr.shape[:2]
    hand = training.get("entities", {}).get("hands", {}).get("right")
    if not hand:
        return None
    grasp = float(hand.get("grasp", 0.0))

    tcp_uv, tcp_forward_uv, tcp_z = compute_tcp_projection(training, T_hand_in_tcp, forward_axis)
    visual_wrist_uv, visual_center_uv, thumb_uv, index_uv = visual_hand_points(training, frame.aria_hands)

    if visual_wrist_uv is not None and visual_center_uv is not None:
        wrist_uv = safe_uv(visual_wrist_uv, width, height)
        gripper_center_uv = safe_uv(visual_center_uv, width, height)
        thumb_uv = safe_uv(thumb_uv, width, height) if thumb_uv is not None else None
        index_uv = safe_uv(index_uv, width, height) if index_uv is not None else None
    elif tcp_uv is not None:
        wrist_uv = safe_uv(tcp_uv + np.asarray([8.0, 26.0]), width, height)
        thumb_uv = None
        index_uv = None
        gripper_center_uv = safe_uv(tcp_uv, width, height)
    else:
        return None

    if orientation_source == "tcp" and tcp_forward_uv is not None and np.all(np.isfinite(tcp_forward_uv)):
        gripper_forward_uv = tcp_forward_uv
    else:
        forward_dir = normalized(gripper_center_uv - wrist_uv, np.asarray([1.0, 0.0]))
        gripper_forward_uv = gripper_center_uv + forward_dir * 80.0

    root_uv, elbow_uv = estimate_root_and_elbow(frame.mask_arm, wrist_uv, width, height)
    return RenderGeometry(
        root_uv=root_uv,
        elbow_uv=elbow_uv,
        wrist_uv=wrist_uv,
        gripper_center_uv=gripper_center_uv,
        gripper_forward_uv=gripper_forward_uv,
        thumb_uv=thumb_uv,
        index_uv=index_uv,
        tcp_uv=tcp_uv,
        tcp_forward_uv=tcp_forward_uv,
        tcp_z=tcp_z,
        grasp=grasp,
    )


def clamp_point(pt: np.ndarray) -> tuple[int, int]:
    return int(round(float(pt[0]))), int(round(float(pt[1])))


def draw_segment(img: np.ndarray, p0: np.ndarray, p1: np.ndarray, radius: int, color: tuple[int, int, int]) -> None:
    p0i, p1i = clamp_point(p0), clamp_point(p1)
    cv2.line(img, p0i, p1i, (18, 22, 28), radius + 8, cv2.LINE_AA)
    cv2.line(img, p0i, p1i, (60, 66, 74), radius + 3, cv2.LINE_AA)
    cv2.line(img, p0i, p1i, color, radius, cv2.LINE_AA)
    highlight = tuple(int(0.55 * c + 0.45 * 255) for c in color)
    cv2.line(img, p0i, p1i, highlight, max(2, radius // 3), cv2.LINE_AA)


def draw_joint(img: np.ndarray, center: np.ndarray, radius: int, color: tuple[int, int, int]) -> None:
    c = clamp_point(center)
    cv2.circle(img, c, radius + 5, (18, 22, 28), -1, cv2.LINE_AA)
    cv2.circle(img, c, radius + 1, (65, 70, 78), -1, cv2.LINE_AA)
    cv2.circle(img, c, radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, c, max(2, radius // 3), (230, 238, 242), -1, cv2.LINE_AA)


def draw_joint_if_visible(img: np.ndarray, center: np.ndarray, radius: int, color: tuple[int, int, int]) -> None:
    h, w = img.shape[:2]
    margin = radius + 8
    if -margin <= center[0] <= w + margin and -margin <= center[1] <= h + margin:
        draw_joint(img, center, radius, color)


def normalized(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return fallback.astype(np.float64)
    return v.astype(np.float64) / n


def draw_gripper(img: np.ndarray, geom: RenderGeometry, scale: float, color: tuple[int, int, int]) -> None:
    center = geom.gripper_center_uv
    if geom.gripper_forward_uv is not None and np.all(np.isfinite(geom.gripper_forward_uv)):
        wrist_to_center = normalized(geom.gripper_forward_uv - center, center - geom.wrist_uv)
    else:
        wrist_to_center = normalized(center - geom.wrist_uv, np.asarray([1.0, 0.0]))
    if geom.thumb_uv is not None and geom.index_uv is not None:
        spread = normalized(geom.index_uv - geom.thumb_uv, np.asarray([0.0, -1.0]))
    else:
        spread = np.asarray([-wrist_to_center[1], wrist_to_center[0]], dtype=np.float64)

    # Open hand gets wider jaws, grasped hand gets a tighter visual gap.
    jaw_gap = (18.0 if geom.grasp > 0.5 else 34.0) * scale
    palm_half = jaw_gap * 0.5
    finger_len = (34.0 if geom.grasp > 0.5 else 42.0) * scale
    finger_w = max(4, int(round(7 * scale)))

    palm_a = center - spread * palm_half - wrist_to_center * 8.0 * scale
    palm_b = center + spread * palm_half - wrist_to_center * 8.0 * scale
    tip_a = palm_a + wrist_to_center * finger_len
    tip_b = palm_b + wrist_to_center * finger_len

    cv2.line(img, clamp_point(geom.wrist_uv), clamp_point(center), (28, 32, 38), max(8, finger_w + 4), cv2.LINE_AA)
    cv2.line(img, clamp_point(geom.wrist_uv), clamp_point(center), color, max(4, finger_w), cv2.LINE_AA)
    cv2.line(img, clamp_point(palm_a), clamp_point(palm_b), (18, 22, 28), finger_w + 5, cv2.LINE_AA)
    cv2.line(img, clamp_point(palm_a), clamp_point(palm_b), color, finger_w, cv2.LINE_AA)
    for base, tip in ((palm_a, tip_a), (palm_b, tip_b)):
        cv2.line(img, clamp_point(base), clamp_point(tip), (18, 22, 28), finger_w + 5, cv2.LINE_AA)
        cv2.line(img, clamp_point(base), clamp_point(tip), color, finger_w, cv2.LINE_AA)
        cv2.circle(img, clamp_point(tip), max(3, finger_w // 2), (225, 235, 238), -1, cv2.LINE_AA)


def draw_debug_pose(img: np.ndarray, geom: RenderGeometry) -> None:
    cv2.circle(img, clamp_point(geom.wrist_uv), 5, (255, 128, 0), -1, cv2.LINE_AA)
    cv2.circle(img, clamp_point(geom.gripper_center_uv), 5, (0, 255, 255), -1, cv2.LINE_AA)
    if geom.gripper_forward_uv is not None and np.all(np.isfinite(geom.gripper_forward_uv)):
        cv2.arrowedLine(
            img,
            clamp_point(geom.gripper_center_uv),
            clamp_point(geom.gripper_forward_uv),
            (0, 220, 0),
            2,
            cv2.LINE_AA,
            tipLength=0.25,
        )
    if geom.thumb_uv is not None:
        cv2.circle(img, clamp_point(geom.thumb_uv), 4, (255, 0, 255), -1, cv2.LINE_AA)
    if geom.index_uv is not None:
        cv2.circle(img, clamp_point(geom.index_uv), 4, (0, 128, 255), -1, cv2.LINE_AA)
    if geom.tcp_uv is None:
        return
    cv2.drawMarker(
        img,
        clamp_point(geom.tcp_uv),
        (0, 180, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=16,
        thickness=2,
        line_type=cv2.LINE_AA,
    )


def render_proxy(frame: FrameInputs, geom: RenderGeometry, args: argparse.Namespace) -> np.ndarray:
    out = frame.background_bgr.copy()
    height, width = out.shape[:2]
    scale = max(0.65, min(1.35, min(width / 640.0, height / 480.0)))
    arm_radius = max(10, int(round(args.arm_radius * scale)))
    joint_radius = max(10, int(round(args.joint_radius * scale)))
    color = tuple(int(v) for v in args.arm_color_bgr)

    overlay = out.copy()
    draw_segment(overlay, geom.root_uv, geom.elbow_uv, arm_radius + 2, color)
    draw_segment(overlay, geom.elbow_uv, geom.wrist_uv, arm_radius, color)
    draw_joint_if_visible(overlay, geom.root_uv, joint_radius, (75, 82, 92))
    draw_joint_if_visible(overlay, geom.elbow_uv, joint_radius + 2, (86, 96, 108))
    draw_joint_if_visible(overlay, geom.wrist_uv, max(8, joint_radius - 3), (92, 103, 116))
    draw_gripper(overlay, geom, scale, (145, 155, 162))
    alpha = float(args.overlay_alpha)
    out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)

    if args.debug_pose:
        draw_debug_pose(out, geom)
    return out


def load_frame(frame_dir: Path, args: argparse.Namespace, camera_config: dict | None = None) -> FrameInputs | None:
    training = read_json(frame_dir / "training_data.json")
    aria_hands = read_json(frame_dir / "aria_hands.json")
    if training is None:
        training = reconstruct_training_from_aria(frame_dir, aria_hands, camera_config)
    if training is None:
        if args.include_all_rgb_frames:
            rgb, bg, mask = load_background(frame_dir, args.background, args.inpaint_radius, args.mask_dilate)
            return FrameInputs(
                frame_dir=frame_dir,
                training={},
                aria_hands=aria_hands,
                rgb_bgr=rgb,
                background_bgr=bg,
                mask_arm=mask,
            )
        return None
    rgb, bg, mask = load_background(frame_dir, args.background, args.inpaint_radius, args.mask_dilate)
    return FrameInputs(
        frame_dir=frame_dir,
        training=training,
        aria_hands=aria_hands,
        rgb_bgr=rgb,
        background_bgr=bg,
        mask_arm=mask,
    )


def make_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def parse_color(value: str) -> tuple[int, int, int]:
    parts = [int(x) for x in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("color must be B,G,R, e.g. 92,117,135")
    if any(v < 0 or v > 255 for v in parts):
        raise argparse.ArgumentTypeError("color values must be in [0, 255]")
    return tuple(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mps-path", type=Path, default=DEFAULT_MPS_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start-index", type=int, default=0, help="Index in the sorted all_data frame list.")
    parser.add_argument("--start-frame-id", type=int, default=None, help="Start from first frame directory >= this id.")
    parser.add_argument("--num-frames", type=int, default=None, help="Number of selected frames to render. Default: all.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--include-all-rgb-frames",
        action="store_true",
        help="Select every rgb.png frame, reconstructing hand pose from aria_hands when training_data.json is absent.",
    )
    parser.add_argument(
        "--use-camera-config-fps",
        action="store_true",
        help="Use preprocess/aria_cam_rgb_config.json fps when available.",
    )
    parser.add_argument(
        "--background",
        choices=["inpaint", "rgb", "woarm", "woarm-kpts"],
        default="inpaint",
        help="Background source. 'inpaint' uses rgb.png + mask_arm.png.",
    )
    parser.add_argument("--inpaint-radius", type=float, default=5.0)
    parser.add_argument("--mask-dilate", type=int, default=5)
    parser.add_argument("--overlay-alpha", type=float, default=0.98)
    parser.add_argument("--arm-radius", type=int, default=14)
    parser.add_argument("--joint-radius", type=int, default=13)
    parser.add_argument("--arm-color-bgr", type=parse_color, default=(94, 121, 139))
    parser.add_argument(
        "--tcp-forward-axis",
        default="+z",
        help="TCP axis used when --orientation-source tcp. G1 URDF gripper center is +Z; use -z to flip.",
    )
    parser.add_argument(
        "--orientation-source",
        choices=["visual", "tcp"],
        default="visual",
        help="Use HumanEgo wrist->midpoint visual direction, or projected TCP axis for diagnosis.",
    )
    parser.add_argument("--debug-pose", action="store_true", help="Draw the projected HumanEgo/G1 TCP target.")
    parser.add_argument("--save-first-frame", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.num_frames is not None and args.num_frames < 1:
        raise ValueError("--num-frames must be >= 1")

    camera_config = read_camera_config(args.mps_path)
    if args.use_camera_config_fps and camera_config is not None and "fps" in camera_config:
        args.fps = float(camera_config["fps"])

    available_frame_dirs = find_rgb_frame_dirs(args.mps_path) if args.include_all_rgb_frames else find_frame_dirs(args.mps_path)
    frame_dirs = select_frame_dirs(
        available_frame_dirs,
        start_index=args.start_index,
        start_frame_id=args.start_frame_id,
        num_frames=args.num_frames,
        stride=args.stride,
    )
    if not frame_dirs:
        raise RuntimeError("No frames selected.")

    T_hand_in_tcp = load_g1_hand_alignment()
    writer: cv2.VideoWriter | None = None
    written = 0
    skipped = 0

    for frame_dir in frame_dirs:
        frame = load_frame(frame_dir, args, camera_config)
        if frame is None:
            skipped += 1
            continue
        geom = make_geometry(frame, T_hand_in_tcp, args.tcp_forward_axis, args.orientation_source)
        if geom is None:
            skipped += 1
            rendered = frame.background_bgr
        else:
            rendered = render_proxy(frame, geom, args)

        if writer is None:
            h, w = rendered.shape[:2]
            writer = make_writer(args.out, args.fps, (w, h))
            if args.save_first_frame is not None:
                args.save_first_frame.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.save_first_frame), rendered)
        writer.write(rendered)
        written += 1

    if writer is not None:
        writer.release()

    report = {
        "mps_path": str(args.mps_path),
        "out": str(args.out),
        "frames_selected": len(frame_dirs),
        "frames_written": written,
        "frames_skipped": skipped,
        "background": args.background,
        "orientation_source": args.orientation_source,
        "tcp_forward_axis": args.tcp_forward_axis,
        "note": "Proxy renderer: G1-style visual arm, not full URDF/MJCF rendering yet.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
