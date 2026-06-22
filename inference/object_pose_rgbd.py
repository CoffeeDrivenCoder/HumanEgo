# -*- coding: utf-8 -*-
"""Online RGB-D object 6D pose estimation for HumanEgo inference.

This module is the real-time counterpart of the offline DINO-SAM + depth lifting
+ PCA pose path used by preprocessing. It estimates static object poses at the
start of an episode from RGB-D frames and returns inference.ObjectState values.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from interfaces import Frame, ObjectState
from preprocess.DINOSAM import DINOSAM
from preprocess.OrientAnything import estimate_frame_pca1, estimate_frame_pca2


@dataclass
class RGBDObjectPoseConfig:
    object_prompts: Dict[str, str]
    dinosam_cfg_path: str
    anchor_key: str = "obj1"
    pose_method: str = "pca1"
    min_valid_depth_m: float = 0.1
    max_valid_depth_m: float = 3.0
    max_points: int = 4096
    min_points: int = 80
    mask_erode_px: int = 2
    depth_window_px: int = 2


def _as_config(cfg: dict) -> RGBDObjectPoseConfig:
    pose_method = cfg.get("pose_method", "pca1")
    if isinstance(pose_method, dict):
        pose_method = pose_method.get("default", "pca1")
    return RGBDObjectPoseConfig(
        object_prompts=dict(cfg["object_prompts"]),
        dinosam_cfg_path=cfg["dinosam_cfg_path"],
        anchor_key=cfg.get("anchor_key", "obj1"),
        pose_method=str(pose_method).lower(),
        min_valid_depth_m=float(cfg.get("min_valid_depth_m", 0.1)),
        max_valid_depth_m=float(cfg.get("max_valid_depth_m", 3.0)),
        max_points=int(cfg.get("max_points", 4096)),
        min_points=int(cfg.get("min_points", 80)),
        mask_erode_px=int(cfg.get("mask_erode_px", 2)),
        depth_window_px=int(cfg.get("depth_window_px", 2)),
    )


class RGBDObjectPoseEstimator:
    """Estimate object poses from one or more RGB-D frames.

    The output convention matches inference.interfaces.ObjectState:
    T_in_cam is object-to-camera in OpenCV camera coordinates, metres.
    """

    def __init__(self, cfg: dict | RGBDObjectPoseConfig):
        self.cfg = _as_config(cfg) if isinstance(cfg, dict) else cfg
        self.detector = DINOSAM(self.cfg.dinosam_cfg_path)

    def estimate(self, frames: List[Frame]) -> Dict[str, ObjectState]:
        if not frames:
            raise ValueError("RGBDObjectPoseEstimator.estimate() requires at least one frame")

        frame = frames[0]
        objs: Dict[str, ObjectState] = {}
        anchor_center: Optional[np.ndarray] = None

        for obj_key, prompt in self.cfg.object_prompts.items():
            mask = self.detector.process_single(frame.rgb, prompt)
            pts_cam = self.mask_to_points(frame.depth_m, frame.K, mask)
            T_obj_in_cam, _info = self.points_to_pose(
                pts_cam,
                is_anchor=(obj_key == self.cfg.anchor_key),
                anchor_center_cam=anchor_center,
            )
            if obj_key == self.cfg.anchor_key:
                anchor_center = T_obj_in_cam[:3, 3].copy()

            kpts_local = self.points_to_local(pts_cam, T_obj_in_cam)
            objs[obj_key] = ObjectState(
                T_in_cam=T_obj_in_cam.astype(np.float32),
                kpts_local=kpts_local.astype(np.float32),
            )

        return objs

    def mask_to_points(self, depth_m: np.ndarray, K: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if depth_m is None:
            raise ValueError("depth_m is required for RGB-D object pose estimation")
        if mask is None:
            raise ValueError("DINO-SAM returned no mask")

        depth = np.asarray(depth_m, dtype=np.float64)
        mask_u8 = (mask > 0).astype(np.uint8)
        if self.cfg.mask_erode_px > 0:
            k = 2 * self.cfg.mask_erode_px + 1
            kernel = np.ones((k, k), np.uint8)
            mask_u8 = cv2.erode(mask_u8, kernel, iterations=1)

        valid = (
            (mask_u8 > 0)
            & np.isfinite(depth)
            & (depth >= self.cfg.min_valid_depth_m)
            & (depth <= self.cfg.max_valid_depth_m)
        )

        if self.cfg.depth_window_px > 0:
            valid = self._fill_valid_from_neighborhood(depth, mask_u8, valid)

        vs, us = np.where(valid)
        if len(us) < self.cfg.min_points:
            raise ValueError(f"Only {len(us)} valid depth points inside object mask")

        if len(us) > self.cfg.max_points:
            idx = np.linspace(0, len(us) - 1, self.cfg.max_points).astype(np.int64)
            us, vs = us[idx], vs[idx]

        z = depth[vs, us]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        x = (us.astype(np.float64) - cx) * z / fx
        y = (vs.astype(np.float64) - cy) * z / fy
        return np.stack([x, y, z], axis=1)

    def _fill_valid_from_neighborhood(
        self, depth: np.ndarray, mask_u8: np.ndarray, valid: np.ndarray
    ) -> np.ndarray:
        if valid.sum() >= self.cfg.min_points:
            return valid
        radius = self.cfg.depth_window_px
        ys, xs = np.where(mask_u8 > 0)
        filled = valid.copy()
        for y, x in zip(ys, xs):
            if filled[y, x]:
                continue
            y0, y1 = max(0, y - radius), min(depth.shape[0], y + radius + 1)
            x0, x1 = max(0, x - radius), min(depth.shape[1], x + radius + 1)
            patch = depth[y0:y1, x0:x1]
            ok = (
                np.isfinite(patch)
                & (patch >= self.cfg.min_valid_depth_m)
                & (patch <= self.cfg.max_valid_depth_m)
            )
            if ok.any():
                depth[y, x] = float(np.median(patch[ok]))
                filled[y, x] = True
        return filled

    def points_to_pose(
        self,
        pts_cam: np.ndarray,
        is_anchor: bool,
        anchor_center_cam: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, dict]:
        if self.cfg.pose_method == "pca1":
            return estimate_frame_pca1(pts_cam, is_anchor=is_anchor, anchor_center_cam=anchor_center_cam)
        if self.cfg.pose_method == "pca2":
            return estimate_frame_pca2(pts_cam, is_anchor=is_anchor, anchor_center_cam=anchor_center_cam)
        raise ValueError(f"Unsupported online pose_method: {self.cfg.pose_method}")

    @staticmethod
    def points_to_local(pts_cam: np.ndarray, T_obj_in_cam: np.ndarray) -> np.ndarray:
        pts_h = np.concatenate([pts_cam, np.ones((len(pts_cam), 1), dtype=np.float64)], axis=1)
        pts_local_h = (np.linalg.inv(T_obj_in_cam) @ pts_h.T).T
        return pts_local_h[:, :3]


class RGBDObjectPosePerception:
    """Small adapter that satisfies the Perception.estimate_objects contract."""

    def __init__(self, cfg: dict):
        self.estimator = RGBDObjectPoseEstimator(cfg)

    def estimate_objects(self, frames: List[Frame]) -> Dict[str, ObjectState]:
        return self.estimator.estimate(frames)

    def make_clean_image(self, frame, ee_poses_in_cam, grippers) -> np.ndarray:
        return frame.rgb.copy()

