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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    fallback_dilate_px: int = 3
    fallback_depth_window_px: int = 6
    object_filters: Dict[str, Dict[str, Any]] | None = None


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
        fallback_dilate_px=int(cfg.get("fallback_dilate_px", 3)),
        fallback_depth_window_px=int(cfg.get("fallback_depth_window_px", 6)),
        object_filters=dict(cfg.get("object_filters", {})),
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
        objects, _debug = self.estimate_with_debug(frames)
        return objects

    def estimate_with_debug(
        self,
        frames: List[Frame],
        debug_dir: str | os.PathLike[str] | None = None,
    ) -> tuple[Dict[str, ObjectState], dict[str, Any]]:
        if not frames:
            raise ValueError("RGBDObjectPoseEstimator.estimate() requires at least one frame")

        frame = frames[0]
        objs: Dict[str, ObjectState] = {}
        anchor_center: Optional[np.ndarray] = None
        debug_path = Path(debug_dir) if debug_dir is not None else None
        if debug_path is not None:
            debug_path.mkdir(parents=True, exist_ok=True)
        debug: dict[str, Any] = {
            "objects": {},
            "config": {
                "pose_method": self.cfg.pose_method,
                "min_points": self.cfg.min_points,
                "max_points": self.cfg.max_points,
                "mask_erode_px": self.cfg.mask_erode_px,
                "depth_window_px": self.cfg.depth_window_px,
                "fallback_dilate_px": self.cfg.fallback_dilate_px,
                "fallback_depth_window_px": self.cfg.fallback_depth_window_px,
                "min_valid_depth_m": self.cfg.min_valid_depth_m,
                "max_valid_depth_m": self.cfg.max_valid_depth_m,
                "object_filters": self.cfg.object_filters or {},
            },
        }

        for obj_key, prompt in self.cfg.object_prompts.items():
            mask, selection_debug = self.segment_object(frame, obj_key, prompt)
            pts_cam, point_info = self.mask_to_points(frame.depth_m, frame.K, mask, return_info=True)
            self.validate_point_info(obj_key, point_info)
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
            object_debug = {
                "prompt": prompt,
                "T_in_cam": T_obj_in_cam.tolist(),
                "kpts_local_count": int(len(kpts_local)),
                "segmentation": selection_debug,
                "points": {k: v for k, v in point_info.items() if not k.startswith("_")},
                "saved_files": {},
            }
            if debug_path is not None:
                object_debug["saved_files"] = self.save_debug_images(
                    debug_path,
                    obj_key,
                    frame.rgb,
                    frame.depth_m,
                    frame.K,
                    mask,
                    point_info,
                    T_obj_in_cam,
                    selection_debug,
                )
            debug["objects"][obj_key] = object_debug

        return objs, debug

    def filter_cfg(self, obj_key: str) -> dict[str, Any]:
        filters = self.cfg.object_filters or {}
        return dict(filters.get(obj_key, {}))

    def segment_object(self, frame: Frame, obj_key: str, prompt: str) -> tuple[np.ndarray, dict[str, Any]]:
        cfg = self.filter_cfg(obj_key)
        use_candidates = bool(cfg.get("use_dinosam_candidates", False))
        if use_candidates and hasattr(self.detector, "process_single_candidates"):
            candidates = self.detector.process_single_candidates(frame.rgb, prompt)
            try:
                selected_mask, debug = self.select_mask_candidate(
                    obj_key=obj_key,
                    candidates=candidates,
                    image_shape=frame.rgb.shape[:2],
                )
                return selected_mask, debug
            except ValueError as exc:
                if bool(cfg.get("plate_circle_fallback", False)):
                    union_mask = self.union_candidate_masks(candidates, frame.rgb.shape[:2])
                    fallback_mask, fallback_debug = self.plate_circle_fallback_mask(
                        frame.rgb,
                        union_mask,
                        cfg,
                    )
                    fallback_debug["candidate_filter_error"] = str(exc)
                    fallback_debug["candidate_count"] = len(candidates)
                    return fallback_mask, fallback_debug
                raise

        mask = self.detector.process_single(frame.rgb, prompt)
        debug = {
            "mode": "combined_mask",
            "selected": True,
            "raw_mask_pixels": int((mask > 0).sum()) if mask is not None else 0,
        }
        return mask, debug

    @staticmethod
    def union_candidate_masks(candidates: list[dict[str, Any]], image_shape: tuple[int, int]) -> np.ndarray:
        h, w = image_shape
        union = np.zeros((h, w), dtype=np.uint8)
        for candidate in candidates:
            mask = np.asarray(candidate.get("mask"), dtype=np.uint8)
            if mask.shape != union.shape:
                continue
            union = cv2.bitwise_or(union, (mask > 0).astype(np.uint8) * 255)
        return union

    @staticmethod
    def plate_circle_fallback_mask(
        rgb_bgr: np.ndarray,
        search_mask: np.ndarray | None,
        cfg: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        h, w = rgb_bgr.shape[:2]
        gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
        search_u8 = None
        if search_mask is not None and np.asarray(search_mask).shape == (h, w) and int((search_mask > 0).sum()) > 0:
            search_u8 = (search_mask > 0).astype(np.uint8) * 255

        work = gray.copy()
        if search_u8 is not None:
            background_value = int(np.median(gray[search_u8 > 0])) if np.any(search_u8 > 0) else int(np.median(gray))
            work[search_u8 == 0] = background_value
        blur_ksize = int(cfg.get("plate_circle_blur_ksize", 5))
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        work = cv2.medianBlur(work, max(3, blur_ksize))

        min_radius = int(cfg.get("plate_circle_min_radius_px", 10))
        max_radius = int(cfg.get("plate_circle_max_radius_px", 55))
        dp = float(cfg.get("plate_circle_dp", 1.2))
        min_dist = float(cfg.get("plate_circle_min_dist_px", max(20, min_radius * 2)))
        param1 = float(cfg.get("plate_circle_param1", 80))
        param2 = float(cfg.get("plate_circle_param2", 18))
        circles = cv2.HoughCircles(
            work,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        edges = cv2.Canny(work, max(1.0, param1 * 0.45), param1)
        target_radius = float(cfg.get("plate_circle_target_radius_px", (min_radius + max_radius) * 0.5))
        candidates: list[dict[str, Any]] = []
        if circles is not None:
            for idx, circle in enumerate(np.round(circles[0]).astype(int)):
                cx, cy, radius = [int(v) for v in circle[:3]]
                if not (0 <= cx < w and 0 <= cy < h):
                    continue
                disk = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(disk, (cx, cy), radius, 255, -1)
                ring = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(ring, (cx, cy), radius, 255, max(2, int(round(radius * 0.12))))
                ring_pixels = ring > 0
                edge_support = float(np.mean(edges[ring_pixels] > 0)) if np.any(ring_pixels) else 0.0
                if search_u8 is not None:
                    center_in_search = bool(search_u8[cy, cx] > 0)
                    disk_overlap = float(np.mean(search_u8[disk > 0] > 0)) if np.any(disk > 0) else 0.0
                else:
                    center_in_search = True
                    disk_overlap = 1.0
                radius_penalty = abs(np.log(max(radius, 1) / max(target_radius, 1.0)))
                score = 3.0 * edge_support + 1.0 * disk_overlap - 0.35 * radius_penalty
                if not center_in_search:
                    score -= 1.0
                candidates.append(
                    {
                        "idx": idx,
                        "center_xy": [cx, cy],
                        "radius_px": radius,
                        "edge_support": edge_support,
                        "disk_overlap": disk_overlap,
                        "center_in_search": center_in_search,
                        "score": float(score),
                    }
                )

        min_score = float(cfg.get("plate_circle_min_score", 0.10))
        min_edge_support = float(cfg.get("plate_circle_min_edge_support", 0.015))
        min_disk_overlap = float(cfg.get("plate_circle_min_disk_overlap", 0.25))
        accepted = [
            item
            for item in candidates
            if item["score"] >= min_score
            and item["edge_support"] >= min_edge_support
            and item["disk_overlap"] >= min_disk_overlap
        ]
        if not accepted:
            debug = {
                "mode": "plate_circle_fallback",
                "selected": False,
                "candidates": candidates,
                "min_score": min_score,
                "min_edge_support": min_edge_support,
                "min_disk_overlap": min_disk_overlap,
            }
            raise ValueError(f"plate circle fallback found no acceptable circle: {debug}")

        best = max(accepted, key=lambda item: float(item["score"]))
        mask_scale = float(cfg.get("plate_circle_mask_scale", 1.08))
        mask_radius = max(1, int(round(float(best["radius_px"]) * mask_scale)))
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = [int(v) for v in best["center_xy"]]
        cv2.circle(mask, (cx, cy), mask_radius, 255, -1)
        debug = {
            "mode": "plate_circle_fallback",
            "selected": True,
            "selected_circle": best,
            "mask_radius_px": mask_radius,
            "candidates": candidates,
        }
        return mask, debug

    def select_mask_candidate(
        self,
        obj_key: str,
        candidates: list[dict[str, Any]],
        image_shape: tuple[int, int],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        h, w = image_shape
        cfg = self.filter_cfg(obj_key)
        rows: list[dict[str, Any]] = []
        for idx, candidate in enumerate(candidates):
            mask = np.asarray(candidate.get("mask"), dtype=np.uint8)
            stats = self.mask_candidate_stats(mask, image_shape)
            row = {
                "idx": idx,
                "box_xyxy": candidate.get("box_xyxy"),
                "box_confidence": candidate.get("box_confidence"),
                "sam_score": candidate.get("sam_score"),
                "latency_s": candidate.get("latency_s"),
                **stats,
            }
            reasons = self.reject_2d_candidate(row, cfg, image_shape)
            row["reject_reasons"] = reasons
            row["accepted"] = not reasons
            row["score"] = self.score_candidate(row, cfg, image_shape) if not reasons else None
            rows.append(row)

        accepted = [row for row in rows if row["accepted"]]
        if not accepted:
            empty = np.zeros((h, w), dtype=np.uint8)
            debug = {
                "mode": "dinosam_candidates",
                "selected": False,
                "candidate_count": len(candidates),
                "candidates": rows,
            }
            raise ValueError(f"{obj_key} no DINO-SAM candidate passed 2D filters: {debug}")

        best = max(accepted, key=lambda row: float(row["score"]))
        best_idx = int(best["idx"])
        selected = np.asarray(candidates[best_idx]["mask"], dtype=np.uint8)
        selected = (selected > 0).astype(np.uint8) * 255
        debug = {
            "mode": "dinosam_candidates",
            "selected": True,
            "selected_idx": best_idx,
            "candidate_count": len(candidates),
            "selected_score": best["score"],
            "selected_stats": {k: v for k, v in best.items() if k not in {"reject_reasons", "accepted", "score"}},
            "candidates": rows,
        }
        return selected, debug

    @staticmethod
    def mask_candidate_stats(mask: np.ndarray, image_shape: tuple[int, int]) -> dict[str, Any]:
        h, w = image_shape
        mask_u8 = (mask > 0).astype(np.uint8)
        area = int(mask_u8.sum())
        if area <= 0:
            return {
                "area_px": 0,
                "area_frac": 0.0,
                "bbox_xyxy": None,
                "bbox_area_px": 0,
                "bbox_area_frac": 0.0,
                "bbox_wh": None,
                "bbox_aspect": None,
                "fill_ratio": 0.0,
                "circularity": 0.0,
            }
        ys, xs = np.where(mask_u8 > 0)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        bw, bh = x2 - x1 + 1, y2 - y1 + 1
        bbox_area = int(bw * bh)
        contours, _hier = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
        circularity = 0.0 if perimeter <= 1e-9 else float(4.0 * np.pi * area / (perimeter * perimeter))
        return {
            "area_px": area,
            "area_frac": float(area / max(h * w, 1)),
            "bbox_xyxy": [x1, y1, x2, y2],
            "bbox_area_px": bbox_area,
            "bbox_area_frac": float(bbox_area / max(h * w, 1)),
            "bbox_wh": [bw, bh],
            "bbox_aspect": float(bw / max(bh, 1)),
            "fill_ratio": float(area / max(bbox_area, 1)),
            "circularity": circularity,
        }

    @staticmethod
    def reject_2d_candidate(row: dict[str, Any], cfg: dict[str, Any], image_shape: tuple[int, int]) -> list[str]:
        h, w = image_shape
        reasons: list[str] = []
        area = int(row.get("area_px") or 0)
        if area <= 0:
            return ["empty_mask"]
        min_area = cfg.get("min_mask_pixels")
        max_area = cfg.get("max_mask_pixels")
        if min_area is not None and area < int(min_area):
            reasons.append(f"area_px<{int(min_area)}")
        if max_area is not None and area > int(max_area):
            reasons.append(f"area_px>{int(max_area)}")

        bbox = row.get("bbox_xyxy")
        if bbox:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            bw, bh = x2 - x1 + 1, y2 - y1 + 1
            max_bbox_frac = cfg.get("max_bbox_area_frac")
            if max_bbox_frac is not None and row["bbox_area_frac"] > float(max_bbox_frac):
                reasons.append(f"bbox_area_frac>{float(max_bbox_frac)}")
            max_bbox_wh = cfg.get("max_bbox_wh")
            if max_bbox_wh is not None:
                max_w, max_h = [float(v) for v in max_bbox_wh]
                if bw > max_w:
                    reasons.append(f"bbox_w>{max_w:g}")
                if bh > max_h:
                    reasons.append(f"bbox_h>{max_h:g}")
            min_bbox_wh = cfg.get("min_bbox_wh")
            if min_bbox_wh is not None:
                min_w, min_h = [float(v) for v in min_bbox_wh]
                if bw < min_w:
                    reasons.append(f"bbox_w<{min_w:g}")
                if bh < min_h:
                    reasons.append(f"bbox_h<{min_h:g}")
            min_aspect = cfg.get("min_bbox_aspect")
            max_aspect = cfg.get("max_bbox_aspect")
            aspect = float(row.get("bbox_aspect") or 0.0)
            if min_aspect is not None and aspect < float(min_aspect):
                reasons.append(f"bbox_aspect<{float(min_aspect):g}")
            if max_aspect is not None and aspect > float(max_aspect):
                reasons.append(f"bbox_aspect>{float(max_aspect):g}")

            roi_norm = cfg.get("roi_norm")
            if roi_norm is not None:
                rx1, ry1, rx2, ry2 = [float(v) for v in roi_norm]
                roi = [rx1 * w, ry1 * h, rx2 * w, ry2 * h]
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                if not (roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]):
                    reasons.append("bbox_center_outside_roi")

        min_fill = cfg.get("min_fill_ratio")
        if min_fill is not None and float(row.get("fill_ratio") or 0.0) < float(min_fill):
            reasons.append(f"fill_ratio<{float(min_fill):g}")
        min_circularity = cfg.get("min_circularity")
        if min_circularity is not None and float(row.get("circularity") or 0.0) < float(min_circularity):
            reasons.append(f"circularity<{float(min_circularity):g}")
        return reasons

    @staticmethod
    def score_candidate(row: dict[str, Any], cfg: dict[str, Any], image_shape: tuple[int, int]) -> float:
        score = 0.0
        box_conf = row.get("box_confidence")
        sam_score = row.get("sam_score")
        if box_conf is not None:
            score += 2.0 * float(box_conf)
        if sam_score is not None:
            score += 1.0 * float(sam_score)
        score += 0.6 * float(row.get("circularity") or 0.0)
        score += 0.3 * float(row.get("fill_ratio") or 0.0)

        target_area = cfg.get("target_mask_pixels")
        if target_area is not None:
            area = max(float(row.get("area_px") or 1.0), 1.0)
            score -= abs(np.log(area / max(float(target_area), 1.0)))

        roi_norm = cfg.get("roi_norm")
        bbox = row.get("bbox_xyxy")
        if roi_norm is not None and bbox:
            h, w = image_shape
            rx1, ry1, rx2, ry2 = [float(v) for v in roi_norm]
            roi_center = np.array([(rx1 + rx2) * 0.5 * w, (ry1 + ry2) * 0.5 * h], dtype=np.float64)
            x1, y1, x2, y2 = [float(v) for v in bbox]
            center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)
            roi_diag = max(float(np.linalg.norm([(rx2 - rx1) * w, (ry2 - ry1) * h])), 1.0)
            score -= 0.5 * float(np.linalg.norm(center - roi_center) / roi_diag)
        return float(score)

    def validate_point_info(self, obj_key: str, point_info: dict[str, Any]) -> None:
        cfg = self.filter_cfg(obj_key)
        if not cfg:
            return
        reasons: list[str] = []
        raw_mask_pixels = int(point_info.get("raw_mask_pixels") or 0)
        valid_depth_pixels = int(point_info.get("valid_depth_pixels") or 0)
        max_raw_mask = cfg.get("max_raw_mask_pixels", cfg.get("max_mask_pixels"))
        if max_raw_mask is not None and raw_mask_pixels > int(max_raw_mask):
            reasons.append(f"raw_mask_pixels>{int(max_raw_mask)}")
        min_valid_depth = cfg.get("min_valid_depth_pixels")
        if min_valid_depth is not None and valid_depth_pixels < int(min_valid_depth):
            reasons.append(f"valid_depth_pixels<{int(min_valid_depth)}")
        min_valid_ratio = cfg.get("min_valid_depth_ratio")
        if min_valid_ratio is not None and float(point_info.get("valid_depth_ratio_raw_mask") or 0.0) < float(min_valid_ratio):
            reasons.append(f"valid_depth_ratio<{float(min_valid_ratio):g}")
        max_depth_range = cfg.get("max_depth_range_m")
        if max_depth_range is not None:
            depth_range = float(point_info.get("valid_depth_max_m") or 0.0) - float(point_info.get("valid_depth_min_m") or 0.0)
            if depth_range > float(max_depth_range):
                reasons.append(f"depth_range>{float(max_depth_range):g}")
        max_extent = cfg.get("max_extent_m")
        if max_extent is not None:
            extent = np.asarray(point_info.get("points_extent_m") or [], dtype=np.float64).reshape(-1)
            limit = np.asarray(max_extent, dtype=np.float64).reshape(-1)
            if extent.size == 3 and limit.size == 3:
                for axis, value, max_value in zip("xyz", extent, limit):
                    if float(value) > float(max_value):
                        reasons.append(f"extent_{axis}>{float(max_value):g}")
        if reasons:
            raise ValueError(f"{obj_key} failed RGB-D object filters: {', '.join(reasons)}")

    def mask_to_points(
        self,
        depth_m: np.ndarray,
        K: np.ndarray,
        mask: np.ndarray,
        return_info: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        if depth_m is None:
            raise ValueError("depth_m is required for RGB-D object pose estimation")
        if mask is None:
            raise ValueError("DINO-SAM returned no mask")

        depth = np.asarray(depth_m, dtype=np.float64)
        mask_u8 = (mask > 0).astype(np.uint8)
        raw_mask_u8 = mask_u8.copy()
        stage = "raw"
        if self.cfg.mask_erode_px > 0:
            k = 2 * self.cfg.mask_erode_px + 1
            kernel = np.ones((k, k), np.uint8)
            mask_u8 = cv2.erode(mask_u8, kernel, iterations=1)
            stage = "eroded"

        valid = self._valid_depth_mask(depth, mask_u8)
        if self.cfg.depth_window_px > 0:
            valid = self._fill_valid_from_neighborhood(depth, mask_u8, valid, self.cfg.depth_window_px)
            stage = f"{stage}_depth_window"

        if valid.sum() < self.cfg.min_points and self.cfg.mask_erode_px > 0:
            mask_u8 = (mask > 0).astype(np.uint8)
            valid = self._valid_depth_mask(depth, mask_u8)
            if self.cfg.depth_window_px > 0:
                valid = self._fill_valid_from_neighborhood(depth, mask_u8, valid, self.cfg.depth_window_px)
            stage = "raw_retry"

        if valid.sum() < self.cfg.min_points and self.cfg.fallback_dilate_px > 0:
            k = 2 * self.cfg.fallback_dilate_px + 1
            kernel = np.ones((k, k), np.uint8)
            dilated = cv2.dilate((mask > 0).astype(np.uint8), kernel, iterations=1)
            valid = self._valid_depth_mask(depth, dilated)
            if self.cfg.fallback_depth_window_px > 0:
                valid = self._fill_valid_from_neighborhood(depth, dilated, valid, self.cfg.fallback_depth_window_px)
            mask_u8 = dilated
            stage = "dilated_fallback"

        vs, us = np.where(valid)
        if len(us) < self.cfg.min_points:
            raise ValueError(f"Only {len(us)} valid depth points inside object mask")

        depth_values = depth[vs, us]
        info: dict[str, Any] = {
            "stage": stage,
            "raw_mask_pixels": int(raw_mask_u8.sum()),
            "used_mask_pixels": int(mask_u8.sum()),
            "valid_depth_pixels": int(len(us)),
            "valid_depth_ratio_raw_mask": float(len(us) / max(int(raw_mask_u8.sum()), 1)),
            "valid_depth_min_m": float(np.min(depth_values)),
            "valid_depth_median_m": float(np.median(depth_values)),
            "valid_depth_max_m": float(np.max(depth_values)),
            "valid_uv_bbox": [int(np.min(us)), int(np.min(vs)), int(np.max(us)), int(np.max(vs))],
            "_valid_mask": valid.copy(),
            "_used_mask": mask_u8.copy(),
        }

        if len(us) > self.cfg.max_points:
            idx = np.linspace(0, len(us) - 1, self.cfg.max_points).astype(np.int64)
            us, vs = us[idx], vs[idx]

        z = depth[vs, us]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        x = (us.astype(np.float64) - cx) * z / fx
        y = (vs.astype(np.float64) - cy) * z / fy
        pts = np.stack([x, y, z], axis=1)
        info.update(
            {
                "sampled_points": int(len(pts)),
                "points_center_m": pts.mean(axis=0).tolist(),
                "points_min_m": pts.min(axis=0).tolist(),
                "points_max_m": pts.max(axis=0).tolist(),
                "points_extent_m": (pts.max(axis=0) - pts.min(axis=0)).tolist(),
            }
        )
        if return_info:
            return pts, info
        return pts

    def _valid_depth_mask(self, depth: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        return (
            (mask_u8 > 0)
            & np.isfinite(depth)
            & (depth >= self.cfg.min_valid_depth_m)
            & (depth <= self.cfg.max_valid_depth_m)
        )

    def _fill_valid_from_neighborhood(
        self, depth: np.ndarray, mask_u8: np.ndarray, valid: np.ndarray, radius: int
    ) -> np.ndarray:
        if valid.sum() >= self.cfg.min_points:
            return valid
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

    def save_debug_images(
        self,
        debug_dir: Path,
        obj_key: str,
        rgb_bgr: np.ndarray,
        depth_m: np.ndarray,
        K: np.ndarray,
        mask: np.ndarray,
        point_info: dict[str, Any],
        T_obj_in_cam: np.ndarray,
        selection_debug: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        files: dict[str, str] = {}
        mask_u8 = (mask > 0).astype(np.uint8) * 255

        mask_path = debug_dir / f"{obj_key}_mask.png"
        cv2.imwrite(str(mask_path), mask_u8)
        files["mask"] = str(mask_path)

        mask_overlay = rgb_bgr.copy()
        mask_color = np.zeros_like(mask_overlay)
        mask_color[:, :, 1] = mask_u8
        mask_overlay = cv2.addWeighted(mask_overlay, 0.65, mask_color, 0.35, 0.0)
        bbox = point_info.get("valid_uv_bbox")
        if bbox:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(mask_overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        uv = self.project_point(K, T_obj_in_cam[:3, 3])
        if uv is not None:
            cv2.drawMarker(mask_overlay, uv, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=22, thickness=2)
            cv2.putText(mask_overlay, obj_key, (uv[0] + 8, uv[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        mask_overlay_path = debug_dir / f"{obj_key}_mask_overlay.jpg"
        cv2.imwrite(str(mask_overlay_path), mask_overlay)
        files["mask_overlay"] = str(mask_overlay_path)

        if selection_debug and selection_debug.get("mode") == "plate_circle_fallback":
            circle_vis = rgb_bgr.copy()
            for cand in selection_debug.get("candidates") or []:
                center = cand.get("center_xy")
                radius = cand.get("radius_px")
                if center is None or radius is None:
                    continue
                cx, cy = [int(v) for v in center]
                radius = int(radius)
                cv2.circle(circle_vis, (cx, cy), radius, (0, 200, 255), 1, lineType=cv2.LINE_AA)
                cv2.putText(
                    circle_vis,
                    f"{cand.get('score', 0.0):.2f}",
                    (cx + 4, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 200, 255),
                    1,
                    cv2.LINE_AA,
                )
            selected = selection_debug.get("selected_circle") or {}
            center = selected.get("center_xy")
            radius = selected.get("radius_px")
            if center is not None and radius is not None:
                cx, cy = [int(v) for v in center]
                cv2.circle(circle_vis, (cx, cy), int(radius), (0, 0, 255), 2, lineType=cv2.LINE_AA)
                cv2.drawMarker(circle_vis, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            circle_path = debug_dir / f"{obj_key}_circle_candidates.jpg"
            cv2.imwrite(str(circle_path), circle_vis)
            files["circle_candidates"] = str(circle_path)

        valid = point_info.get("_valid_mask")
        if isinstance(valid, np.ndarray):
            valid_overlay = rgb_bgr.copy()
            valid_color = np.zeros_like(valid_overlay)
            valid_color[:, :, 2] = valid.astype(np.uint8) * 255
            valid_overlay = cv2.addWeighted(valid_overlay, 0.65, valid_color, 0.35, 0.0)
            valid_overlay_path = debug_dir / f"{obj_key}_valid_depth_overlay.jpg"
            cv2.imwrite(str(valid_overlay_path), valid_overlay)
            files["valid_depth_overlay"] = str(valid_overlay_path)

        point_debug_path = debug_dir / f"{obj_key}_point_debug.json"
        import json

        point_debug_path.write_text(
            json.dumps({k: v for k, v in point_info.items() if not k.startswith("_")}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        files["point_debug"] = str(point_debug_path)
        return files

    @staticmethod
    def project_point(K: np.ndarray, point_cam: np.ndarray) -> tuple[int, int] | None:
        x, y, z = [float(v) for v in point_cam[:3]]
        if not np.isfinite([x, y, z]).all() or z <= 1e-9:
            return None
        u = float(K[0, 0]) * x / z + float(K[0, 2])
        v = float(K[1, 1]) * y / z + float(K[1, 2])
        if not np.isfinite([u, v]).all():
            return None
        return int(round(u)), int(round(v))

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
