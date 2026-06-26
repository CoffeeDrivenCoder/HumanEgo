#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HTTP inference server for HumanEgo on G1.

The server owns the heavy models. The robot client sends RGB + current G1 state,
then receives a converted arm_right_link7 target. No robot SDK is imported here.
"""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, PROJECT_ROOT / "inference", SCRIPTS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from g1_humanego_dry_run import (  # noqa: E402
    build_target_preview,
    choose_device,
    json_safe,
    load_cfg,
    load_fixed_objects,
    matrix_json,
    resolve_project_path,
)
from interfaces import ObjectState  # noqa: E402
from interfaces import Frame  # noqa: E402
from policy import ICTPolicy  # noqa: E402
from g1_artifacts import artifact_dir  # noqa: E402


DEFAULT_CFG = PROJECT_ROOT / "cfg" / "inference" / "g1_serve_bread_right.yaml"
DEFAULT_OUT_DIR = artifact_dir("server")


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def decode_rgb_jpeg(payload: dict[str, Any]) -> np.ndarray:
    encoded = payload.get("rgb_jpeg_b64")
    if not encoded:
        raise ValueError("request missing rgb_jpeg_b64")
    data = base64.b64decode(encoded)
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode rgb_jpeg_b64")
    return image


def decode_depth_npz(payload: dict[str, Any]) -> tuple[np.ndarray | None, dict[str, Any]]:
    encoded = payload.get("depth_m_npz_b64")
    if not encoded:
        return None, {"present": False}
    raw = base64.b64decode(encoded)
    with np.load(BytesIO(raw)) as data:
        depth = np.asarray(data["depth"])
    info = payload.get("depth_encoding") or {}
    encoding = info.get("encoding", "z16")
    if encoding == "z16":
        depth_m = depth.astype(np.float32) / 1000.0
    elif encoding in {"float16", "float32"}:
        depth_m = depth.astype(np.float32)
    else:
        raise ValueError(f"unsupported depth encoding from client: {encoding}")
    return depth_m, {
        "present": True,
        "encoding": encoding,
        "shape": list(depth_m.shape),
        "raw_npz_bytes": len(raw),
    }


def objects_from_payload(payload: dict[str, Any]) -> dict[str, ObjectState] | None:
    raw = payload.get("objects")
    if not raw:
        return None
    objects: dict[str, ObjectState] = {}
    for key, item in raw.items():
        T = np.asarray(item["T_in_cam"], dtype=np.float32).reshape(4, 4)
        kpts = np.asarray(item.get("kpts_local", []), dtype=np.float32)
        if kpts.size == 0:
            kpts = np.zeros((0, 3), dtype=np.float32)
        objects[key] = ObjectState(T_in_cam=T, kpts_local=kpts.reshape(-1, 3))
    return objects


def read_matrix(mapping: dict[str, Any], key: str) -> np.ndarray:
    if key not in mapping:
        raise ValueError(f"request.current missing {key}")
    return np.asarray(mapping[key], dtype=np.float64).reshape(4, 4)


def matrix_from(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return None
    if arr.size != 16:
        return None
    return arr.reshape(4, 4)


def project_point(K: np.ndarray, point_cam: np.ndarray) -> tuple[int, int] | None:
    x, y, z = [float(v) for v in point_cam[:3]]
    if not np.isfinite([x, y, z]).all() or z <= 1e-9:
        return None
    u = float(K[0, 0]) * x / z + float(K[0, 2])
    v = float(K[1, 1]) * y / z + float(K[1, 2])
    if not np.isfinite([u, v]).all():
        return None
    return int(round(u)), int(round(v))


def draw_text(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.45,
) -> None:
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_marker(
    image: np.ndarray,
    uv: tuple[int, int],
    label: str,
    color: tuple[int, int, int],
    radius: int = 7,
) -> None:
    h, w = image.shape[:2]
    u, v = uv
    u_clip = int(np.clip(u, 0, w - 1))
    v_clip = int(np.clip(v, 0, h - 1))
    in_view = 0 <= u < w and 0 <= v < h
    cv2.circle(image, (u_clip, v_clip), radius, color, 2, cv2.LINE_AA)
    cv2.drawMarker(image, (u_clip, v_clip), color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    suffix = "" if in_view else " off"
    draw_text(image, f"{label}{suffix}", (min(u_clip + 9, w - 120), max(v_clip - 8, 16)), color)


def first_right_preview(response: dict[str, Any]) -> dict[str, Any]:
    return ((((response.get("policy_preview") or {}).get("sides") or {}).get("right") or [{}])[0]) or {}


def save_depth_colormap(depth_m: np.ndarray | None, run_dir: Path) -> str | None:
    if depth_m is None:
        return None
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return None
    lo, hi = np.percentile(depth[valid], [2, 98])
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        lo, hi = float(np.min(depth[valid])), float(np.max(depth[valid]))
    norm = np.zeros(depth.shape, dtype=np.uint8)
    norm[valid] = np.clip((depth[valid] - lo) / max(float(hi - lo), 1e-6) * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    color[~valid] = (0, 0, 0)
    draw_text(color, f"depth_m p02={lo:.3f} p98={hi:.3f}", (10, 22), (255, 255, 255), 0.55)
    out = run_dir / "depth_colormap.jpg"
    cv2.imwrite(str(out), color)
    return str(out)


def collect_projection_poses(response: dict[str, Any]) -> dict[str, np.ndarray]:
    input_summary = response.get("input_summary") or {}
    poses: dict[str, np.ndarray] = {}
    for key, item in (input_summary.get("objects") or {}).items():
        T = matrix_from((item or {}).get("T_in_cam"))
        if T is not None:
            poses[str(key)] = T
    T_current = matrix_from(input_summary.get("current_T_tcp_in_cam"))
    if T_current is not None:
        poses["tcp"] = T_current
    T_target = matrix_from(first_right_preview(response).get("T_tcp_target_in_cam"))
    if T_target is not None:
        poses["target"] = T_target
    return poses


def save_projection_layers(response: dict[str, Any], rgb_bgr: np.ndarray, run_dir: Path) -> dict[str, str]:
    input_summary = response.get("input_summary") or {}
    K = np.asarray(input_summary.get("K"), dtype=np.float64).reshape(3, 3)
    poses = collect_projection_poses(response)
    colors = {
        "obj1": (30, 220, 30),
        "obj2": (40, 170, 255),
        "tcp": (255, 220, 40),
        "target": (255, 40, 220),
    }
    layer_labels = {
        "objects": {"obj1", "obj2"},
        "tcp": {"tcp", "target"},
        "all": {"obj1", "obj2", "tcp", "target"},
    }
    files: dict[str, str] = {}
    for layer, labels in layer_labels.items():
        image = rgb_bgr.copy()
        projected: dict[str, tuple[int, int]] = {}
        for label, T in poses.items():
            if label not in labels:
                continue
            uv = project_point(K, T[:3, 3])
            if uv is None:
                continue
            projected[label] = uv
            draw_marker(image, uv, label, colors.get(label, (255, 255, 255)))
        if {"tcp", "target"}.issubset(projected):
            cv2.arrowedLine(
                image,
                projected["tcp"],
                projected["target"],
                colors["target"],
                2,
                cv2.LINE_AA,
                tipLength=0.12,
            )
        draw_text(image, f"projection_{layer} | {run_dir.name}", (10, 22), (255, 255, 255), 0.55)
        out = run_dir / f"vision_projection_{layer}.jpg"
        cv2.imwrite(str(out), image)
        files[layer] = str(out)
    return files


def object_warning_summary(response: dict[str, Any]) -> dict[str, Any]:
    object_debug = ((response.get("input_summary") or {}).get("object_debug") or {}).get("objects") or {}
    objects = (response.get("input_summary") or {}).get("objects") or {}
    summary: dict[str, Any] = {}
    for key, item in object_debug.items():
        points = item.get("points") or {}
        T = matrix_from((objects.get(key) or {}).get("T_in_cam"))
        extent = [float(v) for v in points.get("points_extent_m") or []]
        depth_min = points.get("valid_depth_min_m")
        depth_max = points.get("valid_depth_max_m")
        depth_range = None
        if depth_min is not None and depth_max is not None:
            depth_range = float(depth_max) - float(depth_min)
        warnings: list[str] = []
        if depth_range is not None and depth_range > 0.45:
            warnings.append(f"large_depth_range:{depth_range:.3f}m")
        if extent and max(extent) > 0.45:
            warnings.append(f"large_extent:{max(extent):.3f}m")
        valid_ratio = float(points.get("valid_depth_ratio_raw_mask") or 0.0)
        if valid_ratio < 0.75:
            warnings.append(f"low_valid_depth_ratio:{valid_ratio:.3f}")
        summary[key] = {
            "prompt": item.get("prompt"),
            "xyz_cam_m": None if T is None else [float(v) for v in T[:3, 3]],
            "mask_pixels": points.get("raw_mask_pixels"),
            "valid_depth_pixels": points.get("valid_depth_pixels"),
            "valid_depth_ratio": valid_ratio,
            "depth_min_m": depth_min,
            "depth_median_m": points.get("valid_depth_median_m"),
            "depth_max_m": depth_max,
            "depth_range_m": depth_range,
            "extent_m": extent,
            "valid_uv_bbox": points.get("valid_uv_bbox"),
            "warnings": warnings,
        }
    return summary


def make_contact_sheet(panels: list[tuple[str, np.ndarray]], out_path: Path, cell_w: int = 360) -> str | None:
    if not panels:
        return None
    prepared = []
    for title, image in panels:
        if image is None:
            continue
        h, w = image.shape[:2]
        scale = cell_w / max(float(w), 1.0)
        cell_h = max(1, int(round(h * scale)))
        resized = cv2.resize(image, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        header = np.zeros((28, cell_w, 3), dtype=np.uint8)
        draw_text(header, title[:44], (8, 19), (255, 255, 255), 0.48)
        prepared.append(np.vstack([header, resized]))
    if not prepared:
        return None
    cell_h = max(img.shape[0] for img in prepared)
    padded = []
    for img in prepared:
        if img.shape[0] < cell_h:
            pad = np.zeros((cell_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
            img = np.vstack([img, pad])
        padded.append(img)
    cols = 2
    rows = []
    for i in range(0, len(padded), cols):
        row_items = padded[i : i + cols]
        if len(row_items) < cols:
            row_items.append(np.zeros_like(padded[0]))
        rows.append(np.hstack(row_items))
    sheet = np.vstack(rows)
    cv2.imwrite(str(out_path), sheet)
    return str(out_path)


def save_vision_diagnostics(
    response: dict[str, Any],
    rgb_bgr: np.ndarray,
    depth_m: np.ndarray | None,
    run_dir: Path,
) -> dict[str, Any]:
    files: dict[str, Any] = {}
    projection_files = save_projection_layers(response, rgb_bgr, run_dir)
    files["projection"] = projection_files
    depth_path = save_depth_colormap(depth_m, run_dir)
    if depth_path:
        files["depth_colormap"] = depth_path

    panels: list[tuple[str, np.ndarray]] = [("rgb", rgb_bgr)]
    for label, path in projection_files.items():
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is not None:
            panels.append((f"projection_{label}", image))
    if depth_path:
        depth_img = cv2.imread(depth_path, cv2.IMREAD_COLOR)
        if depth_img is not None:
            panels.append(("depth_colormap", depth_img))
    object_debug = ((response.get("input_summary") or {}).get("object_debug") or {}).get("objects") or {}
    for obj_key in sorted(object_debug.keys()):
        saved = object_debug[obj_key].get("saved_files") or {}
        for kind in ("mask_overlay", "valid_depth_overlay", "circle_candidates"):
            path = saved.get(kind)
            if not path:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None:
                panels.append((f"{obj_key}_{kind}", image))
    contact = make_contact_sheet(panels, run_dir / "vision_contact_sheet.jpg")
    if contact:
        files["contact_sheet"] = contact

    summary = {
        "object_source_used": (response.get("input_summary") or {}).get("object_source_used"),
        "object_error": (response.get("input_summary") or {}).get("object_error"),
        "objects": object_warning_summary(response),
        "files": files,
    }
    summary_path = run_dir / "vision_summary.json"
    summary_path.write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    summary["files"]["summary"] = str(summary_path)
    return summary


def safe_request_id(value: Any) -> str:
    request_id = str(value or utc_stamp())
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in request_id)


@dataclass
class InferenceRuntime:
    cfg_path: Path
    out_dir: Path
    device: str
    save_rgb: bool
    object_source_override: str

    def __post_init__(self) -> None:
        self.cfg = load_cfg(self.cfg_path)
        self.policy = ICTPolicy(self.cfg["policy"], device=self.device)
        self.perception_cfg = self.cfg.get("perception", {})
        self.anchor_key = self.perception_cfg.get("anchor_key", "obj1")
        self.object_source = str(
            self.object_source_override
            or self.perception_cfg.get("object_source", self.perception_cfg.get("dry_run_object_source", "fixed"))
        ).lower()
        self.allow_fixed_fallback = bool(self.perception_cfg.get("allow_fixed_object_fallback", True))
        self.fixed_objects = load_fixed_objects(self.perception_cfg)
        self.object_pose_estimator = None
        if self.object_source == "rgbd":
            from object_pose_rgbd import RGBDObjectPoseEstimator

            self.object_pose_estimator = RGBDObjectPoseEstimator(self.perception_cfg)
        self.T_align = np.asarray(self.cfg["robot"]["T_align"], dtype=np.float64).reshape(4, 4)
        self.max_step_m = float(self.cfg.get("control", {}).get("max_pos_step", 0.03))
        self.preview_steps = int(self.cfg.get("server", {}).get("preview_steps", 3))
        self.lock = threading.Lock()
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        safe_id = safe_request_id(payload.get("request_id"))
        run_dir = self.out_dir / safe_id
        run_dir.mkdir(parents=True, exist_ok=True)
        rgb_bgr = decode_rgb_jpeg(payload)
        depth_m, depth_info = decode_depth_npz(payload)
        h, w = rgb_bgr.shape[:2]
        K = np.asarray(payload["K"], dtype=np.float64).reshape(3, 3)
        current = payload.get("current") or {}

        T_base_camera = read_matrix(current, "T_base_camera")
        T_link7_in_base = read_matrix(current, "T_link7_in_base")
        T_tcp_in_link7 = read_matrix(current, "T_tcp_in_link7")
        T_tcp_in_cam = read_matrix(current, "T_tcp_in_cam")
        T_hand_in_cam = T_tcp_in_cam @ self.T_align
        gripper = float(current.get("gripper", 0.0))

        object_source_used = "payload"
        object_error = None
        object_debug = None
        objects = objects_from_payload(payload)
        if objects is None and self.object_source == "rgbd":
            object_source_used = "rgbd"
            try:
                if depth_m is None:
                    raise ValueError("object_source=rgbd requires depth_m_npz_b64 from client")
                assert self.object_pose_estimator is not None
                frame = Frame(rgb=rgb_bgr, depth_m=depth_m.astype(np.float32), K=K.astype(np.float32))
                objects, object_debug = self.object_pose_estimator.estimate_with_debug(
                    [frame],
                    debug_dir=run_dir / "object_debug",
                )
            except Exception as exc:
                object_error = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                if not self.allow_fixed_fallback:
                    raise
                objects = self.fixed_objects
                object_source_used = "fixed_fallback_after_rgbd_error"
        elif objects is None:
            objects = self.fixed_objects
            object_source_used = "fixed"

        anchor = objects.get(self.anchor_key)
        if anchor is None:
            raise ValueError(f"anchor_key {self.anchor_key!r} not found in objects {sorted(objects.keys())}")

        with self.lock:
            x_rgb = self.policy.prepare_image(rgb_bgr)
            x_ict, ict_mask = self.policy.build_ict({"right": T_hand_in_cam}, {"right": gripper}, objects, self.anchor_key)
            anchor_uv = self.policy.compute_anchor_uv(anchor, K, w, h)
            traj, done_prob = self.policy.infer(x_rgb, x_ict, ict_mask, anchor_uv)

        preview = build_target_preview(
            traj=traj,
            done_prob=done_prob,
            policy=self.policy,
            anchor=anchor,
            T_align=self.T_align,
            T_base_camera=T_base_camera,
            T_tcp_in_link7=T_tcp_in_link7,
            T_link7_current_in_base=T_link7_in_base,
            max_step_m=self.max_step_m,
            max_steps=int(payload.get("preview_steps", self.preview_steps)),
        )

        response = {
            "ok": True,
            "server_time_utc": datetime.now(timezone.utc).isoformat(),
            "request_id": payload.get("request_id"),
            "control_command": None,
            "control_sent": False,
            "policy": {
                "sides": self.policy.sides,
                "frame_mode": self.policy.frame_mode,
                "action_mode": self.policy.action_mode,
                "pred_horizon": self.policy.pred_horizon,
                "use_region_attn": self.policy.use_region_attn,
            },
            "input_summary": {
                "rgb_shape": list(rgb_bgr.shape),
                "depth": depth_info,
                "K": K.tolist(),
                "current_T_tcp_in_cam": matrix_json(T_tcp_in_cam),
                "current_T_hand_in_cam": matrix_json(T_hand_in_cam),
                "gripper": gripper,
                "objects": {
                    key: {
                        "T_in_cam": matrix_json(obj.T_in_cam),
                        "kpts_local_count": int(len(obj.kpts_local)),
                    }
                    for key, obj in objects.items()
                },
                "object_source_used": object_source_used,
                "object_error": object_error,
                "object_debug": object_debug,
            },
            "policy_preview": preview,
            "latency_s": time.time() - started,
        }
        try:
            response["vision_summary"] = save_vision_diagnostics(response, rgb_bgr, depth_m, run_dir)
        except Exception as exc:
            response["vision_summary"] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        self._log_request(payload, response, rgb_bgr, run_dir=run_dir)
        return response

    def _log_request(
        self,
        payload: dict[str, Any],
        response: dict[str, Any],
        rgb_bgr: np.ndarray,
        run_dir: Path,
    ) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        summary = dict(payload)
        summary.pop("rgb_jpeg_b64", None)
        summary.pop("depth_m_npz_b64", None)
        (run_dir / "request_summary.json").write_text(
            json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (run_dir / "response.json").write_text(
            json.dumps(json_safe(response), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if self.save_rgb:
            cv2.imwrite(str(run_dir / "rgb_bgr.jpg"), rgb_bgr)


def make_handler(runtime: InferenceRuntime):
    class Handler(BaseHTTPRequestHandler):
        server_version = "G1HumanEgoInferenceServer/1.0"

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(json_safe(payload), ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path not in {"/", "/health"}:
                self._send_json(404, {"ok": False, "error": "use GET /health or POST /infer"})
                return
            self._send_json(
                200,
                {
                    "ok": True,
                    "message": "POST G1 state to /infer",
                    "cfg_path": str(runtime.cfg_path),
                    "device": runtime.device,
                    "object_source": runtime.object_source,
                    "policy_sides": runtime.policy.sides,
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/infer":
                self._send_json(404, {"ok": False, "error": "use POST /infer"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    raise ValueError("empty request body")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                response = runtime.infer(payload)
                self._send_json(200, response)
            except Exception as exc:
                self._send_json(
                    500,
                    {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--cfg", default=str(DEFAULT_CFG))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--object-source", default="", choices=["", "fixed", "rgbd"])
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--save-rgb", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    cfg_path = resolve_project_path(args.cfg)
    device = choose_device(args.device)
    runtime = InferenceRuntime(
        cfg_path=cfg_path,
        out_dir=Path(args.out_dir).expanduser().resolve(),
        device=device,
        save_rgb=bool(args.save_rgb),
        object_source_override=args.object_source,
    )
    server = ReusableThreadingHTTPServer((args.host, args.port), make_handler(runtime))
    print(f"Listening on http://{args.host}:{args.port}/infer", flush=True)
    print(f"Config: {cfg_path}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Object source: {runtime.object_source}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping HumanEgo inference server.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
