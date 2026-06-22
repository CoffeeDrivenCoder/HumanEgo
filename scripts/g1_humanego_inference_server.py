#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HTTP inference server for HumanEgo on G1.

The server owns the heavy models. The robot client sends RGB + current G1 state,
then receives a converted arm_right_link7 target. No robot SDK is imported here.
"""

from __future__ import annotations

import argparse
import base64
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
from policy import ICTPolicy  # noqa: E402


DEFAULT_CFG = PROJECT_ROOT / "cfg" / "inference" / "g1_serve_bread_right.yaml"


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


@dataclass
class InferenceRuntime:
    cfg_path: Path
    out_dir: Path
    device: str
    save_rgb: bool

    def __post_init__(self) -> None:
        self.cfg = load_cfg(self.cfg_path)
        self.policy = ICTPolicy(self.cfg["policy"], device=self.device)
        self.anchor_key = self.cfg.get("perception", {}).get("anchor_key", "obj1")
        self.fixed_objects = load_fixed_objects(self.cfg.get("perception", {}))
        self.T_align = np.asarray(self.cfg["robot"]["T_align"], dtype=np.float64).reshape(4, 4)
        self.max_step_m = float(self.cfg.get("control", {}).get("max_pos_step", 0.03))
        self.preview_steps = int(self.cfg.get("server", {}).get("preview_steps", 3))
        self.lock = threading.Lock()
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        rgb_bgr = decode_rgb_jpeg(payload)
        h, w = rgb_bgr.shape[:2]
        K = np.asarray(payload["K"], dtype=np.float64).reshape(3, 3)
        current = payload.get("current") or {}

        T_base_camera = read_matrix(current, "T_base_camera")
        T_link7_in_base = read_matrix(current, "T_link7_in_base")
        T_tcp_in_link7 = read_matrix(current, "T_tcp_in_link7")
        T_tcp_in_cam = read_matrix(current, "T_tcp_in_cam")
        T_hand_in_cam = T_tcp_in_cam @ self.T_align
        gripper = float(current.get("gripper", 0.0))

        objects = objects_from_payload(payload) or self.fixed_objects
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
            },
            "policy_preview": preview,
            "latency_s": time.time() - started,
        }
        self._log_request(payload, response, rgb_bgr)
        return response

    def _log_request(self, payload: dict[str, Any], response: dict[str, Any], rgb_bgr: np.ndarray) -> None:
        request_id = str(payload.get("request_id") or utc_stamp())
        safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in request_id)
        run_dir = self.out_dir / safe_id
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
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "g1_humanego_server_runs"))
    parser.add_argument("--save-rgb", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    cfg_path = resolve_project_path(args.cfg)
    device = choose_device(args.device)
    runtime = InferenceRuntime(
        cfg_path=cfg_path,
        out_dir=Path(args.out_dir).expanduser().resolve(),
        device=device,
        save_rgb=bool(args.save_rgb),
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime))
    print(f"Listening on http://{args.host}:{args.port}/infer", flush=True)
    print(f"Config: {cfg_path}", flush=True)
    print(f"Device: {device}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping HumanEgo inference server.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
