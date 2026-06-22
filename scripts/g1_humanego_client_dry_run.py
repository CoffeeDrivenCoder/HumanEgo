#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G1 robot-side client for server-run HumanEgo inference.

Robot side responsibilities:
  * read RGB-D/camera intrinsics
  * read current link7/TCP/base-camera state
  * POST compact state to the server
  * save the returned target preview

This dry-run client sends no robot control commands.
"""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
import os
import sys
import time
import traceback
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = PROJECT_ROOT / "cfg" / "inference" / "g1_serve_bread_right.yaml"

for path in (PROJECT_ROOT, PROJECT_ROOT / "inference"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def log(message: str) -> None:
    print(f"[g1_humanego_client] {message}", flush=True)


def resolve_project_path(path: str | os.PathLike[str]) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    for base in (Path.cwd(), PROJECT_ROOT):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (PROJECT_ROOT / path).resolve()


def load_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def make_zip(src_dir: Path) -> Path:
    zip_path = src_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir.parent))
    return zip_path


def post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    data = json.dumps(json_safe(payload), ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Content-Length": str(len(data)),
            "Connection": "close",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return {"ok": True, "status": resp.status, "num_bytes": len(body), "json": json.loads(body)}


def upload_zip(zip_path: Path, upload_url: str, timeout_s: float = 60.0) -> Dict[str, Any]:
    data = zip_path.read_bytes()
    req = urllib.request.Request(
        upload_url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/zip",
            "Content-Length": str(len(data)),
            "X-G1-Diagnostics-Filename": zip_path.name,
            "Connection": "close",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return {"ok": True, "status": resp.status, "response": resp.read().decode("utf-8", errors="replace")}


def encode_jpeg_b64(image_bgr: np.ndarray, quality: int) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("failed to JPEG-encode RGB image")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def encode_depth_npz_b64(depth_m: np.ndarray, encoding: str = "z16") -> tuple[str, dict[str, Any]]:
    depth = np.asarray(depth_m)
    bio = BytesIO()
    if encoding == "z16":
        depth_mm = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        depth_mm = np.clip(depth_mm * 1000.0, 0.0, 65535.0).astype(np.uint16)
        np.savez_compressed(bio, depth=depth_mm)
        info = {"encoding": "z16", "unit": "mm", "shape": list(depth_mm.shape), "dtype": str(depth_mm.dtype)}
    elif encoding == "float16":
        depth_f16 = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)
        np.savez_compressed(bio, depth=depth_f16)
        info = {"encoding": "float16", "unit": "m", "shape": list(depth_f16.shape), "dtype": str(depth_f16.dtype)}
    elif encoding == "float32":
        depth_f32 = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        np.savez_compressed(bio, depth=depth_f32)
        info = {"encoding": "float32", "unit": "m", "shape": list(depth_f32.shape), "dtype": str(depth_f32.dtype)}
    else:
        raise ValueError(f"unsupported depth encoding: {encoding}")
    raw = bio.getvalue()
    return base64.b64encode(raw).decode("ascii"), {**info, "npz_bytes": len(raw)}


def resize_depth_to_shape(depth_m: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.shape[:2] == (target_h, target_w):
        return depth
    return cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def resize_image_and_K(
    image_bgr: np.ndarray,
    K: np.ndarray,
    target_width: int,
    target_height: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    src_h, src_w = image_bgr.shape[:2]
    target_width = int(target_width)
    target_height = int(target_height)
    if target_width <= 0 and target_height <= 0:
        return image_bgr, np.asarray(K, dtype=np.float64).copy(), {
            "resized": False,
            "source_shape": list(image_bgr.shape),
            "sent_shape": list(image_bgr.shape),
            "scale_x": 1.0,
            "scale_y": 1.0,
        }
    if target_width <= 0:
        target_width = max(1, int(round(src_w * (target_height / float(src_h)))))
    if target_height <= 0:
        target_height = max(1, int(round(src_h * (target_width / float(src_w)))))

    scale_x = target_width / float(src_w)
    scale_y = target_height / float(src_h)
    K_send = np.asarray(K, dtype=np.float64).copy()
    K_send[0, 0] *= scale_x
    K_send[0, 2] *= scale_x
    K_send[1, 1] *= scale_y
    K_send[1, 2] *= scale_y

    interpolation = cv2.INTER_AREA if target_width < src_w or target_height < src_h else cv2.INTER_LINEAR
    image_send = cv2.resize(image_bgr, (target_width, target_height), interpolation=interpolation)
    return image_send, K_send, {
        "resized": True,
        "source_shape": list(image_bgr.shape),
        "sent_shape": list(image_send.shape),
        "scale_x": scale_x,
        "scale_y": scale_y,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default=str(DEFAULT_CFG))
    parser.add_argument("--server-url", default="http://111.0.22.33:30003/infer")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "g1_humanego_client_runs"))
    parser.add_argument("--tag", default="client_dry_run")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--sleep-s", type=float, default=0.5)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--send-width", type=int, default=0)
    parser.add_argument("--send-height", type=int, default=0)
    parser.add_argument("--preview-steps", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--upload-timeout-s", type=float, default=60.0)
    parser.add_argument("--send-depth", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--depth-encoding", choices=["z16", "float16", "float32"], default="z16")
    parser.add_argument("--save-depth", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--close-camera", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--upload-url", default="")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    cfg_path = resolve_project_path(args.cfg)
    cfg = load_cfg(cfg_path)
    run_dir = Path(args.out_dir).expanduser().resolve() / f"g1_humanego_client_{utc_stamp()}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "ok": False,
        "control_sent": False,
        "note": "Robot-side server-client dry-run. No control commands are sent.",
        "args": vars(args),
        "cfg_path": str(cfg_path),
        "requests": [],
    }

    cam = None
    arm = None
    try:
        from G1Camera import G1HeadRGBDCamera
        from G1RobotArm import G1RobotArmReadOnly

        log("initializing G1 camera")
        cam = G1HeadRGBDCamera(resolve_project_path(cfg["camera"]["cfg_path"]))
        log("initializing read-only G1 arm state adapter")
        arm = G1RobotArmReadOnly(side="right")

        for idx in range(max(1, int(args.steps))):
            log(f"step {idx}: reading RGB-D frame")
            frame = cam.get_frame()
            log(f"step {idx}: reading robot TCP/base-camera state")
            state = arm.get_debug_state()
            request_id = f"{utc_stamp()}_{args.tag}_{idx:03d}"
            log(f"step {idx}: encoding JPEG request payload")
            rgb_send, K_send, image_send_info = resize_image_and_K(
                frame.rgb,
                frame.K,
                args.send_width,
                args.send_height,
            )
            jpeg_b64 = encode_jpeg_b64(rgb_send, args.jpeg_quality)
            log(
                f"step {idx}: sending RGB {image_send_info['sent_shape']} "
                f"jpeg_b64_bytes={len(jpeg_b64)}"
            )
            depth_send_info = {"sent": False}
            depth_b64 = None
            if args.send_depth:
                depth_send = resize_depth_to_shape(frame.depth_m, rgb_send.shape[:2])
                depth_b64, depth_send_info = encode_depth_npz_b64(depth_send, args.depth_encoding)
                depth_send_info["sent"] = True
                depth_send_info["base64_bytes"] = len(depth_b64)
                log(
                    f"step {idx}: sending depth {depth_send_info['shape']} "
                    f"encoding={args.depth_encoding} base64_bytes={len(depth_b64)}"
                )
            payload = {
                "request_id": request_id,
                "client_time_utc": datetime.now(timezone.utc).isoformat(),
                "preview_steps": int(args.preview_steps),
                "K": np.asarray(K_send, dtype=np.float64).tolist(),
                "rgb_jpeg_b64": jpeg_b64,
                "frame_summary": {
                    "rgb_shape": list(rgb_send.shape),
                    "source_rgb_shape": list(frame.rgb.shape),
                    "image_send": image_send_info,
                    "depth_shape": list(frame.depth_m.shape),
                    "depth_send": depth_send_info,
                    "depth_valid_ratio": float(np.isfinite(frame.depth_m).mean()),
                },
                "current": {
                    "T_head_pitch_camera": np.asarray(state["T_head_pitch_camera"], dtype=np.float64).tolist(),
                    "T_base_camera": np.asarray(state["T_base_camera"], dtype=np.float64).tolist(),
                    "T_base_in_cam": np.asarray(state["T_base_in_cam"], dtype=np.float64).tolist(),
                    "T_link7_in_base": np.asarray(state["T_link7_in_base"], dtype=np.float64).tolist(),
                    "T_tcp_in_link7": np.asarray(state["T_tcp_in_link7"], dtype=np.float64).tolist(),
                    "T_tcp_in_base": np.asarray(state["T_tcp_in_base"], dtype=np.float64).tolist(),
                    "T_tcp_in_cam": np.asarray(state["T_tcp_in_cam"], dtype=np.float64).tolist(),
                    "gripper": float(state["gripper"]),
                    "gripper_state": json_safe(state["gripper_state"]),
                    "corobot_fk": json_safe(state["corobot_fk"]),
                },
            }
            if depth_b64 is not None:
                payload["depth_m_npz_b64"] = depth_b64
                payload["depth_encoding"] = depth_send_info
            iter_dir = run_dir / f"iter_{idx:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(iter_dir / "rgb_sent_bgr.jpg"), rgb_send)
            if args.save_depth:
                np.save(iter_dir / "depth_m.npy", frame.depth_m)
            request_summary = dict(payload)
            request_summary.pop("rgb_jpeg_b64", None)
            (iter_dir / "request_summary.json").write_text(
                json.dumps(json_safe(request_summary), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            started = time.time()
            try:
                log(f"step {idx}: POST {args.server_url}")
                server_result = post_json(args.server_url, payload, args.timeout_s)
                elapsed = time.time() - started
                log(
                    f"step {idx}: server response ok "
                    f"status={server_result['status']} bytes={server_result['num_bytes']} "
                    f"duration={elapsed:.3f}s"
                )
                (iter_dir / "server_response.json").write_text(
                    json.dumps(json_safe(server_result["json"]), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                item = {
                    "idx": idx,
                    "request_id": request_id,
                    "ok": True,
                    "server_status": server_result["status"],
                    "server_response_bytes": server_result["num_bytes"],
                    "duration_s": elapsed,
                    "server_response": server_result["json"],
                }
            except Exception as exc:
                item = {
                    "idx": idx,
                    "request_id": request_id,
                    "ok": False,
                    "duration_s": time.time() - started,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            report["requests"].append(item)
            if idx + 1 < int(args.steps):
                time.sleep(float(args.sleep_s))

        report["ok"] = all(item.get("ok") for item in report["requests"])
    except Exception as exc:
        report.update({"ok": False, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
    finally:
        if cam is not None and args.close_camera:
            try:
                log("closing G1 camera")
                cam.close()
                log("G1 camera closed")
            except Exception as exc:
                report["camera_close_error"] = f"{type(exc).__name__}: {exc}"
        elif cam is not None:
            report["camera_close_skipped"] = True
            log("skipping G1 camera close for dry-run to avoid SDK shutdown blocking")
        if arm is not None:
            try:
                log("closing read-only arm adapter")
                arm.close()
                log("read-only arm adapter closed")
            except Exception as exc:
                report["arm_close_error"] = f"{type(exc).__name__}: {exc}"

    (run_dir / "client_dry_run_report.json").write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log("building local result zip")
    zip_path = make_zip(run_dir)
    upload = None
    if args.upload_url:
        try:
            log(f"uploading result zip to {args.upload_url}")
            upload = upload_zip(zip_path, args.upload_url, args.upload_timeout_s)
            log(f"upload complete status={upload.get('status')}")
        except Exception as exc:
            upload = {"ok": False, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        (run_dir / "upload_result.json").write_text(json.dumps(upload, ensure_ascii=False, indent=2), encoding="utf-8")
        log("rebuilding zip with upload_result.json")
        zip_path = make_zip(run_dir)

    print(json.dumps({"run_dir": str(run_dir), "zip_path": str(zip_path), "upload": upload}, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
