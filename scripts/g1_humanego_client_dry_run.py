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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default=str(DEFAULT_CFG))
    parser.add_argument("--server-url", default="http://111.0.22.33:30003/infer")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "g1_humanego_client_runs"))
    parser.add_argument("--tag", default="client_dry_run")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--sleep-s", type=float, default=0.5)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--preview-steps", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--upload-timeout-s", type=float, default=60.0)
    parser.add_argument("--save-depth", action=argparse.BooleanOptionalAction, default=False)
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
            payload = {
                "request_id": request_id,
                "client_time_utc": datetime.now(timezone.utc).isoformat(),
                "preview_steps": int(args.preview_steps),
                "K": np.asarray(frame.K, dtype=np.float64).tolist(),
                "rgb_jpeg_b64": encode_jpeg_b64(frame.rgb, args.jpeg_quality),
                "frame_summary": {
                    "rgb_shape": list(frame.rgb.shape),
                    "depth_shape": list(frame.depth_m.shape),
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
            iter_dir = run_dir / f"iter_{idx:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(iter_dir / "rgb_bgr.jpg"), frame.rgb)
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
        if cam is not None:
            try:
                cam.close()
            except Exception as exc:
                report["camera_close_error"] = f"{type(exc).__name__}: {exc}"
        if arm is not None:
            try:
                arm.close()
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
