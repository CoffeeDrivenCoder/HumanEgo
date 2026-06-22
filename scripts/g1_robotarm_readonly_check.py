#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only G1RobotArm check: current TCP pose and target conversion."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = PROJECT_ROOT / "cfg" / "inference" / "g1_serve_bread_right.yaml"


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


def matrix_json(T: Any) -> list[list[float]]:
    arr = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return [[round(float(v), 9) for v in row] for row in arr.tolist()]


def scalar_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: scalar_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scalar_json(v) for v in value]
    return value


def make_zip(src_dir: Path) -> Path:
    zip_path = src_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir.parent))
    return zip_path


def upload_zip(zip_path: Path, upload_url: str) -> Dict[str, Any]:
    data = zip_path.read_bytes()
    req = urllib.request.Request(
        upload_url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/zip",
            "Content-Length": str(len(data)),
            "X-G1-Diagnostics-Filename": zip_path.name,
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return {"ok": True, "status": resp.status, "response": resp.read().decode("utf-8", errors="replace")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default=str(DEFAULT_CFG))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "g1_robotarm_readonly_runs"))
    parser.add_argument("--tag", default="robotarm_readonly")
    parser.add_argument("--side", default="right", choices=["right", "left"])
    parser.add_argument("--upload-url", default="")
    parser.add_argument("--sample-offset-cam", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--urdf-path", default="")
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir).expanduser().resolve() / f"g1_robotarm_{stamp}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {"ok": False, "args": vars(args)}
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "inference"))
        sys.path.insert(0, str(PROJECT_ROOT))
        from G1Geometry import fixed_T_tcp_in_link7
        from G1RobotArm import G1RobotArmReadOnly

        cfg_path = resolve_project_path(args.cfg)
        cfg = load_cfg(cfg_path)
        T_hand_in_tcp = np.asarray(cfg["robot"]["T_align"], dtype=np.float64).reshape(4, 4)
        arm = G1RobotArmReadOnly(side=args.side, urdf_path=args.urdf_path or None)

        state = arm.get_debug_state()
        T_tcp_in_cam = state["T_tcp_in_cam"]
        T_hand_in_cam = T_tcp_in_cam @ T_hand_in_tcp

        # A sample target near the current hand pose. This is only a conversion
        # check and is never sent to the robot.
        T_sample_hand_target_in_cam = T_hand_in_cam.copy()
        T_sample_hand_target_in_cam[:3, 3] += np.asarray(args.sample_offset_cam, dtype=np.float64)
        T_sample_tcp_target_in_cam = T_sample_hand_target_in_cam @ np.linalg.inv(T_hand_in_tcp)
        T_sample_link7_target_in_cam = T_sample_tcp_target_in_cam @ np.linalg.inv(fixed_T_tcp_in_link7(args.side))

        np.save(run_dir / "T_base_camera.npy", state["T_base_camera"])
        np.save(run_dir / "T_tcp_in_base.npy", state["T_tcp_in_base"])
        np.save(run_dir / "T_tcp_in_cam.npy", T_tcp_in_cam)
        np.save(run_dir / "T_hand_in_cam.npy", T_hand_in_cam)
        np.save(run_dir / "T_sample_tcp_target_in_cam.npy", T_sample_tcp_target_in_cam)

        report.update(
            {
                "ok": True,
                "cfg_path": str(cfg_path),
                "note": "Read-only check. No control commands were sent.",
                "parameter_source": state["parameter_source"],
                "corobot_fk": scalar_json(state["corobot_fk"]),
                "gripper_state": scalar_json(state["gripper_state"]),
                "gripper_normalized_proxy": state["gripper"],
                "T_head_pitch_camera": matrix_json(state["T_head_pitch_camera"]),
                "T_base_camera": matrix_json(state["T_base_camera"]),
                "T_base_in_cam": matrix_json(state["T_base_in_cam"]),
                "T_link7_in_base": matrix_json(state["T_link7_in_base"]),
                "T_tcp_in_link7": matrix_json(state["T_tcp_in_link7"]),
                "T_tcp_in_base": matrix_json(state["T_tcp_in_base"]),
                "T_tcp_in_cam": matrix_json(T_tcp_in_cam),
                "T_hand_in_tcp": matrix_json(T_hand_in_tcp),
                "T_hand_in_cam": matrix_json(T_hand_in_cam),
                "sample_target": {
                    "offset_cam_m": args.sample_offset_cam,
                    "T_sample_hand_target_in_cam": matrix_json(T_sample_hand_target_in_cam),
                    "T_sample_tcp_target_in_cam": matrix_json(T_sample_tcp_target_in_cam),
                    "T_sample_link7_target_in_cam": matrix_json(T_sample_link7_target_in_cam),
                },
                "blocked_next": [
                    "Verify set_end_effector_pose_control target frame before sending any target.",
                    "Verify gripper open/close raw range before using policy grasp output.",
                ],
            }
        )
    except Exception as exc:
        report.update({"ok": False, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})

    (run_dir / "robotarm_readonly_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = make_zip(run_dir)
    upload = None
    if args.upload_url:
        try:
            upload = upload_zip(zip_path, args.upload_url)
        except Exception as exc:
            upload = {"ok": False, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        (run_dir / "upload_result.json").write_text(json.dumps(upload, ensure_ascii=False, indent=2), encoding="utf-8")
        zip_path = make_zip(run_dir)

    print(json.dumps({"run_dir": str(run_dir), "zip_path": str(zip_path), "upload": upload}, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
