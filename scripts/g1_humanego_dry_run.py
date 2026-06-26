#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run HumanEgo on G1 once/slowly and print converted G1 targets.

This script is intentionally read-only: it never calls
set_end_effector_pose_control or move_gripper. It is the deployment dry-run
between "all interfaces verified" and "send policy targets to the robot".
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = PROJECT_ROOT / "cfg" / "inference" / "g1_serve_bread_right.yaml"

for path in (PROJECT_ROOT, PROJECT_ROOT / "inference"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from g1_artifacts import artifact_dir, run_dir as artifact_run_dir  # noqa: E402


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


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


def matrix_json(T: Any, decimals: int = 9) -> list[list[float]]:
    arr = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return [[round(float(v), decimals) for v in row] for row in arr.tolist()]


def make_zip(src_dir: Path) -> Path:
    zip_path = src_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir.parent))
    return zip_path


def upload_zip(zip_path: Path, upload_url: str, timeout_s: float = 20.0) -> Dict[str, Any]:
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
        return {
            "ok": True,
            "status": resp.status,
            "response": resp.read().decode("utf-8", errors="replace"),
        }


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def check_dry_run_prerequisites(cfg: dict[str, Any], requested_device: str) -> dict[str, Any]:
    """Fail early with actionable setup information before touching robot APIs."""
    missing: list[dict[str, Any]] = []
    warnings: list[str] = []

    if importlib.util.find_spec("torch") is None:
        missing.append(
            {
                "type": "python_module",
                "name": "torch",
                "message": "HumanEgo policy inference requires PyTorch, but the active Python cannot import torch.",
            }
        )
    else:
        try:
            import torch

            if requested_device == "cuda" and not torch.cuda.is_available():
                missing.append(
                    {
                        "type": "device",
                        "name": "cuda",
                        "message": "--device cuda was requested, but torch.cuda.is_available() is false.",
                    }
                )
            elif requested_device == "auto" and not torch.cuda.is_available():
                warnings.append("CUDA is not available in this Python; dry-run will use CPU if dependencies are otherwise present.")
        except Exception as exc:
            missing.append(
                {
                    "type": "python_module",
                    "name": "torch",
                    "message": f"torch is installed but failed to import: {type(exc).__name__}: {exc}",
                }
            )

    ckpt_path = resolve_project_path(cfg["policy"]["ckpt"])
    ckpt_dir = ckpt_path.parent
    required_files = {
        "checkpoint": ckpt_path,
        "config": ckpt_dir / "config.json",
        "dataset_stats": ckpt_dir / "dataset_stats.json",
    }
    for label, path in required_files.items():
        if not path.exists():
            missing.append(
                {
                    "type": "file",
                    "name": label,
                    "path": str(path),
                    "message": f"Required HumanEgo policy {label} file does not exist.",
                }
            )

    return {
        "ok": not missing,
        "missing": missing,
        "warnings": warnings,
        "policy_ckpt": str(ckpt_path),
        "policy_dir": str(ckpt_dir),
        "python": sys.executable,
    }


def quat_xyzw_to_R(q: Any) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    n = float(np.sqrt(x * x + y * y + z * z + w * w))
    if n <= 0:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def R_to_quat_xyzw(R: np.ndarray) -> list[float]:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(R)))
        if idx == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= max(np.linalg.norm(q), 1e-12)
    return [float(v) for v in q]


def pose_dict_from_T(T: np.ndarray) -> dict[str, float]:
    qx, qy, qz, qw = R_to_quat_xyzw(T[:3, :3])
    return {
        "x": float(T[0, 3]),
        "y": float(T[1, 3]),
        "z": float(T[2, 3]),
        "qx": qx,
        "qy": qy,
        "qz": qz,
        "qw": qw,
    }


def T_from_pose_dict(pose: dict[str, float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_R([pose["qx"], pose["qy"], pose["qz"], pose["qw"]])
    T[:3, 3] = [pose["x"], pose["y"], pose["z"]]
    return T


def clipped_translation_target(T_start: np.ndarray, T_target: np.ndarray, max_step_m: float) -> tuple[np.ndarray, dict[str, Any]]:
    T = np.asarray(T_target, dtype=np.float64).copy()
    delta = T[:3, 3] - T_start[:3, 3]
    norm = float(np.linalg.norm(delta))
    clipped = False
    if max_step_m > 0 and norm > max_step_m:
        T[:3, 3] = T_start[:3, 3] + delta / norm * max_step_m
        clipped = True
    return T, {
        "raw_delta_m": delta.tolist(),
        "raw_delta_norm_m": norm,
        "max_step_m": float(max_step_m),
        "clipped": clipped,
        "clipped_delta_m": (T[:3, 3] - T_start[:3, 3]).tolist(),
        "clipped_delta_norm_m": float(np.linalg.norm(T[:3, 3] - T_start[:3, 3])),
    }


def load_fixed_objects(perception_cfg: dict[str, Any]) -> dict[str, Any]:
    from interfaces import ObjectState

    raw = (
        perception_cfg.get("dry_run_object_poses_cam")
        or perception_cfg.get("fixed_object_poses_cam")
        or {}
    )
    if not raw:
        raise ValueError("fixed object source requires perception.dry_run_object_poses_cam")

    out = {}
    for key, item in raw.items():
        T = np.asarray(item["T_in_cam"] if isinstance(item, dict) else item, dtype=np.float32).reshape(4, 4)
        kpts = np.asarray(item.get("kpts_local", []), dtype=np.float32) if isinstance(item, dict) else np.zeros((0, 3), np.float32)
        if kpts.size == 0:
            kpts = np.zeros((0, 3), dtype=np.float32)
        out[key] = ObjectState(T_in_cam=T, kpts_local=kpts.reshape(-1, 3))
    return out


def estimate_rgbd_objects(cam: Any, perception_cfg: dict[str, Any], n_frames: int):
    from object_pose_rgbd import RGBDObjectPoseEstimator

    estimator = RGBDObjectPoseEstimator(perception_cfg)
    frames = [cam.get_frame() for _ in range(max(1, n_frames))]
    return estimator.estimate(frames)


def object_summary(objs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: {
            "T_in_cam": matrix_json(obj.T_in_cam),
            "p_cam_m": [float(v) for v in np.asarray(obj.T_in_cam)[:3, 3]],
            "kpts_local_count": int(len(obj.kpts_local)),
        }
        for key, obj in objs.items()
    }


def build_target_preview(
    *,
    traj: dict[str, Any],
    done_prob: float,
    policy: Any,
    anchor: Any,
    T_align: np.ndarray,
    T_base_camera: np.ndarray,
    T_tcp_in_link7: np.ndarray,
    T_link7_current_in_base: np.ndarray,
    max_step_m: float,
    max_steps: int,
) -> dict[str, Any]:
    T_link7_inv_tcp = np.linalg.inv(T_tcp_in_link7)
    preview: dict[str, Any] = {"done_prob": float(done_prob), "sides": {}}
    for side, (pos, o6d, grasp) in traj.items():
        side_items: list[dict[str, Any]] = []
        n = min(max_steps, len(pos))
        for k in range(n):
            T_tcp_target_in_cam = np.asarray(
                policy.decode_ee_in_cam(pos[k], o6d[k], anchor, T_align),
                dtype=np.float64,
            )
            T_tcp_target_in_base = T_base_camera @ T_tcp_target_in_cam
            T_link7_target_in_base = T_tcp_target_in_base @ T_link7_inv_tcp
            T_link7_safe, limit_info = clipped_translation_target(
                T_link7_current_in_base,
                T_link7_target_in_base,
                max_step_m,
            )
            grasp_value = float(np.asarray(grasp[k]).reshape(-1)[0])
            side_items.append(
                {
                    "step": k,
                    "gripper_humanego_0_open_1_closed": grasp_value,
                    "gripper_g1_raw_0_open_120_closed": float(np.clip(grasp_value, 0.0, 1.0) * 120.0),
                    "T_tcp_target_in_cam": matrix_json(T_tcp_target_in_cam),
                    "T_tcp_target_in_base": matrix_json(T_tcp_target_in_base),
                    "T_link7_target_in_base": matrix_json(T_link7_target_in_base),
                    "right_pose_flat_raw": pose_dict_from_T(T_link7_target_in_base),
                    "safety_translation_limit": limit_info,
                    "T_link7_target_in_base_limited": matrix_json(T_link7_safe),
                    "right_pose_flat_limited": pose_dict_from_T(T_link7_safe),
                }
            )
        preview["sides"][side] = side_items
    return preview


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default=str(DEFAULT_CFG))
    parser.add_argument("--out-dir", default=str(artifact_dir("client_local_model")))
    parser.add_argument("--tag", default="humanego_dry_run")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--steps", type=int, default=1, help="Number of policy dry-run iterations.")
    parser.add_argument("--sleep-s", type=float, default=0.5)
    parser.add_argument("--object-source", choices=["fixed", "rgbd"], default="")
    parser.add_argument("--preview-steps", type=int, default=3)
    parser.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--upload-url", default="")
    parser.add_argument("--upload-timeout-s", type=float, default=20.0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    cfg_path = resolve_project_path(args.cfg)
    cfg = load_cfg(cfg_path)
    out_base = Path(args.out_dir).expanduser().resolve()
    default_base = artifact_dir("client_local_model")
    if out_base == default_base:
        run_dir = artifact_run_dir("client_local_model", args.tag, prefix="humanego_dry_run")
    else:
        run_dir = out_base / f"g1_humanego_{utc_stamp()}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "ok": False,
        "control_sent": False,
        "note": "Dry-run only. No G1 control commands are sent.",
        "args": vars(args),
        "cfg_path": str(cfg_path),
    }

    cam = None
    arm = None
    try:
        preflight = check_dry_run_prerequisites(cfg, args.device)
        report["preflight"] = preflight
        if not preflight["ok"]:
            raise RuntimeError(
                "Dry-run prerequisites are missing. See report['preflight']['missing'] for exact setup gaps."
            )

        from G1Camera import G1HeadRGBDCamera
        from G1RobotArm import G1RobotArmReadOnly
        from policy import ICTPolicy

        device = choose_device(args.device)
        report["device"] = device

        cam = G1HeadRGBDCamera(resolve_project_path(cfg["camera"]["cfg_path"]))
        arm = G1RobotArmReadOnly(side="right")
        policy = ICTPolicy(cfg["policy"], device=device)
        if policy.sides != ["right"]:
            raise ValueError(f"expected a single right-hand checkpoint, got policy.sides={policy.sides}")

        perception_cfg = cfg.get("perception", {})
        object_source = args.object_source or perception_cfg.get("dry_run_object_source", "fixed")
        if object_source == "rgbd":
            objects = estimate_rgbd_objects(cam, perception_cfg, int(perception_cfg.get("n_init_frames", 1)))
        else:
            objects = load_fixed_objects(perception_cfg)

        anchor_key = perception_cfg.get("anchor_key", "obj1")
        anchor = objects.get(anchor_key)
        if anchor is None:
            raise ValueError(f"anchor_key {anchor_key!r} not found in objects: {sorted(objects.keys())}")

        T_align = np.asarray(cfg["robot"]["T_align"], dtype=np.float64).reshape(4, 4)
        max_step_m = float(cfg.get("control", {}).get("max_pos_step", 0.03))
        preview_steps = max(1, int(args.preview_steps))

        report.update(
            {
                "policy": {
                    "sides": policy.sides,
                    "frame_mode": policy.frame_mode,
                    "action_mode": policy.action_mode,
                    "pred_horizon": policy.pred_horizon,
                    "use_region_attn": policy.use_region_attn,
                    "use_pcd_features": policy.use_pcd_features,
                },
                "object_source": object_source,
                "objects": object_summary(objects),
                "T_align_T_hand_in_tcp": matrix_json(T_align),
                "max_pos_step_m": max_step_m,
                "iterations": [],
            }
        )

        for step_idx in range(max(1, int(args.steps))):
            frame = cam.get_frame()
            state = arm.get_debug_state()
            T_tcp_in_cam = np.asarray(state["T_tcp_in_cam"], dtype=np.float64)
            T_hand_in_cam = T_tcp_in_cam @ T_align
            hands_in_cam = {"right": T_hand_in_cam}
            grippers = {"right": float(state["gripper"])}

            clean_bgr = frame.rgb.copy()
            x_rgb = policy.prepare_image(clean_bgr)
            x_ict, ict_mask = policy.build_ict(hands_in_cam, grippers, objects, anchor_key)
            anchor_uv = policy.compute_anchor_uv(anchor, frame.K, frame.rgb.shape[1], frame.rgb.shape[0])
            traj, done_prob = policy.infer(x_rgb, x_ict, ict_mask, anchor_uv)

            target_preview = build_target_preview(
                traj=traj,
                done_prob=done_prob,
                policy=policy,
                anchor=anchor,
                T_align=T_align,
                T_base_camera=np.asarray(state["T_base_camera"], dtype=np.float64),
                T_tcp_in_link7=np.asarray(state["T_tcp_in_link7"], dtype=np.float64),
                T_link7_current_in_base=np.asarray(state["T_link7_in_base"], dtype=np.float64),
                max_step_m=max_step_m,
                max_steps=preview_steps,
            )

            iter_dir = run_dir / f"iter_{step_idx:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            if args.save_images:
                cv2.imwrite(str(iter_dir / "rgb_bgr.png"), frame.rgb)
                depth_vis = np.asarray(frame.depth_m, dtype=np.float32)
                np.save(iter_dir / "depth_m.npy", depth_vis)

            np.save(iter_dir / "T_base_camera.npy", state["T_base_camera"])
            np.save(iter_dir / "T_tcp_current_in_cam.npy", T_tcp_in_cam)
            np.save(iter_dir / "T_hand_current_in_cam.npy", T_hand_in_cam)

            iteration = {
                "step_idx": step_idx,
                "frame": {
                    "rgb_shape": list(frame.rgb.shape),
                    "depth_shape": list(frame.depth_m.shape),
                    "depth_valid_ratio": float(np.isfinite(frame.depth_m).mean()),
                    "K": np.asarray(frame.K, dtype=np.float64).tolist(),
                },
                "current": {
                    "T_base_camera": matrix_json(state["T_base_camera"]),
                    "T_base_in_cam": matrix_json(state["T_base_in_cam"]),
                    "T_link7_in_base": matrix_json(state["T_link7_in_base"]),
                    "T_tcp_in_link7": matrix_json(state["T_tcp_in_link7"]),
                    "T_tcp_in_cam": matrix_json(T_tcp_in_cam),
                    "T_hand_in_cam": matrix_json(T_hand_in_cam),
                    "gripper": json_safe(state.get("gripper")),
                    "gripper_state": json_safe(state.get("gripper_state")),
                    "corobot_fk": json_safe(state.get("corobot_fk")),
                },
                "ict": {
                    "shape": list(x_ict.shape),
                    "mask": ict_mask.detach().cpu().numpy().astype(bool).tolist(),
                },
                "anchor_uv": None if anchor_uv is None else anchor_uv.detach().cpu().numpy().tolist(),
                "policy_preview": target_preview,
            }
            report["iterations"].append(iteration)
            (iter_dir / "iteration_report.json").write_text(
                json.dumps(json_safe(iteration), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if step_idx + 1 < int(args.steps):
                time.sleep(float(args.sleep_s))

        report["ok"] = True
    except Exception as exc:
        report.update(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
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

    (run_dir / "humanego_dry_run_report.json").write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    zip_path = make_zip(run_dir)
    upload = None
    if args.upload_url:
        try:
            upload = upload_zip(zip_path, args.upload_url, args.upload_timeout_s)
        except Exception as exc:
            upload = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        (run_dir / "upload_result.json").write_text(json.dumps(upload, ensure_ascii=False, indent=2), encoding="utf-8")
        zip_path = make_zip(run_dir)

    print(json.dumps({"run_dir": str(run_dir), "zip_path": str(zip_path), "upload": upload}, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
