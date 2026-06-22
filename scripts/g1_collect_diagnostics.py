#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect G1 camera/control diagnostics for HumanEgo deployment.

Default mode is read-only: it reads camera frames, robot states, motion status,
and calibration files, then writes a zip package. Optional control probes are
guarded by --enable-control --confirm-control RUN_CONTROL.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import os
import platform
import shutil
import socket
import sys
import time
import traceback
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - script should still report import failure
    np = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_G1_CAMERA_CFG = PROJECT_ROOT / "cfg" / "inference" / "g1_head_rgbd.yaml"
DEFAULT_G1_PARAMETER_PY = PROJECT_ROOT / "G1" / "parameter.py"
FALLBACK_G1_PARAMETER_PY = PROJECT_ROOT / "scripts" / "parameters.py"
DEFAULT_DINOSAM_CFG = PROJECT_ROOT / "cfg" / "preprocess" / "base" / "DINOSAM.yaml"

# Robot-side runtime dependencies. a2d_sdk is provided by the G1 SDK/runtime,
# not by this HumanEgo repository.
REQUIRED_G1_MODULES = ["a2d_sdk.robot"]
OPTIONAL_COMMON_MODULES = ["cv2"]
OPTIONAL_OBJECT_POSE_MODULES = ["preprocess.DINOSAM", "preprocess.OrientAnything"]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_safe(value: Any, *, max_items: int = 80) -> Any:
    if np is not None:
        if isinstance(value, np.ndarray):
            info: Dict[str, Any] = {
                "type": "ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            if value.size and np.issubdtype(value.dtype, np.number):
                finite = value[np.isfinite(value)] if np.issubdtype(value.dtype, np.floating) else value
                if finite.size:
                    info.update(
                        {
                            "min": float(np.min(finite)),
                            "max": float(np.max(finite)),
                            "mean": float(np.mean(finite)),
                        }
                    )
            if value.size <= max_items:
                info["values"] = value.tolist()
            return info
        if isinstance(value, np.generic):
            return value.item()

    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": type(value).__name__, "num_bytes": len(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v, max_items=max_items) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        seq = list(value)
        if len(seq) > max_items:
            return {
                "type": type(value).__name__,
                "len": len(seq),
                "head": [json_safe(v, max_items=max_items) for v in seq[: max_items // 2]],
                "tail": [json_safe(v, max_items=max_items) for v in seq[-max_items // 2 :]],
            }
        return [json_safe(v, max_items=max_items) for v in seq]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(json_safe(data), ensure_ascii=False, indent=2), encoding="utf-8")


def call_and_capture(name: str, fn, *args, **kwargs) -> Dict[str, Any]:
    started = time.time()
    item: Dict[str, Any] = {"ok": False, "name": name, "started_unix": started}
    try:
        value = fn(*args, **kwargs)
        item.update(
            {
                "ok": True,
                "duration_s": time.time() - started,
                "value": json_safe(value),
            }
        )
        return item
    except Exception as exc:
        item.update(
            {
                "ok": False,
                "duration_s": time.time() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return item


def method_signature(obj: Any, method_name: str) -> Dict[str, Any]:
    if not hasattr(obj, method_name):
        return {"exists": False}
    method = getattr(obj, method_name)
    out: Dict[str, Any] = {"exists": True, "repr": repr(method)}
    try:
        out["signature"] = str(inspect.signature(method))
    except Exception as exc:
        out["signature_error"] = f"{type(exc).__name__}: {exc}"
    return out


def list_public_methods(obj: Any) -> List[str]:
    methods = []
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(obj, name)
        except Exception:
            continue
        if callable(attr):
            methods.append(name)
    return sorted(methods)


def import_symbol(path: str):
    module_name, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def module_status(module_name: str) -> Dict[str, Any]:
    item: Dict[str, Any] = {"module": module_name, "ok": False}
    try:
        spec = importlib.util.find_spec(module_name)
        item["find_spec"] = spec is not None
        if spec is not None:
            item["origin"] = spec.origin
        module = importlib.import_module(module_name)
        item["ok"] = True
        item["file"] = getattr(module, "__file__", None)
    except Exception as exc:
        item.update(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    return item


def collect_runtime_dependencies(args: argparse.Namespace) -> Dict[str, Any]:
    for path in (PROJECT_ROOT, PROJECT_ROOT / "inference"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    modules = {
        "required_g1": REQUIRED_G1_MODULES,
        "optional_common": OPTIONAL_COMMON_MODULES,
        "optional_object_pose": OPTIONAL_OBJECT_POSE_MODULES if args.run_object_pose else [],
    }
    return {
        group: {module: module_status(module) for module in names}
        for group, names in modules.items()
    }


def import_python_file(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state_tuple_to_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, tuple) and len(value) == 2:
        return {"data": json_safe(value[0]), "timestamp_ns": json_safe(value[1])}
    return {"raw": json_safe(value)}


def array_from_any(value: Any):
    if np is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return np.frombuffer(value, dtype=np.uint8)
    try:
        return np.asarray(value)
    except Exception:
        return None


def maybe_save_rgb(path_prefix: Path, image: Any) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"saved": []}
    arr = array_from_any(image)
    if arr is None:
        return meta

    if np is not None:
        np.save(str(path_prefix) + ".npy", arr)
        meta["saved"].append(str(path_prefix) + ".npy")
        meta["array"] = json_safe(arr)

    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        try:
            import cv2

            bgr = arr[:, :, :3]
            # G1 examples treat RGB as RGB. cv2.imwrite expects BGR.
            if arr.shape[2] >= 3:
                bgr = bgr[:, :, ::-1]
            out = str(path_prefix) + ".png"
            cv2.imwrite(out, bgr)
            meta["saved"].append(out)
        except Exception as exc:
            meta["image_save_error"] = f"{type(exc).__name__}: {exc}"
    elif isinstance(image, (bytes, bytearray, memoryview)):
        out = str(path_prefix) + ".bin"
        Path(out).write_bytes(bytes(image))
        meta["saved"].append(out)

    return meta


def wait_latest_image(camera_group: Any, camera_name: str, tries: int, sleep_s: float) -> Tuple[Any, Any, Dict[str, Any]]:
    attempts: List[Dict[str, Any]] = []
    last_img = None
    last_ts = None
    for idx in range(max(1, tries)):
        started = time.time()
        try:
            img, ts = camera_group.get_latest_image(camera_name)
            last_img, last_ts = img, ts
            attempt = {
                "idx": idx,
                "ok": True,
                "duration_s": time.time() - started,
                "timestamp_ns": json_safe(ts),
                "is_none": img is None,
                "summary": json_safe(img),
            }
            attempts.append(attempt)
            if img is not None:
                return img, ts, {"ok": True, "attempts": attempts, "num_attempts": idx + 1}
        except Exception as exc:
            attempts.append(
                {
                    "idx": idx,
                    "ok": False,
                    "duration_s": time.time() - started,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        time.sleep(sleep_s)
    return last_img, last_ts, {"ok": False, "attempts": attempts, "num_attempts": len(attempts)}


def describe_camera_packet(camera_group: Any, camera_name: str) -> Dict[str, Any]:
    if not hasattr(camera_group, "get_latest_packet"):
        return {"exists": False}
    item: Dict[str, Any] = {"exists": True, "ok": False}
    try:
        packet = camera_group.get_latest_packet(camera_name)
        if packet is None:
            item.update({"ok": True, "packet": None})
            return item
        info = {
            "packet_type": str(getattr(packet, "type", None)),
            "encoding_format": str(getattr(packet, "encoding_format", None)),
            "color_format": str(getattr(packet, "color_format", None)),
            "width": int(getattr(packet, "image_width", 0) or 0),
            "height": int(getattr(packet, "image_height", 0) or 0),
            "send_timestamp": int(getattr(packet, "send_timestamp", 0) or 0),
        }
        if hasattr(packet, "get_image_data"):
            data = packet.get_image_data()
            info["image_data_bytes"] = 0 if data is None else int(len(data))
        item.update({"ok": True, "packet": info})
    except Exception as exc:
        item.update(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    return item


def collect_camera_runtime_stats(camera_group: Any, camera_name: str) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    if hasattr(camera_group, "get_fps"):
        stats["fps"] = call_and_capture(f"get_fps({camera_name})", camera_group.get_fps, camera_name)
    if hasattr(camera_group, "get_latency_stats"):
        stats["latency_stats"] = call_and_capture(
            f"get_latency_stats({camera_name})",
            camera_group.get_latency_stats,
            camera_name,
            window_seconds=5.0,
        )
    stats["packet"] = describe_camera_packet(camera_group, camera_name)
    return stats


def depth_stats(depth_image: Any) -> Dict[str, Any]:
    if np is None:
        return {"error": "numpy unavailable"}
    depth = np.asarray(depth_image)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[:, :, 0]
    finite = np.isfinite(depth)
    nonzero = depth > 0
    valid = finite & nonzero
    valid_values = depth[valid]
    stats: Dict[str, Any] = {
        "shape": [int(v) for v in depth.shape],
        "dtype": str(depth.dtype),
        "min_raw": int(np.min(depth)) if depth.size else None,
        "max_raw": int(np.max(depth)) if depth.size else None,
        "nonzero_pixels": int(np.count_nonzero(nonzero)),
        "total_pixels": int(depth.size),
        "valid_ratio": float(np.count_nonzero(valid) / depth.size) if depth.size else 0.0,
    }
    if valid_values.size:
        stats.update(
            {
                "valid_min_raw": int(np.min(valid_values)),
                "valid_median_raw": float(np.median(valid_values)),
                "valid_mean_raw": float(np.mean(valid_values)),
                "valid_max_raw": int(np.max(valid_values)),
            }
        )
    return stats


def reshape_depth_candidates(depth_vec, shape_value: Any) -> List[Tuple[str, Any]]:
    if np is None:
        return []
    candidates: List[Tuple[str, Any]] = []
    if shape_value is not None:
        try:
            dims = list(shape_value)
            if len(dims) >= 2:
                a, b = int(dims[0]), int(dims[1])
                for label, shape in (
                    (f"shape_{a}_{b}", (a, b)),
                    (f"shape_{b}_{a}", (b, a)),
                ):
                    if depth_vec.size == shape[0] * shape[1]:
                        candidates.append((label, depth_vec.reshape(shape)))
        except Exception:
            pass
    return candidates


def maybe_save_depth(path_prefix: Path, depth_raw: Any, shape_value: Any) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"saved": []}
    if np is None:
        return meta

    if isinstance(depth_raw, (bytes, bytearray, memoryview)):
        depth_vec = np.frombuffer(depth_raw, dtype=np.uint16)
    else:
        arr = np.asarray(depth_raw)
        depth_vec = arr.reshape(-1) if arr.ndim > 1 else arr

    np.save(str(path_prefix) + "_raw.npy", depth_vec)
    meta["saved"].append(str(path_prefix) + "_raw.npy")
    meta["raw"] = json_safe(depth_vec)

    chosen = None
    for label, arr in reshape_depth_candidates(depth_vec, shape_value):
        np.save(str(path_prefix) + f"_{label}.npy", arr)
        meta["saved"].append(str(path_prefix) + f"_{label}.npy")
        meta.setdefault("reshape_candidates", {})[label] = json_safe(arr)
        if chosen is None:
            chosen = arr

    if chosen is None and not isinstance(depth_raw, (bytes, bytearray, memoryview)):
        arr = np.asarray(depth_raw)
        if arr.ndim >= 2:
            chosen = arr.squeeze()

    if chosen is not None and chosen.ndim == 2:
        try:
            import cv2

            png = str(path_prefix) + ".png"
            cv2.imwrite(png, chosen.astype(np.uint16))
            meta["saved"].append(png)
            vis = chosen.astype(np.float32)
            finite = np.isfinite(vis)
            if finite.any():
                lo, hi = np.percentile(vis[finite], [2, 98])
                if hi > lo:
                    vis = np.clip((vis - lo) / (hi - lo), 0, 1)
                else:
                    vis = np.zeros_like(vis)
            vis_png = str(path_prefix) + "_vis.png"
            cv2.imwrite(vis_png, (vis * 255).astype(np.uint8))
            meta["saved"].append(vis_png)
        except Exception as exc:
            meta["image_save_error"] = f"{type(exc).__name__}: {exc}"

    return meta


def collect_parameters(args: argparse.Namespace, report: Dict[str, Any]) -> None:
    section: Dict[str, Any] = {"ok": False}
    report["parameters"] = section
    path = Path(args.g1_parameter_py).expanduser().resolve()
    if not path.exists() and FALLBACK_G1_PARAMETER_PY.exists():
        path = FALLBACK_G1_PARAMETER_PY.resolve()
    section["parameter_py"] = str(path)
    try:
        module = import_python_file(path, "g1_parameter_runtime")
        params = module.load_all_parameters(args.parameter_camera_name)
        section.update(
            {
                "ok": True,
                "camera_name": args.parameter_camera_name,
                "intrinsics": params.get("intrinsics"),
                "extrinsics": params.get("extrinsics"),
            }
        )
    except Exception as exc:
        section.update(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def collect_camera(args: argparse.Namespace, report: Dict[str, Any], out_dir: Path) -> None:
    section: Dict[str, Any] = {"ok": False, "samples": []}
    report["camera"] = section
    frames_dir = ensure_dir(out_dir / "camera_frames")
    camera_group = None
    try:
        CosineCamera = import_symbol(args.cosine_camera_symbol)
        section["import"] = {"ok": True, "symbol": args.cosine_camera_symbol}
        camera_group = CosineCamera(args.camera_names)
        section["ok"] = True
        section["public_methods"] = list_public_methods(camera_group)
        section["method_signatures"] = {
            name: method_signature(camera_group, name)
            for name in (
                "get_latest_image",
                "get_image_shape",
                "get_image_nearest",
                "get_latest_packet",
                "get_fps",
                "get_latency_stats",
            )
        }
        if args.camera_warmup_s > 0:
            time.sleep(args.camera_warmup_s)

        for sample_idx in range(args.samples):
            sample: Dict[str, Any] = {"sample_idx": sample_idx, "cameras": {}}
            for camera_name in args.camera_names:
                cam_item: Dict[str, Any] = {"camera_name": camera_name}
                cam_item["runtime_stats"] = collect_camera_runtime_stats(camera_group, camera_name)
                shape_value = None
                if hasattr(camera_group, "get_image_shape"):
                    shape_item = call_and_capture(f"get_image_shape({camera_name})", camera_group.get_image_shape, camera_name)
                    cam_item["shape_call"] = shape_item
                    if shape_item.get("ok"):
                        try:
                            shape_value = camera_group.get_image_shape(camera_name)
                        except Exception:
                            shape_value = shape_item.get("value")

                image, ts, wait_item = wait_latest_image(
                    camera_group,
                    camera_name,
                    args.camera_tries,
                    args.camera_sleep_s,
                )
                cam_item["latest_wait"] = wait_item

                if image is not None:
                    safe_name = camera_name.replace("/", "_")
                    prefix = frames_dir / f"{sample_idx:03d}_{safe_name}_{ts}"
                    if "depth" in camera_name:
                        cam_item["depth_stats"] = depth_stats(image)
                        cam_item["artifact"] = maybe_save_depth(prefix, image, shape_value)
                    else:
                        cam_item["artifact"] = maybe_save_rgb(prefix, image)
                    cam_item["timestamp_ns"] = json_safe(ts)
                    cam_item["raw_summary"] = json_safe(image)

                sample["cameras"][camera_name] = cam_item

            if (
                hasattr(camera_group, "get_image_nearest")
                and args.rgb_name in args.camera_names
                and args.depth_name in args.camera_names
            ):
                nearest: Dict[str, Any] = {}
                try:
                    _rgb, rgb_ts, rgb_wait = wait_latest_image(
                        camera_group,
                        args.rgb_name,
                        args.camera_tries,
                        args.camera_sleep_s,
                    )
                    nearest["rgb_wait"] = rgb_wait
                    if _rgb is None or rgb_ts is None:
                        nearest["ok"] = False
                        nearest["error"] = "cannot call get_image_nearest because RGB frame/timestamp is None"
                    else:
                        depth, depth_ts = camera_group.get_image_nearest(args.depth_name, int(rgb_ts))
                        nearest["ok"] = depth is not None
                        nearest["rgb_ts_ns"] = json_safe(rgb_ts)
                        nearest["depth_ts_ns"] = json_safe(depth_ts)
                        try:
                            nearest["delta_ns"] = abs(int(depth_ts) - int(rgb_ts))
                        except Exception:
                            pass
                        if depth is not None:
                            nearest["depth_stats"] = depth_stats(depth)
                            prefix = frames_dir / f"{sample_idx:03d}_{args.depth_name}_nearest_{depth_ts}"
                            nearest["artifact"] = maybe_save_depth(prefix, depth, None)
                except Exception as exc:
                    nearest.update(
                        {
                            "ok": False,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                sample["nearest_depth_to_rgb"] = nearest

            section["samples"].append(sample)
            time.sleep(args.interval)
    except Exception as exc:
        section.update(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if camera_group is not None:
            for name in ("close", "shutdown", "stop"):
                if hasattr(camera_group, name):
                    section[f"{name}_call"] = call_and_capture(name, getattr(camera_group, name))
                    break


def parse_pose_value(value: Any) -> Optional[Dict[str, float]]:
    if value is None:
        return None
    if isinstance(value, dict):
        if all(k in value for k in ("x", "y", "z", "qx", "qy", "qz", "qw")):
            return {k: float(value[k]) for k in ("x", "y", "z", "qx", "qy", "qz", "qw")}
        pos = value.get("position") or value.get("pos")
        ori = value.get("orientation") or value.get("quat") or value.get("quaternion")
        if isinstance(pos, dict) and isinstance(ori, dict):
            keys = ("x", "y", "z")
            qkeys = ("qx", "qy", "qz", "qw")
            if all(k in pos for k in keys) and all(k in ori for k in keys + ("w",)):
                return {
                    "x": float(pos["x"]),
                    "y": float(pos["y"]),
                    "z": float(pos["z"]),
                    "qx": float(ori.get("qx", ori["x"])),
                    "qy": float(ori.get("qy", ori["y"])),
                    "qz": float(ori.get("qz", ori["z"])),
                    "qw": float(ori.get("qw", ori["w"])),
                }
        for nested_key in ("pose", "frame_pose", "transform"):
            parsed = parse_pose_value(value.get(nested_key))
            if parsed is not None:
                return parsed
    if isinstance(value, (list, tuple)) and len(value) >= 7:
        vals = [float(v) for v in value[:7]]
        return {
            "x": vals[0],
            "y": vals[1],
            "z": vals[2],
            "qx": vals[3],
            "qy": vals[4],
            "qz": vals[5],
            "qw": vals[6],
        }
    return None


def iter_named_frames(status: Any) -> Iterable[Tuple[str, Any]]:
    if not isinstance(status, dict):
        return
    frames = status.get("frames")
    if isinstance(frames, dict):
        for name, value in frames.items():
            yield str(name), value
    elif isinstance(frames, list):
        for idx, item in enumerate(frames):
            if isinstance(item, dict):
                name = (
                    item.get("name")
                    or item.get("frame_name")
                    or item.get("link_name")
                    or item.get("id")
                    or f"frame_{idx}"
                )
                yield str(name), item
            else:
                yield f"frame_{idx}", item

    names = status.get("frame_names")
    poses = status.get("frame_poses")
    if isinstance(names, list) and isinstance(poses, list):
        for name, pose in zip(names, poses):
            yield str(name), pose


def find_pose_in_motion_status(status: Any, hint: str) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
    hint_lower = hint.lower()
    fallback: Tuple[Optional[str], Optional[Dict[str, float]]] = (None, None)
    for name, value in iter_named_frames(status):
        pose = parse_pose_value(value)
        if pose is None:
            continue
        if fallback[1] is None:
            fallback = (name, pose)
        if hint_lower and hint_lower in name.lower():
            return name, pose
    return fallback


def collect_robot(args: argparse.Namespace, report: Dict[str, Any]) -> None:
    section: Dict[str, Any] = {"ok": False, "state_samples": []}
    report["robot"] = section
    robot = None
    controller = None
    try:
        RobotDds = import_symbol(args.robot_dds_symbol)
        RobotController = import_symbol(args.robot_controller_symbol)
        section["imports"] = {
            "RobotDds": args.robot_dds_symbol,
            "RobotController": args.robot_controller_symbol,
        }

        robot = RobotDds()
        controller = RobotController()
        section["ok"] = True
        time.sleep(args.warmup)

        section["robot_public_methods"] = list_public_methods(robot)
        section["controller_public_methods"] = list_public_methods(controller)
        section["robot_method_signatures"] = {
            name: method_signature(robot, name)
            for name in (
                "head_joint_states",
                "waist_joint_states",
                "arm_joint_states",
                "gripper_states",
                "move_gripper",
                "move_arm",
                "move_head",
                "move_waist",
                "shutdown",
            )
        }
        section["controller_method_signatures"] = {
            name: method_signature(controller, name)
            for name in (
                "get_motion_status",
                "set_motion_control_mode",
                "set_motion_stop",
                "set_end_effector_pose_control",
                "trajectory_tracking_control",
            )
        }

        state_methods = [
            "body_pose_joint_states",
            "head_joint_states",
            "waist_joint_states",
            "arm_joint_states",
            "gripper_states",
            "hand_joint_states",
            "hand_force_states",
            "whole_body_status",
        ]

        latest_status = None
        for sample_idx in range(args.samples):
            sample: Dict[str, Any] = {"sample_idx": sample_idx, "states": {}}
            for method_name in state_methods:
                if not hasattr(robot, method_name):
                    sample["states"][method_name] = {"exists": False}
                    continue
                started = time.time()
                try:
                    value = getattr(robot, method_name)()
                    sample["states"][method_name] = {
                        "ok": True,
                        "name": method_name,
                        "duration_s": time.time() - started,
                        "value": state_tuple_to_dict(value),
                    }
                except Exception as exc:
                    sample["states"][method_name] = {
                        "ok": False,
                        "name": method_name,
                        "duration_s": time.time() - started,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }

            if hasattr(controller, "get_motion_status"):
                started = time.time()
                try:
                    latest_status = controller.get_motion_status()
                    sample["motion_status"] = {
                        "ok": True,
                        "name": "get_motion_status",
                        "duration_s": time.time() - started,
                        "value": json_safe(latest_status),
                    }
                except Exception as exc:
                    sample["motion_status"] = {
                        "ok": False,
                        "name": "get_motion_status",
                        "duration_s": time.time() - started,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }

            ts_ns = int(time.time() * 1e9)
            nearest_methods = [
                "head_joint_states_nearest",
                "waist_joint_states_nearest",
                "arm_joint_states_nearest",
                "gripper_joint_states_nearest",
                "hand_joint_states_nearest",
            ]
            sample["nearest_states"] = {}
            for method_name in nearest_methods:
                if hasattr(robot, method_name):
                    sample["nearest_states"][method_name] = call_and_capture(
                        f"{method_name}({ts_ns})", getattr(robot, method_name), ts_ns
                    )

            section["state_samples"].append(sample)
            time.sleep(args.interval)

        control_allowed = args.enable_control and args.confirm_control == "RUN_CONTROL"
        section["control_probe"] = {
            "enabled": bool(args.enable_control),
            "allowed": bool(control_allowed),
            "note": "No control command is sent unless --enable-control --confirm-control RUN_CONTROL is used.",
        }
        if args.enable_control and not control_allowed:
            section["control_probe"]["skipped_reason"] = "missing --confirm-control RUN_CONTROL"

        if control_allowed and args.test_gripper_hold and hasattr(robot, "gripper_states"):
            gripper_value, _ts = robot.gripper_states()
            section["control_probe"]["gripper_hold"] = call_and_capture(
                "move_gripper(current)", robot.move_gripper, gripper_value
            )

        if control_allowed and args.test_ee_hold:
            ee_item: Dict[str, Any] = {}
            section["control_probe"]["ee_hold"] = ee_item
            if not args.ee_frame_hint:
                ee_item["ok"] = False
                ee_item["skipped_reason"] = "missing --ee-frame-hint; inspect motion_status first, then rerun with an explicit frame hint"
            else:
                if latest_status is None and hasattr(controller, "get_motion_status"):
                    latest_status = controller.get_motion_status()
                frame_name, pose = find_pose_in_motion_status(latest_status, args.ee_frame_hint)
                ee_item["source_frame"] = frame_name
                ee_item["pose"] = pose
                if pose is None:
                    ee_item["ok"] = False
                    ee_item["error"] = "could not parse a 7D pose from motion_status"
                elif not hasattr(controller, "set_end_effector_pose_control"):
                    ee_item["ok"] = False
                    ee_item["error"] = "controller has no set_end_effector_pose_control"
                else:
                    kwargs: Dict[str, Any] = {
                        "lifetime": args.ee_lifetime,
                        "control_group": [f"{args.control_side}_arm"],
                    }
                    if args.control_side == "left":
                        kwargs["left_pose"] = pose
                    else:
                        kwargs["right_pose"] = pose
                    ee_item["call"] = call_and_capture(
                        "set_end_effector_pose_control(hold_current)",
                        controller.set_end_effector_pose_control,
                        **kwargs,
                    )
    except Exception as exc:
        section.update(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if robot is not None and hasattr(robot, "shutdown"):
            section["robot_shutdown"] = call_and_capture("robot.shutdown", robot.shutdown)


def parse_object_prompts(entries: List[str]) -> Dict[str, str]:
    if not entries:
        return {
            "obj1": "piece of bread .",
            "obj2": "a plate .",
        }
    prompts: Dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"--object-prompt must be KEY=TEXT, got: {entry}")
        key, text = entry.split("=", 1)
        prompts[key.strip()] = text.strip()
    return prompts


def collect_object_pose(args: argparse.Namespace, report: Dict[str, Any], out_dir: Path) -> None:
    section: Dict[str, Any] = {"ok": False}
    report["object_pose"] = section
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "inference"))
        sys.path.insert(0, str(PROJECT_ROOT))
        from G1Camera import G1HeadRGBDCamera
        from object_pose_rgbd import RGBDObjectPoseEstimator

        camera = G1HeadRGBDCamera(args.g1_camera_cfg)
        try:
            frame = camera.get_frame()
            section["frame"] = {
                "rgb": json_safe(frame.rgb),
                "depth_m": json_safe(frame.depth_m),
                "K": json_safe(frame.K),
            }
            frame_dir = ensure_dir(out_dir / "object_pose_frame")
            if np is not None:
                np.save(frame_dir / "rgb_bgr.npy", frame.rgb)
                np.save(frame_dir / "depth_m.npy", frame.depth_m)
                np.save(frame_dir / "K.npy", frame.K)

            cfg = {
                "object_prompts": parse_object_prompts(args.object_prompt),
                "dinosam_cfg_path": args.dinosam_cfg,
                "anchor_key": args.anchor_key,
                "pose_method": args.pose_method,
            }
            section["cfg"] = cfg
            estimator = RGBDObjectPoseEstimator(cfg)
            objs = estimator.estimate([frame])
            section["ok"] = True
            section["objects"] = {
                key: {
                    "T_in_cam": obj.T_in_cam,
                    "kpts_local": obj.kpts_local,
                }
                for key, obj in objs.items()
            }
        finally:
            camera.close()
    except Exception as exc:
        section.update(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


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
    started = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return {
            "ok": True,
            "url": upload_url,
            "status": resp.status,
            "duration_s": time.time() - started,
            "response": body,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect G1 diagnostics for HumanEgo.")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "g1_diagnostics_runs"))
    parser.add_argument("--tag", default="", help="Optional tag included in output folder name.")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--warmup", type=float, default=1.0)

    parser.add_argument("--skip-camera", action="store_true")
    parser.add_argument("--camera-names", nargs="+", default=["head", "head_depth"])
    parser.add_argument("--rgb-name", default="head")
    parser.add_argument("--depth-name", default="head_depth")
    parser.add_argument("--cosine-camera-symbol", default="a2d_sdk.robot.CosineCamera")
    parser.add_argument("--camera-warmup-s", type=float, default=2.0)
    parser.add_argument("--camera-tries", type=int, default=30)
    parser.add_argument("--camera-sleep-s", type=float, default=0.1)

    parser.add_argument("--skip-robot", action="store_true")
    parser.add_argument("--robot-dds-symbol", default="a2d_sdk.robot.RobotDds")
    parser.add_argument("--robot-controller-symbol", default="a2d_sdk.robot.RobotController")

    parser.add_argument("--g1-parameter-py", default=str(DEFAULT_G1_PARAMETER_PY))
    parser.add_argument("--parameter-camera-name", default="head")

    parser.add_argument("--run-object-pose", action="store_true")
    parser.add_argument("--g1-camera-cfg", default=str(DEFAULT_G1_CAMERA_CFG))
    parser.add_argument("--dinosam-cfg", default=str(DEFAULT_DINOSAM_CFG))
    parser.add_argument("--object-prompt", action="append", default=[], help='Repeatable, e.g. obj1="piece of bread ."')
    parser.add_argument("--anchor-key", default="obj1")
    parser.add_argument("--pose-method", default="pca1", choices=["pca1", "pca2"])

    parser.add_argument("--upload-url", default="", help="Example: http://192.168.1.10:8765/upload")

    parser.add_argument("--enable-control", action="store_true")
    parser.add_argument("--confirm-control", default="")
    parser.add_argument("--test-gripper-hold", action="store_true")
    parser.add_argument("--test-ee-hold", action="store_true")
    parser.add_argument("--control-side", choices=["left", "right"], default="right")
    parser.add_argument("--ee-frame-hint", default="")
    parser.add_argument("--ee-lifetime", type=float, default=0.2)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    tag = f"_{args.tag}" if args.tag else ""
    run_dir = ensure_dir(Path(args.out_dir).expanduser().resolve() / f"g1_diag_{utc_stamp()}{tag}")

    report: Dict[str, Any] = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
            "cwd": os.getcwd(),
            "project_root": str(PROJECT_ROOT),
            "argv": sys.argv,
        },
        "args": vars(args),
    }
    report["runtime_dependencies"] = collect_runtime_dependencies(args)

    a2d_status = report["runtime_dependencies"]["required_g1"].get("a2d_sdk.robot", {})
    if not a2d_status.get("ok") and (not args.skip_camera or not args.skip_robot):
        print(
            "[g1_collect_diagnostics] WARNING: cannot import a2d_sdk.robot. "
            "Run this script inside the G1 SDK Python environment, or source the "
            "SDK env script before running. The failure will be saved in diagnostics.json.",
            file=sys.stderr,
        )

    collect_parameters(args, report)
    if not args.skip_camera:
        collect_camera(args, report, run_dir)
    if not args.skip_robot:
        collect_robot(args, report)
    if args.run_object_pose:
        collect_object_pose(args, report, run_dir)

    report_path = run_dir / "diagnostics.json"
    write_json(report_path, report)
    zip_path = make_zip(run_dir)

    upload_result = None
    if args.upload_url:
        try:
            upload_result = upload_zip(zip_path, args.upload_url)
        except Exception as exc:
            upload_result = {
                "ok": False,
                "url": args.upload_url,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        write_json(run_dir / "upload_result.json", upload_result)
        # Rebuild zip so upload_result is included for local inspection too.
        zip_path = make_zip(run_dir)

    print(json.dumps({"run_dir": str(run_dir), "zip_path": str(zip_path), "upload": upload_result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
