#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render the real G1 Omnipicker gripper mesh on HumanEgo serve_bread frames.

This is the next step after the 2D proxy renderer: it uses the released G1 URDF
Omnipicker visual meshes and places `gripper_r_center_link` at the HumanEgo TCP
pose. It intentionally renders only the gripper for now; full arm rendering needs
G1 right-arm IK and a shoulder/base placement policy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MPS_PATH = Path("/data/wangk/data/serve_bread/aria/mps_serve_bread_006_vrs")
DEFAULT_G1_ZIP = PROJECT_ROOT / "G1" / "G1_URDF_Omnipicker.zip"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "render_g1_mesh" / "g1_gripper_006_tcp_pz.mp4"

NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}


@dataclass
class MeshPart:
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    color_bgr: tuple[int, int, int]


@dataclass
class RenderMesh:
    vertices_link: np.ndarray
    faces: np.ndarray
    face_colors_bgr: np.ndarray


@dataclass
class FrameInputs:
    frame_dir: Path
    training: dict
    aria_hands: dict | None
    background_bgr: np.ndarray
    mask_arm: np.ndarray | None


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


def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def T_from_rpy_xyz(rpy: Iterable[float], xyz: Iterable[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_to_R(*[float(v) for v in rpy])
    T[:3, 3] = np.asarray(list(xyz), dtype=np.float64)
    return T


def axis_angle_T(axis: Iterable[float], angle: float) -> np.ndarray:
    axis = np.asarray(list(axis), dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return np.eye(4, dtype=np.float64)
    axis = axis / norm
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    R = np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


def find_rgb_frame_dirs(mps_path: Path) -> list[Path]:
    all_data = mps_path / "preprocess" / "all_data"
    if not all_data.is_dir():
        raise FileNotFoundError(f"Missing all_data directory: {all_data}")
    return sorted(
        [p for p in all_data.iterdir() if p.is_dir() and p.name.isdigit() and (p / "rgb.png").exists()],
        key=lambda p: int(p.name),
    )


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


def load_background(frame_dir: Path, mode: str, inpaint_radius: float, dilate_px: int) -> tuple[np.ndarray, np.ndarray | None]:
    rgb_bgr = imread_required(frame_dir / "rgb.png")
    mask_path = frame_dir / "mask_arm.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None

    if mode == "rgb":
        return rgb_bgr.copy(), mask
    if mode == "woarm-kpts":
        bg_path = frame_dir / "rgb_WoArm_WArmObjKpts.png"
        return imread_required(bg_path), mask
    if mode != "inpaint":
        raise ValueError(f"Unknown background mode: {mode}")
    if mask is None:
        return rgb_bgr.copy(), None

    mask_u8 = (mask > 0).astype(np.uint8) * 255
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        mask_u8 = cv2.dilate(mask_u8, np.ones((k, k), np.uint8), iterations=1)
    return cv2.inpaint(rgb_bgr, mask_u8, inpaint_radius, cv2.INPAINT_TELEA), mask_u8


def reconstruct_training_from_aria(aria_hands: dict | None, camera_config: dict | None) -> dict | None:
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
    except Exception:
        return None
    return {
        "metadata": {"k": camera_config["k"], "c2w": c2w},
        "entities": {
            "hands": {
                "right": {
                    "T_hand_to_world": T_hand_to_world.tolist(),
                    "grasp": 1.0 if hand_r.get("grasp_state", False) else 0.0,
                }
            }
        },
    }


def load_frame(frame_dir: Path, args: argparse.Namespace, camera_config: dict | None) -> FrameInputs | None:
    aria_hands = read_json(frame_dir / "aria_hands.json")
    training = read_json(frame_dir / "training_data.json")
    if training is None:
        training = reconstruct_training_from_aria(aria_hands, camera_config)
    if training is None:
        if not args.include_all_rgb_frames:
            return None
        training = {}
    bg, mask = load_background(frame_dir, args.background, args.inpaint_radius, args.mask_dilate)
    return FrameInputs(frame_dir=frame_dir, training=training, aria_hands=aria_hands, background_bgr=bg, mask_arm=mask)


def fixed_T_hand_in_tcp() -> np.ndarray:
    return np.asarray(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def T_tcp_world_from_training(training: dict) -> tuple[np.ndarray, float] | tuple[None, None]:
    try:
        hand = training["entities"]["hands"]["right"]
        T_hand_w = np.asarray(hand["T_hand_to_world"], dtype=np.float64).reshape(4, 4)
        T_tcp_w = T_hand_w @ np.linalg.inv(fixed_T_hand_in_tcp())
        return T_tcp_w, float(hand.get("grasp", 0.0))
    except Exception:
        return None, None


def parse_float_list(text: str | None) -> list[float]:
    if not text:
        return []
    return [float(x) for x in text.split()]


def parse_dae_mesh(path: Path, color_bgr: tuple[int, int, int]) -> MeshPart:
    tree = ET.parse(path)
    root = tree.getroot()
    arrays: dict[str, np.ndarray] = {}
    vertices_sources: dict[str, str] = {}

    for source in root.findall(".//c:source", NS):
        source_id = source.attrib.get("id")
        float_array = source.find("c:float_array", NS)
        accessor = source.find("c:technique_common/c:accessor", NS)
        if source_id and float_array is not None and accessor is not None:
            vals = np.asarray(parse_float_list(float_array.text), dtype=np.float64)
            stride = int(accessor.attrib.get("stride", "1"))
            if stride > 0 and len(vals) >= stride:
                arrays[source_id] = vals.reshape(-1, stride)

    for vertices in root.findall(".//c:vertices", NS):
        vertices_id = vertices.attrib.get("id")
        inp = vertices.find("c:input[@semantic='POSITION']", NS)
        if vertices_id and inp is not None:
            vertices_sources[vertices_id] = inp.attrib["source"].lstrip("#")

    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vertex_offset = 0
    for mesh in root.findall(".//c:mesh", NS):
        for triangles in mesh.findall("c:triangles", NS):
            inputs = triangles.findall("c:input", NS)
            vertex_input = None
            max_offset = 0
            for inp in inputs:
                offset = int(inp.attrib.get("offset", "0"))
                max_offset = max(max_offset, offset)
                if inp.attrib.get("semantic") == "VERTEX":
                    vertex_input = inp
            if vertex_input is None:
                continue
            source_id = vertex_input.attrib["source"].lstrip("#")
            pos_id = vertices_sources.get(source_id, source_id)
            vertices = arrays[pos_id][:, :3]
            p = np.asarray([int(x) for x in (triangles.findtext("c:p", default="", namespaces=NS) or "").split()], dtype=np.int64)
            stride = max_offset + 1
            if len(p) < stride * 3:
                continue
            face_indices = p.reshape(-1, stride * 3)
            faces = face_indices[:, [int(vertex_input.attrib.get("offset", "0")) + stride * i for i in range(3)]]
            all_vertices.append(vertices)
            all_faces.append(faces + vertex_offset)
            vertex_offset += len(vertices)

    if not all_vertices or not all_faces:
        raise ValueError(f"No triangles parsed from {path}")
    vertices = np.concatenate(all_vertices, axis=0)
    faces = np.concatenate(all_faces, axis=0).astype(np.int32)
    return MeshPart(path.stem, vertices, faces, color_bgr)


def extract_g1_assets(zip_path: Path) -> Path:
    extract_dir = Path(tempfile.mkdtemp(prefix="g1_omnipicker_"))
    with zipfile.ZipFile(zip_path) as z:
        wanted = [n for n in z.namelist() if n.endswith(".dae")]
        for name in wanted:
            z.extract(name, extract_dir)
    return extract_dir / "G1_URDF_Omnipicker"


def gripper_joint_angle(grasp: float, open_angle: float, closed_angle: float) -> float:
    g = float(np.clip(grasp, 0.0, 1.0))
    return (1.0 - g) * open_angle + g * closed_angle


def build_gripper_mesh(asset_root: Path, grasp: float) -> RenderMesh:
    mesh_dir = asset_root / "meshes" / "omnipicker"
    colors = {
        "gripper_base_link": (120, 128, 132),
        "inner_link1": (80, 98, 220),
        "inner_link2": (80, 98, 220),
        "inner_link3": (88, 150, 96),
        "inner_link4": (150, 158, 162),
        "outer_link1": (80, 98, 220),
        "outer_link2": (80, 98, 220),
        "outer_link3": (88, 150, 96),
        "outer_link4": (150, 158, 162),
    }
    parts = {name: parse_dae_mesh(mesh_dir / f"{name}.dae", color) for name, color in colors.items()}

    open_angle = 0.62
    closed_angle = 0.10
    q_outer = gripper_joint_angle(grasp, open_angle, closed_angle)
    q_inner = -q_outer

    T_identity = np.eye(4, dtype=np.float64)
    joint_origin = {
        "inner_link1": T_from_rpy_xyz([-2.9951, -math.pi / 2.0, -0.15964], [0, -0.0195, 0.0565]),
        "inner_link2": T_from_rpy_xyz([-2.9951, -math.pi / 2.0, -0.15964], [0, -0.021633, 0.07387]),
        "outer_link1": T_from_rpy_xyz([-2.9951, -math.pi / 2.0, -0.15964], [0, 0.0195, 0.0565]),
        "outer_link2": T_from_rpy_xyz([-2.9951, -math.pi / 2.0, -0.15964], [0, 0.021633, 0.07387]),
        "inner_link3": T_from_rpy_xyz([0, 0, 0], [0.030852, 0.018551, 0]),
        "inner_link4": T_from_rpy_xyz([0, 0, 0], [0.018118, -0.01574, 0]),
        "outer_link3": T_from_rpy_xyz([0, 0, 0], [0.030852, -0.018551, 0]),
        "outer_link4": T_from_rpy_xyz([0, 0, 0], [0.018118, 0.01574, 0]),
    }
    R_inner = axis_angle_T([0, 0, -1], q_inner)
    R_outer = axis_angle_T([0, 0, -1], q_outer)
    T = {
        "gripper_base_link": T_identity,
        "inner_link1": joint_origin["inner_link1"] @ R_inner,
        "inner_link2": joint_origin["inner_link2"],
        "outer_link1": joint_origin["outer_link1"] @ R_outer,
        "outer_link2": joint_origin["outer_link2"],
    }
    T["inner_link3"] = T["inner_link1"] @ joint_origin["inner_link3"]
    T["inner_link4"] = T["inner_link3"] @ joint_origin["inner_link4"]
    T["outer_link3"] = T["outer_link1"] @ joint_origin["outer_link3"]
    T["outer_link4"] = T["outer_link3"] @ joint_origin["outer_link4"]

    vertices_all: list[np.ndarray] = []
    faces_all: list[np.ndarray] = []
    colors_all: list[np.ndarray] = []
    offset = 0
    for name in [
        "gripper_base_link",
        "inner_link1",
        "inner_link2",
        "inner_link3",
        "inner_link4",
        "outer_link1",
        "outer_link2",
        "outer_link3",
        "outer_link4",
    ]:
        part = parts[name]
        vertices_h = np.column_stack([part.vertices, np.ones(len(part.vertices))])
        vertices = (T[name] @ vertices_h.T).T[:, :3]
        vertices_all.append(vertices)
        faces_all.append(part.faces + offset)
        colors_all.append(np.tile(np.asarray(part.color_bgr, dtype=np.uint8), (len(part.faces), 1)))
        offset += len(vertices)

    return RenderMesh(
        vertices_link=np.concatenate(vertices_all, axis=0),
        faces=np.concatenate(faces_all, axis=0).astype(np.int32),
        face_colors_bgr=np.concatenate(colors_all, axis=0),
    )


def decimate_mesh_faces(mesh: RenderMesh, max_faces: int | None) -> RenderMesh:
    if max_faces is None or max_faces <= 0 or len(mesh.faces) <= max_faces:
        return mesh
    centers = np.mean(mesh.vertices_link[mesh.faces], axis=1)
    # Keep a deterministic spatially spread sample instead of taking the first N
    # faces, which often over-represents one DAE material block.
    keys = (
        np.sin(centers[:, 0] * 4096.0)
        + np.sin(centers[:, 1] * 8192.0 + 0.7)
        + np.sin(centers[:, 2] * 16384.0 + 1.3)
    )
    keep = np.argsort(keys)[:max_faces]
    return RenderMesh(
        vertices_link=mesh.vertices_link,
        faces=mesh.faces[keep],
        face_colors_bgr=mesh.face_colors_bgr[keep],
    )


def transform_mesh_to_tcp(mesh: RenderMesh, T_tcp_world: np.ndarray, scale: float) -> np.ndarray:
    # URDF fixed transform: gripper_base_link -> gripper_r_center_link.
    T_center_in_base = T_from_rpy_xyz([0.0, 0.0, -math.pi / 2.0], [0.0, 0.0, 0.14308])
    T_base_world = T_tcp_world @ np.linalg.inv(T_center_in_base)
    vertices = mesh.vertices_link * scale
    vertices_h = np.column_stack([vertices, np.ones(len(vertices))])
    return (T_base_world @ vertices_h.T).T[:, :3]


def project_world_vertices(vertices_world: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices_h = np.column_stack([vertices_world, np.ones(len(vertices_world))])
    vertices_cam = (np.linalg.inv(c2w) @ vertices_h.T).T[:, :3]
    z = vertices_cam[:, 2]
    uv = np.full((len(vertices_cam), 2), np.nan, dtype=np.float64)
    valid = z > 1e-6
    if np.any(valid):
        proj = (K @ vertices_cam[valid].T).T
        uv[valid] = proj[:, :2] / proj[:, 2:3]
    return uv, z, vertices_cam


def shade_color(color_bgr: np.ndarray, normal_cam: np.ndarray) -> tuple[int, int, int]:
    n = normal_cam / (np.linalg.norm(normal_cam) + 1e-9)
    light = np.asarray([-0.25, -0.45, -1.0], dtype=np.float64)
    light = light / np.linalg.norm(light)
    intensity = 0.42 + 0.58 * max(0.0, float(np.dot(n, -light)))
    color = np.clip(color_bgr.astype(np.float64) * intensity + 18.0, 0, 255)
    return tuple(int(v) for v in color)


def rasterize_mesh(
    background: np.ndarray,
    mesh: RenderMesh,
    vertices_world: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    alpha: float,
) -> np.ndarray:
    out = background.copy()
    h, w = out.shape[:2]
    uv, z, vertices_cam = project_world_vertices(vertices_world, K, c2w)
    depth = np.full((h, w), np.inf, dtype=np.float32)

    face_centers_z = np.nanmean(z[mesh.faces], axis=1)
    order = np.argsort(face_centers_z)[::-1]
    for face_idx in order:
        face = mesh.faces[face_idx]
        pts = uv[face]
        zs = z[face]
        if not np.all(np.isfinite(pts)) or np.any(zs <= 1e-6):
            continue
        if np.max(pts[:, 0]) < 0 or np.min(pts[:, 0]) >= w or np.max(pts[:, 1]) < 0 or np.min(pts[:, 1]) >= h:
            continue

        tri_cam = vertices_cam[face]
        normal = np.cross(tri_cam[1] - tri_cam[0], tri_cam[2] - tri_cam[0])
        if np.linalg.norm(normal) < 1e-10:
            continue
        color = shade_color(mesh.face_colors_bgr[face_idx], normal)

        poly = np.round(pts).astype(np.int32)
        x0 = max(0, int(np.floor(np.min(pts[:, 0]))))
        x1 = min(w - 1, int(np.ceil(np.max(pts[:, 0]))))
        y0 = max(0, int(np.floor(np.min(pts[:, 1]))))
        y1 = min(h - 1, int(np.ceil(np.max(pts[:, 1]))))
        if x1 < x0 or y1 < y0:
            continue

        mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
        cv2.fillConvexPoly(mask, poly - np.asarray([x0, y0]), 255, cv2.LINE_AA)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        px = xs.astype(np.float64) + x0
        py = ys.astype(np.float64) + y0
        denom = (
            (pts[1, 1] - pts[2, 1]) * (pts[0, 0] - pts[2, 0])
            + (pts[2, 0] - pts[1, 0]) * (pts[0, 1] - pts[2, 1])
        )
        if abs(float(denom)) < 1e-9:
            continue
        w0 = ((pts[1, 1] - pts[2, 1]) * (px - pts[2, 0]) + (pts[2, 0] - pts[1, 0]) * (py - pts[2, 1])) / denom
        w1 = ((pts[2, 1] - pts[0, 1]) * (px - pts[2, 0]) + (pts[0, 0] - pts[2, 0]) * (py - pts[2, 1])) / denom
        w2 = 1.0 - w0 - w1
        pix_z = w0 * zs[0] + w1 * zs[1] + w2 * zs[2]

        yy = ys + y0
        xx = xs + x0
        update = pix_z < depth[yy, xx]
        if not np.any(update):
            continue
        yy = yy[update]
        xx = xx[update]
        depth[yy, xx] = pix_z[update].astype(np.float32)
        old = out[yy, xx].astype(np.float64)
        new = np.asarray(color, dtype=np.float64)
        out[yy, xx] = np.clip(alpha * new + (1.0 - alpha) * old, 0, 255).astype(np.uint8)

    return out


def rasterize_mesh_painter(
    background: np.ndarray,
    mesh: RenderMesh,
    vertices_world: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    alpha: float,
) -> np.ndarray:
    overlay = background.copy()
    h, w = overlay.shape[:2]
    uv, z, vertices_cam = project_world_vertices(vertices_world, K, c2w)

    face_centers_z = np.nanmean(z[mesh.faces], axis=1)
    order = np.argsort(face_centers_z)[::-1]
    for face_idx in order:
        face = mesh.faces[face_idx]
        pts = uv[face]
        zs = z[face]
        if not np.all(np.isfinite(pts)) or np.any(zs <= 1e-6):
            continue
        if np.max(pts[:, 0]) < 0 or np.min(pts[:, 0]) >= w or np.max(pts[:, 1]) < 0 or np.min(pts[:, 1]) >= h:
            continue

        tri_cam = vertices_cam[face]
        normal = np.cross(tri_cam[1] - tri_cam[0], tri_cam[2] - tri_cam[0])
        if np.linalg.norm(normal) < 1e-10:
            continue
        color = shade_color(mesh.face_colors_bgr[face_idx], normal)
        cv2.fillConvexPoly(overlay, np.round(pts).astype(np.int32), color, cv2.LINE_AA)

    return cv2.addWeighted(overlay, alpha, background, 1.0 - alpha, 0)


def make_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def render_frame(frame: FrameInputs, meshes_by_grasp: dict[str, RenderMesh], args: argparse.Namespace) -> np.ndarray:
    T_tcp_w, grasp = T_tcp_world_from_training(frame.training)
    if T_tcp_w is None:
        return frame.background_bgr
    K = np.asarray(frame.training["metadata"]["k"], dtype=np.float64).reshape(3, 3)
    c2w = np.asarray(frame.training["metadata"]["c2w"], dtype=np.float64).reshape(4, 4)
    mesh = meshes_by_grasp["closed"] if grasp and grasp > 0.5 else meshes_by_grasp["open"]
    vertices_world = transform_mesh_to_tcp(mesh, T_tcp_w, args.mesh_scale)
    if args.rasterizer == "zbuffer":
        return rasterize_mesh(frame.background_bgr, mesh, vertices_world, K, c2w, args.overlay_alpha)
    return rasterize_mesh_painter(frame.background_bgr, mesh, vertices_world, K, c2w, args.overlay_alpha)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mps-path", type=Path, default=DEFAULT_MPS_PATH)
    parser.add_argument("--g1-zip", type=Path, default=DEFAULT_G1_ZIP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--start-frame-id", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--include-all-rgb-frames", action="store_true")
    parser.add_argument("--use-camera-config-fps", action="store_true")
    parser.add_argument("--background", choices=["inpaint", "rgb", "woarm-kpts"], default="inpaint")
    parser.add_argument("--inpaint-radius", type=float, default=5.0)
    parser.add_argument("--mask-dilate", type=int, default=5)
    parser.add_argument("--overlay-alpha", type=float, default=0.96)
    parser.add_argument("--rasterizer", choices=["painter", "zbuffer"], default="painter")
    parser.add_argument("--mesh-scale", type=float, default=1.0, help="Debug scale multiplier for the gripper mesh.")
    parser.add_argument("--max-faces", type=int, default=4500, help="Deterministically sample mesh faces for fast preview rendering. Use 0 for all faces.")
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

    asset_root = extract_g1_assets(args.g1_zip)
    meshes_by_grasp = {
        "open": decimate_mesh_faces(build_gripper_mesh(asset_root, grasp=0.0), args.max_faces),
        "closed": decimate_mesh_faces(build_gripper_mesh(asset_root, grasp=1.0), args.max_faces),
    }

    frame_dirs = select_frame_dirs(
        find_rgb_frame_dirs(args.mps_path),
        start_index=args.start_index,
        start_frame_id=args.start_frame_id,
        num_frames=args.num_frames,
        stride=args.stride,
    )
    if not frame_dirs:
        raise RuntimeError("No frames selected.")

    writer: cv2.VideoWriter | None = None
    written = 0
    skipped = 0
    for frame_dir in frame_dirs:
        frame = load_frame(frame_dir, args, camera_config)
        if frame is None:
            skipped += 1
            continue
        rendered = render_frame(frame, meshes_by_grasp, args)
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
        "fps": args.fps,
        "background": args.background,
        "renderer": "G1 Omnipicker gripper mesh at HumanEgo TCP pose (tcp +Z / pz convention)",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
