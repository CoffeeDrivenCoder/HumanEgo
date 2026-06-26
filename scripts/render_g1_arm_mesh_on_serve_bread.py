#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render G1 right-arm visual meshes plus Omnipicker gripper on serve_bread.

The official G1 arm visual assets in the provided package are binary FBX files.
This script parses the FBX mesh arrays directly, solves a 7-DoF right-arm IK for
each HumanEgo TCP target, and composites the arm + real Omnipicker gripper over
the inpainted egocentric video.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))

from render_g1_gripper_mesh_on_serve_bread import (  # noqa: E402
    DEFAULT_G1_ZIP,
    DEFAULT_MPS_PATH,
    FrameInputs,
    MeshPart,
    RenderMesh,
    axis_angle_T,
    build_gripper_mesh,
    decimate_mesh_faces,
    find_rgb_frame_dirs,
    load_frame,
    make_writer,
    rasterize_mesh_painter,
    read_camera_config,
    select_frame_dirs,
    T_from_rpy_xyz,
    T_tcp_world_from_training,
)


DEFAULT_OUT = PROJECT_ROOT / "outputs" / "render_g1_arm_mesh" / "g1_right_arm_006_tcp_pz_full.mp4"
T_OPENGL_CAMERA_FROM_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


@dataclass
class FbxNode:
    name: str
    props: list[object]
    children: list["FbxNode"]


@dataclass
class ArmState:
    q: np.ndarray | None = None
    base_pos_cam: np.ndarray | None = None


@dataclass
class GpuRenderer:
    pyrender: object
    trimesh: object
    renderer: object
    width: int
    height: int


ARM_LINK_NAMES = [
    "arm_r_base_link",
    "arm_r_link1",
    "arm_r_link2",
    "arm_r_link3",
    "arm_r_link4",
    "arm_r_link5",
    "arm_r_link6",
    "arm_r_end_link",
]

ARM_MESH_FILES = {
    "arm_r_base_link": "arm_r_base_link.fbx",
    "arm_r_link1": "arm_r_link1.fbx",
    "arm_r_link2": "arm_r_link2.fbx",
    "arm_r_link3": "arm_r_link3.fbx",
    "arm_r_link4": "arm_r_link4.fbx",
    "arm_r_link5": "arm_r_link5.fbx",
    "arm_r_link6": "arm_r_link6.fbx",
    "arm_r_end_link": "arm_r_link7.fbx",
}

JOINT_SPECS = [
    ("arm_r_link1", [0.0, 0.0, 0.0], [0.0, 0.0, 0.188], [0.0, 0.0, 1.0], -3.14, 3.14),
    ("arm_r_link2", [-math.pi / 2.0, 0.0, -math.pi / 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], -1.48, 2.09),
    ("arm_r_link3", [-math.pi / 2.0, -math.pi / 2.0, math.pi], [0.0, -0.305, 0.0], [0.0, 0.0, 1.0], -3.10, 3.10),
    ("arm_r_link4", [-math.pi / 2.0, 0.0, math.pi], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], -1.48, 1.48),
    ("arm_r_link5", [-math.pi / 2.0, 0.0, -math.pi], [0.0, -0.1975, 0.0], [0.0, 0.0, 1.0], -3.10, 3.10),
    ("arm_r_link6", [-math.pi / 2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], -1.74, 1.74),
    ("arm_r_end_link", [math.pi / 2.0, 0.0, 0.0], [0.0, -0.1805, 0.0], [0.0, 0.0, 1.0], -3.10, 3.10),
]

JOINT_LOWER = np.asarray([j[4] for j in JOINT_SPECS], dtype=np.float64)
JOINT_UPPER = np.asarray([j[5] for j in JOINT_SPECS], dtype=np.float64)
HOME_Q = np.asarray([0.0, 0.25, 0.0, -0.65, 0.0, 0.45, 0.0], dtype=np.float64)

T_GRIPPER_BASE_IN_END = T_from_rpy_xyz([0.0, 0.0, -math.pi / 2.0], [0.0, 0.0, 0.0])
T_TCP_IN_GRIPPER_BASE = T_from_rpy_xyz([0.0, 0.0, -math.pi / 2.0], [0.0, 0.0, 0.14308])


def extract_assets(zip_path: Path) -> Path:
    extract_dir = Path(tempfile.mkdtemp(prefix="g1_arm_assets_"))
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith((".fbx", ".dae")):
                z.extract(name, extract_dir)
    return extract_dir / "G1_URDF_Omnipicker"


def read_fbx_property(data: bytes, pos: int) -> tuple[object, int]:
    code = chr(data[pos])
    pos += 1
    if code == "C":
        return bool(data[pos]), pos + 1
    if code == "Y":
        return struct.unpack_from("<h", data, pos)[0], pos + 2
    if code == "I":
        return struct.unpack_from("<i", data, pos)[0], pos + 4
    if code == "F":
        return struct.unpack_from("<f", data, pos)[0], pos + 4
    if code == "D":
        return struct.unpack_from("<d", data, pos)[0], pos + 8
    if code == "L":
        return struct.unpack_from("<q", data, pos)[0], pos + 8
    if code in {"S", "R"}:
        length = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        raw = data[pos : pos + length]
        pos += length
        if code == "S":
            return raw.decode("utf-8", "replace"), pos
        return raw, pos
    if code in {"f", "d", "i", "l", "b", "c"}:
        import zlib

        length, encoding, compressed_length = struct.unpack_from("<III", data, pos)
        pos += 12
        raw = data[pos : pos + compressed_length]
        pos += compressed_length
        if encoding == 1:
            raw = zlib.decompress(raw)
        dtype = {
            "f": "<f4",
            "d": "<f8",
            "i": "<i4",
            "l": "<i8",
            "b": "?",
            "c": "?",
        }[code]
        return np.frombuffer(raw, dtype=np.dtype(dtype), count=length).copy(), pos
    raise ValueError(f"Unsupported FBX property code {code!r} at byte {pos - 1}")


def read_fbx_node(data: bytes, pos: int, use_64: bool) -> tuple[FbxNode | None, int]:
    if use_64:
        end_offset, num_props, _prop_len = struct.unpack_from("<QQQ", data, pos)
        pos += 24
    else:
        end_offset, num_props, _prop_len = struct.unpack_from("<III", data, pos)
        pos += 12
    name_len = data[pos]
    pos += 1
    if end_offset == 0 and num_props == 0 and name_len == 0:
        return None, pos
    name = data[pos : pos + name_len].decode("utf-8", "replace")
    pos += name_len

    props: list[object] = []
    for _ in range(num_props):
        prop, pos = read_fbx_property(data, pos)
        props.append(prop)

    children: list[FbxNode] = []
    while pos < end_offset:
        child, next_pos = read_fbx_node(data, pos, use_64)
        pos = next_pos
        if child is None:
            break
        children.append(child)
    return FbxNode(name=name, props=props, children=children), pos


def parse_fbx_tree(path: Path) -> list[FbxNode]:
    data = path.read_bytes()
    if not data.startswith(b"Kaydara FBX Binary"):
        raise ValueError(f"Only binary FBX is supported: {path}")
    version = struct.unpack_from("<I", data, 23)[0]
    use_64 = version >= 7500
    pos = 27
    nodes: list[FbxNode] = []
    while pos < len(data) - 32:
        node, pos = read_fbx_node(data, pos, use_64)
        if node is None:
            break
        nodes.append(node)
    return nodes


def find_nodes(node: FbxNode, name: str) -> list[FbxNode]:
    out = [node] if node.name == name else []
    for child in node.children:
        out.extend(find_nodes(child, name))
    return out


def child_named(node: FbxNode, name: str) -> FbxNode | None:
    for child in node.children:
        if child.name == name:
            return child
    return None


def parse_fbx_mesh(path: Path, color_bgr: tuple[int, int, int]) -> MeshPart:
    roots = parse_fbx_tree(path)
    geometries: list[FbxNode] = []
    for root in roots:
        geometries.extend(find_nodes(root, "Geometry"))
    if not geometries:
        raise ValueError(f"No Geometry node found in {path}")
    geom = geometries[0]
    vertices_node = child_named(geom, "Vertices")
    poly_node = child_named(geom, "PolygonVertexIndex")
    if vertices_node is None or poly_node is None:
        raise ValueError(f"Missing Vertices or PolygonVertexIndex in {path}")
    vertices = np.asarray(vertices_node.props[0], dtype=np.float64).reshape(-1, 3)
    poly = np.asarray(poly_node.props[0], dtype=np.int64)

    faces: list[list[int]] = []
    cur: list[int] = []
    for idx in poly:
        if idx < 0:
            cur.append(int(-idx - 1))
            if len(cur) >= 3:
                for i in range(1, len(cur) - 1):
                    faces.append([cur[0], cur[i], cur[i + 1]])
            cur = []
        else:
            cur.append(int(idx))
    if not faces:
        raise ValueError(f"No faces parsed from {path}")
    return MeshPart(name=path.stem, vertices=vertices, faces=np.asarray(faces, dtype=np.int32), color_bgr=color_bgr)


def fbx_part_to_mesh(part: MeshPart) -> RenderMesh:
    return RenderMesh(
        vertices_link=part.vertices,
        faces=part.faces,
        face_colors_bgr=np.tile(np.asarray(part.color_bgr, dtype=np.uint8), (len(part.faces), 1)),
    )


def load_arm_meshes(asset_root: Path, max_faces_per_link: int) -> dict[str, RenderMesh]:
    mesh_dir = asset_root / "meshes" / "G1"
    colors = {
        "arm_r_base_link": (118, 124, 126),
        "arm_r_link1": (145, 150, 152),
        "arm_r_link2": (132, 138, 140),
        "arm_r_link3": (150, 154, 156),
        "arm_r_link4": (125, 132, 136),
        "arm_r_link5": (150, 154, 156),
        "arm_r_link6": (126, 134, 138),
        "arm_r_end_link": (160, 164, 166),
    }
    meshes: dict[str, RenderMesh] = {}
    for link_name in ARM_LINK_NAMES:
        part = parse_fbx_mesh(mesh_dir / ARM_MESH_FILES[link_name], colors[link_name])
        meshes[link_name] = decimate_mesh_faces(fbx_part_to_mesh(part), max_faces_per_link)
    return meshes


def build_cylinder_mesh(
    radius: float,
    length: float,
    z0: float,
    color_bgr: tuple[int, int, int],
    segments: int = 48,
    radius_end: float | None = None,
) -> RenderMesh:
    if radius_end is None:
        radius_end = radius
    angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    circle0 = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    circle1 = np.column_stack([radius_end * np.cos(angles), radius_end * np.sin(angles)])
    z_vals = np.asarray([z0, z0 + length], dtype=np.float64)
    vertices = []
    for z, circle in zip(z_vals, [circle0, circle1]):
        vertices.extend([[x, y, z] for x, y in circle])
    vertices.append([0.0, 0.0, z0])
    vertices.append([0.0, 0.0, z0 + length])
    vertices = np.asarray(vertices, dtype=np.float64)

    faces: list[list[int]] = []
    bottom_center = 2 * segments
    top_center = 2 * segments + 1
    for i in range(segments):
        j = (i + 1) % segments
        b0, b1 = i, j
        t0, t1 = i + segments, j + segments
        faces.append([b0, b1, t1])
        faces.append([b0, t1, t0])
        faces.append([bottom_center, b1, b0])
        faces.append([top_center, t0, t1])
    faces_arr = np.asarray(faces, dtype=np.int32)
    return RenderMesh(
        vertices_link=vertices,
        faces=faces_arr,
        face_colors_bgr=np.tile(np.asarray(color_bgr, dtype=np.uint8), (len(faces_arr), 1)),
    )


def arm_fk(q: np.ndarray) -> dict[str, np.ndarray]:
    T = np.eye(4, dtype=np.float64)
    out = {"arm_r_base_link": T.copy()}
    for i, spec in enumerate(JOINT_SPECS):
        child, rpy, xyz, axis, _lo, _hi = spec
        T = T @ T_from_rpy_xyz(rpy, xyz) @ axis_angle_T(axis, float(q[i]))
        out[child] = T.copy()
    out["gripper_r_base_link"] = out["arm_r_end_link"] @ T_GRIPPER_BASE_IN_END
    out["gripper_r_center_link"] = out["gripper_r_base_link"] @ T_TCP_IN_GRIPPER_BASE
    return out


def project_cam_to_uv(point_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    q = K @ point_cam.reshape(3)
    return q[:2] / max(q[2], 1e-9)


def unproject_uv_depth(uv: np.ndarray, depth: float, K: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.asarray([(uv[0] - cx) / fx * depth, (uv[1] - cy) / fy * depth, depth], dtype=np.float64)


def estimate_base_cam(
    frame: FrameInputs,
    T_tcp_world: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    state: ArmState,
    smooth: float,
) -> np.ndarray:
    h, w = frame.background_bgr.shape[:2]
    T_tcp_cam = np.linalg.inv(c2w) @ T_tcp_world
    tcp_cam = T_tcp_cam[:3, 3]
    root_y = float(np.clip(project_cam_to_uv(tcp_cam, K)[1] + 95.0, 0.0, h - 1.0))
    if frame.mask_arm is not None and np.any(frame.mask_arm > 0):
        ys, xs = np.where(frame.mask_arm > 0)
        pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
        x_hi = np.percentile(pts[:, 0], 96)
        root_band = pts[pts[:, 0] >= x_hi]
        if len(root_band) > 10:
            root_y = float(np.median(root_band[:, 1]))

    root_uv = np.asarray([w + 45.0, root_y], dtype=np.float64)
    root_depth = float(np.clip(tcp_cam[2] + 0.28, 0.45, 1.25))
    base_pos = unproject_uv_depth(root_uv, root_depth, K)
    direction = tcp_cam - base_pos
    dist = float(np.linalg.norm(direction))
    if dist < 0.42:
        direction = direction / (dist + 1e-9)
        base_pos = tcp_cam - direction * 0.42
    if state.base_pos_cam is not None:
        base_pos = smooth * state.base_pos_cam + (1.0 - smooth) * base_pos
    state.base_pos_cam = base_pos.copy()
    return base_pos


def look_at_base_pose_cam(base_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    z_axis = target_pos - base_pos
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-9)
    y_ref = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = np.cross(y_ref, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    T[:3, 3] = base_pos
    return T


def solve_arm_ik(T_tcp_target_base: np.ndarray, seed_q: np.ndarray | None, max_nfev: int) -> np.ndarray:
    if seed_q is None:
        seed_q = np.clip(HOME_Q, JOINT_LOWER, JOINT_UPPER)
    seed_q = np.clip(seed_q, JOINT_LOWER, JOINT_UPPER)

    def residual(q: np.ndarray) -> np.ndarray:
        T_cur = arm_fk(q)["gripper_r_center_link"]
        pos_err = T_cur[:3, 3] - T_tcp_target_base[:3, 3]
        rot_err = Rotation.from_matrix(T_cur[:3, :3].T @ T_tcp_target_base[:3, :3]).as_rotvec()
        smooth = q - seed_q
        home = q - HOME_Q
        return np.concatenate([12.0 * pos_err, 0.9 * rot_err, 0.035 * smooth, 0.010 * home])

    result = least_squares(
        residual,
        seed_q,
        bounds=(JOINT_LOWER, JOINT_UPPER),
        max_nfev=max_nfev,
        xtol=1e-4,
        ftol=1e-4,
        gtol=1e-4,
        verbose=0,
    )
    return np.clip(result.x, JOINT_LOWER, JOINT_UPPER)


def combine_meshes(
    link_meshes: dict[str, RenderMesh],
    gripper_mesh: RenderMesh,
    adapter_mesh: RenderMesh | None,
    T_base_world: np.ndarray,
    link_T_base: dict[str, np.ndarray],
    visible_links: set[str],
) -> tuple[RenderMesh, np.ndarray]:
    vertices_world_all: list[np.ndarray] = []
    faces_all: list[np.ndarray] = []
    colors_all: list[np.ndarray] = []
    offset = 0

    for link_name in ARM_LINK_NAMES:
        if link_name not in visible_links:
            continue
        mesh = link_meshes[link_name]
        vertices_h = np.column_stack([mesh.vertices_link, np.ones(len(mesh.vertices_link))])
        vertices_world = (T_base_world @ link_T_base[link_name] @ vertices_h.T).T[:, :3]
        vertices_world_all.append(vertices_world)
        faces_all.append(mesh.faces + offset)
        colors_all.append(mesh.face_colors_bgr)
        offset += len(mesh.vertices_link)

    if adapter_mesh is not None:
        vertices_h = np.column_stack([adapter_mesh.vertices_link, np.ones(len(adapter_mesh.vertices_link))])
        vertices_world = (T_base_world @ link_T_base["gripper_r_base_link"] @ vertices_h.T).T[:, :3]
        vertices_world_all.append(vertices_world)
        faces_all.append(adapter_mesh.faces + offset)
        colors_all.append(adapter_mesh.face_colors_bgr)
        offset += len(adapter_mesh.vertices_link)

    vertices_h = np.column_stack([gripper_mesh.vertices_link, np.ones(len(gripper_mesh.vertices_link))])
    vertices_world = (T_base_world @ link_T_base["gripper_r_base_link"] @ vertices_h.T).T[:, :3]
    vertices_world_all.append(vertices_world)
    faces_all.append(gripper_mesh.faces + offset)
    colors_all.append(gripper_mesh.face_colors_bgr)

    vertices_world_cat = np.concatenate(vertices_world_all, axis=0)
    faces_cat = np.concatenate(faces_all, axis=0).astype(np.int32)
    colors_cat = np.concatenate(colors_all, axis=0)
    mesh = RenderMesh(vertices_link=np.zeros_like(vertices_world_cat), faces=faces_cat, face_colors_bgr=colors_cat)
    return mesh, vertices_world_cat


def init_gpu_renderer(width: int, height: int) -> GpuRenderer:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender
    import trimesh

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    return GpuRenderer(pyrender=pyrender, trimesh=trimesh, renderer=renderer, width=width, height=height)


def render_mesh_gpu(
    background_bgr: np.ndarray,
    mesh: RenderMesh,
    vertices_world: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    gpu: GpuRenderer,
    alpha: float,
) -> np.ndarray:
    pyrender = gpu.pyrender
    trimesh = gpu.trimesh
    h, w = background_bgr.shape[:2]
    if gpu.width != w or gpu.height != h:
        gpu.renderer.delete()
        new_gpu = init_gpu_renderer(w, h)
        gpu.pyrender = new_gpu.pyrender
        gpu.trimesh = new_gpu.trimesh
        gpu.renderer = new_gpu.renderer
        gpu.width = new_gpu.width
        gpu.height = new_gpu.height

    vertices_h = np.column_stack([vertices_world, np.ones(len(vertices_world))])
    vertices_cam_cv = (np.linalg.inv(c2w) @ vertices_h.T).T
    vertices_cam_gl = (T_OPENGL_CAMERA_FROM_OPENCV @ vertices_cam_cv.T).T[:, :3]

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.46, 0.46, 0.46])
    tri = trimesh.Trimesh(vertices=vertices_cam_gl, faces=mesh.faces, process=False)
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.18,
        roughnessFactor=0.72,
        baseColorFactor=(0.54, 0.56, 0.56, float(np.clip(alpha, 0.0, 1.0))),
    )
    scene.add(pyrender.Mesh.from_trimesh(tri, material=material, smooth=True), pose=np.eye(4))
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    scene.add(pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.02, zfar=3.0), pose=np.eye(4))

    light_pose = np.eye(4)
    light_pose[:3, 3] = [0.2, 0.6, 0.8]
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.1), pose=light_pose)
    light_pose2 = np.eye(4)
    light_pose2[:3, 3] = [-0.6, -0.2, 0.8]
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=0.9), pose=light_pose2)

    color_rgba, depth = gpu.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    fg_bgr = color_rgba[..., :3][..., ::-1].astype(np.float32)
    mask = depth > 0
    out = background_bgr.astype(np.float32).copy()
    if np.any(mask):
        # Pyrender returns alpha 255 for covered pixels with this material; use
        # the user alpha for compositing so painter/gpu outputs have comparable opacity.
        out[mask] = alpha * fg_bgr[mask] + (1.0 - alpha) * out[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def render_frame(
    frame: FrameInputs,
    link_meshes: dict[str, RenderMesh],
    gripper_meshes: dict[str, RenderMesh],
    adapter_mesh: RenderMesh | None,
    state: ArmState,
    args: argparse.Namespace,
    gpu: GpuRenderer | None = None,
) -> np.ndarray:
    T_tcp_world, grasp = T_tcp_world_from_training(frame.training)
    if T_tcp_world is None:
        return frame.background_bgr
    K = np.asarray(frame.training["metadata"]["k"], dtype=np.float64).reshape(3, 3)
    c2w = np.asarray(frame.training["metadata"]["c2w"], dtype=np.float64).reshape(4, 4)
    T_tcp_cam = np.linalg.inv(c2w) @ T_tcp_world
    base_pos_cam = estimate_base_cam(frame, T_tcp_world, K, c2w, state, args.base_smooth)
    T_base_cam = look_at_base_pose_cam(base_pos_cam, T_tcp_cam[:3, 3])
    T_base_world = c2w @ T_base_cam
    T_tcp_target_base = np.linalg.inv(T_base_world) @ T_tcp_world

    q = solve_arm_ik(T_tcp_target_base, state.q, args.ik_max_nfev)
    state.q = q
    link_T_base = arm_fk(q)
    gripper_mesh = gripper_meshes["closed"] if grasp and grasp > 0.5 else gripper_meshes["open"]
    mesh, vertices_world = combine_meshes(
        link_meshes,
        gripper_mesh,
        adapter_mesh,
        T_base_world,
        link_T_base,
        args.visible_link_names,
    )
    if args.renderer == "gpu":
        if gpu is None:
            raise RuntimeError("GPU renderer requested but not initialized")
        return render_mesh_gpu(frame.background_bgr, mesh, vertices_world, K, c2w, gpu, args.overlay_alpha)
    return rasterize_mesh_painter(frame.background_bgr, mesh, vertices_world, K, c2w, args.overlay_alpha)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mps-path", type=Path, default=DEFAULT_MPS_PATH)
    parser.add_argument("--g1-zip", type=Path, default=DEFAULT_G1_ZIP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--start-frame-id", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--include-all-rgb-frames", action="store_true")
    parser.add_argument("--use-camera-config-fps", action="store_true")
    parser.add_argument("--background", choices=["inpaint", "rgb", "woarm-kpts"], default="inpaint")
    parser.add_argument("--inpaint-radius", type=float, default=5.0)
    parser.add_argument("--mask-dilate", type=int, default=5)
    parser.add_argument("--overlay-alpha", type=float, default=0.96)
    parser.add_argument("--renderer", choices=["painter", "gpu"], default="painter")
    parser.add_argument("--max-arm-faces-per-link", type=int, default=3000)
    parser.add_argument("--max-gripper-faces", type=int, default=12000)
    parser.add_argument("--ik-max-nfev", type=int, default=45)
    parser.add_argument("--base-smooth", type=float, default=0.85)
    parser.add_argument("--wrist-adapter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wrist-adapter-radius", type=float, default=0.032)
    parser.add_argument("--wrist-adapter-radius-end", type=float, default=0.024)
    parser.add_argument("--wrist-adapter-length", type=float, default=0.155)
    parser.add_argument("--wrist-adapter-z0", type=float, default=-0.160)
    parser.add_argument(
        "--visible-links",
        default="arm_r_link4,arm_r_link5,arm_r_link6,arm_r_end_link",
        help="Comma-separated arm links to render. IK still solves the full right arm chain.",
    )
    parser.add_argument("--save-first-frame", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.num_frames is not None and args.num_frames < 1:
        raise ValueError("--num-frames must be >= 1")
    args.visible_link_names = {x.strip() for x in args.visible_links.split(",") if x.strip()}
    unknown_links = args.visible_link_names.difference(ARM_LINK_NAMES)
    if unknown_links:
        raise ValueError(f"Unknown --visible-links entries: {sorted(unknown_links)}")

    camera_config = read_camera_config(args.mps_path)
    if args.use_camera_config_fps and camera_config is not None and "fps" in camera_config:
        args.fps = float(camera_config["fps"])

    asset_root = extract_assets(args.g1_zip)
    link_meshes = load_arm_meshes(asset_root, args.max_arm_faces_per_link)
    gripper_meshes = {
        "open": decimate_mesh_faces(build_gripper_mesh(asset_root, grasp=0.0), args.max_gripper_faces),
        "closed": decimate_mesh_faces(build_gripper_mesh(asset_root, grasp=1.0), args.max_gripper_faces),
    }
    adapter_mesh = None
    if args.wrist_adapter:
        adapter_mesh = build_cylinder_mesh(
            radius=args.wrist_adapter_radius,
            length=args.wrist_adapter_length,
            z0=args.wrist_adapter_z0,
            color_bgr=(118, 126, 130),
            radius_end=args.wrist_adapter_radius_end,
        )

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
    gpu: GpuRenderer | None = None
    state = ArmState()
    written = 0
    skipped = 0
    for frame_dir in frame_dirs:
        frame = load_frame(frame_dir, args, camera_config)
        if frame is None:
            skipped += 1
            continue
        if args.renderer == "gpu" and gpu is None:
            h0, w0 = frame.background_bgr.shape[:2]
            gpu = init_gpu_renderer(w0, h0)
        rendered = render_frame(frame, link_meshes, gripper_meshes, adapter_mesh, state, args, gpu)
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
    if gpu is not None:
        gpu.renderer.delete()

    report = {
        "mps_path": str(args.mps_path),
        "out": str(args.out),
        "frames_selected": len(frame_dirs),
        "frames_written": written,
        "frames_skipped": skipped,
        "fps": args.fps,
        "background": args.background,
        "renderer": "G1 right-arm FBX visual meshes + real Omnipicker DAE gripper, tcp +Z / pz convention",
        "note": f"{args.renderer} renderer; FBX meshes are parsed directly from the official G1 package.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
