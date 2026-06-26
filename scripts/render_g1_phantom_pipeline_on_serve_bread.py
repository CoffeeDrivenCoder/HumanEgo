#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render G1 on serve_bread with a Phantom/Masquerade-style MuJoCo pipeline.

This script consumes the G1 MJCF produced by ``convert_g1_urdf_to_mjcf.py``.
For every HumanEgo frame it:

1. reads the right-hand TCP target,
2. solves the G1 right-arm IK,
3. places the MJCF arm root in the Aria camera frame,
4. renders RGB + depth + segmentation mask with MuJoCo,
5. composites the robot over the inpainted human video using a robot mask.

The output contract mirrors Phantom's TwinRobot output: robot RGB, robot depth,
robot mask, and gripper mask are available per frame before overlay.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
DEFAULT_MPS_PATH = Path("/data/wangk/data/serve_bread/aria/mps_serve_bread_006_vrs")
DEFAULT_MJCF = PROJECT_ROOT / "outputs" / "g1_mjcf" / "g1_omnipicker_right_arm_local.xml"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "render_g1_phantom_pipeline" / "g1_phantom_pipeline_006_tcp_pz_full.mp4"

sys.path.insert(0, str(SCRIPT_DIR))

from render_g1_arm_mesh_on_serve_bread import (  # noqa: E402
    ArmState,
    T_GRIPPER_BASE_IN_END,
    T_TCP_IN_GRIPPER_BASE,
    arm_fk,
    estimate_base_cam,
    look_at_base_pose_cam,
    solve_arm_ik,
)
from render_g1_gripper_mesh_on_serve_bread import (  # noqa: E402
    T_from_rpy_xyz,
    axis_angle_T,
    T_tcp_world_from_training,
    find_rgb_frame_dirs,
    load_frame,
    make_writer,
    read_camera_config,
    select_frame_dirs,
)


RIGHT_ARM_JOINTS = [
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
]

GRIPPER_JOINTS = [
    "idx71_gripper_r_inner_joint1",
    "idx72_gripper_r_inner_joint3",
    "idx73_gripper_r_inner_joint4",
    "idx79_gripper_r_inner_joint2",
    "idx81_gripper_r_outer_joint1",
    "idx82_gripper_r_outer_joint3",
    "idx83_gripper_r_outer_joint4",
    "idx89_gripper_r_outer_joint2",
]

GRIPPER_GEOM_NAMES = {
    "gripper_base_link",
    "inner_link1",
    "inner_link2",
    "inner_link3",
    "inner_link4",
    "outer_link1",
    "outer_link2",
    "outer_link3",
    "outer_link4",
}

GRIPPER_HINGE_AXIS = [0.0, 0.0, -1.0]
GRIPPER_JOINT_ORIGIN = {
    "inner1": T_from_rpy_xyz([-2.9951, -math.pi / 2.0, -0.15964], [0.0, -0.0195, 0.0565]),
    "inner2": T_from_rpy_xyz([-2.9951, -math.pi / 2.0, -0.15964], [0.0, -0.021633, 0.07387]),
    "inner3": T_from_rpy_xyz([0.0, 0.0, 0.0], [0.030852, 0.018551, 0.0]),
    "inner4": T_from_rpy_xyz([0.0, 0.0, 0.0], [0.018118, -0.01574, 0.0]),
    "outer1": T_from_rpy_xyz([-2.9951, -math.pi / 2.0, -0.15964], [0.0, 0.0195, 0.0565]),
    "outer2": T_from_rpy_xyz([-2.9951, -math.pi / 2.0, -0.15964], [0.0, 0.021633, 0.07387]),
    "outer3": T_from_rpy_xyz([0.0, 0.0, 0.0], [0.030852, -0.018551, 0.0]),
    "outer4": T_from_rpy_xyz([0.0, 0.0, 0.0], [0.018118, 0.01574, 0.0]),
}
GRIPPER_LOOP_POINTS = {
    "inner2": np.asarray([0.03123003, -0.010280428, 0.0], dtype=np.float64),
    "inner4": np.asarray([0.0, -0.010280428, 0.0], dtype=np.float64),
    "outer2": np.asarray([0.03123003, 0.010280428, 0.0], dtype=np.float64),
    "outer4": np.asarray([0.0, 0.010280428, 0.0], dtype=np.float64),
}


@dataclass
class MujocoRenderResult:
    rgb_bgr: np.ndarray
    depth: np.ndarray
    robot_mask: np.ndarray
    gripper_mask: np.ndarray


def quat_wxyz_from_R(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_quat(scalar_first=True)


def set_camera_from_arbitrary_K(model, camera_id: int, K: np.ndarray, width: int, height: int) -> None:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    model.cam_resolution[camera_id] = np.asarray([width, height], dtype=np.int32)
    model.cam_sensorsize[camera_id] = np.asarray([width, height], dtype=np.float64)
    # MuJoCo's fixed camera intrinsic array follows the MJCF focalpixel /
    # principalpixel convention: the principal point is an offset from image
    # center, not OpenCV's absolute pixel coordinate.
    model.cam_intrinsic[camera_id] = np.asarray([fx, fy, width / 2.0 - cx, cy - height / 2.0], dtype=np.float64)
    model.cam_projection[camera_id] = 0


def set_fixed_camera_pose_cv(model, camera_id: int, c2w_cv: np.ndarray) -> None:
    # MuJoCo camera convention looks along local -Z with +Y up. OpenCV camera
    # convention looks along +Z with +Y down. This fixed rotation maps MuJoCo
    # camera coordinates into OpenCV camera coordinates.
    R_cv_from_mj = np.diag([1.0, -1.0, -1.0])
    R_w_mj = c2w_cv[:3, :3] @ R_cv_from_mj
    model.cam_pos[camera_id] = c2w_cv[:3, 3]
    model.cam_quat[camera_id] = quat_wxyz_from_R(R_w_mj)


def transform_point(T: np.ndarray, point: np.ndarray) -> np.ndarray:
    return (T @ np.r_[point, 1.0])[:3]


def gripper_loop_residual(side: str, q_main: float, q_linkage: np.ndarray) -> np.ndarray:
    q3, q4, q2 = [float(v) for v in q_linkage]
    T1 = GRIPPER_JOINT_ORIGIN[f"{side}1"] @ axis_angle_T(GRIPPER_HINGE_AXIS, q_main)
    T3 = T1 @ GRIPPER_JOINT_ORIGIN[f"{side}3"] @ axis_angle_T(GRIPPER_HINGE_AXIS, q3)
    T4 = T3 @ GRIPPER_JOINT_ORIGIN[f"{side}4"] @ axis_angle_T(GRIPPER_HINGE_AXIS, q4)
    T2 = GRIPPER_JOINT_ORIGIN[f"{side}2"] @ axis_angle_T(GRIPPER_HINGE_AXIS, q2)
    return transform_point(T2, GRIPPER_LOOP_POINTS[f"{side}2"]) - transform_point(
        T4,
        GRIPPER_LOOP_POINTS[f"{side}4"],
    )


@lru_cache(maxsize=128)
def solve_gripper_loop(side: str, q_main_key: float) -> tuple[float, float, float]:
    q_main = float(q_main_key)
    x0 = np.asarray([-q_main, q_main, -q_main], dtype=np.float64)

    def residual(x: np.ndarray) -> np.ndarray:
        # Scale the loop-anchor error from meters to solver-friendly units, with
        # a tiny prior that selects a stable elbow configuration across frames.
        return np.r_[100.0 * gripper_loop_residual(side, q_main, x), 1e-3 * (x - x0)]

    result = least_squares(
        residual,
        x0,
        bounds=(-2.0, 2.0),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=200,
    )
    return tuple(float(v) for v in result.x)


def gripper_qpos_from_grasp(grasp: float) -> dict[str, float]:
    g = float(np.clip(grasp, 0.0, 1.0))
    open_angle = 0.62
    closed_angle = 0.10
    q_outer = (1.0 - g) * open_angle + g * closed_angle
    q_inner = -q_outer
    # The original URDF uses mimic plus custom loop_joint tags. MuJoCo imports a
    # tree, so we solve the removed loop constraints and drive the passive tree
    # joints to keep the complete Omnipicker visual chain closed.
    q_outer3, q_outer4, q_outer2 = solve_gripper_loop("outer", round(q_outer, 5))
    q_inner3, q_inner4, q_inner2 = solve_gripper_loop("inner", round(q_inner, 5))
    return {
        "idx71_gripper_r_inner_joint1": q_inner,
        "idx72_gripper_r_inner_joint3": q_inner3,
        "idx73_gripper_r_inner_joint4": q_inner4,
        "idx79_gripper_r_inner_joint2": q_inner2,
        "idx81_gripper_r_outer_joint1": q_outer,
        "idx82_gripper_r_outer_joint3": q_outer3,
        "idx83_gripper_r_outer_joint4": q_outer4,
        "idx89_gripper_r_outer_joint2": q_outer2,
    }


class G1MujocoTwin:
    def __init__(self, mjcf_path: Path, width: int, height: int):
        os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "aria")
        if self.camera_id < 0:
            raise RuntimeError(f"MJCF has no camera named 'aria': {mjcf_path}")
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.width = width
        self.height = height
        self.root_qpos_adr = int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "g1_root_freejoint")])
        self.joint_qpos_adr = {
            name: int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)])
            for name in RIGHT_ARM_JOINTS + GRIPPER_JOINTS
        }
        self.gripper_geom_ids = {
            self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in GRIPPER_GEOM_NAMES
        }
        self.gripper_geom_ids.discard(-1)
        self.robot_geom_ids = set(range(self.model.ngeom))

    def close(self) -> None:
        self.renderer.close()

    def render(
        self,
        T_base_world: np.ndarray,
        q_arm: np.ndarray,
        grasp: float,
        K: np.ndarray,
        c2w: np.ndarray,
        width: int,
        height: int,
    ) -> MujocoRenderResult:
        if width != self.width or height != self.height:
            self.renderer.close()
            self.renderer = self.mujoco.Renderer(self.model, height=height, width=width)
            self.width = width
            self.height = height

        set_camera_from_arbitrary_K(self.model, self.camera_id, K, width, height)
        set_fixed_camera_pose_cv(self.model, self.camera_id, c2w)

        self.data.qpos[:] = 0.0
        self.data.qpos[self.root_qpos_adr : self.root_qpos_adr + 3] = T_base_world[:3, 3]
        self.data.qpos[self.root_qpos_adr + 3 : self.root_qpos_adr + 7] = quat_wxyz_from_R(T_base_world[:3, :3])
        for name, q in zip(RIGHT_ARM_JOINTS, q_arm):
            self.data.qpos[self.joint_qpos_adr[name]] = float(q)
        for name, q in gripper_qpos_from_grasp(grasp).items():
            self.data.qpos[self.joint_qpos_adr[name]] = float(q)

        self.mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="aria")
        rgb = self.renderer.render()

        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera="aria")
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()

        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera="aria")
        seg = self.renderer.render().copy()
        self.renderer.disable_segmentation_rendering()

        obj_id = seg[..., 0].astype(np.int32)
        obj_type = seg[..., 1].astype(np.int32)
        geom_type = int(self.mujoco.mjtObj.mjOBJ_GEOM)
        robot_mask = (obj_type == geom_type) & np.isin(obj_id, list(self.robot_geom_ids))
        gripper_mask = (obj_type == geom_type) & np.isin(obj_id, list(self.gripper_geom_ids))
        return MujocoRenderResult(
            rgb_bgr=rgb[..., ::-1].astype(np.uint8),
            depth=depth.astype(np.float32),
            robot_mask=robot_mask,
            gripper_mask=gripper_mask,
        )


def soften_mask(mask: np.ndarray, blur: int, erode: int = 0) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8) * 255
    if erode > 0:
        k = 2 * erode + 1
        mask_u8 = cv2.erode(mask_u8, np.ones((k, k), np.uint8), iterations=1)
    if blur > 0:
        k = 2 * blur + 1
        mask_u8 = cv2.GaussianBlur(mask_u8, (k, k), 0)
    return mask_u8.astype(np.float32) / 255.0


def match_robot_appearance(robot_bgr: np.ndarray, background_bgr: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0 or not np.any(mask):
        return robot_bgr
    out = robot_bgr.astype(np.float32)
    bg = background_bgr.astype(np.float32)
    idx = mask.astype(bool)
    bg_mean = bg[idx].mean(axis=0)
    fg_mean = out[idx].mean(axis=0)
    shift = (bg_mean - fg_mean) * float(np.clip(strength, 0.0, 1.0))
    out[idx] += shift
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_robot(
    background_bgr: np.ndarray,
    robot_result: MujocoRenderResult,
    alpha: float,
    mask_blur: int,
    color_match: float,
) -> np.ndarray:
    mask = robot_result.robot_mask | robot_result.gripper_mask
    fg = match_robot_appearance(robot_result.rgb_bgr, background_bgr, mask, color_match)
    soft = soften_mask(mask, mask_blur)[..., None] * float(np.clip(alpha, 0.0, 1.0))
    out = soft * fg.astype(np.float32) + (1.0 - soft) * background_bgr.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def robot_mask_is_valid(mask: np.ndarray, max_mask_frac: float, max_bbox_frac: float) -> bool:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return False
    h, w = mask.shape[:2]
    mask_frac = float(len(xs)) / float(h * w)
    bbox_frac = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)) / float(h * w)
    return mask_frac <= max_mask_frac and bbox_frac <= max_bbox_frac


def compute_frame_pose(frame, state: ArmState, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray] | None:
    T_tcp_world, grasp = T_tcp_world_from_training(frame.training)
    if T_tcp_world is None:
        return None
    K = np.asarray(frame.training["metadata"]["k"], dtype=np.float64).reshape(3, 3)
    c2w = np.asarray(frame.training["metadata"]["c2w"], dtype=np.float64).reshape(4, 4)
    T_tcp_cam = np.linalg.inv(c2w) @ T_tcp_world
    base_pos_cam = estimate_base_cam(frame, T_tcp_world, K, c2w, state, args.base_smooth)
    T_base_cam = look_at_base_pose_cam(base_pos_cam, T_tcp_cam[:3, 3])
    T_base_world = c2w @ T_base_cam
    T_tcp_target_base = np.linalg.inv(T_base_world) @ T_tcp_world
    q = solve_arm_ik(T_tcp_target_base, state.q, args.ik_max_nfev)
    state.q = q
    return T_base_world, q, float(grasp or 0.0), K, c2w


def render_frame(frame, twin: G1MujocoTwin, state: ArmState, args: argparse.Namespace) -> np.ndarray:
    pose = compute_frame_pose(frame, state, args)
    if pose is None:
        return frame.background_bgr
    T_base_world, q, grasp, K, c2w = pose
    h, w = frame.background_bgr.shape[:2]
    robot = twin.render(T_base_world, q, grasp, K, c2w, width=w, height=h)
    if not robot_mask_is_valid(
        robot.robot_mask | robot.gripper_mask,
        max_mask_frac=args.max_mask_frac,
        max_bbox_frac=args.max_bbox_frac,
    ):
        return frame.background_bgr
    return overlay_robot(frame.background_bgr, robot, args.overlay_alpha, args.mask_blur, args.color_match)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mps-path", type=Path, default=DEFAULT_MPS_PATH)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
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
    parser.add_argument("--overlay-alpha", type=float, default=1.0)
    parser.add_argument("--mask-blur", type=int, default=1)
    parser.add_argument("--color-match", type=float, default=0.18)
    parser.add_argument("--max-mask-frac", type=float, default=0.28)
    parser.add_argument("--max-bbox-frac", type=float, default=0.65)
    parser.add_argument("--ik-max-nfev", type=int, default=45)
    parser.add_argument("--base-smooth", type=float, default=0.85)
    parser.add_argument("--save-first-frame", type=Path, default=None)
    parser.add_argument("--save-debug-frame", type=Path, default=None)
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
    twin: G1MujocoTwin | None = None
    state = ArmState()
    written = 0
    skipped = 0
    debug_saved = False
    for frame_dir in frame_dirs:
        frame = load_frame(frame_dir, args, camera_config)
        if frame is None:
            skipped += 1
            continue
        if twin is None:
            h0, w0 = frame.background_bgr.shape[:2]
            twin = G1MujocoTwin(args.mjcf, width=w0, height=h0)
        rendered = render_frame(frame, twin, state, args)
        if writer is None:
            h, w = rendered.shape[:2]
            writer = make_writer(args.out, args.fps, (w, h))
            if args.save_first_frame is not None:
                args.save_first_frame.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.save_first_frame), rendered)
        if args.save_debug_frame is not None and not debug_saved and np.any(rendered != frame.background_bgr):
            args.save_debug_frame.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.save_debug_frame), rendered)
            debug_saved = True
        writer.write(rendered)
        written += 1

    if writer is not None:
        writer.release()
    if twin is not None:
        twin.close()

    report = {
        "mps_path": str(args.mps_path),
        "mjcf": str(args.mjcf),
        "out": str(args.out),
        "frames_selected": len(frame_dirs),
        "frames_written": written,
        "frames_skipped": skipped,
        "fps": args.fps,
        "background": args.background,
        "renderer": "MuJoCo G1 MJCF renderer + Phantom-style robot mask overlay",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
