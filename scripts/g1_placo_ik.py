#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Placo-based FK/IK utilities for G1 link7 targets.

This module mirrors the LeRobot UMI/G2 approach: placo frame tasks, masked
non-arm joints, velocity regularization, and explicit post-solve safety checks.
It is intentionally independent from the online HumanEgo control path.
"""

from __future__ import annotations

import tempfile
import time
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from g1_urdf_ik import DEFAULT_G1_ZIP, normalize_waist_states, pose_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF_IN_ZIP = "G1_URDF_Omnipicker/urdf/G1/G1_omnipicker_omnipicker.urdf"
G1_LEFT_JOINT_NAMES = [f"idx{idx}_arm_l_joint{j}" for idx, j in zip(range(21, 28), range(1, 8))]
G1_RIGHT_JOINT_NAMES = [f"idx{idx}_arm_r_joint{j}" for idx, j in zip(range(61, 68), range(1, 8))]
G1_ARM_JOINT_NAMES = {"left": G1_LEFT_JOINT_NAMES, "right": G1_RIGHT_JOINT_NAMES}
G1_FRAME_NAMES = {"left": "arm_l_end_link", "right": "arm_r_end_link"}
IK_ORIENTATION_TASK_MIN_WEIGHT = 1e-6


@dataclass
class G1PlacoIkResult:
    side: str
    success: bool
    q_init: np.ndarray
    q_solution: np.ndarray
    position_error_m: float
    rotation_error_deg: float
    q_delta_abs_max_rad: float
    q_delta_norm_rad: float
    within_joint_limits: bool
    duration_s: float
    iterations: int
    converged_by_position: bool
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "success": bool(self.success),
            "q_init": self.q_init.tolist(),
            "q_solution": self.q_solution.tolist(),
            "q_delta": (self.q_solution - self.q_init).tolist(),
            "q_delta_abs_max_rad": float(self.q_delta_abs_max_rad),
            "q_delta_norm_rad": float(self.q_delta_norm_rad),
            "position_error_m": float(self.position_error_m),
            "rotation_error_deg": float(self.rotation_error_deg),
            "within_joint_limits": bool(self.within_joint_limits),
            "duration_s": float(self.duration_s),
            "iterations": int(self.iterations),
            "converged_by_position": bool(self.converged_by_position),
            "error": self.error,
        }


def side_from_name(side: str) -> str:
    side = side.lower()
    if side in {"left", "l"}:
        return "left"
    if side in {"right", "r"}:
        return "right"
    raise ValueError(f"side must be left/right, got {side!r}")


def prepare_placo_urdf(
    urdf_zip: str | Path = DEFAULT_G1_ZIP,
    urdf_in_zip: str = DEFAULT_URDF_IN_ZIP,
    cache_dir: str | Path | None = None,
) -> Path:
    """Extract G1 URDF zip and patch package:// mesh paths for placo."""
    urdf_zip = Path(urdf_zip).expanduser().resolve()
    if cache_dir is None:
        root = Path(tempfile.mkdtemp(prefix="g1_placo_urdf_"))
    else:
        root = Path(cache_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)

    marker = root / ".source_zip"
    patched = root / "g1_placo_patched.urdf"
    if patched.exists() and marker.exists() and marker.read_text(encoding="utf-8") == str(urdf_zip):
        return patched

    with zipfile.ZipFile(urdf_zip) as zf:
        zf.extractall(root)
    package_root = root / "G1_URDF_Omnipicker"
    src_urdf = root / urdf_in_zip
    text = src_urdf.read_text(encoding="utf-8")
    text = text.replace("package://genie_robot_description/", package_root.as_uri() + "/")
    patched.write_text(text, encoding="utf-8")
    marker.write_text(str(urdf_zip), encoding="utf-8")
    return patched


def load_joint_limits_rad(urdf_path: str | Path) -> dict[str, tuple[float, float]]:
    root = ET.parse(urdf_path).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        limit = joint.find("limit")
        if not name or limit is None:
            continue
        lower = limit.attrib.get("lower")
        upper = limit.attrib.get("upper")
        if lower is None or upper is None:
            continue
        limits[name] = (float(lower), float(upper))
    return limits


def configure_split_frame_task(frame_task: Any, task_name: str, position_weight: float, orientation_weight: float) -> None:
    frame_task.position().configure(f"{task_name}_position", "soft", float(position_weight))
    frame_task.orientation().configure(
        f"{task_name}_orientation",
        "soft",
        max(float(orientation_weight), IK_ORIENTATION_TASK_MIN_WEIGHT),
    )


def rotation_angle_deg(R_delta: np.ndarray) -> float:
    value = (float(np.trace(R_delta)) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


class G1PlacoKinematics:
    def __init__(
        self,
        urdf_zip: str | Path = DEFAULT_G1_ZIP,
        *,
        cache_dir: str | Path | None = PROJECT_ROOT / "artifacts" / "g1_placo_urdf_cache",
        max_iterations: int = 20,
        dt: float = 5e-2,
        regularization_weight: float = 1e-3,
        manipulability_weight: float = 0.0,
        enable_joint_limits: bool = False,
    ):
        import placo  # type: ignore[import-not-found]

        self.urdf_path = prepare_placo_urdf(urdf_zip, cache_dir=cache_dir)
        self.joint_limits = load_joint_limits_rad(self.urdf_path)
        self.max_iterations = int(max_iterations)
        self.robot = placo.RobotWrapper(str(self.urdf_path))
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)
        self.solver.enable_joint_limits(bool(enable_joint_limits))
        self.all_joint_names = list(self.robot.joint_names())
        self._robot_joint_name_set = set(self.all_joint_names)

        enabled = set(G1_LEFT_JOINT_NAMES + G1_RIGHT_JOINT_NAMES)
        for joint_name in self.all_joint_names:
            if joint_name not in enabled:
                self.solver.mask_dof(joint_name)

        self.tips = {
            "left": self.solver.add_frame_task(G1_FRAME_NAMES["left"], np.eye(4)),
            "right": self.solver.add_frame_task(G1_FRAME_NAMES["right"], np.eye(4)),
        }
        self.solver.dt = float(dt)
        if regularization_weight > 0.0:
            self.solver.add_regularization_task(float(regularization_weight))
        if manipulability_weight > 0.0:
            self.solver.add_manipulability_task(G1_FRAME_NAMES["left"], "both", 1.0).configure(
                "manip_left", "soft", float(manipulability_weight)
            )
            self.solver.add_manipulability_task(G1_FRAME_NAMES["right"], "both", 1.0).configure(
                "manip_right", "soft", float(manipulability_weight)
            )

    def set_full_state(
        self,
        *,
        side: str,
        q_side: np.ndarray,
        waist_states: list[float] | tuple[float, ...] | None = None,
    ) -> None:
        side = side_from_name(side)
        waist_values = normalize_waist_states(waist_states)
        for name, value in waist_values.items():
            if name in self._robot_joint_name_set:
                self.robot.set_joint(name, float(value))
        q_side = np.asarray(q_side, dtype=np.float64).reshape(7)
        for idx, joint_name in enumerate(G1_ARM_JOINT_NAMES[side]):
            self.robot.set_joint(joint_name, float(q_side[idx]))
        self.robot.update_kinematics()

    def link7_fk(
        self,
        side: str,
        q_side: np.ndarray,
        waist_states: list[float] | tuple[float, ...] | None = None,
    ) -> np.ndarray:
        side = side_from_name(side)
        self.set_full_state(side=side, q_side=q_side, waist_states=waist_states)
        return np.array(self.robot.get_T_world_frame(G1_FRAME_NAMES[side]), dtype=np.float64)

    def within_limits(self, side: str, q_side: np.ndarray) -> bool:
        side = side_from_name(side)
        q_side = np.asarray(q_side, dtype=np.float64).reshape(7)
        for idx, joint_name in enumerate(G1_ARM_JOINT_NAMES[side]):
            limit = self.joint_limits.get(joint_name)
            if limit is None:
                continue
            lower, upper = limit
            value = float(q_side[idx])
            if value < lower or value > upper:
                return False
        return True

    def _hold_inactive_arm(self, side: str) -> None:
        inactive = "right" if side == "left" else "left"
        current_pose = np.array(self.robot.get_T_world_frame(G1_FRAME_NAMES[inactive]), dtype=np.float64)
        self.tips[inactive].T_world_frame = current_pose
        configure_split_frame_task(self.tips[inactive], G1_FRAME_NAMES[inactive], 1.0, 1e-6)

    def solve_link7_ik(
        self,
        side: str,
        target_T_link7_in_base: np.ndarray,
        q_init: np.ndarray,
        waist_states: list[float] | tuple[float, ...] | None = None,
        *,
        position_weight: float = 1.0,
        orientation_weight: float = 0.2,
        position_tolerance_m: float = 1e-3,
        rotation_tolerance_deg: float = 1.0,
        max_joint_delta_rad: float | None = None,
    ) -> G1PlacoIkResult:
        side = side_from_name(side)
        target_T = np.asarray(target_T_link7_in_base, dtype=np.float64).reshape(4, 4)
        q_init = np.asarray(q_init, dtype=np.float64).reshape(7)
        started = time.perf_counter()
        iterations = 0
        converged_by_position = False
        error = None
        try:
            self.set_full_state(side=side, q_side=q_init, waist_states=waist_states)
            self.tips[side].T_world_frame = target_T
            configure_split_frame_task(self.tips[side], G1_FRAME_NAMES[side], position_weight, orientation_weight)
            self._hold_inactive_arm(side)

            for iteration in range(self.max_iterations):
                iterations = iteration + 1
                self.solver.solve(True)
                self.robot.update_kinematics()
                if self.tips[side].position().error_norm() <= float(position_tolerance_m):
                    converged_by_position = True
                    break
        except Exception as exc:  # placo raises RuntimeError for infeasible QPs.
            error = f"{type(exc).__name__}: {exc}"

        q_solution = np.asarray(
            [self.robot.get_joint(name) for name in G1_ARM_JOINT_NAMES[side]],
            dtype=np.float64,
        )
        q_delta = q_solution - q_init
        achieved_T = self.link7_fk(side, q_solution, waist_states=waist_states)
        err = pose_error(achieved_T, target_T)
        q_delta_abs_max = float(np.max(np.abs(q_delta)))
        within_limits = self.within_limits(side, q_solution)
        delta_ok = True if max_joint_delta_rad is None else q_delta_abs_max <= float(max_joint_delta_rad)
        success = bool(
            error is None
            and within_limits
            and delta_ok
            and float(err["position_error_m"]) <= float(position_tolerance_m)
            and float(err["rotation_error_deg"]) <= float(rotation_tolerance_deg)
        )
        return G1PlacoIkResult(
            side=side,
            success=success,
            q_init=q_init,
            q_solution=q_solution,
            position_error_m=float(err["position_error_m"]),
            rotation_error_deg=float(err["rotation_error_deg"]),
            q_delta_abs_max_rad=q_delta_abs_max,
            q_delta_norm_rad=float(np.linalg.norm(q_delta)),
            within_joint_limits=within_limits,
            duration_s=time.perf_counter() - started,
            iterations=iterations,
            converged_by_position=converged_by_position,
            error=error,
        )
