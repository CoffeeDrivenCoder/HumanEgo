#!/usr/bin/env bash
set -euo pipefail

# Run one HumanEgo raw/full step through G1 trajectory_tracking_control
# with control_type=DELTA_POSE. Gripper is disabled by default so this isolates
# EE control/IK tracking from gripper side effects.

cd "$(dirname "$0")/.."

export G1_HUMANEGO_TAG="${G1_HUMANEGO_TAG:-delta_pose_step_raw_full_no_gripper}"
export G1_HUMANEGO_CONTROL_MODE="prompt"
export G1_HUMANEGO_CONFIRM="${G1_HUMANEGO_CONFIRM:-RUN_CONTROL}"
export G1_HUMANEGO_MAX_STEPS="${G1_HUMANEGO_MAX_STEPS:-1}"
export G1_HUMANEGO_SEND_DEPTH="${G1_HUMANEGO_SEND_DEPTH:-true}"
export G1_HUMANEGO_OBJECT_LOCK="${G1_HUMANEGO_OBJECT_LOCK:-base_after_first}"
export G1_HUMANEGO_OBJECT_LOCK_REQUIRE_CLEAN="${G1_HUMANEGO_OBJECT_LOCK_REQUIRE_CLEAN:-true}"
export G1_HUMANEGO_TARGET_SOURCE="raw"
export G1_HUMANEGO_TARGET_ADAPTER="full"
export G1_HUMANEGO_TARGET_Z_BIAS_M="0.0"
export G1_HUMANEGO_EE_CONTROL_MODE="delta_pose"
export G1_HUMANEGO_DELTA_POSE_ROTATION_FRAME="${G1_HUMANEGO_DELTA_POSE_ROTATION_FRAME:-base}"
export G1_HUMANEGO_DELTA_POSE_REFERENCE_TIME="${G1_HUMANEGO_DELTA_POSE_REFERENCE_TIME:-1.0}"
export G1_HUMANEGO_EXECUTE_GRIPPER="${G1_HUMANEGO_EXECUTE_GRIPPER:-false}"
export G1_HUMANEGO_EXECUTE_S="${G1_HUMANEGO_EXECUTE_S:-1.0}"
export G1_HUMANEGO_SETTLE_S="${G1_HUMANEGO_SETTLE_S:-1.0}"
export G1_HUMANEGO_APPROACH_OBJECT_KEY="${G1_HUMANEGO_APPROACH_OBJECT_KEY:-obj1}"

bash scripts/run_g1_humanego_interactive_step_to_public_server.sh "$@"
