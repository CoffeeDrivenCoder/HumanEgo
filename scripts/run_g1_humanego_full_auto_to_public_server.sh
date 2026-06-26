#!/usr/bin/env bash
set -euo pipefail

# Run complete HumanEgo closed-loop control on the G1 robot/client side.
# This uses the model's raw 6D target pose and gripper output continuously.
# Stop with Ctrl+C; completed steps are written under the interactive artifact run.

cd "$(dirname "$0")/.."

export G1_HUMANEGO_TAG="${G1_HUMANEGO_TAG:-full_auto_raw_pose_gripper}"
export G1_HUMANEGO_CONTROL_MODE="auto"
export G1_HUMANEGO_CONFIRM="${G1_HUMANEGO_CONFIRM:-RUN_CONTROL}"
export G1_HUMANEGO_MAX_STEPS="${G1_HUMANEGO_MAX_STEPS:-100}"
export G1_HUMANEGO_TRACKING_GATE="${G1_HUMANEGO_TRACKING_GATE:-true}"
export G1_HUMANEGO_TRACKING_MIN_RATIO="${G1_HUMANEGO_TRACKING_MIN_RATIO:-0.30}"
export G1_HUMANEGO_TRACKING_MIN_COS="${G1_HUMANEGO_TRACKING_MIN_COS:-0.50}"
export G1_HUMANEGO_TRACKING_MIN_TARGET_M="${G1_HUMANEGO_TRACKING_MIN_TARGET_M:-0.01}"
export G1_HUMANEGO_TRACKING_BAD_STEPS="${G1_HUMANEGO_TRACKING_BAD_STEPS:-2}"
export G1_HUMANEGO_SEND_DEPTH="${G1_HUMANEGO_SEND_DEPTH:-true}"
export G1_HUMANEGO_OBJECT_LOCK="${G1_HUMANEGO_OBJECT_LOCK:-base_after_first}"
export G1_HUMANEGO_OBJECT_LOCK_REQUIRE_CLEAN="${G1_HUMANEGO_OBJECT_LOCK_REQUIRE_CLEAN:-true}"
export G1_HUMANEGO_TARGET_SOURCE="raw"
export G1_HUMANEGO_TARGET_ADAPTER="full"
export G1_HUMANEGO_TARGET_Z_BIAS_M="0.0"
export G1_HUMANEGO_EXECUTE_GRIPPER="true"
export G1_HUMANEGO_GRIPPER_SOURCE="${G1_HUMANEGO_GRIPPER_SOURCE:-model}"
export G1_HUMANEGO_EXECUTE_S="${G1_HUMANEGO_EXECUTE_S:-1.5}"
export G1_HUMANEGO_SEND_HZ="${G1_HUMANEGO_SEND_HZ:-10}"
export G1_HUMANEGO_SETTLE_S="${G1_HUMANEGO_SETTLE_S:-0.5}"
export G1_HUMANEGO_GRIPPER_SETTLE_S="${G1_HUMANEGO_GRIPPER_SETTLE_S:-0.5}"
export G1_HUMANEGO_APPROACH_OBJECT_KEY="${G1_HUMANEGO_APPROACH_OBJECT_KEY:-obj1}"

bash scripts/run_g1_humanego_interactive_step_to_public_server.sh "$@"
