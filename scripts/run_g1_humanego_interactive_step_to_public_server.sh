#!/usr/bin/env bash
set -euo pipefail

# Run this on the G1 robot/client side.
# Interactive one-step HumanEgo control. It executes only after operator presses
# Enter at each prompt, and only when G1_HUMANEGO_CONFIRM=RUN_CONTROL.

cd "$(dirname "$0")/.."

if [ -f "a2d_sdk/env.sh" ]; then
  source a2d_sdk/env.sh || echo "[WARN] failed to source a2d_sdk/env.sh; continuing with current environment" >&2
fi

SERVER_URL="${G1_HUMANEGO_SERVER_URL:-http://111.0.22.33:30003/infer}"
# Interactive control should not wait on diagnostics upload by default.
# Set G1_HUMANEGO_UPLOAD_URL or pass --upload-url to upload the saved zip.
UPLOAD_URL="${G1_HUMANEGO_UPLOAD_URL:-${G1_DIAG_UPLOAD_URL:-}}"
SESSION="${G1_ARTIFACT_SESSION:-$(date -u +%Y%m%d)}"
OUT_DIR="${G1_HUMANEGO_INTERACTIVE_OUT_DIR:-./artifacts/g1_humanego/${SESSION}/interactive}"
TAG="${G1_HUMANEGO_TAG:-interactive_step}"
CFG="${G1_HUMANEGO_CFG:-cfg/inference/g1_serve_bread_right.yaml}"
CONFIRM="${G1_HUMANEGO_CONFIRM:-}"
MAX_STEPS="${G1_HUMANEGO_MAX_STEPS:-20}"
CONTROL_MODE="${G1_HUMANEGO_CONTROL_MODE:-prompt}"
TRACKING_GATE="${G1_HUMANEGO_TRACKING_GATE:-false}"
TRACKING_MIN_RATIO="${G1_HUMANEGO_TRACKING_MIN_RATIO:-0.30}"
TRACKING_MIN_COS="${G1_HUMANEGO_TRACKING_MIN_COS:-0.50}"
TRACKING_MIN_TARGET_M="${G1_HUMANEGO_TRACKING_MIN_TARGET_M:-0.01}"
TRACKING_BAD_STEPS="${G1_HUMANEGO_TRACKING_BAD_STEPS:-2}"
TARGET_SOURCE="${G1_HUMANEGO_TARGET_SOURCE:-position_keep_orientation}"
TARGET_ADAPTER="${G1_HUMANEGO_TARGET_ADAPTER:-full}"
OBJECT_LOCK="${G1_HUMANEGO_OBJECT_LOCK:-none}"
OBJECT_LOCK_REQUIRE_CLEAN="${G1_HUMANEGO_OBJECT_LOCK_REQUIRE_CLEAN:-true}"
AXIS_STEP_M="${G1_HUMANEGO_AXIS_STEP_M:-0.01}"
TARGET_Z_BIAS_M="${G1_HUMANEGO_TARGET_Z_BIAS_M:-0.0}"
MAX_ORIENTATION_DEG="${G1_HUMANEGO_MAX_ORIENTATION_DEG:-10}"
PROBE_AXIS="${G1_HUMANEGO_PROBE_AXIS:-z}"
PROBE_DEG="${G1_HUMANEGO_PROBE_DEG:-10}"
PROBE_FRAME="${G1_HUMANEGO_PROBE_FRAME:-local}"
LIFETIME="${G1_HUMANEGO_LIFETIME:-1.5}"
SEND_HZ="${G1_HUMANEGO_SEND_HZ:-20}"
EXECUTE_S="${G1_HUMANEGO_EXECUTE_S:-3.0}"
SETTLE_S="${G1_HUMANEGO_SETTLE_S:-1.5}"
EE_CONTROL_MODE="${G1_HUMANEGO_EE_CONTROL_MODE:-absolute_pose}"
DELTA_POSE_ROTATION_FRAME="${G1_HUMANEGO_DELTA_POSE_ROTATION_FRAME:-base}"
DELTA_POSE_REFERENCE_TIME="${G1_HUMANEGO_DELTA_POSE_REFERENCE_TIME:-}"
EXECUTE_GRIPPER="${G1_HUMANEGO_EXECUTE_GRIPPER:-false}"
GRIPPER_SOURCE="${G1_HUMANEGO_GRIPPER_SOURCE:-model}"
GRIPPER_TARGET="${G1_HUMANEGO_GRIPPER_TARGET:-}"
GRIPPER_MIN="${G1_HUMANEGO_GRIPPER_MIN:-0.0}"
GRIPPER_MAX="${G1_HUMANEGO_GRIPPER_MAX:-1.0}"
GRIPPER_SETTLE_S="${G1_HUMANEGO_GRIPPER_SETTLE_S:-0.5}"
JPEG_QUALITY="${G1_HUMANEGO_JPEG_QUALITY:-75}"
SEND_WIDTH="${G1_HUMANEGO_SEND_WIDTH:-640}"
SEND_HEIGHT="${G1_HUMANEGO_SEND_HEIGHT:-400}"
SEND_DEPTH="${G1_HUMANEGO_SEND_DEPTH:-true}"
DEPTH_ENCODING="${G1_HUMANEGO_DEPTH_ENCODING:-z16}"
APPROACH_OBJECT_KEY="${G1_HUMANEGO_APPROACH_OBJECT_KEY:-obj1}"
TIMEOUT_S="${G1_HUMANEGO_TIMEOUT_S:-120}"
UPLOAD_TIMEOUT_S="${G1_HUMANEGO_UPLOAD_TIMEOUT_S:-20}"

SEND_DEPTH_ARG="--no-send-depth"
if [[ "$SEND_DEPTH" == "true" || "$SEND_DEPTH" == "1" ]]; then
  SEND_DEPTH_ARG="--send-depth"
fi

EXECUTE_GRIPPER_ARG="--no-execute-gripper"
if [[ "$EXECUTE_GRIPPER" == "true" || "$EXECUTE_GRIPPER" == "1" ]]; then
  EXECUTE_GRIPPER_ARG="--execute-gripper"
fi

TRACKING_GATE_ARG="--no-tracking-gate"
if [[ "$TRACKING_GATE" == "true" || "$TRACKING_GATE" == "1" ]]; then
  TRACKING_GATE_ARG="--tracking-gate"
fi

GRIPPER_TARGET_ARGS=()
if [[ -n "$GRIPPER_TARGET" ]]; then
  GRIPPER_TARGET_ARGS=(--gripper-target "$GRIPPER_TARGET")
fi

DELTA_POSE_REFERENCE_ARGS=()
if [[ -n "$DELTA_POSE_REFERENCE_TIME" ]]; then
  DELTA_POSE_REFERENCE_ARGS=(--delta-pose-reference-time "$DELTA_POSE_REFERENCE_TIME")
fi

OBJECT_LOCK_CLEAN_ARG="--object-lock-require-clean"
if [[ "$OBJECT_LOCK_REQUIRE_CLEAN" == "false" || "$OBJECT_LOCK_REQUIRE_CLEAN" == "0" ]]; then
  OBJECT_LOCK_CLEAN_ARG="--no-object-lock-require-clean"
fi

python3 scripts/g1_humanego_interactive_step_client.py \
  --cfg "$CFG" \
  --server-url "$SERVER_URL" \
  --out-dir "$OUT_DIR" \
  --tag "$TAG" \
  --confirm-control "$CONFIRM" \
  --max-steps "$MAX_STEPS" \
  --control-mode "$CONTROL_MODE" \
  "$TRACKING_GATE_ARG" \
  --tracking-min-ratio "$TRACKING_MIN_RATIO" \
  --tracking-min-cos "$TRACKING_MIN_COS" \
  --tracking-min-target-m "$TRACKING_MIN_TARGET_M" \
  --tracking-bad-steps "$TRACKING_BAD_STEPS" \
  --target-source "$TARGET_SOURCE" \
  --approach-object-key "$APPROACH_OBJECT_KEY" \
  --object-lock "$OBJECT_LOCK" \
  "$OBJECT_LOCK_CLEAN_ARG" \
  --target-adapter "$TARGET_ADAPTER" \
  --axis-step-m "$AXIS_STEP_M" \
  --target-z-bias-m "$TARGET_Z_BIAS_M" \
  --max-orientation-deg "$MAX_ORIENTATION_DEG" \
  --probe-axis "$PROBE_AXIS" \
  --probe-deg "$PROBE_DEG" \
  --probe-frame "$PROBE_FRAME" \
  --lifetime "$LIFETIME" \
  --send-hz "$SEND_HZ" \
  --execute-s "$EXECUTE_S" \
  --settle-s "$SETTLE_S" \
  --ee-control-mode "$EE_CONTROL_MODE" \
  --delta-pose-rotation-frame "$DELTA_POSE_ROTATION_FRAME" \
  "${DELTA_POSE_REFERENCE_ARGS[@]}" \
  "$EXECUTE_GRIPPER_ARG" \
  --gripper-source "$GRIPPER_SOURCE" \
  "${GRIPPER_TARGET_ARGS[@]}" \
  --gripper-min "$GRIPPER_MIN" \
  --gripper-max "$GRIPPER_MAX" \
  --gripper-settle-s "$GRIPPER_SETTLE_S" \
  --jpeg-quality "$JPEG_QUALITY" \
  --send-width "$SEND_WIDTH" \
  --send-height "$SEND_HEIGHT" \
  "$SEND_DEPTH_ARG" \
  --depth-encoding "$DEPTH_ENCODING" \
  --timeout-s "$TIMEOUT_S" \
  --upload-timeout-s "$UPLOAD_TIMEOUT_S" \
  --upload-url "$UPLOAD_URL" \
  "$@"
