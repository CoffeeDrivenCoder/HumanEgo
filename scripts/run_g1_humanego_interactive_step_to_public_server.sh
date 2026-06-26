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
TARGET_SOURCE="${G1_HUMANEGO_TARGET_SOURCE:-position_keep_orientation}"
TARGET_ADAPTER="${G1_HUMANEGO_TARGET_ADAPTER:-full}"
AXIS_STEP_M="${G1_HUMANEGO_AXIS_STEP_M:-0.01}"
MAX_ORIENTATION_DEG="${G1_HUMANEGO_MAX_ORIENTATION_DEG:-10}"
PROBE_AXIS="${G1_HUMANEGO_PROBE_AXIS:-z}"
PROBE_DEG="${G1_HUMANEGO_PROBE_DEG:-10}"
PROBE_FRAME="${G1_HUMANEGO_PROBE_FRAME:-local}"
LIFETIME="${G1_HUMANEGO_LIFETIME:-1.5}"
SEND_HZ="${G1_HUMANEGO_SEND_HZ:-20}"
EXECUTE_S="${G1_HUMANEGO_EXECUTE_S:-3.0}"
SETTLE_S="${G1_HUMANEGO_SETTLE_S:-1.5}"
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

GRIPPER_TARGET_ARGS=()
if [[ -n "$GRIPPER_TARGET" ]]; then
  GRIPPER_TARGET_ARGS=(--gripper-target "$GRIPPER_TARGET")
fi

python3 scripts/g1_humanego_interactive_step_client.py \
  --cfg "$CFG" \
  --server-url "$SERVER_URL" \
  --out-dir "$OUT_DIR" \
  --tag "$TAG" \
  --confirm-control "$CONFIRM" \
  --max-steps "$MAX_STEPS" \
  --target-source "$TARGET_SOURCE" \
  --approach-object-key "$APPROACH_OBJECT_KEY" \
  --target-adapter "$TARGET_ADAPTER" \
  --axis-step-m "$AXIS_STEP_M" \
  --max-orientation-deg "$MAX_ORIENTATION_DEG" \
  --probe-axis "$PROBE_AXIS" \
  --probe-deg "$PROBE_DEG" \
  --probe-frame "$PROBE_FRAME" \
  --lifetime "$LIFETIME" \
  --send-hz "$SEND_HZ" \
  --execute-s "$EXECUTE_S" \
  --settle-s "$SETTLE_S" \
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
