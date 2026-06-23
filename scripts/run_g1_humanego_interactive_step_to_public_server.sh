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
TAG="${G1_HUMANEGO_TAG:-interactive_step}"
CFG="${G1_HUMANEGO_CFG:-cfg/inference/g1_serve_bread_right.yaml}"
CONFIRM="${G1_HUMANEGO_CONFIRM:-}"
MAX_STEPS="${G1_HUMANEGO_MAX_STEPS:-20}"
TARGET_SOURCE="${G1_HUMANEGO_TARGET_SOURCE:-position_keep_orientation}"
TARGET_ADAPTER="${G1_HUMANEGO_TARGET_ADAPTER:-full}"
AXIS_STEP_M="${G1_HUMANEGO_AXIS_STEP_M:-0.01}"
LIFETIME="${G1_HUMANEGO_LIFETIME:-0.5}"
SEND_HZ="${G1_HUMANEGO_SEND_HZ:-10}"
EXECUTE_S="${G1_HUMANEGO_EXECUTE_S:-1.0}"
SETTLE_S="${G1_HUMANEGO_SETTLE_S:-1.0}"
JPEG_QUALITY="${G1_HUMANEGO_JPEG_QUALITY:-75}"
SEND_WIDTH="${G1_HUMANEGO_SEND_WIDTH:-320}"
SEND_HEIGHT="${G1_HUMANEGO_SEND_HEIGHT:-240}"
SEND_DEPTH="${G1_HUMANEGO_SEND_DEPTH:-true}"
DEPTH_ENCODING="${G1_HUMANEGO_DEPTH_ENCODING:-z16}"
APPROACH_OBJECT_KEY="${G1_HUMANEGO_APPROACH_OBJECT_KEY:-obj1}"
TIMEOUT_S="${G1_HUMANEGO_TIMEOUT_S:-120}"
UPLOAD_TIMEOUT_S="${G1_HUMANEGO_UPLOAD_TIMEOUT_S:-20}"

SEND_DEPTH_ARG="--no-send-depth"
if [[ "$SEND_DEPTH" == "true" || "$SEND_DEPTH" == "1" ]]; then
  SEND_DEPTH_ARG="--send-depth"
fi

python3 scripts/g1_humanego_interactive_step_client.py \
  --cfg "$CFG" \
  --server-url "$SERVER_URL" \
  --tag "$TAG" \
  --confirm-control "$CONFIRM" \
  --max-steps "$MAX_STEPS" \
  --target-source "$TARGET_SOURCE" \
  --approach-object-key "$APPROACH_OBJECT_KEY" \
  --target-adapter "$TARGET_ADAPTER" \
  --axis-step-m "$AXIS_STEP_M" \
  --lifetime "$LIFETIME" \
  --send-hz "$SEND_HZ" \
  --execute-s "$EXECUTE_S" \
  --settle-s "$SETTLE_S" \
  --jpeg-quality "$JPEG_QUALITY" \
  --send-width "$SEND_WIDTH" \
  --send-height "$SEND_HEIGHT" \
  "$SEND_DEPTH_ARG" \
  --depth-encoding "$DEPTH_ENCODING" \
  --timeout-s "$TIMEOUT_S" \
  --upload-timeout-s "$UPLOAD_TIMEOUT_S" \
  --upload-url "$UPLOAD_URL" \
  "$@"
