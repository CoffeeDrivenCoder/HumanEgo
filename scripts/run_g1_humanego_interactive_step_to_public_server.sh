#!/usr/bin/env bash
set -euo pipefail

# Run this on the G1 robot/client side.
# Interactive one-step HumanEgo control. It executes only after operator presses
# Enter at each prompt, and only when G1_HUMANEGO_CONFIRM=RUN_CONTROL.

cd "$(dirname "$0")/.."

SERVER_URL="${G1_HUMANEGO_SERVER_URL:-http://111.0.22.33:30003/infer}"
UPLOAD_URL="${G1_DIAG_UPLOAD_URL:-http://111.0.22.33:30002/upload}"
TAG="${G1_HUMANEGO_TAG:-interactive_step}"
CFG="${G1_HUMANEGO_CFG:-cfg/inference/g1_serve_bread_right.yaml}"
CONFIRM="${G1_HUMANEGO_CONFIRM:-}"
MAX_STEPS="${G1_HUMANEGO_MAX_STEPS:-20}"
TARGET_SOURCE="${G1_HUMANEGO_TARGET_SOURCE:-limited}"
LIFETIME="${G1_HUMANEGO_LIFETIME:-0.5}"
SETTLE_S="${G1_HUMANEGO_SETTLE_S:-1.0}"
JPEG_QUALITY="${G1_HUMANEGO_JPEG_QUALITY:-75}"
SEND_WIDTH="${G1_HUMANEGO_SEND_WIDTH:-320}"
SEND_HEIGHT="${G1_HUMANEGO_SEND_HEIGHT:-240}"
SEND_DEPTH="${G1_HUMANEGO_SEND_DEPTH:-false}"
DEPTH_ENCODING="${G1_HUMANEGO_DEPTH_ENCODING:-z16}"

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
  --lifetime "$LIFETIME" \
  --settle-s "$SETTLE_S" \
  --jpeg-quality "$JPEG_QUALITY" \
  --send-width "$SEND_WIDTH" \
  --send-height "$SEND_HEIGHT" \
  "$SEND_DEPTH_ARG" \
  --depth-encoding "$DEPTH_ENCODING" \
  --upload-url "$UPLOAD_URL" \
  "$@"
