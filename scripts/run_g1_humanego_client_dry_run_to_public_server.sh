#!/usr/bin/env bash
set -euo pipefail

# Run this on the G1 robot/client side inside the G1 SDK Python environment.
# Server runs HumanEgo; robot sends data and receives target preview.
# This dry-run sends no robot control commands.

cd "$(dirname "$0")/.."

SERVER_URL="${G1_HUMANEGO_SERVER_URL:-http://111.0.22.33:30003/infer}"
UPLOAD_URL="${G1_DIAG_UPLOAD_URL:-http://111.0.22.33:30002/upload}"
TAG="${G1_HUMANEGO_TAG:-client_dry_run}"
CFG="${G1_HUMANEGO_CFG:-cfg/inference/g1_serve_bread_right.yaml}"
STEPS="${G1_HUMANEGO_STEPS:-1}"
PREVIEW_STEPS="${G1_HUMANEGO_PREVIEW_STEPS:-1}"
JPEG_QUALITY="${G1_HUMANEGO_JPEG_QUALITY:-75}"
SEND_WIDTH="${G1_HUMANEGO_SEND_WIDTH:-320}"
SEND_HEIGHT="${G1_HUMANEGO_SEND_HEIGHT:-240}"
TIMEOUT_S="${G1_HUMANEGO_TIMEOUT_S:-120}"
UPLOAD_TIMEOUT_S="${G1_HUMANEGO_UPLOAD_TIMEOUT_S:-60}"
SAVE_DEPTH="${G1_HUMANEGO_SAVE_DEPTH:-false}"
SEND_DEPTH="${G1_HUMANEGO_SEND_DEPTH:-false}"
DEPTH_ENCODING="${G1_HUMANEGO_DEPTH_ENCODING:-z16}"
CLOSE_CAMERA="${G1_HUMANEGO_CLOSE_CAMERA:-false}"

SAVE_DEPTH_ARG="--no-save-depth"
if [[ "$SAVE_DEPTH" == "true" || "$SAVE_DEPTH" == "1" ]]; then
  SAVE_DEPTH_ARG="--save-depth"
fi

CLOSE_CAMERA_ARG="--no-close-camera"
if [[ "$CLOSE_CAMERA" == "true" || "$CLOSE_CAMERA" == "1" ]]; then
  CLOSE_CAMERA_ARG="--close-camera"
fi

SEND_DEPTH_ARG="--no-send-depth"
if [[ "$SEND_DEPTH" == "true" || "$SEND_DEPTH" == "1" ]]; then
  SEND_DEPTH_ARG="--send-depth"
fi

python3 scripts/g1_humanego_client_dry_run.py \
  --cfg "$CFG" \
  --server-url "$SERVER_URL" \
  --tag "$TAG" \
  --steps "$STEPS" \
  --preview-steps "$PREVIEW_STEPS" \
  --jpeg-quality "$JPEG_QUALITY" \
  --send-width "$SEND_WIDTH" \
  --send-height "$SEND_HEIGHT" \
  --timeout-s "$TIMEOUT_S" \
  --upload-timeout-s "$UPLOAD_TIMEOUT_S" \
  "$SAVE_DEPTH_ARG" \
  "$SEND_DEPTH_ARG" \
  --depth-encoding "$DEPTH_ENCODING" \
  "$CLOSE_CAMERA_ARG" \
  --upload-url "$UPLOAD_URL" \
  "$@"
