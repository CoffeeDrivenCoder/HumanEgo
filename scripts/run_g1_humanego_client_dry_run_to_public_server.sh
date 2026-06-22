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
PREVIEW_STEPS="${G1_HUMANEGO_PREVIEW_STEPS:-3}"
JPEG_QUALITY="${G1_HUMANEGO_JPEG_QUALITY:-85}"

python3 scripts/g1_humanego_client_dry_run.py \
  --cfg "$CFG" \
  --server-url "$SERVER_URL" \
  --tag "$TAG" \
  --steps "$STEPS" \
  --preview-steps "$PREVIEW_STEPS" \
  --jpeg-quality "$JPEG_QUALITY" \
  --upload-url "$UPLOAD_URL" \
  "$@"
