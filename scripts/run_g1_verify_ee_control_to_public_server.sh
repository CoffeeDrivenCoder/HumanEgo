#!/usr/bin/env bash
set -euo pipefail

# Run this on the G1 robot/client side inside the G1 SDK Python environment.
# Default mode is read-only observe. To send a control probe, set:
#   G1_EE_MODE=hold or move
#   G1_EE_CONFIRM=RUN_CONTROL

cd "$(dirname "$0")/.."

if [ -f "a2d_sdk/env.sh" ]; then
  source a2d_sdk/env.sh || echo "[WARN] failed to source a2d_sdk/env.sh; continuing with current environment" >&2
fi

UPLOAD_URL="${G1_DIAG_UPLOAD_URL:-http://111.0.22.33:30002/upload}"
TAG="${G1_EE_TAG:-ee_control_frame}"
MODE="${G1_EE_MODE:-observe}"
CONFIRM="${G1_EE_CONFIRM:-}"
SIDE="${G1_EE_SIDE:-right}"
DELTA_AXIS="${G1_EE_DELTA_AXIS:-z}"
DELTA_M="${G1_EE_DELTA_M:-0.01}"
LIFETIME="${G1_EE_LIFETIME:-0.5}"
SETTLE_S="${G1_EE_SETTLE_S:-1.0}"
UPLOAD_TIMEOUT_S="${G1_DIAG_UPLOAD_TIMEOUT_S:-20}"

python3 scripts/g1_verify_ee_control_frame.py \
  --tag "$TAG" \
  --mode "$MODE" \
  --confirm-control "$CONFIRM" \
  --side "$SIDE" \
  --delta-axis "$DELTA_AXIS" \
  --delta-m "$DELTA_M" \
  --lifetime "$LIFETIME" \
  --settle-s "$SETTLE_S" \
  --upload-timeout-s "$UPLOAD_TIMEOUT_S" \
  --upload-url "$UPLOAD_URL" \
  "$@"
