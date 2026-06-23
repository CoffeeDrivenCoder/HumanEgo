#!/usr/bin/env bash
set -euo pipefail

# Run this on the G1 robot/client side inside the G1 SDK Python environment.

cd "$(dirname "$0")/.."

if [ -f "a2d_sdk/env.sh" ]; then
  source a2d_sdk/env.sh || echo "[WARN] failed to source a2d_sdk/env.sh; continuing with current environment" >&2
fi

UPLOAD_URL="${G1_DIAG_UPLOAD_URL:-http://111.0.22.33:30002/upload}"
TAG="${G1_T_VERIFY_TAG:-transform_verify}"
SIDE="both"
UPLOAD_TIMEOUT_S="${G1_DIAG_UPLOAD_TIMEOUT_S:-20}"

python3 scripts/g1_verify_camera_transform.py \
  --tag "$TAG" \
  --side "$SIDE" \
  --upload-timeout-s "$UPLOAD_TIMEOUT_S" \
  --upload-url "$UPLOAD_URL" \
  "$@"
