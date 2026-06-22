#!/usr/bin/env bash
set -euo pipefail

# Run this on the G1 robot/client side inside the G1 SDK Python environment.

cd "$(dirname "$0")/.."

UPLOAD_URL="${G1_DIAG_UPLOAD_URL:-http://111.0.22.33:30002/upload}"
TAG="${G1_T_VERIFY_TAG:-transform_verify}"
SIDE="both"

python3 scripts/g1_verify_camera_transform.py \
  --tag "$TAG" \
  --side "$SIDE" \
  --upload-url "$UPLOAD_URL" \
  "$@"
