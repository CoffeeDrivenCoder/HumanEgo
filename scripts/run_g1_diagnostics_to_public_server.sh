#!/usr/bin/env bash
set -euo pipefail

# Run this on the G1 robot/client side inside the G1 SDK Python environment.
# Public upload URL:
#   server local 8000 <- public 111.0.22.33:30002

cd "$(dirname "$0")/.."

UPLOAD_URL="${G1_DIAG_UPLOAD_URL:-http://111.0.22.33:30002/upload}"
TAG="${G1_DIAG_TAG:-first_g1_check}"
SAMPLES="${G1_DIAG_SAMPLES:-3}"

python3 -c "from a2d_sdk.robot import CosineCamera, RobotDds, RobotController; print('a2d_sdk ok')"

python3 scripts/g1_collect_diagnostics.py \
  --samples "$SAMPLES" \
  --tag "$TAG" \
  --upload-url "$UPLOAD_URL" \
  "$@"
