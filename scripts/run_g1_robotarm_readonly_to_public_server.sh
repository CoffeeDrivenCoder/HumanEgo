#!/usr/bin/env bash
set -euo pipefail

# Run this on the G1 robot/client side inside the G1 SDK Python environment.
# This script is read-only and sends no robot control command.

cd "$(dirname "$0")/.."

UPLOAD_URL="${G1_DIAG_UPLOAD_URL:-http://111.0.22.33:30002/upload}"
TAG="${G1_ROBOTARM_TAG:-robotarm_readonly}"

python3 scripts/g1_robotarm_readonly_check.py \
  --tag "$TAG" \
  --upload-url "$UPLOAD_URL" \
  "$@"
