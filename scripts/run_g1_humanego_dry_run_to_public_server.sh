#!/usr/bin/env bash
set -euo pipefail

# Run this on the G1 robot/client side inside the G1 SDK Python environment.
# This is read-only: it runs HumanEgo inference and target conversion, but sends
# no end-effector or gripper control commands.

cd "$(dirname "$0")/.."

if [ -f "a2d_sdk/env.sh" ]; then
  source a2d_sdk/env.sh || echo "[WARN] failed to source a2d_sdk/env.sh; continuing with current environment" >&2
fi

SESSION="${G1_ARTIFACT_SESSION:-$(date -u +%Y%m%d)}"
OUT_DIR="${G1_HUMANEGO_DRY_RUN_OUT_DIR:-./artifacts/g1_humanego/${SESSION}/client_local_model}"
UPLOAD_URL="${G1_HUMANEGO_UPLOAD_URL:-${G1_DIAG_UPLOAD_URL:-}}"
TAG="${G1_HUMANEGO_TAG:-humanego_dry_run}"
CFG="${G1_HUMANEGO_CFG:-cfg/inference/g1_serve_bread_right.yaml}"
STEPS="${G1_HUMANEGO_STEPS:-1}"
DEVICE="${G1_HUMANEGO_DEVICE:-auto}"
OBJECT_SOURCE="${G1_HUMANEGO_OBJECT_SOURCE:-fixed}"
PREVIEW_STEPS="${G1_HUMANEGO_PREVIEW_STEPS:-3}"
PYTHON_BIN="${G1_HUMANEGO_PYTHON:-python3}"

"$PYTHON_BIN" scripts/g1_humanego_dry_run.py \
  --cfg "$CFG" \
  --out-dir "$OUT_DIR" \
  --tag "$TAG" \
  --steps "$STEPS" \
  --device "$DEVICE" \
  --object-source "$OBJECT_SOURCE" \
  --preview-steps "$PREVIEW_STEPS" \
  --upload-url "$UPLOAD_URL" \
  "$@"
