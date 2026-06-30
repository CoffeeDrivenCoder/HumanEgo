#!/usr/bin/env bash
set -euo pipefail

# Direct CoRobot G01Env EEF_ABS control probe.
# Modes:
#   G1_EEF_ABS_MODE=hold  keeps current link7 pose
#   G1_EEF_ABS_MODE=move  moves a small interpolated absolute-pose trajectory

cd "$(dirname "$0")/.."

if [ -f "a2d_sdk/env.sh" ]; then
  source a2d_sdk/env.sh || echo "[WARN] failed to source a2d_sdk/env.sh; continuing with current environment" >&2
fi

SESSION="${G1_ARTIFACT_SESSION:-$(date -u +%Y%m%d)}"
OUT_DIR="${G1_EEF_ABS_OUT_DIR:-./artifacts/g1_humanego/${SESSION}/diagnostics}"
TAG="${G1_EEF_ABS_TAG:-corobot_eef_abs_verify}"
SIDE="${G1_EEF_ABS_SIDE:-right}"
MODE="${G1_EEF_ABS_MODE:-hold}"
CONTROL_MODE="${G1_EEF_ABS_CONTROL_MODE:-prompt}"
CONFIRM="${G1_EEF_ABS_CONFIRM:-}"
DELTA_AXIS="${G1_EEF_ABS_DELTA_AXIS:-z}"
DELTA_M="${G1_EEF_ABS_DELTA_M:--0.01}"
ROTATION_AXIS="${G1_EEF_ABS_ROTATION_AXIS:-z}"
ROTATION_DEG="${G1_EEF_ABS_ROTATION_DEG:-0.0}"
NUM_POINTS="${G1_EEF_ABS_NUM_POINTS:-30}"
DURATION_S="${G1_EEF_ABS_DURATION_S:-2.0}"
SETTLE_S="${G1_EEF_ABS_SETTLE_S:-1.0}"
UPLOAD_URL="${G1_EEF_ABS_UPLOAD_URL:-${G1_DIAG_UPLOAD_URL:-}}"
UPLOAD_TIMEOUT_S="${G1_EEF_ABS_UPLOAD_TIMEOUT_S:-20}"

python3 scripts/g1_verify_corobot_eef_abs_control.py \
  --out-dir "$OUT_DIR" \
  --tag "$TAG" \
  --side "$SIDE" \
  --mode "$MODE" \
  --control-mode "$CONTROL_MODE" \
  --confirm-control "$CONFIRM" \
  --delta-axis "$DELTA_AXIS" \
  --delta-m "$DELTA_M" \
  --rotation-axis "$ROTATION_AXIS" \
  --rotation-deg "$ROTATION_DEG" \
  --num-points "$NUM_POINTS" \
  --duration-s "$DURATION_S" \
  --settle-s "$SETTLE_S" \
  --upload-url "$UPLOAD_URL" \
  --upload-timeout-s "$UPLOAD_TIMEOUT_S" \
  "$@"
