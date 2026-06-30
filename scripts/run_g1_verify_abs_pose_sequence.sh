#!/usr/bin/env bash
set -euo pipefail

# Direct G1 SDK ABS_POSE trajectory_tracking_control probe.

cd "$(dirname "$0")/.."

if [ -f "a2d_sdk/env.sh" ]; then
  source a2d_sdk/env.sh || echo "[WARN] failed to source a2d_sdk/env.sh; continuing with current environment" >&2
fi

SESSION="${G1_ARTIFACT_SESSION:-$(date -u +%Y%m%d)}"
OUT_DIR="${G1_ABS_POSE_OUT_DIR:-./artifacts/g1_humanego/${SESSION}/diagnostics}"
TAG="${G1_ABS_POSE_TAG:-abs_pose_sequence}"
SIDE="${G1_ABS_POSE_SIDE:-right}"
MODE="${G1_ABS_POSE_MODE:-hold}"
CONTROL_MODE="${G1_ABS_POSE_CONTROL_MODE:-prompt}"
CONFIRM="${G1_ABS_POSE_CONFIRM:-}"
DELTA_AXIS="${G1_ABS_POSE_DELTA_AXIS:-z}"
DELTA_M="${G1_ABS_POSE_DELTA_M:--0.01}"
ROTATION_AXIS="${G1_ABS_POSE_ROTATION_AXIS:-z}"
ROTATION_DEG="${G1_ABS_POSE_ROTATION_DEG:-0.0}"
NUM_POINTS="${G1_ABS_POSE_NUM_POINTS:-30}"
REFERENCE_TIME="${G1_ABS_POSE_REFERENCE_TIME:-2.0}"
EXECUTE_S="${G1_ABS_POSE_EXECUTE_S:-2.0}"
SETTLE_S="${G1_ABS_POSE_SETTLE_S:-1.0}"
UPLOAD_URL="${G1_ABS_POSE_UPLOAD_URL:-${G1_DIAG_UPLOAD_URL:-}}"
UPLOAD_TIMEOUT_S="${G1_ABS_POSE_UPLOAD_TIMEOUT_S:-20}"

python3 scripts/g1_verify_abs_pose_sequence.py \
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
  --reference-time "$REFERENCE_TIME" \
  --execute-s "$EXECUTE_S" \
  --settle-s "$SETTLE_S" \
  --upload-url "$UPLOAD_URL" \
  --upload-timeout-s "$UPLOAD_TIMEOUT_S" \
  "$@"
