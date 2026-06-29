#!/usr/bin/env bash
set -euo pipefail

# Pure G1 SDK DELTA_POSE sequence probe. This does not call HumanEgo inference.
# It sends a small sequence around the current link7 pose and records commanded
# versus observed motion. Requires G1_DELTA_POSE_CONFIRM=RUN_CONTROL to move.

cd "$(dirname "$0")/.."

if [ -f "a2d_sdk/env.sh" ]; then
  source a2d_sdk/env.sh || echo "[WARN] failed to source a2d_sdk/env.sh; continuing with current environment" >&2
fi

SESSION="${G1_ARTIFACT_SESSION:-$(date -u +%Y%m%d)}"
OUT_DIR="${G1_DELTA_POSE_OUT_DIR:-./artifacts/g1_humanego/${SESSION}/diagnostics}"
TAG="${G1_DELTA_POSE_TAG:-delta_pose_sequence}"
CONTROL_MODE="${G1_DELTA_POSE_CONTROL_MODE:-auto}"
CONFIRM="${G1_DELTA_POSE_CONFIRM:-}"
SIDE="${G1_DELTA_POSE_SIDE:-right}"
STEP_M="${G1_DELTA_POSE_STEP_M:-0.01}"
ROT_DEG="${G1_DELTA_POSE_ROT_DEG:-5.0}"
ROTATION_FRAME="${G1_DELTA_POSE_ROTATION_FRAME:-base}"
REFERENCE_TIME="${G1_DELTA_POSE_REFERENCE_TIME:-0.5}"
EXECUTE_S="${G1_DELTA_POSE_EXECUTE_S:-0.5}"
SETTLE_S="${G1_DELTA_POSE_SETTLE_S:-0.5}"
SEQUENCE_JSON="${G1_DELTA_POSE_SEQUENCE_JSON:-}"
SEQUENCE_FILE="${G1_DELTA_POSE_SEQUENCE_FILE:-}"
UPLOAD_URL="${G1_DELTA_POSE_UPLOAD_URL:-${G1_DIAG_UPLOAD_URL:-}}"
UPLOAD_TIMEOUT_S="${G1_DIAG_UPLOAD_TIMEOUT_S:-20}"

SEQUENCE_ARGS=()
if [[ -n "$SEQUENCE_JSON" ]]; then
  SEQUENCE_ARGS+=(--sequence-json "$SEQUENCE_JSON")
fi
if [[ -n "$SEQUENCE_FILE" ]]; then
  SEQUENCE_ARGS+=(--sequence-file "$SEQUENCE_FILE")
fi

python3 scripts/g1_verify_delta_pose_sequence.py \
  --out-dir "$OUT_DIR" \
  --tag "$TAG" \
  --control-mode "$CONTROL_MODE" \
  --confirm-control "$CONFIRM" \
  --side "$SIDE" \
  --step-m "$STEP_M" \
  --rot-deg "$ROT_DEG" \
  --rotation-frame "$ROTATION_FRAME" \
  --reference-time "$REFERENCE_TIME" \
  --execute-s "$EXECUTE_S" \
  --settle-s "$SETTLE_S" \
  "${SEQUENCE_ARGS[@]}" \
  --upload-timeout-s "$UPLOAD_TIMEOUT_S" \
  --upload-url "$UPLOAD_URL" \
  "$@"
