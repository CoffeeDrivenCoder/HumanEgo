#!/usr/bin/env bash
set -euo pipefail

# Validate URDF IK -> SDK ABS_JOINT -> SDK link7 pose on G1.
# Requires G1_IK_JOINT_CONFIRM=RUN_CONTROL to move.

cd "$(dirname "$0")/.."

if [ -f "a2d_sdk/env.sh" ]; then
  source a2d_sdk/env.sh || echo "[WARN] failed to source a2d_sdk/env.sh; continuing with current environment" >&2
fi

SESSION="${G1_ARTIFACT_SESSION:-$(date -u +%Y%m%d)}"
OUT_DIR="${G1_IK_JOINT_OUT_DIR:-./artifacts/g1_humanego/${SESSION}/diagnostics}"
TAG="${G1_IK_JOINT_TAG:-g1_urdf_ik_joint_control}"
URDF_ZIP="${G1_IK_URDF_ZIP:-G1/G1_URDF_Omnipicker.zip}"
SIDE="${G1_IK_JOINT_SIDE:-right}"
ARM_STATE_MAPPING="${G1_IK_ARM_STATE_MAPPING:-left_first}"
WAIST_HEIGHT_OFFSET_M="${G1_IK_WAIST_HEIGHT_OFFSET_M:--0.300}"
CONTROL_MODE="${G1_IK_JOINT_CONTROL_MODE:-prompt}"
CONFIRM="${G1_IK_JOINT_CONFIRM:-}"
DELTA_AXIS="${G1_IK_JOINT_DELTA_AXIS:-z}"
DELTA_M="${G1_IK_JOINT_DELTA_M:-0.01}"
ROTATION_AXIS="${G1_IK_JOINT_ROTATION_AXIS:-z}"
ROTATION_DEG="${G1_IK_JOINT_ROTATION_DEG:-2.0}"
NUM_POINTS="${G1_IK_JOINT_NUM_POINTS:-20}"
REFERENCE_TIME="${G1_IK_JOINT_REFERENCE_TIME:-1.0}"
EXECUTE_S="${G1_IK_JOINT_EXECUTE_S:-1.0}"
SETTLE_S="${G1_IK_JOINT_SETTLE_S:-0.5}"
MAX_NFEV="${G1_IK_MAX_NFEV:-300}"
MAX_JOINT_DELTA_RAD="${G1_IK_JOINT_MAX_DELTA_RAD:-0.35}"
UPLOAD_URL="${G1_IK_JOINT_UPLOAD_URL:-${G1_DIAG_UPLOAD_URL:-}}"
UPLOAD_TIMEOUT_S="${G1_IK_JOINT_UPLOAD_TIMEOUT_S:-20}"

python3 scripts/g1_verify_urdf_ik_joint_control.py \
  --urdf-zip "$URDF_ZIP" \
  --out-dir "$OUT_DIR" \
  --tag "$TAG" \
  --side "$SIDE" \
  --arm-state-mapping "$ARM_STATE_MAPPING" \
  --waist-height-offset-m "$WAIST_HEIGHT_OFFSET_M" \
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
  --max-nfev "$MAX_NFEV" \
  --max-joint-delta-rad "$MAX_JOINT_DELTA_RAD" \
  --upload-timeout-s "$UPLOAD_TIMEOUT_S" \
  --upload-url "$UPLOAD_URL" \
  "$@"
