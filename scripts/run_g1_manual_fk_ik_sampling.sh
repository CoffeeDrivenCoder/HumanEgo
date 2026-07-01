#!/usr/bin/env bash
set -euo pipefail

# Read-only manual FK/IK sampling validation.
# Move the arm by hand, press Enter, and the script records SDK joints/link7
# plus URDF FK/IK consistency metrics. No robot control command is sent.

cd "$(dirname "$0")/.."

if [ -f "a2d_sdk/env.sh" ]; then
  source a2d_sdk/env.sh || echo "[WARN] failed to source a2d_sdk/env.sh; continuing with current environment" >&2
fi

SESSION="${G1_ARTIFACT_SESSION:-$(date -u +%Y%m%d)}"
OUT_DIR="${G1_MANUAL_FK_IK_OUT_DIR:-./artifacts/g1_humanego/${SESSION}/diagnostics}"
TAG="${G1_MANUAL_FK_IK_TAG:-g1_manual_fk_ik_sampling}"
URDF_ZIP="${G1_IK_URDF_ZIP:-G1/G1_URDF_Omnipicker.zip}"
SIDE="${G1_MANUAL_FK_IK_SIDE:-both}"
SAMPLES="${G1_MANUAL_FK_IK_SAMPLES:-5}"
ARM_STATE_MAPPING="${G1_IK_ARM_STATE_MAPPING:-left_first}"
WAIST_HEIGHT_OFFSET_M="${G1_IK_WAIST_HEIGHT_OFFSET_M:--0.300}"
MAX_NFEV="${G1_IK_MAX_NFEV:-300}"
FK_POSITION_TOLERANCE_M="${G1_MANUAL_FK_IK_FK_POSITION_TOLERANCE_M:-0.001}"
FK_ROTATION_TOLERANCE_DEG="${G1_MANUAL_FK_IK_FK_ROTATION_TOLERANCE_DEG:-0.1}"
IK_POSITION_TOLERANCE_M="${G1_MANUAL_FK_IK_IK_POSITION_TOLERANCE_M:-0.001}"
IK_ROTATION_TOLERANCE_DEG="${G1_MANUAL_FK_IK_IK_ROTATION_TOLERANCE_DEG:-1.0}"
IK_Q_TOLERANCE_RAD="${G1_MANUAL_FK_IK_IK_Q_TOLERANCE_RAD:-0.05}"
SAMPLE_INTERVAL_S="${G1_MANUAL_FK_IK_SAMPLE_INTERVAL_S:-0.0}"
UPLOAD_URL="${G1_MANUAL_FK_IK_UPLOAD_URL:-${G1_DIAG_UPLOAD_URL:-}}"
UPLOAD_TIMEOUT_S="${G1_MANUAL_FK_IK_UPLOAD_TIMEOUT_S:-20}"

NO_PROMPT_ARGS=()
if [[ "${G1_MANUAL_FK_IK_NO_PROMPT:-false}" == "true" || "${G1_MANUAL_FK_IK_NO_PROMPT:-false}" == "1" ]]; then
  NO_PROMPT_ARGS+=(--no-prompt)
fi

python3 scripts/g1_manual_fk_ik_sampling.py \
  --urdf-zip "$URDF_ZIP" \
  --out-dir "$OUT_DIR" \
  --tag "$TAG" \
  --side "$SIDE" \
  --samples "$SAMPLES" \
  --arm-state-mapping "$ARM_STATE_MAPPING" \
  --waist-height-offset-m "$WAIST_HEIGHT_OFFSET_M" \
  --max-nfev "$MAX_NFEV" \
  --fk-position-tolerance-m "$FK_POSITION_TOLERANCE_M" \
  --fk-rotation-tolerance-deg "$FK_ROTATION_TOLERANCE_DEG" \
  --ik-position-tolerance-m "$IK_POSITION_TOLERANCE_M" \
  --ik-rotation-tolerance-deg "$IK_ROTATION_TOLERANCE_DEG" \
  --ik-q-tolerance-rad "$IK_Q_TOLERANCE_RAD" \
  --sample-interval-s "$SAMPLE_INTERVAL_S" \
  "${NO_PROMPT_ARGS[@]}" \
  --upload-timeout-s "$UPLOAD_TIMEOUT_S" \
  --upload-url "$UPLOAD_URL" \
  "$@"
