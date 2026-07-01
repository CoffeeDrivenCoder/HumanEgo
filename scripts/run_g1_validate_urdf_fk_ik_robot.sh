#!/usr/bin/env bash
set -euo pipefail

# Read-only robot-side validation of URDF FK/IK against SDK motion_status.
# This script does not send any control command.

cd "$(dirname "$0")/.."

if [ -f "a2d_sdk/env.sh" ]; then
  source a2d_sdk/env.sh || echo "[WARN] failed to source a2d_sdk/env.sh; continuing with current environment" >&2
fi

SESSION="${G1_ARTIFACT_SESSION:-$(date -u +%Y%m%d)}"
OUT_DIR="${G1_IK_VALIDATE_OUT_DIR:-./artifacts/g1_humanego/${SESSION}/diagnostics}"
TAG="${G1_IK_VALIDATE_TAG:-g1_urdf_fk_ik_robot_validate}"
URDF_ZIP="${G1_IK_URDF_ZIP:-G1/G1_URDF_Omnipicker.zip}"
ARM_STATE_MAPPING="${G1_IK_ARM_STATE_MAPPING:-left_first}"
TRY_BOTH_MAPPINGS="${G1_IK_TRY_BOTH_MAPPINGS:-true}"
MAX_NFEV="${G1_IK_MAX_NFEV:-300}"
WAIST_HEIGHT_OFFSET_M="${G1_IK_WAIST_HEIGHT_OFFSET_M:-0.0}"
UPLOAD_URL="${G1_IK_UPLOAD_URL:-${G1_DIAG_UPLOAD_URL:-}}"
UPLOAD_TIMEOUT_S="${G1_IK_UPLOAD_TIMEOUT_S:-20}"

TRY_BOTH_ARG="--try-both-mappings"
if [[ "$TRY_BOTH_MAPPINGS" == "false" || "$TRY_BOTH_MAPPINGS" == "0" ]]; then
  TRY_BOTH_ARG="--no-try-both-mappings"
fi

python3 scripts/g1_validate_urdf_fk_ik_robot.py \
  --urdf-zip "$URDF_ZIP" \
  --out-dir "$OUT_DIR" \
  --tag "$TAG" \
  --arm-state-mapping "$ARM_STATE_MAPPING" \
  "$TRY_BOTH_ARG" \
  --max-nfev "$MAX_NFEV" \
  --waist-height-offset-m "$WAIST_HEIGHT_OFFSET_M" \
  --upload-timeout-s "$UPLOAD_TIMEOUT_S" \
  --upload-url "$UPLOAD_URL" \
  "$@"
