#!/usr/bin/env bash
set -euo pipefail

# Run this on the G1 robot/client side inside the G1 SDK Python environment.
# Default mode is read-only observe. To send a small control probe, set:
#   G1_GRIPPER_MODE=delta
#   G1_GRIPPER_CONFIRM=RUN_CONTROL

cd "$(dirname "$0")/.."

if [ -f "a2d_sdk/env.sh" ]; then
  source a2d_sdk/env.sh || echo "[WARN] failed to source a2d_sdk/env.sh; continuing with current environment" >&2
fi

SESSION="${G1_ARTIFACT_SESSION:-$(date -u +%Y%m%d)}"
OUT_DIR="${G1_GRIPPER_OUT_DIR:-./artifacts/g1_humanego/${SESSION}/diagnostics/gripper_control}"
UPLOAD_URL="${G1_GRIPPER_UPLOAD_URL:-${G1_DIAG_UPLOAD_URL:-}}"
TAG="${G1_GRIPPER_TAG:-gripper_control}"
MODE="${G1_GRIPPER_MODE:-observe}"
CONFIRM="${G1_GRIPPER_CONFIRM:-}"
SIDE="${G1_GRIPPER_SIDE:-right}"
DELTA_RAW="${G1_GRIPPER_DELTA_RAW:-${G1_GRIPPER_DELTA:-0.05}}"
TARGET_RAW="${G1_GRIPPER_TARGET_RAW:-}"
PAYLOAD_FORMAT="${G1_GRIPPER_PAYLOAD_FORMAT:-auto}"
MIN_RAW="${G1_GRIPPER_MIN_RAW:-0.0}"
MAX_RAW="${G1_GRIPPER_MAX_RAW:-1.0}"
SETTLE_S="${G1_GRIPPER_SETTLE_S:-1.0}"
SAMPLES="${G1_GRIPPER_SAMPLES:-3}"
SAMPLE_INTERVAL_S="${G1_GRIPPER_SAMPLE_INTERVAL_S:-0.1}"
CHANGE_THRESHOLD="${G1_GRIPPER_CHANGE_THRESHOLD:-0.005}"
PROMPT="${G1_GRIPPER_PROMPT:-true}"
CLOSE_ROBOT="${G1_GRIPPER_CLOSE_ROBOT:-false}"
UPLOAD_TIMEOUT_S="${G1_DIAG_UPLOAD_TIMEOUT_S:-20}"

PROMPT_ARG="--prompt"
if [[ "$PROMPT" == "false" || "$PROMPT" == "0" ]]; then
  PROMPT_ARG="--no-prompt"
fi

CLOSE_ARG="--no-close-robot"
if [[ "$CLOSE_ROBOT" == "true" || "$CLOSE_ROBOT" == "1" ]]; then
  CLOSE_ARG="--close-robot"
fi

TARGET_ARGS=()
if [[ -n "$TARGET_RAW" ]]; then
  TARGET_ARGS=(--target-raw "$TARGET_RAW")
fi

python3 scripts/g1_verify_gripper_control.py \
  --out-dir "$OUT_DIR" \
  --tag "$TAG" \
  --mode "$MODE" \
  --confirm-control "$CONFIRM" \
  --side "$SIDE" \
  --delta-raw "$DELTA_RAW" \
  "${TARGET_ARGS[@]}" \
  --payload-format "$PAYLOAD_FORMAT" \
  --min-raw "$MIN_RAW" \
  --max-raw "$MAX_RAW" \
  --settle-s "$SETTLE_S" \
  --samples "$SAMPLES" \
  --sample-interval-s "$SAMPLE_INTERVAL_S" \
  --change-threshold "$CHANGE_THRESHOLD" \
  "$PROMPT_ARG" \
  "$CLOSE_ARG" \
  --upload-timeout-s "$UPLOAD_TIMEOUT_S" \
  --upload-url "$UPLOAD_URL" \
  "$@"
