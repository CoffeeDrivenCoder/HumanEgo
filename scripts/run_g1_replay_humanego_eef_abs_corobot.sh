#!/usr/bin/env bash
set -euo pipefail

# Replay recorded HumanEgo link7 targets through a CoRobot-style EEF_ABS action.
# This requires a robot-side CoRobot endpoint that accepts raw action JSON.

cd "$(dirname "$0")/.."

if [ -f "a2d_sdk/env.sh" ]; then
  source a2d_sdk/env.sh || echo "[WARN] failed to source a2d_sdk/env.sh; continuing with current environment" >&2
fi

SESSION="${G1_ARTIFACT_SESSION:-$(date -u +%Y%m%d)}"
OUT_DIR="${G1_EEF_ABS_OUT_DIR:-./artifacts/g1_humanego/${SESSION}/diagnostics}"
TAG="${G1_EEF_ABS_TAG:-humanego_eef_abs_corobot_replay}"
SIDE="${G1_EEF_ABS_SIDE:-right}"
CONTROL_MODE="${G1_EEF_ABS_CONTROL_MODE:-prompt}"
CONFIRM="${G1_EEF_ABS_CONFIRM:-}"
MAX_ACTIONS="${G1_EEF_ABS_MAX_ACTIONS:-10}"
DURATION_S="${G1_EEF_ABS_DURATION_S:-2.0}"
SETTLE_S="${G1_EEF_ABS_SETTLE_S:-1.0}"
COROBOT_ACTION_URL="${G1_EEF_ABS_COROBOT_ACTION_URL:-}"
COROBOT_TIMEOUT_S="${G1_EEF_ABS_COROBOT_TIMEOUT_S:-10}"
UPLOAD_URL="${G1_EEF_ABS_UPLOAD_URL:-${G1_DIAG_UPLOAD_URL:-}}"
UPLOAD_TIMEOUT_S="${G1_EEF_ABS_UPLOAD_TIMEOUT_S:-20}"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <interactive_step_report.json | humanego_action_replay_sequence.json | run_dir> [extra args]" >&2
  exit 2
fi

COROBOT_URL_ARGS=()
if [[ -n "$COROBOT_ACTION_URL" ]]; then
  COROBOT_URL_ARGS=(--corobot-action-url "$COROBOT_ACTION_URL")
fi

python3 scripts/g1_replay_humanego_eef_abs_corobot.py \
  "$1" \
  --out-dir "$OUT_DIR" \
  --tag "$TAG" \
  --side "$SIDE" \
  --control-mode "$CONTROL_MODE" \
  --confirm-control "$CONFIRM" \
  --max-actions "$MAX_ACTIONS" \
  --duration-s "$DURATION_S" \
  --settle-s "$SETTLE_S" \
  "${COROBOT_URL_ARGS[@]}" \
  --corobot-timeout-s "$COROBOT_TIMEOUT_S" \
  --upload-url "$UPLOAD_URL" \
  --upload-timeout-s "$UPLOAD_TIMEOUT_S" \
  "${@:2}"
