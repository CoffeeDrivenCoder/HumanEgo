#!/usr/bin/env bash
set -euo pipefail

# Replay fixed HumanEgo DELTA_POSE action_data exported from an interactive run.
# Usage:
#   G1_REPLAY_SEQUENCE_JSON=/path/to/humanego_action_replay_sequence.json \
#   G1_REPLAY_CONFIRM=RUN_CONTROL \
#   bash scripts/run_g1_replay_humanego_action_sequence.sh

cd "$(dirname "$0")/.."

if [ -f "a2d_sdk/env.sh" ]; then
  source a2d_sdk/env.sh || echo "[WARN] failed to source a2d_sdk/env.sh; continuing with current environment" >&2
fi

SEQUENCE_JSON="${G1_REPLAY_SEQUENCE_JSON:-${1:-}}"
if [[ -z "$SEQUENCE_JSON" ]]; then
  echo "ERROR: set G1_REPLAY_SEQUENCE_JSON or pass the sequence JSON path as the first argument" >&2
  exit 2
fi

SESSION="${G1_ARTIFACT_SESSION:-$(date -u +%Y%m%d)}"
OUT_DIR="${G1_REPLAY_OUT_DIR:-./artifacts/g1_humanego/${SESSION}/diagnostics}"
TAG="${G1_REPLAY_TAG:-humanego_action_replay}"
CONTROL_MODE="${G1_REPLAY_CONTROL_MODE:-prompt}"
CONFIRM="${G1_REPLAY_CONFIRM:-}"
SIDE="${G1_REPLAY_SIDE:-}"
MAX_ACTIONS="${G1_REPLAY_MAX_ACTIONS:-0}"
REFERENCE_TIME="${G1_REPLAY_REFERENCE_TIME:-}"
FALLBACK_REFERENCE_TIME="${G1_REPLAY_FALLBACK_REFERENCE_TIME:-1.0}"
EXECUTE_S="${G1_REPLAY_EXECUTE_S:-1.0}"
SETTLE_S="${G1_REPLAY_SETTLE_S:-1.0}"
UPLOAD_URL="${G1_REPLAY_UPLOAD_URL:-${G1_DIAG_UPLOAD_URL:-}}"
UPLOAD_TIMEOUT_S="${G1_REPLAY_UPLOAD_TIMEOUT_S:-20}"

REFERENCE_ARGS=()
if [[ -n "$REFERENCE_TIME" ]]; then
  REFERENCE_ARGS=(--reference-time "$REFERENCE_TIME")
fi

SIDE_ARGS=()
if [[ -n "$SIDE" ]]; then
  SIDE_ARGS=(--side "$SIDE")
fi

python3 scripts/g1_replay_humanego_action_sequence.py "$SEQUENCE_JSON" \
  --out-dir "$OUT_DIR" \
  --tag "$TAG" \
  "${SIDE_ARGS[@]}" \
  --control-mode "$CONTROL_MODE" \
  --confirm-control "$CONFIRM" \
  --max-actions "$MAX_ACTIONS" \
  "${REFERENCE_ARGS[@]}" \
  --fallback-reference-time "$FALLBACK_REFERENCE_TIME" \
  --execute-s "$EXECUTE_S" \
  --settle-s "$SETTLE_S" \
  --upload-timeout-s "$UPLOAD_TIMEOUT_S" \
  --upload-url "$UPLOAD_URL"

