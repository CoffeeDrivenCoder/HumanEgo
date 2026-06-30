#!/usr/bin/env bash
set -euo pipefail

# Replay recorded HumanEgo link7 targets through direct G1 SDK ABS_POSE.

cd "$(dirname "$0")/.."

if [ -f "a2d_sdk/env.sh" ]; then
  source a2d_sdk/env.sh || echo "[WARN] failed to source a2d_sdk/env.sh; continuing with current environment" >&2
fi

if [ "$#" -lt 1 ] && [ -z "${G1_ABS_POSE_REPLAY_SOURCE:-}" ]; then
  echo "usage: $0 <interactive_step_report.json | humanego_action_replay_sequence.json | autoregressive_rollout.json | run_dir> [extra args]" >&2
  echo "or set G1_ABS_POSE_REPLAY_SOURCE=/path/to/source" >&2
  exit 2
fi

SOURCE="${G1_ABS_POSE_REPLAY_SOURCE:-$1}"
if [ "$#" -gt 0 ]; then
  shift
fi

SESSION="${G1_ARTIFACT_SESSION:-$(date -u +%Y%m%d)}"
OUT_DIR="${G1_ABS_POSE_REPLAY_OUT_DIR:-./artifacts/g1_humanego/${SESSION}/diagnostics}"
TAG="${G1_ABS_POSE_REPLAY_TAG:-humanego_abs_pose_replay}"
SIDE="${G1_ABS_POSE_REPLAY_SIDE:-right}"
CONTROL_MODE="${G1_ABS_POSE_REPLAY_CONTROL_MODE:-prompt}"
CONFIRM="${G1_ABS_POSE_REPLAY_CONFIRM:-}"
MAX_ACTIONS="${G1_ABS_POSE_REPLAY_MAX_ACTIONS:-10}"
TARGET_MODE="${G1_ABS_POSE_REPLAY_TARGET_MODE:-full}"
INTERP_POINTS="${G1_ABS_POSE_REPLAY_INTERP_POINTS:-30}"
REFERENCE_TIME="${G1_ABS_POSE_REPLAY_REFERENCE_TIME:-2.0}"
EXECUTE_S="${G1_ABS_POSE_REPLAY_EXECUTE_S:-2.0}"
SETTLE_S="${G1_ABS_POSE_REPLAY_SETTLE_S:-1.0}"
UPLOAD_URL="${G1_ABS_POSE_REPLAY_UPLOAD_URL:-${G1_DIAG_UPLOAD_URL:-}}"
UPLOAD_TIMEOUT_S="${G1_ABS_POSE_REPLAY_UPLOAD_TIMEOUT_S:-20}"

python3 scripts/g1_replay_humanego_abs_pose_sequence.py "$SOURCE" \
  --out-dir "$OUT_DIR" \
  --tag "$TAG" \
  --side "$SIDE" \
  --control-mode "$CONTROL_MODE" \
  --confirm-control "$CONFIRM" \
  --max-actions "$MAX_ACTIONS" \
  --target-mode "$TARGET_MODE" \
  --interp-points "$INTERP_POINTS" \
  --reference-time "$REFERENCE_TIME" \
  --execute-s "$EXECUTE_S" \
  --settle-s "$SETTLE_S" \
  --upload-url "$UPLOAD_URL" \
  --upload-timeout-s "$UPLOAD_TIMEOUT_S" \
  "$@"
