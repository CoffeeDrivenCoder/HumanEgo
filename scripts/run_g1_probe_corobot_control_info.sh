#!/usr/bin/env bash
set -euo pipefail

# Read-only CoRobot/G1 control capability probe. This does not send motion
# commands; it imports modules, scans local CoRobot package files, and uses
# read-only/OPTIONS HTTP probes.

cd "$(dirname "$0")/.."

if [ -f "a2d_sdk/env.sh" ]; then
  source a2d_sdk/env.sh || echo "[WARN] failed to source a2d_sdk/env.sh; continuing with current environment" >&2
fi

SESSION="${G1_ARTIFACT_SESSION:-$(date -u +%Y%m%d)}"
OUT_DIR="${G1_COROBOT_PROBE_OUT_DIR:-./artifacts/g1_humanego/${SESSION}/diagnostics}"
TAG="${G1_COROBOT_PROBE_TAG:-corobot_control_probe}"
COROBOT_BASE_URL="${G1_COROBOT_BASE_URL:-${COROBOT_BASE_URL:-http://localhost:8765}}"
HTTP_TIMEOUT_S="${G1_COROBOT_PROBE_HTTP_TIMEOUT_S:-2.0}"
UPLOAD_URL="${G1_COROBOT_PROBE_UPLOAD_URL:-${G1_DIAG_UPLOAD_URL:-}}"
UPLOAD_TIMEOUT_S="${G1_COROBOT_PROBE_UPLOAD_TIMEOUT_S:-20}"

python3 scripts/g1_probe_corobot_control_info.py \
  --out-dir "$OUT_DIR" \
  --tag "$TAG" \
  --corobot-base-url "$COROBOT_BASE_URL" \
  --http-timeout-s "$HTTP_TIMEOUT_S" \
  --upload-url "$UPLOAD_URL" \
  --upload-timeout-s "$UPLOAD_TIMEOUT_S" \
  "$@"
