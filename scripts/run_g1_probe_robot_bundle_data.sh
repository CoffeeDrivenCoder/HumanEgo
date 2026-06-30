#!/usr/bin/env bash
set -euo pipefail

# Read-only probe for RobotBundleData / bundle class compatibility.

cd "$(dirname "$0")/.."

if [ -f "a2d_sdk/env.sh" ]; then
  source a2d_sdk/env.sh || echo "[WARN] failed to source a2d_sdk/env.sh; continuing with current environment" >&2
fi

SESSION="${G1_ARTIFACT_SESSION:-$(date -u +%Y%m%d)}"
OUT_DIR="${G1_BUNDLE_PROBE_OUT_DIR:-./artifacts/g1_humanego/${SESSION}/diagnostics}"
TAG="${G1_BUNDLE_PROBE_TAG:-robot_bundle_data_probe}"
UPLOAD_URL="${G1_BUNDLE_PROBE_UPLOAD_URL:-${G1_DIAG_UPLOAD_URL:-}}"
UPLOAD_TIMEOUT_S="${G1_BUNDLE_PROBE_UPLOAD_TIMEOUT_S:-20}"

python3 scripts/g1_probe_robot_bundle_data.py \
  --out-dir "$OUT_DIR" \
  --tag "$TAG" \
  --upload-url "$UPLOAD_URL" \
  --upload-timeout-s "$UPLOAD_TIMEOUT_S" \
  "$@"
