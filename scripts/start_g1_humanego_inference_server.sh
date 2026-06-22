#!/usr/bin/env bash
set -euo pipefail

# Run this on the public/server machine.
# Public mapping provided by user:
#   public 111.0.22.33:30003 -> server local 50051

cd "$(dirname "$0")/.."

HOST="${G1_HUMANEGO_SERVER_HOST:-0.0.0.0}"
PORT="${G1_HUMANEGO_SERVER_PORT:-50051}"
CFG="${G1_HUMANEGO_CFG:-cfg/inference/g1_serve_bread_right.yaml}"
DEVICE="${G1_HUMANEGO_DEVICE:-auto}"
OBJECT_SOURCE="${G1_HUMANEGO_OBJECT_SOURCE:-}"
OUT_DIR="${G1_HUMANEGO_SERVER_OUT_DIR:-./g1_humanego_server_runs}"

python3 scripts/g1_humanego_inference_server.py \
  --host "$HOST" \
  --port "$PORT" \
  --cfg "$CFG" \
  --device "$DEVICE" \
  --object-source "$OBJECT_SOURCE" \
  --out-dir "$OUT_DIR" \
  "$@"
