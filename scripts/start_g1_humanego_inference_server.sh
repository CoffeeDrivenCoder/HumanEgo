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

if [[ "$OBJECT_SOURCE" == "rgbd" ]]; then
  HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub"
  DINO_CACHE_DIR="$HF_CACHE_DIR/models--IDEA-Research--grounding-dino-tiny"
  SAM2_CACHE_DIR="$HF_CACHE_DIR/models--facebook--sam2-hiera-tiny"

  if [[ -z "${HUMANEGO_DINO_MODEL_PATH:-}" && -r "$DINO_CACHE_DIR/refs/main" ]]; then
    DINO_REF="$(cat "$DINO_CACHE_DIR/refs/main")"
    DINO_SNAPSHOT="$DINO_CACHE_DIR/snapshots/$DINO_REF"
    if [[ -d "$DINO_SNAPSHOT" ]]; then
      export HUMANEGO_DINO_MODEL_PATH="$DINO_SNAPSHOT"
    fi
  fi

  if [[ -z "${HUMANEGO_SAM2_CHECKPOINT:-}" && -r "$SAM2_CACHE_DIR/refs/main" ]]; then
    SAM2_REF="$(cat "$SAM2_CACHE_DIR/refs/main")"
    SAM2_CKPT="$SAM2_CACHE_DIR/snapshots/$SAM2_REF/sam2_hiera_tiny.pt"
    if [[ -f "$SAM2_CKPT" ]]; then
      export HUMANEGO_SAM2_CHECKPOINT="$SAM2_CKPT"
    fi
  fi

  if [[ -n "${HUMANEGO_DINO_MODEL_PATH:-}" && -n "${HUMANEGO_SAM2_CHECKPOINT:-}" && -z "${HUMANEGO_HF_LOCAL_ONLY:-}" ]]; then
    export HUMANEGO_HF_LOCAL_ONLY=1
  fi
fi

python3 scripts/g1_humanego_inference_server.py \
  --host "$HOST" \
  --port "$PORT" \
  --cfg "$CFG" \
  --device "$DEVICE" \
  --object-source "$OBJECT_SOURCE" \
  --out-dir "$OUT_DIR" \
  "$@"
