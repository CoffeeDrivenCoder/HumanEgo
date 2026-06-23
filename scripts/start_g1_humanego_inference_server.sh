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
RESTART="${G1_HUMANEGO_RESTART:-0}"

port_is_busy() {
  python3 - "$HOST" "$PORT" <<'PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
probe_host = "" if host in {"0.0.0.0", "::"} else host
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind((probe_host, port))
except OSError:
    sys.exit(0)
finally:
    sock.close()
sys.exit(1)
PY
}

matching_server_pids() {
  python3 - "$PWD" "$PORT" <<'PY'
import os
import sys

project_root = os.path.realpath(sys.argv[1])
port = sys.argv[2]

for name in os.listdir("/proc"):
    if not name.isdigit() or int(name) == os.getpid():
        continue
    proc_dir = os.path.join("/proc", name)
    try:
        raw = open(os.path.join(proc_dir, "cmdline"), "rb").read()
        cmd = raw.decode("utf-8", errors="ignore").replace("\0", " ").strip()
        cwd = os.path.realpath(os.readlink(os.path.join(proc_dir, "cwd")))
    except Exception:
        continue
    if "g1_humanego_inference_server.py" not in cmd:
        continue
    if f"--port {port}" not in cmd:
        continue
    if cwd == project_root:
        print(name)
PY
}

if port_is_busy; then
  if [[ "$RESTART" == "1" || "$RESTART" == "true" || "$RESTART" == "yes" ]]; then
    PIDS="$(matching_server_pids || true)"
    if [[ -z "$PIDS" ]]; then
      echo "[g1_humanego_server] port $PORT is busy, but no matching HumanEgo server was found." >&2
      echo "[g1_humanego_server] Inspect it with: ss -ltnp | grep ':$PORT'" >&2
      exit 98
    fi
    echo "[g1_humanego_server] stopping previous HumanEgo server on port $PORT: $PIDS" >&2
    kill $PIDS
    for _ in {1..20}; do
      if ! port_is_busy; then
        break
      fi
      sleep 0.5
    done
    if port_is_busy; then
      echo "[g1_humanego_server] previous server did not stop after SIGTERM; forcing stop: $PIDS" >&2
      kill -9 $PIDS
      sleep 0.5
    fi
  else
    echo "[g1_humanego_server] port $PORT is already in use; server was not started." >&2
    echo "[g1_humanego_server] If this is the old HumanEgo server, restart with:" >&2
    echo "  G1_HUMANEGO_RESTART=1 G1_HUMANEGO_OBJECT_SOURCE=${OBJECT_SOURCE:-rgbd} bash scripts/start_g1_humanego_inference_server.sh" >&2
    echo "[g1_humanego_server] Or inspect it with: ss -ltnp | grep ':$PORT'" >&2
    exit 98
  fi
fi

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
