#!/usr/bin/env bash
set -euo pipefail

# Run this on the public server after SSH login.
# Public mapping provided by user:
#   public 111.0.22.33:30002 -> server local 8000

cd "$(dirname "$0")/.."

python3 scripts/g1_diagnostics_receiver.py \
  --host 0.0.0.0 \
  --port "${G1_DIAG_RECEIVER_PORT:-8000}" \
  --out-dir "${G1_DIAG_OUT_DIR:-./g1_diagnostics_uploads}" \
  --unpack
