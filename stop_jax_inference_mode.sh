#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/piper_jax_inference_v1"
STOPPER="$ROOT/stop_all_rlt_ros.sh"

if [[ ! -f "$STOPPER" ]]; then
  echo "Unified JAX/RLT stopper is missing: $STOPPER" >&2
  exit 2
fi

exec bash "$STOPPER"
