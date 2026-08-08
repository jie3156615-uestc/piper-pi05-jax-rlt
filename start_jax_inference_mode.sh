#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper for the historical pure-inference entrypoint.
# RLT has its own explicit launcher: start_jax_rlt_mode.sh.
exec bash "$HOME/piper_jax_inference_v1/start_pure_inference_mode.sh"
