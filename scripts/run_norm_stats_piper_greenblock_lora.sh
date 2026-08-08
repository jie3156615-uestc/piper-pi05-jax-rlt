#!/usr/bin/env bash
set -euo pipefail
source /home/cwzk/openpi_jax_piper_lora_v1_20260707/scripts/activate_openpi_jax_piper_lora.sh
python scripts/compute_norm_stats.py --config-name pi05_piper_greenblock_5090_lora_delta 2>&1 | tee /home/cwzk/openpi_jax_piper_lora_v1_20260707/logs/compute_norm_stats_piper_greenblock_lora.log
