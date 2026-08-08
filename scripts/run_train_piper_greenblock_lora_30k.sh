#!/usr/bin/env bash
set -euo pipefail
source /home/cwzk/openpi_jax_piper_lora_v1_20260707/scripts/activate_openpi_jax_piper_lora.sh
exec python scripts/train.py pi05_piper_greenblock_5090_lora_delta \
  --exp-name piper_greenblock_5090_lora_delta_30k_20260707 \
  --batch-size ${BATCH_SIZE:-8} \
  --num-train-steps ${STEPS:-30000} \
  --save-interval 5000 \
  --keep-period 5000 \
  --no-wandb-enabled
