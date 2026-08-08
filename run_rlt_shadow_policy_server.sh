#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/openpi_jax_piper_lora_v1_20260707"

export PYTHONPATH="$HOME/piper_jax_inference_v1:$HOME/openpi_jax_piper_lora_v1_20260707/src:$HOME/openpi_jax_piper_lora_v1_20260707/packages/openpi-client/src:${PYTHONPATH:-}"
export PIPER_POLICY_CONFIG="${PIPER_POLICY_CONFIG:-pi05_piper_greenblock_5090_jax_delta_v1}"
export PIPER_POLICY_CHECKPOINT="${PIPER_POLICY_CHECKPOINT:-$HOME/openpi_checkpoints/pi05_piper_greenblock_5090_jax_delta_v1/piper_greenblock_5090_delta_sft_30k_20260707/20000}"
export PIPER_RL_TOKEN_CHECKPOINT="${PIPER_RL_TOKEN_CHECKPOINT:-$HOME/openpi_rlt/rl_tokens/pi05_piper_greenblock_5090_full20k_rlt_token_v1_20260709/step_010000}"
export PIPER_RLT_ACTOR_MODE="${PIPER_RLT_ACTOR_MODE:-none}"
export PIPER_RLT_SHADOW_PORT="${PIPER_RLT_SHADOW_PORT:-8001}"

echo "Starting fail-open RLT shadow policy service"
echo "  base_config : $PIPER_POLICY_CONFIG"
echo "  base_ckpt   : $PIPER_POLICY_CHECKPOINT"
echo "  token_ckpt  : $PIPER_RL_TOKEN_CHECKPOINT"
echo "  actor_mode  : $PIPER_RLT_ACTOR_MODE"
echo "  listen      : 127.0.0.1:$PIPER_RLT_SHADOW_PORT"
echo "  actor output: service metadata only; the authorized runtime mux decides whether it is live"

exec "$HOME/openpi_jax_piper_lora_v1_20260707/.venv/bin/python" -u -m piper_runtime.rlt_shadow_policy_service
