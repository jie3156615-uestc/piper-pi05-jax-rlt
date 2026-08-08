#!/usr/bin/env bash
set -euo pipefail

SELECTED_FILE="${PIPER_RLT_SELECTED_ACTOR_FILE:-$HOME/openpi_rlt/online_current/selected_actor_checkpoint.txt}"
mkdir -p "$(dirname "$SELECTED_FILE")"
if [[ ! -f "$SELECTED_FILE" ]]; then
  printf 'NONE\n' > "$SELECTED_FILE"
fi

SELECTED="$(grep -v '^[[:space:]]*$' "$SELECTED_FILE" | head -n 1 | tr -d '\r' || true)"
export PIPER_RL_TOKEN_CHECKPOINT="${PIPER_RL_TOKEN_CHECKPOINT:-$HOME/openpi_rlt/rl_tokens/pi05_piper_greenblock_5090_full20k_rlt_token_v1_20260709/step_010000_encoder_only}"
export PIPER_RLT_SHADOW_PORT="${PIPER_RLT_SHADOW_PORT:-8001}"

if [[ -z "$SELECTED" || "$SELECTED" == "NONE" || "$SELECTED" == "none" ]]; then
  export PIPER_RLT_ACTOR_MODE=none
  unset PIPER_RLT_ACTOR_CHECKPOINT || true
  echo "RLT online policy: token-only warmup (no Actor checkpoint selected)"
else
  if [[ ! -f "$SELECTED/learner.msgpack" || ! -f "$SELECTED/metadata.json" ]]; then
    echo "Selected Actor checkpoint is incomplete: $SELECTED" >&2
    exit 2
  fi
  export PIPER_RLT_ACTOR_MODE=checkpoint
  export PIPER_RLT_ACTOR_CHECKPOINT="$SELECTED"
  echo "RLT online policy: Actor checkpoint $SELECTED"
fi

exec bash "$HOME/piper_jax_inference_v1/run_rlt_shadow_policy_server.sh"
