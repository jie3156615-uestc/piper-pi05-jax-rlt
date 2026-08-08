#!/usr/bin/env bash
set -euo pipefail

readonly BOOTSTRAP_LINEAGE_MODE="persistent_gripper_v3_bootstrap_warm_start"
readonly BOOTSTRAP_REPLAY_POLICY="immutable_migrated_v5_warmup_plus_persistent_v5_online"
readonly FRESH_ZERO_LINEAGE_MODE="persistent_gripper_v3_fresh_zero"
readonly FRESH_ZERO_REPLAY_POLICY="fresh_persistent_v5_online_only"
readonly EXPECTED_ACTOR_PROFILE="persistent_c10_filtered_actual_v2"
readonly EXPECTED_RAW_SCHEMA="piper_joint_delta_v5_c10_n10_stride10_behavior_ref50_rank1_joint_r005_d1_0015_d2_001_cone15_gripper_close_knot_r005"
readonly EXPECTED_EXECUTION_SCHEMA="piper_joint_delta_v5_c10_n10_stride10_behavior_ref50_persistent_filtered_actual_r005_d1_0015_d2_001_cone15_gripper_close_assist_r005_d1_0005_d2_0003_boundary0005"
readonly EXPECTED_PROJECTION="rank1_joint_v1_r005_d1_0015_d2_001_cone15_scale33_min020_gripper_close_knot_r005"
readonly EXPECTED_GOVERNOR="persistent_governor_v3_joint_r005_d1_0015_d2_001_cone15_boundary060_gripper_close_r005_d1_0005_d2_0003_boundary0005"
readonly EXPECTED_GRIPPER_MODE="close_only_persistent_v1"
readonly EXPECTED_SHADOW_SERVICE="openpi-rlt-shadow-policy-gripper-v3.service"
readonly EXPECTED_HUMAN_GRIPPER_Q_FILTER_MODE="critic_min_advantage_v1"

require_config_value() {
  local key="$1"
  local expected="$2"
  local actual="${!key-}"
  if [[ "$actual" != "$expected" ]]; then
    echo "Gripper-close v3 config mismatch: ${key}=${actual@Q}, expected ${expected@Q}." >&2
    exit 2
  fi
}

CONFIG_FILE="${RLT_ONLINE_CONFIG:?RLT_ONLINE_CONFIG must name the immutable config.env for this session}"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Missing online RLT configuration: $CONFIG_FILE" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -x "$SCRIPT_DIR/.venv/bin/python" \
   && -d "$SCRIPT_DIR/src/openpi" \
   && -d "$SCRIPT_DIR/piper_runtime" ]]; then
  DEFAULT_WORKSPACE="$SCRIPT_DIR"
  DEFAULT_RUNTIME="$SCRIPT_DIR"
else
  DEFAULT_WORKSPACE=""
  DEFAULT_RUNTIME=""
fi
HOOK_SESSION_ROOT="${RLT_SESSION_ROOT:-}"
WORKSPACE_OVERRIDE="${RLT_V3_WORKSPACE_OVERRIDE:-${OPENPI_WORKSPACE:-${RLT_WORKSPACE:-$DEFAULT_WORKSPACE}}}"
RUNTIME_OVERRIDE="${RLT_V3_RUNTIME_OVERRIDE:-${PIPER_RLT_RUNTIME:-${RLT_RUNTIME:-$DEFAULT_RUNTIME}}}"
UPDATER_OVERRIDE="${RLT_V3_UPDATER_OVERRIDE:-}"
# shellcheck disable=SC1090
source "$CONFIG_FILE"

SESSION_ROOT="${RLT_SESSION_ROOT:?RLT_SESSION_ROOT is required by the after-episode hook}"
STATE_ROOT="${RLT_STATE_ROOT:-$SESSION_ROOT/.online_rlt}"
WORKSPACE="${WORKSPACE_OVERRIDE:-${RLT_WORKSPACE:-$HOME/openpi_jax_piper_lora_v1_20260707}}"
RUNTIME="${RUNTIME_OVERRIDE:-${RLT_RUNTIME:-$HOME/piper_jax_inference_v1}}"
PYTHON="$WORKSPACE/.venv/bin/python"
UPDATER="${UPDATER_OVERRIDE:-$WORKSPACE/scripts/piper_rlt/tools/run_online_rlt_update.py}"

case "${RLT_LINEAGE_MODE-}:${RLT_REPLAY_TRAINING_POLICY-}" in
  "${BOOTSTRAP_LINEAGE_MODE}:${BOOTSTRAP_REPLAY_POLICY}")
    require_config_value RLT_ALLOW_WARM_START_OBJECTIVE_MIGRATION "1"
    require_config_value RLT_WARMUP_EPISODES "30"
    ;;
  "${FRESH_ZERO_LINEAGE_MODE}:${FRESH_ZERO_REPLAY_POLICY}")
    require_config_value RLT_ALLOW_WARM_START_OBJECTIVE_MIGRATION "0"
    [[ "${RLT_WARMUP_EPISODES-}" =~ ^[1-9][0-9]*$ ]] || {
      echo "Fresh-zero RLT_WARMUP_EPISODES must be a positive integer." >&2
      exit 2
    }
    require_config_value RLT_MIN_NEW_PERSISTENT_COMMITTED_EPISODES "$RLT_WARMUP_EPISODES"
    if [[ -n "${RLT_WARM_START_ACTOR_CHECKPOINT-}" ]]; then
      echo "Fresh-zero lineage must not define RLT_WARM_START_ACTOR_CHECKPOINT." >&2
      exit 2
    fi
    ;;
  *)
    echo "Unsupported gripper-close v3 lineage/replay pair: ${RLT_LINEAGE_MODE-}:${RLT_REPLAY_TRAINING_POLICY-}" >&2
    exit 2
    ;;
esac
require_config_value RLT_ACTOR_EXECUTION_PROFILE "$EXPECTED_ACTOR_PROFILE"
require_config_value RLT_ACTOR_MODEL_ACTION_SCHEMA_FINGERPRINT "$EXPECTED_RAW_SCHEMA"
require_config_value RLT_EXECUTION_ACTION_SCHEMA_FINGERPRINT "$EXPECTED_EXECUTION_SCHEMA"
require_config_value RLT_ACTOR_PROJECTION_PROFILE "$EXPECTED_PROJECTION"
require_config_value RLT_ACTOR_GOVERNOR_FINGERPRINT "$EXPECTED_GOVERNOR"
require_config_value RLT_GRIPPER_RESIDUAL_MODE "$EXPECTED_GRIPPER_MODE"
require_config_value RLT_GRIPPER_RESIDUAL_MAX "0.005"
require_config_value RLT_GRIPPER_RESIDUAL_D1_MAX_M "0.0005"
require_config_value RLT_GRIPPER_RESIDUAL_D2_MAX_M "0.0003"
require_config_value RLT_GRIPPER_MAX_BOUNDARY_JUMP_M "0.0005"
require_config_value RLT_GRIPPER_COMMAND_MIN_M "0.0"
require_config_value RLT_GRIPPER_COMMAND_MAX_M "0.08"
require_config_value RLT_GRIPPER_RELEASE_REFERENCE_M "0.05"
require_config_value RLT_GRIPPER_RELEASE_DELTA_M "0.002"
require_config_value RLT_FREEZE_GRIPPER_RESIDUAL "0"
require_config_value RLT_BETA_BC "20.0"
require_config_value RLT_BETA_HUMAN_BC "0.0"
require_config_value RLT_BETA_HUMAN_GRIPPER_BC "1.0"
require_config_value RLT_HUMAN_GRIPPER_BC_SCALE_M "0.005"
require_config_value RLT_HUMAN_GRIPPER_Q_FILTER_MODE "$EXPECTED_HUMAN_GRIPPER_Q_FILTER_MODE"
require_config_value RLT_HUMAN_GRIPPER_Q_FILTER_MARGIN "0.0"
require_config_value RLT_MIN_SUCCESS_HUMAN_EPISODES "0"
require_config_value RLT_MIN_ADMITTED_HUMAN_EPISODES "1"
require_config_value RLT_SHADOW_SERVICE "$EXPECTED_SHADOW_SERVICE"
RLT_ACTION_SCHEMA_FINGERPRINT="$RLT_ACTOR_MODEL_ACTION_SCHEMA_FINGERPRINT"

[[ -x "$PYTHON" ]] || {
  echo "Gripper-close v3 requires the selected workspace .venv Python: $PYTHON" >&2
  exit 2
}
[[ -f "$UPDATER" ]] || {
  echo "Gripper-close v3 updater is missing: $UPDATER" >&2
  exit 2
}
echo "[online-rlt] gripper-close v3 learner: all admitted-human gripper candidates (reward 1/0 labels preserved), Q-filter=${RLT_HUMAN_GRIPPER_Q_FILTER_MODE} margin=${RLT_HUMAN_GRIPPER_Q_FILTER_MARGIN}, beta=1; close-only r/d1/d2/boundary=5/0.5/0.3/0.5mm."

if [[ -n "$HOOK_SESSION_ROOT" && "$(realpath -m "$HOOK_SESSION_ROOT")" != "$(realpath -m "$SESSION_ROOT")" ]]; then
  echo "Hook/config session mismatch; refusing to read or train another session's replay." >&2
  exit 2
fi
if [[ "$(realpath -m "$STATE_ROOT")" != "$(realpath -m "$SESSION_ROOT")"/* ]]; then
  echo "Hook state root is outside its session root; refusing unsafe cross-session update." >&2
  exit 2
fi

export PYTHONPATH="$WORKSPACE/src:$WORKSPACE/packages/openpi-client/src:$RUNTIME:${PYTHONPATH:-}"
mkdir -p "$STATE_ROOT"

# Level 1 is intentionally report-only. Episode JSONL files can be hundreds
# of MB, so fully re-auditing all trajectories after warmup episodes 1..29
# would create O(N^2) I/O. This candidate count is never used to train; it only
# decides when the strict updater is worth invoking. The strict audit below is
# still authoritative for gate activity and transition construction.
LIGHT_PROBE_JSON="$STATE_ROOT/warmup_candidate_progress.json"
"$PYTHON" "$WORKSPACE/scripts/piper_rlt/tools/probe_online_rlt_warmup_reports.py" \
  --session-root "$SESSION_ROOT" \
  --output "$LIGHT_PROBE_JSON"
readarray -t LIGHT_FIELDS < <(
  "$PYTHON" "$WORKSPACE/scripts/piper_rlt/tools/print_json_fields.py" \
    --input "$LIGHT_PROBE_JSON" \
    --field candidate_episodes \
    --field candidate_successes \
    --field candidate_failures
)
CANDIDATE_EPISODES="${LIGHT_FIELDS[0]:-0}"
CANDIDATE_SUCCESSES="${LIGHT_FIELDS[1]:-0}"
CANDIDATE_FAILURES="${LIGHT_FIELDS[2]:-0}"
INCREMENTAL_CANDIDATE_EPISODES="$CANDIDATE_EPISODES"
BASE_STATE_FIELDS="$(
  "$PYTHON" - "$STATE_ROOT/online_state.json" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.is_file():
    print("0\n0\n0\n0")
else:
    state = json.loads(path.read_text(encoding="utf-8"))
    frozen_base_ids = list(state.get("frozen_base_episode_ids") or [])
    bootstrap_ids = list(state.get("bootstrap_gripper_episode_ids") or [])
    bootstrap_count = int(state.get("bootstrap_gripper_episode_count", len(bootstrap_ids)))
    bootstrap_quality = dict(state.get("bootstrap_gripper_quality") or {})
    bootstrap_quality_count = int(bootstrap_quality.get("episodes", len(bootstrap_ids)))
    bootstrap_successes = int(bootstrap_quality.get("reward_positive_episodes", 0))
    bootstrap_failures = int(bootstrap_quality.get("reward_negative_episodes", 0))

    if bootstrap_count != len(bootstrap_ids):
        raise SystemExit(
            "bootstrap_gripper_episode_count does not match bootstrap_gripper_episode_ids"
        )
    if bootstrap_quality_count != len(bootstrap_ids):
        raise SystemExit(
            "bootstrap_gripper_quality.episodes does not match bootstrap_gripper_episode_ids"
        )
    if bootstrap_ids and bootstrap_successes + bootstrap_failures != len(bootstrap_ids):
        raise SystemExit(
            "bootstrap gripper reward counts do not cover every bootstrap episode"
        )

    print(len(frozen_base_ids))
    print(len(bootstrap_ids))
    print(bootstrap_successes)
    print(bootstrap_failures)
PY
)" || {
  echo "[online-rlt] invalid bootstrap provenance in online_state.json; refusing the learner update." >&2
  exit 2
}
readarray -t BASE_FIELDS <<<"$BASE_STATE_FIELDS"
FROZEN_BASE_EPISODES="${BASE_FIELDS[0]:-0}"
BOOTSTRAP_EPISODES="${BASE_FIELDS[1]:-0}"
BOOTSTRAP_SUCCESSES="${BASE_FIELDS[2]:-0}"
BOOTSTRAP_FAILURES="${BASE_FIELDS[3]:-0}"
if [[ "${RLT_ACTOR_EXECUTION_PROFILE:-}" == "persistent_c10_filtered_actual_v2" \
   && "$FROZEN_BASE_EPISODES" != "0" ]]; then
  echo "[online-rlt] persistent-v2 state contains forbidden frozen-base episodes; refusing mixed replay." >&2
  exit 2
fi
CANDIDATE_EPISODES=$((CANDIDATE_EPISODES + FROZEN_BASE_EPISODES + BOOTSTRAP_EPISODES))
CANDIDATE_SUCCESSES=$((CANDIDATE_SUCCESSES + BOOTSTRAP_SUCCESSES))
CANDIDATE_FAILURES=$((CANDIDATE_FAILURES + BOOTSTRAP_FAILURES))
NEXT_STRICT_CANDIDATE_FILE="$STATE_ROOT/next_strict_candidate_count.txt"
if (( CANDIDATE_EPISODES < RLT_WARMUP_EPISODES )); then
  echo "[online-rlt] cumulative learner readiness: bootstrap=${BOOTSTRAP_EPISODES} + newly-admitted=${INCREMENTAL_CANDIDATE_EPISODES} = ${CANDIDATE_EPISODES}/${RLT_WARMUP_EPISODES}, R+${CANDIDATE_SUCCESSES}/R-${CANDIDATE_FAILURES}."
  echo "[online-rlt] report-only check complete; no trajectory scan, RL-token cache, replay build, or A-C update was run."
  exit 0
fi
SELECTED_ACTOR="$(head -n 1 "$RLT_SELECTED_ACTOR_FILE" 2>/dev/null || true)"
if [[ -z "$SELECTED_ACTOR" || "$SELECTED_ACTOR" == "NONE" ]]; then
  NEXT_STRICT_CANDIDATE="$(head -n 1 "$NEXT_STRICT_CANDIDATE_FILE" 2>/dev/null || true)"
  if [[ "$NEXT_STRICT_CANDIDATE" =~ ^[0-9]+$ ]] \
    && (( CANDIDATE_EPISODES < NEXT_STRICT_CANDIDATE )); then
    echo "[online-rlt] warmup candidate progress: ${CANDIDATE_EPISODES}/${NEXT_STRICT_CANDIDATE} before the next strict audit (last strict trainable count was below 30)."
    echo "[online-rlt] report-only check complete; no repeated full trajectory scan was run."
    exit 0
  fi
fi

# Keep every learner setting in one immutable argument list. The cheap dry-run
# is the authoritative full-trajectory audit. It runs only after the report
# prefilter reaches the warmup target and still performs no enrichment/training.
UPDATE_ARGS=(
  --session-root "$SESSION_ROOT"
  --state-root "$STATE_ROOT"
  --workspace "$WORKSPACE"
  --runtime "$RUNTIME"
  --phase-checkpoint "$RLT_PHASE_CHECKPOINT"
  --selected-checkpoint-file "$RLT_SELECTED_ACTOR_FILE"
  --shadow-service "$RLT_SHADOW_SERVICE"
  --policy-host "$RLT_POLICY_HOST"
  --policy-port "$RLT_POLICY_PORT"
  --warmup-episodes "$RLT_WARMUP_EPISODES"
  --min-success "$RLT_MIN_SUCCESS"
  --min-failure "$RLT_MIN_FAILURE"
  --min-success-human-episodes "$RLT_MIN_SUCCESS_HUMAN_EPISODES"
  --min-admitted-human-episodes "$RLT_MIN_ADMITTED_HUMAN_EPISODES"
  --update-every "$RLT_UPDATE_EVERY"
  --min-warmup-transitions "$RLT_MIN_WARMUP_TRANSITIONS"
  --utd "$RLT_UTD"
  --min-update-steps "$RLT_MIN_UPDATE_STEPS"
  --max-update-steps "$RLT_MAX_UPDATE_STEPS"
  --batch-size "$RLT_BATCH_SIZE"
  --beta-bc "$RLT_BETA_BC"
  --beta-human-bc "$RLT_BETA_HUMAN_BC"
  --beta-human-gripper-bc "$RLT_BETA_HUMAN_GRIPPER_BC"
  --human-gripper-bc-scale-m "$RLT_HUMAN_GRIPPER_BC_SCALE_M"
  --human-gripper-q-filter-mode "$RLT_HUMAN_GRIPPER_Q_FILTER_MODE"
  --human-gripper-q-filter-margin "$RLT_HUMAN_GRIPPER_Q_FILTER_MARGIN"
  --reference-dropout "$RLT_REFERENCE_DROPOUT"
  --target-policy-noise-std "$RLT_TARGET_POLICY_NOISE_STD"
  --target-policy-noise-clip "$RLT_TARGET_POLICY_NOISE_CLIP"
  --residual-max "$RLT_RESIDUAL_MAX"
  --residual-d1-max-rad "$RLT_RESIDUAL_D1_MAX_RAD"
  --residual-d2-max-rad "$RLT_RESIDUAL_D2_MAX_RAD"
  --direction-cone-deg "$RLT_DIRECTION_CONE_DEG"
  --action-schema-fingerprint "$RLT_ACTION_SCHEMA_FINGERPRINT"
  --actor-projection-profile "$RLT_ACTOR_PROJECTION_PROFILE"
  --gripper-residual-max "$RLT_GRIPPER_RESIDUAL_MAX"
  --gripper-residual-mode "$RLT_GRIPPER_RESIDUAL_MODE"
  --gripper-residual-d1-max-m "$RLT_GRIPPER_RESIDUAL_D1_MAX_M"
  --gripper-residual-d2-max-m "$RLT_GRIPPER_RESIDUAL_D2_MAX_M"
  --gripper-max-boundary-jump-m "$RLT_GRIPPER_MAX_BOUNDARY_JUMP_M"
  --gripper-command-min-m "$RLT_GRIPPER_COMMAND_MIN_M"
  --gripper-command-max-m "$RLT_GRIPPER_COMMAND_MAX_M"
  --gripper-release-reference-m "$RLT_GRIPPER_RELEASE_REFERENCE_M"
  --gripper-release-delta-m "$RLT_GRIPPER_RELEASE_DELTA_M"
  --success-fraction "$RLT_SUCCESS_FRACTION"
  --human-fraction "$RLT_HUMAN_FRACTION"
  --validation-fraction "$RLT_VALIDATION_FRACTION"
  --max-validation-td-error "$RLT_MAX_VALIDATION_TD_ERROR"
  --max-actor-q-advantage "$RLT_MAX_ACTOR_Q_ADVANTAGE"
)
UPDATE_ARGS+=(
  --execution-action-schema-fingerprint "$RLT_EXECUTION_ACTION_SCHEMA_FINGERPRINT"
  --actor-execution-profile "$RLT_ACTOR_EXECUTION_PROFILE"
  --execution-filter-profile "$RLT_EXECUTION_FILTER_PROFILE"
  --execution-filter-tau-s "$RLT_EXECUTION_FILTER_TAU_S"
  --control-hz "$RLT_CONTROL_HZ"
  --actor-live-max-boundary-jump-rad "$RLT_ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD"
  --actor-projection-scale-steps "$RLT_ACTOR_PROJECTION_SCALE_STEPS"
  --actor-min-projection-scale "$RLT_ACTOR_MIN_PROJECTION_SCALE"
  --actor-direction-static-threshold-rad "$RLT_ACTOR_DIRECTION_STATIC_THRESHOLD_RAD"
  --actor-governor-fingerprint "$RLT_ACTOR_GOVERNOR_FINGERPRINT"
  --min-new-persistent-committed-episodes "$RLT_MIN_NEW_PERSISTENT_COMMITTED_EPISODES"
)
if [[ -n "${RLT_WARM_START_ACTOR_CHECKPOINT:-}" ]]; then
  UPDATE_ARGS+=(--warm-start-actor-checkpoint "$RLT_WARM_START_ACTOR_CHECKPOINT")
fi
case "${RLT_ALLOW_WARM_START_OBJECTIVE_MIGRATION:-0}" in
  0)
    ;;
  1)
    if [[ -z "${RLT_WARM_START_ACTOR_CHECKPOINT:-}" ]]; then
      echo "Objective migration authorization requires the config-bound v3 Actor warm-start." >&2
      exit 2
    fi
    UPDATE_ARGS+=(--allow-warm-start-objective-migration)
    ;;
  *)
    echo "RLT_ALLOW_WARM_START_OBJECTIVE_MIGRATION must be 0 or 1." >&2
    exit 2
    ;;
esac

PROBE_JSON="$STATE_ROOT/readiness_probe.json"
PROBE_TMP="$STATE_ROOT/.readiness_probe.json.tmp"
if ! "$PYTHON" -u "$UPDATER" "${UPDATE_ARGS[@]}" --dry-run >"$PROBE_TMP"; then
  echo "[online-rlt] warmup readiness audit failed; no cache/training was started and the incumbent is unchanged." >&2
  exit 1
fi
mv -f "$PROBE_TMP" "$PROBE_JSON"

readarray -t PROBE_FIELDS < <(
  "$PYTHON" "$WORKSPACE/scripts/piper_rlt/tools/print_json_fields.py" \
    --input "$PROBE_JSON" \
    --field outcome \
    --field episodes \
    --field warmup_episodes \
    --field successes \
    --field failures \
    --field human_intervention_episodes \
    --field audited_trainable_transitions
)
PROBE_OUTCOME="${PROBE_FIELDS[0]:-unknown}"
TRAINABLE_EPISODES="${PROBE_FIELDS[1]:-0}"
WARMUP_TARGET="${PROBE_FIELDS[2]:-$RLT_WARMUP_EPISODES}"
SUCCESSES="${PROBE_FIELDS[3]:-0}"
FAILURES="${PROBE_FIELDS[4]:-0}"
ADMITTED_HUMAN="${PROBE_FIELDS[5]:-0}"
TRAINABLE_TRANSITIONS="${PROBE_FIELDS[6]:-0}"

if [[ "$PROBE_OUTCOME" == "incumbent_requires_clean_retrain_after_quarantine" ]]; then
  echo "[online-rlt] replay provenance invalidated the incumbent; unloading it before any further rollout." >&2
  exec "$PYTHON" -u "$UPDATER" "${UPDATE_ARGS[@]}"
fi

if [[ "$PROBE_OUTCOME" != "dry_run_would_update" ]]; then
  if [[ -z "$SELECTED_ACTOR" || "$SELECTED_ACTOR" == "NONE" ]]; then
    TRAINABLE_DEFICIT=$((WARMUP_TARGET - TRAINABLE_EPISODES))
    if (( TRAINABLE_DEFICIT < 1 )); then
      TRAINABLE_DEFICIT=1
    fi
    NEXT_STRICT_CANDIDATE=$((CANDIDATE_EPISODES + TRAINABLE_DEFICIT))
    printf '%s\n' "$NEXT_STRICT_CANDIDATE" >"$NEXT_STRICT_CANDIDATE_FILE"
  fi
  echo "[online-rlt] warmup progress: ${TRAINABLE_EPISODES}/${WARMUP_TARGET} admitted episodes, R+${SUCCESSES}/R-${FAILURES}, admitted-human=${ADMITTED_HUMAN}, audited-TD-C10=${TRAINABLE_TRANSITIONS}."
  echo "[online-rlt] no new admitted episode since the last attempt; no cache/replay/A-C update is run. The phase-filtered replay count is reported only by a formal update."
  exit 0
fi

rm -f "$NEXT_STRICT_CANDIDATE_FILE"
echo "[online-rlt] learner gate is ready at ${TRAINABLE_EPISODES} admitted episodes (R+${SUCCESSES}/R-${FAILURES}); preparing the formal A-C update."

# Only an episode boundary that is actually eligible to update reaches this
# expensive path. The readiness audit already identified the only physical
# same-plan C10 rows that can enter replay. Cache every row's phase probability
# but request fresh Tokens only for those strict candidates. Token-only batches
# never invoke Pi0.5, never advance its RNG and never touch ROS/CAN.
ENRICHMENT_DIR="$STATE_ROOT/enrichment"
ENRICHMENT_CACHE="$ENRICHMENT_DIR/fresh_token_cache.jsonl"
mkdir -p "$ENRICHMENT_DIR"
echo "[online-rlt] incrementally refreshing strict-C10 RL-token rows (Token-only batches; no Pi0.5 RNG or robot commands)..."
if ! "$PYTHON" -u \
  "$WORKSPACE/scripts/piper_rlt/tools/generate_external_rlt_enrichment_cache.py" \
  --readiness-probe "$PROBE_JSON" \
  --dataset-root "$SESSION_ROOT" \
  --output "$ENRICHMENT_CACHE" \
  --phase-checkpoint "$RLT_PHASE_CHECKPOINT" \
  --phase-device cpu \
  --policy-host "$RLT_POLICY_HOST" \
  --policy-port "$RLT_POLICY_PORT" \
  --base-fingerprint "$RLT_BASE_FINGERPRINT" \
  --token-fingerprint "$RLT_TOKEN_FINGERPRINT" \
  --incremental \
  --token-only \
  --token-batch-size 4 \
  --skip-invalid-episodes \
  >"$ENRICHMENT_DIR/cache_update.log" 2>&1; then
  echo "[online-rlt] RL-token cache refresh failed; keeping the incumbent Actor unchanged." >&2
  tail -n 40 "$ENRICHMENT_DIR/cache_update.log" >&2 || true
  exit 1
fi
CACHE_ROWS="$(wc -l < "$ENRICHMENT_CACHE")"
echo "[online-rlt] RL-token cache ready: ${CACHE_ROWS} aligned rows"

exec "$PYTHON" -u "$UPDATER" "${UPDATE_ARGS[@]}" --enrichment-cache "$ENRICHMENT_CACHE"
