#!/usr/bin/env bash
set -eo pipefail

readonly EXPECTED_ACTOR_PROFILE="persistent_c10_filtered_actual_v2"
readonly EXPECTED_RAW_SCHEMA="piper_joint_delta_v5_c10_n10_stride10_behavior_ref50_rank1_joint_r005_d1_0015_d2_001_cone15_gripper_close_knot_r005"
readonly EXPECTED_EXECUTION_SCHEMA="piper_joint_delta_v5_c10_n10_stride10_behavior_ref50_persistent_filtered_actual_r005_d1_0015_d2_001_cone15_gripper_close_assist_r005_d1_0005_d2_0003_boundary0005"
readonly EXPECTED_PROJECTION="rank1_joint_v1_r005_d1_0015_d2_001_cone15_scale33_min020_gripper_close_knot_r005"
readonly EXPECTED_GOVERNOR="persistent_governor_v3_joint_r005_d1_0015_d2_001_cone15_boundary060_gripper_close_r005_d1_0005_d2_0003_boundary0005"
readonly EXPECTED_GRIPPER_MODE="close_only_persistent_v1"

require_value() {
  local label="$1"
  local actual="$2"
  local expected="$3"
  if [[ "$actual" != "$expected" ]]; then
    echo "Gripper-close v3 launch mismatch: ${label}=${actual@Q}, expected ${expected@Q}." >&2
    exit 2
  fi
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -x "$SCRIPT_DIR/.venv/bin/python" \
   && -d "$SCRIPT_DIR/src/openpi" \
   && -d "$SCRIPT_DIR/piper_runtime" ]]; then
  DEFAULT_WORKSPACE="$SCRIPT_DIR"
  DEFAULT_RUNTIME="$SCRIPT_DIR"
else
  DEFAULT_WORKSPACE="$HOME/openpi_jax_piper_lora_v1_20260707"
  DEFAULT_RUNTIME="$HOME/piper_jax_inference_v1"
fi
WORKSPACE="${RLT_V3_WORKSPACE_OVERRIDE:-${OPENPI_WORKSPACE:-${RLT_WORKSPACE:-$DEFAULT_WORKSPACE}}}"
RUNTIME="${RLT_V3_RUNTIME_OVERRIDE:-${PIPER_RLT_RUNTIME:-${RLT_RUNTIME:-$DEFAULT_RUNTIME}}}"
PROJECT_PYTHON="$WORKSPACE/.venv/bin/python"
ROS_PYTHON="${PIPER_RLT_ROS_PYTHON:-$HOME/venvs/pika/bin/python}"
[[ -x "$PROJECT_PYTHON" ]] || {
  echo "Gripper-close v3 requires the selected workspace .venv Python: $PROJECT_PYTHON" >&2
  exit 2
}
[[ -x "$ROS_PYTHON" ]] || {
  echo "Gripper-close v3 ROS runtime Python is missing: $ROS_PYTHON" >&2
  exit 2
}
[[ -d "$RUNTIME/piper_runtime" ]] || {
  echo "Gripper-close v3 runtime package is missing: $RUNTIME/piper_runtime" >&2
  exit 2
}

cd "$RUNTIME"
source "$HOME/pika_ros/install/setup.bash"
set -u

SESSION_ID="${SESSION_ID:-rlt_session_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/rlt_takeover_sessions/${SESSION_ID}}"
EPISODE_PREFIX="${EPISODE_PREFIX:-episode}"
MAX_EPISODES="${MAX_EPISODES:-30}"
DURATION="${DURATION:-120}"
PUBLISH="${PUBLISH:-1}"
RESET_HOME="${RESET_HOME:-1}"
RESET_BEFORE_FIRST="${RESET_BEFORE_FIRST:-0}"
RESET_HOME_TARGET="${RESET_HOME_TARGET:-0,0,0,0,0,0,0}"
RESET_HOME_HOLD="${RESET_HOME_HOLD:-4.0}"
MODEL_SAFETY_PROFILE="${MODEL_SAFETY_PROFILE:-native}"
MODEL_SMOOTHING_TAU="${MODEL_SMOOTHING_TAU:-0.05}"
MODEL_MAX_JOINT_STEP_DEG="${MODEL_MAX_JOINT_STEP_DEG:-3.0}"
MODEL_MAX_GRIPPER_STEP="${MODEL_MAX_GRIPPER_STEP:-0.02}"
HARDWARE_IO="${HARDWARE_IO:-native_sdk}"
MODEL_EXECUTE_STEPS="${MODEL_EXECUTE_STEPS:-50}"
# The isolated base-only lane measured 80.9 ms steady p95 / 82.8 ms steady
# maximum across 100 requests on this 5090. Five 30 Hz frames retain two full
# scheduling frames of margin while staying close to pure H50's four frames.
MODEL_PREFETCH_LEAD_STEPS="${MODEL_PREFETCH_LEAD_STEPS:-5}"
HUMAN_END_TIMEOUT="${HUMAN_END_TIMEOUT:-1.5}"
ACTOR_SHADOW="${ACTOR_SHADOW:-0}"
ACTOR_LIVE="${ACTOR_LIVE:-0}"
ACTOR_LIVE_MAX_CHUNKS="${ACTOR_LIVE_MAX_CHUNKS:-0}"
ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD="${ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD:-0.06}"
ACTOR_SHADOW_EXPECTED_Z_DIM="${ACTOR_SHADOW_EXPECTED_Z_DIM:-2048}"
ACTOR_SHADOW_MAX_LATENCY="${ACTOR_SHADOW_MAX_LATENCY:-0.333}"
ACTION_SCHEMA_FINGERPRINT="${ACTION_SCHEMA_FINGERPRINT:-$EXPECTED_RAW_SCHEMA}"
EXECUTION_ACTION_SCHEMA_FINGERPRINT="${EXECUTION_ACTION_SCHEMA_FINGERPRINT:-$EXPECTED_EXECUTION_SCHEMA}"
ACTOR_PROJECTION_PROFILE="${ACTOR_PROJECTION_PROFILE:-$EXPECTED_PROJECTION}"
ACTOR_EXECUTION_PROFILE="${ACTOR_EXECUTION_PROFILE:-$EXPECTED_ACTOR_PROFILE}"
ACTOR_GOVERNOR_FINGERPRINT="${ACTOR_GOVERNOR_FINGERPRINT:-$EXPECTED_GOVERNOR}"
ACTOR_RESIDUAL_MAX_RAD="${ACTOR_RESIDUAL_MAX_RAD:-0.005}"
ACTOR_RESIDUAL_D1_MAX_RAD="${ACTOR_RESIDUAL_D1_MAX_RAD:-0.0015}"
ACTOR_RESIDUAL_D2_MAX_RAD="${ACTOR_RESIDUAL_D2_MAX_RAD:-0.001}"
ACTOR_DIRECTION_CONE_DEG="${ACTOR_DIRECTION_CONE_DEG:-15.0}"
ACTOR_GRIPPER_RESIDUAL_MODE="${ACTOR_GRIPPER_RESIDUAL_MODE:-$EXPECTED_GRIPPER_MODE}"
ACTOR_GRIPPER_RESIDUAL_MAX_CLOSE_M="${ACTOR_GRIPPER_RESIDUAL_MAX_CLOSE_M:-0.005}"
ACTOR_GRIPPER_RESIDUAL_D1_MAX_M="${ACTOR_GRIPPER_RESIDUAL_D1_MAX_M:-0.0005}"
ACTOR_GRIPPER_RESIDUAL_D2_MAX_M="${ACTOR_GRIPPER_RESIDUAL_D2_MAX_M:-0.0003}"
ACTOR_GRIPPER_MAX_BOUNDARY_JUMP_M="${ACTOR_GRIPPER_MAX_BOUNDARY_JUMP_M:-0.0005}"
ACTOR_GRIPPER_COMMAND_MIN_M="${ACTOR_GRIPPER_COMMAND_MIN_M:-0.0}"
ACTOR_GRIPPER_COMMAND_MAX_M="${ACTOR_GRIPPER_COMMAND_MAX_M:-0.08}"
ACTOR_GRIPPER_RELEASE_REFERENCE_M="${ACTOR_GRIPPER_RELEASE_REFERENCE_M:-0.05}"
ACTOR_GRIPPER_RELEASE_DELTA_M="${ACTOR_GRIPPER_RELEASE_DELTA_M:-0.002}"
RLT_FREEZE_GRIPPER_RESIDUAL="${RLT_FREEZE_GRIPPER_RESIDUAL:-0}"
RLT_BETA_HUMAN_GRIPPER_BC="${RLT_BETA_HUMAN_GRIPPER_BC:-1.0}"
RLT_HUMAN_GRIPPER_Q_FILTER_MODE="${RLT_HUMAN_GRIPPER_Q_FILTER_MODE:-critic_min_advantage_v1}"
RLT_HUMAN_GRIPPER_Q_FILTER_MARGIN="${RLT_HUMAN_GRIPPER_Q_FILTER_MARGIN:-0.0}"
PHASE_CLASSIFIER_CHECKPOINT="${PHASE_CLASSIFIER_CHECKPOINT:-$HOME/rlt_phase_classifiers/greenblock_box_resnet18_v4_manual_intervals/phase_classifier.pt}"
PHASE_CLASSIFIER_DEVICE="${PHASE_CLASSIFIER_DEVICE:-cpu}"
PHASE_ENTER_THRESHOLD="${PHASE_ENTER_THRESHOLD:-0.5}"
PHASE_ENTER_FRAMES="${PHASE_ENTER_FRAMES:-3}"
PHASE_CLASSIFIER_PERIOD="${PHASE_CLASSIFIER_PERIOD:-1}"
MANUAL_TRAINING_ADMISSION="${MANUAL_TRAINING_ADMISSION:-0}"

if [[ "$MANUAL_TRAINING_ADMISSION" != "0" && "$MANUAL_TRAINING_ADMISSION" != "1" ]]; then
  echo "MANUAL_TRAINING_ADMISSION must be 0 or 1; got: $MANUAL_TRAINING_ADMISSION" >&2
  exit 2
fi
require_value ACTION_SCHEMA_FINGERPRINT "$ACTION_SCHEMA_FINGERPRINT" "$EXPECTED_RAW_SCHEMA"
require_value EXECUTION_ACTION_SCHEMA_FINGERPRINT "$EXECUTION_ACTION_SCHEMA_FINGERPRINT" "$EXPECTED_EXECUTION_SCHEMA"
require_value ACTOR_PROJECTION_PROFILE "$ACTOR_PROJECTION_PROFILE" "$EXPECTED_PROJECTION"
require_value ACTOR_EXECUTION_PROFILE "$ACTOR_EXECUTION_PROFILE" "$EXPECTED_ACTOR_PROFILE"
require_value ACTOR_GOVERNOR_FINGERPRINT "$ACTOR_GOVERNOR_FINGERPRINT" "$EXPECTED_GOVERNOR"
require_value ACTOR_GRIPPER_RESIDUAL_MODE "$ACTOR_GRIPPER_RESIDUAL_MODE" "$EXPECTED_GRIPPER_MODE"
require_value ACTOR_GRIPPER_RESIDUAL_MAX_CLOSE_M "$ACTOR_GRIPPER_RESIDUAL_MAX_CLOSE_M" "0.005"
require_value ACTOR_GRIPPER_RESIDUAL_D1_MAX_M "$ACTOR_GRIPPER_RESIDUAL_D1_MAX_M" "0.0005"
require_value ACTOR_GRIPPER_RESIDUAL_D2_MAX_M "$ACTOR_GRIPPER_RESIDUAL_D2_MAX_M" "0.0003"
require_value ACTOR_GRIPPER_MAX_BOUNDARY_JUMP_M "$ACTOR_GRIPPER_MAX_BOUNDARY_JUMP_M" "0.0005"
require_value ACTOR_GRIPPER_COMMAND_MIN_M "$ACTOR_GRIPPER_COMMAND_MIN_M" "0.0"
require_value ACTOR_GRIPPER_COMMAND_MAX_M "$ACTOR_GRIPPER_COMMAND_MAX_M" "0.08"
require_value ACTOR_GRIPPER_RELEASE_REFERENCE_M "$ACTOR_GRIPPER_RELEASE_REFERENCE_M" "0.05"
require_value ACTOR_GRIPPER_RELEASE_DELTA_M "$ACTOR_GRIPPER_RELEASE_DELTA_M" "0.002"
require_value RLT_FREEZE_GRIPPER_RESIDUAL "$RLT_FREEZE_GRIPPER_RESIDUAL" "0"
require_value RLT_BETA_HUMAN_GRIPPER_BC "$RLT_BETA_HUMAN_GRIPPER_BC" "1.0"
require_value RLT_HUMAN_GRIPPER_Q_FILTER_MODE "$RLT_HUMAN_GRIPPER_Q_FILTER_MODE" "critic_min_advantage_v1"
require_value RLT_HUMAN_GRIPPER_Q_FILTER_MARGIN "$RLT_HUMAN_GRIPPER_Q_FILTER_MARGIN" "0.0"

echo "RLT online session"
echo "  session_id    : ${SESSION_ID}"
echo "  output_dir    : ${OUTPUT_DIR}"
echo "  episode_prefix: ${EPISODE_PREFIX}"
echo "  max_episodes  : ${MAX_EPISODES}"
echo "  duration      : ${DURATION}s per episode"
echo "  at time limit : hold current pose and wait indefinitely for operator reward 1/0"
echo "  publish       : ${PUBLISH}"
echo "  reset_home    : ${RESET_HOME}"
echo "  reset_first   : ${RESET_BEFORE_FIRST} (operator-confirmed before the first episode)"
echo "  reset_target  : ${RESET_HOME_TARGET}"
echo "  reset_smooth_s: ${RESET_HOME_HOLD}"
echo "  model_safety  : ${MODEL_SAFETY_PROFILE}"
echo "  model_smooth  : tau=${MODEL_SMOOTHING_TAU}s, joint<=${MODEL_MAX_JOINT_STEP_DEG}deg/frame, gripper<=${MODEL_MAX_GRIPPER_STEP}m/frame"
echo "  hardware_io   : ${HARDWARE_IO}"
echo "  model_steps   : ${MODEL_EXECUTE_STEPS}"
echo "  H50 prefetch  : ${MODEL_PREFETCH_LEAD_STEPS} frames early on an independent base-policy lane"
echo "  H50 hold      : last-target 50Hz bridge keepalive; pre-hold velocity history preserved"
echo "  human_end_s   : ${HUMAN_END_TIMEOUT}"
echo "  actor_shadow  : ${ACTOR_SHADOW} (SFT execute=${MODEL_EXECUTE_STEPS}; Actor C=10)"
echo "  actor_live    : ${ACTOR_LIVE} (RLT source only after frozen phase gate; human remains highest priority)"
echo "  actor_live_max: ${ACTOR_LIVE_MAX_CHUNKS} C=10 chunk(s) per episode (0=unlimited)"
echo "  actor_boundary: <=${ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD} rad at each new C10"
echo "  shadow_z_dim  : ${ACTOR_SHADOW_EXPECTED_Z_DIM}"
echo "  shadow_max_s  : ${ACTOR_SHADOW_MAX_LATENCY}"
echo "  action_schema : ${ACTION_SCHEMA_FINGERPRINT}"
echo "  exec_schema   : ${EXECUTION_ACTION_SCHEMA_FINGERPRINT}"
echo "  actor_project : ${ACTOR_PROJECTION_PROFILE}"
echo "  actor_governor: ${ACTOR_GOVERNOR_FINGERPRINT}"
echo "  actor_execute : ${ACTOR_EXECUTION_PROFILE}"
echo "  actor_limits  : r=${ACTOR_RESIDUAL_MAX_RAD}, d1=${ACTOR_RESIDUAL_D1_MAX_RAD}, d2=${ACTOR_RESIDUAL_D2_MAX_RAD}, cone=${ACTOR_DIRECTION_CONE_DEG}deg"
echo "  gripper learn : all admitted-human intervention candidates, reward 1/0 labels preserved, beta=${RLT_BETA_HUMAN_GRIPPER_BC}"
echo "  gripper Q-gate: ${RLT_HUMAN_GRIPPER_Q_FILTER_MODE}, margin=${RLT_HUMAN_GRIPPER_Q_FILTER_MARGIN}"
echo "  gripper mode  : close-only residual; release/opening remains Pi0.5-authoritative"
echo "  gripper limits: close<=${ACTOR_GRIPPER_RESIDUAL_MAX_CLOSE_M}m, d1<=${ACTOR_GRIPPER_RESIDUAL_D1_MAX_M}m, d2<=${ACTOR_GRIPPER_RESIDUAL_D2_MAX_M}m, C10-boundary<=${ACTOR_GRIPPER_MAX_BOUNDARY_JUMP_M}m"
echo "  gripper range : [${ACTOR_GRIPPER_COMMAND_MIN_M},${ACTOR_GRIPPER_COMMAND_MAX_M}]m"
echo "  release guard : reference>=${ACTOR_GRIPPER_RELEASE_REFERENCE_M}m or opening delta>=${ACTOR_GRIPPER_RELEASE_DELTA_M}m forces residual=0"
echo "  phase_ckpt    : ${PHASE_CLASSIFIER_CHECKPOINT}"
echo "  phase_device  : ${PHASE_CLASSIFIER_DEVICE}"
echo "  phase_enter   : p>=${PHASE_ENTER_THRESHOLD} for ${PHASE_ENTER_FRAMES} classifier frames"
echo "  phase_period  : every ${PHASE_CLASSIFIER_PERIOD} control frame(s)"
echo "  train_admit   : ${MANUAL_TRAINING_ADMISSION} (1=operator must explicitly admit every rewarded episode)"
echo
echo "Inside each episode:"
echo "  s        = arm human takeover; model stops and waits"
echo "  Pika double-click gripper once  = start teleop; replay_include begins on fresh Pika commands"
echo "  Pika double-click gripper again = stop teleop; after ${HUMAN_END_TIMEOUT}s without Pika commands, wait for reward"
echo "  second s/e = fallback: finish motion phase and wait for reward"
echo "  1 / 0    = reward-positive / reward-negative episode label; both are trainable after admission"
echo "  q        = emergency-stop current episode with reward 0"
echo
if [[ "$MANUAL_TRAINING_ADMISSION" == "1" ]]; then
  echo "After reward 1/0 (the report is fail-closed/excluded until your decision):"
  echo "  t / y    = admit this episode and run the A-C update"
  echo "  r / n    = exclude it and replay with the same Actor (no learner hook)"
  echo "  e / q    = exclude it and stop the session"
  echo
fi
echo "After each rewarded episode:"
echo "  type n + Enter to publish reset target, then start the next episode after resetting the scene"
echo "  type e/q + Enter to stop the session"
echo

ARGS=(
  -u -m piper_runtime.rlt_online_session
  --episode-prefix "$EPISODE_PREFIX"
  --output-dir "$OUTPUT_DIR"
  --max-episodes "$MAX_EPISODES"
  --duration "$DURATION"
  --model-safety-profile "$MODEL_SAFETY_PROFILE"
  --model-smoothing-tau "$MODEL_SMOOTHING_TAU"
  --model-max-joint-step-deg "$MODEL_MAX_JOINT_STEP_DEG"
  --model-max-gripper-step "$MODEL_MAX_GRIPPER_STEP"
  --hardware-io "$HARDWARE_IO"
  --model-execute-steps "$MODEL_EXECUTE_STEPS"
  --model-prefetch-lead-steps "$MODEL_PREFETCH_LEAD_STEPS"
  --human-end-timeout "$HUMAN_END_TIMEOUT"
  --action-schema-fingerprint "$ACTION_SCHEMA_FINGERPRINT"
  --execution-action-schema-fingerprint "$EXECUTION_ACTION_SCHEMA_FINGERPRINT"
  --actor-projection-profile "$ACTOR_PROJECTION_PROFILE"
  --actor-execution-profile "$ACTOR_EXECUTION_PROFILE"
  --actor-governor-fingerprint "$ACTOR_GOVERNOR_FINGERPRINT"
  --actor-residual-max-rad "$ACTOR_RESIDUAL_MAX_RAD"
  --actor-live-max-boundary-jump-rad "$ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD"
  --actor-residual-d1-max-rad "$ACTOR_RESIDUAL_D1_MAX_RAD"
  --actor-residual-d2-max-rad "$ACTOR_RESIDUAL_D2_MAX_RAD"
  --actor-direction-cone-deg "$ACTOR_DIRECTION_CONE_DEG"
  --actor-gripper-residual-mode "$ACTOR_GRIPPER_RESIDUAL_MODE"
  --actor-gripper-residual-max-close-m "$ACTOR_GRIPPER_RESIDUAL_MAX_CLOSE_M"
  --actor-gripper-residual-d1-max-m "$ACTOR_GRIPPER_RESIDUAL_D1_MAX_M"
  --actor-gripper-residual-d2-max-m "$ACTOR_GRIPPER_RESIDUAL_D2_MAX_M"
  --actor-gripper-max-boundary-jump-m "$ACTOR_GRIPPER_MAX_BOUNDARY_JUMP_M"
  --actor-gripper-command-min-m "$ACTOR_GRIPPER_COMMAND_MIN_M"
  --actor-gripper-command-max-m "$ACTOR_GRIPPER_COMMAND_MAX_M"
  --actor-gripper-release-reference-m "$ACTOR_GRIPPER_RELEASE_REFERENCE_M"
  --actor-gripper-release-delta-m "$ACTOR_GRIPPER_RELEASE_DELTA_M"
  --reset-home-target "$RESET_HOME_TARGET"
  --reset-home-hold "$RESET_HOME_HOLD"
)

if [[ -n "$PHASE_CLASSIFIER_CHECKPOINT" && "$PHASE_CLASSIFIER_CHECKPOINT" != "0" && "$PHASE_CLASSIFIER_CHECKPOINT" != "none" ]]; then
  ARGS+=(
    --phase-classifier-checkpoint "$PHASE_CLASSIFIER_CHECKPOINT"
    --phase-classifier-device "$PHASE_CLASSIFIER_DEVICE"
    --phase-enter-threshold "$PHASE_ENTER_THRESHOLD"
    --phase-enter-frames "$PHASE_ENTER_FRAMES"
    --phase-classifier-period "$PHASE_CLASSIFIER_PERIOD"
  )
fi

if [[ "$ACTOR_SHADOW" == "1" ]]; then
  ARGS+=(
    --actor-shadow
    --actor-shadow-expected-z-dim "$ACTOR_SHADOW_EXPECTED_Z_DIM"
    --actor-shadow-max-latency "$ACTOR_SHADOW_MAX_LATENCY"
  )
fi

if [[ "$ACTOR_LIVE" == "1" ]]; then
  ARGS+=(
    --actor-live
    --actor-live-authorization I_UNDERSTAND_RLT_ACTOR_CONTROLS_ARM_IN_PHASE
    --actor-live-max-chunks "$ACTOR_LIVE_MAX_CHUNKS"
  )
fi

if [[ "$PUBLISH" == "1" ]]; then
  ARGS+=(--publish --publish-authorization I_UNDERSTAND_RLT_PUBLISHES_ARM_COMMANDS)
fi

if [[ "$RESET_HOME" != "1" ]]; then
  ARGS+=(--no-reset-home)
fi

if [[ "$RESET_BEFORE_FIRST" == "1" ]]; then
  ARGS+=(--reset-before-first)
fi

if [[ "$MANUAL_TRAINING_ADMISSION" == "1" ]]; then
  ARGS+=(--manual-training-admission)
fi

if [[ "${RLT_GRIPPER_V3_PRINT_ARGS_ONLY:-0}" == "1" ]]; then
  printf 'RLT_GRIPPER_V3_SESSION_COMMAND='
  printf '%q ' "$ROS_PYTHON" "${ARGS[@]}" "$@"
  printf '\n'
  exit 0
fi

PYTHONPATH="$RUNTIME:$WORKSPACE/src:$WORKSPACE/packages/openpi-client/src:${PYTHONPATH:-}" \
  exec "$ROS_PYTHON" "${ARGS[@]}" "$@"
