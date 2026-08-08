#!/usr/bin/env bash
set -euo pipefail

readonly STAGING="${RLT_V3_STAGING:-$HOME/gripper_close_v3_staging_20260727}"
readonly LINEAGE_ROOT="${RLT_EVAL_LINEAGE_ROOT:-$HOME/rlt_online_sessions/greenblock_rlt_fresh_v3_20260729}"
readonly STATE_ROOT="${RLT_EVAL_STATE_ROOT:-$LINEAGE_ROOT/.online_rlt_persistent_gripper_v3_fresh}"
readonly CONFIG_FILE="$STATE_ROOT/config.env"
readonly SESSION_WRAPPER="$STAGING/run_rlt_online_session_gripper_close_v3.sh"
readonly NATIVE_SERVICE="rlt-native-sdk-command-gripper-v3.service"
readonly PURE_CONTROLLER_SERVICE="rlt-piper-controller.service"
readonly TAKEOVER_SOURCE_SERVICE="rlt-takeover-sources.service"
readonly TELEOP_SERVICE="rlt-teleop-controller.service"

STEP_VALUE="latest"
EPISODES_VALUE="20"
ACTOR_LIVE_MAX_CHUNKS_VALUE="0"
DURATION_VALUE="120"
RESET_SECONDS_VALUE="${RLT_EVAL_RESET_SECONDS:-4}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  run_fixed_checkpoint_eval_gripper_close_v3.sh \
    [--step N|latest] \
    [--episodes N] \
    [--actor-live-max-chunks N] \
    [--duration SECONDS] \
    [--reset-seconds SECONDS] \
    [--dry-run]

The evaluator freezes one formally accepted Actor checkpoint, writes all
rollouts under ~/rlt_eval_sessions, never starts Pika/Sense or the online
learner hook, prints a final success-rate summary, and restores the original
selected checkpoint.
EOF
}

while (($#)); do
  case "$1" in
    --step)
      [[ $# -ge 2 ]] || { echo "--step requires a value" >&2; exit 2; }
      STEP_VALUE="$2"
      shift 2
      ;;
    --episodes)
      [[ $# -ge 2 ]] || { echo "--episodes requires a value" >&2; exit 2; }
      EPISODES_VALUE="$2"
      shift 2
      ;;
    --actor-live-max-chunks)
      [[ $# -ge 2 ]] || { echo "--actor-live-max-chunks requires a value" >&2; exit 2; }
      ACTOR_LIVE_MAX_CHUNKS_VALUE="$2"
      shift 2
      ;;
    --duration)
      [[ $# -ge 2 ]] || { echo "--duration requires a value" >&2; exit 2; }
      DURATION_VALUE="$2"
      shift 2
      ;;
    --reset-seconds)
      [[ $# -ge 2 ]] || { echo "--reset-seconds requires a value" >&2; exit 2; }
      RESET_SECONDS_VALUE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ "$STEP_VALUE" == "latest" || "$STEP_VALUE" =~ ^[0-9]+$ ]] || {
  echo "--step must be a non-negative integer or latest: $STEP_VALUE" >&2
  exit 2
}
[[ "$EPISODES_VALUE" =~ ^[1-9][0-9]*$ ]] || {
  echo "--episodes must be a positive integer: $EPISODES_VALUE" >&2
  exit 2
}
[[ "$ACTOR_LIVE_MAX_CHUNKS_VALUE" =~ ^[0-9]+$ ]] || {
  echo "--actor-live-max-chunks must be a non-negative integer: $ACTOR_LIVE_MAX_CHUNKS_VALUE" >&2
  exit 2
}
[[ "$DURATION_VALUE" =~ ^[1-9][0-9]*$ ]] || {
  echo "--duration must be a positive integer: $DURATION_VALUE" >&2
  exit 2
}
[[ "$RESET_SECONDS_VALUE" =~ ^[1-9][0-9]*$ ]] \
  && ((RESET_SECONDS_VALUE <= 30)) || {
  echo "--reset-seconds must be an integer in [1, 30]: $RESET_SECONDS_VALUE" >&2
  exit 2
}
[[ -f "$CONFIG_FILE" ]] || {
  echo "RLT state config is missing: $CONFIG_FILE" >&2
  exit 2
}
[[ -f "$SESSION_WRAPPER" ]] || {
  echo "Gripper-close v3 session wrapper is missing: $SESSION_WRAPPER" >&2
  exit 2
}

# shellcheck disable=SC1090
source "$CONFIG_FILE"

PROJECT_PYTHON="$RLT_WORKSPACE/.venv/bin/python"
ROS_PYTHON="${PIPER_RLT_PIKA_PYTHON:-$HOME/venvs/pika/bin/python}"
[[ -x "$PROJECT_PYTHON" ]] || {
  echo "Project Python is missing: $PROJECT_PYTHON" >&2
  exit 2
}
[[ -x "$ROS_PYTHON" ]] || {
  echo "Pika/ROS Python is missing: $ROS_PYTHON" >&2
  exit 2
}

if [[ "$STEP_VALUE" == "latest" ]]; then
  CHECKPOINT="$(grep -v '^[[:space:]]*$' "$RLT_SELECTED_ACTOR_FILE" | head -n 1 | tr -d '\r' || true)"
  [[ "$CHECKPOINT" =~ /step_([0-9]{8})$ ]] || {
    echo "Selected checkpoint does not end in step_XXXXXXXX: $CHECKPOINT" >&2
    exit 2
  }
  STEP_NUMBER="$((10#${BASH_REMATCH[1]}))"
else
  STEP_NUMBER="$((10#$STEP_VALUE))"
  printf -v STEP_DIR "step_%08d" "$STEP_NUMBER"
  CHECKPOINT="$STATE_ROOT/learner/$STEP_DIR"
fi
printf -v STEP_DIR "step_%08d" "$STEP_NUMBER"

"$PROJECT_PYTHON" - \
  "$CHECKPOINT" \
  "$STATE_ROOT" \
  "$STEP_NUMBER" \
  "$RLT_EXECUTION_ACTION_SCHEMA_FINGERPRINT" \
  "$RLT_ACTOR_EXECUTION_PROFILE" \
  "$RLT_ACTOR_GOVERNOR_FINGERPRINT" \
  "$RLT_BASE_FINGERPRINT" \
  "$RLT_TOKEN_FINGERPRINT" \
  "$RLT_PHASE_FINGERPRINT" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

checkpoint = Path(sys.argv[1]).resolve()
state_root = Path(sys.argv[2]).resolve()
expected_step = int(sys.argv[3])
expected = {
    "action_schema": sys.argv[4],
    "actor_execution_profile": sys.argv[5],
    "actor_governor": sys.argv[6],
    "base_checkpoint": sys.argv[7],
    "rl_token": sys.argv[8],
    "phase_classifier": sys.argv[9],
}
if state_root not in checkpoint.parents:
    raise SystemExit(f"checkpoint is outside the selected lineage state: {checkpoint}")
for name in ("learner.msgpack", "metadata.json"):
    if not (checkpoint / name).is_file():
        raise SystemExit(f"incomplete checkpoint: {checkpoint / name}")
metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
if int(metadata.get("update_step", -1)) != expected_step:
    raise SystemExit(
        f"checkpoint step mismatch: {metadata.get('update_step')} != {expected_step}"
    )
fingerprints = metadata.get("fingerprints")
if not isinstance(fingerprints, dict):
    raise SystemExit("checkpoint metadata has no fingerprint contract")
for key, value in expected.items():
    if fingerprints.get(key) != value:
        raise SystemExit(
            f"checkpoint fingerprint mismatch for {key}: "
            f"{fingerprints.get(key)!r} != {value!r}"
        )

validation_ok = False
for path in (state_root / "learner").glob("validation_*.json"):
    if path.name.endswith(".stdout.json"):
        continue
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if int(report.get("update_step", -1)) == expected_step and report.get("passed") is True:
        validation_ok = True
        break
if not validation_ok:
    raise SystemExit(
        f"step {expected_step} has no passing validation report; "
        "refusing physical evaluation"
    )

actor_name = f"checkpoint:{checkpoint}"
smoke_ok = False
for path in state_root.glob("shadow_acceptance_*.json"):
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    shadow = report.get("shadow")
    if (
        isinstance(shadow, dict)
        and shadow.get("actor_name") == actor_name
        and shadow.get("actor_status") == "ok"
        and shadow.get("latency_ok") is True
        and report.get("actions_finite") is True
        and report.get("a_actor_finite") is True
    ):
        smoke_ok = True
        break
if not smoke_ok:
    raise SystemExit(
        f"step {expected_step} has no successful shadow smoke report; "
        "refusing physical evaluation"
    )
print(f"Validated accepted checkpoint: {checkpoint}")
PY

if ((DRY_RUN == 1)); then
  echo "Dry run passed for $STEP_DIR; no service, selector, ROS, CAN, or robot state was changed."
  exit 0
fi

ACTIVE_JOBS="$(
  "$PROJECT_PYTHON" - <<'PY'
from pathlib import Path

exact_names = {
    "rlt_online_session.py",
    "rlt_takeover_rollout.py",
    "run_online_rlt_update.py",
    "train_real_rlt_jax.py",
    "generate_external_rlt_enrichment_cache.py",
}
exact_modules = {
    "piper_runtime.rlt_online_session",
    "piper_runtime.rlt_takeover_rollout",
}
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        args = (entry / "cmdline").read_bytes().split(b"\0")
        values = [value.decode("utf-8", "replace") for value in args if value]
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if any(Path(value).name in exact_names for value in values) or any(
        value in exact_modules for value in values
    ):
        print(f"{entry.name} {' '.join(values)}")
PY
)"
if [[ -n "$ACTIVE_JOBS" ]]; then
  echo "An online rollout/learner is active. Exit it cleanly before fixed-checkpoint evaluation:" >&2
  printf '%s\n' "$ACTIVE_JOBS" >&2
  exit 2
fi

SELECTOR_DIR="$(dirname "$RLT_SELECTED_ACTOR_FILE")"
mkdir -p "$SELECTOR_DIR"
exec 9>"$SELECTOR_DIR/session.lock"
if ! flock -n 9; then
  echo "Another RLT session owns $SELECTOR_DIR/session.lock." >&2
  exit 2
fi

if ! systemctl --user cat "$RLT_SHADOW_SERVICE" 2>/dev/null \
  | grep -Fq "PIPER_RLT_SELECTED_ACTOR_FILE=$RLT_SELECTED_ACTOR_FILE"; then
  echo "$RLT_SHADOW_SERVICE is not bound to the expected selector." >&2
  exit 2
fi
for legacy_unit in \
  rlt-native-sdk-command.service \
  openpi-rlt-shadow-policy.service \
  openpi-piper-policy.service; do
  if systemctl --user is-active --quiet "$legacy_unit"; then
    echo "Legacy/conflicting service is active: $legacy_unit" >&2
    exit 2
  fi
done

ORIGINAL_SELECTED="$(grep -v '^[[:space:]]*$' "$RLT_SELECTED_ACTOR_FILE" | head -n 1 | tr -d '\r' || true)"
ORIGINAL_SELECTED="${ORIGINAL_SELECTED:-NONE}"
SHADOW_WAS_ACTIVE=0
NATIVE_WAS_ACTIVE=0
PURE_CONTROLLER_WAS_ACTIVE=0
TAKEOVER_SOURCE_WAS_ACTIVE=0
TELEOP_WAS_ACTIVE=0
systemctl --user is-active --quiet "$RLT_SHADOW_SERVICE" && SHADOW_WAS_ACTIVE=1
systemctl --user is-active --quiet "$NATIVE_SERVICE" && NATIVE_WAS_ACTIVE=1
systemctl --user is-active --quiet "$PURE_CONTROLLER_SERVICE" && PURE_CONTROLLER_WAS_ACTIVE=1
systemctl --user is-active --quiet "$TAKEOVER_SOURCE_SERVICE" && TAKEOVER_SOURCE_WAS_ACTIVE=1
systemctl --user is-active --quiet "$TELEOP_SERVICE" && TELEOP_WAS_ACTIVE=1
RESTORED=0

write_selector() {
  local value="$1"
  local temporary="${RLT_SELECTED_ACTOR_FILE}.fixed-eval.$$"
  printf '%s\n' "$value" >"$temporary"
  mv -f "$temporary" "$RLT_SELECTED_ACTOR_FILE"
}

wait_for_policy() {
  timeout 180 bash -c \
    "until systemctl --user is-active --quiet '$RLT_SHADOW_SERVICE' \
      && ss -ltn | grep -q '$RLT_POLICY_HOST:$RLT_POLICY_PORT'; do sleep 2; done"
}

restore_runtime() {
  local status=$?
  if ((RESTORED == 0)); then
    RESTORED=1
    write_selector "$ORIGINAL_SELECTED"
    if ((SHADOW_WAS_ACTIVE == 1)); then
      systemctl --user restart "$RLT_SHADOW_SERVICE" >/dev/null 2>&1 || true
      wait_for_policy >/dev/null 2>&1 || true
    else
      systemctl --user stop "$RLT_SHADOW_SERVICE" >/dev/null 2>&1 || true
    fi
    if ((NATIVE_WAS_ACTIVE == 0)); then
      systemctl --user stop "$NATIVE_SERVICE" >/dev/null 2>&1 || true
    fi
    if ((PURE_CONTROLLER_WAS_ACTIVE == 0)); then
      systemctl --user stop "$PURE_CONTROLLER_SERVICE" >/dev/null 2>&1 || true
    fi
    if ((TAKEOVER_SOURCE_WAS_ACTIVE == 1)); then
      systemctl --user start "$TAKEOVER_SOURCE_SERVICE" >/dev/null 2>&1 || true
    fi
    if ((TELEOP_WAS_ACTIVE == 1)); then
      systemctl --user start "$TELEOP_SERVICE" >/dev/null 2>&1 || true
    fi
    echo "Restored online selector: $ORIGINAL_SELECTED"
  fi
  return "$status"
}
trap restore_runtime EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

EVAL_ID="gripper_close_v3_${STEP_DIR}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${RLT_EVAL_OUTPUT_DIR:-$HOME/rlt_eval_sessions/$EVAL_ID}"
mkdir -p "$OUTPUT_DIR"

# Autonomous evaluation intentionally does not probe, open, or launch the
# Pika/Sense device chain. It uses the single Piper controller only for
# feedback/reset and the persistent native bridge for model commands.
if pgrep -af '[/]opt/ros/noetic/bin/roslaunch .*run_data_capture' >/dev/null \
  || pgrep -af '[/]opt/ros/noetic/bin/roslaunch .*open_sensor_gripper.launch' >/dev/null; then
  echo "A camera/data-capture launch owns the RealSense devices." >&2
  exit 3
fi
systemctl --user stop "$TELEOP_SERVICE" "$TAKEOVER_SOURCE_SERVICE"
systemctl --user start rlt-roscore.service
set +u
source /opt/ros/noetic/setup.bash
set -u
for _ in $(seq 1 50); do
  ROS_MASTER_URI=http://localhost:11311 rosparam list >/dev/null 2>&1 && break
  sleep 0.2
done
systemctl --user restart "$PURE_CONTROLLER_SERVICE"
if ! timeout 20 bash -lc '
  source "$HOME/pika_ros/install/setup.bash"
  while true; do
    controller_count="$(pgrep -fc "[p]iper_ctrl_single_node.py" || true)"
    if rosservice list 2>/dev/null | grep -qx /enable_srv \
      && rostopic list 2>/dev/null | grep -qx /joint_states_single \
      && [[ "$controller_count" == "1" ]]; then
      exit 0
    fi
    sleep 0.2
  done
'; then
  echo "Autonomous Piper controller failed to become ready." >&2
  systemctl --user status "$PURE_CONTROLLER_SERVICE" --no-pager -l >&2 || true
  exit 3
fi

systemctl --user start "$NATIVE_SERVICE"
if ! timeout 15 bash -lc '
  source "$HOME/pika_ros/install/setup.bash"
  until rosnode list 2>/dev/null | grep -qx /rlt_native_sdk_command_bridge; do
    sleep 0.2
  done
'; then
  echo "Piper native SDK command bridge is unavailable." >&2
  exit 3
fi

bash "$RLT_RUNTIME/enable_piper_arm.sh"
write_selector "$CHECKPOINT"
systemctl --user restart "$RLT_SHADOW_SERVICE"
if ! wait_for_policy; then
  echo "Fixed checkpoint policy service did not become ready." >&2
  systemctl --user status "$RLT_SHADOW_SERVICE" --no-pager -l >&2 || true
  exit 3
fi

export RLT_V3_WORKSPACE_OVERRIDE="$RLT_WORKSPACE"
export RLT_V3_RUNTIME_OVERRIDE="$RLT_RUNTIME"
export PIPER_RLT_ROS_PYTHON="$ROS_PYTHON"
export SESSION_ID="$EVAL_ID"
export OUTPUT_DIR
export EPISODE_PREFIX=eval
export MAX_EPISODES="$EPISODES_VALUE"
export DURATION="$DURATION_VALUE"
export PUBLISH=1
export RESET_HOME=1
export RESET_BEFORE_FIRST=1
export RESET_HOME_TARGET="${RESET_HOME_TARGET:-0.020019,0.152472,-0.228603,-0.020857,0.632856,0,0.06517}"
export RESET_HOME_HOLD="$RESET_SECONDS_VALUE"
export HARDWARE_IO=ros_bridge
export MODEL_EXECUTE_STEPS="$RLT_MODEL_EXECUTE_STEPS"
export MODEL_SMOOTHING_TAU="$RLT_EXECUTION_FILTER_TAU_S"
export MODEL_MAX_JOINT_STEP_DEG="${MODEL_MAX_JOINT_STEP_DEG:-3.0}"
export MODEL_MAX_GRIPPER_STEP="${MODEL_MAX_GRIPPER_STEP:-0.02}"
export ACTOR_SHADOW=1
export ACTOR_LIVE=1
export ACTOR_LIVE_MAX_CHUNKS="$ACTOR_LIVE_MAX_CHUNKS_VALUE"
export ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD="$RLT_ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD"
export ACTION_SCHEMA_FINGERPRINT="$RLT_ACTOR_MODEL_ACTION_SCHEMA_FINGERPRINT"
export EXECUTION_ACTION_SCHEMA_FINGERPRINT="$RLT_EXECUTION_ACTION_SCHEMA_FINGERPRINT"
export ACTOR_PROJECTION_PROFILE="$RLT_ACTOR_PROJECTION_PROFILE"
export ACTOR_EXECUTION_PROFILE="$RLT_ACTOR_EXECUTION_PROFILE"
export ACTOR_GOVERNOR_FINGERPRINT="$RLT_ACTOR_GOVERNOR_FINGERPRINT"
export ACTOR_RESIDUAL_MAX_RAD="$RLT_RESIDUAL_MAX"
export ACTOR_RESIDUAL_D1_MAX_RAD="$RLT_RESIDUAL_D1_MAX_RAD"
export ACTOR_RESIDUAL_D2_MAX_RAD="$RLT_RESIDUAL_D2_MAX_RAD"
export ACTOR_DIRECTION_CONE_DEG="$RLT_DIRECTION_CONE_DEG"
export ACTOR_GRIPPER_RESIDUAL_MODE="$RLT_GRIPPER_RESIDUAL_MODE"
export ACTOR_GRIPPER_RESIDUAL_MAX_CLOSE_M="$RLT_GRIPPER_RESIDUAL_MAX"
export ACTOR_GRIPPER_RESIDUAL_D1_MAX_M="$RLT_GRIPPER_RESIDUAL_D1_MAX_M"
export ACTOR_GRIPPER_RESIDUAL_D2_MAX_M="$RLT_GRIPPER_RESIDUAL_D2_MAX_M"
export ACTOR_GRIPPER_MAX_BOUNDARY_JUMP_M="$RLT_GRIPPER_MAX_BOUNDARY_JUMP_M"
export ACTOR_GRIPPER_COMMAND_MIN_M="$RLT_GRIPPER_COMMAND_MIN_M"
export ACTOR_GRIPPER_COMMAND_MAX_M="$RLT_GRIPPER_COMMAND_MAX_M"
export ACTOR_GRIPPER_RELEASE_REFERENCE_M="$RLT_GRIPPER_RELEASE_REFERENCE_M"
export ACTOR_GRIPPER_RELEASE_DELTA_M="$RLT_GRIPPER_RELEASE_DELTA_M"
export RLT_FREEZE_GRIPPER_RESIDUAL
export RLT_BETA_HUMAN_GRIPPER_BC
export RLT_HUMAN_GRIPPER_Q_FILTER_MODE
export RLT_HUMAN_GRIPPER_Q_FILTER_MARGIN
export PHASE_CLASSIFIER_CHECKPOINT="$RLT_PHASE_CHECKPOINT"
export MANUAL_TRAINING_ADMISSION=0

echo "Starting fixed-checkpoint evaluation:"
echo "  checkpoint : $CHECKPOINT"
echo "  episodes   : $EPISODES_VALUE"
echo "  output      : $OUTPUT_DIR"
echo "  learner     : disabled"
echo "  Pika/Sense  : disabled and not started"
echo "  Actor C10   : $ACTOR_LIVE_MAX_CHUNKS_VALUE (0=unlimited)"
echo "  soft reset  : ${RESET_SECONDS_VALUE}s smoothstep to the unchanged SFT start pose"

set +e
bash "$SESSION_WRAPPER" \
  --episode-start-index 0 \
  --policy-host "$RLT_POLICY_HOST" \
  --policy-port "$RLT_POLICY_PORT"
SESSION_STATUS=$?
set -e

"$PROJECT_PYTHON" - \
  "$OUTPUT_DIR" \
  "$CHECKPOINT" \
  "$STEP_NUMBER" \
  "$EPISODES_VALUE" \
  "$SESSION_STATUS" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

output = Path(sys.argv[1]).resolve()
checkpoint = str(Path(sys.argv[2]).resolve())
step = int(sys.argv[3])
requested = int(sys.argv[4])
session_status = int(sys.argv[5])
reports = []
for path in sorted(output.glob("eval_[0-9]*/report.json")):
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if report.get("outcome") == "episode_done" and report.get("terminal_reward") in (
        0,
        0.0,
        1,
        1.0,
    ):
        reports.append((path, report))
successes = sum(float(report["terminal_reward"]) > 0 for _, report in reports)
failures = len(reports) - successes
summary = {
    "format": "openpi_piper_fixed_checkpoint_eval_v1",
    "checkpoint": checkpoint,
    "step": step,
    "requested_episodes": requested,
    "completed_episodes": len(reports),
    "successes": successes,
    "failures": failures,
    "success_rate": None if not reports else successes / len(reports),
    "session_status": session_status,
    "actor_live_chunks_started": sum(
        int(report.get("actor_live_chunks_started", 0)) for _, report in reports
    ),
    "actor_live_chunks_completed": sum(
        int(report.get("actor_live_chunks_completed", 0)) for _, report in reports
    ),
    "episode_reports": [str(path) for path, _ in reports],
    "created_unix": time.time(),
}
summary_path = output / "eval_summary.json"
temporary = output / ".eval_summary.json.tmp"
temporary.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
temporary.replace(summary_path)
rate = "N/A" if summary["success_rate"] is None else f"{100.0 * summary['success_rate']:.2f}%"
print()
print("Fixed-checkpoint evaluation summary")
print(f"  checkpoint : step_{step:08d}")
print(f"  completed  : {len(reports)}/{requested}")
print(f"  success    : {successes}")
print(f"  failure    : {failures}")
print(f"  rate       : {rate}")
print(f"  report     : {summary_path}")
PY

exit "$SESSION_STATUS"
