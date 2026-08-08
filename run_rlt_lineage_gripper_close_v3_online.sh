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
readonly EXPECTED_HUMAN_GRIPPER_Q_FILTER_MODE="critic_min_advantage_v1"
readonly V3_NATIVE_SERVICE="rlt-native-sdk-command-gripper-v3.service"
readonly V3_SHADOW_SERVICE="openpi-rlt-shadow-policy-gripper-v3.service"

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
RUNTIME="${RLT_V3_RUNTIME_OVERRIDE:-${PIPER_RLT_RUNTIME:-${RLT_RUNTIME:-$DEFAULT_RUNTIME}}}"
SESSIONS_ROOT="${RLT_SESSIONS_ROOT:-$HOME/rlt_online_sessions}"
WORKSPACE="${RLT_V3_WORKSPACE_OVERRIDE:-${OPENPI_WORKSPACE:-${RLT_WORKSPACE:-$DEFAULT_WORKSPACE}}}"
RUNTIME="$(realpath -m "$RUNTIME")"
WORKSPACE="$(realpath -m "$WORKSPACE")"
PROJECT_PYTHON="$WORKSPACE/.venv/bin/python"
PIKA_PYTHON="${PIPER_RLT_PIKA_PYTHON:-$HOME/venvs/pika/bin/python}"
VALIDATOR="${RLT_V3_VALIDATOR_OVERRIDE:-$WORKSPACE/scripts/piper_rlt/tools/validate_close_assist_v3_lineage.py}"
FRESH_VALIDATOR="${RLT_V3_FRESH_VALIDATOR_OVERRIDE:-$WORKSPACE/scripts/piper_rlt/tools/init_fresh_close_assist_v3_lineage.py}"
ONLINE_SESSION_SCRIPT="$SCRIPT_DIR/run_rlt_online_session_gripper_close_v3.sh"
UPDATE_HOOK_SCRIPT="$SCRIPT_DIR/run_rlt_online_update_hook_gripper_close_v3.sh"

[[ -x "$PROJECT_PYTHON" ]] || {
  echo "OpenPI project Python is missing or not executable: $PROJECT_PYTHON" >&2
  exit 2
}
[[ -x "$PIKA_PYTHON" ]] || {
  echo "ROS Noetic/Piper ABI Python is missing or not executable: $PIKA_PYTHON" >&2
  exit 2
}
[[ -f "$VALIDATOR" ]] || {
  echo "Gripper-close v3 validator is missing: $VALIDATOR" >&2
  exit 2
}
[[ -f "$FRESH_VALIDATOR" ]] || {
  echo "Fresh-zero gripper-close v3 validator is missing: $FRESH_VALIDATOR" >&2
  exit 2
}
[[ -f "$ONLINE_SESSION_SCRIPT" && -f "$UPDATE_HOOK_SCRIPT" ]] || {
  echo "The gripper-close v3 session/hook pair is incomplete under: $SCRIPT_DIR" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage:
  run_rlt_lineage_gripper_close_v3_online.sh \
    --lineage NAME \
    --latest-episode episode_XXXXXX|NONE \
    [--state-dir .online_rlt_NAME] \
    [--max-episodes N] \
    [--actor-live-max-chunks N] \
    [--actor-execution-profile persistent_c10_filtered_actual_v2] \
    [--dry-run]

--latest-episode is an optimistic-concurrency guard. It must equal the newest
completed episode directory on disk; it never deletes, renames, or overwrites
episodes. The next rollout is always newest+1.
EOF
}

LINEAGE=""
EXPECTED_LATEST=""
STATE_DIR=""
MAX_EPISODES_VALUE="1000000"
ACTOR_LIVE_MAX_CHUNKS_VALUE="0"
ACTOR_EXECUTION_PROFILE_VALUE="${ACTOR_EXECUTION_PROFILE:-$EXPECTED_ACTOR_PROFILE}"
DRY_RUN=0

while (($#)); do
  case "$1" in
    --lineage)
      [[ $# -ge 2 ]] || { echo "--lineage requires a value" >&2; exit 2; }
      LINEAGE="$2"
      shift 2
      ;;
    --latest-episode)
      [[ $# -ge 2 ]] || { echo "--latest-episode requires a value" >&2; exit 2; }
      EXPECTED_LATEST="$2"
      shift 2
      ;;
    --state-dir)
      [[ $# -ge 2 ]] || { echo "--state-dir requires a value" >&2; exit 2; }
      STATE_DIR="$2"
      shift 2
      ;;
    --max-episodes)
      [[ $# -ge 2 ]] || { echo "--max-episodes requires a value" >&2; exit 2; }
      MAX_EPISODES_VALUE="$2"
      shift 2
      ;;
    --actor-live-max-chunks)
      [[ $# -ge 2 ]] || { echo "--actor-live-max-chunks requires a value" >&2; exit 2; }
      ACTOR_LIVE_MAX_CHUNKS_VALUE="$2"
      shift 2
      ;;
    --actor-execution-profile)
      [[ $# -ge 2 ]] || { echo "--actor-execution-profile requires a value" >&2; exit 2; }
      ACTOR_EXECUTION_PROFILE_VALUE="$2"
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

[[ "$LINEAGE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "Invalid or missing --lineage: $LINEAGE" >&2
  exit 2
}
[[ "$EXPECTED_LATEST" == "NONE" || "$EXPECTED_LATEST" =~ ^episode_[0-9]{6}$ ]] || {
  echo "--latest-episode must be NONE or look like episode_000051; got: $EXPECTED_LATEST" >&2
  exit 2
}
[[ "$MAX_EPISODES_VALUE" =~ ^[1-9][0-9]*$ ]] || {
  echo "--max-episodes must be a positive integer; got: $MAX_EPISODES_VALUE" >&2
  exit 2
}
[[ "$ACTOR_LIVE_MAX_CHUNKS_VALUE" =~ ^[0-9]+$ ]] || {
  echo "--actor-live-max-chunks must be a non-negative integer; got: $ACTOR_LIVE_MAX_CHUNKS_VALUE" >&2
  exit 2
}
if [[ "$ACTOR_EXECUTION_PROFILE_VALUE" != "$EXPECTED_ACTOR_PROFILE" ]]; then
  echo "Gripper-close v3 only supports --actor-execution-profile $EXPECTED_ACTOR_PROFILE; got: $ACTOR_EXECUTION_PROFILE_VALUE" >&2
  exit 2
fi

SESSION_ROOT="$SESSIONS_ROOT/$LINEAGE"
[[ -d "$SESSION_ROOT" ]] || {
  echo "Lineage directory does not exist: $SESSION_ROOT" >&2
  exit 2
}

if [[ -n "$STATE_DIR" ]]; then
  [[ "$STATE_DIR" =~ ^[.][A-Za-z0-9._-]+$ ]] || {
    echo "--state-dir must be one direct hidden directory name; got: $STATE_DIR" >&2
    exit 2
  }
  STATE_ROOT="$SESSION_ROOT/$STATE_DIR"
else
  mapfile -t STATE_CANDIDATES < <(
    find "$SESSION_ROOT" -mindepth 1 -maxdepth 1 -type d -name '.online_rlt*' -print \
      | while IFS= read -r candidate; do
          [[ -f "$candidate/online_state.json" && -f "$candidate/config.env" ]] && printf '%s\n' "$candidate"
        done
  )
  if [[ "${#STATE_CANDIDATES[@]}" != "1" ]]; then
    echo "Expected exactly one resumable online state under $SESSION_ROOT; found ${#STATE_CANDIDATES[@]}." >&2
    printf '  %s\n' "${STATE_CANDIDATES[@]:-<none>}" >&2
    echo "Pass --state-dir explicitly when this lineage intentionally has multiple states." >&2
    exit 2
  fi
  STATE_ROOT="${STATE_CANDIDATES[0]}"
  STATE_DIR="$(basename "$STATE_ROOT")"
fi

STATE_FILE="$STATE_ROOT/online_state.json"
CONFIG_FILE="$STATE_ROOT/config.env"
[[ -f "$STATE_FILE" && -f "$CONFIG_FILE" ]] || {
  echo "Incomplete online state: $STATE_ROOT" >&2
  exit 2
}

# Select the authoritative read-only validator from the immutable lineage
# contract before ROS, services, selectors, CAN, or robot state can change.
CONFIG_LINEAGE_MODE="$(sed -n 's/^RLT_LINEAGE_MODE=//p' "$CONFIG_FILE")"
case "$CONFIG_LINEAGE_MODE" in
  "$BOOTSTRAP_LINEAGE_MODE")
    "$PROJECT_PYTHON" "$VALIDATOR" \
      --session-root "$SESSION_ROOT" \
      --state-root "$STATE_ROOT" \
      --config "$CONFIG_FILE" \
      --read-only >/dev/null
    ;;
  "$FRESH_ZERO_LINEAGE_MODE")
    "$PROJECT_PYTHON" "$FRESH_VALIDATOR" \
      --session-root "$SESSION_ROOT" \
      --state-root "$STATE_ROOT" \
      --config "$CONFIG_FILE" \
      --validate-existing >/dev/null
    ;;
  *)
    echo "Unsupported gripper-close v3 lineage mode: $CONFIG_LINEAGE_MODE" >&2
    exit 2
    ;;
esac
PREFLIGHT_OUTPUT="$("$PROJECT_PYTHON" - "$SESSION_ROOT" "$STATE_ROOT" "$EXPECTED_LATEST" <<'PY'
import json
from pathlib import Path
import re
import shlex
import sys

session = Path(sys.argv[1]).resolve()
state_root = Path(sys.argv[2]).resolve()
expected_name = sys.argv[3]

episodes = []
for path in session.glob("episode_[0-9]*"):
    match = re.fullmatch(r"episode_([0-9]+)", path.name)
    if path.is_dir() and match:
        episodes.append((int(match.group(1)), path))
state_path = state_root / "online_state.json"
state = json.loads(state_path.read_text(encoding="utf-8"))
if Path(state.get("session_root", "")).resolve() != session:
    raise SystemExit("online_state.json is bound to a different session root")
floor = int(state.get("episode_index_floor", 0))
if not episodes:
    if expected_name != "NONE" or state.get("lineage_mode") not in {
        "persistent_gripper_v3_bootstrap_warm_start",
        "persistent_gripper_v3_fresh_zero",
    }:
        raise SystemExit("lineage has no episode directories")
    latest_index = floor - 1
    actual_name = "NONE"
else:
    latest_index, latest_path = max(episodes)
    actual_name = f"episode_{latest_index:06d}"
    if actual_name != expected_name:
        raise SystemExit(
            f"stale --latest-episode: requested {expected_name}, actual newest directory is {actual_name}; "
            "inspect the active/last session and retry with the real completed boundary"
        )
    report_path = latest_path / "report.json"
    if not report_path.is_file():
        raise SystemExit(f"newest episode has no report.json and is not safely resumable: {latest_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("outcome") != "episode_done" or report.get("terminal_reward") not in (0, 0.0, 1, 1.0):
        raise SystemExit(f"newest episode is not a completed rewarded boundary: {report_path}")
checkpoint_value = state.get("latest_checkpoint") or state.get("deployment_checkpoint")
if not checkpoint_value:
    if state.get("lineage_mode") != "persistent_gripper_v3_fresh_zero":
        raise SystemExit("online state has no promoted Actor checkpoint")
    checkpoint_text = "NONE"
else:
    checkpoint = Path(checkpoint_value).resolve()
    if state_root not in checkpoint.parents:
        raise SystemExit(f"latest checkpoint is outside this lineage state: {checkpoint}")
    if not (checkpoint / "learner.msgpack").is_file() or not (checkpoint / "metadata.json").is_file():
        raise SystemExit(f"latest promoted checkpoint is incomplete: {checkpoint}")
    checkpoint_text = str(checkpoint)

config = {}
for raw in (state_root / "config.env").read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    parsed = shlex.split(value, posix=True)
    config[key] = parsed[0] if parsed else ""
if Path(config.get("RLT_SESSION_ROOT", "")).resolve() != session:
    raise SystemExit("config.env is bound to a different session root")
if Path(config.get("RLT_STATE_ROOT", "")).resolve() != state_root:
    raise SystemExit("config.env is bound to a different state root")

print(actual_name)
print(f"episode_{max(latest_index + 1, floor):06d}")
print(checkpoint_text)
print(state.get("last_update_episode_count", 0))
print(state.get("update_index", 0))
PY
)"
mapfile -t PREFLIGHT <<<"$PREFLIGHT_OUTPUT"

ACTUAL_LATEST="${PREFLIGHT[0]}"
NEXT_EPISODE="${PREFLIGHT[1]}"
LATEST_CHECKPOINT="${PREFLIGHT[2]}"
TRAINED_EPISODE_COUNT="${PREFLIGHT[3]}"
UPDATE_INDEX="${PREFLIGHT[4]}"

echo "Validated online RLT lineage:"
echo "  lineage               : $LINEAGE"
echo "  session root          : $SESSION_ROOT"
echo "  state root            : $STATE_ROOT"
echo "  newest completed dir  : $ACTUAL_LATEST"
echo "  next rollout          : $NEXT_EPISODE"
echo "  promoted Actor        : $LATEST_CHECKPOINT"
echo "  trained episodes      : $TRAINED_EPISODE_COUNT"
echo "  accepted update index : $UPDATE_INDEX"

# Load and re-check the immutable lineage contract. The validator is
# authoritative; these shell checks additionally make the physical launch
# command self-describing and keep v2 aliases from entering the environment.
# shellcheck disable=SC1090
source "$CONFIG_FILE"

require_config_value() {
  local key="$1"
  local expected="$2"
  local actual="${!key-}"
  if [[ "$actual" != "$expected" ]]; then
    echo "Gripper-close v3 config mismatch: ${key}=${actual@Q}, expected ${expected@Q}." >&2
    exit 2
  fi
}

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
    [[ -z "${RLT_WARM_START_ACTOR_CHECKPOINT-}" ]] || {
      echo "Fresh-zero lineage must not define RLT_WARM_START_ACTOR_CHECKPOINT." >&2
      exit 2
    }
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
require_config_value RLT_SHADOW_SERVICE "$V3_SHADOW_SERVICE"

POLICY_IMPORT_AUDIT="$(
  PYTHONPATH="$RUNTIME:$WORKSPACE/src:$WORKSPACE/packages/openpi-client/src:${PYTHONPATH:-}" \
    "$PROJECT_PYTHON" - \
      "$RUNTIME" \
      "$EXPECTED_RAW_SCHEMA" \
      "$EXPECTED_EXECUTION_SCHEMA" \
      "$EXPECTED_PROJECTION" \
      "$EXPECTED_ACTOR_PROFILE" \
      "$EXPECTED_GOVERNOR" \
      "$EXPECTED_GRIPPER_MODE" <<'PY'
from pathlib import Path
import sys

import piper_runtime.rlt_shadow_policy_service as service
from piper_runtime.rlt_actor_protocol import (
    RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT,
)
from piper_runtime.rlt_actor_protocol import (
    RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE,
)
from piper_runtime.rlt_online_session import SessionRuntimeConfig
from piper_runtime.rlt_takeover_rollout import TakeoverRuntimeConfig

runtime = Path(sys.argv[1]).resolve()
expected_schema = sys.argv[2]
expected_execution_schema = sys.argv[3]
expected_projection = sys.argv[4]
expected_actor_profile = sys.argv[5]
expected_governor = sys.argv[6]
expected_gripper_mode = sys.argv[7]
service_path = Path(service.__file__).resolve()
if runtime not in service_path.parents:
    raise SystemExit(
        f"shadow policy import escaped selected runtime: {service_path} not under {runtime}"
    )
if RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT != expected_schema:
    raise SystemExit(
        "selected runtime raw Actor schema is not gripper-close v5: "
        f"{RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT!r}"
    )
if RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE != expected_projection:
    raise SystemExit(
        "selected runtime projection profile is not gripper-close v3: "
        f"{RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE!r}"
    )
contract_kwargs = {
    "action_schema_fingerprint": expected_schema,
    "execution_action_schema_fingerprint": expected_execution_schema,
    "actor_projection_profile": expected_projection,
    "actor_execution_profile": expected_actor_profile,
    "actor_governor_fingerprint": expected_governor,
    "actor_gripper_residual_mode": expected_gripper_mode,
    "actor_live_max_boundary_jump_rad": 0.06,
}
SessionRuntimeConfig(**contract_kwargs).validate()
TakeoverRuntimeConfig(**contract_kwargs).validate()
print(f"runtime_module={service_path}")
print(f"raw_actor_schema={expected_schema}")
print(f"actor_projection_profile={expected_projection}")
print("session_and_rollout_contract=validated")
PY
)"
ROS_RUNTIME_IMPORT_AUDIT="$(
  PYTHONPATH="$RUNTIME:$WORKSPACE/src:$WORKSPACE/packages/openpi-client/src:$HOME/pika_ros/install/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages:/usr/lib/python3/dist-packages:${PYTHONPATH:-}" \
    "$PIKA_PYTHON" - \
      "$EXPECTED_RAW_SCHEMA" \
      "$EXPECTED_EXECUTION_SCHEMA" \
      "$EXPECTED_PROJECTION" \
      "$EXPECTED_ACTOR_PROFILE" \
      "$EXPECTED_GOVERNOR" \
      "$EXPECTED_GRIPPER_MODE" <<'PY'
import sys

import catkin_pkg
import cv2
import pyrealsense2
import rospkg
import rospy
import torch
from sensor_msgs.msg import JointState

from piper_runtime.rlt_online_session import SessionRuntimeConfig
from piper_runtime.rlt_takeover_rollout import TakeoverRuntimeConfig
from piper_runtime.ros_command_io import require_ros_modules

expected_schema = sys.argv[1]
expected_execution_schema = sys.argv[2]
expected_projection = sys.argv[3]
expected_actor_profile = sys.argv[4]
expected_governor = sys.argv[5]
expected_gripper_mode = sys.argv[6]
resolved_rospy, resolved_joint_state = require_ros_modules()
if resolved_rospy is not rospy or resolved_joint_state is not JointState:
    raise SystemExit("ROS module identity mismatch")
contract_kwargs = {
    "action_schema_fingerprint": expected_schema,
    "execution_action_schema_fingerprint": expected_execution_schema,
    "actor_projection_profile": expected_projection,
    "actor_execution_profile": expected_actor_profile,
    "actor_governor_fingerprint": expected_governor,
    "actor_gripper_residual_mode": expected_gripper_mode,
    "actor_live_max_boundary_jump_rad": 0.06,
}
SessionRuntimeConfig(**contract_kwargs).validate()
TakeoverRuntimeConfig(**contract_kwargs).validate()
print(f"ros_runtime_python={sys.executable}")
print(f"rospkg={rospkg.__file__}")
print(f"catkin_pkg={catkin_pkg.__file__}")
print(f"torch={torch.__version__}")
print(f"opencv={cv2.__version__}")
print(f"pyrealsense2={pyrealsense2.__file__}")
print("ros_session_and_rollout_import=validated")
PY
)"

echo "Gripper-close v3 admitted-human Q-filter contract:"
echo "  raw Actor schema      : $RLT_ACTOR_MODEL_ACTION_SCHEMA_FINGERPRINT"
echo "  execution schema      : $RLT_EXECUTION_ACTION_SCHEMA_FINGERPRINT"
echo "  governor              : $RLT_ACTOR_GOVERNOR_FINGERPRINT"
echo "  admitted-human grip BC: reward 1/0 labels preserved; beta=$RLT_BETA_HUMAN_GRIPPER_BC, scale=$RLT_HUMAN_GRIPPER_BC_SCALE_M m"
echo "  local advantage gate  : mode=$RLT_HUMAN_GRIPPER_Q_FILTER_MODE, margin=$RLT_HUMAN_GRIPPER_Q_FILTER_MARGIN"
echo "  close assist          : r<=${RLT_GRIPPER_RESIDUAL_MAX}m, d1<=${RLT_GRIPPER_RESIDUAL_D1_MAX_M}m, d2<=${RLT_GRIPPER_RESIDUAL_D2_MAX_M}m, boundary<=${RLT_GRIPPER_MAX_BOUNDARY_JUMP_M}m"
echo "  command/release guard : [${RLT_GRIPPER_COMMAND_MIN_M},${RLT_GRIPPER_COMMAND_MAX_M}]m; release ref=${RLT_GRIPPER_RELEASE_REFERENCE_M}m, delta=${RLT_GRIPPER_RELEASE_DELTA_M}m"
echo "  session script        : $ONLINE_SESSION_SCRIPT"
echo "  update hook           : $UPDATE_HOOK_SCRIPT"
echo "  policy import audit   :"
while IFS= read -r audit_line; do
  echo "    $audit_line"
done <<<"$POLICY_IMPORT_AUDIT"
echo "  ROS runtime audit     :"
while IFS= read -r audit_line; do
  echo "    $audit_line"
done <<<"$ROS_RUNTIME_IMPORT_AUDIT"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run passed; v3 validator and launch-contract checks passed."
  echo "No ROS service, selector, CAN, or robot command was changed."
  exit 0
fi

# The state-root flock below prevents duplicate launches of one lineage, but
# separate lineages have separate lock files. Refuse every live rollout or
# learner globally before any service/CAN/robot mutation so a clean branch can
# never overlap the source branch.
ACTIVE_RLT_JOBS="$(
  "$PROJECT_PYTHON" - <<'PY'
from pathlib import Path

exact_names = {
    "rlt_online_session.py",
    "rlt_takeover_rollout.py",
    "train_real_rlt_jax.py",
    "run_online_rlt_update.py",
    "run_rlt_online_update_hook_gripper_close_v3.sh",
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
        text_args = [value.decode("utf-8", "replace") for value in args if value]
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    matched = any(Path(value).name in exact_names for value in text_args)
    matched = matched or any(value in exact_modules for value in text_args)
    if matched:
        print(f"{entry.name} {' '.join(text_args)}")
PY
)"
if [[ -n "$ACTIVE_RLT_JOBS" ]]; then
  echo "Another Piper RLT rollout/learner is still active; refusing a second control chain:" >&2
  printf '%s\n' "$ACTIVE_RLT_JOBS" >&2
  echo "Exit the existing session cleanly, verify pgrep is empty, then retry." >&2
  exit 2
fi

[[ -f "$RUNTIME/start_rlt_mode.sh" \
   && -f "$RUNTIME/enable_piper_arm.sh" \
   && -d "$RUNTIME/piper_runtime" ]] || {
  echo "Selected runtime is missing required RLT hardware/service assets: $RUNTIME" >&2
  exit 2
}
CURRENT_ROOT="$(dirname "$RLT_SELECTED_ACTOR_FILE")"
mkdir -p "$CURRENT_ROOT" "$RUNTIME/logs" "$HOME/.config/systemd/user"
exec 9>"$CURRENT_ROOT/session.lock"
if ! flock -n 9; then
  echo "Another Piper RLT online session owns $CURRENT_ROOT/session.lock." >&2
  exit 2
fi

for legacy_unit in \
  rlt-native-sdk-command.service \
  openpi-rlt-shadow-policy.service \
  openpi-piper-policy.service; do
  if systemctl --user is-active --quiet "$legacy_unit"; then
    echo "Legacy service $legacy_unit is active; refusing to stop or reuse it." >&2
    echo "Stop it deliberately, then rerun the isolated gripper-v3 launcher." >&2
    exit 2
  fi
done

for service_path in "$WORKSPACE" "$RUNTIME" "$PROJECT_PYTHON" "$PIKA_PYTHON" "$RLT_SELECTED_ACTOR_FILE"; do
  if [[ "$service_path" =~ [[:space:]] ]]; then
    echo "v3 systemd service paths may not contain whitespace: $service_path" >&2
    exit 2
  fi
done

SERVICE_HELPER_DIR="$RUNTIME/.gripper_close_v3_service"
POLICY_HELPER="$SERVICE_HELPER_DIR/run_selected_policy_server.sh"
NATIVE_UNIT="$HOME/.config/systemd/user/$V3_NATIVE_SERVICE"
SHADOW_UNIT="$HOME/.config/systemd/user/$V3_SHADOW_SERVICE"
POLICY_CONFIG_VALUE="${PIPER_POLICY_CONFIG:-pi05_piper_greenblock_5090_jax_delta_v1}"
POLICY_CHECKPOINT_VALUE="${PIPER_POLICY_CHECKPOINT:-$HOME/openpi_checkpoints/pi05_piper_greenblock_5090_jax_delta_v1/piper_greenblock_5090_delta_sft_30k_20260707/20000}"
TOKEN_CHECKPOINT_VALUE="${PIPER_RL_TOKEN_CHECKPOINT:-$HOME/openpi_rlt/rl_tokens/pi05_piper_greenblock_5090_full20k_rlt_token_v1_20260709/step_010000_encoder_only}"
mkdir -p "$SERVICE_HELPER_DIR" "$RUNTIME/jax_cache_gripper_v3"

install -m 0755 /dev/stdin "$POLICY_HELPER" <<'POLICY_HELPER_EOF'
#!/usr/bin/env bash
set -euo pipefail

SELECTED_FILE="${PIPER_RLT_SELECTED_ACTOR_FILE:?}"
PROJECT_PYTHON="${PIPER_RLT_PROJECT_PYTHON:?}"
EXPECTED_RUNTIME="${PIPER_RLT_EXPECTED_RUNTIME:?}"
EXPECTED_RAW_SCHEMA="${PIPER_RLT_EXPECTED_RAW_SCHEMA:?}"
EXPECTED_PROJECTION="${PIPER_RLT_EXPECTED_PROJECTION_PROFILE:?}"
IMPORT_AUDIT="$("$PROJECT_PYTHON" - "$EXPECTED_RUNTIME" "$EXPECTED_RAW_SCHEMA" "$EXPECTED_PROJECTION" <<'PY'
from pathlib import Path
import sys

import piper_runtime.rlt_shadow_policy_service as service
from piper_runtime.rlt_actor_protocol import (
    RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT,
)
from piper_runtime.rlt_actor_protocol import (
    RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE,
)

runtime = Path(sys.argv[1]).resolve()
expected_schema = sys.argv[2]
expected_projection = sys.argv[3]
service_path = Path(service.__file__).resolve()
if runtime not in service_path.parents:
    raise SystemExit(
        f"shadow policy import escaped selected runtime: {service_path} not under {runtime}"
    )
if RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT != expected_schema:
    raise SystemExit(
        "selected runtime raw Actor schema is not gripper-close v5: "
        f"{RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT!r}"
    )
if RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE != expected_projection:
    raise SystemExit(
        "selected runtime projection profile is not gripper-close v3: "
        f"{RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE!r}"
    )
print(f"runtime_module={service_path}")
print(f"raw_actor_schema={expected_schema}")
print(f"actor_projection_profile={expected_projection}")
PY
)"
printf 'Gripper-v3 policy import audit: %s\n' "$IMPORT_AUDIT"
SELECTED="$(grep -v '^[[:space:]]*$' "$SELECTED_FILE" | head -n 1 | tr -d '\r' || true)"
if [[ -z "$SELECTED" || "$SELECTED" == "NONE" || "$SELECTED" == "none" ]]; then
  export PIPER_RLT_ACTOR_MODE=none
  unset PIPER_RLT_ACTOR_CHECKPOINT || true
else
  if [[ ! -f "$SELECTED/learner.msgpack" || ! -f "$SELECTED/metadata.json" ]]; then
    echo "Selected gripper-v3 Actor checkpoint is incomplete: $SELECTED" >&2
    exit 2
  fi
  export PIPER_RLT_ACTOR_MODE=checkpoint
  export PIPER_RLT_ACTOR_CHECKPOINT="$SELECTED"
fi
exec "$PROJECT_PYTHON" -u -m piper_runtime.rlt_shadow_policy_service
POLICY_HELPER_EOF

install -m 0644 /dev/stdin "$NATIVE_UNIT" <<NATIVE_UNIT_EOF
[Unit]
Description=Persistent Piper native SDK command bridge for gripper-close v3 RLT
After=network.target

[Service]
Type=simple
WorkingDirectory=$RUNTIME
Environment="PYTHONPATH=$RUNTIME:$WORKSPACE/src:$WORKSPACE/packages/openpi-client/src:$HOME/pika_ros/install/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages"
ExecStart=$PIKA_PYTHON -u -m piper_runtime.native_sdk_command_bridge
Restart=on-failure
RestartSec=2
TimeoutStopSec=10
StandardOutput=append:$RUNTIME/logs/gripper_v3_native_sdk_command_bridge.log
StandardError=append:$RUNTIME/logs/gripper_v3_native_sdk_command_bridge.log

[Install]
WantedBy=default.target
NATIVE_UNIT_EOF

install -m 0644 /dev/stdin "$SHADOW_UNIT" <<SHADOW_UNIT_EOF
[Unit]
Description=OpenPI Piper policy and selected gripper-close v3 RLT Actor
After=network.target

[Service]
Type=simple
WorkingDirectory=$WORKSPACE
Environment="PYTHONPATH=$RUNTIME:$WORKSPACE/src:$WORKSPACE/packages/openpi-client/src"
Environment="XLA_PYTHON_CLIENT_PREALLOCATE=false"
Environment="JAX_COMPILATION_CACHE_DIR=$RUNTIME/jax_cache_gripper_v3"
Environment="PIPER_RLT_SELECTED_ACTOR_FILE=$RLT_SELECTED_ACTOR_FILE"
Environment="PIPER_RLT_PROJECT_PYTHON=$PROJECT_PYTHON"
Environment="PIPER_RLT_EXPECTED_RUNTIME=$RUNTIME"
Environment="PIPER_RLT_EXPECTED_RAW_SCHEMA=$EXPECTED_RAW_SCHEMA"
Environment="PIPER_RLT_EXPECTED_PROJECTION_PROFILE=$EXPECTED_PROJECTION"
Environment="PIPER_POLICY_CONFIG=$POLICY_CONFIG_VALUE"
Environment="PIPER_POLICY_CHECKPOINT=$POLICY_CHECKPOINT_VALUE"
Environment="PIPER_RL_TOKEN_CHECKPOINT=$TOKEN_CHECKPOINT_VALUE"
Environment="PIPER_RLT_SHADOW_PORT=$RLT_POLICY_PORT"
ExecStart=/usr/bin/bash $POLICY_HELPER
Restart=on-failure
RestartSec=5
TimeoutStopSec=30
StandardOutput=append:$RUNTIME/logs/gripper_v3_shadow_policy_server.log
StandardError=append:$RUNTIME/logs/gripper_v3_shadow_policy_server.log

[Install]
WantedBy=default.target
SHADOW_UNIT_EOF

export RLT_V3_WORKSPACE_OVERRIDE="$WORKSPACE"
export RLT_V3_RUNTIME_OVERRIDE="$RUNTIME"
export PIPER_RLT_ROS_PYTHON="$PIKA_PYTHON"
export RLT_ONLINE_CONFIG="$CONFIG_FILE"
export RLT_SESSION_ROOT="$SESSION_ROOT"
export RLT_STATE_ROOT="$STATE_ROOT"
export SESSION_ID="$LINEAGE"
export OUTPUT_DIR="$SESSION_ROOT"
export MAX_EPISODES="$MAX_EPISODES_VALUE"
export DURATION="${DURATION:-120}"
export PUBLISH="${PUBLISH:-1}"
export RESET_HOME="${RESET_HOME:-1}"
export RESET_BEFORE_FIRST="${RESET_BEFORE_FIRST:-0}"
export RESET_HOME_TARGET="${RESET_HOME_TARGET:-0.020019,0.152472,-0.228603,-0.020857,0.632856,0,0.06517}"
export RESET_HOME_HOLD="${RESET_HOME_HOLD:-4.0}"
export HARDWARE_IO=ros_bridge
export MODEL_EXECUTE_STEPS="$RLT_MODEL_EXECUTE_STEPS"
export MODEL_SMOOTHING_TAU="$RLT_EXECUTION_FILTER_TAU_S"
export MODEL_MAX_JOINT_STEP_DEG="${MODEL_MAX_JOINT_STEP_DEG:-3.0}"
export MODEL_MAX_GRIPPER_STEP="${MODEL_MAX_GRIPPER_STEP:-0.02}"
export ACTOR_SHADOW=1
export ACTOR_LIVE=1
export ACTOR_LIVE_MAX_CHUNKS="$ACTOR_LIVE_MAX_CHUNKS_VALUE"
export MANUAL_TRAINING_ADMISSION=1
export ACTION_SCHEMA_FINGERPRINT="$RLT_ACTOR_MODEL_ACTION_SCHEMA_FINGERPRINT"
export EXECUTION_ACTION_SCHEMA_FINGERPRINT="$RLT_EXECUTION_ACTION_SCHEMA_FINGERPRINT"
export ACTOR_PROJECTION_PROFILE="$RLT_ACTOR_PROJECTION_PROFILE"
export ACTOR_EXECUTION_PROFILE="$RLT_ACTOR_EXECUTION_PROFILE"
export ACTOR_GOVERNOR_FINGERPRINT="$RLT_ACTOR_GOVERNOR_FINGERPRINT"
export ACTOR_RESIDUAL_MAX_RAD="$RLT_RESIDUAL_MAX"
export ACTOR_RESIDUAL_D1_MAX_RAD="$RLT_RESIDUAL_D1_MAX_RAD"
export ACTOR_RESIDUAL_D2_MAX_RAD="$RLT_RESIDUAL_D2_MAX_RAD"
export ACTOR_DIRECTION_CONE_DEG="$RLT_DIRECTION_CONE_DEG"
export ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD="$RLT_ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD"
export ACTOR_GRIPPER_RESIDUAL_MODE="$RLT_GRIPPER_RESIDUAL_MODE"
export ACTOR_GRIPPER_RESIDUAL_MAX_CLOSE_M="$RLT_GRIPPER_RESIDUAL_MAX"
export ACTOR_GRIPPER_RESIDUAL_D1_MAX_M="$RLT_GRIPPER_RESIDUAL_D1_MAX_M"
export ACTOR_GRIPPER_RESIDUAL_D2_MAX_M="$RLT_GRIPPER_RESIDUAL_D2_MAX_M"
export ACTOR_GRIPPER_MAX_BOUNDARY_JUMP_M="$RLT_GRIPPER_MAX_BOUNDARY_JUMP_M"
export ACTOR_GRIPPER_COMMAND_MIN_M="$RLT_GRIPPER_COMMAND_MIN_M"
export ACTOR_GRIPPER_COMMAND_MAX_M="$RLT_GRIPPER_COMMAND_MAX_M"
export ACTOR_GRIPPER_RELEASE_REFERENCE_M="$RLT_GRIPPER_RELEASE_REFERENCE_M"
export ACTOR_GRIPPER_RELEASE_DELTA_M="$RLT_GRIPPER_RELEASE_DELTA_M"
export RLT_BETA_HUMAN_GRIPPER_BC
export RLT_HUMAN_GRIPPER_Q_FILTER_MODE
export RLT_HUMAN_GRIPPER_Q_FILTER_MARGIN
export PHASE_CLASSIFIER_CHECKPOINT="$RLT_PHASE_CHECKPOINT"

EPISODE_START_INDEX="$((10#${NEXT_EPISODE#episode_}))"

echo "Starting hardware preflight; rollout begins at $NEXT_EPISODE only after every check passes."
echo "  Actor live chunks: $ACTOR_LIVE_MAX_CHUNKS_VALUE (0 means unlimited; v3 runtime guards remain active)"
echo "  online update    : only after operator admission, via the v3 hook above"
echo "  training admit   : explicit operator t/r/e decision after every reward"

V3_NATIVE_WAS_ACTIVE=0
V3_SHADOW_WAS_ACTIVE=0
systemctl --user is-active --quiet "$V3_NATIVE_SERVICE" \
  && V3_NATIVE_WAS_ACTIVE=1
systemctl --user is-active --quiet "$V3_SHADOW_SERVICE" \
  && V3_SHADOW_WAS_ACTIVE=1

cleanup_v3_services_on_failure() {
  local status=$?
  if ((status != 0)); then
    echo "Gripper-v3 launch/session failed (status=$status); cleaning services started by this invocation." >&2
    if ((V3_SHADOW_WAS_ACTIVE == 0)); then
      systemctl --user stop "$V3_SHADOW_SERVICE" >/dev/null 2>&1 || true
    fi
    if ((V3_NATIVE_WAS_ACTIVE == 0)); then
      systemctl --user stop "$V3_NATIVE_SERVICE" >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup_v3_services_on_failure EXIT

bash "$RUNTIME/start_rlt_mode.sh"

systemctl --user daemon-reload
if systemctl --user is-active --quiet "$V3_NATIVE_SERVICE"; then
  if bash -lc 'source "$HOME/pika_ros/install/setup.bash"; rosnode list 2>/dev/null | grep -qx /rlt_native_sdk_command_bridge'; then
    echo "Reusing $V3_NATIVE_SERVICE."
  else
    systemctl --user restart "$V3_NATIVE_SERVICE"
  fi
else
  systemctl --user start "$V3_NATIVE_SERVICE"
fi
if ! timeout 15 bash -lc '
  source "$HOME/pika_ros/install/setup.bash"
  until rosnode list 2>/dev/null | grep -qx /rlt_native_sdk_command_bridge; do sleep 0.2; done
'; then
  echo "Persistent native SDK bridge did not register with ROS." >&2
  systemctl --user status "$V3_NATIVE_SERVICE" --no-pager -l >&2 || true
  exit 3
fi

if ! bash -lc 'source "$HOME/pika_ros/install/setup.bash"; timeout 5 rostopic echo -n 1 /pika_pose >/dev/null 2>&1'; then
  echo "Pika takeover pose is unavailable; refusing an online session." >&2
  exit 3
fi
if ! bash -lc 'source "$HOME/pika_ros/install/setup.bash"; timeout 5 rostopic echo -n 1 /sensor/gripper/joint_state >/dev/null 2>&1'; then
  echo "Sense gripper feedback is unavailable; human gripper imitation cannot be recorded safely." >&2
  exit 3
fi

bash "$RUNTIME/enable_piper_arm.sh"
printf '%s\n' "$LATEST_CHECKPOINT" >"$RLT_SELECTED_ACTOR_FILE"

systemctl --user restart "$V3_SHADOW_SERVICE"
if ! timeout 180 bash -c \
  "until ss -ltn | grep -q '${RLT_POLICY_HOST}:${RLT_POLICY_PORT}'; do sleep 2; done"; then
  echo "$V3_SHADOW_SERVICE did not open ${RLT_POLICY_HOST}:${RLT_POLICY_PORT}." >&2
  systemctl --user status "$V3_SHADOW_SERVICE" --no-pager -l >&2 || true
  tail -n 60 "$RUNTIME/logs/gripper_v3_shadow_policy_server.log" >&2 || true
  exit 3
fi

bash "$ONLINE_SESSION_SCRIPT" \
  --episode-start-index "$EPISODE_START_INDEX" \
  --policy-host "$RLT_POLICY_HOST" \
  --policy-port "$RLT_POLICY_PORT" \
  --after-episode-command "bash $UPDATE_HOOK_SCRIPT"
