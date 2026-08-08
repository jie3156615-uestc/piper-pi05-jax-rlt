#!/usr/bin/env bash
set -euo pipefail

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
LAUNCHER="$SCRIPT_DIR/run_rlt_lineage_gripper_close_v3_online.sh"
SESSIONS_ROOT="${RLT_SESSIONS_ROOT:-$HOME/rlt_online_sessions}"

LINEAGE="${RLT_GRIPPER_V3_LINEAGE:-greenblock_rlt_gripper_close_v3_restart_from_bootstrap_20260728}"
STATE_DIR="${RLT_GRIPPER_V3_STATE_DIR:-.online_rlt_persistent_gripper_v3}"
LATEST_EPISODE="AUTO"
ACTOR_LIVE_MAX_CHUNKS_VALUE="0"
MAX_EPISODES_VALUE="1000000"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  run_greenblock_gripper_close_v3_current.sh \
    [--lineage NAME] \
    [--state-dir .online_rlt_NAME] \
    [--latest-episode AUTO|NONE|episode_XXXXXX] \
    [--actor-live-max-chunks N] \
    [--max-episodes N] \
    [--dry-run]

AUTO reads the newest episode directory, then the guarded v3 launcher repeats
the check before touching ROS/CAN. Explicit values remain optimistic-concurrency
guards and never select a historical checkpoint. The default lineage is the
clean 2026-07-28 bootstrap restart; select the 2026-07-27 lineage explicitly
only when intentionally continuing that older experiment.
EOF
}

while (($#)); do
  case "$1" in
    --lineage)
      [[ $# -ge 2 ]] || { echo "--lineage requires a value" >&2; exit 2; }
      LINEAGE="$2"
      shift 2
      ;;
    --state-dir)
      [[ $# -ge 2 ]] || { echo "--state-dir requires a value" >&2; exit 2; }
      STATE_DIR="$2"
      shift 2
      ;;
    --latest-episode)
      [[ $# -ge 2 ]] || { echo "--latest-episode requires a value" >&2; exit 2; }
      LATEST_EPISODE="$2"
      shift 2
      ;;
    --actor-live-max-chunks)
      [[ $# -ge 2 ]] || { echo "--actor-live-max-chunks requires a value" >&2; exit 2; }
      ACTOR_LIVE_MAX_CHUNKS_VALUE="$2"
      shift 2
      ;;
    --max-episodes)
      [[ $# -ge 2 ]] || { echo "--max-episodes requires a value" >&2; exit 2; }
      MAX_EPISODES_VALUE="$2"
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
  echo "Invalid --lineage/RLT_GRIPPER_V3_LINEAGE: $LINEAGE" >&2
  exit 2
}
[[ "$STATE_DIR" =~ ^[.][A-Za-z0-9._-]+$ ]] || {
  echo "--state-dir must be one direct hidden directory name: $STATE_DIR" >&2
  exit 2
}
[[ "$LATEST_EPISODE" == "AUTO" \
   || "$LATEST_EPISODE" == "NONE" \
   || "$LATEST_EPISODE" =~ ^episode_[0-9]{6}$ ]] || {
  echo "--latest-episode must be AUTO, NONE, or episode_XXXXXX: $LATEST_EPISODE" >&2
  exit 2
}
[[ "$ACTOR_LIVE_MAX_CHUNKS_VALUE" =~ ^[0-9]+$ ]] || {
  echo "--actor-live-max-chunks must be a non-negative integer." >&2
  exit 2
}
[[ "$MAX_EPISODES_VALUE" =~ ^[1-9][0-9]*$ ]] || {
  echo "--max-episodes must be a positive integer." >&2
  exit 2
}
[[ -x "$PROJECT_PYTHON" ]] || {
  echo "Gripper-close v3 requires the selected workspace .venv Python: $PROJECT_PYTHON" >&2
  exit 2
}
[[ -f "$LAUNCHER" ]] || {
  echo "Gripper-close v3 guarded launcher is missing: $LAUNCHER" >&2
  exit 2
}

SESSION_ROOT="$SESSIONS_ROOT/$LINEAGE"
if [[ "$LATEST_EPISODE" == "AUTO" ]]; then
  LATEST_EPISODE="$("$PROJECT_PYTHON" - "$SESSION_ROOT" <<'PY'
from pathlib import Path
import re
import sys

session_root = Path(sys.argv[1])
episodes = []
for path in session_root.glob("episode_[0-9]*"):
    match = re.fullmatch(r"episode_([0-9]+)", path.name)
    if path.is_dir() and match:
        episodes.append((int(match.group(1)), path.name))
print(max(episodes)[1] if episodes else "NONE")
PY
)"
fi

ARGS=(
  --lineage "$LINEAGE"
  --state-dir "$STATE_DIR"
  --latest-episode "$LATEST_EPISODE"
  --actor-live-max-chunks "$ACTOR_LIVE_MAX_CHUNKS_VALUE"
  --max-episodes "$MAX_EPISODES_VALUE"
)
if [[ "$DRY_RUN" == "1" ]]; then
  ARGS+=(--dry-run)
fi

echo "Launching guarded gripper-close v3 lineage:"
echo "  lineage         : $LINEAGE"
echo "  state dir       : $STATE_DIR"
echo "  latest guard    : $LATEST_EPISODE"
echo "  live max chunks : $ACTOR_LIVE_MAX_CHUNKS_VALUE"
echo "  workspace       : $WORKSPACE"
echo "  runtime         : $RUNTIME"

export RLT_V3_WORKSPACE_OVERRIDE="$WORKSPACE"
export RLT_V3_RUNTIME_OVERRIDE="$RUNTIME"
exec bash "$LAUNCHER" "${ARGS[@]}"
