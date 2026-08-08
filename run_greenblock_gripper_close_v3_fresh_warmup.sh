#!/usr/bin/env bash
set -euo pipefail

STAGING="${RLT_V3_WORKSPACE_OVERRIDE:-$HOME/gripper_close_v3_staging_20260727}"
INITIALIZER="$STAGING/scripts/piper_rlt/tools/init_fresh_close_assist_v3_lineage.py"
LAUNCHER="$STAGING/run_rlt_lineage_gripper_close_v3_online.sh"
PYTHON="$STAGING/.venv/bin/python"
SESSIONS_ROOT="${RLT_SESSIONS_ROOT:-$HOME/rlt_online_sessions}"
STATE_DIR=".online_rlt_persistent_gripper_v3_fresh"
LINEAGE=""
WARMUP=""
MAX_EPISODES=""
ACTOR_LIVE_MAX_CHUNKS="0"
CREATE_ONLY=0

usage() {
  cat <<'EOF'
Usage:
  run_greenblock_gripper_close_v3_fresh_warmup.sh \
    --name NAME \
    --warmup-episodes N \
    [--max-episodes N] \
    [--actor-live-max-chunks N] \
    [--state-dir .online_rlt_NAME] \
    [--create-only]

Creates a completely independent current gripper-close v3 lineage:
episode_000000, empty replay, no inherited Actor/Critic/optimizer/checkpoint.
If --max-episodes is omitted it equals --warmup-episodes.
EOF
}

while (($#)); do
  case "$1" in
    --name|--lineage) LINEAGE="${2:?missing value for $1}"; shift 2 ;;
    --warmup-episodes) WARMUP="${2:?missing value for $1}"; shift 2 ;;
    --max-episodes) MAX_EPISODES="${2:?missing value for $1}"; shift 2 ;;
    --actor-live-max-chunks) ACTOR_LIVE_MAX_CHUNKS="${2:?missing value for $1}"; shift 2 ;;
    --state-dir) STATE_DIR="${2:?missing value for $1}"; shift 2 ;;
    --create-only) CREATE_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$LINEAGE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "--name is required and must be filesystem-safe" >&2; exit 2; }
[[ "$WARMUP" =~ ^[1-9][0-9]*$ ]] || { echo "--warmup-episodes must be a positive integer" >&2; exit 2; }
MAX_EPISODES="${MAX_EPISODES:-$WARMUP}"
[[ "$MAX_EPISODES" =~ ^[1-9][0-9]*$ ]] || { echo "--max-episodes must be a positive integer" >&2; exit 2; }
[[ "$ACTOR_LIVE_MAX_CHUNKS" =~ ^[0-9]+$ ]] || { echo "--actor-live-max-chunks must be a non-negative integer" >&2; exit 2; }
[[ "$STATE_DIR" =~ ^[.][A-Za-z0-9._-]+$ ]] || { echo "--state-dir must be one hidden directory name" >&2; exit 2; }
[[ -x "$PYTHON" && -f "$INITIALIZER" && -f "$LAUNCHER" ]] || { echo "gripper-close v3 staging is incomplete: $STAGING" >&2; exit 2; }

SESSION_ROOT="$SESSIONS_ROOT/$LINEAGE"
STATE_ROOT="$SESSION_ROOT/$STATE_DIR"
"$PYTHON" "$INITIALIZER" \
  --session-root "$SESSION_ROOT" \
  --state-root "$STATE_ROOT" \
  --workspace "$STAGING" \
  --runtime "$STAGING" \
  --warmup-episodes "$WARMUP" \
  --create

echo "Fresh-zero lineage created: $LINEAGE"
echo "  first rollout     : episode_000000"
echo "  warmup threshold  : $WARMUP admitted trainable episodes"
echo "  inherited training: NONE"
if (( CREATE_ONLY )); then
  exit 0
fi

exec bash "$LAUNCHER" \
  --lineage "$LINEAGE" \
  --state-dir "$STATE_DIR" \
  --latest-episode NONE \
  --max-episodes "$MAX_EPISODES" \
  --actor-live-max-chunks "$ACTOR_LIVE_MAX_CHUNKS"
