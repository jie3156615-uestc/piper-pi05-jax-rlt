#!/usr/bin/env bash
set -euo pipefail

STAGING="${RLT_V3_STAGING:-$HOME/gripper_close_v3_staging_20260727}"
LAUNCHER="$STAGING/run_greenblock_gripper_close_v3_current.sh"

LINEAGE="${RLT_LINEAGE:-greenblock_rlt_fresh_v3_20260729}"
STATE_DIR="${RLT_STATE_DIR:-.online_rlt_persistent_gripper_v3_fresh}"
MAX_EPISODES="${RLT_MAX_EPISODES:-1000000}"
ACTOR_LIVE_MAX_CHUNKS="${RLT_ACTOR_LIVE_MAX_CHUNKS:-0}"

[[ -f "$LAUNCHER" ]] || {
  echo "RLT launcher is missing: $LAUNCHER" >&2
  exit 2
}
[[ "$LINEAGE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "Invalid RLT_LINEAGE: $LINEAGE" >&2
  exit 2
}
[[ "$STATE_DIR" =~ ^[.][A-Za-z0-9._-]+$ ]] || {
  echo "Invalid RLT_STATE_DIR: $STATE_DIR" >&2
  exit 2
}
[[ "$MAX_EPISODES" =~ ^[1-9][0-9]*$ ]] || {
  echo "RLT_MAX_EPISODES must be a positive integer." >&2
  exit 2
}
[[ "$ACTOR_LIVE_MAX_CHUNKS" =~ ^[0-9]+$ ]] || {
  echo "RLT_ACTOR_LIVE_MAX_CHUNKS must be a non-negative integer." >&2
  exit 2
}

echo "Starting current Piper RLT lineage:"
echo "  lineage         : $LINEAGE"
echo "  state dir       : $STATE_DIR"
echo "  latest episode  : AUTO"
echo "  max episodes    : $MAX_EPISODES"
echo "  Actor C10 limit : $ACTOR_LIVE_MAX_CHUNKS (0=unlimited)"

exec bash "$LAUNCHER" \
  --lineage "$LINEAGE" \
  --state-dir "$STATE_DIR" \
  --latest-episode AUTO \
  --max-episodes "$MAX_EPISODES" \
  --actor-live-max-chunks "$ACTOR_LIVE_MAX_CHUNKS"
