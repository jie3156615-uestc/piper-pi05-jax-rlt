#!/usr/bin/env bash
set -euo pipefail

# Offline-only helper for creating another clean gripper-close v3 branch from
# the immutable 30-episode bootstrap and accepted step_00000288 checkpoint.
# It never starts ROS, CAN, a policy service, or a robot command publisher.

readonly STAGING_ROOT="/home/cwzk/gripper_close_v3_staging_20260727"
readonly SOURCE_SESSION_ROOT="/home/cwzk/rlt_online_sessions/greenblock_rlt_persistent_v2_from_v8_ep369_20260724"
readonly SOURCE_STATE_ROOT="$SOURCE_SESSION_ROOT/.online_rlt_persistent_v2"
readonly SOURCE_ACTOR_CHECKPOINT="$SOURCE_STATE_ROOT/provenance/source_actor_checkpoint_step_00012627"
readonly BOOTSTRAP_REPLAY="$STAGING_ROOT/replay_gripper_close_v3.npz"
readonly BOOTSTRAP_MIGRATION_REPORT="$STAGING_ROOT/replay_gripper_close_v3_migration.json"
readonly INITIAL_V3_ROOT="$STAGING_ROOT/artifacts/initial_gripper_close_v3_from_v2_ep407_20260727"
readonly INITIAL_V3_CHECKPOINT="$INITIAL_V3_ROOT/step_00000288"
readonly INITIAL_V3_VALIDATION="$INITIAL_V3_ROOT/initial_v3_acceptance.json"
readonly FORK_TOOL="$STAGING_ROOT/scripts/piper_rlt/tools/fork_close_assist_v3_lineage.py"
readonly PROJECT_PYTHON="$STAGING_ROOT/.venv/bin/python"

TARGET_LINEAGE=""
TARGET_STATE_DIR=".online_rlt_persistent_gripper_v3"
CREATE=0

usage() {
  cat <<'EOF'
Usage:
  create_greenblock_gripper_close_v3_restart_branch.sh \
    --target-lineage NAME [--target-state-dir .NAME] [--create]

Without --create this performs a read-only preflight. With --create it creates
one isolated lineage whose first rollout is episode_000408. The branch reuses
the audited 30-episode warmup and accepted step_00000288 Actor/Critic state;
it does not reuse rejected online candidates and does not recollect warmup.

This helper is intentionally not a historical-episode rewind tool. A rollout
directory is not a checkpoint. Continue an existing lineage with the launcher
--latest-episode guard; fork a historical model only at an audited promotion
boundary with a dedicated v3-to-v3 migration.
EOF
}

need_value() {
  [[ $# -ge 2 ]] || {
    echo "$1 requires a value" >&2
    exit 2
  }
}

while (($#)); do
  case "$1" in
    --target-lineage)
      need_value "$1" "$#"
      TARGET_LINEAGE="$2"
      shift 2
      ;;
    --target-state-dir)
      need_value "$1" "$#"
      TARGET_STATE_DIR="$2"
      shift 2
      ;;
    --create)
      CREATE=1
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

[[ "$TARGET_LINEAGE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "--target-lineage must be a non-empty safe lineage name" >&2
  exit 2
}
[[ "$TARGET_STATE_DIR" =~ ^[.][A-Za-z0-9._-]+$ ]] || {
  echo "--target-state-dir must be one direct hidden directory name" >&2
  exit 2
}
[[ -x "$PROJECT_PYTHON" && -f "$FORK_TOOL" ]] || {
  echo "The self-contained gripper-v3 fork environment is incomplete" >&2
  exit 2
}

TARGET_SESSION_ROOT="$HOME/rlt_online_sessions/$TARGET_LINEAGE"
ARGS=(
  "$PROJECT_PYTHON"
  "$FORK_TOOL"
  --source-session-root "$SOURCE_SESSION_ROOT"
  --source-state-root "$SOURCE_STATE_ROOT"
  --source-actor-checkpoint "$SOURCE_ACTOR_CHECKPOINT"
  --bootstrap-replay "$BOOTSTRAP_REPLAY"
  --bootstrap-migration-report "$BOOTSTRAP_MIGRATION_REPORT"
  --initial-v3-checkpoint "$INITIAL_V3_CHECKPOINT"
  --initial-v3-validation-report "$INITIAL_V3_VALIDATION"
  --target-session-root "$TARGET_SESSION_ROOT"
  --workspace "$STAGING_ROOT"
  --runtime "$STAGING_ROOT"
  --target-state-dir "$TARGET_STATE_DIR"
  --expected-source-latest-episode episode_000406
  --expected-target-episode-floor 408
  --expected-initial-v3-checkpoint-step 288
)
if [[ "$CREATE" == "1" ]]; then
  ARGS+=(--create)
fi

echo "Gripper-close v3 clean-branch operation:"
echo "  mode           : $([[ "$CREATE" == "1" ]] && echo create || echo read-only preflight)"
echo "  target lineage : $TARGET_LINEAGE"
echo "  target state   : $TARGET_STATE_DIR"
echo "  first rollout  : episode_000408"
echo "  initial Actor  : step_00000288"
echo "  hardware       : untouched"

exec "${ARGS[@]}"
