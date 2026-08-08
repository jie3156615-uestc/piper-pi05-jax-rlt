#!/usr/bin/env bash
set -euo pipefail

# Prepare the first trained gripper-close v3 checkpoint and fork its immutable
# online lineage.  This entry point is intentionally independent of the live
# v2 workspace/runtime and refuses to coexist with rollout or learner jobs.

readonly DEFAULT_STAGING_ROOT="/home/cwzk/gripper_close_v3_staging_20260727"
readonly DEFAULT_SOURCE_LINEAGE="greenblock_rlt_persistent_v2_from_v8_ep369_20260724"
readonly DEFAULT_TARGET_LINEAGE="greenblock_rlt_gripper_close_v3_from_v2_ep407_20260727"
readonly DEFAULT_SOURCE_STATE_DIR=".online_rlt_persistent_v2"
readonly DEFAULT_TARGET_STATE_DIR=".online_rlt_persistent_gripper_v3"
readonly DEFAULT_SOURCE_LATEST_EPISODE="episode_000406"
readonly DEFAULT_EPISODE_FLOOR="408"

readonly EXPECTED_BOOTSTRAP_SHA256="658d981065227b48656e7b792fe7da2ce390f91c4584b22eb35bb54007cbbff5"
readonly EXPECTED_SOURCE_REPLAY_SHA256="1e8ba723d20619cfff5d15df3cba1c6936ab6e038cbc4c403bda3bf3a073be5c"
readonly EXPECTED_SOURCE_ACTOR_LEARNER_SHA256="11a182b44fbb840750650156da74a93dda0f0a84fc20d4f3192198168dcd88ef"
readonly EXPECTED_SOURCE_ACTOR_STEP="12627"
readonly REJECTED_SOURCE_STEP="144"
readonly EXPECTED_BOOTSTRAP_EPISODES="30"
readonly EXPECTED_BOOTSTRAP_REWARD_POSITIVE_EPISODES="24"
readonly EXPECTED_BOOTSTRAP_REWARD_NEGATIVE_EPISODES="6"
readonly EXPECTED_BOOTSTRAP_HUMAN_EPISODES="29"
readonly EXPECTED_BOOTSTRAP_TRAIN_TRANSITIONS="144"
readonly HUMAN_GRIPPER_Q_FILTER_MODE="critic_min_advantage_v1"
readonly HUMAN_GRIPPER_Q_FILTER_MARGIN="0.0"
readonly BOOTSTRAP_TEACHER_POLICY="all_admitted_human_dim6_clip_delta_to_[-0.005,0]_critic_min_advantage_q_filter_reward_independent"

readonly ACTION_SCHEMA_FINGERPRINT="piper_joint_delta_v5_c10_n10_stride10_behavior_ref50_persistent_filtered_actual_r005_d1_0015_d2_001_cone15_gripper_close_assist_r005_d1_0005_d2_0003_boundary0005"
readonly ACTOR_EXECUTION_PROFILE="persistent_c10_filtered_actual_v2"
readonly EXECUTION_FILTER_PROFILE="exp_one_minus_exp_neg_dt_over_tau_v1"
readonly ACTOR_GOVERNOR_FINGERPRINT="persistent_governor_v3_joint_r005_d1_0015_d2_001_cone15_boundary060_gripper_close_r005_d1_0005_d2_0003_boundary0005"
readonly GRIPPER_RESIDUAL_MODE="close_only_persistent_v1"

STAGING_ROOT="${GRIPPER_V3_STAGING_ROOT:-$DEFAULT_STAGING_ROOT}"
SOURCE_SESSION_ROOT=""
SOURCE_STATE_ROOT=""
SOURCE_ACTOR_CHECKPOINT=""
BOOTSTRAP_REPLAY=""
BOOTSTRAP_MIGRATION_REPORT=""
OUTPUT_DIR=""
ACCEPTANCE_REPORT=""
TARGET_SESSION_ROOT=""
TARGET_STATE_DIR="$DEFAULT_TARGET_STATE_DIR"
EXPECTED_SOURCE_LATEST_EPISODE="$DEFAULT_SOURCE_LATEST_EPISODE"
EXPECTED_EPISODE_FLOOR="$DEFAULT_EPISODE_FLOOR"
DRY_RUN=0
VERIFY_EXISTING=0

usage() {
  cat <<'EOF'
Usage:
  prepare_gripper_close_v3_initial.sh [options]

Creates the first fully trained gripper-close v3 checkpoint from the audited
30-episode migrated replay, validates it, then creates and read-only validates
the new online lineage.  It never controls hardware.

Modes:
  (default)            Refuse existing output/lineage and create once.
  --dry-run            Read-only preflight; print every command, write nothing.
  --verify-existing    Read-only verification of already-created artifacts.

Path guards/overrides:
  --staging-root PATH
  --source-session-root PATH
  --source-state-root PATH
  --source-actor-checkpoint PATH
  --bootstrap-replay PATH
  --bootstrap-migration-report PATH
  --output-dir PATH
  --acceptance-report PATH
  --target-session-root PATH
  --target-state-dir .NAME
  --expected-source-latest-episode episode_XXXXXX
  --expected-episode-floor N

Defaults:
  staging  /home/cwzk/gripper_close_v3_staging_20260727
  source   ~/rlt_online_sessions/greenblock_rlt_persistent_v2_from_v8_ep369_20260724
  target   ~/rlt_online_sessions/greenblock_rlt_gripper_close_v3_from_v2_ep407_20260727
  floor    408

The workspace and runtime are always the same self-contained --staging-root.
There are deliberately no separate workspace/runtime override flags.
EOF
}

need_value() {
  local option="$1"
  local count="$2"
  if ((count < 2)); then
    echo "$option requires a value" >&2
    exit 2
  fi
}

while (($#)); do
  case "$1" in
    --staging-root)
      need_value "$1" "$#"; STAGING_ROOT="$2"; shift 2 ;;
    --source-session-root)
      need_value "$1" "$#"; SOURCE_SESSION_ROOT="$2"; shift 2 ;;
    --source-state-root)
      need_value "$1" "$#"; SOURCE_STATE_ROOT="$2"; shift 2 ;;
    --source-actor-checkpoint)
      need_value "$1" "$#"; SOURCE_ACTOR_CHECKPOINT="$2"; shift 2 ;;
    --bootstrap-replay)
      need_value "$1" "$#"; BOOTSTRAP_REPLAY="$2"; shift 2 ;;
    --bootstrap-migration-report)
      need_value "$1" "$#"; BOOTSTRAP_MIGRATION_REPORT="$2"; shift 2 ;;
    --output-dir)
      need_value "$1" "$#"; OUTPUT_DIR="$2"; shift 2 ;;
    --acceptance-report)
      need_value "$1" "$#"; ACCEPTANCE_REPORT="$2"; shift 2 ;;
    --target-session-root)
      need_value "$1" "$#"; TARGET_SESSION_ROOT="$2"; shift 2 ;;
    --target-state-dir)
      need_value "$1" "$#"; TARGET_STATE_DIR="$2"; shift 2 ;;
    --expected-source-latest-episode)
      need_value "$1" "$#"; EXPECTED_SOURCE_LATEST_EPISODE="$2"; shift 2 ;;
    --expected-episode-floor)
      need_value "$1" "$#"; EXPECTED_EPISODE_FLOOR="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    --verify-existing)
      VERIFY_EXISTING=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if [[ "$DRY_RUN" == "1" && "$VERIFY_EXISTING" == "1" ]]; then
  echo "--dry-run and --verify-existing are mutually exclusive" >&2
  exit 2
fi
[[ "$TARGET_STATE_DIR" =~ ^[.][A-Za-z0-9._-]+$ ]] || {
  echo "--target-state-dir must be one direct hidden directory name" >&2
  exit 2
}
[[ "$EXPECTED_SOURCE_LATEST_EPISODE" =~ ^episode_[0-9]{6}$ ]] || {
  echo "--expected-source-latest-episode must be episode_XXXXXX" >&2
  exit 2
}
[[ "$EXPECTED_EPISODE_FLOOR" =~ ^[1-9][0-9]*$ ]] || {
  echo "--expected-episode-floor must be a positive integer" >&2
  exit 2
}

STAGING_ROOT="${STAGING_ROOT/#\~/$HOME}"
SOURCE_SESSION_ROOT="${SOURCE_SESSION_ROOT:-$HOME/rlt_online_sessions/$DEFAULT_SOURCE_LINEAGE}"
SOURCE_STATE_ROOT="${SOURCE_STATE_ROOT:-$SOURCE_SESSION_ROOT/$DEFAULT_SOURCE_STATE_DIR}"
SOURCE_ACTOR_CHECKPOINT="${SOURCE_ACTOR_CHECKPOINT:-$SOURCE_STATE_ROOT/provenance/source_actor_checkpoint_step_00012627}"
BOOTSTRAP_REPLAY="${BOOTSTRAP_REPLAY:-$STAGING_ROOT/replay_gripper_close_v3.npz}"
BOOTSTRAP_MIGRATION_REPORT="${BOOTSTRAP_MIGRATION_REPORT:-$STAGING_ROOT/replay_gripper_close_v3_migration.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$STAGING_ROOT/artifacts/initial_gripper_close_v3_from_v2_ep407_20260727}"
ACCEPTANCE_REPORT="${ACCEPTANCE_REPORT:-$OUTPUT_DIR/initial_v3_acceptance.json}"
TARGET_SESSION_ROOT="${TARGET_SESSION_ROOT:-$HOME/rlt_online_sessions/$DEFAULT_TARGET_LINEAGE}"

PROJECT_PYTHON="$STAGING_ROOT/.venv/bin/python"
TRAIN_SCRIPT="$STAGING_ROOT/scripts/train_real_rlt_jax.py"
VALIDATE_ACTOR_SCRIPT="$STAGING_ROOT/scripts/validate_real_rlt_actor_jax.py"
FORK_SCRIPT="$STAGING_ROOT/scripts/piper_rlt/tools/fork_close_assist_v3_lineage.py"
VALIDATE_LINEAGE_SCRIPT="$STAGING_ROOT/scripts/piper_rlt/tools/validate_close_assist_v3_lineage.py"
TARGET_STATE_ROOT="$TARGET_SESSION_ROOT/$TARGET_STATE_DIR"

die() {
  echo "GRIPPER_CLOSE_V3_INITIAL_FAIL: $*" >&2
  exit 1
}

# Keep --dry-run and read-only audits from creating import bytecode.
export PYTHONDONTWRITEBYTECODE=1

for path in \
  "$PROJECT_PYTHON" \
  "$TRAIN_SCRIPT" \
  "$VALIDATE_ACTOR_SCRIPT" \
  "$FORK_SCRIPT" \
  "$VALIDATE_LINEAGE_SCRIPT"; do
  [[ -e "$path" ]] || die "self-contained staging artifact is missing: $path"
done
[[ -x "$PROJECT_PYTHON" ]] || die "staging Python is not executable: $PROJECT_PYTHON"
[[ -d "$STAGING_ROOT/src/openpi" ]] || die "staging openpi package is missing"
[[ -d "$STAGING_ROOT/piper_runtime" ]] || die "staging runtime is missing"
[[ -f "$SOURCE_STATE_ROOT/online_state.json" ]] || die "source online_state.json is missing"
[[ -d "$SOURCE_ACTOR_CHECKPOINT" ]] || die "accepted source Actor checkpoint is missing"
[[ -f "$BOOTSTRAP_REPLAY" ]] || die "migrated v5 bootstrap replay is missing"
[[ -f "$BOOTSTRAP_MIGRATION_REPORT" ]] || die "bootstrap migration report is missing"

assert_no_active_rlt_jobs() {
  command -v pgrep >/dev/null 2>&1 || die "pgrep is required for the fail-closed process guard"
  local pattern
  local active
  pattern='[r]lt_online_session|[r]lt_takeover_rollout|[r]lt_shadow_policy_service|[t]rain_real_rlt_jax|[r]un_online_rlt_update|[r]un_rlt_online_update_hook'
  active="$(pgrep -af "$pattern" || true)"
  if [[ -n "$active" ]]; then
    printf '%s\n' "Refusing to prepare v3 while an RLT rollout/train/update job is active:" >&2
    printf '%s\n' "$active" >&2
    printf '%s\n' \
      "Stop the old online session and openpi-rlt-shadow-policy.service cleanly; this script will not stop either one for you." >&2
    die "stop the active job cleanly, then retry"
  fi
}

assert_no_active_rlt_jobs

# This audit is intentionally duplicated before the dedicated fork tool:
# training must never begin from a drifted replay, a rejected Actor, or an
# unaudited split count.
mapfile -t AUDIT_FIELDS < <(
  "$PROJECT_PYTHON" - \
    "$BOOTSTRAP_REPLAY" \
    "$BOOTSTRAP_MIGRATION_REPORT" \
    "$SOURCE_STATE_ROOT/online_state.json" \
    "$SOURCE_ACTOR_CHECKPOINT" \
    "$EXPECTED_BOOTSTRAP_SHA256" \
    "$EXPECTED_SOURCE_REPLAY_SHA256" \
    "$EXPECTED_SOURCE_ACTOR_LEARNER_SHA256" \
    "$ACTION_SCHEMA_FINGERPRINT" \
    "$ACTOR_GOVERNOR_FINGERPRINT" \
    "$GRIPPER_RESIDUAL_MODE" <<'PY'
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

(
    replay_path_raw,
    migration_path_raw,
    state_path_raw,
    actor_path_raw,
    expected_replay_sha,
    expected_source_replay_sha,
    expected_actor_learner_sha,
    expected_schema,
    expected_governor,
    expected_gripper_mode,
) = sys.argv[1:]

replay_path = Path(replay_path_raw).expanduser().resolve()
migration_path = Path(migration_path_raw).expanduser().resolve()
state_path = Path(state_path_raw).expanduser().resolve()
actor_path = Path(actor_path_raw).expanduser().resolve()


def fail(message: str) -> None:
    raise SystemExit(f"bootstrap/source audit failed: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


actual_replay_sha = sha256(replay_path)
if actual_replay_sha != expected_replay_sha:
    fail(f"bootstrap replay SHA {actual_replay_sha} != {expected_replay_sha}")

with np.load(replay_path, allow_pickle=False) as replay:
    required = {
        "episode_id",
        "episode_split",
        "success_mask",
        "human_mask",
        "reward",
        "action_schema_fingerprint",
        "actor_governor_fingerprint",
        "gripper_residual_mode",
    }
    missing = sorted(required.difference(replay.files))
    if missing:
        fail(f"bootstrap replay lacks arrays {missing}")
    episode_id = np.asarray(replay["episode_id"]).astype(str)
    episode_split = np.asarray(replay["episode_split"]).astype(str)
    success_mask = np.asarray(replay["success_mask"]).astype(bool)
    human_mask = np.asarray(replay["human_mask"]).astype(bool)
    reward = np.asarray(replay["reward"])
    schema = np.asarray(replay["action_schema_fingerprint"]).astype(str)
    governor = np.asarray(replay["actor_governor_fingerprint"]).astype(str)
    gripper_mode = np.asarray(replay["gripper_residual_mode"]).astype(str)

if episode_id.ndim != 1 or not len(episode_id):
    fail("bootstrap episode_id is empty or malformed")
n = len(episode_id)
if episode_split.shape != (n,) or success_mask.shape != (n,):
    fail("transition-aligned bootstrap fields have inconsistent shapes")
if human_mask.shape != (n, 10):
    fail(f"human_mask shape {human_mask.shape} != {(n, 10)}")
if reward.shape[0] != n or not np.all(np.isfinite(reward)):
    fail("bootstrap rewards are malformed or non-finite")
if set(schema.tolist()) != {expected_schema}:
    fail("bootstrap action schema is not exclusively gripper-close v5")
if set(governor.tolist()) != {expected_governor}:
    fail("bootstrap governor is not exclusively gripper-close v3")
if set(gripper_mode.tolist()) != {expected_gripper_mode}:
    fail("bootstrap gripper residual mode is not close-only")
if set(episode_split.tolist()).difference({"train", "validation"}):
    fail(
        "bootstrap contains unknown episode_split labels "
        f"{sorted(set(episode_split.tolist()))}"
    )

episode_ids = sorted(set(episode_id.tolist()))
reward_positive_ids = sorted(set(episode_id[success_mask].tolist()))
reward_negative_ids = sorted(
    set(episode_ids).difference(reward_positive_ids)
)
human_ids = sorted(set(episode_id[np.any(human_mask, axis=1)].tolist()))
reward_positive_human_steps = int(
    np.count_nonzero(human_mask & success_mask[:, None])
)
reward_negative_human_steps = int(
    np.count_nonzero(human_mask & ~success_mask[:, None])
)
quality = (
    len(episode_ids),
    len(reward_positive_ids),
    len(reward_negative_ids),
    len(human_ids),
)
if quality != (30, 24, 6, 29):
    fail(f"bootstrap admitted quality {quality} != (30, R+24, R-6, H29)")
if reward_positive_human_steps <= 0 or reward_negative_human_steps <= 0:
    fail(
        "bootstrap does not preserve admitted-human steps with both "
        "reward-positive and reward-negative labels"
    )
train_count = int(np.count_nonzero(episode_split == "train"))
validation_count = int(np.count_nonzero(episode_split == "validation"))
if train_count != 144:
    fail(f"derived train split count {train_count} != audited expectation 144")
if validation_count <= 0:
    fail("bootstrap has no held-out validation transitions")

migration = json.loads(migration_path.read_text(encoding="utf-8"))
if migration.get("output_sha256") != actual_replay_sha:
    fail("migration report output SHA does not bind the bootstrap replay")
if migration.get("source_sha256") != expected_source_replay_sha:
    fail("migration report source replay SHA is not the audited source")
if Path(str(migration.get("output_replay", ""))).expanduser().resolve() != replay_path:
    fail("migration report is bound to another output path")
if migration.get("target_schema") != expected_schema:
    fail("migration report target schema mismatch")
if migration.get("target_governor") != expected_governor:
    fail("migration report target governor mismatch")
if migration.get("gripper_residual_mode") != expected_gripper_mode:
    fail("migration report gripper mode mismatch")
if migration.get("source_mutated") is not False:
    fail("migration report does not prove source_mutated=false")
if migration.get("teacher_policy") != (
    "all_admitted_human_dim6_clip_delta_to_[-0.005,0]_"
    "critic_min_advantage_q_filter_reward_independent"
):
    fail("migration report is not the admitted-human Q-filter contract")
for key, expected in {
    "transitions": n,
    "episodes": 30,
    "successful_episodes": 24,
    "reward1_human_steps": reward_positive_human_steps,
    "reward0_human_steps": reward_negative_human_steps,
}.items():
    if int(migration.get(key, -1)) != expected:
        fail(f"migration report {key} mismatch")
contract = migration.get("contract")
if not isinstance(contract, dict):
    fail("migration report lacks gripper contract")
for key, expected in {
    "execution_gripper_residual_max_close_m": 0.005,
    "execution_gripper_d1_max_m": 0.0005,
    "execution_gripper_d2_max_m": 0.0003,
    "execution_gripper_boundary_limit_m": 0.0005,
    "execution_gripper_command_min_m": 0.0,
    "execution_gripper_command_max_m": 0.08,
    "execution_gripper_release_reference_m": 0.05,
    "execution_gripper_release_delta_m": 0.002,
}.items():
    try:
        actual = float(contract[key])
    except (KeyError, TypeError, ValueError):
        fail(f"migration report lacks numeric contract {key}")
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
        fail(f"migration contract {key}={actual} != {expected}")

state = json.loads(state_path.read_text(encoding="utf-8"))
state_actor_raw = state.get("initial_actor_warm_start_checkpoint")
if not state_actor_raw:
    fail("source state lacks initial_actor_warm_start_checkpoint")
state_actor = Path(str(state_actor_raw)).expanduser().resolve()
if state_actor != actor_path:
    fail(f"explicit source Actor {actor_path} != state-bound Actor {state_actor}")
state_attempt_ids = sorted(str(value) for value in state.get("last_attempt_episode_ids", []))
if state_attempt_ids != episode_ids:
    fail("bootstrap episode IDs differ from source last_attempt_episode_ids")
if int(state.get("last_attempt_episode_count", -1)) != 30:
    fail("source last_attempt_episode_count is not 30")
if int(state.get("last_attempt_train_transition_count", -1)) != train_count:
    fail("source last-attempt train count differs from derived bootstrap count")

actor_metadata_path = actor_path / "metadata.json"
actor_learner_path = actor_path / "learner.msgpack"
actor_metadata = json.loads(actor_metadata_path.read_text(encoding="utf-8"))
if int(actor_metadata.get("update_step", -1)) != 12627:
    fail("source Actor is not the accepted step12627")
actor_learner_sha = sha256(actor_learner_path)
if actor_learner_sha != expected_actor_learner_sha:
    fail("source Actor learner.msgpack SHA differs from accepted step12627")
source_config = actor_metadata.get("config", {})
for key, expected in {
    "beta_bc": 40.0,
    "beta_human_bc": 0.0,
    "beta_human_gripper_bc": 0.0,
}.items():
    actual = float(source_config.get(key, 0.0))
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
        fail(f"source Actor objective {key}={actual} != {expected}")

rejected_raw = state.get("latest_rejected_checkpoint")
if not rejected_raw:
    fail("source state lacks rejected-candidate provenance")
rejected = Path(str(rejected_raw)).expanduser().resolve()
if rejected == actor_path:
    fail("accepted source Actor aliases the rejected candidate")
rejected_metadata = json.loads(
    (rejected / "metadata.json").read_text(encoding="utf-8")
)
if int(rejected_metadata.get("update_step", -1)) != 144:
    fail("state-bound rejected candidate is not step144")
if sha256(rejected / "learner.msgpack") == actor_learner_sha:
    fail("rejected step144 aliases the accepted step12627 payload")
if Path(str(state.get("deployment_checkpoint", ""))).expanduser().resolve() != actor_path:
    fail("source deployment checkpoint is not the accepted step12627 Actor")

fingerprints = actor_metadata.get("fingerprints")
if not isinstance(fingerprints, dict):
    fail("accepted source Actor lacks fingerprints")
fingerprint_values = []
for key in ("base_checkpoint", "rl_token", "phase_classifier"):
    value = fingerprints.get(key)
    if not isinstance(value, str) or not value or "\n" in value or "\t" in value:
        fail(f"source Actor fingerprint {key} is missing or unsafe")
    fingerprint_values.append(value)

print(train_count)
print(validation_count)
print(actual_replay_sha)
print(actor_learner_sha)
print(*fingerprint_values, sep="\n")
PY
)

[[ "${#AUDIT_FIELDS[@]}" -eq 7 ]] || die "bootstrap/source audit returned an incomplete result"
TRAIN_TRANSITIONS="${AUDIT_FIELDS[0]}"
VALIDATION_TRANSITIONS="${AUDIT_FIELDS[1]}"
AUDITED_REPLAY_SHA="${AUDIT_FIELDS[2]}"
AUDITED_ACTOR_SHA="${AUDIT_FIELDS[3]}"
BASE_FINGERPRINT="${AUDIT_FIELDS[4]}"
RL_TOKEN_FINGERPRINT="${AUDIT_FIELDS[5]}"
PHASE_FINGERPRINT="${AUDIT_FIELDS[6]}"
[[ "$TRAIN_TRANSITIONS" =~ ^[1-9][0-9]*$ ]] || die "derived train split count is invalid"
[[ "$TRAIN_TRANSITIONS" == "$EXPECTED_BOOTSTRAP_TRAIN_TRANSITIONS" ]] || {
  die "derived train split count $TRAIN_TRANSITIONS != $EXPECTED_BOOTSTRAP_TRAIN_TRANSITIONS"
}
CRITIC_BURN_IN_STEPS="$TRAIN_TRANSITIONS"
TRAIN_STEPS=$((CRITIC_BURN_IN_STEPS * 2))

FINAL_CHECKPOINT="$OUTPUT_DIR/step_$(printf '%08d' "$TRAIN_STEPS")"
TRAINING_SUMMARY="$OUTPUT_DIR/training_summary.json"
WARM_START_REPORT="$OUTPUT_DIR/warm_start_actor_only_migration.json"

TRAIN_COMMAND=(
  "$PROJECT_PYTHON" "$TRAIN_SCRIPT"
  --replay-npz "$BOOTSTRAP_REPLAY"
  --output-dir "$OUTPUT_DIR"
  --steps "$TRAIN_STEPS"
  --actor-start-step "$CRITIC_BURN_IN_STEPS"
  --batch-size 256
  --seed 0
  --split train
  --success-fraction 0.5
  --human-fraction 0.5
  --beta-bc 20
  --beta-human-bc 0
  --beta-human-gripper-bc 1
  --human-gripper-bc-scale-m 0.005
  --human-gripper-q-filter-mode "$HUMAN_GRIPPER_Q_FILTER_MODE"
  --human-gripper-q-filter-margin "$HUMAN_GRIPPER_Q_FILTER_MARGIN"
  --reference-dropout 0.5
  --target-policy-noise-std 0.1
  --target-policy-noise-clip 0.2
  --actor-residual-parameterization rank1_bump
  --actor-residual-max-rad 0.005
  --actor-residual-d1-max-rad 0.0015
  --actor-residual-d2-max-rad 0.001
  --actor-direction-cone-deg 15
  --actor-execution-profile "$ACTOR_EXECUTION_PROFILE"
  --execution-filter-profile "$EXECUTION_FILTER_PROFILE"
  --execution-filter-tau-s 0.05
  --actor-max-boundary-jump-rad 0.06
  --actor-direction-static-threshold-rad 0.001
  --actor-projection-scale-steps 33
  --actor-min-projection-scale 0.2
  --actor-governor-fingerprint "$ACTOR_GOVERNOR_FINGERPRINT"
  --residual-max 0.005
  --gripper-residual-max 0.005
  --no-freeze-gripper-residual
  --gripper-residual-mode "$GRIPPER_RESIDUAL_MODE"
  --gripper-residual-d1-max-m 0.0005
  --gripper-residual-d2-max-m 0.0003
  --gripper-boundary-max-m 0.0005
  --gripper-command-min-m 0
  --gripper-command-max-m 0.08
  --gripper-release-reference-m 0.05
  --gripper-release-delta-m 0.002
  --log-every 100
  --warm-start-actor-checkpoint "$SOURCE_ACTOR_CHECKPOINT"
  --warm-start-dry-run-report "$WARM_START_REPORT"
  --allow-warm-start-objective-migration
  --base-checkpoint-fingerprint "$BASE_FINGERPRINT"
  --rl-token-fingerprint "$RL_TOKEN_FINGERPRINT"
  --phase-fingerprint "$PHASE_FINGERPRINT"
  --action-schema-fingerprint "$ACTION_SCHEMA_FINGERPRINT"
  --require-gpu
)

VALIDATE_COMMAND=(
  "$PROJECT_PYTHON" "$VALIDATE_ACTOR_SCRIPT"
  --checkpoint "$FINAL_CHECKPOINT"
  --replay-npz "$BOOTSTRAP_REPLAY"
  --split validation
  --samples 512
  --seed 0
  --output-json "$ACCEPTANCE_REPORT"
  --max-joint-residual-limit 0.005
  --max-gripper-residual-limit 0.005
  --required-action-schema-fingerprint "$ACTION_SCHEMA_FINGERPRINT"
  --max-rank1-fit-error-rad 1e-5
  --max-residual-abs-rad 0.005
  --max-residual-d1-rad 0.0015
  --max-residual-d2-rad 0.001
  --max-direction-cone-deg 15
  --max-active-normalized-residual-step 0.30
  --max-actor-joint-d1-p95-rad 0.025
  --max-chunk-boundary-normalized-residual-jump-p95 1.75
  --max-chunk-boundary-actor-command-joint-d1-p95-rad 0.05235987755982989
)
VALIDATE_READ_ONLY_COMMAND=()
for ((validation_arg_index = 0; validation_arg_index < ${#VALIDATE_COMMAND[@]}; validation_arg_index++)); do
  if [[ "${VALIDATE_COMMAND[$validation_arg_index]}" == "--output-json" ]]; then
    ((validation_arg_index += 1))
    continue
  fi
  VALIDATE_READ_ONLY_COMMAND+=("${VALIDATE_COMMAND[$validation_arg_index]}")
done

FORK_BASE_COMMAND=(
  "$PROJECT_PYTHON" "$FORK_SCRIPT"
  --source-session-root "$SOURCE_SESSION_ROOT"
  --source-state-root "$SOURCE_STATE_ROOT"
  --source-actor-checkpoint "$SOURCE_ACTOR_CHECKPOINT"
  --bootstrap-replay "$BOOTSTRAP_REPLAY"
  --bootstrap-migration-report "$BOOTSTRAP_MIGRATION_REPORT"
  --initial-v3-checkpoint "$FINAL_CHECKPOINT"
  --initial-v3-validation-report "$ACCEPTANCE_REPORT"
  --target-session-root "$TARGET_SESSION_ROOT"
  --workspace "$STAGING_ROOT"
  --runtime "$STAGING_ROOT"
  --target-state-dir "$TARGET_STATE_DIR"
  --expected-source-latest-episode "$EXPECTED_SOURCE_LATEST_EPISODE"
  --expected-target-episode-floor "$EXPECTED_EPISODE_FLOOR"
  --expected-initial-v3-checkpoint-step "$TRAIN_STEPS"
)
FORK_CREATE_COMMAND=("${FORK_BASE_COMMAND[@]}" --create)
VALIDATE_LINEAGE_COMMAND=(
  "$PROJECT_PYTHON" "$VALIDATE_LINEAGE_SCRIPT"
  --session-root "$TARGET_SESSION_ROOT"
  --state-root "$TARGET_STATE_ROOT"
  --expected-episode-floor "$EXPECTED_EPISODE_FLOOR"
  --read-only
)

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

print_contract() {
  cat <<EOF
Gripper-close v3 initial preparation:
  staging workspace/runtime : $STAGING_ROOT
  source session            : $SOURCE_SESSION_ROOT
  accepted Actor source     : $SOURCE_ACTOR_CHECKPOINT (step $EXPECTED_SOURCE_ACTOR_STEP)
  rejected source excluded  : step $REJECTED_SOURCE_STEP
  bootstrap replay          : $BOOTSTRAP_REPLAY
  bootstrap SHA256          : $AUDITED_REPLAY_SHA
  bootstrap quality         : ${EXPECTED_BOOTSTRAP_EPISODES} admitted episodes, R+${EXPECTED_BOOTSTRAP_REWARD_POSITIVE_EPISODES}/R-${EXPECTED_BOOTSTRAP_REWARD_NEGATIVE_EPISODES}/H${EXPECTED_BOOTSTRAP_HUMAN_EPISODES}
  train/validation rows      : $TRAIN_TRANSITIONS/$VALIDATION_TRANSITIONS
  learner schedule           : Critic-only 0..$((CRITIC_BURN_IN_STEPS - 1)); Actor starts at $CRITIC_BURN_IN_STEPS; final step=$TRAIN_STEPS
  target objective          : beta=20/0/1
  human gripper objective   : all admitted human, reward labels preserved; Q-filter=${HUMAN_GRIPPER_Q_FILTER_MODE}, margin=${HUMAN_GRIPPER_Q_FILTER_MARGIN}
  gripper governor          : close-only max=5mm d1=0.5mm d2=0.3mm boundary=0.5mm release=Pi0.5-priority
  final checkpoint          : $FINAL_CHECKPOINT
  training summary          : $TRAINING_SUMMARY
  acceptance report         : $ACCEPTANCE_REPORT
  target lineage            : $TARGET_SESSION_ROOT
  target episode floor      : $EXPECTED_EPISODE_FLOOR
EOF
}

verify_completed_artifacts() {
  local require_lineage="${1:-0}"
  [[ -d "$OUTPUT_DIR" ]] || die "existing training output is missing: $OUTPUT_DIR"
  [[ -d "$FINAL_CHECKPOINT" ]] || die "existing final checkpoint is missing: $FINAL_CHECKPOINT"
  [[ -f "$TRAINING_SUMMARY" ]] || die "existing training_summary.json is missing"
  [[ -f "$ACCEPTANCE_REPORT" ]] || die "existing acceptance report is missing"
  if [[ "$require_lineage" == "1" ]]; then
    [[ -d "$TARGET_STATE_ROOT" ]] || die "existing target lineage state is missing"
  fi

  "$PROJECT_PYTHON" - \
    "$TRAINING_SUMMARY" \
    "$ACCEPTANCE_REPORT" \
    "$FINAL_CHECKPOINT" \
    "$BOOTSTRAP_REPLAY" \
    "$TRAIN_STEPS" \
    "$CRITIC_BURN_IN_STEPS" \
    "$EXPECTED_BOOTSTRAP_SHA256" \
    "$ACTION_SCHEMA_FINGERPRINT" <<'PY'
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

summary_path, acceptance_path, checkpoint_raw, replay_raw, step_raw, actor_start_raw, replay_sha, schema = sys.argv[1:]
summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
acceptance = json.loads(Path(acceptance_path).read_text(encoding="utf-8"))
checkpoint = Path(checkpoint_raw).expanduser().resolve()
replay = Path(replay_raw).expanduser().resolve()
step = int(step_raw)
actor_start_step = int(actor_start_raw)
actor_updates = max(0, (step - actor_start_step) // 2)

if summary.get("format") != "openpi_real_rlt_jax_training_summary_v1":
    raise SystemExit("invalid training summary format")
for key, expected in {
    "start_step": 0,
    "final_step": step,
    "additional_steps": step,
    "actor_start_step": actor_start_step,
    "actor_updates_expected": actor_updates,
    "critic_updates": step,
}.items():
    if int(summary.get(key, -1)) != expected:
        raise SystemExit(f"training summary {key} mismatch")
if Path(str(summary.get("checkpoint", ""))).expanduser().resolve() != checkpoint:
    raise SystemExit("training summary checkpoint mismatch")
metrics = summary.get("final_metrics")
if not isinstance(metrics, dict):
    raise SystemExit("training summary lacks final metrics")
for key in (
    "actor_loss",
    "actor_human_gripper_bc_loss",
    "actor_admitted_human_gripper_steps",
    "actor_human_gripper_q_filter_selected_steps",
    "actor_human_gripper_q_filter_fraction",
    "actor_reward1_human_gripper_q_filter_fraction",
    "actor_reward0_human_gripper_q_filter_fraction",
    "actor_reward1_human_gripper_q_advantage_mean",
    "actor_reward0_human_gripper_q_advantage_mean",
    "actor_gripper_residual_min_m",
    "actor_gripper_residual_max_m",
    "critic_loss",
    "critic_reward1_q_mean",
    "critic_reward0_q_mean",
    "critic_reward1_reward0_q_gap",
):
    try:
        value = float(metrics[key])
    except (KeyError, TypeError, ValueError):
        raise SystemExit(f"training summary lacks finite metric {key}")
    if not math.isfinite(value):
        raise SystemExit(f"training summary metric {key} is non-finite")
if float(metrics["actor_gripper_residual_max_m"]) > 1.0e-7:
    raise SystemExit("trained Actor produced an opening gripper residual")
if float(metrics["actor_gripper_residual_min_m"]) < -0.0050001:
    raise SystemExit("trained Actor exceeded the 5mm close-only residual bound")

if acceptance.get("format") != "openpi_real_rlt_actor_acceptance":
    raise SystemExit("invalid acceptance report format")
if acceptance.get("passed") is not True:
    raise SystemExit("initial v3 Actor acceptance did not pass")
if int(acceptance.get("update_step", -1)) != step:
    raise SystemExit("acceptance/checkpoint step mismatch")
if Path(str(acceptance.get("checkpoint", ""))).expanduser().resolve() != checkpoint:
    raise SystemExit("acceptance checkpoint path mismatch")
if Path(str(acceptance.get("replay", ""))).expanduser().resolve() != replay:
    raise SystemExit("acceptance replay path mismatch")
if acceptance.get("replay_sha256") != replay_sha:
    raise SystemExit("acceptance replay SHA mismatch")
if acceptance.get("replay_split") != "validation":
    raise SystemExit("acceptance did not use the held-out validation split")
if acceptance.get("action_schema_fingerprint") != schema:
    raise SystemExit("acceptance action schema mismatch")
for key in (
    "gripper_close_only_contract",
    "rank1_checkpoint_parameterization_contract",
    "rank1_residual_contract",
    "direction_cone_contract",
    "action_schema_fingerprint_contract",
):
    if acceptance.get(key) is not True:
        raise SystemExit(f"acceptance contract {key} did not pass")
print(
    "GRIPPER_CLOSE_V3_EXISTING_ARTIFACTS_PASS "
    f"step={step} actor_start={actor_start_step} "
    f"actor_updates={actor_updates} critic_updates={step}"
)
PY
}

export PYTHONPATH="$STAGING_ROOT/src:$STAGING_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export RLT_V3_WORKSPACE_OVERRIDE="$STAGING_ROOT"
export RLT_V3_RUNTIME_OVERRIDE="$STAGING_ROOT"

print_contract

if [[ "$VERIFY_EXISTING" == "1" ]]; then
  verify_completed_artifacts 1
  echo "Recomputing Actor acceptance read-only:"
  print_command "${VALIDATE_READ_ONLY_COMMAND[@]}"
  "${VALIDATE_READ_ONLY_COMMAND[@]}"
  echo "Validating existing lineage read-only:"
  print_command "${VALIDATE_LINEAGE_COMMAND[@]}"
  "${VALIDATE_LINEAGE_COMMAND[@]}"
  echo "Final training_summary.json:"
  "$PROJECT_PYTHON" -m json.tool "$TRAINING_SUMMARY"
  echo "Final Actor acceptance report:"
  "$PROJECT_PYTHON" -m json.tool "$ACCEPTANCE_REPORT"
  echo "GRIPPER_CLOSE_V3_INITIAL_VERIFY_PASS"
  exit 0
fi

if [[ -e "$OUTPUT_DIR" || -L "$OUTPUT_DIR" ]]; then
  die "training output already exists; use --verify-existing for read-only inspection: $OUTPUT_DIR"
fi
if [[ -e "$TARGET_SESSION_ROOT" || -L "$TARGET_SESSION_ROOT" ]]; then
  die "target lineage already exists; use --verify-existing for read-only inspection: $TARGET_SESSION_ROOT"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY RUN: no directory or file will be created."
  echo "Training command:"
  print_command "${TRAIN_COMMAND[@]}"
  echo "Acceptance command:"
  print_command "${VALIDATE_COMMAND[@]}"
  echo "Fork read-only preflight command:"
  print_command "${FORK_BASE_COMMAND[@]}"
  echo "Fork create-once command:"
  print_command "${FORK_CREATE_COMMAND[@]}"
  echo "Final lineage read-only validation command:"
  print_command "${VALIDATE_LINEAGE_COMMAND[@]}"
  echo "GRIPPER_CLOSE_V3_INITIAL_DRY_RUN_PASS"
  exit 0
fi

echo "Starting isolated GPU-required initial v3 training."
print_command "${TRAIN_COMMAND[@]}"
"${TRAIN_COMMAND[@]}"

[[ -d "$FINAL_CHECKPOINT" ]] || die "training did not create the expected final checkpoint"
[[ -f "$TRAINING_SUMMARY" ]] || die "training did not create training_summary.json"

echo "Running held-out Actor acceptance."
print_command "${VALIDATE_COMMAND[@]}"
"${VALIDATE_COMMAND[@]}"
verify_completed_artifacts 0

echo "Running read-only fork preflight."
print_command "${FORK_BASE_COMMAND[@]}"
"${FORK_BASE_COMMAND[@]}"

# Close the race in which a rollout/update could be started while the offline
# learner was running.  Lineage creation remains prohibited until the host is
# idle again.
assert_no_active_rlt_jobs
[[ ! -e "$TARGET_SESSION_ROOT" && ! -L "$TARGET_SESSION_ROOT" ]] || {
  die "target lineage appeared after preflight; refusing to overwrite it"
}

echo "Creating the immutable v3 lineage exactly once."
print_command "${FORK_CREATE_COMMAND[@]}"
"${FORK_CREATE_COMMAND[@]}"

echo "Validating the new lineage read-only."
print_command "${VALIDATE_LINEAGE_COMMAND[@]}"
"${VALIDATE_LINEAGE_COMMAND[@]}"

echo "Final training_summary.json:"
"$PROJECT_PYTHON" -m json.tool "$TRAINING_SUMMARY"
echo "Final Actor acceptance report:"
"$PROJECT_PYTHON" -m json.tool "$ACCEPTANCE_REPORT"
echo "GRIPPER_CLOSE_V3_INITIAL_PREPARE_PASS lineage=$TARGET_SESSION_ROOT floor=$EXPECTED_EPISODE_FLOOR"
