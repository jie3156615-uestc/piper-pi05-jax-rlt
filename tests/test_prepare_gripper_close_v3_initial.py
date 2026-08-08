#!/usr/bin/env python3
"""Side-effect-free static contract test for initial gripper-close v3 prep."""

from __future__ import annotations

import os
from pathlib import Path


SCRIPT = Path(
    os.environ.get(
        "GRIPPER_V3_PREP_SCRIPT",
        Path(__file__).with_name("prepare_gripper_close_v3_initial.sh"),
    )
).resolve()


def require(text: str, needle: str) -> None:
    if needle not in text:
        raise AssertionError(f"prepare script lacks {needle!r}")


def reject(text: str, needle: str) -> None:
    if needle in text:
        raise AssertionError(f"prepare script contains forbidden token {needle!r}")


def main() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    for needle in (
        "set -euo pipefail",
        "/home/cwzk/gripper_close_v3_staging_20260727",
        "greenblock_rlt_gripper_close_v3_from_v2_ep407_20260727",
        'DEFAULT_SOURCE_LATEST_EPISODE="episode_000406"',
        'DEFAULT_EPISODE_FLOOR="408"',
        'EXPECTED_SOURCE_ACTOR_STEP="12627"',
        'REJECTED_SOURCE_STEP="144"',
        "658d981065227b48656e7b792fe7da2ce390f91c4584b22eb35bb54007cbbff5",
        'EXPECTED_BOOTSTRAP_EPISODES="30"',
        'EXPECTED_BOOTSTRAP_REWARD_POSITIVE_EPISODES="24"',
        'EXPECTED_BOOTSTRAP_REWARD_NEGATIVE_EPISODES="6"',
        'EXPECTED_BOOTSTRAP_HUMAN_EPISODES="29"',
        'EXPECTED_BOOTSTRAP_TRAIN_TRANSITIONS="144"',
        '"episode_split",',
        'replay["episode_split"]',
        '{"train", "validation"}',
        'HUMAN_GRIPPER_Q_FILTER_MODE="critic_min_advantage_v1"',
        'HUMAN_GRIPPER_Q_FILTER_MARGIN="0.0"',
        "all_admitted_human_dim6_clip_delta_to_[-0.005,0]_critic_min_advantage_q_filter_reward_independent",
        "assert_no_active_rlt_jobs",
        "[r]lt_online_session",
        "[r]lt_takeover_rollout",
        "[r]lt_shadow_policy_service",
        "[t]rain_real_rlt_jax",
        "[r]un_online_rlt_update",
        "[r]un_rlt_online_update_hook",
        '--steps "$TRAIN_STEPS"',
        '--actor-start-step "$CRITIC_BURN_IN_STEPS"',
        "--require-gpu",
        "--warm-start-actor-checkpoint",
        '"$SOURCE_ACTOR_CHECKPOINT"',
        "--beta-bc 20",
        "--beta-human-bc 0",
        "--beta-human-gripper-bc 1",
        '--human-gripper-q-filter-mode "$HUMAN_GRIPPER_Q_FILTER_MODE"',
        '--human-gripper-q-filter-margin "$HUMAN_GRIPPER_Q_FILTER_MARGIN"',
        "--no-freeze-gripper-residual",
        "--gripper-residual-mode",
        "close_only_persistent_v1",
        "--gripper-residual-max 0.005",
        "--gripper-residual-d1-max-m 0.0005",
        "--gripper-residual-d2-max-m 0.0003",
        "--gripper-boundary-max-m 0.0005",
        "--gripper-release-reference-m 0.05",
        "--gripper-release-delta-m 0.002",
        "training_summary.json",
        "actor_admitted_human_gripper_steps",
        "actor_reward0_human_gripper_q_filter_fraction",
        "critic_reward1_reward0_q_gap",
        "initial_v3_acceptance.json",
        "fork_close_assist_v3_lineage.py",
        "validate_close_assist_v3_lineage.py",
        "--expected-initial-v3-checkpoint-step",
        '--workspace "$STAGING_ROOT"',
        '--runtime "$STAGING_ROOT"',
        "--expected-episode-floor",
        "--expected-target-episode-floor",
        "--dry-run",
        "--verify-existing",
        "GRIPPER_CLOSE_V3_INITIAL_DRY_RUN_PASS",
        "GRIPPER_CLOSE_V3_INITIAL_VERIFY_PASS",
        "GRIPPER_CLOSE_V3_INITIAL_PREPARE_PASS",
        "openpi-rlt-shadow-policy.service",
        'RLT_V3_WORKSPACE_OVERRIDE="$STAGING_ROOT"',
        'RLT_V3_RUNTIME_OVERRIDE="$STAGING_ROOT"',
        '[[ -e "$OUTPUT_DIR" || -L "$OUTPUT_DIR" ]]',
        '[[ -e "$TARGET_SESSION_ROOT" || -L "$TARGET_SESSION_ROOT" ]]',
    ):
        require(text, needle)

    reject(text, "--warm-start-actor-checkpoint \"$SOURCE_STATE_ROOT/learner/step_00000144\"")
    reject(text, "--steps 144")
    reject(text, 'RLT_V3_WORKSPACE_OVERRIDE="$HOME/openpi_jax_piper_lora_v1_20260707"')
    reject(text, 'RLT_V3_RUNTIME_OVERRIDE="$HOME/piper_jax_inference_v1"')
    reject(text, "systemctl stop")
    reject(text, "pkill ")
    reject(text, "kill -")
    reject(text, 'replay["split"]')
    reject(text, '{"train", "val"}')

    train_start = text.index("TRAIN_COMMAND=(")
    train_end = text.index("\n)", train_start)
    train_command = text[train_start:train_end]
    require(train_command, "--require-gpu")
    require(train_command, '--steps "$TRAIN_STEPS"')
    require(train_command, '--actor-start-step "$CRITIC_BURN_IN_STEPS"')
    require(train_command, '--warm-start-actor-checkpoint "$SOURCE_ACTOR_CHECKPOINT"')
    reject(train_command, "step_00000144")

    dry_start = text.index('if [[ "$DRY_RUN" == "1" ]]')
    dry_end = text.index("\nfi", dry_start)
    dry_block = text[dry_start:dry_end]
    for needle in (
        'print_command "${TRAIN_COMMAND[@]}"',
        'print_command "${VALIDATE_COMMAND[@]}"',
        'print_command "${FORK_BASE_COMMAND[@]}"',
        'print_command "${FORK_CREATE_COMMAND[@]}"',
        'print_command "${VALIDATE_LINEAGE_COMMAND[@]}"',
    ):
        require(dry_block, needle)
    dry_lines = {line.strip() for line in dry_block.splitlines()}
    if '"${TRAIN_COMMAND[@]}"' in dry_lines:
        raise AssertionError("dry-run block executes the training command")
    if '"${FORK_CREATE_COMMAND[@]}"' in dry_lines:
        raise AssertionError("dry-run block executes the fork-create command")

    print(
        "GRIPPER_CLOSE_V3_INITIAL_STATIC_PASS "
        "source=step12627 rejected=step144 "
        "bootstrap=30/R+24/R-6/H29 rows=144 "
        "schedule=critic288/actor_start144/actor72 "
        "objective=20/0/1+admitted-human-Q-filter "
        "gripper=close-only-5mm gpu=required"
    )


if __name__ == "__main__":
    main()
