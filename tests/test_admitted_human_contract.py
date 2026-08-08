#!/usr/bin/env python3
"""Static audit for the reward-independent admitted-human gripper-v3 contract."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TOOLS = Path(
    os.environ.get("GRIPPER_V3_TOOLS_ROOT", ROOT / ".codex_gripper_v3_tools")
).resolve()
SCRIPTS = Path(
    os.environ.get(
        "GRIPPER_V3_SCRIPT_ROOT",
        ROOT / ".codex_gripper_v3_scripts",
    )
).resolve()
DOCS = Path(
    os.environ.get("GRIPPER_V3_DOC_ROOT", ROOT / ".codex_doc_update")
).resolve()


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"{label}: missing {needle!r}")


def reject(text: str, needle: str, label: str) -> None:
    if needle in text:
        raise AssertionError(f"{label}: contains forbidden {needle!r}")


def main() -> None:
    fork = (TOOLS / "fork_close_assist_v3_lineage.py").read_text(
        encoding="utf-8"
    )
    validator = (TOOLS / "validate_close_assist_v3_lineage.py").read_text(
        encoding="utf-8"
    )
    updater = (TOOLS / "run_online_rlt_update.py").read_text(encoding="utf-8")
    prepare = (SCRIPTS / "prepare_gripper_close_v3_initial.sh").read_text(
        encoding="utf-8"
    )
    readme = (DOCS / "README_GRIPPER_CLOSE_V3.md").read_text(encoding="utf-8")

    for text, label in (
        (fork, "fork"),
        (validator, "validator"),
        (prepare, "prepare"),
    ):
        require(text, "episode_split", label)
        require(text, '"validation"', label)
        reject(text, 'replay["split"]', label)
        reject(text, '{"train", "val"}', label)
    require(
        fork,
        "highest_source_episode_directory_index",
        "fork source floor provenance",
    )
    require(
        validator,
        "highest source episode directory + 1",
        "validator source floor contract",
    )

    for text, label in (
        (fork, "fork"),
        (prepare, "prepare"),
    ):
        require(
            text,
            "all_admitted_human_dim6_clip_delta_to_[-0.005,0]_",
            label,
        )
        require(
            text,
            "critic_min_advantage_q_filter_reward_independent",
            label,
        )
        require(text, "reward1_human_steps", label)
        require(text, "reward0_human_steps", label)

    for text, label in (
        (fork, "fork"),
        (updater, "updater"),
    ):
        require(text, "critic_min_advantage_v1", label)
        require(text, "human_gripper_q_filter_margin", label)
    require(prepare, "critic_min_advantage_v1", "prepare")
    require(prepare, "HUMAN_GRIPPER_Q_FILTER_MARGIN", "prepare")
    require(validator, "HUMAN_GRIPPER_Q_FILTER_MODE", "validator")
    require(validator, "HUMAN_GRIPPER_Q_FILTER_MARGIN", "validator")

    require(updater, "--min-admitted-human-episodes", "updater")
    require(
        updater,
        "and human_episodes\n        >= int(getattr(args, "
        '"min_admitted_human_episodes", 0))',
        "updater admitted-human readiness",
    )
    require(updater, "--human-gripper-q-filter-mode", "updater")
    require(updater, "--human-gripper-q-filter-margin", "updater")
    require(updater, "reward1_reward0_exec_q_gap", "updater reward Q guard")

    require(prepare, 'CRITIC_BURN_IN_STEPS="$TRAIN_TRANSITIONS"', "prepare")
    require(prepare, "TRAIN_STEPS=$((CRITIC_BURN_IN_STEPS * 2))", "prepare")
    require(prepare, '--actor-start-step "$CRITIC_BURN_IN_STEPS"', "prepare")
    require(fork, "EXPECTED_INITIAL_CHECKPOINT_STEP = 288", "fork")
    require(fork, "EXPECTED_INITIAL_CRITIC_BURN_IN_STEPS = 144", "fork")

    require(readme, "248条完整C10", "readme")
    require(readme, "169条transition", "readme")
    require(readme, "Critic updates: 288", "readme")
    require(readme, "Actor start:    step 144", "readme")

    print(
        "GRIPPER_CLOSE_V3_ADMITTED_HUMAN_STATIC_PASS "
        "replay=R+andR- critic=all actor=Q-filter "
        "rows=144/25 chunks=248 td=169 schedule=288/144/72"
    )


if __name__ == "__main__":
    main()
