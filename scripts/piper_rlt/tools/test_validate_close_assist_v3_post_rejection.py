#!/usr/bin/env python3
"""Regression test for a valid rejected online-update attempt.

The expensive replay/checkpoint provenance readers are stubbed after their
inputs have been bound to a synthetic lineage.  The state-machine portion of
``validate()`` remains real: a rejected attempt may advance ``attempt_index``
and ``last_attempt_episode_ids`` without advancing the accepted/deployed
checkpoint or ``trained_episode_ids``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import validate_close_assist_v3_lineage as validator  # noqa: E402


def _json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _bytes(path: Path, payload: bytes = b"fixture\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_valid_post_rejection_keeps_initial_actor_deployed(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    session = tmp_path / "greenblock_rlt_gripper_close_v3"
    state_root = session / ".online_rlt_persistent_gripper_v3"
    state_root.mkdir(parents=True)

    workspace = tmp_path / "workspace"
    _bytes(workspace / ".venv" / "bin" / "python")
    (workspace / "src" / "openpi").mkdir(parents=True)
    runtime = tmp_path / "runtime"
    (runtime / "piper_runtime").mkdir(parents=True)

    selected_actor = state_root / "selected_actor_checkpoint.txt"
    _bytes(selected_actor)
    source_actor = state_root / "provenance" / "source_actor_step_00012627"
    source_actor.mkdir(parents=True)
    source_rejected = (
        state_root / "provenance" / "source_rejected_step_00000144"
    )
    source_rejected.mkdir(parents=True)
    initial_checkpoint = state_root / "learner" / "step_00000288"
    initial_checkpoint.mkdir(parents=True)
    rejected_checkpoint = state_root / "learner" / "step_00000299"
    rejected_checkpoint.mkdir(parents=True)

    bootstrap_ids = [
        f"episode_{index:06d}" for index in range(371, 401)
    ]
    rejected_attempt_ids = [*bootstrap_ids, "episode_000413"]
    bootstrap_replay = state_root / "bootstrap_gripper_v5_replay.npz"
    rejected_replay = state_root / "replays" / "attempt_000002.npz"
    bootstrap_replay.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        bootstrap_replay,
        episode_id=np.asarray(bootstrap_ids, dtype="U14"),
    )
    rejected_replay.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        rejected_replay,
        episode_id=np.asarray(rejected_attempt_ids, dtype="U14"),
    )
    bootstrap_sha = _sha256(bootstrap_replay)
    rejected_replay_sha = _sha256(rejected_replay)

    legacy_replay = state_root / "provenance" / "legacy_source_replay.npz"
    legacy_replay.parent.mkdir(parents=True, exist_ok=True)
    legacy_ids = ["episode_000308"]
    np.savez(
        legacy_replay,
        episode_id=np.asarray(legacy_ids, dtype="U14"),
    )

    bootstrap_report = (
        state_root / "provenance" / "bootstrap_migration.json"
    )
    _json(bootstrap_report, {"fixture": True})

    initial_validation = (
        state_root / "provenance" / "initial_v3_acceptance.json"
    )
    initial_validation_payload = {
        "format": "openpi_real_rlt_actor_acceptance",
        "passed": True,
        "checkpoint": str(initial_checkpoint),
        "replay": str(bootstrap_replay),
        "update_step": 288,
        "replay_sha256": bootstrap_sha,
        "replay_split": "validation",
        "samples": 30,
        "fingerprints": {
            "replay_sha256": bootstrap_sha,
            "action_schema": validator.ACTION_SCHEMA_FINGERPRINT,
            "actor_execution_profile": validator.ACTOR_EXECUTION_PROFILE,
            "execution_filter_profile": validator.EXECUTION_FILTER_PROFILE,
            "actor_governor": validator.PERSISTENT_GOVERNOR_FINGERPRINT,
            "warm_start_actor_source_schema": validator.SOURCE_ACTOR_SCHEMA,
        },
    }
    _json(initial_validation, initial_validation_payload)

    source_actor_tree = "source-actor-tree"
    source_actor_learner = "source-actor-learner"
    source_rejected_tree = "source-rejected-tree"
    source_rejected_learner = "source-rejected-learner"
    initial_tree = "initial-v3-tree"
    initial_learner = "initial-v3-learner"

    contract = validator._contract_payload()
    target = {
        "session_root": str(session),
        "state_root": str(state_root),
        "workspace": str(workspace),
        "runtime": str(runtime),
        "selected_actor_file": str(selected_actor),
        "shadow_service": "openpi-rlt-shadow-policy-gripper-v3.service",
        "episode_index_floor": 408,
        "first_episode_id": "episode_000408",
        "initial_update_index": 1,
        "initial_attempt_index": 1,
        "bootstrap_gripper_replay_copy": str(bootstrap_replay),
        "legacy_replay_provenance_copy": str(legacy_replay),
        "initial_v3_checkpoint_copy": str(initial_checkpoint),
        "initial_v3_validation_report_copy": str(initial_validation),
    }
    source = {
        "latest_complete_episode": "episode_000406",
        "latest_complete_episode_index": 406,
        "highest_source_episode_directory": "episode_000407",
        "highest_source_episode_directory_index": 407,
        "last_attempt_episode_ids": bootstrap_ids,
        "actor": {
            "tree_sha256": source_actor_tree,
            "learner_msgpack_sha256": source_actor_learner,
        },
        "rejected_candidate": {
            "step": validator.EXPECTED_REJECTED_STEP,
            "checkpoint": str(source_rejected),
            "tree_sha256": source_rejected_tree,
            "learner_msgpack_sha256": source_rejected_learner,
            "action_schema": validator.SOURCE_FROZEN_EXECUTION_SCHEMA,
        },
        "rejected_replay": {
            "path": str(bootstrap_replay),
            "sha256": bootstrap_sha,
        },
    }
    manifest = {
        "format": validator.FORK_FORMAT,
        "created": True,
        "preflight_passed": True,
        "contract": contract,
        "target": target,
        "source": source,
        "bootstrap": {
            "episode_ids": bootstrap_ids,
            "path": str(bootstrap_replay),
            "sha256": bootstrap_sha,
        },
        "initial_v3_checkpoint": {
            "checkpoint": str(initial_checkpoint),
            "tree_sha256": initial_tree,
        },
        "initial_v3_validation": {"passed": True},
    }
    manifest_path = state_root / "fork_manifest.json"
    _json(manifest_path, manifest)

    state = {
        "format": validator.STATE_FORMAT,
        "lineage_mode": validator.LINEAGE_MODE,
        **{
            key: value
            for key, value in contract.items()
            if key not in {"state_format", "lineage_mode"}
        },
        "session_root": str(session),
        "workspace": str(workspace),
        "runtime": str(runtime),
        "selected_checkpoint_file": str(selected_actor),
        "shadow_service": "openpi-rlt-shadow-policy-gripper-v3.service",
        "fork_manifest": str(manifest_path),
        "fork_manifest_sha256": _sha256(manifest_path),
        "replay_training_policy": validator.REPLAY_POLICY,
        "frozen_base_replay": None,
        "frozen_base_episode_ids": [],
        "legacy_replay_training_rows": 0,
        "legacy_replay_policy": (
            "immutable_provenance_only_never_merged"
        ),
        "episode_index_floor": 408,
        "initial_actor_warm_start_checkpoint": str(source_actor),
        "initial_actor_warm_start_checkpoint_tree_sha256": (
            source_actor_tree
        ),
        "initial_actor_warm_start_learner_msgpack_sha256": (
            source_actor_learner
        ),
        "source_actor_checkpoint_step": (
            validator.EXPECTED_SOURCE_ACTOR_STEP
        ),
        "source_rejected_checkpoint_step": (
            validator.EXPECTED_REJECTED_STEP
        ),
        "source_rejected_checkpoint_tree_sha256": source_rejected_tree,
        "source_rejected_checkpoint_learner_msgpack_sha256": (
            source_rejected_learner
        ),
        "source_rejected_checkpoint_disposition": (
            "rejected_never_warm_start_never_deploy"
        ),
        "bootstrap_gripper_episode_ids": bootstrap_ids,
        "bootstrap_gripper_episode_count": 30,
        "bootstrap_gripper_replay": str(bootstrap_replay),
        "bootstrap_gripper_replay_sha256": bootstrap_sha,
        "bootstrap_gripper_train_transition_count": 144,
        "bootstrap_gripper_quality": {
            "episodes": 30,
            "reward_positive_episodes": 24,
            "reward_negative_episodes": 6,
            "admitted_human_episodes": 29,
        },
        "bootstrap_gripper_migration_report": str(bootstrap_report),
        "bootstrap_gripper_migration_report_sha256": _sha256(
            bootstrap_report
        ),
        "legacy_source_replay": str(legacy_replay),
        "legacy_source_replay_sha256": _sha256(legacy_replay),
        "legacy_source_replay_episode_ids": legacy_ids,
        "initial_v3_checkpoint": str(initial_checkpoint),
        "initial_v3_checkpoint_step": 288,
        "initial_v3_checkpoint_tree_sha256": initial_tree,
        "initial_v3_checkpoint_learner_msgpack_sha256": initial_learner,
        "initial_v3_validation_report": str(initial_validation),
        "initial_v3_validation_report_sha256": _sha256(
            initial_validation
        ),
        "initial_v3_validation_passed": True,
        "initial_v3_validation_samples": 30,
        # The accepted state remains the 30-episode bootstrap/step 288.
        "trained_episode_ids": bootstrap_ids,
        "last_update_episode_count": 30,
        "last_train_transition_count": 144,
        "update_index": 1,
        "latest_replay": str(bootstrap_replay),
        "latest_replay_sha256": bootstrap_sha,
        "latest_checkpoint": str(initial_checkpoint),
        "deployment_checkpoint": str(initial_checkpoint),
        # Attempt 2 includes episode 413, but step 299 was rejected.
        "last_attempt_episode_ids": rejected_attempt_ids,
        "last_attempt_episode_count": 31,
        "last_attempt_train_transition_count": 155,
        "attempt_index": 2,
        "latest_rejected_replay": str(rejected_replay),
        "latest_rejected_checkpoint": str(rejected_checkpoint),
        "latest_rejection_reason": (
            "success_failure_exec_q_gap=-0.0923 below 0.0"
        ),
    }
    _json(state_root / "online_state.json", state)

    source_actor_audit = {
        "tree_sha256": source_actor_tree,
        "learner_msgpack_sha256": source_actor_learner,
        "objective_weights": dict(validator.EXPECTED_SOURCE_BETAS),
    }
    initial_audit = {
        "tree_sha256": initial_tree,
        "learner_msgpack_sha256": initial_learner,
        "objective_weights": dict(validator.EXPECTED_TARGET_BETAS),
        "step": 288,
    }
    bootstrap_audit = {
        "sha256": bootstrap_sha,
        "transitions": 150,
        "train_transitions": 144,
        "validation_transitions": 6,
        "episode_ids": bootstrap_ids,
        "reward_positive_human_steps": 100,
        "reward_negative_human_steps": 20,
    }

    monkeypatch.setattr(
        validator,
        "_audit_source_actor",
        lambda path: source_actor_audit,
    )
    monkeypatch.setattr(
        validator,
        "_checkpoint_manifest",
        lambda path: (
            [
                {
                    "path": "learner.msgpack",
                    "sha256": source_rejected_learner,
                    "size_bytes": 1,
                }
            ],
            source_rejected_tree,
        ),
    )
    monkeypatch.setattr(
        validator,
        "_replay_audit",
        lambda path, *, expected_episode_ids: bootstrap_audit,
    )
    monkeypatch.setattr(
        validator,
        "_validate_bootstrap_report_copy",
        lambda *args, **kwargs: {"valid": True},
    )
    monkeypatch.setattr(
        validator,
        "_audit_initial_v3_checkpoint",
        lambda *args, **kwargs: initial_audit,
    )
    monkeypatch.setattr(
        validator,
        "_validate_objective_migration",
        lambda **kwargs: {"valid": True},
    )

    def fake_replay_validation(
        path: Path,
        *,
        expected_sha256: str,
        expected_episode_ids: list[str],
    ) -> dict[str, Any]:
        resolved = path.resolve()
        if resolved == rejected_replay.resolve():
            assert expected_sha256 == rejected_replay_sha
            assert expected_episode_ids == rejected_attempt_ids
            train_count = 155
        else:
            assert resolved == bootstrap_replay.resolve()
            assert expected_sha256 == bootstrap_sha
            assert expected_episode_ids == bootstrap_ids
            train_count = 144
        return {
            "path": str(resolved),
            "sha256": expected_sha256,
            "episode_ids": expected_episode_ids,
            "split_counts": {"train": train_count, "validation": 1},
        }

    monkeypatch.setattr(
        validator,
        "_validate_v5_replay_generic",
        fake_replay_validation,
    )

    def fake_checkpoint_validation(
        path: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        resolved = path.resolve()
        if resolved == rejected_checkpoint.resolve():
            step = 299
        else:
            assert resolved == initial_checkpoint.resolve()
            step = 288
        return {"path": str(resolved), "step": step, "valid": True}

    monkeypatch.setattr(
        validator,
        "_validate_checkpoint_v3",
        fake_checkpoint_validation,
    )
    monkeypatch.setattr(
        validator,
        "_validate_config",
        lambda **kwargs: {"fixture": "valid"},
    )
    monkeypatch.setattr(
        validator,
        "_validate_no_old_v2_mix",
        lambda **kwargs: ["fixture:v5_only"],
    )

    report = validator.validate(
        session,
        state_root,
        expected_episode_floor=408,
    )

    assert report["valid"] is True
    assert report["state_episode_ids"]["trained"] == bootstrap_ids
    assert (
        report["state_episode_ids"]["last_attempt"]
        == rejected_attempt_ids
    )
    assert report["latest_checkpoint"]["step"] == 288
    assert report["deployment_checkpoint"]["step"] == 288
    assert report["latest_rejected_attempt"]["checkpoint"]["step"] == 299
    assert (
        report["latest_rejected_attempt"]["replay"]["episode_ids"]
        == rejected_attempt_ids
    )
