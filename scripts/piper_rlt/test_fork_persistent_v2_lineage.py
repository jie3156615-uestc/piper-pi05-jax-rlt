from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


TOOLS = Path(__file__).parent / "tools"
sys.path.insert(0, str(TOOLS))
import fork_persistent_v2_lineage as fork  # noqa: E402
import validate_persistent_v2_lineage as validate_lineage  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    source_session = tmp_path / "source"
    source_state = source_session / ".online_rlt_v4"
    checkpoint = source_state / "learner" / "step_00012627"
    replay = source_state / "replays" / "latest" / "replay.npz"
    checkpoint.mkdir(parents=True)
    replay.parent.mkdir(parents=True)
    ids = np.asarray(["episode_000308", "episode_000352"])
    np.savez_compressed(
        replay,
        episode_id=ids,
        reward=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    replay_sha = _sha256(replay)
    (checkpoint / "learner.msgpack").write_bytes(b"legacy-actor-critic")
    (checkpoint / "metadata.json").write_text(
        json.dumps(
            {
                "format": "openpi_real_rlt_jax_learner",
                "update_step": 12627,
                "config": {
                    "beta_bc": 40.0,
                    "beta_human_bc": 0.0,
                    "chunk_stride": 2,
                    "actor_residual_parameterization": "rank1_bump",
                    "freeze_gripper_residual": True,
                },
                "fingerprints": {
                    "action_schema": fork.SOURCE_ACTION_SCHEMA,
                    "replay_sha256": replay_sha,
                },
            }
        ),
        encoding="utf-8",
    )
    for index in (308, 352, 368, 369):
        episode = source_session / f"episode_{index:06d}"
        episode.mkdir(parents=True)
    (source_state / "online_state.json").write_text(
        json.dumps(
            {
                "session_root": str(source_session.resolve()),
                "latest_checkpoint": str(checkpoint.resolve()),
                "latest_replay": str(replay.resolve()),
                "latest_replay_sha256": replay_sha,
                "trained_episode_ids": ids.tolist(),
            }
        ),
        encoding="utf-8",
    )
    target = tmp_path / "target"
    args = argparse.Namespace(
        source_session_root=source_session,
        source_state_root=source_state,
        source_checkpoint=checkpoint,
        target_session_root=target,
        target_state_dir=".online_rlt_persistent_v2",
        min_new_persistent_episodes=30,
        expected_source_latest_episode="episode_000369",
        expected_source_checkpoint_step=12627,
        create=False,
    )
    return args


def test_fork_preflight_is_read_only(tmp_path):
    args = _fixture(tmp_path)
    report = fork.preflight(args)
    assert report["preflight_passed"] is True
    assert report["created"] is False
    assert report["target"]["first_episode_id"] == "episode_000370"
    assert report["migration"]["legacy_replay_training_rows"] == 0
    assert report["migration"]["explicitly_excluded_ep368_ep369"] == [
        "episode_000368",
        "episode_000369",
    ]
    assert not args.target_session_root.exists()


def test_fork_create_isolated_state_and_readonly_provenance(tmp_path):
    args = _fixture(tmp_path)
    args.create = True
    report = fork.create_lineage(fork.preflight(args), args)
    state_root = args.target_session_root / args.target_state_dir
    state = json.loads(
        (state_root / "online_state.json").read_text(encoding="utf-8")
    )
    assert report["created"] is True
    assert state["latest_checkpoint"] is None
    assert state["latest_replay"] is None
    assert state["trained_episode_ids"] == []
    assert "frozen_base_replay" not in state
    assert state["replay_training_policy"] == "persistent_only_no_legacy_merge"
    assert state["min_new_persistent_committed_episodes"] == 30
    assert state["episode_index_floor"] == 370
    provenance_replay = Path(state["legacy_source_replay"])
    assert provenance_replay.is_file()
    assert _sha256(provenance_replay) == state["legacy_source_replay_sha256"]
    validation = validate_lineage.validate(
        args.target_session_root,
        state_root,
        state_root / "config.env",
    )
    assert validation["valid"] is True
    assert validation["legacy_replay_training_rows"] == 0


def test_fork_rejects_warmup_weaker_than_30(tmp_path):
    args = _fixture(tmp_path)
    args.min_new_persistent_episodes = 29
    with pytest.raises(ValueError, match="cannot be weakened"):
        fork.preflight(args)


def test_fork_rejects_stale_latest_episode_guard(tmp_path):
    args = _fixture(tmp_path)
    args.expected_source_latest_episode = "episode_000368"
    with pytest.raises(ValueError, match="stale source latest episode"):
        fork.preflight(args)
