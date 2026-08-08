from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from openpi.rlt.real.config import Source
from openpi.rlt.real.config import PERSISTENT_ACTOR_EXECUTION_PROFILE
from openpi.rlt.real.external_episode import ExternalEpisodeContract
from openpi.rlt.real.external_episode import ExternalEpisodeError
from openpi.rlt.real.external_episode import load_episode_jsonl


def _step(index: int, **overrides):
    step = {
        "episode_id": "ep_test",
        "t": index,
        "global_image": f"camera_global/{index:06d}.jpg",
        "wrist_image": f"camera_wrist/{index:06d}.jpg",
        "z_rl": [float(index), 0.1, 0.2, 0.3],
        "state": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "a_ref": [[0.01 * index] * 7 for _ in range(10)],
        "a_exec": [0.02 * index] * 7,
        "a_human": None,
        "a_actor": None,
        "source": Source.HUMAN_PIKA,
        "reward": 0.0,
        "done": False,
        "phase_probability": 0.25,
        "gate_active": False,
        "timestamp_ns": 1_780_000_000_000_000_000 + index,
    }
    step.update(overrides)
    return step


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_images(root: Path, count: int) -> None:
    for subdir in ["camera_global", "camera_wrist"]:
        (root / subdir).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            (root / subdir / f"{index:06d}.jpg").write_bytes(b"image")


def test_load_episode_jsonl_preserves_camera_refs_and_timestamp(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode.jsonl"
    _write_images(tmp_path, 3)
    _write_jsonl(episode_path, [_step(0), _step(1), _step(2, reward=1.0, done=True)])

    records = load_episode_jsonl(episode_path, dataset_root=tmp_path)

    assert len(records) == 3
    assert records[0].episode_id == "ep_test"
    assert records[0].global_image == "camera_global/000000.jpg"
    assert records[0].wrist_image == "camera_wrist/000000.jpg"
    assert records[0].timestamp_ns == 1_780_000_000_000_000_000
    assert records[-1].reward == 1.0
    assert records[-1].done is True
    np.testing.assert_allclose(records[1].a_exec, [0.02] * 7)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows[0].pop("global_image"), "global_image"),
        (lambda rows: rows[1].__setitem__("t", 0), "monotonic"),
        (lambda rows: rows[0].__setitem__("reward", 1.0), "terminal"),
        (lambda rows: rows[0].__setitem__("source", "unknown"), "source"),
        (lambda rows: rows[0].__setitem__("state", [0.0] * 6), "state"),
        (lambda rows: rows[0].__setitem__("a_exec", [0.0] * 6), "a_exec"),
    ],
)
def test_load_episode_jsonl_rejects_invalid_contract(tmp_path: Path, mutate, message: str) -> None:
    episode_path = tmp_path / "episode.jsonl"
    _write_images(tmp_path, 3)
    rows = [_step(0), _step(1), _step(2, reward=0.0, done=True)]
    mutate(rows)
    _write_jsonl(episode_path, rows)

    with pytest.raises(ExternalEpisodeError, match=message):
        load_episode_jsonl(episode_path, dataset_root=tmp_path)


def test_load_episode_jsonl_can_skip_image_existence_check(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode.jsonl"
    _write_jsonl(episode_path, [_step(0), _step(1, reward=0.0, done=True)])

    records = load_episode_jsonl(
        episode_path,
        dataset_root=tmp_path,
        contract=ExternalEpisodeContract(check_image_exists=False),
    )

    assert len(records) == 2


def test_load_episode_jsonl_reads_replay_filter_metadata(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode.jsonl"
    _write_images(tmp_path, 2)
    rows = [
        _step(
            0,
            policy_metadata={"replay_include": False, "mux_reason": "waiting_for_reward_hold"},
            keyboard={"waiting_for_reward": True},
        ),
        _step(1, done=True),
    ]
    _write_jsonl(episode_path, rows)

    records = load_episode_jsonl(episode_path, dataset_root=tmp_path)

    assert records[0].replay_include is False
    assert records[0].waiting_for_reward is True


def test_persistent_session_profile_allows_non_replay_non_committed_rows(
    tmp_path: Path,
) -> None:
    episode_path = tmp_path / "episode.jsonl"
    _write_images(tmp_path, 2)
    rows = [
        _step(
            0,
            source=Source.PI05,
            actor_execution_profile=PERSISTENT_ACTOR_EXECUTION_PROFILE,
            policy_metadata={
                "replay_include": False,
                "actor_execution_profile": PERSISTENT_ACTOR_EXECUTION_PROFILE,
                "actor_execution_committed_this_step": False,
            },
        ),
        _step(
            1,
            source=Source.STOP,
            done=True,
            actor_execution_profile=PERSISTENT_ACTOR_EXECUTION_PROFILE,
            policy_metadata={
                "replay_include": False,
                "actor_execution_profile": PERSISTENT_ACTOR_EXECUTION_PROFILE,
                "actor_execution_committed_this_step": False,
            },
        ),
    ]
    _write_jsonl(episode_path, rows)

    records = load_episode_jsonl(episode_path, dataset_root=tmp_path)

    assert len(records) == 2
    assert records[0].actor_execution_profile == PERSISTENT_ACTOR_EXECUTION_PROFILE
    assert records[0].actor_canonical_decision is None


def test_persistent_committed_replay_row_still_requires_execution_evidence(
    tmp_path: Path,
) -> None:
    episode_path = tmp_path / "episode.jsonl"
    _write_images(tmp_path, 2)
    rows = [
        _step(
            0,
            actor_execution_profile=PERSISTENT_ACTOR_EXECUTION_PROFILE,
            policy_metadata={
                "replay_include": True,
                "actor_execution_profile": PERSISTENT_ACTOR_EXECUTION_PROFILE,
                "actor_execution_committed_this_step": True,
            },
        ),
        _step(1, done=True),
    ]
    _write_jsonl(episode_path, rows)

    with pytest.raises(
        ExternalEpisodeError,
        match="missing required persistent-v2 field: actor_canonical_decision",
    ):
        load_episode_jsonl(episode_path, dataset_root=tmp_path)
