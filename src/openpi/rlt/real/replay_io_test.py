from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from openpi.rlt.real.config import Source
from openpi.rlt.real.replay import RealStepRecord
from openpi.rlt.real.replay import chunk_real_episode
from openpi.rlt.real.replay_io import transitions_to_arrays
from openpi.rlt.real.replay_io import write_replay_npz


def _record(index: int) -> RealStepRecord:
    return RealStepRecord(
        episode_id="ep_io",
        t=index,
        z_rl=np.array([index, index + 1, index + 2, index + 3], dtype=np.float32),
        state=np.full(7, index, dtype=np.float32),
        a_ref=np.full((10, 7), 0.1, dtype=np.float32),
        a_exec=np.full(7, 0.2 + index, dtype=np.float32),
        a_human=np.full(7, 0.3, dtype=np.float32) if index == 0 else None,
        a_actor=np.full((10, 7), 0.4, dtype=np.float32) if index == 0 else None,
        source=Source.HUMAN_PIKA if index == 0 else Source.PI05,
        reward=1.0 if index == 9 else 0.0,
        done=index == 9,
        phase_probability=0.7,
        gate_active=True,
        global_image=f"camera_global/{index:06d}.jpg",
        wrist_image=f"camera_wrist/{index:06d}.jpg",
        timestamp_ns=1000 + index,
    )


def test_transitions_to_arrays_preserves_training_semantics() -> None:
    transitions = chunk_real_episode([_record(index) for index in range(10)], chunk_length=10, stride=10, n_step=10, gamma=0.99)

    arrays = transitions_to_arrays(transitions)

    assert arrays["z_rl"].shape == (1, 4)
    assert arrays["state"].shape == (1, 7)
    assert arrays["a_ref"].shape == (1, 10, 7)
    assert arrays["a_exec"].shape == (1, 10, 7)
    assert arrays["a_exec_absolute"].shape == (1, 10, 7)
    assert arrays["source_chunk"].shape == (1, 10)
    assert arrays["human_mask"].shape == (1, 10)
    assert arrays["actor_mask"].shape == (1, 10)
    assert arrays["step_mask"].shape == (1, 10)
    assert arrays["source"].tolist() == [Source.HUMAN_PIKA]
    assert arrays["episode_id"].tolist() == ["ep_io"]
    np.testing.assert_allclose(arrays["reward"], [0.99**9])
    np.testing.assert_array_equal(arrays["done"], [True])


def test_write_replay_npz_writes_arrays_and_manifest(tmp_path: Path) -> None:
    transitions = chunk_real_episode([_record(index) for index in range(10)], chunk_length=10, stride=10, n_step=10, gamma=0.99)

    report = write_replay_npz(transitions, tmp_path, episode_split_by_id={"ep_io": "train"})

    assert report["transitions"] == 1
    assert report["episodes"] == 1
    replay_path = tmp_path / "replay.npz"
    manifest_path = tmp_path / "manifest.json"
    assert replay_path.is_file()
    assert manifest_path.is_file()

    loaded = np.load(replay_path, allow_pickle=False)
    assert loaded["a_exec"].shape == (1, 10, 7)
    assert loaded["source"].tolist() == [Source.HUMAN_PIKA]
    assert loaded["episode_split"].tolist() == ["train"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == "openpi_real_rlt_replay_npz"
    assert manifest["version"] == 2
    assert manifest["transitions"] == 1
    assert manifest["files"]["replay"] == "replay.npz"
