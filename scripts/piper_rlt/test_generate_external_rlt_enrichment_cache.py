from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

from openpi.rlt.real.replay_enrichment import CachedEnrichmentProvider


SCRIPT = Path(__file__).parent / "tools" / "generate_external_rlt_enrichment_cache.py"
SPEC = importlib.util.spec_from_file_location("generate_external_rlt_enrichment_cache", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
PREPARE_SCRIPT = Path(__file__).parent / "tools" / "prepare_external_rlt_replay.py"
PREPARE_SPEC = importlib.util.spec_from_file_location("prepare_external_rlt_replay_for_cache_test", PREPARE_SCRIPT)
assert PREPARE_SPEC is not None and PREPARE_SPEC.loader is not None
PREPARE_MODULE = importlib.util.module_from_spec(PREPARE_SPEC)
sys.modules[PREPARE_SPEC.name] = PREPARE_MODULE
PREPARE_SPEC.loader.exec_module(PREPARE_MODULE)


class FakePhase:
    def __init__(self, probabilities: list[float]):
        self.probabilities = probabilities
        self.calls = 0

    def predict_probability(self, images):
        assert set(images) == {"camera1", "camera2"}
        value = self.probabilities[self.calls]
        self.calls += 1
        return value


class FakePolicy:
    def __init__(self):
        self.calls: list[dict] = []

    def get_server_metadata(self):
        return {
            "rlt_actor_controls_robot": False,
            "base_config": "test_config",
            "base_checkpoint": "/models/full20k",
            "token_checkpoint": "/models/token",
        }

    def infer(self, observation):
        self.calls.append(observation)
        index = len(self.calls)
        return {
            "actions": np.full((50, 7), index, dtype=np.float32),
            "z_rl": np.arange(4, dtype=np.float32) + index,
            "rlt_shadow": {"token_status": "ok", "actor_controls_robot": False},
        }


def _write_episode(root: Path) -> Path:
    episode = root / "session_a" / "episode_001"
    image_dir = episode / "images"
    image_dir.mkdir(parents=True)
    rows = []
    for t in range(5):
        global_rel = f"images/global_{t}.png"
        wrist_rel = f"images/wrist_{t}.png"
        Image.fromarray(np.full((12, 16, 3), t, dtype=np.uint8)).save(episode / global_rel)
        Image.fromarray(np.full((12, 16, 3), t + 20, dtype=np.uint8)).save(episode / wrist_rel)
        terminal = t == 4
        rows.append(
            {
                "episode_id": "legacy_id",
                "t": t,
                "z_rl": [0.0],
                "state": [0.1 * t] * 7,
                "a_ref": [[0.0] * 7 for _ in range(10)],
                "a_exec": [0.2 * t] * 7,
                "a_human": None,
                "a_actor": None,
                "source": "stop" if terminal else "pi05",
                "reward": 1.0 if terminal else 0.0,
                "done": terminal,
                "global_image": global_rel,
                "wrist_image": wrist_rel,
                "policy_metadata": {"replay_include": not terminal},
            }
        )
    path = episode / "episode.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (episode / "report.json").write_text(
        json.dumps({"outcome": "episode_done", "terminal_reward": 1.0}), encoding="utf-8"
    )
    return path


def _config(root: Path, output: Path):
    return MODULE.CacheGenerationConfig(
        output=output,
        dataset_root=root,
        phase_enter_threshold=0.5,
        phase_enter_frames=2,
        expected_z_dim=4,
        base_fingerprint="base-full20k-sha",
        token_fingerprint="token-sha",
        phase_fingerprint="phase-sha",
        expected_base_config="test_config",
        expected_base_checkpoint="/models/full20k",
        expected_token_checkpoint="/models/token",
    )


def test_cache_all_phase_rows_but_policy_only_latched_valid_rows_and_resume(tmp_path: Path) -> None:
    episode = _write_episode(tmp_path)
    output = tmp_path / "cache.jsonl"
    phase = FakePhase([0.1, 0.8, 0.9, 0.2, 0.2])
    policy = FakePolicy()
    report = MODULE.generate_enrichment_cache(
        [episode], config=_config(tmp_path, output), phase_predictor=phase, policy_client_factory=lambda: policy
    )

    assert phase.calls == 5
    assert len(policy.calls) == 2  # t=2 enters the latch; t=4 is terminal/stop and invalid.
    assert report["counts"]["rows"] == 5
    assert report["counts"]["policy_rows"] == 2
    assert report["safety"]["commands_published"] is False
    assert not Path(str(output) + ".partial").exists()

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [(row["episode_id"], row["t"]) for row in rows] == [
        ("session_a/episode_001", index) for index in range(5)
    ]
    assert ["a_ref_absolute" in row for row in rows] == [False, False, True, True, False]
    assert np.asarray(rows[2]["a_ref_absolute"]).shape == (10, 7)
    assert np.asarray(rows[2]["z_rl"]).shape == (4,)
    cache = CachedEnrichmentProvider.from_jsonl(output)
    replay_report = PREPARE_MODULE.prepare_external_replay(
        [episode],
        output_dir=tmp_path / "replay",
        dataset_root=tmp_path,
        reference_provider=cache,
        phase_provider=cache.phase_probability,
        phase_enter_threshold=0.5,
        phase_enter_frames=2,
        base_fingerprint="base-full20k-sha",
        token_fingerprint="token-sha",
        phase_fingerprint="phase-sha",
    )
    assert replay_report["statistics"]["transitions"] == 1
    replay = np.load(tmp_path / "replay" / "replay.npz")
    assert replay["z_rl"].shape == (1, 4)
    assert replay["a_ref"].shape == (1, 10, 7)

    # A complete rerun is a true cache hit: no ResNet or policy inference.
    second_phase = FakePhase([])
    second_policy = FakePolicy()
    second = MODULE.generate_enrichment_cache(
        [episode],
        config=_config(tmp_path, output),
        phase_predictor=second_phase,
        policy_client_factory=lambda: second_policy,
    )
    assert second_phase.calls == 0
    assert len(second_policy.calls) == 0
    assert second["counts"]["phase_cache_hits"] == 5
    assert second["counts"]["policy_cache_hits"] == 2


def test_cache_rejects_changed_observation_on_resume(tmp_path: Path) -> None:
    episode = _write_episode(tmp_path)
    output = tmp_path / "cache.jsonl"
    MODULE.generate_enrichment_cache(
        [episode],
        config=_config(tmp_path, output),
        phase_predictor=FakePhase([0.1, 0.8, 0.9, 0.2, 0.2]),
        policy_client_factory=FakePolicy,
    )
    Image.fromarray(np.full((12, 16, 3), 255, dtype=np.uint8)).save(episode.parent / "images/global_0.png")
    with pytest.raises(ValueError, match="stale/incompatible cache entry"):
        MODULE.generate_enrichment_cache(
            [episode],
            config=_config(tmp_path, output),
            phase_predictor=FakePhase([]),
            policy_client_factory=FakePolicy,
        )


def test_incremental_cache_skips_immutable_completed_episode(
    tmp_path: Path,
) -> None:
    episode = _write_episode(tmp_path)
    output = tmp_path / "cache.jsonl"
    config = dataclasses.replace(
        _config(tmp_path, output),
        incremental=True,
    )
    first = MODULE.generate_enrichment_cache(
        [episode],
        config=config,
        phase_predictor=FakePhase([0.1, 0.8, 0.9, 0.2, 0.2]),
        policy_client_factory=FakePolicy,
    )
    original = output.read_bytes()

    second = MODULE.generate_enrichment_cache(
        [episode],
        config=config,
        phase_predictor=FakePhase([]),
        policy_client_factory=FakePolicy,
    )

    assert output.read_bytes() == original
    assert second["counts"]["episodes_reused"] == 1
    assert second["counts"]["episodes_processed"] == 0
    assert second["counts"]["rows"] == first["counts"]["rows"]
    assert second["counts"]["policy_requests"] == 0


def test_missing_report_is_recorded_when_skipping_invalid(tmp_path: Path) -> None:
    episode = _write_episode(tmp_path)
    (episode.parent / "report.json").unlink()
    config = dataclasses.replace(_config(tmp_path, tmp_path / "cache.jsonl"), skip_invalid_episodes=True)
    report = MODULE.generate_enrichment_cache(
        [episode], config=config, phase_predictor=FakePhase([]), policy_client_factory=FakePolicy
    )
    assert report["counts"]["rows"] == 0
    assert report["episodes"]["accepted"] == []
    assert "terminal report does not exist" in report["episodes"]["skipped"][0]["reason"]


def test_quarantined_episode_is_never_enriched_for_training(tmp_path: Path) -> None:
    episode = _write_episode(tmp_path)
    (episode.parent / "report.json").write_text(
        json.dumps(
            {
                "outcome": "episode_done",
                "terminal_reward": 0.0,
                "exclude_from_training": True,
                "exclusion_reason": "controller fault",
            }
        ),
        encoding="utf-8",
    )
    config = dataclasses.replace(_config(tmp_path, tmp_path / "cache.jsonl"), skip_invalid_episodes=True)
    report = MODULE.generate_enrichment_cache(
        [episode], config=config, phase_predictor=FakePhase([]), policy_client_factory=FakePolicy
    )

    assert report["counts"]["rows"] == 0
    assert report["episodes"]["accepted"] == []
    assert "quarantined from training" in report["episodes"]["skipped"][0]["reason"]
