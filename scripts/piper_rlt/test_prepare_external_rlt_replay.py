from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from openpi.rlt.real.config import Source
from openpi.rlt.real.replay import RealStepRecord

TOOLS_DIR = Path(__file__).resolve().parent / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import prepare_external_rlt_replay as replay_tool  # noqa: E402
from prepare_external_rlt_replay import _apply_split_registry  # noqa: E402
from prepare_external_rlt_replay import main  # noqa: E402


def _step(index: int) -> dict:
    return {
        "episode_id": "ep_cli",
        "t": index,
        "global_image": f"camera_global/{index:06d}.jpg",
        "wrist_image": f"camera_wrist/{index:06d}.jpg",
        "z_rl": [float(index), float(index + 1), float(index + 2), float(index + 3)],
        "state": [0.0] * 7,
        "a_ref": [[0.1] * 7 for _ in range(10)],
        "a_exec": [0.2] * 7,
        "a_human": [0.2] * 7,
        "a_actor": None,
        "source": Source.HUMAN_PIKA,
        "reward": 1.0 if index == 9 else 0.0,
        "done": index == 9,
        "phase_probability": 0.8,
        "gate_active": True,
        "timestamp_ns": 1000 + index,
    }


def _write_episode(root: Path) -> Path:
    for subdir in ["camera_global", "camera_wrist"]:
        (root / subdir).mkdir(parents=True)
        for index in range(10):
            (root / subdir / f"{index:06d}.jpg").write_bytes(b"image")
    episode_path = root / "episode.jsonl"
    episode_path.write_text("\n".join(json.dumps(_step(index)) for index in range(10)) + "\n", encoding="utf-8")
    return episode_path


def test_prepare_external_rlt_replay_cli_writes_npz_and_manifest(tmp_path: Path) -> None:
    episode_path = _write_episode(tmp_path)
    output = tmp_path / "replay_out"
    episode_manifest = tmp_path / "enrichment.manifest.json"
    episode_manifest.write_text(
        json.dumps({"complete": True, "episodes": {"accepted": [str(episode_path)]}}),
        encoding="utf-8",
    )

    report = main(
        [
            "--episode-manifest",
            str(episode_manifest),
            "--dataset-root",
            str(tmp_path),
            "--output",
            str(output),
            "--chunk-length",
            "10",
            "--stride",
            "2",
            "--n-step",
            "10",
            "--allow-logged-reference",
            "--allow-logged-phase",
        ]
    )

    assert report["input_episodes"] == 1
    assert report["transitions"] == 4
    assert (output / "replay.npz").is_file()
    assert (output / "manifest.json").is_file()

    replay = np.load(output / "replay.npz", allow_pickle=False)
    assert replay["a_exec"].shape == (4, 10, 7)
    assert replay["source"].tolist() == [Source.HUMAN_PIKA] * 4

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_episode_files"] == [str(episode_path)]
    assert manifest["chunking"]["chunk_length"] == 10
    assert manifest["chunking"]["stride"] == 2
    assert manifest["coordinate_contract"]["training_action"]["gripper_6"] == "absolute_command"


def test_prepare_external_replay_uses_cache_only_after_gate_entry(tmp_path: Path) -> None:
    episode_path = _write_episode(tmp_path)
    output = tmp_path / "recomputed"
    cache_path = tmp_path / "cache.jsonl"
    rows = []
    for index in range(10):
        row = {
            "episode_id": tmp_path.name,
            "t": index,
            "phase_probability": 0.9,
        }
        if index >= 2:
            row.update(
                {
                    "a_ref_absolute": [[0.6] * 6 + [0.07] for _ in range(10)],
                    "z_rl": [float(index), float(index + 1)],
                }
            )
        rows.append(row)
    cache_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    report = main(
        [
            "--episode-jsonl",
            str(episode_path),
            "--dataset-root",
            str(tmp_path.parent),
            "--output",
            str(output),
            "--enrichment-cache",
            str(cache_path),
            "--base-fingerprint",
            "base-sha",
            "--token-fingerprint",
            "token-sha",
            "--phase-fingerprint",
            "phase-sha",
        ]
    )

    replay = np.load(output / "replay.npz", allow_pickle=False)
    assert report["provenance"]["reference_recomputed"] is True
    assert report["provenance"]["base_fingerprint"] == "base-sha"
    np.testing.assert_allclose(replay["a_ref_absolute"][0, :, :6], 0.6, atol=1e-6)
    np.testing.assert_allclose(replay["a_ref_absolute"][0, :, 6], 0.07, atol=1e-6)
    np.testing.assert_allclose(replay["a_ref_original_absolute"][0], 0.1, atol=1e-6)
    np.testing.assert_allclose(replay["z_rl"][0], [2.0, 3.0])


def test_online_cache_refreshes_token_without_replacing_logged_reference(tmp_path: Path) -> None:
    episode_path = _write_episode(tmp_path)
    output = tmp_path / "fresh_token_exact_reference"
    cache_path = tmp_path / "cache.jsonl"
    rows = []
    for index in range(10):
        rows.append(
            {
                "episode_id": tmp_path.name,
                "t": index,
                "phase_probability": 1.0,
                "a_ref_absolute": [[0.9] * 7 for _ in range(10)],
                "z_rl": [100.0 + index, 200.0 + index],
            }
        )
    cache_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    report = main(
        [
            "--episode-jsonl",
            str(episode_path),
            "--dataset-root",
            str(tmp_path.parent),
            "--output",
            str(output),
            "--enrichment-cache",
            str(cache_path),
            "--preserve-logged-reference",
            "--base-fingerprint",
            "base-sha",
            "--token-fingerprint",
            "token-sha",
            "--phase-fingerprint",
            "phase-sha",
            "--phase-enter-frames",
            "1",
        ]
    )

    with np.load(output / "replay.npz", allow_pickle=False) as replay:
        np.testing.assert_allclose(replay["a_ref_absolute"], 0.1)
        np.testing.assert_allclose(replay["z_rl"][0], [100.0, 200.0])
    assert report["provenance"]["reference_recomputed"] is False
    assert report["provenance"]["token_recomputed"] is True
    assert report["provenance"]["logged_reference_preserved_with_fresh_token"] is True


def test_online_split_registry_never_moves_heldout_episodes(tmp_path: Path) -> None:
    registry = tmp_path / "split_registry.json"
    first, _ = _apply_split_registry(
        {"train": ["train_a"], "validation": ["val_a"], "test": []},
        episode_ids=["train_a", "val_a"],
        registry_path=registry,
        seed="seed",
    )
    second, strategy = _apply_split_registry(
        {"train": ["val_a"], "validation": ["train_a", "new_a"], "test": []},
        episode_ids=["train_a", "val_a", "new_a"],
        registry_path=registry,
        seed="seed",
    )

    assert first["validation"] == ["val_a"]
    assert second["validation"] == ["val_a"]
    assert second["train"] == ["new_a", "train_a"]
    assert "persistent_registry" in strategy


def test_persistent_v2_bootstraps_only_across_real_contiguous_c10(
    tmp_path: Path, monkeypatch
) -> None:
    records = [
        RealStepRecord(
            episode_id="persistent_ep",
            t=index,
            z_rl=np.zeros(4, dtype=np.float32),
            state=np.zeros(7, dtype=np.float32),
            a_ref=np.zeros((10, 7), dtype=np.float32),
            a_exec=np.zeros(7, dtype=np.float32),
            a_human=None,
            a_actor=np.zeros(7, dtype=np.float32),
            source=Source.RLT,
            reward=1.0 if index == 19 else 0.0,
            done=index == 19,
            gate_active=True,
        )
        for index in range(20)
    ]
    chunk_calls = []

    def fake_enrich_episode(input_records, **_kwargs):
        return SimpleNamespace(
            records=list(input_records),
            statistics={"gate_enter_t": 0},
        )

    def fake_chunk_real_episode(input_records, **kwargs):
        snapshot = list(input_records)
        chunk_calls.append((snapshot, kwargs))
        assert len(snapshot) == 20
        assert not any(record.done for record in snapshot[:-1])
        assert snapshot[-1].done
        return [
            SimpleNamespace(t=0, done=False, discount=kwargs["gamma"] ** 10),
            SimpleNamespace(t=10, done=True, discount=0.0),
        ]

    monkeypatch.setattr(replay_tool, "enrich_episode", fake_enrich_episode)
    monkeypatch.setattr(replay_tool, "chunk_real_episode", fake_chunk_real_episode)
    audit = {
        "complete_c10_row_indices_by_plan": {
            "plan_0": list(range(10)),
            "plan_1": list(range(10, 20)),
        },
        "complete_human_c10_chunks": 0,
        "complete_actor_c10_chunks": 2,
    }

    transitions, statistics = replay_tool._persistent_v2_transitions(
        records,
        audit=audit,
        episode_root=tmp_path,
        reference_provider=None,
        phase_provider=None,
        allow_logged_reference=True,
        allow_logged_phase=True,
        preserve_logged_reference=False,
        phase_enter_threshold=0.5,
        phase_enter_frames=3,
        gamma=0.99,
    )

    assert len(chunk_calls) == 1
    assert len(transitions) == 2
    assert transitions[0].done is False
    assert transitions[0].discount == 0.99**10
    assert transitions[1].done is True
    assert transitions[1].discount == 0.0
    assert statistics["persistent_v2_transition_count"] == 2
    assert statistics["persistent_v2_dropped_no_bootstrap_c10"] == 0


def test_persistent_v2_drops_whole_pre_gate_c10_without_requesting_missing_token(
    tmp_path: Path, monkeypatch
) -> None:
    records = [
        RealStepRecord(
            episode_id="persistent_gate_ep",
            t=index,
            z_rl=np.zeros(4, dtype=np.float32),
            state=np.zeros(7, dtype=np.float32),
            a_ref=np.zeros((10, 7), dtype=np.float32),
            a_exec=np.zeros(7, dtype=np.float32),
            a_human=np.zeros(7, dtype=np.float32),
            a_actor=None,
            source=Source.HUMAN_PIKA,
            reward=1.0 if index == 29 else 0.0,
            done=index == 29,
            gate_active=index >= 12,
        )
        for index in range(30)
    ]
    provider_calls: list[int] = []

    def reference_provider(record, _episode_root):
        provider_calls.append(record.t)
        if record.t < 12:
            raise AssertionError("pre-gate cache row must not require a_ref/z_rl")
        return replay_tool.ReferenceTokenValue(
            a_ref=np.zeros((10, 7), dtype=np.float32),
            z_rl=np.ones(4, dtype=np.float32),
            action_space=replay_tool.ABSOLUTE_ACTION_SPACE,
        )

    chunk_calls = []

    def fake_chunk_real_episode(input_records, **_kwargs):
        snapshot = list(input_records)
        chunk_calls.append(snapshot)
        assert [record.t for record in snapshot] == list(range(20, 30))
        assert snapshot[-1].done
        return [SimpleNamespace(t=20, done=True, discount=0.0)]

    monkeypatch.setattr(replay_tool, "chunk_real_episode", fake_chunk_real_episode)
    audit = {
        "complete_c10_row_indices_by_plan": {
            "pre_gate": list(range(10)),
            "partial_gate": list(range(10, 20)),
            "active": list(range(20, 30)),
        },
        "complete_human_c10_chunks": 3,
        "complete_actor_c10_chunks": 0,
    }

    transitions, statistics = replay_tool._persistent_v2_transitions(
        records,
        audit=audit,
        episode_root=tmp_path,
        reference_provider=reference_provider,
        phase_provider=lambda record, _root: 0.1 if record.t < 10 else 0.9,
        allow_logged_reference=False,
        allow_logged_phase=False,
        preserve_logged_reference=True,
        phase_enter_threshold=0.5,
        phase_enter_frames=3,
        gamma=0.99,
    )

    assert provider_calls == list(range(12, 30))
    assert len(chunk_calls) == 1
    assert len(transitions) == 1
    assert statistics["persistent_v2_transition_count"] == 1
    assert statistics["persistent_v2_dropped_phase_gate_c10"] == 2
    reasons = {
        item["plan_id"]: item["drop_reason"]
        for item in statistics["persistent_v2_plans"]
    }
    assert reasons == {
        "pre_gate": "phase_gate_inactive_complete_c10",
        "partial_gate": "phase_gate_partial_complete_c10",
        "active": None,
    }


def test_persistent_v2_all_pre_gate_plans_return_no_transitions_safely(
    tmp_path: Path, monkeypatch
) -> None:
    records = [
        RealStepRecord(
            episode_id="persistent_never_active",
            t=index,
            z_rl=np.zeros(4, dtype=np.float32),
            state=np.zeros(7, dtype=np.float32),
            a_ref=np.zeros((10, 7), dtype=np.float32),
            a_exec=np.zeros(7, dtype=np.float32),
            a_human=np.zeros(7, dtype=np.float32),
            a_actor=None,
            source=Source.HUMAN_PIKA,
            reward=1.0 if index == 9 else 0.0,
            done=index == 9,
        )
        for index in range(10)
    ]

    def reference_provider(*_args):
        raise AssertionError("a never-active phase must not request a policy token")

    def chunk_real_episode(*_args, **_kwargs):
        raise AssertionError("a never-active phase must not be chunked")

    monkeypatch.setattr(replay_tool, "chunk_real_episode", chunk_real_episode)
    transitions, statistics = replay_tool._persistent_v2_transitions(
        records,
        audit={
            "complete_c10_row_indices_by_plan": {
                "pre_gate": list(range(10)),
            },
            "complete_human_c10_chunks": 1,
            "complete_actor_c10_chunks": 0,
        },
        episode_root=tmp_path,
        reference_provider=reference_provider,
        phase_provider=lambda _record, _root: 0.1,
        allow_logged_reference=False,
        allow_logged_phase=False,
        preserve_logged_reference=True,
        phase_enter_threshold=0.5,
        phase_enter_frames=3,
        gamma=0.99,
    )

    assert transitions == []
    assert statistics["valid_action_rows"] == 0
    assert statistics["persistent_v2_transition_count"] == 0
    assert statistics["persistent_v2_dropped_phase_gate_c10"] == 1
    assert statistics["terminal_reward"] == 1.0
