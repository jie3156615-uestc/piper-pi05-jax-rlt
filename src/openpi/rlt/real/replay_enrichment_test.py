from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openpi.rlt.real.config import Source
from openpi.rlt.real.replay import RealStepRecord
from openpi.rlt.real.replay import chunk_real_episode
from openpi.rlt.real.replay_enrichment import DELTA_ACTION_SPACE
from openpi.rlt.real.replay_enrichment import ReferenceTokenValue
from openpi.rlt.real.replay_enrichment import assign_episode_splits
from openpi.rlt.real.replay_enrichment import assign_stratified_episode_splits
from openpi.rlt.real.replay_enrichment import enrich_episode


def _record(t: int, **overrides) -> RealStepRecord:
    state = np.asarray([float(t)] * 6 + [0.03], dtype=np.float32)
    values = dict(
        episode_id="session/episode_0",
        t=t,
        z_rl=np.asarray([0.0], dtype=np.float32),
        state=state,
        a_ref=np.full((10, 7), 90.0 + t, dtype=np.float32),
        a_exec=np.asarray([float(t) + 0.5] * 6 + [0.04], dtype=np.float32),
        a_human=None,
        a_actor=None,
        source=Source.PI05,
        reward=0.0,
        done=False,
        replay_include=True,
    )
    values.update(overrides)
    return RealStepRecord(**values)


def _reference_provider(record: RealStepRecord, episode_root: Path) -> ReferenceTokenValue:
    del episode_root
    # First six values are deltas and must be converted back to absolute before
    # chunk-level training-coordinate conversion.
    reference = np.full((10, 7), 0.25, dtype=np.float32)
    reference[:, 6] = 0.05
    return ReferenceTokenValue(
        a_ref=reference,
        z_rl=np.asarray([record.t, record.t + 1], dtype=np.float32),
        action_space=DELTA_ACTION_SPACE,
    )


def _phase_provider(record: RealStepRecord, episode_root: Path) -> float:
    del episode_root
    return 0.9 if record.t >= 3 else 0.1


def test_enrichment_filters_invalid_rows_migrates_terminal_and_latches() -> None:
    records = [_record(0, source=Source.SAFETY_BLOCK, replay_include=False)]
    records.extend(_record(t) for t in range(1, 6))
    records.append(_record(6, source=Source.SAFETY_BLOCK, replay_include=False))
    records.extend(
        _record(
            t,
            source=Source.HUMAN_PIKA if t in {8, 9} else Source.PI05,
            a_human=(np.asarray([float(t) + 0.4] * 6 + [0.02], dtype=np.float32) if t in {8, 9} else None),
        )
        for t in range(7, 14)
    )
    records.append(_record(14, source=Source.STOP, replay_include=False, reward=1.0, done=True))

    enriched = enrich_episode(
        records,
        episode_root=".",
        reference_provider=_reference_provider,
        phase_provider=_phase_provider,
        enter_threshold=0.5,
        enter_frames=3,
    )

    assert [[record.t for record in segment] for segment in enriched.segments] == [[5], list(range(7, 14))]
    assert enriched.records[-1].done is True
    assert enriched.records[-1].reward == 1.0
    assert enriched.records[-2].done is False
    assert enriched.statistics["filtered_by_reason"] == {
        "gate_inactive": 4,
        "source:safety_block": 2,
        "source:stop": 1,
    }
    assert enriched.statistics["gate_enter_t"] == 5
    assert all(record.gate_active for record in enriched.records if record.t >= 5)
    np.testing.assert_allclose(enriched.records[0].z_rl, [5, 6])
    # Logged reference remains available independently of recomputed reference.
    np.testing.assert_allclose(enriched.records[0].a_ref_original, 95.0)


def test_chunk_uses_training_coordinates_masks_and_does_not_bridge_gap() -> None:
    records = [_record(t) for t in range(5)]
    records.extend(
        _record(
            t,
            source=Source.HUMAN_PIKA if t == 7 else Source.PI05,
            a_human=(np.asarray([7.4] * 6 + [0.02], dtype=np.float32) if t == 7 else None),
            done=t == 13,
            reward=1.0 if t == 13 else 0.0,
            a_ref_original=np.full((10, 7), 100.0 + t, dtype=np.float32),
        )
        for t in range(7, 14)
    )
    transitions = chunk_real_episode(records, chunk_length=10, stride=2, n_step=10, gamma=0.99)

    # The nonterminal five-row prefix is too short; every emitted chunk starts
    # in the post-gap terminal segment.
    assert [transition.t for transition in transitions] == [7, 9, 11, 13]
    first = transitions[0]
    np.testing.assert_allclose(first.a_exec[0, :6], 0.5)
    np.testing.assert_allclose(first.a_exec[0, 6], 0.04)
    np.testing.assert_allclose(first.a_exec_absolute[0, :6], 7.5)
    assert first.source_chunk.tolist() == [Source.HUMAN_PIKA] + [Source.PI05] * 6 + ["pad"] * 3
    assert first.human_mask.tolist() == [True] + [False] * 9
    assert first.actor_mask.tolist() == [False] * 10
    np.testing.assert_allclose(first.a_human[1:], 0.0)
    np.testing.assert_allclose(first.a_actor, 0.0)
    assert first.step_mask.tolist() == [True] * 7 + [False] * 3
    np.testing.assert_allclose(first.a_ref_original_absolute, 107.0)


def test_enrichment_requires_frozen_providers_by_default() -> None:
    records = [_record(0), _record(1, source=Source.STOP, replay_include=False, done=True)]
    with pytest.raises(ValueError, match="frozen-base"):
        enrich_episode(records, episode_root=".", reference_provider=None, phase_provider=_phase_provider)
    with pytest.raises(ValueError, match="frozen phase"):
        enrich_episode(records, episode_root=".", reference_provider=_reference_provider, phase_provider=None)


def test_replay_include_false_and_reward_wait_are_strictly_excluded() -> None:
    records = [
        _record(0, replay_include=False),
        _record(1, waiting_for_reward=True),
        _record(2),
        _record(3, source=Source.STOP, replay_include=False, done=True),
    ]
    enriched = enrich_episode(
        records,
        episode_root=".",
        reference_provider=_reference_provider,
        phase_provider=lambda record, root: 1.0,
        enter_frames=1,
    )
    assert [record.t for record in enriched.records] == [2]
    assert enriched.records[0].done is True
    assert enriched.statistics["filtered_by_reason"] == {
        "replay_include_false": 1,
        "source:stop": 1,
        "waiting_for_reward": 1,
    }


def test_episode_split_is_deterministic_and_disjoint() -> None:
    episode_ids = [f"session/episode_{index}" for index in range(20)]
    first = assign_episode_splits(episode_ids, validation_fraction=0.2, test_fraction=0.1, seed="seed")
    second = assign_episode_splits(list(reversed(episode_ids)), validation_fraction=0.2, test_fraction=0.1, seed="seed")
    assert first == second
    assert len(first["train"]) == 14
    assert len(first["validation"]) == 4
    assert len(first["test"]) == 2
    assert set(first["train"]).isdisjoint(first["validation"])
    assert set().union(*map(set, first.values())) == set(episode_ids)


def test_episode_split_stratifies_sparse_terminal_reward() -> None:
    labels = {
        **{f"success_{index}": "success" for index in range(6)},
        **{f"failure_{index}": "failure" for index in range(10)},
    }
    split = assign_stratified_episode_splits(labels, validation_fraction=0.15, seed="seed")

    validation_labels = {labels[episode_id] for episode_id in split["validation"]}
    assert validation_labels == {"success", "failure"}
    assert set(split["train"]).isdisjoint(split["validation"])
    assert set().union(*map(set, split.values())) == set(labels)


def test_episode_without_gate_entry_is_reported_and_skipped() -> None:
    records = [
        _record(0),
        _record(1, source=Source.STOP, replay_include=False, done=True),
    ]
    enriched = enrich_episode(
        records,
        episode_root=".",
        reference_provider=_reference_provider,
        phase_provider=lambda record, root: 0.1,
    )
    assert enriched.records == []
    assert enriched.segments == []
    assert enriched.statistics["skipped_reason"] == "no_gate_active_valid_action"


def test_nonterminal_chunk_requires_real_n_step_bootstrap_state() -> None:
    exactly_c = [_record(t) for t in range(10)]
    c_plus_one = [_record(t) for t in range(11)]

    assert chunk_real_episode(exactly_c, chunk_length=10, stride=2, n_step=10, gamma=0.99) == []
    transitions = chunk_real_episode(c_plus_one, chunk_length=10, stride=2, n_step=10, gamma=0.99)

    assert [transition.t for transition in transitions] == [0]
    assert transitions[0].done is False
    assert transitions[0].next_state[0] == 10.0
