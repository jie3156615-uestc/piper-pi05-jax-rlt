from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from openpi.rlt.real.config import GRIPPER_RESIDUAL_CLOSE_ASSIST
from openpi.rlt.real.config import HUMAN_EXECUTION_PROFILE
from openpi.rlt.real.config import PERSISTENT_ACTION_SCHEMA_FINGERPRINT
from openpi.rlt.real.config import PERSISTENT_ACTOR_EXECUTION_PROFILE
from openpi.rlt.real.config import PERSISTENT_EXECUTION_FILTER_PROFILE
from openpi.rlt.real.config import (
    PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT,
)
from openpi.rlt.real.config import Source
from openpi.rlt.real.external_episode import ExternalEpisodeContract
from openpi.rlt.real.external_episode import load_episode_jsonl
from openpi.rlt.real.replay import RealStepRecord
from openpi.rlt.real.replay import chunk_real_episode
from openpi.rlt.real.replay_io import transitions_to_arrays


_BLEND = np.asarray(
    [
        0.0,
        0.011532794797541024,
        0.07641111619163743,
        0.2098765432098765,
        0.3966874968246709,
        0.603312503175329,
        0.790123456790123,
        0.9235888838083626,
        0.9884672052024577,
        1.0,
    ],
    dtype=np.float64,
)


def _close_records(
    *,
    episode_id: str = "close_ep",
    start_t: int = 0,
    terminal: bool = True,
) -> list[RealStepRecord]:
    tau = 0.05
    dt = np.full(10, 1.0 / 30.0, dtype=np.float64)
    alpha = 1.0 - np.exp(-dt / tau)
    reference = np.zeros((10, 7), dtype=np.float64)
    reference[:, 6] = 0.02
    base = reference.copy()
    carry = np.zeros(7, dtype=np.float64)
    previous = np.zeros(7, dtype=np.float64)
    decision = np.zeros(7, dtype=np.float64)
    decision[6] = -0.002
    planned = np.zeros((10, 7), dtype=np.float64)
    planned[:, 6] = _BLEND * decision[6]
    actual_residual = planned.copy()
    actual = base + actual_residual
    boundary = np.zeros(7, dtype=np.float64)
    boundary[6] = 0.02
    d1 = np.diff(np.r_[carry[6], actual_residual[:, 6]])
    d2 = d1 - np.r_[carry[6] - previous[6], d1[:-1]]
    certificate = {
        "actor_filtered_actual_gripper_residual_max": np.abs(
            actual_residual[:, 6]
        ),
        "actor_filtered_actual_gripper_residual_d1_max_m": np.abs(d1),
        "actor_filtered_actual_gripper_residual_d2_max_m": np.abs(d2),
        "actor_filtered_actual_gripper_boundary_jump_max_m": np.abs(d1),
    }
    records: list[RealStepRecord] = []
    for offset in range(10):
        done = bool(terminal and offset == 9)
        records.append(
            RealStepRecord(
                episode_id=episode_id,
                t=start_t + offset,
                z_rl=np.full(4, float(offset), dtype=np.float32),
                state=np.zeros(7, dtype=np.float32),
                a_ref=reference.astype(np.float32),
                a_exec=actual[offset].astype(np.float32),
                a_human=None,
                a_actor=actual[offset].astype(np.float32),
                source=Source.RLT,
                reward=1.0 if done else 0.0,
                done=done,
                gate_active=True,
                replay_include=True,
                actor_execution_profile=PERSISTENT_ACTOR_EXECUTION_PROFILE,
                actor_execution_plan_id=f"{episode_id}/plan_0",
                actor_execution_plan_offset=offset,
                actor_execution_committed=True,
                actor_canonical_decision=decision.astype(np.float32),
                actor_persistent_carry_in=carry.astype(np.float32),
                actor_persistent_carry_out=actual_residual[offset].astype(
                    np.float32
                ),
                actor_persistent_previous_carry=previous.astype(np.float32),
                actor_persistent_planned_residual=planned[offset].astype(
                    np.float32
                ),
                actor_execution_boundary_anchor=boundary.astype(np.float32),
                actor_filtered_base_action=base[offset].astype(np.float32),
                actor_filtered_actual_action=actual[offset].astype(np.float32),
                actor_filtered_actual_residual=actual_residual[offset].astype(
                    np.float32
                ),
                actor_execution_filter_tau_s=tau,
                actor_execution_filter_dt_s=float(dt[offset]),
                actor_execution_filter_alpha=float(alpha[offset]),
                actor_execution_projection_scale=1.0,
                action_schema_fingerprint=(
                    PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT
                ),
                execution_filter_profile=PERSISTENT_EXECUTION_FILTER_PROFILE,
                safety_reasons=("model_low_pass",),
                actor_execution_residual_max_rad=0.005,
                actor_execution_d1_max_rad=0.0015,
                actor_execution_d2_max_rad=0.001,
                actor_execution_direction_cone_deg=15.0,
                actor_execution_boundary_limit_rad=0.06,
                actor_execution_projection_scale_steps=33,
                actor_execution_min_projection_scale=0.2,
                actor_execution_direction_static_threshold_rad=0.001,
                actor_gripper_residual_mode=GRIPPER_RESIDUAL_CLOSE_ASSIST,
                actor_gripper_release_intent=False,
                execution_gripper_residual_max_close_m=0.005,
                execution_gripper_d1_max_m=0.0005,
                execution_gripper_d2_max_m=0.0003,
                execution_gripper_boundary_limit_m=0.0005,
                execution_gripper_command_min_m=0.0,
                execution_gripper_command_max_m=0.08,
                execution_gripper_release_reference_m=0.05,
                execution_gripper_release_delta_m=0.002,
                **{
                    name: float(values[offset])
                    for name, values in certificate.items()
                },
            )
        )
    return records


def test_close_assist_replay_preserves_planned_gripper_without_second_filter():
    records = _close_records()

    transitions = chunk_real_episode(
        records, chunk_length=10, stride=10, n_step=10, gamma=0.99
    )
    arrays = transitions_to_arrays(transitions)

    assert len(transitions) == 1
    transition = transitions[0]
    np.testing.assert_allclose(
        transition.filtered_actual_residual[:, 6],
        transition.actor_persistent_planned_residual[:, 6],
        rtol=0.0,
        atol=1e-8,
    )
    assert transition.success_mask
    np.testing.assert_array_equal(arrays["success_mask"], [True])
    assert arrays["actor_persistent_planned_residual"].shape == (1, 10, 7)
    assert arrays["actor_gripper_release_intent"].shape == (1, 10)
    assert arrays["gripper_residual_mode"].tolist() == [
        GRIPPER_RESIDUAL_CLOSE_ASSIST
    ]
    assert arrays["execution_gripper_d1_max_m"].shape == (1, 10)
    np.testing.assert_allclose(
        arrays["actor_filtered_actual_gripper_residual_max"][0],
        np.abs(transition.filtered_actual_residual[:, 6]),
        rtol=0.0,
        atol=1e-8,
    )


def test_close_assist_replay_rejects_a_second_gripper_low_pass():
    records = _close_records()
    row = records[5]
    wrong_residual = row.actor_filtered_actual_residual.copy()
    wrong_residual[6] *= 0.5
    wrong_actual = row.actor_filtered_base_action + wrong_residual
    records[5] = dataclasses.replace(
        row,
        a_exec=wrong_actual,
        actor_filtered_actual_action=wrong_actual,
        actor_filtered_actual_residual=wrong_residual,
        actor_persistent_carry_out=wrong_residual,
    )

    with pytest.raises(ValueError, match="must equal the planned knot"):
        chunk_real_episode(
            records, chunk_length=10, stride=10, n_step=10, gamma=0.99
        )


def test_human_close_assist_c10_derives_missing_actor_certificate():
    records = _close_records()
    human_records = []
    for record in records:
        human_action = record.actor_filtered_actual_action.copy()
        human_action[6] = 0.015
        human_residual = human_action - record.actor_filtered_base_action
        human_records.append(
            dataclasses.replace(
                record,
                source=Source.HUMAN_PIKA,
                a_exec=human_action,
                a_human=human_action,
                a_actor=None,
                actor_execution_profile=HUMAN_EXECUTION_PROFILE,
                actor_canonical_decision=np.zeros(7, dtype=np.float32),
                actor_persistent_carry_in=np.zeros(7, dtype=np.float32),
                actor_persistent_carry_out=np.zeros(7, dtype=np.float32),
                actor_persistent_previous_carry=np.zeros(7, dtype=np.float32),
                # Reproduces older rollout logs that retained a shadow Actor
                # knot even though the selected/physical source was human.
                actor_persistent_planned_residual=(
                    record.actor_persistent_planned_residual
                ),
                actor_filtered_actual_action=human_action,
                actor_filtered_actual_residual=human_residual,
                safety_reasons=(),
                actor_filtered_actual_gripper_residual_max=None,
                actor_filtered_actual_gripper_residual_d1_max_m=None,
                actor_filtered_actual_gripper_residual_d2_max_m=None,
                actor_filtered_actual_gripper_boundary_jump_max_m=None,
                actor_gripper_release_intent=bool(record.t % 2),
            )
        )

    transitions = chunk_real_episode(
        human_records, chunk_length=10, stride=10, n_step=10, gamma=0.99
    )
    arrays = transitions_to_arrays(transitions)

    assert len(transitions) == 1
    np.testing.assert_allclose(
        transitions[0].actor_persistent_planned_residual, 0.0
    )
    np.testing.assert_array_equal(
        transitions[0].actor_gripper_release_intent, False
    )
    assert arrays["actor_filtered_actual_gripper_residual_max"][0, 0] == (
        pytest.approx(0.005)
    )
    assert np.all(
        np.isfinite(
            arrays["actor_filtered_actual_gripper_boundary_jump_max_m"]
        )
    )


def test_replay_rejects_frozen_and_close_schema_mixing_before_chunking():
    close = _close_records(terminal=False)
    mixed = close + [
        dataclasses.replace(
            record,
            t=record.t + 10,
            actor_execution_plan_id="close_ep/plan_1",
            action_schema_fingerprint=PERSISTENT_ACTION_SCHEMA_FINGERPRINT,
            done=record.actor_execution_plan_offset == 9,
            reward=1.0 if record.actor_execution_plan_offset == 9 else 0.0,
        )
        for record in close
    ]

    with pytest.raises(ValueError, match="cannot be mixed"):
        chunk_real_episode(
            mixed, chunk_length=10, stride=10, n_step=10, gamma=0.99
        )


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def test_external_episode_reads_close_assist_evidence_and_alias(
    tmp_path: Path,
):
    rows = []
    for record in _close_records():
        row = {
            field.name: _jsonable(getattr(record, field.name))
            for field in dataclasses.fields(record)
            if getattr(record, field.name) is not None
        }
        # Exercise the runtime alias rather than only the canonical parser key.
        row["actor_governor_safe_residual_this_step"] = row.pop(
            "actor_persistent_planned_residual"
        )
        rows.append(row)
    path = tmp_path / "episode.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    loaded = load_episode_jsonl(
        path,
        contract=ExternalEpisodeContract(
            require_images=False,
            check_image_exists=False,
            execution_profile=PERSISTENT_ACTOR_EXECUTION_PROFILE,
            action_schema_fingerprint=(
                PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT
            ),
            require_committed_execution=True,
        ),
    )

    assert loaded[0].actor_gripper_residual_mode == (
        GRIPPER_RESIDUAL_CLOSE_ASSIST
    )
    np.testing.assert_allclose(
        loaded[7].actor_persistent_planned_residual,
        rows[7]["actor_governor_safe_residual_this_step"],
    )
