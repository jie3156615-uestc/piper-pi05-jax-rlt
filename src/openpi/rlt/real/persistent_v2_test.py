from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.rlt.real.agent_jax import RealRLTLearner
from openpi.rlt.real.agent_jax import ReplayBatchSampler
from openpi.rlt.real.agent_jax import admitted_human_gripper_q_filter
from openpi.rlt.real.agent_jax import q_filtered_human_gripper_bc_loss
from openpi.rlt.real.actor_runtime import ACTION_SCHEMA_FINGERPRINT
from openpi.rlt.real.actor_runtime import EXPECTED_CHECKPOINT_FINGERPRINTS
from openpi.rlt.real.actor_runtime import JaxCheckpointShadowActor
from openpi.rlt.real.config import HUMAN_EXECUTION_PROFILE
from openpi.rlt.real.config import GRIPPER_RESIDUAL_CLOSE_ASSIST
from openpi.rlt.real.config import MIN_GRIPPER_CLOSE_CRITIC_BURN_IN_UPDATES
from openpi.rlt.real.config import PERSISTENT_ACTION_SCHEMA_FINGERPRINT
from openpi.rlt.real.config import PERSISTENT_ACTOR_EXECUTION_PROFILE
from openpi.rlt.real.config import PERSISTENT_EXECUTION_FILTER_PROFILE
from openpi.rlt.real.config import PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT
from openpi.rlt.real.config import PERSISTENT_GOVERNOR_PROFILE
from openpi.rlt.real.config import RANK1_ACTOR_OUTPUT_SCHEMA_FINGERPRINT
from openpi.rlt.real.config import RealRLTConfig
from openpi.rlt.real.config import Source
from openpi.rlt.real.external_episode import ExternalEpisodeContract
from openpi.rlt.real.external_episode import ExternalEpisodeError
from openpi.rlt.real.external_episode import validate_episode_records
from openpi.rlt.real.networks_jax import persistent_filtered_candidate_action
from openpi.rlt.real.networks_jax import persistent_governed_residual
from openpi.rlt.real.replay import RealStepRecord
from openpi.rlt.real.replay import chunk_real_episode
from openpi.rlt.real.replay_io import transitions_to_arrays


RUNTIME_ROOT = Path(
    os.environ.get("OPENPI_PIPER_RUNTIME_ROOT", "/home/cwzk/piper_jax_inference_v1")
)
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))


def _persistent_cfg(**changes) -> RealRLTConfig:
    values = {
        "actor_execution_profile": PERSISTENT_ACTOR_EXECUTION_PROFILE,
        "chunk_stride": 10,
        "hidden_dim": 32,
        "projection_dim": 16,
        "batch_size": 4,
        "reference_dropout": 0.0,
        "seed": 11,
    }
    values.update(changes)
    return dataclasses.replace(RealRLTConfig(), **values)


def _runtime_plan(
    a_ref: np.ndarray,
    direction: np.ndarray,
    carry: np.ndarray,
    previous: np.ndarray,
    boundary: np.ndarray,
    cfg: RealRLTConfig,
):
    if not RUNTIME_ROOT.is_dir():
        pytest.skip("5090 runtime checkout is unavailable")
    from piper_runtime.rlt_residual_governor import ActorResidualGovernorConfig
    from piper_runtime.rlt_residual_governor import RANK1_BUMP_WINDOW
    from piper_runtime.rlt_residual_governor import govern_persistent_actor_plan

    raw_actor = np.asarray(a_ref, dtype=np.float64).copy()
    raw_actor[:, :6] += RANK1_BUMP_WINDOW[:, None] * direction[None, :6]
    return govern_persistent_actor_plan(
        plan_id="golden",
        behavior_plan_id="base",
        behavior_start_offset=0,
        behavior_ref=a_ref,
        raw_actor=raw_actor,
        boundary_anchor=boundary,
        carry_in=carry,
        previous_carry=previous,
        config=ActorResidualGovernorConfig(
            actor_residual_max_rad=cfg.actor_residual_max_rad,
            actor_residual_d1_max_rad=cfg.actor_residual_d1_max_rad,
            actor_residual_d2_max_rad=cfg.actor_residual_d2_max_rad,
            actor_direction_cone_deg=cfg.actor_direction_cone_deg,
            max_boundary_jump_rad=cfg.actor_max_boundary_jump_rad,
            direction_static_threshold_rad=cfg.actor_direction_static_threshold_rad,
            min_projection_scale=cfg.actor_min_projection_scale,
            projection_scale_steps=cfg.actor_projection_scale_steps,
        ),
    )


@pytest.mark.parametrize("seed", [1, 4, 19, 33])
def test_jax_governor_matches_runtime_numpy_random_and_boundary_golden(seed: int):
    cfg = _persistent_cfg()
    rng = np.random.default_rng(seed)
    ref_steps = rng.normal(scale=0.012, size=(10, 6))
    ref = np.zeros((10, 7), dtype=np.float32)
    ref[:, :6] = np.cumsum(ref_steps, axis=0)
    carry = np.zeros(7, dtype=np.float32)
    carry[:6] = rng.uniform(-0.0015, 0.0015, size=6)
    previous = carry.copy()
    previous[:6] += rng.uniform(-0.0002, 0.0002, size=6)
    direction = np.zeros(7, dtype=np.float32)
    direction[:6] = rng.uniform(-0.005, 0.005, size=6)
    boundary = np.zeros(7, dtype=np.float32)
    boundary[:6] = -carry[:6]

    runtime = _runtime_plan(ref, direction, carry, previous, boundary, cfg)
    residual, scale, approved = persistent_governed_residual(
        jnp.asarray(ref[None]),
        jnp.asarray(direction[None]),
        jnp.asarray(carry[None]),
        jnp.asarray(previous[None]),
        jnp.asarray(boundary[None]),
        residual_max_rad=cfg.actor_residual_max_rad,
        d1_max_rad=cfg.actor_residual_d1_max_rad,
        d2_max_rad=cfg.actor_residual_d2_max_rad,
        direction_cone_deg=cfg.actor_direction_cone_deg,
        boundary_jump_max_rad=cfg.actor_max_boundary_jump_rad,
        direction_static_threshold_rad=cfg.actor_direction_static_threshold_rad,
        projection_scale_steps=cfg.actor_projection_scale_steps,
    )

    assert bool(np.asarray(approved)[0]) == bool(runtime.approved)
    np.testing.assert_allclose(np.asarray(scale)[0], runtime.projection_scale, atol=2e-6)
    np.testing.assert_allclose(np.asarray(residual)[0], runtime.safe_residual, atol=2e-6)


def test_zero_direction_and_zero_carry_candidate_equals_filtered_base_not_raw_reference():
    cfg = _persistent_cfg()
    a_ref = np.full((1, 10, 7), 0.4, dtype=np.float32)
    base = np.full((1, 10, 7), -0.2, dtype=np.float32)
    zeros = np.zeros((1, 7), dtype=np.float32)
    alpha = np.full((1, 10), 0.48, dtype=np.float32)

    candidate, evidence = persistent_filtered_candidate_action(
        jnp.asarray(base),
        jnp.asarray(a_ref),
        jnp.asarray(zeros),
        jnp.asarray(zeros),
        jnp.asarray(zeros),
        jnp.asarray(zeros),
        jnp.asarray(alpha),
        cfg,
    )

    np.testing.assert_array_equal(np.asarray(evidence["filtered_residual"]), np.zeros((1, 10, 7)))
    np.testing.assert_array_equal(np.asarray(candidate), base)
    assert not np.array_equal(np.asarray(candidate), a_ref)


def _make_plan_records(
    *,
    episode_id: str,
    start_t: int,
    plan_number: int,
    profile: str,
    carry_in: np.ndarray,
    previous_carry: np.ndarray,
    direction: np.ndarray,
    terminal_offset: int | None = None,
    human_delta: float = 0.0,
) -> list[RealStepRecord]:
    tau = 0.05
    dt = np.linspace(0.028, 0.036, 10, dtype=np.float64)
    alpha = 1.0 - np.exp(-dt / tau)
    base = np.zeros((10, 7), dtype=np.float64)
    base[:, :6] = np.linspace(0.0, 0.09, 10)[:, None]
    ref = base.copy()
    boundary = np.zeros(7, dtype=np.float64)
    boundary[:6] = -carry_in[:6]
    if profile == PERSISTENT_ACTOR_EXECUTION_PROFILE:
        runtime = _runtime_plan(ref, direction, carry_in, previous_carry, boundary, _persistent_cfg())
        planned = np.asarray(runtime.safe_residual, dtype=np.float64)
        projection_scale = float(runtime.projection_scale)
        current = carry_in.astype(np.float64).copy()
        residual_rows = []
        for planned_row, row_alpha in zip(planned, alpha):
            current = (1.0 - row_alpha) * current + row_alpha * planned_row
            current[6] = 0.0
            residual_rows.append(current.copy())
        residual = np.asarray(residual_rows)
        actual = base + residual
    else:
        direction = np.zeros(7, dtype=np.float64)
        carry_in = np.zeros(7, dtype=np.float64)
        previous_carry = np.zeros(7, dtype=np.float64)
        projection_scale = 0.0
        residual = np.zeros((10, 7), dtype=np.float64)
        residual[:, :6] = human_delta
        actual = base + residual

    records = []
    for offset in range(10):
        done = terminal_offset == offset
        records.append(
            RealStepRecord(
                episode_id=episode_id,
                t=start_t + offset,
                z_rl=np.full(4, 0.1 * (start_t + offset), dtype=np.float32),
                state=np.zeros(7, dtype=np.float32),
                a_ref=ref.astype(np.float32),
                a_exec=actual[offset].astype(np.float32),
                a_human=actual[offset].astype(np.float32) if profile == HUMAN_EXECUTION_PROFILE else None,
                a_actor=None,
                source=Source.HUMAN_PIKA if profile == HUMAN_EXECUTION_PROFILE else Source.RLT,
                reward=1.0 if done else 0.0,
                done=done,
                actor_execution_profile=profile,
                actor_execution_plan_id=f"{episode_id}/plan_{plan_number}",
                actor_execution_plan_offset=offset,
                actor_execution_committed=True,
                actor_canonical_decision=np.asarray(direction, dtype=np.float32),
                actor_persistent_carry_in=np.asarray(carry_in, dtype=np.float32),
                actor_persistent_carry_out=(
                    np.zeros(7, dtype=np.float32)
                    if profile == HUMAN_EXECUTION_PROFILE
                    else residual[offset].astype(np.float32)
                ),
                actor_persistent_previous_carry=np.asarray(previous_carry, dtype=np.float32),
                actor_execution_boundary_anchor=boundary.astype(np.float32),
                actor_filtered_base_action=base[offset].astype(np.float32),
                actor_filtered_actual_action=actual[offset].astype(np.float32),
                actor_filtered_actual_residual=residual[offset].astype(np.float32),
                actor_execution_filter_tau_s=tau,
                actor_execution_filter_dt_s=float(dt[offset]),
                actor_execution_filter_alpha=float(alpha[offset]),
                actor_execution_projection_scale=projection_scale,
                action_schema_fingerprint=PERSISTENT_ACTION_SCHEMA_FINGERPRINT,
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
            )
        )
    return records


def test_replay_builds_only_complete_plan_and_preserves_filtered_actual_contract():
    zero = np.zeros(7, dtype=np.float32)
    direction = np.array([0.001, -0.0005, 0.0008, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    first = _make_plan_records(
        episode_id="ep",
        start_t=0,
        plan_number=0,
        profile=PERSISTENT_ACTOR_EXECUTION_PROFILE,
        carry_in=zero,
        previous_carry=zero,
        direction=direction,
    )
    carry = first[-1].actor_persistent_carry_out
    second = _make_plan_records(
        episode_id="ep",
        start_t=10,
        plan_number=1,
        profile=PERSISTENT_ACTOR_EXECUTION_PROFILE,
        carry_in=carry,
        previous_carry=first[-2].actor_persistent_carry_out,
        direction=-direction,
        terminal_offset=9,
    )

    transitions = chunk_real_episode(
        first + second, chunk_length=10, stride=10, n_step=10, gamma=0.99
    )
    arrays = transitions_to_arrays(transitions)

    assert len(transitions) == 2
    assert arrays["a_base_filtered"].shape == (2, 10, 7)
    assert arrays["actor_canonical_decision"].shape == (2, 7)
    np.testing.assert_allclose(arrays["a_exec"], arrays["a_filtered_actual"], atol=1e-7)
    np.testing.assert_array_equal(arrays["actor_execution_plan_offset"][0], np.arange(10))
    assert transitions[0].discount == pytest.approx(0.99**10)
    assert transitions[1].discount == 0.0


def test_persistent_replay_rejects_unmodelled_safety_modification():
    zero = np.zeros(7, dtype=np.float32)
    records = _make_plan_records(
        episode_id="unsafe",
        start_t=0,
        plan_number=0,
        profile=PERSISTENT_ACTOR_EXECUTION_PROFILE,
        carry_in=zero,
        previous_carry=zero,
        direction=zero,
        terminal_offset=9,
    )
    records[4] = dataclasses.replace(
        records[4],
        safety_reasons=("model_low_pass", "joint_step_clamp"),
    )

    with pytest.raises(ValueError, match="nonlinear/unmodelled safety"):
        chunk_real_episode(
            records, chunk_length=10, stride=10, n_step=10, gamma=0.99
        )


@pytest.mark.parametrize("value", [None, 0.004])
def test_persistent_replay_rejects_missing_or_drifted_runtime_envelope(value):
    zero = np.zeros(7, dtype=np.float32)
    records = _make_plan_records(
        episode_id="envelope",
        start_t=0,
        plan_number=0,
        profile=PERSISTENT_ACTOR_EXECUTION_PROFILE,
        carry_in=zero,
        previous_carry=zero,
        direction=zero,
        terminal_offset=9,
    )
    records[6] = dataclasses.replace(
        records[6],
        actor_execution_residual_max_rad=value,
    )

    expected = "ten finite values" if value is None else "runtime envelope drift"
    with pytest.raises((TypeError, ValueError), match=expected):
        chunk_real_episode(
            records, chunk_length=10, stride=10, n_step=10, gamma=0.99
        )


def test_cross_profile_bootstrap_resets_carry_and_human_delta_is_retained():
    zero = np.zeros(7, dtype=np.float32)
    direction = np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    actor = _make_plan_records(
        episode_id="mixed",
        start_t=0,
        plan_number=0,
        profile=PERSISTENT_ACTOR_EXECUTION_PROFILE,
        carry_in=zero,
        previous_carry=zero,
        direction=direction,
    )
    human = _make_plan_records(
        episode_id="mixed",
        start_t=10,
        plan_number=1,
        profile=HUMAN_EXECUTION_PROFILE,
        carry_in=zero,
        previous_carry=zero,
        direction=zero,
        human_delta=0.025,
    )
    actor_after = _make_plan_records(
        episode_id="mixed",
        start_t=20,
        plan_number=2,
        profile=PERSISTENT_ACTOR_EXECUTION_PROFILE,
        carry_in=zero,
        previous_carry=zero,
        direction=-direction,
        terminal_offset=9,
    )

    transitions = chunk_real_episode(
        actor + human + actor_after,
        chunk_length=10,
        stride=10,
        n_step=10,
        gamma=0.99,
    )

    assert [transition.actor_execution_profile for transition in transitions] == [
        PERSISTENT_ACTOR_EXECUTION_PROFILE,
        HUMAN_EXECUTION_PROFILE,
        PERSISTENT_ACTOR_EXECUTION_PROFILE,
    ]
    assert np.max(np.abs(transitions[1].filtered_actual_residual[..., :6])) == pytest.approx(0.025)
    np.testing.assert_array_equal(
        transitions[0].next_actor_persistent_carry_in, np.zeros(7)
    )
    assert transitions[1].human_mask.all()


def test_external_episode_allows_human_gripper_difference_but_rejects_actor_residual():
    zero = np.zeros(7, dtype=np.float32)
    contract = ExternalEpisodeContract(
        require_images=False,
        check_image_exists=False,
        execution_profile=PERSISTENT_ACTOR_EXECUTION_PROFILE,
        require_committed_execution=True,
    )
    human = _make_plan_records(
        episode_id="human_gripper",
        start_t=0,
        plan_number=0,
        profile=HUMAN_EXECUTION_PROFILE,
        carry_in=zero,
        previous_carry=zero,
        direction=zero,
        terminal_offset=9,
    )
    human_with_gripper = []
    for record in human:
        residual = record.actor_filtered_actual_residual.copy()
        residual[6] = 0.03
        actual = record.actor_filtered_base_action + residual
        human_with_gripper.append(
            dataclasses.replace(
                record,
                a_exec=actual,
                a_human=actual,
                actor_filtered_actual_action=actual,
                actor_filtered_actual_residual=residual,
            )
        )

    validate_episode_records(human_with_gripper, contract=contract)

    actor = _make_plan_records(
        episode_id="actor_gripper",
        start_t=0,
        plan_number=0,
        profile=PERSISTENT_ACTOR_EXECUTION_PROFILE,
        carry_in=zero,
        previous_carry=zero,
        direction=zero,
        terminal_offset=9,
    )
    residual = actor[0].actor_filtered_actual_residual.copy()
    residual[6] = 0.001
    actual = actor[0].actor_filtered_base_action + residual
    actor[0] = dataclasses.replace(
        actor[0],
        a_exec=actual,
        actor_filtered_actual_action=actual,
        actor_filtered_actual_residual=residual,
    )

    with pytest.raises(ExternalEpisodeError, match="actual gripper residual"):
        validate_episode_records(actor, contract=contract)


def test_partial_terminal_reward_moves_to_last_complete_plan_without_fake_actions():
    zero = np.zeros(7, dtype=np.float32)
    complete = _make_plan_records(
        episode_id="partial",
        start_t=0,
        plan_number=0,
        profile=PERSISTENT_ACTOR_EXECUTION_PROFILE,
        carry_in=zero,
        previous_carry=zero,
        direction=zero,
    )
    partial = _make_plan_records(
        episode_id="partial",
        start_t=10,
        plan_number=1,
        profile=PERSISTENT_ACTOR_EXECUTION_PROFILE,
        carry_in=zero,
        previous_carry=zero,
        direction=zero,
        terminal_offset=3,
    )[:4]

    transitions = chunk_real_episode(
        complete + partial,
        chunk_length=10,
        stride=10,
        n_step=10,
        gamma=0.99,
    )

    assert len(transitions) == 1
    transition = transitions[0]
    assert transition.done
    assert transition.discount == 0.0
    assert transition.reward == pytest.approx(0.99**13)
    assert transition.terminal_reward_migration_steps == 4
    assert transition.terminal_reward_migration_offset == 13
    np.testing.assert_array_equal(
        transition.a_exec_absolute,
        np.stack([record.a_exec for record in complete]),
    )
    np.testing.assert_array_equal(transition.next_state, partial[-1].state)


def _persistent_replay(n: int = 12) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    z = rng.normal(size=(n, 5)).astype(np.float32)
    state = rng.normal(scale=0.1, size=(n, 7)).astype(np.float32)
    ref = rng.normal(scale=0.05, size=(n, 10, 7)).astype(np.float32)
    base = ref + rng.normal(scale=0.002, size=ref.shape).astype(np.float32)
    carry = np.zeros((n, 7), dtype=np.float32)
    previous = np.zeros((n, 7), dtype=np.float32)
    boundary = np.zeros((n, 7), dtype=np.float32)
    direction = np.zeros((n, 7), dtype=np.float32)
    dt = np.full((n, 10), 1.0 / 30.0, dtype=np.float32)
    tau = np.full((n, 10), 0.05, dtype=np.float32)
    alpha = (1.0 - np.exp(-dt / tau)).astype(np.float32)
    actual = base.copy()
    return {
        "episode_id": np.asarray([f"ep_{idx // 3}" for idx in range(n)]),
        "episode_split": np.asarray(["train"] * n),
        "z_rl": z,
        "state": state,
        "a_ref": ref,
        "a_exec": actual,
        "a_human": np.zeros_like(actual),
        "human_mask": np.zeros((n, 10), dtype=np.bool_),
        "reward": np.zeros(n, dtype=np.float32),
        "discount": np.zeros(n, dtype=np.float32),
        "next_z_rl": np.roll(z, -1, axis=0),
        "next_state": np.roll(state, -1, axis=0),
        "next_a_ref": np.roll(ref, -1, axis=0),
        "done": np.ones(n, dtype=np.bool_),
        "actor_execution_profile": np.asarray(
            [PERSISTENT_ACTOR_EXECUTION_PROFILE] * n
        ),
        "action_schema_fingerprint": np.asarray(
            [PERSISTENT_ACTION_SCHEMA_FINGERPRINT] * n
        ),
        "execution_filter_profile": np.asarray(
            [PERSISTENT_EXECUTION_FILTER_PROFILE] * n
        ),
        "actor_canonical_decision": direction,
        "actor_persistent_carry_in": carry,
        "actor_persistent_carry_out": carry,
        "actor_persistent_previous_carry": previous,
        "actor_execution_boundary_anchor": boundary,
        "a_base_filtered": base,
        "a_filtered_actual": actual,
        "filtered_actual_residual": np.zeros_like(actual),
        "execution_filter_tau_s": tau,
        "execution_filter_dt_s": dt,
        "execution_filter_alpha": alpha,
        "execution_projection_scale": np.ones((n, 10), dtype=np.float32),
        "next_a_base_filtered": np.roll(base, -1, axis=0),
        "next_execution_filter_alpha": np.roll(alpha, -1, axis=0),
        "next_actor_persistent_previous_carry": np.roll(previous, -1, axis=0),
        "next_actor_persistent_carry_in": np.roll(carry, -1, axis=0),
        "next_actor_execution_boundary_anchor": np.roll(boundary, -1, axis=0),
        "execution_residual_max_rad": np.full((n, 10), 0.005, dtype=np.float32),
        "execution_d1_max_rad": np.full((n, 10), 0.0015, dtype=np.float32),
        "execution_d2_max_rad": np.full((n, 10), 0.001, dtype=np.float32),
        "execution_direction_cone_deg": np.full((n, 10), 15.0, dtype=np.float32),
        "execution_boundary_limit_rad": np.full((n, 10), 0.06, dtype=np.float32),
        "execution_projection_scale_steps": np.full((n, 10), 33.0, dtype=np.float32),
        "execution_min_projection_scale": np.full((n, 10), 0.2, dtype=np.float32),
        "execution_direction_static_threshold_rad": np.full(
            (n, 10), 0.001, dtype=np.float32
        ),
    }


def _legacy_replay(n: int = 12) -> dict[str, np.ndarray]:
    replay = _persistent_replay(n)
    replay["a_exec"] = replay["a_exec"] + 0.03
    for key in list(replay):
        if key.startswith(("actor_persistent_", "actor_execution_", "next_actor_", "execution_")):
            del replay[key]
    for key in (
        "action_schema_fingerprint",
        "execution_filter_profile",
        "a_base_filtered",
        "a_filtered_actual",
        "filtered_actual_residual",
        "execution_filter_tau_s",
        "execution_filter_dt_s",
        "execution_filter_alpha",
        "execution_projection_scale",
        "next_a_base_filtered",
        "next_execution_filter_alpha",
        "actor_canonical_decision",
    ):
        replay.pop(key, None)
    return replay


def test_admitted_human_gripper_q_filter_uses_critic_advantage_not_reward_label() -> None:
    # Index 1 represents a reward-0 admitted human chunk.  It is deliberately
    # selected because its human gripper candidate has higher Q.  Index 2 is a
    # reward-1 chunk, but equal Q is not enough for strict "better than".
    reward = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    selected, advantage = admitted_human_gripper_q_filter(
        jnp.asarray([2.0, 3.0, 4.0, 9.0]),
        jnp.asarray([1.0, 2.0, 4.0, 0.0]),
        jnp.asarray([1.0, 1.0, 1.0, 0.0]),
        margin=0.0,
    )

    np.testing.assert_array_equal(np.asarray(selected), [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(np.asarray(advantage), [1.0, 1.0, 0.0, 9.0])
    assert reward[1] == 0.0
    assert float(np.asarray(selected)[1]) == 1.0

    selected_with_margin, _ = admitted_human_gripper_q_filter(
        jnp.asarray([2.0, 3.0, 4.0, 9.0]),
        jnp.asarray([1.0, 2.0, 4.0, 0.0]),
        jnp.asarray([1.0, 1.0, 1.0, 0.0]),
        margin=1.0,
    )
    np.testing.assert_array_equal(
        np.asarray(selected_with_margin),
        [0.0, 0.0, 0.0, 0.0],
    )


def test_reward0_close_teacher_gets_bc_gradient_only_when_q_is_better() -> None:
    actor = jnp.full((2, 10), 0.02, dtype=jnp.float32)
    teacher = jnp.full((2, 10), 0.015, dtype=jnp.float32)
    human_mask = jnp.ones((2, 10), dtype=jnp.float32)

    def selected_loss(value: jnp.ndarray) -> jnp.ndarray:
        loss, _, _, _ = q_filtered_human_gripper_bc_loss(
            value,
            teacher,
            human_mask,
            # Row 0 is the admitted reward-0 example and is deliberately
            # valued above Actor; row 1 is valued below Actor.
            jnp.asarray([2.0, 0.0], dtype=jnp.float32),
            jnp.asarray([1.0, 1.0], dtype=jnp.float32),
            margin=0.0,
            scale_m=0.005,
        )
        return loss

    loss, selected_chunks, selected_steps, _ = (
        q_filtered_human_gripper_bc_loss(
            actor,
            teacher,
            human_mask,
            jnp.asarray([2.0, 0.0], dtype=jnp.float32),
            jnp.asarray([1.0, 1.0], dtype=jnp.float32),
            margin=0.0,
            scale_m=0.005,
        )
    )
    gradient = np.asarray(jax.grad(selected_loss)(actor))

    assert float(loss) == pytest.approx(1.0)
    np.testing.assert_array_equal(np.asarray(selected_chunks), [1.0, 0.0])
    assert float(jnp.sum(selected_steps[0])) == 10.0
    assert np.any(np.abs(gradient[0]) > 0.0)
    np.testing.assert_array_equal(gradient[1], np.zeros(10, dtype=np.float32))

    def rejected_loss(value: jnp.ndarray) -> jnp.ndarray:
        loss, _, _, _ = q_filtered_human_gripper_bc_loss(
            value,
            teacher,
            human_mask,
            jnp.zeros(2, dtype=jnp.float32),
            jnp.ones(2, dtype=jnp.float32),
            margin=0.0,
            scale_m=0.005,
        )
        return loss

    assert float(rejected_loss(actor)) == 0.0
    np.testing.assert_array_equal(
        np.asarray(jax.grad(rejected_loss)(actor)),
        np.zeros((2, 10), dtype=np.float32),
    )


def test_admitted_human_gripper_bc_q_filters_both_reward_labels() -> None:
    replay = _persistent_replay(12)
    success = np.arange(12) < 6
    replay["success_mask"] = success
    replay["reward"] = success.astype(np.float32)
    replay["human_mask"] = np.ones((12, 10), dtype=np.bool_)
    replay["a_ref"][..., 6] = 0.02
    replay["a_base_filtered"][..., 6] = 0.02
    replay["a_filtered_actual"][..., 6] = 0.02
    replay["a_exec"][..., 6] = 0.02
    replay["next_a_ref"][..., 6] = 0.02
    replay["next_a_base_filtered"][..., 6] = 0.02
    replay["actor_execution_boundary_anchor"][..., 6] = 0.02
    replay["next_actor_execution_boundary_anchor"][..., 6] = 0.02
    replay["a_human"] = replay["a_base_filtered"].copy()
    replay["a_human"][success, :, 6] = 0.0
    # Reward-0 human motion remains admitted training data.  The one-sided
    # physical contract turns opening into a zero close residual; the Critic
    # Q-filter, rather than the binary reward label, decides whether to clone.
    replay["a_human"][~success, :, 6] = 0.08
    replay["action_schema_fingerprint"] = np.asarray(
        [PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT] * 12
    )
    cfg = _persistent_cfg(
        policy_delay=1,
        freeze_gripper_residual=False,
        gripper_residual_mode=GRIPPER_RESIDUAL_CLOSE_ASSIST,
        actor_start_step=MIN_GRIPPER_CLOSE_CRITIC_BURN_IN_UPDATES,
        beta_human_gripper_bc=1.0,
        human_gripper_bc_scale_m=0.005,
    )
    for name, value in {
        "execution_gripper_residual_max_close_m": cfg.actor_gripper_residual_max_close_m,
        "execution_gripper_d1_max_m": cfg.actor_gripper_residual_d1_max_m,
        "execution_gripper_d2_max_m": cfg.actor_gripper_residual_d2_max_m,
        "execution_gripper_boundary_limit_m": cfg.actor_gripper_max_boundary_jump_m,
        "execution_gripper_command_min_m": cfg.gripper_command_min_m,
        "execution_gripper_command_max_m": cfg.gripper_command_max_m,
        "execution_gripper_release_reference_m": cfg.gripper_release_reference_m,
        "execution_gripper_release_delta_m": cfg.gripper_release_delta_m,
    }.items():
        replay[name] = np.full((12, 10), value, dtype=np.float32)

    learner = RealRLTLearner.create(
        replay,
        config=cfg,
        residual_limit=np.full((10, 7), 0.005, dtype=np.float32),
    )
    learner.update_step = cfg.actor_start_step
    batch = ReplayBatchSampler(replay).sample(128)
    metrics = learner.update(batch)

    assert metrics["actor_updated"] == 1.0
    expected_fraction = float(np.mean(batch["human_mask"]))
    assert metrics["actor_admitted_human_gripper_fraction"] == pytest.approx(
        expected_fraction,
        abs=1e-7,
    )
    assert metrics["actor_admitted_human_gripper_chunks"] == pytest.approx(
        float(len(batch["reward"]))
    )
    assert metrics["actor_admitted_human_gripper_steps"] == pytest.approx(
        float(np.count_nonzero(batch["human_mask"]))
    )
    assert metrics["actor_reward1_human_gripper_chunks"] > 0.0
    assert metrics["actor_reward0_human_gripper_chunks"] > 0.0
    assert (
        metrics[
            "actor_reward1_human_gripper_q_filter_selected_chunks"
        ]
        <= metrics["actor_reward1_human_gripper_chunks"]
    )
    assert (
        metrics[
            "actor_reward0_human_gripper_q_filter_selected_chunks"
        ]
        <= metrics["actor_reward0_human_gripper_chunks"]
    )
    assert (
        0.0
        <= metrics["actor_human_gripper_q_filter_fraction"]
        <= 1.0
    )
    assert (
        metrics["actor_human_gripper_q_filter_selected_steps"]
        <= metrics["actor_admitted_human_gripper_steps"]
    )
    assert metrics["actor_human_gripper_supervised_steps"] == pytest.approx(
        metrics["actor_human_gripper_q_filter_selected_steps"]
    )
    assert learner.config.beta_human_bc == 0.0


def test_close_assist_burns_in_critic_then_resume_starts_actor_once_eligible(
    tmp_path: Path,
) -> None:
    replay = _persistent_replay(12)
    replay["action_schema_fingerprint"] = np.asarray(
        [PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT] * 12
    )
    cfg = _persistent_cfg(
        policy_delay=2,
        freeze_gripper_residual=False,
        gripper_residual_mode=GRIPPER_RESIDUAL_CLOSE_ASSIST,
        actor_start_step=MIN_GRIPPER_CLOSE_CRITIC_BURN_IN_UPDATES,
        beta_human_gripper_bc=1.0,
    )
    for name, value in {
        "execution_gripper_residual_max_close_m": cfg.actor_gripper_residual_max_close_m,
        "execution_gripper_d1_max_m": cfg.actor_gripper_residual_d1_max_m,
        "execution_gripper_d2_max_m": cfg.actor_gripper_residual_d2_max_m,
        "execution_gripper_boundary_limit_m": cfg.actor_gripper_max_boundary_jump_m,
        "execution_gripper_command_min_m": cfg.gripper_command_min_m,
        "execution_gripper_command_max_m": cfg.gripper_command_max_m,
        "execution_gripper_release_reference_m": cfg.gripper_release_reference_m,
        "execution_gripper_release_delta_m": cfg.gripper_release_delta_m,
    }.items():
        replay[name] = np.full((12, 10), value, dtype=np.float32)

    learner = RealRLTLearner.create(
        replay,
        config=cfg,
        residual_limit=np.full((10, 7), 0.005, dtype=np.float32),
    )
    sampler = ReplayBatchSampler(replay)
    actor_before = [
        np.asarray(leaf).copy()
        for leaf in jax.tree_util.tree_leaves(learner.actor_state.params)
    ]
    learner.update_step = cfg.actor_start_step - 1
    boundary_metrics = learner.update(sampler.sample(4))

    assert boundary_metrics["update_step"] == cfg.actor_start_step
    assert boundary_metrics["actor_burn_in_active"] == 1.0
    assert boundary_metrics["actor_updated"] == 0.0
    for before, after in zip(
        actor_before,
        jax.tree_util.tree_leaves(learner.actor_state.params),
    ):
        np.testing.assert_array_equal(before, np.asarray(after))

    checkpoint = tmp_path / "burn_in_boundary"
    learner.save_checkpoint(checkpoint)
    resumed = RealRLTLearner.load_checkpoint(checkpoint)
    odd_metrics = resumed.update(sampler.sample(4))
    actor_metrics = resumed.update(sampler.sample(4))

    assert odd_metrics["update_step"] == cfg.actor_start_step + 1
    assert odd_metrics["actor_burn_in_active"] == 0.0
    assert odd_metrics["actor_updated"] == 0.0
    assert actor_metrics["update_step"] == cfg.actor_start_step + 2
    assert actor_metrics["actor_burn_in_active"] == 0.0
    assert actor_metrics["actor_updated"] == 1.0


def test_actor_only_checkpoint_warm_start_is_exact_and_everything_else_is_fresh(tmp_path):
    old_replay = _legacy_replay()
    old = RealRLTLearner.create(
        old_replay,
        config=dataclasses.replace(
            RealRLTConfig(),
            hidden_dim=32,
            projection_dim=16,
            batch_size=4,
            seed=5,
            beta_bc=40.0,
            beta_human_bc=0.0,
        ),
        fingerprints={
            "base_checkpoint": "base",
            "rl_token": "token",
            "phase_classifier": "phase",
            "action_schema": RANK1_ACTOR_OUTPUT_SCHEMA_FINGERPRINT,
        },
    )
    old.update(ReplayBatchSampler(old_replay).sample(4))
    old_checkpoint = tmp_path / "old"
    old.save_checkpoint(old_checkpoint)
    replay = _persistent_replay()
    fingerprints = {
        "base_checkpoint": "base",
        "rl_token": "token",
        "phase_classifier": "phase",
        "action_schema": PERSISTENT_ACTION_SCHEMA_FINGERPRINT,
        "actor_execution_profile": PERSISTENT_ACTOR_EXECUTION_PROFILE,
        "execution_filter_profile": PERSISTENT_EXECUTION_FILTER_PROFILE,
        "actor_governor": PERSISTENT_GOVERNOR_PROFILE,
    }

    with pytest.raises(ValueError, match="objective weights exactly"):
        RealRLTLearner.warm_start_actor_for_persistent_v2(
            old_checkpoint,
            replay,
            config=_persistent_cfg(seed=5, beta_bc=1.0),
            fingerprints=fingerprints,
            expected_source_fingerprints={
                "base_checkpoint": "base",
                "rl_token": "token",
                "phase_classifier": "phase",
            },
        )

    exploratory = RealRLTLearner.warm_start_actor_for_persistent_v2(
        old_checkpoint,
        replay,
        config=_persistent_cfg(seed=5, beta_bc=20.0, beta_human_bc=0.0),
        fingerprints=fingerprints,
        expected_source_fingerprints={
            "base_checkpoint": "base",
            "rl_token": "token",
            "phase_classifier": "phase",
        },
        allow_objective_migration=True,
    )
    assert exploratory.update_step == 0
    assert exploratory.config.beta_bc == 20.0
    assert exploratory.config.beta_human_bc == 0.0
    assert exploratory.fingerprints["warm_start_objective_weights"] == (
        "explicit_target_beta_migration_v1"
    )
    assert exploratory.fingerprints["warm_start_source_beta_bc"] == "40.0"
    assert exploratory.fingerprints["warm_start_source_beta_human_bc"] == "0.0"
    assert exploratory.fingerprints["warm_start_target_beta_bc"] == "20.0"
    assert exploratory.fingerprints["warm_start_target_beta_human_bc"] == "0.0"
    assert (
        exploratory.fingerprints["warm_start_objective_migration_authorized"]
        == "true"
    )
    assert all(
        np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(
            jax.tree_util.tree_leaves(old.actor_state.params),
            jax.tree_util.tree_leaves(exploratory.actor_state.params),
        )
    )
    assert all(
        np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(
            jax.tree_util.tree_leaves(exploratory.actor_state.params),
            jax.tree_util.tree_leaves(exploratory.actor_state.target_params),
        )
    )

    migrated = RealRLTLearner.warm_start_actor_for_persistent_v2(
        old_checkpoint,
        replay,
        config=_persistent_cfg(seed=5, beta_bc=40.0, beta_human_bc=0.0),
        fingerprints=fingerprints,
        expected_source_fingerprints={
            "base_checkpoint": "base",
            "rl_token": "token",
            "phase_classifier": "phase",
        },
    )

    assert migrated.update_step == 0
    assert migrated.config.beta_bc == 40.0
    assert migrated.config.beta_human_bc == 0.0
    assert migrated.fingerprints["warm_start_objective_weights"] == (
        "preserve_source_beta_bc_and_beta_human_bc_exactly_v1"
    )
    assert (
        migrated.fingerprints["warm_start_objective_migration_authorized"]
        == "false"
    )
    assert all(
        np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(
            jax.tree_util.tree_leaves(old.actor_state.params),
            jax.tree_util.tree_leaves(migrated.actor_state.params),
        )
    )
    assert all(
        np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(
            jax.tree_util.tree_leaves(migrated.actor_state.params),
            jax.tree_util.tree_leaves(migrated.actor_state.target_params),
        )
    )
    assert any(
        not np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(
            jax.tree_util.tree_leaves(old.critic_state.params),
            jax.tree_util.tree_leaves(migrated.critic_state.params),
        )
    )
    np.testing.assert_array_equal(migrated.normalization.z_rl.mean, old.normalization.z_rl.mean)
    assert not np.array_equal(
        migrated.normalization.candidate_action.mean,
        old.normalization.candidate_action.mean,
    )
    checkpoint = tmp_path / "persistent"
    migrated.save_checkpoint(checkpoint)
    restored = RealRLTLearner.load_checkpoint(
        checkpoint,
        expected_fingerprints={
            "action_schema": PERSISTENT_ACTION_SCHEMA_FINGERPRINT,
            "actor_execution_profile": PERSISTENT_ACTOR_EXECUTION_PROFILE,
        },
    )
    assert restored.update_step == 0
    assert restored.config.actor_execution_profile == PERSISTENT_ACTOR_EXECUTION_PROFILE


def test_persistent_learner_rejects_legacy_replay_and_updates_on_new_contract():
    replay = _persistent_replay()
    with pytest.raises(ValueError, match="cannot be mixed"):
        RealRLTLearner.create(_legacy_replay(), config=_persistent_cfg())

    learner = RealRLTLearner.create(replay, config=_persistent_cfg(policy_delay=1))
    batch = ReplayBatchSampler(replay).sample(4)
    metrics = learner.update(batch)
    candidate = learner.persistent_candidate_action(
        batch["z_rl"],
        batch["state"],
        batch["a_ref"],
        a_base_filtered=batch["a_base_filtered"],
        carry_in=batch["actor_persistent_carry_in"],
        previous_carry=batch["actor_persistent_previous_carry"],
        boundary_anchor=batch["actor_execution_boundary_anchor"],
        filter_alpha=batch["execution_filter_alpha"],
    )

    assert np.isfinite(metrics["critic_loss"])
    assert metrics["actor_updated"] == 1.0
    assert candidate.shape == (4, 10, 7)


def test_actor_runtime_loads_legacy_v3_and_persistent_v4_but_outputs_only_v3(tmp_path):
    legacy_replay = _legacy_replay()
    legacy = RealRLTLearner.create(
        legacy_replay,
        config=dataclasses.replace(
            RealRLTConfig(), hidden_dim=32, projection_dim=16, batch_size=4
        ),
        fingerprints=EXPECTED_CHECKPOINT_FINGERPRINTS,
    )
    legacy_checkpoint = tmp_path / "legacy"
    legacy.save_checkpoint(legacy_checkpoint)
    legacy_actor = JaxCheckpointShadowActor(legacy_checkpoint)
    assert legacy_actor.provenance_metadata() == {
        "actor_checkpoint_action_schema_fingerprint": ACTION_SCHEMA_FINGERPRINT,
        "actor_output_protocol_action_schema_fingerprint": ACTION_SCHEMA_FINGERPRINT,
        "actor_checkpoint_kind": "legacy_v3_rank1_source",
    }

    persistent_fingerprints = {
        **{
            key: value
            for key, value in EXPECTED_CHECKPOINT_FINGERPRINTS.items()
            if key != "action_schema"
        },
        "action_schema": PERSISTENT_ACTION_SCHEMA_FINGERPRINT,
        "actor_execution_profile": PERSISTENT_ACTOR_EXECUTION_PROFILE,
        "execution_filter_profile": PERSISTENT_EXECUTION_FILTER_PROFILE,
        "actor_governor": PERSISTENT_GOVERNOR_PROFILE,
    }
    persistent = RealRLTLearner.warm_start_actor_for_persistent_v2(
        legacy_checkpoint,
        _persistent_replay(),
        config=_persistent_cfg(),
        fingerprints=persistent_fingerprints,
        expected_source_fingerprints=EXPECTED_CHECKPOINT_FINGERPRINTS,
    )
    persistent_checkpoint = tmp_path / "persistent"
    persistent.save_checkpoint(persistent_checkpoint)
    persistent_actor = JaxCheckpointShadowActor(persistent_checkpoint)
    metadata = persistent_actor.provenance_metadata()
    assert metadata["actor_checkpoint_action_schema_fingerprint"] == PERSISTENT_ACTION_SCHEMA_FINGERPRINT
    assert metadata["actor_output_protocol_action_schema_fingerprint"] == ACTION_SCHEMA_FINGERPRINT
    assert metadata["actor_checkpoint_kind"] == "persistent_v4_checkpoint_with_v3_rank1_output"


def test_actor_runtime_rejects_unknown_checkpoint_schema(tmp_path):
    learner = RealRLTLearner.create(
        _legacy_replay(),
        config=dataclasses.replace(
            RealRLTConfig(), hidden_dim=32, projection_dim=16, batch_size=4
        ),
        fingerprints={
            **EXPECTED_CHECKPOINT_FINGERPRINTS,
            "action_schema": "unknown",
        },
    )
    checkpoint = tmp_path / "unknown"
    learner.save_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="unsupported Actor checkpoint action schema"):
        JaxCheckpointShadowActor(checkpoint)
