from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.rlt.real.agent_jax import admitted_human_gripper_candidate
from openpi.rlt.real.config import GRIPPER_RESIDUAL_CLOSE_ASSIST
from openpi.rlt.real.config import MIN_GRIPPER_CLOSE_CRITIC_BURN_IN_UPDATES
from openpi.rlt.real.config import PERSISTENT_ACTOR_EXECUTION_PROFILE
from openpi.rlt.real.config import RealRLTConfig
from openpi.rlt.real.networks_jax import persistent_filter_residual
from openpi.rlt.real.networks_jax import persistent_governed_residual
from openpi.rlt.real.networks_jax import rank1_direction_limit
from openpi.rlt.real.networks_jax import straight_through_positive


def _config(**changes) -> RealRLTConfig:
    defaults = {
        "actor_execution_profile": PERSISTENT_ACTOR_EXECUTION_PROFILE,
        "chunk_stride": 10,
        "freeze_gripper_residual": False,
        "gripper_residual_mode": GRIPPER_RESIDUAL_CLOSE_ASSIST,
        "beta_human_gripper_bc": 1.0,
        "actor_start_step": MIN_GRIPPER_CLOSE_CRITIC_BURN_IN_UPDATES,
    }
    defaults.update(changes)
    return dataclasses.replace(RealRLTConfig(), **defaults)


def _govern(
    cfg: RealRLTConfig,
    *,
    reference_gripper: np.ndarray | float = 0.02,
    direction_gripper: float = -0.005,
    carry_gripper: float = 0.0,
    previous_gripper: float = 0.0,
) -> np.ndarray:
    a_ref = np.zeros((1, 10, 7), dtype=np.float32)
    a_ref[0, :, 0] = np.linspace(0.001, 0.010, 10)
    a_ref[0, :, 6] = reference_gripper
    direction = np.zeros((1, 7), dtype=np.float32)
    direction[0, 6] = direction_gripper
    carry = np.zeros((1, 7), dtype=np.float32)
    carry[0, 6] = carry_gripper
    previous = np.zeros((1, 7), dtype=np.float32)
    previous[0, 6] = previous_gripper
    boundary = np.zeros((1, 7), dtype=np.float32)
    boundary[0, 6] = float(a_ref[0, 0, 6]) + carry_gripper
    residual, _, _ = persistent_governed_residual(
        jnp.asarray(a_ref),
        jnp.asarray(direction),
        jnp.asarray(carry),
        jnp.asarray(previous),
        jnp.asarray(boundary),
        residual_max_rad=cfg.actor_residual_max_rad,
        d1_max_rad=cfg.actor_residual_d1_max_rad,
        d2_max_rad=cfg.actor_residual_d2_max_rad,
        direction_cone_deg=cfg.actor_direction_cone_deg,
        boundary_jump_max_rad=cfg.actor_max_boundary_jump_rad,
        direction_static_threshold_rad=cfg.actor_direction_static_threshold_rad,
        projection_scale_steps=cfg.actor_projection_scale_steps,
        gripper_residual_mode=cfg.gripper_residual_mode,
        gripper_residual_max_close_m=cfg.actor_gripper_residual_max_close_m,
        gripper_d1_max_m=cfg.actor_gripper_residual_d1_max_m,
        gripper_d2_max_m=cfg.actor_gripper_residual_d2_max_m,
        gripper_boundary_jump_max_m=cfg.actor_gripper_max_boundary_jump_m,
        gripper_command_min_m=cfg.gripper_command_min_m,
        gripper_command_max_m=cfg.gripper_command_max_m,
        gripper_release_reference_m=cfg.gripper_release_reference_m,
        gripper_release_delta_m=cfg.gripper_release_delta_m,
    )
    return np.asarray(residual[0])


def test_close_assist_requires_explicit_one_way_mode() -> None:
    with pytest.raises(ValueError, match="disabled"):
        _config(freeze_gripper_residual=True)
    with pytest.raises(ValueError, match="persistent"):
        _config(actor_execution_profile="rank1_bump_v1")


def test_rank1_head_gets_a_separate_metre_limit() -> None:
    cfg = _config()
    replay_limit = jnp.full((10, 7), 0.05)

    direction_limit = np.asarray(rank1_direction_limit(replay_limit, cfg))

    np.testing.assert_allclose(
        direction_limit[:6],
        cfg.actor_residual_max_rad,
        atol=1e-8,
    )
    assert direction_limit[6] == pytest.approx(
        cfg.actor_gripper_residual_max_close_m
    )


def test_one_sided_gripper_head_keeps_recovery_gradient() -> None:
    values = jnp.asarray([-0.4, 0.0, 0.4], dtype=jnp.float32)

    forward = np.asarray(straight_through_positive(values))
    gradient = np.asarray(
        jax.grad(lambda x: jnp.sum(straight_through_positive(x)))(values)
    )

    np.testing.assert_allclose(forward, [0.0, 0.0, 0.4], atol=1e-7)
    np.testing.assert_allclose(gradient, [1.0, 1.0, 1.0], atol=1e-7)


def test_gripper_close_knot_is_one_sided_rate_limited_and_persistent() -> None:
    cfg = _config()

    first = _govern(cfg)
    first_gripper = first[:, 6]

    assert first_gripper[0] == pytest.approx(0.0, abs=1e-9)
    assert first_gripper[-1] < -1e-4
    assert np.max(first_gripper) <= 1e-9
    assert np.min(first_gripper) >= -cfg.actor_gripper_residual_max_close_m - 1e-9
    assert (
        np.max(np.abs(np.diff(np.r_[0.0, first_gripper])))
        <= cfg.actor_gripper_residual_d1_max_m + 1e-8
    )

    second = _govern(
        cfg,
        carry_gripper=float(first_gripper[-1]),
        previous_gripper=float(first_gripper[-2]),
    )
    assert second[-1, 6] < first_gripper[-1]
    assert second[0, 6] == pytest.approx(first_gripper[-1], abs=1e-8)


def test_pi05_release_intent_forces_close_carry_toward_zero() -> None:
    cfg = _config()
    opening_reference = np.linspace(0.02, 0.065, 10, dtype=np.float32)

    release = _govern(
        cfg,
        reference_gripper=opening_reference,
        direction_gripper=-cfg.actor_gripper_residual_max_close_m,
        carry_gripper=-0.002,
        previous_gripper=-0.002,
    )

    assert release[0, 6] == pytest.approx(-0.002, abs=1e-8)
    assert release[-1, 6] > release[0, 6]
    assert np.max(release[:, 6]) <= 1e-9
    command = opening_reference + release[:, 6]
    assert np.min(command) >= cfg.gripper_command_min_m - 1e-8
    assert np.max(command) <= cfg.gripper_command_max_m + 1e-8


def test_native_filter_does_not_double_filter_governed_gripper() -> None:
    planned = np.zeros((1, 10, 7), dtype=np.float32)
    planned[0, :, 0] = 0.004
    planned[0, :, 6] = np.linspace(0.0, -0.002, 10)
    carry = np.zeros((1, 7), dtype=np.float32)
    alpha = np.full((1, 10), 0.5, dtype=np.float32)

    filtered = np.asarray(
        persistent_filter_residual(
            jnp.asarray(planned),
            jnp.asarray(carry),
            jnp.asarray(alpha),
            freeze_gripper_residual=False,
        )
    )

    np.testing.assert_array_equal(filtered[..., 6], planned[..., 6])
    assert filtered[0, 0, 0] == pytest.approx(0.002)


def test_admitted_human_candidate_uses_exact_persistent_gripper_contract() -> None:
    cfg = _config()
    a_ref = np.zeros((2, 10, 7), dtype=np.float32)
    a_ref[..., 0] = np.linspace(0.001, 0.010, 10)
    a_ref[..., 6] = 0.02
    a_base = a_ref.copy()
    actor_direction = np.zeros((2, 7), dtype=np.float32)
    actor_direction[:, 1] = 0.001
    a_human = a_base.copy()
    # Deliberately discontinuous raw Pika values.  They must be reduced to one
    # target knot and governed, never passed directly to the Critic.
    a_human[:, ::2, 6] = 0.0
    human_mask = np.zeros((2, 10), dtype=np.float32)
    human_mask[:, ::2] = 1.0
    carry = np.zeros((2, 7), dtype=np.float32)
    previous = np.zeros((2, 7), dtype=np.float32)
    boundary = a_base[:, 0, :].copy()
    alpha = np.full((2, 10), 0.5, dtype=np.float32)

    candidate, human_direction, execution = (
        admitted_human_gripper_candidate(
            jnp.asarray(actor_direction),
            jnp.asarray(a_human),
            jnp.asarray(human_mask),
            jnp.asarray(a_ref),
            jnp.asarray(a_base),
            jnp.asarray(carry),
            jnp.asarray(previous),
            jnp.asarray(boundary),
            jnp.asarray(alpha),
            cfg,
        )
    )
    candidate = np.asarray(candidate)
    human_direction = np.asarray(human_direction)
    residual = np.asarray(execution["filtered_residual"])
    gripper_residual = residual[..., 6]

    np.testing.assert_allclose(
        human_direction[:, 6],
        -cfg.actor_gripper_residual_max_close_m,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        human_direction[:, :6],
        actor_direction[:, :6],
        atol=1e-8,
    )
    assert np.max(gripper_residual) <= 1e-9
    assert (
        np.min(gripper_residual)
        >= -cfg.actor_gripper_residual_max_close_m - 1e-9
    )
    assert (
        np.max(np.abs(np.diff(gripper_residual, axis=1)))
        <= cfg.actor_gripper_residual_d1_max_m + 1e-8
    )
    assert (
        np.max(np.abs(np.diff(gripper_residual, n=2, axis=1)))
        <= cfg.actor_gripper_residual_d2_max_m + 1e-8
    )
    assert np.min(candidate[..., 6]) >= cfg.gripper_command_min_m - 1e-9
    assert np.max(candidate[..., 6]) <= cfg.gripper_command_max_m + 1e-9


@pytest.mark.parametrize(
    ("reference_gripper", "carry_gripper", "previous_gripper"),
    (
        (np.full(10, 0.02, dtype=np.float32), 0.0, 0.0),
        (np.linspace(0.02, 0.065, 10, dtype=np.float32), -0.002, -0.002),
    ),
)
def test_jax_and_runtime_gripper_governors_are_numerically_identical(
    reference_gripper: np.ndarray,
    carry_gripper: float,
    previous_gripper: float,
) -> None:
    runtime = pytest.importorskip("piper_runtime.rlt_residual_governor")
    cfg = _config()
    a_ref = np.zeros((1, 10, 7), dtype=np.float32)
    a_ref[0, :, 0] = np.linspace(0.002, 0.020, 10)
    a_ref[0, :, 6] = reference_gripper
    direction = np.zeros((1, 7), dtype=np.float32)
    direction[0, 1] = 0.001
    direction[0, 6] = -0.005
    carry = np.zeros((1, 7), dtype=np.float32)
    carry[0, 6] = carry_gripper
    previous = np.zeros((1, 7), dtype=np.float32)
    previous[0, 6] = previous_gripper
    boundary = np.zeros((1, 7), dtype=np.float32)
    boundary[0, 6] = float(reference_gripper[0]) + carry_gripper
    jax_residual, _, _ = persistent_governed_residual(
        jnp.asarray(a_ref),
        jnp.asarray(direction),
        jnp.asarray(carry),
        jnp.asarray(previous),
        jnp.asarray(boundary),
        residual_max_rad=cfg.actor_residual_max_rad,
        d1_max_rad=cfg.actor_residual_d1_max_rad,
        d2_max_rad=cfg.actor_residual_d2_max_rad,
        direction_cone_deg=cfg.actor_direction_cone_deg,
        boundary_jump_max_rad=cfg.actor_max_boundary_jump_rad,
        direction_static_threshold_rad=cfg.actor_direction_static_threshold_rad,
        projection_scale_steps=cfg.actor_projection_scale_steps,
        gripper_residual_mode=cfg.gripper_residual_mode,
        gripper_residual_max_close_m=cfg.actor_gripper_residual_max_close_m,
        gripper_d1_max_m=cfg.actor_gripper_residual_d1_max_m,
        gripper_d2_max_m=cfg.actor_gripper_residual_d2_max_m,
        gripper_boundary_jump_max_m=cfg.actor_gripper_max_boundary_jump_m,
        gripper_command_min_m=cfg.gripper_command_min_m,
        gripper_command_max_m=cfg.gripper_command_max_m,
        gripper_release_reference_m=cfg.gripper_release_reference_m,
        gripper_release_delta_m=cfg.gripper_release_delta_m,
    )
    raw_actor = a_ref[0].copy()
    raw_actor += runtime.RANK1_BUMP_WINDOW[:, None] * direction[0][None, :]
    runtime_plan = runtime.govern_persistent_actor_plan(
        plan_id="parity",
        behavior_plan_id="behavior",
        behavior_start_offset=0,
        behavior_ref=a_ref[0],
        raw_actor=raw_actor,
        boundary_anchor=boundary[0],
        carry_in=carry[0],
        previous_carry=previous[0],
        config=runtime.ActorResidualGovernorConfig(
            max_boundary_jump_rad=cfg.actor_max_boundary_jump_rad,
            gripper_residual_mode=runtime.GRIPPER_RESIDUAL_CLOSE_ASSIST,
            gripper_residual_max_close_m=cfg.actor_gripper_residual_max_close_m,
            gripper_residual_d1_max_m=cfg.actor_gripper_residual_d1_max_m,
            gripper_residual_d2_max_m=cfg.actor_gripper_residual_d2_max_m,
            gripper_max_boundary_jump_m=cfg.actor_gripper_max_boundary_jump_m,
            gripper_command_min_m=cfg.gripper_command_min_m,
            gripper_command_max_m=cfg.gripper_command_max_m,
            gripper_release_reference_m=cfg.gripper_release_reference_m,
            gripper_release_delta_m=cfg.gripper_release_delta_m,
        ),
    )

    assert runtime_plan.approved
    np.testing.assert_allclose(
        runtime_plan.safe_residual,
        np.asarray(jax_residual[0]),
        rtol=0.0,
        atol=2e-7,
    )
