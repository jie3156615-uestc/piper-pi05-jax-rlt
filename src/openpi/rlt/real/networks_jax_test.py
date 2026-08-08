import jax
import jax.numpy as jnp
import numpy as np

from openpi.rlt.real.config import RealRLTConfig
from openpi.rlt.real.networks_jax import (
    ResidualActor,
    TwinCritic,
    apply_direction_cone_scale,
    rank1_bump_residual,
    rank1_bump_window,
    rank1_direction_from_residual,
    rank1_direction_limit,
    residual_temporal_metrics,
)


def test_residual_actor_initial_output_is_close_to_original_reference():
    cfg = RealRLTConfig()
    actor = ResidualActor(cfg)
    direction_limit = rank1_direction_limit(jnp.full((10, 7), 0.05), cfg)
    params = actor.init(
        jax.random.PRNGKey(0),
        jnp.ones((2, 32)),
        jnp.ones((2, 7)),
        jnp.zeros((2, 10, 7)),
        jnp.ones((2, 10, 7)),
        direction_limit,
    )

    action = actor.apply(
        params,
        jnp.ones((2, 32)),
        jnp.ones((2, 7)),
        jnp.zeros((2, 10, 7)),
        jnp.ones((2, 10, 7)),
        direction_limit,
    )

    assert action.shape == (2, 10, 7)
    assert float(jnp.max(jnp.abs(action - 1.0))) < 1e-5


def test_rank1_bump_has_exact_endpoints_and_hard_rmax_d1_d2_limits():
    cfg = RealRLTConfig()
    direction_limit = rank1_direction_limit(jnp.full((10, 7), 0.05), cfg)
    direction = jnp.broadcast_to(direction_limit, (3, 7))
    residual = rank1_bump_residual(direction)
    metrics = residual_temporal_metrics(residual)

    np.testing.assert_allclose(
        np.asarray(rank1_bump_window()),
        [0.0, 0.2, 0.5, 0.8, 1.0, 1.0, 0.8, 0.5, 0.2, 0.0],
        rtol=0.0,
        atol=1e-7,
    )
    assert float(jnp.max(jnp.abs(residual[..., :6]))) <= cfg.actor_residual_max_rad + 1e-7
    assert float(metrics["residual_d1_abs_max"]) <= cfg.actor_residual_d1_max_rad + 1e-7
    assert float(metrics["residual_d2_abs_max"]) <= cfg.actor_residual_d2_max_rad + 1e-7
    assert float(metrics["residual_endpoint_abs_max"]) == 0.0
    assert float(metrics["residual_rank1_error_abs_max"]) < 1e-7
    assert bool(jnp.all(direction_limit[:6] == cfg.actor_residual_max_rad))
    assert float(direction_limit[6]) == 0.0
    recovered = rank1_direction_from_residual(residual)
    assert bool(jnp.allclose(recovered, direction, atol=1e-7))


def test_direction_cone_uses_one_whole_chunk_scale_and_preserves_rank1():
    a_ref = jnp.zeros((1, 10, 7))
    a_ref = a_ref.at[0, :, 0].set(0.002 * jnp.arange(1, 11))
    direction = jnp.asarray([[0.0, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0]])
    residual = rank1_bump_residual(direction)

    safe_residual, scale = apply_direction_cone_scale(a_ref, residual, cone_deg=15.0)

    assert 0.0 < float(scale[0]) < 1.0
    recovered = rank1_direction_from_residual(safe_residual)
    assert bool(jnp.allclose(safe_residual, rank1_bump_residual(recovered), atol=1e-8))
    ref_velocity = jnp.diff(jnp.concatenate([jnp.zeros((1, 1, 6)), a_ref[..., :6]], axis=1), axis=1)
    candidate = a_ref + safe_residual
    candidate_velocity = jnp.diff(
        jnp.concatenate([jnp.zeros((1, 1, 6)), candidate[..., :6]], axis=1), axis=1
    )
    dot = jnp.sum(ref_velocity * candidate_velocity, axis=-1)
    cosine = dot / (jnp.linalg.norm(ref_velocity, axis=-1) * jnp.linalg.norm(candidate_velocity, axis=-1))
    assert bool(jnp.all(dot >= -1e-10))
    assert bool(jnp.all(cosine >= jnp.cos(jnp.deg2rad(15.0)) - 1e-6))


def test_direction_cone_caps_candidate_velocity_when_reference_is_static():
    a_ref = jnp.zeros((1, 10, 7))
    residual = rank1_bump_residual(jnp.asarray([[0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))

    safe_residual, scale = apply_direction_cone_scale(a_ref, residual, cone_deg=15.0)

    candidate_velocity = jnp.diff(
        jnp.concatenate([jnp.zeros((1, 1, 6)), safe_residual[..., :6]], axis=1), axis=1
    )
    assert 0.0 < float(scale[0]) < 1.0
    assert float(jnp.max(jnp.linalg.norm(candidate_velocity, axis=-1))) <= 1e-3 + 1e-7


def test_twin_critic_depends_on_candidate_action():
    cfg = RealRLTConfig()
    critic = TwinCritic(cfg)
    params = critic.init(
        jax.random.PRNGKey(1),
        jnp.ones((2, 32)),
        jnp.ones((2, 7)),
        jnp.ones((2, 10, 7)),
        jnp.zeros((2, 10, 7)),
    )

    q_zero = critic.apply(
        params,
        jnp.ones((2, 32)),
        jnp.ones((2, 7)),
        jnp.ones((2, 10, 7)),
        jnp.zeros((2, 10, 7)),
    )
    q_one = critic.apply(
        params,
        jnp.ones((2, 32)),
        jnp.ones((2, 7)),
        jnp.ones((2, 10, 7)),
        jnp.ones((2, 10, 7)),
    )

    assert q_zero[0].shape == (2,)
    assert q_zero[1].shape == (2,)
    assert bool(jnp.any(jnp.abs(q_zero[0] - q_one[0]) > 1e-7))
