import jax
import jax.numpy as jnp
import numpy as np

from openpi.rlt.real.agent_jax import (
    apply_reference_dropout,
    apply_target_policy_smoothing,
    create_train_state,
    critic_batch_view,
)
from openpi.rlt.real.config import RealRLTConfig
from openpi.rlt.real.networks_jax import (
    rank1_bump_residual,
    rank1_direction_from_residual,
    residual_temporal_metrics,
)


def test_reference_dropout_only_changes_actor_input():
    a_ref = jnp.ones((8, 10, 7))
    dropped = apply_reference_dropout(jax.random.PRNGKey(0), a_ref, dropout=1.0)

    np.testing.assert_allclose(np.asarray(dropped), np.zeros((8, 10, 7)))
    np.testing.assert_allclose(np.asarray(a_ref), np.ones((8, 10, 7)))


def test_critic_batch_view_uses_executed_action_not_actor_action():
    batch = {
        "a_ref": jnp.ones((2, 10, 7)),
        "a_exec": jnp.full((2, 10, 7), 2.0),
        "a_actor": jnp.full((2, 10, 7), 9.0),
    }

    view = critic_batch_view(batch)

    np.testing.assert_allclose(np.asarray(view["candidate_action"]), np.full((2, 10, 7), 2.0))


def test_create_train_state_smoke():
    cfg = RealRLTConfig()
    residual_limit = jnp.full((10, 7), 0.05)

    actor_state, critic_state = create_train_state(jax.random.PRNGKey(2), cfg, z_dim=32, residual_limit=residual_limit)

    assert actor_state.params is not None
    assert critic_state.params is not None


def test_target_policy_smoothing_is_clipped_and_respects_frozen_dimension():
    a_ref = jnp.zeros((4, 10, 7), dtype=jnp.float32)
    a_ref = a_ref.at[:, :, 0].set(0.01 * jnp.arange(1, 11))
    direction = jnp.asarray([[0.002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    action = a_ref + rank1_bump_residual(jnp.broadcast_to(direction, (4, 7)))
    limit = jnp.asarray([0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.0])

    smoothed = apply_target_policy_smoothing(
        jax.random.PRNGKey(11),
        action,
        a_ref,
        limit,
        noise_std=jnp.asarray(10.0),
        noise_clip=jnp.asarray(0.2),
    )

    residual = smoothed - a_ref
    recovered = rank1_direction_from_residual(residual)
    temporal = residual_temporal_metrics(residual)
    assert bool(jnp.all(jnp.abs(recovered[..., :6]) <= 0.005 + 1e-7))
    assert bool(jnp.allclose(residual, rank1_bump_residual(recovered), atol=1e-7))
    assert float(temporal["residual_d1_abs_max"]) <= 0.0015 + 1e-7
    assert float(temporal["residual_d2_abs_max"]) <= 0.001 + 1e-7
    assert float(temporal["residual_endpoint_abs_max"]) == 0.0
    np.testing.assert_array_equal(np.asarray(smoothed[..., 6]), np.asarray(a_ref[..., 6]))
