from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest
from flax import serialization

from openpi.rlt.real.actor_runtime import ACTION_SCHEMA_FINGERPRINT
from openpi.rlt.real.actor_runtime import create_shadow_actor
from openpi.rlt.real.actor_runtime import EXPECTED_CHECKPOINT_FINGERPRINTS
from openpi.rlt.real.agent_jax import (
    RealRLTLearner,
    ReplayBatchSampler,
    ReplaySamplingConfig,
    estimate_residual_limit,
    filter_replay_by_split,
)
from openpi.rlt.real.config import RealRLTConfig


def _config(**changes) -> RealRLTConfig:
    defaults = {
        "hidden_dim": 32,
        "projection_dim": 16,
        "batch_size": 16,
        "reference_dropout": 0.5,
        "seed": 7,
    }
    defaults.update(changes)
    return dataclasses.replace(RealRLTConfig(), **defaults)


def _synthetic_replay(n: int = 64, *, z_dim: int = 8) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(4)
    z_rl = rng.normal(size=(n, z_dim)).astype(np.float32)
    state = rng.normal(scale=0.2, size=(n, 7)).astype(np.float32)
    a_ref = rng.normal(scale=0.15, size=(n, 10, 7)).astype(np.float32)
    correction = 0.03 * np.tanh(z_rl[:, :1, None])
    a_exec = a_ref + correction.astype(np.float32)
    next_z_rl = np.roll(z_rl, -1, axis=0)
    next_state = np.roll(state, -1, axis=0)
    next_a_ref = np.roll(a_ref, -1, axis=0)
    success = np.arange(n) % 2 == 0
    human = np.arange(n) % 3 == 0
    reward = success.astype(np.float32)
    return {
        "episode_id": np.asarray([f"ep_{idx // 8}" for idx in range(n)]),
        "z_rl": z_rl,
        "state": state,
        "a_ref": a_ref,
        "a_exec": a_exec,
        "reward": reward,
        "discount": np.zeros(n, dtype=np.float32),
        "next_z_rl": next_z_rl,
        "next_state": next_state,
        "next_a_ref": next_a_ref,
        "success_mask": success,
        "human_mask": human,
        "source": np.where(human, "human_pika", "pi05"),
        "a_human": np.where(human[:, None, None], a_exec, 0.0).astype(np.float32),
    }


def _batch(replay: dict[str, np.ndarray], size: int = 16) -> dict[str, np.ndarray]:
    return ReplayBatchSampler(replay, config=ReplaySamplingConfig(seed=2)).sample(size)


def test_balanced_sampler_honors_success_failure_and_human_masks():
    replay = _synthetic_replay(120)
    sampler = ReplayBatchSampler(
        replay,
        config=ReplaySamplingConfig(success_fraction=0.7, failure_fraction=0.3, human_fraction=0.4, seed=3),
    )

    batch = sampler.sample(20_000)

    assert abs(float(batch["success_mask"].mean()) - 0.7) < 0.02
    assert abs(float(batch["failure_mask"].mean()) - 0.3) < 0.02
    assert abs(float(batch["human_mask"].mean()) - 0.4) < 0.02


def test_sampler_reduces_per_step_human_mask_to_transition_mask():
    replay = _synthetic_replay(12)
    step_mask = np.zeros((12, 10), dtype=np.bool_)
    step_mask[[1, 5, 9], 3:6] = True
    replay["human_mask"] = step_mask

    sampler = ReplayBatchSampler(replay)

    expected = np.zeros(12, dtype=np.bool_)
    expected[[1, 5, 9]] = True
    np.testing.assert_array_equal(sampler.human_mask, expected)


def test_sampler_synthesizes_optional_human_arrays_for_old_replay():
    replay = _synthetic_replay(12)
    del replay["a_human"]
    del replay["human_mask"]
    del replay["source"]

    batch = ReplayBatchSampler(replay).sample(5)

    assert batch["a_human"].shape == (5, 10, 7)
    assert batch["human_mask"].shape == (5, 10)
    np.testing.assert_array_equal(batch["a_human"], np.zeros((5, 10, 7)))
    np.testing.assert_array_equal(batch["human_mask"], np.zeros((5, 10)))


def test_replay_split_filter_is_whole_episode_and_rejects_leakage():
    replay = _synthetic_replay(16)
    replay["episode_split"] = np.asarray(["train"] * 8 + ["validation"] * 8)

    train = filter_replay_by_split(replay, "train")
    validation = filter_replay_by_split(replay, "validation")

    assert len(train["reward"]) == 8
    assert len(validation["reward"]) == 8
    assert set(train["episode_id"]) == {"ep_0"}
    assert set(validation["episode_id"]) == {"ep_1"}

    replay["episode_split"][7] = "validation"
    with np.testing.assert_raises_regex(ValueError, "span multiple"):
        filter_replay_by_split(replay, "train")


def test_residual_limit_is_estimated_from_actual_executed_corrections():
    replay = _synthetic_replay()

    limit = estimate_residual_limit(replay, percentile=100.0)

    expected = np.max(np.abs(replay["a_exec"] - replay["a_ref"]), axis=0)
    np.testing.assert_allclose(limit, np.maximum(expected, 1e-3), atol=1e-7)


def test_residual_limit_estimator_allows_only_explicit_frozen_gripper_zero():
    replay = _synthetic_replay()
    maximum = np.full((10, 7), 0.05, dtype=np.float32)
    maximum[:, 6] = 0.0

    with np.testing.assert_raises_regex(ValueError, "positive"):
        estimate_residual_limit(replay, maximum=maximum)
    limit = estimate_residual_limit(
        replay,
        maximum=maximum,
        allow_zero_last_action=True,
    )

    assert np.all(limit[:, :6] > 0.0)
    np.testing.assert_array_equal(limit[:, 6], np.zeros(10))


def test_delayed_actor_update_and_target_soft_update():
    replay = _synthetic_replay()
    learner = RealRLTLearner.create(replay, config=_config(policy_delay=2, tau=0.005))
    batch = _batch(replay)
    actor_before = jax.tree_util.tree_map(np.asarray, learner.actor_state.params)
    actor_target_before = jax.tree_util.tree_map(np.asarray, learner.actor_state.target_params)

    first = learner.update(batch)
    actor_after_first = jax.tree_util.tree_map(np.asarray, learner.actor_state.params)
    second = learner.update(batch)
    actor_after_second = jax.tree_util.tree_map(np.asarray, learner.actor_state.params)
    actor_target_after = jax.tree_util.tree_map(np.asarray, learner.actor_state.target_params)

    assert first["actor_updated"] == 0.0
    assert second["actor_updated"] == 1.0
    assert all(
        np.array_equal(a, b) for a, b in zip(jax.tree.leaves(actor_before), jax.tree.leaves(actor_after_first))
    )
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(actor_after_first), jax.tree.leaves(actor_after_second))
    )
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(actor_target_before), jax.tree.leaves(actor_target_after))
    )


def test_synthetic_small_overfit_reduces_terminal_critic_loss():
    replay = _synthetic_replay(32)
    learner = RealRLTLearner.create(
        replay,
        config=_config(critic_lr=1e-3, actor_lr=1e-4, reference_dropout=0.0),
    )
    batch = _batch(replay, 32)

    first = learner.update(batch)["critic_loss"]
    last = first
    for _ in range(79):
        last = learner.update(batch)["critic_loss"]

    assert np.isfinite(last)
    assert last < first * 0.35


def test_checkpoint_roundtrip_restores_optimizer_targets_norm_rng_and_predictions(tmp_path):
    replay = _synthetic_replay()
    learner = RealRLTLearner.create(
        replay,
        config=_config(),
        fingerprints={"base_checkpoint": "full20k", "rl_token": "token-step10000", "replay_sha256": "abc"},
    )
    batch = _batch(replay)
    learner.update(batch)
    learner.update(batch)
    checkpoint = tmp_path / "step_00000002"
    learner.save_checkpoint(checkpoint)

    restored = RealRLTLearner.load_checkpoint(
        checkpoint,
        expected_fingerprints={"base_checkpoint": "full20k", "rl_token": "token-step10000"},
    )
    before = learner.act(batch["z_rl"], batch["state"], batch["a_ref"])
    after = restored.act(batch["z_rl"], batch["state"], batch["a_ref"])
    np.testing.assert_array_equal(before, after)
    assert restored.update_step == learner.update_step
    np.testing.assert_array_equal(restored.normalization.z_rl.mean, learner.normalization.z_rl.mean)
    assert (checkpoint / "learner.msgpack").is_file()
    assert (checkpoint / "metadata.json").is_file()

    # Identical continuation proves optimizer state and RNG were restored, not
    # merely inference parameters.
    learner.update(batch)
    restored.update(batch)
    learner.update(batch)
    restored.update(batch)
    np.testing.assert_allclose(
        learner.act(batch["z_rl"], batch["state"], batch["a_ref"]),
        restored.act(batch["z_rl"], batch["state"], batch["a_ref"]),
        atol=1e-7,
    )


def test_checkpoint_roundtrip_loads_through_actor_shadow_factory(tmp_path):
    replay = _synthetic_replay()
    learner = RealRLTLearner.create(
        replay,
        config=_config(),
        fingerprints=EXPECTED_CHECKPOINT_FINGERPRINTS,
    )
    batch = _batch(replay)
    learner.update(batch)
    learner.update(batch)
    checkpoint = tmp_path / "actor_checkpoint"
    learner.save_checkpoint(checkpoint)

    actor = create_shadow_actor(checkpoint)
    expected = learner.act(batch["z_rl"][0], batch["state"][0], batch["a_ref"][0])[0]
    actual = actor.predict(
        z_rl=batch["z_rl"][0],
        state=batch["state"][0],
        a_ref=batch["a_ref"][0],
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (10, 7)


@pytest.mark.parametrize(
    "fingerprint_name",
    ("base_checkpoint", "rl_token", "phase_classifier", "action_schema"),
)
def test_actor_shadow_factory_rejects_mismatched_training_identity(
    tmp_path, fingerprint_name
):
    replay = _synthetic_replay()
    fingerprints = dict(EXPECTED_CHECKPOINT_FINGERPRINTS)
    fingerprints[fingerprint_name] = "wrong-training-identity"
    learner = RealRLTLearner.create(
        replay,
        config=_config(),
        fingerprints=fingerprints,
    )
    checkpoint = tmp_path / fingerprint_name
    learner.save_checkpoint(checkpoint)

    with pytest.raises(ValueError, match="fingerprint"):
        create_shadow_actor(checkpoint)


def test_critic_is_sensitive_to_candidate_action_after_training():
    replay = _synthetic_replay(32)
    learner = RealRLTLearner.create(replay, config=_config(reference_dropout=0.0))
    batch = _batch(replay, 32)
    for _ in range(20):
        learner.update(batch)

    q_exec = learner.q_values(batch["z_rl"], batch["state"], batch["a_ref"], batch["a_exec"])
    changed_action = batch["a_exec"].copy()
    changed_action[..., 2] += 0.05
    q_changed = learner.q_values(batch["z_rl"], batch["state"], batch["a_ref"], changed_action)

    sensitivity = np.mean(np.abs(q_exec[0] - q_changed[0])) + np.mean(np.abs(q_exec[1] - q_changed[1]))
    assert sensitivity > 1e-6


def test_actor_always_respects_residual_limit():
    replay = _synthetic_replay()
    limit = np.full((10, 7), 0.0125, dtype=np.float32)
    learner = RealRLTLearner.create(replay, config=_config(), residual_limit=limit)
    batch = _batch(replay)
    for _ in range(6):
        learner.update(batch)

    action = learner.act(batch["z_rl"], batch["state"], batch["a_ref"], reference_visible=False)
    residual = action - batch["a_ref"]

    assert np.all(np.abs(residual) <= limit[None] + 1e-6)
    assert np.max(np.abs(residual[..., :6])) <= learner.config.actor_residual_max_rad + 1e-6
    assert np.max(np.abs(np.diff(residual[..., :6], axis=1))) <= learner.config.actor_residual_d1_max_rad + 1e-6
    assert np.max(np.abs(np.diff(residual[..., :6], n=2, axis=1))) <= learner.config.actor_residual_d2_max_rad + 1e-6
    np.testing.assert_array_equal(residual[:, (0, -1), :], np.zeros((len(residual), 2, 7)))
    np.testing.assert_array_equal(residual[..., 6], np.zeros((len(residual), 10)))


def test_actor_bc_regularizer_is_full_chunk_squared_l2_norm():
    replay = _synthetic_replay()
    learner = RealRLTLearner.create(replay, config=_config(reference_dropout=0.0))
    batch = _batch(replay)
    metrics = {}
    for _ in range(4):
        metrics = learner.update(batch)

    assert metrics["actor_updated"] == 1.0
    assert metrics["actor_bc_mse"] > 0.0
    np.testing.assert_allclose(metrics["actor_bc_loss"], metrics["actor_bc_mse"] * 70.0, rtol=1e-5)


def test_actor_optimizes_q1_not_clipped_twin_minimum():
    replay = _synthetic_replay()
    learner = RealRLTLearner.create(
        replay,
        config=_config(policy_delay=1, actor_lr=1e-12, reference_dropout=0.0),
    )
    batch = _batch(replay)

    metrics = learner.update(batch)
    action = learner.act(batch["z_rl"], batch["state"], batch["a_ref"])
    q1, q2 = learner.q_values(batch["z_rl"], batch["state"], batch["a_ref"], action)

    # Metrics are evaluated before the optimizer/target-state replacement, so
    # allow the tiny post-update numerical drift but require an unambiguous Q1
    # match rather than the clipped twin minimum.
    q1_distance = abs(metrics["actor_q_mean"] - float(np.mean(q1)))
    qmin_distance = abs(metrics["actor_q_mean"] - float(np.mean(np.minimum(q1, q2))))
    assert q1_distance < 2e-3
    assert q1_distance < qmin_distance


def test_optional_human_bc_is_additive_and_uses_step_mask():
    replay = _synthetic_replay()
    batch = _batch(replay)
    batch["human_mask"] = np.zeros((len(batch["reward"]), 10), dtype=np.float32)
    batch["human_mask"][:, 2:5] = 1.0
    batch["a_human"] = batch["a_ref"].copy()
    batch["a_human"][:, 2:5, :6] += 0.04
    learner = RealRLTLearner.create(
        replay,
        config=_config(policy_delay=1, beta_human_bc=1.0, reference_dropout=0.0),
    )

    metrics = learner.update(batch)

    assert metrics["actor_human_bc_loss"] > 0.0
    np.testing.assert_allclose(metrics["actor_human_fraction"], 0.3, atol=1e-7)
    assert metrics["actor_bc_loss"] >= 0.0


def test_freeze_gripper_residual_is_exact_and_survives_checkpoint(tmp_path):
    replay = _synthetic_replay()
    learner = RealRLTLearner.create(
        replay,
        config=_config(freeze_gripper_residual=True, policy_delay=1),
        residual_limit=np.full((10, 7), 0.02, dtype=np.float32),
    )
    batch = _batch(replay)
    for _ in range(3):
        learner.update(batch)

    action = learner.act(batch["z_rl"], batch["state"], batch["a_ref"])
    np.testing.assert_array_equal(action[..., 6], batch["a_ref"][..., 6])
    np.testing.assert_array_equal(np.asarray(learner.residual_limit)[..., 6], np.zeros(10))

    checkpoint = tmp_path / "frozen_gripper"
    learner.save_checkpoint(checkpoint)
    restored = RealRLTLearner.load_checkpoint(checkpoint)
    restored_action = restored.act(batch["z_rl"], batch["state"], batch["a_ref"])
    np.testing.assert_array_equal(restored_action[..., 6], batch["a_ref"][..., 6])


def test_checkpoint_without_new_config_fields_loads_with_safe_defaults(tmp_path):
    replay = _synthetic_replay()
    learner = RealRLTLearner.create(
        replay,
        config=_config(
            actor_residual_parameterization="legacy_full_chunk",
            freeze_gripper_residual=False,
        ),
    )
    checkpoint = tmp_path / "legacy_config"
    learner.save_checkpoint(checkpoint)
    learner_path = checkpoint / "learner.msgpack"
    payload = serialization.msgpack_restore(learner_path.read_bytes())
    for name in (
        "beta_human_bc",
        "target_policy_noise_std",
        "target_policy_noise_clip",
        "actor_residual_parameterization",
        "actor_residual_max_rad",
        "actor_residual_d1_max_rad",
        "actor_residual_d2_max_rad",
        "actor_direction_cone_deg",
        "freeze_gripper_residual",
    ):
        payload["config"].pop(name)
    payload["checkpoint_version"] = 1
    learner_path.write_bytes(serialization.msgpack_serialize(payload))

    restored = RealRLTLearner.load_checkpoint(checkpoint)

    assert restored.config.beta_human_bc == 0.0
    assert restored.config.target_policy_noise_std == 0.1
    assert restored.config.target_policy_noise_clip == 0.2
    assert restored.config.actor_residual_parameterization == "legacy_full_chunk"
    assert not restored.config.freeze_gripper_residual


def test_actor_update_reports_hard_rank1_temporal_metrics():
    replay = _synthetic_replay()
    learner = RealRLTLearner.create(replay, config=_config(policy_delay=1, reference_dropout=0.0))

    metrics = learner.update(_batch(replay))

    assert metrics["actor_residual_d1_abs_max"] <= learner.config.actor_residual_d1_max_rad + 1e-7
    assert metrics["actor_residual_d2_abs_max"] <= learner.config.actor_residual_d2_max_rad + 1e-7
    assert metrics["actor_residual_endpoint_abs_max"] == 0.0
    assert metrics["actor_residual_rank1_error_abs_max"] < 1e-7
    assert metrics["target_residual_d1_abs_max"] <= learner.config.actor_residual_d1_max_rad + 1e-7
    assert metrics["target_residual_d2_abs_max"] <= learner.config.actor_residual_d2_max_rad + 1e-7
    assert metrics["target_residual_endpoint_abs_max"] == 0.0
    assert metrics["target_residual_rank1_error_abs_max"] < 1e-7
