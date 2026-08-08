import pytest

from openpi.rlt.real.config import (
    HUMAN_GRIPPER_Q_FILTER_CRITIC_MIN_ADVANTAGE,
    MIN_GRIPPER_CLOSE_CRITIC_BURN_IN_UPDATES,
    GRIPPER_RESIDUAL_CLOSE_ASSIST,
    PERSISTENT_ACTOR_EXECUTION_PROFILE,
    RealRLTConfig,
    Source,
)


def test_default_real_rlt_config_matches_hardware_contract():
    cfg = RealRLTConfig()

    assert cfg.control_hz == 30
    assert cfg.action_horizon == 50
    assert cfg.chunk_length == 10
    assert cfg.execute_steps == 50
    assert cfg.n_step == 10
    assert cfg.action_dim == 7
    assert cfg.state_dim == 7
    assert cfg.reference_dropout == 0.5
    assert cfg.actor_output_mode == "residual"
    assert cfg.actor_residual_parameterization == "rank1_bump"
    assert cfg.actor_residual_max_rad == 0.005
    assert cfg.actor_residual_d1_max_rad == 0.0015
    assert cfg.actor_residual_d2_max_rad == 0.001
    assert cfg.actor_direction_cone_deg == 15.0
    assert cfg.beta_bc == 1.0
    assert cfg.beta_human_bc == 0.0
    assert cfg.actor_start_step == 0
    assert (
        cfg.human_gripper_q_filter_mode
        == HUMAN_GRIPPER_Q_FILTER_CRITIC_MIN_ADVANTAGE
    )
    assert cfg.human_gripper_q_filter_margin == 0.0
    assert cfg.target_policy_noise_std == 0.1
    assert cfg.target_policy_noise_clip == 0.2
    assert cfg.freeze_gripper_residual
    assert cfg.chunk_stride == 2


def test_source_names_are_stable_for_replay_logs():
    assert Source.PI05 == "pi05"
    assert Source.RLT == "rlt"
    assert Source.HUMAN_PIKA == "human_pika"
    assert Source.STOP == "stop"
    assert Source.SAFETY_BLOCK == "safety_block"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"gamma": 1.01}, "gamma"),
        ({"tau": 0.0}, "tau"),
        ({"policy_delay": 0}, "policy_delay"),
        ({"chunk_length": 0}, "chunk_length"),
        ({"actor_lr": 0.0}, "actor_lr"),
        ({"critic_lr": -1e-4}, "critic_lr"),
        ({"batch_size": 0}, "batch_size"),
        ({"actor_output_mode": "absolute"}, "actor_output_mode"),
        ({"actor_residual_parameterization": "full_chunk"}, "actor_residual_parameterization"),
        ({"actor_residual_max_rad": 0.0}, "actor_residual_max_rad"),
        ({"actor_residual_d1_max_rad": 0.0}, "actor_residual_d1_max_rad"),
        ({"actor_residual_d2_max_rad": -1.0}, "actor_residual_d2_max_rad"),
        ({"actor_direction_cone_deg": 0.0}, "actor_direction_cone_deg"),
        ({"actor_direction_cone_deg": 91.0}, "actor_direction_cone_deg"),
        ({"freeze_gripper_residual": False}, "freeze_gripper_residual"),
        (
            {"human_gripper_q_filter_mode": "reward_only"},
            "human_gripper_q_filter_mode",
        ),
        (
            {"human_gripper_q_filter_margin": -0.1},
            "human_gripper_q_filter_margin",
        ),
        (
            {"human_gripper_q_filter_margin": float("nan")},
            "human_gripper_q_filter_margin",
        ),
        ({"actor_start_step": -1}, "actor_start_step"),
    ],
)
def test_real_rlt_config_rejects_unsafe_values(changes, message):
    with pytest.raises(ValueError, match=message):
        RealRLTConfig(**changes)


def test_legacy_parameterization_remains_loadable_for_read_only_reproduction():
    cfg = RealRLTConfig(
        actor_residual_parameterization="legacy_full_chunk",
        freeze_gripper_residual=False,
    )

    assert cfg.actor_residual_parameterization == "legacy_full_chunk"
    assert not cfg.freeze_gripper_residual


def test_close_assist_requires_full_fresh_critic_burn_in() -> None:
    common = {
        "actor_execution_profile": PERSISTENT_ACTOR_EXECUTION_PROFILE,
        "chunk_stride": 10,
        "freeze_gripper_residual": False,
        "gripper_residual_mode": GRIPPER_RESIDUAL_CLOSE_ASSIST,
    }
    with pytest.raises(ValueError, match="Critic-only burn-in"):
        RealRLTConfig(
            **common,
            actor_start_step=MIN_GRIPPER_CLOSE_CRITIC_BURN_IN_UPDATES - 1,
        )

    cfg = RealRLTConfig(
        **common,
        actor_start_step=MIN_GRIPPER_CLOSE_CRITIC_BURN_IN_UPDATES,
    )
    assert cfg.actor_start_step == MIN_GRIPPER_CLOSE_CRITIC_BURN_IN_UPDATES
