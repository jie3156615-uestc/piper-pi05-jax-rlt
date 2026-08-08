from __future__ import annotations

import dataclasses
from typing import Literal


@dataclasses.dataclass(frozen=True)
class RLTTokenConfig:
    """Configuration for the RL-token bottleneck module."""

    embedding_dim: int = 2048
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_seq_len: int = 1024


@dataclasses.dataclass(frozen=True)
class RLTActorCriticConfig:
    """Configuration for the lightweight online actor-critic."""

    state_dim: int = 8
    action_dim: int = 7
    chunk_length: int = 5
    hidden_dim: int = 256
    num_hidden_layers: int = 2
    fixed_std: float = 0.05
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    beta_bc: float = 1.0
    reference_dropout: float = 0.5
    actor_output_mode: Literal["absolute", "residual"] = "absolute"
    absolute_delta_clip: float = 0.0
    residual_scale: float = 0.05
    target_policy_noise: float = 0.0
    target_noise_clip: float = 0.1
    grad_clip_norm: float = 10.0


@dataclasses.dataclass(frozen=True)
class RLTReplayConfig:
    capacity: int = 200_000
    batch_size: int = 256
    chunk_stride: int = 2
    positive_sample_fraction: float = 0.0
    demo_sample_fraction: float = 0.0
    online_positive_sample_fraction: float = 0.0
    # Retained for backwards CLI/checkpoint compatibility. Chunk returns are
    # controlled by actor_critic.chunk_length in the current RLT implementation.
    n_step: int = 10


@dataclasses.dataclass(frozen=True)
class RLTLiberoTrainConfig:
    """Top-level settings for LIBERO RLT simulation training."""

    train_config_name: str = "pi05_libero"
    checkpoint_dir: str = "gs://openpi-assets/checkpoints/pi05_libero"
    rlt_token_checkpoint: str | None = None
    actor_critic_checkpoint: str | None = None
    resume_checkpoint: str | None = None
    output_dir: str = "checkpoints/rlt/pi05_libero"
    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    seed: int = 7
    device: str = "cuda"
    num_episodes: int = 200
    warmup_episodes: int = 20
    max_steps: int | None = None
    num_steps_wait: int = 10
    action_horizon: int = 10
    updates_per_env_step: int = 5
    start_learning_after: int = 1_000
    eval_interval_episodes: int = 25
    save_interval_episodes: int = 25
    critical_phase_mode: Literal["full", "window"] = "full"
    critical_start_step: int = 0
    critical_end_step: int | None = None
    wandb_enabled: bool = False
    project_name: str = "openpi-rlt"


@dataclasses.dataclass(frozen=True)
class RLTIsaacLabTrainConfig:
    """Top-level settings for Isaac Lab RLT simulation training."""

    train_config_name: str = "pi05_isaac_lab_block_stacking_mimic_stack_aligned"
    checkpoint_dir: str = (
        "checkpoints/pi05_isaac_lab_block_stacking_mimic_stack_aligned/"
        "isaac_mimic_stack_aligned_pi05_bs16_30k_20260518_071546/30000"
    )
    rlt_token_checkpoint: str | None = None
    actor_critic_checkpoint: str | None = None
    resume_checkpoint: str | None = None
    output_dir: str = "checkpoints/rlt/pi05_isaac_lab_stack_absolute"
    env_host: str = "127.0.0.1"
    env_port: int = 8792
    prompt: str = "stack the cubes"
    allow_prompt_mismatch: bool = False
    seed: int = 7
    device: str = "cuda"
    num_episodes: int = 200
    warmup_episodes: int = 20
    min_warmup_success_episodes: int = 0
    warmup_success_only_replay: bool = False
    max_warmup_attempt_episodes: int = 0
    cache_warmup_rollouts: bool = False
    warmup_cache_dir: str | None = None
    max_steps: int = 1800
    action_horizon: int = 10
    updates_per_env_step: int = 5
    online_update_interval_steps: int = 1
    start_learning_after: int = 1_000
    offline_pretrain_updates: int = 0
    rl_takeover_ramp_episodes: int = 0
    rl_takeover_max_probability: float = 1.0
    rl_takeover_success_gate: float = 0.0
    rl_takeover_gate_window: int = 0
    rl_takeover_gate_increment: float = 0.25
    feature_batch_size: int = 16
    save_interval_episodes: int = 25
    reward_success: float = 1.0
    reward_failure: float = 0.0
    env_connect_timeout_s: float = 120.0
    env_request_timeout_s: float = 240.0
    num_envs: int = 1
    vectorized_rollout: bool = False
    enable_diagnostics: bool = True
    diagnostic_window_steps: int = 200
    diagnostic_min_steps: int = 200
    diagnostic_no_grasp_steps: int = 900
    diagnostic_no_stack_steps: int = 1200
    diagnostic_lost_subtask_steps: int = 400
    diagnostic_stuck_action_norm: float = 0.03
    diagnostic_stuck_eef_range_m: float = 0.015
    diagnostic_stuck_cube_range_m: float = 0.01
    diagnostic_trace_stride: int = 10
    early_abort_on_diagnostic_failure: bool = False
    early_abort_min_steps: int = 900
    wandb_enabled: bool = False
    project_name: str = "openpi-rlt-isaac"
