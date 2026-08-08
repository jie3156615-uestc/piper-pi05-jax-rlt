from __future__ import annotations

import dataclasses
import math


LEGACY_ACTOR_EXECUTION_PROFILE = "rank1_bump_v1"
PERSISTENT_ACTOR_EXECUTION_PROFILE = "persistent_c10_filtered_actual_v2"
HUMAN_EXECUTION_PROFILE = "human_pika_filtered_actual_v2"
PERSISTENT_EXECUTION_FILTER_PROFILE = "exp_one_minus_exp_neg_dt_over_tau_v1"
RANK1_ACTOR_OUTPUT_SCHEMA_FINGERPRINT = (
    "piper_joint_delta_v3_c10_n10_stride2_behavior_ref50_"
    "rank1_bump_r005_d1_0015_d2_001_cone15_"
    "gripper_absolute_frozen_residual"
)
PERSISTENT_ACTION_SCHEMA_FINGERPRINT = (
    "piper_joint_delta_v4_c10_n10_stride10_behavior_ref50_"
    "persistent_filtered_actual_r005_d1_0015_d2_001_cone15_"
    "gripper_absolute_frozen_residual"
)
PERSISTENT_GOVERNOR_PROFILE = (
    "persistent_governor_v2_r005_d1_0015_d2_001_cone15_"
    "boundary060_static001_scale33_min020"
)
PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT = (
    "piper_joint_delta_v5_c10_n10_stride10_behavior_ref50_"
    "persistent_filtered_actual_r005_d1_0015_d2_001_cone15_"
    "gripper_close_assist_r005_d1_0005_d2_0003_boundary0005"
)
RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT = (
    "piper_joint_delta_v5_c10_n10_stride10_behavior_ref50_"
    "rank1_joint_r005_d1_0015_d2_001_cone15_"
    "gripper_close_knot_r005"
)
RANK1_GRIPPER_CLOSE_PROJECTION_PROFILE = (
    "rank1_joint_v1_r005_d1_0015_d2_001_cone15_"
    "scale33_min020_gripper_close_knot_r005"
)
PERSISTENT_GRIPPER_CLOSE_GOVERNOR_PROFILE = (
    "persistent_governor_v3_joint_r005_d1_0015_d2_001_cone15_"
    "boundary060_gripper_close_r005_d1_0005_d2_0003_boundary0005"
)
GRIPPER_RESIDUAL_FROZEN = "frozen"
GRIPPER_RESIDUAL_CLOSE_ASSIST = "close_only_persistent_v1"
HUMAN_GRIPPER_Q_FILTER_CRITIC_MIN_ADVANTAGE = "critic_min_advantage_v1"
MIN_GRIPPER_CLOSE_CRITIC_BURN_IN_UPDATES = 144


class Source:
    PI05 = "pi05"
    RLT = "rlt"
    HUMAN_PIKA = "human_pika"
    STOP = "stop"
    SAFETY_BLOCK = "safety_block"


@dataclasses.dataclass(frozen=True)
class RealRLTConfig:
    control_hz: int = 30
    action_horizon: int = 50
    chunk_length: int = 10
    # Pi0.5 keeps one H=50 behavior slice for all five governed C=10 Actor
    # windows.  ``chunk_length`` is the Actor horizon; ``execute_steps`` is the
    # behavior-policy execution horizon and must not be confused with C.
    execute_steps: int = 50
    n_step: int = 10
    chunk_stride: int = 2
    state_dim: int = 7
    action_dim: int = 7
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    # Actor/Q-filter updates are disabled through this absolute learner step.
    # Checkpoint resume preserves update_step, so burn-in is never repeated.
    actor_start_step: int = 0
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    weight_decay: float = 1e-4
    beta_bc: float = 1.0
    beta_human_bc: float = 0.0
    # This term is deliberately separate from ``beta_human_bc``.  It only
    # behavior-clones dim 6 on admitted human transitions that pass the
    # Critic advantage filter, so enabling gripper imitation cannot silently
    # change the six governed arm joints.  Reward 0 is a negative Critic label,
    # not a reason to discard an admitted human transition.
    beta_human_gripper_bc: float = 0.0
    human_gripper_bc_scale_m: float = 0.005
    human_gripper_q_filter_mode: str = (
        HUMAN_GRIPPER_Q_FILTER_CRITIC_MIN_ADVANTAGE
    )
    human_gripper_q_filter_margin: float = 0.0
    reference_dropout: float = 0.5
    # TD3 target smoothing is expressed as a fraction of each action element's
    # residual limit.  This keeps joint-radian and gripper-metre scales separate.
    target_policy_noise_std: float = 0.1
    target_policy_noise_clip: float = 0.2
    actor_output_mode: str = "residual"
    # V3 real-robot actor contract.  The actor predicts one joint-space
    # direction and applies it through the fixed C10 bump in networks_jax.
    # These limits are hard parameterization limits, not training penalties.
    actor_residual_parameterization: str = "rank1_bump"
    # Execution is separate from the Actor head parameterization.  Legacy
    # checkpoints keep their unchanged seven-output rank1 direction head.  A
    # persistent-v2 learner explicitly maps that direction through the same
    # carry-aware governor and low-pass used on the robot.
    actor_execution_profile: str = LEGACY_ACTOR_EXECUTION_PROFILE
    execution_filter_profile: str = PERSISTENT_EXECUTION_FILTER_PROFILE
    execution_filter_tau_s: float = 0.05
    actor_residual_max_rad: float = 0.005
    actor_residual_d1_max_rad: float = 0.0015
    actor_residual_d2_max_rad: float = 0.001
    actor_direction_cone_deg: float = 15.0
    actor_max_boundary_jump_rad: float = 0.06
    actor_direction_static_threshold_rad: float = 0.001
    actor_projection_scale_steps: int = 33
    actor_min_projection_scale: float = 0.2
    freeze_gripper_residual: bool = True
    gripper_residual_mode: str = GRIPPER_RESIDUAL_FROZEN
    actor_gripper_residual_max_close_m: float = 0.005
    actor_gripper_residual_d1_max_m: float = 0.0005
    actor_gripper_residual_d2_max_m: float = 0.0003
    actor_gripper_max_boundary_jump_m: float = 0.0005
    gripper_command_min_m: float = 0.0
    gripper_command_max_m: float = 0.08
    gripper_release_reference_m: float = 0.05
    gripper_release_delta_m: float = 0.002
    grad_clip_norm: float = 10.0
    hidden_dim: int = 256
    projection_dim: int = 128
    residual_limit_default: float = 0.05
    normalization_clip: float = 10.0
    batch_size: int = 256
    seed: int = 0

    def __post_init__(self) -> None:
        positive_integer_fields = (
            "control_hz",
            "action_horizon",
            "chunk_length",
            "execute_steps",
            "n_step",
            "chunk_stride",
            "state_dim",
            "action_dim",
            "policy_delay",
            "hidden_dim",
            "projection_dim",
            "batch_size",
            "actor_projection_scale_steps",
        )
        for name in positive_integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.actor_start_step, bool)
            or not isinstance(self.actor_start_step, int)
            or self.actor_start_step < 0
        ):
            raise ValueError("actor_start_step must be a non-negative integer")
        if self.chunk_length > self.action_horizon:
            raise ValueError("chunk_length cannot exceed action_horizon")
        if self.execute_steps > self.action_horizon:
            raise ValueError("execute_steps cannot exceed action_horizon")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        if self.actor_lr <= 0.0 or self.critic_lr <= 0.0:
            raise ValueError("actor_lr and critic_lr must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if not 0.0 <= self.reference_dropout <= 1.0:
            raise ValueError("reference_dropout must be in [0, 1]")
        if (
            self.beta_bc < 0.0
            or self.beta_human_bc < 0.0
            or self.beta_human_gripper_bc < 0.0
        ):
            raise ValueError("BC weights must be non-negative")
        if self.human_gripper_bc_scale_m <= 0.0:
            raise ValueError("human_gripper_bc_scale_m must be positive")
        if (
            self.human_gripper_q_filter_mode
            != HUMAN_GRIPPER_Q_FILTER_CRITIC_MIN_ADVANTAGE
        ):
            raise ValueError(
                "human_gripper_q_filter_mode must be "
                f"{HUMAN_GRIPPER_Q_FILTER_CRITIC_MIN_ADVANTAGE!r}"
            )
        if (
            not math.isfinite(self.human_gripper_q_filter_margin)
            or self.human_gripper_q_filter_margin < 0.0
        ):
            raise ValueError(
                "human_gripper_q_filter_margin must be finite and non-negative"
            )
        if self.target_policy_noise_std < 0.0 or self.target_policy_noise_clip < 0.0:
            raise ValueError("target policy noise parameters must be non-negative")
        if self.actor_output_mode != "residual":
            raise ValueError("actor_output_mode must remain 'residual' for real RLT")
        if self.actor_residual_parameterization not in {"rank1_bump", "legacy_full_chunk"}:
            raise ValueError(
                "actor_residual_parameterization must be 'rank1_bump' or 'legacy_full_chunk' for real RLT"
            )
        for name in (
            "actor_residual_max_rad",
            "actor_residual_d1_max_rad",
            "actor_residual_d2_max_rad",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.actor_direction_cone_deg <= 90.0:
            raise ValueError("actor_direction_cone_deg must be in (0, 90]")
        if self.gripper_residual_mode not in {
            GRIPPER_RESIDUAL_FROZEN,
            GRIPPER_RESIDUAL_CLOSE_ASSIST,
        }:
            raise ValueError(
                "gripper_residual_mode must be 'frozen' or "
                "'close_only_persistent_v1'"
            )
        if self.gripper_residual_mode == GRIPPER_RESIDUAL_FROZEN:
            if (
                self.actor_residual_parameterization == "rank1_bump"
                and not self.freeze_gripper_residual
            ):
                raise ValueError(
                    "freeze_gripper_residual must be enabled when gripper_residual_mode='frozen'"
                )
            if self.beta_human_gripper_bc != 0.0:
                raise ValueError(
                    "beta_human_gripper_bc must be zero while the gripper residual is frozen"
                )
        else:
            if self.freeze_gripper_residual:
                raise ValueError(
                    "freeze_gripper_residual must be disabled for close-only gripper assistance"
                )
            if self.actor_residual_parameterization != "rank1_bump":
                raise ValueError(
                    "close-only gripper assistance requires the checkpoint-compatible rank1 Actor head"
                )
            if self.actor_execution_profile != PERSISTENT_ACTOR_EXECUTION_PROFILE:
                raise ValueError(
                    "close-only gripper assistance requires persistent C10 execution"
                )
            if (
                self.actor_start_step
                < MIN_GRIPPER_CLOSE_CRITIC_BURN_IN_UPDATES
            ):
                raise ValueError(
                    "close-only gripper assistance requires at least "
                    f"{MIN_GRIPPER_CLOSE_CRITIC_BURN_IN_UPDATES} Critic-only "
                    "burn-in updates before Actor/Q-filter training"
                )
        if self.actor_execution_profile not in {
            LEGACY_ACTOR_EXECUTION_PROFILE,
            PERSISTENT_ACTOR_EXECUTION_PROFILE,
        }:
            raise ValueError(
                "actor_execution_profile must be "
                f"{LEGACY_ACTOR_EXECUTION_PROFILE!r} or {PERSISTENT_ACTOR_EXECUTION_PROFILE!r}"
            )
        if (
            self.actor_execution_profile == PERSISTENT_ACTOR_EXECUTION_PROFILE
            and self.actor_residual_parameterization != "rank1_bump"
        ):
            raise ValueError("persistent-v2 execution requires the checkpoint-compatible rank1_bump Actor head")
        if self.actor_execution_profile == PERSISTENT_ACTOR_EXECUTION_PROFILE and self.chunk_stride != 10:
            raise ValueError("persistent-v2 replay requires chunk_stride=10 at physical Actor plan boundaries")
        if self.execution_filter_profile != PERSISTENT_EXECUTION_FILTER_PROFILE:
            raise ValueError(
                f"execution_filter_profile must remain {PERSISTENT_EXECUTION_FILTER_PROFILE!r}"
            )
        if self.execution_filter_tau_s <= 0.0:
            raise ValueError("execution_filter_tau_s must be positive")
        if self.actor_max_boundary_jump_rad <= 0.0:
            raise ValueError("actor_max_boundary_jump_rad must be positive")
        if self.actor_direction_static_threshold_rad <= 0.0:
            raise ValueError("actor_direction_static_threshold_rad must be positive")
        if self.actor_projection_scale_steps < 2:
            raise ValueError("actor_projection_scale_steps must be at least 2")
        if not 0.0 <= self.actor_min_projection_scale <= 1.0:
            raise ValueError("actor_min_projection_scale must be in [0, 1]")
        for name in (
            "actor_gripper_residual_max_close_m",
            "actor_gripper_residual_d1_max_m",
            "actor_gripper_residual_d2_max_m",
            "actor_gripper_max_boundary_jump_m",
            "gripper_command_max_m",
            "gripper_release_reference_m",
            "gripper_release_delta_m",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.gripper_command_min_m < 0.0:
            raise ValueError("gripper_command_min_m must be non-negative")
        if self.gripper_command_min_m >= self.gripper_command_max_m:
            raise ValueError("gripper command bounds are invalid")
        if not (
            self.gripper_command_min_m
            < self.gripper_release_reference_m
            < self.gripper_command_max_m
        ):
            raise ValueError("gripper_release_reference_m must lie inside command bounds")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive")
        if self.residual_limit_default <= 0.0:
            raise ValueError("residual_limit_default must be positive")
        if self.normalization_clip <= 0.0:
            raise ValueError("normalization_clip must be positive")
