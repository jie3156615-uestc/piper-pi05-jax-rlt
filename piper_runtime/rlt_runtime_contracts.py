"""Shared validation and public constants for the Piper RLT runtime contract."""

from __future__ import annotations

import math
from typing import Any

from piper_runtime.rlt_actor_protocol import ACTION_SCHEMA_FINGERPRINT
from piper_runtime.rlt_actor_protocol import ACTOR_PROJECTION_PROFILE
from piper_runtime.rlt_actor_protocol import (
    RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT,
)
from piper_runtime.rlt_actor_protocol import (
    RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE,
)
from piper_runtime.rlt_actor_protocol import (
    SUPPORTED_RAW_ACTOR_ACTION_SCHEMA_FINGERPRINTS,
)
from piper_runtime.rlt_residual_governor import ActorResidualGovernorConfig
from piper_runtime.rlt_residual_governor import GRIPPER_RESIDUAL_CLOSE_ASSIST
from piper_runtime.rlt_residual_governor import PERSISTENT_C10_EXECUTION_CONTRACT
from piper_runtime.rlt_residual_governor import (
    PERSISTENT_FILTERED_ACTUAL_ACTION_SCHEMA_FINGERPRINT,
)
from piper_runtime.rlt_residual_governor import (
    PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT,
)
from piper_runtime.rlt_residual_governor import (
    PERSISTENT_FILTERED_ACTUAL_GOVERNOR_FINGERPRINT,
)
from piper_runtime.rlt_residual_governor import (
    PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT,
)
from piper_runtime.rlt_residual_governor import (
    PERSISTENT_GRIPPER_CLOSE_GOVERNOR_FINGERPRINT,
)


PUBLISH_AUTHORIZATION = "I_UNDERSTAND_RLT_PUBLISHES_ARM_COMMANDS"
ACTOR_LIVE_AUTHORIZATION = "I_UNDERSTAND_RLT_ACTOR_CONTROLS_ARM_IN_PHASE"
LEGACY_ACTOR_EXECUTION_PROFILE = "rank1_bump_v1"


def validate_actor_runtime_contract(
    config: Any,
    *,
    context: str,
    phase_classifier_checkpoint: Any,
    chunk_length: int,
    action_dim: int,
    actor_prefetch_lead_steps: int | None = None,
) -> int | None:
    """Validate the Actor contract shared by session and rollout configs."""

    if context not in {"session", "runtime"}:
        raise ValueError(f"unsupported Actor contract context: {context!r}")
    if config.actor_shadow_expected_z_dim < 1:
        raise ValueError("actor_shadow_expected_z_dim must be positive")
    if config.actor_shadow_max_latency_s <= 0:
        raise ValueError("actor_shadow_max_latency_s must be positive")
    if actor_prefetch_lead_steps is not None and (
        actor_prefetch_lead_steps < 1 or actor_prefetch_lead_steps > chunk_length
    ):
        raise ValueError("actor_prefetch_lead_steps must be in [1, chunk_length]")
    if context == "runtime" and (
        not math.isfinite(config.actor_live_max_boundary_jump_rad)
        or config.actor_live_max_boundary_jump_rad <= 0.0
    ):
        raise ValueError(
            "actor_live_max_boundary_jump_rad must be finite and positive"
        )

    ActorResidualGovernorConfig(
        chunk_length=chunk_length,
        action_dim=action_dim,
        actor_residual_max_rad=config.actor_residual_max_rad,
        actor_residual_d1_max_rad=config.actor_residual_d1_max_rad,
        actor_residual_d2_max_rad=config.actor_residual_d2_max_rad,
        actor_direction_cone_deg=config.actor_direction_cone_deg,
        max_boundary_jump_rad=config.actor_live_max_boundary_jump_rad,
        gripper_residual_mode=config.actor_gripper_residual_mode,
        gripper_residual_max_close_m=config.actor_gripper_residual_max_close_m,
        gripper_residual_d1_max_m=config.actor_gripper_residual_d1_max_m,
        gripper_residual_d2_max_m=config.actor_gripper_residual_d2_max_m,
        gripper_max_boundary_jump_m=config.actor_gripper_max_boundary_jump_m,
        gripper_command_min_m=config.actor_gripper_command_min_m,
        gripper_command_max_m=config.actor_gripper_command_max_m,
        gripper_release_reference_m=config.actor_gripper_release_reference_m,
        gripper_release_delta_m=config.actor_gripper_release_delta_m,
    ).validate()

    if (
        config.action_schema_fingerprint
        not in SUPPORTED_RAW_ACTOR_ACTION_SCHEMA_FINGERPRINTS
    ):
        raise ValueError(
            f"{context} action schema mismatch: unsupported raw Actor schema "
            f"{config.action_schema_fingerprint!r}"
        )
    close_assist = (
        config.actor_gripper_residual_mode == GRIPPER_RESIDUAL_CLOSE_ASSIST
    )
    if (
        close_assist
        and config.actor_execution_profile
        != PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT
    ):
        raise ValueError(
            "gripper close-assist requires the filtered-actual persistent "
            "execution profile"
        )
    expected_raw_actor_schema = (
        RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT
        if close_assist
        else ACTION_SCHEMA_FINGERPRINT
    )
    if config.action_schema_fingerprint != expected_raw_actor_schema:
        raise ValueError(
            f"{context} raw Actor schema/gripper mode mismatch: "
            f"{config.action_schema_fingerprint!r} != "
            f"{expected_raw_actor_schema!r}"
        )
    expected_execution_schema = (
        (
            PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT
            if close_assist
            else PERSISTENT_FILTERED_ACTUAL_ACTION_SCHEMA_FINGERPRINT
        )
        if config.actor_execution_profile
        == PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT
        else ACTION_SCHEMA_FINGERPRINT
    )
    if config.execution_action_schema_fingerprint != expected_execution_schema:
        raise ValueError(
            f"{context} execution action schema mismatch for "
            f"{config.actor_execution_profile!r}: "
            f"{config.execution_action_schema_fingerprint!r} != "
            f"{expected_execution_schema!r}"
        )
    if (
        config.actor_execution_profile
        == PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT
    ):
        required_limits = {
            "actor_residual_max_rad": 0.005,
            "actor_residual_d1_max_rad": 0.0015,
            "actor_residual_d2_max_rad": 0.001,
            "actor_direction_cone_deg": 15.0,
            "actor_live_max_boundary_jump_rad": 0.06,
        }
        if close_assist:
            required_limits.update(
                {
                    "actor_gripper_residual_max_close_m": 0.005,
                    "actor_gripper_residual_d1_max_m": 0.0005,
                    "actor_gripper_residual_d2_max_m": 0.0003,
                    "actor_gripper_max_boundary_jump_m": 0.0005,
                    "actor_gripper_command_min_m": 0.0,
                    "actor_gripper_command_max_m": 0.08,
                    "actor_gripper_release_reference_m": 0.05,
                    "actor_gripper_release_delta_m": 0.002,
                }
            )
        for name, expected in required_limits.items():
            actual = float(getattr(config, name))
            if not math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "filtered-actual safety contract mismatch: "
                    f"{name}={actual!r}, expected {expected!r}"
                )
        expected_governor_fingerprint = (
            PERSISTENT_GRIPPER_CLOSE_GOVERNOR_FINGERPRINT
            if close_assist
            else PERSISTENT_FILTERED_ACTUAL_GOVERNOR_FINGERPRINT
        )
        if config.actor_governor_fingerprint != expected_governor_fingerprint:
            raise ValueError(
                f"{context} Actor governor fingerprint mismatch: "
                f"{config.actor_governor_fingerprint!r} != "
                f"{expected_governor_fingerprint!r}"
            )
    expected_projection_profile = (
        RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE
        if close_assist
        else ACTOR_PROJECTION_PROFILE
    )
    if config.actor_projection_profile != expected_projection_profile:
        raise ValueError(
            f"{context} Actor projection profile mismatch: "
            f"{config.actor_projection_profile!r} != "
            f"{expected_projection_profile!r}"
        )
    if config.actor_execution_profile not in {
        LEGACY_ACTOR_EXECUTION_PROFILE,
        PERSISTENT_C10_EXECUTION_CONTRACT,
        PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT,
    }:
        raise ValueError(
            "actor_execution_profile must be "
            f"{LEGACY_ACTOR_EXECUTION_PROFILE!r} or "
            f"{PERSISTENT_C10_EXECUTION_CONTRACT!r} or "
            f"{PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT!r}"
        )

    actor_live_max_chunks = config.actor_live_max_chunks
    if actor_live_max_chunks is not None:
        actor_live_max_chunks = int(actor_live_max_chunks)
        if actor_live_max_chunks < 0:
            raise ValueError("actor_live_max_chunks must be non-negative")
        if actor_live_max_chunks == 0:
            actor_live_max_chunks = None
    if context == "runtime" and config.actor_shadow and chunk_length != 10:
        raise ValueError("Piper RLT actor shadow is fixed to C=10")
    if config.actor_live and not config.actor_shadow:
        message = "actor_live requires actor_shadow"
        if context == "runtime":
            message += " so z_rl/a_actor are validated and logged"
        raise ValueError(message)
    if config.actor_live and phase_classifier_checkpoint is None:
        raise ValueError("actor_live requires the frozen phase classifier")
    if (
        config.actor_live
        and config.actor_live_authorization != ACTOR_LIVE_AUTHORIZATION
    ):
        raise ValueError(
            f"actor live authorization must equal {ACTOR_LIVE_AUTHORIZATION!r}"
        )
    return actor_live_max_chunks
