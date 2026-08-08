from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class RLTPolicyOutput:
    a_ref: np.ndarray
    z_rl: np.ndarray
    a_actor: np.ndarray | None
    metadata: dict[str, Any]


PIPER_ACTION_SPACE = "joint_absolute_gripper_absolute"
RLT_LEARNING_ACTION_SPACE = "joint_delta6_gripper_absolute"


def extract_rlt_policy_output(
    response: dict[str, Any],
    *,
    chunk_length: int = 10,
    action_dim: int = 7,
    fallback_z_rl_dim: int = 1,
    state_snapshot: np.ndarray | None = None,
    read_actor: bool = False,
    actor_chunk_length: int | None = None,
) -> RLTPolicyOutput:
    actor_chunk_length = chunk_length if actor_chunk_length is None else int(actor_chunk_length)
    if actor_chunk_length < 1:
        raise ValueError("actor_chunk_length must be positive")
    actions = _extract_actions(response, chunk_length=chunk_length, action_dim=action_dim)
    z_rl, z_source, z_error = _extract_z_rl(response, fallback_z_rl_dim=fallback_z_rl_dim)
    actor, actor_source, actor_space, actor_error = _extract_actor(
        response,
        chunk_length=actor_chunk_length,
        action_dim=action_dim,
        state_snapshot=state_snapshot,
        enabled=read_actor,
    )
    actor_name, behavior_actor_checkpoint = _extract_actor_identity(response)
    behavior_ref_metadata = _extract_actor_behavior_ref_metadata(response)
    return RLTPolicyOutput(
        a_ref=actions[:chunk_length, :action_dim].astype(np.float32),
        z_rl=z_rl.astype(np.float32),
        a_actor=actor,
        metadata={
            "action_source": "response.actions",
            "z_rl_source": z_source,
            "z_rl_error": z_error,
            "actor_status": "ok" if actor is not None else "unavailable",
            "actor_source": actor_source,
            "actor_response_action_space": actor_space,
            "actor_logged_action_space": PIPER_ACTION_SPACE if actor is not None else None,
            "actor_error": actor_error,
            "actor_name": actor_name,
            "behavior_actor_checkpoint": behavior_actor_checkpoint,
            **behavior_ref_metadata,
            "chunk_length": int(chunk_length),
            "actor_chunk_length": int(actor_chunk_length),
            "action_dim": int(action_dim),
        },
    )


def _extract_actor_behavior_ref_metadata(response: dict[str, Any]) -> dict[str, Any]:
    shadow = response.get("rlt_shadow")
    shadow = shadow if isinstance(shadow, dict) else {}
    source = response.get(
        "a_actor_behavior_ref_source",
        shadow.get("behavior_ref_source"),
    )
    contract = response.get(
        "a_actor_behavior_ref_contract",
        shadow.get("actor_contract"),
    )
    action_schema = response.get(
        "a_actor_action_schema_fingerprint",
        shadow.get("action_schema_fingerprint"),
    )
    projection_profile = response.get(
        "a_actor_projection_profile",
        shadow.get("actor_projection_profile"),
    )
    plan_id = response.get(
        "a_actor_behavior_ref_plan_id",
        shadow.get("behavior_ref_plan_id"),
    )
    raw_offset = response.get(
        "a_actor_behavior_ref_start_offset",
        shadow.get("behavior_ref_start_offset"),
    )
    error = None
    start_offset = None
    if raw_offset is not None:
        try:
            start_offset = int(raw_offset)
            if start_offset < 0:
                raise ValueError("must be non-negative")
        except (TypeError, ValueError) as exc:
            error = f"invalid actor behavior_ref start offset: {exc}"
    shadow_mode = shadow.get("mode")
    actor_only_mode = shadow_mode == "actor_only"
    base_only_mode = shadow_mode == "base_only"
    return {
        "actor_behavior_ref_source": None if source is None else str(source),
        "actor_behavior_ref_contract": None if contract is None else str(contract),
        "actor_action_schema_fingerprint": (
            None if action_schema is None else str(action_schema)
        ),
        "actor_projection_profile": (
            None if projection_profile is None else str(projection_profile)
        ),
        "actor_behavior_ref_plan_id": None if plan_id is None else str(plan_id),
        "actor_behavior_ref_start_offset": start_offset,
        "actor_behavior_ref_error": error,
        # Flatten proof that an enrichment response did not call Pi0.5 or the
        # token encoder.  Episode JSONL should preserve these fields so an
        # Actor-only rollout can be audited without parsing nested wire data.
        "actor_only_mode": actor_only_mode,
        "actor_only_protocol": (
            shadow.get("actor_only_protocol") if actor_only_mode else None
        ),
        "actor_only_base_policy_called": (
            shadow.get("base_policy_called") if actor_only_mode else None
        ),
        "actor_only_base_rng_advanced": (
            shadow.get("base_rng_advanced") if actor_only_mode else None
        ),
        "actor_only_token_encoder_called": (
            shadow.get("token_encoder_called") if actor_only_mode else None
        ),
        "actor_only_actor_called": (
            shadow.get("actor_called") if actor_only_mode else None
        ),
        "actor_only_latency_s": (
            shadow.get("shadow_latency_s") if actor_only_mode else None
        ),
        "base_only_mode": base_only_mode,
        "base_only_base_policy_called": (
            shadow.get("base_policy_called") if base_only_mode else None
        ),
        "base_only_base_rng_advanced": (
            shadow.get("base_rng_advanced") if base_only_mode else None
        ),
        "base_only_token_encoder_called": (
            shadow.get("token_encoder_called") if base_only_mode else None
        ),
        "base_only_actor_called": (
            shadow.get("actor_called") if base_only_mode else None
        ),
        "base_only_latency_s": (
            shadow.get("base_policy_latency_s") if base_only_mode else None
        ),
    }


def _extract_actor_identity(response: dict[str, Any]) -> tuple[str | None, str | None]:
    """Extract the exact Actor identity advertised by the policy response."""

    shadow = response.get("rlt_shadow")
    shadow = shadow if isinstance(shadow, dict) else {}
    response_metadata = response.get("metadata")
    response_metadata = response_metadata if isinstance(response_metadata, dict) else {}

    actor_name_raw = next(
        (
            value
            for value in (
                response.get("actor_name"),
                shadow.get("actor_name"),
                response_metadata.get("rlt_actor_name"),
            )
            if value is not None
        ),
        None,
    )
    actor_name = None if actor_name_raw is None else str(actor_name_raw)

    checkpoint_raw = next(
        (
            value
            for value in (
                response.get("behavior_actor_checkpoint"),
                response.get("actor_checkpoint"),
                shadow.get("behavior_actor_checkpoint"),
                shadow.get("actor_checkpoint"),
            )
            if value is not None
        ),
        None,
    )
    if checkpoint_raw is None and actor_name not in {None, "", "none"}:
        checkpoint_raw = actor_name
    checkpoint = None if checkpoint_raw is None else str(checkpoint_raw)
    return actor_name, checkpoint


def delta_chunk_to_piper_targets(delta_chunk: np.ndarray, state_snapshot: np.ndarray) -> np.ndarray:
    """Convert policy joint-delta actions to Piper absolute joint targets.

    The current JAX LoRA policy emits joint deltas for the six arm joints while
    the ROS Piper command topic expects absolute joint targets. The gripper
    channel is kept as the policy's absolute opening target, matching the
    existing Piper/OpenPI data convention used by the runtime.
    """

    chunk = np.asarray(delta_chunk, dtype=np.float32)
    state = np.asarray(state_snapshot, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] < 7:
        raise ValueError(f"delta_chunk must have shape (N, >=7), got {chunk.shape}")
    if state.shape != (7,):
        raise ValueError(f"state_snapshot must have shape (7,), got {state.shape}")
    if not np.all(np.isfinite(chunk[:, :7])) or not np.all(np.isfinite(state)):
        raise ValueError("delta chunk and state snapshot must be finite")
    targets = chunk[:, :7].astype(np.float32, copy=True)
    targets[:, :6] = state[:6][None, :] + targets[:, :6]
    return targets


def piper_targets_to_delta_chunk(target_chunk: np.ndarray, state_snapshot: np.ndarray) -> np.ndarray:
    """Convert absolute Piper targets to the RLT learning action domain.

    The first six channels become deltas from the state at the start of the
    chunk.  The gripper remains an absolute opening.  This is the inverse of
    :func:`delta_chunk_to_piper_targets` and is the only conversion the shadow
    actor service should use before invoking a JAX actor.
    """

    chunk = np.asarray(target_chunk, dtype=np.float32)
    state = np.asarray(state_snapshot, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] < 7:
        raise ValueError(f"target_chunk must have shape (N, >=7), got {chunk.shape}")
    if state.shape != (7,):
        raise ValueError(f"state_snapshot must have shape (7,), got {state.shape}")
    if not np.all(np.isfinite(chunk[:, :7])) or not np.all(np.isfinite(state)):
        raise ValueError("target chunk and state snapshot must be finite")
    delta = chunk[:, :7].astype(np.float32, copy=True)
    delta[:, :6] = delta[:, :6] - state[:6][None, :]
    return delta


def _extract_actions(response: dict[str, Any], *, chunk_length: int, action_dim: int) -> np.ndarray:
    if "actions" not in response:
        raise ValueError("policy response is missing actions")
    actions = np.asarray(response["actions"], dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"policy actions must be 2-D, got {actions.shape}")
    if actions.shape[0] < chunk_length or actions.shape[1] < action_dim:
        raise ValueError(
            f"policy actions must have at least ({chunk_length}, {action_dim}), got {actions.shape}"
        )
    if not np.all(np.isfinite(actions[:chunk_length, :action_dim])):
        raise ValueError("policy actions contain non-finite values")
    return actions


def _extract_z_rl(response: dict[str, Any], *, fallback_z_rl_dim: int) -> tuple[np.ndarray, str, str | None]:
    for key in ("z_rl", "rl_token", "rl_latent"):
        if key in response and response[key] is not None:
            try:
                z_rl = np.asarray(response[key], dtype=np.float32).reshape(-1)
            except (TypeError, ValueError) as exc:
                return _fallback_z_rl(fallback_z_rl_dim), "zeros_invalid_policy_response", f"{key}: {exc}"
            if z_rl.size == 0 or not np.all(np.isfinite(z_rl)):
                return (
                    _fallback_z_rl(fallback_z_rl_dim),
                    "zeros_invalid_policy_response",
                    f"{key} must be a non-empty finite vector",
                )
            return z_rl, key, None
    return _fallback_z_rl(fallback_z_rl_dim), "zeros_missing_from_policy_response", None


def _fallback_z_rl(fallback_z_rl_dim: int) -> np.ndarray:
    if fallback_z_rl_dim <= 0:
        raise ValueError("fallback_z_rl_dim must be positive when policy response does not contain z_rl")
    return np.zeros(fallback_z_rl_dim, dtype=np.float32)


def _extract_actor(
    response: dict[str, Any],
    *,
    chunk_length: int,
    action_dim: int,
    state_snapshot: np.ndarray | None,
    enabled: bool,
) -> tuple[np.ndarray | None, str | None, str | None, str | None]:
    if not enabled:
        return None, None, None, None
    actor_key = next(
        (key for key in ("a_actor", "actor_actions", "rlt_actions") if response.get(key) is not None),
        None,
    )
    if actor_key is None:
        return None, None, None, "policy response does not contain a shadow actor chunk"
    try:
        actor = np.asarray(response[actor_key], dtype=np.float32)
    except (TypeError, ValueError) as exc:
        return None, actor_key, None, f"invalid actor array: {exc}"
    if actor.ndim != 2 or actor.shape[0] < chunk_length or actor.shape[1] < action_dim:
        return (
            None,
            actor_key,
            None,
            f"actor chunk must have at least ({chunk_length}, {action_dim}), got {actor.shape}",
        )
    actor = actor[:chunk_length, :action_dim]
    if not np.all(np.isfinite(actor)):
        return None, actor_key, None, "actor chunk contains non-finite values"

    action_space = str(response.get("a_actor_action_space", PIPER_ACTION_SPACE))
    if action_space == PIPER_ACTION_SPACE:
        absolute_actor = actor.astype(np.float32, copy=True)
    elif action_space == RLT_LEARNING_ACTION_SPACE:
        if state_snapshot is None:
            return None, actor_key, action_space, "state_snapshot is required to convert actor deltas"
        try:
            absolute_actor = delta_chunk_to_piper_targets(actor, state_snapshot)
        except ValueError as exc:
            return None, actor_key, action_space, f"actor action conversion failed: {exc}"
    else:
        return None, actor_key, action_space, f"unsupported actor action space: {action_space!r}"
    return absolute_actor, actor_key, action_space, None
