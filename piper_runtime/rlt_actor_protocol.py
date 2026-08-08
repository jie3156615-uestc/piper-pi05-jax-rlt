from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np


BEHAVIOR_REF_KEY = "rlt/behavior_ref"
BEHAVIOR_REF_PLAN_ID_KEY = "rlt/behavior_ref_plan_id"
BEHAVIOR_REF_START_OFFSET_KEY = "rlt/behavior_ref_start_offset"
BEHAVIOR_REF_CONTRACT_KEY = "rlt/behavior_ref_contract"
ACTOR_CONDITIONING_STATE_KEY = "rlt/actor_conditioning_state"
ACTOR_ONLY_MODE_KEY = "rlt/actor_only_mode"
ACTOR_ONLY_Z_RL_KEY = "rlt/actor_only_z_rl"
BASE_ONLY_MODE_KEY = "rlt/base_only_mode"
TOKEN_BATCH_MODE_KEY = "rlt/token_batch_mode"
TOKEN_BATCH_OBSERVATIONS_KEY = "rlt/token_batch_observations"
RANK1_BUMP_CONTRACT = "rank1_bump_v1"
ACTOR_ONLY_PROTOCOL = "actor_only_v1"
ACTOR_ENRICHMENT_ONLY_PROTOCOL = "actor_enrichment_only_v1"
BASE_ONLY_PROTOCOL = "base_only_v1"
TOKEN_BATCH_PROTOCOL = "token_batch_v1"
ACTION_SCHEMA_FINGERPRINT = (
    "piper_joint_delta_v3_c10_n10_stride2_behavior_ref50_"
    "rank1_bump_r005_d1_0015_d2_001_cone15_gripper_absolute_frozen_residual"
)
RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT = (
    "piper_joint_delta_v5_c10_n10_stride10_behavior_ref50_"
    "rank1_joint_r005_d1_0015_d2_001_cone15_"
    "gripper_close_knot_r005"
)
SUPPORTED_RAW_ACTOR_ACTION_SCHEMA_FINGERPRINTS = frozenset(
    {
        ACTION_SCHEMA_FINGERPRINT,
        RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT,
    }
)
ACTOR_PROJECTION_PROFILE = (
    "rank1_bump_v1_r005_d1_0015_d2_001_cone15_"
    "scale33_min020_gripper_frozen"
)
RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE = (
    "rank1_joint_v1_r005_d1_0015_d2_001_cone15_"
    "scale33_min020_gripper_close_knot_r005"
)
ACTOR_PROJECTION_PROFILE_BY_ACTION_SCHEMA = {
    ACTION_SCHEMA_FINGERPRINT: ACTOR_PROJECTION_PROFILE,
    RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT: (
        RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE
    ),
}

_BEHAVIOR_REFERENCE_KEYS = frozenset(
    {
        BEHAVIOR_REF_KEY,
        BEHAVIOR_REF_PLAN_ID_KEY,
        BEHAVIOR_REF_START_OFFSET_KEY,
        BEHAVIOR_REF_CONTRACT_KEY,
        ACTOR_CONDITIONING_STATE_KEY,
    }
)
_ACTOR_ONLY_KEYS = frozenset({ACTOR_ONLY_MODE_KEY, ACTOR_ONLY_Z_RL_KEY})
_TOKEN_BATCH_KEYS = frozenset(
    {TOKEN_BATCH_MODE_KEY, TOKEN_BATCH_OBSERVATIONS_KEY}
)
_PROTOCOL_KEYS = (
    _BEHAVIOR_REFERENCE_KEYS
    | _ACTOR_ONLY_KEYS
    | {BASE_ONLY_MODE_KEY}
    | _TOKEN_BATCH_KEYS
)


@dataclasses.dataclass(frozen=True)
class BehaviorReferenceTarget:
    """The exact behavior-plan slice on which an Actor prediction is conditioned."""

    actions: np.ndarray
    plan_id: str
    start_offset: int
    conditioning_state: np.ndarray
    contract: str = RANK1_BUMP_CONTRACT

    def validate(self, *, chunk_length: int = 10, action_dim: int = 7) -> "BehaviorReferenceTarget":
        actions = np.asarray(self.actions, dtype=np.float32)
        if actions.shape != (chunk_length, action_dim):
            raise ValueError(
                f"behavior_ref must have shape ({chunk_length}, {action_dim}), got {actions.shape}"
            )
        if not np.all(np.isfinite(actions)):
            raise ValueError("behavior_ref must contain only finite values")
        plan_id = str(self.plan_id).strip()
        if not plan_id:
            raise ValueError("behavior_ref plan_id must be non-empty")
        start_offset = int(self.start_offset)
        if start_offset < 0:
            raise ValueError("behavior_ref start_offset must be non-negative")
        if self.contract != RANK1_BUMP_CONTRACT:
            raise ValueError(
                f"unsupported behavior_ref contract {self.contract!r}; expected {RANK1_BUMP_CONTRACT!r}"
            )
        conditioning_state = np.asarray(self.conditioning_state, dtype=np.float32)
        if conditioning_state.shape != (action_dim,) or not np.all(np.isfinite(conditioning_state)):
            raise ValueError(
                f"actor conditioning_state must be finite with shape ({action_dim},), "
                f"got {conditioning_state.shape}"
            )
        return dataclasses.replace(
            self,
            actions=actions.copy(),
            plan_id=plan_id,
            start_offset=start_offset,
            conditioning_state=conditioning_state.copy(),
        )


@dataclasses.dataclass(frozen=True)
class ActorOnlyRequest:
    """One Actor refresh that must not invoke Pi0.5 or advance its RNG.

    ``actor_only_v1`` consumes a caller-provided cached token.
    ``actor_enrichment_only_v1`` computes a fresh token from the observation,
    but still never invokes the base policy.
    """

    target: BehaviorReferenceTarget
    z_rl: np.ndarray | None = None
    protocol: str = ACTOR_ONLY_PROTOCOL

    def validate(
        self,
        *,
        chunk_length: int = 10,
        action_dim: int = 7,
        expected_z_dim: int = 2048,
    ) -> "ActorOnlyRequest":
        target = self.target.validate(
            chunk_length=chunk_length,
            action_dim=action_dim,
        )
        if self.protocol not in {
            ACTOR_ONLY_PROTOCOL,
            ACTOR_ENRICHMENT_ONLY_PROTOCOL,
        }:
            raise ValueError(
                f"unsupported Actor-only protocol {self.protocol!r}; "
                f"expected {ACTOR_ONLY_PROTOCOL!r} or "
                f"{ACTOR_ENRICHMENT_ONLY_PROTOCOL!r}"
            )
        if self.protocol == ACTOR_ONLY_PROTOCOL:
            if self.z_rl is None:
                raise ValueError("cached-token Actor-only request requires z_rl")
            z_rl = np.asarray(self.z_rl, dtype=np.float32).reshape(-1)
            if z_rl.shape != (expected_z_dim,) or not np.all(np.isfinite(z_rl)):
                raise ValueError(
                    f"Actor-only z_rl must be finite with shape ({expected_z_dim},), "
                    f"got {z_rl.shape}"
                )
        else:
            if self.z_rl is not None:
                raise ValueError(
                    "fresh-token Actor enrichment request must not carry cached z_rl"
                )
            z_rl = None
        return dataclasses.replace(
            self,
            target=target,
            z_rl=None if z_rl is None else z_rl.copy(),
        )


def add_behavior_reference(
    observation: dict[str, Any],
    target: BehaviorReferenceTarget,
) -> dict[str, Any]:
    """Return an observation carrying a validated Actor behavior-reference target."""

    target = target.validate()
    result = dict(observation)
    result[BEHAVIOR_REF_KEY] = target.actions.copy()
    result[BEHAVIOR_REF_PLAN_ID_KEY] = target.plan_id
    result[BEHAVIOR_REF_START_OFFSET_KEY] = target.start_offset
    result[BEHAVIOR_REF_CONTRACT_KEY] = target.contract
    result[ACTOR_CONDITIONING_STATE_KEY] = target.conditioning_state.copy()
    return result


def add_actor_only_request(
    observation: dict[str, Any],
    request: ActorOnlyRequest,
    *,
    expected_z_dim: int = 2048,
) -> dict[str, Any]:
    """Attach a validated Actor-only request to an observation.

    The receiver must echo ``target.actions`` as its base ``actions`` payload
    and run only the lightweight Actor on the supplied token.  In particular,
    it must not call Pi0.5 or the token encoder.
    """

    request = request.validate(expected_z_dim=expected_z_dim)
    result = add_behavior_reference(observation, request.target)
    result[ACTOR_ONLY_MODE_KEY] = request.protocol
    assert request.z_rl is not None
    result[ACTOR_ONLY_Z_RL_KEY] = request.z_rl.copy()
    return result


def add_actor_enrichment_only_request(
    observation: dict[str, Any],
    target: BehaviorReferenceTarget,
) -> dict[str, Any]:
    """Attach an exact-reference Token+Actor request that skips Pi0.5."""

    request = ActorOnlyRequest(
        target=target,
        z_rl=None,
        protocol=ACTOR_ENRICHMENT_ONLY_PROTOCOL,
    ).validate()
    result = add_behavior_reference(observation, request.target)
    result[ACTOR_ONLY_MODE_KEY] = request.protocol
    return result


def add_base_only_request(observation: dict[str, Any]) -> dict[str, Any]:
    """Request exactly one Pi0.5 inference with Token/Actor work disabled."""

    result = dict(observation)
    result[BASE_ONLY_MODE_KEY] = BASE_ONLY_PROTOCOL
    return result


def add_token_batch_request(
    observations: list[dict[str, Any]],
    *,
    max_batch_size: int = 16,
) -> dict[str, Any]:
    """Request a batch of fresh RL tokens without Pi0.5 or Actor inference."""

    if not isinstance(observations, list) or not observations:
        raise ValueError("token batch requires a non-empty observation list")
    if len(observations) > int(max_batch_size):
        raise ValueError(
            f"token batch size {len(observations)} exceeds {int(max_batch_size)}"
        )
    cleaned: list[dict[str, Any]] = []
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            raise TypeError(f"token batch observation {index} is not a mapping")
        if any(key in observation for key in _PROTOCOL_KEYS):
            raise ValueError(
                f"token batch observation {index} contains runtime protocol fields"
            )
        cleaned.append(dict(observation))
    return {
        TOKEN_BATCH_MODE_KEY: TOKEN_BATCH_PROTOCOL,
        TOKEN_BATCH_OBSERVATIONS_KEY: cleaned,
    }


def extract_token_batch_request(
    request: dict[str, Any],
    *,
    max_batch_size: int = 16,
) -> list[dict[str, Any]] | None:
    """Validate and extract a fail-closed Token-only batch request."""

    mode_present = TOKEN_BATCH_MODE_KEY in request
    observations_present = TOKEN_BATCH_OBSERVATIONS_KEY in request
    if not mode_present and not observations_present:
        return None
    if not mode_present or not observations_present:
        raise ValueError("incomplete token batch protocol payload")
    if str(request[TOKEN_BATCH_MODE_KEY]) != TOKEN_BATCH_PROTOCOL:
        raise ValueError(
            f"unsupported token batch protocol "
            f"{request[TOKEN_BATCH_MODE_KEY]!r}"
        )
    if any(key in request for key in _BEHAVIOR_REFERENCE_KEYS | _ACTOR_ONLY_KEYS):
        raise ValueError("token batch request must not carry Actor protocol fields")
    if BASE_ONLY_MODE_KEY in request:
        raise ValueError("token batch and base-only markers are mutually exclusive")
    observations = request[TOKEN_BATCH_OBSERVATIONS_KEY]
    if not isinstance(observations, list) or not observations:
        raise ValueError("token batch requires a non-empty observation list")
    if len(observations) > int(max_batch_size):
        raise ValueError(
            f"token batch size {len(observations)} exceeds {int(max_batch_size)}"
        )
    result: list[dict[str, Any]] = []
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            raise TypeError(f"token batch observation {index} is not a mapping")
        if any(key in observation for key in _PROTOCOL_KEYS):
            raise ValueError(
                f"token batch observation {index} contains runtime protocol fields"
            )
        result.append(dict(observation))
    return result


def is_base_only_request(observation: dict[str, Any]) -> bool:
    """Validate and identify the explicit base-only protocol marker."""

    if BASE_ONLY_MODE_KEY not in observation:
        return False
    protocol = str(observation[BASE_ONLY_MODE_KEY])
    if protocol != BASE_ONLY_PROTOCOL:
        raise ValueError(
            f"unsupported base-only protocol {protocol!r}; "
            f"expected {BASE_ONLY_PROTOCOL!r}"
        )
    if ACTOR_ONLY_MODE_KEY in observation or ACTOR_ONLY_Z_RL_KEY in observation:
        raise ValueError("base-only and Actor-only protocol markers are mutually exclusive")
    if any(key in observation for key in _BEHAVIOR_REFERENCE_KEYS):
        raise ValueError("base-only request must not carry a behavior_ref payload")
    return True


def strip_behavior_reference(observation: dict[str, Any]) -> dict[str, Any]:
    """Remove runtime-only protocol fields before invoking Pi0.5/token transforms."""

    return {key: value for key, value in observation.items() if key not in _PROTOCOL_KEYS}


def extract_behavior_reference(
    observation: dict[str, Any],
    *,
    chunk_length: int = 10,
    action_dim: int = 7,
) -> BehaviorReferenceTarget | None:
    """Read the all-or-none behavior-reference payload from a policy request."""

    present = {key for key in _BEHAVIOR_REFERENCE_KEYS if key in observation}
    if not present:
        return None
    if present != _BEHAVIOR_REFERENCE_KEYS:
        missing = sorted(_BEHAVIOR_REFERENCE_KEYS - present)
        raise ValueError(f"incomplete behavior_ref protocol payload; missing {missing}")
    return BehaviorReferenceTarget(
        actions=np.asarray(observation[BEHAVIOR_REF_KEY], dtype=np.float32),
        plan_id=str(observation[BEHAVIOR_REF_PLAN_ID_KEY]),
        start_offset=int(observation[BEHAVIOR_REF_START_OFFSET_KEY]),
        conditioning_state=np.asarray(observation[ACTOR_CONDITIONING_STATE_KEY], dtype=np.float32),
        contract=str(observation[BEHAVIOR_REF_CONTRACT_KEY]),
    ).validate(chunk_length=chunk_length, action_dim=action_dim)


def extract_actor_only_request(
    observation: dict[str, Any],
    *,
    chunk_length: int = 10,
    action_dim: int = 7,
    expected_z_dim: int = 2048,
) -> ActorOnlyRequest | None:
    """Extract an all-or-none Actor-only request.

    Actor-only fields without the exact behavior-reference payload are rejected
    instead of falling through to a stochastic Pi0.5 call.
    """

    mode_present = ACTOR_ONLY_MODE_KEY in observation
    z_present = ACTOR_ONLY_Z_RL_KEY in observation
    if not mode_present and not z_present:
        return None
    if not mode_present:
        raise ValueError(
            f"incomplete Actor-only protocol payload; missing {ACTOR_ONLY_MODE_KEY!r}"
        )
    protocol = str(observation[ACTOR_ONLY_MODE_KEY])
    if protocol == ACTOR_ONLY_PROTOCOL and not z_present:
        raise ValueError(
            f"incomplete Actor-only protocol payload; missing {ACTOR_ONLY_Z_RL_KEY!r}"
        )
    if protocol == ACTOR_ENRICHMENT_ONLY_PROTOCOL and z_present:
        raise ValueError(
            "fresh-token Actor enrichment request must not carry cached z_rl"
        )
    target = extract_behavior_reference(
        observation,
        chunk_length=chunk_length,
        action_dim=action_dim,
    )
    if target is None:
        raise ValueError("Actor-only request is missing the behavior_ref payload")
    return ActorOnlyRequest(
        target=target,
        z_rl=(
            None
            if not z_present
            else np.asarray(observation[ACTOR_ONLY_Z_RL_KEY], dtype=np.float32)
        ),
        protocol=protocol,
    ).validate(
        chunk_length=chunk_length,
        action_dim=action_dim,
        expected_z_dim=expected_z_dim,
    )
