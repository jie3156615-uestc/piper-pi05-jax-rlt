from __future__ import annotations

import dataclasses
import time
from typing import Any, Protocol

import numpy as np

from piper_runtime.rlt_actor_protocol import ACTION_SCHEMA_FINGERPRINT
from piper_runtime.rlt_actor_protocol import (
    ACTOR_PROJECTION_PROFILE_BY_ACTION_SCHEMA,
)
from piper_runtime.rlt_actor_protocol import ACTOR_ONLY_MODE_KEY
from piper_runtime.rlt_actor_protocol import ACTOR_ONLY_Z_RL_KEY
from piper_runtime.rlt_actor_protocol import ACTOR_PROJECTION_PROFILE
from piper_runtime.rlt_actor_protocol import BASE_ONLY_MODE_KEY
from piper_runtime.rlt_actor_protocol import RANK1_BUMP_CONTRACT
from piper_runtime.rlt_actor_protocol import TOKEN_BATCH_MODE_KEY
from piper_runtime.rlt_actor_protocol import TOKEN_BATCH_PROTOCOL
from piper_runtime.rlt_actor_protocol import ActorOnlyRequest
from piper_runtime.rlt_actor_protocol import extract_actor_only_request
from piper_runtime.rlt_actor_protocol import extract_behavior_reference
from piper_runtime.rlt_actor_protocol import extract_token_batch_request
from piper_runtime.rlt_actor_protocol import is_base_only_request
from piper_runtime.rlt_actor_protocol import strip_behavior_reference
from piper_runtime.rlt_policy_adapter import PIPER_ACTION_SPACE
from piper_runtime.rlt_policy_adapter import RLT_LEARNING_ACTION_SPACE
from piper_runtime.rlt_policy_adapter import delta_chunk_to_piper_targets
from piper_runtime.rlt_policy_adapter import piper_targets_to_delta_chunk


class TokenEncoder(Protocol):
    def encode(self, observation: dict[str, Any]) -> np.ndarray: ...

    def encode_batch(
        self, observations: list[dict[str, Any]]
    ) -> np.ndarray: ...


class ShadowActor(Protocol):
    def predict(self, *, z_rl: np.ndarray, state: np.ndarray, a_ref: np.ndarray) -> np.ndarray: ...


@dataclasses.dataclass(frozen=True)
class ShadowPolicyConfig:
    chunk_length: int = 10
    action_dim: int = 7
    expected_z_dim: int = 2048
    max_shadow_latency_s: float = 0.333
    actor_action_schema_fingerprint: str = ACTION_SCHEMA_FINGERPRINT
    actor_projection_profile: str = ACTOR_PROJECTION_PROFILE

    def validate(self) -> "ShadowPolicyConfig":
        if self.chunk_length != 10:
            raise ValueError("Piper RLT shadow policy is fixed to C=10")
        if self.action_dim != 7:
            raise ValueError("Piper RLT action dimension must be 7")
        if self.expected_z_dim < 1:
            raise ValueError("expected_z_dim must be positive")
        if self.max_shadow_latency_s <= 0:
            raise ValueError("max_shadow_latency_s must be positive")
        expected_projection = ACTOR_PROJECTION_PROFILE_BY_ACTION_SCHEMA.get(
            self.actor_action_schema_fingerprint
        )
        if expected_projection is None:
            raise ValueError(
                "unsupported shadow Actor action schema: "
                f"{self.actor_action_schema_fingerprint!r}"
            )
        if self.actor_projection_profile != expected_projection:
            raise ValueError(
                "shadow Actor schema/projection mismatch: "
                f"{self.actor_projection_profile!r} != "
                f"{expected_projection!r}"
            )
        return self


class PassThroughShadowActor:
    """Dry/mock actor that predicts exactly the reference learning action."""

    def predict(self, *, z_rl: np.ndarray, state: np.ndarray, a_ref: np.ndarray) -> np.ndarray:
        del z_rl, state
        return np.asarray(a_ref, dtype=np.float32).copy()


class ShadowAugmentedPolicy:
    """Fail-open wrapper that adds RL-token and Actor shadow fields.

    The wrapped base policy is called first.  Its ``actions`` field is never
    changed.  Shadow failures are reported in ``rlt_shadow`` and cannot escape
    from :meth:`infer`, so a Token/Actor failure cannot interrupt Pi0.5.
    """

    def __init__(
        self,
        base_policy: Any,
        *,
        token_encoder: TokenEncoder | None,
        actor: ShadowActor | None,
        config: ShadowPolicyConfig = ShadowPolicyConfig(),
        clock: Any = time.monotonic,
        actor_name: str = "none",
    ) -> None:
        self.base_policy = base_policy
        self.token_encoder = token_encoder
        self.actor = actor
        self.config = config.validate()
        self.clock = clock
        self.actor_name = str(actor_name)

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = dict(getattr(self.base_policy, "metadata", {}))
        metadata.update(
            {
                "rlt_mode": "actor_shadow",
                "rlt_actor_controls_robot": False,
                "rlt_chunk_length": self.config.chunk_length,
                "rlt_learning_action_space": RLT_LEARNING_ACTION_SPACE,
                "rlt_wire_actor_action_space": PIPER_ACTION_SPACE,
                "rlt_actor_name": self.actor_name,
                "behavior_actor_checkpoint": None if self.actor is None else self.actor_name,
                "rlt_action_schema_fingerprint": (
                    self.config.actor_action_schema_fingerprint
                ),
                "rlt_actor_projection_profile": (
                    self.config.actor_projection_profile
                ),
                "rlt_token_batch_protocol": TOKEN_BATCH_PROTOCOL,
                "rlt_token_batch_max_size": 16,
            }
        )
        return metadata

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        if TOKEN_BATCH_MODE_KEY in observation:
            try:
                token_batch = extract_token_batch_request(observation)
                if token_batch is None:  # pragma: no cover - guarded by key.
                    raise ValueError("token batch request payload is missing")
            except Exception as exc:
                return {
                    "rlt_shadow": {
                        "mode": "token_batch",
                        "actor_controls_robot": False,
                        "base_policy_called": False,
                        "base_rng_advanced": False,
                        "token_encoder_called": False,
                        "actor_called": False,
                        "token_status": "invalid_token_batch_protocol",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                }
            return self._infer_token_batch(token_batch)

        if BASE_ONLY_MODE_KEY in observation:
            try:
                if not is_base_only_request(observation):  # pragma: no cover
                    raise ValueError("base-only request payload is missing")
            except Exception as exc:
                return {
                    "actor_name": self.actor_name,
                    "behavior_actor_checkpoint": (
                        None if self.actor is None else self.actor_name
                    ),
                    "rlt_shadow": {
                        "mode": "base_only",
                        "actor_controls_robot": False,
                        "base_policy_called": False,
                        "base_rng_advanced": False,
                        "token_encoder_called": False,
                        "actor_called": False,
                        "base_status": "invalid_base_only_protocol",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                }
            return self._infer_base_only(strip_behavior_reference(observation))

        actor_only_requested = bool(
            ACTOR_ONLY_MODE_KEY in observation or ACTOR_ONLY_Z_RL_KEY in observation
        )
        if actor_only_requested:
            try:
                actor_only = extract_actor_only_request(
                    observation,
                    chunk_length=self.config.chunk_length,
                    action_dim=self.config.action_dim,
                    expected_z_dim=self.config.expected_z_dim,
                )
                if actor_only is None:  # pragma: no cover - guarded by key check.
                    raise ValueError("Actor-only request payload is missing")
            except Exception as exc:
                # An invalid explicit Actor-only request must fail closed.  It
                # must never fall through to Pi0.5 and accidentally advance the
                # stochastic policy RNG.
                return {
                    "actor_name": self.actor_name,
                    "behavior_actor_checkpoint": (
                        None if self.actor is None else self.actor_name
                    ),
                    "rlt_shadow": {
                        "mode": "actor_only",
                        "actor_controls_robot": False,
                        "base_policy_called": False,
                        "base_rng_advanced": False,
                        "token_encoder_called": False,
                        "actor_called": False,
                        "actor_status": "invalid_actor_only_protocol",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                }
            return self._infer_actor_only(
                actor_only,
                policy_observation=strip_behavior_reference(observation),
            )

        # Runtime-only behavior-reference fields must never leak into Pi0.5 or
        # token input transforms.  Only the Actor consumes this side channel.
        policy_observation = strip_behavior_reference(observation)
        target = None
        target_error: Exception | None = None
        try:
            target = extract_behavior_reference(
                observation,
                chunk_length=self.config.chunk_length,
                action_dim=self.config.action_dim,
            )
        except Exception as exc:
            target_error = exc

        base_output = dict(self.base_policy.infer(policy_observation))
        # Base actions are the control-critical path.  Validate them before any
        # optional work, but never modify their values.
        actions = _require_chunk(
            base_output.get("actions"),
            min_rows=self.config.chunk_length,
            action_dim=self.config.action_dim,
            label="base actions",
        )
        result = dict(base_output)
        result["actor_name"] = self.actor_name
        result["behavior_actor_checkpoint"] = None if self.actor is None else self.actor_name
        status: dict[str, Any] = {
            "mode": "actor_shadow",
            "actor_controls_robot": False,
            "control_invariant": "response.actions_are_unmodified_pi05",
            "chunk_length": self.config.chunk_length,
            "actor_name": self.actor_name,
            "behavior_actor_checkpoint": None if self.actor is None else self.actor_name,
            "token_status": "disabled" if self.token_encoder is None else "pending",
            "actor_status": "disabled" if self.actor is None else "pending",
            "learning_action_space": RLT_LEARNING_ACTION_SPACE,
            "wire_actor_action_space": PIPER_ACTION_SPACE,
            "actor_contract": RANK1_BUMP_CONTRACT,
            "action_schema_fingerprint": (
                self.config.actor_action_schema_fingerprint
            ),
            "actor_projection_profile": self.config.actor_projection_profile,
            "behavior_ref_source": (
                "request_behavior_ref" if target is not None else "response_actions"
            ),
            "behavior_ref_plan_id": None if target is None else target.plan_id,
            "behavior_ref_start_offset": None if target is None else target.start_offset,
        }
        shadow_started = self.clock()

        if target_error is not None:
            status.update(
                actor_status="invalid_behavior_ref_protocol",
                error=f"{type(target_error).__name__}: {target_error}",
            )
            status["shadow_latency_s"] = max(0.0, self.clock() - shadow_started)
            result["rlt_shadow"] = status
            return result

        try:
            state = (
                _extract_state(policy_observation)
                if target is None
                else target.conditioning_state.astype(np.float32, copy=True)
            )
            ref_absolute = (
                actions[: self.config.chunk_length, : self.config.action_dim]
                if target is None
                else target.actions
            )
            ref_learning = piper_targets_to_delta_chunk(ref_absolute, state)
        except Exception as exc:
            status.update(actor_status="invalid_reference", error=f"{type(exc).__name__}: {exc}")
            status["shadow_latency_s"] = max(0.0, self.clock() - shadow_started)
            result["rlt_shadow"] = status
            return result

        if self.token_encoder is None:
            status["shadow_latency_s"] = max(0.0, self.clock() - shadow_started)
            result["rlt_shadow"] = status
            return result

        try:
            token_started = self.clock()
            z_rl = np.asarray(self.token_encoder.encode(policy_observation), dtype=np.float32).reshape(-1)
            status["token_latency_s"] = max(0.0, self.clock() - token_started)
            if z_rl.shape != (self.config.expected_z_dim,):
                raise ValueError(f"z_rl must have shape ({self.config.expected_z_dim},), got {z_rl.shape}")
            if not np.all(np.isfinite(z_rl)):
                raise ValueError("z_rl contains non-finite values")
            result["z_rl"] = z_rl
            status["token_status"] = "ok"
        except Exception as exc:
            status.update(token_status="error", actor_status="skipped_no_token", error=f"{type(exc).__name__}: {exc}")
            status["shadow_latency_s"] = max(0.0, self.clock() - shadow_started)
            result["rlt_shadow"] = status
            return result

        if self.actor is not None:
            try:
                actor_started = self.clock()
                actor_learning = _require_chunk(
                    self.actor.predict(z_rl=z_rl, state=state, a_ref=ref_learning),
                    min_rows=self.config.chunk_length,
                    action_dim=self.config.action_dim,
                    label="shadow actor",
                )[: self.config.chunk_length, : self.config.action_dim]
                status["actor_latency_s"] = max(0.0, self.clock() - actor_started)
                actor_absolute = delta_chunk_to_piper_targets(actor_learning, state)
                result["a_actor"] = actor_absolute
                result["a_actor_action_space"] = PIPER_ACTION_SPACE
                result["a_actor_behavior_ref_contract"] = RANK1_BUMP_CONTRACT
                result["a_actor_action_schema_fingerprint"] = (
                    self.config.actor_action_schema_fingerprint
                )
                result["a_actor_projection_profile"] = (
                    self.config.actor_projection_profile
                )
                result["a_actor_behavior_ref_source"] = status["behavior_ref_source"]
                result["a_actor_behavior_ref_plan_id"] = status["behavior_ref_plan_id"]
                result["a_actor_behavior_ref_start_offset"] = status[
                    "behavior_ref_start_offset"
                ]
                status["actor_status"] = "ok"
            except Exception as exc:
                status.update(actor_status="error", actor_error=f"{type(exc).__name__}: {exc}")

        shadow_latency_s = max(0.0, self.clock() - shadow_started)
        status["shadow_latency_s"] = shadow_latency_s
        status["latency_ok"] = shadow_latency_s <= self.config.max_shadow_latency_s
        if not status["latency_ok"] and status["actor_status"] == "ok":
            # Keep z_rl for diagnostics, but do not expose a late Actor chunk as
            # a candidate for future live integration.
            result.pop("a_actor", None)
            result.pop("a_actor_action_space", None)
            status["actor_status"] = "discarded_late"
        result["rlt_shadow"] = status
        return result

    def _infer_token_batch(
        self,
        observations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Encode saved observations in one JAX batch without Pi0.5/Actor."""

        started = self.clock()
        status: dict[str, Any] = {
            "mode": "token_batch",
            "token_batch_protocol": TOKEN_BATCH_PROTOCOL,
            "actor_controls_robot": False,
            "base_policy_called": False,
            "base_rng_advanced": False,
            "token_encoder_called": self.token_encoder is not None,
            "actor_called": False,
            "token_status": "pending",
            "actor_status": "skipped_token_batch",
            "batch_size": len(observations),
        }
        if self.token_encoder is None:
            status.update(
                token_status="disabled",
                error="token encoder is disabled",
            )
            status["token_latency_s"] = max(0.0, self.clock() - started)
            return {"rlt_shadow": status}
        try:
            encode_batch = getattr(self.token_encoder, "encode_batch")
            z_rl = np.asarray(
                encode_batch(observations),
                dtype=np.float32,
            )
            expected_shape = (len(observations), self.config.expected_z_dim)
            if z_rl.shape != expected_shape:
                raise ValueError(
                    f"z_rl_batch must have shape {expected_shape}, got "
                    f"{z_rl.shape}"
                )
            if not np.all(np.isfinite(z_rl)):
                raise ValueError("z_rl_batch contains non-finite values")
            status["token_status"] = "ok"
            status["token_latency_s"] = max(0.0, self.clock() - started)
            return {
                "z_rl_batch": z_rl.copy(),
                "rlt_shadow": status,
            }
        except Exception as exc:
            status.update(
                token_status="error",
                error=f"{type(exc).__name__}: {exc}",
            )
            status["token_latency_s"] = max(0.0, self.clock() - started)
            return {"rlt_shadow": status}

    def _infer_base_only(self, policy_observation: dict[str, Any]) -> dict[str, Any]:
        """Run Pi0.5 once while skipping Token/Actor enrichment completely."""

        started = self.clock()
        base_output = dict(self.base_policy.infer(policy_observation))
        _require_chunk(
            base_output.get("actions"),
            min_rows=self.config.chunk_length,
            action_dim=self.config.action_dim,
            label="base-only actions",
        )
        result = dict(base_output)
        result["actor_name"] = self.actor_name
        result["behavior_actor_checkpoint"] = (
            None if self.actor is None else self.actor_name
        )
        result["rlt_shadow"] = {
            "mode": "base_only",
            "actor_controls_robot": False,
            "control_invariant": "response.actions_are_unmodified_pi05",
            "base_policy_called": True,
            # The wrapped flow policy samples exactly once in this request.
            "base_rng_advanced": True,
            "token_encoder_called": False,
            "actor_called": False,
            "base_status": "ok",
            "token_status": "skipped_base_only",
            "actor_status": "skipped_base_only",
            "base_policy_latency_s": max(0.0, self.clock() - started),
        }
        return result

    def _infer_actor_only(
        self,
        request: ActorOnlyRequest,
        *,
        policy_observation: dict[str, Any],
    ) -> dict[str, Any]:
        """Run Token+Actor or cached-token Actor without invoking Pi0.5."""

        target = request.target
        state = target.conditioning_state.astype(np.float32, copy=True)
        ref_absolute = target.actions.astype(np.float32, copy=True)
        result: dict[str, Any] = {
            # Keep the wire response compatible with the existing policy
            # adapter.  These are the exact request targets, not newly sampled
            # Pi0.5 actions.
            "actions": ref_absolute.copy(),
            "actor_name": self.actor_name,
            "behavior_actor_checkpoint": None if self.actor is None else self.actor_name,
        }
        status: dict[str, Any] = {
            "mode": "actor_only",
            "actor_only_protocol": request.protocol,
            "actor_controls_robot": False,
            "control_invariant": "response.actions_echo_request_behavior_ref",
            "base_policy_called": False,
            "base_rng_advanced": False,
            "token_encoder_called": request.z_rl is None,
            "actor_called": False,
            "chunk_length": self.config.chunk_length,
            "actor_name": self.actor_name,
            "behavior_actor_checkpoint": None if self.actor is None else self.actor_name,
            "token_status": "provided" if request.z_rl is not None else "pending",
            "actor_status": "disabled" if self.actor is None else "pending",
            "learning_action_space": RLT_LEARNING_ACTION_SPACE,
            "wire_actor_action_space": PIPER_ACTION_SPACE,
            "actor_contract": RANK1_BUMP_CONTRACT,
            "action_schema_fingerprint": (
                self.config.actor_action_schema_fingerprint
            ),
            "actor_projection_profile": self.config.actor_projection_profile,
            "behavior_ref_source": "request_behavior_ref",
            "behavior_ref_plan_id": target.plan_id,
            "behavior_ref_start_offset": target.start_offset,
        }
        started = self.clock()
        if request.z_rl is not None:
            z_rl = request.z_rl.astype(np.float32, copy=True)
        elif self.token_encoder is None:
            status.update(
                token_status="disabled",
                actor_status="skipped_no_token",
                error="token encoder is disabled",
            )
            status["shadow_latency_s"] = max(0.0, self.clock() - started)
            status["latency_ok"] = True
            result["rlt_shadow"] = status
            return result
        else:
            try:
                token_started = self.clock()
                z_rl = np.asarray(
                    self.token_encoder.encode(policy_observation),
                    dtype=np.float32,
                ).reshape(-1)
                status["token_latency_s"] = max(0.0, self.clock() - token_started)
                if z_rl.shape != (self.config.expected_z_dim,):
                    raise ValueError(
                        f"z_rl must have shape ({self.config.expected_z_dim},), "
                        f"got {z_rl.shape}"
                    )
                if not np.all(np.isfinite(z_rl)):
                    raise ValueError("z_rl contains non-finite values")
                status["token_status"] = "ok"
            except Exception as exc:
                status.update(
                    token_status="error",
                    actor_status="skipped_no_token",
                    error=f"{type(exc).__name__}: {exc}",
                )
                status["shadow_latency_s"] = max(0.0, self.clock() - started)
                status["latency_ok"] = (
                    status["shadow_latency_s"] <= self.config.max_shadow_latency_s
                )
                result["rlt_shadow"] = status
                return result
        result["z_rl"] = z_rl.copy()
        if self.actor is not None:
            try:
                ref_learning = piper_targets_to_delta_chunk(ref_absolute, state)
                actor_started = self.clock()
                status["actor_called"] = True
                actor_learning = _require_chunk(
                    self.actor.predict(z_rl=z_rl, state=state, a_ref=ref_learning),
                    min_rows=self.config.chunk_length,
                    action_dim=self.config.action_dim,
                    label="Actor-only actor",
                )[: self.config.chunk_length, : self.config.action_dim]
                status["actor_latency_s"] = max(0.0, self.clock() - actor_started)
                actor_absolute = delta_chunk_to_piper_targets(actor_learning, state)
                result.update(
                    {
                        "a_actor": actor_absolute,
                        "a_actor_action_space": PIPER_ACTION_SPACE,
                        "a_actor_behavior_ref_contract": RANK1_BUMP_CONTRACT,
                        "a_actor_action_schema_fingerprint": (
                            self.config.actor_action_schema_fingerprint
                        ),
                        "a_actor_projection_profile": (
                            self.config.actor_projection_profile
                        ),
                        "a_actor_behavior_ref_source": "request_behavior_ref",
                        "a_actor_behavior_ref_plan_id": target.plan_id,
                        "a_actor_behavior_ref_start_offset": target.start_offset,
                    }
                )
                status["actor_status"] = "ok"
            except Exception as exc:
                status.update(
                    actor_status="error",
                    actor_error=f"{type(exc).__name__}: {exc}",
                )
        latency_s = max(0.0, self.clock() - started)
        status["shadow_latency_s"] = latency_s
        status["latency_ok"] = latency_s <= self.config.max_shadow_latency_s
        if not status["latency_ok"] and status["actor_status"] == "ok":
            result.pop("a_actor", None)
            result.pop("a_actor_action_space", None)
            status["actor_status"] = "discarded_late"
        result["rlt_shadow"] = status
        return result


def _extract_state(observation: dict[str, Any]) -> np.ndarray:
    raw = observation.get("observation/state", observation.get("state"))
    state = np.asarray(raw, dtype=np.float32)
    if state.shape != (7,) or not np.all(np.isfinite(state)):
        raise ValueError(f"observation state must be finite with shape (7,), got {state.shape}")
    return state


def _require_chunk(value: Any, *, min_rows: int, action_dim: int, label: str) -> np.ndarray:
    chunk = np.asarray(value, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[0] < min_rows or chunk.shape[1] < action_dim:
        raise ValueError(f"{label} must have at least ({min_rows}, {action_dim}), got {chunk.shape}")
    if not np.all(np.isfinite(chunk[:min_rows, :action_dim])):
        raise ValueError(f"{label} contains non-finite values")
    return chunk
