from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np


RANK1_BUMP_WINDOW = np.asarray(
    [0.0, 0.2, 0.5, 0.8, 1.0, 1.0, 0.8, 0.5, 0.2, 0.0],
    dtype=np.float64,
)
PERSISTENT_C10_EXECUTION_CONTRACT = "persistent_c10_from_rank1_v1"
PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT = "persistent_c10_filtered_actual_v2"
PERSISTENT_FILTERED_ACTUAL_ACTION_SCHEMA_FINGERPRINT = (
    "piper_joint_delta_v4_c10_n10_stride10_behavior_ref50_"
    "persistent_filtered_actual_r005_d1_0015_d2_001_cone15_"
    "gripper_absolute_frozen_residual"
)
PERSISTENT_FILTERED_ACTUAL_GOVERNOR_FINGERPRINT = (
    "persistent_governor_v2_r005_d1_0015_d2_001_cone15_"
    "boundary060_static001_scale33_min020"
)
GRIPPER_RESIDUAL_FROZEN = "frozen"
GRIPPER_RESIDUAL_CLOSE_ASSIST = "close_only_persistent_v1"
PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT = (
    "piper_joint_delta_v5_c10_n10_stride10_behavior_ref50_"
    "persistent_filtered_actual_r005_d1_0015_d2_001_cone15_"
    "gripper_close_assist_r005_d1_0005_d2_0003_boundary0005"
)
PERSISTENT_GRIPPER_CLOSE_GOVERNOR_FINGERPRINT = (
    "persistent_governor_v3_joint_r005_d1_0015_d2_001_cone15_"
    "boundary060_gripper_close_r005_d1_0005_d2_0003_boundary0005"
)

# Ten samples from a quintic smoothstep over [0, 1].  The first C10 row is
# exactly the actually executed carry and the last row is exactly the new knot;
# this makes phase entry/re-entry and every C10 boundary position-continuous.
# The old Actor still
# predicts one rank1-bump direction from an unchanged C10 checkpoint.  This
# execution-only compatibility layer interprets that direction as the next
# persistent residual knot, then moves from the actually executed carry to the
# new knot with zero continuous-time endpoint velocity/acceleration.
_PERSISTENT_C10_BLEND_WINDOW = np.asarray(
    [
        0.0,
        0.011532794797541024,
        0.07641111619163743,
        0.2098765432098765,
        0.3966874968246709,
        0.603312503175329,
        0.790123456790123,
        0.9235888838083626,
        0.9884672052024577,
        1.0,
    ],
    dtype=np.float64,
)

# Incoming policy chunks are float32.  Comparisons are performed in float64,
# but values exactly on a configured boundary can retain ~1e-10 conversion
# noise.  This tolerance is nine orders below a radian and prevents an exact
# 0.005-rad candidate from being spuriously demoted by one projection grid bin.
_NUMERIC_TOLERANCE = 1e-9


@dataclasses.dataclass(frozen=True)
class ActorResidualGovernorConfig:
    chunk_length: int = 10
    action_dim: int = 7
    actor_residual_max_rad: float = 0.005
    actor_residual_d1_max_rad: float = 0.0015
    actor_residual_d2_max_rad: float = 0.001
    actor_direction_cone_deg: float = 15.0
    max_boundary_jump_rad: float = 0.02
    direction_static_threshold_rad: float = 0.001
    min_projection_scale: float = 0.2
    projection_scale_steps: int = 33
    rank1_fit_tolerance_rad: float = 1e-5
    gripper_residual_tolerance: float = 1e-6
    gripper_residual_mode: str = GRIPPER_RESIDUAL_FROZEN
    gripper_residual_max_close_m: float = 0.005
    gripper_residual_d1_max_m: float = 0.0005
    gripper_residual_d2_max_m: float = 0.0003
    gripper_max_boundary_jump_m: float = 0.0005
    gripper_rank1_fit_tolerance_m: float = 1e-5
    gripper_command_min_m: float = 0.0
    gripper_command_max_m: float = 0.08
    gripper_release_reference_m: float = 0.05
    gripper_release_delta_m: float = 0.002

    def validate(self) -> "ActorResidualGovernorConfig":
        if self.chunk_length != 10:
            raise ValueError("rank1_bump residual governor is fixed to C=10")
        if self.action_dim != 7:
            raise ValueError("Piper residual governor action dimension must be 7")
        for name in (
            "actor_residual_max_rad",
            "actor_residual_d1_max_rad",
            "actor_residual_d2_max_rad",
            "max_boundary_jump_rad",
            "direction_static_threshold_rad",
            "rank1_fit_tolerance_rad",
            "gripper_residual_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.actor_direction_cone_deg)
            or self.actor_direction_cone_deg <= 0.0
            or self.actor_direction_cone_deg >= 90.0
        ):
            raise ValueError("actor_direction_cone_deg must be finite and in (0, 90)")
        if (
            not math.isfinite(self.min_projection_scale)
            or self.min_projection_scale < 0.0
            or self.min_projection_scale > 1.0
        ):
            raise ValueError("min_projection_scale must be finite and in [0, 1]")
        if self.projection_scale_steps != 33:
            raise ValueError("rank1_bump projection uses the fixed 33-point scale grid")
        if self.gripper_residual_mode not in {
            GRIPPER_RESIDUAL_FROZEN,
            GRIPPER_RESIDUAL_CLOSE_ASSIST,
        }:
            raise ValueError(
                "gripper_residual_mode must be frozen or close_only_persistent_v1"
            )
        for name in (
            "gripper_residual_max_close_m",
            "gripper_residual_d1_max_m",
            "gripper_residual_d2_max_m",
            "gripper_max_boundary_jump_m",
            "gripper_rank1_fit_tolerance_m",
            "gripper_command_max_m",
            "gripper_release_reference_m",
            "gripper_release_delta_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.gripper_command_min_m)
            or self.gripper_command_min_m < 0.0
            or self.gripper_command_min_m >= self.gripper_command_max_m
        ):
            raise ValueError("gripper command bounds are invalid")
        if not (
            self.gripper_command_min_m
            < self.gripper_release_reference_m
            < self.gripper_command_max_m
        ):
            raise ValueError("gripper_release_reference_m must lie inside command bounds")
        return self


@dataclasses.dataclass(frozen=True)
class GovernedActorPlan:
    plan_id: str
    behavior_plan_id: str
    behavior_start_offset: int
    approved: bool
    rejection_reason: str | None
    safe_actions: np.ndarray
    raw_residual: np.ndarray
    safe_residual: np.ndarray
    rank1_direction: np.ndarray
    projection_scale: float
    raw_rank1_fit_error_max_rad: float
    raw_gripper_residual_max: float
    safe_residual_max_rad: float
    safe_residual_d1_max_rad: float
    safe_residual_d2_max_rad: float
    direction_min_cosine: float | None
    direction_violation_count: int
    boundary_jump_max_rad: float
    boundary_jump_limit_rad: float

    def action_at(self, offset: int) -> np.ndarray:
        index = int(offset)
        if index < 0 or index >= int(self.safe_actions.shape[0]):
            raise IndexError(f"Actor plan offset {index} is outside the cached C10 chunk")
        return self.safe_actions[index].astype(np.float32, copy=True)

    def metadata(self) -> dict[str, Any]:
        return {
            "actor_governor_contract": "rank1_bump_v1",
            "actor_governor_plan_id": self.plan_id,
            "actor_governor_behavior_plan_id": self.behavior_plan_id,
            "actor_governor_behavior_start_offset": self.behavior_start_offset,
            "actor_governor_approved": self.approved,
            "actor_governor_rejection_reason": self.rejection_reason,
            "actor_governor_projection_scale": self.projection_scale,
            "actor_governor_rank1_direction": self.rank1_direction.tolist(),
            "actor_governor_raw_rank1_fit_error_max_rad": self.raw_rank1_fit_error_max_rad,
            "actor_governor_raw_gripper_residual_max": self.raw_gripper_residual_max,
            "actor_governor_residual_max_rad": self.safe_residual_max_rad,
            "actor_governor_residual_d1_max_rad": self.safe_residual_d1_max_rad,
            "actor_governor_residual_d2_max_rad": self.safe_residual_d2_max_rad,
            "actor_governor_direction_min_cosine": self.direction_min_cosine,
            "actor_governor_direction_violation_count": self.direction_violation_count,
            "actor_governor_boundary_jump_max_rad": self.boundary_jump_max_rad,
            "actor_governor_boundary_jump_limit_rad": self.boundary_jump_limit_rad,
            "actor_governor_first_residual": self.safe_residual[0].tolist(),
            "actor_governor_last_residual": self.safe_residual[-1].tolist(),
        }


class ActorResidualGovernor:
    """Projects and caches a complete Actor C10 plan before any row can execute."""

    def __init__(self, config: ActorResidualGovernorConfig = ActorResidualGovernorConfig()) -> None:
        self.config = config.validate()
        self._plans: dict[str, GovernedActorPlan] = {}

    def prepare_plan(
        self,
        *,
        plan_id: str,
        behavior_plan_id: str,
        behavior_start_offset: int,
        behavior_ref: np.ndarray,
        raw_actor: np.ndarray,
        boundary_anchor: np.ndarray,
    ) -> GovernedActorPlan:
        plan_key = str(plan_id)
        cached = self._plans.get(plan_key)
        if cached is not None:
            if (
                cached.behavior_plan_id != str(behavior_plan_id)
                or cached.behavior_start_offset != int(behavior_start_offset)
            ):
                raise ValueError("Actor plan_id was reused for a different behavior_ref target")
            return cached
        plan = govern_actor_plan(
            plan_id=plan_key,
            behavior_plan_id=str(behavior_plan_id),
            behavior_start_offset=int(behavior_start_offset),
            behavior_ref=behavior_ref,
            raw_actor=raw_actor,
            boundary_anchor=boundary_anchor,
            config=self.config,
        )
        self._plans[plan_key] = plan
        return plan

    def get(self, plan_id: str) -> GovernedActorPlan | None:
        return self._plans.get(str(plan_id))


def govern_actor_plan(
    *,
    plan_id: str,
    behavior_plan_id: str,
    behavior_start_offset: int,
    behavior_ref: np.ndarray,
    raw_actor: np.ndarray,
    boundary_anchor: np.ndarray,
    config: ActorResidualGovernorConfig = ActorResidualGovernorConfig(),
) -> GovernedActorPlan:
    """Project one raw Actor chunk onto the shared rank1-bump safety contract.

    A single six-joint direction is fitted to the raw residual.  The fixed C10
    bump supplies its time profile, and one scalar from ``linspace(0, 1, 33)``
    is applied to the entire chunk.  A scale is eligible only when every row,
    including the entry step from ``boundary_anchor``, satisfies all limits.
    """

    config = config.validate()
    reference = _require_reference(behavior_ref, config=config)
    anchor = np.asarray(boundary_anchor, dtype=np.float64)
    if anchor.shape != (config.action_dim,) or not np.all(np.isfinite(anchor)):
        raise ValueError(
            f"boundary_anchor must be finite with shape ({config.action_dim},), got {anchor.shape}"
        )
    raw = np.asarray(raw_actor, dtype=np.float64)
    if raw.shape != reference.shape or not np.all(np.isfinite(raw)):
        return _rejected_invalid_actor(
            plan_id=plan_id,
            behavior_plan_id=behavior_plan_id,
            behavior_start_offset=behavior_start_offset,
            reference=reference,
            raw=raw,
        )

    raw_residual = raw - reference
    joint_raw_residual = raw_residual[:, :6]
    denominator = float(np.dot(RANK1_BUMP_WINDOW, RANK1_BUMP_WINDOW))
    direction = (
        np.sum(RANK1_BUMP_WINDOW[:, None] * joint_raw_residual, axis=0) / denominator
    )
    rank1_fit = RANK1_BUMP_WINDOW[:, None] * direction[None, :]
    fit_error = joint_raw_residual - rank1_fit
    raw_fit_error_max = float(np.max(np.abs(fit_error)))
    raw_gripper_max = float(np.max(np.abs(raw_residual[:, 6])))
    if raw_fit_error_max > config.rank1_fit_tolerance_rad + 1e-12:
        return _rejected_contract_actor(
            plan_id=plan_id,
            behavior_plan_id=behavior_plan_id,
            behavior_start_offset=behavior_start_offset,
            reference=reference,
            raw_residual=raw_residual,
            direction=direction,
            fit_error_max=raw_fit_error_max,
            raw_gripper_max=raw_gripper_max,
            reason="rank1_contract_violation",
        )
    if raw_gripper_max > config.gripper_residual_tolerance + 1e-12:
        return _rejected_contract_actor(
            plan_id=plan_id,
            behavior_plan_id=behavior_plan_id,
            behavior_start_offset=behavior_start_offset,
            reference=reference,
            raw_residual=raw_residual,
            direction=direction,
            fit_error_max=raw_fit_error_max,
            raw_gripper_max=raw_gripper_max,
            reason="gripper_residual_not_frozen",
        )

    selected_scale = 0.0
    selected_residual = np.zeros_like(reference)
    selected_metrics = _constraint_metrics(
        reference=reference,
        safe_residual=selected_residual,
        boundary_anchor=anchor,
        config=config,
    )
    for scale in np.linspace(0.0, 1.0, config.projection_scale_steps, dtype=np.float64)[::-1]:
        candidate_residual = np.zeros_like(reference)
        candidate_residual[:, :6] = scale * rank1_fit
        metrics = _constraint_metrics(
            reference=reference,
            safe_residual=candidate_residual,
            boundary_anchor=anchor,
            config=config,
        )
        if metrics["valid"]:
            selected_scale = float(scale)
            selected_residual = candidate_residual
            selected_metrics = metrics
            break

    approved = selected_scale + 1e-12 >= config.min_projection_scale
    if approved:
        rejection_reason = None
    elif selected_metrics["boundary_jump_max"] > config.max_boundary_jump_rad + _NUMERIC_TOLERANCE:
        rejection_reason = "boundary_jump_exceeds_limit"
    else:
        rejection_reason = "projection_scale_below_minimum"
    if not approved:
        # Rejection is atomic: callers receive the behavior reference as the
        # cached safe payload and must route the whole C10 through Pi0.5.
        selected_residual = np.zeros_like(reference)
        selected_metrics = _constraint_metrics(
            reference=reference,
            safe_residual=selected_residual,
            boundary_anchor=anchor,
            config=config,
        )
    safe_actions = reference.copy()
    safe_actions[:, :6] += selected_residual[:, :6]
    # Gripper residual is frozen by construction.
    safe_actions[:, 6] = reference[:, 6]
    return GovernedActorPlan(
        plan_id=str(plan_id),
        behavior_plan_id=str(behavior_plan_id),
        behavior_start_offset=int(behavior_start_offset),
        approved=approved,
        rejection_reason=rejection_reason,
        safe_actions=safe_actions.astype(np.float32),
        raw_residual=raw_residual.astype(np.float32),
        safe_residual=selected_residual.astype(np.float32),
        rank1_direction=direction.astype(np.float32),
        projection_scale=selected_scale,
        raw_rank1_fit_error_max_rad=raw_fit_error_max,
        raw_gripper_residual_max=raw_gripper_max,
        safe_residual_max_rad=float(selected_metrics["residual_max"]),
        safe_residual_d1_max_rad=float(selected_metrics["d1_max"]),
        safe_residual_d2_max_rad=float(selected_metrics["d2_max"]),
        direction_min_cosine=selected_metrics["min_cosine"],
        direction_violation_count=int(selected_metrics["direction_violations"]),
        boundary_jump_max_rad=float(selected_metrics["boundary_jump_max"]),
        boundary_jump_limit_rad=float(config.max_boundary_jump_rad),
    )


def _require_reference(value: np.ndarray, *, config: ActorResidualGovernorConfig) -> np.ndarray:
    reference = np.asarray(value, dtype=np.float64)
    expected = (config.chunk_length, config.action_dim)
    if reference.shape != expected or not np.all(np.isfinite(reference)):
        raise ValueError(f"behavior_ref must be finite with shape {expected}, got {reference.shape}")
    return reference


def _constraint_metrics(
    *,
    reference: np.ndarray,
    safe_residual: np.ndarray,
    boundary_anchor: np.ndarray,
    config: ActorResidualGovernorConfig,
) -> dict[str, Any]:
    joint_residual = safe_residual[:, :6]
    padded = np.concatenate(
        [np.zeros((1, 6), dtype=np.float64), joint_residual, np.zeros((1, 6), dtype=np.float64)],
        axis=0,
    )
    residual_d1 = np.diff(padded, axis=0)
    residual_d2 = np.diff(residual_d1, axis=0)
    residual_max = float(np.max(np.abs(joint_residual)))
    d1_max = float(np.max(np.abs(residual_d1)))
    d2_max = float(np.max(np.abs(residual_d2)))

    safe = reference.copy()
    safe[:, :6] += joint_residual
    ref_steps = np.diff(
        np.concatenate([boundary_anchor[None, :6], reference[:, :6]], axis=0),
        axis=0,
    )
    actor_steps = np.diff(
        np.concatenate([boundary_anchor[None, :6], safe[:, :6]], axis=0),
        axis=0,
    )
    boundary_jump_max = float(np.max(np.abs(actor_steps[0])))
    cone_cosine = math.cos(math.radians(config.actor_direction_cone_deg))
    direction_violations = 0
    cosines: list[float] = []
    # Keep the runtime compatible with the ROS/Pika Python environment, which
    # may still be Python 3.9 (``zip(strict=...)`` was added in Python 3.10).
    if ref_steps.shape != actor_steps.shape:
        raise ValueError("reference and Actor step arrays must have the same shape")
    for ref_step, actor_step in zip(ref_steps, actor_steps):
        ref_norm = float(np.linalg.norm(ref_step))
        actor_norm = float(np.linalg.norm(actor_step))
        if ref_norm >= config.direction_static_threshold_rad:
            dot = float(np.dot(ref_step, actor_step))
            cosine = -1.0 if actor_norm <= 1e-12 else dot / (ref_norm * actor_norm)
            cosines.append(cosine)
            if dot < -_NUMERIC_TOLERANCE or cosine + _NUMERIC_TOLERANCE < cone_cosine:
                direction_violations += 1
        elif actor_norm > config.direction_static_threshold_rad + _NUMERIC_TOLERANCE:
            direction_violations += 1

    valid = bool(
        residual_max <= config.actor_residual_max_rad + _NUMERIC_TOLERANCE
        and d1_max <= config.actor_residual_d1_max_rad + _NUMERIC_TOLERANCE
        and d2_max <= config.actor_residual_d2_max_rad + _NUMERIC_TOLERANCE
        and boundary_jump_max <= config.max_boundary_jump_rad + _NUMERIC_TOLERANCE
        and direction_violations == 0
        and np.allclose(joint_residual[0], 0.0, rtol=0.0, atol=1e-12)
        and np.allclose(joint_residual[-1], 0.0, rtol=0.0, atol=1e-12)
    )
    return {
        "valid": valid,
        "residual_max": residual_max,
        "d1_max": d1_max,
        "d2_max": d2_max,
        "min_cosine": None if not cosines else float(min(cosines)),
        "direction_violations": direction_violations,
        "boundary_jump_max": boundary_jump_max,
    }


def _rejected_invalid_actor(
    *,
    plan_id: str,
    behavior_plan_id: str,
    behavior_start_offset: int,
    reference: np.ndarray,
    raw: np.ndarray,
) -> GovernedActorPlan:
    zeros = np.zeros_like(reference, dtype=np.float32)
    raw_residual = zeros.copy()
    if raw.shape == reference.shape:
        raw_residual = (raw - reference).astype(np.float32)
    return GovernedActorPlan(
        plan_id=str(plan_id),
        behavior_plan_id=str(behavior_plan_id),
        behavior_start_offset=int(behavior_start_offset),
        approved=False,
        rejection_reason="invalid_actor_chunk",
        safe_actions=reference.astype(np.float32),
        raw_residual=raw_residual,
        safe_residual=zeros,
        rank1_direction=np.zeros(6, dtype=np.float32),
        projection_scale=0.0,
        raw_rank1_fit_error_max_rad=float("inf"),
        raw_gripper_residual_max=float("inf"),
        safe_residual_max_rad=0.0,
        safe_residual_d1_max_rad=0.0,
        safe_residual_d2_max_rad=0.0,
        direction_min_cosine=None,
        direction_violation_count=0,
        boundary_jump_max_rad=float("inf"),
        boundary_jump_limit_rad=ActorResidualGovernorConfig().max_boundary_jump_rad,
    )


def _rejected_contract_actor(
    *,
    plan_id: str,
    behavior_plan_id: str,
    behavior_start_offset: int,
    reference: np.ndarray,
    raw_residual: np.ndarray,
    direction: np.ndarray,
    fit_error_max: float,
    raw_gripper_max: float,
    reason: str,
) -> GovernedActorPlan:
    zeros = np.zeros_like(reference, dtype=np.float32)
    return GovernedActorPlan(
        plan_id=str(plan_id),
        behavior_plan_id=str(behavior_plan_id),
        behavior_start_offset=int(behavior_start_offset),
        approved=False,
        rejection_reason=reason,
        safe_actions=reference.astype(np.float32),
        raw_residual=raw_residual.astype(np.float32),
        safe_residual=zeros,
        rank1_direction=direction.astype(np.float32),
        projection_scale=0.0,
        raw_rank1_fit_error_max_rad=float(fit_error_max),
        raw_gripper_residual_max=float(raw_gripper_max),
        safe_residual_max_rad=0.0,
        safe_residual_d1_max_rad=0.0,
        safe_residual_d2_max_rad=0.0,
        direction_min_cosine=None,
        direction_violation_count=0,
        boundary_jump_max_rad=float("inf"),
        boundary_jump_limit_rad=ActorResidualGovernorConfig().max_boundary_jump_rad,
    )


@dataclasses.dataclass(frozen=True)
class PersistentGovernedActorPlan:
    """A C10 execution plan derived explicitly from a legacy rank1 Actor.

    ``approved`` means every command in ``safe_actions`` can be used by the
    runtime. ``target_update_accepted`` is separate: an unsafe or too-small new
    Actor update may be rejected while the already-executed carry is held. This
    prevents a rejected prediction from dropping the residual to zero at a C10
    boundary.
    """

    plan_id: str
    behavior_plan_id: str
    behavior_start_offset: int
    approved: bool
    target_update_accepted: bool
    rejection_reason: str | None
    safe_actions: np.ndarray
    raw_residual: np.ndarray
    safe_residual: np.ndarray
    rank1_direction: np.ndarray
    boundary_anchor: np.ndarray
    carry_in: np.ndarray
    previous_carry: np.ndarray
    carry_out: np.ndarray
    projection_scale: float
    raw_rank1_fit_error_max_rad: float
    raw_gripper_residual_max: float
    safe_residual_max_rad: float
    safe_residual_d1_max_rad: float
    safe_residual_d2_max_rad: float
    direction_min_cosine: float | None
    direction_violation_count: int
    boundary_jump_max_rad: float
    boundary_jump_limit_rad: float
    hold_only: bool = False
    safe_gripper_residual_max_m: float = 0.0
    safe_gripper_d1_max_m: float = 0.0
    safe_gripper_d2_max_m: float = 0.0
    gripper_boundary_jump_max_m: float = 0.0
    gripper_release_intent: bool = False
    gripper_residual_mode: str = GRIPPER_RESIDUAL_FROZEN

    def action_at(self, offset: int) -> np.ndarray:
        index = int(offset)
        if index < 0 or index >= int(self.safe_actions.shape[0]):
            raise IndexError(f"Actor plan offset {index} is outside the cached C10 chunk")
        return self.safe_actions[index].astype(np.float32, copy=True)

    def residual_at(self, offset: int) -> np.ndarray:
        index = int(offset)
        if index < 0 or index >= int(self.safe_residual.shape[0]):
            raise IndexError(f"Actor residual offset {index} is outside the cached C10 chunk")
        return self.safe_residual[index].astype(np.float32, copy=True)

    def metadata(self) -> dict[str, Any]:
        return {
            "actor_governor_contract": PERSISTENT_C10_EXECUTION_CONTRACT,
            "actor_governor_legacy_input_contract": "rank1_bump_v1",
            "actor_governor_plan_id": self.plan_id,
            "actor_governor_behavior_plan_id": self.behavior_plan_id,
            "actor_governor_behavior_start_offset": self.behavior_start_offset,
            "actor_governor_approved": self.approved,
            "actor_governor_rejection_reason": self.rejection_reason,
            "actor_governor_projection_scale": self.projection_scale,
            "actor_governor_rank1_direction": self.rank1_direction.tolist(),
            "actor_governor_raw_rank1_fit_error_max_rad": (
                self.raw_rank1_fit_error_max_rad
            ),
            "actor_governor_raw_gripper_residual_max": self.raw_gripper_residual_max,
            "actor_governor_residual_max_rad": self.safe_residual_max_rad,
            "actor_governor_residual_d1_max_rad": self.safe_residual_d1_max_rad,
            "actor_governor_residual_d2_max_rad": self.safe_residual_d2_max_rad,
            "actor_governor_direction_min_cosine": self.direction_min_cosine,
            "actor_governor_direction_violation_count": self.direction_violation_count,
            "actor_governor_boundary_jump_max_rad": self.boundary_jump_max_rad,
            "actor_governor_boundary_jump_limit_rad": self.boundary_jump_limit_rad,
            "actor_governor_first_residual": self.safe_residual[0].tolist(),
            "actor_governor_last_residual": self.safe_residual[-1].tolist(),
            "actor_execution_boundary_anchor": self.boundary_anchor.tolist(),
            "actor_execution_contract": PERSISTENT_C10_EXECUTION_CONTRACT,
            "actor_persistent_carry_in": self.carry_in.tolist(),
            "actor_persistent_previous_carry": self.previous_carry.tolist(),
            "actor_persistent_carry_out": self.carry_out.tolist(),
            "actor_persistent_target_update_accepted": self.target_update_accepted,
            "actor_persistent_hold": self.hold_only,
            "actor_gripper_residual_mode": self.gripper_residual_mode,
            "actor_gripper_residual_max_m": self.safe_gripper_residual_max_m,
            "actor_gripper_residual_d1_max_m": self.safe_gripper_d1_max_m,
            "actor_gripper_residual_d2_max_m": self.safe_gripper_d2_max_m,
            "actor_gripper_boundary_jump_max_m": (
                self.gripper_boundary_jump_max_m
            ),
            "actor_gripper_release_intent": self.gripper_release_intent,
        }


@dataclasses.dataclass(frozen=True)
class FilteredActualExecutionCertificate:
    """Pre-publish proof for one post-safety-filter persistent Actor row.

    The real and base-only safety filters evolve independently from the same
    state at phase entry.  Their output difference is therefore the physical
    Actor contribution, including the filter's memory, rather than the
    instantaneous ``alpha * planned_residual`` from a per-frame clone.
    """

    plan_id: str
    offset: int
    approved: bool
    rejection_reason: str | None
    filtered_base_action: np.ndarray
    filtered_actual_action: np.ndarray
    actual_residual: np.ndarray
    carry_in: np.ndarray
    previous_carry: np.ndarray
    residual_max_rad: float
    residual_d1_max_rad: float
    residual_d2_max_rad: float
    direction_min_cosine: float | None
    direction_violation_count: int
    boundary_jump_max_rad: float
    gripper_residual_max: float
    gripper_d1_max_m: float = 0.0
    gripper_d2_max_m: float = 0.0
    gripper_boundary_jump_max_m: float = 0.0

    def metadata(self) -> dict[str, Any]:
        return {
            "actor_filtered_actual_certificate_approved": self.approved,
            "actor_filtered_actual_certificate_rejection_reason": (
                self.rejection_reason
            ),
            "actor_filtered_actual_residual_max_rad": self.residual_max_rad,
            "actor_filtered_actual_residual_d1_max_rad": self.residual_d1_max_rad,
            "actor_filtered_actual_residual_d2_max_rad": self.residual_d2_max_rad,
            "actor_filtered_actual_direction_min_cosine": (
                self.direction_min_cosine
            ),
            "actor_filtered_actual_direction_violation_count": (
                self.direction_violation_count
            ),
            "actor_filtered_actual_boundary_jump_max_rad": (
                self.boundary_jump_max_rad
            ),
            "actor_filtered_actual_gripper_residual_max": (
                self.gripper_residual_max
            ),
            "actor_filtered_actual_gripper_residual_d1_max_m": (
                self.gripper_d1_max_m
            ),
            "actor_filtered_actual_gripper_residual_d2_max_m": (
                self.gripper_d2_max_m
            ),
            "actor_filtered_actual_gripper_boundary_jump_max_m": (
                self.gripper_boundary_jump_max_m
            ),
        }


class PersistentActorResidualGovernor:
    """Stateful, execution-committed C10 carry for old rank1 checkpoints.

    Planning never advances the carry.  The control loop must call
    :meth:`mark_executed` only after the final mux/safety-selected command is
    actually published.  Human takeover, phase exit, episode reset, or any
    external command source must call :meth:`reset_execution_state`.
    """

    def __init__(self, config: ActorResidualGovernorConfig = ActorResidualGovernorConfig()) -> None:
        self.config = config.validate()
        self._plans: dict[str, PersistentGovernedActorPlan] = {}
        self._current_residual = np.zeros(self.config.action_dim, dtype=np.float64)
        self._previous_residual = np.zeros(self.config.action_dim, dtype=np.float64)
        self._last_executed_plan_id: str | None = None
        self._last_executed_offset: int | None = None
        self._last_reset_reason = "initial"

    @property
    def current_residual(self) -> np.ndarray:
        return self._current_residual.astype(np.float32, copy=True)

    @property
    def previous_residual(self) -> np.ndarray:
        return self._previous_residual.astype(np.float32, copy=True)

    @property
    def last_reset_reason(self) -> str:
        return self._last_reset_reason

    def prepare_plan(
        self,
        *,
        plan_id: str,
        behavior_plan_id: str,
        behavior_start_offset: int,
        behavior_ref: np.ndarray,
        raw_actor: np.ndarray,
        boundary_anchor: np.ndarray,
    ) -> PersistentGovernedActorPlan:
        plan_key = str(plan_id)
        cached = self._plans.get(plan_key)
        if cached is not None:
            if (
                cached.behavior_plan_id != str(behavior_plan_id)
                or cached.behavior_start_offset != int(behavior_start_offset)
            ):
                raise ValueError("Actor plan_id was reused for a different behavior_ref target")
            return cached
        plan = govern_persistent_actor_plan(
            plan_id=plan_key,
            behavior_plan_id=str(behavior_plan_id),
            behavior_start_offset=int(behavior_start_offset),
            behavior_ref=behavior_ref,
            raw_actor=raw_actor,
            boundary_anchor=boundary_anchor,
            carry_in=self._current_residual,
            previous_carry=self._previous_residual,
            config=self.config,
        )
        self._plans[plan_key] = plan
        return plan

    def prepare_hold_plan(
        self,
        *,
        plan_id: str,
        behavior_plan_id: str,
        behavior_start_offset: int,
        behavior_ref: np.ndarray,
        boundary_anchor: np.ndarray,
    ) -> PersistentGovernedActorPlan:
        """Hold the last executed correction when a fresh Actor is unavailable."""

        plan_key = str(plan_id)
        cached = self._plans.get(plan_key)
        if cached is not None:
            return cached
        plan = govern_persistent_hold_plan(
            plan_id=plan_key,
            behavior_plan_id=str(behavior_plan_id),
            behavior_start_offset=int(behavior_start_offset),
            behavior_ref=behavior_ref,
            boundary_anchor=boundary_anchor,
            carry_in=self._current_residual,
            previous_carry=self._previous_residual,
            config=self.config,
        )
        self._plans[plan_key] = plan
        return plan

    def get(self, plan_id: str) -> PersistentGovernedActorPlan | None:
        return self._plans.get(str(plan_id))

    def mark_executed(self, plan_id: str, offset: int) -> np.ndarray:
        """Commit exactly one residual after its command was published."""

        plan_key = str(plan_id)
        plan = self._plans.get(plan_key)
        if plan is None:
            raise KeyError(f"unknown persistent Actor plan_id {plan_key!r}")
        index = int(offset)
        if index < 0 or index >= self.config.chunk_length:
            raise IndexError(f"Actor plan offset {index} is outside the cached C10 chunk")
        if self._last_executed_plan_id == plan_key:
            expected = 0 if self._last_executed_offset is None else self._last_executed_offset + 1
            if index != expected:
                raise ValueError(
                    f"persistent Actor plan offsets must execute in order; "
                    f"expected {expected}, got {index}"
                )
        elif index != 0:
            raise ValueError("a new persistent Actor plan must begin at offset 0")
        next_residual = np.asarray(plan.safe_residual[index], dtype=np.float64)
        self._previous_residual = self._current_residual.copy()
        self._current_residual = next_residual.copy()
        self._last_executed_plan_id = plan_key
        self._last_executed_offset = index
        return self.current_residual

    def certify_filtered_execution(
        self,
        *,
        plan_id: str,
        offset: int,
        filtered_base_action: np.ndarray,
        filtered_actual_action: np.ndarray,
        base_boundary_anchor: np.ndarray,
        actual_boundary_anchor: np.ndarray,
    ) -> FilteredActualExecutionCertificate:
        """Certify the *actual* post-filter residual without mutating carry.

        This method is intentionally called before publication.  A rejected
        certificate lets the runtime publish the already-filtered base-only
        fallback and synchronize the real filter to the shadow filter.  The
        matching :meth:`mark_filtered_executed` call is allowed only after the
        certified absolute command was successfully published.
        """

        plan_key = str(plan_id)
        plan = self._plans.get(plan_key)
        if plan is None:
            raise KeyError(f"unknown persistent Actor plan_id {plan_key!r}")
        index = int(offset)
        if index < 0 or index >= self.config.chunk_length:
            raise IndexError(
                f"Actor plan offset {index} is outside the cached C10 chunk"
            )
        if not plan.approved:
            raise ValueError("cannot certify a rejected persistent Actor plan")
        self._validate_execution_sequence(plan_key, index)

        base = _require_action_vector(
            filtered_base_action,
            config=self.config,
            label="filtered_base_action",
        )
        actual = _require_action_vector(
            filtered_actual_action,
            config=self.config,
            label="filtered_actual_action",
        )
        base_anchor = _require_action_vector(
            base_boundary_anchor,
            config=self.config,
            label="base_boundary_anchor",
        )
        actual_anchor = _require_action_vector(
            actual_boundary_anchor,
            config=self.config,
            label="actual_boundary_anchor",
        )
        actual_residual = actual - base
        joint_residual = actual_residual[:6]
        current = self._current_residual.copy()
        previous = self._previous_residual.copy()

        residual_d1 = joint_residual - current[:6]
        previous_d1 = current[:6] - previous[:6]
        residual_d2 = residual_d1 - previous_d1
        residual_max = float(np.max(np.abs(joint_residual)))
        d1_max = float(np.max(np.abs(residual_d1)))
        d2_max = float(np.max(np.abs(residual_d2)))
        gripper_residual_max = abs(float(actual_residual[6]))
        gripper_d1_max = abs(float(actual_residual[6] - current[6]))
        gripper_previous_d1 = float(current[6] - previous[6])
        gripper_d2_max = abs(
            float(actual_residual[6] - current[6] - gripper_previous_d1)
        )
        gripper_boundary_jump = gripper_d1_max

        base_step = base[:6] - base_anchor[:6]
        actual_step = actual[:6] - actual_anchor[:6]
        boundary_jump_max = float(np.max(np.abs(actual_step)))
        ref_norm = float(np.linalg.norm(base_step))
        actor_norm = float(np.linalg.norm(actual_step))
        direction_violations = 0
        direction_min_cosine: float | None = None
        if ref_norm >= self.config.direction_static_threshold_rad:
            dot = float(np.dot(base_step, actual_step))
            direction_min_cosine = (
                -1.0
                if actor_norm <= 1e-12
                else dot / (ref_norm * actor_norm)
            )
            cone_cosine = math.cos(
                math.radians(self.config.actor_direction_cone_deg)
            )
            if (
                dot < -_NUMERIC_TOLERANCE
                or direction_min_cosine + _NUMERIC_TOLERANCE < cone_cosine
            ):
                direction_violations = 1
        elif (
            actor_norm
            > self.config.direction_static_threshold_rad + _NUMERIC_TOLERANCE
        ):
            direction_violations = 1

        violations: list[str] = []
        if residual_max > self.config.actor_residual_max_rad + _NUMERIC_TOLERANCE:
            violations.append("residual_max")
        if d1_max > self.config.actor_residual_d1_max_rad + _NUMERIC_TOLERANCE:
            violations.append("residual_d1")
        if d2_max > self.config.actor_residual_d2_max_rad + _NUMERIC_TOLERANCE:
            violations.append("residual_d2")
        if (
            boundary_jump_max
            > self.config.max_boundary_jump_rad + _NUMERIC_TOLERANCE
        ):
            violations.append("boundary_jump")
        if direction_violations:
            violations.append("direction_cone")
        if self.config.gripper_residual_mode == GRIPPER_RESIDUAL_FROZEN:
            if (
                gripper_residual_max
                > self.config.gripper_residual_tolerance + _NUMERIC_TOLERANCE
            ):
                violations.append("gripper_residual")
        else:
            expected_gripper_residual = float(plan.safe_residual[index, 6])
            if (
                abs(float(actual_residual[6]) - expected_gripper_residual)
                > 2e-6
            ):
                violations.append("gripper_execution_mismatch")
            if actual_residual[6] > self.config.gripper_residual_tolerance:
                violations.append("gripper_open_residual")
            if (
                gripper_residual_max
                > self.config.gripper_residual_max_close_m + _NUMERIC_TOLERANCE
            ):
                violations.append("gripper_residual")
            if (
                gripper_d1_max
                > self.config.gripper_residual_d1_max_m + _NUMERIC_TOLERANCE
            ):
                violations.append("gripper_residual_d1")
            if (
                gripper_d2_max
                > self.config.gripper_residual_d2_max_m + _NUMERIC_TOLERANCE
            ):
                violations.append("gripper_residual_d2")
            if (
                gripper_boundary_jump
                > self.config.gripper_max_boundary_jump_m + _NUMERIC_TOLERANCE
            ):
                violations.append("gripper_boundary_jump")
            if (
                actual[6]
                < self.config.gripper_command_min_m - _NUMERIC_TOLERANCE
                or actual[6]
                > self.config.gripper_command_max_m + _NUMERIC_TOLERANCE
            ):
                violations.append("gripper_command_range")

        return FilteredActualExecutionCertificate(
            plan_id=plan_key,
            offset=index,
            approved=not violations,
            rejection_reason=None if not violations else ",".join(violations),
            filtered_base_action=base.astype(np.float32),
            filtered_actual_action=actual.astype(np.float32),
            actual_residual=actual_residual.astype(np.float32),
            carry_in=current.astype(np.float32),
            previous_carry=previous.astype(np.float32),
            residual_max_rad=residual_max,
            residual_d1_max_rad=d1_max,
            residual_d2_max_rad=d2_max,
            direction_min_cosine=direction_min_cosine,
            direction_violation_count=direction_violations,
            boundary_jump_max_rad=boundary_jump_max,
            gripper_residual_max=gripper_residual_max,
            gripper_d1_max_m=gripper_d1_max,
            gripper_d2_max_m=gripper_d2_max,
            gripper_boundary_jump_max_m=gripper_boundary_jump,
        )

    def mark_filtered_executed(
        self,
        certificate: FilteredActualExecutionCertificate,
    ) -> np.ndarray:
        """Commit a pre-certified post-filter residual after publication."""

        if not isinstance(certificate, FilteredActualExecutionCertificate):
            raise TypeError(
                "filtered execution commit requires its exact certificate"
            )
        if not certificate.approved:
            raise ValueError("cannot commit a rejected filtered execution")
        plan_key = str(certificate.plan_id)
        if plan_key not in self._plans:
            raise KeyError(f"unknown persistent Actor plan_id {plan_key!r}")
        index = int(certificate.offset)
        self._validate_execution_sequence(plan_key, index)
        if not np.allclose(
            self._current_residual,
            certificate.carry_in,
            rtol=0.0,
            atol=1e-9,
        ) or not np.allclose(
            self._previous_residual,
            certificate.previous_carry,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                "persistent carry changed between filtered certification and commit"
            )
        next_residual = np.asarray(
            certificate.actual_residual,
            dtype=np.float64,
        )
        self._previous_residual = self._current_residual.copy()
        self._current_residual = next_residual.copy()
        self._last_executed_plan_id = plan_key
        self._last_executed_offset = index
        return self.current_residual

    def _validate_execution_sequence(self, plan_key: str, index: int) -> None:
        if self._last_executed_plan_id == plan_key:
            expected = (
                0
                if self._last_executed_offset is None
                else self._last_executed_offset + 1
            )
            if index != expected:
                raise ValueError(
                    "persistent Actor plan offsets must execute in order; "
                    f"expected {expected}, got {index}"
                )
        elif index != 0:
            raise ValueError("a new persistent Actor plan must begin at offset 0")

    def reset_execution_state(self, reason: str) -> None:
        reset_reason = str(reason).strip()
        if not reset_reason:
            raise ValueError("persistent Actor reset reason must be non-empty")
        self._previous_residual.fill(0.0)
        self._current_residual.fill(0.0)
        self._last_executed_plan_id = None
        self._last_executed_offset = None
        self._last_reset_reason = reset_reason


def govern_persistent_actor_plan(
    *,
    plan_id: str,
    behavior_plan_id: str,
    behavior_start_offset: int,
    behavior_ref: np.ndarray,
    raw_actor: np.ndarray,
    boundary_anchor: np.ndarray,
    carry_in: np.ndarray,
    previous_carry: np.ndarray,
    config: ActorResidualGovernorConfig = ActorResidualGovernorConfig(),
) -> PersistentGovernedActorPlan:
    """Convert a legacy rank1 direction into a persistent C10 residual knot."""

    config = config.validate()
    reference = _require_reference(behavior_ref, config=config)
    anchor = _require_action_vector(
        boundary_anchor,
        config=config,
        label="boundary_anchor",
    )
    carry = _require_action_vector(carry_in, config=config, label="carry_in")
    previous = _require_action_vector(
        previous_carry,
        config=config,
        label="previous_carry",
    )
    if (
        config.gripper_residual_mode == GRIPPER_RESIDUAL_FROZEN
        and abs(float(carry[6])) > config.gripper_residual_tolerance
    ):
        raise ValueError("persistent carry gripper residual must stay frozen at zero")
    if (
        config.gripper_residual_mode == GRIPPER_RESIDUAL_FROZEN
        and abs(float(previous[6])) > config.gripper_residual_tolerance
    ):
        raise ValueError("previous persistent carry gripper residual must stay frozen at zero")
    raw = np.asarray(raw_actor, dtype=np.float64)
    if raw.shape != reference.shape or not np.all(np.isfinite(raw)):
        return _persistent_hold_or_reject(
            plan_id=plan_id,
            behavior_plan_id=behavior_plan_id,
            behavior_start_offset=behavior_start_offset,
            reference=reference,
            boundary_anchor=anchor,
            carry=carry,
            previous=previous,
            raw_residual=np.zeros_like(reference),
            direction=np.zeros(
                7
                if config.gripper_residual_mode
                == GRIPPER_RESIDUAL_CLOSE_ASSIST
                else 6,
                dtype=np.float64,
            ),
            fit_error=float("inf"),
            raw_gripper_max=float("inf"),
            reason="invalid_actor_chunk",
            config=config,
        )

    raw_residual = raw - reference
    joint_raw_residual = raw_residual[:, :6]
    denominator = float(np.dot(RANK1_BUMP_WINDOW, RANK1_BUMP_WINDOW))
    joint_direction = (
        np.sum(RANK1_BUMP_WINDOW[:, None] * joint_raw_residual, axis=0) / denominator
    )
    rank1_fit = RANK1_BUMP_WINDOW[:, None] * joint_direction[None, :]
    fit_error = float(np.max(np.abs(joint_raw_residual - rank1_fit)))
    raw_gripper_max = float(np.max(np.abs(raw_residual[:, 6])))
    gripper_direction = float(
        np.dot(RANK1_BUMP_WINDOW, raw_residual[:, 6]) / denominator
    )
    gripper_fit_error = float(
        np.max(
            np.abs(
                raw_residual[:, 6]
                - RANK1_BUMP_WINDOW * gripper_direction
            )
        )
    )
    direction = (
        np.concatenate(
            [joint_direction, np.asarray([gripper_direction], dtype=np.float64)]
        )
        if config.gripper_residual_mode == GRIPPER_RESIDUAL_CLOSE_ASSIST
        else joint_direction
    )
    if fit_error > config.rank1_fit_tolerance_rad + 1e-12:
        return _persistent_hold_or_reject(
            plan_id=plan_id,
            behavior_plan_id=behavior_plan_id,
            behavior_start_offset=behavior_start_offset,
            reference=reference,
            boundary_anchor=anchor,
            carry=carry,
            previous=previous,
            raw_residual=raw_residual,
            direction=direction,
            fit_error=fit_error,
            raw_gripper_max=raw_gripper_max,
            reason="rank1_contract_violation",
            config=config,
        )
    if (
        config.gripper_residual_mode == GRIPPER_RESIDUAL_FROZEN
        and raw_gripper_max > config.gripper_residual_tolerance + 1e-12
    ):
        return _persistent_hold_or_reject(
            plan_id=plan_id,
            behavior_plan_id=behavior_plan_id,
            behavior_start_offset=behavior_start_offset,
            reference=reference,
            boundary_anchor=anchor,
            carry=carry,
            previous=previous,
            raw_residual=raw_residual,
            direction=direction,
            fit_error=fit_error,
            raw_gripper_max=raw_gripper_max,
            reason="gripper_residual_not_frozen",
            config=config,
        )
    if config.gripper_residual_mode == GRIPPER_RESIDUAL_CLOSE_ASSIST:
        if gripper_fit_error > config.gripper_rank1_fit_tolerance_m + 1e-12:
            return _persistent_hold_or_reject(
                plan_id=plan_id,
                behavior_plan_id=behavior_plan_id,
                behavior_start_offset=behavior_start_offset,
                reference=reference,
                boundary_anchor=anchor,
                carry=carry,
                previous=previous,
                raw_residual=raw_residual,
                direction=direction,
                fit_error=fit_error,
                raw_gripper_max=raw_gripper_max,
                reason="gripper_rank1_contract_violation",
                config=config,
            )
        if gripper_direction > config.gripper_residual_tolerance:
            return _persistent_hold_or_reject(
                plan_id=plan_id,
                behavior_plan_id=behavior_plan_id,
                behavior_start_offset=behavior_start_offset,
                reference=reference,
                boundary_anchor=anchor,
                carry=carry,
                previous=previous,
                raw_residual=raw_residual,
                direction=direction,
                fit_error=fit_error,
                raw_gripper_max=raw_gripper_max,
                reason="gripper_open_residual_forbidden",
                config=config,
            )
        if (
            gripper_direction
            < -config.gripper_residual_max_close_m
            - config.gripper_residual_tolerance
        ):
            return _persistent_hold_or_reject(
                plan_id=plan_id,
                behavior_plan_id=behavior_plan_id,
                behavior_start_offset=behavior_start_offset,
                reference=reference,
                boundary_anchor=anchor,
                carry=carry,
                previous=previous,
                raw_residual=raw_residual,
                direction=direction,
                fit_error=fit_error,
                raw_gripper_max=raw_gripper_max,
                reason="gripper_close_residual_exceeds_limit",
                config=config,
            )

    selected_scale: float | None = None
    selected_residual: np.ndarray | None = None
    selected_metrics: dict[str, Any] | None = None
    for scale in np.linspace(0.0, 1.0, config.projection_scale_steps, dtype=np.float64)[::-1]:
        target = carry[:6] + float(scale) * (joint_direction - carry[:6])
        residual = np.zeros_like(reference)
        residual[:, :6] = (
            carry[None, :6]
            + _PERSISTENT_C10_BLEND_WINDOW[:, None] * (target - carry[:6])[None, :]
        )
        if config.gripper_residual_mode == GRIPPER_RESIDUAL_CLOSE_ASSIST:
            # Joint projection is independent, but the shared metric function
            # still needs a physically continuous provisional gripper path.
            residual[:, 6] = carry[6]
        metrics = _persistent_constraint_metrics(
            reference=reference,
            safe_residual=residual,
            boundary_anchor=anchor,
            carry_in=carry,
            previous_carry=previous,
            config=config,
        )
        if metrics["valid"]:
            selected_scale = float(scale)
            selected_residual = residual
            selected_metrics = metrics
            break
    if selected_scale is None or selected_residual is None or selected_metrics is None:
        return _persistent_rejected_plan(
            plan_id=plan_id,
            behavior_plan_id=behavior_plan_id,
            behavior_start_offset=behavior_start_offset,
            reference=reference,
            raw_residual=raw_residual,
            direction=direction,
            carry=carry,
            previous=previous,
            boundary_anchor=anchor,
            fit_error=fit_error,
            raw_gripper_max=raw_gripper_max,
            reason="persistent_carry_not_safe_for_behavior_ref",
            config=config,
        )
    gripper_result = _persistent_gripper_residual(
        reference=reference,
        base_boundary_gripper=float(anchor[6] - carry[6]),
        direction_gripper=gripper_direction,
        carry_in=carry,
        previous_carry=previous,
        config=config,
        hold_only=False,
    )
    if gripper_result is None:
        return _persistent_rejected_plan(
            plan_id=plan_id,
            behavior_plan_id=behavior_plan_id,
            behavior_start_offset=behavior_start_offset,
            reference=reference,
            raw_residual=raw_residual,
            direction=direction,
            carry=carry,
            previous=previous,
            boundary_anchor=anchor,
            fit_error=fit_error,
            raw_gripper_max=raw_gripper_max,
            reason="persistent_gripper_carry_not_safe_for_behavior_ref",
            config=config,
        )
    gripper_residual, gripper_metrics = gripper_result
    selected_residual[:, 6] = gripper_residual
    selected_metrics = _persistent_constraint_metrics(
        reference=reference,
        safe_residual=selected_residual,
        boundary_anchor=anchor,
        carry_in=carry,
        previous_carry=previous,
        config=config,
    )
    selected_metrics.update(gripper_metrics)
    update_accepted = selected_scale + 1e-12 >= config.min_projection_scale
    reason = None if update_accepted else "target_update_projection_below_minimum"
    return _make_persistent_plan(
        plan_id=plan_id,
        behavior_plan_id=behavior_plan_id,
        behavior_start_offset=behavior_start_offset,
        reference=reference,
        raw_residual=raw_residual,
        direction=direction,
        carry=carry,
        previous=previous,
        boundary_anchor=anchor,
        residual=selected_residual,
        projection_scale=selected_scale,
        metrics=selected_metrics,
        fit_error=fit_error,
        raw_gripper_max=raw_gripper_max,
        target_update_accepted=update_accepted,
        rejection_reason=reason,
        hold_only=not update_accepted,
        config=config,
    )


def _persistent_gripper_residual(
    *,
    reference: np.ndarray,
    base_boundary_gripper: float,
    direction_gripper: float,
    carry_in: np.ndarray,
    previous_carry: np.ndarray,
    config: ActorResidualGovernorConfig,
    hold_only: bool,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Plan one independent close-only gripper knot in physical metres."""

    if config.gripper_residual_mode == GRIPPER_RESIDUAL_FROZEN:
        return np.zeros(config.chunk_length, dtype=np.float64), {
            "gripper_residual_max": 0.0,
            "gripper_d1_max": 0.0,
            "gripper_d2_max": 0.0,
            "gripper_boundary_jump_max": 0.0,
            "gripper_release_intent": False,
        }
    reference_gripper = np.asarray(reference[:, 6], dtype=np.float64)
    carry = float(carry_in[6])
    previous = float(previous_carry[6])
    opening_delta = float(reference_gripper[-1] - base_boundary_gripper)
    release_intent = bool(
        np.max(reference_gripper)
        >= config.gripper_release_reference_m - _NUMERIC_TOLERANCE
        or opening_delta
        >= config.gripper_release_delta_m - _NUMERIC_TOLERANCE
    )
    if release_intent:
        desired = 0.0
    elif hold_only:
        desired = carry
    else:
        desired = float(
            np.clip(
                direction_gripper,
                -config.gripper_residual_max_close_m,
                0.0,
            )
        )
    for scale in np.linspace(
        0.0,
        1.0,
        config.projection_scale_steps,
        dtype=np.float64,
    )[::-1]:
        target = carry + float(scale) * (desired - carry)
        residual = carry + _PERSISTENT_C10_BLEND_WINDOW * (target - carry)
        history = np.concatenate(
            [
                np.asarray([previous, carry], dtype=np.float64),
                residual,
                residual[-1:],
            ]
        )
        d1 = np.diff(history)
        d2 = np.diff(d1)
        command = reference_gripper + residual
        boundary_jump = abs(float(residual[0] - carry))
        metrics = {
            "gripper_residual_max": float(np.max(np.abs(residual))),
            "gripper_d1_max": float(np.max(np.abs(d1))),
            "gripper_d2_max": float(np.max(np.abs(d2))),
            "gripper_boundary_jump_max": boundary_jump,
            "gripper_release_intent": release_intent,
        }
        valid = bool(
            np.max(residual) <= config.gripper_residual_tolerance
            and metrics["gripper_residual_max"]
            <= config.gripper_residual_max_close_m + _NUMERIC_TOLERANCE
            and metrics["gripper_d1_max"]
            <= config.gripper_residual_d1_max_m + _NUMERIC_TOLERANCE
            and metrics["gripper_d2_max"]
            <= config.gripper_residual_d2_max_m + _NUMERIC_TOLERANCE
            and boundary_jump
            <= config.gripper_max_boundary_jump_m + _NUMERIC_TOLERANCE
            and np.min(command)
            >= config.gripper_command_min_m - _NUMERIC_TOLERANCE
            and np.max(command)
            <= config.gripper_command_max_m + _NUMERIC_TOLERANCE
        )
        if valid:
            return residual, metrics
    return None


def govern_persistent_hold_plan(
    *,
    plan_id: str,
    behavior_plan_id: str,
    behavior_start_offset: int,
    behavior_ref: np.ndarray,
    boundary_anchor: np.ndarray,
    carry_in: np.ndarray,
    previous_carry: np.ndarray,
    config: ActorResidualGovernorConfig = ActorResidualGovernorConfig(),
) -> PersistentGovernedActorPlan:
    """Apply the committed carry to a C10 reference without a fresh Actor."""

    config = config.validate()
    reference = _require_reference(behavior_ref, config=config)
    anchor = _require_action_vector(boundary_anchor, config=config, label="boundary_anchor")
    carry = _require_action_vector(carry_in, config=config, label="carry_in")
    previous = _require_action_vector(
        previous_carry,
        config=config,
        label="previous_carry",
    )
    residual = np.zeros_like(reference)
    residual[:, :6] = carry[None, :6]
    gripper_result = _persistent_gripper_residual(
        reference=reference,
        base_boundary_gripper=float(anchor[6] - carry[6]),
        direction_gripper=float(carry[6]),
        carry_in=carry,
        previous_carry=previous,
        config=config,
        hold_only=True,
    )
    if gripper_result is None:
        return _persistent_rejected_plan(
            plan_id=plan_id,
            behavior_plan_id=behavior_plan_id,
            behavior_start_offset=behavior_start_offset,
            reference=reference,
            raw_residual=np.zeros_like(reference),
            direction=(
                carry.copy()
                if config.gripper_residual_mode
                == GRIPPER_RESIDUAL_CLOSE_ASSIST
                else carry[:6]
            ),
            carry=carry,
            previous=previous,
            boundary_anchor=anchor,
            fit_error=0.0,
            raw_gripper_max=0.0,
            reason="persistent_gripper_hold_not_safe_for_behavior_ref",
            config=config,
        )
    gripper_residual, gripper_metrics = gripper_result
    residual[:, 6] = gripper_residual
    metrics = _persistent_constraint_metrics(
        reference=reference,
        safe_residual=residual,
        boundary_anchor=anchor,
        carry_in=carry,
        previous_carry=previous,
        config=config,
    )
    metrics.update(gripper_metrics)
    if not metrics["valid"]:
        return _persistent_rejected_plan(
            plan_id=plan_id,
            behavior_plan_id=behavior_plan_id,
            behavior_start_offset=behavior_start_offset,
            reference=reference,
            raw_residual=np.zeros_like(reference),
            direction=carry[:6],
            carry=carry,
            previous=previous,
            boundary_anchor=anchor,
            fit_error=0.0,
            raw_gripper_max=0.0,
            reason="persistent_hold_not_safe_for_behavior_ref",
            config=config,
        )
    return _make_persistent_plan(
        plan_id=plan_id,
        behavior_plan_id=behavior_plan_id,
        behavior_start_offset=behavior_start_offset,
        reference=reference,
        raw_residual=np.zeros_like(reference),
        direction=(
            carry.copy()
            if config.gripper_residual_mode == GRIPPER_RESIDUAL_CLOSE_ASSIST
            else carry[:6]
        ),
        carry=carry,
        previous=previous,
        boundary_anchor=anchor,
        residual=residual,
        projection_scale=0.0,
        metrics=metrics,
        fit_error=0.0,
        raw_gripper_max=0.0,
        target_update_accepted=False,
        rejection_reason=None,
        hold_only=True,
        config=config,
    )


def _persistent_hold_or_reject(
    *,
    plan_id: str,
    behavior_plan_id: str,
    behavior_start_offset: int,
    reference: np.ndarray,
    boundary_anchor: np.ndarray,
    carry: np.ndarray,
    previous: np.ndarray,
    raw_residual: np.ndarray,
    direction: np.ndarray,
    fit_error: float,
    raw_gripper_max: float,
    reason: str,
    config: ActorResidualGovernorConfig,
) -> PersistentGovernedActorPlan:
    residual = np.zeros_like(reference)
    residual[:, :6] = carry[None, :6]
    gripper_result = _persistent_gripper_residual(
        reference=reference,
        base_boundary_gripper=float(boundary_anchor[6] - carry[6]),
        direction_gripper=float(carry[6]),
        carry_in=carry,
        previous_carry=previous,
        config=config,
        hold_only=True,
    )
    if gripper_result is None:
        return _persistent_rejected_plan(
            plan_id=plan_id,
            behavior_plan_id=behavior_plan_id,
            behavior_start_offset=behavior_start_offset,
            reference=reference,
            raw_residual=raw_residual,
            direction=direction,
            carry=carry,
            previous=previous,
            boundary_anchor=boundary_anchor,
            fit_error=fit_error,
            raw_gripper_max=raw_gripper_max,
            reason=reason,
            config=config,
        )
    gripper_residual, gripper_metrics = gripper_result
    residual[:, 6] = gripper_residual
    metrics = _persistent_constraint_metrics(
        reference=reference,
        safe_residual=residual,
        boundary_anchor=boundary_anchor,
        carry_in=carry,
        previous_carry=previous,
        config=config,
    )
    metrics.update(gripper_metrics)
    if not metrics["valid"]:
        return _persistent_rejected_plan(
            plan_id=plan_id,
            behavior_plan_id=behavior_plan_id,
            behavior_start_offset=behavior_start_offset,
            reference=reference,
            raw_residual=raw_residual,
            direction=direction,
            carry=carry,
            previous=previous,
            boundary_anchor=boundary_anchor,
            fit_error=fit_error,
            raw_gripper_max=raw_gripper_max,
            reason=reason,
            config=config,
        )
    return _make_persistent_plan(
        plan_id=plan_id,
        behavior_plan_id=behavior_plan_id,
        behavior_start_offset=behavior_start_offset,
        reference=reference,
        raw_residual=raw_residual,
        direction=direction,
        carry=carry,
        previous=previous,
        boundary_anchor=boundary_anchor,
        residual=residual,
        projection_scale=0.0,
        metrics=metrics,
        fit_error=fit_error,
        raw_gripper_max=raw_gripper_max,
        target_update_accepted=False,
        rejection_reason=reason,
        hold_only=True,
        config=config,
    )


def _make_persistent_plan(
    *,
    plan_id: str,
    behavior_plan_id: str,
    behavior_start_offset: int,
    reference: np.ndarray,
    raw_residual: np.ndarray,
    direction: np.ndarray,
    carry: np.ndarray,
    previous: np.ndarray,
    boundary_anchor: np.ndarray,
    residual: np.ndarray,
    projection_scale: float,
    metrics: dict[str, Any],
    fit_error: float,
    raw_gripper_max: float,
    target_update_accepted: bool,
    rejection_reason: str | None,
    hold_only: bool,
    config: ActorResidualGovernorConfig,
) -> PersistentGovernedActorPlan:
    safe = reference + residual
    if config.gripper_residual_mode == GRIPPER_RESIDUAL_CLOSE_ASSIST:
        if np.any(safe[:, 6] < config.gripper_command_min_m - _NUMERIC_TOLERANCE):
            raise ValueError("governed gripper command is below its physical range")
        if np.any(safe[:, 6] > config.gripper_command_max_m + _NUMERIC_TOLERANCE):
            raise ValueError("governed gripper command is above its physical range")
    else:
        safe[:, 6] = reference[:, 6]
    return PersistentGovernedActorPlan(
        plan_id=str(plan_id),
        behavior_plan_id=str(behavior_plan_id),
        behavior_start_offset=int(behavior_start_offset),
        approved=True,
        target_update_accepted=bool(target_update_accepted),
        rejection_reason=rejection_reason,
        safe_actions=safe.astype(np.float32),
        raw_residual=np.asarray(raw_residual, dtype=np.float32),
        safe_residual=np.asarray(residual, dtype=np.float32),
        rank1_direction=np.asarray(direction, dtype=np.float32),
        boundary_anchor=np.asarray(boundary_anchor, dtype=np.float32),
        carry_in=carry.astype(np.float32),
        previous_carry=np.asarray(previous, dtype=np.float32),
        carry_out=np.asarray(residual[-1], dtype=np.float32),
        projection_scale=float(projection_scale),
        raw_rank1_fit_error_max_rad=float(fit_error),
        raw_gripper_residual_max=float(raw_gripper_max),
        safe_residual_max_rad=float(metrics["residual_max"]),
        safe_residual_d1_max_rad=float(metrics["d1_max"]),
        safe_residual_d2_max_rad=float(metrics["d2_max"]),
        direction_min_cosine=metrics["min_cosine"],
        direction_violation_count=int(metrics["direction_violations"]),
        boundary_jump_max_rad=float(metrics["boundary_jump_max"]),
        boundary_jump_limit_rad=float(config.max_boundary_jump_rad),
        hold_only=bool(hold_only),
        safe_gripper_residual_max_m=float(
            metrics.get("gripper_residual_max", 0.0)
        ),
        safe_gripper_d1_max_m=float(metrics.get("gripper_d1_max", 0.0)),
        safe_gripper_d2_max_m=float(metrics.get("gripper_d2_max", 0.0)),
        gripper_boundary_jump_max_m=float(
            metrics.get("gripper_boundary_jump_max", 0.0)
        ),
        gripper_release_intent=bool(
            metrics.get("gripper_release_intent", False)
        ),
        gripper_residual_mode=config.gripper_residual_mode,
    )


def _persistent_rejected_plan(
    *,
    plan_id: str,
    behavior_plan_id: str,
    behavior_start_offset: int,
    reference: np.ndarray,
    raw_residual: np.ndarray,
    direction: np.ndarray,
    carry: np.ndarray,
    previous: np.ndarray,
    boundary_anchor: np.ndarray,
    fit_error: float,
    raw_gripper_max: float,
    reason: str,
    config: ActorResidualGovernorConfig,
) -> PersistentGovernedActorPlan:
    zeros = np.zeros_like(reference, dtype=np.float32)
    return PersistentGovernedActorPlan(
        plan_id=str(plan_id),
        behavior_plan_id=str(behavior_plan_id),
        behavior_start_offset=int(behavior_start_offset),
        approved=False,
        target_update_accepted=False,
        rejection_reason=reason,
        safe_actions=reference.astype(np.float32),
        raw_residual=np.asarray(raw_residual, dtype=np.float32),
        safe_residual=zeros,
        rank1_direction=np.asarray(direction, dtype=np.float32),
        boundary_anchor=np.asarray(boundary_anchor, dtype=np.float32),
        carry_in=carry.astype(np.float32),
        previous_carry=np.asarray(previous, dtype=np.float32),
        carry_out=np.zeros(config.action_dim, dtype=np.float32),
        projection_scale=0.0,
        raw_rank1_fit_error_max_rad=float(fit_error),
        raw_gripper_residual_max=float(raw_gripper_max),
        safe_residual_max_rad=0.0,
        safe_residual_d1_max_rad=0.0,
        safe_residual_d2_max_rad=0.0,
        direction_min_cosine=None,
        direction_violation_count=0,
        boundary_jump_max_rad=float("inf"),
        boundary_jump_limit_rad=float(config.max_boundary_jump_rad),
        hold_only=False,
        gripper_residual_mode=config.gripper_residual_mode,
    )


def _persistent_constraint_metrics(
    *,
    reference: np.ndarray,
    safe_residual: np.ndarray,
    boundary_anchor: np.ndarray,
    carry_in: np.ndarray,
    previous_carry: np.ndarray,
    config: ActorResidualGovernorConfig,
) -> dict[str, Any]:
    joint_residual = np.asarray(safe_residual[:, :6], dtype=np.float64)
    history_and_terminal = np.concatenate(
        [
            previous_carry[None, :6],
            carry_in[None, :6],
            joint_residual,
            joint_residual[-1:, :],
        ],
        axis=0,
    )
    residual_d1 = np.diff(history_and_terminal, axis=0)
    residual_d2 = np.diff(residual_d1, axis=0)
    residual_max = float(
        np.max(
            np.abs(
                np.concatenate(
                    [carry_in[None, :6], joint_residual],
                    axis=0,
                )
            )
        )
    )
    d1_max = float(np.max(np.abs(residual_d1)))
    d2_max = float(np.max(np.abs(residual_d2)))

    safe = reference.copy()
    safe[:, :6] += joint_residual
    # ``boundary_anchor`` is the last actually executed target and therefore
    # already contains ``carry_in``.  Compare reference motion against the
    # corresponding base-policy anchor; using the executed anchor for both
    # would falsely report a direction-cone violation whenever a non-zero
    # persistent carry crosses a C10/H50 boundary.
    reference_boundary_anchor = boundary_anchor[:6] - carry_in[:6]
    ref_steps = np.diff(
        np.concatenate([reference_boundary_anchor[None, :], reference[:, :6]], axis=0),
        axis=0,
    )
    actor_steps = np.diff(
        np.concatenate([boundary_anchor[None, :6], safe[:, :6]], axis=0),
        axis=0,
    )
    boundary_jump_max = float(np.max(np.abs(actor_steps[0])))
    cone_cosine = math.cos(math.radians(config.actor_direction_cone_deg))
    direction_violations = 0
    cosines: list[float] = []
    for ref_step, actor_step in zip(ref_steps, actor_steps):
        ref_norm = float(np.linalg.norm(ref_step))
        actor_norm = float(np.linalg.norm(actor_step))
        if ref_norm >= config.direction_static_threshold_rad:
            dot = float(np.dot(ref_step, actor_step))
            cosine = -1.0 if actor_norm <= 1e-12 else dot / (ref_norm * actor_norm)
            cosines.append(cosine)
            if dot < -_NUMERIC_TOLERANCE or cosine + _NUMERIC_TOLERANCE < cone_cosine:
                direction_violations += 1
        elif actor_norm > config.direction_static_threshold_rad + _NUMERIC_TOLERANCE:
            direction_violations += 1
    valid = bool(
        residual_max <= config.actor_residual_max_rad + _NUMERIC_TOLERANCE
        and d1_max <= config.actor_residual_d1_max_rad + _NUMERIC_TOLERANCE
        and d2_max <= config.actor_residual_d2_max_rad + _NUMERIC_TOLERANCE
        and boundary_jump_max <= config.max_boundary_jump_rad + _NUMERIC_TOLERANCE
        and direction_violations == 0
    )
    gripper = np.asarray(safe_residual[:, 6], dtype=np.float64)
    gripper_history = np.concatenate(
        [
            np.asarray(
                [previous_carry[6], carry_in[6]],
                dtype=np.float64,
            ),
            gripper,
            gripper[-1:],
        ]
    )
    gripper_d1 = np.diff(gripper_history)
    gripper_d2 = np.diff(gripper_d1)
    gripper_residual_max = float(np.max(np.abs(gripper)))
    gripper_d1_max = float(np.max(np.abs(gripper_d1)))
    gripper_d2_max = float(np.max(np.abs(gripper_d2)))
    gripper_boundary_jump = abs(float(gripper[0] - carry_in[6]))
    gripper_command = reference[:, 6] + gripper
    if config.gripper_residual_mode == GRIPPER_RESIDUAL_FROZEN:
        gripper_valid = bool(
            gripper_residual_max
            <= config.gripper_residual_tolerance + _NUMERIC_TOLERANCE
        )
    else:
        gripper_valid = bool(
            np.max(gripper) <= config.gripper_residual_tolerance
            and gripper_residual_max
            <= config.gripper_residual_max_close_m + _NUMERIC_TOLERANCE
            and gripper_d1_max
            <= config.gripper_residual_d1_max_m + _NUMERIC_TOLERANCE
            and gripper_d2_max
            <= config.gripper_residual_d2_max_m + _NUMERIC_TOLERANCE
            and gripper_boundary_jump
            <= config.gripper_max_boundary_jump_m + _NUMERIC_TOLERANCE
            and np.min(gripper_command)
            >= config.gripper_command_min_m - _NUMERIC_TOLERANCE
            and np.max(gripper_command)
            <= config.gripper_command_max_m + _NUMERIC_TOLERANCE
        )
    valid = bool(valid and gripper_valid)
    return {
        "valid": valid,
        "residual_max": residual_max,
        "d1_max": d1_max,
        "d2_max": d2_max,
        "min_cosine": None if not cosines else float(min(cosines)),
        "direction_violations": direction_violations,
        "boundary_jump_max": boundary_jump_max,
        "gripper_residual_max": gripper_residual_max,
        "gripper_d1_max": gripper_d1_max,
        "gripper_d2_max": gripper_d2_max,
        "gripper_boundary_jump_max": gripper_boundary_jump,
    }


def _require_action_vector(
    value: np.ndarray,
    *,
    config: ActorResidualGovernorConfig,
    label: str,
) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (config.action_dim,) or not np.all(np.isfinite(vector)):
        raise ValueError(
            f"{label} must be finite with shape ({config.action_dim},), got {vector.shape}"
        )
    return vector
