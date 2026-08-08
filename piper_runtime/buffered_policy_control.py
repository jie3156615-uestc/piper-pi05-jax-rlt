"""Buffered asynchronous policy execution for smooth Piper control."""

from __future__ import annotations

import dataclasses
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from piper_runtime.hardware_control import HardwareSafetyConfig
from piper_runtime.hardware_control import StatefulSafetyFilter
from piper_runtime.observation import build_observation


MIN_BUFFER_STEPS = 10
PREFETCH_THRESHOLD = 15
TARGET_BUFFER_STEPS = 30
MAX_STALE_MS = 150
BLEND_STEPS = 5
MODEL_HZ = 30.0
PUBLISH_HZ = 50.0
MAX_SERVO_JOINT_STEP = math.radians(3.0) * MODEL_HZ / PUBLISH_HZ
MAX_SERVO_GRIPPER_STEP = 0.02 * MODEL_HZ / PUBLISH_HZ
BLEND_GUARD_MODEL_STEPS = int(math.ceil(BLEND_STEPS * MODEL_HZ / PUBLISH_HZ))
GRIPPER_INTENT_DELTA_M = 0.008
GRIPPER_REVERSAL_CONFIRMATIONS = 2
# Absolute time alignment already preserves the old plan's long-horizon
# intent.  A moderate recency weight lets new visual evidence correct it while
# making an expiring old contribution too light to cause a boundary jump.
TEMPORAL_ENSEMBLE_DECAY = 0.75
# A strict short-horizon plan contains 10 model actions (333 ms at 30 Hz),
# resampled to 17 Piper servo targets.  The measured hot p99 policy latency is
# about 111 ms, so eight remaining servo points leave about 160 ms for the
# whole pipeline while making the next observation fresher than the former
# ten-point trigger.  This remains an internal invariant, not a CLI surface.
STRICT_PREFETCH_SERVO_STEPS = 8
STRICT_HANDOFF_PREPARE_SERVO_STEPS = 2


@dataclasses.dataclass(frozen=True)
class BufferedAction:
    target: np.ndarray
    chunk_id: int
    chunk_step: int

    def __post_init__(self) -> None:
        target = np.asarray(self.target, dtype=np.float64)
        if target.shape != (7,):
            raise ValueError("buffered action target must have shape (7,)")
        object.__setattr__(self, "target", target.copy())

    @property
    def joint_targets(self) -> np.ndarray:
        return self.target[:6]

    @property
    def gripper_target(self) -> float:
        return float(self.target[6])


@dataclasses.dataclass(frozen=True)
class H50HandoffPlan:
    """One prefetched absolute H50 plan time-aligned to its actual handoff."""

    actions: np.ndarray
    joint_rebase_offset: np.ndarray
    bridge_steps: int
    action_start_index: int
    observation_age_s: float
    raw_boundary_jump_rad: float
    bridge_excursion_rad: float
    bridge_target_correction_rad: float
    bridge_target_correction_clipped: bool


class H50HandoffRejected(RuntimeError):
    """The prefetched plan is too stale or inconsistent for physical use."""


def prepare_h50_handoff_plan(
    actions: np.ndarray,
    *,
    observation_state: np.ndarray,
    handoff_target: np.ndarray,
    previous_target: np.ndarray,
    observation_age_s: float,
    bridge_steps: int = BLEND_STEPS,
    model_hz: float = MODEL_HZ,
    max_bridge_excursion_rad: float = 0.12,
    model_smoothing_tau_s: float = 0.05,
    max_bridge_target_correction_rad: float = 0.12,
) -> H50HandoffPlan:
    """Time-align a prefetched absolute plan without replaying old motion.

    Drop targets whose timestamps elapsed while the previous H50 continued.
    Blend only the first few surviving joint targets from a constant-velocity
    continuation into the original trajectory, then return exactly to the
    policy's absolute targets.  The continuation is expressed in pre-low-pass
    target space so the first published command retains the preceding physical
    velocity instead of creating a stop/restart cusp.  Never translate the
    full suffix: AbsoluteActions has already anchored it to the prefetch
    observation, so a global offset would replay completed motion and
    accumulate endpoint error.

    Observation and gripper motion after prefetch are expected and are not
    rejection conditions.  The gripper remains the policy's unmodified
    absolute trajectory.  Only an implausibly large local bridge excursion
    rejects the standby plan.
    """

    values = np.asarray(actions, dtype=np.float64)
    observation = np.asarray(observation_state, dtype=np.float64)
    handoff = np.asarray(handoff_target, dtype=np.float64)
    previous = np.asarray(previous_target, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or len(values) < 2:
        raise ValueError("H50 handoff actions must have shape (N>=2, 7)")
    for label, vector in (
        ("observation_state", observation),
        ("handoff_target", handoff),
        ("previous_target", previous),
    ):
        if vector.shape != (7,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{label} must be finite with shape (7,)")
    if not np.all(np.isfinite(values)):
        raise ValueError("H50 handoff actions must be finite")
    age = float(observation_age_s)
    if not np.isfinite(age) or age < 0.0:
        raise ValueError("observation_age_s must be finite and non-negative")
    if model_hz <= 0.0:
        raise ValueError("model_hz must be positive")
    if not np.isfinite(model_smoothing_tau_s) or model_smoothing_tau_s < 0.0:
        raise ValueError("model_smoothing_tau_s must be finite and non-negative")
    action_start_index = int(math.floor(age * float(model_hz) + 1e-9))
    if action_start_index >= len(values) - 1:
        raise H50HandoffRejected(
            f"prefetched plan expired: stale_steps={action_start_index}, horizon={len(values)}"
        )
    action_start_index = max(0, action_start_index)
    aligned = values[action_start_index:].copy()
    raw_boundary_jump = float(np.max(np.abs(aligned[0, :6] - handoff[:6])))
    count = min(max(0, int(bridge_steps)), len(aligned))
    bridge_excursion = float(
        np.max(np.abs(aligned[: max(1, count), :6] - handoff[None, :6]))
    )
    if bridge_excursion > float(max_bridge_excursion_rad) + 1e-12:
        raise H50HandoffRejected(
            "boundary bridge excursion exceeds standby limit: "
            f"{bridge_excursion:.6f}>{float(max_bridge_excursion_rad):.6f} rad"
        )
    original_aligned = aligned.copy()
    if count >= 2:
        dt = 1.0 / float(model_hz)
        alpha = (
            1.0
            if model_smoothing_tau_s == 0.0
            else 1.0 - math.exp(-dt / float(model_smoothing_tau_s))
        )
        start_velocity = handoff[:6] - previous[:6]
        for index in range(count):
            u = float(index + 1) / float(count)
            u2 = u * u
            u3 = u2 * u
            weight = 3.0 * u2 - 2.0 * u3
            velocity_preserving_target = (
                handoff[:6]
                + float(index + 1) * start_velocity
                + ((1.0 / alpha) - 1.0) * start_velocity
            )
            aligned[index, :6] = (
                (1.0 - weight) * velocity_preserving_target
                + weight * original_aligned[index, :6]
            )
    correction = (
        aligned[: max(1, count), :6] - original_aligned[: max(1, count), :6]
    )
    clipped_correction = np.clip(
        correction,
        -float(max_bridge_target_correction_rad),
        float(max_bridge_target_correction_rad),
    )
    bridge_target_correction_clipped = not np.allclose(
        correction, clipped_correction, atol=1e-12, rtol=0.0
    )
    aligned[: max(1, count), :6] = (
        original_aligned[: max(1, count), :6] + clipped_correction
    )
    bridge_target_correction = float(np.max(np.abs(clipped_correction)))
    return H50HandoffPlan(
        actions=aligned,
        joint_rebase_offset=np.zeros(6, dtype=np.float64),
        bridge_steps=count,
        action_start_index=action_start_index,
        observation_age_s=age,
        raw_boundary_jump_rad=raw_boundary_jump,
        bridge_excursion_rad=bridge_excursion,
        bridge_target_correction_rad=bridge_target_correction,
        bridge_target_correction_clipped=bridge_target_correction_clipped,
    )


@dataclasses.dataclass(frozen=True)
class MergeResult:
    old_remaining: int
    new_points: int
    buffer_after: int
    blended_points: int
    used_last_published_fallback: bool


class ActionBuffer:
    """Thread-safe future target queue with atomic chunk replacement/blending."""

    def __init__(self, initial_target: np.ndarray) -> None:
        initial = np.asarray(initial_target, dtype=np.float64)
        if initial.shape != (7,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial target must be a finite shape-(7,) vector")
        self._lock = threading.Lock()
        self._actions: deque[BufferedAction] = deque()
        self._last_published_target = initial.copy()
        self._last_update_s: float | None = None

    def initialize(self, actions: np.ndarray, *, chunk_id: int, now_s: float) -> None:
        items = _make_buffered_actions(actions, chunk_id=chunk_id)
        if not items:
            raise ValueError("cannot initialize an empty action buffer")
        with self._lock:
            self._actions = deque(items)
            self._last_update_s = float(now_s)

    def replace_with_blend(
        self,
        actions: np.ndarray,
        *,
        chunk_id: int,
        now_s: float,
        blend_steps: int = BLEND_STEPS,
    ) -> MergeResult:
        new_items = _make_buffered_actions(actions, chunk_id=chunk_id)
        if not new_items:
            raise ValueError("cannot merge an empty action chunk")
        with self._lock:
            old_items = list(self._actions)
            old_remaining = len(old_items)
            blend_count = min(int(blend_steps), len(new_items))
            use_old = len(old_items) >= blend_count and blend_count > 0
            fallback = not use_old
            if blend_count > 0:
                if use_old:
                    old_targets = [old_items[index].target for index in range(blend_count)]
                else:
                    old_targets = [self._last_published_target for _ in range(blend_count)]
                blended_targets = _c1_hermite_blend(
                    old_targets=np.stack(old_targets),
                    new_targets=np.stack([item.target for item in new_items]),
                    blend_count=blend_count,
                )
                blended = [
                    BufferedAction(target, chunk_id, index)
                    for index, target in enumerate(blended_targets)
                ]
                merged = blended + new_items[blend_count:]
            else:  # pragma: no cover - BLEND_STEPS is a positive fixed constant.
                merged = new_items
            # Blending is itself a post-processing boundary. Enforce the
            # existing 3 degree / 0.02 m model-frame caps on the equivalent
            # 50 Hz grid so even the last-published fallback cannot jump.
            continuous_targets = _limit_servo_deltas(
                np.stack([item.target for item in merged]),
                initial_target=self._last_published_target,
            )
            merged = [
                BufferedAction(target, chunk_id, index)
                for index, target in enumerate(continuous_targets)
            ]
            self._actions = deque(merged)
            self._last_update_s = float(now_s)
            return MergeResult(
                old_remaining=old_remaining,
                new_points=len(new_items),
                buffer_after=len(merged),
                blended_points=blend_count,
                used_last_published_fallback=fallback,
            )

    def pop_next(self) -> BufferedAction | None:
        with self._lock:
            if not self._actions:
                return None
            return self._actions.popleft()

    def mark_published(self, target: np.ndarray) -> None:
        value = np.asarray(target, dtype=np.float64)
        if value.shape != (7,) or not np.all(np.isfinite(value)):
            raise ValueError("published target must be finite with shape (7,)")
        with self._lock:
            self._last_published_target = value.copy()

    def remaining(self) -> int:
        with self._lock:
            return len(self._actions)

    def skip_steps(self, count: int) -> int:
        """Discard elapsed servo slots without ever publishing them later."""

        skipped = 0
        with self._lock:
            for _ in range(max(0, int(count))):
                if not self._actions:
                    break
                self._actions.popleft()
                skipped += 1
        return skipped

    def prefetch_due(self) -> bool:
        return self.remaining() <= PREFETCH_THRESHOLD

    def last_published_target(self) -> np.ndarray:
        with self._lock:
            return self._last_published_target.copy()

    def last_update_s(self) -> float | None:
        with self._lock:
            return self._last_update_s


@dataclasses.dataclass(frozen=True)
class StrictInstallResult:
    active_remaining: int
    standby_points: int
    standby_chunk_id: int


class StrictChunkActionBuffer:
    """Two-slot queue for atomic, non-overlapping short-horizon execution.

    The active chunk is immutable once publication starts.  A policy worker
    may prepare exactly one standby chunk, but it cannot replace, blend, or
    phase-search the active chunk.  Promotion happens under the same lock as
    ``pop_next`` so a ready standby chunk follows the active chunk without an
    empty 50 Hz publication slot.
    """

    def __init__(
        self,
        initial_target: np.ndarray,
        *,
        safety_config: HardwareSafetyConfig | None = None,
    ) -> None:
        initial = np.asarray(initial_target, dtype=np.float64)
        if initial.shape != (7,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial target must be a finite shape-(7,) vector")
        self._lock = threading.Lock()
        self._active: deque[BufferedAction] = deque()
        self._standby: deque[BufferedAction] | None = None
        self._standby_anchor: np.ndarray | None = None
        self._last_published_target = initial.copy()
        self._previous_published_target = initial.copy()
        self._active_terminal_target = initial.copy()
        self._last_update_s: float | None = None
        self._initial_chunk_size = 0
        self._safety_config = safety_config or HardwareSafetyConfig()

    def initialize(self, actions: np.ndarray, *, chunk_id: int, now_s: float) -> None:
        items = _make_buffered_actions(actions, chunk_id=chunk_id)
        if not items:
            raise ValueError("cannot initialize an empty strict chunk")
        with self._lock:
            self._active = deque(items)
            self._standby = None
            self._standby_anchor = None
            self._initial_chunk_size = len(items)
            self._active_terminal_target = items[-1].target.copy()
            self._last_update_s = float(now_s)

    def install_standby(
        self,
        actions: np.ndarray,
        *,
        chunk_id: int,
        now_s: float,
        planned_handoff_target: np.ndarray,
    ) -> StrictInstallResult:
        items = _make_buffered_actions(actions, chunk_id=chunk_id)
        if not items:
            raise ValueError("cannot install an empty strict standby chunk")
        anchor = np.asarray(planned_handoff_target, dtype=np.float64)
        if anchor.shape != (7,) or not np.all(np.isfinite(anchor)):
            raise ValueError("strict standby anchor must be finite with shape (7,)")
        with self._lock:
            if self._standby is not None:
                raise RuntimeError("strict standby chunk is already occupied")
            self._standby = deque(items)
            self._standby_anchor = anchor.copy()
            self._last_update_s = float(now_s)
            return StrictInstallResult(
                active_remaining=len(self._active),
                standby_points=len(items),
                standby_chunk_id=int(chunk_id),
            )

    def pop_next(self) -> BufferedAction | None:
        with self._lock:
            self._promote_locked()
            if not self._active:
                return None
            return self._active.popleft()

    def mark_published(self, target: np.ndarray) -> None:
        value = np.asarray(target, dtype=np.float64)
        if value.shape != (7,) or not np.all(np.isfinite(value)):
            raise ValueError("published target must be finite with shape (7,)")
        with self._lock:
            self._previous_published_target = self._last_published_target.copy()
            self._last_published_target = value.copy()
            if not self._active:
                self._active_terminal_target = value.copy()

    def remaining(self) -> int:
        """Return active points only; standby never delays the next prefetch."""

        with self._lock:
            return len(self._active)

    def standby_ready(self) -> bool:
        with self._lock:
            return self._standby is not None

    def skip_steps(self, count: int) -> int:
        """Discard elapsed slots across an atomic boundary without bursts."""

        skipped = 0
        with self._lock:
            for _ in range(max(0, int(count))):
                self._promote_locked()
                if not self._active:
                    break
                self._active.popleft()
                skipped += 1
        return skipped

    def prefetch_due(self) -> bool:
        with self._lock:
            if self._standby is not None:
                return False
            if not self._active:
                return True
            threshold = min(
                STRICT_PREFETCH_SERVO_STEPS,
                max(1, self._initial_chunk_size - 1),
            )
            return len(self._active) <= threshold

    def last_published_target(self) -> np.ndarray:
        with self._lock:
            return self._last_published_target.copy()

    def handoff_target(self) -> np.ndarray:
        """Return the immutable active chunk's final target for standby prep."""

        with self._lock:
            return self._active_terminal_target.copy()

    def last_update_s(self) -> float | None:
        with self._lock:
            return self._last_update_s

    def _promote_locked(self) -> None:
        if not self._active and self._standby is not None:
            if self._standby_anchor is None:  # pragma: no cover - guarded by install_standby.
                raise RuntimeError("strict standby chunk has no handoff anchor")
            values = np.stack([item.target for item in self._standby])
            # Inference and standby preparation finish before the active chunk
            # boundary.  Re-anchor the six relative-joint dimensions at the
            # actual last command at promotion, eliminating the remaining
            # 1-2 servo-slot race.  Gripper is absolute and is not shifted.
            values[:, :6] += (
                self._last_published_target[:6] - self._standby_anchor[:6]
            )[None, :]
            bridge_count = min(BLEND_STEPS, len(values))
            if bridge_count >= 2:
                # Join only the arm joints with a C1 cubic Hermite segment.
                # The previous absolute-target smoothstep occupied 5/17 of
                # every H10 chunk and produced a repeated speed valley.  This
                # bridge reaches the new plan's original fifth point and its
                # local velocity instead of distorting the following path.
                start = self._last_published_target[:6].copy()
                start_velocity = (
                    self._last_published_target[:6] - self._previous_published_target[:6]
                )
                end_index = bridge_count - 1
                end = values[end_index, :6].copy()
                if bridge_count < len(values):
                    end_velocity = values[bridge_count, :6] - end
                else:
                    end_velocity = end - values[end_index - 1, :6]
                duration_steps = float(bridge_count)
                for index in range(bridge_count):
                    u = float(index + 1) / duration_steps
                    u2 = u * u
                    u3 = u2 * u
                    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
                    h10 = u3 - 2.0 * u2 + u
                    h01 = -2.0 * u3 + 3.0 * u2
                    h11 = u3 - u2
                    values[index, :6] = (
                        h00 * start
                        + h10 * duration_steps * start_velocity
                        + h01 * end
                        + h11 * duration_steps * end_velocity
                    )
            # Dimension 6 is an absolute gripper command.  It deliberately
            # bypasses the velocity bridge: real H10 audit data showed the old
            # bridge shifting close/open intent by as much as 22 mm.
            continuous = []
            previous = self._last_published_target.copy()
            for target in values:
                limited = _limit_one_servo_delta(
                    target,
                    previous,
                    config=self._safety_config,
                )
                continuous.append(limited)
                previous = limited
            chunk_id = self._standby[0].chunk_id
            self._active = deque(
                BufferedAction(target, chunk_id, index)
                for index, target in enumerate(continuous)
            )
            self._standby = None
            self._standby_anchor = None
            self._initial_chunk_size = len(self._active)
            self._active_terminal_target = self._active[-1].target.copy()


@dataclasses.dataclass(frozen=True)
class _PlanContribution:
    target: np.ndarray
    chunk_id: int
    activation_gain: float

    def __post_init__(self) -> None:
        target = np.asarray(self.target, dtype=np.float64)
        if target.shape != (7,) or not np.all(np.isfinite(target)):
            raise ValueError("temporal contribution target must be finite with shape (7,)")
        object.__setattr__(self, "target", target.copy())


@dataclasses.dataclass(frozen=True)
class TemporalInsertResult:
    observation_slot: int
    plan_ready_slot: int
    first_slot: int
    last_slot: int
    dropped_past_points: int
    overlap_slots: int
    buffer_horizon_steps: int


class TemporalActionBuffer:
    """Absolute 50 Hz timeline retaining overlapping full-horizon predictions.

    `execute_steps` controls how often a new observation is requested.  It does
    not truncate the policy's 50-step prediction.  A slot may therefore carry
    predictions from several policy calls; targets are combined only when that
    absolute slot is consumed.
    """

    def __init__(
        self,
        initial_target: np.ndarray,
        *,
        replan_interval_steps: int,
        max_ensemble_plans: int,
        safety_config: HardwareSafetyConfig | None = None,
    ) -> None:
        initial = np.asarray(initial_target, dtype=np.float64)
        if initial.shape != (7,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial target must be a finite shape-(7,) vector")
        if int(replan_interval_steps) < 1 or int(max_ensemble_plans) < 1:
            raise ValueError("temporal buffer intervals must be positive")
        self._lock = threading.Lock()
        self._slots: dict[int, list[_PlanContribution]] = {}
        self._current_slot = 0
        self._last_published_target = initial.copy()
        self._last_update_s: float | None = None
        self._replan_interval_steps = int(replan_interval_steps)
        self._next_replan_slot = int(replan_interval_steps)
        self._max_ensemble_plans = int(max_ensemble_plans)
        self._safety_config = safety_config or HardwareSafetyConfig()
        self._reserved_replan = False
        self._last_ensemble_contributors = 0

    def initialize(self, actions: np.ndarray, *, chunk_id: int, now_s: float) -> None:
        values = self._validated_chunk(actions)
        with self._lock:
            self._slots.clear()
            self._current_slot = 0
            for offset, target in enumerate(values):
                fade_out = _smoothstep_activation_gain(len(values) - 1 - offset, BLEND_STEPS)
                self._slots[offset] = [_PlanContribution(target, int(chunk_id), fade_out)]
            self._last_update_s = float(now_s)
            self._next_replan_slot = self._replan_interval_steps
            self._reserved_replan = False

    def insert_plan(
        self,
        actions: np.ndarray,
        *,
        chunk_id: int,
        observation_slot: int,
        now_s: float,
    ) -> TemporalInsertResult:
        """Add a complete prediction at its absolute observation-time slots."""

        values = self._validated_chunk(actions)
        observation_slot = int(observation_slot)
        with self._lock:
            ready_slot = self._current_slot
            dropped = max(0, ready_slot - observation_slot)
            first_offset = min(len(values), dropped)
            overlap_slots = 0
            first_slot = observation_slot + first_offset
            last_slot = first_slot - 1
            for offset in range(first_offset, len(values)):
                absolute_slot = observation_slot + offset
                if absolute_slot < self._current_slot:
                    continue
                activation_index = absolute_slot - first_slot
                fade_in = _smoothstep_activation_gain(activation_index, BLEND_STEPS)
                fade_out = _smoothstep_activation_gain(len(values) - 1 - offset, BLEND_STEPS)
                gain = fade_in * fade_out
                contributions = self._slots.setdefault(absolute_slot, [])
                if contributions:
                    overlap_slots += 1
                contributions.append(_PlanContribution(values[offset], int(chunk_id), gain))
                # Natural overlap is at most ceil(50 / execute_steps), but
                # bound memory if a future policy configuration changes.
                if len(contributions) > self._max_ensemble_plans:
                    del contributions[: len(contributions) - self._max_ensemble_plans]
                last_slot = absolute_slot
            self._last_update_s = float(now_s)
            self._reserved_replan = False
            horizon = self._remaining_locked()
            return TemporalInsertResult(
                observation_slot=observation_slot,
                plan_ready_slot=ready_slot,
                first_slot=first_slot,
                last_slot=last_slot,
                dropped_past_points=first_offset,
                overlap_slots=overlap_slots,
                buffer_horizon_steps=horizon,
            )

    def pop_next(self) -> BufferedAction | None:
        with self._lock:
            slot = self._current_slot
            self._current_slot += 1
            contributions = self._slots.pop(slot, None)
            self._drop_past_locked()
            if not contributions:
                self._last_ensemble_contributors = 0
                return None
            target = _ensemble_contributions(contributions)
            if len(contributions) > 1:
                tau_s = float(self._safety_config.model_smoothing_tau_s)
                alpha = 1.0 if tau_s == 0 else 1.0 - math.exp(-(1.0 / PUBLISH_HZ) / tau_s)
                target[:6] = self._last_published_target[:6] + alpha * (
                    target[:6] - self._last_published_target[:6]
                )
            target = _limit_one_servo_delta(
                target,
                self._last_published_target,
                config=self._safety_config,
            )
            self._last_ensemble_contributors = len(contributions)
            newest_chunk = max(contribution.chunk_id for contribution in contributions)
            return BufferedAction(target, newest_chunk, slot)

    def mark_published(self, target: np.ndarray) -> None:
        value = np.asarray(target, dtype=np.float64)
        if value.shape != (7,) or not np.all(np.isfinite(value)):
            raise ValueError("published target must be finite with shape (7,)")
        with self._lock:
            self._last_published_target = value.copy()

    def remaining(self) -> int:
        with self._lock:
            return self._remaining_locked()

    def current_slot(self) -> int:
        with self._lock:
            return self._current_slot

    def last_ensemble_contributors(self) -> int:
        with self._lock:
            return self._last_ensemble_contributors

    def last_published_target(self) -> np.ndarray:
        with self._lock:
            return self._last_published_target.copy()

    def last_update_s(self) -> float | None:
        with self._lock:
            return self._last_update_s

    def skip_steps(self, count: int) -> int:
        count = max(0, int(count))
        with self._lock:
            self._current_slot += count
            self._drop_past_locked()
        return count

    def prefetch_due(self) -> bool:
        with self._lock:
            return (
                not self._reserved_replan
                and (self._current_slot >= self._next_replan_slot or self._remaining_locked() <= PREFETCH_THRESHOLD)
            )

    def reserve_replan(self) -> int | None:
        """Atomically reserve one policy query; overdue intervals are skipped."""

        with self._lock:
            if self._reserved_replan:
                return None
            low_horizon = self._remaining_locked() <= PREFETCH_THRESHOLD
            if self._current_slot < self._next_replan_slot and not low_horizon:
                return None
            query_slot = self._current_slot
            while self._next_replan_slot <= self._current_slot:
                self._next_replan_slot += self._replan_interval_steps
            if low_horizon and self._next_replan_slot <= query_slot:
                self._next_replan_slot = query_slot + self._replan_interval_steps
            self._reserved_replan = True
            return query_slot

    def cancel_replan_reservation(self) -> None:
        with self._lock:
            self._reserved_replan = False

    def _remaining_locked(self) -> int:
        if not self._slots:
            return 0
        return max(self._slots) - self._current_slot + 1

    def _drop_past_locked(self) -> None:
        stale = [slot for slot in self._slots if slot < self._current_slot]
        for slot in stale:
            del self._slots[slot]

    @staticmethod
    def _validated_chunk(actions: np.ndarray) -> np.ndarray:
        values = np.asarray(actions, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 7 or len(values) < 1:
            raise ValueError("temporal action chunk must have shape (N, 7)")
        if not np.all(np.isfinite(values)):
            raise ValueError("temporal action chunk contains NaN or Inf")
        return values.copy()


def _smoothstep_activation_gain(index: int, blend_steps: int = BLEND_STEPS) -> float:
    if blend_steps <= 1:
        return 1.0
    u = float(np.clip(int(index) / float(blend_steps - 1), 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def _ensemble_contributions(contributions: list[_PlanContribution]) -> np.ndarray:
    ordered = sorted(contributions, key=lambda item: item.chunk_id)
    # An established full-horizon prediction carries the stable task intent.
    # Older predictions are stronger, but every plan fades out over its final
    # five slots so expiration cannot create a discontinuity.
    ages = np.arange(len(ordered), dtype=np.float64)
    weights = np.exp(-TEMPORAL_ENSEMBLE_DECAY * ages)
    weights *= np.asarray([item.activation_gain for item in ordered], dtype=np.float64)
    if float(np.sum(weights)) <= 1e-12:
        return ordered[0].target.copy()
    weights /= np.sum(weights)
    targets = np.stack([item.target for item in ordered])
    result = np.sum(targets * weights[:, None], axis=0)
    # Preserve the oldest still-valid time-aligned gripper event.  Averaging
    # stochastic open/close predictions caused repeated half-closing and
    # erased the late close event that makes execute_steps=50 successful.
    result[6] = float(ordered[int(np.argmax(weights))].target[6])
    return result


def _limit_one_servo_delta(
    target: np.ndarray,
    previous: np.ndarray,
    *,
    config: HardwareSafetyConfig | None = None,
) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    previous = np.asarray(previous, dtype=np.float64)
    desired_delta = target - previous
    desired_delta[:6] = np.clip(desired_delta[:6], -MAX_SERVO_JOINT_STEP, MAX_SERVO_JOINT_STEP)
    desired_delta[6] = float(np.clip(desired_delta[6], -MAX_SERVO_GRIPPER_STEP, MAX_SERVO_GRIPPER_STEP))
    limited = previous + desired_delta
    limits = config or HardwareSafetyConfig()
    limited[:6] = np.clip(limited[:6], limits.joint_min, limits.joint_max)
    limited[6] = float(np.clip(limited[6], limits.gripper_min, limits.gripper_max))
    return limited


def _make_buffered_actions(actions: np.ndarray, *, chunk_id: int) -> list[BufferedAction]:
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError("action chunk must have shape (N, 7)")
    if not np.all(np.isfinite(values)):
        raise ValueError("action chunk contains NaN or Inf")
    return [BufferedAction(target, int(chunk_id), index) for index, target in enumerate(values)]


def _limit_servo_deltas(actions: np.ndarray, *, initial_target: np.ndarray) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float64)
    previous = np.asarray(initial_target, dtype=np.float64).copy()
    output = []
    for target in values:
        limited = previous.copy()
        limited[:6] += np.clip(
            target[:6] - previous[:6],
            -MAX_SERVO_JOINT_STEP,
            MAX_SERVO_JOINT_STEP,
        )
        limited[6] += float(
            np.clip(
                target[6] - previous[6],
                -MAX_SERVO_GRIPPER_STEP,
                MAX_SERVO_GRIPPER_STEP,
            )
        )
        output.append(limited)
        previous = limited
    return np.asarray(output, dtype=np.float64)


def _c1_hermite_blend(
    *,
    old_targets: np.ndarray,
    new_targets: np.ndarray,
    blend_count: int,
) -> np.ndarray:
    """Join old and new targets with matching endpoint position and velocity."""

    old = np.asarray(old_targets, dtype=np.float64)
    new = np.asarray(new_targets, dtype=np.float64)
    count = int(blend_count)
    if count <= 0 or old.shape[0] < count or new.shape[0] < count:
        raise ValueError("invalid Hermite blend inputs")
    if count == 1:
        return new[:1].copy()

    p0 = old[0]
    p1 = new[count - 1]
    old_velocity = old[1] - old[0] if len(old) > 1 else np.zeros_like(p0)
    new_velocity = new[count] - new[count - 1] if len(new) > count else new[count - 1] - new[count - 2]
    duration = float(count - 1)
    output = []
    for index in range(count):
        t = index / duration
        t2 = t * t
        t3 = t2 * t
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2
        target = h00 * p0 + h10 * duration * old_velocity + h01 * p1 + h11 * duration * new_velocity
        output.append(target)
    return np.asarray(output, dtype=np.float64)


def low_pass_and_limit_actions(
    actions: np.ndarray,
    *,
    initial_target: np.ndarray,
    config: HardwareSafetyConfig,
    dt: float = 1.0 / MODEL_HZ,
) -> np.ndarray:
    """Apply the existing model-only low pass, limits and per-frame delta caps."""

    values = np.asarray(actions, dtype=np.float64)
    previous = np.asarray(initial_target, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError("model actions must have shape (N, 7)")
    if previous.shape != (7,):
        raise ValueError("initial target must have shape (7,)")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(previous)):
        raise ValueError("model action processing received NaN or Inf")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("model action dt must be positive")

    tau_s = float(config.model_smoothing_tau_s)
    if not math.isfinite(tau_s) or tau_s < 0:
        raise ValueError("model smoothing tau must be finite and non-negative")
    alpha = 1.0 if tau_s == 0 else 1.0 - math.exp(-dt / tau_s)
    output = []
    for raw_target in values:
        target = raw_target.copy()
        target[:6] = np.clip(target[:6], config.joint_min, config.joint_max)
        target[6] = float(np.clip(target[6], config.gripper_min, config.gripper_max))

        smoothed_joints = previous[:6] + alpha * (target[:6] - previous[:6])
        joint_delta = np.clip(
            smoothed_joints - previous[:6],
            -float(config.model_max_joint_step),
            float(config.model_max_joint_step),
        )
        gripper_delta = float(
            np.clip(
                target[6] - previous[6],
                -float(config.model_max_gripper_step),
                float(config.model_max_gripper_step),
            )
        )
        command = previous.copy()
        command[:6] = np.clip(previous[:6] + joint_delta, config.joint_min, config.joint_max)
        command[6] = float(np.clip(previous[6] + gripper_delta, config.gripper_min, config.gripper_max))
        output.append(command)
        previous = command
    return np.asarray(output, dtype=np.float64)


def resample_actions_causal(
    actions: np.ndarray,
    *,
    initial_target: np.ndarray,
    input_hz: float = MODEL_HZ,
    output_hz: float = PUBLISH_HZ,
) -> np.ndarray:
    """Causally linearly resample model waypoints onto the 50 Hz servo grid."""

    values = np.asarray(actions, dtype=np.float64)
    initial = np.asarray(initial_target, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or len(values) < 1:
        raise ValueError("resampling requires a non-empty shape-(N, 7) chunk")
    if initial.shape != (7,) or not np.all(np.isfinite(values)) or not np.all(np.isfinite(initial)):
        raise ValueError("resampling received invalid targets")
    if input_hz <= 0 or output_hz <= 0:
        raise ValueError("resampling frequencies must be positive")

    duration_s = len(values) / float(input_hz)
    output_count = max(1, int(math.ceil(duration_s * float(output_hz))))
    output = []
    for output_index in range(output_count):
        t_s = min((output_index + 1) / float(output_hz), duration_s)
        scaled = t_s * float(input_hz)
        right_index = min(max(int(math.ceil(scaled)) - 1, 0), len(values) - 1)
        left_target = initial if right_index == 0 else values[right_index - 1]
        segment_start_s = right_index / float(input_hz)
        weight = float(np.clip((t_s - segment_start_s) * float(input_hz), 0.0, 1.0))
        output.append((1.0 - weight) * left_target + weight * values[right_index])
    return np.asarray(output, dtype=np.float64)


@dataclasses.dataclass(frozen=True)
class PlannedChunk:
    chunk_id: int
    actions: np.ndarray
    inference_s: float
    observation_state: np.ndarray
    completion_state: np.ndarray
    action_start_index: int
    raw_horizon_joint_distances: dict[str, float]
    filtered_model_actions: np.ndarray
    planned_model_steps: int
    expected_action_start_index: int


@dataclasses.dataclass(frozen=True)
class StrictPlannedChunk:
    """The exact policy prefix captured from one observation."""

    chunk_id: int
    all_reference_actions: np.ndarray
    reference_actions: np.ndarray
    inference_s: float
    observation_state: np.ndarray
    completion_state: np.ndarray
    observation_time_s: float
    inference_done_time_s: float
    raw_horizon_joint_distances: dict[str, float]


@dataclasses.dataclass(frozen=True)
class StrictPreparedChunk:
    """A strict model prefix prepared for one future atomic handoff."""

    planned: StrictPlannedChunk
    actions: np.ndarray
    rebased_model_actions: np.ndarray
    filtered_model_actions: np.ndarray
    handoff_target: np.ndarray
    action_start_index: int
    alignment_age_s: float


@dataclasses.dataclass(frozen=True)
class TemporalPlannedChunk:
    chunk_id: int
    actions: np.ndarray
    inference_s: float
    observation_state: np.ndarray
    completion_state: np.ndarray
    observation_slot: int
    plan_ready_slot: int
    raw_horizon_joint_distances: dict[str, float]


class StrictPolicyChunkPlanner:
    """Infer H50, then commit one latency-aligned H10 decision.

    The output transform has already converted the six joint deltas back to
    absolute future joint targets.  During asynchronous execution the first
    few predicted targets are stale by the handoff time, so preparation picks
    the ten targets aligned to that future handoff.  Continuity correction is
    relative to the immediately preceding predicted target, not the original
    observation state; this avoids adding already executed motion twice.
    """

    def __init__(
        self,
        *,
        cameras: Any,
        policy: Any,
        feedback_reader: Any,
        prompt: str,
        execute_steps: int,
        safety_config: HardwareSafetyConfig,
        max_inference_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cameras = cameras
        self.policy = policy
        self.feedback_reader = feedback_reader
        self.prompt = str(prompt)
        self.execute_steps = int(execute_steps)
        self.safety_config = safety_config
        self.max_inference_s = float(max_inference_s)
        self.clock = clock

    def plan(self, *, chunk_id: int, warmup_frames: int) -> StrictPlannedChunk:
        images = self.cameras.read(timeout_ms=5000, warmup_frames=int(warmup_frames))
        snapshot = np.asarray(self.feedback_reader.read(), dtype=np.float64)
        if snapshot.shape != (7,) or not np.all(np.isfinite(snapshot)):
            raise RuntimeError("strict policy planner received invalid robot feedback")
        observation_time_s = self.clock()
        observation = build_observation(images, snapshot, prompt=self.prompt)
        response = self.policy.infer(observation)
        inference_done_time_s = self.clock()
        inference_s = inference_done_time_s - observation_time_s
        if inference_s > self.max_inference_s:
            raise RuntimeError(f"policy inference exceeded {self.max_inference_s:.1f} seconds")
        actions = _validated_policy_actions(response)
        reference = actions[: self.execute_steps].copy()
        if len(reference) != self.execute_steps:
            raise RuntimeError("policy response is shorter than strict execution horizon")
        completion_state = np.asarray(self.feedback_reader.read(), dtype=np.float64)
        if completion_state.shape != (7,) or not np.all(np.isfinite(completion_state)):
            raise RuntimeError("strict policy planner received invalid post-inference feedback")
        raw_horizon_joint_distances = {
            f"d{step}": float(np.linalg.norm(actions[step - 1, :6] - snapshot[:6]))
            for step in (10, 20, 30, 50)
        }
        return StrictPlannedChunk(
            chunk_id=int(chunk_id),
            all_reference_actions=actions.copy(),
            reference_actions=reference,
            inference_s=float(inference_s),
            observation_state=snapshot.copy(),
            completion_state=completion_state.copy(),
            observation_time_s=float(observation_time_s),
            inference_done_time_s=float(inference_done_time_s),
            raw_horizon_joint_distances=raw_horizon_joint_distances,
        )

    def prepare_for_handoff(
        self,
        planned: StrictPlannedChunk,
        *,
        handoff_target: np.ndarray,
        handoff_time_s: float | None = None,
    ) -> StrictPreparedChunk:
        handoff = np.asarray(handoff_target, dtype=np.float64)
        if handoff.shape != (7,) or not np.all(np.isfinite(handoff)):
            raise RuntimeError("strict policy handoff target is invalid")
        all_reference = np.asarray(planned.all_reference_actions, dtype=np.float64)
        if handoff_time_s is None:
            alignment_age_s = 0.0
            action_start_index = 0
        else:
            alignment_age_s = max(0.0, float(handoff_time_s) - planned.observation_time_s)
            # action[i] is the target at (i + 1) / MODEL_HZ.  floor(age*Hz)
            # drops only fully elapsed targets; ceil would skip one additional
            # still-future waypoint at every non-grid-aligned handoff.
            action_start_index = int(math.floor(alignment_age_s * MODEL_HZ + 1e-9))
        latest_valid_start = len(all_reference) - self.execute_steps
        if action_start_index > latest_valid_start:
            raise RuntimeError(
                "policy plan expired before an aligned H10 slice was available: "
                f"start={action_start_index}, latest={latest_valid_start}"
            )
        action_start_index = max(0, action_start_index)
        selected = all_reference[
            action_start_index : action_start_index + self.execute_steps
        ].copy()
        if len(selected) != self.execute_steps:
            raise RuntimeError("latency-aligned policy slice is shorter than H10")

        # AbsoluteActions has already added the observation state back to the
        # model output.  Anchor against the prediction immediately preceding
        # the selected H10 slice; anchoring every chunk against observation
        # state double-counted 0..k motion and caused large cumulative drift.
        if action_start_index == 0:
            predicted_anchor = planned.observation_state
        else:
            predicted_anchor = all_reference[action_start_index - 1]
        rebased = selected.copy()
        rebased[:, :6] += (handoff[:6] - predicted_anchor[:6])[None, :]
        filtered = low_pass_and_limit_actions(
            rebased,
            initial_target=handoff,
            config=self.safety_config,
            dt=1.0 / MODEL_HZ,
        )
        resampled = resample_actions_causal(
            filtered,
            initial_target=handoff,
            input_hz=MODEL_HZ,
            output_hz=PUBLISH_HZ,
        )
        return StrictPreparedChunk(
            planned=planned,
            actions=resampled,
            rebased_model_actions=rebased,
            filtered_model_actions=filtered,
            handoff_target=handoff.copy(),
            action_start_index=action_start_index,
            alignment_age_s=float(alignment_age_s),
        )


class PolicyChunkPlanner:
    """Capture the latest observation, infer once, and postprocess without publishing."""

    def __init__(
        self,
        *,
        cameras: Any,
        policy: Any,
        feedback_reader: Any,
        prompt: str,
        execute_steps: int,
        safety_config: HardwareSafetyConfig,
        max_inference_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cameras = cameras
        self.policy = policy
        self.feedback_reader = feedback_reader
        self.prompt = str(prompt)
        self.execute_steps = int(execute_steps)
        self.safety_config = safety_config
        self.max_inference_s = float(max_inference_s)
        self.clock = clock

    def plan(
        self,
        *,
        chunk_id: int,
        warmup_frames: int,
        runtime_alignment_delay_s: float | None = None,
    ) -> PlannedChunk:
        images = self.cameras.read(timeout_ms=5000, warmup_frames=int(warmup_frames))
        snapshot = np.asarray(self.feedback_reader.read(), dtype=np.float64)
        if snapshot.shape != (7,) or not np.all(np.isfinite(snapshot)):
            raise RuntimeError("policy planner received invalid robot feedback")
        observation = build_observation(images, snapshot, prompt=self.prompt)
        started_s = self.clock()
        response = self.policy.infer(observation)
        inference_s = self.clock() - started_s
        if inference_s > self.max_inference_s:
            raise RuntimeError(f"policy inference exceeded {self.max_inference_s:.1f} seconds")
        actions = _validated_policy_actions(response)
        raw_horizon_joint_distances = {
            f"d{step}": float(np.linalg.norm(actions[step - 1, :6] - snapshot[:6]))
            for step in (10, 20, 30, 50)
        }
        completion_state = np.asarray(self.feedback_reader.read(), dtype=np.float64)
        if completion_state.shape != (7,) or not np.all(np.isfinite(completion_state)):
            raise RuntimeError("policy planner received invalid post-inference feedback")

        # While inference is prefetched, the controller consumes the active
        # chunk until its five-point handoff. action[0] is stale by then. Skip
        # the larger of measured inference time and expected handoff delay,
        # then commit N still-valid actions. The five-point blend advances
        # through the selected chunk while its weight rises, so its duration
        # must not be counted a second time here. The initial synchronous plan
        # intentionally starts at zero.
        # Five 50 Hz blending points overlap about three 30 Hz model points.
        # Keep those guard samples internally so the non-overlapped horizon
        # still advances by execute_steps model frames.
        planned_model_steps = min(
            len(actions),
            self.execute_steps + (0 if self.execute_steps == len(actions) else BLEND_GUARD_MODEL_STEPS),
        )
        action_start_index = 0
        if runtime_alignment_delay_s is not None:
            elapsed_to_handoff_s = max(inference_s, float(runtime_alignment_delay_s))
            action_start_index = int(math.ceil(elapsed_to_handoff_s * MODEL_HZ))
            action_start_index = min(action_start_index, len(actions) - planned_model_steps)
        filtered_model_actions = low_pass_and_limit_actions(
            actions,
            initial_target=snapshot,
            config=self.safety_config,
            dt=1.0 / MODEL_HZ,
        )
        filtered = filtered_model_actions[action_start_index : action_start_index + planned_model_steps]
        resampled = resample_actions_causal(
            filtered,
            initial_target=completion_state,
            input_hz=MODEL_HZ,
            output_hz=PUBLISH_HZ,
        )
        return PlannedChunk(
            int(chunk_id),
            resampled,
            float(inference_s),
            snapshot.copy(),
            completion_state.copy(),
            action_start_index,
            raw_horizon_joint_distances,
            filtered_model_actions,
            planned_model_steps,
            action_start_index,
        )

    def align_to_handoff(self, planned: PlannedChunk, *, handoff_state: np.ndarray) -> PlannedChunk:
        """Choose a nearby, non-future phase using actual handoff feedback."""

        state = np.asarray(handoff_state, dtype=np.float64)
        if state.shape != (7,) or not np.all(np.isfinite(state)):
            raise RuntimeError("policy handoff received invalid robot feedback")
        max_start = len(planned.filtered_model_actions) - planned.planned_model_steps
        search_high = min(max_start, planned.expected_action_start_index + 4)
        candidates = planned.filtered_model_actions[: search_high + 1]
        joint_cost = np.linalg.norm(candidates[:, :6] - state[:6], axis=1)
        gripper_cost = 0.5 * np.abs(candidates[:, 6] - state[6])
        # A tiny phase penalty breaks near-ties toward the earlier action and
        # avoids jumping ahead during precise grasp/contact motion.
        phase_penalty = np.arange(len(candidates), dtype=np.float64) * 1e-3
        action_start_index = int(np.argmin(joint_cost + gripper_cost + phase_penalty))
        selected = planned.filtered_model_actions[
            action_start_index : action_start_index + planned.planned_model_steps
        ]
        resampled = resample_actions_causal(
            selected,
            initial_target=state,
            input_hz=MODEL_HZ,
            output_hz=PUBLISH_HZ,
        )
        return dataclasses.replace(
            planned,
            actions=resampled,
            completion_state=state.copy(),
            action_start_index=action_start_index,
        )


class TemporalPolicyChunkPlanner:
    """Create a complete 50-step prediction on the absolute servo timeline."""

    def __init__(
        self,
        *,
        cameras: Any,
        policy: Any,
        feedback_reader: Any,
        action_buffer: TemporalActionBuffer,
        prompt: str,
        safety_config: HardwareSafetyConfig,
        max_inference_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cameras = cameras
        self.policy = policy
        self.feedback_reader = feedback_reader
        self.action_buffer = action_buffer
        self.prompt = str(prompt)
        self.safety_config = safety_config
        self.max_inference_s = float(max_inference_s)
        self.clock = clock

    def plan(self, *, chunk_id: int, warmup_frames: int) -> TemporalPlannedChunk:
        images = self.cameras.read(timeout_ms=5000, warmup_frames=int(warmup_frames))
        snapshot = np.asarray(self.feedback_reader.read(), dtype=np.float64)
        if snapshot.shape != (7,) or not np.all(np.isfinite(snapshot)):
            raise RuntimeError("temporal policy planner received invalid robot feedback")
        # This is the next not-yet-published absolute 50 Hz slot.  The full
        # resampled prediction is indexed from this observation-time slot;
        # inference-latency slots are discarded atomically when it is inserted.
        observation_slot = self.action_buffer.current_slot()
        observation = build_observation(images, snapshot, prompt=self.prompt)
        started_s = self.clock()
        response = self.policy.infer(observation)
        inference_s = self.clock() - started_s
        if inference_s > self.max_inference_s:
            raise RuntimeError(f"policy inference exceeded {self.max_inference_s:.1f} seconds")
        actions = _validated_policy_actions(response)
        raw_horizon_joint_distances = {
            f"d{step}": float(np.linalg.norm(actions[step - 1, :6] - snapshot[:6]))
            for step in (10, 20, 30, 50)
        }
        completion_state = np.asarray(self.feedback_reader.read(), dtype=np.float64)
        if completion_state.shape != (7,) or not np.all(np.isfinite(completion_state)):
            raise RuntimeError("temporal policy planner received invalid post-inference feedback")
        filtered = low_pass_and_limit_actions(
            actions,
            initial_target=snapshot,
            config=self.safety_config,
            dt=1.0 / MODEL_HZ,
        )
        # Do not restart interpolation from completion_state: the filtered
        # waypoints and their absolute time origin both belong to snapshot.
        resampled = resample_actions_causal(
            filtered,
            initial_target=snapshot,
            input_hz=MODEL_HZ,
            output_hz=PUBLISH_HZ,
        )
        return TemporalPlannedChunk(
            chunk_id=int(chunk_id),
            actions=resampled,
            inference_s=float(inference_s),
            observation_state=snapshot.copy(),
            completion_state=completion_state.copy(),
            observation_slot=observation_slot,
            plan_ready_slot=self.action_buffer.current_slot(),
            raw_horizon_joint_distances=raw_horizon_joint_distances,
        )


def _validated_policy_actions(response: Any) -> np.ndarray:
    try:
        actions = np.asarray(response["actions"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("policy response has no valid actions") from exc
    if actions.ndim != 2 or actions.shape[0] != 50 or actions.shape[1] < 7:
        raise RuntimeError(f"policy actions must have shape (50, >=7), got {actions.shape!r}")
    actions = actions[:, :7]
    if not np.all(np.isfinite(actions)):
        raise RuntimeError("policy actions contain NaN or Inf")
    return actions


class PolicyWorker:
    """Single-flight asynchronous planner that only writes completed chunks to the buffer."""

    def __init__(
        self,
        *,
        planner: PolicyChunkPlanner,
        action_buffer: ActionBuffer | StrictChunkActionBuffer | TemporalActionBuffer,
        first_chunk_id: int = 2,
        on_record: Callable[[dict], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.planner = planner
        self.action_buffer = action_buffer
        self.on_record = on_record
        self.clock = clock
        self._next_chunk_id = int(first_chunk_id)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._requested = False
        self._active = False
        self._retry_not_before_s = 0.0
        self.completed_plans = 0
        self.failed_plans = 0
        self._accepted_gripper_intent = 0
        self._pending_gripper_intent = 0
        self._pending_gripper_count = 0

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._run, name="piper-policy-worker", daemon=True)
            self._thread.start()

    def request_prefetch(self) -> bool:
        with self._lock:
            if not self._running or self._stop.is_set() or self._active or self._requested:
                return False
            if self.action_buffer.remaining() >= TARGET_BUFFER_STEPS:
                return False
            if self.clock() < self._retry_not_before_s:
                return False
            self._requested = True
            self._wake.set()
            return True

    def is_busy(self) -> bool:
        with self._lock:
            return self._active or self._requested

    def stop(self, *, timeout_s: float = 6.0) -> None:
        with self._lock:
            self._running = False
            self._stop.set()
            self._wake.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.0, float(timeout_s)))

    def _emit(self, record: dict) -> None:
        if self.on_record is not None:
            self.on_record(record)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=0.1)
            self._wake.clear()
            with self._lock:
                if not self._running or self._stop.is_set():
                    return
                if not self._requested:
                    continue
                self._requested = False
                self._active = True
                chunk_id = self._next_chunk_id
                self._next_chunk_id += 1
            try:
                remaining_at_request = self.action_buffer.remaining()
                expected_handoff_delay_s = max(
                    0.0,
                    (remaining_at_request - BLEND_STEPS) / PUBLISH_HZ,
                )
                planned = self.planner.plan(
                    chunk_id=chunk_id,
                    warmup_frames=1,
                    runtime_alignment_delay_s=expected_handoff_delay_s,
                )
                if self._stop.is_set():
                    return
                # Inference is prefetched at the low watermark, but a ready
                # chunk must not truncate the active execute_steps segment.
                # Handoff only when its final five servo points can be blended.
                while self.action_buffer.remaining() > BLEND_STEPS:
                    if self._stop.wait(0.005):
                        return
                planned = self.planner.align_to_handoff(
                    planned,
                    handoff_state=np.asarray(self.planner.feedback_reader.read(), dtype=np.float64),
                )
                planned, gripper_intent, gripper_reversal_pending = self._stabilize_gripper(planned)
                merge = self.action_buffer.replace_with_blend(
                    planned.actions,
                    chunk_id=chunk_id,
                    now_s=self.clock(),
                    blend_steps=BLEND_STEPS,
                )
                self.completed_plans += 1
                self._emit(
                    {
                        "event": "plan_ready",
                        "chunk_id": chunk_id,
                        "inference_s": planned.inference_s,
                        "new_points": merge.new_points,
                        "old_remaining": merge.old_remaining,
                        "blended_points": merge.blended_points,
                        "buffer_after": merge.buffer_after,
                        "blend_fallback": merge.used_last_published_fallback,
                        "action_start_index": planned.action_start_index,
                        "expected_action_start_index": planned.expected_action_start_index,
                        "observation_state": planned.observation_state.astype(float).tolist(),
                        "completion_state": planned.completion_state.astype(float).tolist(),
                        "first_target": planned.actions[0].astype(float).tolist(),
                        "last_target": planned.actions[-1].astype(float).tolist(),
                        "raw_horizon_joint_distances": planned.raw_horizon_joint_distances,
                        "gripper_intent": gripper_intent,
                        "gripper_reversal_pending": gripper_reversal_pending,
                    }
                )
            except Exception as exc:
                self.failed_plans += 1
                with self._lock:
                    self._retry_not_before_s = self.clock() + MAX_STALE_MS / 1000.0
                self._emit(
                    {
                        "event": "plan_failed",
                        "chunk_id": chunk_id,
                        "error": f"{type(exc).__name__}: {exc}",
                        "buffer_remaining": self.action_buffer.remaining(),
                    }
                )
            finally:
                with self._lock:
                    self._active = False

    def _stabilize_gripper(self, planned: PlannedChunk) -> tuple[PlannedChunk, int, bool]:
        actions = planned.actions.copy()
        current = float(planned.completion_state[6])
        delta = float(np.median(actions[len(actions) // 2 :, 6]) - current)
        requested = 1 if delta >= GRIPPER_INTENT_DELTA_M else -1 if delta <= -GRIPPER_INTENT_DELTA_M else 0
        reversal_pending = False

        if requested == 0:
            self._pending_gripper_intent = 0
            self._pending_gripper_count = 0
        elif self._accepted_gripper_intent in (0, requested):
            self._accepted_gripper_intent = requested
            self._pending_gripper_intent = 0
            self._pending_gripper_count = 0
        else:
            if self._pending_gripper_intent == requested:
                self._pending_gripper_count += 1
            else:
                self._pending_gripper_intent = requested
                self._pending_gripper_count = 1
            if self._pending_gripper_count >= GRIPPER_REVERSAL_CONFIRMATIONS:
                self._accepted_gripper_intent = requested
                self._pending_gripper_intent = 0
                self._pending_gripper_count = 0
            else:
                reversal_pending = True

        if reversal_pending:
            actions[:, 6] = current
        elif self._accepted_gripper_intent > 0:
            actions[:, 6] = np.maximum.accumulate(np.concatenate([[current], actions[:, 6]]))[1:]
        elif self._accepted_gripper_intent < 0:
            actions[:, 6] = np.minimum.accumulate(np.concatenate([[current], actions[:, 6]]))[1:]
        return dataclasses.replace(planned, actions=actions), self._accepted_gripper_intent, reversal_pending


class StrictPolicyWorker:
    """Single-flight RTC worker for exact, non-overlapping short chunks."""

    def __init__(
        self,
        *,
        planner: StrictPolicyChunkPlanner,
        action_buffer: StrictChunkActionBuffer,
        first_chunk_id: int = 2,
        initial_gripper_intent: int = 0,
        on_record: Callable[[dict], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.planner = planner
        self.action_buffer = action_buffer
        self.on_record = on_record
        self.clock = clock
        self._next_chunk_id = int(first_chunk_id)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._requested = False
        self._active = False
        self._retry_not_before_s = 0.0
        self.completed_plans = 0
        self.failed_plans = 0
        self._accepted_gripper_intent = int(np.clip(initial_gripper_intent, -1, 1))
        self._pending_gripper_intent = 0
        self._pending_gripper_count = 0

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._run, name="piper-policy-worker", daemon=True)
            self._thread.start()

    def request_prefetch(self) -> bool:
        with self._lock:
            if not self._running or self._stop.is_set() or self._active or self._requested:
                return False
            if not self.action_buffer.prefetch_due():
                return False
            if self.clock() < self._retry_not_before_s:
                return False
            self._requested = True
            self._wake.set()
            return True

    def is_busy(self) -> bool:
        with self._lock:
            return self._active or self._requested

    def stop(self, *, timeout_s: float = 6.0) -> None:
        with self._lock:
            self._running = False
            self._stop.set()
            self._wake.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.0, float(timeout_s)))

    def _emit(self, record: dict) -> None:
        if self.on_record is not None:
            self.on_record(record)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=0.1)
            self._wake.clear()
            with self._lock:
                if not self._running or self._stop.is_set():
                    return
                if not self._requested:
                    continue
                self._requested = False
                self._active = True
                chunk_id = self._next_chunk_id
                self._next_chunk_id += 1
            try:
                requested_at_s = self.clock()
                active_at_request = self.action_buffer.remaining()
                planned = self.planner.plan(chunk_id=chunk_id, warmup_frames=1)
                if self._stop.is_set():
                    return
                # Keep the active C-step decision immutable.  Lightweight
                # rebasing/filtering is delayed until its final two servo
                # slots so the standby starts from the real handoff target.
                while self.action_buffer.remaining() > STRICT_HANDOFF_PREPARE_SERVO_STEPS:
                    if self._stop.wait(0.002):
                        return
                remaining_before_handoff = self.action_buffer.remaining()
                estimated_handoff_time_s = self.clock() + remaining_before_handoff / PUBLISH_HZ
                prepared = self.planner.prepare_for_handoff(
                    planned,
                    handoff_target=self.action_buffer.handoff_target(),
                    handoff_time_s=estimated_handoff_time_s,
                )
                prepared, gripper_intent, gripper_reversal_pending = self._stabilize_gripper(prepared)
                # Publish immutable plan metadata before making the standby
                # visible to the 50 Hz consumer.  Even when inference finishes
                # after active exhaustion, no command for this chunk can race
                # ahead of its a_ref/plan-id record.
                self._emit(
                    {
                        "event": "plan_metadata",
                        "execution_mode": "latency_aligned_h10",
                        "chunk_id": chunk_id,
                        "action_start_index": prepared.action_start_index,
                        "alignment_age_s": prepared.alignment_age_s,
                        "raw_reference_actions": planned.all_reference_actions[
                            prepared.action_start_index : prepared.action_start_index
                            + self.planner.execute_steps
                        ].astype(float).tolist(),
                        "rebased_reference_actions": prepared.rebased_model_actions.astype(float).tolist(),
                        "post_stabilized_servo_actions": prepared.actions.astype(float).tolist(),
                        "handoff_target": prepared.handoff_target.astype(float).tolist(),
                        "gripper_intent": gripper_intent,
                        "gripper_reversal_pending": gripper_reversal_pending,
                    }
                )
                installed_at_s = self.clock()
                install = self.action_buffer.install_standby(
                    prepared.actions,
                    chunk_id=chunk_id,
                    now_s=installed_at_s,
                    planned_handoff_target=prepared.handoff_target,
                )
                self.completed_plans += 1
                self._emit(
                    {
                        "event": "plan_ready",
                        "execution_mode": "latency_aligned_h10",
                        "chunk_id": chunk_id,
                        "model_action_count": len(planned.reference_actions),
                        "servo_point_count": len(prepared.actions),
                        "action_start_index": prepared.action_start_index,
                        "alignment_age_s": prepared.alignment_age_s,
                        "inference_s": planned.inference_s,
                        "requested_at_s": requested_at_s,
                        "observation_time_s": planned.observation_time_s,
                        "inference_done_time_s": planned.inference_done_time_s,
                        "installed_at_s": installed_at_s,
                        "observation_age_at_install_s": installed_at_s - planned.observation_time_s,
                        "active_at_request": active_at_request,
                        "active_remaining_at_install": install.active_remaining,
                        "standby_points": install.standby_points,
                        "raw_reference_actions": planned.all_reference_actions[
                            prepared.action_start_index : prepared.action_start_index
                            + self.planner.execute_steps
                        ].astype(float).tolist(),
                        "rebased_reference_actions": prepared.rebased_model_actions.astype(float).tolist(),
                        "filtered_model_actions": prepared.filtered_model_actions.astype(float).tolist(),
                        "post_stabilized_servo_actions": prepared.actions.astype(float).tolist(),
                        "gripper_intent": gripper_intent,
                        "gripper_reversal_pending": gripper_reversal_pending,
                        "first_target": prepared.actions[0].astype(float).tolist(),
                        "last_target": prepared.actions[-1].astype(float).tolist(),
                        "handoff_target": prepared.handoff_target.astype(float).tolist(),
                        "raw_horizon_joint_distances": planned.raw_horizon_joint_distances,
                    }
                )
            except Exception as exc:
                self.failed_plans += 1
                with self._lock:
                    self._retry_not_before_s = self.clock() + MAX_STALE_MS / 1000.0
                self._emit(
                    {
                        "event": "plan_failed",
                        "execution_mode": "latency_aligned_h10",
                        "chunk_id": chunk_id,
                        "error": f"{type(exc).__name__}: {exc}",
                        "buffer_remaining": self.action_buffer.remaining(),
                    }
                )
            finally:
                with self._lock:
                    self._active = False

    def _stabilize_gripper(
        self,
        prepared: StrictPreparedChunk,
    ) -> tuple[StrictPreparedChunk, int, bool]:
        actions = prepared.actions.copy()
        current = float(prepared.handoff_target[6])
        delta = float(np.median(actions[len(actions) // 2 :, 6]) - current)
        requested = 1 if delta >= GRIPPER_INTENT_DELTA_M else -1 if delta <= -GRIPPER_INTENT_DELTA_M else 0
        reversal_pending = False

        if requested == 0:
            self._pending_gripper_intent = 0
            self._pending_gripper_count = 0
        elif self._accepted_gripper_intent in (0, requested):
            self._accepted_gripper_intent = requested
            self._pending_gripper_intent = 0
            self._pending_gripper_count = 0
        else:
            if self._pending_gripper_intent == requested:
                self._pending_gripper_count += 1
            else:
                self._pending_gripper_intent = requested
                self._pending_gripper_count = 1
            if self._pending_gripper_count >= GRIPPER_REVERSAL_CONFIRMATIONS:
                self._accepted_gripper_intent = requested
                self._pending_gripper_intent = 0
                self._pending_gripper_count = 0
            else:
                reversal_pending = True

        if reversal_pending:
            actions[:, 6] = current
        elif self._accepted_gripper_intent > 0:
            actions[:, 6] = np.maximum.accumulate(np.concatenate([[current], actions[:, 6]]))[1:]
        elif self._accepted_gripper_intent < 0:
            actions[:, 6] = np.minimum.accumulate(np.concatenate([[current], actions[:, 6]]))[1:]
        return dataclasses.replace(prepared, actions=actions), self._accepted_gripper_intent, reversal_pending


class TemporalPolicyWorker:
    """Single-flight full-horizon worker for temporal-ensemble execution."""

    def __init__(
        self,
        *,
        planner: TemporalPolicyChunkPlanner,
        action_buffer: TemporalActionBuffer,
        first_chunk_id: int = 2,
        on_record: Callable[[dict], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.planner = planner
        self.action_buffer = action_buffer
        self.on_record = on_record
        self.clock = clock
        self._next_chunk_id = int(first_chunk_id)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._requested = False
        self._active = False
        self._retry_not_before_s = 0.0
        self.completed_plans = 0
        self.failed_plans = 0

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._run, name="piper-policy-worker", daemon=True)
            self._thread.start()

    def request_prefetch(self) -> bool:
        with self._lock:
            if not self._running or self._stop.is_set() or self._active or self._requested:
                return False
            if self.clock() < self._retry_not_before_s:
                return False
            reserved_slot = self.action_buffer.reserve_replan()
            if reserved_slot is None:
                return False
            self._requested = True
            self._wake.set()
            return True

    def is_busy(self) -> bool:
        with self._lock:
            return self._active or self._requested

    def stop(self, *, timeout_s: float = 6.0) -> None:
        with self._lock:
            self._running = False
            self._stop.set()
            self._wake.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.0, float(timeout_s)))

    def _emit(self, record: dict) -> None:
        if self.on_record is not None:
            self.on_record(record)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=0.1)
            self._wake.clear()
            with self._lock:
                if not self._running or self._stop.is_set():
                    return
                if not self._requested:
                    continue
                self._requested = False
                self._active = True
                chunk_id = self._next_chunk_id
                self._next_chunk_id += 1
            try:
                planned = self.planner.plan(chunk_id=chunk_id, warmup_frames=1)
                if self._stop.is_set():
                    return
                insert = self.action_buffer.insert_plan(
                    planned.actions,
                    chunk_id=chunk_id,
                    observation_slot=planned.observation_slot,
                    now_s=self.clock(),
                )
                self.completed_plans += 1
                self._emit(
                    {
                        "event": "plan_ready",
                        "chunk_id": chunk_id,
                        "inference_s": planned.inference_s,
                        "new_points": len(planned.actions),
                        "observation_slot": insert.observation_slot,
                        "plan_ready_slot": insert.plan_ready_slot,
                        "plan_first_slot": insert.first_slot,
                        "plan_last_slot": insert.last_slot,
                        "dropped_past_points": insert.dropped_past_points,
                        "overlap_slots": insert.overlap_slots,
                        "ensemble_mode": "full_horizon_absolute_slots",
                        "buffer_horizon_steps": insert.buffer_horizon_steps,
                        "observation_state": planned.observation_state.astype(float).tolist(),
                        "completion_state": planned.completion_state.astype(float).tolist(),
                        "raw_horizon_joint_distances": planned.raw_horizon_joint_distances,
                    }
                )
            except Exception as exc:
                self.failed_plans += 1
                self.action_buffer.cancel_replan_reservation()
                with self._lock:
                    self._retry_not_before_s = self.clock() + MAX_STALE_MS / 1000.0
                self._emit(
                    {
                        "event": "plan_failed",
                        "chunk_id": chunk_id,
                        "error": f"{type(exc).__name__}: {exc}",
                        "buffer_remaining": self.action_buffer.remaining(),
                    }
                )
            finally:
                with self._lock:
                    self._active = False


class ControllerPublisher:
    """Minimal fixed-rate 50 Hz consumer; never captures images or waits for policy inference."""

    def __init__(
        self,
        *,
        action_buffer: ActionBuffer | StrictChunkActionBuffer | TemporalActionBuffer,
        sink: Any,
        safety_config: HardwareSafetyConfig,
        initial_state: np.ndarray,
        safety_profile: str,
        feedback_reader: Any,
        request_prefetch: Callable[[], bool],
        on_record: Callable[[dict], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        output_hz: float = PUBLISH_HZ,
    ) -> None:
        self.action_buffer = action_buffer
        self.sink = sink
        self.safety_config = safety_config
        self.safety = StatefulSafetyFilter(safety_config, initial_state)
        self.safety_profile = str(safety_profile)
        self.feedback_reader = feedback_reader
        self.request_prefetch = request_prefetch
        self.on_record = on_record
        self.clock = clock
        self.output_hz = float(output_hz)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: str | None = None
        self._last_action_s: float | None = None
        self._hold_started_s: float | None = None
        self.publish_count = 0
        self.action_publish_count = 0
        self.hold_publish_count = 0

    @property
    def error(self) -> str | None:
        return self._error

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="piper-controller-50hz", daemon=True)
        self._thread.start()

    def stop(self, *, timeout_s: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, float(timeout_s)))

    def publish_once(self, *, now_s: float | None = None, missed_servo_slots: int = 0) -> dict:
        now = self.clock() if now_s is None else float(now_s)
        item = self.action_buffer.pop_next()
        event = "command"
        reasons: list[str] = []
        if item is not None:
            try:
                # The queue is normally pre-limited point by point.  Apply the
                # same cap again against the command that was actually sent so
                # skipped scheduler slots can never expose a multi-frame jump.
                candidate = _limit_one_servo_delta(
                    item.target,
                    self.action_buffer.last_published_target(),
                    config=self.safety_config,
                )
                if self.safety_profile == "native":
                    filtered = self.safety.filter_native(candidate)
                else:
                    feedback = np.asarray(self.feedback_reader.read(), dtype=np.float64)
                    filtered = self.safety.filter(
                        candidate,
                        snapshot=feedback,
                        feedback=feedback,
                        dt=1.0 / self.output_hz,
                    )
                target = filtered.command
                reasons = list(filtered.reasons)
            except Exception as exc:
                item = None
                target = self.action_buffer.last_published_target()
                event = "invalid_action_hold"
                reasons = [f"{type(exc).__name__}: {exc}"]
        else:
            target = self.action_buffer.last_published_target()
            event = "buffer_empty_hold"

        if not np.all(np.isfinite(target)):
            raise RuntimeError("controller has no finite safe hold target")
        self.sink.send(target)
        self.action_buffer.mark_published(target)
        self.publish_count += 1
        if item is not None:
            self.action_publish_count += 1
            self._last_action_s = now
            self._hold_started_s = None
        else:
            self.hold_publish_count += 1
            if self._hold_started_s is None:
                self._hold_started_s = now

        hold_age_ms = 0.0 if self._hold_started_s is None else (now - self._hold_started_s) * 1000.0
        safe_hold = item is None and hold_age_ms >= MAX_STALE_MS
        remaining = self.action_buffer.remaining()
        prefetch_requested = False
        if self.action_buffer.prefetch_due():
            prefetch_requested = bool(self.request_prefetch())
        ensemble_contributors = (
            self.action_buffer.last_ensemble_contributors()
            if isinstance(self.action_buffer, TemporalActionBuffer)
            else 1 if item is not None else 0
        )
        record = {
            "event": "safe_hold" if safe_hold else event,
            "chunk_id": None if item is None else item.chunk_id,
            "chunk_step": None if item is None else item.chunk_step,
            "command": np.asarray(target, dtype=float).tolist(),
            "reasons": reasons,
            "buffer_remaining": remaining,
            "prefetch_requested": prefetch_requested,
            "hold_age_ms": hold_age_ms,
            "safe_hold": safe_hold,
            "published_at_s": now,
            "ensemble_contributors": ensemble_contributors,
            "missed_servo_slots": int(missed_servo_slots),
        }
        if self.on_record is not None:
            self.on_record(record)
        return record

    def _run(self) -> None:
        period_s = 1.0 / self.output_hz
        next_deadline = self.clock()
        while not self._stop.is_set():
            now = self.clock()
            wait_s = next_deadline - now
            if wait_s > 0:
                self._stop.wait(wait_s)
                if self._stop.is_set():
                    return
                continue
            # If scheduling was delayed, skip elapsed slots before publishing
            # exactly one current target.  Never catch up by bursting old
            # commands at the robot.
            slots_due = max(1, int(math.floor((now - next_deadline) / period_s)) + 1)
            missed_slots = slots_due - 1
            if missed_slots:
                self.action_buffer.skip_steps(missed_slots)
            try:
                self.publish_once(now_s=now, missed_servo_slots=missed_slots)
            except Exception as exc:  # A sink failure is fatal; never catch up with queued commands.
                self._error = f"{type(exc).__name__}: {exc}"
                self._stop.set()
                return
            next_deadline += slots_due * period_s
