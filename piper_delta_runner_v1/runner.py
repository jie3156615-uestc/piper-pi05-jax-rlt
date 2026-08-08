"""Mock-only Piper delta runner with ordered safety filtering."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class UnsafeHardwareEmitterError(RuntimeError):
    """Raised when the offline runner is given a hardware-capable emitter."""


@dataclass
class SafetyProfile:
    mode: str
    joint_min: np.ndarray
    joint_max: np.ndarray
    gripper_min: float
    gripper_max: float
    max_delta: np.ndarray
    max_velocity: np.ndarray
    max_acceleration: np.ndarray
    control_period_s: float
    max_plan_age_s: float
    watchdog_timeout_s: float
    allow_hardware_execution: bool = False
    calibration_status: str = "mock_only"

    def __post_init__(self) -> None:
        for name in ("joint_min", "joint_max", "max_delta", "max_velocity", "max_acceleration"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (6,):
                raise ValueError(f"{name} must have shape (6,), got {value.shape}")
            setattr(self, name, value)
        if self.control_period_s <= 0:
            raise ValueError("control_period_s must be positive")
        if self.max_plan_age_s < 0 or self.watchdog_timeout_s < 0:
            raise ValueError("freshness thresholds must be non-negative")


@dataclass
class Plan:
    plan_id: str
    q_feedback_snapshot: np.ndarray
    predicted_delta_chunk: np.ndarray
    safety_profile: str
    planned_at: float

    def __post_init__(self) -> None:
        self.q_feedback_snapshot = np.asarray(self.q_feedback_snapshot, dtype=float)
        self.predicted_delta_chunk = np.asarray(self.predicted_delta_chunk, dtype=float)
        if self.q_feedback_snapshot.shape != (7,):
            raise ValueError("q_feedback_snapshot must have shape (7,)")
        if self.predicted_delta_chunk.ndim != 2 or self.predicted_delta_chunk.shape[1] != 7:
            raise ValueError("predicted_delta_chunk must have shape (H, 7)")


@dataclass
class ExecutionResult:
    emitted: list[list[float]] = field(default_factory=list)
    audit: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str | None = None


class MockEmitter:
    is_hardware = False

    def __init__(self, feedback_joint: np.ndarray | None = None) -> None:
        self.feedback_joint = np.asarray(
            np.zeros(7, dtype=float) if feedback_joint is None else feedback_joint,
            dtype=float,
        )
        self.emitted: list[dict[str, Any]] = []
        self.stop_reasons: list[str] = []

    def emit(self, value: np.ndarray, *, timestamp: float) -> np.ndarray:
        self.emitted.append({"timestamp": timestamp, "value": value.tolist()})
        return self.feedback_joint.copy()

    def stop(self, reason: str) -> None:
        self.stop_reasons.append(reason)


class DeltaRunner:
    """Restores a chunk from one feedback snapshot and emits mock values only."""

    def __init__(self, safety_profile: SafetyProfile, emitter: Any) -> None:
        if getattr(emitter, "is_hardware", False):
            raise UnsafeHardwareEmitterError("offline v1 runner refuses hardware emitters")
        if safety_profile.allow_hardware_execution:
            raise UnsafeHardwareEmitterError("offline v1 runner refuses hardware-enabled profiles")
        self.safety_profile = safety_profile
        self.emitter = emitter

    def _stop(self, result: ExecutionResult, reason: str, *, timestamp: float) -> ExecutionResult:
        result.stop_reason = reason
        result.events.append({"event": "stop", "reason": reason, "timestamp": timestamp})
        self.emitter.stop(reason)
        return result

    def execute_plan(
        self,
        plan: Plan,
        *,
        now: float,
        network_connected: bool = True,
    ) -> ExecutionResult:
        profile = self.safety_profile
        result = ExecutionResult()
        if plan.safety_profile != profile.mode:
            return self._stop(result, "safety_profile_mismatch", timestamp=now)
        if not np.all(np.isfinite(plan.q_feedback_snapshot)):
            return self._stop(result, "non_finite", timestamp=now)

        previous_target = plan.q_feedback_snapshot.copy()
        previous_velocity = np.zeros(6, dtype=float)
        dt = profile.control_period_s
        for index, raw_delta in enumerate(plan.predicted_delta_chunk):
            timestamp = now + index * dt
            if not np.all(np.isfinite(raw_delta)):
                return self._stop(result, "non_finite", timestamp=timestamp)

            restored_target = plan.q_feedback_snapshot.copy()
            restored_target[:6] += raw_delta[:6]
            restored_target[6] = raw_delta[6]
            if (
                np.any(restored_target[:6] < profile.joint_min)
                or np.any(restored_target[:6] > profile.joint_max)
                or restored_target[6] < profile.gripper_min
                or restored_target[6] > profile.gripper_max
            ):
                return self._stop(result, "hard_limit", timestamp=timestamp)

            reasons: list[str] = []
            clipped_delta = np.clip(raw_delta[:6], -profile.max_delta, profile.max_delta)
            if not np.allclose(clipped_delta, raw_delta[:6]):
                reasons.append("delta_clip")
            filtered_target = plan.q_feedback_snapshot.copy()
            filtered_target[:6] += clipped_delta
            filtered_target[6] = raw_delta[6]
            clipped_target = filtered_target.copy()

            desired_velocity = (filtered_target[:6] - previous_target[:6]) / dt
            limited_velocity = np.clip(desired_velocity, -profile.max_velocity, profile.max_velocity)
            if not np.allclose(limited_velocity, desired_velocity):
                reasons.append("max_velocity")

            minimum_velocity = previous_velocity - profile.max_acceleration * dt
            maximum_velocity = previous_velocity + profile.max_acceleration * dt
            accelerated_velocity = np.clip(limited_velocity, minimum_velocity, maximum_velocity)
            if not np.allclose(accelerated_velocity, limited_velocity):
                reasons.append("max_acceleration")

            filtered_target[:6] = previous_target[:6] + accelerated_velocity * dt
            if not network_connected:
                return self._stop(result, "network_disconnect", timestamp=timestamp)
            if timestamp - plan.planned_at > profile.max_plan_age_s:
                return self._stop(result, "stale_plan", timestamp=timestamp)
            if timestamp - plan.planned_at > profile.watchdog_timeout_s:
                return self._stop(result, "watchdog_timeout", timestamp=timestamp)

            feedback = self.emitter.emit(filtered_target.copy(), timestamp=timestamp)
            emitted = filtered_target.tolist()
            result.emitted.append(emitted)
            result.audit.append(
                {
                    "plan_id": plan.plan_id,
                    "step": index,
                    "raw_delta": raw_delta.tolist(),
                    "restored_target": restored_target.tolist(),
                    "clipped_target": clipped_target.tolist(),
                    "filtered_target": emitted,
                    "emit_value": emitted,
                    "feedback_value": np.asarray(feedback, dtype=float).tolist(),
                    "reasons": reasons,
                    "timestamp": timestamp,
                }
            )
            previous_target = filtered_target
            previous_velocity = accelerated_velocity
        return result

    @staticmethod
    def write_audit_jsonl(path: Path, result: ExecutionResult) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            for record in result.audit:
                stream.write(json.dumps({"event": "emit", **record}, sort_keys=True) + "\n")
            for event in result.events:
                stream.write(json.dumps(event, sort_keys=True) + "\n")
