from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np

from piper_runtime.rlt_keyboard import RLTKeyboardSnapshot

SourceName = Literal["pi05", "rlt", "human_pika", "safety_block", "stop"]


@dataclasses.dataclass(frozen=True)
class TimedCommand:
    value: np.ndarray | None
    timestamp_s: float | None


@dataclasses.dataclass(frozen=True)
class MuxSelection:
    command: np.ndarray | None
    source: SourceName
    reason: str
    should_stop: bool = False


@dataclasses.dataclass(frozen=True)
class _ValidatedCommand:
    value: np.ndarray | None
    status: str


@dataclasses.dataclass
class RLTCommandMux:
    action_dim: int = 7
    freshness_s: float = 0.1

    def select(
        self,
        *,
        now_s: float,
        keyboard: RLTKeyboardSnapshot,
        feedback: TimedCommand | None,
        model: TimedCommand | None,
        human: TimedCommand | None,
        actor: TimedCommand | None = None,
        rlt_active: bool = False,
        allow_actor_live: bool = False,
    ) -> MuxSelection:
        feedback_result = self._validate(feedback, now_s=now_s, label="feedback")

        if keyboard.episode_done:
            return self._hold(
                feedback_result,
                reason="episode_done",
                source="stop",
                should_stop=True,
            )

        if keyboard.waiting_for_reward:
            return self._hold(
                feedback_result,
                reason="waiting_for_reward_hold",
                source="stop",
            )

        # Pressing `s` only arms takeover.  Until the subsequent physical Pika
        # activation edge is observed, neither an old nor a continuously
        # published human command may move the arm.
        if keyboard.mode == "TAKEOVER_ARMED":
            return self._hold(
                feedback_result,
                reason="takeover_armed_hold",
                source="safety_block",
            )

        if keyboard.human_takeover:
            human_result = self._validate(human, now_s=now_s, label="human")
            if human_result.value is not None:
                return MuxSelection(command=human_result.value, source="human_pika", reason="fresh_human")
            return self._hold(feedback_result, reason=f"human_{human_result.status}", source="safety_block")

        if allow_actor_live and rlt_active:
            actor_result = self._validate(actor, now_s=now_s, label="actor")
            if actor_result.value is not None:
                return MuxSelection(command=actor_result.value, source="rlt", reason="fresh_rlt_actor")
            # A missing/late Actor must not freeze the arm. Fall back to the
            # frozen Pi0.5 reference for this control step and record why.
            model_result = self._validate(model, now_s=now_s, label="model")
            if model_result.value is not None:
                return MuxSelection(
                    command=model_result.value,
                    source="pi05",
                    reason=f"rlt_actor_{actor_result.status}_fallback_model",
                )
            return self._hold(
                feedback_result,
                reason=f"rlt_actor_{actor_result.status}_model_{model_result.status}",
                source="safety_block",
            )

        model_result = self._validate(model, now_s=now_s, label="model")
        if model_result.value is not None:
            return MuxSelection(command=model_result.value, source="pi05", reason="fresh_model")
        return self._hold(feedback_result, reason=f"model_{model_result.status}", source="safety_block")

    def _validate(self, command: TimedCommand | None, *, now_s: float, label: str) -> _ValidatedCommand:
        if command is None or command.value is None:
            return _ValidatedCommand(value=None, status="missing")
        if command.timestamp_s is None:
            return _ValidatedCommand(value=None, status="missing_timestamp")
        age_s = now_s - float(command.timestamp_s)
        if age_s < 0:
            return _ValidatedCommand(value=None, status="from_future")
        if age_s > self.freshness_s:
            return _ValidatedCommand(value=None, status="stale")
        try:
            value = np.asarray(command.value, dtype=np.float32)
        except (TypeError, ValueError):
            return _ValidatedCommand(value=None, status="invalid")
        if value.shape != (self.action_dim,):
            return _ValidatedCommand(value=None, status="invalid")
        if not np.all(np.isfinite(value)):
            return _ValidatedCommand(value=None, status="invalid")
        return _ValidatedCommand(value=value.copy(), status="fresh")

    def _hold(
        self,
        feedback: _ValidatedCommand,
        *,
        reason: str,
        source: SourceName,
        should_stop: bool = False,
    ) -> MuxSelection:
        if feedback.value is None:
            return MuxSelection(
                command=None,
                source="stop",
                reason=f"{reason}_no_safe_feedback_{feedback.status}",
                should_stop=True,
            )
        return MuxSelection(command=feedback.value, source=source, reason=reason, should_stop=should_stop)
