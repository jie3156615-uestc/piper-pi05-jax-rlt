from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from piper_runtime.rlt_keyboard import RLTKeyboardSnapshot


class RLTEpisodeLogger:
    """Append-only JSONL logger for real-robot RLT episodes."""

    def __init__(self, path: str | Path, *, episode_id: str):
        self.path = Path(path)
        self.episode_id = episode_id
        self._stream = None
        self._closed = False
        self._terminal_written = False

    def __enter__(self) -> "RLTEpisodeLogger":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        if self._stream is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("w", encoding="utf-8", buffering=1)
        self._closed = False

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        self._closed = True

    def append_step(
        self,
        *,
        t: int,
        global_image: str,
        wrist_image: str,
        z_rl: Any,
        state: Any,
        a_ref: Any,
        a_exec: Any,
        a_human: Any | None,
        a_actor: Any | None,
        a_actor_safe: Any | None = None,
        source: str,
        phase_probability: float,
        gate_active: bool,
        keyboard: RLTKeyboardSnapshot,
        timestamp_ns: int | None,
        gate_state: str | None = None,
        gate_enter_t: int | None = None,
        gate_exit_t: int | None = None,
        gate_reason: str | None = None,
        takeover_active: bool | None = None,
        takeover_started_t: int | None = None,
        takeover_ended_t: int | None = None,
        policy_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._stream is None:
            self.open()
        if self._terminal_written:
            raise RuntimeError("cannot append a step after the terminal episode row has been written")

        reward = 0.0 if keyboard.terminal_reward is None else float(keyboard.terminal_reward)
        done = bool(keyboard.episode_done)
        serialized_policy_metadata = {} if policy_metadata is None else _metadata_to_jsonable(policy_metadata)
        row = {
            "episode_id": self.episode_id,
            "t": int(t),
            "global_image": str(global_image),
            "wrist_image": str(wrist_image),
            "z_rl": _to_jsonable(z_rl),
            "state": _to_jsonable(state),
            "a_ref": _to_jsonable(a_ref),
            "a_exec": _to_jsonable(a_exec),
            "a_human": None if a_human is None else _to_jsonable(a_human),
            "a_actor": None if a_actor is None else _to_jsonable(a_actor),
            # ``a_actor`` remains the raw network proposal for backward
            # compatibility.  ``a_actor_safe`` is the complete atomically
            # governed C10 plan and is the only Actor payload eligible for the
            # command mux.
            "a_actor_raw": None if a_actor is None else _to_jsonable(a_actor),
            "a_actor_safe": (
                None if a_actor_safe is None else _to_jsonable(a_actor_safe)
            ),
            "source": str(source),
            "reward": reward,
            "done": done,
            "phase_probability": float(phase_probability),
            "gate_active": bool(gate_active),
            "gate_state": "ACTIVE" if gate_active else "IDLE" if gate_state is None else str(gate_state),
            "gate_enter_t": None if gate_enter_t is None else int(gate_enter_t),
            "gate_exit_t": None if gate_exit_t is None else int(gate_exit_t),
            "gate_reason": "" if gate_reason is None else str(gate_reason),
            "timestamp_ns": None if timestamp_ns is None else int(timestamp_ns),
            "keyboard": {
                "human_takeover": bool(keyboard.human_takeover),
                "waiting_for_reward": bool(keyboard.waiting_for_reward),
                "terminal_reward": keyboard.terminal_reward,
                "stop_requested": bool(keyboard.stop_requested),
                "episode_done": bool(keyboard.episode_done),
                "last_key": keyboard.last_key,
                "mode": keyboard.mode,
                "takeover_started_s": keyboard.takeover_started_s,
                "takeover_ended_s": keyboard.takeover_ended_s,
            },
            "takeover_active": bool(keyboard.human_takeover) if takeover_active is None else bool(takeover_active),
            "takeover_started_t": None if takeover_started_t is None else int(takeover_started_t),
            "takeover_ended_t": None if takeover_ended_t is None else int(takeover_ended_t),
            "policy_plan_id": serialized_policy_metadata.get("policy_plan_id"),
            "policy_observation_t": serialized_policy_metadata.get("policy_observation_t"),
            "plan_offset": serialized_policy_metadata.get("plan_offset"),
            "behavior_actor_checkpoint": serialized_policy_metadata.get("behavior_actor_checkpoint"),
            "policy_metadata": serialized_policy_metadata,
        }
        assert self._stream is not None
        self._stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self._stream.flush()
        self._terminal_written = done
        return row


def _to_jsonable(value: Any) -> Any:
    array = np.asarray(value)
    if array.dtype.kind in {"f", "i", "u", "b"}:
        return array.tolist()
    raise TypeError(f"expected numeric array-like value, got dtype {array.dtype}")


def _metadata_to_jsonable(metadata: dict[str, Any]) -> dict[str, Any]:
    json.dumps(metadata)
    return dict(metadata)
