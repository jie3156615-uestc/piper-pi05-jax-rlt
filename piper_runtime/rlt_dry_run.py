from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class RLTDecisionInputs:
    stop_pressed: bool
    human_takeover: bool
    phase_gate_active: bool
    a_ref: np.ndarray
    a_actor: np.ndarray | None
    a_human: np.ndarray | None


@dataclasses.dataclass(frozen=True)
class RLTDecision:
    source: str
    a_exec: np.ndarray | None


def choose_action_source(inputs: RLTDecisionInputs) -> RLTDecision:
    if inputs.stop_pressed:
        return RLTDecision(source="stop", a_exec=None)
    if inputs.human_takeover:
        if inputs.a_human is None:
            return RLTDecision(source="safety_block", a_exec=None)
        return RLTDecision(source="human_pika", a_exec=np.asarray(inputs.a_human, dtype=np.float32))
    if inputs.phase_gate_active:
        if inputs.a_actor is None:
            return RLTDecision(source="safety_block", a_exec=None)
        return RLTDecision(source="rlt", a_exec=np.asarray(inputs.a_actor, dtype=np.float32))
    return RLTDecision(source="pi05", a_exec=np.asarray(inputs.a_ref, dtype=np.float32))
