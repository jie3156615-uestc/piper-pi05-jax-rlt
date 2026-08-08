from __future__ import annotations

import numpy as np
import pytest

from piper_runtime.rlt_phase_gate import PhaseGateConfig
from piper_runtime.rlt_phase_gate import SingleLatchPhaseGate
from piper_runtime.rlt_phase_gate import should_run_classifier


def test_single_latch_gate_enters_after_consecutive_high_predictions() -> None:
    gate = SingleLatchPhaseGate(PhaseGateConfig(enter_threshold=0.5, enter_consecutive_frames=3))

    assert gate.update(0.6, t=10).active is False
    assert gate.update(0.7, t=11).active is False
    snapshot = gate.update(0.8, t=12)

    assert snapshot.active is True
    assert snapshot.state == "ACTIVE"
    assert snapshot.enter_t == 12
    assert snapshot.reason == "entered_single_latch"


def test_single_latch_gate_does_not_exit_on_low_probability_after_entry() -> None:
    gate = SingleLatchPhaseGate(PhaseGateConfig(enter_threshold=0.5, enter_consecutive_frames=2))
    gate.update(0.9, t=1)
    gate.update(0.9, t=2)

    snapshot = gate.update(0.01, t=3)

    assert snapshot.active is True
    assert snapshot.state == "ACTIVE"
    assert snapshot.enter_t == 2
    assert snapshot.exit_t is None
    assert snapshot.reason == "locked_until_terminal"


def test_single_latch_gate_exits_only_on_terminal_event() -> None:
    gate = SingleLatchPhaseGate(PhaseGateConfig(enter_threshold=0.5, enter_consecutive_frames=1))
    gate.update(0.9, t=4)

    snapshot = gate.update(None, t=20, terminal=True, terminal_reason="episode_done")

    assert snapshot.active is False
    assert snapshot.state == "EXITED"
    assert snapshot.enter_t == 4
    assert snapshot.exit_t == 20
    assert snapshot.reason == "episode_done"


def test_single_latch_gate_resets_high_count_on_low_probability_before_entry() -> None:
    gate = SingleLatchPhaseGate(PhaseGateConfig(enter_threshold=0.5, enter_consecutive_frames=2))

    assert gate.update(0.9, t=0).high_count == 1
    assert gate.update(0.1, t=1).high_count == 0
    assert gate.update(0.9, t=2).active is False
    assert gate.update(0.9, t=3).active is True
    assert gate.snapshot().enter_t == 3


def test_phase_gate_rejects_invalid_probability() -> None:
    gate = SingleLatchPhaseGate()

    with pytest.raises(ValueError, match="finite"):
        gate.update(float("nan"), t=0)


def test_should_run_classifier_period() -> None:
    assert [should_run_classifier(t=t, period=3) for t in range(7)] == [
        True,
        False,
        False,
        True,
        False,
        False,
        True,
    ]

    with pytest.raises(ValueError, match="period"):
        should_run_classifier(t=0, period=0)


def test_gate_clamps_probability_to_unit_interval() -> None:
    gate = SingleLatchPhaseGate(PhaseGateConfig(enter_threshold=1.0, enter_consecutive_frames=1))

    snapshot = gate.update(np.float32(2.0), t=1)

    assert snapshot.probability == 1.0
    assert snapshot.active is True
