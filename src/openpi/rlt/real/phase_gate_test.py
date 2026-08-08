from openpi.rlt.real.phase_gate import HysteresisGate
from openpi.rlt.real.phase_gate import event_alignment


def test_hysteresis_requires_consecutive_positive_frames():
    gate = HysteresisGate(enter_threshold=0.7, exit_threshold=0.4, enter_frames=3, exit_frames=2)
    outputs = [gate.update(p) for p in [0.8, 0.8, 0.6, 0.8, 0.8, 0.8]]

    assert outputs == [False, False, False, False, False, True]


def test_hysteresis_does_not_flicker_on_single_low_frame():
    gate = HysteresisGate(enter_threshold=0.7, exit_threshold=0.4, enter_frames=2, exit_frames=2)
    for p in [0.9, 0.9]:
        active = gate.update(p)
    assert active is True

    assert gate.update(0.1) is True
    assert gate.update(0.1) is False


def test_event_alignment_reports_lead_lag():
    result = event_alignment(predicted_enter_t=8, manual_enter_t=10, predicted_exit_t=20, manual_exit_t=22)

    assert result["enter_lead_frames"] == 2
    assert result["exit_lag_frames"] == -2
