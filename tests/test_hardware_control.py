import math

import numpy as np
import pytest

from piper_runtime.hardware_control import (
    HardwareSafetyConfig,
    HardwareSafetyError,
    InterpolatedPiperCommandSink,
    PiperCommandSink,
    StatefulSafetyFilter,
)


def make_filter(state=None):
    state = np.zeros(7, dtype=float) if state is None else np.asarray(state, dtype=float)
    return StatefulSafetyFilter(HardwareSafetyConfig(), state)


def test_small_boundary_overrun_is_clipped_to_official_limits():
    safety = make_filter()
    target = np.zeros(7)
    target[1] = math.radians(-1)
    target[2] = math.radians(1)
    result = safety.filter(target, snapshot=np.zeros(7), feedback=np.zeros(7), dt=1 / 30)
    assert result.command[1] >= 0
    assert result.command[2] <= 0
    assert "joint_limit_clip" in result.reasons


def test_large_official_limit_overrun_is_rejected():
    safety = make_filter()
    target = np.zeros(7)
    target[1] = math.radians(-6)
    with pytest.raises(HardwareSafetyError):
        safety.filter(target, snapshot=np.zeros(7), feedback=np.zeros(7), dt=1 / 30)


def test_plan_delta_velocity_acceleration_and_step_are_limited():
    safety = make_filter()
    target = np.zeros(7)
    target[0] = math.radians(20)
    result = safety.filter(target, snapshot=np.zeros(7), feedback=np.zeros(7), dt=1 / 30)
    assert abs(math.degrees(result.command[0])) <= math.degrees(0.03 / 30.0)
    assert "plan_delta_clip" in result.reasons
    assert "max_acceleration" in result.reasons


def test_gripper_changes_by_at_most_two_millimeters():
    safety = make_filter()
    target = np.zeros(7)
    target[6] = 0.03
    result = safety.filter(target, snapshot=np.zeros(7), feedback=np.zeros(7), dt=1 / 30)
    assert result.command[6] == pytest.approx(0.002)
    assert "gripper_step_clip" in result.reasons


def test_pika_native_open_gripper_is_accepted_and_clipped_to_piper_limit():
    safety = make_filter()
    target = np.zeros(7)
    target[6] = 0.09670703858137131
    result = safety.filter(target, snapshot=np.zeros(7), feedback=np.zeros(7), dt=1 / 30)
    assert result.command[6] == pytest.approx(0.002)
    assert "gripper_step_clip" in result.reasons


def test_implausibly_large_gripper_target_is_rejected():
    safety = make_filter()
    target = np.zeros(7)
    target[6] = 0.12
    with pytest.raises(HardwareSafetyError):
        safety.filter(target, snapshot=np.zeros(7), feedback=np.zeros(7), dt=1 / 30)


def test_hold_feedback_resynchronizes_after_external_tracking_jump():
    safety = make_filter()
    feedback = np.zeros(7)
    feedback[4] = math.radians(7)

    result = safety.filter(feedback, snapshot=feedback, feedback=feedback, dt=1 / 30)

    np.testing.assert_allclose(result.command, feedback)
    assert "tracking_resync" in result.reasons


def test_non_hold_target_still_rejects_tracking_jump():
    safety = make_filter()
    feedback = np.zeros(7)
    feedback[4] = math.radians(7)
    target = feedback.copy()
    target[0] = math.radians(1)

    with pytest.raises(HardwareSafetyError, match="tracking error"):
        safety.filter(target, snapshot=feedback, feedback=feedback, dt=1 / 30)


def test_human_native_preserves_large_valid_target_without_dynamic_attenuation():
    safety = make_filter()
    target = np.array([1.0, 1.2, -1.4, 0.8, 0.9, -1.8, 0.06])

    result = safety.filter_human_native(target)

    np.testing.assert_allclose(result.command, target)
    assert result.reasons == []


def test_human_native_clips_only_official_joint_and_gripper_ranges():
    safety = make_filter()
    target = np.array([0.0, math.radians(-1), math.radians(1), 0.0, 0.0, 0.0, 0.096707])

    result = safety.filter_human_native(target)

    assert result.command[1] == pytest.approx(0.0)
    assert result.command[2] == pytest.approx(0.0)
    assert result.command[6] == pytest.approx(0.08)
    assert set(result.reasons) == {"joint_limit_clip", "gripper_range_clip"}


def test_human_native_rejects_hard_limit_overrun():
    safety = make_filter()
    target = np.zeros(7)
    target[1] = math.radians(-6)

    with pytest.raises(HardwareSafetyError, match="official limit"):
        safety.filter_human_native(target)


def test_human_native_rejects_implausible_gripper_target():
    safety = make_filter()
    target = np.zeros(7)
    target[6] = 0.12

    with pytest.raises(HardwareSafetyError, match="gripper"):
        safety.filter_human_native(target)


def test_model_native_low_pass_and_step_limit_only_model_commands():
    config = HardwareSafetyConfig(
        model_smoothing_tau_s=0.05,
        model_max_joint_step=math.radians(3.0),
        model_max_gripper_step=0.02,
    )
    safety = StatefulSafetyFilter(config, np.zeros(7))
    target = np.array([math.radians(10), 0, 0, 0, 0, 0, 0.08])

    result = safety.filter_model_native(target, dt=1 / 30)

    assert result.command[0] == pytest.approx(math.radians(3.0))
    assert result.command[6] == pytest.approx(0.02)
    assert set(result.reasons) >= {
        "model_low_pass",
        "model_joint_step_clip",
        "model_gripper_step_clip",
    }


def test_model_native_smoothing_attenuates_direction_reversal():
    safety = StatefulSafetyFilter(HardwareSafetyConfig(), np.zeros(7))
    positive = np.zeros(7)
    positive[0] = math.radians(2)
    first = safety.filter_model_native(positive, dt=1 / 30).command.copy()
    negative = np.zeros(7)
    negative[0] = math.radians(-2)

    second = safety.filter_model_native(negative, dt=1 / 30).command

    assert abs(second[0] - first[0]) < abs(negative[0] - first[0])


class FakeSdk:
    def __init__(self):
        self.calls = []

    def MotionCtrl_2(self, *args):
        self.calls.append(("motion", args))

    def JointCtrl(self, *args):
        self.calls.append(("joint", args))

    def GripperCtrl(self, *args):
        self.calls.append(("gripper", args))


def test_command_sink_configures_movej_once_then_sends_only_targets():
    sdk = FakeSdk()
    sink = PiperCommandSink(sdk)
    command = np.array([math.radians(1), 0, 0, 0, 0, 0, 0.01])
    sink.configure_motion_mode()
    sink.configure_motion_mode()
    sink.send(command)
    sink.send(command)
    assert sdk.calls[0] == ("motion", (0x01, 0x01, 10, 0x00))
    assert sdk.calls[1] == ("joint", (1000, 0, 0, 0, 0, 0))
    assert sdk.calls[2] == ("gripper", (10000, 1000, 0x01, 0))
    assert [name for name, _args in sdk.calls].count("motion") == 1


def test_interpolated_sink_outputs_convex_50hz_servo_points():
    class CaptureSink:
        def __init__(self):
            self.sent = []

        def send(self, command):
            self.sent.append(np.asarray(command).copy())

    clock = [0.0]
    native = CaptureSink()
    sink = InterpolatedPiperCommandSink(
        native,
        input_hz=30.0,
        output_hz=1.0,
        now_fn=lambda: clock[0],
    )
    try:
        sink.send(np.ones(7))
        assert sink._emit(0.0)
        clock[0] = 1 / 30
        sink.send(np.full(7, 2.0))
        assert sink._emit(clock[0] + 0.02)
        assert sink._emit(clock[0] + 1 / 30)
    finally:
        sink.close()

    np.testing.assert_allclose(native.sent[-2], np.full(7, 1.6), atol=1e-6)
    np.testing.assert_allclose(native.sent[-1], np.full(7, 2.0), atol=1e-6)
