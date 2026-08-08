import numpy as np
import pytest

from piper_runtime.piper_feedback import FeedbackHealthError, PiperFeedbackReader


class Value:
    pass


def make_sdk(hz=200.0, can_fps=3040.0, err_code=0):
    sdk = Value()
    joints = Value()
    joints.Hz = hz
    joints.joint_state = Value()
    for index, raw in enumerate([57295.7795, 0, -57295.7795, 28647.88975, 0, 0], 1):
        setattr(joints.joint_state, "joint_%d" % index, raw)
    gripper = Value()
    gripper.Hz = hz
    gripper.gripper_state = Value()
    gripper.gripper_state.grippers_angle = 40000
    status = Value()
    status.Hz = hz
    status.arm_status = Value()
    status.arm_status.err_code = err_code
    sdk.GetArmJointMsgs = lambda: joints
    sdk.GetArmGripperMsgs = lambda: gripper
    sdk.GetArmStatus = lambda: status
    sdk.GetCanFps = lambda: can_fps
    return sdk


def test_read_converts_sdk_units_without_writing():
    state = PiperFeedbackReader(make_sdk()).read()
    np.testing.assert_allclose(state, [1.0, 0.0, -1.0, 0.5, 0.0, 0.0, 0.04], rtol=1e-6)


@pytest.mark.parametrize("hz,can_fps,err_code", [(0.0, 3040.0, 0), (200.0, 0.0, 0), (200.0, 3040.0, 1)])
def test_read_rejects_unhealthy_feedback(hz, can_fps, err_code):
    with pytest.raises(FeedbackHealthError):
        PiperFeedbackReader(make_sdk(hz=hz, can_fps=can_fps, err_code=err_code)).read()


def test_wait_until_healthy_handles_sdk_startup():
    sdk = make_sdk()
    calls = iter([0.0, 3040.0])
    sdk.GetCanFps = lambda: next(calls)
    state = PiperFeedbackReader(sdk).wait_until_healthy(timeout_s=0.1, poll_s=0.0)
    assert state.shape == (7,)
