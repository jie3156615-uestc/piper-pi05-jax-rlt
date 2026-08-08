import pytest

from piper_runtime.low_amplitude_motion import MotionSafetyError, compute_joint6_target, validate_feedback


def test_joint6_target_is_positive_two_degrees_only():
    origin = [100, 200, 300, 400, 500, 600]
    target = compute_joint6_target(origin, degrees=2.0)
    assert target[:5] == origin[:5]
    assert target[5] - origin[5] == 2000


def test_feedback_guard_rejects_other_joint_motion():
    origin = [0, 0, 0, 0, 0, 0]
    target = compute_joint6_target(origin, degrees=2.0)
    with pytest.raises(MotionSafetyError):
        validate_feedback(origin, [1000, 0, 0, 0, 0, 0], target)


def test_feedback_guard_accepts_expected_joint6_motion():
    origin = [0, 0, 0, 0, 0, 0]
    target = compute_joint6_target(origin, degrees=2.0)
    validate_feedback(origin, [0, 0, 0, 0, 0, 1500], target)
