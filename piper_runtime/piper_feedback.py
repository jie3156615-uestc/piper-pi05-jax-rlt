"""Read-only Piper V2 feedback adapter."""

import time

import numpy as np


ARM_FACTOR = 57295.7795
GRIPPER_FACTOR = 1000000.0


class FeedbackHealthError(RuntimeError):
    pass


def decode_arm_error_code(err_code: int) -> list:
    names = []
    joint_limit_names = [
        "joint_1_angle_limit",
        "joint_2_angle_limit",
        "joint_3_angle_limit",
        "joint_4_angle_limit",
        "joint_5_angle_limit",
        "joint_6_angle_limit",
    ]
    joint_comm_names = [
        "joint_1_communication_error",
        "joint_2_communication_error",
        "joint_3_communication_error",
        "joint_4_communication_error",
        "joint_5_communication_error",
        "joint_6_communication_error",
    ]
    for bit, name in enumerate(joint_limit_names):
        if err_code & (1 << bit):
            names.append(name)
    for bit, name in enumerate(joint_comm_names, start=8):
        if err_code & (1 << bit):
            names.append(name)
    reserved = err_code & ~0x3F3F
    if reserved:
        names.append("reserved_bits_0x%X" % reserved)
    return names


class PiperFeedbackReader:
    def __init__(self, sdk):
        self.sdk = sdk

    def read(self) -> np.ndarray:
        joints = self.sdk.GetArmJointMsgs()
        gripper = self.sdk.GetArmGripperMsgs()
        status = self.sdk.GetArmStatus()
        can_fps = float(self.sdk.GetCanFps())
        frequencies = [float(joints.Hz), float(gripper.Hz), float(status.Hz), can_fps]
        if min(frequencies) <= 0:
            raise FeedbackHealthError("non-positive feedback frequency: %r" % frequencies)
        err_code = int(status.arm_status.err_code)
        if err_code != 0:
            raise FeedbackHealthError("Piper arm error code: %d (%s)" % (err_code, ",".join(decode_arm_error_code(err_code)) or "unknown"))
        joint_state = joints.joint_state
        raw_gripper = gripper.gripper_state.grippers_angle
        if isinstance(raw_gripper, (tuple, list, np.ndarray)):
            raw_gripper = raw_gripper[0]
        state = np.asarray(
            [
                getattr(joint_state, "joint_%d" % index) / ARM_FACTOR
                for index in range(1, 7)
            ]
            + [float(raw_gripper) / GRIPPER_FACTOR],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(state)):
            raise FeedbackHealthError("non-finite Piper feedback")
        return state

    def wait_until_healthy(self, timeout_s: float = 5.0, poll_s: float = 0.02) -> np.ndarray:
        deadline = time.monotonic() + timeout_s
        last_error = None
        while time.monotonic() < deadline:
            try:
                return self.read()
            except FeedbackHealthError as exc:
                last_error = exc
                time.sleep(poll_s)
        raise FeedbackHealthError("Piper feedback did not become healthy: %s" % last_error)
