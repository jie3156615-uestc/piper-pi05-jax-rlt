"""Hardware-capable Piper emitter behind calibration and operator gates."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from piper_delta_runner_v1.runner import DeltaRunner, SafetyProfile


class HardwareAuthorizationError(RuntimeError):
    """Raised when a hardware-capable path is constructed without all gates."""


class PiperSDKEmitter:
    """Emit reviewed absolute joint targets through a Piper SDK-like object."""

    is_hardware = True

    def __init__(
        self,
        *,
        sdk: Any,
        feedback_reader: Callable[[], np.ndarray],
        stop_callback: Callable[[str], None],
        arm_factor: float,
        gripper_factor: float,
    ) -> None:
        self.sdk = sdk
        self.feedback_reader = feedback_reader
        self.stop_callback = stop_callback
        self.arm_factor = float(arm_factor)
        self.gripper_factor = float(gripper_factor)

    def emit(self, value: np.ndarray, *, timestamp: float) -> np.ndarray:
        del timestamp
        target = np.asarray(value, dtype=float)
        if target.shape != (7,):
            raise ValueError(f"absolute target must have shape (7,), got {target.shape}")
        joints = tuple(round(float(value) * self.arm_factor) for value in target[:6])
        gripper = round(abs(float(target[6])) * self.gripper_factor)
        self.sdk.JointCtrl(*joints)
        self.sdk.GripperCtrl(gripper, 1000, 0x01, 0)
        return np.asarray(self.feedback_reader(), dtype=float)

    def stop(self, reason: str) -> None:
        self.stop_callback(reason)


class HardwareDeltaRunner(DeltaRunner):
    """Use the tested delta safety pipeline only after explicit hardware gates."""

    def __init__(
        self,
        safety_profile: SafetyProfile,
        emitter: PiperSDKEmitter,
        *,
        authorization_token: str | None,
    ) -> None:
        if authorization_token != "YES":
            raise HardwareAuthorizationError("Hardware execution requires explicit authorization_token='YES'")
        if safety_profile.mode != "hardware_calibrated":
            raise HardwareAuthorizationError("Hardware execution requires mode='hardware_calibrated'")
        if not safety_profile.allow_hardware_execution:
            raise HardwareAuthorizationError("Hardware execution is disabled by the safety profile")
        if safety_profile.calibration_status != "reviewed_and_approved":
            raise HardwareAuthorizationError("Hardware execution requires reviewed_and_approved calibration")
        if not getattr(emitter, "is_hardware", False):
            raise HardwareAuthorizationError("Hardware runner requires a hardware emitter")
        self.safety_profile = safety_profile
        self.emitter = emitter
