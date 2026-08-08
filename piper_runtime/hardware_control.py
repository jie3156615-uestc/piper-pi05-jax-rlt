"""Stateful safety filtering and Piper SDK command emission."""

import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np


ARM_FACTOR = 57295.7795
GRIPPER_FACTOR = 1000000.0


class HardwareSafetyError(RuntimeError):
    pass


@dataclass(frozen=True)
class HardwareSafetyConfig:
    joint_min: np.ndarray = field(
        default_factory=lambda: np.radians([-150.0, 0.0, -170.0, -100.0, -70.0, -180.0])
    )
    joint_max: np.ndarray = field(
        default_factory=lambda: np.radians([150.0, 180.0, 0.0, 100.0, 70.0, 180.0])
    )
    hard_limit_overrun: float = math.radians(5.0)
    max_plan_delta: float = math.radians(8.0)
    max_velocity: float = 0.03
    max_acceleration: float = 0.2
    max_tracking_error: float = math.radians(5.0)
    gripper_min: float = 0.0
    gripper_max: float = 0.08
    # Pika's serial gripper source can report about 0.098 m when fully open.
    # Piper's executable command path clamps to 0.08 m, so accept the native
    # demonstrator range here and clip to gripper_max before publishing.
    gripper_hard_overrun: float = 0.02
    gripper_max_step: float = 0.002
    piper_move_speed_percent: int = 10
    # Only autonomous model commands use these continuity limits. Pika human
    # takeover stays on the unattenuated native path.
    model_smoothing_tau_s: float = 0.05
    model_max_joint_step: float = math.radians(3.0)
    model_max_gripper_step: float = 0.02


def make_hardware_safety_config(profile: str = "probe") -> HardwareSafetyConfig:
    if profile == "probe":
        return HardwareSafetyConfig()
    if profile == "normal":
        return HardwareSafetyConfig(
            max_plan_delta=math.radians(20.0),
            max_velocity=0.6,
            max_acceleration=6.0,
            gripper_max_step=0.008,
            piper_move_speed_percent=30,
        )
    if profile == "native":
        return HardwareSafetyConfig(
            piper_move_speed_percent=30,
            gripper_max_step=0.08,
        )
    raise ValueError(f"unknown hardware safety profile: {profile}")


@dataclass
class FilterResult:
    command: np.ndarray
    reasons: list


class StatefulSafetyFilter:
    def __init__(self, config: HardwareSafetyConfig, initial_state):
        self.config = config
        self.previous_command = np.asarray(initial_state, dtype=float).copy()
        if self.previous_command.shape != (7,):
            raise ValueError("initial state must have shape (7,)")
        self.previous_velocity = np.zeros(6, dtype=float)

    def filter(self, raw_target, *, snapshot, feedback, dt: float) -> FilterResult:
        target = np.asarray(raw_target, dtype=float).copy()
        snapshot = np.asarray(snapshot, dtype=float)
        feedback = np.asarray(feedback, dtype=float)
        if target.shape != (7,) or snapshot.shape != (7,) or feedback.shape != (7,):
            raise HardwareSafetyError("target, snapshot, and feedback must have shape (7,)")
        if not np.all(np.isfinite(target)) or not np.all(np.isfinite(snapshot)) or not np.all(np.isfinite(feedback)):
            raise HardwareSafetyError("non-finite control data")
        if dt <= 0:
            raise HardwareSafetyError("control period must be positive")
        reasons = []
        tracking_error = np.abs(feedback[:6] - self.previous_command[:6])
        if np.any(tracking_error > self.config.max_tracking_error):
            holding_feedback = np.allclose(target, feedback, atol=1e-6, rtol=0.0) and np.allclose(snapshot, feedback, atol=1e-6, rtol=0.0)
            if not holding_feedback:
                raise HardwareSafetyError("joint tracking error exceeded 5 degrees")
            self.previous_command = feedback.copy()
            self.previous_velocity = np.zeros(6, dtype=float)
            reasons.append("tracking_resync")

        below = self.config.joint_min - target[:6]
        above = target[:6] - self.config.joint_max
        if np.any(below > self.config.hard_limit_overrun) or np.any(above > self.config.hard_limit_overrun):
            raise HardwareSafetyError("policy target exceeded official limit by more than 5 degrees")
        clipped_limits = np.clip(target[:6], self.config.joint_min, self.config.joint_max)
        if not np.allclose(clipped_limits, target[:6]):
            reasons.append("joint_limit_clip")
        target[:6] = clipped_limits

        plan_delta = target[:6] - snapshot[:6]
        clipped_plan_delta = np.clip(plan_delta, -self.config.max_plan_delta, self.config.max_plan_delta)
        if not np.allclose(clipped_plan_delta, plan_delta):
            reasons.append("plan_delta_clip")
        target[:6] = np.clip(snapshot[:6] + clipped_plan_delta, self.config.joint_min, self.config.joint_max)

        desired_velocity = (target[:6] - self.previous_command[:6]) / dt
        limited_velocity = np.clip(desired_velocity, -self.config.max_velocity, self.config.max_velocity)
        if not np.allclose(limited_velocity, desired_velocity):
            reasons.append("max_velocity")
        dv = self.config.max_acceleration * dt
        accelerated_velocity = np.clip(
            limited_velocity,
            self.previous_velocity - dv,
            self.previous_velocity + dv,
        )
        if not np.allclose(accelerated_velocity, limited_velocity):
            reasons.append("max_acceleration")
        command = self.previous_command.copy()
        command[:6] = np.clip(
            self.previous_command[:6] + accelerated_velocity * dt,
            self.config.joint_min,
            self.config.joint_max,
        )

        if target[6] < self.config.gripper_min - self.config.gripper_hard_overrun or target[6] > self.config.gripper_max + self.config.gripper_hard_overrun:
            raise HardwareSafetyError("gripper target exceeded the hard range")
        gripper_target = float(np.clip(target[6], self.config.gripper_min, self.config.gripper_max))
        gripper_delta = np.clip(
            gripper_target - self.previous_command[6],
            -self.config.gripper_max_step,
            self.config.gripper_max_step,
        )
        if not math.isclose(gripper_delta, gripper_target - self.previous_command[6], abs_tol=1e-9):
            reasons.append("gripper_step_clip")
        command[6] = self.previous_command[6] + gripper_delta

        self.previous_command = command.copy()
        self.previous_velocity = accelerated_velocity.copy()
        return FilterResult(command=command, reasons=reasons)

    def filter_native(self, raw_target) -> FilterResult:
        target = np.asarray(raw_target, dtype=float).copy()
        if target.shape != (7,):
            raise HardwareSafetyError("native target must have shape (7,)")
        if not np.all(np.isfinite(target)):
            raise HardwareSafetyError("non-finite native control data")

        below = self.config.joint_min - target[:6]
        above = target[:6] - self.config.joint_max
        if np.any(below > self.config.hard_limit_overrun) or np.any(above > self.config.hard_limit_overrun):
            raise HardwareSafetyError("native target exceeded official limit by more than 5 degrees")
        if target[6] < self.config.gripper_min - self.config.gripper_hard_overrun or target[6] > self.config.gripper_max + self.config.gripper_hard_overrun:
            raise HardwareSafetyError("native gripper target exceeded the hard range")

        self.previous_command = target.copy()
        self.previous_velocity = np.zeros(6, dtype=float)
        return FilterResult(command=target, reasons=[])

    def filter_model_native(self, raw_target, *, dt: float = 1.0 / 30.0) -> FilterResult:
        target = np.asarray(raw_target, dtype=float).copy()
        if target.shape != (7,):
            raise HardwareSafetyError("model native target must have shape (7,)")
        if not np.all(np.isfinite(target)):
            raise HardwareSafetyError("non-finite model native control data")
        if not math.isfinite(dt) or dt <= 0:
            raise HardwareSafetyError("model control period must be positive")

        reasons = []
        clipped_joints = np.clip(target[:6], self.config.joint_min, self.config.joint_max)
        if not np.allclose(clipped_joints, target[:6]):
            reasons.append("model_native_joint_range_clip")
        target[:6] = clipped_joints

        gripper_target = float(np.clip(target[6], self.config.gripper_min, self.config.gripper_max))
        if not math.isclose(gripper_target, float(target[6]), abs_tol=1e-9):
            reasons.append("model_native_gripper_range_clip")
        target[6] = gripper_target

        tau_s = float(self.config.model_smoothing_tau_s)
        if not math.isfinite(tau_s) or tau_s < 0:
            raise HardwareSafetyError("model smoothing tau must be finite and non-negative")
        alpha = 1.0 if tau_s == 0 else 1.0 - math.exp(-dt / tau_s)
        smoothed = target.copy()
        smoothed[:6] = self.previous_command[:6] + alpha * (target[:6] - self.previous_command[:6])
        if not np.allclose(smoothed[:6], target[:6], atol=1e-9, rtol=0.0):
            reasons.append("model_low_pass")

        max_joint_step = float(self.config.model_max_joint_step)
        max_gripper_step = float(self.config.model_max_gripper_step)
        if not math.isfinite(max_joint_step) or max_joint_step <= 0:
            raise HardwareSafetyError("model max joint step must be finite and positive")
        if not math.isfinite(max_gripper_step) or max_gripper_step <= 0:
            raise HardwareSafetyError("model max gripper step must be finite and positive")

        desired_joint_delta = smoothed[:6] - self.previous_command[:6]
        joint_delta = np.clip(desired_joint_delta, -max_joint_step, max_joint_step)
        if not np.allclose(joint_delta, desired_joint_delta, atol=1e-9, rtol=0.0):
            reasons.append("model_joint_step_clip")
        desired_gripper_delta = float(smoothed[6] - self.previous_command[6])
        gripper_delta = float(np.clip(desired_gripper_delta, -max_gripper_step, max_gripper_step))
        if not math.isclose(gripper_delta, desired_gripper_delta, abs_tol=1e-9):
            reasons.append("model_gripper_step_clip")

        command = self.previous_command.copy()
        command[:6] += joint_delta
        command[6] += gripper_delta
        command[:6] = np.clip(command[:6], self.config.joint_min, self.config.joint_max)
        command[6] = float(np.clip(command[6], self.config.gripper_min, self.config.gripper_max))

        self.previous_command = command.copy()
        self.previous_velocity = np.zeros(6, dtype=float)
        return FilterResult(command=command, reasons=reasons)

    def filter_human_native(self, raw_target) -> FilterResult:
        target = np.asarray(raw_target, dtype=float).copy()
        if target.shape != (7,):
            raise HardwareSafetyError("human target must have shape (7,)")
        if not np.all(np.isfinite(target)):
            raise HardwareSafetyError("non-finite human control data")

        below = self.config.joint_min - target[:6]
        above = target[:6] - self.config.joint_max
        if np.any(below > self.config.hard_limit_overrun) or np.any(above > self.config.hard_limit_overrun):
            raise HardwareSafetyError("human target exceeded official limit by more than 5 degrees")

        reasons = []
        clipped_joints = np.clip(target[:6], self.config.joint_min, self.config.joint_max)
        if not np.allclose(clipped_joints, target[:6]):
            reasons.append("joint_limit_clip")
        target[:6] = clipped_joints

        if target[6] < self.config.gripper_min - self.config.gripper_hard_overrun or target[6] > self.config.gripper_max + self.config.gripper_hard_overrun:
            raise HardwareSafetyError("human gripper target exceeded the hard range")
        gripper_target = float(np.clip(target[6], self.config.gripper_min, self.config.gripper_max))
        if not math.isclose(gripper_target, float(target[6]), abs_tol=1e-9):
            reasons.append("gripper_range_clip")
        target[6] = gripper_target

        self.previous_command = target.copy()
        self.previous_velocity = np.zeros(6, dtype=float)
        return FilterResult(command=target, reasons=reasons)


class PiperCommandSink:
    def __init__(self, sdk, move_speed_percent: int = 10):
        if move_speed_percent < 1 or move_speed_percent > 100:
            raise ValueError("move_speed_percent must be in [1, 100]")
        self.sdk = sdk
        self.move_speed_percent = int(move_speed_percent)
        self._configured_speed_percent: int | None = None

    def configure_motion_mode(self, *, force: bool = False) -> None:
        """Enter CAN MOVE_J once, matching the stable LeRobot Piper driver."""
        if force or self._configured_speed_percent != self.move_speed_percent:
            self.sdk.MotionCtrl_2(0x01, 0x01, self.move_speed_percent, 0x00)
            self._configured_speed_percent = self.move_speed_percent

    def send(self, command) -> None:
        command = np.asarray(command, dtype=float)
        if command.shape != (7,) or not np.all(np.isfinite(command)):
            raise HardwareSafetyError("invalid Piper command")
        joints = tuple(round(float(value) * ARM_FACTOR) for value in command[:6])
        gripper = round(float(command[6]) * GRIPPER_FACTOR)
        self.sdk.JointCtrl(*joints)
        self.sdk.GripperCtrl(gripper, 1000, 0x01, 0)

    def hold(self, feedback_reader, duration_s: float = 1.0, hz: float = 30.0) -> np.ndarray:
        state = np.asarray(feedback_reader.read(), dtype=float)
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            self.send(state)
            time.sleep(1.0 / hz)
        return state


class InterpolatedPiperCommandSink:
    """Causally resample 30 Hz absolute waypoints onto one 50 Hz SDK thread."""

    def __init__(
        self,
        sink: PiperCommandSink,
        *,
        input_hz: float = 30.0,
        output_hz: float = 50.0,
        stale_timeout_s: float = 0.15,
        now_fn=time.monotonic,
    ):
        if input_hz <= 0 or output_hz <= 0 or stale_timeout_s <= 0:
            raise ValueError("interpolation rates and stale timeout must be positive")
        self.sink = sink
        self.input_hz = float(input_hz)
        self.output_hz = float(output_hz)
        self.stale_timeout_s = float(stale_timeout_s)
        self.now_fn = now_fn
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._target = None
        self._segment_start = None
        self._last_sent = None
        self._target_received_s = None
        self._error = None
        self.input_count = 0
        self.output_count = 0
        self._thread = threading.Thread(target=self._run, name="piper-model-50hz", daemon=True)
        self._thread.start()

    def send(self, command) -> None:
        value = np.asarray(command, dtype=float)
        if value.shape != (7,) or not np.all(np.isfinite(value)):
            raise HardwareSafetyError("invalid interpolated Piper command")
        with self._lock:
            if self._error is not None:
                raise HardwareSafetyError(f"50 Hz Piper publisher failed: {self._error}")
            if self._stop.is_set():
                raise HardwareSafetyError("50 Hz Piper publisher is closed")
            self._segment_start = value.copy() if self._last_sent is None else self._last_sent.copy()
            self._target = value.copy()
            self._target_received_s = float(self.now_fn())
            self.input_count += 1

    def _emit(self, now_s: float) -> bool:
        with self._lock:
            if self._target is None or self._target_received_s is None:
                return False
            if now_s - self._target_received_s > self.stale_timeout_s:
                return False
            start = self._target if self._segment_start is None else self._segment_start
            alpha = float(np.clip((now_s - self._target_received_s) * self.input_hz, 0.0, 1.0))
            command = ((1.0 - alpha) * start + alpha * self._target).copy()
        try:
            self.sink.send(command)
        except Exception as exc:
            with self._lock:
                self._error = exc
            self._stop.set()
            return False
        with self._lock:
            self._last_sent = command
            self.output_count += 1
        return True

    def _run(self) -> None:
        period_s = 1.0 / self.output_hz
        next_deadline = float(self.now_fn())
        while not self._stop.is_set():
            now_s = float(self.now_fn())
            wait_s = next_deadline - now_s
            if wait_s > 0:
                self._stop.wait(wait_s)
                continue
            self._emit(now_s)
            # Skip expired slots instead of sending a catch-up burst.
            missed = max(1, int(math.floor((now_s - next_deadline) / period_s)) + 1)
            next_deadline += missed * period_s

    def hold(self, feedback_reader, duration_s: float = 1.0, hz: float = 30.0) -> np.ndarray:
        del hz
        state = np.asarray(feedback_reader.read(), dtype=float)
        self.send(state)
        self._stop.wait(max(0.0, float(duration_s)))
        return state

    def close(self, timeout_s: float = 1.0) -> None:
        self._stop.set()
        self._thread.join(timeout=max(0.0, float(timeout_s)))
