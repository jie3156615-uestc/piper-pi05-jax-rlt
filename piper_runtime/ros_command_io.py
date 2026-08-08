from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from piper_runtime.rlt_command_mux import TimedCommand


def joint_state_to_vector(message: Any, *, action_dim: int = 7) -> np.ndarray:
    try:
        position = getattr(message, "position")
    except AttributeError as exc:
        raise ValueError("JointState-like message must expose a position field") from exc
    try:
        vector = np.asarray(position, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("JointState position must be numeric") from exc
    if vector.shape != (action_dim,):
        raise ValueError(f"JointState position must have shape ({action_dim},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("JointState position must contain finite values")
    return vector.copy()


def vector_to_joint_state(
    vector: Any,
    *,
    message_factory: Callable[[], Any] | None = None,
    joint_names: Sequence[str] | None = None,
    action_dim: int = 7,
) -> Any:
    value = np.asarray(vector, dtype=np.float32)
    if value.shape != (action_dim,):
        raise ValueError(f"command vector must have shape ({action_dim},), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("command vector must contain finite values")
    if message_factory is None:
        from sensor_msgs.msg import JointState  # type: ignore

        message_factory = JointState
    message = message_factory()
    message.position = value.astype(float).tolist()
    if joint_names is not None:
        message.name = list(joint_names)
    return message


@dataclasses.dataclass
class FreshJointCommandTracker:
    action_dim: int = 7
    freshness_s: float = 0.1

    def __post_init__(self) -> None:
        self._latest: TimedCommand | None = None

    def update(self, message_or_vector: Any, *, timestamp_s: float | None = None) -> TimedCommand:
        if hasattr(message_or_vector, "position"):
            value = joint_state_to_vector(message_or_vector, action_dim=self.action_dim)
        else:
            value = np.asarray(message_or_vector, dtype=np.float32)
            if value.shape != (self.action_dim,) or not np.all(np.isfinite(value)):
                raise ValueError(f"command vector must be finite with shape ({self.action_dim},)")
        command = TimedCommand(
            value=value.copy(),
            timestamp_s=time.monotonic() if timestamp_s is None else float(timestamp_s),
        )
        self._latest = command
        return command

    def latest(self, *, now_s: float | None = None) -> TimedCommand | None:
        if self._latest is None or self._latest.timestamp_s is None:
            return None
        current = time.monotonic() if now_s is None else float(now_s)
        if current - self._latest.timestamp_s > self.freshness_s:
            return None
        return TimedCommand(
            value=None if self._latest.value is None else self._latest.value.copy(),
            timestamp_s=self._latest.timestamp_s,
        )


def require_ros_modules() -> tuple[Any, Any]:
    try:
        import rospy  # type: ignore
        from sensor_msgs.msg import JointState  # type: ignore
    except ImportError as exc:
        raise RuntimeError("ROS Noetic Python modules are required for live takeover rollout") from exc
    return rospy, JointState
