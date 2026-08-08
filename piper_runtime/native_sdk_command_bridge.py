"""Persistent ROS-to-Piper-SDK command bridge for multi-episode RLT."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import numpy as np

from piper_runtime.hardware_control import PiperCommandSink
from piper_runtime.ros_command_io import joint_state_to_vector


COMMAND_TOPIC = "/rlt/native_sdk_command"
PASSTHROUGH_SERVICE = "/rlt/native_sdk_command_bridge/set_passthrough"
STATUS_TOPIC = "/rlt/native_sdk_command_bridge/status"


class NativeSdkCommandBridgeCore:
    """Resample 30 Hz policy waypoints into a single 50 Hz SDK command stream."""

    def __init__(
        self,
        sink: Any,
        *,
        model_speed_percent: int = 30,
        human_speed_percent: int = 50,
        input_hz: float = 30.0,
        output_hz: float = 50.0,
        stale_timeout_s: float = 0.12,
        now_fn: Any = time.monotonic,
    ):
        self.sink = sink
        self.model_speed_percent = int(model_speed_percent)
        self.human_speed_percent = int(human_speed_percent)
        self.input_hz = float(input_hz)
        self.output_hz = float(output_hz)
        self.stale_timeout_s = float(stale_timeout_s)
        self.now_fn = now_fn
        if self.input_hz <= 0 or self.output_hz <= 0 or self.stale_timeout_s <= 0:
            raise ValueError("bridge rates and stale timeout must be positive")
        self._lock = threading.Lock()
        self.command_count = 0
        self.input_count = 0
        self.last_source: str | None = None
        self._target: np.ndarray | None = None
        self._segment_start: np.ndarray | None = None
        self._last_sent: np.ndarray | None = None
        self._target_received_s: float | None = None
        self._paused = False
        self._closed = False
        self._last_tick_s: float | None = None
        self._last_input_s: float | None = None
        self.repeated_input_count = 0

    def handle_message(self, message: Any) -> None:
        command = joint_state_to_vector(message, action_dim=7)
        source = str(getattr(getattr(message, "header", None), "frame_id", "") or "pi05")
        if source == "human_pika":
            raise ValueError("human Pika commands must use the official ROS pass-through")
        now_s = float(self.now_fn())
        with self._lock:
            if self._closed or self._paused:
                return
            if self._target is not None and np.allclose(
                command,
                self._target,
                rtol=0.0,
                atol=1e-9,
            ):
                self.repeated_input_count += 1
            self._segment_start = command.copy() if self._last_sent is None else self._last_sent.copy()
            self._target = command.copy()
            self._target_received_s = now_s
            self._last_input_s = now_s
            self.input_count += 1
            self.last_source = source

    def tick(self, *, now_s: float | None = None) -> bool:
        current = float(self.now_fn()) if now_s is None else float(now_s)
        with self._lock:
            if self._closed or self._paused or self._target is None or self._target_received_s is None:
                return False
            if current - self._target_received_s > self.stale_timeout_s:
                return False
            start = self._target if self._segment_start is None else self._segment_start
            alpha = float(np.clip((current - self._target_received_s) * self.input_hz, 0.0, 1.0))
            command = ((1.0 - alpha) * start + alpha * self._target).astype(np.float32)
            self.sink.move_speed_percent = self.model_speed_percent
            self.sink.send(command)
            self._last_sent = command.copy()
            self.command_count += 1
            self._last_tick_s = current
            return True

    def telemetry(self, *, now_s: float | None = None) -> dict[str, Any]:
        current = float(self.now_fn()) if now_s is None else float(now_s)
        with self._lock:
            return {
                "output_hz": self.output_hz,
                "input_hz": self.input_hz,
                "command_count": self.command_count,
                "input_count": self.input_count,
                "repeated_input_count": self.repeated_input_count,
                "last_output_age_s": (
                    None
                    if self._last_tick_s is None
                    else max(0.0, current - self._last_tick_s)
                ),
                "last_input_age_s": (
                    None
                    if self._last_input_s is None
                    else max(0.0, current - self._last_input_s)
                ),
                "paused": self._paused,
                "closed": self._closed,
                "source": self.last_source,
            }

    def set_external_passthrough_active(self, active: bool) -> None:
        with self._lock:
            self._paused = bool(active)
            self._target = None
            self._segment_start = None
            self._target_received_s = None
            if active:
                self._last_sent = None

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._paused = True
            self._target = None
            self._segment_start = None
            self._target_received_s = None


def main() -> None:
    import rospy
    from piper_sdk import C_PiperInterface_V2
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from std_srvs.srv import SetBool, SetBoolResponse

    rospy.init_node("rlt_native_sdk_command_bridge", anonymous=False)
    sdk = C_PiperInterface_V2("can0")
    sdk.ConnectPort(False, False, True)
    sink = PiperCommandSink(sdk, move_speed_percent=30)
    sink.configure_motion_mode()
    bridge = NativeSdkCommandBridgeCore(sink)
    rospy.Subscriber(COMMAND_TOPIC, JointState, bridge.handle_message, queue_size=1, tcp_nodelay=True)
    status_publisher = rospy.Publisher(STATUS_TOPIC, String, queue_size=1)

    def tick_and_report(_event: Any) -> None:
        bridge.tick()
        status_publisher.publish(
            String(data=json.dumps(bridge.telemetry(), separators=(",", ":")))
        )

    timer = rospy.Timer(
        rospy.Duration.from_sec(1.0 / bridge.output_hz),
        tick_and_report,
    )

    def set_passthrough(request: Any) -> Any:
        bridge.set_external_passthrough_active(bool(request.data))
        return SetBoolResponse(success=True, message="paused" if request.data else "ready_for_new_target")

    rospy.Service(PASSTHROUGH_SERVICE, SetBool, set_passthrough)
    rospy.loginfo(
        "persistent native SDK command bridge ready: %s (%.0f Hz input -> %.0f Hz output)",
        COMMAND_TOPIC,
        bridge.input_hz,
        bridge.output_hz,
    )

    def disconnect() -> None:
        timer.shutdown()
        bridge.close()
        try:
            sdk.DisconnectPort()
        except Exception:
            pass

    rospy.on_shutdown(disconnect)
    rospy.spin()


if __name__ == "__main__":
    main()
