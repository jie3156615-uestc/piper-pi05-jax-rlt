from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from piper_runtime.native_sdk_command_bridge import NativeSdkCommandBridgeCore


class FakeSink:
    def __init__(self):
        self.move_speed_percent = 0
        self.sent = []

    def send(self, command):
        self.sent.append((self.move_speed_percent, np.asarray(command).copy()))


def message(value: float, source: str):
    return SimpleNamespace(
        position=[value] * 7,
        header=SimpleNamespace(frame_id=source),
    )


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


def test_bridge_resamples_model_waypoints_at_explicit_ticks():
    sink = FakeSink()
    clock = FakeClock()
    bridge = NativeSdkCommandBridgeCore(sink, now_fn=clock)

    bridge.handle_message(message(1.0, "pi05"))
    assert sink.sent == []
    assert bridge.tick(now_s=0.0)

    clock.value = 1 / 30
    bridge.handle_message(message(2.0, "rlt"))
    assert bridge.tick(now_s=clock.value + 0.02)
    assert bridge.tick(now_s=clock.value + 1 / 30)

    assert [speed for speed, _command in sink.sent] == [30, 30, 30]
    np.testing.assert_allclose(sink.sent[0][1], np.ones(7))
    np.testing.assert_allclose(sink.sent[1][1], np.full(7, 1.6), atol=1e-6)
    np.testing.assert_allclose(sink.sent[2][1], np.full(7, 2.0))
    assert bridge.command_count == 3
    assert bridge.input_count == 2
    assert bridge.last_source == "rlt"


def test_bridge_pause_is_a_synchronous_barrier_and_drops_old_target():
    sink = FakeSink()
    clock = FakeClock()
    bridge = NativeSdkCommandBridgeCore(sink, now_fn=clock)
    bridge.handle_message(message(1.0, "pi05"))
    assert bridge.tick(now_s=0.0)

    bridge.set_external_passthrough_active(True)
    assert not bridge.tick(now_s=0.02)
    bridge.set_external_passthrough_active(False)
    assert not bridge.tick(now_s=0.04)

    clock.value = 0.05
    bridge.handle_message(message(2.0, "pi05"))
    assert bridge.tick(now_s=0.05)
    np.testing.assert_allclose(sink.sent[-1][1], np.full(7, 2.0))


def test_bridge_stale_watchdog_and_close_fail_closed():
    sink = FakeSink()
    bridge = NativeSdkCommandBridgeCore(sink, now_fn=lambda: 0.0, stale_timeout_s=0.1)
    bridge.handle_message(message(1.0, "pi05"))
    assert not bridge.tick(now_s=0.11)
    bridge.close()
    assert not bridge.tick(now_s=0.0)


def test_bridge_emits_verifiable_50hz_keepalive_telemetry():
    sink = FakeSink()
    clock = FakeClock()
    bridge = NativeSdkCommandBridgeCore(
        sink,
        now_fn=clock,
        input_hz=30.0,
        output_hz=50.0,
        stale_timeout_s=0.12,
    )
    bridge.handle_message(message(1.0, "pi05"))

    for slot in range(6):
        clock.value = slot / 50.0
        assert bridge.tick(now_s=clock.value)

    # A repeated 30 Hz boundary keepalive refreshes the watchdog while the
    # physical publisher continues on its independent 50 Hz timer.
    clock.value = 0.10
    bridge.handle_message(message(1.0, "pi05"))
    assert bridge.tick(now_s=clock.value)
    telemetry = bridge.telemetry(now_s=clock.value)

    assert telemetry["output_hz"] == 50.0
    assert telemetry["input_hz"] == 30.0
    assert telemetry["command_count"] == 7
    assert telemetry["input_count"] == 2
    assert telemetry["repeated_input_count"] == 1
    assert telemetry["last_output_age_s"] == 0.0
    assert len(sink.sent) == 7
