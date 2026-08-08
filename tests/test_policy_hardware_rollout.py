import numpy as np
import pytest
import time
import concurrent.futures
import threading

from piper_runtime.buffered_policy_control import H50HandoffRejected, prepare_h50_handoff_plan
from piper_runtime.hardware_control import StatefulSafetyFilter, make_hardware_safety_config

from piper_runtime.policy_hardware_rollout import (
    AUTHORIZATION,
    INFERENCE_START_RESET_HZ,
    INFERENCE_START_TARGET,
    WINDOWED_C10_MODE,
    RolloutConfig,
    RolloutConfigurationError,
    _committed_action_count,
    _await_policy_future_with_keepalive,
    _reset_ros_controller_to_inference_start,
    make_stop_handler,
    run_rollout,
)


def test_rollout_requires_explicit_authorization_and_bounded_duration():
    with pytest.raises(RolloutConfigurationError):
        RolloutConfig(duration_s=10, authorization="wrong").validate()
    with pytest.raises(RolloutConfigurationError):
        RolloutConfig(duration_s=601, authorization=AUTHORIZATION).validate()
    with pytest.raises(RolloutConfigurationError):
        RolloutConfig(duration_s=None, authorization=AUTHORIZATION).validate()


def test_rollout_allows_operator_label_control_without_duration():
    config = RolloutConfig(
        duration_s=None,
        authorization=AUTHORIZATION,
        operator_label_control=True,
    ).validate()

    assert config.duration_s is None
    assert config.operator_label_control is True


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += max(0.0, seconds)
        time.sleep(0)


class FakeCameras:
    def __init__(self):
        self.calls = []

    def read(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "camera1": np.full((480, 640, 3), 10, dtype=np.uint8),
            "camera2": np.full((480, 640, 3), 20, dtype=np.uint8),
        }


class FakeFeedback:
    def read(self):
        return np.zeros(7, dtype=np.float32)


class FakeSink:
    def __init__(self):
        self.sent = []
        self.hold_calls = 0

    def send(self, command):
        self.sent.append(np.asarray(command))

    def hold(self, feedback_reader):
        self.hold_calls += 1
        return feedback_reader.read()


class FakePolicy:
    def __init__(self, sink, fail=False, actions=None):
        self.sink = sink
        self.fail = fail
        self.actions = actions
        self.send_counts_at_infer = []

    def infer(self, observation):
        self.send_counts_at_infer.append(len(self.sink.sent))
        if self.fail:
            raise RuntimeError("policy failed")
        if self.actions is not None:
            return {"actions": np.asarray(self.actions, dtype=np.float32)}
        return {"actions": np.zeros((50, 7), dtype=np.float32)}


class FakeOperatorLabelSource:
    def __init__(self, values):
        self.values = list(values)

    def __call__(self):
        if not self.values:
            return None
        return self.values.pop(0)


def test_rollout_replans_after_each_command_by_default_and_holds():
    clock, sink = FakeClock(), FakeSink()
    policy = FakePolicy(sink)
    cameras = FakeCameras()
    streamed = []
    result = run_rollout(
        RolloutConfig(duration_s=0.2, authorization=AUTHORIZATION),
        cameras=cameras,
        policy=policy,
        feedback_reader=FakeFeedback(),
        sink=sink,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        on_record=streamed.append,
    )
    assert policy.send_counts_at_infer[:2] == [0, 1]
    assert sink.hold_calls == 1
    assert result["step_count"] >= 5
    assert len(streamed) == result["step_count"]
    assert cameras.calls[0]["warmup_frames"] == 60


def test_policy_failure_triggers_hold():
    clock, sink = FakeClock(), FakeSink()
    with pytest.raises(RuntimeError, match="policy failed"):
        run_rollout(
            RolloutConfig(duration_s=0.2, authorization=AUTHORIZATION),
            cameras=FakeCameras(),
            policy=FakePolicy(sink, fail=True),
            feedback_reader=FakeFeedback(),
            sink=sink,
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
    assert sink.hold_calls == 1



class FailingHoldSink(FakeSink):
    def hold(self, feedback_reader):
        self.hold_calls += 1
        raise RuntimeError("feedback unhealthy during hold")


def test_feedback_error_during_hold_does_not_mask_original_exception():
    clock, sink = FakeClock(), FailingHoldSink()
    with pytest.raises(RuntimeError, match="policy failed"):
        run_rollout(
            RolloutConfig(duration_s=0.2, authorization=AUTHORIZATION),
            cameras=FakeCameras(),
            policy=FakePolicy(sink, fail=True),
            feedback_reader=FakeFeedback(),
            sink=sink,
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
    assert sink.hold_calls == 1

def test_stop_handler_turns_process_signal_into_controlled_interrupt():
    with pytest.raises(KeyboardInterrupt):
        make_stop_handler()(15, None)


def test_rollout_rejects_unknown_safety_profile():
    with pytest.raises(RolloutConfigurationError):
        RolloutConfig(duration_s=1, authorization=AUTHORIZATION, safety_profile="raw").validate()


def test_rollout_allows_full_native_chunk_execution():
    config = RolloutConfig(
        duration_s=1,
        authorization=AUTHORIZATION,
        safety_profile="native",
        execute_steps=50,
    ).validate()

    assert config.execute_steps == 50
    assert config.safety_profile == "native"


def test_rollout_can_stop_after_one_policy_plan():
    clock, sink = FakeClock(), FakeSink()
    policy = FakePolicy(sink)
    result = run_rollout(
        RolloutConfig(
            duration_s=10.0,
            authorization=AUTHORIZATION,
            safety_profile="native",
            execute_steps=50,
            max_plans=1,
        ),
        cameras=FakeCameras(),
        policy=policy,
        feedback_reader=FakeFeedback(),
        sink=sink,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
    )

    assert result["outcome"] == "max_plans_complete"
    assert result["plan_count"] == 1
    assert result["step_count"] == 50
    assert policy.send_counts_at_infer == [0]


def test_rollout_operator_label_ends_episode_and_holds():
    clock, sink = FakeClock(), FakeSink()
    policy = FakePolicy(sink)
    result = run_rollout(
        RolloutConfig(
            duration_s=None,
            authorization=AUTHORIZATION,
            safety_profile="native",
            execute_steps=50,
            operator_label_control=True,
        ),
        cameras=FakeCameras(),
        policy=policy,
        feedback_reader=FakeFeedback(),
        sink=sink,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        operator_label_source=FakeOperatorLabelSource([None, None, "1"]),
    )

    assert result["outcome"] == "operator_labeled"
    assert result["operator_label"] == "1"
    assert result["step_count"] == 3
    assert sink.hold_calls == 1


def test_h10_is_a_c10_window_over_one_committed_h50_policy_chunk():
    assert _committed_action_count(10) == 50
    assert _committed_action_count(50) == 50
    assert _committed_action_count(5) == 5

    clock, sink = FakeClock(), FakeSink()
    actions = np.zeros((50, 7), dtype=np.float32)
    actions[:, 0] = np.linspace(0.0, 0.2, 50)
    actions[:, 6] = np.linspace(0.065, 0.01, 50)
    policy = FakePolicy(sink, actions=actions)
    streamed = []
    result = run_rollout(
        RolloutConfig(
            duration_s=1.8,
            authorization=AUTHORIZATION,
            safety_profile="native",
            execute_steps=10,
        ),
        cameras=FakeCameras(),
        policy=policy,
        feedback_reader=FakeFeedback(),
        sink=sink,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        on_record=streamed.append,
    )

    # The C10 compatibility mode remains sequential until the same semantics
    # are explicitly migrated into RLT.
    assert policy.send_counts_at_infer[:2] == [0, 50]
    assert result["execution_mode"] == WINDOWED_C10_MODE
    first_chunk = streamed[:50]
    assert [row["c10_window_index"] for row in first_chunk] == [index // 10 for index in range(50)]
    assert [row["c10_window_offset"] for row in first_chunk] == [index % 10 for index in range(50)]
    window_rows = [row for row in first_chunk if row["c10_window_offset"] == 0]
    assert len(window_rows) == 5
    for index, row in enumerate(window_rows):
        np.testing.assert_allclose(row["a_ref_c10"], actions[index * 10 : (index + 1) * 10])


def test_prefetched_h50_is_time_aligned_and_locally_bridged_without_global_rebase():
    raw = np.zeros((50, 7), dtype=np.float64)
    raw[:, :6] = (0.1 + np.linspace(0.0, 0.2, 50))[:, None]
    raw[:, 6] = np.linspace(0.065, 0.01, 50)
    observation = np.full(7, 0.1)
    observation[6] = 0.065
    handoff = observation.copy()
    handoff[:6] = raw[4, :6] + 0.01
    previous = handoff.copy()
    previous[:6] -= 0.002

    prepared = prepare_h50_handoff_plan(
        raw,
        observation_state=observation,
        handoff_target=handoff,
        previous_target=previous,
        observation_age_s=5.0 / 30.0,
    )

    np.testing.assert_allclose(prepared.joint_rebase_offset, np.zeros(6))
    assert prepared.action_start_index == 5
    np.testing.assert_allclose(prepared.actions[5:, :6], raw[10:, :6])
    np.testing.assert_allclose(prepared.actions[:, 6], raw[5:, 6])
    assert prepared.bridge_steps == 5
    np.testing.assert_allclose(prepared.actions[4, :6], raw[9, :6])
    assert prepared.bridge_target_correction_rad > 0.0


def test_velocity_aware_bridge_avoids_a_boundary_stop_after_native_low_pass():
    raw = np.zeros((50, 7), dtype=np.float64)
    raw[:, :6] = np.linspace(0.03, 0.2, 50)[:, None]
    handoff = np.full(7, 0.02)
    handoff[6] = 0.065
    previous = handoff.copy()
    previous[:6] -= 0.004

    prepared = prepare_h50_handoff_plan(
        raw,
        observation_state=np.zeros(7),
        handoff_target=handoff,
        previous_target=previous,
        observation_age_s=0.0,
    )
    safety = StatefulSafetyFilter(make_hardware_safety_config("native"), handoff)
    first = safety.filter_model_native(prepared.actions[0]).command
    previous_speed = np.linalg.norm(handoff[:6] - previous[:6])
    boundary_speed = np.linalg.norm(first[:6] - handoff[:6])

    assert boundary_speed >= 0.5 * previous_speed
    np.testing.assert_allclose(prepared.actions[5:, :6], raw[5:, :6])


def test_velocity_bridge_correction_is_clipped_without_rejecting_the_plan():
    raw = np.zeros((50, 7), dtype=np.float64)
    handoff = np.zeros(7, dtype=np.float64)
    previous = handoff.copy()
    previous[:6] = -0.08

    prepared = prepare_h50_handoff_plan(
        raw,
        observation_state=handoff,
        handoff_target=handoff,
        previous_target=previous,
        observation_age_s=0.0,
    )

    assert prepared.bridge_target_correction_clipped is True
    assert prepared.bridge_target_correction_rad == pytest.approx(0.12)


def test_prefetched_h50_rejects_only_an_implausibly_large_local_bridge():
    raw = np.zeros((50, 7), dtype=np.float64)
    observation = np.zeros(7, dtype=np.float64)
    handoff = np.zeros(7, dtype=np.float64)
    handoff[1] = 0.2
    with pytest.raises(H50HandoffRejected, match="boundary bridge excursion"):
        prepare_h50_handoff_plan(
            raw,
            observation_state=observation,
            handoff_target=handoff,
            previous_target=observation,
            observation_age_s=5.0 / 30.0,
        )


def test_large_observation_motion_is_not_replayed_when_absolute_suffix_is_aligned():
    raw = np.zeros((50, 7), dtype=np.float64)
    raw[:, 1] = 0.2
    observation = np.zeros(7, dtype=np.float64)
    handoff = np.zeros(7, dtype=np.float64)
    handoff[1] = 0.2

    prepared = prepare_h50_handoff_plan(
        raw,
        observation_state=observation,
        handoff_target=handoff,
        previous_target=handoff,
        observation_age_s=5.0 / 30.0,
    )

    np.testing.assert_allclose(prepared.actions[:, 1], 0.2)
    np.testing.assert_allclose(prepared.joint_rebase_offset, 0.0)


def test_policy_wait_republishes_last_target_until_future_is_ready():
    sink = FakeSink()
    future = concurrent.futures.Future()
    timer = threading.Timer(0.005, future.set_result, args=("ready",))
    timer.start()

    try:
        result, keepalives = _await_policy_future_with_keepalive(
            future,
            sink=sink,
            hold_target=np.arange(7, dtype=float),
            timeout_s=1.0,
        )
    finally:
        timer.cancel()

    assert result == "ready"
    assert keepalives == 1
    np.testing.assert_allclose(sink.sent, [np.arange(7, dtype=float)])


def test_h50_prefetch_stages_next_plan_without_a_boundary_command_gap():
    clock, sink = FakeClock(), FakeSink()
    actions = np.zeros((50, 7), dtype=np.float32)
    actions[:, 0] = np.linspace(0.0, 0.015, 50)
    policy = FakePolicy(sink, actions=actions)
    streamed = []

    result = run_rollout(
        RolloutConfig(
            duration_s=1.8,
            authorization=AUTHORIZATION,
            safety_profile="native",
            execute_steps=50,
            h50_prefetch_lead_steps=5,
        ),
        cameras=FakeCameras(),
        policy=policy,
        feedback_reader=FakeFeedback(),
        sink=sink,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        on_record=streamed.append,
    )

    assert result["step_count"] > 50
    assert [row["action_index"] for row in streamed[:52]] == list(range(50)) + [0, 1]
    assert any(row["prefetch_requested_this_step"] for row in streamed[:50])
    assert streamed[50]["plan_boundary_wait_s"] == pytest.approx(0.0)
    assert streamed[50]["plan_prefetch_accepted"] is True
    assert np.max(np.abs(sink.sent[50][:6] - sink.sent[49][:6])) < 0.03


def test_rejected_prefetch_brakes_at_feedback_and_resets_fresh_plan_anchor():
    clock, sink = FakeClock(), FakeSink()
    forward = np.zeros((50, 7), dtype=np.float32)
    forward[:, :6] = 0.1
    conflicting = np.zeros((50, 7), dtype=np.float32)
    conflicting[:, :6] = -0.1
    fresh = np.zeros((50, 7), dtype=np.float32)

    class SequencePolicy:
        def __init__(self):
            self.calls = 0

        def infer(self, observation):
            del observation
            self.calls += 1
            if self.calls == 1:
                return {"actions": forward}
            if self.calls == 2:
                return {"actions": conflicting}
            time.sleep(0.005)
            return {"actions": fresh}

    streamed = []
    run_rollout(
        RolloutConfig(
            duration_s=1.8,
            authorization=AUTHORIZATION,
            safety_profile="native",
            execute_steps=50,
            h50_prefetch_lead_steps=4,
        ),
        cameras=FakeCameras(),
        policy=SequencePolicy(),
        feedback_reader=FakeFeedback(),
        sink=sink,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        on_record=streamed.append,
    )

    first_after_fallback = next(row for row in streamed if row["plan"] == 2)
    assert first_after_fallback["plan_prefetch_accepted"] is False
    assert first_after_fallback["plan_feedback_anchored_hold"] is True
    assert first_after_fallback["plan_fallback_tracking_error_rad"] > 0.09
    np.testing.assert_allclose(first_after_fallback["plan_fallback_hold_target"][:6], 0.0)
    np.testing.assert_allclose(first_after_fallback["command"][:6], 0.0)


def test_mandatory_start_reset_uses_one_shared_verified_7d_target():
    calls = []

    class FakeResetPublisher:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def publish_home(self):
            calls.append(("publish", None))

    _reset_ros_controller_to_inference_start(FakeResetPublisher, reset_s=10.0)

    assert calls[0][0] == "init"
    assert calls[0][1] == {
        "target": INFERENCE_START_TARGET,
        "selected_command_topic": "/rlt/selected_joint_command",
        "hold_s": 10.0,
        "hz": INFERENCE_START_RESET_HZ,
    }
    assert len(INFERENCE_START_TARGET) == 7
    assert calls[1] == ("publish", None)


def test_windowed_h10_and_h50_publish_identical_first_policy_chunk():
    actions = np.zeros((50, 7), dtype=np.float32)
    actions[:, :6] = np.linspace(0.0, 0.15, 50)[:, None]
    actions[:, 6] = np.concatenate([np.full(25, 0.065), np.linspace(0.065, 0.01, 25)])

    outputs = {}
    for execute_steps in (10, 50):
        clock, sink = FakeClock(), FakeSink()
        run_rollout(
            RolloutConfig(
                duration_s=1.67,
                authorization=AUTHORIZATION,
                safety_profile="native",
                execute_steps=execute_steps,
            ),
            cameras=FakeCameras(),
            policy=FakePolicy(sink, actions=actions),
            feedback_reader=FakeFeedback(),
            sink=sink,
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        outputs[execute_steps] = np.stack(sink.sent[:50])

    np.testing.assert_allclose(outputs[10], outputs[50], atol=0.0, rtol=0.0)



def test_native_rollout_clips_model_targets_to_official_joint_limits_without_stopping():
    clock, sink = FakeClock(), FakeSink()
    actions = np.zeros((50, 7), dtype=np.float32)
    actions[:, 1] = np.deg2rad(-6.0)
    policy = FakePolicy(sink, actions=actions)
    streamed = []

    result = run_rollout(
        RolloutConfig(duration_s=0.04, authorization=AUTHORIZATION, safety_profile="native", execute_steps=1),
        cameras=FakeCameras(),
        policy=policy,
        feedback_reader=FakeFeedback(),
        sink=sink,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        on_record=streamed.append,
    )

    assert result["step_count"] >= 1
    assert sink.sent[0][1] == pytest.approx(0.0)
    assert "model_native_joint_range_clip" in streamed[0]["reasons"]
