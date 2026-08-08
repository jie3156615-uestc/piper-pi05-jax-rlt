import math
import threading
import time

import numpy as np
import pytest

from piper_runtime.buffered_policy_control import ActionBuffer
from piper_runtime.buffered_policy_control import BLEND_STEPS
from piper_runtime.buffered_policy_control import ControllerPublisher
from piper_runtime.buffered_policy_control import MAX_STALE_MS
from piper_runtime.buffered_policy_control import MIN_BUFFER_STEPS
from piper_runtime.buffered_policy_control import PREFETCH_THRESHOLD
from piper_runtime.buffered_policy_control import PolicyWorker
from piper_runtime.buffered_policy_control import PlannedChunk
from piper_runtime.buffered_policy_control import PolicyChunkPlanner
from piper_runtime.buffered_policy_control import TARGET_BUFFER_STEPS
from piper_runtime.buffered_policy_control import TemporalActionBuffer
from piper_runtime.buffered_policy_control import low_pass_and_limit_actions
from piper_runtime.buffered_policy_control import resample_actions_causal
from piper_runtime.buffered_policy_control import StrictChunkActionBuffer
from piper_runtime.buffered_policy_control import StrictPolicyChunkPlanner
from piper_runtime.hardware_control import HardwareSafetyConfig


def test_fixed_buffer_defaults_match_real_robot_contract():
    assert MIN_BUFFER_STEPS == 10
    assert PREFETCH_THRESHOLD == 15
    assert TARGET_BUFFER_STEPS == 30
    assert MAX_STALE_MS == 150
    assert BLEND_STEPS == 5


def test_low_pass_uses_dynamic_dt_and_preserves_frame_caps():
    config = HardwareSafetyConfig(
        model_smoothing_tau_s=0.05,
        model_max_joint_step=math.radians(3),
        model_max_gripper_step=0.02,
    )
    raw = np.zeros((1, 7))
    raw[0, 0] = math.radians(10)
    raw[0, 6] = 0.08

    at_30hz = low_pass_and_limit_actions(raw, initial_target=np.zeros(7), config=config, dt=1 / 30)
    at_60hz = low_pass_and_limit_actions(raw, initial_target=np.zeros(7), config=config, dt=1 / 60)

    assert at_30hz[0, 0] == pytest.approx(math.radians(3))
    assert 0 < at_60hz[0, 0] < at_30hz[0, 0]
    assert at_30hz[0, 6] == pytest.approx(0.02)


def test_low_pass_rejects_nan_and_clamps_official_ranges():
    config = HardwareSafetyConfig()
    raw = np.zeros((1, 7))
    raw[0, 1] = -10
    raw[0, 2] = 10
    raw[0, 6] = 10
    result = low_pass_and_limit_actions(raw, initial_target=np.zeros(7), config=config)
    assert result[0, 1] == pytest.approx(0)
    assert result[0, 2] == pytest.approx(0)
    assert result[0, 6] <= config.gripper_max
    raw[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        low_pass_and_limit_actions(raw, initial_target=np.zeros(7), config=config)


def test_ten_model_steps_resample_to_seventeen_50hz_points_without_endpoint_loss():
    actions = np.stack([np.full(7, index + 1.0) for index in range(10)])
    output = resample_actions_causal(actions, initial_target=np.zeros(7), input_hz=30, output_hz=50)
    assert output.shape == (17, 7)
    np.testing.assert_allclose(output[-1], actions[-1])
    assert np.all(np.diff(output[:, 0]) >= 0)


def test_strict_double_buffer_never_truncates_active_chunk():
    buffer = StrictChunkActionBuffer(np.zeros(7))
    active = np.stack([np.full(7, 0.001 * index) for index in range(17)])
    standby = np.stack([np.full(7, 0.100 + 0.001 * index) for index in range(17)])
    buffer.initialize(active, chunk_id=1, now_s=0.0)

    first = []
    for _ in range(7):
        item = buffer.pop_next()
        first.append(item)
        buffer.mark_published(item.target)
    install = buffer.install_standby(
        standby,
        chunk_id=2,
        now_s=0.1,
        planned_handoff_target=active[-1],
    )
    rest = []
    for _ in range(10):
        item = buffer.pop_next()
        rest.append(item)
        buffer.mark_published(item.target)
    promoted = buffer.pop_next()

    np.testing.assert_allclose(np.stack([item.target for item in first + rest]), active)
    assert all(item.chunk_id == 1 for item in first + rest)
    assert install.active_remaining == 10
    assert promoted is not None and promoted.chunk_id == 2 and promoted.chunk_step == 0
    assert np.max(np.abs(promoted.target[:6] - active[-1, :6])) <= math.radians(1.8)
    assert abs(promoted.target[6] - active[-1, 6]) <= 0.012


def test_strict_handoff_target_survives_last_pop_mark_race():
    buffer = StrictChunkActionBuffer(np.zeros(7))
    active = np.zeros((3, 7))
    active[:, 0] = [0.01, 0.02, 0.03]
    buffer.initialize(active, chunk_id=1, now_s=0.0)
    for _ in range(2):
        item = buffer.pop_next()
        buffer.mark_published(item.target)

    final = buffer.pop_next()
    assert buffer.remaining() == 0
    # The publisher has popped the final item but has not called
    # mark_published yet.  Standby preparation must still see its immutable
    # terminal target rather than the previous command.
    np.testing.assert_allclose(buffer.handoff_target(), active[-1])
    buffer.mark_published(final.target)
    np.testing.assert_allclose(buffer.handoff_target(), active[-1])


def test_strict_promotion_uses_c1_joint_bridge_without_blending_gripper():
    buffer = StrictChunkActionBuffer(np.zeros(7))
    active = np.zeros((17, 7))
    active[:, 0] = np.arange(1, 18) * 0.001
    active[:, 6] = np.arange(1, 18) * 0.001
    standby = np.zeros((17, 7))
    standby[:, 0] = 0.10 + np.arange(17) * 0.001
    standby[:, 6] = 0.060
    buffer.initialize(active, chunk_id=1, now_s=0.0)

    for _ in range(15):
        item = buffer.pop_next()
        buffer.mark_published(item.target)
    buffer.install_standby(
        standby,
        chunk_id=2,
        now_s=0.1,
        planned_handoff_target=active[-1],
    )
    for _ in range(2):
        item = buffer.pop_next()
        buffer.mark_published(item.target)
    boundary = active[-1].copy()
    promoted = []
    for _ in range(5):
        item = buffer.pop_next()
        promoted.append(item.target.copy())
        buffer.mark_published(item.target)
    promoted = np.stack(promoted)

    # The fifth joint point lands on the original new-plan endpoint and the
    # per-servo safety cap remains intact.
    assert promoted[-1, 0] == pytest.approx(standby[4, 0])
    deltas = np.diff(np.vstack([boundary, promoted])[:, 0])
    assert np.max(np.abs(deltas)) <= math.radians(1.8)
    # Absolute gripper intent is not velocity-blended with the old chunk.  It
    # is only constrained by the existing 12 mm/servo safety cap.
    assert promoted[0, 6] == pytest.approx(boundary[6] + 0.012)


class _StaticCamera:
    def read(self, **kwargs):
        del kwargs
        return {
            "camera1": np.ones((480, 640, 3), dtype=np.uint8),
            "camera2": np.ones((480, 640, 3), dtype=np.uint8),
        }


class _StaticPolicy:
    def __init__(self, actions):
        self.actions = np.asarray(actions, dtype=np.float32)

    def infer(self, observation):
        del observation
        return {"actions": self.actions.copy()}


def test_strict_planner_uses_only_first_ten_and_suffix_cannot_leak():
    snapshot = np.asarray([0.1, 0.2, -0.1, 0.0, 0.3, 0.0, 0.06])

    class Feedback:
        def read(self):
            return snapshot.copy()

    base = np.repeat(snapshot[None, :], 50, axis=0)
    base[:, 0] += np.arange(50) * 0.001
    changed_suffix = base.copy()
    changed_suffix[10:, :6] += 1.0
    changed_suffix[10:, 6] = 0.0

    def run(actions):
        planner = StrictPolicyChunkPlanner(
            cameras=_StaticCamera(),
            policy=_StaticPolicy(actions),
            feedback_reader=Feedback(),
            prompt="test",
            execute_steps=10,
            safety_config=HardwareSafetyConfig(),
            max_inference_s=1.0,
        )
        planned = planner.plan(chunk_id=1, warmup_frames=0)
        handoff = snapshot.copy()
        handoff[0] += 0.05
        prepared = planner.prepare_for_handoff(planned, handoff_target=handoff)
        return planned, prepared

    planned_a, prepared_a = run(base)
    planned_b, prepared_b = run(changed_suffix)

    assert planned_a.reference_actions.shape == (10, 7)
    np.testing.assert_allclose(planned_a.reference_actions, base[:10])
    np.testing.assert_allclose(planned_a.reference_actions, planned_b.reference_actions)
    np.testing.assert_allclose(prepared_a.actions, prepared_b.actions)
    assert prepared_a.actions.shape == (17, 7)
    expected_delta = base[:10, 0] - snapshot[0]
    np.testing.assert_allclose(
        prepared_a.rebased_model_actions[:, 0],
        prepared_a.handoff_target[0] + expected_delta,
    )


def test_strict_planner_latency_aligns_h10_and_anchors_to_preceding_prediction():
    snapshot = np.asarray([0.1, 0.2, -0.1, 0.0, 0.3, 0.0, 0.06])

    class Feedback:
        def read(self):
            return snapshot.copy()

    actions = np.repeat(snapshot[None, :], 50, axis=0)
    actions[:, 0] += np.arange(1, 51) * 0.01
    actions[:, 6] = np.linspace(0.06, 0.0, 50)
    planner = StrictPolicyChunkPlanner(
        cameras=_StaticCamera(),
        policy=_StaticPolicy(actions),
        feedback_reader=Feedback(),
        prompt="test",
        execute_steps=10,
        safety_config=HardwareSafetyConfig(),
        max_inference_s=1.0,
        clock=lambda: 0.0,
    )
    planned = planner.plan(chunk_id=1, warmup_frames=0)
    handoff = snapshot.copy()
    handoff[0] = 0.5
    prepared = planner.prepare_for_handoff(
        planned,
        handoff_target=handoff,
        handoff_time_s=0.2,
    )

    # ceil(0.2 * 30) = 6: execute exactly actions[6:16], still C=10.
    assert prepared.action_start_index == 6
    assert prepared.rebased_model_actions.shape == (10, 7)
    preceding = actions[5]
    expected_joints = actions[6:16, :6] + (handoff[:6] - preceding[:6])[None, :]
    np.testing.assert_allclose(prepared.rebased_model_actions[:, :6], expected_joints)
    # Gripper is absolute and never joint-rebased.
    np.testing.assert_allclose(prepared.rebased_model_actions[:, 6], actions[6:16, 6])

    non_grid = planner.prepare_for_handoff(
        planned,
        handoff_target=handoff,
        handoff_time_s=0.12,
    )
    assert non_grid.action_start_index == 3

    with pytest.raises(RuntimeError, match="expired"):
        planner.prepare_for_handoff(
            planned,
            handoff_target=handoff,
            handoff_time_s=2.0,
        )


def test_chunk_merge_blends_five_old_and_new_points_then_uses_new_tail():
    buffer = ActionBuffer(np.zeros(7))
    old = np.stack([np.full(7, 0.010 + index * 0.001) for index in range(8)])
    new = np.stack([np.full(7, 0.020 + index * 0.001) for index in range(10)])
    buffer.initialize(old, chunk_id=1, now_s=0)

    merge = buffer.replace_with_blend(new, chunk_id=2, now_s=1)
    merged = [buffer.pop_next() for _ in range(10)]

    assert merge.blended_points == 5
    assert not merge.used_last_published_fallback
    np.testing.assert_allclose(merged[0].target, old[0])
    np.testing.assert_allclose(merged[4].target, new[4])
    np.testing.assert_allclose(merged[5].target, new[5])
    assert all(item.chunk_id == 2 for item in merged)


def test_short_old_buffer_blends_from_last_published_target():
    buffer = ActionBuffer(np.full(7, 0.030))
    buffer.initialize(np.stack([np.full(7, 0.031), np.full(7, 0.032)]), chunk_id=1, now_s=0)
    new = np.stack([np.full(7, 0.040 + index * 0.001) for index in range(8)])

    merge = buffer.replace_with_blend(new, chunk_id=2, now_s=1)
    first = buffer.pop_next()
    for _ in range(3):
        buffer.pop_next()
    fifth = buffer.pop_next()

    assert merge.used_last_published_fallback
    np.testing.assert_allclose(first.target, np.full(7, 0.030))
    np.testing.assert_allclose(fifth.target, new[4])


def test_fallback_blend_cannot_create_a_large_50hz_boundary_jump():
    buffer = ActionBuffer(np.zeros(7))
    far_chunk = np.full((8, 7), 1.0)
    buffer.replace_with_blend(far_chunk, chunk_id=2, now_s=1)
    targets = np.stack([buffer.pop_next().target for _ in range(8)])
    padded = np.vstack([np.zeros(7), targets])
    assert np.max(np.abs(np.diff(padded[:, :6], axis=0))) <= math.radians(1.8) + 1e-12
    assert np.max(np.abs(np.diff(padded[:, 6], axis=0))) <= 0.012 + 1e-12


def test_handoff_alignment_uses_actual_state_instead_of_fixed_time_index():
    filtered = np.zeros((50, 7))
    filtered[:, 0] = np.arange(50) * 0.01
    planned = PlannedChunk(
        2,
        np.zeros((22, 7)),
        0.1,
        np.zeros(7),
        np.zeros(7),
        6,
        {"d10": 0.0, "d20": 0.0, "d30": 0.0, "d50": 0.0},
        filtered,
        13,
        6,
    )
    planner = PolicyChunkPlanner(
        cameras=None,
        policy=None,
        feedback_reader=None,
        prompt="test",
        execute_steps=10,
        safety_config=HardwareSafetyConfig(),
        max_inference_s=1.0,
    )
    handoff = np.zeros(7)
    handoff[0] = 0.021
    aligned = planner.align_to_handoff(planned, handoff_state=handoff)
    assert aligned.expected_action_start_index == 6
    assert aligned.action_start_index == 2
    assert aligned.actions.shape == (22, 7)


class CaptureSink:
    def __init__(self):
        self.sent = []
        self.timestamps = []

    def send(self, command):
        self.sent.append(np.asarray(command).copy())
        self.timestamps.append(time.monotonic())


class ZeroFeedback:
    def read(self):
        return np.zeros(7, dtype=np.float32)


def test_publisher_holds_last_safe_target_and_marks_stale_without_catchup():
    buffer = ActionBuffer(np.zeros(7))
    one = np.zeros((1, 7))
    one[0, 0] = 0.01
    buffer.initialize(one, chunk_id=1, now_s=0)
    sink = CaptureSink()
    requests = []
    publisher = ControllerPublisher(
        action_buffer=buffer,
        sink=sink,
        safety_config=HardwareSafetyConfig(),
        initial_state=np.zeros(7),
        safety_profile="native",
        feedback_reader=ZeroFeedback(),
        request_prefetch=lambda: requests.append(True) or True,
    )

    command = publisher.publish_once(now_s=0.0)
    empty_hold = publisher.publish_once(now_s=0.02)
    stale_hold = publisher.publish_once(now_s=0.18)

    assert command["event"] == "command"
    assert empty_hold["event"] == "buffer_empty_hold"
    assert stale_hold["event"] == "safe_hold"
    np.testing.assert_allclose(sink.sent[-1], sink.sent[0])
    assert publisher.publish_count == 3
    assert requests


class BlockingPlanner:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.feedback_reader = ZeroFeedback()

    def plan(self, *, chunk_id, warmup_frames, runtime_alignment_delay_s=None):
        from piper_runtime.buffered_policy_control import PlannedChunk

        del warmup_frames, runtime_alignment_delay_s
        self.calls += 1
        self.started.set()
        self.release.wait(timeout=2)
        return PlannedChunk(
            chunk_id,
            np.zeros((17, 7)),
            0.1,
            np.zeros(7),
            np.zeros(7),
            0,
            {"d10": 0.0, "d20": 0.0, "d30": 0.0, "d50": 0.0},
            np.zeros((50, 7)),
            13,
            0,
        )

    def align_to_handoff(self, planned, *, handoff_state):
        del handoff_state
        return planned


def test_policy_worker_is_single_flight_and_never_duplicates_inference():
    buffer = ActionBuffer(np.zeros(7))
    buffer.initialize(np.zeros((12, 7)), chunk_id=1, now_s=0)
    planner = BlockingPlanner()
    worker = PolicyWorker(planner=planner, action_buffer=buffer)
    worker.start()
    try:
        assert worker.request_prefetch()
        assert planner.started.wait(timeout=1)
        assert not worker.request_prefetch()
        planner.release.set()
        time.sleep(0.05)
        assert worker.completed_plans == 0
        assert buffer.remaining() == 12
        for _ in range(7):
            buffer.pop_next()
        deadline = time.monotonic() + 1
        while worker.completed_plans < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert planner.calls == 1
        assert worker.completed_plans == 1
    finally:
        planner.release.set()
        worker.stop()


def test_gripper_reversal_requires_two_consistent_plans():
    worker = PolicyWorker(planner=None, action_buffer=ActionBuffer(np.zeros(7)))

    def plan(current, target):
        actions = np.zeros((22, 7))
        actions[:, 6] = np.linspace(current, target, 22)
        state = np.zeros(7)
        state[6] = current
        return PlannedChunk(
            2,
            actions,
            0.1,
            state,
            state,
            0,
            {"d10": 0.0, "d20": 0.0, "d30": 0.0, "d50": 0.0},
            np.zeros((50, 7)),
            13,
            0,
        )

    closing, intent, pending = worker._stabilize_gripper(plan(0.065, 0.02))
    assert intent == -1 and not pending
    assert np.all(np.diff(closing.actions[:, 6]) <= 1e-12)

    held, intent, pending = worker._stabilize_gripper(plan(0.02, 0.065))
    assert intent == -1 and pending
    np.testing.assert_allclose(held.actions[:, 6], 0.02)

    opening, intent, pending = worker._stabilize_gripper(plan(0.02, 0.065))
    assert intent == 1 and not pending
    assert np.all(np.diff(opening.actions[:, 6]) >= -1e-12)


def test_action_buffer_rejects_nonfinite_chunk_before_publication():
    buffer = ActionBuffer(np.zeros(7))
    invalid = np.zeros((10, 7))
    invalid[3, 2] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        buffer.initialize(invalid, chunk_id=1, now_s=0)


def test_temporal_buffer_keeps_long_horizon_when_replanning_every_ten_model_steps():
    buffer = TemporalActionBuffer(
        np.zeros(7),
        replan_interval_steps=17,
        max_ensemble_plans=5,
    )
    initial = np.zeros((84, 7))
    initial[17:, 0] = np.linspace(0.0, 0.8, 67)
    buffer.initialize(initial, chunk_id=1, now_s=0)

    published = []
    for slot in range(70):
        if slot in (17, 34, 51):
            current = buffer.last_published_target()
            replacement = np.repeat(current[None, :], 84, axis=0)
            replacement[17:, 0] += np.linspace(0.0, 0.8, 67)
            buffer.insert_plan(
                replacement,
                chunk_id=2 + slot // 17,
                observation_slot=slot,
                now_s=float(slot),
            )
        item = buffer.pop_next()
        assert item is not None
        buffer.mark_published(item.target)
        published.append(item.target.copy())

    # Prefix-replacement would remain at zero forever; the absolute timeline
    # retains and executes the original plan's post-prefix task intent.
    assert published[-1][0] > 0.25


def test_temporal_buffer_preserves_late_gripper_event_across_replans():
    buffer = TemporalActionBuffer(
        np.asarray([0, 0, 0, 0, 0, 0, 0.065], dtype=float),
        replan_interval_steps=17,
        max_ensemble_plans=5,
    )
    plan = np.zeros((84, 7))
    plan[:, 6] = 0.065
    plan[42:, 6] = 0.010
    buffer.initialize(plan, chunk_id=1, now_s=0)
    commands = []
    for slot in range(70):
        if slot in (17, 34):
            buffer.insert_plan(plan, chunk_id=2 + slot // 17, observation_slot=slot, now_s=slot)
        item = buffer.pop_next()
        assert item is not None
        buffer.mark_published(item.target)
        commands.append(item.target.copy())
    assert min(command[6] for command in commands) < 0.02


def test_temporal_buffer_drops_missed_slots_instead_of_catching_up():
    buffer = TemporalActionBuffer(np.zeros(7), replan_interval_steps=17, max_ensemble_plans=5)
    actions = np.zeros((20, 7))
    actions[:, 0] = np.arange(20) * 0.001
    buffer.initialize(actions, chunk_id=1, now_s=0)
    assert buffer.skip_steps(3) == 3
    item = buffer.pop_next()
    assert item is not None
    assert item.chunk_step == 3
    assert item.target[0] == pytest.approx(0.003)


def test_temporal_nan_plan_is_rejected_atomically():
    buffer = TemporalActionBuffer(np.zeros(7), replan_interval_steps=17, max_ensemble_plans=5)
    original = np.full((20, 7), 0.01)
    buffer.initialize(original, chunk_id=1, now_s=0)
    remaining = buffer.remaining()
    invalid = np.zeros((84, 7))
    invalid[20, 3] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        buffer.insert_plan(invalid, chunk_id=2, observation_slot=0, now_s=1)
    assert buffer.remaining() == remaining
    item = buffer.pop_next()
    assert item is not None
    baseline = TemporalActionBuffer(np.zeros(7), replan_interval_steps=17, max_ensemble_plans=5)
    baseline.initialize(original, chunk_id=1, now_s=0)
    np.testing.assert_allclose(item.target, baseline.pop_next().target)
