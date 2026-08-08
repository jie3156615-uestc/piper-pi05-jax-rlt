from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from piper_runtime.hardware_control import HardwareSafetyConfig
from piper_runtime.hardware_control import StatefulSafetyFilter
from piper_runtime.rlt_command_mux import TimedCommand
from piper_runtime.rlt_episode_logger import RLTEpisodeLogger
from piper_runtime.rlt_keyboard import RLTKeyboardStateMachine
from piper_runtime.rlt_phase_gate import PhaseGateConfig
from piper_runtime.rlt_phase_gate import PhaseGateSnapshot
from piper_runtime.rlt_phase_gate import SingleLatchPhaseGate
from piper_runtime.rlt_policy_worker import PolicyWorkerOutput
from piper_runtime.rlt_actor_protocol import ACTION_SCHEMA_FINGERPRINT
from piper_runtime.rlt_actor_protocol import ACTOR_PROJECTION_PROFILE
from piper_runtime.rlt_actor_protocol import RANK1_BUMP_CONTRACT
from piper_runtime.rlt_actor_protocol import (
    RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT,
)
from piper_runtime.rlt_actor_protocol import (
    RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE,
)
from piper_runtime.rlt_actor_protocol import (
    SUPPORTED_RAW_ACTOR_ACTION_SCHEMA_FINGERPRINTS,
)
from piper_runtime.rlt_residual_governor import GRIPPER_RESIDUAL_CLOSE_ASSIST
from piper_runtime.rlt_residual_governor import RANK1_BUMP_WINDOW
from piper_runtime.rlt_residual_governor import PERSISTENT_C10_EXECUTION_CONTRACT
from piper_runtime.rlt_residual_governor import (
    PERSISTENT_FILTERED_ACTUAL_ACTION_SCHEMA_FINGERPRINT,
)
from piper_runtime.rlt_residual_governor import (
    PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT,
)
from piper_runtime.rlt_residual_governor import (
    PERSISTENT_FILTERED_ACTUAL_GOVERNOR_FINGERPRINT,
)
from piper_runtime.rlt_residual_governor import (
    PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT,
)
from piper_runtime.rlt_residual_governor import (
    PERSISTENT_GRIPPER_CLOSE_GOVERNOR_FINGERPRINT,
)
from piper_runtime.rlt_residual_governor import PersistentActorResidualGovernor
from piper_runtime.rlt_takeover_rollout import ACTOR_LIVE_AUTHORIZATION
from piper_runtime.rlt_takeover_rollout import PUBLISH_AUTHORIZATION
from piper_runtime.rlt_takeover_rollout import FreshArmStatusTracker
from piper_runtime.rlt_takeover_rollout import ArmStatusHealthError
from piper_runtime.rlt_takeover_rollout import RosFeedbackReader
from piper_runtime.rlt_takeover_rollout import decode_ros_arm_error_code
from piper_runtime.rlt_takeover_rollout import PiperSdkCommandPublisher
from piper_runtime.rlt_takeover_rollout import NativePikaJointStatePassthrough
from piper_runtime.rlt_takeover_rollout import RosNativeSdkCommandPublisher
from piper_runtime.rlt_takeover_rollout import RosArbitratedCommandPublisher
from piper_runtime.rlt_takeover_rollout import ScriptedKeySource
from piper_runtime.rlt_takeover_rollout import TakeoverLoopCore
from piper_runtime.rlt_takeover_rollout import TakeoverRuntimeConfig
from piper_runtime.rlt_takeover_rollout import parse_scripted_keys


class FakeCameras:
    def read(self, timeout_ms: int = 5000, warmup_frames: int = 1):
        del timeout_ms, warmup_frames
        return {
            "camera1": np.full((480, 640, 3), 11, dtype=np.uint8),
            "camera2": np.full((480, 640, 3), 22, dtype=np.uint8),
        }


class FakeFeedbackReader:
    def __init__(self):
        self.value = np.zeros(7, dtype=np.float32)

    def read(self):
        return self.value.copy()


class MessagePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class FakeRosTime:
    @staticmethod
    def now():
        return 123.0


class FakeRosModule:
    Time = FakeRosTime


class FakeHumanTracker:
    def latest(self, *, now_s: float):
        return TimedCommand(value=np.full(7, 2.0, dtype=np.float32), timestamp_s=now_s)


class ScheduledHumanTracker:
    def __init__(self, values_by_call: dict[int, np.ndarray]):
        self.values_by_call = {int(k): np.asarray(v, dtype=np.float32) for k, v in values_by_call.items()}
        self.calls = 0

    def latest(self, *, now_s: float):
        value = self.values_by_call.get(self.calls)
        self.calls += 1
        if value is None:
            return None
        return TimedCommand(value=value.copy(), timestamp_s=now_s)


class FixedHumanTracker:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def latest(self, *, now_s: float):
        return TimedCommand(value=self.value.copy(), timestamp_s=now_s)


class FakePolicyWorker:
    planning_mode = "test_synchronous"

    def __init__(self):
        self.submitted = []

    def submit(self, observation, *, timestamp_s: float, observation_t: int | None = None):
        self.submitted.append((observation, timestamp_s, observation_t))

    def latest(self):
        actions = np.full((50, 7), 1.0, dtype=np.float32)
        timestamp = 0.0 if not self.submitted else float(self.submitted[-1][1])
        observation_t = None if not self.submitted else self.submitted[-1][2]
        return PolicyWorkerOutput(
            value={"actions": actions},
            observation_timestamp_s=timestamp,
            completed_timestamp_s=timestamp + 0.01,
            policy_observation_t=observation_t,
        )


class FakeShadowPolicyWorker(FakePolicyWorker):
    def __init__(self, *, invalid_actor: bool = False):
        super().__init__()
        self.invalid_actor = invalid_actor

    def latest(self):
        actions = np.full((50, 7), 1.0, dtype=np.float32)
        observation = {} if not self.submitted else self.submitted[-1][0]
        requested = observation.get("rlt/behavior_ref")
        actor_ref = actions[:10] if requested is None else np.asarray(requested, dtype=np.float32)
        actor = actor_ref.copy()
        actor[:, 0] += RANK1_BUMP_WINDOW * 0.002
        if self.invalid_actor:
            actor[0, 0] = np.nan
        timestamp = 0.0 if not self.submitted else float(self.submitted[-1][1])
        observation_t = None if not self.submitted else self.submitted[-1][2]
        return PolicyWorkerOutput(
            value={
                "actions": actions,
                "z_rl": np.ones(2048, dtype=np.float32),
                "a_actor": actor,
                "a_actor_action_space": "joint_absolute_gripper_absolute",
                "a_actor_behavior_ref_contract": RANK1_BUMP_CONTRACT,
                "a_actor_behavior_ref_source": (
                    "response_actions" if requested is None else "request_behavior_ref"
                ),
                "a_actor_behavior_ref_plan_id": observation.get("rlt/behavior_ref_plan_id"),
                "a_actor_behavior_ref_start_offset": observation.get(
                    "rlt/behavior_ref_start_offset"
                ),
                "a_actor_action_schema_fingerprint": ACTION_SCHEMA_FINGERPRINT,
                "a_actor_projection_profile": ACTOR_PROJECTION_PROFILE,
            },
            observation_timestamp_s=timestamp,
            completed_timestamp_s=timestamp + 0.05,
            policy_observation_t=observation_t,
        )


class ProtocolBaseOnlyWorker(FakePolicyWorker):
    def latest(self):
        output = super().latest()
        output.value["rlt_shadow"] = {
            "mode": "base_only",
            "base_policy_called": True,
            "base_rng_advanced": True,
            "token_encoder_called": False,
            "actor_called": False,
            "base_policy_latency_s": 0.01,
        }
        return output


class SmoothProtocolBaseOnlyWorker(ProtocolBaseOnlyWorker):
    """Small, physically continuous H50 reference for the strict 0.06 limit."""

    def latest(self):
        output = super().latest()
        actions = np.zeros((50, 7), dtype=np.float32)
        actions[:, 0] = np.arange(1, 51, dtype=np.float32) * 0.001
        actions[:, 1] = np.arange(1, 51, dtype=np.float32) * 0.002
        actions[:, 2] = -np.arange(1, 51, dtype=np.float32) * 0.001
        actions[:, 6] = 0.02
        output.value["actions"] = actions
        return output


class ProtocolActorEnrichmentWorker(FakePolicyWorker):
    def latest(self):
        if not self.submitted:
            return None
        observation, timestamp, observation_t = self.submitted[-1]
        reference = np.asarray(observation["rlt/behavior_ref"], dtype=np.float32)
        actor = reference.copy()
        actor[:, 0] += RANK1_BUMP_WINDOW * 0.002
        return PolicyWorkerOutput(
            value={
                "actions": reference.copy(),
                "z_rl": np.ones(2048, dtype=np.float32),
                "a_actor": actor,
                "a_actor_action_space": "joint_absolute_gripper_absolute",
                "a_actor_behavior_ref_contract": RANK1_BUMP_CONTRACT,
                "a_actor_behavior_ref_source": "request_behavior_ref",
                "a_actor_behavior_ref_plan_id": observation[
                    "rlt/behavior_ref_plan_id"
                ],
                "a_actor_behavior_ref_start_offset": observation[
                    "rlt/behavior_ref_start_offset"
                ],
                "a_actor_action_schema_fingerprint": ACTION_SCHEMA_FINGERPRINT,
                "a_actor_projection_profile": ACTOR_PROJECTION_PROFILE,
                "rlt_shadow": {
                    "mode": "actor_only",
                    "actor_only_protocol": "actor_enrichment_only_v1",
                    "base_policy_called": False,
                    "base_rng_advanced": False,
                    "token_encoder_called": True,
                    "actor_called": True,
                    "shadow_latency_s": 0.02,
                },
            },
            observation_timestamp_s=float(timestamp),
            completed_timestamp_s=float(timestamp) + 0.02,
            policy_observation_t=observation_t,
        )


class CloseAssistProtocolActorEnrichmentWorker(FakePolicyWorker):
    def latest(self):
        if not self.submitted:
            return None
        observation, timestamp, observation_t = self.submitted[-1]
        reference = np.asarray(observation["rlt/behavior_ref"], dtype=np.float32)
        actor = reference.copy()
        actor[:, 0] += RANK1_BUMP_WINDOW * 0.002
        actor[:, 6] += RANK1_BUMP_WINDOW * -0.002
        return PolicyWorkerOutput(
            value={
                "actions": reference.copy(),
                "z_rl": np.ones(2048, dtype=np.float32),
                "a_actor": actor,
                "a_actor_action_space": "joint_absolute_gripper_absolute",
                "a_actor_behavior_ref_contract": RANK1_BUMP_CONTRACT,
                "a_actor_behavior_ref_source": "request_behavior_ref",
                "a_actor_behavior_ref_plan_id": observation[
                    "rlt/behavior_ref_plan_id"
                ],
                "a_actor_behavior_ref_start_offset": observation[
                    "rlt/behavior_ref_start_offset"
                ],
                "a_actor_action_schema_fingerprint": (
                    RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT
                ),
                "a_actor_projection_profile": (
                    RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE
                ),
                "rlt_shadow": {
                    "mode": "actor_only",
                    "actor_only_protocol": "actor_enrichment_only_v1",
                    "base_policy_called": False,
                    "base_rng_advanced": False,
                    "token_encoder_called": True,
                    "actor_called": True,
                    "shadow_latency_s": 0.02,
                },
            },
            observation_timestamp_s=float(timestamp),
            completed_timestamp_s=float(timestamp) + 0.02,
            policy_observation_t=observation_t,
        )


class NeverReadyPolicyWorker:
    """Record an independent request lane without ever producing a result."""

    planning_mode = "test_never_ready"

    def __init__(self):
        self.submitted = []

    def submit(self, observation, *, timestamp_s: float, observation_t: int | None = None):
        self.submitted.append((observation, timestamp_s, observation_t))

    def latest(self):
        return None


class DelayedSecondPlanWorker:
    """Expose the H50 standby only after the active 50 commands are consumed."""

    planning_mode = "test_delayed_second_h50"

    def __init__(self, *, clock, second_delay_s: float = 0.40):
        self.clock = clock
        self.second_delay_s = float(second_delay_s)
        self.submitted = []

    def submit(self, observation, *, timestamp_s: float, observation_t: int | None = None):
        self.submitted.append((observation, timestamp_s, observation_t))

    def latest(self):
        if not self.submitted:
            return None
        selected = 0
        if (
            len(self.submitted) >= 2
            and self.clock.now() - float(self.submitted[1][1]) >= self.second_delay_s
        ):
            selected = 1
        timestamp = float(self.submitted[selected][1])
        action_offset = 0.0 if selected == 0 else 0.05
        actions = np.zeros((50, 7), dtype=np.float32)
        actions[:, :6] = (
            action_offset
            + np.arange(50, dtype=np.float32)[:, None] * 0.001
        )
        actions[:, 6] = 0.065
        return PolicyWorkerOutput(
            value={"actions": actions},
            observation_timestamp_s=timestamp,
            completed_timestamp_s=(
                timestamp + 0.01
                if selected == 0
                else timestamp + self.second_delay_s
            ),
            policy_plan_id=f"delayed_plan_{selected}",
            policy_observation_t=self.submitted[selected][2],
            worker_lane="base",
        )


class JumpingShadowPolicyWorker(FakePolicyWorker):
    """Return a safe first Actor plan and a discontinuous prefetched replacement."""

    def latest(self):
        plan_index = max(0, len(self.submitted) - 1)
        actions = np.full((50, 7), 0.01, dtype=np.float32)
        observation = {} if not self.submitted else self.submitted[-1][0]
        requested = observation.get("rlt/behavior_ref")
        actor_ref = actions[:10] if requested is None else np.asarray(requested, dtype=np.float32)
        actor = actor_ref.copy()
        actor[:, 0] += RANK1_BUMP_WINDOW * (0.002 if plan_index == 0 else 0.20)
        timestamp = 0.0 if not self.submitted else float(self.submitted[-1][1])
        observation_t = None if not self.submitted else self.submitted[-1][2]
        return PolicyWorkerOutput(
            value={
                "actions": actions,
                "z_rl": np.ones(2048, dtype=np.float32),
                "a_actor": actor,
                "a_actor_action_space": "joint_absolute_gripper_absolute",
                "a_actor_behavior_ref_contract": RANK1_BUMP_CONTRACT,
                "a_actor_behavior_ref_source": (
                    "response_actions" if requested is None else "request_behavior_ref"
                ),
                "a_actor_behavior_ref_plan_id": observation.get("rlt/behavior_ref_plan_id"),
                "a_actor_behavior_ref_start_offset": observation.get(
                    "rlt/behavior_ref_start_offset"
                ),
                "a_actor_action_schema_fingerprint": ACTION_SCHEMA_FINGERPRINT,
                "a_actor_projection_profile": ACTOR_PROJECTION_PROFILE,
            },
            observation_timestamp_s=timestamp,
            completed_timestamp_s=timestamp + 0.01,
            policy_plan_id=f"plan_{plan_index}",
            policy_observation_t=observation_t,
        )


class FakePhaseClassifier:
    def __init__(self, probabilities: list[float]):
        self.probabilities = list(probabilities)
        self.calls = 0

    def predict_probability(self, images) -> float:
        assert sorted(images) == ["camera1", "camera2"]
        index = min(self.calls, len(self.probabilities) - 1)
        self.calls += 1
        return float(self.probabilities[index])


class ScheduledPhaseGate:
    def __init__(self, active_by_t: dict[int, bool]):
        self.active_by_t = {int(t): bool(active) for t, active in active_by_t.items()}
        self._previous_active = False
        self._enter_t = None
        self._exit_t = None

    def update(self, probability, *, t: int, terminal: bool, terminal_reason: str):
        del probability
        active = bool(self.active_by_t.get(int(t), False)) and not terminal
        if active and not self._previous_active:
            self._enter_t = int(t)
            self._exit_t = None
        elif self._previous_active and not active:
            self._exit_t = int(t)
        self._previous_active = active
        return PhaseGateSnapshot(
            probability=1.0 if active else 0.0,
            active=active,
            state="ACTIVE" if active else "EXITED",
            enter_t=self._enter_t,
            exit_t=self._exit_t,
            reason=terminal_reason if terminal else "scheduled_test_gate",
            high_count=1 if active else 0,
        )


class FakeKeySource:
    def __init__(self, keys_by_t: dict[int, list[str]]):
        self.keys_by_t = keys_by_t
        self.t = 0

    def poll_keys(self):
        return list(self.keys_by_t.get(self.t, []))


class FakeImageWriter:
    def __init__(self):
        self.saved = []

    def save(self, *, t: int, images):
        self.saved.append((t, sorted(images)))
        return f"camera_global/{t:06d}.jpg", f"camera_wrist/{t:06d}.jpg"


class FakePublisher:
    def __init__(self):
        self.commands = []

    def publish(self, command):
        self.commands.append(np.asarray(command, dtype=np.float32).copy())


class FailingPublisher:
    def publish(self, command):
        del command
        raise RuntimeError("injected publish failure")


class FakeSdkSink:
    def __init__(self):
        self.move_speed_percent = 0
        self.sent = []

    def send(self, command):
        self.sent.append((self.move_speed_percent, np.asarray(command, dtype=np.float32).copy()))


class FakeTeleopController:
    def __init__(self):
        self.calls = []

    def set_active(self, active: bool) -> None:
        self.calls.append(bool(active))

    def observed_active(self) -> bool:
        return False


class ScheduledTeleopController:
    """Expose an explicit observed teleop off/on edge to the loop."""

    def __init__(self, active_by_t: dict[int, bool], *, step_source: FakeKeySource):
        self.active_by_t = {int(k): bool(v) for k, v in active_by_t.items()}
        self.step_source = step_source
        self.last_active = False
        self.set_calls = []

    def set_active(self, active: bool) -> None:
        # This records programmatic requests only.  The scheduled observed state
        # models the operator's physical double-click and is intentionally not
        # changed by the request itself.
        self.set_calls.append(bool(active))

    def observed_active(self) -> bool:
        self.last_active = self.active_by_t.get(int(self.step_source.t), self.last_active)
        return self.last_active


def test_native_sdk_publisher_matches_model_speed30_and_original_human_speed50() -> None:
    sink = FakeSdkSink()
    publisher = PiperSdkCommandPublisher(sink)
    publisher.publish_selected(np.ones(7), source="pi05")
    publisher.publish_selected(np.full(7, 2.0), source="human_pika")

    assert [item[0] for item in sink.sent] == [30, 50]
    np.testing.assert_allclose(sink.sent[0][1], np.ones(7))
    np.testing.assert_allclose(sink.sent[1][1], np.full(7, 2.0))


def test_ros_arm_status_tracker_rejects_stale_or_real_controller_errors() -> None:
    tracker = FreshArmStatusTracker(freshness_s=0.5)
    with pytest.raises(RuntimeError, match="no fresh"):
        tracker.require_healthy(now_s=1.0)

    healthy = type("Status", (), {"err_code": 0})()
    tracker.update(healthy, timestamp_s=1.0)
    tracker.require_healthy(now_s=1.4)
    with pytest.raises(RuntimeError, match="no fresh"):
        tracker.require_healthy(now_s=1.6)

    failed = type("Status", (), {"err_code": 7})()
    tracker.update(failed, timestamp_s=2.0)
    with pytest.raises(ArmStatusHealthError, match="joint_1_communication"):
        tracker.require_healthy(now_s=2.1)


def test_ros_arm_error_decode_matches_official_sdk_bit_layout() -> None:
    assert decode_ros_arm_error_code(1) == ["joint_1_communication"]
    assert decode_ros_arm_error_code(63) == [
        "joint_1_communication",
        "joint_2_communication",
        "joint_3_communication",
        "joint_4_communication",
        "joint_5_communication",
        "joint_6_communication",
    ]
    assert decode_ros_arm_error_code(1 << 8) == ["joint_1_angle_limit"]


class SequenceArmHealth:
    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    def require_healthy(self, *, now_s):
        del now_s
        self.calls += 1
        if self.errors:
            error = self.errors.pop(0)
            if error is not None:
                raise error


class FixedFreshTracker:
    def latest(self, *, now_s):
        return TimedCommand(value=np.arange(7, dtype=np.float32), timestamp_s=now_s)


def test_feedback_reader_waits_through_transient_communication_flags() -> None:
    arm = SequenceArmHealth(
        [
            ArmStatusHealthError(63, 0.10),
            ArmStatusHealthError(1, 0.20),
            None,
        ]
    )
    reader = RosFeedbackReader(FixedFreshTracker(), arm, communication_grace_s=0.75)

    np.testing.assert_allclose(reader.read(), np.arange(7, dtype=np.float32))
    assert arm.calls == 3


def test_feedback_reader_rejects_persistent_communication_or_angle_fault() -> None:
    persistent = RosFeedbackReader(
        FixedFreshTracker(),
        SequenceArmHealth([ArmStatusHealthError(63, 0.75)]),
        communication_grace_s=0.75,
    )
    with pytest.raises(ArmStatusHealthError, match="communication"):
        persistent.read()

    angle = RosFeedbackReader(
        FixedFreshTracker(),
        SequenceArmHealth([ArmStatusHealthError(1 << 8, 0.0)]),
        communication_grace_s=0.75,
    )
    with pytest.raises(ArmStatusHealthError, match="angle_limit"):
        angle.read()


def test_s_arms_takeover_without_auto_triggering_pika_teleop(tmp_path: Path) -> None:
    clock = StepClock()
    teleop = FakeTeleopController()
    logger_path = tmp_path / "episode.jsonl"
    config = TakeoverRuntimeConfig(output_dir=tmp_path, episode_id="ep_teleop_arm", publish_commands=False)

    with RLTEpisodeLogger(logger_path, episode_id="ep_teleop_arm") as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({}),
            policy_worker=FakePolicyWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({0: ["s"], 4: ["e"], 5: ["1"]}),
            image_writer=FakeImageWriter(),
            logger=logger,
            teleop_controller=teleop,
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        result = core.run_steps(max_steps=10)

    assert result["outcome"] == "episode_done"
    # Arming may explicitly request the already-off state, but must never
    # programmatically activate teleop; activation belongs to the user's
    # physical double-click.
    assert not any(teleop.calls)
    rows = [json.loads(line) for line in logger_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["keyboard"]["mode"] == "TAKEOVER_ARMED"
    assert rows[0]["source"] == "safety_block"
    assert rows[0]["policy_metadata"]["replay_include"] is False


def test_s_does_not_accept_an_already_continuous_pika_command_stream(tmp_path: Path) -> None:
    """A stream that predates `s` is not the demonstrator's post-arm double-click edge."""

    clock = StepClock()
    logger_path = tmp_path / "episode.jsonl"
    config = TakeoverRuntimeConfig(output_dir=tmp_path, episode_id="ep_existing_pika", publish_commands=False)

    with RLTEpisodeLogger(logger_path, episode_id=config.episode_id) as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            # The stream is already present during the MODEL row, before s.
            human_tracker=FakeHumanTracker(),
            policy_worker=FakePolicyWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({1: ["s"], 4: ["0"]}),
            image_writer=FakeImageWriter(),
            logger=logger,
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        result = core.run_steps(max_steps=6)

    rows = [json.loads(line) for line in logger_path.read_text(encoding="utf-8").splitlines()]
    assert result["outcome"] == "episode_done"
    assert [row["keyboard"]["mode"] for row in rows] == [
        "MODEL",
        "TAKEOVER_ARMED",
        "TAKEOVER_ARMED",
        "TAKEOVER_ARMED",
        "STOPPED",
    ]
    assert [row["source"] for row in rows] == [
        "pi05",
        "safety_block",
        "safety_block",
        "safety_block",
        "stop",
    ]
    # The pre-arm MODEL row remains valid Pi0.5 replay.  Only ARMED/terminal
    # rows are excluded because no new physical takeover edge occurred.
    assert rows[0]["policy_metadata"]["replay_include"] is True
    assert all(row["policy_metadata"]["replay_include"] is False for row in rows[1:])
    assert all(row["a_human"] is None for row in rows)
    assert all(row["takeover_started_t"] is None for row in rows)


def test_explicit_post_s_teleop_off_to_on_edge_starts_human_control(tmp_path: Path) -> None:
    """Continuous commands become eligible only after the physical teleop off->on edge."""

    clock = StepClock()
    logger_path = tmp_path / "episode.jsonl"
    key_source = FakeKeySource({1: ["s"], 4: ["1"]})
    teleop = ScheduledTeleopController({0: False, 1: False, 2: True}, step_source=key_source)
    config = TakeoverRuntimeConfig(output_dir=tmp_path, episode_id="ep_teleop_edge", publish_commands=False)

    with RLTEpisodeLogger(logger_path, episode_id=config.episode_id) as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=FakeHumanTracker(),
            policy_worker=FakePolicyWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=key_source,
            image_writer=FakeImageWriter(),
            logger=logger,
            teleop_controller=teleop,
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        result = core.run_steps(max_steps=6)

    rows = [json.loads(line) for line in logger_path.read_text(encoding="utf-8").splitlines()]
    assert result["outcome"] == "episode_done"
    assert [row["keyboard"]["mode"] for row in rows] == [
        "MODEL",
        "TAKEOVER_ARMED",
        "HUMAN",
        "HUMAN",
        "STOPPED",
    ]
    assert rows[1]["source"] == "safety_block"
    assert rows[1]["policy_metadata"]["replay_include"] is False
    assert rows[2]["source"] == "human_pika"
    # The edge command itself has no post-takeover policy observation/Token yet.
    assert rows[2]["policy_metadata"]["replay_include"] is False
    assert rows[3]["policy_metadata"]["replay_include"] is True
    assert rows[2]["takeover_started_t"] == 2
    assert not any(teleop.set_calls)


def test_fresh_pika_command_starts_takeover_recording_and_command_loss_ends_motion(tmp_path: Path) -> None:
    clock = StepClock()
    logger_path = tmp_path / "episode.jsonl"
    human_value = np.full(7, 2.0, dtype=np.float32)
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_double_click_takeover",
        publish_commands=False,
        human_end_timeout_s=0.05,
    )

    with RLTEpisodeLogger(logger_path, episode_id="ep_double_click_takeover") as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({2: human_value, 3: human_value}),
            policy_worker=FakePolicyWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({0: ["s"], 7: ["1"]}),
            image_writer=FakeImageWriter(),
            logger=logger,
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        result = core.run_steps(max_steps=10)

    rows = [json.loads(line) for line in logger_path.read_text(encoding="utf-8").splitlines()]
    assert result["outcome"] == "episode_done"
    assert [row["keyboard"]["mode"] for row in rows[:8]] == [
        "TAKEOVER_ARMED",
        "TAKEOVER_ARMED",
        "HUMAN",
        "HUMAN",
        "HUMAN",
        "WAIT_FOR_REWARD",
        "WAIT_FOR_REWARD",
        "STOPPED",
    ]
    assert rows[2]["source"] == "human_pika"
    # The command that first flips TAKEOVER_ARMED -> HUMAN arrives after this
    # step's policy request point, so it is deliberately excluded until a
    # post-takeover observation/Token is available on the next row.
    assert rows[2]["policy_metadata"]["replay_include"] is False
    assert rows[2]["policy_metadata"]["policy_replay_alignment_ready"] is False
    assert rows[3]["policy_metadata"]["replay_include"] is True
    assert rows[3]["policy_metadata"]["policy_replay_alignment_ready"] is True
    assert rows[0]["policy_metadata"]["replay_include"] is False
    assert rows[5]["policy_metadata"]["replay_include"] is False
    assert rows[2]["takeover_started_t"] == 2
    assert rows[5]["takeover_ended_t"] == 5
    assert rows[-1]["done"] is True
    assert rows[-1]["reward"] == 1.0


def test_reward_key_can_finish_model_only_episode_and_marks_model_rows_for_replay(tmp_path: Path) -> None:
    clock = StepClock()
    logger_path = tmp_path / "episode.jsonl"
    config = TakeoverRuntimeConfig(output_dir=tmp_path, episode_id="ep_model_reward", publish_commands=False)

    with RLTEpisodeLogger(logger_path, episode_id="ep_model_reward") as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({}),
            policy_worker=FakePolicyWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({2: ["1"]}),
            image_writer=FakeImageWriter(),
            logger=logger,
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        result = core.run_steps(max_steps=10)

    rows = [json.loads(line) for line in logger_path.read_text(encoding="utf-8").splitlines()]
    assert result["outcome"] == "episode_done"
    assert result["steps"] == 3
    assert result["terminal_reward"] == 1.0
    assert [row["source"] for row in rows] == ["pi05", "pi05", "stop"]
    assert rows[0]["policy_metadata"]["replay_include"] is True
    assert rows[1]["policy_metadata"]["replay_include"] is True
    assert rows[2]["policy_metadata"]["replay_include"] is False
    assert rows[2]["done"] is True
    assert rows[2]["reward"] == 1.0


def test_phase_gate_enters_once_and_stays_active_until_reward(tmp_path: Path) -> None:
    clock = StepClock()
    logger_path = tmp_path / "episode.jsonl"
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_phase_gate",
        publish_commands=False,
        phase_enter_frames=2,
    )

    with RLTEpisodeLogger(logger_path, episode_id="ep_phase_gate") as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({}),
            policy_worker=FakePolicyWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({5: ["1"]}),
            image_writer=FakeImageWriter(),
            logger=logger,
            phase_gate=SingleLatchPhaseGate(PhaseGateConfig(enter_threshold=0.5, enter_consecutive_frames=2)),
            phase_classifier=FakePhaseClassifier([0.1, 0.8, 0.9, 0.0, 0.0, 0.0]),
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        result = core.run_steps(max_steps=10)

    rows = [json.loads(line) for line in logger_path.read_text(encoding="utf-8").splitlines()]
    assert result["outcome"] == "episode_done"
    assert [row["gate_active"] for row in rows] == [False, False, True, True, True, False]
    assert [row["gate_state"] for row in rows] == ["IDLE", "IDLE", "ACTIVE", "ACTIVE", "ACTIVE", "EXITED"]
    assert [row["gate_enter_t"] for row in rows] == [None, None, 2, 2, 2, 2]
    assert rows[5]["gate_exit_t"] == 5
    assert rows[5]["gate_reason"] == "episode_done"
    assert rows[3]["phase_probability"] == 0.0
    assert rows[3]["policy_metadata"]["phase_gate_reason"] == "locked_until_terminal"
    assert rows[3]["policy_metadata"]["phase_gate_enter_t"] == 2


def test_actor_shadow_never_replaces_pi05_or_human_even_when_gate_is_active(tmp_path: Path) -> None:
    clock = StepClock()
    logger_path = tmp_path / "episode.jsonl"
    publisher = FakePublisher()
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_actor_shadow",
        publish_commands=True,
        publish_authorization=PUBLISH_AUTHORIZATION,
        actor_shadow=True,
    )

    with RLTEpisodeLogger(logger_path, episode_id="ep_actor_shadow") as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker(
                {
                    2: np.full(7, 2.0, dtype=np.float32),
                    3: np.full(7, 2.0, dtype=np.float32),
                }
            ),
            policy_worker=FakeShadowPolicyWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({1: ["s"], 4: ["1"]}),
            image_writer=FakeImageWriter(),
            logger=logger,
            command_publisher=publisher,
            phase_gate=SingleLatchPhaseGate(PhaseGateConfig(enter_threshold=0.5, enter_consecutive_frames=1)),
            phase_classifier=FakePhaseClassifier([0.99, 0.99, 0.99]),
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        result = core.run_steps(max_steps=5)

    rows = [json.loads(line) for line in logger_path.read_text(encoding="utf-8").splitlines()]
    assert result["published_commands"] == 5
    np.testing.assert_allclose(publisher.commands[0], np.ones(7))
    np.testing.assert_allclose(publisher.commands[1], np.zeros(7))
    np.testing.assert_allclose(publisher.commands[2], np.full(7, 2.0))
    np.testing.assert_allclose(publisher.commands[3], np.full(7, 2.0))
    np.testing.assert_allclose(publisher.commands[4], np.zeros(7))
    assert [row["source"] for row in rows] == [
        "pi05",
        "safety_block",
        "human_pika",
        "human_pika",
        "stop",
    ]
    assert rows[0]["gate_active"] is True
    expected_shadow = np.ones((10, 7), dtype=np.float32)
    expected_shadow[:, 0] += RANK1_BUMP_WINDOW * 0.002
    np.testing.assert_allclose(rows[0]["a_actor"], expected_shadow)
    # Shadow evaluation now applies the same 0.02-rad entry boundary contract
    # as live control.  This synthetic policy jumps from zero to one radian, so
    # the safe shadow payload falls back to the behavior reference.
    np.testing.assert_allclose(rows[0]["a_actor_safe"], np.ones((10, 7)))
    assert rows[0]["policy_metadata"]["actor_governor_rejection_reason"] == (
        "boundary_jump_exceeds_limit"
    )
    assert rows[0]["policy_metadata"]["actor_governor_evaluated_in_shadow"] is True
    assert len(rows[0]["z_rl"]) == 2048
    assert rows[0]["policy_metadata"]["actor_shadow_control_source"] == "pi05_or_human_only"
    assert rows[0]["policy_metadata"]["actor_shadow_would_activate"] is True


def test_actor_shadow_keeps_pure_sft_execute50_outside_live_actor_phase(tmp_path: Path) -> None:
    clock = StepClock()
    logger_path = tmp_path / "episode.jsonl"
    publisher = FakePublisher()
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_sft_execute50",
        publish_commands=True,
        publish_authorization=PUBLISH_AUTHORIZATION,
        actor_shadow=True,
        model_execute_steps=50,
    )

    with RLTEpisodeLogger(logger_path, episode_id="ep_sft_execute50") as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({}),
            policy_worker=FakeShadowPolicyWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({}),
            image_writer=FakeImageWriter(),
            logger=logger,
            command_publisher=publisher,
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        core.run_steps(max_steps=12)

    rows = [json.loads(line) for line in logger_path.read_text(encoding="utf-8").splitlines()]
    assert [row["source"] for row in rows] == ["pi05"] * 12
    assert [row["policy_metadata"]["model_action_index"] for row in rows] == list(range(12))
    assert {row["policy_metadata"]["model_execute_steps"] for row in rows} == {50}
    assert {row["policy_metadata"]["effective_execute_steps"] for row in rows} == {50}
    assert rows[9]["a_actor"] is not None
    assert rows[10]["a_actor"] is None
    np.testing.assert_allclose(publisher.commands, np.ones((12, 7), dtype=np.float32))


def test_pure_sft_prefetches_five_steps_early_and_switches_only_at_h50_boundary(tmp_path: Path) -> None:
    clock = StepClock()
    logger_path = tmp_path / "episode.jsonl"
    policy_worker = FakeShadowPolicyWorker()
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_native_boundary_planning",
        publish_commands=False,
        actor_shadow=True,
        model_execute_steps=50,
    )

    with RLTEpisodeLogger(logger_path, episode_id="ep_native_boundary_planning") as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({}),
            policy_worker=policy_worker,
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({}),
            image_writer=FakeImageWriter(),
            logger=logger,
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        core.run_steps(max_steps=55)

    rows = [json.loads(line) for line in logger_path.read_text(encoding="utf-8").splitlines()]
    assert len(policy_worker.submitted) == 2
    assert [submission[2] for submission in policy_worker.submitted] == [0, 45]
    assert [row["policy_metadata"]["model_action_index"] for row in rows] == [*range(50), *range(5)]
    assert rows[49]["policy_metadata"].get("h50_prefetch_accepted") is None
    assert rows[50]["policy_metadata"]["h50_prefetch_accepted"] is True
    assert rows[50]["policy_metadata"]["h50_handoff_actor_suppressed"] is True
    assert rows[50]["a_actor"] is None


def test_split_enrichment_lane_cannot_block_control_critical_h50_prefetch(
    tmp_path: Path,
) -> None:
    clock = StepClock()
    logger_path = tmp_path / "episode.jsonl"
    base_worker = FakeShadowPolicyWorker()
    enrichment_worker = NeverReadyPolicyWorker()
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_split_policy_workers",
        publish_commands=False,
        actor_shadow=True,
        model_execute_steps=50,
        phase_enter_frames=1,
    )

    with RLTEpisodeLogger(logger_path, episode_id=config.episode_id) as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({}),
            policy_worker=base_worker,
            enrichment_policy_worker=enrichment_worker,
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({}),
            image_writer=FakeImageWriter(),
            logger=logger,
            phase_gate=SingleLatchPhaseGate(
                PhaseGateConfig(
                    enter_threshold=0.5,
                    enter_consecutive_frames=1,
                )
            ),
            phase_classifier=FakePhaseClassifier([1.0]),
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        core.run_steps(max_steps=55)

    rows = [
        json.loads(line)
        for line in logger_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [submission[2] for submission in base_worker.submitted] == [0, 45]
    assert len(enrichment_worker.submitted) == 1
    assert enrichment_worker.submitted[0][2] == 2
    assert "safety_block" not in [row["source"] for row in rows]
    assert rows[50]["policy_metadata"]["h50_prefetch_accepted"] is True
    assert rows[50]["policy_metadata"]["policy_worker_lane"] == "base"
    assert rows[50]["policy_metadata"]["policy_workers_split"] is True


def test_split_lanes_send_base_only_and_exact_actor_enrichment_protocols(
    tmp_path: Path,
) -> None:
    clock = StepClock()
    logger_path = tmp_path / "episode.jsonl"
    base_worker = ProtocolBaseOnlyWorker()
    enrichment_worker = ProtocolActorEnrichmentWorker()
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_split_protocols",
        publish_commands=False,
        actor_shadow=True,
        phase_enter_frames=1,
    )

    with RLTEpisodeLogger(logger_path, episode_id=config.episode_id) as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({}),
            policy_worker=base_worker,
            enrichment_policy_worker=enrichment_worker,
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({}),
            image_writer=FakeImageWriter(),
            logger=logger,
            phase_gate=SingleLatchPhaseGate(
                PhaseGateConfig(
                    enter_threshold=0.5,
                    enter_consecutive_frames=1,
                )
            ),
            phase_classifier=FakePhaseClassifier([1.0] * 12),
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        core.run_steps(max_steps=12)

    assert base_worker.submitted
    assert all(
        observation["rlt/base_only_mode"] == "base_only_v1"
        for observation, _, _ in base_worker.submitted
    )
    assert enrichment_worker.submitted
    for observation, _, _ in enrichment_worker.submitted:
        assert (
            observation["rlt/actor_only_mode"]
            == "actor_enrichment_only_v1"
        )
        assert "rlt/actor_only_z_rl" not in observation
        assert np.asarray(observation["rlt/behavior_ref"]).shape == (10, 7)
    rows = [
        json.loads(line)
        for line in logger_path.read_text(encoding="utf-8").splitlines()
    ]
    actor_only_rows = [
        row
        for row in rows
        if row["policy_metadata"].get("actor_only_mode") is True
    ]
    assert actor_only_rows
    assert all(
        row["policy_metadata"]["actor_only_base_policy_called"] is False
        and row["policy_metadata"]["actor_only_base_rng_advanced"] is False
        for row in actor_only_rows
    )


def test_persistent_actor_carry_covers_phase_across_c10_and_h50(
    tmp_path: Path,
) -> None:
    clock = StepClock()
    logger_path = tmp_path / "episode.jsonl"
    checkpoint = tmp_path / "phase_classifier.pt"
    checkpoint.write_bytes(b"test")
    publisher = FakePublisher()
    base_worker = ProtocolBaseOnlyWorker()
    enrichment_worker = ProtocolActorEnrichmentWorker()
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_persistent_full_h50",
        publish_commands=True,
        publish_authorization=PUBLISH_AUTHORIZATION,
        actor_shadow=True,
        actor_live=True,
        actor_live_authorization=ACTOR_LIVE_AUTHORIZATION,
        actor_live_max_boundary_jump_rad=10.0,
        actor_live_max_chunks=0,
        actor_execution_profile=PERSISTENT_C10_EXECUTION_CONTRACT,
        phase_classifier_checkpoint=checkpoint,
        phase_enter_frames=1,
        model_execute_steps=50,
    )

    with RLTEpisodeLogger(logger_path, episode_id=config.episode_id) as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({}),
            policy_worker=base_worker,
            enrichment_policy_worker=enrichment_worker,
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({}),
            image_writer=FakeImageWriter(),
            logger=logger,
            command_publisher=publisher,
            phase_gate=SingleLatchPhaseGate(
                PhaseGateConfig(
                    enter_threshold=0.5,
                    enter_consecutive_frames=1,
                )
            ),
            phase_classifier=FakePhaseClassifier([1.0] * 55),
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        result = core.run_steps(max_steps=55)

    rows = [
        json.loads(line)
        for line in logger_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 55
    assert [row["source"] for row in rows] == ["rlt"] * 55
    assert all(
        row["policy_metadata"]["actor_execution_profile"]
        == PERSISTENT_C10_EXECUTION_CONTRACT
        for row in rows
    )
    assert all(
        row["policy_metadata"]["actor_governor_safe_residual_this_step"]
        is not None
        for row in rows
    )
    assert rows[0]["policy_metadata"]["actor_persistent_hold"] is True
    assert rows[0]["policy_metadata"]["actor_persistent_zero_carry_hold"] is True
    assert any(
        max(
            abs(float(value))
            for value in row["policy_metadata"][
                "actor_governor_safe_residual_this_step"
            ][:6]
        )
        > 0.0
        for row in rows[10:]
    )
    h50_rows = [
        row
        for row in rows
        if row["policy_metadata"].get("h50_prefetch_accepted") is True
        and row["policy_metadata"].get("model_action_index") == 0
    ]
    assert h50_rows
    assert h50_rows[0]["policy_metadata"]["actor_persistent_hold"] is True
    assert h50_rows[0]["policy_metadata"][
        "h50_handoff_actor_replaced_by_persistent_hold"
    ] is True
    assert result["actor_live_chunks_started"] >= 5


def test_filtered_actual_v2_commits_post_lowpass_commands_with_canonical_metadata(
    tmp_path: Path,
) -> None:
    clock = StepClock()
    logger_path = tmp_path / "episode.jsonl"
    checkpoint = tmp_path / "phase_classifier.pt"
    checkpoint.write_bytes(b"test")
    publisher = FakePublisher()
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_filtered_actual_v2",
        publish_commands=True,
        publish_authorization=PUBLISH_AUTHORIZATION,
        actor_shadow=True,
        actor_live=True,
        actor_live_authorization=ACTOR_LIVE_AUTHORIZATION,
        actor_live_max_boundary_jump_rad=0.06,
        actor_live_max_chunks=0,
        actor_execution_profile=PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT,
        execution_action_schema_fingerprint=(
            PERSISTENT_FILTERED_ACTUAL_ACTION_SCHEMA_FINGERPRINT
        ),
        phase_classifier_checkpoint=checkpoint,
        phase_enter_frames=1,
        model_execute_steps=50,
    )
    safety = StatefulSafetyFilter(
        HardwareSafetyConfig(
            model_smoothing_tau_s=0.05,
            model_max_joint_step=np.radians(3.0),
            model_max_gripper_step=0.02,
        ),
        np.zeros(7),
    )

    with RLTEpisodeLogger(logger_path, episode_id=config.episode_id) as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({}),
            policy_worker=SmoothProtocolBaseOnlyWorker(),
            enrichment_policy_worker=ProtocolActorEnrichmentWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({}),
            image_writer=FakeImageWriter(),
            logger=logger,
            command_publisher=publisher,
            safety_filter=safety,
            phase_gate=SingleLatchPhaseGate(
                PhaseGateConfig(
                    enter_threshold=0.5,
                    enter_consecutive_frames=1,
                )
            ),
            phase_classifier=FakePhaseClassifier([1.0] * 25),
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        result = core.run_steps(max_steps=25)

    rows = [
        json.loads(line)
        for line in logger_path.read_text(encoding="utf-8").splitlines()
    ]
    committed = [
        row
        for row in rows
        if row["source"] == "rlt"
        and row["policy_metadata"]["actor_execution_committed_this_step"]
    ]
    assert len(committed) == 25
    assert result["actor_live_chunks_completed"] >= 2
    assert any("model_low_pass" in row["policy_metadata"]["safety_reasons"] for row in committed)
    for row in committed:
        meta = row["policy_metadata"]
        assert meta["actor_execution_profile"] == (
            PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT
        )
        assert meta["actor_persistent_commit_status"] == (
            "committed_filtered_actual_after_publish"
        )
        assert meta["actor_filtered_actual_certificate_approved"] is True
        assert meta["actor_execution_filter_alpha"] == pytest.approx(
            1.0 - np.exp(-(1.0 / 30.0) / 0.05)
        )
        assert meta["actor_model_action_schema_fingerprint"] == (
            ACTION_SCHEMA_FINGERPRINT
        )
        assert meta["actor_execution_schema_fingerprint"] == (
            PERSISTENT_FILTERED_ACTUAL_ACTION_SCHEMA_FINGERPRINT
        )
        assert meta["action_schema_fingerprint"] == (
            PERSISTENT_FILTERED_ACTUAL_ACTION_SCHEMA_FINGERPRINT
        )
        assert meta["actor_governor_fingerprint"] == (
            PERSISTENT_FILTERED_ACTUAL_GOVERNOR_FINGERPRINT
        )
        assert meta["actor_execution_boundary_limit_rad"] == pytest.approx(0.06)
        assert meta["actor_execution_projection_scale_steps"] == 33
        assert meta["actor_execution_min_projection_scale"] == pytest.approx(0.2)
        assert meta[
            "actor_execution_direction_static_threshold_rad"
        ] == pytest.approx(0.001)
        base = np.asarray(meta["actor_filtered_base_command"], dtype=np.float32)
        residual = np.asarray(
            meta["actor_filtered_actual_residual"],
            dtype=np.float32,
        )
        actual = np.asarray(meta["actor_filtered_actual_action"], dtype=np.float32)
        np.testing.assert_allclose(actual, base + residual, atol=2e-7)
        np.testing.assert_allclose(actual, row["a_exec"], atol=2e-7)
        assert len(meta["actor_canonical_decision"]) == 7
        assert meta["actor_canonical_decision"][6] == 0.0
        assert 0 <= int(meta["actor_execution_plan_offset"]) < 10


def test_close_assist_v3_commits_and_logs_auditable_gripper_residuals(
    tmp_path: Path,
) -> None:
    assert (
        RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT
        in SUPPORTED_RAW_ACTOR_ACTION_SCHEMA_FINGERPRINTS
    )
    clock = StepClock()
    logger_path = tmp_path / "episode.jsonl"
    checkpoint = tmp_path / "phase_classifier.pt"
    checkpoint.write_bytes(b"test")
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_gripper_close_v3",
        publish_commands=True,
        publish_authorization=PUBLISH_AUTHORIZATION,
        actor_shadow=True,
        actor_live=True,
        actor_live_authorization=ACTOR_LIVE_AUTHORIZATION,
        actor_live_max_boundary_jump_rad=0.06,
        actor_live_max_chunks=0,
        actor_execution_profile=PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT,
        action_schema_fingerprint=(
            RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT
        ),
        actor_projection_profile=(
            RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE
        ),
        execution_action_schema_fingerprint=(
            PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT
        ),
        actor_governor_fingerprint=(
            PERSISTENT_GRIPPER_CLOSE_GOVERNOR_FINGERPRINT
        ),
        actor_gripper_residual_mode=GRIPPER_RESIDUAL_CLOSE_ASSIST,
        phase_classifier_checkpoint=checkpoint,
        phase_enter_frames=1,
        model_execute_steps=50,
    )
    safety = StatefulSafetyFilter(
        HardwareSafetyConfig(
            model_smoothing_tau_s=0.05,
            model_max_joint_step=np.radians(3.0),
            model_max_gripper_step=0.02,
        ),
        np.zeros(7),
    )

    with RLTEpisodeLogger(logger_path, episode_id=config.episode_id) as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({}),
            policy_worker=SmoothProtocolBaseOnlyWorker(),
            enrichment_policy_worker=CloseAssistProtocolActorEnrichmentWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({}),
            image_writer=FakeImageWriter(),
            logger=logger,
            command_publisher=FakePublisher(),
            safety_filter=safety,
            phase_gate=SingleLatchPhaseGate(
                PhaseGateConfig(
                    enter_threshold=0.5,
                    enter_consecutive_frames=1,
                )
            ),
            phase_classifier=FakePhaseClassifier([1.0] * 25),
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        core.run_steps(max_steps=25)

    rows = [
        json.loads(line)
        for line in logger_path.read_text(encoding="utf-8").splitlines()
    ]
    committed = [
        row
        for row in rows
        if row["source"] == "rlt"
        and row["policy_metadata"]["actor_execution_committed_this_step"]
    ]
    assert len(committed) == 25
    assert any(
        float(row["policy_metadata"]["actor_filtered_actual_residual"][6]) < 0.0
        for row in committed
    )
    assert any(
        float(row["policy_metadata"]["actor_persistent_committed_residual"][6])
        < 0.0
        for row in committed
    )
    assert any(
        float(row["policy_metadata"]["actor_canonical_decision"][6])
        == pytest.approx(-0.002)
        for row in committed
    )
    for row in committed:
        meta = row["policy_metadata"]
        assert meta["actor_model_action_schema_fingerprint"] == (
            RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT
        )
        assert meta["actor_execution_schema_fingerprint"] == (
            PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT
        )
        assert meta["actor_governor_fingerprint"] == (
            PERSISTENT_GRIPPER_CLOSE_GOVERNOR_FINGERPRINT
        )
        assert meta["actor_gripper_residual_mode"] == (
            GRIPPER_RESIDUAL_CLOSE_ASSIST
        )
        assert len(meta["actor_canonical_decision"]) == 7
        assert meta["actor_canonical_decision"][6] <= 1e-9
        assert len(meta["actor_persistent_planned_residual"]) == 7
        assert meta["actor_filtered_actual_gripper_residual_max"] <= 0.005 + 1e-9
        assert (
            meta["actor_filtered_actual_gripper_residual_d1_max_m"]
            <= 0.0005 + 1e-9
        )
        assert (
            meta["actor_filtered_actual_gripper_residual_d2_max_m"]
            <= 0.0003 + 1e-9
        )
        assert (
            meta["actor_filtered_actual_gripper_boundary_jump_max_m"]
            <= 0.0005 + 1e-9
        )
        assert meta["execution_gripper_command_min_m"] == pytest.approx(0.0)
        assert meta["execution_gripper_command_max_m"] == pytest.approx(0.08)
        assert (
            meta["execution_gripper_command_min_m"]
            <= meta["actor_filtered_actual_gripper_command_m"]
            <= meta["execution_gripper_command_max_m"]
        )
        assert len(meta["actor_persistent_carry_in"]) == 7
        assert len(meta["actor_persistent_previous_carry"]) == 7
        assert len(meta["actor_persistent_carry_out"]) == 7


def test_filtered_actual_v2_requires_distinct_protocol_and_execution_schemas() -> None:
    with pytest.raises(ValueError, match="execution action schema mismatch"):
        TakeoverRuntimeConfig(
            actor_execution_profile=(
                PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT
            ),
        ).validate()
    with pytest.raises(ValueError, match="runtime action schema mismatch"):
        TakeoverRuntimeConfig(
            actor_execution_profile=(
                PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT
            ),
            action_schema_fingerprint=(
                PERSISTENT_FILTERED_ACTUAL_ACTION_SCHEMA_FINGERPRINT
            ),
            execution_action_schema_fingerprint=(
                PERSISTENT_FILTERED_ACTUAL_ACTION_SCHEMA_FINGERPRINT
            ),
        ).validate()
    config = TakeoverRuntimeConfig(
        actor_execution_profile=PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT,
        actor_live_max_boundary_jump_rad=0.06,
        action_schema_fingerprint=ACTION_SCHEMA_FINGERPRINT,
        execution_action_schema_fingerprint=(
            PERSISTENT_FILTERED_ACTUAL_ACTION_SCHEMA_FINGERPRINT
        ),
    ).validate()
    assert config.action_schema_fingerprint == ACTION_SCHEMA_FINGERPRINT
    assert config.execution_action_schema_fingerprint == (
        PERSISTENT_FILTERED_ACTUAL_ACTION_SCHEMA_FINGERPRINT
    )


def test_filtered_actual_v2_rejection_publishes_only_filtered_base(
    tmp_path: Path,
    monkeypatch,
) -> None:
    clock = StepClock()
    logger_path = tmp_path / "episode.jsonl"
    checkpoint = tmp_path / "phase_classifier.pt"
    checkpoint.write_bytes(b"test")
    publisher = FakePublisher()
    rejected_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    original_certify = (
        PersistentActorResidualGovernor.certify_filtered_execution
    )

    def reject_first_nonzero(self, **kwargs):
        certificate = original_certify(self, **kwargs)
        if (
            not rejected_pairs
            and float(np.max(np.abs(certificate.actual_residual[:6])))
            > 1e-8
        ):
            rejected_pairs.append(
                (
                    certificate.filtered_base_action.copy(),
                    certificate.filtered_actual_action.copy(),
                )
            )
            return dataclasses.replace(
                certificate,
                approved=False,
                rejection_reason="injected_post_filter_violation",
            )
        return certificate

    monkeypatch.setattr(
        PersistentActorResidualGovernor,
        "certify_filtered_execution",
        reject_first_nonzero,
    )
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_filtered_actual_reject",
        publish_commands=True,
        publish_authorization=PUBLISH_AUTHORIZATION,
        actor_shadow=True,
        actor_live=True,
        actor_live_authorization=ACTOR_LIVE_AUTHORIZATION,
        actor_live_max_boundary_jump_rad=0.06,
        actor_execution_profile=PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT,
        execution_action_schema_fingerprint=(
            PERSISTENT_FILTERED_ACTUAL_ACTION_SCHEMA_FINGERPRINT
        ),
        phase_classifier_checkpoint=checkpoint,
        phase_enter_frames=1,
    )
    with RLTEpisodeLogger(logger_path, episode_id=config.episode_id) as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({}),
            policy_worker=SmoothProtocolBaseOnlyWorker(),
            enrichment_policy_worker=ProtocolActorEnrichmentWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({}),
            image_writer=FakeImageWriter(),
            logger=logger,
            command_publisher=publisher,
            safety_filter=StatefulSafetyFilter(
                HardwareSafetyConfig(),
                np.zeros(7),
            ),
            phase_gate=SingleLatchPhaseGate(
                PhaseGateConfig(
                    enter_threshold=0.5,
                    enter_consecutive_frames=1,
                )
            ),
            phase_classifier=FakePhaseClassifier([1.0] * 55),
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        core.run_steps(max_steps=55)

    assert rejected_pairs
    rows = [
        json.loads(line)
        for line in logger_path.read_text(encoding="utf-8").splitlines()
    ]
    rejected_index = next(
        index
        for index, row in enumerate(rows)
        if row["policy_metadata"]["actor_filtered_actual_rejected"]
    )
    base, rejected_actor = rejected_pairs[0]
    np.testing.assert_allclose(publisher.commands[rejected_index], base, atol=1e-7)
    assert not np.allclose(
        publisher.commands[rejected_index],
        rejected_actor,
        atol=1e-8,
        rtol=0.0,
    )
    row = rows[rejected_index]
    assert row["source"] == "pi05"
    assert row["policy_metadata"]["actor_requested_source"] == "rlt"
    assert row["policy_metadata"]["actor_delivered_source"] == "pi05"
    assert row["policy_metadata"]["actor_execution_committed_this_step"] is False


def test_filtered_actual_v2_never_commits_before_successful_publish(
    tmp_path: Path,
    monkeypatch,
) -> None:
    clock = StepClock()
    checkpoint = tmp_path / "phase_classifier.pt"
    checkpoint.write_bytes(b"test")
    mark_calls = []
    original_mark = PersistentActorResidualGovernor.mark_filtered_executed

    def record_mark(self, certificate):
        mark_calls.append((certificate.plan_id, certificate.offset))
        return original_mark(self, certificate)

    monkeypatch.setattr(
        PersistentActorResidualGovernor,
        "mark_filtered_executed",
        record_mark,
    )
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_filtered_publish_failure",
        publish_commands=True,
        publish_authorization=PUBLISH_AUTHORIZATION,
        actor_shadow=True,
        actor_live=True,
        actor_live_authorization=ACTOR_LIVE_AUTHORIZATION,
        actor_live_max_boundary_jump_rad=0.06,
        actor_execution_profile=PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT,
        execution_action_schema_fingerprint=(
            PERSISTENT_FILTERED_ACTUAL_ACTION_SCHEMA_FINGERPRINT
        ),
        phase_classifier_checkpoint=checkpoint,
        phase_enter_frames=1,
    )
    logger_path = tmp_path / "episode.jsonl"
    with RLTEpisodeLogger(logger_path, episode_id=config.episode_id) as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({}),
            policy_worker=SmoothProtocolBaseOnlyWorker(),
            enrichment_policy_worker=ProtocolActorEnrichmentWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({}),
            image_writer=FakeImageWriter(),
            logger=logger,
            command_publisher=FailingPublisher(),
            safety_filter=StatefulSafetyFilter(
                HardwareSafetyConfig(),
                np.zeros(7),
            ),
            phase_gate=SingleLatchPhaseGate(
                PhaseGateConfig(
                    enter_threshold=0.5,
                    enter_consecutive_frames=1,
                )
            ),
            phase_classifier=FakePhaseClassifier([1.0]),
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        with pytest.raises(RuntimeError, match="injected publish failure"):
            core.run_steps(max_steps=1)
    assert mark_calls == []


def test_persistent_actor_does_not_commit_before_failed_publish(
    tmp_path: Path,
    monkeypatch,
) -> None:
    clock = StepClock()
    checkpoint = tmp_path / "phase_classifier.pt"
    checkpoint.write_bytes(b"test")
    mark_calls = []
    original_mark_executed = PersistentActorResidualGovernor.mark_executed

    def record_mark_executed(self, plan_id, offset):
        mark_calls.append((plan_id, offset))
        return original_mark_executed(self, plan_id, offset)

    monkeypatch.setattr(
        PersistentActorResidualGovernor,
        "mark_executed",
        record_mark_executed,
    )
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_publish_failure_no_commit",
        publish_commands=True,
        publish_authorization=PUBLISH_AUTHORIZATION,
        actor_shadow=True,
        actor_live=True,
        actor_live_authorization=ACTOR_LIVE_AUTHORIZATION,
        actor_live_max_boundary_jump_rad=10.0,
        actor_execution_profile=PERSISTENT_C10_EXECUTION_CONTRACT,
        phase_classifier_checkpoint=checkpoint,
        phase_enter_frames=1,
    )
    logger_path = tmp_path / "episode.jsonl"
    with RLTEpisodeLogger(logger_path, episode_id=config.episode_id) as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({}),
            policy_worker=ProtocolBaseOnlyWorker(),
            enrichment_policy_worker=ProtocolActorEnrichmentWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({}),
            image_writer=FakeImageWriter(),
            logger=logger,
            command_publisher=FailingPublisher(),
            phase_gate=SingleLatchPhaseGate(
                PhaseGateConfig(
                    enter_threshold=0.5,
                    enter_consecutive_frames=1,
                )
            ),
            phase_classifier=FakePhaseClassifier([1.0]),
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        with pytest.raises(RuntimeError, match="injected publish failure"):
            core.run_steps(max_steps=1)

    assert mark_calls == []


def test_persistent_actor_phase_exit_resets_carry_before_reentry(
    tmp_path: Path,
) -> None:
    clock = StepClock()
    logger_path = tmp_path / "episode.jsonl"
    checkpoint = tmp_path / "phase_classifier.pt"
    checkpoint.write_bytes(b"test")
    active_by_t = {
        **{t: True for t in range(20)},
        **{t: True for t in range(21, 36)},
    }
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_phase_exit_reentry",
        publish_commands=True,
        publish_authorization=PUBLISH_AUTHORIZATION,
        actor_shadow=True,
        actor_live=True,
        actor_live_authorization=ACTOR_LIVE_AUTHORIZATION,
        actor_live_max_boundary_jump_rad=10.0,
        actor_execution_profile=PERSISTENT_C10_EXECUTION_CONTRACT,
        phase_classifier_checkpoint=checkpoint,
        phase_enter_frames=1,
    )

    with RLTEpisodeLogger(logger_path, episode_id=config.episode_id) as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({}),
            policy_worker=ProtocolBaseOnlyWorker(),
            enrichment_policy_worker=ProtocolActorEnrichmentWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({}),
            image_writer=FakeImageWriter(),
            logger=logger,
            command_publisher=FakePublisher(),
            phase_gate=ScheduledPhaseGate(active_by_t),
            phase_classifier=FakePhaseClassifier([1.0] * 36),
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        core.run_steps(max_steps=36)

    rows = [
        json.loads(line)
        for line in logger_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        max(abs(float(value)) for value in row["policy_metadata"][
            "actor_persistent_committed_residual"
        ][:6]) > 0.0
        for row in rows[10:20]
    )
    exit_row = rows[20]
    assert exit_row["policy_metadata"]["actor_persistent_phase_exit_reset"] is True
    assert exit_row["policy_metadata"]["actor_persistent_committed_residual"] == [
        0.0
    ] * 7
    first_reentry_actor_row = next(
        row for row in rows[21:] if row["source"] == "rlt"
    )
    assert first_reentry_actor_row["policy_metadata"][
        "actor_governor_safe_residual_this_step"
    ][:6] == [0.0] * 6


def test_h50_base_priority_suppresses_new_enrichment_inside_guard_window(
    tmp_path: Path,
) -> None:
    clock = StepClock()
    logger_path = tmp_path / "episode.jsonl"
    checkpoint = tmp_path / "phase_classifier.pt"
    checkpoint.write_bytes(b"test")
    base_worker = ProtocolBaseOnlyWorker()
    enrichment_worker = ProtocolActorEnrichmentWorker()
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_base_priority_guard",
        publish_commands=True,
        publish_authorization=PUBLISH_AUTHORIZATION,
        actor_shadow=True,
        actor_live=True,
        actor_live_authorization=ACTOR_LIVE_AUTHORIZATION,
        actor_live_max_boundary_jump_rad=10.0,
        actor_execution_profile=PERSISTENT_C10_EXECUTION_CONTRACT,
        phase_classifier_checkpoint=checkpoint,
        phase_enter_frames=1,
        actor_prefetch_lead_steps=8,
    )
    active_by_t = {t: True for t in range(37, 55)}

    with RLTEpisodeLogger(logger_path, episode_id=config.episode_id) as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({}),
            policy_worker=base_worker,
            enrichment_policy_worker=enrichment_worker,
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({}),
            image_writer=FakeImageWriter(),
            logger=logger,
            command_publisher=FakePublisher(),
            phase_gate=ScheduledPhaseGate(active_by_t),
            phase_classifier=FakePhaseClassifier([1.0] * 55),
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        core.run_steps(max_steps=55)

    rows = [
        json.loads(line)
        for line in logger_path.read_text(encoding="utf-8").splitlines()
    ]
    guarded_rows = rows[37:50]
    assert any(
        row["policy_metadata"][
            "actor_enrichment_suppressed_base_priority"
        ]
        for row in guarded_rows
    )
    assert {
        row["policy_metadata"]["base_priority_guard_steps"]
        for row in guarded_rows
    } == {14}
    assert [submission[2] for submission in base_worker.submitted] == [0, 45]
    assert enrichment_worker.submitted
    assert min(submission[2] for submission in enrichment_worker.submitted) >= 50
    assert all(row["source"] == "rlt" for row in guarded_rows)
    assert all(
        row["policy_metadata"]["actor_persistent_hold"] is True
        for row in guarded_rows
    )


def test_human_takeover_resets_nonzero_persistent_carry(
    tmp_path: Path,
) -> None:
    clock = StepClock()
    logger_path = tmp_path / "episode.jsonl"
    checkpoint = tmp_path / "phase_classifier.pt"
    checkpoint.write_bytes(b"test")
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_human_resets_persistent_carry",
        publish_commands=True,
        publish_authorization=PUBLISH_AUTHORIZATION,
        actor_shadow=True,
        actor_live=True,
        actor_live_authorization=ACTOR_LIVE_AUTHORIZATION,
        actor_live_max_boundary_jump_rad=10.0,
        actor_execution_profile=PERSISTENT_C10_EXECUTION_CONTRACT,
        phase_classifier_checkpoint=checkpoint,
        phase_enter_frames=1,
    )
    human_command = np.full(7, 0.5, dtype=np.float32)

    with RLTEpisodeLogger(logger_path, episode_id=config.episode_id) as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({16: human_command}),
            policy_worker=ProtocolBaseOnlyWorker(),
            enrichment_policy_worker=ProtocolActorEnrichmentWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({15: ["s"]}),
            image_writer=FakeImageWriter(),
            logger=logger,
            command_publisher=FakePublisher(),
            phase_gate=SingleLatchPhaseGate(
                PhaseGateConfig(
                    enter_threshold=0.5,
                    enter_consecutive_frames=1,
                )
            ),
            phase_classifier=FakePhaseClassifier([1.0] * 18),
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        core.run_steps(max_steps=18)

    rows = [
        json.loads(line)
        for line in logger_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        max(abs(float(value)) for value in row["policy_metadata"][
            "actor_persistent_committed_residual"
        ][:6]) > 0.0
        for row in rows[10:15]
    )
    human_row = rows[16]
    assert human_row["source"] == "human_pika"
    assert human_row["policy_metadata"]["actor_persistent_commit_status"] == (
        "reset_for_human"
    )
    assert human_row["policy_metadata"]["actor_persistent_committed_residual"] == [
        0.0
    ] * 7
    assert human_row["policy_metadata"][
        "actor_persistent_human_transition_reset"
    ] is True
    assert human_row["policy_metadata"]["actor_enrichment_min_observation_t"] == 17
    assert human_row["policy_metadata"]["actor_enrichment_fresh_after_reset"] is False


def test_human_takeover_resets_close_assist_gripper_carry(
    tmp_path: Path,
) -> None:
    clock = StepClock()
    logger_path = tmp_path / "episode.jsonl"
    checkpoint = tmp_path / "phase_classifier.pt"
    checkpoint.write_bytes(b"test")
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_human_resets_close_gripper",
        publish_commands=True,
        publish_authorization=PUBLISH_AUTHORIZATION,
        actor_shadow=True,
        actor_live=True,
        actor_live_authorization=ACTOR_LIVE_AUTHORIZATION,
        actor_live_max_boundary_jump_rad=0.06,
        actor_execution_profile=PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT,
        action_schema_fingerprint=(
            RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT
        ),
        actor_projection_profile=(
            RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE
        ),
        execution_action_schema_fingerprint=(
            PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT
        ),
        actor_governor_fingerprint=(
            PERSISTENT_GRIPPER_CLOSE_GOVERNOR_FINGERPRINT
        ),
        actor_gripper_residual_mode=GRIPPER_RESIDUAL_CLOSE_ASSIST,
        phase_classifier_checkpoint=checkpoint,
        phase_enter_frames=1,
    )
    human_command = np.zeros(7, dtype=np.float32)
    human_command[6] = 0.02

    with RLTEpisodeLogger(logger_path, episode_id=config.episode_id) as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({16: human_command}),
            policy_worker=SmoothProtocolBaseOnlyWorker(),
            enrichment_policy_worker=CloseAssistProtocolActorEnrichmentWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({15: ["s"]}),
            image_writer=FakeImageWriter(),
            logger=logger,
            command_publisher=FakePublisher(),
            safety_filter=StatefulSafetyFilter(
                HardwareSafetyConfig(
                    model_smoothing_tau_s=0.05,
                    model_max_joint_step=np.radians(3.0),
                    model_max_gripper_step=0.02,
                ),
                np.zeros(7),
            ),
            phase_gate=SingleLatchPhaseGate(
                PhaseGateConfig(
                    enter_threshold=0.5,
                    enter_consecutive_frames=1,
                )
            ),
            phase_classifier=FakePhaseClassifier([1.0] * 18),
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        core.run_steps(max_steps=18)

    rows = [
        json.loads(line)
        for line in logger_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        float(row["policy_metadata"]["actor_persistent_committed_residual"][6])
        < 0.0
        for row in rows[1:15]
        if row["source"] == "rlt"
    )
    human_row = rows[16]
    assert human_row["source"] == "human_pika"
    assert human_row["policy_metadata"]["actor_persistent_commit_status"] == (
        "reset_for_human"
    )
    assert human_row["policy_metadata"]["actor_persistent_committed_residual"] == [
        0.0
    ] * 7
    assert human_row["policy_metadata"]["actor_persistent_current_residual"] == [
        0.0
    ] * 7
    assert human_row["policy_metadata"]["actor_persistent_previous_residual"] == [
        0.0
    ] * 7
    assert core._last_exec_actor_residual is None
    assert core._previous_exec_actor_residual is None


def test_human_filtered_actual_gap_restarts_canonical_c10_at_zero(
    tmp_path: Path,
) -> None:
    clock = StepClock()
    logger_path = tmp_path / "episode.jsonl"
    checkpoint = tmp_path / "phase_classifier.pt"
    checkpoint.write_bytes(b"test")
    human = np.array([0.1, 0.5, -0.5, 0.0, 0.0, 0.0, 0.02], dtype=np.float32)
    human_schedule = {t: human for t in range(16, 26)}
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_human_filtered_gap",
        publish_commands=True,
        publish_authorization=PUBLISH_AUTHORIZATION,
        actor_shadow=True,
        actor_live=True,
        actor_live_authorization=ACTOR_LIVE_AUTHORIZATION,
        actor_live_max_boundary_jump_rad=0.06,
        actor_execution_profile=PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT,
        execution_action_schema_fingerprint=(
            PERSISTENT_FILTERED_ACTUAL_ACTION_SCHEMA_FINGERPRINT
        ),
        phase_classifier_checkpoint=checkpoint,
        phase_enter_frames=1,
    )
    with RLTEpisodeLogger(logger_path, episode_id=config.episode_id) as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker(human_schedule),
            policy_worker=SmoothProtocolBaseOnlyWorker(),
            enrichment_policy_worker=ProtocolActorEnrichmentWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({15: ["s"]}),
            image_writer=FakeImageWriter(),
            logger=logger,
            command_publisher=FakePublisher(),
            safety_filter=StatefulSafetyFilter(
                HardwareSafetyConfig(),
                np.zeros(7),
            ),
            phase_gate=SingleLatchPhaseGate(
                PhaseGateConfig(
                    enter_threshold=0.5,
                    enter_consecutive_frames=1,
                )
            ),
            phase_classifier=FakePhaseClassifier([1.0] * 26),
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        original_latest = core._latest_policy_command

        def drop_one_human_counterfactual(**kwargs):
            output = original_latest(**kwargs)
            if core.key_source.t == 20:
                return (*output[:4], None)
            return output

        core._latest_policy_command = drop_one_human_counterfactual
        core.run_steps(max_steps=26)

    rows = [
        json.loads(line)
        for line in logger_path.read_text(encoding="utf-8").splitlines()
    ]
    before_gap = rows[19]["policy_metadata"]
    gap = rows[20]["policy_metadata"]
    after_gap = rows[21]["policy_metadata"]
    assert rows[19]["source"] == rows[20]["source"] == rows[21]["source"] == (
        "human_pika"
    )
    assert before_gap["actor_execution_plan_offset"] == 3
    assert gap["human_base_counterfactual_valid"] is False
    assert gap["actor_execution_committed_this_step"] is False
    assert after_gap["human_base_counterfactual_valid"] is True
    assert after_gap["actor_execution_plan_offset"] == 0
    assert (
        after_gap["actor_execution_plan_id"]
        != before_gap["actor_execution_plan_id"]
    )


def test_late_h50_uses_model_keepalive_without_erasing_velocity_history(
    tmp_path: Path,
) -> None:
    clock = StepClock()
    logger_path = tmp_path / "episode.jsonl"
    worker = DelayedSecondPlanWorker(clock=clock, second_delay_s=0.40)
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_late_h50_keepalive",
        publish_commands=False,
        model_execute_steps=50,
    )

    with RLTEpisodeLogger(logger_path, episode_id=config.episode_id) as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({}),
            policy_worker=worker,
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({}),
            image_writer=FakeImageWriter(),
            logger=logger,
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        core.run_steps(max_steps=65)

    rows = [
        json.loads(line)
        for line in logger_path.read_text(encoding="utf-8").splitlines()
    ]
    keepalive_rows = [
        row
        for row in rows
        if row["policy_metadata"].get("h50_boundary_keepalive") is True
    ]
    assert keepalive_rows
    assert {row["source"] for row in keepalive_rows} == {"pi05"}
    assert all(
        row["policy_metadata"]["h50_keepalive_preserves_velocity_history"] is True
        for row in keepalive_rows
    )
    assert "safety_block" not in [row["source"] for row in rows]
    activation = next(
        row
        for row in rows
        if row["t"] > 50
        and row["policy_metadata"].get("model_action_index") == 0
    )
    assert activation["policy_metadata"]["h50_prefetch_accepted"] is True
    assert activation["policy_metadata"][
        "h50_boundary_keepalive_count_before_handoff"
    ] == len(keepalive_rows)
    assert activation["policy_metadata"]["h50_handoff_history_source"] == (
        "executed_model_history_minus_actor_carry"
    )


def test_h50_velocity_handoff_becomes_exact_replay_reference_and_suppresses_stale_actor(
    tmp_path: Path,
) -> None:
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_h50_handoff_contract",
        actor_shadow=True,
        model_execute_steps=50,
    ).validate()
    logger_path = tmp_path / "unused.jsonl"
    with RLTEpisodeLogger(logger_path, episode_id=config.episode_id) as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({}),
            policy_worker=FakeShadowPolicyWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({}),
            image_writer=FakeImageWriter(),
            logger=logger,
        )
        raw = np.zeros((50, 7), dtype=np.float32)
        raw[:, :6] = np.linspace(0.02, 0.08, 50, dtype=np.float32)[:, None]
        raw[:, 6] = 0.065
        core._prefetched_model_plan = raw.copy()
        core._prefetched_model_z_rl = np.ones(2048, dtype=np.float32)
        core._prefetched_model_metadata = {
            "policy_plan_id": "prefetched",
            "policy_observation_timestamp_s": 100.0,
            "actor_shadow_ready": True,
            "actor_behavior_ref_echo_verified": True,
        }
        core._prefetched_model_key = (100.0, 100.05)
        core._last_exec_command = np.full(7, 0.02, dtype=np.float32)
        core._last_exec_command[6] = 0.065
        core._previous_exec_command = core._last_exec_command.copy()
        core._previous_exec_command[:6] -= 0.004
        core._last_exec_source = "pi05"

        activated = core._activate_prefetched_model_plan(
            now_s=100.1334,
            state_snapshot=np.zeros(7, dtype=np.float32),
            chunk_length=10,
        )

    assert activated is True
    assert core._model_plan_limit == 46
    assert core._model_plan_actor is None
    assert core._enrichment_plan_actor is None
    assert core._model_plan_metadata["h50_handoff_actor_suppressed"] is True
    assert core._model_plan_metadata["actor_shadow_ready"] is False
    assert core._model_plan_metadata["actor_behavior_ref_source"] == (
        "h50_velocity_aligned_behavior"
    )
    np.testing.assert_allclose(core._enrichment_plan, core._model_plan)
    np.testing.assert_allclose(core._model_plan[5:], raw[9:])


def test_h50_handoff_removes_close_assist_gripper_carry_but_frozen_stays_legacy(
    tmp_path: Path,
) -> None:
    close_config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_close_handoff",
        actor_live_max_boundary_jump_rad=0.06,
        actor_execution_profile=PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT,
        action_schema_fingerprint=(
            RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT
        ),
        actor_projection_profile=(
            RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE
        ),
        execution_action_schema_fingerprint=(
            PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT
        ),
        actor_governor_fingerprint=(
            PERSISTENT_GRIPPER_CLOSE_GOVERNOR_FINGERPRINT
        ),
        actor_gripper_residual_mode=GRIPPER_RESIDUAL_CLOSE_ASSIST,
    ).validate()
    logger_path = tmp_path / "unused_close_handoff.jsonl"
    with RLTEpisodeLogger(
        logger_path,
        episode_id=close_config.episode_id,
    ) as logger:
        core = TakeoverLoopCore(
            config=close_config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({}),
            policy_worker=FakeShadowPolicyWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({}),
            image_writer=FakeImageWriter(),
            logger=logger,
        )
        first = np.zeros(7, dtype=np.float32)
        first[6] = 0.040
        second = np.zeros(7, dtype=np.float32)
        second[6] = 0.039
        first_residual = np.zeros(7, dtype=np.float32)
        first_residual[6] = -0.002
        second_residual = np.zeros(7, dtype=np.float32)
        second_residual[6] = -0.003
        core._record_handoff_execution(
            first,
            source="rlt",
            is_boundary_keepalive=False,
            actor_residual=first_residual,
        )
        core._record_handoff_execution(
            second,
            source="rlt",
            is_boundary_keepalive=False,
            actor_residual=second_residual,
        )
        handoff, previous, source = core._handoff_base_targets(
            state_snapshot=np.zeros(7, dtype=np.float32)
        )

    assert source == "executed_model_history_minus_actor_carry"
    assert core._last_exec_actor_residual[6] == pytest.approx(-0.003)
    assert core._previous_exec_actor_residual[6] == pytest.approx(-0.002)
    assert handoff[6] == pytest.approx(0.042)
    assert previous[6] == pytest.approx(0.042)

    frozen_config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_frozen_handoff",
    ).validate()
    frozen_logger_path = tmp_path / "unused_frozen_handoff.jsonl"
    with RLTEpisodeLogger(
        frozen_logger_path,
        episode_id=frozen_config.episode_id,
    ) as logger:
        frozen_core = TakeoverLoopCore(
            config=frozen_config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({}),
            policy_worker=FakeShadowPolicyWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({}),
            image_writer=FakeImageWriter(),
            logger=logger,
        )
        frozen_core._record_handoff_execution(
            first,
            source="rlt",
            is_boundary_keepalive=False,
            actor_residual=first_residual,
        )
        frozen_handoff, _, _ = frozen_core._handoff_base_targets(
            state_snapshot=np.zeros(7, dtype=np.float32)
        )

    assert frozen_core._last_exec_actor_residual[6] == 0.0
    assert frozen_handoff[6] == pytest.approx(first[6])


def test_authorized_live_actor_controls_only_inside_latched_phase(tmp_path: Path) -> None:
    clock = StepClock()
    logger_path = tmp_path / "episode.jsonl"
    checkpoint = tmp_path / "phase_classifier.pt"
    checkpoint.write_bytes(b"test")
    publisher = FakePublisher()
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_actor_live",
        publish_commands=True,
        publish_authorization=PUBLISH_AUTHORIZATION,
        actor_shadow=True,
        actor_live=True,
        actor_live_authorization=ACTOR_LIVE_AUTHORIZATION,
        actor_live_max_boundary_jump_rad=10.0,
        phase_classifier_checkpoint=checkpoint,
    )

    with RLTEpisodeLogger(logger_path, episode_id="ep_actor_live") as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({}),
            policy_worker=FakeShadowPolicyWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({1: ["1"]}),
            image_writer=FakeImageWriter(),
            logger=logger,
            command_publisher=publisher,
            phase_gate=SingleLatchPhaseGate(PhaseGateConfig(enter_threshold=0.5, enter_consecutive_frames=1)),
            phase_classifier=FakePhaseClassifier([0.99, 0.99]),
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        result = core.run_steps(max_steps=3)

    rows = [json.loads(line) for line in logger_path.read_text(encoding="utf-8").splitlines()]
    assert result["outcome"] == "episode_done"
    assert rows[0]["source"] == "rlt"
    assert rows[0]["policy_metadata"]["replay_include"] is True
    assert rows[0]["policy_metadata"]["actor_live_enabled"] is True
    # The rank1 envelope is exactly zero at the entry row, so the first live
    # command is continuous with Pi0.5 by construction.
    np.testing.assert_allclose(publisher.commands[0], np.ones(7, dtype=np.float32))


def test_live_actor_synchronously_refreshes_at_c10_boundary(tmp_path: Path) -> None:
    clock = StepClock()
    logger_path = tmp_path / "episode.jsonl"
    checkpoint = tmp_path / "phase_classifier.pt"
    checkpoint.write_bytes(b"test")
    publisher = FakePublisher()
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_actor_c10_boundary",
        publish_commands=True,
        publish_authorization=PUBLISH_AUTHORIZATION,
        actor_shadow=True,
        actor_live=True,
        actor_live_authorization=ACTOR_LIVE_AUTHORIZATION,
        actor_live_max_boundary_jump_rad=10.0,
        phase_classifier_checkpoint=checkpoint,
        model_execute_steps=50,
    )

    with RLTEpisodeLogger(logger_path, episode_id="ep_actor_c10_boundary") as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({}),
            policy_worker=FakeShadowPolicyWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({}),
            image_writer=FakeImageWriter(),
            logger=logger,
            command_publisher=publisher,
            phase_gate=SingleLatchPhaseGate(PhaseGateConfig(enter_threshold=0.5, enter_consecutive_frames=1)),
            phase_classifier=FakePhaseClassifier([0.99] * 11),
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        core.run_steps(max_steps=11)

    rows = [json.loads(line) for line in logger_path.read_text(encoding="utf-8").splitlines()]
    assert [row["source"] for row in rows[:10]] == ["rlt"] * 10
    assert {row["policy_metadata"]["effective_execute_steps"] for row in rows[:10]} == {10}
    assert [row["source"] for row in rows] == ["rlt"] * 11
    # The native SFT behavior plan remains execute_steps=50; only the separate
    # C=10 actor/enrichment plan refreshes at this boundary.
    assert [row["policy_metadata"]["model_action_index"] for row in rows] == list(range(11))
    assert [row["policy_metadata"]["plan_offset"] for row in rows] == [*range(10), 0]
    assert rows[10]["a_actor"] is not None
    assert rows[10]["policy_metadata"]["actor_shadow_ready"] is True
    expected_actor = np.ones((11, 7), dtype=np.float32)
    expected_actor[:10, 0] += RANK1_BUMP_WINDOW * 0.002
    np.testing.assert_allclose(publisher.commands, expected_actor)


def test_async_sft_boundary_result_stages_actor_until_partial_c10_chunk_finishes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "phase_classifier.pt"
    checkpoint.write_bytes(b"test")
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_async_boundary",
        actor_shadow=True,
        actor_live=True,
        actor_live_authorization=ACTOR_LIVE_AUTHORIZATION,
        phase_classifier_checkpoint=checkpoint,
        model_execute_steps=50,
    ).validate()
    logger_path = tmp_path / "unused.jsonl"
    with RLTEpisodeLogger(logger_path, episode_id=config.episode_id) as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({}),
            policy_worker=FakeShadowPolicyWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({}),
            image_writer=FakeImageWriter(),
            logger=logger,
        )
        old_ref = np.full((50, 7), 1.0, dtype=np.float32)
        old_actor = np.full((10, 7), 9.0, dtype=np.float32)
        core._model_plan = old_ref.copy()
        core._model_plan_index = 50
        core._model_plan_limit = 50
        core._install_enrichment_plan(
            plan=old_ref,
            z_rl=np.ones(2048, dtype=np.float32),
            actor=old_actor,
            metadata={"policy_plan_id": "old", "actor_shadow_ready": True},
            key=(0.0, 0.1),
            chunk_length=10,
        )
        # The Actor is already four actions into its C=10 plan when the
        # asynchronous replacement for the exhausted SFT 50-step plan lands.
        core._enrichment_plan_index = 4
        core._policy_request_pending = True
        core._policy_request_reason = "boundary_miss"
        core._policy_request_lead_steps = 0
        replacement = PolicyWorkerOutput(
            value={
                "actions": np.full((50, 7), 2.0, dtype=np.float32),
                "z_rl": np.full(2048, 2.0, dtype=np.float32),
                "a_actor": np.full((10, 7), 8.0, dtype=np.float32),
                "a_actor_action_space": "joint_absolute_gripper_absolute",
            },
            observation_timestamp_s=1.0,
            completed_timestamp_s=1.05,
            policy_plan_id="replacement",
            policy_observation_t=50,
        )

        core._consume_latest_policy_output(
            latest=replacement,
            now_s=1.05,
            state_snapshot=np.zeros(7, dtype=np.float32),
            action_dim=7,
            chunk_length=10,
            required_actions=50,
            actor_phase_active=True,
        )

        # A new H50 behavior boundary invalidates a previously interrupted
        # Actor suffix.  The replacement C10 starts atomically at offset zero;
        # the old partial plan is never resumed or rebased.
        np.testing.assert_allclose(core._model_plan, np.full((50, 7), 2.0))
        assert core._model_plan_index == 0
        np.testing.assert_allclose(core._enrichment_plan_actor, np.full((10, 7), 8.0))
        np.testing.assert_allclose(core._enrichment_plan, np.full((10, 7), 2.0))
        assert core._enrichment_plan_index == 0
        assert core._prefetched_enrichment_actor is None
        assert core._enrichment_plan_metadata["actor_chunk_switch_mode"] == "installed_now"
        assert core._enrichment_plan_metadata["policy_worker_plan_id"] == "replacement"


def test_c10_actor_residual_is_rebased_to_executed_fifty_step_reference(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "phase_classifier.pt"
    checkpoint.write_bytes(b"test")
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_actor_reference_rebase",
        actor_shadow=True,
        actor_live=True,
        actor_live_authorization=ACTOR_LIVE_AUTHORIZATION,
        phase_classifier_checkpoint=checkpoint,
        model_execute_steps=50,
    ).validate()
    logger_path = tmp_path / "unused.jsonl"
    with RLTEpisodeLogger(logger_path, episode_id=config.episode_id) as logger:
        worker = FakePolicyWorker()
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({}),
            policy_worker=worker,
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({}),
            image_writer=FakeImageWriter(),
            logger=logger,
        )
        behavior = np.zeros((50, 7), dtype=np.float32)
        behavior[:, :6] = np.arange(50, dtype=np.float32)[:, None] * 0.001
        behavior[:, 6] = 0.04
        core._model_plan = behavior.copy()
        core._model_plan_index = 10
        core._model_plan_limit = 50
        fresh_reference = np.full((50, 7), 0.5, dtype=np.float32)
        fresh_actor = fresh_reference[:10].copy()
        fresh_actor[:, :6] += 0.01
        fresh_actor[:, 6] += 0.02  # malformed gripper residual must stay frozen
        core._install_enrichment_plan(
            plan=fresh_reference,
            z_rl=np.ones(2048, dtype=np.float32),
            actor=fresh_actor,
            metadata={"policy_plan_id": "fresh", "actor_shadow_ready": True},
            key=(0.0, 0.01),
            chunk_length=10,
        )
        # Ignore FakePolicyWorker's synthetic result so this call exercises the
        # already installed behavior/enrichment pair only.
        core._last_policy_output_key = (0.0, 0.01)

        a_ref, _, a_actor, metadata, model_command = core._latest_policy_command(
            now_s=1.0,
            action_dim=7,
            chunk_length=10,
            state_snapshot=np.zeros(7, dtype=np.float32),
            actor_phase_active=True,
            enrichment_active=True,
        )

    expected_ref = behavior[10:20]
    np.testing.assert_allclose(a_ref, expected_ref)
    assert a_actor is not None
    np.testing.assert_allclose(a_actor, fresh_actor)
    np.testing.assert_allclose(model_command.value, behavior[10])
    assert metadata["actor_reference_rebased"] is False
    assert metadata["actor_reference_mode"] == "behavior_ref_protocol_mismatch"
    assert metadata["actor_behavior_ref_alignment_ready"] is False
    assert metadata["actor_shadow_ready"] is False


def test_live_actor_rejects_entire_discontinuous_c10_plan(tmp_path: Path) -> None:
    clock = StepClock()
    logger_path = tmp_path / "episode.jsonl"
    checkpoint = tmp_path / "phase_classifier.pt"
    checkpoint.write_bytes(b"test")
    publisher = FakePublisher()
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_actor_boundary_guard",
        publish_commands=True,
        publish_authorization=PUBLISH_AUTHORIZATION,
        actor_shadow=True,
        actor_live=True,
        actor_live_authorization=ACTOR_LIVE_AUTHORIZATION,
        actor_live_max_boundary_jump_rad=0.02,
        phase_classifier_checkpoint=checkpoint,
        model_execute_steps=50,
    )

    with RLTEpisodeLogger(logger_path, episode_id=config.episode_id) as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({}),
            policy_worker=JumpingShadowPolicyWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({}),
            image_writer=FakeImageWriter(),
            logger=logger,
            command_publisher=publisher,
            phase_gate=SingleLatchPhaseGate(
                PhaseGateConfig(enter_threshold=0.5, enter_consecutive_frames=1)
            ),
            phase_classifier=FakePhaseClassifier([0.99] * 12),
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        result = core.run_steps(max_steps=12)

    rows = [json.loads(line) for line in logger_path.read_text(encoding="utf-8").splitlines()]
    assert result["actor_live_chunks_started"] == 1
    assert [row["source"] for row in rows[:10]] == ["rlt"] * 10
    assert [row["source"] for row in rows[10:]] == ["pi05"] * 2
    boundary = rows[10]["policy_metadata"]
    assert boundary["actor_live_boundary_checked"] is True
    assert boundary["actor_live_boundary_jump_max_rad"] is None
    assert boundary["actor_governor_projection_scale"] < 0.2
    assert boundary["actor_governor_rejection_reason"] == "projection_scale_below_minimum"
    assert boundary["actor_live_suppression_reason"] == (
        "actor_residual_governor_rejected:projection_scale_below_minimum"
    )
    assert boundary["actor_live_boundary_plan_rejected"] is True
    assert rows[11]["policy_metadata"]["actor_live_suppression_reason"] == (
        "actor_residual_governor_rejected:projection_scale_below_minimum"
    )
    expected_first = np.full((10, 7), 0.01, dtype=np.float32)
    expected_first[:, 0] += RANK1_BUMP_WINDOW * 0.002
    np.testing.assert_allclose(publisher.commands[:10], expected_first)
    np.testing.assert_allclose(publisher.commands[10:], np.full((2, 7), 0.01))


def test_actor_live_one_chunk_canary_falls_back_to_pi05_for_rest_of_episode(tmp_path: Path) -> None:
    clock = StepClock()
    logger_path = tmp_path / "episode.jsonl"
    checkpoint = tmp_path / "phase_classifier.pt"
    checkpoint.write_bytes(b"test")
    publisher = FakePublisher()
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_actor_one_chunk",
        publish_commands=True,
        publish_authorization=PUBLISH_AUTHORIZATION,
        actor_shadow=True,
        actor_live=True,
        actor_live_authorization=ACTOR_LIVE_AUTHORIZATION,
        actor_live_max_boundary_jump_rad=10.0,
        actor_live_max_chunks=1,
        phase_classifier_checkpoint=checkpoint,
        model_execute_steps=50,
    )

    with RLTEpisodeLogger(logger_path, episode_id=config.episode_id) as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({}),
            policy_worker=FakeShadowPolicyWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({}),
            image_writer=FakeImageWriter(),
            logger=logger,
            command_publisher=publisher,
            phase_gate=SingleLatchPhaseGate(PhaseGateConfig(enter_threshold=0.5, enter_consecutive_frames=1)),
            phase_classifier=FakePhaseClassifier([0.99] * 22),
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        result = core.run_steps(max_steps=22)

    rows = [json.loads(line) for line in logger_path.read_text(encoding="utf-8").splitlines()]
    assert [row["source"] for row in rows[:10]] == ["rlt"] * 10
    assert [row["source"] for row in rows[10:]] == ["pi05"] * 12
    expected_actor = np.ones((10, 7), dtype=np.float32)
    expected_actor[:, 0] += RANK1_BUMP_WINDOW * 0.002
    np.testing.assert_allclose(publisher.commands[:10], expected_actor)
    np.testing.assert_allclose(publisher.commands[10:], np.ones((12, 7), dtype=np.float32))
    assert rows[9]["policy_metadata"]["actor_live_chunks_completed"] == 1
    assert rows[10]["policy_metadata"]["actor_live_suppressed"] is True
    assert rows[10]["policy_metadata"]["actor_live_suppression_reason"] == "max_chunks_reached"
    assert rows[10]["policy_metadata"]["mux_reason"] == "rlt_actor_missing_fallback_model"
    assert rows[10]["policy_metadata"]["effective_execute_steps"] == 50
    assert result["actor_live_chunks_started"] == 1
    assert result["actor_live_chunks_completed"] == 1


def test_actor_live_max_chunks_zero_is_unlimited_and_negative_is_rejected() -> None:
    assert TakeoverRuntimeConfig(actor_live_max_chunks=0).validate().actor_live_max_chunks is None
    with pytest.raises(ValueError, match="actor_live_max_chunks"):
        TakeoverRuntimeConfig(actor_live_max_chunks=-1).validate()


def test_actor_live_unlimited_keeps_c10_sequence_gate_and_runs_multiple_chunks(tmp_path: Path) -> None:
    clock = StepClock()
    logger_path = tmp_path / "episode.jsonl"
    checkpoint = tmp_path / "phase_classifier.pt"
    checkpoint.write_bytes(b"test")
    publisher = FakePublisher()
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_actor_unlimited",
        publish_commands=True,
        publish_authorization=PUBLISH_AUTHORIZATION,
        actor_shadow=True,
        actor_live=True,
        actor_live_authorization=ACTOR_LIVE_AUTHORIZATION,
        actor_live_max_boundary_jump_rad=10.0,
        actor_live_max_chunks=0,
        phase_classifier_checkpoint=checkpoint,
        model_execute_steps=50,
    )

    with RLTEpisodeLogger(logger_path, episode_id=config.episode_id) as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({}),
            policy_worker=FakeShadowPolicyWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({}),
            image_writer=FakeImageWriter(),
            logger=logger,
            command_publisher=publisher,
            phase_gate=SingleLatchPhaseGate(PhaseGateConfig(enter_threshold=0.5, enter_consecutive_frames=1)),
            phase_classifier=FakePhaseClassifier([0.99] * 22),
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        result = core.run_steps(max_steps=22)

    rows = [json.loads(line) for line in logger_path.read_text(encoding="utf-8").splitlines()]
    assert [row["source"] for row in rows[:20]] == ["rlt"] * 20
    assert result["actor_live_chunks_started"] >= 2
    assert result["actor_live_chunks_completed"] >= 2
    assert all(
        row["policy_metadata"]["actor_live_suppression_reason"]
        not in {"active_chunk_sequence_mismatch", "awaiting_c10_boundary"}
        for row in rows[:20]
    )


def test_invalid_actor_shadow_does_not_interrupt_pi05_control(tmp_path: Path) -> None:
    clock = StepClock()
    logger_path = tmp_path / "episode.jsonl"
    publisher = FakePublisher()
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_bad_actor",
        publish_commands=True,
        publish_authorization=PUBLISH_AUTHORIZATION,
        actor_shadow=True,
    )

    with RLTEpisodeLogger(logger_path, episode_id="ep_bad_actor") as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({}),
            policy_worker=FakeShadowPolicyWorker(invalid_actor=True),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({1: ["1"]}),
            image_writer=FakeImageWriter(),
            logger=logger,
            command_publisher=publisher,
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        core.run_steps(max_steps=3)

    rows = [json.loads(line) for line in logger_path.read_text(encoding="utf-8").splitlines()]
    np.testing.assert_allclose(publisher.commands[0], np.ones(7))
    assert rows[0]["source"] == "pi05"
    assert rows[0]["a_actor"] is None
    assert "non-finite" in rows[0]["policy_metadata"]["actor_error"]


def test_reward_key_can_finish_active_human_takeover_without_waiting_for_second_double_click(tmp_path: Path) -> None:
    clock = StepClock()
    logger_path = tmp_path / "episode.jsonl"
    human_value = np.full(7, 2.0, dtype=np.float32)
    config = TakeoverRuntimeConfig(output_dir=tmp_path, episode_id="ep_human_reward", publish_commands=False)

    with RLTEpisodeLogger(logger_path, episode_id="ep_human_reward") as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({1: human_value, 2: human_value, 3: human_value}),
            policy_worker=FakePolicyWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({0: ["s"], 3: ["1"]}),
            image_writer=FakeImageWriter(),
            logger=logger,
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        result = core.run_steps(max_steps=10)

    rows = [json.loads(line) for line in logger_path.read_text(encoding="utf-8").splitlines()]
    assert result["outcome"] == "episode_done"
    assert result["steps"] == 4
    assert result["terminal_reward"] == 1.0
    assert [row["keyboard"]["mode"] for row in rows] == ["TAKEOVER_ARMED", "HUMAN", "HUMAN", "STOPPED"]
    assert [row["source"] for row in rows] == ["safety_block", "human_pika", "human_pika", "stop"]
    assert rows[0]["policy_metadata"]["replay_include"] is False
    assert rows[1]["policy_metadata"]["replay_include"] is False
    assert rows[1]["policy_metadata"]["policy_replay_alignment_ready"] is False
    assert rows[2]["policy_metadata"]["replay_include"] is True
    assert rows[2]["policy_metadata"]["policy_replay_alignment_ready"] is True
    assert rows[3]["policy_metadata"]["replay_include"] is False
    assert rows[1]["takeover_started_t"] == 1
    assert rows[3]["takeover_ended_t"] == 3
    assert rows[3]["done"] is True
    assert rows[3]["reward"] == 1.0


def test_shadow_loop_logs_takeover_without_publishing(tmp_path: Path) -> None:
    clock = StepClock()
    key_source = FakeKeySource({0: ["s"], 3: ["s"], 4: ["1"]})
    publisher = FakePublisher()
    logger_path = tmp_path / "episode.jsonl"
    config = TakeoverRuntimeConfig(output_dir=tmp_path, episode_id="ep_shadow", publish_commands=False)

    with RLTEpisodeLogger(logger_path, episode_id="ep_shadow") as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker(
                {
                    1: np.full(7, 2.0, dtype=np.float32),
                    2: np.full(7, 2.0, dtype=np.float32),
                }
            ),
            policy_worker=FakePolicyWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=key_source,
            image_writer=FakeImageWriter(),
            logger=logger,
            command_publisher=publisher,
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        result = core.run_steps(max_steps=10)

    rows = [json.loads(line) for line in logger_path.read_text(encoding="utf-8").splitlines()]
    assert result["outcome"] == "episode_done"
    assert result["steps"] == 5
    assert result["published_commands"] == 0
    assert publisher.commands == []
    assert [row["source"] for row in rows] == [
        "safety_block",
        "human_pika",
        "human_pika",
        "stop",
        "stop",
    ]
    assert rows[0]["a_human"] is None
    assert rows[1]["a_human"] == [2.0] * 7
    assert rows[0]["a_ref"] == [[1.0] * 7 for _ in range(10)]
    assert rows[1]["policy_metadata"]["safety_profile"] == "human_native"
    assert rows[1]["policy_metadata"]["safety_reasons"] == []
    assert rows[-1]["done"] is True
    assert rows[-1]["reward"] == 1.0
    assert rows[-1]["keyboard"]["mode"] == "STOPPED"


def test_motion_limit_holds_until_operator_reward_when_enabled(tmp_path: Path) -> None:
    clock = StepClock()
    policy_worker = FakePolicyWorker()
    logger_path = tmp_path / "episode.jsonl"
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_wait_after_limit",
        wait_for_reward_after_max_steps=True,
    )

    with RLTEpisodeLogger(logger_path, episode_id=config.episode_id) as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=FakeHumanTracker(),
            policy_worker=policy_worker,
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({5: ["0"]}),
            image_writer=FakeImageWriter(),
            logger=logger,
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        result = core.run_steps(max_steps=3)

    rows = [json.loads(line) for line in logger_path.read_text(encoding="utf-8").splitlines()]
    assert result["outcome"] == "episode_done"
    assert result["steps"] == 6
    assert result["terminal_reward"] == 0.0
    assert [row["keyboard"]["mode"] for row in rows] == [
        "MODEL",
        "MODEL",
        "MODEL",
        "WAIT_FOR_REWARD",
        "WAIT_FOR_REWARD",
        "STOPPED",
    ]
    assert [row["source"] for row in rows[-3:]] == ["stop", "stop", "stop"]
    assert [row["policy_metadata"]["model_motion_budget_exhausted"] for row in rows] == [
        False,
        False,
        False,
        True,
        True,
        True,
    ]
    assert rows[-1]["done"] is True
    assert rows[-1]["reward"] == 0.0
    # No new policy plan is requested while the arm is holding for a score.
    assert len(policy_worker.submitted) == 1


def test_publish_mode_requires_explicit_authorization(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="publish authorization"):
        TakeoverRuntimeConfig(output_dir=tmp_path, publish_commands=True).validate()

    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        publish_commands=True,
        publish_authorization=PUBLISH_AUTHORIZATION,
    ).validate()

    assert config.publish_commands is True


def test_publish_mode_sends_selected_commands_when_authorized(tmp_path: Path) -> None:
    clock = StepClock()
    publisher = FakePublisher()
    logger_path = tmp_path / "episode.jsonl"
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_publish",
        publish_commands=True,
        publish_authorization=PUBLISH_AUTHORIZATION,
    )

    with RLTEpisodeLogger(logger_path, episode_id="ep_publish") as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({1: np.full(7, 2.0, dtype=np.float32)}),
            policy_worker=FakePolicyWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({0: ["s"], 2: ["e"], 3: ["1"]}),
            image_writer=FakeImageWriter(),
            logger=logger,
            command_publisher=publisher,
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        result = core.run_steps(max_steps=5)

    assert result["published_commands"] == 4
    assert len(publisher.commands) == 4
    np.testing.assert_allclose(publisher.commands[0], [0.0] * 7)
    np.testing.assert_allclose(publisher.commands[1], [2.0] * 7)
    np.testing.assert_allclose(publisher.commands[2], [0.0] * 7)
    np.testing.assert_allclose(publisher.commands[3], [0.0] * 7)


def test_native_pika_passthrough_preserves_original_message_and_rate() -> None:
    now = [10.0]
    direct = MessagePublisher()
    passthrough = NativePikaJointStatePassthrough(
        direct,
        heartbeat_timeout_s=0.12,
        now_fn=lambda: now[0],
    )
    messages = [object() for _ in range(5)]

    assert passthrough.forward(messages[0]) is False
    passthrough.set_active(True)
    for message in messages:
        now[0] += 0.02  # Native Pika/IK is 50 Hz.
        assert passthrough.forward(message) is True

    assert direct.messages == messages
    assert passthrough.forwarded_count == 5


def test_native_pika_passthrough_fails_closed_without_loop_heartbeat() -> None:
    now = [20.0]
    direct = MessagePublisher()
    passthrough = NativePikaJointStatePassthrough(
        direct,
        heartbeat_timeout_s=0.12,
        now_fn=lambda: now[0],
    )
    passthrough.set_active(True)
    now[0] += 0.13

    assert passthrough.forward(object()) is False
    assert direct.messages == []


def test_native_sdk_publisher_does_not_duplicate_downsampled_human_command(monkeypatch) -> None:
    monkeypatch.setattr(
        "piper_runtime.rlt_takeover_rollout.vector_to_joint_state",
        lambda *args, **kwargs: SimpleNamespace(header=SimpleNamespace(stamp=None, frame_id="")),
    )
    native = MessagePublisher()
    passthrough = NativePikaJointStatePassthrough(MessagePublisher())
    publisher = RosNativeSdkCommandPublisher(
        native,
        FakeRosModule,
        action_dim=7,
        human_passthrough=passthrough,
    )

    publisher.publish_selected(np.zeros(7, dtype=np.float32), source="human_pika")
    assert native.messages == []

    publisher.publish_selected(np.zeros(7, dtype=np.float32), source="pi05")
    assert len(native.messages) == 1


def test_ros_arbitrated_publisher_does_not_duplicate_native_rate_human_command(monkeypatch) -> None:
    monkeypatch.setattr(
        "piper_runtime.rlt_takeover_rollout.vector_to_joint_state",
        lambda *args, **kwargs: SimpleNamespace(header=SimpleNamespace(stamp=None, frame_id="")),
    )
    selected = MessagePublisher()
    passthrough = NativePikaJointStatePassthrough(selected)
    publisher = RosArbitratedCommandPublisher(
        selected,
        action_dim=7,
        human_passthrough=passthrough,
    )

    publisher.publish_selected(np.zeros(7, dtype=np.float32), source="human_pika")
    assert selected.messages == []

    publisher.publish_selected(np.zeros(7, dtype=np.float32), source="pi05")
    assert len(selected.messages) == 1


def test_ros_arbitrated_50hz_servo_is_causal_pauses_for_human_and_counts_actual_outputs(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "piper_runtime.rlt_takeover_rollout.vector_to_joint_state",
        lambda command, **kwargs: SimpleNamespace(
            position=np.asarray(command, dtype=np.float32).copy(),
            header=SimpleNamespace(stamp=None, frame_id=""),
        ),
    )

    class NoopThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            return None

        def join(self, timeout=None):
            del timeout

    monkeypatch.setattr(
        "piper_runtime.rlt_takeover_rollout.threading.Thread",
        NoopThread,
    )
    clock = StepClock()
    selected = MessagePublisher()
    passthrough = NativePikaJointStatePassthrough(
        selected,
        now_fn=clock.now,
    )
    publisher = RosArbitratedCommandPublisher(
        selected,
        action_dim=7,
        human_passthrough=passthrough,
        rospy_module=FakeRosModule,
        input_hz=30.0,
        output_hz=50.0,
        now_fn=clock.now,
    )
    try:
        publisher.publish_selected(np.zeros(7, dtype=np.float32), source="pi05")
        assert publisher._servo_emit(clock.now())

        clock.current += 1.0 / 30.0
        received_s = clock.now()
        publisher.publish_selected(np.ones(7, dtype=np.float32), source="rlt")
        assert publisher._servo_emit(received_s)
        assert publisher._servo_emit(received_s + 0.02)
        assert publisher._servo_emit(received_s + 1.0 / 30.0)

        np.testing.assert_allclose(selected.messages[0].position, np.zeros(7))
        np.testing.assert_allclose(selected.messages[1].position, np.zeros(7))
        np.testing.assert_allclose(
            selected.messages[2].position,
            np.full(7, 0.6),
            atol=1e-6,
        )
        np.testing.assert_allclose(selected.messages[3].position, np.ones(7))
        assert {
            message.header.frame_id for message in selected.messages
        } == {"pi05_50hz"}

        publisher.set_external_passthrough_active(True)
        assert not publisher._servo_emit(received_s + 0.04)
        publisher.publish_selected(
            np.full(7, 9.0, dtype=np.float32),
            source="human_pika",
        )
        publisher.publish_selected(
            np.full(7, 8.0, dtype=np.float32),
            source="pi05",
        )
        assert len(selected.messages) == 4

        publisher.set_external_passthrough_active(False)
        assert not publisher._servo_emit(received_s + 0.06)
        clock.current = received_s + 0.06
        publisher.publish_selected(
            np.full(7, 2.0, dtype=np.float32),
            source="pi05",
        )
        assert publisher._servo_emit(clock.now())

        telemetry = publisher.telemetry()
        assert telemetry["command_publisher_hz"] == 50.0
        assert telemetry["command_publisher_input_count"] == 3
        assert telemetry["command_publisher_count"] == 5
        assert telemetry["command_publisher_paused_for_human"] is False
        assert telemetry["command_publisher_error"] is None
    finally:
        publisher.close()


def test_native_sdk_publisher_uses_synchronous_passthrough_arbitration() -> None:
    calls = []

    def service(active):
        calls.append(bool(active))
        return SimpleNamespace(success=True, message="ok")

    publisher = RosNativeSdkCommandPublisher(
        MessagePublisher(),
        FakeRosModule,
        action_dim=7,
        passthrough_service=service,
    )
    publisher.set_external_passthrough_active(False)
    publisher.set_external_passthrough_active(False)
    publisher.set_external_passthrough_active(True)
    publisher.set_external_passthrough_active(True)
    publisher.set_external_passthrough_active(False)

    assert calls == [False, True, False]


def test_publish_mode_uses_human_native_profile_and_logs_clipping(tmp_path: Path) -> None:
    clock = StepClock()
    publisher = FakePublisher()
    logger_path = tmp_path / "episode.jsonl"
    human_target = np.array([1.0, 1.2, -1.4, 0.8, 0.9, -1.8, 0.096707], dtype=np.float32)
    config = TakeoverRuntimeConfig(
        output_dir=tmp_path,
        episode_id="ep_human_native",
        publish_commands=True,
        publish_authorization=PUBLISH_AUTHORIZATION,
    )

    with RLTEpisodeLogger(logger_path, episode_id="ep_human_native") as logger:
        core = TakeoverLoopCore(
            config=config,
            cameras=FakeCameras(),
            feedback_reader=FakeFeedbackReader(),
            human_tracker=ScheduledHumanTracker({1: human_target}),
            policy_worker=FakePolicyWorker(),
            keyboard=RLTKeyboardStateMachine(toggle_debounce_s=0.01),
            key_source=FakeKeySource({0: ["s"], 2: ["e"], 3: ["1"]}),
            image_writer=FakeImageWriter(),
            logger=logger,
            command_publisher=publisher,
            safety_filter=StatefulSafetyFilter(HardwareSafetyConfig(), np.zeros(7)),
            now_fn=clock.now,
            sleep_fn=clock.sleep,
        )
        result = core.run_steps(max_steps=5)

    rows = [json.loads(line) for line in logger_path.read_text(encoding="utf-8").splitlines()]
    assert result["published_commands"] == 4
    np.testing.assert_allclose(publisher.commands[0], np.zeros(7))
    np.testing.assert_allclose(publisher.commands[1], [1.0, 1.2, -1.4, 0.8, 0.9, -1.8, 0.08])
    assert rows[1]["policy_metadata"]["safety_profile"] == "human_native"
    assert rows[1]["policy_metadata"]["safety_reasons"] == ["gripper_range_clip"]
    assert rows[2]["policy_metadata"]["safety_profile"] == "hold"
    assert "tracking_resync" in rows[2]["policy_metadata"]["safety_reasons"]
    assert rows[2]["policy_metadata"]["h50_feedback_anchored_hold"] is True
    assert rows[2]["policy_metadata"]["h50_feedback_hold_tracking_error_rad"] > 0.03
    assert rows[3]["done"] is True
    assert rows[3]["reward"] == 1.0


def test_scripted_key_source_parses_and_replays_keys() -> None:
    schedule = parse_scripted_keys("0:s,3:s,4:1")
    key_source = ScriptedKeySource(schedule)

    key_source.t = 0
    assert key_source.poll_keys() == ["s"]
    key_source.t = 1
    assert key_source.poll_keys() == []

    key_source.t = 3
    assert key_source.poll_keys() == ["s"]

    key_source.t = 4
    assert key_source.poll_keys() == ["1"]


def test_parse_scripted_keys_rejects_malformed_entries() -> None:
    with pytest.raises(ValueError, match="scripted key"):
        parse_scripted_keys("bad")

    with pytest.raises(ValueError, match="allowed"):
        parse_scripted_keys("0:x")


class StepClock:
    def __init__(self):
        self.current = 100.0

    def now(self) -> float:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.current += float(seconds)
