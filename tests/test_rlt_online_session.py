from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from piper_runtime.rlt_online_session import EpisodeOutcome
from piper_runtime.rlt_online_session import InteractiveResetWaiter
from piper_runtime.rlt_online_session import InteractiveTrainingAdmissionWaiter
from piper_runtime.rlt_online_session import PUBLISH_AUTHORIZATION
from piper_runtime.rlt_online_session import RLTOnlineSession
from piper_runtime.rlt_online_session import build_smooth_reset_trajectory
from piper_runtime.rlt_online_session import _build_default_reset_waiter
from piper_runtime.rlt_online_session import _ensure_piper_enabled_for_reset
from piper_runtime.rlt_online_session import SessionRuntimeConfig
from piper_runtime.rlt_online_session import TrainingAdmissionDecision
from piper_runtime.rlt_online_session import build_arg_parser
from piper_runtime.rlt_online_session import config_from_args
from piper_runtime.rlt_online_session import build_rollout_command
from piper_runtime.rlt_actor_protocol import ACTION_SCHEMA_FINGERPRINT
from piper_runtime.rlt_actor_protocol import ACTOR_PROJECTION_PROFILE
from piper_runtime.rlt_actor_protocol import (
    RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT,
)
from piper_runtime.rlt_actor_protocol import (
    RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE,
)
from piper_runtime.rlt_residual_governor import GRIPPER_RESIDUAL_CLOSE_ASSIST
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
from piper_runtime.rlt_takeover_rollout import ACTOR_LIVE_AUTHORIZATION
from piper_runtime.rlt_takeover_rollout import LEGACY_ACTOR_EXECUTION_PROFILE


class RecordingEpisodeRunner:
    def __init__(self, rewards: list[float]):
        self.rewards = list(rewards)
        self.calls: list[dict[str, object]] = []

    def __call__(self, command: list[str], *, episode_id: str, episode_root: Path) -> EpisodeOutcome:
        call_index = len(self.calls)
        reward = self.rewards[call_index]
        self.calls.append(
            {
                "command": list(command),
                "episode_id": episode_id,
                "episode_root": Path(episode_root),
            }
        )
        return EpisodeOutcome(
            outcome="episode_done",
            steps=100 + call_index,
            published_commands=0,
            last_source="stop",
            terminal_reward=reward,
            episode_root=str(episode_root),
            logger_path=str(episode_root / "episode.jsonl"),
            report_path=str(episode_root / "report.json"),
        )


class EmergencyStopEpisodeRunner:
    def __init__(self):
        self.calls = 0

    def __call__(self, command: list[str], *, episode_id: str, episode_root: Path) -> EpisodeOutcome:
        del command, episode_id
        self.calls += 1
        return EpisodeOutcome(
            outcome="episode_done",
            steps=10,
            published_commands=10,
            last_source="stop",
            terminal_reward=0.0,
            episode_root=str(episode_root),
            logger_path=str(episode_root / "episode.jsonl"),
            report_path=str(episode_root / "report.json"),
            termination_reason="operator_emergency_stop",
            exclude_from_training=True,
            exclusion_category="operator_emergency_stop",
        )


class RecordingHookRunner:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def __call__(self, command: str, *, env: dict[str, str]) -> int:
        self.calls.append({"command": command, "env": dict(env)})
        return 0


class FailingHookRunner(RecordingHookRunner):
    def __call__(self, command: str, *, env: dict[str, str]) -> int:
        self.calls.append({"command": command, "env": dict(env)})
        return 7


class RecordingResetWaiter:
    def __init__(self, decisions: list[bool]):
        self.decisions = list(decisions)
        self.calls: list[dict[str, object]] = []

    def __call__(self, *, episode_index: int, outcome: EpisodeOutcome) -> bool:
        self.calls.append({"episode_index": episode_index, "outcome": outcome})
        if not self.decisions:
            return True
        return self.decisions.pop(0)


class RecordingTrainingAdmissionWaiter:
    def __init__(self, decisions: list[TrainingAdmissionDecision]):
        self.decisions = list(decisions)
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        *,
        episode_index: int,
        episode_id: str,
        outcome: EpisodeOutcome,
    ) -> TrainingAdmissionDecision:
        self.calls.append(
            {
                "episode_index": episode_index,
                "episode_id": episode_id,
                "outcome": outcome,
            }
        )
        return self.decisions.pop(0)


class RecordingHomeResetPublisher:
    def __init__(self):
        self.calls = 0

    def publish_home(self) -> None:
        self.calls += 1


class OrderedEpisodeRunner(RecordingEpisodeRunner):
    def __init__(self, rewards: list[float], events: list[str]):
        super().__init__(rewards)
        self.events = events

    def __call__(self, command: list[str], *, episode_id: str, episode_root: Path) -> EpisodeOutcome:
        self.events.append("episode")
        return super().__call__(command, episode_id=episode_id, episode_root=episode_root)


class OrderedHomeResetPublisher(RecordingHomeResetPublisher):
    def __init__(self, events: list[str]):
        super().__init__()
        self.events = events

    def publish_home(self) -> None:
        self.events.append("reset")
        super().publish_home()


class FakeEnableRos:
    def __init__(self, *, response: object | None = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.waited: list[tuple[str, float]] = []
        self.calls: list[tuple[str, object, dict[str, object]]] = []

    def wait_for_service(self, name: str, timeout: float) -> None:
        self.waited.append((name, timeout))

    def ServiceProxy(self, name: str, service_type: object):
        def call(**kwargs):
            self.calls.append((name, service_type, dict(kwargs)))
            if self.error is not None:
                raise self.error
            return self.response

        return call


def test_terminal_reward_finishes_current_episode_but_session_continues(tmp_path: Path) -> None:
    runner = RecordingEpisodeRunner([1.0, 0.0])
    reset_waiter = RecordingResetWaiter([True])
    config = SessionRuntimeConfig(
        output_dir=tmp_path,
        episode_prefix="warmup",
        max_episodes=2,
        wait_reset=True,
        publish_commands=False,
    ).validate()

    result = RLTOnlineSession(
        config=config,
        episode_runner=runner,
        hook_runner=RecordingHookRunner(),
        reset_waiter=reset_waiter,
    ).run()

    assert result["outcome"] == "max_episodes"
    assert result["episodes_completed"] == 2
    assert [call["episode_id"] for call in runner.calls] == ["warmup_000000", "warmup_000001"]
    assert [call["episode_index"] for call in reset_waiter.calls] == [0]

    session_rows = [json.loads(line) for line in (tmp_path / "session.jsonl").read_text(encoding="utf-8").splitlines()]
    completed = [row for row in session_rows if row["event"] == "episode_completed"]
    assert [row["terminal_reward"] for row in completed] == [1.0, 0.0]
    assert completed[0]["episode_id"] == "warmup_000000"
    assert completed[1]["episode_id"] == "warmup_000001"


def test_q_emergency_stop_ends_session_without_hook_or_reset_prompt(tmp_path: Path) -> None:
    runner = EmergencyStopEpisodeRunner()
    hook_runner = RecordingHookRunner()
    reset_waiter = RecordingResetWaiter([True])
    config = SessionRuntimeConfig(
        output_dir=tmp_path,
        max_episodes=100,
        wait_reset=True,
        after_episode_commands=("update-rlt",),
    ).validate()

    result = RLTOnlineSession(
        config=config,
        episode_runner=runner,
        hook_runner=hook_runner,
        reset_waiter=reset_waiter,
    ).run()

    assert result["outcome"] == "operator_emergency_stop"
    assert result["episodes_completed"] == 0
    assert runner.calls == 1
    assert hook_runner.calls == []
    assert reset_waiter.calls == []


def test_resume_start_index_skips_existing_and_partial_episode_directories(tmp_path: Path) -> None:
    runner = RecordingEpisodeRunner([1.0, 0.0])
    config = SessionRuntimeConfig(
        output_dir=tmp_path,
        episode_prefix="episode",
        episode_start_index=2,
        max_episodes=2,
        wait_reset=False,
    ).validate()

    result = RLTOnlineSession(
        config=config,
        episode_runner=runner,
        hook_runner=RecordingHookRunner(),
        reset_waiter=RecordingResetWaiter([]),
    ).run()

    assert result["episodes_completed"] == 2
    assert [call["episode_id"] for call in runner.calls] == ["episode_000002", "episode_000003"]


def test_after_episode_hooks_receive_episode_environment(tmp_path: Path) -> None:
    runner = RecordingEpisodeRunner([1.0])
    hook_runner = RecordingHookRunner()
    config = SessionRuntimeConfig(
        output_dir=tmp_path,
        episode_prefix="rollout",
        max_episodes=1,
        wait_reset=False,
        after_episode_commands=("ingest-replay", "train-actor-critic"),
    ).validate()

    result = RLTOnlineSession(
        config=config,
        episode_runner=runner,
        hook_runner=hook_runner,
        reset_waiter=RecordingResetWaiter([]),
    ).run()

    assert result["episodes_completed"] == 1
    assert [call["command"] for call in hook_runner.calls] == ["ingest-replay", "train-actor-critic"]
    first_env = hook_runner.calls[0]["env"]
    assert first_env["RLT_EPISODE_ID"] == "rollout_000000"
    assert first_env["RLT_EPISODE_INDEX"] == "0"
    assert first_env["RLT_TERMINAL_REWARD"] == "1.0"
    assert first_env["RLT_EPISODE_ROOT"] == str(tmp_path / "rollout_000000")
    assert first_env["RLT_SESSION_ROOT"] == str(tmp_path)


def test_operator_can_exclude_replay_then_admit_with_same_actor_lineage(tmp_path: Path) -> None:
    runner = RecordingEpisodeRunner([0.0, 1.0])
    hook_runner = RecordingHookRunner()
    admission_waiter = RecordingTrainingAdmissionWaiter(
        [
            TrainingAdmissionDecision(include_in_training=False, replay_next=True),
            TrainingAdmissionDecision(include_in_training=True),
        ]
    )
    config = SessionRuntimeConfig(
        output_dir=tmp_path,
        max_episodes=2,
        wait_reset=False,
        manual_training_admission=True,
        after_episode_commands=("update-rlt",),
    ).validate()

    result = RLTOnlineSession(
        config=config,
        episode_runner=runner,
        hook_runner=hook_runner,
        reset_waiter=RecordingResetWaiter([]),
        training_admission_waiter=admission_waiter,
    ).run()

    assert result["episodes_completed"] == 2
    assert result["episodes_admitted"] == 1
    assert [call["episode_id"] for call in runner.calls] == ["episode_000000", "episode_000001"]
    assert [call["env"]["RLT_EPISODE_ID"] for call in hook_runner.calls] == ["episode_000001"]

    excluded = json.loads((tmp_path / "episode_000000" / "report.json").read_text(encoding="utf-8"))
    admitted = json.loads((tmp_path / "episode_000001" / "report.json").read_text(encoding="utf-8"))
    assert excluded["exclude_from_training"] is True
    assert excluded["training_admission"] == "operator_excluded_replay"
    assert excluded["exclusion_category"] == "operator_not_admitted_replay"
    assert admitted["exclude_from_training"] is False
    assert admitted["training_admission"] == "operator_admitted"
    assert admitted["replay_of_episode_id"] == "episode_000000"

    rows = [json.loads(line) for line in (tmp_path / "session.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(row["event"] == "hook_skipped_episode_not_admitted" for row in rows) == 1
    second_start = [row for row in rows if row["event"] == "episode_started"][1]
    assert second_start["replay_of_episode_id"] == "episode_000000"


def test_operator_can_exclude_episode_and_stop_without_learner_hook(tmp_path: Path) -> None:
    runner = RecordingEpisodeRunner([1.0])
    hook_runner = RecordingHookRunner()
    admission_waiter = RecordingTrainingAdmissionWaiter(
        [TrainingAdmissionDecision(include_in_training=False, stop_session=True)]
    )
    config = SessionRuntimeConfig(
        output_dir=tmp_path,
        max_episodes=10,
        wait_reset=False,
        manual_training_admission=True,
        after_episode_commands=("update-rlt",),
    ).validate()

    result = RLTOnlineSession(
        config=config,
        episode_runner=runner,
        hook_runner=hook_runner,
        reset_waiter=RecordingResetWaiter([]),
        training_admission_waiter=admission_waiter,
    ).run()

    assert result["outcome"] == "operator_excluded_episode_and_stopped"
    assert result["episodes_completed"] == 1
    assert result["episodes_admitted"] == 0
    assert hook_runner.calls == []
    report = json.loads((tmp_path / "episode_000000" / "report.json").read_text(encoding="utf-8"))
    assert report["exclude_from_training"] is True
    assert report["training_admission"] == "operator_excluded_stop"


def test_interactive_training_admission_accepts_train_replay_and_stop_commands() -> None:
    outcome = EpisodeOutcome(
        outcome="episode_done",
        steps=1,
        published_commands=1,
        last_source="stop",
        terminal_reward=1.0,
        episode_root="/tmp/episode",
        logger_path="/tmp/episode/episode.jsonl",
        report_path="/tmp/episode/report.json",
    )
    for command, expected in (
        ("t", TrainingAdmissionDecision(include_in_training=True)),
        ("r", TrainingAdmissionDecision(include_in_training=False, replay_next=True)),
        ("e", TrainingAdmissionDecision(include_in_training=False, stop_session=True)),
    ):
        waiter = InteractiveTrainingAdmissionWaiter(input_fn=lambda _prompt="", value=command: value)
        assert waiter(episode_index=4, episode_id="episode_000004", outcome=outcome) == expected


def test_failed_learner_hook_keeps_incumbent_and_next_episode_continues(tmp_path: Path) -> None:
    runner = RecordingEpisodeRunner([1.0, 0.0])
    config = SessionRuntimeConfig(
        output_dir=tmp_path,
        max_episodes=2,
        wait_reset=False,
        after_episode_commands=("update-rlt",),
    ).validate()

    result = RLTOnlineSession(
        config=config,
        episode_runner=runner,
        hook_runner=FailingHookRunner(),
        reset_waiter=RecordingResetWaiter([]),
    ).run()

    assert result["outcome"] == "max_episodes"
    assert result["episodes_completed"] == 2
    rows = [json.loads(line) for line in (tmp_path / "session.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(row["event"] == "hook_failed_keep_incumbent" for row in rows) == 2


def test_rollout_command_includes_publish_authorization_only_when_enabled(tmp_path: Path) -> None:
    shadow = SessionRuntimeConfig(output_dir=tmp_path, episode_prefix="shadow", max_episodes=1).validate()
    shadow_command = build_rollout_command(
        shadow,
        episode_id="shadow_000000",
        python_executable="/venv/bin/python",
    )
    assert "--publish" not in shadow_command
    assert "--publish-authorization" not in shadow_command

    publish = SessionRuntimeConfig(
        output_dir=tmp_path,
        episode_prefix="publish",
        max_episodes=1,
        publish_commands=True,
        publish_authorization=PUBLISH_AUTHORIZATION,
    ).validate()
    publish_command = build_rollout_command(
        publish,
        episode_id="publish_000000",
        python_executable="/venv/bin/python",
    )
    assert "--publish" in publish_command
    assert publish_command[publish_command.index("--publish-authorization") + 1] == PUBLISH_AUTHORIZATION


def test_online_rollout_waits_for_operator_reward_after_motion_limit_by_default(tmp_path: Path) -> None:
    config = SessionRuntimeConfig(output_dir=tmp_path).validate()
    command = build_rollout_command(config, episode_id="episode_000000", python_executable="python")
    assert "--wait-for-reward-after-max-steps" in command

    no_wait = dataclasses.replace(config, wait_for_reward_after_max_steps=False)
    no_wait_command = build_rollout_command(
        no_wait,
        episode_id="episode_000001",
        python_executable="python",
    )
    assert "--wait-for-reward-after-max-steps" not in no_wait_command


def test_rollout_command_forwards_phase_classifier_args(tmp_path: Path) -> None:
    checkpoint = tmp_path / "phase_classifier.pt"
    checkpoint.write_bytes(b"checkpoint-placeholder")
    config = SessionRuntimeConfig(
        output_dir=tmp_path,
        episode_prefix="phase",
        max_episodes=1,
        phase_classifier_checkpoint=checkpoint,
        phase_classifier_device="cpu",
        phase_enter_threshold=0.6,
        phase_enter_frames=4,
        phase_classifier_period=2,
    ).validate()

    command = build_rollout_command(config, episode_id="phase_000000", python_executable="/venv/bin/python")

    assert command[command.index("--phase-classifier-checkpoint") + 1] == str(checkpoint)
    assert command[command.index("--phase-classifier-device") + 1] == "cpu"
    assert command[command.index("--phase-enter-threshold") + 1] == "0.6"
    assert command[command.index("--phase-enter-frames") + 1] == "4"
    assert command[command.index("--phase-classifier-period") + 1] == "2"


def test_actor_shadow_preserves_sft_execute50_and_forwards_actor_c10_contract(tmp_path: Path) -> None:
    config = SessionRuntimeConfig(
        output_dir=tmp_path,
        actor_shadow=True,
        model_execute_steps=50,
        model_prefetch_lead_steps=4,
        actor_shadow_expected_z_dim=2048,
        actor_shadow_max_latency_s=0.25,
    ).validate()

    assert config.model_execute_steps == 50
    assert config.model_prefetch_lead_steps == 4
    command = build_rollout_command(config, episode_id="shadow_000000", python_executable="python")
    assert "--actor-shadow" in command
    assert command[command.index("--model-execute-steps") + 1] == "50"
    assert command[command.index("--model-prefetch-lead-steps") + 1] == "4"
    assert command[command.index("--hardware-io") + 1] == "native_sdk"
    assert command[command.index("--actor-shadow-expected-z-dim") + 1] == "2048"
    assert command[command.index("--actor-shadow-max-latency") + 1] == "0.25"
    assert command[command.index("--action-schema-fingerprint") + 1] == ACTION_SCHEMA_FINGERPRINT
    assert (
        command[
            command.index("--execution-action-schema-fingerprint") + 1
        ]
        == ACTION_SCHEMA_FINGERPRINT
    )
    assert command[command.index("--actor-projection-profile") + 1] == ACTOR_PROJECTION_PROFILE
    assert (
        command[command.index("--actor-execution-profile") + 1]
        == LEGACY_ACTOR_EXECUTION_PROFILE
    )
    assert command[command.index("--actor-residual-max-rad") + 1] == "0.005"
    assert command[command.index("--actor-residual-d1-max-rad") + 1] == "0.0015"
    assert command[command.index("--actor-residual-d2-max-rad") + 1] == "0.001"
    assert command[command.index("--actor-direction-cone-deg") + 1] == "15"


def test_session_default_h50_prefetch_lead_matches_measured_base_only_budget(
    tmp_path: Path,
) -> None:
    args = build_arg_parser().parse_args(["--output-dir", str(tmp_path)])
    config = config_from_args(args)
    command = build_rollout_command(
        config,
        episode_id="lead_000000",
        python_executable="python",
    )

    assert config.model_prefetch_lead_steps == 5
    assert command[command.index("--model-prefetch-lead-steps") + 1] == "5"


def test_session_cli_accepts_and_forwards_directional_trust_contract(tmp_path: Path) -> None:
    args = build_arg_parser().parse_args(
        [
            "--output-dir", str(tmp_path),
            "--actor-shadow",
            "--action-schema-fingerprint", ACTION_SCHEMA_FINGERPRINT,
            "--actor-projection-profile", ACTOR_PROJECTION_PROFILE,
            "--actor-residual-max-rad", "0.005",
            "--actor-residual-d1-max-rad", "0.0015",
            "--actor-residual-d2-max-rad", "0.001",
            "--actor-direction-cone-deg", "15.0",
            "--manual-training-admission",
        ]
    )
    config = config_from_args(args)
    command = build_rollout_command(config, episode_id="shadow_000000", python_executable="python")

    assert config.action_schema_fingerprint == ACTION_SCHEMA_FINGERPRINT
    assert config.actor_projection_profile == ACTOR_PROJECTION_PROFILE
    assert config.manual_training_admission is True
    assert command[command.index("--actor-residual-max-rad") + 1] == "0.005"
    assert command[command.index("--actor-direction-cone-deg") + 1] == "15"


def test_session_explicitly_forwards_persistent_actor_execution_profile(
    tmp_path: Path,
) -> None:
    args = build_arg_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--actor-execution-profile",
            PERSISTENT_C10_EXECUTION_CONTRACT,
        ]
    )
    config = config_from_args(args)
    command = build_rollout_command(
        config,
        episode_id="persistent_000000",
        python_executable="python",
    )

    assert config.actor_execution_profile == PERSISTENT_C10_EXECUTION_CONTRACT
    assert (
        command[command.index("--actor-execution-profile") + 1]
        == PERSISTENT_C10_EXECUTION_CONTRACT
    )


def test_session_cli_forwards_v2_protocol_and_execution_schemas_separately(
    tmp_path: Path,
) -> None:
    args = build_arg_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--action-schema-fingerprint",
            ACTION_SCHEMA_FINGERPRINT,
            "--execution-action-schema-fingerprint",
            PERSISTENT_FILTERED_ACTUAL_ACTION_SCHEMA_FINGERPRINT,
            "--actor-execution-profile",
            PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT,
            "--actor-live-max-boundary-jump-rad",
            "0.06",
        ]
    )
    config = config_from_args(args)
    command = build_rollout_command(
        config,
        episode_id="persistent_v2_000000",
        python_executable="python",
    )
    assert config.action_schema_fingerprint == ACTION_SCHEMA_FINGERPRINT
    assert config.execution_action_schema_fingerprint == (
        PERSISTENT_FILTERED_ACTUAL_ACTION_SCHEMA_FINGERPRINT
    )
    assert config.actor_execution_profile == (
        PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT
    )
    assert command[command.index("--action-schema-fingerprint") + 1] == (
        ACTION_SCHEMA_FINGERPRINT
    )
    assert command[
        command.index("--execution-action-schema-fingerprint") + 1
    ] == PERSISTENT_FILTERED_ACTUAL_ACTION_SCHEMA_FINGERPRINT
    assert command[command.index("--actor-execution-profile") + 1] == (
        PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT
    )
    assert command[
        command.index("--actor-live-max-boundary-jump-rad") + 1
    ] == "0.06"


def test_session_cli_forwards_close_assist_v3_contract(
    tmp_path: Path,
) -> None:
    args = build_arg_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--action-schema-fingerprint",
            RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT,
            "--execution-action-schema-fingerprint",
            PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT,
            "--actor-projection-profile",
            RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE,
            "--actor-execution-profile",
            PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT,
            "--actor-governor-fingerprint",
            PERSISTENT_GRIPPER_CLOSE_GOVERNOR_FINGERPRINT,
            "--actor-gripper-residual-mode",
            GRIPPER_RESIDUAL_CLOSE_ASSIST,
            "--actor-live-max-boundary-jump-rad",
            "0.06",
        ]
    )
    config = config_from_args(args)
    command = build_rollout_command(
        config,
        episode_id="persistent_close_v3_000000",
        python_executable="python",
    )

    expected = {
        "--action-schema-fingerprint": (
            RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT
        ),
        "--execution-action-schema-fingerprint": (
            PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT
        ),
        "--actor-projection-profile": (
            RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE
        ),
        "--actor-governor-fingerprint": (
            PERSISTENT_GRIPPER_CLOSE_GOVERNOR_FINGERPRINT
        ),
        "--actor-gripper-residual-mode": GRIPPER_RESIDUAL_CLOSE_ASSIST,
        "--actor-gripper-residual-max-close-m": "0.005",
        "--actor-gripper-residual-d1-max-m": "0.0005",
        "--actor-gripper-residual-d2-max-m": "0.0003",
        "--actor-gripper-max-boundary-jump-m": "0.0005",
        "--actor-gripper-command-min-m": "0",
        "--actor-gripper-command-max-m": "0.08",
        "--actor-gripper-release-reference-m": "0.05",
        "--actor-gripper-release-delta-m": "0.002",
    }
    for flag, value in expected.items():
        assert command[command.index(flag) + 1] == value


def test_session_close_assist_schema_and_governor_mismatch_fail_closed(
    tmp_path: Path,
) -> None:
    common = {
        "output_dir": tmp_path,
        "actor_execution_profile": (
            PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT
        ),
        "actor_live_max_boundary_jump_rad": 0.06,
        "actor_gripper_residual_mode": GRIPPER_RESIDUAL_CLOSE_ASSIST,
        "actor_projection_profile": (
            RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE
        ),
        "execution_action_schema_fingerprint": (
            PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT
        ),
        "actor_governor_fingerprint": (
            PERSISTENT_GRIPPER_CLOSE_GOVERNOR_FINGERPRINT
        ),
    }
    with pytest.raises(ValueError, match="raw Actor schema/gripper mode mismatch"):
        SessionRuntimeConfig(**common).validate()
    with pytest.raises(ValueError, match="governor fingerprint mismatch"):
        SessionRuntimeConfig(
            **{
                **common,
                "actor_governor_fingerprint": (
                    PERSISTENT_FILTERED_ACTUAL_GOVERNOR_FINGERPRINT
                ),
            },
            action_schema_fingerprint=(
                RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT
            ),
        ).validate()


def test_session_close_assist_projection_pair_is_strict(
    tmp_path: Path,
) -> None:
    common = {
        "output_dir": tmp_path,
        "actor_execution_profile": (
            PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT
        ),
        "actor_live_max_boundary_jump_rad": 0.06,
        "action_schema_fingerprint": (
            RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT
        ),
        "execution_action_schema_fingerprint": (
            PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT
        ),
        "actor_governor_fingerprint": (
            PERSISTENT_GRIPPER_CLOSE_GOVERNOR_FINGERPRINT
        ),
        "actor_gripper_residual_mode": GRIPPER_RESIDUAL_CLOSE_ASSIST,
    }
    validated = SessionRuntimeConfig(
        **common,
        actor_projection_profile=(
            RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE
        ),
    ).validate()
    assert (
        validated.actor_projection_profile
        == RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE
    )
    with pytest.raises(ValueError, match="projection profile mismatch"):
        SessionRuntimeConfig(
            **common,
            actor_projection_profile=ACTOR_PROJECTION_PROFILE,
        ).validate()


def test_session_v2_schema_pair_mismatch_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="execution action schema mismatch"):
        SessionRuntimeConfig(
            output_dir=tmp_path,
            actor_execution_profile=(
                PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT
            ),
        ).validate()
    with pytest.raises(ValueError, match="session action schema mismatch"):
        SessionRuntimeConfig(
            output_dir=tmp_path,
            action_schema_fingerprint=(
                PERSISTENT_FILTERED_ACTUAL_ACTION_SCHEMA_FINGERPRINT
            ),
            execution_action_schema_fingerprint=(
                PERSISTENT_FILTERED_ACTUAL_ACTION_SCHEMA_FINGERPRINT
            ),
            actor_execution_profile=(
                PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT
            ),
        ).validate()


def test_rollout_command_forwards_model_smoothing_contract(tmp_path: Path) -> None:
    config = SessionRuntimeConfig(
        output_dir=tmp_path,
        model_smoothing_tau_s=0.12,
        model_max_joint_step_deg=0.8,
        model_max_gripper_step=0.003,
    ).validate()

    command = build_rollout_command(config, episode_id="smooth_000000", python_executable="python")

    assert command[command.index("--model-smoothing-tau") + 1] == "0.12"
    assert command[command.index("--model-max-joint-step-deg") + 1] == "0.8"
    assert command[command.index("--model-max-gripper-step") + 1] == "0.003"


def test_actor_live_requires_phase_and_explicit_authorization(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="phase classifier"):
        SessionRuntimeConfig(
            output_dir=tmp_path,
            actor_shadow=True,
            actor_live=True,
            actor_live_authorization=ACTOR_LIVE_AUTHORIZATION,
        ).validate()

    checkpoint = tmp_path / "phase.pt"
    checkpoint.write_bytes(b"test")
    config = SessionRuntimeConfig(
        output_dir=tmp_path,
        actor_shadow=True,
        actor_live=True,
        actor_live_authorization=ACTOR_LIVE_AUTHORIZATION,
        actor_live_max_chunks=1,
        phase_classifier_checkpoint=checkpoint,
    ).validate()
    command = build_rollout_command(config, episode_id="live_000000", python_executable="python")

    assert "--actor-live" in command
    assert command[command.index("--actor-live-authorization") + 1] == ACTOR_LIVE_AUTHORIZATION
    assert command[command.index("--actor-live-max-chunks") + 1] == "1"

    unlimited = dataclasses.replace(config, actor_live_max_chunks=0).validate()
    unlimited_command = build_rollout_command(
        unlimited,
        episode_id="live_000001",
        python_executable="python",
    )
    assert unlimited.actor_live_max_chunks is None
    assert "--actor-live-max-chunks" not in unlimited_command

    with pytest.raises(ValueError, match="actor_live_max_chunks"):
        dataclasses.replace(config, actor_live_max_chunks=-1).validate()


def test_publish_session_requires_explicit_authorization(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="publish authorization"):
        SessionRuntimeConfig(output_dir=tmp_path, publish_commands=True).validate()


def test_interactive_waiter_publishes_home_reset_before_next_episode() -> None:
    reset_publisher = RecordingHomeResetPublisher()
    inputs = iter(["n"])
    printed: list[str] = []
    waiter = InteractiveResetWaiter(
        reset_publisher=reset_publisher,
        input_fn=lambda prompt="": next(inputs),
        print_fn=lambda message: printed.append(str(message)),
    )
    outcome = EpisodeOutcome(
        outcome="episode_done",
        steps=5,
        published_commands=5,
        last_source="stop",
        terminal_reward=1.0,
        episode_root="/tmp/episode",
        logger_path="/tmp/episode/episode.jsonl",
        report_path="/tmp/episode/report.json",
    )

    assert waiter(episode_index=0, outcome=outcome) is True
    assert reset_publisher.calls == 1
    assert any("reset" in message.lower() for message in printed)


def test_reset_enable_gate_requires_positive_service_confirmation() -> None:
    service_type = object()
    enabled = FakeEnableRos(response=type("Response", (), {"enable_response": True})())

    _ensure_piper_enabled_for_reset(enabled, service_type)

    assert enabled.waited == [("/enable_srv", 3.0)]
    assert enabled.calls == [("/enable_srv", service_type, {"enable_request": True})]

    disabled = FakeEnableRos(response=type("Response", (), {"enable_response": False})())
    with pytest.raises(RuntimeError, match="did not enable"):
        _ensure_piper_enabled_for_reset(disabled, service_type)


def test_reset_enable_gate_wraps_service_failure() -> None:
    ros = FakeEnableRos(error=TimeoutError("service unavailable"))

    with pytest.raises(RuntimeError, match="failed to call /enable_srv"):
        _ensure_piper_enabled_for_reset(ros, object())


def test_interactive_waiter_keeps_session_alive_when_reset_needs_retry() -> None:
    class FailsOnce:
        def __init__(self):
            self.calls = 0

        def publish_home(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("feedback not converged")

    reset = FailsOnce()
    inputs = iter(["n", "n"])
    printed = []
    waiter = InteractiveResetWaiter(
        reset_publisher=reset,
        input_fn=lambda prompt="": next(inputs),
        print_fn=lambda message: printed.append(str(message)),
    )
    outcome = EpisodeOutcome(
        outcome="episode_done",
        steps=5,
        published_commands=5,
        last_source="stop",
        terminal_reward=1.0,
        episode_root="/tmp/episode",
        logger_path="/tmp/episode/episode.jsonl",
        report_path="/tmp/episode/report.json",
    )

    assert waiter(episode_index=0, outcome=outcome) is True
    assert reset.calls == 2
    assert any("No next episode was started" in message for message in printed)


def test_initial_reset_happens_before_first_resumed_episode(tmp_path: Path) -> None:
    events: list[str] = []
    runner = OrderedEpisodeRunner([1.0], events)
    reset = OrderedHomeResetPublisher(events)
    waiter = InteractiveResetWaiter(reset_publisher=reset, input_fn=lambda prompt="": "n")
    config = SessionRuntimeConfig(
        output_dir=tmp_path,
        episode_start_index=14,
        max_episodes=1,
        wait_reset=False,
        reset_before_first_episode=True,
    ).validate()

    result = RLTOnlineSession(
        config=config,
        episode_runner=runner,
        hook_runner=RecordingHookRunner(),
        reset_waiter=waiter,
    ).run()

    assert result["outcome"] == "max_episodes"
    assert events == ["reset", "episode"]
    assert runner.calls[0]["episode_id"] == "episode_000014"
    rows = [json.loads(line) for line in (tmp_path / "session.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows[:2]] == ["waiting_for_initial_reset", "initial_reset_completed"]


def test_operator_can_stop_before_initial_reset_without_starting_episode(tmp_path: Path) -> None:
    runner = RecordingEpisodeRunner([1.0])
    waiter = InteractiveResetWaiter(
        reset_publisher=RecordingHomeResetPublisher(),
        input_fn=lambda prompt="": "e",
    )
    config = SessionRuntimeConfig(
        output_dir=tmp_path,
        episode_start_index=14,
        max_episodes=1,
        reset_before_first_episode=True,
    ).validate()

    result = RLTOnlineSession(
        config=config,
        episode_runner=runner,
        hook_runner=RecordingHookRunner(),
        reset_waiter=waiter,
    ).run()

    assert result["outcome"] == "operator_stopped_before_first_episode"
    assert result["episodes_completed"] == 0
    assert runner.calls == []


def test_session_config_validates_home_reset_target(tmp_path: Path) -> None:
    config = SessionRuntimeConfig(output_dir=tmp_path, reset_home_target=(0, 0, 0, 0, 0, 0, 0)).validate()

    assert config.reset_home_target == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="reset_home_target"):
        SessionRuntimeConfig(output_dir=tmp_path, reset_home_target=(0, 0, 0)).validate()

    with pytest.raises(ValueError, match="finite"):
        SessionRuntimeConfig(output_dir=tmp_path, reset_home_target=(0, 0, 0, 0, 0, 0, float("nan"))).validate()


def test_home_reset_trajectory_smoothly_interpolates_from_current_state() -> None:
    start = (1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 0.08)
    target = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    trajectory = build_smooth_reset_trajectory(start=start, target=target, steps=5)

    assert len(trajectory) == 5
    np.testing.assert_allclose(trajectory[0], start)
    np.testing.assert_allclose(trajectory[-1], target)
    np.testing.assert_allclose(
        trajectory[1],
        np.asarray(start) + (np.asarray(target) - np.asarray(start)) * 0.15625,
    )
    np.testing.assert_allclose(
        trajectory[2],
        np.asarray(start) + (np.asarray(target) - np.asarray(start)) * 0.5,
    )


def test_native_sdk_session_resets_through_the_same_persistent_bridge(tmp_path: Path) -> None:
    config = SessionRuntimeConfig(
        output_dir=tmp_path,
        publish_commands=True,
        publish_authorization=PUBLISH_AUTHORIZATION,
        hardware_io="native_sdk",
        reset_home_on_continue=True,
    ).validate()

    waiter = _build_default_reset_waiter(config)

    assert waiter.reset_publisher is not None
    assert waiter.reset_publisher.selected_command_topic == "/rlt/native_sdk_command"
