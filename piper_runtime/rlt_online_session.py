from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from piper_runtime.native_sdk_command_bridge import COMMAND_TOPIC as NATIVE_SDK_COMMAND_TOPIC
from piper_runtime.observation import DEFAULT_PROMPT
from piper_runtime.rlt_actor_protocol import ACTION_SCHEMA_FINGERPRINT
from piper_runtime.rlt_actor_protocol import ACTOR_PROJECTION_PROFILE
from piper_runtime.rlt_residual_governor import GRIPPER_RESIDUAL_CLOSE_ASSIST
from piper_runtime.rlt_residual_governor import GRIPPER_RESIDUAL_FROZEN
from piper_runtime.rlt_residual_governor import PERSISTENT_C10_EXECUTION_CONTRACT
from piper_runtime.rlt_residual_governor import (
    PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT,
)
from piper_runtime.rlt_residual_governor import (
    PERSISTENT_FILTERED_ACTUAL_GOVERNOR_FINGERPRINT,
)
from piper_runtime.rlt_runtime_contracts import LEGACY_ACTOR_EXECUTION_PROFILE
from piper_runtime.rlt_runtime_contracts import PUBLISH_AUTHORIZATION
from piper_runtime.rlt_runtime_contracts import validate_actor_runtime_contract
from piper_runtime.ros_command_io import joint_state_to_vector
from piper_runtime.ros_command_io import require_ros_modules
from piper_runtime.ros_command_io import vector_to_joint_state


@dataclasses.dataclass(frozen=True)
class SessionRuntimeConfig:
    output_dir: Path = Path("~/rlt_takeover_sessions/session").expanduser()
    episode_prefix: str = "episode"
    episode_start_index: int = 0
    max_episodes: int = 1
    duration_s: float = 120.0
    max_steps: int | None = None
    wait_for_reward_after_max_steps: bool = True
    prompt: str = DEFAULT_PROMPT
    policy_host: str = "127.0.0.1"
    policy_port: int = 8000
    freshness_s: float = 0.1
    publish_commands: bool = False
    publish_authorization: str | None = None
    model_safety_profile: str = "native"
    model_smoothing_tau_s: float = 0.05
    model_max_joint_step_deg: float = 3.0
    model_max_gripper_step: float = 0.02
    hardware_io: str = "native_sdk"
    model_execute_steps: int = 50
    model_prefetch_lead_steps: int = 5
    human_end_timeout_s: float = 1.5
    phase_classifier_checkpoint: Path | None = None
    phase_classifier_device: str = "cpu"
    phase_enter_threshold: float = 0.5
    phase_enter_frames: int = 3
    phase_classifier_period: int = 1
    actor_shadow: bool = False
    actor_live: bool = False
    actor_live_authorization: str | None = None
    actor_live_max_chunks: int | None = None
    actor_live_max_boundary_jump_rad: float = 0.02
    actor_residual_max_rad: float = 0.005
    actor_residual_d1_max_rad: float = 0.0015
    actor_residual_d2_max_rad: float = 0.001
    actor_direction_cone_deg: float = 15.0
    action_schema_fingerprint: str = ACTION_SCHEMA_FINGERPRINT
    execution_action_schema_fingerprint: str = ACTION_SCHEMA_FINGERPRINT
    actor_projection_profile: str = ACTOR_PROJECTION_PROFILE
    actor_execution_profile: str = LEGACY_ACTOR_EXECUTION_PROFILE
    actor_governor_fingerprint: str = (
        PERSISTENT_FILTERED_ACTUAL_GOVERNOR_FINGERPRINT
    )
    actor_gripper_residual_mode: str = GRIPPER_RESIDUAL_FROZEN
    actor_gripper_residual_max_close_m: float = 0.005
    actor_gripper_residual_d1_max_m: float = 0.0005
    actor_gripper_residual_d2_max_m: float = 0.0003
    actor_gripper_max_boundary_jump_m: float = 0.0005
    actor_gripper_command_min_m: float = 0.0
    actor_gripper_command_max_m: float = 0.08
    actor_gripper_release_reference_m: float = 0.05
    actor_gripper_release_delta_m: float = 0.002
    actor_shadow_expected_z_dim: int = 2048
    actor_shadow_max_latency_s: float = 0.333
    wait_reset: bool = True
    start_human: bool = False
    after_episode_commands: tuple[str, ...] = ()
    stop_on_hook_failure: bool = False
    manual_training_admission: bool = False
    reset_before_first_episode: bool = False
    reset_home_on_continue: bool = True
    reset_home_target: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    reset_home_hold_s: float = 2.0
    reset_home_hz: float = 30.0
    selected_command_topic: str = "/rlt/selected_joint_command"

    def validate(self) -> "SessionRuntimeConfig":
        output_dir = Path(self.output_dir).expanduser()
        episode_prefix = str(self.episode_prefix).strip()
        after_episode_commands = tuple(str(command).strip() for command in self.after_episode_commands if str(command).strip())
        reset_home_target = tuple(float(value) for value in self.reset_home_target)

        if not episode_prefix:
            raise ValueError("episode_prefix must be non-empty")
        if any(char in episode_prefix for char in "/\\:"):
            raise ValueError("episode_prefix must not contain path separators or ':'")
        if self.episode_start_index < 0:
            raise ValueError("episode_start_index must be non-negative")
        if self.max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        if self.duration_s <= 0 or self.duration_s > 600:
            raise ValueError("duration_s must be in (0, 600]")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when provided")
        if self.freshness_s <= 0:
            raise ValueError("freshness_s must be positive")
        if self.publish_commands and self.publish_authorization != PUBLISH_AUTHORIZATION:
            raise ValueError(f"publish authorization must equal {PUBLISH_AUTHORIZATION!r}")
        if self.model_safety_profile not in {"native", "autonomous"}:
            raise ValueError("model_safety_profile must be 'native' or 'autonomous'")
        if not math.isfinite(self.model_smoothing_tau_s) or self.model_smoothing_tau_s < 0:
            raise ValueError("model_smoothing_tau_s must be finite and non-negative")
        if not math.isfinite(self.model_max_joint_step_deg) or self.model_max_joint_step_deg <= 0:
            raise ValueError("model_max_joint_step_deg must be finite and positive")
        if not math.isfinite(self.model_max_gripper_step) or self.model_max_gripper_step <= 0:
            raise ValueError("model_max_gripper_step must be finite and positive")
        if self.hardware_io not in {"native_sdk", "ros_bridge"}:
            raise ValueError("hardware_io must be 'native_sdk' or 'ros_bridge'")
        if self.model_execute_steps < 1 or self.model_execute_steps > 50:
            raise ValueError("model_execute_steps must be between 1 and 50")
        if (
            self.model_prefetch_lead_steps < 1
            or self.model_prefetch_lead_steps >= self.model_execute_steps
        ):
            raise ValueError("model_prefetch_lead_steps must be in [1, model_execute_steps-1]")
        if self.human_end_timeout_s <= 0 or self.human_end_timeout_s > 30:
            raise ValueError("human_end_timeout_s must be in (0, 30]")
        phase_classifier_checkpoint = None
        if self.phase_classifier_checkpoint is not None:
            phase_classifier_checkpoint = Path(self.phase_classifier_checkpoint).expanduser()
            if not phase_classifier_checkpoint.exists():
                raise ValueError(f"phase classifier checkpoint does not exist: {phase_classifier_checkpoint}")
        if not 0.0 <= float(self.phase_enter_threshold) <= 1.0:
            raise ValueError("phase_enter_threshold must be in [0, 1]")
        if int(self.phase_enter_frames) < 1:
            raise ValueError("phase_enter_frames must be >= 1")
        if int(self.phase_classifier_period) < 1:
            raise ValueError("phase_classifier_period must be >= 1")
        if not str(self.phase_classifier_device).strip():
            raise ValueError("phase_classifier_device must be non-empty")
        actor_live_max_chunks = validate_actor_runtime_contract(
            self,
            context="session",
            phase_classifier_checkpoint=phase_classifier_checkpoint,
            chunk_length=10,
            action_dim=7,
        )
        if len(reset_home_target) != 7:
            raise ValueError("reset_home_target must contain exactly 7 values")
        if not all(math.isfinite(value) for value in reset_home_target):
            raise ValueError("reset_home_target must contain finite values")
        if self.reset_home_hold_s <= 0 or self.reset_home_hold_s > 30:
            raise ValueError("reset_home_hold_s must be in (0, 30]")
        if self.reset_home_hz <= 0 or self.reset_home_hz > 100:
            raise ValueError("reset_home_hz must be in (0, 100]")
        if not str(self.selected_command_topic).startswith("/"):
            raise ValueError("selected_command_topic must be an absolute ROS topic")

        return dataclasses.replace(
            self,
            output_dir=output_dir,
            episode_prefix=episode_prefix,
            after_episode_commands=after_episode_commands,
            phase_classifier_checkpoint=phase_classifier_checkpoint,
            phase_classifier_device=str(self.phase_classifier_device).strip(),
            phase_enter_threshold=float(self.phase_enter_threshold),
            phase_enter_frames=int(self.phase_enter_frames),
            phase_classifier_period=int(self.phase_classifier_period),
            actor_live_max_chunks=actor_live_max_chunks,
            reset_home_target=reset_home_target,
            selected_command_topic=str(self.selected_command_topic),
        )


@dataclasses.dataclass(frozen=True)
class EpisodeOutcome:
    outcome: str
    steps: int
    published_commands: int
    last_source: str
    terminal_reward: float | None
    episode_root: str
    logger_path: str
    report_path: str
    termination_reason: str | None = None
    exclude_from_training: bool = False
    exclusion_category: str | None = None
    exclusion_reason: str | None = None
    training_admission: str | None = None
    replay_of_episode_id: str | None = None

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any],
        *,
        episode_root: Path,
        logger_path: Path | None = None,
        report_path: Path | None = None,
    ) -> "EpisodeOutcome":
        return cls(
            outcome=str(data.get("outcome", "")),
            steps=int(data.get("steps", 0)),
            published_commands=int(data.get("published_commands", 0)),
            last_source=str(data.get("last_source", "")),
            terminal_reward=None if data.get("terminal_reward") is None else float(data["terminal_reward"]),
            episode_root=str(data.get("episode_root", episode_root)),
            logger_path=str(data.get("logger_path", logger_path or episode_root / "episode.jsonl")),
            report_path=str(data.get("report_path", report_path or episode_root / "report.json")),
            termination_reason=(
                None if data.get("termination_reason") is None else str(data["termination_reason"])
            ),
            exclude_from_training=bool(data.get("exclude_from_training", False)),
            exclusion_category=(
                None if data.get("exclusion_category") is None else str(data["exclusion_category"])
            ),
            exclusion_reason=(
                None if data.get("exclusion_reason") is None else str(data["exclusion_reason"])
            ),
            training_admission=(
                None if data.get("training_admission") is None else str(data["training_admission"])
            ),
            replay_of_episode_id=(
                None if data.get("replay_of_episode_id") is None else str(data["replay_of_episode_id"])
            ),
        )


@dataclasses.dataclass(frozen=True)
class TrainingAdmissionDecision:
    include_in_training: bool
    replay_next: bool = False
    stop_session: bool = False

    def validate(self) -> "TrainingAdmissionDecision":
        if self.include_in_training and (self.replay_next or self.stop_session):
            raise ValueError("an admitted episode cannot also request replay or stop")
        if self.replay_next and self.stop_session:
            raise ValueError("training admission cannot request replay and stop together")
        return self


def build_episode_id(config: SessionRuntimeConfig, episode_index: int) -> str:
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    return f"{config.episode_prefix}_{episode_index:06d}"


def build_rollout_command(
    config: SessionRuntimeConfig,
    *,
    episode_id: str,
    python_executable: str | None = None,
) -> list[str]:
    python = python_executable or sys.executable
    command = [
        python,
        "-u",
        "-m",
        "piper_runtime.rlt_takeover_rollout",
        "--episode-id",
        episode_id,
        "--output-dir",
        str(config.output_dir),
        "--duration",
        _format_float(config.duration_s),
        "--prompt",
        config.prompt,
        "--policy-host",
        config.policy_host,
        "--policy-port",
        str(config.policy_port),
        "--freshness",
        _format_float(config.freshness_s),
        "--model-safety-profile",
        config.model_safety_profile,
        "--model-smoothing-tau",
        _format_float(config.model_smoothing_tau_s),
        "--model-max-joint-step-deg",
        _format_float(config.model_max_joint_step_deg),
        "--model-max-gripper-step",
        _format_float(config.model_max_gripper_step),
        "--hardware-io",
        config.hardware_io,
        "--model-execute-steps",
        str(config.model_execute_steps),
        "--model-prefetch-lead-steps",
        str(config.model_prefetch_lead_steps),
        "--human-end-timeout",
        _format_float(config.human_end_timeout_s),
        "--action-schema-fingerprint",
        config.action_schema_fingerprint,
        "--execution-action-schema-fingerprint",
        config.execution_action_schema_fingerprint,
        "--actor-projection-profile",
        config.actor_projection_profile,
        "--actor-execution-profile",
        config.actor_execution_profile,
        "--actor-residual-max-rad",
        _format_float(config.actor_residual_max_rad),
        "--actor-live-max-boundary-jump-rad",
        _format_float(config.actor_live_max_boundary_jump_rad),
        "--actor-residual-d1-max-rad",
        _format_float(config.actor_residual_d1_max_rad),
        "--actor-residual-d2-max-rad",
        _format_float(config.actor_residual_d2_max_rad),
        "--actor-direction-cone-deg",
        _format_float(config.actor_direction_cone_deg),
        "--actor-governor-fingerprint",
        config.actor_governor_fingerprint,
        "--actor-gripper-residual-mode",
        config.actor_gripper_residual_mode,
        "--actor-gripper-residual-max-close-m",
        _format_float(config.actor_gripper_residual_max_close_m),
        "--actor-gripper-residual-d1-max-m",
        _format_float(config.actor_gripper_residual_d1_max_m),
        "--actor-gripper-residual-d2-max-m",
        _format_float(config.actor_gripper_residual_d2_max_m),
        "--actor-gripper-max-boundary-jump-m",
        _format_float(config.actor_gripper_max_boundary_jump_m),
        "--actor-gripper-command-min-m",
        _format_float(config.actor_gripper_command_min_m),
        "--actor-gripper-command-max-m",
        _format_float(config.actor_gripper_command_max_m),
        "--actor-gripper-release-reference-m",
        _format_float(config.actor_gripper_release_reference_m),
        "--actor-gripper-release-delta-m",
        _format_float(config.actor_gripper_release_delta_m),
    ]
    if config.phase_classifier_checkpoint is not None:
        command.extend(
            [
                "--phase-classifier-checkpoint",
                str(config.phase_classifier_checkpoint),
                "--phase-classifier-device",
                config.phase_classifier_device,
                "--phase-enter-threshold",
                _format_float(config.phase_enter_threshold),
                "--phase-enter-frames",
                str(config.phase_enter_frames),
                "--phase-classifier-period",
                str(config.phase_classifier_period),
            ]
        )
    if config.actor_shadow:
        command.extend(
            [
                "--actor-shadow",
                "--actor-shadow-expected-z-dim",
                str(config.actor_shadow_expected_z_dim),
                "--actor-shadow-max-latency",
                _format_float(config.actor_shadow_max_latency_s),
            ]
        )
    if config.actor_live:
        command.extend(
            [
                "--actor-live",
                "--actor-live-authorization",
                str(config.actor_live_authorization),
            ]
        )
        if config.actor_live_max_chunks is not None:
            command.extend(["--actor-live-max-chunks", str(config.actor_live_max_chunks)])
    if config.max_steps is not None:
        command.extend(["--max-steps", str(config.max_steps)])
    if config.wait_for_reward_after_max_steps:
        command.append("--wait-for-reward-after-max-steps")
    if config.start_human:
        command.append("--start-human")
    if config.publish_commands:
        command.extend(["--publish", "--publish-authorization", str(config.publish_authorization)])
    return command


class SubprocessEpisodeRunner:
    def __call__(self, command: list[str], *, episode_id: str, episode_root: Path) -> EpisodeOutcome:
        del episode_id
        completed = subprocess.run(command, check=False)
        report_path = episode_root / "report.json"
        if completed.returncode != 0:
            raise RuntimeError(f"episode subprocess failed with exit code {completed.returncode}")
        if not report_path.exists():
            raise RuntimeError(f"episode subprocess completed but report was not written: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return EpisodeOutcome.from_mapping(
            report,
            episode_root=episode_root,
            logger_path=episode_root / "episode.jsonl",
            report_path=report_path,
        )


class ShellHookRunner:
    def __call__(self, command: str, *, env: dict[str, str]) -> int:
        hook_env = os.environ.copy()
        hook_env.update(env)
        return subprocess.run(command, shell=True, env=hook_env, check=False).returncode


class InteractiveResetWaiter:
    def __init__(self, reset_publisher: Any | None = None, input_fn: Any = input, print_fn: Any = print):
        self.reset_publisher = reset_publisher
        self.input_fn = input_fn
        self.print_fn = print_fn

    def __call__(self, *, episode_index: int, outcome: EpisodeOutcome) -> bool:
        self.print_fn(
            f"\nEpisode {episode_index} finished with reward={outcome.terminal_reward}. "
            "Reset the scene, then type 'n' + Enter for next episode, or 'e'/'q' + Enter to stop."
        )
        return self._wait_for_reset(
            success_message="Home reset converged and arm status is stable; starting the next episode."
        )

    def before_first(self, *, episode_index: int) -> bool:
        self.print_fn(
            f"\nEpisode {episode_index} is ready to resume. Reset the physical scene and clear the arm's "
            "swept volume, then type 'n' + Enter to publish the controlled start-pose reset, "
            "or 'e'/'q' + Enter to stop."
        )
        return self._wait_for_reset(
            success_message="Initial reset converged and arm status is stable; starting the episode."
        )

    def _wait_for_reset(self, *, success_message: str) -> bool:
        while True:
            value = self.input_fn("> ").strip().lower()
            if value == "n":
                if self.reset_publisher is not None:
                    self.print_fn("Re-enabling Piper and publishing the configured reset target...")
                    try:
                        self.reset_publisher.publish_home()
                    except Exception as exc:
                        self.print_fn(
                            f"Home reset did not complete safely: {type(exc).__name__}: {exc}. "
                            "No next episode was started; type 'n' to retry reset or 'e'/'q' to stop."
                        )
                        continue
                    self.print_fn(success_message)
                return True
            if value in {"e", "q", "stop"}:
                return False
            self.print_fn("Expected 'n' to continue or 'e'/'q' to stop.")


class InteractiveTrainingAdmissionWaiter:
    """Ask whether a rewarded rollout may enter replay/A-C training."""

    def __init__(self, input_fn: Any = input, print_fn: Any = print):
        self.input_fn = input_fn
        self.print_fn = print_fn

    def __call__(
        self,
        *,
        episode_index: int,
        episode_id: str,
        outcome: EpisodeOutcome,
    ) -> TrainingAdmissionDecision:
        self.print_fn(
            f"\nEpisode {episode_index} ({episode_id}) has reward={outcome.terminal_reward}. "
            "It is currently excluded from training until you decide."
        )
        self.print_fn("  t / y = admit this episode, run the online A-C update")
        self.print_fn("  r / n = exclude it and replay the test with the same Actor")
        self.print_fn("  e / q = exclude it and stop the session")
        while True:
            value = self.input_fn("train/replay/stop> ").strip().lower()
            if value in {"t", "train", "y", "yes"}:
                return TrainingAdmissionDecision(include_in_training=True).validate()
            if value in {"r", "replay", "n", "no"}:
                return TrainingAdmissionDecision(
                    include_in_training=False,
                    replay_next=True,
                ).validate()
            if value in {"e", "q", "stop"}:
                return TrainingAdmissionDecision(
                    include_in_training=False,
                    stop_session=True,
                ).validate()
            self.print_fn("Expected 't' to train, 'r' to exclude/replay, or 'e'/'q' to exclude/stop.")


def _write_training_admission(
    outcome: EpisodeOutcome,
    *,
    admission: str,
    include_in_training: bool,
    exclusion_category: str | None,
    exclusion_reason: str | None,
    replay_of_episode_id: str | None,
) -> EpisodeOutcome:
    """Atomically persist an operator admission decision in report.json."""

    report_path = Path(outcome.report_path)
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        # Test/fake runners may return an outcome without materializing the
        # production report. Creating it here keeps the admission contract
        # identical and makes this helper independently testable.
        report = dataclasses.asdict(outcome)
    report.update(
        {
            "exclude_from_training": not include_in_training,
            "exclusion_category": exclusion_category,
            "exclusion_reason": exclusion_reason,
            "training_admission": admission,
            "replay_of_episode_id": replay_of_episode_id,
        }
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.admission.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)
    return EpisodeOutcome.from_mapping(
        report,
        episode_root=Path(outcome.episode_root),
        logger_path=Path(outcome.logger_path),
        report_path=report_path,
    )


def _ensure_piper_enabled_for_reset(rospy: Any, enable_service_type: Any) -> None:
    """Enable every Piper joint driver immediately before a reset trajectory."""
    service_name = "/enable_srv"
    try:
        rospy.wait_for_service(service_name, timeout=3.0)
        response = rospy.ServiceProxy(service_name, enable_service_type)(enable_request=True)
    except Exception as exc:
        raise RuntimeError("failed to call /enable_srv before home reset") from exc
    if not bool(getattr(response, "enable_response", False)):
        raise RuntimeError("Piper drivers did not enable before home reset")


class RosHomeResetPublisher:
    def __init__(
        self,
        *,
        target: tuple[float, ...],
        selected_command_topic: str,
        hold_s: float,
        hz: float,
        feedback_topic: str = "/joint_states_single",
    ):
        self.target = tuple(float(value) for value in target)
        self.selected_command_topic = str(selected_command_topic)
        self.hold_s = float(hold_s)
        self.hz = float(hz)
        self.feedback_topic = str(feedback_topic)
        if len(self.target) != 7:
            raise ValueError("home reset target must contain exactly 7 values")
        if not all(math.isfinite(value) for value in self.target):
            raise ValueError("home reset target must contain finite values")

    def publish_home(self) -> None:
        rospy, _JointState = require_ros_modules()
        if not rospy.core.is_initialized():
            rospy.init_node("rlt_online_session_reset", anonymous=True, disable_signals=True)
        from piper_msgs.srv import Enable

        # The arm may disable while services warm up or while the operator
        # resets the scene. Renew and verify enable before the first reset
        # position command; err_code=0 alone does not prove driver enable.
        _ensure_piper_enabled_for_reset(rospy, Enable)
        publisher = rospy.Publisher(self.selected_command_topic, _JointState, queue_size=1)
        start = self._read_current_state(rospy, _JointState)
        interval_s = 1.0 / self.hz
        publish_count = max(1, int(round(self.hold_s * self.hz)))
        trajectory = build_smooth_reset_trajectory(start=start, target=self.target, steps=publish_count)
        time.sleep(0.2)
        for command in trajectory:
            if rospy.is_shutdown():
                break
            message = vector_to_joint_state(command, action_dim=7)
            message.header.stamp = rospy.Time.now()
            message.header.frame_id = "reset"
            publisher.publish(message)
            time.sleep(interval_s)
        self._wait_until_target_reached(rospy, _JointState, publisher, interval_s=interval_s)

    def _wait_until_target_reached(
        self,
        rospy: Any,
        joint_state_type: Any,
        publisher: Any,
        *,
        interval_s: float,
    ) -> None:
        from piper_msgs.msg import PiperStatusMsg

        target = np.asarray(self.target, dtype=np.float32)
        joint_tolerance = math.radians(3.0)
        gripper_tolerance = 0.005
        deadline = time.monotonic() + 6.0
        stable_samples = 0
        last_state: np.ndarray | None = None
        while time.monotonic() < deadline:
            message = vector_to_joint_state(target, action_dim=7)
            message.header.stamp = rospy.Time.now()
            message.header.frame_id = "reset"
            publisher.publish(message)
            try:
                feedback = rospy.wait_for_message(self.feedback_topic, joint_state_type, timeout=0.25)
                last_state = joint_state_to_vector(feedback, action_dim=7)
                arm_status = rospy.wait_for_message("/arm_status", PiperStatusMsg, timeout=0.25)
                reached = bool(
                    np.all(np.abs(last_state[:6] - target[:6]) <= joint_tolerance)
                    and abs(float(last_state[6] - target[6])) <= gripper_tolerance
                )
                healthy = int(arm_status.err_code) == 0 and int(arm_status.ctrl_mode) == 1
                stable_samples = stable_samples + 1 if reached and healthy else 0
                if stable_samples >= 5:
                    return
            except Exception:
                stable_samples = 0
            time.sleep(interval_s)
        raise RuntimeError(
            "home reset feedback did not converge to target; last_state=%s target=%s"
            % (None if last_state is None else last_state.tolist(), target.tolist())
        )

    def _read_current_state(self, rospy: Any, joint_state_type: Any) -> tuple[float, ...]:
        try:
            message = rospy.wait_for_message(self.feedback_topic, joint_state_type, timeout=1.0)
            return tuple(float(value) for value in joint_state_to_vector(message, action_dim=7))
        except Exception as exc:
            raise RuntimeError(f"cannot start home reset without fresh feedback on {self.feedback_topic}") from exc


@dataclasses.dataclass
class RLTOnlineSession:
    config: SessionRuntimeConfig
    episode_runner: Any = dataclasses.field(default_factory=SubprocessEpisodeRunner)
    hook_runner: Any = dataclasses.field(default_factory=ShellHookRunner)
    reset_waiter: Any | None = None
    training_admission_waiter: Any | None = None
    python_executable: str | None = None
    now_ns_fn: Any = time.time_ns

    def run(self) -> dict[str, Any]:
        config = self.config.validate()
        config.output_dir.mkdir(parents=True, exist_ok=True)
        session_log = config.output_dir / "session.jsonl"
        reset_waiter = self.reset_waiter or _build_default_reset_waiter(config)
        training_admission_waiter = self.training_admission_waiter
        if config.manual_training_admission and training_admission_waiter is None:
            training_admission_waiter = InteractiveTrainingAdmissionWaiter()
        episodes_completed = 0
        episodes_admitted = 0
        replay_origin_episode_id: str | None = None
        final_outcome = "max_episodes"

        if config.reset_before_first_episode:
            before_first = getattr(reset_waiter, "before_first", None)
            if before_first is None:
                raise RuntimeError("reset_before_first_episode requires an initial-reset-capable waiter")
            _append_session_event(
                session_log,
                event="waiting_for_initial_reset",
                now_ns_fn=self.now_ns_fn,
                episode_index=config.episode_start_index,
            )
            if not bool(before_first(episode_index=config.episode_start_index)):
                result = {
                    "outcome": "operator_stopped_before_first_episode",
                    "episodes_completed": 0,
                    "episodes_admitted": 0,
                    "session_root": str(config.output_dir),
                    "session_log": str(session_log),
                }
                _append_session_event(session_log, event="session_finished", now_ns_fn=self.now_ns_fn, **result)
                return result
            _append_session_event(
                session_log,
                event="initial_reset_completed",
                now_ns_fn=self.now_ns_fn,
                episode_index=config.episode_start_index,
            )

        for episode_offset in range(config.max_episodes):
            episode_index = config.episode_start_index + episode_offset
            episode_id = build_episode_id(config, episode_index)
            episode_root = config.output_dir / episode_id
            command = build_rollout_command(config, episode_id=episode_id, python_executable=self.python_executable)
            _append_session_event(
                session_log,
                event="episode_started",
                now_ns_fn=self.now_ns_fn,
                episode_index=episode_index,
                episode_id=episode_id,
                episode_root=str(episode_root),
                command=command,
                replay_of_episode_id=replay_origin_episode_id,
            )

            outcome = self.episode_runner(command, episode_id=episode_id, episode_root=episode_root)
            _append_session_event(
                session_log,
                event="episode_completed",
                now_ns_fn=self.now_ns_fn,
                episode_index=episode_index,
                episode_id=episode_id,
                **dataclasses.asdict(outcome),
            )

            # q is a session-level emergency stop.  The rollout already holds
            # the arm and marks this trajectory as excluded; do not run the
            # learner hook or ask for a second q at the reset prompt.
            if (
                outcome.termination_reason == "operator_emergency_stop"
                or outcome.exclusion_category == "operator_emergency_stop"
            ):
                final_outcome = "operator_emergency_stop"
                break

            if outcome.outcome != "episode_done" or outcome.terminal_reward is None:
                final_outcome = "episode_incomplete"
                break

            episodes_completed += 1
            stop_after_admission = False
            if config.manual_training_admission:
                # Fail closed: immediately quarantine the rewarded report
                # before waiting for console input. A crash or disconnect at
                # this prompt therefore cannot leak the episode into replay.
                outcome = _write_training_admission(
                    outcome,
                    admission="pending_operator_decision",
                    include_in_training=False,
                    exclusion_category="pending_operator_training_admission",
                    exclusion_reason="awaiting explicit operator train/replay decision",
                    replay_of_episode_id=replay_origin_episode_id,
                )
                _append_session_event(
                    session_log,
                    event="training_admission_pending",
                    now_ns_fn=self.now_ns_fn,
                    episode_index=episode_index,
                    episode_id=episode_id,
                    replay_of_episode_id=replay_origin_episode_id,
                )
                if training_admission_waiter is None:
                    raise RuntimeError("manual training admission requires a decision waiter")
                decision = training_admission_waiter(
                    episode_index=episode_index,
                    episode_id=episode_id,
                    outcome=outcome,
                ).validate()
                if decision.include_in_training:
                    outcome = _write_training_admission(
                        outcome,
                        admission="operator_admitted",
                        include_in_training=True,
                        exclusion_category=None,
                        exclusion_reason=None,
                        replay_of_episode_id=replay_origin_episode_id,
                    )
                    episodes_admitted += 1
                    replay_origin_episode_id = None
                elif decision.replay_next:
                    outcome = _write_training_admission(
                        outcome,
                        admission="operator_excluded_replay",
                        include_in_training=False,
                        exclusion_category="operator_not_admitted_replay",
                        exclusion_reason="operator requested another test with the unchanged Actor",
                        replay_of_episode_id=replay_origin_episode_id,
                    )
                    if replay_origin_episode_id is None:
                        replay_origin_episode_id = episode_id
                else:
                    outcome = _write_training_admission(
                        outcome,
                        admission="operator_excluded_stop",
                        include_in_training=False,
                        exclusion_category="operator_not_admitted_stop",
                        exclusion_reason="operator excluded this test and stopped the session",
                        replay_of_episode_id=replay_origin_episode_id,
                    )
                    stop_after_admission = decision.stop_session
                _append_session_event(
                    session_log,
                    event="training_admission_decided",
                    now_ns_fn=self.now_ns_fn,
                    episode_index=episode_index,
                    episode_id=episode_id,
                    training_admission=outcome.training_admission,
                    exclude_from_training=outcome.exclude_from_training,
                    replay_of_episode_id=outcome.replay_of_episode_id,
                    next_replay_origin_episode_id=replay_origin_episode_id,
                )
                if stop_after_admission:
                    final_outcome = "operator_excluded_episode_and_stopped"
                    break
            else:
                episodes_admitted += 1

            hook_failed = False
            if outcome.exclude_from_training:
                _append_session_event(
                    session_log,
                    event="hook_skipped_episode_not_admitted",
                    now_ns_fn=self.now_ns_fn,
                    episode_index=episode_index,
                    episode_id=episode_id,
                    training_admission=outcome.training_admission,
                )
            else:
                hook_failed = self._run_after_episode_hooks(
                    session_log=session_log,
                    config=config,
                    episode_index=episode_index,
                    episode_id=episode_id,
                    outcome=outcome,
                )
            if hook_failed:
                # Learner/replay maintenance must never tear down a healthy
                # robot data session. Keep the incumbent Actor (the updater is
                # transactional), log the failure, and let the operator reset
                # for the next episode. A strict debugging mode remains
                # available when explicitly requested.
                _append_session_event(
                    session_log,
                    event="hook_failed_keep_incumbent",
                    now_ns_fn=self.now_ns_fn,
                    episode_index=episode_index,
                    episode_id=episode_id,
                )
                if config.stop_on_hook_failure:
                    final_outcome = "hook_failed"
                    break

            has_next_episode = episode_offset + 1 < config.max_episodes
            if has_next_episode and config.wait_reset:
                _append_session_event(
                    session_log,
                    event="waiting_for_reset",
                    now_ns_fn=self.now_ns_fn,
                    episode_index=episode_index,
                    episode_id=episode_id,
                )
                if not reset_waiter(episode_index=episode_index, outcome=outcome):
                    final_outcome = "operator_stopped"
                    break

        result = {
            "outcome": final_outcome,
            "episodes_completed": episodes_completed,
            "episodes_admitted": episodes_admitted,
            "session_root": str(config.output_dir),
            "session_log": str(session_log),
        }
        _append_session_event(session_log, event="session_finished", now_ns_fn=self.now_ns_fn, **result)
        return result

    def _run_after_episode_hooks(
        self,
        *,
        session_log: Path,
        config: SessionRuntimeConfig,
        episode_index: int,
        episode_id: str,
        outcome: EpisodeOutcome,
    ) -> bool:
        hook_env = {
            "RLT_SESSION_ROOT": str(config.output_dir),
            "RLT_EPISODE_INDEX": str(episode_index),
            "RLT_EPISODE_ID": episode_id,
            "RLT_EPISODE_ROOT": outcome.episode_root,
            "RLT_LOGGER_PATH": outcome.logger_path,
            "RLT_REPORT_PATH": outcome.report_path,
            "RLT_TERMINAL_REWARD": "" if outcome.terminal_reward is None else str(float(outcome.terminal_reward)),
        }
        for command in config.after_episode_commands:
            _append_session_event(
                session_log,
                event="hook_started",
                now_ns_fn=self.now_ns_fn,
                episode_index=episode_index,
                episode_id=episode_id,
                hook_command=command,
            )
            return_code = int(self.hook_runner(command, env=hook_env))
            _append_session_event(
                session_log,
                event="hook_completed",
                now_ns_fn=self.now_ns_fn,
                episode_index=episode_index,
                episode_id=episode_id,
                hook_command=command,
                return_code=return_code,
            )
            if return_code != 0:
                return True
        return False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent multi-episode Piper RLT session wrapper")
    parser.add_argument("--output-dir", type=Path, default=Path("~/rlt_takeover_sessions/session").expanduser())
    parser.add_argument("--episode-prefix", default="episode")
    parser.add_argument("--episode-start-index", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--no-wait-for-reward-after-max-steps",
        action="store_true",
        help="End an unscored episode at its motion limit instead of holding for an operator 1/0 reward.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--freshness", type=float, default=0.1)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--publish-authorization", default=None)
    parser.add_argument("--model-safety-profile", default="native", choices=["native", "autonomous"])
    parser.add_argument("--model-smoothing-tau", type=float, default=0.05)
    parser.add_argument("--model-max-joint-step-deg", type=float, default=3.0)
    parser.add_argument("--model-max-gripper-step", type=float, default=0.02)
    parser.add_argument("--hardware-io", default="native_sdk", choices=["native_sdk", "ros_bridge"])
    parser.add_argument("--model-execute-steps", "--execute-steps", dest="model_execute_steps", type=int, default=50)
    parser.add_argument("--model-prefetch-lead-steps", type=int, default=5)
    parser.add_argument("--human-end-timeout", type=float, default=1.5)
    parser.add_argument("--phase-classifier-checkpoint", type=Path, default=None)
    parser.add_argument("--phase-classifier-device", default="cpu")
    parser.add_argument("--phase-enter-threshold", type=float, default=0.5)
    parser.add_argument("--phase-enter-frames", type=int, default=3)
    parser.add_argument("--phase-classifier-period", type=int, default=1)
    parser.add_argument("--actor-shadow", action="store_true")
    parser.add_argument("--actor-live", action="store_true")
    parser.add_argument("--actor-live-authorization", default=None)
    parser.add_argument(
        "--actor-live-max-chunks",
        type=int,
        default=0,
        help="Maximum C=10 Actor chunks per episode; 0 means unlimited.",
    )
    parser.add_argument("--actor-shadow-expected-z-dim", type=int, default=2048)
    parser.add_argument("--actor-shadow-max-latency", type=float, default=0.333)
    parser.add_argument("--actor-live-max-boundary-jump-rad", type=float, default=0.02)
    parser.add_argument("--actor-residual-max-rad", type=float, default=0.005)
    parser.add_argument("--actor-residual-d1-max-rad", type=float, default=0.0015)
    parser.add_argument("--actor-residual-d2-max-rad", type=float, default=0.001)
    parser.add_argument("--actor-direction-cone-deg", type=float, default=15.0)
    parser.add_argument(
        "--actor-gripper-residual-mode",
        choices=[GRIPPER_RESIDUAL_FROZEN, GRIPPER_RESIDUAL_CLOSE_ASSIST],
        default=GRIPPER_RESIDUAL_FROZEN,
    )
    parser.add_argument(
        "--actor-gripper-residual-max-close-m",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--actor-gripper-residual-d1-max-m",
        type=float,
        default=0.0005,
    )
    parser.add_argument(
        "--actor-gripper-residual-d2-max-m",
        type=float,
        default=0.0003,
    )
    parser.add_argument(
        "--actor-gripper-max-boundary-jump-m",
        type=float,
        default=0.0005,
    )
    parser.add_argument("--actor-gripper-command-min-m", type=float, default=0.0)
    parser.add_argument("--actor-gripper-command-max-m", type=float, default=0.08)
    parser.add_argument(
        "--actor-gripper-release-reference-m",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--actor-gripper-release-delta-m",
        type=float,
        default=0.002,
    )
    parser.add_argument(
        "--action-schema-fingerprint",
        default=ACTION_SCHEMA_FINGERPRINT,
    )
    parser.add_argument(
        "--execution-action-schema-fingerprint",
        default=ACTION_SCHEMA_FINGERPRINT,
    )
    parser.add_argument(
        "--actor-projection-profile",
        default=ACTOR_PROJECTION_PROFILE,
    )
    parser.add_argument(
        "--actor-governor-fingerprint",
        default=PERSISTENT_FILTERED_ACTUAL_GOVERNOR_FINGERPRINT,
    )
    parser.add_argument(
        "--actor-execution-profile",
        default=LEGACY_ACTOR_EXECUTION_PROFILE,
        choices=[
            LEGACY_ACTOR_EXECUTION_PROFILE,
            PERSISTENT_C10_EXECUTION_CONTRACT,
            PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT,
        ],
        help=(
            "Actor execution semantics. The persistent profile carries each "
            "safe residual through every phase-active H50 frame."
        ),
    )
    parser.add_argument("--start-human", action="store_true")
    parser.add_argument("--no-wait-reset", action="store_true")
    parser.add_argument(
        "--reset-before-first",
        action="store_true",
        help="Require operator confirmation and publish the controlled reset before the first episode.",
    )
    parser.add_argument("--no-reset-home", action="store_true")
    parser.add_argument("--reset-home-target", default="0,0,0,0,0,0,0")
    parser.add_argument("--reset-home-hold", type=float, default=2.0)
    parser.add_argument("--reset-home-hz", type=float, default=30.0)
    parser.add_argument("--selected-command-topic", default="/rlt/selected_joint_command")
    parser.add_argument(
        "--after-episode-command",
        action="append",
        default=[],
        help="Shell command to run after each rewarded episode. May be repeated.",
    )
    parser.add_argument(
        "--stop-on-hook-failure",
        action="store_true",
        help="Strict/debug mode: stop the rollout session if an after-episode learner hook fails.",
    )
    parser.add_argument(
        "--manual-training-admission",
        action="store_true",
        help=(
            "After reward 1/0, require an explicit operator train/replay/stop decision. "
            "Episodes remain excluded until admitted."
        ),
    )
    return parser


def config_from_args(args: argparse.Namespace) -> SessionRuntimeConfig:
    return SessionRuntimeConfig(
        output_dir=args.output_dir,
        episode_prefix=args.episode_prefix,
        episode_start_index=args.episode_start_index,
        max_episodes=args.max_episodes,
        duration_s=args.duration,
        max_steps=args.max_steps,
        wait_for_reward_after_max_steps=not bool(args.no_wait_for_reward_after_max_steps),
        prompt=args.prompt,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        freshness_s=args.freshness,
        publish_commands=bool(args.publish),
        publish_authorization=args.publish_authorization,
        model_safety_profile=args.model_safety_profile,
        model_smoothing_tau_s=args.model_smoothing_tau,
        model_max_joint_step_deg=args.model_max_joint_step_deg,
        model_max_gripper_step=args.model_max_gripper_step,
        hardware_io=args.hardware_io,
        model_execute_steps=args.model_execute_steps,
        model_prefetch_lead_steps=args.model_prefetch_lead_steps,
        human_end_timeout_s=args.human_end_timeout,
        phase_classifier_checkpoint=args.phase_classifier_checkpoint,
        phase_classifier_device=args.phase_classifier_device,
        phase_enter_threshold=args.phase_enter_threshold,
        phase_enter_frames=args.phase_enter_frames,
        phase_classifier_period=args.phase_classifier_period,
        actor_shadow=bool(args.actor_shadow),
        actor_live=bool(args.actor_live),
        actor_live_authorization=args.actor_live_authorization,
        actor_live_max_chunks=args.actor_live_max_chunks,
        actor_live_max_boundary_jump_rad=args.actor_live_max_boundary_jump_rad,
        actor_residual_max_rad=args.actor_residual_max_rad,
        actor_residual_d1_max_rad=args.actor_residual_d1_max_rad,
        actor_residual_d2_max_rad=args.actor_residual_d2_max_rad,
        actor_direction_cone_deg=args.actor_direction_cone_deg,
        actor_gripper_residual_mode=args.actor_gripper_residual_mode,
        actor_gripper_residual_max_close_m=(
            args.actor_gripper_residual_max_close_m
        ),
        actor_gripper_residual_d1_max_m=args.actor_gripper_residual_d1_max_m,
        actor_gripper_residual_d2_max_m=args.actor_gripper_residual_d2_max_m,
        actor_gripper_max_boundary_jump_m=(
            args.actor_gripper_max_boundary_jump_m
        ),
        actor_gripper_command_min_m=args.actor_gripper_command_min_m,
        actor_gripper_command_max_m=args.actor_gripper_command_max_m,
        actor_gripper_release_reference_m=(
            args.actor_gripper_release_reference_m
        ),
        actor_gripper_release_delta_m=args.actor_gripper_release_delta_m,
        action_schema_fingerprint=args.action_schema_fingerprint,
        execution_action_schema_fingerprint=(
            args.execution_action_schema_fingerprint
        ),
        actor_projection_profile=args.actor_projection_profile,
        actor_execution_profile=args.actor_execution_profile,
        actor_governor_fingerprint=args.actor_governor_fingerprint,
        actor_shadow_expected_z_dim=args.actor_shadow_expected_z_dim,
        actor_shadow_max_latency_s=args.actor_shadow_max_latency,
        wait_reset=not bool(args.no_wait_reset),
        start_human=bool(args.start_human),
        after_episode_commands=tuple(args.after_episode_command),
        stop_on_hook_failure=bool(args.stop_on_hook_failure),
        manual_training_admission=bool(args.manual_training_admission),
        reset_before_first_episode=bool(args.reset_before_first),
        reset_home_on_continue=not bool(args.no_reset_home),
        reset_home_target=parse_reset_home_target(args.reset_home_target),
        reset_home_hold_s=args.reset_home_hold,
        reset_home_hz=args.reset_home_hz,
        selected_command_topic=args.selected_command_topic,
    ).validate()


def parse_reset_home_target(value: str | tuple[float, ...] | list[float]) -> tuple[float, ...]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        try:
            target = tuple(float(part) for part in parts)
        except ValueError as exc:
            raise ValueError("reset_home_target must be a comma-separated list of numbers") from exc
    else:
        target = tuple(float(part) for part in value)
    if len(target) != 7:
        raise ValueError("reset_home_target must contain exactly 7 values")
    if not all(math.isfinite(part) for part in target):
        raise ValueError("reset_home_target must contain finite values")
    return target


def _build_default_reset_waiter(config: SessionRuntimeConfig) -> InteractiveResetWaiter:
    reset_publisher = None
    if config.publish_commands and config.reset_home_on_continue:
        reset_publisher = RosHomeResetPublisher(
            target=config.reset_home_target,
            selected_command_topic=(
                NATIVE_SDK_COMMAND_TOPIC if config.hardware_io == "native_sdk" else config.selected_command_topic
            ),
            hold_s=config.reset_home_hold_s,
            hz=config.reset_home_hz,
        )
    return InteractiveResetWaiter(reset_publisher=reset_publisher)


def build_smooth_reset_trajectory(*, start: Any, target: Any, steps: int) -> list[np.ndarray]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    start_array = np.asarray(start, dtype=np.float32)
    target_array = np.asarray(target, dtype=np.float32)
    if start_array.shape != (7,) or target_array.shape != (7,):
        raise ValueError("start and target must have shape (7,)")
    if not np.all(np.isfinite(start_array)) or not np.all(np.isfinite(target_array)):
        raise ValueError("start and target must be finite")
    if steps == 1:
        return [target_array.copy()]
    trajectory: list[np.ndarray] = []
    for index in range(int(steps)):
        tau = index / float(steps - 1)
        alpha = (3.0 * tau * tau) - (2.0 * tau * tau * tau)
        trajectory.append((start_array + (target_array - start_array) * alpha).astype(np.float32))
    return trajectory


def _append_session_event(path: Path, *, event: str, now_ns_fn: Any, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"event": event, "timestamp_ns": int(now_ns_fn()), **fields}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _format_float(value: float) -> str:
    return f"{float(value):g}"


def main() -> None:
    parser = build_arg_parser()
    config = config_from_args(parser.parse_args())
    result = RLTOnlineSession(config=config).run()
    print(result)


if __name__ == "__main__":
    main()
