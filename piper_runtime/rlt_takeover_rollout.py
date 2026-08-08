from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from piper_runtime.cameras import DualRealSenseReader
from piper_runtime.buffered_policy_control import H50HandoffRejected
from piper_runtime.buffered_policy_control import prepare_h50_handoff_plan
from piper_runtime.hardware_control import HardwareSafetyConfig
from piper_runtime.hardware_control import HardwareSafetyError
from piper_runtime.hardware_control import StatefulSafetyFilter
from piper_runtime.observation import DEFAULT_PROMPT
from piper_runtime.observation import build_observation
from piper_runtime.rlt_actor_protocol import ACTION_SCHEMA_FINGERPRINT
from piper_runtime.rlt_actor_protocol import ACTOR_PROJECTION_PROFILE
from piper_runtime.rlt_actor_protocol import BehaviorReferenceTarget
from piper_runtime.rlt_actor_protocol import RANK1_BUMP_CONTRACT
from piper_runtime.rlt_actor_protocol import add_actor_enrichment_only_request
from piper_runtime.rlt_actor_protocol import add_base_only_request
from piper_runtime.rlt_actor_protocol import add_behavior_reference
from piper_runtime.native_sdk_command_bridge import COMMAND_TOPIC as NATIVE_SDK_COMMAND_TOPIC
from piper_runtime.native_sdk_command_bridge import PASSTHROUGH_SERVICE as NATIVE_SDK_PASSTHROUGH_SERVICE
from piper_runtime.native_sdk_command_bridge import STATUS_TOPIC as NATIVE_SDK_STATUS_TOPIC
from piper_runtime.rlt_command_mux import RLTCommandMux
from piper_runtime.rlt_command_mux import TimedCommand
from piper_runtime.rlt_episode_logger import RLTEpisodeLogger
from piper_runtime.rlt_keyboard import RLTKeyboardStateMachine
from piper_runtime.rlt_phase_gate import PhaseGateConfig
from piper_runtime.rlt_phase_gate import PhaseGateSnapshot
from piper_runtime.rlt_phase_gate import SingleLatchPhaseGate
from piper_runtime.rlt_phase_gate import TorchPhaseClassifier
from piper_runtime.rlt_phase_gate import should_run_classifier
from piper_runtime.rlt_policy_adapter import extract_rlt_policy_output
from piper_runtime.rlt_policy_worker import RLTPolicyWorker
from piper_runtime.rlt_residual_governor import ActorResidualGovernor
from piper_runtime.rlt_residual_governor import ActorResidualGovernorConfig
from piper_runtime.rlt_residual_governor import GRIPPER_RESIDUAL_CLOSE_ASSIST
from piper_runtime.rlt_residual_governor import GRIPPER_RESIDUAL_FROZEN
from piper_runtime.rlt_residual_governor import PERSISTENT_C10_EXECUTION_CONTRACT
from piper_runtime.rlt_residual_governor import (
    PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT,
)
from piper_runtime.rlt_residual_governor import (
    PERSISTENT_FILTERED_ACTUAL_GOVERNOR_FINGERPRINT,
)
from piper_runtime.rlt_residual_governor import PersistentActorResidualGovernor
from piper_runtime.rlt_runtime_contracts import ACTOR_LIVE_AUTHORIZATION
from piper_runtime.rlt_runtime_contracts import LEGACY_ACTOR_EXECUTION_PROFILE
from piper_runtime.rlt_runtime_contracts import PUBLISH_AUTHORIZATION
from piper_runtime.rlt_runtime_contracts import validate_actor_runtime_contract
from piper_runtime.ros_command_io import FreshJointCommandTracker
from piper_runtime.ros_command_io import require_ros_modules
from piper_runtime.ros_command_io import vector_to_joint_state


H50_FALLBACK_FEEDBACK_HOLD_THRESHOLD_RAD = 0.03


@dataclasses.dataclass(frozen=True)
class TakeoverRuntimeConfig:
    control_hz: float = 30.0
    action_dim: int = 7
    chunk_length: int = 10
    model_execute_steps: int = 50
    # The split base lane uses ``base_only_v1``.  On the deployed 5090 service,
    # 100-request validation measured 80.9 ms steady p95 / 82.8 ms steady max.
    # Five 30 Hz frames preserve two full frames of scheduling margin.
    model_prefetch_lead_steps: int = 5
    human_end_timeout_s: float = 1.5
    freshness_s: float = 0.1
    duration_s: float = 30.0
    max_steps: int | None = None
    output_dir: Path = Path("~/rlt_takeover_data").expanduser()
    episode_id: str = "episode_000"
    prompt: str = DEFAULT_PROMPT
    policy_host: str = "127.0.0.1"
    policy_port: int = 8000
    human_command_topic: str = "/joint_states_gripper"
    feedback_topic: str = "/joint_states_single"
    selected_command_topic: str = "/rlt/selected_joint_command"
    publish_commands: bool = False
    publish_authorization: str | None = None
    model_safety_profile: str = "native"
    model_smoothing_tau_s: float = 0.05
    model_max_joint_step_deg: float = 3.0
    model_max_gripper_step: float = 0.02
    hardware_io: str = "native_sdk"
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
    # The unchanged Actor head/service speaks the legacy rank1 protocol schema.
    # V2 execution logs a separate post-filter physical-action schema.
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
    actor_prefetch_lead_steps: int = 8
    scripted_keys: str | None = None
    start_human: bool = False
    wait_for_reward_after_max_steps: bool = False
    dry_import_check: bool = False

    def validate(self) -> "TakeoverRuntimeConfig":
        if self.control_hz != 30.0:
            raise ValueError("Piper RLT takeover rollout is fixed at 30 Hz")
        if self.action_dim != 7:
            raise ValueError("Piper runtime command dimension must be 7")
        if self.chunk_length < 1:
            raise ValueError("chunk_length must be positive")
        if self.model_execute_steps < 1 or self.model_execute_steps > 50:
            raise ValueError("model_execute_steps must be between 1 and 50")
        if (
            self.model_prefetch_lead_steps < 1
            or self.model_prefetch_lead_steps >= self.model_execute_steps
        ):
            raise ValueError("model_prefetch_lead_steps must be in [1, model_execute_steps-1]")
        if self.human_end_timeout_s <= 0 or self.human_end_timeout_s > 30:
            raise ValueError("human_end_timeout_s must be in (0, 30]")
        if self.freshness_s <= 0:
            raise ValueError("freshness_s must be positive")
        if self.duration_s <= 0 or self.duration_s > 600:
            raise ValueError("duration_s must be in (0, 600]")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when provided")
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        if self.publish_commands and self.publish_authorization != PUBLISH_AUTHORIZATION:
            raise ValueError(f"live publish authorization must equal {PUBLISH_AUTHORIZATION!r}")
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
        phase_classifier_checkpoint = None
        if self.phase_classifier_checkpoint is not None:
            phase_classifier_checkpoint = Path(self.phase_classifier_checkpoint).expanduser()
            if not phase_classifier_checkpoint.exists():
                raise ValueError(f"phase classifier checkpoint does not exist: {phase_classifier_checkpoint}")
        PhaseGateConfig(
            enter_threshold=self.phase_enter_threshold,
            enter_consecutive_frames=self.phase_enter_frames,
            classifier_period=self.phase_classifier_period,
        ).validate()
        if not str(self.phase_classifier_device).strip():
            raise ValueError("phase_classifier_device must be non-empty")
        actor_live_max_chunks = validate_actor_runtime_contract(
            self,
            context="runtime",
            phase_classifier_checkpoint=phase_classifier_checkpoint,
            chunk_length=self.chunk_length,
            action_dim=self.action_dim,
            actor_prefetch_lead_steps=self.actor_prefetch_lead_steps,
        )
        return dataclasses.replace(
            self,
            phase_classifier_checkpoint=phase_classifier_checkpoint,
            phase_classifier_device=str(self.phase_classifier_device).strip(),
            actor_live_max_chunks=actor_live_max_chunks,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Piper RLT pi0.5/Pika human takeover rollout")
    parser.add_argument("--output-dir", type=Path, default=Path("~/rlt_takeover_data").expanduser())
    parser.add_argument("--episode-id", default="episode_000")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--freshness", type=float, default=0.1)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--wait-for-reward-after-max-steps",
        action="store_true",
        help=(
            "When the model motion budget is exhausted, hold the current pose and keep the episode open "
            "until the operator enters reward 1/0 (or q)."
        ),
    )
    parser.add_argument(
        "--model-execute-steps",
        "--execute-steps",
        dest="model_execute_steps",
        type=int,
        default=50,
        help="How many absolute policy actions to play from each pi0.5/SFT chunk before switching to a newer chunk.",
    )
    parser.add_argument(
        "--model-prefetch-lead-steps",
        type=int,
        default=5,
        help=(
            "Begin the next H50 request this many 30 Hz frames before the active behavior plan ends. "
            "The result is staged and velocity-aligned only at the real H50 boundary."
        ),
    )
    parser.add_argument(
        "--human-end-timeout",
        type=float,
        default=1.5,
        help=(
            "Seconds without fresh Pika demonstrator commands before an armed/active human takeover "
            "is treated as finished and the episode waits for 1/0 reward."
        ),
    )
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--publish-authorization", default=None)
    parser.add_argument(
        "--model-safety-profile",
        default="native",
        choices=["native", "autonomous"],
        help=(
            "How to execute pi0.5/SFT model commands. "
            "'native' matches the direct inference path by bypassing RLT velocity/acceleration smoothing; "
            "'autonomous' keeps the older RLT conservative filter."
        ),
    )
    parser.add_argument("--model-smoothing-tau", type=float, default=0.05)
    parser.add_argument("--model-max-joint-step-deg", type=float, default=3.0)
    parser.add_argument("--model-max-gripper-step", type=float, default=0.02)
    parser.add_argument(
        "--hardware-io",
        default="native_sdk",
        choices=["native_sdk", "ros_bridge"],
        help="Use the original piper_sdk feedback/command path, or the legacy ROS command bridge.",
    )
    parser.add_argument(
        "--phase-classifier-checkpoint",
        type=Path,
        default=None,
        help="Optional ResNet phase classifier checkpoint. When provided, gate_active enters once and latches.",
    )
    parser.add_argument(
        "--phase-classifier-device",
        default="cpu",
        help="Torch device for phase classifier. Default cpu avoids RTX 5090 PyTorch CUDA incompatibility.",
    )
    parser.add_argument("--phase-enter-threshold", type=float, default=0.5)
    parser.add_argument("--phase-enter-frames", type=int, default=3)
    parser.add_argument(
        "--phase-classifier-period",
        type=int,
        default=1,
        help="Run the phase classifier every N control steps; skipped steps reuse gate state.",
    )
    parser.add_argument(
        "--actor-shadow",
        action="store_true",
        help=(
            "Read and log z_rl/a_actor from the policy response using C=10. "
            "The actor is never connected to the command mux or publisher."
        ),
    )
    parser.add_argument("--actor-shadow-expected-z-dim", type=int, default=2048)
    parser.add_argument("--actor-prefetch-lead-steps", type=int, default=8)
    parser.add_argument(
        "--actor-shadow-max-latency",
        type=float,
        default=0.333,
        help="Mark shadow predictions slower than this many seconds as late; pi0.5 control remains unchanged.",
    )
    parser.add_argument(
        "--actor-live",
        action="store_true",
        help=(
            "Allow a validated Actor chunk to control only while the frozen phase gate is latched. "
            "Human takeover remains highest priority; invalid/late Actor output falls back to Pi0.5."
        ),
    )
    parser.add_argument("--actor-live-authorization", default=None)
    parser.add_argument(
        "--actor-live-max-chunks",
        type=int,
        default=0,
        help=(
            "Maximum complete C=10 Actor chunks allowed to control per episode. "
            "0 means unlimited; use 1 for a one-chunk live canary."
        ),
    )
    parser.add_argument(
        "--actor-live-max-boundary-jump-rad",
        type=float,
        default=0.02,
        help=(
            "Fail closed for an entire C=10 Actor plan when its first six-joint "
            "absolute command differs from the previous executed target by more "
            "than this many radians."
        ),
    )
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
        help=(
            "Physical/replay execution schema. Keep the Actor protocol schema "
            "on --action-schema-fingerprint; v2 requires its distinct v4 "
            "post-filter schema here."
        ),
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
            "Explicit runtime interpretation of the unchanged legacy C10 Actor. "
            "persistent_c10_from_rank1_v1 carries each safe residual knot across "
            "C10/H50 boundaries; the default preserves old endpoint-zero behavior."
        ),
    )
    parser.add_argument(
        "--start-human",
        action="store_true",
        help="Start the episode already in Pika human takeover mode; the first typed 's' then ends the motion phase.",
    )
    parser.add_argument(
        "--scripted-keys",
        default=None,
        help="Non-interactive key schedule like '0:s,30:s,35:1' for shadow self-checks.",
    )
    parser.add_argument("--dry-import-check", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> TakeoverRuntimeConfig:
    return TakeoverRuntimeConfig(
        output_dir=args.output_dir,
        episode_id=args.episode_id,
        prompt=args.prompt,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        freshness_s=args.freshness,
        duration_s=args.duration,
        max_steps=args.max_steps,
        model_execute_steps=args.model_execute_steps,
        model_prefetch_lead_steps=args.model_prefetch_lead_steps,
        human_end_timeout_s=args.human_end_timeout,
        publish_commands=bool(args.publish),
        publish_authorization=args.publish_authorization,
        model_safety_profile=args.model_safety_profile,
        model_smoothing_tau_s=args.model_smoothing_tau,
        model_max_joint_step_deg=args.model_max_joint_step_deg,
        model_max_gripper_step=args.model_max_gripper_step,
        hardware_io=args.hardware_io,
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
        actor_prefetch_lead_steps=args.actor_prefetch_lead_steps,
        scripted_keys=args.scripted_keys,
        start_human=bool(args.start_human),
        wait_for_reward_after_max_steps=bool(args.wait_for_reward_after_max_steps),
        dry_import_check=bool(args.dry_import_check),
    ).validate()


@dataclasses.dataclass
class TakeoverLoopCore:
    config: TakeoverRuntimeConfig
    cameras: Any
    feedback_reader: Any
    human_tracker: Any
    policy_worker: Any
    keyboard: RLTKeyboardStateMachine
    key_source: Any
    image_writer: Any
    logger: RLTEpisodeLogger
    command_publisher: Any | None = None
    teleop_controller: Any | None = None
    human_passthrough: Any | None = None
    safety_filter: StatefulSafetyFilter | None = None
    phase_gate: SingleLatchPhaseGate | None = None
    phase_classifier: Any | None = None
    # Live RLT supplies a second websocket/worker so a C10 enrichment request
    # cannot occupy or coalesce away the control-critical H50 request.  Tests
    # and old callers may omit it and retain the historical shared-worker path.
    enrichment_policy_worker: Any | None = None
    now_fn: Any = time.monotonic
    sleep_fn: Any = time.sleep
    _model_plan: np.ndarray | None = dataclasses.field(default=None, init=False, repr=False)
    _model_plan_z_rl: np.ndarray | None = dataclasses.field(default=None, init=False, repr=False)
    _model_plan_actor: np.ndarray | None = dataclasses.field(default=None, init=False, repr=False)
    _model_plan_metadata: dict[str, Any] = dataclasses.field(default_factory=dict, init=False, repr=False)
    _model_plan_key: tuple[float, float] | None = dataclasses.field(default=None, init=False, repr=False)
    _model_plan_index: int = dataclasses.field(default=0, init=False, repr=False)
    _model_plan_limit: int = dataclasses.field(default=0, init=False, repr=False)
    _enrichment_plan: np.ndarray | None = dataclasses.field(default=None, init=False, repr=False)
    _enrichment_plan_z_rl: np.ndarray | None = dataclasses.field(default=None, init=False, repr=False)
    _enrichment_plan_actor: np.ndarray | None = dataclasses.field(default=None, init=False, repr=False)
    _enrichment_plan_metadata: dict[str, Any] = dataclasses.field(default_factory=dict, init=False, repr=False)
    _enrichment_plan_key: tuple[float, float] | None = dataclasses.field(default=None, init=False, repr=False)
    _enrichment_plan_index: int = dataclasses.field(default=0, init=False, repr=False)
    _enrichment_plan_limit: int = dataclasses.field(default=0, init=False, repr=False)
    _prefetched_enrichment_plan: np.ndarray | None = dataclasses.field(default=None, init=False, repr=False)
    _prefetched_enrichment_z_rl: np.ndarray | None = dataclasses.field(default=None, init=False, repr=False)
    _prefetched_enrichment_actor: np.ndarray | None = dataclasses.field(default=None, init=False, repr=False)
    _prefetched_enrichment_metadata: dict[str, Any] | None = dataclasses.field(default=None, init=False, repr=False)
    _prefetched_enrichment_key: tuple[float, float] | None = dataclasses.field(default=None, init=False, repr=False)
    _prefetched_model_plan: np.ndarray | None = dataclasses.field(default=None, init=False, repr=False)
    _prefetched_model_z_rl: np.ndarray | None = dataclasses.field(default=None, init=False, repr=False)
    _prefetched_model_metadata: dict[str, Any] | None = dataclasses.field(default=None, init=False, repr=False)
    _prefetched_model_key: tuple[float, float] | None = dataclasses.field(default=None, init=False, repr=False)
    _last_policy_output_key: tuple[float, float] | None = dataclasses.field(default=None, init=False, repr=False)
    _policy_request_pending: bool = dataclasses.field(default=False, init=False, repr=False)
    _policy_request_reason: str | None = dataclasses.field(default=None, init=False, repr=False)
    _policy_request_lead_steps: int | None = dataclasses.field(default=None, init=False, repr=False)
    _policy_request_behavior_target: BehaviorReferenceTarget | None = dataclasses.field(
        default=None, init=False, repr=False
    )
    _policy_wait_started_s: float | None = dataclasses.field(default=None, init=False, repr=False)
    _first_model_ready_announced: bool = dataclasses.field(default=False, init=False, repr=False)
    _last_exec_command: np.ndarray | None = dataclasses.field(default=None, init=False, repr=False)
    _previous_exec_command: np.ndarray | None = dataclasses.field(default=None, init=False, repr=False)
    _last_exec_source: str | None = dataclasses.field(default=None, init=False, repr=False)
    _last_feedback_hold_metadata: dict[str, Any] = dataclasses.field(
        default_factory=dict, init=False, repr=False
    )
    _base_last_policy_output_key: tuple[float, float] | None = dataclasses.field(
        default=None, init=False, repr=False
    )
    _base_request_pending: bool = dataclasses.field(default=False, init=False, repr=False)
    _base_request_reason: str | None = dataclasses.field(default=None, init=False, repr=False)
    _base_request_lead_steps: int | None = dataclasses.field(default=None, init=False, repr=False)
    _base_wait_started_s: float | None = dataclasses.field(default=None, init=False, repr=False)
    _actor_enrichment_min_observation_t: int | None = dataclasses.field(
        default=None, init=False, repr=False
    )
    _handoff_last_command: np.ndarray | None = dataclasses.field(default=None, init=False, repr=False)
    _handoff_previous_command: np.ndarray | None = dataclasses.field(default=None, init=False, repr=False)
    # The persistent Actor execution profile owns these residuals.  Keeping
    # explicit mirrors here lets H50 align its *base* target without counting
    # a carried Actor offset twice.
    _last_exec_actor_residual: np.ndarray | None = dataclasses.field(
        default=None, init=False, repr=False
    )
    _previous_exec_actor_residual: np.ndarray | None = dataclasses.field(
        default=None, init=False, repr=False
    )
    _boundary_keepalive_count: int = dataclasses.field(default=0, init=False, repr=False)
    # V2 owns an independently evolving base-only safety-filter trajectory.
    # It is synchronized from the real filter once at Actor/phase entry, never
    # cloned from the real filter on every frame.
    _actor_base_safety_filter: StatefulSafetyFilter | None = dataclasses.field(
        default=None,
        init=False,
        repr=False,
    )
    _human_base_safety_filter: StatefulSafetyFilter | None = dataclasses.field(
        default=None,
        init=False,
        repr=False,
    )

    def run_steps(self, *, max_steps: int) -> dict[str, Any]:
        config = self.config.validate()
        self.config = config
        dt = 1.0 / config.control_hz
        mux = RLTCommandMux(action_dim=config.action_dim, freshness_s=config.freshness_s)
        actor_governor_config = ActorResidualGovernorConfig(
            chunk_length=config.chunk_length,
            action_dim=config.action_dim,
            actor_residual_max_rad=config.actor_residual_max_rad,
            actor_residual_d1_max_rad=config.actor_residual_d1_max_rad,
            actor_residual_d2_max_rad=config.actor_residual_d2_max_rad,
            actor_direction_cone_deg=config.actor_direction_cone_deg,
            max_boundary_jump_rad=config.actor_live_max_boundary_jump_rad,
            gripper_residual_mode=config.actor_gripper_residual_mode,
            gripper_residual_max_close_m=(
                config.actor_gripper_residual_max_close_m
            ),
            gripper_residual_d1_max_m=(
                config.actor_gripper_residual_d1_max_m
            ),
            gripper_residual_d2_max_m=(
                config.actor_gripper_residual_d2_max_m
            ),
            gripper_max_boundary_jump_m=(
                config.actor_gripper_max_boundary_jump_m
            ),
            gripper_command_min_m=config.actor_gripper_command_min_m,
            gripper_command_max_m=config.actor_gripper_command_max_m,
            gripper_release_reference_m=(
                config.actor_gripper_release_reference_m
            ),
            gripper_release_delta_m=config.actor_gripper_release_delta_m,
        )
        persistent_actor_execution = bool(
            config.actor_execution_profile
            in {
                PERSISTENT_C10_EXECUTION_CONTRACT,
                PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT,
            }
        )
        filtered_actual_execution = bool(
            config.actor_execution_profile
            == PERSISTENT_FILTERED_ACTUAL_EXECUTION_CONTRACT
        )
        actor_governor = (
            PersistentActorResidualGovernor(actor_governor_config)
            if persistent_actor_execution
            else ActorResidualGovernor(actor_governor_config)
        )
        if persistent_actor_execution:
            # A new episode/reset must never inherit residual carry from a
            # previous physical scene, even if the governor implementation is
            # later made reusable across episode objects.
            actor_governor.reset_execution_state("episode_start")
        published_commands = 0
        takeover_started_t: int | None = None
        takeover_ended_t: int | None = None
        last_human_command_s: float | None = None
        last_source = "none"
        outcome = "max_steps"
        first_capture = True
        announced_mode = "MODEL"
        takeover_activation_armed = False
        takeover_seen_inactive = False
        actor_live_max_chunks = config.actor_live_max_chunks
        actor_live_chunks_started = 0
        actor_live_chunks_completed = 0
        actor_live_active_plan_id: str | None = None
        actor_live_active_steps = 0
        actor_live_next_offset = 0
        actor_live_limit_announced = False
        actor_live_approved_plan_ids: set[str] = set()
        actor_live_rejected_plan_ids: set[str] = set()
        previous_phase_active = False
        previous_keyboard_mode = "MODEL"
        last_exec_command: np.ndarray | None = None
        self._last_exec_command = None
        self._previous_exec_command = None
        self._last_exec_source = None
        self._handoff_last_command = None
        self._handoff_previous_command = None
        self._last_exec_actor_residual = None
        self._previous_exec_actor_residual = None
        self._boundary_keepalive_count = 0
        self._actor_base_safety_filter = None
        self._human_base_safety_filter = None
        self._actor_enrichment_min_observation_t = None
        human_canonical_plan_serial = 0
        human_canonical_plan_id: str | None = None
        human_canonical_plan_offset = 0
        human_canonical_boundary_anchor: np.ndarray | None = None

        t = 0
        motion_budget_exhausted = False
        while t < max_steps or config.wait_for_reward_after_max_steps:
            if t >= max_steps and not motion_budget_exhausted:
                self.keyboard.end_motion_phase(now_s=self.now_fn())
                motion_budget_exhausted = True
                print(
                    f"[{config.episode_id}] model motion limit reached after {max_steps} control steps; "
                    "holding current pose and waiting for operator reward 1/0 (q = emergency failure)",
                    flush=True,
                )
            if hasattr(self.key_source, "t"):
                self.key_source.t = t
            step_started = self.now_fn()
            for key in self.key_source.poll_keys():
                self.keyboard.press(str(key), now_s=step_started)
            keyboard_snapshot = self.keyboard.snapshot(now_s=step_started)

            warming_cameras = first_capture
            camera_read_started_s = self.now_fn()
            images = self.cameras.read(timeout_ms=5000, warmup_frames=60 if first_capture else 1)
            first_capture = False
            if warming_cameras:
                print(
                    f"[{config.episode_id}] dual camera streams ready after "
                    f"{self.now_fn() - camera_read_started_s:.2f}s; requesting first policy plan now",
                    flush=True,
                )
            state = np.asarray(self.feedback_reader.read(), dtype=np.float32)
            if state.shape != (config.action_dim,) or not np.all(np.isfinite(state)):
                raise RuntimeError(f"feedback state must be finite with shape ({config.action_dim},)")
            observation = build_observation(images, state, prompt=config.prompt)

            phase_snapshot = self._update_phase_gate(
                t=t,
                images=images,
                keyboard=keyboard_snapshot,
            )
            persistent_phase_exit_reset = bool(
                persistent_actor_execution
                and previous_phase_active
                and not phase_snapshot.active
            )
            if persistent_phase_exit_reset:
                # The next phase entry starts from zero carry. Reusing the last
                # insertion correction after the phase classifier exits would
                # apply a stale task-space intent to a new visual context.
                actor_governor.reset_execution_state("phase_exit")
                actor_live_active_plan_id = None
                actor_live_active_steps = 0
                actor_live_next_offset = 0
                self._actor_base_safety_filter = None
                self._invalidate_actor_enrichment(
                    reason="phase_exit",
                    min_observation_t=t + 1,
                )
            previous_phase_active = bool(phase_snapshot.active)
            enrichment_active = bool(
                phase_snapshot.active or keyboard_snapshot.mode == "HUMAN"
            )
            request_reasons: list[tuple[str, str]] = []
            request_allowed = bool(
                keyboard_snapshot.mode in {"MODEL", "HUMAN"}
                or (phase_snapshot.active and keyboard_snapshot.mode == "TAKEOVER_ARMED")
            )
            if request_allowed:
                current_policy_observation_t = self._enrichment_plan_metadata.get("policy_observation_t")
                phase_requires_fresh = bool(
                    phase_snapshot.active
                    and phase_snapshot.enter_t is not None
                    and (
                        current_policy_observation_t is None
                        or int(current_policy_observation_t) < int(phase_snapshot.enter_t)
                    )
                )
                human_requires_fresh = bool(
                    keyboard_snapshot.mode == "HUMAN"
                    and takeover_started_t is not None
                    and (
                        current_policy_observation_t is None
                        or int(current_policy_observation_t) < int(takeover_started_t)
                    )
                )
                request_reasons = self._policy_request_kinds(
                    enrichment_active=enrichment_active,
                    force_enrichment_refresh=phase_requires_fresh or human_requires_fresh,
                )
            behavior_remaining_before_requests = self._effective_plan_remaining()
            base_priority_guard_steps = (
                int(config.model_prefetch_lead_steps)
                + int(config.actor_prefetch_lead_steps)
                + 1
            )
            base_request_scheduled_this_step = any(
                lane == "base" for lane, _reason in request_reasons
            )
            base_priority_active = bool(
                self.enrichment_policy_worker is not None
                and (
                    self._base_request_pending
                    or base_request_scheduled_this_step
                    or (
                        behavior_remaining_before_requests is not None
                        and behavior_remaining_before_requests
                        <= base_priority_guard_steps
                    )
                )
            )
            enrichment_suppressed_base_priority = False
            enrichment_suppression_base_priority_reason = None
            for request_lane, request_reason in request_reasons:
                if request_lane == "enrichment" and base_priority_active:
                    # Both websocket clients ultimately execute synchronous JAX
                    # inference on one server event loop. Do not let a new token
                    # /Actor job occupy that loop inside the H50 standby window.
                    # Persistent carry (or zero carry) fills the skipped C10.
                    enrichment_suppressed_base_priority = True
                    if self._base_request_pending:
                        enrichment_suppression_base_priority_reason = (
                            "base_request_pending"
                        )
                    elif base_request_scheduled_this_step:
                        enrichment_suppression_base_priority_reason = (
                            "base_request_scheduled"
                        )
                    else:
                        enrichment_suppression_base_priority_reason = (
                            "behavior_remaining_inside_base_priority_guard"
                        )
                    continue
                # Timestamp after camera acquisition and deep-copy the observation:
                # RealSense arrays may otherwise be reused while the async worker is
                # still performing inference.
                request_timestamp_s = self.now_fn()
                policy_observation = copy.deepcopy(observation)
                behavior_target = (
                    None
                    if request_lane == "base"
                    else self._actor_request_behavior_target(
                        request_reason=request_reason,
                        state_snapshot=state,
                    )
                )
                if request_lane == "base":
                    policy_observation = add_base_only_request(policy_observation)
                elif request_lane == "enrichment":
                    # A split enrichment request is useful only when it can be
                    # tied to an exact executable C10 slice. Never fall through
                    # to the stochastic base policy when that target is absent.
                    if behavior_target is None:
                        continue
                    policy_observation = add_actor_enrichment_only_request(
                        policy_observation,
                        behavior_target,
                    )
                elif behavior_target is not None:
                    policy_observation = add_behavior_reference(
                        policy_observation,
                        behavior_target,
                    )
                request_worker = (
                    self.policy_worker
                    if request_lane in {"base", "shared"}
                    else self.enrichment_policy_worker
                )
                if request_worker is None:  # pragma: no cover - validate scheduling invariant
                    raise RuntimeError(f"missing policy worker for {request_lane!r} request lane")
                request_worker.submit(
                    policy_observation,
                    timestamp_s=request_timestamp_s,
                    observation_t=t,
                )
                request_lead_steps = (
                    self._enrichment_plan_remaining()
                    if request_lane == "enrichment"
                    else self._effective_plan_remaining()
                )
                self._mark_policy_request_submitted(
                    lane=request_lane,
                    reason=request_reason,
                    lead_steps=request_lead_steps,
                    behavior_target=behavior_target,
                )
                # Preserve the previous startup contract: the first episode row is
                # recorded only after a real plan exists. Later C=10 requests are
                # prefetched without blocking the 30 Hz command loop.
                if (
                    request_lane in {"base", "shared"}
                    and self._model_plan is None
                    and hasattr(request_worker, "wait_for_result")
                ):
                    ready = request_worker.wait_for_result(
                        observation_timestamp_s=request_timestamp_s,
                        timeout_s=5.0,
                    )
                    if ready is None:
                        raise RuntimeError("timed out waiting for the initial policy plan")

            a_ref, z_rl, a_actor, policy_metadata, model_command = self._latest_policy_command(
                now_s=self.now_fn(),
                action_dim=config.action_dim,
                chunk_length=config.chunk_length,
                state_snapshot=state,
                actor_phase_active=bool(phase_snapshot.active),
                enrichment_active=enrichment_active,
            )
            policy_metadata.setdefault("h50_boundary_keepalive", False)
            policy_metadata.setdefault("h50_boundary_keepalive_count", self._boundary_keepalive_count)
            policy_metadata.setdefault("h50_keepalive_preserves_velocity_history", False)
            policy_metadata.update(
                {
                    "actor_enrichment_suppressed_base_priority": (
                        enrichment_suppressed_base_priority
                    ),
                    "actor_enrichment_suppression_base_priority_reason": (
                        enrichment_suppression_base_priority_reason
                    ),
                    "base_priority_guard_steps": base_priority_guard_steps,
                    "base_priority_behavior_remaining": (
                        behavior_remaining_before_requests
                    ),
                    "base_priority_request_pending": bool(
                        self._base_request_pending
                    ),
                    "base_priority_request_scheduled_this_step": (
                        base_request_scheduled_this_step
                    ),
                }
            )
            if model_command is None:
                model_command = self._model_boundary_keepalive(now_s=self.now_fn())
                if model_command is not None:
                    self._boundary_keepalive_count += 1
                    policy_metadata.update(
                        {
                            "h50_boundary_keepalive": True,
                            "h50_boundary_keepalive_count": self._boundary_keepalive_count,
                            "h50_keepalive_target": model_command.value.astype(float).tolist(),
                            "h50_keepalive_preserves_velocity_history": True,
                        }
                    )
            if model_command is not None and not self._first_model_ready_announced:
                latency_s = float(policy_metadata.get("policy_inference_latency_s", 0.0))
                print(
                    f"[{config.episode_id}] first policy plan ready after {latency_s:.2f}s; "
                    "model command publication and replay recording start now",
                    flush=True,
                )
                self._first_model_ready_announced = True
            # ROS callbacks run concurrently with the 30 Hz loop.  A human
            # message can arrive after ``step_started`` but before selection;
            # compare it against a timestamp sampled after reading the tracker
            # so a genuinely fresh command is never classified as "future".
            human_now_s = self.now_fn()
            human_command = self.human_tracker.latest(now_s=human_now_s)
            teleop_active = self._teleop_observed_active(human_command)
            takeover_activation_edge_accepted = False
            if human_command is not None and human_command.value is not None:
                last_human_command_s = human_now_s

            if keyboard_snapshot.mode == "TAKEOVER_ARMED":
                if not takeover_activation_armed:
                    takeover_activation_armed = True
                    takeover_seen_inactive = not teleop_active
                    # A previous episode may have left the original teleop
                    # publisher active. Close that old activation once; the
                    # operator's next physical double-click then creates the
                    # required new off->on edge.
                    if teleop_active:
                        self._request_teleop_inactive()
                elif not teleop_active:
                    takeover_seen_inactive = True

                if (
                    takeover_seen_inactive
                    and teleop_active
                    and human_command is not None
                    and human_command.value is not None
                ):
                    keyboard_snapshot = self.keyboard.start_takeover(now_s=human_now_s)
                    takeover_activation_armed = False
                    takeover_activation_edge_accepted = True
            elif (
                keyboard_snapshot.mode == "HUMAN"
                and last_human_command_s is not None
                and human_now_s - last_human_command_s >= config.human_end_timeout_s
            ):
                keyboard_snapshot = self.keyboard.end_motion_phase(now_s=human_now_s)

            if self.human_passthrough is not None:
                # Keep the high-rate pass-through alive only while the RLT
                # state machine has accepted the physical Pika activation.
                # Its own short heartbeat timeout fails closed if this loop
                # exits unexpectedly.
                human_active = keyboard_snapshot.mode == "HUMAN"
                set_external_passthrough = getattr(
                    self.command_publisher, "set_external_passthrough_active", None
                )
                if human_active:
                    # Synchronously silence the periodic SDK publisher before
                    # allowing the official Pika controller to write CAN.
                    if set_external_passthrough is not None:
                        set_external_passthrough(True)
                    self.human_passthrough.set_active(True)
                else:
                    # Reverse the order on exit: close Pika first, then allow
                    # only a newly submitted model/hold target to resume SDK IO.
                    self.human_passthrough.set_active(False)
                    if set_external_passthrough is not None:
                        set_external_passthrough(False)

            persistent_human_transition_had_active_plan = False
            persistent_human_transition_reset = bool(
                persistent_actor_execution
                and (
                    (
                        previous_keyboard_mode != "HUMAN"
                        and keyboard_snapshot.mode == "HUMAN"
                    )
                    or (
                        previous_keyboard_mode == "HUMAN"
                        and keyboard_snapshot.mode != "HUMAN"
                    )
                )
            )
            if persistent_human_transition_reset:
                persistent_human_transition_had_active_plan = bool(
                    actor_live_active_plan_id is not None
                )
                transition_reason = (
                    "human_takeover_enter"
                    if keyboard_snapshot.mode == "HUMAN"
                    else "human_takeover_exit"
                )
                actor_governor.reset_execution_state(transition_reason)
                actor_live_active_plan_id = None
                actor_live_active_steps = 0
                actor_live_next_offset = 0
                self._actor_base_safety_filter = None
                self._human_base_safety_filter = None
                human_canonical_plan_id = None
                human_canonical_plan_offset = 0
                human_canonical_boundary_anchor = None
                self._invalidate_actor_enrichment(
                    reason=transition_reason,
                    min_observation_t=t + 1,
                )
            previous_keyboard_mode = str(keyboard_snapshot.mode)

            if keyboard_snapshot.mode != announced_mode:
                mode_messages = {
                    "TAKEOVER_ARMED": "takeover armed; model held, double-click Pika to start human replay recording",
                    "HUMAN": (
                        "post-arm Pika double-click activation detected; "
                        "human control and replay recording active"
                    ),
                    "WAIT_FOR_REWARD": "motion phase ended; holding arm and waiting for 1/0 reward",
                    "STOPPED": "terminal key received; closing episode record",
                }
                message = mode_messages.get(keyboard_snapshot.mode, f"mode -> {keyboard_snapshot.mode}")
                print(f"[{config.episode_id}] {message}", flush=True)
                announced_mode = keyboard_snapshot.mode

            if keyboard_snapshot.takeover_started_s is not None and takeover_started_t is None:
                takeover_started_t = t
            if keyboard_snapshot.takeover_ended_s is not None and takeover_ended_t is None:
                takeover_ended_t = t

            # A boundary miss can hold while the async plan finishes. Re-read
            # feedback immediately before selection so every executable command
            # uses the current controller state and clock domain.
            execution_state = np.asarray(self.feedback_reader.read(), dtype=np.float32)
            if execution_state.shape != (config.action_dim,) or not np.all(np.isfinite(execution_state)):
                raise RuntimeError(f"execution feedback must be finite with shape ({config.action_dim},)")
            selection_now_s = self.now_fn()
            feedback_command = TimedCommand(value=execution_state, timestamp_s=selection_now_s)
            policy_observation_t = policy_metadata.get("policy_observation_t")
            gate_alignment_ready = bool(
                not phase_snapshot.active
                or (
                    policy_observation_t is not None
                    and phase_snapshot.enter_t is not None
                    and int(policy_observation_t) >= int(phase_snapshot.enter_t)
                )
            )
            # A plan inferred before the phase entry may be useful as SFT
            # behavior, but its residual Actor must never control the arm.
            actor_enrichment_fresh_after_reset = bool(
                self._actor_enrichment_min_observation_t is None
                or (
                    policy_observation_t is not None
                    and int(policy_observation_t)
                    >= int(self._actor_enrichment_min_observation_t)
                )
            )
            actor_ready = bool(
                policy_metadata.get("actor_shadow_ready", False)
                and gate_alignment_ready
                and actor_enrichment_fresh_after_reset
            )
            raw_actor_plan_id = policy_metadata.get("policy_plan_id")
            actor_plan_id = None if raw_actor_plan_id is None else str(raw_actor_plan_id)
            raw_actor_plan_offset = policy_metadata.get("plan_offset")
            try:
                actor_plan_offset = None if raw_actor_plan_offset is None else int(raw_actor_plan_offset)
            except (TypeError, ValueError):
                actor_plan_offset = None
            actor_payload_present = bool(a_actor is not None and len(a_actor))
            actor_plan_available = bool(actor_ready and actor_payload_present)
            actor_candidate = bool(config.actor_live and actor_plan_available)
            persistent_hold_plan = False
            persistent_zero_carry_hold = False
            persistent_behavior_boundary_restart = False
            actor_live_suppressed = False
            actor_live_suppression_reason: str | None = None
            actor_live_control_allowed = False
            actor_boundary_checked = False
            actor_boundary_anchor = None
            actor_boundary_jump_per_joint = None
            actor_boundary_jump_max_rad = None
            governed_plan = None
            governed_actor_action = None
            actor_governor_input_error = None
            if persistent_actor_execution and config.actor_live and phase_snapshot.active:
                current_behavior_plan_id = policy_metadata.get(
                    "actor_behavior_ref_current_plan_id"
                )
                raw_behavior_offset = policy_metadata.get("behavior_plan_offset")
                try:
                    current_behavior_offset = int(raw_behavior_offset)
                except (TypeError, ValueError):
                    current_behavior_offset = None

                active_persistent_plan = (
                    None
                    if actor_live_active_plan_id is None
                    else actor_governor.get(actor_live_active_plan_id)
                )
                if (
                    active_persistent_plan is not None
                    and current_behavior_plan_id is not None
                    and str(active_persistent_plan.behavior_plan_id)
                    != str(current_behavior_plan_id)
                ):
                    # A short/padded tail can end before its logical C10. Keep
                    # the actually committed carry, but never execute padded
                    # actions against the next H50 behavior plan.
                    actor_live_active_plan_id = None
                    actor_live_active_steps = 0
                    actor_live_next_offset = 0
                    active_persistent_plan = None
                    persistent_behavior_boundary_restart = True

                if (
                    active_persistent_plan is not None
                    and actor_live_next_offset < config.chunk_length
                ):
                    governed_plan = active_persistent_plan
                    actor_plan_id = active_persistent_plan.plan_id
                    actor_plan_offset = actor_live_next_offset
                    actor_plan_available = True
                    actor_candidate = True
                    persistent_hold_plan = bool(
                        getattr(active_persistent_plan, "hold_only", False)
                    )
                elif (
                    not actor_plan_available
                    or actor_plan_offset != 0
                ):
                    # A late/missing Actor prediction must not drop the
                    # already-executed correction. Build a C10 carry plan from
                    # the exact current H50 suffix; _slice_action_chunk has
                    # already padded a short tail, but only real model rows are
                    # ever selected before the next behavior plan replaces it.
                    if (
                        current_behavior_plan_id is not None
                        and current_behavior_offset is not None
                    ):
                        actor_boundary_anchor = (
                            execution_state.copy()
                            if last_exec_command is None
                            or last_source == "human_pika"
                            else last_exec_command.copy()
                        )
                        hold_plan_id = (
                            f"{config.episode_id}/persistent_hold/"
                            f"{current_behavior_plan_id}/{current_behavior_offset}"
                        )
                        try:
                            governed_plan = actor_governor.prepare_hold_plan(
                                plan_id=hold_plan_id,
                                behavior_plan_id=str(current_behavior_plan_id),
                                behavior_start_offset=current_behavior_offset,
                                behavior_ref=np.asarray(a_ref, dtype=np.float32),
                                boundary_anchor=actor_boundary_anchor,
                            )
                        except (TypeError, ValueError) as exc:
                            actor_governor_input_error = (
                                "persistent_hold_governor_input_error:"
                                f"{type(exc).__name__}: {exc}"
                            )
                        if governed_plan is not None:
                            actor_plan_id = governed_plan.plan_id
                            actor_plan_offset = 0
                            actor_plan_available = bool(governed_plan.approved)
                            actor_candidate = bool(
                                config.actor_live and governed_plan.approved
                            )
                            persistent_hold_plan = True
                            persistent_zero_carry_hold = bool(
                                np.allclose(
                                    governed_plan.carry_in[:6],
                                    0.0,
                                    rtol=0.0,
                                    atol=1e-12,
                                )
                            )

            if (
                config.actor_live
                and actor_payload_present
                and not actor_ready
                and not persistent_hold_plan
            ):
                actor_live_suppressed = True
                actor_live_suppression_reason = (
                    "actor_behavior_ref_mismatch"
                    if not policy_metadata.get(
                        "actor_behavior_ref_alignment_ready", False
                    )
                    else "actor_gate_or_shadow_not_ready"
                )
            # Evaluate the complete C10 governor in shadow mode too.  This is
            # what makes a shadow rollout a meaningful canary: it records the
            # exact projection/rejection result without permitting any Actor
            # command to reach the mux.
            if actor_plan_available:
                if actor_plan_id is None:
                    actor_governor_input_error = "missing_actor_plan_id"
                else:
                    governed_plan = actor_governor.get(actor_plan_id)
                if actor_plan_id is not None and governed_plan is None:
                    if actor_plan_offset != 0:
                        actor_governor_input_error = "actor_plan_governor_not_cached"
                    else:
                        actor_boundary_checked = True
                        actor_boundary_anchor = (
                            execution_state.copy()
                            if last_exec_command is None or last_source == "human_pika"
                            else last_exec_command.copy()
                        )
                        target_behavior_plan_id = policy_metadata.get(
                            "actor_behavior_ref_plan_id"
                        )
                        target_behavior_start_offset = policy_metadata.get(
                            "actor_behavior_ref_start_offset"
                        )
                        try:
                            if (
                                target_behavior_plan_id is None
                                or target_behavior_start_offset is None
                            ):
                                raise ValueError("missing behavior_ref target identity")
                            governed_plan = actor_governor.prepare_plan(
                                plan_id=actor_plan_id,
                                behavior_plan_id=str(target_behavior_plan_id),
                                behavior_start_offset=int(
                                    target_behavior_start_offset
                                ),
                                behavior_ref=np.asarray(a_ref, dtype=np.float32),
                                raw_actor=np.asarray(a_actor, dtype=np.float32),
                                boundary_anchor=actor_boundary_anchor,
                            )
                        except (TypeError, ValueError) as exc:
                            actor_governor_input_error = (
                                f"actor_governor_input_error:{type(exc).__name__}: {exc}"
                            )
                        if governed_plan is not None and not governed_plan.approved:
                            actor_live_rejected_plan_ids.add(actor_plan_id)
                        elif governed_plan is not None:
                            actor_live_approved_plan_ids.add(actor_plan_id)
                            actor_boundary_jump_per_joint = np.abs(
                                governed_plan.safe_actions[0, :6]
                                - actor_boundary_anchor[:6]
                            )
                            actor_boundary_jump_max_rad = float(
                                np.max(actor_boundary_jump_per_joint)
                            )
                if governed_plan is not None and governed_plan.approved and actor_plan_offset is not None:
                    try:
                        governed_actor_action = governed_plan.action_at(
                            actor_plan_offset
                        )
                    except IndexError:
                        actor_governor_input_error = (
                            "actor_plan_offset_outside_governed_chunk"
                        )
                actor_live_control_allowed = bool(
                    actor_candidate
                    and governed_plan is not None
                    and governed_plan.approved
                    and governed_actor_action is not None
                    and actor_governor_input_error is None
                )
                if config.actor_live and actor_candidate and not actor_live_control_allowed:
                    actor_live_suppressed = True
                    if actor_governor_input_error is not None:
                        actor_live_suppression_reason = actor_governor_input_error
                    elif governed_plan is not None and not governed_plan.approved:
                        actor_live_suppression_reason = (
                            "actor_residual_governor_rejected:"
                            f"{governed_plan.rejection_reason}"
                        )
                    else:
                        actor_live_suppression_reason = "actor_plan_governor_rejected"
            elif config.actor_live and actor_ready:
                actor_live_suppressed = True
                actor_live_suppression_reason = "actor_payload_missing"
            if (
                actor_live_control_allowed
                and config.actor_live
                and phase_snapshot.active
            ):
                if (
                    actor_live_max_chunks is not None
                    and actor_live_active_plan_id is None
                    and actor_live_chunks_started >= actor_live_max_chunks
                ):
                    actor_live_control_allowed = False
                    actor_live_suppressed = True
                    actor_live_suppression_reason = "max_chunks_reached"
                elif actor_candidate and actor_live_active_plan_id is not None:
                    if (
                        actor_plan_id != actor_live_active_plan_id
                        or actor_plan_offset != actor_live_next_offset
                    ):
                        actor_live_control_allowed = False
                        actor_live_suppressed = True
                        actor_live_suppression_reason = "active_chunk_sequence_mismatch"
                elif actor_candidate and actor_live_active_plan_id is None:
                    # Live control starts only at an actual Actor chunk boundary.
                    # Falling into a suffix after human control or a delayed
                    # result would otherwise execute a partial C=10 chunk.  This
                    # continuity rule also applies when the chunk limit is
                    # configured as zero/unlimited.
                    if actor_plan_id is None or actor_plan_offset != 0:
                        actor_live_control_allowed = False
                        actor_live_suppressed = True
                        actor_live_suppression_reason = "awaiting_c10_boundary"
            actor_command = None
            if actor_live_control_allowed and governed_actor_action is not None:
                actor_command = TimedCommand(
                    value=governed_actor_action,
                    timestamp_s=selection_now_s,
                )
            if governed_plan is not None:
                policy_metadata.update(governed_plan.metadata())
                policy_metadata["actor_governor_raw_residual_this_step"] = (
                    None
                    if actor_plan_offset is None
                    or actor_plan_offset < 0
                    or actor_plan_offset >= config.chunk_length
                    else governed_plan.raw_residual[actor_plan_offset].tolist()
                )
                policy_metadata["actor_governor_safe_action_this_step"] = (
                    None
                    if governed_actor_action is None
                    else governed_actor_action.tolist()
                )
                policy_metadata["actor_governor_safe_residual_this_step"] = (
                    None
                    if actor_plan_offset is None
                    or actor_plan_offset < 0
                    or actor_plan_offset >= config.chunk_length
                    else governed_plan.safe_residual[
                        actor_plan_offset
                    ].tolist()
                )
                # Canonical replay field: one full 7-D planned residual for
                # this row/offset.  Ten rows with the same plan id form C10.
                policy_metadata["actor_persistent_planned_residual"] = (
                    policy_metadata["actor_governor_safe_residual_this_step"]
                )
            policy_metadata["actor_governor_input_error"] = actor_governor_input_error
            policy_metadata["actor_governor_evaluated_in_shadow"] = bool(
                governed_plan is not None and not config.actor_live
            )
            selection = mux.select(
                now_s=selection_now_s,
                keyboard=keyboard_snapshot,
                feedback=feedback_command,
                model=model_command,
                human=human_command,
                actor=actor_command,
                rlt_active=bool(phase_snapshot.active),
                allow_actor_live=bool(config.actor_live),
            )
            actor_live_chunk_interrupted = bool(
                persistent_human_transition_had_active_plan
            )
            actor_live_chunk_interruption_reason: str | None = (
                "actor_chunk_interrupted_by_human_transition"
                if persistent_human_transition_had_active_plan
                else None
            )
            # Chunk progress is committed only after the final safety filter and
            # successful command publication below.  Selection alone is not
            # evidence that the arm received this Actor row.

            if actor_live_suppression_reason == "max_chunks_reached" and not actor_live_limit_announced:
                print(
                    f"[{config.episode_id}] Actor live limit reached after "
                    f"{actor_live_chunks_started} C={config.chunk_length} chunk(s); "
                    "phase remains active but commands now fall back to Pi0.5",
                    flush=True,
                )
                actor_live_limit_announced = True

            policy_metadata.update(
                {
                    "effective_execute_steps": (
                        config.chunk_length if selection.source == "rlt" else config.model_execute_steps
                    ),
                    "actor_live_max_chunks": 0 if actor_live_max_chunks is None else actor_live_max_chunks,
                    "actor_live_chunks_started": actor_live_chunks_started,
                    "actor_live_chunks_completed": actor_live_chunks_completed,
                    "actor_live_chunks_remaining": (
                        None
                        if actor_live_max_chunks is None
                        else max(0, actor_live_max_chunks - actor_live_chunks_started)
                    ),
                    "actor_live_chunk_plan_id": actor_live_active_plan_id,
                    "actor_live_chunk_steps": actor_live_active_steps,
                    "actor_live_control_allowed_this_step": bool(actor_live_control_allowed),
                    "actor_live_selected_this_step": selection.source == "rlt",
                    "actor_live_suppressed": actor_live_suppressed,
                    "actor_live_suppression_reason": actor_live_suppression_reason,
                    "actor_live_chunk_interrupted": actor_live_chunk_interrupted,
                    "actor_live_chunk_interruption_reason": actor_live_chunk_interruption_reason,
                    "actor_live_boundary_checked": actor_boundary_checked,
                    "actor_live_boundary_anchor": (
                        None if actor_boundary_anchor is None else actor_boundary_anchor.tolist()
                    ),
                    "actor_live_boundary_jump_per_joint_rad": (
                        None
                        if actor_boundary_jump_per_joint is None
                        else actor_boundary_jump_per_joint.tolist()
                    ),
                    "actor_live_boundary_jump_max_rad": actor_boundary_jump_max_rad,
                    "actor_live_boundary_jump_threshold_rad": (
                        config.actor_live_max_boundary_jump_rad
                    ),
                    "actor_residual_max_rad": config.actor_residual_max_rad,
                    "actor_residual_d1_max_rad": config.actor_residual_d1_max_rad,
                    "actor_residual_d2_max_rad": config.actor_residual_d2_max_rad,
                    "actor_direction_cone_deg": config.actor_direction_cone_deg,
                    "action_schema_fingerprint": config.action_schema_fingerprint,
                    "actor_projection_profile": config.actor_projection_profile,
                    "actor_execution_profile": config.actor_execution_profile,
                    "actor_enrichment_fresh_after_reset": (
                        actor_enrichment_fresh_after_reset
                    ),
                    "actor_enrichment_min_observation_t": (
                        self._actor_enrichment_min_observation_t
                    ),
                    "actor_persistent_human_transition_reset": (
                        persistent_human_transition_reset
                    ),
                    "actor_persistent_hold": persistent_hold_plan,
                    "actor_persistent_zero_carry_hold": (
                        persistent_zero_carry_hold
                    ),
                    "actor_persistent_behavior_boundary_restart": (
                        persistent_behavior_boundary_restart
                    ),
                    "h50_handoff_actor_replaced_by_persistent_hold": bool(
                        persistent_hold_plan
                        and policy_metadata.get(
                            "h50_handoff_actor_suppressed", False
                        )
                    ),
                    "actor_persistent_current_residual": (
                        actor_governor.current_residual.tolist()
                        if persistent_actor_execution
                        else None
                    ),
                    "actor_persistent_previous_residual": (
                        actor_governor.previous_residual.tolist()
                        if persistent_actor_execution
                        else None
                    ),
                    "actor_live_boundary_plan_approved": bool(
                        actor_plan_id is not None
                        and actor_plan_id in actor_live_approved_plan_ids
                    ),
                    "actor_live_boundary_plan_rejected": bool(
                        actor_plan_id is not None
                        and actor_plan_id in actor_live_rejected_plan_ids
                    ),
                }
            )
            planned_actor_residual = (
                None
                if governed_plan is None
                or actor_plan_offset is None
                or actor_plan_offset < 0
                or actor_plan_offset >= int(governed_plan.safe_residual.shape[0])
                else governed_plan.safe_residual[actor_plan_offset]
            )
            requested_selection_source = selection.source
            selected_command = (
                execution_state if selection.command is None else selection.command
            )
            actual_filter_boundary_anchor = (
                np.asarray(
                    self.safety_filter.previous_command,
                    dtype=np.float32,
                ).copy()
                if self.safety_filter is not None
                else (
                    execution_state.copy()
                    if last_exec_command is None
                    else last_exec_command.copy()
                )
            )
            actor_filtered_base_command = None
            actor_filtered_base_boundary_anchor = None
            actor_filtered_actual_certificate = None
            actor_filtered_actual_rejected = False
            human_filtered_base_command = None
            human_filtered_base_boundary_anchor = None
            human_filtered_base_valid = False
            human_filtered_base_error = None

            # V2 maintains a true base-only counterfactual across frames.  The
            # shadow filter starts from the real pre-Actor filter state once,
            # then sees only the uncorrected behavior target while the real
            # filter sees behavior+residual.
            if (
                filtered_actual_execution
                and selection.source == "rlt"
                and governed_plan is not None
                and governed_actor_action is not None
                and planned_actor_residual is not None
                and actor_plan_id is not None
                and actor_plan_offset is not None
            ):
                if (
                    self._actor_base_safety_filter is None
                    and self.safety_filter is not None
                ):
                    self._actor_base_safety_filter = copy.deepcopy(
                        self.safety_filter
                    )
                base_target = (
                    np.asarray(governed_actor_action, dtype=np.float32)
                    - np.asarray(planned_actor_residual, dtype=np.float32)
                )
                # Frozen-v2 keeps the absolute gripper pass-through contract.
                # Close-assist instead needs the true counterfactual base
                # opening so the physical residual can be certified/logged.
                if (
                    config.actor_gripper_residual_mode
                    == GRIPPER_RESIDUAL_FROZEN
                ):
                    base_target[6] = np.asarray(
                        governed_actor_action,
                        dtype=np.float32,
                    )[6]
                if self._actor_base_safety_filter is None:
                    actor_filtered_base_boundary_anchor = (
                        actual_filter_boundary_anchor.copy()
                    )
                    actor_filtered_base_command = base_target.copy()
                else:
                    actor_filtered_base_boundary_anchor = np.asarray(
                        self._actor_base_safety_filter.previous_command,
                        dtype=np.float32,
                    ).copy()
                    actor_filtered_base_command = self._filter_model_with(
                        self._actor_base_safety_filter,
                        base_target,
                        snapshot=execution_state,
                        feedback=execution_state,
                        dt=dt,
                    )

            # Human intervention remains trainable.  It gets an independent
            # model-only counterfactual, while its actual command still follows
            # the untouched native Pika path.
            if filtered_actual_execution and selection.source == "human_pika":
                if (
                    self._human_base_safety_filter is None
                    and self.safety_filter is not None
                ):
                    self._human_base_safety_filter = copy.deepcopy(
                        self.safety_filter
                    )
                if model_command is not None and model_command.value is not None:
                    try:
                        if self._human_base_safety_filter is None:
                            human_filtered_base_boundary_anchor = (
                                actual_filter_boundary_anchor.copy()
                            )
                            human_filtered_base_command = np.asarray(
                                model_command.value,
                                dtype=np.float32,
                            ).copy()
                        else:
                            human_filtered_base_boundary_anchor = np.asarray(
                                self._human_base_safety_filter.previous_command,
                                dtype=np.float32,
                            ).copy()
                            human_filtered_base_command = self._filter_model_with(
                                self._human_base_safety_filter,
                                np.asarray(
                                    model_command.value,
                                    dtype=np.float32,
                                ),
                                snapshot=execution_state,
                                feedback=execution_state,
                                dt=dt,
                            )
                        human_filtered_base_valid = True
                    except (HardwareSafetyError, TypeError, ValueError) as exc:
                        # Counterfactual logging must never interrupt native
                        # human control. It fails closed for training instead.
                        human_filtered_base_error = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        self._human_base_safety_filter = None

            exec_command, safety_profile, safety_reasons = self._filter_command(
                selected_command,
                source=selection.source,
                snapshot=execution_state,
                feedback=execution_state,
                dt=dt,
            )
            last_exec_command = np.asarray(exec_command, dtype=np.float32).copy()

            if (
                filtered_actual_execution
                and selection.source == "rlt"
                and actor_filtered_base_command is not None
                and actor_filtered_base_boundary_anchor is not None
                and actor_plan_id is not None
                and actor_plan_offset is not None
            ):
                try:
                    actor_filtered_actual_certificate = (
                        actor_governor.certify_filtered_execution(
                            plan_id=actor_plan_id,
                            offset=actor_plan_offset,
                            filtered_base_action=actor_filtered_base_command,
                            filtered_actual_action=last_exec_command,
                            base_boundary_anchor=(
                                actor_filtered_base_boundary_anchor
                            ),
                            actual_boundary_anchor=(
                                actual_filter_boundary_anchor
                            ),
                        )
                    )
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    actor_governor_input_error = (
                        "filtered_actual_certificate_error:"
                        f"{type(exc).__name__}: {exc}"
                    )
                if (
                    actor_filtered_actual_certificate is None
                    or not actor_filtered_actual_certificate.approved
                ):
                    # The real filter has already evaluated the Actor target,
                    # but nothing has been published. Replace that state and
                    # output with the independently filtered base trajectory.
                    last_exec_command = np.asarray(
                        actor_filtered_base_command,
                        dtype=np.float32,
                    ).copy()
                    if self._actor_base_safety_filter is not None:
                        self.safety_filter = copy.deepcopy(
                            self._actor_base_safety_filter
                        )
                    actor_filtered_actual_rejected = True
                    rejection_reason = (
                        actor_governor_input_error
                        if actor_filtered_actual_certificate is None
                        else actor_filtered_actual_certificate.rejection_reason
                    )
                    safety_reasons = list(safety_reasons) + [
                        "actor_filtered_actual_certificate_rejected:"
                        f"{rejection_reason}"
                    ]
                    selection = dataclasses.replace(
                        selection,
                        command=last_exec_command.copy(),
                        source="pi05",
                        reason="rlt_filtered_actual_rejected_fallback_model",
                    )
            # One authoritative delivered command feeds publication, logging,
            # handoff history, and replay. In particular, a rejected Actor
            # candidate must never survive through the stale pre-fallback
            # ``exec_command`` local.
            exec_command = last_exec_command.copy()

            self._previous_exec_command = (
                last_exec_command.copy()
                if self._last_exec_command is None
                else self._last_exec_command.copy()
            )
            self._last_exec_command = last_exec_command.copy()
            self._last_exec_source = selection.source
            actor_action_survived_filter = bool(
                selection.source == "rlt"
                and governed_plan is not None
                and governed_actor_action is not None
                and actor_plan_id is not None
                and actor_plan_offset is not None
                and (
                    (
                        filtered_actual_execution
                        and actor_filtered_actual_certificate is not None
                        and actor_filtered_actual_certificate.approved
                    )
                    or (
                        not filtered_actual_execution
                        and np.allclose(
                            last_exec_command,
                            governed_actor_action,
                            rtol=0.0,
                            atol=1e-6,
                        )
                    )
                )
            )
            selected_command_survived_filter = bool(
                np.allclose(
                    last_exec_command,
                    np.asarray(selected_command, dtype=np.float32),
                    rtol=0.0,
                    atol=1e-6,
                )
            )
            h50_boundary_keepalive = bool(
                policy_metadata.get("h50_boundary_keepalive", False)
            )
            persistent_commit_candidate = bool(
                persistent_actor_execution and actor_action_survived_filter
            )
            persistent_commit_status = None
            persistent_commit_error = None
            command_delivery_succeeded = False
            command_delivery_mode = "not_requested"
            if config.publish_commands and self.command_publisher is not None:
                publish_selected = getattr(self.command_publisher, "publish_selected", None)
                if publish_selected is None:
                    self.command_publisher.publish(exec_command)
                else:
                    publish_selected(exec_command, source=selection.source)
                published_commands += 1
                command_delivery_succeeded = True
                command_delivery_mode = "published"
            elif config.publish_commands:
                command_delivery_mode = "missing_publisher"
            else:
                # Offline replay may select an Actor row, but it is not physical
                # execution. Never advance persistent carry or live chunk
                # counters without a successful publisher call.
                command_delivery_mode = "simulated_no_publish"

            executed_actor_residual = None
            if persistent_actor_execution:
                if persistent_commit_candidate:
                    if command_delivery_succeeded:
                        try:
                            if filtered_actual_execution:
                                if actor_filtered_actual_certificate is None:
                                    raise ValueError(
                                        "missing filtered-actual certificate"
                                    )
                                actor_governor.mark_filtered_executed(
                                    actor_filtered_actual_certificate
                                )
                            else:
                                actor_governor.mark_executed(
                                    actor_plan_id,
                                    actor_plan_offset,
                                )
                            persistent_commit_status = (
                                "committed_filtered_actual_after_publish"
                                if filtered_actual_execution
                                else "committed_after_publish"
                            )
                            executed_actor_residual = (
                                actor_filtered_actual_certificate.actual_residual
                                if filtered_actual_execution
                                else planned_actor_residual
                            )
                        except (KeyError, IndexError, ValueError) as exc:
                            persistent_commit_status = "reset_after_commit_error"
                            persistent_commit_error = (
                                f"{type(exc).__name__}: {exc}"
                            )
                            actor_governor.reset_execution_state("commit_error")
                            actor_live_active_plan_id = None
                            actor_live_active_steps = 0
                            actor_live_next_offset = 0
                            self._actor_base_safety_filter = None
                    else:
                        persistent_commit_status = (
                            "not_committed_missing_publisher"
                            if command_delivery_mode == "missing_publisher"
                            else "not_committed_simulation_no_publish"
                        )
                elif selection.source == "rlt":
                    # A safety filter changed the planned absolute target. Its
                    # actual residual is no longer the certified persistent
                    # trajectory, so fail closed before the next frame instead
                    # of carrying a fictional offset.
                    persistent_commit_status = "reset_after_safety_modification"
                    actor_governor.reset_execution_state(
                        "safety_modified_actor_command"
                    )
                    actor_live_active_plan_id = None
                    actor_live_active_steps = 0
                    actor_live_next_offset = 0
                    self._actor_base_safety_filter = None
                elif selection.source == "human_pika":
                    persistent_commit_status = "reset_for_human"
                    actor_governor.reset_execution_state("human_takeover")
                    actor_live_active_plan_id = None
                    actor_live_active_steps = 0
                    actor_live_next_offset = 0
                elif phase_snapshot.active and h50_boundary_keepalive:
                    if (
                        selected_command_survived_filter
                        and command_delivery_succeeded
                    ):
                        # The keepalive republishes the last physical target,
                        # which already contains the committed Actor carry. No
                        # new Actor row is consumed, but the carry remains real.
                        persistent_commit_status = (
                            "preserved_through_h50_keepalive"
                        )
                        executed_actor_residual = (
                            actor_governor.current_residual
                        )
                    else:
                        # A brake/feedback substitute is not the carried target.
                        # Retaining the residual here would corrupt the next H50
                        # handoff and claim phase coverage that did not happen.
                        persistent_commit_status = (
                            "reset_after_filtered_h50_keepalive"
                            if not selected_command_survived_filter
                            else "reset_after_undelivered_h50_keepalive"
                        )
                        actor_governor.reset_execution_state(
                            persistent_commit_status
                        )
                        actor_live_active_plan_id = None
                        actor_live_active_steps = 0
                        actor_live_next_offset = 0
                        self._actor_base_safety_filter = None
                elif (
                    phase_snapshot.active
                ):
                    persistent_commit_status = "reset_after_actor_fallback"
                    actor_governor.reset_execution_state(
                        f"phase_actor_fallback_to_{selection.source}"
                    )
                    actor_live_active_plan_id = None
                    actor_live_active_steps = 0
                    actor_live_next_offset = 0
                    self._actor_base_safety_filter = None

            actor_execution_committed = bool(
                config.actor_live
                and actor_action_survived_filter
                and command_delivery_succeeded
                and (
                    not persistent_actor_execution
                    or persistent_commit_status
                    in {
                        "committed_after_publish",
                        "committed_filtered_actual_after_publish",
                    }
                )
            )
            if actor_execution_committed:
                executed_actor_residual = (
                    actor_filtered_actual_certificate.actual_residual
                    if filtered_actual_execution
                    and actor_filtered_actual_certificate is not None
                    else planned_actor_residual
                )
                if actor_live_active_plan_id is None:
                    actor_live_chunks_started += 1
                    actor_live_active_plan_id = actor_plan_id
                    actor_live_active_steps = 0
                    actor_live_next_offset = (
                        0 if actor_plan_offset is None else actor_plan_offset
                    )
                actor_live_active_steps += 1
                actor_live_next_offset = (
                    actor_live_next_offset + 1
                    if actor_plan_offset is None
                    else actor_plan_offset + 1
                )
                if actor_live_active_steps >= config.chunk_length:
                    actor_live_chunks_completed += 1
                    actor_live_active_plan_id = None
                    actor_live_active_steps = 0
                    actor_live_next_offset = 0
            elif actor_live_active_plan_id is not None:
                # Human takeover, terminal hold, a filtered Actor target, or a
                # failed/missing publisher interrupts the in-flight chunk. Its
                # already consumed canary budget is deliberately not returned.
                actor_live_chunk_interrupted = True
                actor_live_chunk_interruption_reason = (
                    f"actor_chunk_interrupted_by_{selection.source}"
                    f"_{command_delivery_mode}"
                )
                actor_live_active_plan_id = None
                actor_live_active_steps = 0
                actor_live_next_offset = 0

            filter_tau_s = float(config.model_smoothing_tau_s)
            model_lowpass_alpha = (
                1.0
                if filter_tau_s == 0.0
                else 1.0 - math.exp(-dt / filter_tau_s)
            )
            actor_canonical_decision = None
            if selection.source == "rlt" and governed_plan is not None:
                actor_canonical_decision_array = np.asarray(
                    governed_plan.rank1_direction,
                    dtype=np.float32,
                ).reshape(-1)
                if actor_canonical_decision_array.shape == (6,):
                    # Frozen-v2 governors historically expose only the six
                    # joint knots.  Preserve their canonical 7-D replay shape
                    # with a zero gripper while close-assist already supplies
                    # its real seventh knot.
                    actor_canonical_decision_array = np.pad(
                        actor_canonical_decision_array,
                        (0, 1),
                    )
                if actor_canonical_decision_array.shape != (config.action_dim,):
                    raise RuntimeError(
                        "governed Actor canonical decision must have shape "
                        f"({config.action_dim},), got "
                        f"{actor_canonical_decision_array.shape}"
                    )
                actor_canonical_decision = (
                    actor_canonical_decision_array.tolist()
                )
            canonical_execution_metadata: dict[str, Any] = {
                "actor_requested_source": requested_selection_source,
                "actor_delivered_source": selection.source,
                "actor_delivery_fallback_reason": (
                    "filtered_actual_certificate_rejected"
                    if actor_filtered_actual_rejected
                    else None
                ),
                "actor_execution_profile": config.actor_execution_profile,
                "actor_execution_schema_fingerprint": (
                    config.execution_action_schema_fingerprint
                ),
                "actor_model_action_schema_fingerprint": (
                    config.action_schema_fingerprint
                ),
                "actor_execution_filter_tau_s": filter_tau_s,
                "actor_execution_filter_dt_s": dt,
                "actor_execution_filter_alpha": model_lowpass_alpha,
                "actor_execution_residual_max_rad": (
                    config.actor_residual_max_rad
                ),
                "actor_execution_d1_max_rad": (
                    config.actor_residual_d1_max_rad
                ),
                "actor_execution_d2_max_rad": (
                    config.actor_residual_d2_max_rad
                ),
                "actor_execution_direction_cone_deg": (
                    config.actor_direction_cone_deg
                ),
                "actor_execution_boundary_limit_rad": (
                    config.actor_live_max_boundary_jump_rad
                ),
                "actor_execution_projection_scale_steps": (
                    actor_governor_config.projection_scale_steps
                ),
                "actor_execution_min_projection_scale": (
                    actor_governor_config.min_projection_scale
                ),
                "actor_execution_direction_static_threshold_rad": (
                    actor_governor_config.direction_static_threshold_rad
                ),
                "actor_governor_fingerprint": (
                    config.actor_governor_fingerprint
                    if filtered_actual_execution
                    else None
                ),
                "actor_gripper_residual_mode": (
                    config.actor_gripper_residual_mode
                ),
                "execution_gripper_residual_max_close_m": (
                    config.actor_gripper_residual_max_close_m
                ),
                "execution_gripper_d1_max_m": (
                    config.actor_gripper_residual_d1_max_m
                ),
                "execution_gripper_d2_max_m": (
                    config.actor_gripper_residual_d2_max_m
                ),
                "execution_gripper_boundary_limit_m": (
                    config.actor_gripper_max_boundary_jump_m
                ),
                "execution_gripper_command_min_m": (
                    config.actor_gripper_command_min_m
                ),
                "execution_gripper_command_max_m": (
                    config.actor_gripper_command_max_m
                ),
                "execution_gripper_release_reference_m": (
                    config.actor_gripper_release_reference_m
                ),
                "execution_gripper_release_delta_m": (
                    config.actor_gripper_release_delta_m
                ),
                # Compatibility aliases for rollout/audit tooling.
                "actor_projection_scale_steps": (
                    actor_governor_config.projection_scale_steps
                ),
                "actor_min_projection_scale": (
                    actor_governor_config.min_projection_scale
                ),
                "actor_direction_static_threshold_rad": (
                    actor_governor_config.direction_static_threshold_rad
                ),
                "execution_filter_profile": (
                    "exp_one_minus_exp_neg_dt_over_tau_v1"
                ),
                "action_schema_fingerprint": (
                    config.execution_action_schema_fingerprint
                ),
                # Compatibility aliases used by the lineage/updater guard.
                "model_smoothing_tau_s": filter_tau_s,
                "control_hz": config.control_hz,
                "model_lowpass_alpha": model_lowpass_alpha,
                "actor_execution_plan_id": (
                    actor_plan_id if selection.source == "rlt" else None
                ),
                "actor_execution_plan_offset": (
                    actor_plan_offset if selection.source == "rlt" else None
                ),
                "actor_canonical_decision": actor_canonical_decision,
                "actor_persistent_carry_in": (
                    None
                    if selection.source != "rlt"
                    or governed_plan is None
                    or not hasattr(governed_plan, "carry_in")
                    else np.asarray(
                        governed_plan.carry_in,
                        dtype=np.float32,
                    ).tolist()
                ),
                "actor_persistent_previous_carry": (
                    None
                    if selection.source != "rlt"
                    or governed_plan is None
                    or not hasattr(governed_plan, "previous_carry")
                    else np.asarray(
                        governed_plan.previous_carry,
                        dtype=np.float32,
                    ).tolist()
                ),
                "actor_persistent_carry_out": (
                    actor_governor.current_residual.tolist()
                    if persistent_actor_execution
                    and selection.source == "rlt"
                    and actor_execution_committed
                    else None
                ),
                "actor_execution_boundary_anchor": (
                    None
                    if selection.source != "rlt"
                    or governed_plan is None
                    or not hasattr(governed_plan, "boundary_anchor")
                    else np.asarray(
                        governed_plan.boundary_anchor,
                        dtype=np.float32,
                    ).tolist()
                ),
                "actor_execution_projection_scale": (
                    None
                    if selection.source != "rlt" or governed_plan is None
                    else float(governed_plan.projection_scale)
                ),
                "actor_filtered_base_command": (
                    None
                    if actor_filtered_base_command is None
                    else np.asarray(
                        actor_filtered_base_command,
                        dtype=np.float32,
                    ).tolist()
                ),
                "actor_filtered_base_action": (
                    None
                    if actor_filtered_base_command is None
                    else np.asarray(
                        actor_filtered_base_command,
                        dtype=np.float32,
                    ).tolist()
                ),
                "actor_filtered_actual_action": (
                    last_exec_command.tolist()
                    if selection.source == "rlt"
                    else None
                ),
                "actor_filtered_actual_residual": (
                    None
                    if actor_filtered_actual_certificate is None
                    else actor_filtered_actual_certificate.actual_residual.tolist()
                ),
                "actor_actual_residual": (
                    None
                    if actor_filtered_actual_certificate is None
                    else actor_filtered_actual_certificate.actual_residual.tolist()
                ),
                "actor_filtered_actual_rejected": (
                    actor_filtered_actual_rejected
                ),
            }
            if actor_filtered_actual_certificate is not None:
                canonical_execution_metadata.update(
                    actor_filtered_actual_certificate.metadata()
                )
                canonical_execution_metadata.update(
                    {
                        "actor_filtered_base_gripper_command_m": float(
                            actor_filtered_actual_certificate.filtered_base_action[
                                6
                            ]
                        ),
                        "actor_filtered_actual_gripper_command_m": float(
                            actor_filtered_actual_certificate.filtered_actual_action[
                                6
                            ]
                        ),
                    }
                )

            # Preserve human intervention as an explicit, fail-closed C10
            # execution contract instead of silently dropping it because it
            # has no Actor plan.  The actual side remains native Pika; only the
            # counterfactual base side uses the model safety filter.
            human_canonical_committed = bool(
                filtered_actual_execution
                and selection.source == "human_pika"
                and human_filtered_base_valid
                and command_delivery_succeeded
            )
            if filtered_actual_execution and selection.source == "human_pika":
                if human_canonical_committed:
                    if human_canonical_plan_id is None:
                        human_canonical_plan_id = (
                            f"{config.episode_id}/human_filtered_actual/"
                            f"plan_{human_canonical_plan_serial:08d}"
                        )
                        human_canonical_plan_offset = 0
                        human_canonical_boundary_anchor = np.asarray(
                            human_filtered_base_boundary_anchor,
                            dtype=np.float32,
                        ).copy()
                    human_residual = (
                        last_exec_command
                        - np.asarray(
                            human_filtered_base_command,
                            dtype=np.float32,
                        )
                    )
                    canonical_execution_metadata.update(
                        {
                            "actor_execution_profile": (
                                "human_pika_filtered_actual_v2"
                            ),
                            "actor_execution_plan_id": human_canonical_plan_id,
                            "actor_execution_plan_offset": (
                                human_canonical_plan_offset
                            ),
                            "actor_canonical_decision": [0.0] * config.action_dim,
                            "actor_persistent_carry_in": [0.0] * config.action_dim,
                            "actor_persistent_previous_carry": (
                                [0.0] * config.action_dim
                            ),
                            "actor_persistent_carry_out": [0.0] * config.action_dim,
                            "actor_persistent_planned_residual": (
                                [0.0] * config.action_dim
                            ),
                            "actor_gripper_release_intent": False,
                            "actor_execution_boundary_anchor": (
                                human_canonical_boundary_anchor.tolist()
                            ),
                            "actor_execution_projection_scale": 0.0,
                            "actor_filtered_base_command": (
                                np.asarray(
                                    human_filtered_base_command,
                                    dtype=np.float32,
                                ).tolist()
                            ),
                            "actor_filtered_base_action": (
                                np.asarray(
                                    human_filtered_base_command,
                                    dtype=np.float32,
                                ).tolist()
                            ),
                            "actor_filtered_actual_action": (
                                last_exec_command.tolist()
                            ),
                            "actor_filtered_actual_residual": (
                                human_residual.tolist()
                            ),
                            "actor_actual_residual": human_residual.tolist(),
                            "human_base_counterfactual_valid": True,
                            "actor_execution_actual_filter_profile": (
                                "human_native"
                            ),
                        }
                    )
                    human_canonical_plan_offset += 1
                    if human_canonical_plan_offset >= config.chunk_length:
                        human_canonical_plan_serial += 1
                        human_canonical_plan_id = None
                        human_canonical_plan_offset = 0
                        human_canonical_boundary_anchor = None
                else:
                    if human_canonical_plan_id is not None:
                        human_canonical_plan_serial += 1
                    human_canonical_plan_id = None
                    human_canonical_plan_offset = 0
                    human_canonical_boundary_anchor = None
                    canonical_execution_metadata.update(
                        {
                            "actor_execution_profile": (
                                "human_pika_filtered_actual_v2"
                            ),
                            "human_base_counterfactual_valid": False,
                            "human_base_counterfactual_error": (
                                human_filtered_base_error
                                or "missing_fresh_model_counterfactual"
                                if not human_filtered_base_valid
                                else "physical_command_not_delivered"
                            ),
                        }
                    )

            if persistent_actor_execution:
                policy_metadata.update(
                    {
                        "actor_persistent_commit_status": (
                            persistent_commit_status
                        ),
                        "actor_persistent_commit_error": persistent_commit_error,
                        "actor_persistent_committed_residual": (
                            actor_governor.current_residual.tolist()
                        ),
                        "actor_persistent_phase_exit_reset": (
                            persistent_phase_exit_reset
                        ),
                        "actor_full_phase_coverage_this_step": bool(
                            not phase_snapshot.active
                            or actor_execution_committed
                            or (
                                h50_boundary_keepalive
                                and persistent_commit_status
                                == "preserved_through_h50_keepalive"
                            )
                        ),
                    }
                )
            policy_metadata.update(canonical_execution_metadata)
            policy_metadata.update(
                {
                    "actor_command_delivery_mode": command_delivery_mode,
                    "actor_command_delivery_succeeded": (
                        command_delivery_succeeded
                    ),
                    "actor_execution_committed_this_step": (
                        actor_execution_committed or human_canonical_committed
                    ),
                    "actor_physical_execution_committed_this_step": bool(
                        (actor_execution_committed or human_canonical_committed)
                        and command_delivery_mode == "published"
                    ),
                    "actor_simulated_execution_this_step": bool(
                        selection.source == "rlt"
                        and actor_action_survived_filter
                        and command_delivery_mode == "simulated_no_publish"
                    ),
                    "actor_live_chunks_started": actor_live_chunks_started,
                    "actor_live_chunks_completed": actor_live_chunks_completed,
                    "actor_live_chunk_steps": actor_live_active_steps,
                    "actor_live_chunk_interrupted": actor_live_chunk_interrupted,
                    "actor_live_chunk_interruption_reason": (
                        actor_live_chunk_interruption_reason
                    ),
                }
            )
            self._record_handoff_execution(
                last_exec_command,
                source=selection.source,
                is_boundary_keepalive=h50_boundary_keepalive,
                actor_residual=executed_actor_residual,
            )
            policy_metadata.update(self._last_feedback_hold_metadata)
            policy_metadata.update(self._command_publisher_telemetry())

            human_alignment_ready = bool(
                keyboard_snapshot.mode != "HUMAN"
                or (
                    policy_observation_t is not None
                    and takeover_started_t is not None
                    and int(policy_observation_t) >= int(takeover_started_t)
                )
            )
            replay_alignment_ready = bool(
                policy_metadata.get("policy_enrichment_fresh", True)
                and gate_alignment_ready
                and human_alignment_ready
            )
            policy_metadata["policy_replay_alignment_ready"] = replay_alignment_ready
            token_ready_for_replay = bool(
                not config.actor_shadow
                or (
                    policy_metadata.get("actor_shadow_z_is_true") is True
                    and policy_metadata.get("actor_shadow_z_shape_ok") is True
                )
            ) and replay_alignment_ready
            replay_include = token_ready_for_replay and (
                (keyboard_snapshot.mode == "MODEL" and selection.source in {"pi05", "rlt"})
                or (keyboard_snapshot.mode == "HUMAN" and selection.source == "human_pika")
            ) and not keyboard_snapshot.episode_done
            global_image, wrist_image = self.image_writer.save(t=t, images=images)
            self.logger.append_step(
                t=t,
                global_image=global_image,
                wrist_image=wrist_image,
                z_rl=z_rl,
                state=state,
                a_ref=a_ref,
                a_exec=exec_command,
                # A continuously published or stale Pika stream is diagnostic
                # input only. It becomes a recorded human action exclusively
                # after the post-s physical activation edge enters HUMAN.
                a_human=(
                    None
                    if keyboard_snapshot.mode != "HUMAN" or human_command is None
                    else human_command.value
                ),
                a_actor=a_actor,
                a_actor_safe=(
                    None if governed_plan is None else governed_plan.safe_actions
                ),
                source=selection.source,
                phase_probability=0.0 if phase_snapshot.probability is None else phase_snapshot.probability,
                gate_active=phase_snapshot.active,
                gate_state=phase_snapshot.state,
                gate_enter_t=phase_snapshot.enter_t,
                gate_exit_t=phase_snapshot.exit_t,
                gate_reason=phase_snapshot.reason,
                keyboard=keyboard_snapshot,
                timestamp_ns=int(time.time_ns()),
                takeover_active=keyboard_snapshot.mode == "HUMAN",
                takeover_started_t=takeover_started_t,
                takeover_ended_t=takeover_ended_t,
                policy_metadata={
                    **policy_metadata,
                    "mux_reason": selection.reason,
                    "safety_profile": safety_profile,
                    "safety_reasons": safety_reasons,
                    "replay_include": replay_include,
                    "takeover_recording_active": keyboard_snapshot.mode == "HUMAN",
                    "pika_teleop_observed_active": teleop_active,
                    "pika_takeover_seen_inactive_after_s": takeover_seen_inactive,
                    "pika_activation_edge_accepted": takeover_activation_edge_accepted,
                    "human_end_timeout_s": config.human_end_timeout_s,
                    "phase_gate_state": phase_snapshot.state,
                    "phase_gate_reason": phase_snapshot.reason,
                    "phase_gate_high_count": phase_snapshot.high_count,
                    "phase_gate_enter_t": phase_snapshot.enter_t,
                    "phase_gate_exit_t": phase_snapshot.exit_t,
                    "phase_classifier_checkpoint": None
                    if config.phase_classifier_checkpoint is None
                    else str(config.phase_classifier_checkpoint),
                    "phase_classifier_period": config.phase_classifier_period,
                    "actor_shadow_enabled": config.actor_shadow,
                    "actor_live_enabled": config.actor_live,
                    "actor_shadow_control_source": (
                        "rlt_when_gate_active_else_pi05_or_human" if config.actor_live else "pi05_or_human_only"
                    ),
                    "actor_shadow_gate_active": bool(phase_snapshot.active),
                    "actor_shadow_would_activate": bool(phase_snapshot.active and a_actor is not None),
                    "model_motion_budget_exhausted": motion_budget_exhausted,
                },
            )
            last_source = selection.source
            if keyboard_snapshot.episode_done or selection.should_stop:
                if self.human_passthrough is not None:
                    self.human_passthrough.set_active(False)
                outcome = "episode_done" if keyboard_snapshot.episode_done else "stopped"
                emergency_stop = bool(keyboard_snapshot.stop_requested)
                return {
                    "outcome": outcome,
                    "steps": t + 1,
                    "published_commands": published_commands,
                    "last_source": last_source,
                    "terminal_reward": keyboard_snapshot.terminal_reward,
                    "termination_reason": (
                        "operator_emergency_stop"
                        if emergency_stop
                        else "operator_reward"
                        if keyboard_snapshot.terminal_reward is not None
                        else "command_mux_stop"
                    ),
                    "exclude_from_training": emergency_stop,
                    "exclusion_category": "operator_emergency_stop" if emergency_stop else None,
                    "exclusion_reason": (
                        "q is a safety abort, not a task-level reward=0 failure" if emergency_stop else None
                    ),
                    "actor_live_chunks_started": actor_live_chunks_started,
                    "actor_live_chunks_completed": actor_live_chunks_completed,
                    "human_passthrough_commands": (
                        0 if self.human_passthrough is None else self.human_passthrough.forwarded_count
                    ),
                }
            self.sleep_fn(max(0.0, dt - (self.now_fn() - step_started)))
            t += 1

        if self.human_passthrough is not None:
            self.human_passthrough.set_active(False)
        return {
            "outcome": outcome,
            "steps": max_steps,
            "published_commands": published_commands,
            "last_source": last_source,
            "terminal_reward": self.keyboard.snapshot(now_s=self.now_fn()).terminal_reward,
            "actor_live_chunks_started": actor_live_chunks_started,
            "actor_live_chunks_completed": actor_live_chunks_completed,
            "human_passthrough_commands": (
                0 if self.human_passthrough is None else self.human_passthrough.forwarded_count
            ),
        }

    def _teleop_observed_active(self, human_command: TimedCommand | None) -> bool:
        """Return the physical Pika teleop state, with a command-edge fallback."""
        if self.teleop_controller is not None:
            observed_active = getattr(self.teleop_controller, "observed_active", None)
            if observed_active is None:
                observed_active = getattr(self.teleop_controller, "_observed_active", None)
            if callable(observed_active):
                try:
                    return bool(observed_active())
                except Exception:
                    # Falling back to command freshness is fail-closed with the
                    # quiet->fresh gate; it never accepts a stream already
                    # present when `s` is pressed.
                    pass
        return bool(human_command is not None and human_command.value is not None)

    def _request_teleop_inactive(self) -> None:
        if self.teleop_controller is None:
            return
        set_active = getattr(self.teleop_controller, "set_active", None)
        if not callable(set_active):
            return
        try:
            set_active(False)
        except Exception as exc:
            print(
                f"[{self.config.episode_id}] could not close a pre-existing Pika teleop activation: {exc}; "
                "remaining armed until an explicit off->on edge is observed",
                flush=True,
            )

    def _update_phase_gate(
        self,
        *,
        t: int,
        images: dict[str, np.ndarray],
        keyboard: Any,
    ) -> PhaseGateSnapshot:
        if self.phase_gate is None:
            return PhaseGateSnapshot(
                probability=None,
                active=False,
                state="DISABLED",
                enter_t=None,
                exit_t=None,
                reason="no_phase_classifier",
                high_count=0,
            )
        terminal = bool(keyboard.episode_done or keyboard.stop_requested)
        terminal_reason = "episode_done" if keyboard.episode_done else "stop_requested" if keyboard.stop_requested else "terminal"
        probability: float | None = None
        if (
            not terminal
            and self.phase_classifier is not None
            and should_run_classifier(t=t, period=self.config.phase_classifier_period)
        ):
            probability = float(self.phase_classifier.predict_probability(images))
        return self.phase_gate.update(probability, t=t, terminal=terminal, terminal_reason=terminal_reason)

    def _latest_policy_command(
        self,
        *,
        now_s: float,
        action_dim: int,
        chunk_length: int,
        state_snapshot: np.ndarray,
        actor_phase_active: bool = False,
        enrichment_active: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any], TimedCommand | None]:
        split_workers = self.enrichment_policy_worker is not None
        base_latest = self.policy_worker.latest()
        enrichment_latest = (
            None
            if not split_workers
            else self.enrichment_policy_worker.latest()
        )
        execute_steps = self.config.model_execute_steps
        required_actions = max(chunk_length, execute_steps)
        self._consume_latest_policy_output(
            latest=base_latest,
            now_s=now_s,
            state_snapshot=state_snapshot,
            action_dim=action_dim,
            chunk_length=chunk_length,
            required_actions=required_actions,
            actor_phase_active=actor_phase_active,
            request_lane="base" if split_workers else "shared",
        )
        if split_workers:
            self._consume_latest_policy_output(
                latest=enrichment_latest,
                now_s=now_s,
                state_snapshot=state_snapshot,
                action_dim=action_dim,
                chunk_length=chunk_length,
                required_actions=required_actions,
                actor_phase_active=actor_phase_active,
                request_lane="enrichment",
            )
        if self._effective_plan_remaining() == 0:
            self._activate_prefetched_model_plan(
                now_s=now_s,
                state_snapshot=state_snapshot,
                chunk_length=chunk_length,
            )
        if enrichment_active and self._enrichment_plan_remaining() == 0:
            self._activate_prefetched_enrichment_plan()

        behavior_index = int(self._model_plan_index)
        behavior_available = bool(
            self._model_plan is not None
            and self._model_plan_limit > 0
            and behavior_index < self._model_plan_limit
        )
        if not behavior_available and self._model_plan is not None:
            if self.enrichment_policy_worker is None:
                if self._policy_wait_started_s is None:
                    self._policy_wait_started_s = now_s
            elif self._base_wait_started_s is None:
                self._base_wait_started_s = now_s

        use_enrichment = bool(enrichment_active and self._enrichment_plan is not None)
        if use_enrichment:
            payload_plan = self._enrichment_plan
            payload_z_rl = self._enrichment_plan_z_rl
            payload_actor = self._enrichment_plan_actor
            payload_metadata = self._enrichment_plan_metadata
            plan_offset = int(self._enrichment_plan_index)
            payload_limit = int(self._enrichment_plan_limit)
        else:
            payload_plan = self._model_plan
            payload_z_rl = self._model_plan_z_rl
            payload_actor = self._model_plan_actor
            payload_metadata = self._model_plan_metadata
            plan_offset = behavior_index
            payload_limit = int(self._model_plan_limit)

        if payload_plan is None:
            a_ref = np.zeros((chunk_length, action_dim), dtype=np.float32)
            z_rl = np.zeros(self.config.actor_shadow_expected_z_dim if self.config.actor_shadow else 1, dtype=np.float32)
            metadata = {
                "policy_status": "missing_plan",
                "policy_error": None if base_latest is None else base_latest.error,
                "policy_plan_id": None,
                "policy_observation_t": None,
                "plan_offset": None,
                "behavior_actor_checkpoint": None,
                "model_action_index": behavior_index,
                "model_execute_steps": execute_steps,
                "actor_execute_steps": chunk_length,
                "effective_execute_steps": execute_steps,
                "actor_shadow_ready": False,
                "policy_enrichment_fresh": False,
            }
            return a_ref, z_rl, None, metadata, None

        enrichment_a_ref = _slice_action_chunk(
            payload_plan, plan_offset, chunk_length, action_dim
        )
        behavior_a_ref = enrichment_a_ref
        if behavior_available:
            assert self._model_plan is not None
            behavior_a_ref = _slice_action_chunk(
                self._model_plan,
                behavior_index,
                chunk_length,
                action_dim,
            )
        a_ref = behavior_a_ref
        z_rl = np.zeros(1, dtype=np.float32) if payload_z_rl is None else payload_z_rl
        actor_index_valid = bool(
            payload_actor is not None
            and plan_offset < payload_limit
            and plan_offset < int(payload_actor.shape[0])
        )
        a_actor = None if not actor_index_valid else _slice_action_chunk(
            payload_actor, plan_offset, chunk_length, action_dim
        )
        current_behavior_plan_id = self._model_plan_metadata.get("policy_plan_id")
        target_behavior_plan_id = payload_metadata.get("actor_behavior_ref_plan_id")
        target_behavior_start_offset = payload_metadata.get(
            "actor_behavior_ref_start_offset"
        )
        expected_behavior_offset = None
        if target_behavior_start_offset is not None:
            expected_behavior_offset = int(target_behavior_start_offset) + int(plan_offset)
        actor_behavior_ref_alignment_ready = bool(
            behavior_available
            and payload_metadata.get("actor_behavior_ref_echo_verified", False)
            and target_behavior_plan_id is not None
            and str(target_behavior_plan_id) == str(current_behavior_plan_id)
            and expected_behavior_offset == behavior_index
            and np.allclose(
                enrichment_a_ref[0, :action_dim],
                behavior_a_ref[0, :action_dim],
                rtol=0.0,
                atol=1e-7,
            )
        )
        actor_reference_mode = (
            "behavior_ref_protocol_exact"
            if actor_behavior_ref_alignment_ready
            else "behavior_ref_protocol_mismatch"
        )
        actor_ready_this_step = bool(
            payload_metadata.get("actor_shadow_ready", False)
            and a_actor is not None
            and actor_behavior_ref_alignment_ready
        )
        actor_controls_this_step = bool(self.config.actor_live and actor_phase_active and actor_ready_this_step)
        enrichment_fresh = bool(not enrichment_active or (use_enrichment and plan_offset < payload_limit))
        metadata = {
            **payload_metadata,
            "plan_offset": plan_offset,
            "model_action_index": behavior_index,
            "behavior_plan_offset": behavior_index,
            "model_execute_steps": execute_steps,
            "actor_execute_steps": chunk_length,
            "effective_execute_steps": chunk_length if actor_controls_this_step else execute_steps,
            "actor_shadow_ready": actor_ready_this_step,
            "policy_enrichment_active": bool(enrichment_active),
            "policy_enrichment_fresh": enrichment_fresh,
            "actor_reference_mode": actor_reference_mode,
            "actor_reference_rebased": False,
            "actor_reference_rebase_jump_per_joint_rad": None,
            "actor_reference_rebase_jump_max_rad": None,
            "actor_behavior_ref_current_plan_id": current_behavior_plan_id,
            "actor_behavior_ref_current_offset": behavior_index,
            "actor_behavior_ref_expected_offset": expected_behavior_offset,
            "actor_behavior_ref_alignment_ready": actor_behavior_ref_alignment_ready,
        }
        model_command = None
        if behavior_available:
            assert self._model_plan is not None
            command = self._model_plan[behavior_index].astype(np.float32, copy=True)
            model_command = TimedCommand(value=command, timestamp_s=now_s)
            self._model_plan_index += 1
        if self._enrichment_plan is not None:
            self._enrichment_plan_index += 1
        return a_ref, z_rl, a_actor, metadata, model_command

    def _consume_latest_policy_output(
        self,
        *,
        latest: Any,
        now_s: float,
        state_snapshot: np.ndarray,
        action_dim: int,
        chunk_length: int,
        required_actions: int,
        actor_phase_active: bool,
        request_lane: str = "shared",
    ) -> None:
        if latest is None:
            return
        latest_key = (float(latest.observation_timestamp_s), float(latest.completed_timestamp_s))
        last_output_key = (
            self._base_last_policy_output_key
            if request_lane == "base"
            else self._last_policy_output_key
        )
        if latest_key == last_output_key:
            return
        if request_lane == "base":
            self._base_last_policy_output_key = latest_key
        else:
            self._last_policy_output_key = latest_key
        (
            request_reason,
            request_lead_steps,
            requested_target,
            request_wait_started_s,
        ) = self._policy_request_context(request_lane)
        if latest.value is None or latest.error is not None:
            self._clear_policy_request(request_lane)
            return

        request_wait_s = 0.0 if request_wait_started_s is None else max(
            0.0, now_s - request_wait_started_s
        )
        response_action_rows = (
            chunk_length if request_lane == "enrichment" else required_actions
        )
        output = extract_rlt_policy_output(
            latest.value,
            chunk_length=response_action_rows,
            actor_chunk_length=chunk_length,
            action_dim=action_dim,
            fallback_z_rl_dim=self.config.actor_shadow_expected_z_dim if self.config.actor_shadow else 1,
            state_snapshot=state_snapshot,
            read_actor=self.config.actor_shadow,
        )
        policy_plan = output.a_ref[:, :action_dim].astype(np.float32, copy=True)
        enrichment_z_rl = output.z_rl.astype(np.float32, copy=True)
        enrichment_actor = None if output.a_actor is None else output.a_actor.astype(np.float32, copy=True)
        policy_latency_s = max(
            0.0,
            float(latest.completed_timestamp_s) - float(latest.observation_timestamp_s),
        )
        z_is_true = output.metadata.get("z_rl_source") in {"z_rl", "rl_token", "rl_latent"}
        z_shape_ok = enrichment_z_rl.shape == (self.config.actor_shadow_expected_z_dim,)
        shadow_late = bool(
            self.config.actor_shadow and policy_latency_s > self.config.actor_shadow_max_latency_s
        )
        worker_plan_id = getattr(latest, "policy_plan_id", None)
        if worker_plan_id is None:
            worker_plan_id = (
                f"timestamp_{float(latest.observation_timestamp_s):.9f}_"
                f"{float(latest.completed_timestamp_s):.9f}"
            )
        policy_plan_id = (
            f"{self.config.episode_id}/{worker_plan_id}"
            if request_lane == "shared"
            else f"{self.config.episode_id}/{request_lane}/{worker_plan_id}"
        )
        policy_observation_t = getattr(latest, "policy_observation_t", None)
        behavior_remaining = self._effective_plan_remaining()
        install_behavior = bool(
            request_lane != "enrichment"
            and (
                self._model_plan is None
                or behavior_remaining is None
                or behavior_remaining <= 0
                or request_reason in {"initial", "boundary_miss"}
            )
        )
        if requested_target is not None:
            actor_reference_plan = requested_target.actions.astype(np.float32, copy=True)
            actor_behavior_plan_id = requested_target.plan_id
            actor_behavior_start_offset = requested_target.start_offset
            actor_behavior_ref_echo_ok = bool(
                output.metadata.get("actor_behavior_ref_error") is None
                and output.metadata.get("actor_behavior_ref_source") == "request_behavior_ref"
                and output.metadata.get("actor_behavior_ref_contract") == RANK1_BUMP_CONTRACT
                and output.metadata.get("actor_action_schema_fingerprint")
                == self.config.action_schema_fingerprint
                and output.metadata.get("actor_projection_profile")
                == self.config.actor_projection_profile
                and output.metadata.get("actor_behavior_ref_plan_id") == requested_target.plan_id
                and output.metadata.get("actor_behavior_ref_start_offset")
                == requested_target.start_offset
            )
            actor_behavior_ref_source = "request_behavior_ref"
        else:
            actor_reference_plan = policy_plan[:chunk_length].astype(np.float32, copy=True)
            actor_behavior_plan_id = str(policy_plan_id)
            actor_behavior_start_offset = 0
            # Initial/boundary inference creates the H50 plan itself, so its
            # first C10 is the exact behavior target.  A non-installing fresh
            # response is never allowed to control an older behavior plan.
            actor_behavior_ref_echo_ok = bool(
                install_behavior
                and output.metadata.get("actor_behavior_ref_error") is None
                and output.metadata.get("actor_behavior_ref_source") == "response_actions"
                and output.metadata.get("actor_behavior_ref_contract") == RANK1_BUMP_CONTRACT
                and output.metadata.get("actor_action_schema_fingerprint")
                == self.config.action_schema_fingerprint
                and output.metadata.get("actor_projection_profile")
                == self.config.actor_projection_profile
            )
            actor_behavior_ref_source = "response_actions"
        metadata = {
            **output.metadata,
            "policy_status": "ok",
            "policy_action_space": "joint_absolute_gripper_absolute",
            "executable_action_space": "joint_absolute_gripper_absolute",
            "policy_plan_id": str(policy_plan_id),
            "policy_worker_plan_id": str(worker_plan_id),
            "policy_observation_t": None if policy_observation_t is None else int(policy_observation_t),
            "policy_observation_timestamp_s": float(latest.observation_timestamp_s),
            "policy_completed_timestamp_s": float(latest.completed_timestamp_s),
            "policy_inference_latency_s": policy_latency_s,
            "policy_planning_mode": str(
                getattr(
                    self.policy_worker
                    if request_lane in {"base", "shared"}
                    else self.enrichment_policy_worker,
                    "planning_mode",
                    "test_synchronous",
                )
            ),
            "policy_worker_lane": request_lane,
            "policy_worker_reported_lane": str(
                getattr(latest, "worker_lane", request_lane)
            ),
            "policy_workers_split": bool(self.enrichment_policy_worker is not None),
            "policy_request_reason": request_reason,
            "policy_prefetch_lead_steps": request_lead_steps,
            "policy_prefetch_hit": bool(
                request_reason in {
                    "actor_prefetch",
                    "replay_enrichment_prefetch",
                    "behavior_prefetch",
                }
                and request_wait_s == 0.0
            ),
            "policy_boundary_wait_s": request_wait_s,
            "actor_shadow_latency_ok": not shadow_late,
            "actor_shadow_z_is_true": bool(z_is_true),
            "actor_shadow_z_shape_ok": bool(z_shape_ok),
            "actor_shadow_ready": bool(
                self.config.actor_shadow
                and enrichment_actor is not None
                and z_is_true
                and z_shape_ok
                and not shadow_late
                and actor_behavior_ref_echo_ok
            ),
            "actor_behavior_ref_source": actor_behavior_ref_source,
            "actor_behavior_ref_contract": RANK1_BUMP_CONTRACT,
            "actor_behavior_ref_plan_id": actor_behavior_plan_id,
            "actor_behavior_ref_start_offset": actor_behavior_start_offset,
            "actor_behavior_ref_echo_verified": actor_behavior_ref_echo_ok,
            "actor_behavior_ref_requested": requested_target is not None,
            "action_schema_fingerprint": self.config.action_schema_fingerprint,
            "actor_projection_profile": self.config.actor_projection_profile,
            "actor_response_action_schema_fingerprint": output.metadata.get(
                "actor_action_schema_fingerprint"
            ),
            "actor_response_projection_profile": output.metadata.get(
                "actor_projection_profile"
            ),
        }
        if request_reason == "behavior_prefetch":
            # Stage the raw H50.  Its exact executable suffix is unknown until
            # the real boundary supplies the latest executed target and age.
            # Do not expose the response Actor: it was conditioned on the raw
            # response actions, not the velocity-aligned behavior reference.
            self._prefetched_model_plan = policy_plan.copy()
            self._prefetched_model_z_rl = enrichment_z_rl.copy()
            self._prefetched_model_metadata = dict(metadata)
            self._prefetched_model_key = latest_key
            self._clear_policy_request(request_lane)
            return
        actor_chunk_offset = int(self._enrichment_plan_index)
        actor_chunk_partial = bool(
            self.config.actor_live
            and actor_phase_active
            and self._enrichment_plan_actor is not None
            and self._enrichment_plan_metadata.get("actor_shadow_ready", False)
            and 0 < actor_chunk_offset < int(self._enrichment_plan_limit)
        )
        current_behavior_plan_id = self._model_plan_metadata.get("policy_plan_id")
        current_behavior_index = int(self._model_plan_index)
        target_is_future_on_current_behavior = bool(
            requested_target is not None
            and str(current_behavior_plan_id) == requested_target.plan_id
            and current_behavior_index < requested_target.start_offset
        )
        stage_enrichment = bool(
            target_is_future_on_current_behavior
            or (request_reason == "boundary_miss" and actor_chunk_partial and not install_behavior)
        )
        metadata["actor_chunk_switch_mode"] = "staged_at_c10_boundary" if stage_enrichment else "installed_now"
        metadata["actor_chunk_offset_when_result_arrived"] = actor_chunk_offset
        if stage_enrichment:
            # Never replace a partially executed C=10 plan merely because a
            # fast prefetch or an asynchronously completed SFT boundary plan
            # arrived.  The SFT behavior plan may switch immediately below,
            # but the Actor/reference enrichment pair switches atomically only
            # after the current C=10 Actor chunk reaches its boundary.
            self._prefetched_enrichment_plan = actor_reference_plan
            self._prefetched_enrichment_z_rl = enrichment_z_rl
            self._prefetched_enrichment_actor = enrichment_actor
            self._prefetched_enrichment_metadata = metadata
            self._prefetched_enrichment_key = latest_key
        else:
            self._install_enrichment_plan(
                plan=actor_reference_plan,
                z_rl=enrichment_z_rl,
                actor=enrichment_actor,
                metadata=metadata,
                key=latest_key,
                chunk_length=chunk_length,
            )

        if install_behavior:
            self._model_plan = policy_plan.copy()
            self._model_plan_z_rl = enrichment_z_rl.copy()
            self._model_plan_actor = None if enrichment_actor is None else enrichment_actor.copy()
            self._model_plan_metadata = dict(metadata)
            self._model_plan_key = latest_key
            self._model_plan_index = 0
            self._model_plan_limit = min(int(policy_plan.shape[0]), int(self.config.model_execute_steps))

        self._clear_policy_request(request_lane)

    def _install_enrichment_plan(
        self,
        *,
        plan: np.ndarray,
        z_rl: np.ndarray,
        actor: np.ndarray | None,
        metadata: dict[str, Any],
        key: tuple[float, float],
        chunk_length: int,
    ) -> None:
        self._enrichment_plan = plan
        self._enrichment_plan_z_rl = z_rl
        self._enrichment_plan_actor = actor
        self._enrichment_plan_metadata = metadata
        self._enrichment_plan_key = key
        self._enrichment_plan_index = 0
        self._enrichment_plan_limit = min(int(plan.shape[0]), int(chunk_length))
        # A forced phase/takeover refresh supersedes any older staged plan.
        self._prefetched_enrichment_plan = None
        self._prefetched_enrichment_z_rl = None
        self._prefetched_enrichment_actor = None
        self._prefetched_enrichment_metadata = None
        self._prefetched_enrichment_key = None

    def _invalidate_actor_enrichment(
        self,
        *,
        reason: str,
        min_observation_t: int,
    ) -> None:
        """Atomically invalidate Actor/token payloads across control epochs."""

        self._actor_enrichment_min_observation_t = int(min_observation_t)
        self._enrichment_plan_z_rl = None
        self._enrichment_plan_actor = None
        self._enrichment_plan_metadata = dict(self._enrichment_plan_metadata)
        self._enrichment_plan_metadata.update(
            {
                "actor_shadow_ready": False,
                "actor_enrichment_invalidated": True,
                "actor_enrichment_invalidation_reason": str(reason),
                "actor_enrichment_min_observation_t": int(min_observation_t),
            }
        )
        self._prefetched_enrichment_plan = None
        self._prefetched_enrichment_z_rl = None
        self._prefetched_enrichment_actor = None
        self._prefetched_enrichment_metadata = None
        self._prefetched_enrichment_key = None

    def _clear_prefetched_model_plan(self) -> None:
        self._prefetched_model_plan = None
        self._prefetched_model_z_rl = None
        self._prefetched_model_metadata = None
        self._prefetched_model_key = None

    def _activate_prefetched_model_plan(
        self,
        *,
        now_s: float,
        state_snapshot: np.ndarray,
        chunk_length: int,
    ) -> bool:
        if (
            self._prefetched_model_plan is None
            or self._prefetched_model_z_rl is None
            or self._prefetched_model_metadata is None
            or self._prefetched_model_key is None
        ):
            return False
        raw_plan = self._prefetched_model_plan
        z_rl = self._prefetched_model_z_rl
        metadata = dict(self._prefetched_model_metadata)
        key = self._prefetched_model_key
        handoff_target, previous_target, handoff_history_source = self._handoff_base_targets(
            state_snapshot=state_snapshot,
        )
        observation_timestamp_s = float(
            metadata.get("policy_observation_timestamp_s", now_s)
        )
        observation_age_s = max(0.0, float(now_s) - observation_timestamp_s)
        try:
            prepared = prepare_h50_handoff_plan(
                raw_plan,
                observation_state=np.asarray(state_snapshot, dtype=np.float64),
                handoff_target=handoff_target,
                previous_target=previous_target,
                observation_age_s=observation_age_s,
                model_smoothing_tau_s=self.config.model_smoothing_tau_s,
            )
        except H50HandoffRejected as exc:
            self._model_plan_metadata = {
                **self._model_plan_metadata,
                "h50_prefetch_accepted": False,
                "h50_prefetch_fallback_reason": str(exc),
                "h50_feedback_hold_threshold_rad": (
                    H50_FALLBACK_FEEDBACK_HOLD_THRESHOLD_RAD
                ),
                "h50_handoff_history_source": handoff_history_source,
                "h50_boundary_keepalive_count_before_handoff": self._boundary_keepalive_count,
            }
            self._clear_prefetched_model_plan()
            return False

        executable_plan = prepared.actions.astype(np.float32, copy=True)
        metadata.update(
            {
                "h50_prefetch_accepted": True,
                "h50_prefetch_fallback_reason": None,
                "h50_policy_action_start_index": prepared.action_start_index,
                "h50_observation_age_at_handoff_s": prepared.observation_age_s,
                "h50_bridge_steps": prepared.bridge_steps,
                "h50_raw_boundary_jump_rad": prepared.raw_boundary_jump_rad,
                "h50_bridge_excursion_rad": prepared.bridge_excursion_rad,
                "h50_bridge_target_correction_rad": (
                    prepared.bridge_target_correction_rad
                ),
                "h50_bridge_target_correction_clipped": (
                    prepared.bridge_target_correction_clipped
                ),
                "h50_feedback_hold_threshold_rad": (
                    H50_FALLBACK_FEEDBACK_HOLD_THRESHOLD_RAD
                ),
                "h50_handoff_history_source": handoff_history_source,
                "h50_handoff_actor_carry_removed": bool(
                    handoff_history_source == "executed_model_history_minus_actor_carry"
                    and self._last_exec_actor_residual is not None
                    and np.any(np.abs(self._last_exec_actor_residual) > 0.0)
                ),
                "h50_boundary_keepalive_count_before_handoff": self._boundary_keepalive_count,
                "h50_handoff_actor_suppressed": True,
                "actor_shadow_ready": False,
                "actor_behavior_ref_source": "h50_velocity_aligned_behavior",
                "actor_behavior_ref_plan_id": metadata.get("policy_plan_id"),
                "actor_behavior_ref_start_offset": 0,
                "actor_behavior_ref_echo_verified": False,
                "actor_chunk_switch_mode": "h50_first_c10_actor_suppressed",
            }
        )
        self._model_plan = executable_plan
        self._model_plan_z_rl = z_rl.copy()
        self._model_plan_actor = None
        self._model_plan_metadata = dict(metadata)
        self._model_plan_key = key
        self._model_plan_index = 0
        self._model_plan_limit = min(
            int(executable_plan.shape[0]), int(self.config.model_execute_steps)
        )
        self._install_enrichment_plan(
            plan=executable_plan,
            z_rl=z_rl,
            actor=None,
            metadata=metadata,
            key=key,
            chunk_length=chunk_length,
        )
        self._boundary_keepalive_count = 0
        self._clear_prefetched_model_plan()
        return True

    def _activate_prefetched_enrichment_plan(self) -> bool:
        if (
            self._prefetched_enrichment_plan is None
            or self._prefetched_enrichment_z_rl is None
            or self._prefetched_enrichment_metadata is None
            or self._prefetched_enrichment_key is None
        ):
            return False
        target_plan_id = self._prefetched_enrichment_metadata.get(
            "actor_behavior_ref_plan_id"
        )
        target_offset = self._prefetched_enrichment_metadata.get(
            "actor_behavior_ref_start_offset"
        )
        current_plan_id = self._model_plan_metadata.get("policy_plan_id")
        current_offset = int(self._model_plan_index)
        aligned = bool(
            target_plan_id is not None
            and target_offset is not None
            and str(target_plan_id) == str(current_plan_id)
            and int(target_offset) == current_offset
        )
        if not aligned:
            target_is_still_future = bool(
                target_plan_id is not None
                and target_offset is not None
                and str(target_plan_id) == str(current_plan_id)
                and current_offset < int(target_offset)
            )
            if target_is_still_future:
                return False
            # A delayed response or behavior-plan replacement invalidates the
            # complete C10 atomically; never rebase it onto a different slice.
            self._prefetched_enrichment_plan = None
            self._prefetched_enrichment_z_rl = None
            self._prefetched_enrichment_actor = None
            self._prefetched_enrichment_metadata = None
            self._prefetched_enrichment_key = None
            return False
        plan = self._prefetched_enrichment_plan
        z_rl = self._prefetched_enrichment_z_rl
        actor = self._prefetched_enrichment_actor
        metadata = self._prefetched_enrichment_metadata
        key = self._prefetched_enrichment_key
        self._install_enrichment_plan(
            plan=plan,
            z_rl=z_rl,
            actor=actor,
            metadata=metadata,
            key=key,
            chunk_length=self.config.chunk_length,
        )
        return True

    def _effective_plan_remaining(self) -> int | None:
        if self._model_plan is None:
            return None
        return max(0, int(self._model_plan_limit) - int(self._model_plan_index))

    def _enrichment_plan_remaining(self) -> int | None:
        if self._enrichment_plan is None:
            return None
        return max(0, int(self._enrichment_plan_limit) - int(self._enrichment_plan_index))

    def _actor_request_behavior_target(
        self,
        *,
        request_reason: str,
        state_snapshot: np.ndarray,
    ) -> BehaviorReferenceTarget | None:
        """Bind an Actor request to the exact H50 slice it may later modify."""

        if request_reason in {"initial", "boundary_miss", "behavior_prefetch"} or self._model_plan is None:
            return None
        behavior_plan_id = self._model_plan_metadata.get("policy_plan_id")
        if behavior_plan_id is None:
            return None
        behavior_index = int(self._model_plan_index)
        is_prefetch = request_reason in {"actor_prefetch", "replay_enrichment_prefetch"}
        if is_prefetch:
            enrichment_remaining = self._enrichment_plan_remaining()
            if enrichment_remaining is None:
                return None
            target_offset = behavior_index + int(enrichment_remaining)
        else:
            target_offset = behavior_index
        target_end = target_offset + int(self.config.chunk_length)
        if target_offset < 0 or target_end > int(self._model_plan_limit):
            return None
        behavior_ref = self._model_plan[target_offset:target_end, : self.config.action_dim]
        if behavior_ref.shape != (self.config.chunk_length, self.config.action_dim):
            return None
        if is_prefetch and target_offset > 0:
            conditioning_state = self._model_plan[target_offset - 1, : self.config.action_dim]
        else:
            conditioning_state = np.asarray(state_snapshot, dtype=np.float32)
        return BehaviorReferenceTarget(
            actions=np.asarray(behavior_ref, dtype=np.float32),
            plan_id=str(behavior_plan_id),
            start_offset=target_offset,
            conditioning_state=np.asarray(conditioning_state, dtype=np.float32),
        ).validate(chunk_length=self.config.chunk_length, action_dim=self.config.action_dim)

    def _policy_request_context(
        self,
        lane: str,
    ) -> tuple[str | None, int | None, BehaviorReferenceTarget | None, float | None]:
        if lane == "base":
            return (
                self._base_request_reason,
                self._base_request_lead_steps,
                None,
                self._base_wait_started_s,
            )
        return (
            self._policy_request_reason,
            self._policy_request_lead_steps,
            self._policy_request_behavior_target,
            self._policy_wait_started_s,
        )

    def _mark_policy_request_submitted(
        self,
        *,
        lane: str,
        reason: str,
        lead_steps: int | None,
        behavior_target: BehaviorReferenceTarget | None,
    ) -> None:
        if lane == "base":
            self._base_request_pending = True
            self._base_request_reason = reason
            self._base_request_lead_steps = lead_steps
            return
        self._policy_request_pending = True
        self._policy_request_reason = reason
        self._policy_request_lead_steps = lead_steps
        self._policy_request_behavior_target = behavior_target

    def _clear_policy_request(self, lane: str) -> None:
        if lane == "base":
            self._base_request_pending = False
            self._base_request_reason = None
            self._base_request_lead_steps = None
            self._base_wait_started_s = None
            return
        self._policy_request_pending = False
        self._policy_request_reason = None
        self._policy_request_lead_steps = None
        self._policy_request_behavior_target = None
        self._policy_wait_started_s = None

    def _base_policy_request_kind(self) -> str | None:
        if self._base_request_pending:
            return None
        remaining = self._effective_plan_remaining()
        if remaining is None:
            return "initial"
        if remaining <= 0:
            if self._prefetched_model_plan is not None:
                return None
            return "boundary_miss"
        if (
            self.config.model_execute_steps == 50
            and self._prefetched_model_plan is None
            and remaining <= self.config.model_prefetch_lead_steps
        ):
            return "behavior_prefetch"
        return None

    def _enrichment_policy_request_kind(
        self,
        *,
        enrichment_active: bool,
        force_enrichment_refresh: bool,
    ) -> str | None:
        if self._policy_request_pending or not enrichment_active or self._model_plan is None:
            return None
        if force_enrichment_refresh:
            return "replay_enrichment"
        if self._prefetched_enrichment_plan is not None:
            return None
        enrichment_remaining = self._enrichment_plan_remaining()
        if enrichment_remaining is None or enrichment_remaining <= 0:
            return "replay_enrichment"
        if enrichment_remaining <= self.config.actor_prefetch_lead_steps:
            return "replay_enrichment_prefetch"
        return None

    def _policy_request_kinds(
        self,
        *,
        enrichment_active: bool,
        force_enrichment_refresh: bool,
    ) -> list[tuple[str, str]]:
        if self.enrichment_policy_worker is None:
            reason = self._policy_request_kind(
                enrichment_active=enrichment_active,
                force_enrichment_refresh=force_enrichment_refresh,
            )
            return [] if reason is None else [("shared", reason)]

        # Submit the H50 lane first.  Its worker and websocket are independent,
        # so a token/Actor request can neither replace its pending observation
        # nor prevent the control-critical standby from being requested.
        requests: list[tuple[str, str]] = []
        base_reason = self._base_policy_request_kind()
        if base_reason is not None:
            requests.append(("base", base_reason))
        enrichment_reason = self._enrichment_policy_request_kind(
            enrichment_active=enrichment_active,
            force_enrichment_refresh=force_enrichment_refresh,
        )
        if enrichment_reason is not None:
            requests.append(("enrichment", enrichment_reason))
        return requests

    def _policy_request_kind(
        self,
        *,
        enrichment_active: bool = False,
        force_enrichment_refresh: bool = False,
    ) -> str | None:
        if self._policy_request_pending:
            return None
        remaining = self._effective_plan_remaining()
        if remaining is None:
            return "initial"
        if remaining <= 0:
            if self._prefetched_model_plan is not None:
                return None
            return "boundary_miss"
        if (
            self.config.model_execute_steps == 50
            and self._prefetched_model_plan is None
            and remaining <= self.config.model_prefetch_lead_steps
        ):
            return "behavior_prefetch"
        if enrichment_active:
            if force_enrichment_refresh:
                return "replay_enrichment"
            if self._prefetched_enrichment_plan is not None:
                return None
            enrichment_remaining = self._enrichment_plan_remaining()
            if enrichment_remaining is None or enrichment_remaining <= 0:
                return "replay_enrichment"
            if enrichment_remaining <= self.config.actor_prefetch_lead_steps:
                return "replay_enrichment_prefetch"
        return None

    def _model_boundary_keepalive(self, *, now_s: float) -> TimedCommand | None:
        """Repeat the last safe model target while an H50 standby is late.

        This mirrors pure inference's boundary keepalive.  Crucially, the
        repeated target is not appended to the handoff history, so the local
        pre-boundary velocity remains available when the standby is aligned.
        """

        if self._handoff_last_command is not None:
            target = self._handoff_last_command
        elif self._last_exec_command is not None and self._last_exec_source in {"pi05", "rlt"}:
            # Compatibility for direct/unit callers that predate the explicit
            # handoff history.
            target = self._last_exec_command
        else:
            return None
        return TimedCommand(
            value=np.asarray(target, dtype=np.float32).copy(),
            timestamp_s=float(now_s),
        )

    def _record_handoff_execution(
        self,
        command: np.ndarray,
        *,
        source: str,
        is_boundary_keepalive: bool,
        actor_residual: np.ndarray | None,
    ) -> None:
        # Human control changes the physical anchor outside the model plan.
        # The next H50 must therefore use fresh feedback rather than an old
        # model velocity.  Safety/terminal holds, however, must not erase the
        # last real model velocity as they did in the former RLT path.
        if source == "human_pika":
            self._handoff_last_command = None
            self._handoff_previous_command = None
            self._last_exec_actor_residual = None
            self._previous_exec_actor_residual = None
            return
        if source not in {"pi05", "rlt"} or is_boundary_keepalive:
            return
        value = np.asarray(command, dtype=np.float32)
        if value.shape != (self.config.action_dim,) or not np.all(np.isfinite(value)):
            return
        residual = np.zeros(self.config.action_dim, dtype=np.float32)
        if source == "rlt" and actor_residual is not None:
            candidate = np.asarray(actor_residual, dtype=np.float32)
            if candidate.shape == residual.shape and np.all(np.isfinite(candidate)):
                residual = candidate.copy()
                if (
                    self.config.actor_gripper_residual_mode
                    == GRIPPER_RESIDUAL_FROZEN
                ):
                    residual[6] = 0.0
        self._handoff_previous_command = (
            value.copy()
            if self._handoff_last_command is None
            else self._handoff_last_command.copy()
        )
        self._previous_exec_actor_residual = (
            residual.copy()
            if self._last_exec_actor_residual is None
            else self._last_exec_actor_residual.copy()
        )
        self._handoff_last_command = value.copy()
        self._last_exec_actor_residual = residual.copy()

    def _handoff_base_targets(
        self,
        *,
        state_snapshot: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        if self._handoff_last_command is not None:
            handoff = self._handoff_last_command.astype(np.float64, copy=True)
            previous = (
                handoff.copy()
                if self._handoff_previous_command is None
                else self._handoff_previous_command.astype(np.float64, copy=True)
            )
            last_residual = (
                np.zeros(self.config.action_dim, dtype=np.float64)
                if self._last_exec_actor_residual is None
                else self._last_exec_actor_residual.astype(np.float64, copy=True)
            )
            previous_residual = (
                last_residual.copy()
                if self._previous_exec_actor_residual is None
                else self._previous_exec_actor_residual.astype(np.float64, copy=True)
            )
            return (
                handoff - last_residual,
                previous - previous_residual,
                "executed_model_history_minus_actor_carry",
            )
        if self._last_exec_command is not None and self._last_exec_source in {"pi05", "rlt"}:
            handoff = self._last_exec_command.astype(np.float64, copy=True)
            previous = (
                handoff.copy()
                if self._previous_exec_command is None
                else self._previous_exec_command.astype(np.float64, copy=True)
            )
            return handoff, previous, "legacy_executed_model_history"
        state = np.asarray(state_snapshot, dtype=np.float64).copy()
        return state, state.copy(), "fresh_feedback"

    def _command_publisher_telemetry(self) -> dict[str, Any]:
        telemetry = getattr(self.command_publisher, "telemetry", None)
        if telemetry is not None:
            try:
                values = dict(telemetry())
            except Exception as exc:  # pragma: no cover - logging must fail open
                values = {"command_publisher_telemetry_error": f"{type(exc).__name__}: {exc}"}
        else:
            values = {}
        values.setdefault(
            "command_publisher_hz",
            50.0 if self.config.hardware_io == "native_sdk" else self.config.control_hz,
        )
        values.setdefault("command_publisher_count", None)
        values.setdefault("command_publisher_hold_count", self._boundary_keepalive_count)
        return values

    def _filter_model_with(
        self,
        safety_filter: StatefulSafetyFilter,
        command: np.ndarray,
        *,
        snapshot: np.ndarray,
        feedback: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Advance one explicit model filter without touching runtime metadata."""

        if self.config.model_safety_profile == "native":
            result = safety_filter.filter_model_native(command, dt=dt)
        else:
            result = safety_filter.filter(
                command,
                snapshot=snapshot,
                feedback=feedback,
                dt=dt,
            )
        return result.command.astype(np.float32)

    def _filter_command(
        self,
        command: np.ndarray,
        *,
        source: str,
        snapshot: np.ndarray,
        feedback: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, str, list[str]]:
        if source == "human_pika":
            safety_profile = "human_native"
        elif source in {"pi05", "rlt"}:
            safety_profile = f"model_{self.config.model_safety_profile}"
        else:
            safety_profile = "hold"
        if self.safety_filter is None:
            self._last_feedback_hold_metadata = {
                "h50_feedback_anchored_hold": False,
                "h50_feedback_hold_tracking_error_rad": 0.0,
            }
            return np.asarray(command, dtype=np.float32).copy(), safety_profile, []
        self._last_feedback_hold_metadata = {
            "h50_feedback_anchored_hold": False,
            "h50_feedback_hold_tracking_error_rad": 0.0,
        }
        if source not in {"human_pika", "pi05", "rlt"}:
            tracking_error = float(
                np.max(
                    np.abs(
                        np.asarray(feedback, dtype=np.float64)[:6]
                        - self.safety_filter.previous_command[:6]
                    )
                )
            )
            self._last_feedback_hold_metadata = {
                "h50_feedback_anchored_hold": bool(
                    tracking_error > H50_FALLBACK_FEEDBACK_HOLD_THRESHOLD_RAD
                ),
                "h50_feedback_hold_tracking_error_rad": tracking_error,
            }
            if tracking_error > H50_FALLBACK_FEEDBACK_HOLD_THRESHOLD_RAD:
                self.safety_filter = StatefulSafetyFilter(
                    self.safety_filter.config,
                    np.asarray(feedback, dtype=np.float64),
                )
                result = self.safety_filter.filter_native(feedback)
                return (
                    result.command.astype(np.float32),
                    safety_profile,
                    ["tracking_resync", "feedback_anchor_resync"],
                )
        if source == "human_pika":
            result = self.safety_filter.filter_human_native(command)
        elif source in {"pi05", "rlt"} and self.config.model_safety_profile == "native":
            result = self.safety_filter.filter_model_native(command, dt=dt)
        else:
            result = self.safety_filter.filter(command, snapshot=snapshot, feedback=feedback, dt=dt)
        return result.command.astype(np.float32), safety_profile, list(result.reasons)


class EpisodeImageWriter:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.global_dir = self.root / "camera_global"
        self.wrist_dir = self.root / "camera_wrist"
        self.global_dir.mkdir(parents=True, exist_ok=True)
        self.wrist_dir.mkdir(parents=True, exist_ok=True)

    def save(self, *, t: int, images: dict[str, np.ndarray]) -> tuple[str, str]:
        import cv2

        global_rel = Path("camera_global") / f"{t:06d}.jpg"
        wrist_rel = Path("camera_wrist") / f"{t:06d}.jpg"
        _write_rgb_image(self.root / global_rel, images["camera1"], cv2)
        _write_rgb_image(self.root / wrist_rel, images["camera2"], cv2)
        return global_rel.as_posix(), wrist_rel.as_posix()


def _slice_action_chunk(plan: np.ndarray | None, start: int, length: int, action_dim: int) -> np.ndarray:
    if plan is None or length <= 0:
        return np.zeros((max(0, int(length)), int(action_dim)), dtype=np.float32)
    actions = np.asarray(plan, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < action_dim or actions.shape[0] == 0:
        return np.zeros((int(length), int(action_dim)), dtype=np.float32)
    start = max(0, int(start))
    stop = min(actions.shape[0], start + int(length))
    chunk = actions[start:stop, :action_dim].astype(np.float32, copy=True)
    if chunk.shape[0] == length:
        return chunk
    if chunk.shape[0] == 0:
        pad_row = actions[-1:, :action_dim].astype(np.float32, copy=True)
    else:
        pad_row = chunk[-1:, :]
    pad = np.repeat(pad_row, int(length) - chunk.shape[0], axis=0)
    return np.concatenate([chunk, pad], axis=0)


def _write_rgb_image(path: Path, image: np.ndarray, cv2_module: Any) -> None:
    bgr = image[:, :, ::-1]
    ok = cv2_module.imwrite(str(path), bgr)
    if not ok:
        raise RuntimeError(f"failed to write image: {path}")


class NonBlockingTerminalKeySource:
    def __init__(self, stream: Any = None):
        self.stream = stream if stream is not None else sys.stdin
        self._fd: int | None = None
        self._old_settings: Any | None = None

    def __enter__(self) -> "NonBlockingTerminalKeySource":
        import termios
        import tty

        self._fd = self.stream.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fd is not None and self._old_settings is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def poll_keys(self) -> list[str]:
        import select

        if self._fd is None:
            return []
        keys: list[str] = []
        while True:
            readable, _, _ = select.select([self.stream], [], [], 0)
            if not readable:
                return keys
            char = self.stream.read(1)
            if not char:
                return keys
            keys.append(char)


class NullKeySource:
    def poll_keys(self) -> list[str]:
        return []


class ScriptedKeySource:
    def __init__(self, schedule: dict[int, list[str]]):
        self.schedule = {int(k): list(v) for k, v in schedule.items()}
        self.t = 0

    def poll_keys(self) -> list[str]:
        return list(self.schedule.get(int(self.t), []))


def parse_scripted_keys(value: str | None) -> dict[int, list[str]]:
    if value is None or value.strip() == "":
        return {}
    schedule: dict[int, list[str]] = {}
    allowed = {"s", "1", "0", "e", "q"}
    for raw_entry in value.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(f"scripted key entry must be '<step>:<key>', got {entry!r}")
        raw_step, raw_key = entry.split(":", 1)
        try:
            step = int(raw_step)
        except ValueError as exc:
            raise ValueError(f"scripted key step must be an integer, got {raw_step!r}") from exc
        key = raw_key.strip().lower()
        if step < 0:
            raise ValueError("scripted key step must be non-negative")
        if key not in allowed:
            raise ValueError(f"scripted key allowed values are {sorted(allowed)}, got {key!r}")
        schedule.setdefault(step, []).append(key)
    return schedule


def decode_ros_arm_error_code(code: int) -> list[str]:
    labels = []
    for index in range(6):
        if code & (1 << index):
            labels.append(f"joint_{index + 1}_communication")
        if code & (1 << (index + 8)):
            labels.append(f"joint_{index + 1}_angle_limit")
    known_mask = 0x003F | 0x3F00
    unknown = int(code) & ~known_mask
    if unknown:
        labels.append(f"unknown_bits_0x{unknown:04x}")
    return labels


class ArmStatusHealthError(RuntimeError):
    def __init__(self, code: int, nonzero_duration_s: float):
        self.code = int(code)
        self.nonzero_duration_s = float(nonzero_duration_s)
        self.communication_only = self.code != 0 and (self.code & ~0x003F) == 0
        labels = ",".join(decode_ros_arm_error_code(self.code)) or "unknown"
        super().__init__(
            f"ROS Piper arm error code: {self.code} ({labels}); "
            f"nonzero_for={self.nonzero_duration_s:.3f}s"
        )


@dataclasses.dataclass
class FreshArmStatusTracker:
    freshness_s: float = 0.5
    _err_code: int | None = dataclasses.field(default=None, init=False, repr=False)
    _timestamp_s: float | None = dataclasses.field(default=None, init=False, repr=False)
    _nonzero_since_s: float | None = dataclasses.field(default=None, init=False, repr=False)
    _lock: Any = dataclasses.field(default_factory=threading.Lock, init=False, repr=False)

    def update(self, message: Any, *, timestamp_s: float | None = None) -> None:
        code = int(getattr(message, "err_code"))
        stamp = time.monotonic() if timestamp_s is None else float(timestamp_s)
        with self._lock:
            if code == 0:
                self._nonzero_since_s = None
            elif self._err_code in (None, 0) or self._nonzero_since_s is None:
                self._nonzero_since_s = stamp
            self._err_code = code
            self._timestamp_s = stamp

    def require_healthy(self, *, now_s: float | None = None) -> None:
        current = time.monotonic() if now_s is None else float(now_s)
        with self._lock:
            code = self._err_code
            timestamp_s = self._timestamp_s
            nonzero_since_s = self._nonzero_since_s
        if timestamp_s is None or current - timestamp_s > self.freshness_s:
            raise RuntimeError("no fresh ROS Piper arm status on /arm_status")
        if code not in (None, 0):
            duration = 0.0 if nonzero_since_s is None else max(0.0, current - nonzero_since_s)
            raise ArmStatusHealthError(code, duration)


class RosFeedbackReader:
    def __init__(
        self,
        tracker: FreshJointCommandTracker,
        arm_status: FreshArmStatusTracker | None = None,
        *,
        communication_grace_s: float = 0.75,
    ):
        self.tracker = tracker
        self.arm_status = arm_status
        self.communication_grace_s = float(communication_grace_s)

    def read(self) -> np.ndarray:
        if self.arm_status is not None:
            while True:
                try:
                    self.arm_status.require_healthy(now_s=time.monotonic())
                    break
                except ArmStatusHealthError as exc:
                    # Codes 1..63 are joint communication flags, not angle
                    # limits. Pause command production while a short CAN status
                    # glitch clears, but fail closed if it persists.
                    if not exc.communication_only or exc.nonzero_duration_s >= self.communication_grace_s:
                        raise
                    time.sleep(0.01)
        latest = self.tracker.latest(now_s=time.monotonic())
        if latest is None or latest.value is None:
            raise RuntimeError("no fresh ROS feedback on /joint_states_single")
        return latest.value


class RosCommandPublisher:
    def __init__(self, publisher: Any, *, action_dim: int):
        self.publisher = publisher
        self.action_dim = action_dim
        self._publish_count = 0
        self._last_publish_s: float | None = None

    def publish(self, command: np.ndarray) -> None:
        self.publisher.publish(vector_to_joint_state(command, action_dim=self.action_dim))
        self._publish_count += 1
        self._last_publish_s = time.monotonic()

    def telemetry(self) -> dict[str, Any]:
        now_s = time.monotonic()
        return {
            "command_publisher_input_count": self._publish_count,
            "command_publisher_last_publish_age_s": (
                None
                if self._last_publish_s is None
                else max(0.0, now_s - self._last_publish_s)
            ),
        }


class NativePikaJointStatePassthrough:
    """Forward accepted Pika messages to the official Piper ROS controller.

    The RLT loop remains 30 Hz for replay/learning, while native Pika teleop
    keeps its original 50 Hz JointState stream and the official
    piper_ctrl_single_node command primitive.
    """

    def __init__(self, publisher: Any, *, heartbeat_timeout_s: float = 0.12, now_fn: Any = time.monotonic):
        self.publisher = publisher
        self.heartbeat_timeout_s = float(heartbeat_timeout_s)
        self.now_fn = now_fn
        self._lock = threading.Lock()
        self._active = False
        self._heartbeat_s = float("-inf")
        self.forwarded_count = 0

    def set_active(self, active: bool) -> None:
        now_s = float(self.now_fn())
        with self._lock:
            self._active = bool(active)
            if active:
                self._heartbeat_s = now_s

    def forward(self, message: Any) -> bool:
        now_s = float(self.now_fn())
        with self._lock:
            allowed = self._active and now_s - self._heartbeat_s <= self.heartbeat_timeout_s
            if not allowed:
                return False
            self.publisher.publish(message)
            self.forwarded_count += 1
            return True


class RosArbitratedCommandPublisher(RosCommandPublisher):
    """Use one official ROS Piper controller for model, reset, and Pika.

    Accepted Pika messages are already forwarded at their native rate by
    ``NativePikaJointStatePassthrough``.  The 30 Hz decision loop must not send
    a second downsampled copy of the same human command.
    """

    def __init__(
        self,
        publisher: Any,
        *,
        action_dim: int,
        human_passthrough: NativePikaJointStatePassthrough | None = None,
        rospy_module: Any | None = None,
        input_hz: float = 30.0,
        output_hz: float | None = None,
        stale_timeout_s: float = 0.15,
        now_fn: Any = time.monotonic,
    ):
        super().__init__(publisher, action_dim=action_dim)
        self.human_passthrough = human_passthrough
        self.rospy = rospy_module
        self.input_hz = float(input_hz)
        self.output_hz = None if output_hz is None else float(output_hz)
        self.stale_timeout_s = float(stale_timeout_s)
        self.now_fn = now_fn
        if self.input_hz <= 0.0 or self.stale_timeout_s <= 0.0:
            raise ValueError("ROS interpolation input rate and stale timeout must be positive")
        if self.output_hz is not None and self.output_hz <= 0.0:
            raise ValueError("ROS interpolation output rate must be positive")
        self._servo_lock = threading.Lock()
        self._servo_stop = threading.Event()
        self._servo_target: np.ndarray | None = None
        self._servo_segment_start: np.ndarray | None = None
        self._servo_last_sent: np.ndarray | None = None
        self._servo_target_received_s: float | None = None
        self._servo_last_output_s: float | None = None
        self._servo_output_count = 0
        self._servo_paused = False
        self._servo_error: Exception | None = None
        self._servo_thread: threading.Thread | None = None
        if self.output_hz is not None:
            self._servo_thread = threading.Thread(
                target=self._servo_loop,
                name="rlt-ros-model-50hz",
                daemon=True,
            )
            self._servo_thread.start()

    def publish_selected(self, command: np.ndarray, *, source: str) -> None:
        if source == "human_pika" and self.human_passthrough is not None:
            return
        if self.output_hz is None:
            super().publish(command)
            return
        value = np.asarray(command, dtype=np.float32)
        if value.shape != (self.action_dim,) or not np.all(np.isfinite(value)):
            raise ValueError("interpolated ROS command must be finite with shape (7,)")
        with self._servo_lock:
            if self._servo_error is not None:
                raise RuntimeError(f"50 Hz ROS command publisher failed: {self._servo_error}")
            if self._servo_stop.is_set():
                raise RuntimeError("50 Hz ROS command publisher is closed")
            if self._servo_paused:
                return
            self._servo_segment_start = (
                value.copy()
                if self._servo_last_sent is None
                else self._servo_last_sent.copy()
            )
            self._servo_target = value.copy()
            self._servo_target_received_s = float(self.now_fn())
            self._publish_count += 1
            self._last_publish_s = float(self.now_fn())

    def publish(self, command: np.ndarray) -> None:
        self.publish_selected(command, source="pi05")

    def set_external_passthrough_active(self, active: bool) -> None:
        if self.output_hz is None:
            return
        with self._servo_lock:
            active = bool(active)
            if self._servo_paused == active:
                return
            self._servo_paused = active
            self._servo_target = None
            self._servo_segment_start = None
            self._servo_target_received_s = None
            # Human Pika now owns the official controller.  Do not use a
            # pre-takeover interpolated sample as the next model anchor.
            if active:
                self._servo_last_sent = None

    def _servo_emit(self, now_s: float) -> bool:
        with self._servo_lock:
            if (
                self._servo_paused
                or self._servo_target is None
                or self._servo_target_received_s is None
            ):
                return False
            if now_s - self._servo_target_received_s > self.stale_timeout_s:
                return False
            start = (
                self._servo_target
                if self._servo_segment_start is None
                else self._servo_segment_start
            )
            alpha = float(
                np.clip(
                    (now_s - self._servo_target_received_s) * self.input_hz,
                    0.0,
                    1.0,
                )
            )
            command = (
                (1.0 - alpha) * start + alpha * self._servo_target
            ).astype(np.float32)
            try:
                message = vector_to_joint_state(command, action_dim=self.action_dim)
                if self.rospy is not None:
                    message.header.stamp = self.rospy.Time.now()
                message.header.frame_id = "pi05_50hz"
                self.publisher.publish(message)
            except Exception as exc:  # pragma: no cover - surfaced on next 30 Hz input
                self._servo_error = exc
                self._servo_stop.set()
                return False
            self._servo_last_sent = command.copy()
            self._servo_last_output_s = now_s
            self._servo_output_count += 1
        return True

    def _servo_loop(self) -> None:
        assert self.output_hz is not None
        period_s = 1.0 / self.output_hz
        next_deadline = float(self.now_fn())
        while not self._servo_stop.is_set():
            now_s = float(self.now_fn())
            wait_s = next_deadline - now_s
            if wait_s > 0.0:
                self._servo_stop.wait(wait_s)
                continue
            self._servo_emit(now_s)
            missed = max(
                1,
                int(math.floor((now_s - next_deadline) / period_s)) + 1,
            )
            next_deadline += missed * period_s

    def telemetry(self) -> dict[str, Any]:
        if self.output_hz is None:
            return super().telemetry()
        now_s = float(self.now_fn())
        with self._servo_lock:
            return {
                "command_publisher_hz": self.output_hz,
                "command_publisher_transport": "in_process_ros_30_to_50hz",
                "command_publisher_input_count": self._publish_count,
                "command_publisher_count": self._servo_output_count,
                "command_publisher_last_publish_age_s": (
                    None
                    if self._servo_last_output_s is None
                    else max(0.0, now_s - self._servo_last_output_s)
                ),
                "command_publisher_paused_for_human": self._servo_paused,
                "command_publisher_error": (
                    None if self._servo_error is None else str(self._servo_error)
                ),
            }

    def close(self, *, timeout_s: float = 1.0) -> None:
        self._servo_stop.set()
        if self._servo_thread is not None:
            self._servo_thread.join(timeout=max(0.0, float(timeout_s)))


class NativeSdkBridgeTelemetryTracker:
    """Latest actual 50 Hz SDK bridge counters for episode audit rows."""

    def __init__(self, *, now_fn: Any = time.monotonic):
        self.now_fn = now_fn
        self._lock = threading.Lock()
        self._values: dict[str, Any] | None = None
        self._received_s: float | None = None

    def update(self, message: Any) -> None:
        try:
            values = json.loads(str(getattr(message, "data", "")))
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(values, dict):
            return
        now_s = float(self.now_fn())
        with self._lock:
            self._values = dict(values)
            self._received_s = now_s

    def latest(self) -> dict[str, Any] | None:
        now_s = float(self.now_fn())
        with self._lock:
            if self._values is None:
                return None
            values = dict(self._values)
            values["telemetry_age_s"] = (
                None
                if self._received_s is None
                else max(0.0, now_s - self._received_s)
            )
            return values


class RosNativeSdkCommandPublisher(RosCommandPublisher):
    """Send a source-tagged command to the persistent SDK bridge."""

    def __init__(
        self,
        publisher: Any,
        rospy_module: Any,
        *,
        action_dim: int,
        human_passthrough: NativePikaJointStatePassthrough | None = None,
        passthrough_service: Any | None = None,
        bridge_telemetry: NativeSdkBridgeTelemetryTracker | None = None,
    ):
        super().__init__(publisher, action_dim=action_dim)
        self.rospy = rospy_module
        self.human_passthrough = human_passthrough
        self.passthrough_service = passthrough_service
        self.bridge_telemetry = bridge_telemetry
        self._external_passthrough_active: bool | None = None

    def set_external_passthrough_active(self, active: bool) -> None:
        active = bool(active)
        if self._external_passthrough_active == active:
            return
        if self.passthrough_service is None:
            raise RuntimeError("native SDK bridge passthrough arbitration service is unavailable")
        response = self.passthrough_service(active)
        if not bool(getattr(response, "success", False)):
            raise RuntimeError(
                "native SDK bridge rejected passthrough arbitration: "
                f"{getattr(response, 'message', '')}"
            )
        self._external_passthrough_active = active

    def publish_selected(self, command: np.ndarray, *, source: str) -> None:
        if source == "human_pika" and self.human_passthrough is not None:
            # The original 50 Hz JointState has already been forwarded to
            # piper_ctrl_single_node. Do not send a second, downsampled SDK
            # command from the 30 Hz RLT loop.
            return
        message = vector_to_joint_state(command, action_dim=self.action_dim)
        message.header.stamp = self.rospy.Time.now()
        message.header.frame_id = str(source)
        self.publisher.publish(message)
        self._publish_count += 1
        self._last_publish_s = time.monotonic()

    def telemetry(self) -> dict[str, Any]:
        values = super().telemetry()
        bridge = None if self.bridge_telemetry is None else self.bridge_telemetry.latest()
        values.update(
            {
                "command_publisher_hz": (
                    50.0 if bridge is None else float(bridge.get("output_hz", 50.0))
                ),
                "command_publisher_transport": "persistent_native_sdk_bridge_30_to_50hz",
                "command_publisher_count": (
                    None if bridge is None else int(bridge.get("command_count", 0))
                ),
                "command_publisher_hold_count": (
                    None if bridge is None else int(bridge.get("repeated_input_count", 0))
                ),
                "command_publisher_last_publish_age_s": (
                    None if bridge is None else bridge.get("last_output_age_s")
                ),
                "command_publisher_telemetry_age_s": (
                    None if bridge is None else bridge.get("telemetry_age_s")
                ),
            }
        )
        return values

    def publish(self, command: np.ndarray) -> None:
        self.publish_selected(command, source="pi05")


class PiperSdkCommandPublisher:
    """Publish selected commands through the same SDK primitive as native inference."""

    def __init__(self, sink: Any, *, model_speed_percent: int = 30, human_speed_percent: int = 50):
        self.sink = sink
        self.model_speed_percent = int(model_speed_percent)
        self.human_speed_percent = int(human_speed_percent)

    def publish_selected(self, command: np.ndarray, *, source: str) -> None:
        requested_speed = (
            self.human_speed_percent if source == "human_pika" else self.model_speed_percent
        )
        speed_changed = self.sink.move_speed_percent != requested_speed
        self.sink.move_speed_percent = requested_speed
        configure_motion_mode = getattr(self.sink, "configure_motion_mode", None)
        if speed_changed and configure_motion_mode is not None:
            configure_motion_mode()
        self.sink.send(command)

    def publish(self, command: np.ndarray) -> None:
        self.publish_selected(command, source="pi05")


class PikaTeleopTriggerController:
    def __init__(
        self,
        rospy_module: Any,
        trigger_type: Any,
        ctrl_pose_type: Any,
        *,
        service_name: str = "/teleop_trigger",
        ctrl_pose_topic: str = "/piper_IK/ctrl_end_pose",
        teleop_status_type: Any | None = None,
        teleop_status_topic: str = "/teleop_status",
        observed_freshness_s: float = 0.2,
    ):
        self.rospy = rospy_module
        self.trigger_type = trigger_type
        self.service_name = service_name
        self.observed_freshness_s = float(observed_freshness_s)
        self._last_ctrl_pose_s = 0.0
        self._proxy = None
        self._desired_active = False
        self._status_seen = False
        self._status_active = False
        self.rospy.Subscriber(ctrl_pose_topic, ctrl_pose_type, self._ctrl_pose_callback, queue_size=1)
        if teleop_status_type is not None:
            self.rospy.Subscriber(
                teleop_status_topic,
                teleop_status_type,
                self._teleop_status_callback,
                queue_size=1,
            )

    def _ctrl_pose_callback(self, _msg: Any) -> None:
        self._last_ctrl_pose_s = time.monotonic()

    def _teleop_status_callback(self, msg: Any) -> None:
        self._status_seen = True
        self._status_active = not bool(getattr(msg, "quit", False)) and not bool(getattr(msg, "fail", False))

    def _observed_active(self) -> bool:
        if self._status_seen:
            return self._status_active
        return (time.monotonic() - self._last_ctrl_pose_s) <= self.observed_freshness_s

    def observed_active(self) -> bool:
        return self._observed_active()

    def set_active(self, active: bool) -> None:
        desired = bool(active)
        if self.observed_active() == desired:
            self._desired_active = desired
            return
        if self._proxy is None:
            self.rospy.wait_for_service(self.service_name, timeout=2.0)
            self._proxy = self.rospy.ServiceProxy(self.service_name, self.trigger_type)
        self._proxy()
        self._desired_active = desired


def run(config: TakeoverRuntimeConfig) -> dict[str, object]:
    config = config.validate()
    if config.dry_import_check:
        return {
            "outcome": "dry_import_check",
            "control_hz": config.control_hz,
            "action_dim": config.action_dim,
            "selected_command_topic": config.selected_command_topic,
            "model_safety_profile": config.model_safety_profile,
            "hardware_io": config.hardware_io,
            "model_execute_steps": config.model_execute_steps,
            "model_prefetch_lead_steps": config.model_prefetch_lead_steps,
            "human_end_timeout_s": config.human_end_timeout_s,
            "phase_classifier_checkpoint": None
            if config.phase_classifier_checkpoint is None
            else str(config.phase_classifier_checkpoint),
            "phase_enter_threshold": config.phase_enter_threshold,
            "phase_enter_frames": config.phase_enter_frames,
            "phase_classifier_period": config.phase_classifier_period,
            "phase_classifier_device": config.phase_classifier_device,
            "actor_shadow": config.actor_shadow,
            "actor_live": config.actor_live,
            "actor_live_max_chunks": 0 if config.actor_live_max_chunks is None else config.actor_live_max_chunks,
            "actor_shadow_expected_z_dim": config.actor_shadow_expected_z_dim,
            "actor_shadow_max_latency_s": config.actor_shadow_max_latency_s,
        }

    return _run_live_ros(config)


def _run_live_ros(config: TakeoverRuntimeConfig) -> dict[str, Any]:
    rospy, JointState = require_ros_modules()
    from geometry_msgs.msg import PoseStamped
    from data_msgs.msg import TeleopStatus
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from std_msgs.msg import String
    from std_srvs.srv import SetBool, Trigger

    rospy.init_node("rlt_takeover_rollout", anonymous=True)
    feedback_tracker = FreshJointCommandTracker(action_dim=config.action_dim, freshness_s=config.freshness_s)
    arm_status_tracker = FreshArmStatusTracker()
    human_tracker = FreshJointCommandTracker(action_dim=config.action_dim, freshness_s=config.freshness_s)
    from piper_msgs.msg import PiperStatusMsg

    rospy.Subscriber(config.feedback_topic, JointState, lambda msg: feedback_tracker.update(msg), queue_size=1)
    rospy.Subscriber("/arm_status", PiperStatusMsg, lambda msg: arm_status_tracker.update(msg), queue_size=1)
    ros_publisher = None
    command_publisher = None
    human_passthrough = None
    bridge_telemetry = None
    feedback_reader = RosFeedbackReader(feedback_tracker, arm_status_tracker)
    if config.hardware_io == "native_sdk" and config.publish_commands:
        bridge_telemetry = NativeSdkBridgeTelemetryTracker()
        rospy.Subscriber(
            NATIVE_SDK_STATUS_TOPIC,
            String,
            bridge_telemetry.update,
            queue_size=1,
            tcp_nodelay=True,
        )
        ros_publisher = rospy.Publisher(NATIVE_SDK_COMMAND_TOPIC, JointState, queue_size=1)
        human_direct_publisher = rospy.Publisher(config.selected_command_topic, JointState, queue_size=1)
        deadline = time.monotonic() + 5.0
        while (
            (ros_publisher.get_num_connections() < 1 or human_direct_publisher.get_num_connections() < 1)
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        if ros_publisher.get_num_connections() < 1:
            raise RuntimeError("persistent native SDK command bridge is not connected")
        if human_direct_publisher.get_num_connections() < 1:
            raise RuntimeError("official Piper ROS controller is not connected for native Pika pass-through")
        try:
            rospy.wait_for_service(NATIVE_SDK_PASSTHROUGH_SERVICE, timeout=5.0)
        except Exception as exc:
            raise RuntimeError("native SDK bridge passthrough arbitration service is unavailable") from exc
        passthrough_service = rospy.ServiceProxy(
            NATIVE_SDK_PASSTHROUGH_SERVICE,
            SetBool,
            persistent=True,
        )
        human_passthrough = NativePikaJointStatePassthrough(human_direct_publisher)
        print(
            f"[{config.episode_id}] native Pika pass-through ready: original JointState stream -> "
            f"{config.selected_command_topic}; RLT replay remains {config.control_hz:.0f} Hz",
            flush=True,
        )
        command_publisher = RosNativeSdkCommandPublisher(
            ros_publisher,
            rospy,
            action_dim=config.action_dim,
            human_passthrough=human_passthrough,
            passthrough_service=passthrough_service,
            bridge_telemetry=bridge_telemetry,
        )
        command_publisher.set_external_passthrough_active(False)
    elif config.publish_commands:
        ros_publisher = rospy.Publisher(config.selected_command_topic, JointState, queue_size=1)
        deadline = time.monotonic() + 5.0
        while ros_publisher.get_num_connections() < 1 and time.monotonic() < deadline:
            time.sleep(0.05)
        if ros_publisher.get_num_connections() < 1:
            raise RuntimeError("official Piper ROS controller is not connected")
        human_passthrough = NativePikaJointStatePassthrough(ros_publisher)
        command_publisher = RosArbitratedCommandPublisher(
            ros_publisher,
            action_dim=config.action_dim,
            human_passthrough=human_passthrough,
            rospy_module=rospy,
            input_hz=config.control_hz,
            output_hz=50.0,
        )
        print(
            f"[{config.episode_id}] single-owner ROS arbitration ready: model/reset -> "
            f"{config.selected_command_topic}; accepted Pika remains native-rate pass-through",
            flush=True,
        )
    def on_human_command(message: Any) -> None:
        human_tracker.update(message)
        if human_passthrough is not None:
            human_passthrough.forward(message)

    rospy.Subscriber(
        config.human_command_topic,
        JointState,
        on_human_command,
        queue_size=1,
        tcp_nodelay=True,
    )
    teleop_controller = PikaTeleopTriggerController(
        rospy,
        Trigger,
        PoseStamped,
        teleop_status_type=TeleopStatus,
    )

    base_policy = WebsocketClientPolicy(
        config.policy_host,
        config.policy_port,
        connect_timeout_s=30.0,
        request_timeout_s=5.0,
    )
    policy_worker = RLTPolicyWorker(
        policy_fn=base_policy.infer,
        worker_lane="base",
    )
    enrichment_policy_worker = None
    if config.actor_shadow:
        enrichment_policy = WebsocketClientPolicy(
            config.policy_host,
            config.policy_port,
            connect_timeout_s=30.0,
            request_timeout_s=5.0,
        )
        enrichment_policy_worker = RLTPolicyWorker(
            policy_fn=enrichment_policy.infer,
            worker_lane="enrichment",
        )
    episode_root = config.output_dir / config.episode_id
    logger_path = episode_root / "episode.jsonl"
    max_steps = config.max_steps or int(round(config.duration_s * config.control_hz))
    cameras = DualRealSenseReader()
    policy_worker.start()
    if enrichment_policy_worker is not None:
        enrichment_policy_worker.start()
    try:
        print(f"[{config.episode_id}] starting dual cameras and warming up image streams", flush=True)
        cameras.start()
        print(
            f"[{config.episode_id}] waiting for fresh Piper feedback with err_code=0 stable for 0.8s",
            flush=True,
        )
        initial_state = _wait_for_initial_feedback(feedback_reader, timeout_s=12.0, stable_s=0.8)
        print(
            f"[{config.episode_id}] hardware feedback stable; running camera warmup and first policy inference "
            "(no command is published until the first plan is ready)",
            flush=True,
        )
        safety_filter = (
            StatefulSafetyFilter(
                HardwareSafetyConfig(
                    model_smoothing_tau_s=config.model_smoothing_tau_s,
                    model_max_joint_step=math.radians(config.model_max_joint_step_deg),
                    model_max_gripper_step=config.model_max_gripper_step,
                ),
                initial_state,
            )
            if config.publish_commands
            else None
        )
        phase_gate = None
        phase_classifier = None
        if config.phase_classifier_checkpoint is not None:
            phase_gate = SingleLatchPhaseGate(
                PhaseGateConfig(
                    enter_threshold=config.phase_enter_threshold,
                    enter_consecutive_frames=config.phase_enter_frames,
                    classifier_period=config.phase_classifier_period,
                )
            )
            phase_classifier = TorchPhaseClassifier(
                config.phase_classifier_checkpoint,
                device=config.phase_classifier_device,
            )
        scripted_schedule = parse_scripted_keys(config.scripted_keys)
        key_context = _key_source_context(ScriptedKeySource(scripted_schedule) if scripted_schedule else NonBlockingTerminalKeySource())
        keyboard = RLTKeyboardStateMachine(toggle_debounce_s=0.25)
        if config.start_human:
            keyboard.press("s", now_s=time.monotonic())
        with key_context as key_source, RLTEpisodeLogger(logger_path, episode_id=config.episode_id) as logger:
            core = TakeoverLoopCore(
                config=config,
                cameras=cameras,
                feedback_reader=feedback_reader,
                human_tracker=human_tracker,
                policy_worker=policy_worker,
                keyboard=keyboard,
                key_source=key_source,
                image_writer=EpisodeImageWriter(episode_root),
                logger=logger,
                command_publisher=command_publisher,
                teleop_controller=teleop_controller,
                human_passthrough=human_passthrough,
                safety_filter=safety_filter,
                phase_gate=phase_gate,
                phase_classifier=phase_classifier,
                enrichment_policy_worker=enrichment_policy_worker,
            )
            result = core.run_steps(max_steps=max_steps)
    finally:
        if human_passthrough is not None:
            human_passthrough.set_active(False)
        if command_publisher is not None:
            set_external_passthrough = getattr(
                command_publisher, "set_external_passthrough_active", None
            )
            if set_external_passthrough is not None:
                try:
                    set_external_passthrough(False)
                except Exception:
                    pass
            close_publisher = getattr(command_publisher, "close", None)
            if close_publisher is not None:
                try:
                    close_publisher()
                except Exception:
                    pass
        policy_worker.stop(timeout_s=2.0)
        if enrichment_policy_worker is not None:
            enrichment_policy_worker.stop(timeout_s=2.0)
        cameras.stop()
    report_path = episode_root / "report.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**result, "episode_root": str(episode_root), "logger_path": str(logger_path), "report_path": str(report_path)}


class _key_source_context:
    def __init__(self, key_source: Any):
        self.key_source = key_source

    def __enter__(self) -> Any:
        enter = getattr(self.key_source, "__enter__", None)
        if enter is None:
            return self.key_source
        return enter()

    def __exit__(self, exc_type, exc, tb) -> None:
        exit_fn = getattr(self.key_source, "__exit__", None)
        if exit_fn is not None:
            exit_fn(exc_type, exc, tb)


def _wait_for_initial_feedback(
    feedback_reader: RosFeedbackReader,
    *,
    timeout_s: float,
    stable_s: float = 0.8,
) -> np.ndarray:
    wait_until_healthy = getattr(feedback_reader, "wait_until_healthy", None)
    if wait_until_healthy is not None:
        return np.asarray(wait_until_healthy(timeout_s=timeout_s), dtype=np.float32)
    if stable_s < 0:
        raise ValueError("stable_s must be non-negative")
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    healthy_since: float | None = None
    latest_state: np.ndarray | None = None
    while time.monotonic() < deadline:
        try:
            latest_state = np.asarray(feedback_reader.read(), dtype=np.float32)
            now_s = time.monotonic()
            if healthy_since is None:
                healthy_since = now_s
            if now_s - healthy_since >= stable_s:
                return latest_state
        except Exception as exc:
            last_error = exc
            healthy_since = None
        time.sleep(0.05)
    raise RuntimeError(f"timed out waiting for fresh Piper feedback: {last_error}")


def main() -> None:
    parser = build_arg_parser()
    config = config_from_args(parser.parse_args())
    result = run(config)
    print(result)


if __name__ == "__main__":
    main()
