"""Mock-only policy execution; this module has no hardware output path."""

import time
import uuid
from pathlib import Path

import numpy as np

from piper_delta_runner_v1.runner import DeltaRunner, MockEmitter, Plan, SafetyProfile
from piper_runtime.observation import absolute_actions_to_delta, build_observation


def make_mock_profile() -> SafetyProfile:
    return SafetyProfile(
        mode="mock_readonly",
        joint_min=np.full(6, -3.2, dtype=float),
        joint_max=np.full(6, 3.2, dtype=float),
        gripper_min=-0.1,
        gripper_max=0.1,
        max_delta=np.full(6, 0.05, dtype=float),
        max_velocity=np.full(6, 0.5, dtype=float),
        max_acceleration=np.full(6, 1.0, dtype=float),
        control_period_s=0.02,
        max_plan_age_s=2.0,
        watchdog_timeout_s=2.0,
        allow_hardware_execution=False,
        calibration_status="mock_only",
    )


def execute_dry_run(
    *,
    policy,
    images,
    state_snapshot,
    profile: SafetyProfile,
    audit_path: Path,
    now: float = None,
):
    if now is None:
        now = time.monotonic()
    snapshot = np.asarray(state_snapshot, dtype=np.float32).copy()
    observation = build_observation(images, snapshot)
    response = policy.infer(observation)
    delta_chunk = absolute_actions_to_delta(response, snapshot)
    plan = Plan(
        plan_id=str(uuid.uuid4()),
        q_feedback_snapshot=snapshot,
        predicted_delta_chunk=delta_chunk,
        safety_profile=profile.mode,
        planned_at=now,
    )
    result = DeltaRunner(profile, MockEmitter(feedback_joint=snapshot)).execute_plan(plan, now=now)
    DeltaRunner.write_audit_jsonl(Path(audit_path), result)
    return result
