"""Bounded, supervised real-hardware rollout for the Piper JAX policy."""

import argparse
import concurrent.futures
import json
import select
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from piper_runtime.hardware_control import InterpolatedPiperCommandSink
from piper_runtime.hardware_control import PiperCommandSink, StatefulSafetyFilter, make_hardware_safety_config
from piper_runtime.buffered_policy_control import H50HandoffRejected, prepare_h50_handoff_plan
from piper_runtime.observation import DEFAULT_PROMPT
from piper_runtime.observation import build_observation


AUTHORIZATION = "I_UNDERSTAND_POLICY_MOVES_ARM"
POLICY_CHUNK_STEPS = 50
RLT_ACTION_CHUNK_STEPS = 10
WINDOWED_C10_MODE = "committed_h50_windowed_c10"
INFERENCE_START_TARGET = (
    0.020019,
    0.152472,
    -0.228603,
    -0.020857,
    0.632856,
    0.0,
    0.06517,
)
INFERENCE_START_RESET_S = 4.0
INFERENCE_START_RESET_HZ = 50.0
# Four model frames (133 ms) cover the measured ~87-106 ms hot inference
# latency while reducing observation staleness versus the former five-frame
# lead.  The boundary keepalive below safely covers an occasional late result.
DEFAULT_H50_PREFETCH_LEAD_STEPS = 4
H50_BOUNDARY_KEEPALIVE_HZ = 30.0
H50_FALLBACK_FEEDBACK_HOLD_THRESHOLD_RAD = 0.03


class RolloutConfigurationError(ValueError):
    pass


def make_stop_handler():
    def _handle(signum, frame):
        del frame
        raise KeyboardInterrupt("received signal %d" % signum)

    return _handle


def _normalize_operator_label(value):
    if value is None:
        return None
    label = str(value).strip().lower()
    if label in {"1", "0", "e", "q"}:
        return label
    return "invalid"


class StdinOperatorLabelSource:
    def __init__(self, stream=None):
        self.stream = sys.stdin if stream is None else stream
        self._prompted = False

    def __call__(self):
        if not self._prompted:
            print(
                "[policy-rollout] operator controls episode end: "
                "type 1 + Enter for success, 0 + Enter for failure, e/q + Enter to stop",
                flush=True,
            )
            self._prompted = True
        try:
            readable, _, _ = select.select([self.stream], [], [], 0.0)
        except (OSError, ValueError):
            return None
        if not readable:
            return None
        return self.stream.readline()


@dataclass(frozen=True)
class RolloutConfig:
    duration_s: Optional[float]
    authorization: str
    control_hz: float = 30.0
    execute_steps: int = 1
    max_plans: Optional[int] = None
    operator_label_control: bool = False
    max_inference_s: float = 1.0
    prompt: str = DEFAULT_PROMPT
    safety_profile: str = "probe"
    h50_prefetch_lead_steps: int = DEFAULT_H50_PREFETCH_LEAD_STEPS

    def validate(self):
        if self.authorization != AUTHORIZATION:
            raise RolloutConfigurationError("explicit hardware authorization is required")
        if self.duration_s is None:
            if not self.operator_label_control:
                raise RolloutConfigurationError(
                    "duration is required unless operator_label_control is enabled"
                )
        elif self.duration_s <= 0 or self.duration_s > 600:
            raise RolloutConfigurationError("duration must be in (0, 600] seconds")
        if self.control_hz != 30.0:
            raise RolloutConfigurationError("hardware rollout is fixed at 30 Hz")
        if self.execute_steps < 1 or self.execute_steps > 50:
            raise RolloutConfigurationError("execute_steps must be between 1 and 50")
        if self.max_plans is not None and self.max_plans < 1:
            raise RolloutConfigurationError("max_plans must be positive when set")
        if self.safety_profile not in {"probe", "normal", "native"}:
            raise RolloutConfigurationError("unknown safety_profile: %s" % self.safety_profile)
        if self.h50_prefetch_lead_steps < 1 or self.h50_prefetch_lead_steps >= POLICY_CHUNK_STEPS:
            raise RolloutConfigurationError("h50_prefetch_lead_steps must be in [1, 49]")
        return self


def _validated_absolute_actions(response):
    try:
        actions = np.asarray(response["actions"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("policy response has no valid actions") from exc
    if actions.ndim != 2 or actions.shape[0] != 50 or actions.shape[1] < 7:
        raise RuntimeError("policy actions must have shape (50, >=7), got %r" % (actions.shape,))
    actions = actions[:, :7]
    if not np.all(np.isfinite(actions)):
        raise RuntimeError("policy actions contain non-finite values")
    return actions


def _committed_action_count(execute_steps):
    """Keep the proven H50 policy trajectory while exposing C=10 windows.

    In the RLT-compatible H10 mode, ``execute_steps`` is the Actor/control
    window, not the SFT replanning horizon.  The base policy therefore commits
    its complete 50-step prediction and that trajectory is consumed as five
    consecutive, non-overlapping C10 windows.  This preserves the approach,
    grasp, transport and release timing that was lost when an independent
    policy call replaced the trajectory every ten steps.
    """

    return POLICY_CHUNK_STEPS if int(execute_steps) == RLT_ACTION_CHUNK_STEPS else int(execute_steps)


def _c10_window(actions, action_index):
    values = np.asarray(actions, dtype=float)
    if values.shape != (POLICY_CHUNK_STEPS, 7):
        raise ValueError("C10 window source must have shape (50, 7)")
    start = (int(action_index) // RLT_ACTION_CHUNK_STEPS) * RLT_ACTION_CHUNK_STEPS
    end = start + RLT_ACTION_CHUNK_STEPS
    if start < 0 or end > len(values):
        raise IndexError("C10 window lies outside the committed H50 trajectory")
    return start, values[start:end].copy()


@dataclass(frozen=True)
class InferredPolicyPlan:
    actions: np.ndarray
    observation_state: np.ndarray
    inference_s: float
    observation_time_s: float
    completed_time_s: float


def _infer_policy_plan(
    *,
    cameras,
    policy,
    feedback_reader,
    prompt,
    max_inference_s,
    warmup_frames,
    now_fn,
):
    images = cameras.read(timeout_ms=5000, warmup_frames=int(warmup_frames))
    snapshot = np.asarray(feedback_reader.read(), dtype=float)
    if snapshot.shape != (7,) or not np.all(np.isfinite(snapshot)):
        raise RuntimeError("policy observation feedback must be finite with shape (7,)")
    observation = build_observation(images, snapshot, prompt=prompt)
    inference_started = now_fn()
    response = policy.infer(observation)
    completed = now_fn()
    inference_s = completed - inference_started
    if inference_s > max_inference_s:
        raise RuntimeError("policy inference exceeded %.1f seconds" % max_inference_s)
    return InferredPolicyPlan(
        actions=_validated_absolute_actions(response),
        observation_state=snapshot.copy(),
        inference_s=float(inference_s),
        observation_time_s=float(inference_started),
        completed_time_s=float(completed),
    )


def _await_policy_future_with_keepalive(
    future,
    *,
    sink,
    hold_target,
    timeout_s,
    keepalive_hz=H50_BOUNDARY_KEEPALIVE_HZ,
):
    """Wait for policy inference without creating a command-publication hole."""

    period_s = 1.0 / float(keepalive_hz)
    deadline = time.monotonic() + float(timeout_s)
    keepalive_count = 0
    while True:
        if future.done():
            return future.result(), keepalive_count
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0.0:
            raise concurrent.futures.TimeoutError()
        sink.send(np.asarray(hold_target, dtype=float))
        keepalive_count += 1
        try:
            return future.result(timeout=min(period_s, remaining_s)), keepalive_count
        except concurrent.futures.TimeoutError:
            pass


def _reset_ros_controller_to_inference_start(reset_publisher_type=None, reset_s=INFERENCE_START_RESET_S):
    """Move to and verify the single prescribed start before camera/model use."""

    reset_s = float(reset_s)
    if reset_s <= 0.0 or reset_s > 30.0:
        raise RolloutConfigurationError("reset_s must be in (0, 30] seconds")
    if reset_publisher_type is None:
        from piper_runtime.rlt_online_session import RosHomeResetPublisher

        reset_publisher_type = RosHomeResetPublisher
    print(
        "[policy-rollout] mandatory start reset: "
        f"target={list(INFERENCE_START_TARGET)}, duration={reset_s:.1f}s; "
        "keep the swept volume clear",
        flush=True,
    )
    publisher = reset_publisher_type(
        target=INFERENCE_START_TARGET,
        selected_command_topic="/rlt/selected_joint_command",
        hold_s=reset_s,
        hz=INFERENCE_START_RESET_HZ,
    )
    publisher.publish_home()
    print(
        "[policy-rollout] mandatory start reset converged in all 6 joints and the gripper; "
        "camera/model startup begins now",
        flush=True,
    )


def run_rollout(
    config: RolloutConfig,
    *,
    cameras,
    policy,
    feedback_reader,
    sink,
    now_fn=time.monotonic,
    sleep_fn=time.sleep,
    on_record=None,
    operator_label_source: Optional[Callable[[], Optional[str]]] = None,
):
    config.validate()
    dt = 1.0 / config.control_hz
    started = now_fn()
    deadline = None if config.duration_s is None else started + config.duration_s

    def deadline_reached():
        return deadline is not None and now_fn() >= deadline

    if hasattr(feedback_reader, "wait_until_healthy"):
        initial_state = np.asarray(feedback_reader.wait_until_healthy(timeout_s=5.0), dtype=float)
    else:
        initial_state = np.asarray(feedback_reader.read(), dtype=float)
    safety = StatefulSafetyFilter(make_hardware_safety_config(config.safety_profile), initial_state)
    records = []
    plan_count = 0
    step_count = 0
    operator_label = None
    operator_outcome = None
    windowed_c10 = config.execute_steps == RLT_ACTION_CHUNK_STEPS
    committed_action_count = _committed_action_count(config.execute_steps)
    # Only the pure H50 path is under evaluation.  The committed-H50/C10
    # compatibility path remains byte-for-byte sequential until RLT migration.
    asynchronous_h50 = config.execute_steps == POLICY_CHUNK_STEPS
    executor = (
        concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="h50-prefetch")
        if asynchronous_h50
        else None
    )
    pending_plan = None
    last_command = initial_state.copy()
    previous_command = initial_state.copy()
    try:
        current_plan = _infer_policy_plan(
            cameras=cameras,
            policy=policy,
            feedback_reader=feedback_reader,
            prompt=config.prompt,
            max_inference_s=config.max_inference_s,
            warmup_frames=60,
            now_fn=now_fn,
        )
        actions = current_plan.actions.copy()
        raw_actions = current_plan.actions.copy()
        plan_handoff = {
            "prefetch_hit": None,
            "boundary_wait_s": 0.0,
            "observation_age_at_handoff_s": 0.0,
            "joint_rebase_offset": np.zeros(6, dtype=float),
            "bridge_steps": 0,
            "action_start_index": 0,
            "prefetch_accepted": None,
            "prefetch_ready_at_boundary": None,
            "fallback_reason": None,
            "raw_boundary_jump_rad": 0.0,
            "bridge_excursion_rad": 0.0,
            "boundary_keepalive_count": 0,
            "bridge_target_correction_rad": 0.0,
            "bridge_target_correction_clipped": False,
            "fallback_tracking_error_rad": 0.0,
            "feedback_anchored_hold": False,
            "fallback_hold_target": initial_state.copy(),
        }
        while not deadline_reached() and operator_outcome is None:
            plan_count += 1
            if plan_count == 1:
                if windowed_c10:
                    print(
                        "[policy-rollout] first 50-step policy plan ready after "
                        f"{current_plan.inference_s:.3f}s; committing the proven H50 trajectory as "
                        "five consecutive C10 windows with asynchronous H50 standby; "
                        "command publication starts now",
                        flush=True,
                    )
                elif asynchronous_h50:
                    print(
                        "[policy-rollout] first 50-step policy plan ready after "
                        f"{current_plan.inference_s:.3f}s; asynchronous H50 standby is enabled; "
                        "command publication starts now",
                        flush=True,
                    )
                else:
                    print(
                        "[policy-rollout] first 50-step policy plan ready after "
                        f"{current_plan.inference_s:.3f}s; command publication starts now",
                        flush=True,
                    )
            prefetch_requested_at_s = None
            active_action_count = len(actions) if asynchronous_h50 else committed_action_count
            for action_index in range(active_action_count):
                if deadline_reached() or operator_outcome is not None:
                    break
                prefetch_requested_this_step = False
                remaining_including_current = active_action_count - action_index
                if (
                    asynchronous_h50
                    and (config.max_plans is None or plan_count < config.max_plans)
                    and pending_plan is None
                    and remaining_including_current <= config.h50_prefetch_lead_steps
                ):
                    prefetch_requested_at_s = now_fn()
                    pending_plan = executor.submit(
                        _infer_policy_plan,
                        cameras=cameras,
                        policy=policy,
                        feedback_reader=feedback_reader,
                        prompt=config.prompt,
                        max_inference_s=config.max_inference_s,
                        warmup_frames=1,
                        now_fn=now_fn,
                    )
                    prefetch_requested_this_step = True
                step_started = now_fn()
                feedback = np.asarray(feedback_reader.read(), dtype=float)
                if config.safety_profile == "native":
                    filtered = safety.filter_model_native(actions[action_index])
                else:
                    filtered = safety.filter(
                        actions[action_index],
                        snapshot=current_plan.observation_state,
                        feedback=feedback,
                        dt=dt,
                    )
                sink.send(filtered.command)
                previous_command = last_command.copy()
                last_command = np.asarray(filtered.command, dtype=float).copy()
                step_count += 1
                window_start = None
                window_ref = None
                if windowed_c10:
                    window_start, window_ref = _c10_window(actions, action_index)
                record = {
                        "event": "command",
                    "plan": plan_count,
                    "action_index": action_index,
                    "elapsed_s": now_fn() - started,
                    "inference_s": current_plan.inference_s,
                    "snapshot": current_plan.observation_state.tolist(),
                    "feedback": feedback.tolist(),
                    "policy_raw_target": raw_actions[action_index].tolist(),
                    "raw_target": actions[action_index].tolist(),
                    "command": filtered.command.tolist(),
                    "reasons": filtered.reasons,
                    "safety_profile": config.safety_profile,
                    "h50_prefetch_lead_steps": config.h50_prefetch_lead_steps,
                    "prefetch_requested_this_step": prefetch_requested_this_step,
                    "plan_prefetch_hit": plan_handoff["prefetch_hit"],
                    "plan_boundary_wait_s": plan_handoff["boundary_wait_s"],
                    "plan_observation_age_at_handoff_s": plan_handoff[
                        "observation_age_at_handoff_s"
                    ],
                    "plan_joint_rebase_offset": plan_handoff["joint_rebase_offset"].tolist(),
                    "plan_bridge_steps": plan_handoff["bridge_steps"],
                    "policy_action_index": plan_handoff["action_start_index"] + action_index,
                    "plan_prefetch_accepted": plan_handoff["prefetch_accepted"],
                    "plan_prefetch_ready_at_boundary": plan_handoff[
                        "prefetch_ready_at_boundary"
                    ],
                    "plan_prefetch_fallback_reason": plan_handoff["fallback_reason"],
                    "plan_boundary_raw_jump_rad": plan_handoff["raw_boundary_jump_rad"],
                    "plan_boundary_bridge_excursion_rad": plan_handoff[
                        "bridge_excursion_rad"
                    ],
                    "plan_boundary_keepalive_count": plan_handoff[
                        "boundary_keepalive_count"
                    ],
                    "plan_bridge_target_correction_rad": plan_handoff[
                        "bridge_target_correction_rad"
                    ],
                    "plan_bridge_target_correction_clipped": plan_handoff[
                        "bridge_target_correction_clipped"
                    ],
                    "plan_fallback_tracking_error_rad": plan_handoff[
                        "fallback_tracking_error_rad"
                    ],
                    "plan_feedback_anchored_hold": plan_handoff[
                        "feedback_anchored_hold"
                    ],
                    "plan_fallback_hold_target": plan_handoff[
                        "fallback_hold_target"
                    ].tolist(),
                }
                if windowed_c10:
                    window_offset = action_index - window_start
                    record.update(
                        {
                            "execution_mode": WINDOWED_C10_MODE,
                            "policy_chunk_steps": POLICY_CHUNK_STEPS,
                            "actor_chunk_steps": RLT_ACTION_CHUNK_STEPS,
                            "c10_window_index": window_start // RLT_ACTION_CHUNK_STEPS,
                            "c10_window_offset": window_offset,
                        }
                    )
                    # One exact reference record per window is sufficient for
                    # replay/Actor alignment without repeating 10x7 values on
                    # every 30 Hz command row.
                    if window_offset == 0:
                        record["a_ref_c10"] = window_ref.tolist()
                records.append(record)
                if on_record is not None:
                    on_record(record)
                if config.operator_label_control and operator_label_source is not None:
                    label = _normalize_operator_label(operator_label_source())
                    if label in {"1", "0"}:
                        operator_label = label
                        operator_outcome = "operator_labeled"
                        print(
                            f"[policy-rollout] operator labeled episode success={label}; holding now",
                            flush=True,
                        )
                    elif label in {"e", "q"}:
                        operator_outcome = "operator_stopped"
                        print("[policy-rollout] operator stopped episode without a label", flush=True)
                    elif label == "invalid":
                        print(
                            "[policy-rollout] invalid operator input; type 1, 0, e, or q then Enter",
                            flush=True,
                        )
                sleep_fn(max(0.0, dt - (now_fn() - step_started)))
            if deadline_reached() or operator_outcome is not None:
                break
            if config.max_plans is not None and plan_count >= config.max_plans:
                break
            if asynchronous_h50:
                if pending_plan is None:
                    prefetch_requested_at_s = now_fn()
                    pending_plan = executor.submit(
                        _infer_policy_plan,
                        cameras=cameras,
                        policy=policy,
                        feedback_reader=feedback_reader,
                        prompt=config.prompt,
                        max_inference_s=config.max_inference_s,
                        warmup_frames=1,
                        now_fn=now_fn,
                    )
                prefetch_ready = pending_plan.done()
                wait_started = now_fn()
                try:
                    next_plan, wait_keepalives = _await_policy_future_with_keepalive(
                        pending_plan,
                        sink=sink,
                        hold_target=last_command,
                        timeout_s=config.max_inference_s + 0.25,
                    )
                except concurrent.futures.TimeoutError as exc:
                    raise RuntimeError("prefetched H50 was not ready before the safety timeout") from exc
                pending_plan = None
                observation_age_s = max(0.0, now_fn() - next_plan.observation_time_s)
                try:
                    prepared = prepare_h50_handoff_plan(
                        next_plan.actions,
                        observation_state=next_plan.observation_state,
                        handoff_target=last_command,
                        previous_target=previous_command,
                        observation_age_s=observation_age_s,
                    )
                except H50HandoffRejected as rejected:
                    # Accuracy wins over continuity.  Hold at the last safe
                    # command at 30 Hz while obtaining a boundary-fresh
                    # observation.  This prevents load droop on fallback.
                    boundary_feedback = np.asarray(feedback_reader.read(), dtype=float)
                    if boundary_feedback.shape != (7,) or not np.all(np.isfinite(boundary_feedback)):
                        raise RuntimeError("fallback boundary feedback must be finite with shape (7,)")
                    fallback_tracking_error = float(
                        np.max(np.abs(boundary_feedback[:6] - last_command[:6]))
                    )
                    fallback_hold_target = last_command.copy()
                    feedback_anchored_hold = (
                        fallback_tracking_error
                        > H50_FALLBACK_FEEDBACK_HOLD_THRESHOLD_RAD
                    )
                    if feedback_anchored_hold:
                        fallback_hold_target[:6] = boundary_feedback[:6]
                        # The next fresh policy is observed at this physical
                        # pose.  Reset the pure-inference low-pass state to the
                        # same anchor instead of jumping back toward a stale
                        # command after the wait.
                        safety = StatefulSafetyFilter(
                            make_hardware_safety_config(config.safety_profile),
                            fallback_hold_target,
                        )
                    fresh_future = executor.submit(
                        _infer_policy_plan,
                        cameras=cameras,
                        policy=policy,
                        feedback_reader=feedback_reader,
                        prompt=config.prompt,
                        max_inference_s=config.max_inference_s,
                        warmup_frames=1,
                        now_fn=now_fn,
                    )
                    fresh_plan, fallback_keepalives = _await_policy_future_with_keepalive(
                        fresh_future,
                        sink=sink,
                        hold_target=fallback_hold_target,
                        timeout_s=config.max_inference_s + 0.25,
                    )
                    current_plan = fresh_plan
                    raw_actions = fresh_plan.actions.copy()
                    actions = fresh_plan.actions.copy()
                    if feedback_anchored_hold:
                        previous_command = fallback_hold_target.copy()
                        last_command = fallback_hold_target.copy()
                    plan_handoff = {
                        "prefetch_hit": False,
                        "boundary_wait_s": max(0.0, now_fn() - wait_started),
                        "observation_age_at_handoff_s": 0.0,
                        "joint_rebase_offset": np.zeros(6, dtype=float),
                        "bridge_steps": 0,
                        "action_start_index": 0,
                        "prefetch_accepted": False,
                        "prefetch_ready_at_boundary": bool(prefetch_ready),
                        "fallback_reason": str(rejected),
                        "prefetch_requested_at_s": prefetch_requested_at_s,
                        "raw_boundary_jump_rad": 0.0,
                        "bridge_excursion_rad": 0.0,
                        "boundary_keepalive_count": wait_keepalives + fallback_keepalives,
                        "bridge_target_correction_rad": 0.0,
                        "bridge_target_correction_clipped": False,
                        "fallback_tracking_error_rad": fallback_tracking_error,
                        "feedback_anchored_hold": feedback_anchored_hold,
                        "fallback_hold_target": fallback_hold_target.copy(),
                    }
                else:
                    current_plan = next_plan
                    raw_actions = next_plan.actions[prepared.action_start_index :].copy()
                    actions = prepared.actions.copy()
                    plan_handoff = {
                        "prefetch_hit": bool(prefetch_ready),
                        "boundary_wait_s": max(0.0, now_fn() - wait_started),
                        "observation_age_at_handoff_s": prepared.observation_age_s,
                        "joint_rebase_offset": prepared.joint_rebase_offset.copy(),
                        "bridge_steps": int(prepared.bridge_steps),
                        "action_start_index": int(prepared.action_start_index),
                        "prefetch_accepted": True,
                        "prefetch_ready_at_boundary": bool(prefetch_ready),
                        "fallback_reason": None,
                        "prefetch_requested_at_s": prefetch_requested_at_s,
                        "raw_boundary_jump_rad": prepared.raw_boundary_jump_rad,
                        "bridge_excursion_rad": prepared.bridge_excursion_rad,
                        "boundary_keepalive_count": wait_keepalives,
                        "bridge_target_correction_rad": prepared.bridge_target_correction_rad,
                        "bridge_target_correction_clipped": (
                            prepared.bridge_target_correction_clipped
                        ),
                        "fallback_tracking_error_rad": 0.0,
                        "feedback_anchored_hold": False,
                        "fallback_hold_target": last_command.copy(),
                    }
            else:
                current_plan = _infer_policy_plan(
                    cameras=cameras,
                    policy=policy,
                    feedback_reader=feedback_reader,
                    prompt=config.prompt,
                    max_inference_s=config.max_inference_s,
                    warmup_frames=1,
                    now_fn=now_fn,
                )
                actions = current_plan.actions.copy()
                raw_actions = current_plan.actions.copy()
                plan_handoff = {
                    "prefetch_hit": False,
                    "boundary_wait_s": current_plan.inference_s,
                    "observation_age_at_handoff_s": 0.0,
                    "joint_rebase_offset": np.zeros(6, dtype=float),
                    "bridge_steps": 0,
                    "action_start_index": 0,
                    "prefetch_accepted": False,
                    "prefetch_ready_at_boundary": False,
                    "fallback_reason": "sequential_mode",
                    "raw_boundary_jump_rad": 0.0,
                    "bridge_excursion_rad": 0.0,
                    "boundary_keepalive_count": 0,
                    "bridge_target_correction_rad": 0.0,
                    "bridge_target_correction_clipped": False,
                    "fallback_tracking_error_rad": 0.0,
                    "feedback_anchored_hold": False,
                    "fallback_hold_target": last_command.copy(),
                }
        if operator_outcome is not None:
            outcome = operator_outcome
        elif config.max_plans is not None and plan_count >= config.max_plans:
            outcome = "max_plans_complete"
        else:
            outcome = "duration_complete"
        return {
            "outcome": outcome,
            "elapsed_s": now_fn() - started,
            "plan_count": plan_count,
            "step_count": step_count,
            "operator_label": operator_label,
            "execution_mode": WINDOWED_C10_MODE if windowed_c10 else "legacy_execute_horizon",
            "records": records,
        }
    except Exception as exc:
        exc.partial_rollout_result = {
            "outcome": "stopped",
            "elapsed_s": now_fn() - started,
            "plan_count": plan_count,
            "step_count": step_count,
            "operator_label": operator_label,
            "execution_mode": WINDOWED_C10_MODE if windowed_c10 else "legacy_execute_horizon",
            "records": records,
        }
        raise
    finally:
        try:
            sink.hold(feedback_reader)
        except Exception as hold_exc:
            if records:
                hold_record = {
                    "event": "hold_failed",
                    "elapsed_s": now_fn() - started,
                    "error": "%s: %s" % (type(hold_exc).__name__, hold_exc),
                }
                records.append(hold_record)
                if on_record is not None:
                    on_record(hold_record)
        if executor is not None:
            # Ubuntu 20.04/Python 3.8 has no cancel_futures argument.
            executor.shutdown(wait=True)


def _enable_with_timeout(piper, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while not piper.EnablePiper():
        if time.monotonic() >= deadline:
            raise RuntimeError("Piper enable timeout")
        time.sleep(0.01)


def main():
    parser = argparse.ArgumentParser(description="Authorized Piper policy hardware rollout")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--execute-steps", type=int, default=1)
    parser.add_argument(
        "--max-plans",
        type=int,
        default=None,
        help="Stop after this many policy plans. Use 1 for one reset -> one inference run.",
    )
    parser.add_argument(
        "--h50-prefetch-lead-steps",
        type=int,
        default=DEFAULT_H50_PREFETCH_LEAD_STEPS,
        help="Request the next H50 this many 30 Hz steps before the active H50 ends.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--safety-profile", default="probe", choices=["probe", "normal", "native"])
    parser.add_argument(
        "--hardware-io",
        default="ros_controller",
        choices=["ros_controller", "native_sdk"],
        help="Use the official Piper ROS controller by default; native_sdk is diagnostic only.",
    )
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--operator-label-control",
        action="store_true",
        help="Run until the operator types 1/0 + Enter; e/q stops without recording a label.",
    )
    parser.add_argument(
        "--reset-seconds",
        type=float,
        default=INFERENCE_START_RESET_S,
        help="Smooth reset duration before camera/model startup when using ros_controller.",
    )
    args = parser.parse_args()
    config = RolloutConfig(
        duration_s=args.duration,
        authorization=args.authorization,
        execute_steps=args.execute_steps,
        max_plans=args.max_plans,
        operator_label_control=bool(args.operator_label_control),
        prompt=args.prompt,
        safety_profile=args.safety_profile,
        h50_prefetch_lead_steps=args.h50_prefetch_lead_steps,
    ).validate()
    if args.reset_seconds <= 0.0 or args.reset_seconds > 30.0:
        raise RolloutConfigurationError("--reset-seconds must be in (0, 30]")
    stop_handler = make_stop_handler()
    for signal_name in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signal_name, stop_handler)

    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from piper_runtime.cameras import DualRealSenseReader

    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    cameras = DualRealSenseReader()
    piper = None
    native_sink = None
    if args.hardware_io == "ros_controller":
        # Keep the historical `PYTHONPATH=.` launch command working while
        # loading ROS Noetic and this catkin workspace from the pika venv.
        for ros_python_path in (
            Path.home() / "pika_ros/install/lib/python3/dist-packages",
            Path("/opt/ros/noetic/lib/python3/dist-packages"),
        ):
            rendered = str(ros_python_path)
            if rendered not in sys.path:
                sys.path.insert(0, rendered)
        import rospy
        from piper_msgs.msg import PiperStatusMsg
        from piper_msgs.srv import Enable
        from sensor_msgs.msg import JointState

        from piper_runtime.rlt_takeover_rollout import FreshArmStatusTracker
        from piper_runtime.rlt_takeover_rollout import RosFeedbackReader
        from piper_runtime.rlt_takeover_rollout import _wait_for_initial_feedback
        from piper_runtime.ros_command_io import FreshJointCommandTracker
        from piper_runtime.ros_command_io import vector_to_joint_state

        rospy.init_node("piper_policy_hardware_rollout", anonymous=True)
        feedback_tracker = FreshJointCommandTracker(action_dim=7, freshness_s=0.2)
        arm_status_tracker = FreshArmStatusTracker(freshness_s=0.5)
        rospy.Subscriber(
            "/joint_states_single",
            JointState,
            lambda msg: feedback_tracker.update(msg),
            queue_size=1,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            "/arm_status",
            PiperStatusMsg,
            lambda msg: arm_status_tracker.update(msg),
            queue_size=1,
            tcp_nodelay=True,
        )
        command_publisher = rospy.Publisher(
            "/rlt/selected_joint_command",
            JointState,
            queue_size=1,
        )

        class RosJointStateSink:
            def send(self, command):
                message = vector_to_joint_state(command, action_dim=7)
                message.header.stamp = rospy.Time.now()
                message.header.frame_id = "pi05"
                command_publisher.publish(message)

        feedback_reader = RosFeedbackReader(feedback_tracker, arm_status_tracker)
        sink = InterpolatedPiperCommandSink(
            RosJointStateSink(),
            input_hz=config.control_hz,
            output_hz=50.0,
        )
        rospy.wait_for_service("/enable_srv", timeout=8.0)
        enable_response = rospy.ServiceProxy("/enable_srv", Enable)(True)
        if not bool(getattr(enable_response, "enable_response", False)):
            raise RuntimeError("official Piper ROS controller failed to enable the arm")
        _wait_for_initial_feedback(feedback_reader, timeout_s=12.0, stable_s=0.8)
    else:
        from piper_sdk import C_PiperInterface_V2
        from piper_runtime.piper_feedback import PiperFeedbackReader

        piper = C_PiperInterface_V2("can0")
        piper.ConnectPort(False, False, True)
        feedback_reader = PiperFeedbackReader(piper)
        native_sink = PiperCommandSink(
            piper,
            move_speed_percent=make_hardware_safety_config(config.safety_profile).piper_move_speed_percent,
        )
        sink = InterpolatedPiperCommandSink(
            native_sink,
            input_hz=config.control_hz,
            output_hz=50.0,
        )
    result = None
    error = None
    start_reset_completed = False
    started_wall = time.time()
    audit_stream = args.audit.open("w", encoding="utf-8", buffering=1)

    def persist_record(record):
        audit_stream.write(json.dumps(record, sort_keys=True) + "\n")
        audit_stream.flush()

    try:
        if args.hardware_io == "native_sdk":
            feedback_reader.wait_until_healthy(timeout_s=5.0)
        else:
            # Both execute_steps=50 and windowed C10 must begin from the same
            # verified 7-D state.  Failure aborts before cameras or policy are
            # started; it is never treated as a warning.
            _reset_ros_controller_to_inference_start(reset_s=args.reset_seconds)
            start_reset_completed = True
        print(
            f"[policy-rollout] hardware path ready: {args.hardware_io}; starting dual cameras",
            flush=True,
        )
        cameras.start()
        print("[policy-rollout] dual cameras ready; connecting to policy server", flush=True)
        policy = WebsocketClientPolicy("127.0.0.1", 8000, connect_timeout_s=30.0, request_timeout_s=5.0)
        if args.hardware_io == "native_sdk":
            _enable_with_timeout(piper)
            native_sink.configure_motion_mode()
        result = run_rollout(
            config,
            cameras=cameras,
            policy=policy,
            feedback_reader=feedback_reader,
            sink=sink,
            operator_label_source=StdinOperatorLabelSource() if config.operator_label_control else None,
            on_record=persist_record,
        )
    except KeyboardInterrupt:
        error = "operator_interrupt"
    except Exception as exc:
        result = getattr(exc, "partial_rollout_result", result)
        error = "%s: %s" % (type(exc).__name__, exc)
        traceback.print_exc()
    finally:
        try:
            final_state = feedback_reader.read().astype(float).tolist()
        except Exception:
            final_state = None
        cameras.stop()
        sink.close()
        if piper is not None:
            piper.DisconnectPort()
        audit_stream.close()

    records = [] if result is None else result["records"]
    reasons = sorted({reason for record in records for reason in record["reasons"]})
    report = {
        "outcome": "stopped" if error is not None else (result["outcome"] if result is not None else "stopped"),
        "error": error,
        "requested_duration_s": config.duration_s,
        "execution_mode": None if result is None else result.get("execution_mode"),
        "start_reset_required": args.hardware_io == "ros_controller",
        "start_reset_completed": start_reset_completed,
        "start_reset_target": list(INFERENCE_START_TARGET),
        "start_reset_seconds": args.reset_seconds,
        "wall_elapsed_s": time.time() - started_wall,
        "plan_count": 0 if result is None else result["plan_count"],
        "max_plans": config.max_plans,
        "step_count": 0 if result is None else result["step_count"],
        "operator_label_control": config.operator_label_control,
        "operator_label": None if result is None else result.get("operator_label"),
        "h50_prefetch_lead_steps": config.h50_prefetch_lead_steps,
        "final_state": final_state,
        "filter_reasons": reasons,
        "last_record": records[-1] if records else None,
        "audit_path": str(args.audit),
        "hardware_io": args.hardware_io,
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if error is not None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
