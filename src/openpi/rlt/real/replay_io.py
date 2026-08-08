from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import numpy as np

from openpi.rlt.real.config import GRIPPER_RESIDUAL_CLOSE_ASSIST
from openpi.rlt.real.config import PERSISTENT_ACTION_SCHEMA_FINGERPRINT
from openpi.rlt.real.config import PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT
from openpi.rlt.real.config import PERSISTENT_GRIPPER_CLOSE_GOVERNOR_PROFILE
from openpi.rlt.real.config import PERSISTENT_GOVERNOR_PROFILE
from openpi.rlt.real.replay import RealTransition


def transitions_to_arrays(
    transitions: list[RealTransition],
    *,
    episode_split_by_id: Mapping[str, str] | None = None,
) -> dict[str, np.ndarray]:
    if not transitions:
        raise ValueError("cannot serialize an empty transition list")

    arrays = {
        "episode_id": _string_array([transition.episode_id for transition in transitions]),
        "t": np.asarray([transition.t for transition in transitions], dtype=np.int64),
        "z_rl": _stack("z_rl", [transition.z_rl for transition in transitions]),
        "state": _stack("state", [transition.state for transition in transitions]),
        "a_ref": _stack("a_ref", [transition.a_ref for transition in transitions]),
        "a_exec": _stack("a_exec", [transition.a_exec for transition in transitions]),
        "a_human": _stack("a_human", [transition.a_human for transition in transitions]),
        "a_actor": _stack("a_actor", [transition.a_actor for transition in transitions]),
        "source": _string_array([transition.source for transition in transitions]),
        "source_chunk": _stack_strings([transition.source_chunk for transition in transitions]),
        "human_mask": _stack_bool("human_mask", [transition.human_mask for transition in transitions]),
        "actor_mask": _stack_bool("actor_mask", [transition.actor_mask for transition in transitions]),
        "step_mask": _stack_bool("step_mask", [transition.step_mask for transition in transitions]),
        "a_ref_absolute": _stack("a_ref_absolute", [transition.a_ref_absolute for transition in transitions]),
        "a_ref_original_absolute": _stack(
            "a_ref_original_absolute", [transition.a_ref_original_absolute for transition in transitions]
        ),
        "a_exec_absolute": _stack("a_exec_absolute", [transition.a_exec_absolute for transition in transitions]),
        "a_human_absolute": _stack("a_human_absolute", [transition.a_human_absolute for transition in transitions]),
        "a_actor_absolute": _stack("a_actor_absolute", [transition.a_actor_absolute for transition in transitions]),
        "reward": np.asarray([transition.reward for transition in transitions], dtype=np.float32),
        "discount": np.asarray([transition.discount for transition in transitions], dtype=np.float32),
        "next_z_rl": _stack("next_z_rl", [transition.next_z_rl for transition in transitions]),
        "next_state": _stack("next_state", [transition.next_state for transition in transitions]),
        "next_a_ref": _stack("next_a_ref", [transition.next_a_ref for transition in transitions]),
        "next_a_ref_absolute": _stack(
            "next_a_ref_absolute", [transition.next_a_ref_absolute for transition in transitions]
        ),
        "done": np.asarray([transition.done for transition in transitions], dtype=np.bool_),
        "success_mask": np.asarray(
            [transition.success_mask for transition in transitions],
            dtype=np.bool_,
        ),
        "phase_probability": np.asarray([transition.phase_probability for transition in transitions], dtype=np.float32),
        "gate_active": np.asarray([transition.gate_active for transition in transitions], dtype=np.bool_),
        "policy_plan_id": _string_array([transition.policy_plan_id for transition in transitions]),
        "policy_observation_t": np.asarray(
            [transition.policy_observation_t for transition in transitions], dtype=np.int64
        ),
        "plan_offset": np.asarray([transition.plan_offset for transition in transitions], dtype=np.int64),
        "behavior_actor_checkpoint": _string_array(
            [transition.behavior_actor_checkpoint for transition in transitions]
        ),
    }
    persistent = [transition.actor_execution_profile is not None for transition in transitions]
    if any(persistent):
        if not all(persistent):
            raise ValueError("legacy and persistent-v2 transitions cannot be serialized into one replay")
        schema_values = {
            str(transition.action_schema_fingerprint)
            for transition in transitions
        }
        if not schema_values.issubset(
            {
                PERSISTENT_ACTION_SCHEMA_FINGERPRINT,
                PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT,
            }
        ):
            raise ValueError(
                "unsupported persistent action schema(s): "
                f"{sorted(schema_values)}"
            )
        if len(schema_values) != 1:
            raise ValueError(
                "frozen-v2 and gripper-close transitions cannot be serialized "
                f"into one replay: {sorted(schema_values)}"
            )
        close_assist = (
            schema_values
            == {PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT}
        )
        required = {
            "actor_execution_plan_offset": [t.actor_execution_plan_offset for t in transitions],
            "actor_canonical_decision": [t.actor_canonical_decision for t in transitions],
            "actor_persistent_carry_in": [t.actor_persistent_carry_in for t in transitions],
            "actor_persistent_carry_out": [t.actor_persistent_carry_out for t in transitions],
            "actor_persistent_previous_carry": [t.actor_persistent_previous_carry for t in transitions],
            "actor_execution_boundary_anchor": [t.actor_execution_boundary_anchor for t in transitions],
            "a_base_filtered": [t.a_base_filtered for t in transitions],
            "a_filtered_actual": [t.a_filtered_actual for t in transitions],
            "filtered_actual_residual": [t.filtered_actual_residual for t in transitions],
            "execution_filter_tau_s": [t.execution_filter_tau_s for t in transitions],
            "execution_filter_dt_s": [t.execution_filter_dt_s for t in transitions],
            "execution_filter_alpha": [t.execution_filter_alpha for t in transitions],
            "execution_projection_scale": [t.execution_projection_scale for t in transitions],
            "next_a_base_filtered": [t.next_a_base_filtered for t in transitions],
            "next_execution_filter_alpha": [t.next_execution_filter_alpha for t in transitions],
            "next_actor_persistent_previous_carry": [
                t.next_actor_persistent_previous_carry for t in transitions
            ],
            "next_actor_persistent_carry_in": [
                t.next_actor_persistent_carry_in for t in transitions
            ],
            "next_actor_execution_boundary_anchor": [
                t.next_actor_execution_boundary_anchor for t in transitions
            ],
            "execution_residual_max_rad": [t.execution_residual_max_rad for t in transitions],
            "execution_d1_max_rad": [t.execution_d1_max_rad for t in transitions],
            "execution_d2_max_rad": [t.execution_d2_max_rad for t in transitions],
            "execution_direction_cone_deg": [
                t.execution_direction_cone_deg for t in transitions
            ],
            "execution_boundary_limit_rad": [
                t.execution_boundary_limit_rad for t in transitions
            ],
            "execution_projection_scale_steps": [
                t.execution_projection_scale_steps for t in transitions
            ],
            "execution_min_projection_scale": [
                t.execution_min_projection_scale for t in transitions
            ],
            "execution_direction_static_threshold_rad": [
                t.execution_direction_static_threshold_rad for t in transitions
            ],
        }
        if close_assist:
            required.update(
                {
                    "actor_persistent_planned_residual": [
                        t.actor_persistent_planned_residual
                        for t in transitions
                    ],
                    "actor_gripper_release_intent": [
                        t.actor_gripper_release_intent for t in transitions
                    ],
                    "execution_gripper_residual_max_close_m": [
                        t.execution_gripper_residual_max_close_m
                        for t in transitions
                    ],
                    "execution_gripper_d1_max_m": [
                        t.execution_gripper_d1_max_m for t in transitions
                    ],
                    "execution_gripper_d2_max_m": [
                        t.execution_gripper_d2_max_m for t in transitions
                    ],
                    "execution_gripper_boundary_limit_m": [
                        t.execution_gripper_boundary_limit_m
                        for t in transitions
                    ],
                    "execution_gripper_command_min_m": [
                        t.execution_gripper_command_min_m
                        for t in transitions
                    ],
                    "execution_gripper_command_max_m": [
                        t.execution_gripper_command_max_m
                        for t in transitions
                    ],
                    "execution_gripper_release_reference_m": [
                        t.execution_gripper_release_reference_m
                        for t in transitions
                    ],
                    "execution_gripper_release_delta_m": [
                        t.execution_gripper_release_delta_m
                        for t in transitions
                    ],
                    "actor_filtered_actual_gripper_residual_max": [
                        t.actor_filtered_actual_gripper_residual_max
                        for t in transitions
                    ],
                    "actor_filtered_actual_gripper_residual_d1_max_m": [
                        t.actor_filtered_actual_gripper_residual_d1_max_m
                        for t in transitions
                    ],
                    "actor_filtered_actual_gripper_residual_d2_max_m": [
                        t.actor_filtered_actual_gripper_residual_d2_max_m
                        for t in transitions
                    ],
                    "actor_filtered_actual_gripper_boundary_jump_max_m": [
                        t.actor_filtered_actual_gripper_boundary_jump_max_m
                        for t in transitions
                    ],
                }
            )
        missing = sorted(name for name, values in required.items() if any(value is None for value in values))
        if missing:
            raise ValueError(f"persistent-v2 transitions are missing arrays: {missing}")
        arrays.update(
            {
                "actor_execution_profile": _string_array(
                    [str(t.actor_execution_profile) for t in transitions]
                ),
                "actor_execution_plan_id": _string_array(
                    [str(t.actor_execution_plan_id) for t in transitions]
                ),
                "action_schema_fingerprint": _string_array(
                    [str(t.action_schema_fingerprint) for t in transitions]
                ),
                "execution_filter_profile": _string_array(
                    [str(t.execution_filter_profile) for t in transitions]
                ),
                "terminal_reward_migration_steps": np.asarray(
                    [t.terminal_reward_migration_steps for t in transitions],
                    dtype=np.int64,
                ),
                "terminal_reward_migration_offset": np.asarray(
                    [t.terminal_reward_migration_offset for t in transitions],
                    dtype=np.int64,
                ),
            }
        )
        if close_assist:
            modes = {str(t.gripper_residual_mode) for t in transitions}
            if modes != {GRIPPER_RESIDUAL_CLOSE_ASSIST}:
                raise ValueError(
                    "gripper-close transitions require exactly one explicit "
                    f"residual mode, got {sorted(modes)}"
                )
            arrays["gripper_residual_mode"] = _string_array(
                [str(t.gripper_residual_mode) for t in transitions]
            )
            arrays["actor_governor_fingerprint"] = _string_array(
                [PERSISTENT_GRIPPER_CLOSE_GOVERNOR_PROFILE] * len(transitions)
            )
        for name, values in required.items():
            if name == "actor_execution_plan_offset":
                arrays[name] = np.stack(
                    [np.asarray(value, dtype=np.int64) for value in values]
                )
            elif name == "actor_gripper_release_intent":
                arrays[name] = _stack_bool(name, values)
            else:
                arrays[name] = _stack(name, values)
    if episode_split_by_id is not None:
        valid_splits = {"train", "validation", "test"}
        missing = sorted({transition.episode_id for transition in transitions}.difference(episode_split_by_id))
        if missing:
            raise ValueError(f"episode split is missing transition episodes: {missing}")
        values = [str(episode_split_by_id[transition.episode_id]) for transition in transitions]
        invalid = sorted(set(values).difference(valid_splits))
        if invalid:
            raise ValueError(f"unsupported episode split labels: {invalid}")
        arrays["episode_split"] = _string_array(values)
    return arrays


def write_replay_npz(
    transitions: list[RealTransition],
    output_dir: str | Path,
    *,
    replay_name: str = "replay.npz",
    manifest_name: str = "manifest.json",
    manifest_metadata: dict[str, Any] | None = None,
    episode_split_by_id: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = transitions_to_arrays(transitions, episode_split_by_id=episode_split_by_id)
    replay_path = output_dir / replay_name
    np.savez_compressed(replay_path, **arrays)

    report = _build_manifest(arrays, replay_name=replay_name)
    if manifest_metadata:
        report.update(manifest_metadata)
    manifest_path = output_dir / manifest_name
    manifest_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def _stack(name: str, values: list[np.ndarray]) -> np.ndarray:
    try:
        return np.stack([np.asarray(value, dtype=np.float32) for value in values]).astype(np.float32)
    except ValueError as exc:
        shapes = [tuple(np.asarray(value).shape) for value in values]
        raise ValueError(f"{name} values have inconsistent shapes: {shapes}") from exc


def _string_array(values: list[str]) -> np.ndarray:
    width = max(1, *(len(value) for value in values))
    return np.asarray(values, dtype=f"<U{width}")


def _stack_strings(values: list[np.ndarray]) -> np.ndarray:
    rows = [[str(item) for item in np.asarray(value).tolist()] for value in values]
    width = max(1, *(len(item) for row in rows for item in row))
    try:
        return np.asarray(rows, dtype=f"<U{width}")
    except ValueError as exc:
        raise ValueError("source_chunk values have inconsistent shapes") from exc


def _stack_bool(name: str, values: list[np.ndarray]) -> np.ndarray:
    try:
        return np.stack([np.asarray(value, dtype=np.bool_) for value in values]).astype(np.bool_)
    except ValueError as exc:
        shapes = [tuple(np.asarray(value).shape) for value in values]
        raise ValueError(f"{name} values have inconsistent shapes: {shapes}") from exc


def _build_manifest(arrays: dict[str, np.ndarray], *, replay_name: str) -> dict[str, Any]:
    episode_ids = arrays["episode_id"].tolist()
    sources = arrays["source"].tolist()
    persistent_schema = None
    if "action_schema_fingerprint" in arrays:
        schema_values = sorted(
            set(arrays["action_schema_fingerprint"].astype(str).tolist())
        )
        if len(schema_values) != 1:
            raise ValueError(
                "persistent replay must contain exactly one action schema"
            )
        persistent_schema = schema_values[0]
    close_assist = (
        persistent_schema
        == PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT
    )
    manifest = {
        "format": "openpi_real_rlt_replay_npz",
        "version": (
            4
            if close_assist
            else (3 if "actor_execution_profile" in arrays else 2)
        ),
        "transitions": int(len(episode_ids)),
        "episodes": int(len(set(episode_ids))),
        "successful_transitions": int(
            np.count_nonzero(arrays["success_mask"])
        ),
        "successful_episodes": int(
            len(
                {
                    episode_id
                    for episode_id, success
                    in zip(episode_ids, arrays["success_mask"].tolist())
                    if success
                }
            )
        ),
        "sources": sorted(set(sources)),
        "files": {"replay": replay_name},
        "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()},
    }
    if "actor_execution_profile" in arrays:
        profiles = sorted(set(arrays["actor_execution_profile"].astype(str).tolist()))
        schemas = sorted(set(arrays["action_schema_fingerprint"].astype(str).tolist()))
        filters = sorted(set(arrays["execution_filter_profile"].astype(str).tolist()))
        if len(schemas) != 1 or len(filters) != 1:
            raise ValueError("persistent-v2 replay must contain exactly one action schema and filter profile")
        if schemas[0] not in {
            PERSISTENT_ACTION_SCHEMA_FINGERPRINT,
            PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT,
        }:
            raise ValueError(
                f"unsupported persistent action schema: {schemas[0]!r}"
            )
        tau = np.asarray(arrays["execution_filter_tau_s"], dtype=np.float64)
        manifest["execution_contract"] = {
            "profiles": profiles,
            "action_schema_fingerprint": schemas[0],
            "execution_filter_profile": filters[0],
            "execution_filter_tau_s_min": float(np.min(tau)),
            "execution_filter_tau_s_max": float(np.max(tau)),
            "filter_alpha_source": (
                "per_physical_step_logged_and_verified_for_joints_only"
                if close_assist
                else "per_physical_step_logged_and_verified"
            ),
            "actor_candidate_action": (
                "a_base_filtered_plus_joint_governed_quintic_lowpass_and_"
                "gripper_governed_planned_carry_without_second_lowpass"
                if close_assist
                else "a_base_filtered_plus_governed_quintic_lowpass_residual"
            ),
            "critic_behavior_action": "a_exec_final_hardware_safety_output",
            "actor_governor_fingerprint": (
                PERSISTENT_GRIPPER_CLOSE_GOVERNOR_PROFILE
                if close_assist
                else PERSISTENT_GOVERNOR_PROFILE
            ),
            "governor": {
                "residual_max_rad": 0.005,
                "d1_max_rad": 0.0015,
                "d2_max_rad": 0.001,
                "direction_cone_deg": 15.0,
                "boundary_limit_rad": 0.06,
                "direction_static_threshold_rad": 0.001,
                "projection_scale_steps": 33,
                "min_projection_scale": 0.2,
            },
            "partial_terminal_rule": (
                "terminal_reward_moves_to_last_complete_committed_c10;"
                "reward_discount=gamma**physical_offset;bootstrap_discount=0;"
                "no_unexecuted_action_is_materialized"
            ),
            "terminal_reward_migrations": int(
                np.count_nonzero(arrays["terminal_reward_migration_steps"])
            ),
            "terminal_reward_migration_steps_max": int(
                np.max(arrays["terminal_reward_migration_steps"])
            ),
        }
        if close_assist:
            mode_values = sorted(
                set(arrays["gripper_residual_mode"].astype(str).tolist())
            )
            if mode_values != [GRIPPER_RESIDUAL_CLOSE_ASSIST]:
                raise ValueError(
                    "gripper-close replay mode mismatch: "
                    f"{mode_values}"
                )
            manifest["execution_contract"]["gripper"] = {
                "residual_mode": GRIPPER_RESIDUAL_CLOSE_ASSIST,
                "residual_max_close_m": 0.005,
                "d1_max_m_per_control_frame": 0.0005,
                "d2_max_m_per_control_frame2": 0.0003,
                "boundary_limit_m": 0.0005,
                "command_range_m": [0.0, 0.08],
                "release_reference_m": 0.05,
                "release_delta_m": 0.002,
                "planned_residual_source": (
                    "per_physical_step_logged_and_c10_reassembled"
                ),
                "filter_contract": (
                    "planned_gripper_residual_is_committed_directly_without_"
                    "joint_lowpass"
                ),
                "actual_residual_abs_max_observed_m": float(
                    np.max(
                        arrays[
                            "actor_filtered_actual_gripper_residual_max"
                        ]
                    )
                ),
                "actual_d1_abs_max_observed_m": float(
                    np.max(
                        arrays[
                            "actor_filtered_actual_gripper_residual_d1_max_m"
                        ]
                    )
                ),
                "actual_d2_abs_max_observed_m": float(
                    np.max(
                        arrays[
                            "actor_filtered_actual_gripper_residual_d2_max_m"
                        ]
                    )
                ),
                "actual_boundary_jump_abs_max_observed_m": float(
                    np.max(
                        arrays[
                            "actor_filtered_actual_gripper_boundary_jump_max_m"
                        ]
                    )
                ),
            }
    return manifest
