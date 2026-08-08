from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from openpi.rlt.real.config import HUMAN_EXECUTION_PROFILE
from openpi.rlt.real.config import GRIPPER_RESIDUAL_CLOSE_ASSIST
from openpi.rlt.real.config import GRIPPER_RESIDUAL_FROZEN
from openpi.rlt.real.config import PERSISTENT_ACTOR_EXECUTION_PROFILE
from openpi.rlt.real.config import PERSISTENT_ACTION_SCHEMA_FINGERPRINT
from openpi.rlt.real.config import PERSISTENT_EXECUTION_FILTER_PROFILE
from openpi.rlt.real.config import PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT
from openpi.rlt.real.config import Source
from openpi.rlt.real.replay import RealStepRecord


class ExternalEpisodeError(ValueError):
    """Raised when an external real-robot episode violates the RLT data contract."""


@dataclasses.dataclass(frozen=True)
class ExternalEpisodeContract:
    state_dim: int = 7
    action_dim: int = 7
    chunk_length: int = 10
    require_images: bool = True
    check_image_exists: bool = True
    require_terminal: bool = True
    z_rl_dim: int | None = None
    execution_profile: str | None = None
    action_schema_fingerprint: str | None = None
    require_committed_execution: bool = False


_VALID_SOURCES = {
    Source.PI05,
    Source.RLT,
    Source.HUMAN_PIKA,
    Source.STOP,
    Source.SAFETY_BLOCK,
}


def load_episode_jsonl(
    path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    contract: ExternalEpisodeContract | None = None,
) -> list[RealStepRecord]:
    contract = contract or ExternalEpisodeContract()
    path = Path(path)
    root = Path(dataset_root) if dataset_root is not None else path.parent
    rows = _read_jsonl_rows(path)
    records = [_row_to_record(row, index=index, contract=contract) for index, row in enumerate(rows)]
    validate_episode_records(records, dataset_root=root, contract=contract)
    return records


def validate_episode_records(
    records: list[RealStepRecord],
    *,
    dataset_root: str | Path | None = None,
    contract: ExternalEpisodeContract | None = None,
) -> None:
    contract = contract or ExternalEpisodeContract()
    if not records:
        raise ExternalEpisodeError("episode is empty")

    episode_ids = {record.episode_id for record in records}
    if len(episode_ids) != 1:
        raise ExternalEpisodeError(f"episode_id must be constant within one JSONL file, got {sorted(episode_ids)}")

    previous_t: int | None = None
    done_indices: list[int] = []
    previous_timestamp: int | None = None
    for index, record in enumerate(records):
        if previous_t is not None and record.t <= previous_t:
            raise ExternalEpisodeError("timestep values must be strictly monotonic")
        previous_t = record.t

        if record.source not in _VALID_SOURCES:
            raise ExternalEpisodeError(f"unknown source: {record.source}")

        _require_vector(record.state, contract.state_dim, "state")
        _require_vector_or_chunk(record.a_ref, contract.action_dim, "a_ref")
        _require_vector(record.a_exec, contract.action_dim, "a_exec")
        if record.a_human is not None:
            _require_vector(record.a_human, contract.action_dim, "a_human")
        if record.a_actor is not None:
            _require_vector_or_chunk(record.a_actor, contract.action_dim, "a_actor")
        _require_z_rl(record.z_rl, contract)
        _validate_execution_evidence(record, contract)

        if contract.require_images:
            _require_image_ref(record.global_image, "global_image", dataset_root, contract)
            _require_image_ref(record.wrist_image, "wrist_image", dataset_root, contract)

        if record.timestamp_ns is not None:
            if previous_timestamp is not None and record.timestamp_ns < previous_timestamp:
                raise ExternalEpisodeError("timestamp_ns values must be monotonic")
            previous_timestamp = record.timestamp_ns

        if record.done:
            done_indices.append(index)
        elif float(record.reward) != 0.0:
            raise ExternalEpisodeError("nonzero reward is only allowed on a terminal row")

    if len(done_indices) > 1:
        raise ExternalEpisodeError("episode may contain at most one terminal row")
    if done_indices and done_indices[0] != len(records) - 1:
        raise ExternalEpisodeError("terminal row must be the final row in an episode")
    if contract.require_terminal and not done_indices:
        raise ExternalEpisodeError("episode must contain one terminal row")
    committed_schemas = {
        str(record.action_schema_fingerprint)
        for record in records
        if record.actor_execution_profile
        in {PERSISTENT_ACTOR_EXECUTION_PROFILE, HUMAN_EXECUTION_PROFILE}
        and record.actor_execution_committed is True
        and record.action_schema_fingerprint is not None
    }
    if len(committed_schemas) > 1:
        raise ExternalEpisodeError(
            "frozen-v2 and gripper-close action schemas cannot be mixed in "
            f"one episode: {sorted(committed_schemas)}"
        )
    if (
        contract.action_schema_fingerprint is not None
        and committed_schemas
        != {str(contract.action_schema_fingerprint)}
    ):
        raise ExternalEpisodeError(
            "episode action schema mismatch: "
            f"{sorted(committed_schemas)} != "
            f"{[str(contract.action_schema_fingerprint)]}"
        )


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ExternalEpisodeError(f"episode JSONL does not exist: {path}") from exc

    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ExternalEpisodeError(f"invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ExternalEpisodeError(f"line {line_number} must contain a JSON object")
        rows.append(row)
    if not rows:
        raise ExternalEpisodeError(f"episode JSONL is empty: {path}")
    return rows


def _row_to_record(row: dict[str, Any], *, index: int, contract: ExternalEpisodeContract) -> RealStepRecord:
    policy_metadata = row.get("policy_metadata") or {}
    keyboard = row.get("keyboard") or {}
    replay_include = policy_metadata.get("replay_include")
    # Older capture versions did not emit replay_include. Missing is treated as
    # eligible and still subject to source/wait filtering; explicit false is
    # always rejected by enrichment.
    if replay_include is None:
        replay_include = True
    mux_reason = str(policy_metadata.get("mux_reason", ""))
    return RealStepRecord(
        episode_id=str(_required(row, "episode_id")),
        t=int(_required(row, "t")),
        z_rl=_array(_required(row, "z_rl"), "z_rl"),
        state=_array(_required(row, "state"), "state"),
        a_ref=_array(_required(row, "a_ref"), "a_ref"),
        a_exec=_array(_required(row, "a_exec"), "a_exec"),
        a_human=_optional_array(row.get("a_human"), "a_human"),
        a_actor=_optional_array(row.get("a_actor"), "a_actor"),
        source=str(_required(row, "source")),
        reward=float(row.get("reward", 0.0)),
        done=bool(row.get("done", False)),
        phase_probability=float(row.get("phase_probability", 0.0)),
        gate_active=bool(row.get("gate_active", False)),
        global_image=_optional_str(row.get("global_image")),
        wrist_image=_optional_str(row.get("wrist_image")),
        timestamp_ns=_optional_int(row.get("timestamp_ns")),
        replay_include=bool(replay_include),
        waiting_for_reward=bool(keyboard.get("waiting_for_reward", False)) or mux_reason == "waiting_for_reward_hold",
        policy_plan_id=_optional_str(row.get("policy_plan_id", policy_metadata.get("policy_plan_id"))),
        policy_observation_t=_optional_int(
            row.get("policy_observation_t", policy_metadata.get("policy_observation_t"))
        ),
        plan_offset=_optional_int(row.get("plan_offset", policy_metadata.get("plan_offset"))),
        behavior_actor_checkpoint=_optional_str(
            row.get("behavior_actor_checkpoint", policy_metadata.get("behavior_actor_checkpoint"))
        ),
        actor_execution_profile=_optional_str(
            row.get("actor_execution_profile", policy_metadata.get("actor_execution_profile"))
        ),
        actor_execution_plan_id=_optional_str(
            row.get("actor_execution_plan_id", policy_metadata.get("actor_execution_plan_id"))
        ),
        actor_execution_plan_offset=_optional_int(
            row.get("actor_execution_plan_offset", policy_metadata.get("actor_execution_plan_offset"))
        ),
        actor_execution_committed=_optional_bool(
            row.get(
                "actor_execution_committed",
                row.get(
                    "actor_execution_committed_this_step",
                    policy_metadata.get(
                        "actor_execution_committed_this_step"
                    ),
                ),
            )
        ),
        actor_canonical_decision=_optional_array(
            row.get("actor_canonical_decision", policy_metadata.get("actor_canonical_decision")),
            "actor_canonical_decision",
        ),
        actor_persistent_carry_in=_optional_array(
            row.get("actor_persistent_carry_in", policy_metadata.get("actor_persistent_carry_in")),
            "actor_persistent_carry_in",
        ),
        actor_persistent_carry_out=_optional_array(
            row.get("actor_persistent_carry_out", policy_metadata.get("actor_persistent_carry_out")),
            "actor_persistent_carry_out",
        ),
        actor_persistent_previous_carry=_optional_array(
            row.get(
                "actor_persistent_previous_carry",
                policy_metadata.get("actor_persistent_previous_carry"),
            ),
            "actor_persistent_previous_carry",
        ),
        actor_persistent_planned_residual=_optional_array(
            _metadata_value(
                row,
                policy_metadata,
                "actor_persistent_planned_residual",
                "actor_governor_safe_residual_this_step",
                "actor_safe_residual",
            ),
            "actor_persistent_planned_residual",
        ),
        actor_execution_boundary_anchor=_optional_array(
            row.get(
                "actor_execution_boundary_anchor",
                policy_metadata.get("actor_execution_boundary_anchor"),
            ),
            "actor_execution_boundary_anchor",
        ),
        actor_filtered_base_action=_optional_array(
            row.get("actor_filtered_base_action", policy_metadata.get("actor_filtered_base_action")),
            "actor_filtered_base_action",
        ),
        actor_filtered_actual_action=_optional_array(
            row.get(
                "actor_filtered_actual_action",
                policy_metadata.get("actor_filtered_actual_action"),
            ),
            "actor_filtered_actual_action",
        ),
        actor_filtered_actual_residual=_optional_array(
            row.get(
                "actor_filtered_actual_residual",
                policy_metadata.get("actor_filtered_actual_residual"),
            ),
            "actor_filtered_actual_residual",
        ),
        actor_execution_filter_tau_s=_optional_float(
            row.get(
                "actor_execution_filter_tau_s",
                policy_metadata.get("actor_execution_filter_tau_s"),
            )
        ),
        actor_execution_filter_dt_s=_optional_float(
            row.get(
                "actor_execution_filter_dt_s",
                policy_metadata.get("actor_execution_filter_dt_s"),
            )
        ),
        actor_execution_filter_alpha=_optional_float(
            row.get(
                "actor_execution_filter_alpha",
                policy_metadata.get("actor_execution_filter_alpha"),
            )
        ),
        actor_execution_projection_scale=_optional_float(
            row.get(
                "actor_execution_projection_scale",
                policy_metadata.get("actor_execution_projection_scale"),
            )
        ),
        action_schema_fingerprint=_optional_str(
            row.get("action_schema_fingerprint", policy_metadata.get("action_schema_fingerprint"))
        ),
        execution_filter_profile=_optional_str(
            row.get(
                "execution_filter_profile",
                policy_metadata.get("execution_filter_profile"),
            )
        ),
        safety_reasons=_string_tuple(
            row.get("safety_reasons", policy_metadata.get("safety_reasons", ()))
        ),
        actor_execution_residual_max_rad=_optional_float(
            row.get(
                "actor_execution_residual_max_rad",
                policy_metadata.get("actor_execution_residual_max_rad"),
            )
        ),
        actor_execution_d1_max_rad=_optional_float(
            row.get(
                "actor_execution_d1_max_rad",
                policy_metadata.get("actor_execution_d1_max_rad"),
            )
        ),
        actor_execution_d2_max_rad=_optional_float(
            row.get(
                "actor_execution_d2_max_rad",
                policy_metadata.get("actor_execution_d2_max_rad"),
            )
        ),
        actor_execution_direction_cone_deg=_optional_float(
            row.get(
                "actor_execution_direction_cone_deg",
                policy_metadata.get("actor_execution_direction_cone_deg"),
            )
        ),
        actor_execution_boundary_limit_rad=_optional_float(
            row.get(
                "actor_execution_boundary_limit_rad",
                policy_metadata.get("actor_execution_boundary_limit_rad"),
            )
        ),
        actor_execution_projection_scale_steps=_optional_int(
            row.get(
                "actor_execution_projection_scale_steps",
                policy_metadata.get("actor_execution_projection_scale_steps"),
            )
        ),
        actor_execution_min_projection_scale=_optional_float(
            row.get(
                "actor_execution_min_projection_scale",
                policy_metadata.get("actor_execution_min_projection_scale"),
            )
        ),
        actor_execution_direction_static_threshold_rad=_optional_float(
            row.get(
                "actor_execution_direction_static_threshold_rad",
                policy_metadata.get("actor_execution_direction_static_threshold_rad"),
            )
        ),
        actor_gripper_residual_mode=_optional_str(
            _metadata_value(
                row,
                policy_metadata,
                "actor_gripper_residual_mode",
                "gripper_residual_mode",
            )
        ),
        actor_gripper_release_intent=_optional_bool(
            _metadata_value(
                row,
                policy_metadata,
                "actor_gripper_release_intent",
            )
        ),
        execution_gripper_residual_max_close_m=_optional_float(
            _metadata_value(
                row,
                policy_metadata,
                "execution_gripper_residual_max_close_m",
                "actor_gripper_residual_max_m",
            )
        ),
        execution_gripper_d1_max_m=_optional_float(
            _metadata_value(
                row,
                policy_metadata,
                "execution_gripper_d1_max_m",
                "actor_gripper_residual_d1_max_m",
            )
        ),
        execution_gripper_d2_max_m=_optional_float(
            _metadata_value(
                row,
                policy_metadata,
                "execution_gripper_d2_max_m",
                "actor_gripper_residual_d2_max_m",
            )
        ),
        execution_gripper_boundary_limit_m=_optional_float(
            _metadata_value(
                row,
                policy_metadata,
                "execution_gripper_boundary_limit_m",
                "actor_gripper_boundary_jump_max_m",
            )
        ),
        execution_gripper_command_min_m=_optional_float(
            _metadata_value(
                row,
                policy_metadata,
                "execution_gripper_command_min_m",
                "actor_gripper_command_min_m",
            )
        ),
        execution_gripper_command_max_m=_optional_float(
            _metadata_value(
                row,
                policy_metadata,
                "execution_gripper_command_max_m",
                "actor_gripper_command_max_m",
            )
        ),
        execution_gripper_release_reference_m=_optional_float(
            _metadata_value(
                row,
                policy_metadata,
                "execution_gripper_release_reference_m",
                "actor_gripper_release_reference_m",
            )
        ),
        execution_gripper_release_delta_m=_optional_float(
            _metadata_value(
                row,
                policy_metadata,
                "execution_gripper_release_delta_m",
                "actor_gripper_release_delta_m",
            )
        ),
        actor_filtered_actual_gripper_residual_max=_optional_float(
            _metadata_value(
                row,
                policy_metadata,
                "actor_filtered_actual_gripper_residual_max",
            )
        ),
        actor_filtered_actual_gripper_residual_d1_max_m=_optional_float(
            _metadata_value(
                row,
                policy_metadata,
                "actor_filtered_actual_gripper_residual_d1_max_m",
            )
        ),
        actor_filtered_actual_gripper_residual_d2_max_m=_optional_float(
            _metadata_value(
                row,
                policy_metadata,
                "actor_filtered_actual_gripper_residual_d2_max_m",
            )
        ),
        actor_filtered_actual_gripper_boundary_jump_max_m=_optional_float(
            _metadata_value(
                row,
                policy_metadata,
                "actor_filtered_actual_gripper_boundary_jump_max_m",
            )
        ),
    )


def _required(row: dict[str, Any], key: str) -> Any:
    if key not in row:
        raise ExternalEpisodeError(f"missing required field: {key}")
    return row[key]


def _metadata_value(
    row: dict[str, Any],
    policy_metadata: dict[str, Any],
    *keys: str,
) -> Any:
    """Read one canonical value while accepting explicitly named log aliases."""

    for container in (row, policy_metadata):
        for key in keys:
            if key in container:
                return container[key]
    return None


def _array(value: Any, name: str) -> np.ndarray:
    try:
        return np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ExternalEpisodeError(f"{name} must be numeric") from exc


def _optional_array(value: Any, name: str) -> np.ndarray | None:
    if value is None:
        return None
    return _array(value, name)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ExternalEpisodeError(f"boolean execution evidence expected, got {value!r}")
    return value


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple)):
        raise ExternalEpisodeError(f"safety_reasons must be a list of strings, got {value!r}")
    return tuple(str(item) for item in value)


def _validate_execution_evidence(
    record: RealStepRecord,
    contract: ExternalEpisodeContract,
) -> None:
    profile = record.actor_execution_profile
    if contract.execution_profile is not None and profile not in {
        None,
        contract.execution_profile,
        HUMAN_EXECUTION_PROFILE,
    }:
        raise ExternalEpisodeError(
            "actor_execution_profile mismatch: "
            f"expected {contract.execution_profile!r}, got {profile!r}"
        )
    if profile not in {PERSISTENT_ACTOR_EXECUTION_PROFILE, HUMAN_EXECUTION_PROFILE}:
        if contract.require_committed_execution:
            raise ExternalEpisodeError("missing persistent-v2 actor_execution_profile")
        return
    # A rollout declares its session-wide execution profile on every row,
    # including Pi0.5 warmup, holds, safety blocks, and the terminal marker.
    # Those rows are intentionally not physical C10 evidence.  Only a row that
    # is both replay-eligible and explicitly committed may be required to carry
    # the full persistent execution proof below; the episode-level auditor
    # separately accepts only complete 0..9 committed plans.
    if not record.replay_include or record.actor_execution_committed is not True:
        if contract.require_committed_execution and record.replay_include:
            raise ExternalEpisodeError("persistent-v2 replay row is not physically committed")
        return
    # Preserve the frozen-v2 diagnostic order: malformed committed rows report
    # their first missing physical certificate before any schema-specific
    # checks.  This keeps old capture/audit tooling fully compatible.
    common_required_vectors = {
        "actor_canonical_decision": record.actor_canonical_decision,
        "actor_persistent_carry_in": record.actor_persistent_carry_in,
        "actor_persistent_carry_out": record.actor_persistent_carry_out,
        "actor_persistent_previous_carry": record.actor_persistent_previous_carry,
        "actor_execution_boundary_anchor": record.actor_execution_boundary_anchor,
        "actor_filtered_base_action": record.actor_filtered_base_action,
        "actor_filtered_actual_action": record.actor_filtered_actual_action,
        "actor_filtered_actual_residual": record.actor_filtered_actual_residual,
    }
    for name, value in common_required_vectors.items():
        if value is None:
            raise ExternalEpisodeError(
                f"missing required persistent-v2 field: {name}"
            )
        _require_vector(value, contract.action_dim, name)
        if not np.all(np.isfinite(value)):
            raise ExternalEpisodeError(f"{name} contains NaN or inf")
    action_schema = record.action_schema_fingerprint
    if action_schema not in {
        PERSISTENT_ACTION_SCHEMA_FINGERPRINT,
        PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT,
    }:
        raise ExternalEpisodeError(
            "unsupported persistent action_schema_fingerprint: "
            f"{action_schema!r}"
        )
    if (
        contract.action_schema_fingerprint is not None
        and action_schema != contract.action_schema_fingerprint
    ):
        raise ExternalEpisodeError(
            "persistent action_schema_fingerprint mismatch: "
            f"{action_schema!r} != {contract.action_schema_fingerprint!r}"
        )
    close_assist = (
        action_schema == PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT
    )
    if close_assist:
        if record.actor_gripper_residual_mode != GRIPPER_RESIDUAL_CLOSE_ASSIST:
            raise ExternalEpisodeError(
                "gripper-close schema requires "
                f"actor_gripper_residual_mode={GRIPPER_RESIDUAL_CLOSE_ASSIST!r}"
            )
        if not isinstance(record.actor_gripper_release_intent, bool):
            raise ExternalEpisodeError(
                "gripper-close schema requires boolean "
                "actor_gripper_release_intent"
            )
    elif record.actor_gripper_residual_mode not in {
        None,
        "",
        GRIPPER_RESIDUAL_FROZEN,
    }:
        raise ExternalEpisodeError(
            "frozen-v2 schema cannot declare close-assist gripper mode"
        )

    required_vectors = dict(common_required_vectors)
    if close_assist:
        required_vectors["actor_persistent_planned_residual"] = (
            record.actor_persistent_planned_residual
        )
    for name, value in required_vectors.items():
        if value is None:
            raise ExternalEpisodeError(f"missing required persistent-v2 field: {name}")
        _require_vector(value, contract.action_dim, name)
        if not np.all(np.isfinite(value)):
            raise ExternalEpisodeError(f"{name} contains NaN or inf")

    if not record.actor_execution_plan_id:
        raise ExternalEpisodeError("missing required persistent-v2 field: actor_execution_plan_id")
    if record.actor_execution_plan_offset is None or not (
        0 <= record.actor_execution_plan_offset < contract.chunk_length
    ):
        raise ExternalEpisodeError(
            "actor_execution_plan_offset must be within "
            f"[0, {contract.chunk_length}), got {record.actor_execution_plan_offset}"
        )
    if record.execution_filter_profile != PERSISTENT_EXECUTION_FILTER_PROFILE:
        raise ExternalEpisodeError(
            "persistent-v2 execution_filter_profile mismatch: "
            f"{record.execution_filter_profile!r}"
        )
    envelope_expected = {
        "actor_execution_residual_max_rad": 0.005,
        "actor_execution_d1_max_rad": 0.0015,
        "actor_execution_d2_max_rad": 0.001,
        "actor_execution_direction_cone_deg": 15.0,
        "actor_execution_boundary_limit_rad": 0.06,
        "actor_execution_projection_scale_steps": 33,
        "actor_execution_min_projection_scale": 0.2,
        "actor_execution_direction_static_threshold_rad": 0.001,
    }
    if close_assist:
        envelope_expected.update(
            {
                "execution_gripper_residual_max_close_m": 0.005,
                "execution_gripper_d1_max_m": 0.0005,
                "execution_gripper_d2_max_m": 0.0003,
                "execution_gripper_boundary_limit_m": 0.0005,
                "execution_gripper_command_min_m": 0.0,
                "execution_gripper_command_max_m": 0.08,
                "execution_gripper_release_reference_m": 0.05,
                "execution_gripper_release_delta_m": 0.002,
            }
        )
    for name, expected in envelope_expected.items():
        value = getattr(record, name)
        if value is None or not np.isclose(
            float(value), float(expected), rtol=0.0, atol=1e-12
        ):
            raise ExternalEpisodeError(
                f"persistent-v2 runtime envelope mismatch for {name}: "
                f"{value!r} != {expected!r}"
            )
    for name in (
        "actor_execution_filter_tau_s",
        "actor_execution_filter_dt_s",
        "actor_execution_filter_alpha",
        "actor_execution_projection_scale",
    ):
        value = getattr(record, name)
        if value is None or not np.isfinite(value):
            raise ExternalEpisodeError(f"missing or non-finite persistent-v2 field: {name}")
    if record.actor_execution_filter_tau_s <= 0.0 or record.actor_execution_filter_dt_s <= 0.0:
        raise ExternalEpisodeError("execution filter tau/dt must be positive")
    expected_alpha = 1.0 - np.exp(
        -record.actor_execution_filter_dt_s / record.actor_execution_filter_tau_s
    )
    if not np.isclose(record.actor_execution_filter_alpha, expected_alpha, rtol=2e-5, atol=1e-7):
        raise ExternalEpisodeError(
            "actor_execution_filter_alpha does not equal 1-exp(-dt/tau): "
            f"{record.actor_execution_filter_alpha} != {expected_alpha}"
        )
    if not 0.0 <= record.actor_execution_projection_scale <= 1.0:
        raise ExternalEpisodeError("actor_execution_projection_scale must be in [0, 1]")
    if not close_assist and abs(float(record.actor_canonical_decision[6])) > 1e-7:
        raise ExternalEpisodeError("persistent-v2 canonical gripper decision must be exactly frozen")
    # The Actor residual channel never controls the gripper: Pi0.5's absolute
    # gripper target passes through unchanged.  Human execution is different.
    # Its ``actual_residual`` is audit evidence for
    # ``human_actual - counterfactual_base`` and may therefore contain a
    # nonzero gripper difference when the operator opens or closes the Pika
    # gripper.  Rejecting that difference would discard the intervention data
    # while conflating it with an Actor command.
    if (
        profile == PERSISTENT_ACTOR_EXECUTION_PROFILE
        and not close_assist
        and abs(float(record.actor_filtered_actual_residual[6])) > 1e-7
    ):
        raise ExternalEpisodeError("persistent-v2 actual gripper residual must be exactly frozen")
    if not np.allclose(
        record.a_exec,
        record.actor_filtered_actual_action,
        rtol=0.0,
        atol=2e-6,
    ):
        raise ExternalEpisodeError("a_exec does not match the final filtered actual action")
    if profile == PERSISTENT_ACTOR_EXECUTION_PROFILE:
        if not np.allclose(
            record.actor_filtered_actual_action,
            record.actor_filtered_base_action + record.actor_filtered_actual_residual,
            rtol=0.0,
            atol=2e-6,
        ):
            raise ExternalEpisodeError(
                "actor_filtered_actual_action must equal filtered base plus actual residual"
            )
        if not np.allclose(
            record.actor_persistent_carry_out,
            record.actor_filtered_actual_residual,
            rtol=0.0,
            atol=2e-6,
        ):
            raise ExternalEpisodeError(
                "actor_persistent_carry_out must equal the committed actual residual"
            )
        if close_assist:
            _validate_close_assist_row(record)
    else:
        for name, value in (
            ("actor_canonical_decision", record.actor_canonical_decision),
            ("actor_persistent_carry_in", record.actor_persistent_carry_in),
            ("actor_persistent_carry_out", record.actor_persistent_carry_out),
            ("actor_persistent_previous_carry", record.actor_persistent_previous_carry),
        ):
            if not np.allclose(value, 0.0, rtol=0.0, atol=1e-7):
                raise ExternalEpisodeError(f"human execution requires zero {name}")
        if not np.allclose(
            record.actor_filtered_actual_action,
            record.actor_filtered_base_action + record.actor_filtered_actual_residual,
            rtol=0.0,
            atol=2e-6,
        ):
            raise ExternalEpisodeError(
                "human actual residual audit must equal actual minus filtered base"
            )
        if close_assist:
            # This field was populated from the shadow Actor before the
            # source-specific human metadata override in older rollouts.
            # It is not physical human execution evidence and is ignored for
            # HUMAN_EXECUTION_PROFILE.  New rollouts explicitly log zero.
            _validate_close_assist_row(record)


def _validate_close_assist_row(record: RealStepRecord) -> None:
    """Validate per-row evidence; C10 temporal certificates are checked later."""

    assert record.actor_persistent_planned_residual is not None
    planned_gripper = float(record.actor_persistent_planned_residual[6])
    actual_residual_gripper = float(record.actor_filtered_actual_residual[6])
    actual_gripper = float(record.actor_filtered_actual_action[6])
    close_limit = 0.005
    tolerance = 3e-6
    if not -close_limit - tolerance <= planned_gripper <= tolerance:
        raise ExternalEpisodeError(
            "planned close-assist gripper residual must stay in [-0.005, 0]"
        )
    if not -tolerance <= actual_gripper <= 0.08 + tolerance:
        raise ExternalEpisodeError(
            "executed gripper command leaves the certified [0, 0.08] m range"
        )
    if record.actor_execution_profile == PERSISTENT_ACTOR_EXECUTION_PROFILE:
        for name, value in (
            ("actor_canonical_decision", record.actor_canonical_decision[6]),
            ("actor_persistent_carry_in", record.actor_persistent_carry_in[6]),
            (
                "actor_persistent_previous_carry",
                record.actor_persistent_previous_carry[6],
            ),
            ("actor_persistent_carry_out", record.actor_persistent_carry_out[6]),
        ):
            scalar = float(value)
            if not -close_limit - tolerance <= scalar <= tolerance:
                raise ExternalEpisodeError(
                    f"{name} gripper residual leaves [-0.005, 0]"
                )
        if not np.isclose(
            actual_residual_gripper,
            planned_gripper,
            rtol=0.0,
            atol=tolerance,
        ):
            raise ExternalEpisodeError(
                "close-assist gripper residual must equal its planned knot; "
                "the joint low-pass must not filter it again"
            )
    certificate_limits = {
        "actor_filtered_actual_gripper_residual_max": 0.005,
        "actor_filtered_actual_gripper_residual_d1_max_m": 0.0005,
        "actor_filtered_actual_gripper_residual_d2_max_m": 0.0003,
        "actor_filtered_actual_gripper_boundary_jump_max_m": 0.0005,
    }
    for name, limit in certificate_limits.items():
        value = getattr(record, name)
        # Actor rows must carry the governor's execution-time certificate.
        # Human rows predate that Actor-only logger field; their certificate
        # is deterministically reconstructed from the complete physical C10
        # in replay.py and then serialized with the replay.
        if (
            value is None
            and record.actor_execution_profile == HUMAN_EXECUTION_PROFILE
        ):
            continue
        if value is None or not np.isfinite(value) or float(value) < 0.0:
            raise ExternalEpisodeError(
                f"missing or invalid gripper execution certificate: {name}"
            )
        # Human Pika commands are deliberately not constrained by the Actor
        # residual envelope.  Their certificate remains audit evidence and
        # may be much larger than the learned close-assist limit.
        if (
            record.actor_execution_profile
            == PERSISTENT_ACTOR_EXECUTION_PROFILE
            and float(value) > limit + tolerance
        ):
            raise ExternalEpisodeError(
                f"gripper execution certificate exceeds {name}={limit}"
            )
    if (
        record.actor_filtered_actual_gripper_residual_max is not None
        and
        float(record.actor_filtered_actual_gripper_residual_max)
        + tolerance
        < abs(actual_residual_gripper)
    ):
        raise ExternalEpisodeError(
            "gripper residual certificate is smaller than this executed row"
        )


def _require_vector(value: np.ndarray, expected_dim: int, name: str) -> None:
    if value.shape != (expected_dim,):
        raise ExternalEpisodeError(f"{name} must have shape ({expected_dim},), got {value.shape}")


def _require_vector_or_chunk(value: np.ndarray, expected_dim: int, name: str) -> None:
    if value.shape == (expected_dim,):
        return
    if value.ndim == 2 and value.shape[0] > 0 and value.shape[1] == expected_dim:
        return
    raise ExternalEpisodeError(f"{name} must have shape ({expected_dim},) or (N, {expected_dim}), got {value.shape}")


def _require_z_rl(value: np.ndarray, contract: ExternalEpisodeContract) -> None:
    if value.ndim != 1 or value.shape[0] == 0:
        raise ExternalEpisodeError(f"z_rl must be a non-empty 1-D vector, got {value.shape}")
    if contract.z_rl_dim is not None and value.shape != (contract.z_rl_dim,):
        raise ExternalEpisodeError(f"z_rl must have shape ({contract.z_rl_dim},), got {value.shape}")


def _require_image_ref(
    value: str | None,
    name: str,
    dataset_root: str | Path | None,
    contract: ExternalEpisodeContract,
) -> None:
    if not value:
        raise ExternalEpisodeError(f"missing required field: {name}")
    image_path = Path(value)
    if image_path.is_absolute() or ".." in image_path.parts:
        raise ExternalEpisodeError(f"{name} must be a relative path inside the dataset root: {value}")
    if contract.check_image_exists and dataset_root is not None:
        resolved = Path(dataset_root) / image_path
        if not resolved.is_file():
            raise ExternalEpisodeError(f"{name} file does not exist: {resolved}")
