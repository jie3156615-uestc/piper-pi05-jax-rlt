#!/usr/bin/env python3
"""Fail-closed contracts for persistent C10 RLT with gripper close assistance."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ACTOR_EXECUTION_PROFILE = "persistent_c10_filtered_actual_v2"
HUMAN_EXECUTION_PROFILE = "human_pika_filtered_actual_v2"
STRICT_ACTOR_CANARY = "strict_actor_canary"
TRAINING_EPISODE_CONTRACT = "training_episode_contract"
ACTION_SCHEMA_FINGERPRINT = (
    "piper_joint_delta_v5_c10_n10_stride10_behavior_ref50_"
    "persistent_filtered_actual_r005_d1_0015_d2_001_cone15_"
    "gripper_close_assist_r005_d1_0005_d2_0003_boundary0005"
)
ACTOR_MODEL_ACTION_SCHEMA_FINGERPRINT = (
    "piper_joint_delta_v5_c10_n10_stride10_behavior_ref50_"
    "rank1_joint_r005_d1_0015_d2_001_cone15_"
    "gripper_close_knot_r005"
)
ACTOR_PROJECTION_PROFILE = (
    "rank1_joint_v1_r005_d1_0015_d2_001_cone15_"
    "scale33_min020_gripper_close_knot_r005"
)
EXECUTION_FILTER_PROFILE = "exp_one_minus_exp_neg_dt_over_tau_v1"
EXECUTION_FILTER_TAU_S = 0.05
CONTROL_HZ = 30.0
CONTROL_DT_S = 1.0 / CONTROL_HZ
EXECUTION_FILTER_ALPHA = 1.0 - math.exp(-CONTROL_DT_S / EXECUTION_FILTER_TAU_S)
CHUNK_LENGTH = 10
CHUNK_STRIDE = 10
ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD = 0.06
ACTOR_PROJECTION_SCALE_STEPS = 33
ACTOR_MIN_PROJECTION_SCALE = 0.2
ACTOR_DIRECTION_STATIC_THRESHOLD_RAD = 0.001
PERSISTENT_GOVERNOR_FINGERPRINT = (
    "persistent_governor_v3_joint_r005_d1_0015_d2_001_cone15_"
    "boundary060_gripper_close_r005_d1_0005_d2_0003_boundary0005"
)
GRIPPER_RESIDUAL_MODE = "close_only_persistent_v1"
GRIPPER_RESIDUAL_MAX_CLOSE_M = 0.005
GRIPPER_RESIDUAL_D1_MAX_M = 0.0005
GRIPPER_RESIDUAL_D2_MAX_M = 0.0003
GRIPPER_MAX_BOUNDARY_JUMP_M = 0.0005
GRIPPER_COMMAND_MIN_M = 0.0
GRIPPER_COMMAND_MAX_M = 0.08
GRIPPER_RELEASE_REFERENCE_M = 0.05
GRIPPER_RELEASE_DELTA_M = 0.002
DEFAULT_MIN_NEW_COMMITTED_EPISODES = 30
DEFAULT_MIN_PHASE_RLT_PUBLISHED_ROWS = 10
DEFAULT_MIN_PHYSICAL_COMMITTED_ROWS = 10
NONZERO_RESIDUAL_EPS_RAD = 1.0e-8


class PersistentV2ContractError(ValueError):
    """The trajectory cannot be admitted into persistent-v2 replay."""


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("policy_metadata")
    return value if isinstance(value, dict) else {}


def _finite_vector(
    value: Any,
    *,
    length: int,
    label: str,
    context: str,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise PersistentV2ContractError(
            f"{context}: {label} must be a finite vector with shape ({length},)"
        )
    return result


def _exact_float(
    metadata: dict[str, Any],
    key: str,
    expected: float,
    *,
    context: str,
    atol: float = 1.0e-12,
) -> None:
    try:
        actual = float(metadata[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise PersistentV2ContractError(
            f"{context}: missing/invalid {key!r}"
        ) from exc
    if not math.isfinite(actual) or not math.isclose(
        actual, expected, rel_tol=0.0, abs_tol=atol
    ):
        raise PersistentV2ContractError(
            f"{context}: {key} mismatch: {actual!r} != {expected!r}"
        )


def _require_row_binding(
    row: dict[str, Any],
    *,
    context: str,
    expected_execution_profile: str | None,
    expected_action_schema: str,
    expected_projection_profile: str,
    expected_filter_profile: str,
    expected_filter_tau_s: float,
    expected_control_hz: float,
) -> None:
    metadata = _metadata(row)
    exact_strings = {
        "action_schema_fingerprint": expected_action_schema,
        "actor_execution_schema_fingerprint": expected_action_schema,
        "actor_model_action_schema_fingerprint": (
            ACTOR_MODEL_ACTION_SCHEMA_FINGERPRINT
        ),
        "actor_projection_profile": expected_projection_profile,
        "execution_filter_profile": expected_filter_profile,
    }
    if expected_execution_profile is not None:
        exact_strings["actor_execution_profile"] = expected_execution_profile
    exact_strings["actor_governor_fingerprint"] = (
        PERSISTENT_GOVERNOR_FINGERPRINT
    )
    for key, expected in exact_strings.items():
        actual = metadata.get(key)
        if actual != expected:
            raise PersistentV2ContractError(
                f"{context}: {key} mismatch: {actual!r} != {expected!r}"
            )
    _exact_float(
        metadata,
        "model_smoothing_tau_s",
        expected_filter_tau_s,
        context=context,
    )
    _exact_float(
        metadata,
        "control_hz",
        expected_control_hz,
        context=context,
    )
    expected_alpha = 1.0 - math.exp(
        -(1.0 / expected_control_hz) / expected_filter_tau_s
    )
    _exact_float(
        metadata,
        "model_lowpass_alpha",
        expected_alpha,
        context=context,
        atol=1.0e-9,
    )
    _exact_float(
        metadata,
        "actor_execution_filter_tau_s",
        expected_filter_tau_s,
        context=context,
    )
    _exact_float(
        metadata,
        "actor_execution_filter_dt_s",
        1.0 / expected_control_hz,
        context=context,
        atol=1.0e-12,
    )
    _exact_float(
        metadata,
        "actor_execution_filter_alpha",
        expected_alpha,
        context=context,
        atol=1.0e-9,
    )
    _exact_float(
        metadata,
        "actor_live_boundary_jump_threshold_rad",
        ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD,
        context=context,
    )
    try:
        projection_scale_steps = int(metadata["actor_projection_scale_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PersistentV2ContractError(
            f"{context}: missing/invalid actor_projection_scale_steps"
        ) from exc
    if projection_scale_steps != ACTOR_PROJECTION_SCALE_STEPS:
        raise PersistentV2ContractError(
            f"{context}: actor_projection_scale_steps mismatch: "
            f"{projection_scale_steps} != {ACTOR_PROJECTION_SCALE_STEPS}"
        )
    _exact_float(
        metadata,
        "actor_min_projection_scale",
        ACTOR_MIN_PROJECTION_SCALE,
        context=context,
    )
    _exact_float(
        metadata,
        "actor_direction_static_threshold_rad",
        ACTOR_DIRECTION_STATIC_THRESHOLD_RAD,
        context=context,
    )
    for key, expected in (
        ("actor_execution_residual_max_rad", 0.005),
        ("actor_execution_d1_max_rad", 0.0015),
        ("actor_execution_d2_max_rad", 0.001),
        ("actor_execution_direction_cone_deg", 15.0),
        (
            "actor_execution_boundary_limit_rad",
            ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD,
        ),
        (
            "actor_execution_min_projection_scale",
            ACTOR_MIN_PROJECTION_SCALE,
        ),
        (
            "actor_execution_direction_static_threshold_rad",
            ACTOR_DIRECTION_STATIC_THRESHOLD_RAD,
        ),
    ):
        _exact_float(metadata, key, expected, context=context)
    try:
        execution_scale_steps = int(
            metadata["actor_execution_projection_scale_steps"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PersistentV2ContractError(
            f"{context}: missing/invalid actor_execution_projection_scale_steps"
        ) from exc
    if execution_scale_steps != ACTOR_PROJECTION_SCALE_STEPS:
        raise PersistentV2ContractError(
            f"{context}: actor_execution_projection_scale_steps mismatch: "
            f"{execution_scale_steps} != {ACTOR_PROJECTION_SCALE_STEPS}"
        )


def load_episode_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise PersistentV2ContractError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise PersistentV2ContractError(
                    f"{path}:{line_number}: row must be an object"
                )
            rows.append(row)
    return rows


def audit_persistent_v2_rows(
    rows: Iterable[dict[str, Any]],
    *,
    episode_id: str,
    expected_execution_profile: str = ACTOR_EXECUTION_PROFILE,
    expected_action_schema: str = ACTION_SCHEMA_FINGERPRINT,
    expected_projection_profile: str = ACTOR_PROJECTION_PROFILE,
    expected_filter_profile: str = EXECUTION_FILTER_PROFILE,
    expected_filter_tau_s: float = EXECUTION_FILTER_TAU_S,
    expected_control_hz: float = CONTROL_HZ,
    mode: str = TRAINING_EPISODE_CONTRACT,
    min_phase_rlt_published_rows: int = DEFAULT_MIN_PHASE_RLT_PUBLISHED_ROWS,
    min_physical_committed_rows: int = DEFAULT_MIN_PHYSICAL_COMMITTED_ROWS,
    require_complete_chunk: bool = True,
) -> dict[str, Any]:
    rows = list(rows)
    if mode not in {STRICT_ACTOR_CANARY, TRAINING_EPISODE_CONTRACT}:
        raise ValueError(f"unsupported persistent-v2 audit mode: {mode}")
    if (
        not rows
        or rows[-1].get("done") is not True
        or float(rows[-1].get("reward", -1.0)) not in {0.0, 1.0}
    ):
        raise PersistentV2ContractError(
            f"{episode_id}: terminal reward/done contract failed"
        )
    if min_phase_rlt_published_rows <= 0 or min_physical_committed_rows <= 0:
        raise ValueError("physical readiness thresholds must be positive")

    eligible_rows = 0
    phase_rlt_published_rows = 0
    actor_physical_committed_rows = 0
    human_physical_committed_rows = 0
    nonzero_actual_rows = 0
    nonzero_committed_rows = 0
    maximum_actual_residual_rad = 0.0
    maximum_committed_residual_rad = 0.0
    maximum_actual_gripper_close_m = 0.0
    maximum_committed_gripper_close_m = 0.0
    positive_gripper_residual_rows = 0
    gripper_release_rows = 0
    completed_counter_max = 0
    offsets_by_plan: dict[str, list[int]] = defaultdict(list)
    indices_by_plan: dict[str, list[int]] = defaultdict(list)
    t_by_plan: dict[str, list[int]] = defaultdict(list)
    kind_by_plan: dict[str, str] = {}
    invalid_plan_reasons: dict[str, list[str]] = defaultdict(list)
    bad_safety_reset_rows: list[int] = []
    disallowed_actor_safety_rows: list[dict[str, Any]] = []
    committed_row_indices: list[int] = []

    for row_index, row in enumerate(rows):
        metadata = _metadata(row)
        source = str(row.get("source", ""))
        include = bool(metadata.get("replay_include", False))
        if not include or source not in {"pi05", "rlt", "human_pika"}:
            continue
        eligible_rows += 1
        context = f"{episode_id} row={row_index} t={row.get('t')}"
        _require_row_binding(
            row,
            context=context,
            expected_execution_profile=None,
            expected_action_schema=expected_action_schema,
            expected_projection_profile=expected_projection_profile,
            expected_filter_profile=expected_filter_profile,
            expected_filter_tau_s=expected_filter_tau_s,
            expected_control_hz=expected_control_hz,
        )
        if source == "pi05":
            continue
        if source not in {"rlt", "human_pika"}:
            continue
        expected_source_profile = (
            expected_execution_profile
            if source == "rlt"
            else HUMAN_EXECUTION_PROFILE
        )
        if metadata.get("actor_execution_profile") != expected_source_profile:
            raise PersistentV2ContractError(
                f"{context}: source-specific execution profile mismatch: "
                f"{metadata.get('actor_execution_profile')!r} != "
                f"{expected_source_profile!r}"
            )
        published = bool(
            metadata.get("actor_command_delivery_mode") == "published"
            and metadata.get("actor_command_delivery_succeeded") is True
        )
        physical_committed = bool(
            published
            and metadata.get("actor_physical_execution_committed_this_step") is True
            and metadata.get("actor_execution_committed_this_step") is True
        )
        status = metadata.get("actor_persistent_commit_status")
        if status == "reset_after_safety_modification":
            bad_safety_reset_rows.append(row_index)
        if source == "rlt":
            published_phase_rlt = bool(row.get("gate_active") and published)
            phase_rlt_published_rows += int(published_phase_rlt)
            physical_committed = bool(
                physical_committed
                and published_phase_rlt
                and status == "committed_filtered_actual_after_publish"
            )
        else:
            if metadata.get("safety_profile") != "human_native":
                raise PersistentV2ContractError(
                    f"{context}: human row safety_profile must be human_native"
                )
            if (
                metadata.get("actor_execution_actual_filter_profile")
                != "human_native"
            ):
                raise PersistentV2ContractError(
                    f"{context}: human actual-filter profile must be human_native"
                )
            if metadata.get("human_base_counterfactual_valid") is not True:
                raise PersistentV2ContractError(
                    f"{context}: human_base_counterfactual_valid must be true"
                )
            for zero_key in (
                "actor_persistent_committed_residual",
                "actor_persistent_current_residual",
                "actor_persistent_previous_residual",
            ):
                zero_value = _finite_vector(
                    metadata.get(zero_key),
                    length=7,
                    label=zero_key,
                    context=context,
                )
                if np.any(np.abs(zero_value) > NONZERO_RESIDUAL_EPS_RAD):
                    raise PersistentV2ContractError(
                        f"{context}: human row has nonzero Actor carry in {zero_key}"
                    )
        if not physical_committed:
            continue
        committed_row_indices.append(row_index)
        if source == "rlt":
            actor_physical_committed_rows += 1
        else:
            human_physical_committed_rows += 1
        safety_reasons_raw = metadata.get("safety_reasons")
        if safety_reasons_raw in (None, "", []):
            safety_reasons: list[str] = []
        elif isinstance(safety_reasons_raw, str):
            safety_reasons = [safety_reasons_raw]
        elif isinstance(safety_reasons_raw, list) and all(
            isinstance(item, str) for item in safety_reasons_raw
        ):
            safety_reasons = list(safety_reasons_raw)
        else:
            raise PersistentV2ContractError(
                f"{context}: safety_reasons must be a string list"
            )
        disallowed_safety_reasons = (
            sorted(set(safety_reasons).difference({"model_low_pass"}))
            if source == "rlt"
            else []
        )
        if source == "rlt" and disallowed_safety_reasons:
            disallowed_actor_safety_rows.append(
                {
                    "row": row_index,
                    "reasons": disallowed_safety_reasons,
                }
            )

        if source == "rlt":
            if metadata.get("actor_gripper_residual_mode") != GRIPPER_RESIDUAL_MODE:
                raise PersistentV2ContractError(
                    f"{context}: actor_gripper_residual_mode mismatch"
                )
            if metadata.get("actor_filtered_actual_certificate_approved") is not True:
                raise PersistentV2ContractError(
                    f"{context}: filtered-actual certificate was not approved"
                )
            committed_residual = _finite_vector(
                metadata.get("actor_persistent_committed_residual"),
                length=7,
                label="actor_persistent_committed_residual",
                context=context,
            )
            committed_max = float(np.max(np.abs(committed_residual[:6])))
            maximum_committed_residual_rad = max(
                maximum_committed_residual_rad, committed_max
            )
            nonzero_committed_rows += int(
                committed_max > NONZERO_RESIDUAL_EPS_RAD
            )
            committed_gripper = float(committed_residual[6])
            maximum_committed_gripper_close_m = max(
                maximum_committed_gripper_close_m,
                max(0.0, -committed_gripper),
            )
            positive_gripper_residual_rows += int(
                committed_gripper > NONZERO_RESIDUAL_EPS_RAD
            )
            if (
                committed_gripper > NONZERO_RESIDUAL_EPS_RAD
                or committed_gripper
                < -GRIPPER_RESIDUAL_MAX_CLOSE_M - NONZERO_RESIDUAL_EPS_RAD
            ):
                raise PersistentV2ContractError(
                    f"{context}: committed gripper residual is outside the "
                    "close-only 5 mm envelope"
                )
            actual_residual = _finite_vector(
                metadata.get("actor_filtered_actual_residual"),
                length=7,
                label="actor_filtered_actual_residual",
                context=context,
            )
            actual_residual_alias = _finite_vector(
                metadata.get("actor_actual_residual"),
                length=7,
                label="actor_actual_residual",
                context=context,
            )
            if not np.allclose(
                actual_residual,
                actual_residual_alias,
                rtol=0.0,
                atol=1.0e-9,
            ):
                raise PersistentV2ContractError(
                    f"{context}: filtered actual residual alias mismatch"
                )
            actual_max = float(np.max(np.abs(actual_residual[:6])))
            maximum_actual_residual_rad = max(
                maximum_actual_residual_rad, actual_max
            )
            nonzero_actual_rows += int(actual_max > NONZERO_RESIDUAL_EPS_RAD)
            actual_gripper = float(actual_residual[6])
            maximum_actual_gripper_close_m = max(
                maximum_actual_gripper_close_m,
                max(0.0, -actual_gripper),
            )
            if (
                actual_gripper > NONZERO_RESIDUAL_EPS_RAD
                or actual_gripper
                < -GRIPPER_RESIDUAL_MAX_CLOSE_M - NONZERO_RESIDUAL_EPS_RAD
            ):
                raise PersistentV2ContractError(
                    f"{context}: physical gripper residual is outside the "
                    "close-only 5 mm envelope"
                )
            if not math.isclose(
                actual_gripper,
                committed_gripper,
                rel_tol=0.0,
                abs_tol=2.0e-6,
            ):
                raise PersistentV2ContractError(
                    f"{context}: committed and physical gripper residual differ"
                )
            a_exec = _finite_vector(
                row.get("a_exec"),
                length=7,
                label="a_exec",
                context=context,
            )
            if not (
                GRIPPER_COMMAND_MIN_M - NONZERO_RESIDUAL_EPS_RAD
                <= float(a_exec[6])
                <= GRIPPER_COMMAND_MAX_M + NONZERO_RESIDUAL_EPS_RAD
            ):
                raise PersistentV2ContractError(
                    f"{context}: absolute gripper command is outside [0, 0.08] m"
                )
            for metric_key, metric_limit in (
                (
                    "actor_filtered_actual_gripper_residual_d1_max_m",
                    GRIPPER_RESIDUAL_D1_MAX_M,
                ),
                (
                    "actor_filtered_actual_gripper_residual_d2_max_m",
                    GRIPPER_RESIDUAL_D2_MAX_M,
                ),
                (
                    "actor_filtered_actual_gripper_boundary_jump_max_m",
                    GRIPPER_MAX_BOUNDARY_JUMP_M,
                ),
            ):
                try:
                    metric_value = float(metadata[metric_key])
                except (KeyError, TypeError, ValueError) as exc:
                    raise PersistentV2ContractError(
                        f"{context}: missing/invalid {metric_key}"
                    ) from exc
                if (
                    not math.isfinite(metric_value)
                    or metric_value < 0.0
                    or metric_value > metric_limit + NONZERO_RESIDUAL_EPS_RAD
                ):
                    raise PersistentV2ContractError(
                        f"{context}: {metric_key} exceeds its physical envelope"
                    )
            gripper_release_rows += int(
                bool(metadata.get("actor_gripper_release_intent", False))
            )
        plan_id = metadata.get("actor_execution_plan_id")
        if not isinstance(plan_id, str) or not plan_id:
            raise PersistentV2ContractError(
                f"{context}: committed row lacks actor_execution_plan_id"
            )
        try:
            chunk_offset = int(metadata["actor_execution_plan_offset"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PersistentV2ContractError(
                f"{context}: committed row lacks actor_execution_plan_offset"
            ) from exc
        if not 0 <= chunk_offset < CHUNK_LENGTH:
            raise PersistentV2ContractError(
                f"{context}: C10 offset {chunk_offset} is outside 0..9"
            )
        plan_kind = "actor" if source == "rlt" else "human"
        previous_kind = kind_by_plan.setdefault(plan_id, plan_kind)
        if previous_kind != plan_kind:
            invalid_plan_reasons[plan_id].append("mixed_actor_human_plan_id")
        offsets_by_plan[plan_id].append(chunk_offset)
        indices_by_plan[plan_id].append(row_index)
        try:
            t_by_plan[plan_id].append(int(row["t"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise PersistentV2ContractError(
                f"{context}: committed row lacks integer t"
            ) from exc
        try:
            completed_counter_max = max(
                completed_counter_max,
                int(metadata.get("actor_live_chunks_completed", 0)),
            )
        except (TypeError, ValueError) as exc:
            raise PersistentV2ContractError(
                f"{context}: actor_live_chunks_completed is invalid"
            ) from exc

    if eligible_rows == 0:
        raise PersistentV2ContractError(
            f"{episode_id}: no replay-included executable rows"
        )
    for item in disallowed_actor_safety_rows:
        row_index = int(item["row"])
        plan_id = _metadata(rows[row_index]).get("actor_execution_plan_id")
        if isinstance(plan_id, str):
            invalid_plan_reasons[plan_id].append(
                "nonlinear_or_nonreproducible_actor_safety_modification"
            )
    for row_index in bad_safety_reset_rows:
        plan_id = _metadata(rows[row_index]).get("actor_execution_plan_id")
        if isinstance(plan_id, str):
            invalid_plan_reasons[plan_id].append(
                "reset_after_safety_modification"
            )

    complete_plan_ids: list[str] = []
    partial_plan_ids: list[str] = []
    for plan_id in sorted(offsets_by_plan, key=lambda key: indices_by_plan[key][0]):
        offsets = offsets_by_plan[plan_id]
        indices = indices_by_plan[plan_id]
        timesteps = t_by_plan[plan_id]
        complete = bool(
            offsets == list(range(CHUNK_LENGTH))
            and indices
            == list(range(indices[0], indices[0] + CHUNK_LENGTH))
            and timesteps
            == list(range(timesteps[0], timesteps[0] + CHUNK_LENGTH))
            and not invalid_plan_reasons.get(plan_id)
        )
        (complete_plan_ids if complete else partial_plan_ids).append(plan_id)

    complete_actor_plan_ids = [
        plan_id
        for plan_id in complete_plan_ids
        if kind_by_plan[plan_id] == "actor"
    ]
    complete_human_plan_ids = [
        plan_id
        for plan_id in complete_plan_ids
        if kind_by_plan[plan_id] == "human"
    ]
    if mode == STRICT_ACTOR_CANARY:
        if bad_safety_reset_rows:
            raise PersistentV2ContractError(
                f"{episode_id}: reset_after_safety_modification appeared at rows "
                f"{bad_safety_reset_rows[:10]}"
            )
        if disallowed_actor_safety_rows:
            raise PersistentV2ContractError(
                f"{episode_id}: Actor C10 contains nonlinear/non-reproducible "
                f"safety modification(s): {disallowed_actor_safety_rows[:10]}"
            )
        if phase_rlt_published_rows < min_phase_rlt_published_rows:
            raise PersistentV2ContractError(
                f"{episode_id}: only {phase_rlt_published_rows} phase-active "
                f"published RLT rows; need {min_phase_rlt_published_rows}"
            )
        if actor_physical_committed_rows < min_physical_committed_rows:
            raise PersistentV2ContractError(
                f"{episode_id}: only {actor_physical_committed_rows} physically "
                f"committed Actor rows; need {min_physical_committed_rows}"
            )
        if not complete_actor_plan_ids:
            raise PersistentV2ContractError(
                f"{episode_id}: no complete admissible Actor C10"
            )
        if nonzero_actual_rows == 0 or nonzero_committed_rows == 0:
            raise PersistentV2ContractError(
                f"{episode_id}: Actor residual was never nonzero in both actual "
                "and committed coordinates"
            )
    elif require_complete_chunk and not complete_plan_ids:
        raise PersistentV2ContractError(
            f"{episode_id}: no complete admissible Actor or human C10; partial "
            f"plans excluded={partial_plan_ids}"
        )
    if completed_counter_max < len(complete_actor_plan_ids):
        raise PersistentV2ContractError(
            f"{episode_id}: runtime completed-chunk counter {completed_counter_max} "
            f"is below audited complete Actor chunks {len(complete_actor_plan_ids)}"
        )

    return {
        "format": "openpi_piper_persistent_gripper_v3_episode_audit",
        "episode_id": episode_id,
        "reward": float(rows[-1]["reward"]),
        "rows": len(rows),
        "eligible_rows": eligible_rows,
        "phase_active_rlt_published_rows": phase_rlt_published_rows,
        "audit_mode": mode,
        "actor_physical_committed_rows": actor_physical_committed_rows,
        "human_physical_committed_rows": human_physical_committed_rows,
        "nonzero_actual_residual_rows": nonzero_actual_rows,
        "nonzero_committed_residual_rows": nonzero_committed_rows,
        "maximum_actual_residual_rad": maximum_actual_residual_rad,
        "maximum_committed_residual_rad": maximum_committed_residual_rad,
        "maximum_actual_gripper_close_m": maximum_actual_gripper_close_m,
        "maximum_committed_gripper_close_m": maximum_committed_gripper_close_m,
        "positive_gripper_residual_rows": positive_gripper_residual_rows,
        "gripper_release_rows": gripper_release_rows,
        "gripper_residual_mode": GRIPPER_RESIDUAL_MODE,
        "gripper_residual_max_close_m": GRIPPER_RESIDUAL_MAX_CLOSE_M,
        "gripper_residual_d1_max_m": GRIPPER_RESIDUAL_D1_MAX_M,
        "gripper_residual_d2_max_m": GRIPPER_RESIDUAL_D2_MAX_M,
        "gripper_max_boundary_jump_m": GRIPPER_MAX_BOUNDARY_JUMP_M,
        "complete_c10_chunks": len(complete_plan_ids),
        "complete_c10_plan_ids": complete_plan_ids,
        "complete_actor_c10_chunks": len(complete_actor_plan_ids),
        "complete_actor_c10_plan_ids": complete_actor_plan_ids,
        "complete_human_c10_chunks": len(complete_human_plan_ids),
        "complete_human_c10_plan_ids": complete_human_plan_ids,
        "partial_or_excluded_c10_plan_ids": partial_plan_ids,
        "excluded_plan_reasons": {
            plan_id: sorted(set(reasons))
            for plan_id, reasons in sorted(invalid_plan_reasons.items())
        },
        "complete_c10_row_indices_by_plan": {
            plan_id: indices_by_plan[plan_id]
            for plan_id in complete_plan_ids
        },
        "committed_row_indices": committed_row_indices,
        "actor_live_chunks_completed_counter_max": completed_counter_max,
        "actor_execution_profile": expected_execution_profile,
        "action_schema_fingerprint": expected_action_schema,
        "actor_model_action_schema_fingerprint": (
            ACTOR_MODEL_ACTION_SCHEMA_FINGERPRINT
        ),
        "actor_projection_profile": expected_projection_profile,
        "execution_filter_profile": expected_filter_profile,
        "execution_filter_tau_s": expected_filter_tau_s,
        "control_hz": expected_control_hz,
        "control_dt_s": 1.0 / expected_control_hz,
        "execution_filter_alpha": 1.0
        - math.exp(-(1.0 / expected_control_hz) / expected_filter_tau_s),
        "chunk_length": CHUNK_LENGTH,
        "chunk_stride": CHUNK_STRIDE,
        "actor_live_max_boundary_jump_rad": (
            ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD
        ),
        "actor_projection_scale_steps": ACTOR_PROJECTION_SCALE_STEPS,
        "actor_min_projection_scale": ACTOR_MIN_PROJECTION_SCALE,
        "actor_direction_static_threshold_rad": (
            ACTOR_DIRECTION_STATIC_THRESHOLD_RAD
        ),
        "actor_governor_fingerprint": PERSISTENT_GOVERNOR_FINGERPRINT,
        "ready_for_actor_canary": bool(
            mode == STRICT_ACTOR_CANARY and complete_actor_plan_ids
        ),
        "ready_for_training_admission": bool(complete_plan_ids),
        "ready_for_admission": True,
    }


def audit_persistent_v2_episode(
    path: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    path = Path(path)
    return audit_persistent_v2_rows(
        load_episode_rows(path),
        episode_id=path.parent.name,
        **kwargs,
    )
