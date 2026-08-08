from __future__ import annotations

"""Offline acceptance audit for the synchronized H50/C10 RLT runtime.

The audit reads episode JSONL only.  It never imports ROS, opens a camera, or
publishes a robot command, so it is safe to run while the hardware is offline.

The checks deliberately distinguish the 50-step behavior horizon from the
10-step legacy Actor checkpoint horizon:

* an H50 boundary is a reset of ``model_action_index`` to zero;
* a C10 Actor refresh may change ``policy_plan_id`` and must not be counted as
  an H50 behavior boundary;
* an old Actor checkpoint remains compatible when it still emits finite
  ``(10, 7)`` chunks that are selected for live control.
"""

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PERSISTENT_C10_EXECUTION_CONTRACT = "persistent_c10_from_rank1_v1"
LEGACY_RANK1_INPUT_CONTRACT = "rank1_bump_v1"


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("policy_metadata")
    return value if isinstance(value, dict) else {}


def _field(row: dict[str, Any], key: str, default: Any = None) -> Any:
    value = row.get(key)
    if value is not None:
        return value
    return _metadata(row).get(key, default)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _model_index(row: dict[str, Any]) -> int | None:
    raw = _field(row, "model_action_index")
    try:
        return None if raw is None else int(raw)
    except (TypeError, ValueError):
        return None


def _actor_offset(row: dict[str, Any]) -> int | None:
    raw = _field(row, "plan_offset")
    try:
        return None if raw is None else int(raw)
    except (TypeError, ValueError):
        return None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _finite_vector(value: Any, length: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        return None
    return vector


def _actor_residual_at_row(
    row: dict[str, Any],
    *,
    action_dim: int,
    actor_chunk_length: int,
) -> np.ndarray | None:
    """Return the governed Actor residual at the row's C10 offset."""

    offset = _actor_offset(row)
    if offset is None or offset < 0 or offset >= actor_chunk_length:
        return None
    # Persistent execution deliberately differs from the legacy rank1 bump at
    # the C10 endpoints.  The explicit *safe* residual is therefore the only
    # unambiguous per-step continuity signal in the synchronized runtime.
    safe_metadata_residual = _field(
        row, "actor_governor_safe_residual_this_step"
    )
    if safe_metadata_residual is not None:
        try:
            residual = np.asarray(safe_metadata_residual, dtype=np.float64)
        except (TypeError, ValueError):
            residual = np.empty((0,), dtype=np.float64)
        if residual.shape == (action_dim,) and np.all(np.isfinite(residual)):
            return residual
    # Historical v4 logs did not expose the safe per-step residual.  Their raw
    # residual field is still preferable to potentially misaligned replay
    # arrays, but it must only be used as a backward-compatibility fallback.
    metadata_residual = _field(row, "actor_governor_raw_residual_this_step")
    if metadata_residual is not None:
        try:
            residual = np.asarray(metadata_residual, dtype=np.float64)
        except (TypeError, ValueError):
            residual = np.empty((0,), dtype=np.float64)
        if residual.shape == (action_dim,) and np.all(np.isfinite(residual)):
            return residual
    safe = row.get("a_actor_safe")
    reference = row.get("a_ref")
    if safe is None or reference is None:
        return None
    try:
        safe_array = np.asarray(safe, dtype=np.float64)
        reference_array = np.asarray(reference, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    expected = (actor_chunk_length, action_dim)
    if safe_array.shape != expected or reference_array.shape != expected:
        return None
    if not np.all(np.isfinite(safe_array)) or not np.all(np.isfinite(reference_array)):
        return None
    return safe_array[offset] - reference_array[offset]


def _is_model_motion_row(row: dict[str, Any]) -> bool:
    keyboard = row.get("keyboard")
    mode = keyboard.get("mode") if isinstance(keyboard, dict) else None
    if mode is not None and mode != "MODEL":
        return False
    return not bool(row.get("done", False))


def _h50_boundaries(rows: list[dict[str, Any]]) -> list[int]:
    """Indices of non-initial H50 activations.

    ``policy_plan_id`` is intentionally not used: the legacy C10 Actor service
    assigns a new inference id every ten steps while retaining the same H50
    behavior reference.
    """

    boundaries: list[int] = []
    previous_index: int | None = None
    seen_behavior = False
    previous_episode: Any = None
    for row_index, row in enumerate(rows):
        episode = row.get("episode_id")
        if previous_episode is not None and episode != previous_episode:
            previous_index = None
            seen_behavior = False
        model_index = _model_index(row)
        if model_index is None:
            previous_episode = episode
            continue
        if model_index == 0:
            if seen_behavior and previous_index not in {None, 0}:
                boundaries.append(row_index)
            seen_behavior = True
        previous_index = model_index
        previous_episode = episode
    return boundaries


def _full_h50_segments(
    rows: list[dict[str, Any]],
    *,
    horizon: int,
) -> list[list[dict[str, Any]]]:
    """Return complete model-index 0..H-1 segments without boundary holds."""

    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    last_index: int | None = None
    last_episode: Any = None
    for row in rows:
        episode = row.get("episode_id")
        if last_episode is not None and episode != last_episode:
            if current:
                segments.append(current)
            current = []
            last_index = None
        index = _model_index(row)
        if index is None:
            last_episode = episode
            continue
        if index == 0 and last_index not in {None, 0}:
            if current:
                segments.append(current)
            current = []
        if 0 <= index < horizon:
            current.append(row)
        last_index = index
        last_episode = episode
    if current:
        segments.append(current)
    complete: list[list[dict[str, Any]]] = []
    for segment in segments:
        indices = [_model_index(row) for row in segment]
        if indices and indices[0] == 0 and set(indices) >= set(range(horizon)):
            complete.append(segment)
    return complete


def audit_rlt_inference_parity(
    rows: list[dict[str, Any]],
    *,
    expected_checkpoint: str | None = None,
    behavior_horizon: int = 50,
    actor_chunk_length: int = 10,
    action_dim: int = 7,
    min_h50_prefetch_hit_rate: float = 1.0,
    min_actor_phase_coverage: float = 1.0,
    max_boundary_wait_s: float = 0.04,
    max_actor_boundary_residual_jump_rad: float = 0.001,
    require_50hz_evidence: bool = True,
    require_persistent_actor_contract: bool = True,
    require_actor_only_rng_isolation: bool = True,
    require_physical_actor_execution: bool = True,
) -> dict[str, Any]:
    if behavior_horizon < 1 or actor_chunk_length < 1 or action_dim < 1:
        raise ValueError("horizon, Actor chunk length, and action dimension must be positive")
    if not 0.0 <= min_h50_prefetch_hit_rate <= 1.0:
        raise ValueError("min_h50_prefetch_hit_rate must be in [0, 1]")
    if not 0.0 <= min_actor_phase_coverage <= 1.0:
        raise ValueError("min_actor_phase_coverage must be in [0, 1]")
    if max_boundary_wait_s < 0.0 or max_actor_boundary_residual_jump_rad < 0.0:
        raise ValueError("audit limits must be non-negative")

    violations: list[str] = []
    boundaries = _h50_boundaries(rows)
    boundary_reports: list[dict[str, Any]] = []
    for boundary_index in boundaries:
        row = rows[boundary_index]
        metadata = _metadata(row)
        previous_index = boundary_index - 1
        safety_rows: list[int] = []
        boundary_episode = row.get("episode_id")
        # Include all repeated end-of-plan rows immediately before activation,
        # not merely the last row.  This catches the old 4--12 frame droop gap.
        while previous_index >= 0:
            previous = rows[previous_index]
            if previous.get("episode_id") != boundary_episode:
                break
            previous_model_index = _model_index(previous)
            if previous_model_index is not None and previous_model_index < behavior_horizon - 1:
                break
            if previous.get("source") == "safety_block":
                safety_rows.append(int(previous.get("t", previous_index)))
            previous_index -= 1
        request_reason = metadata.get("policy_request_reason")
        prefetch_hit = bool(
            metadata.get("policy_prefetch_hit") is True
            and request_reason == "behavior_prefetch"
        )
        wait_s = _finite_float(metadata.get("policy_boundary_wait_s"), 0.0)
        accepted = metadata.get("h50_prefetch_accepted")
        report = {
            "row_index": boundary_index,
            "t": row.get("t"),
            "request_reason": request_reason,
            "prefetch_hit": prefetch_hit,
            "handoff_accepted": accepted,
            "boundary_wait_s": wait_s,
            "preceding_safety_block_t": list(reversed(safety_rows)),
        }
        boundary_reports.append(report)
        if not prefetch_hit:
            violations.append(
                f"t={row.get('t')}: H50 boundary was not a ready behavior-prefetch hit "
                f"(reason={request_reason!r})"
            )
        if wait_s > max_boundary_wait_s + 1e-9:
            violations.append(
                f"t={row.get('t')}: H50 boundary wait {wait_s:.6f}s exceeds "
                f"{max_boundary_wait_s:.6f}s"
            )
        if safety_rows:
            violations.append(
                f"t={row.get('t')}: safety_block rows preceded the H50 activation: "
                f"{list(reversed(safety_rows))}"
            )

    hit_count = sum(report["prefetch_hit"] for report in boundary_reports)
    hit_rate = 1.0 if not boundary_reports else hit_count / len(boundary_reports)
    if boundary_reports and hit_rate + 1e-12 < min_h50_prefetch_hit_rate:
        violations.append(
            f"H50 prefetch hit rate {hit_count}/{len(boundary_reports)}={hit_rate:.3f} "
            f"is below {min_h50_prefetch_hit_rate:.3f}"
        )
    if not boundary_reports:
        violations.append("episode contains no non-initial H50 boundary; parity result is inconclusive")

    phase_rows = [
        row
        for row in rows
        if row.get("gate_active") is True
        and _is_model_motion_row(row)
        and (_model_index(row) is not None)
        and 0 <= int(_model_index(row)) < behavior_horizon
    ]
    actor_selected_rows = [row for row in phase_rows if row.get("source") == "rlt"]
    actor_physical_rows = [
        row
        for row in actor_selected_rows
        if _field(row, "actor_physical_execution_committed_this_step") is True
    ]
    actor_coverage = (
        0.0 if not phase_rows else len(actor_selected_rows) / len(phase_rows)
    )
    actor_physical_coverage = (
        0.0 if not phase_rows else len(actor_physical_rows) / len(phase_rows)
    )
    if not phase_rows:
        violations.append("no phase-active model rows; Actor coverage is inconclusive")
    elif actor_coverage + 1e-12 < min_actor_phase_coverage:
        violations.append(
            f"Actor phase coverage {len(actor_selected_rows)}/{len(phase_rows)}="
            f"{actor_coverage:.3f} is below {min_actor_phase_coverage:.3f}"
        )
    if (
        require_physical_actor_execution
        and phase_rows
        and actor_physical_coverage + 1e-12 < min_actor_phase_coverage
    ):
        violations.append(
            f"physically committed Actor phase coverage "
            f"{len(actor_physical_rows)}/{len(phase_rows)}="
            f"{actor_physical_coverage:.3f} is below "
            f"{min_actor_phase_coverage:.3f}; simulated commits are not "
            "physical execution evidence"
        )

    actor_deciles: list[dict[str, Any]] = []
    for start in range(0, behavior_horizon, actor_chunk_length):
        stop = min(behavior_horizon, start + actor_chunk_length)
        eligible = [
            row
            for row in phase_rows
            if start <= int(_model_index(row)) < stop
        ]
        selected = [row for row in eligible if row.get("source") == "rlt"]
        actor_deciles.append(
            {
                "start": start,
                "stop": stop,
                "eligible_rows": len(eligible),
                "actor_rows": len(selected),
                "coverage": None if not eligible else len(selected) / len(eligible),
            }
        )

    full_segments = _full_h50_segments(rows, horizon=behavior_horizon)
    full_phase_segments = [
        segment
        for segment in full_segments
        if all(row.get("gate_active") is True for row in segment)
    ]
    segment_reports: list[dict[str, Any]] = []
    for segment in full_phase_segments:
        by_index: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in segment:
            index = _model_index(row)
            if index is not None:
                by_index[index].append(row)
        missing = [
            index
            for index in range(behavior_horizon)
            if not any(row.get("source") == "rlt" for row in by_index.get(index, []))
        ]
        segment_reports.append(
            {
                "start_t": segment[0].get("t"),
                "end_t": segment[-1].get("t"),
                "missing_actor_offsets": missing,
            }
        )
        if missing:
            violations.append(
                f"full phase-active H50 starting t={segment[0].get('t')} "
                f"has no Actor command at offsets {missing}"
            )

    actor_boundary_jumps: list[float] = []
    actor_carry_boundary_jumps: list[float] = []
    actor_exec_boundary_jumps: list[float] = []
    for previous, current in zip(rows, rows[1:]):
        if (
            previous.get("episode_id") != current.get("episode_id")
            or
            previous.get("source") != "rlt"
            or current.get("source") != "rlt"
            or _actor_offset(current) != 0
            or _actor_offset(previous) != actor_chunk_length - 1
        ):
            continue
        previous_residual = _actor_residual_at_row(
            previous,
            action_dim=action_dim,
            actor_chunk_length=actor_chunk_length,
        )
        current_residual = _actor_residual_at_row(
            current,
            action_dim=action_dim,
            actor_chunk_length=actor_chunk_length,
        )
        if previous_residual is not None and current_residual is not None:
            jump = float(
                np.max(np.abs(current_residual[:6] - previous_residual[:6]))
            )
            actor_boundary_jumps.append(jump)
            if jump > max_actor_boundary_residual_jump_rad + 1e-9:
                violations.append(
                    f"t={current.get('t')}: Actor residual boundary jump "
                    f"{jump:.6f}rad exceeds {max_actor_boundary_residual_jump_rad:.6f}rad"
                )
        previous_carry_out = _finite_vector(
            _field(previous, "actor_persistent_carry_out"),
            action_dim,
        )
        current_carry_in = _finite_vector(
            _field(current, "actor_persistent_carry_in"),
            action_dim,
        )
        if previous_carry_out is not None and current_carry_in is not None:
            carry_jump = float(
                np.max(np.abs(current_carry_in[:6] - previous_carry_out[:6]))
            )
            actor_carry_boundary_jumps.append(carry_jump)
            if carry_jump > 1e-7:
                violations.append(
                    f"t={current.get('t')}: persistent Actor carry changed at a "
                    f"C10 boundary by {carry_jump:.9f}rad"
                )
        elif require_persistent_actor_contract:
            violations.append(
                f"t={current.get('t')}: persistent Actor carry-in/out metadata "
                "is missing at a C10 boundary"
            )
        try:
            previous_exec = np.asarray(previous.get("a_exec"), dtype=np.float64)
            current_exec = np.asarray(current.get("a_exec"), dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if (
            previous_exec.shape == (action_dim,)
            and current_exec.shape == (action_dim,)
            and np.all(np.isfinite(previous_exec))
            and np.all(np.isfinite(current_exec))
        ):
            actor_exec_boundary_jumps.append(
                float(np.max(np.abs(current_exec[:6] - previous_exec[:6])))
            )

    compatibility_rows = []
    incompatible_rows = []
    accepted_update_rows = 0
    persistent_hold_rows = 0
    checkpoints: set[str] = set()
    schemas: set[str] = set()
    persistent_contract_rows = 0
    legacy_input_contract_rows = 0
    for row in actor_selected_rows:
        checkpoint = str(_field(row, "behavior_actor_checkpoint", "") or "")
        if checkpoint:
            checkpoints.add(checkpoint)
        schema = str(_field(row, "action_schema_fingerprint", "") or "")
        if schema:
            schemas.add(schema)
        execution_contract = _field(row, "actor_execution_contract")
        legacy_input_contract = _field(
            row, "actor_governor_legacy_input_contract"
        )
        if execution_contract == PERSISTENT_C10_EXECUTION_CONTRACT:
            persistent_contract_rows += 1
        if legacy_input_contract == LEGACY_RANK1_INPUT_CONTRACT:
            legacy_input_contract_rows += 1
        persistent_hold = _field(row, "actor_persistent_hold") is True
        target_update_accepted = (
            _field(row, "actor_persistent_target_update_accepted") is True
        )
        persistent_hold_rows += int(persistent_hold)
        accepted_update_rows += int(target_update_accepted)
        try:
            actor = np.asarray(row.get("a_actor"), dtype=np.float64)
        except (TypeError, ValueError):
            actor = np.empty((0, 0), dtype=np.float64)
        raw_payload_compatible = bool(
            actor.shape == (actor_chunk_length, action_dim)
            and np.all(np.isfinite(actor))
            and _field(row, "actor_shadow_ready") is True
        )
        # A latency/rejection hold is still a valid live persistent command:
        # it intentionally has no fresh legacy Actor payload.  Compatibility
        # is judged on accepted target-update rows, while hold rows must carry
        # the explicit persistent-hold contract.
        compatible = bool(
            raw_payload_compatible
            if require_persistent_actor_contract and not persistent_hold
            else persistent_hold or raw_payload_compatible
        )
        compatibility_rows.append(compatible)
        if not compatible:
            incompatible_rows.append(int(row.get("t", -1)))
        if expected_checkpoint and expected_checkpoint not in checkpoint:
            violations.append(
                f"t={row.get('t')}: Actor checkpoint {checkpoint!r} does not match "
                f"{expected_checkpoint!r}"
            )
    if actor_selected_rows and not all(compatibility_rows):
        violations.append(
            f"legacy C{actor_chunk_length} Actor payload is incompatible at t={incompatible_rows}"
        )
    if require_persistent_actor_contract and actor_selected_rows:
        if accepted_update_rows == 0:
            violations.append(
                "persistent Actor selected commands but accepted no legacy C10 "
                "target update; correction effectiveness is inconclusive"
            )
        if persistent_contract_rows != len(actor_selected_rows):
            violations.append(
                "not every live Actor row advertises "
                f"{PERSISTENT_C10_EXECUTION_CONTRACT!r}: "
                f"{persistent_contract_rows}/{len(actor_selected_rows)}"
            )
        if legacy_input_contract_rows != len(actor_selected_rows):
            violations.append(
                "not every persistent Actor row records the legacy rank1 input "
                f"contract: {legacy_input_contract_rows}/{len(actor_selected_rows)}"
            )

    actor_only_refresh_rows = [
        row
        for row in actor_selected_rows
        if _actor_offset(row) == 0
        and _field(
            row,
            "actor_only_mode",
            _field(
                row,
                "actor_inference_mode",
                _field(row, "rlt_shadow_mode", None),
            ),
        )
        in {True, "actor_only"}
    ]
    actor_only_isolated_rows = 0
    for row in actor_only_refresh_rows:
        mode = _field(
            row,
            "actor_only_mode",
            _field(
                row,
                "actor_inference_mode",
                _field(row, "rlt_shadow_mode", None),
            ),
        )
        base_called = _field(
            row,
            "actor_only_base_policy_called",
            _field(row, "base_policy_called", None),
        )
        rng_advanced = _field(
            row,
            "actor_only_base_rng_advanced",
            _field(row, "base_rng_advanced", None),
        )
        token_called = _field(
            row,
            "actor_only_token_encoder_called",
            _field(row, "token_encoder_called", None),
        )
        actor_called = _field(
            row,
            "actor_only_actor_called",
            _field(row, "actor_called", None),
        )
        protocol = _field(row, "actor_only_protocol", None)
        token_contract_ok = bool(
            (
                protocol == "actor_only_v1"
                and token_called is False
            )
            or (
                protocol == "actor_enrichment_only_v1"
                and token_called is True
            )
        )
        isolated = bool(
            mode in {True, "actor_only"}
            and base_called is False
            and rng_advanced is False
            and actor_called is True
            and token_contract_ok
        )
        actor_only_isolated_rows += int(isolated)
        if require_actor_only_rng_isolation and not isolated:
            violations.append(
                f"t={row.get('t')}: Actor refresh lacks proof that Pi0.5/token "
                "followed the declared Actor-only protocol and the base RNG "
                "was not advanced"
            )
    if (
        require_actor_only_rng_isolation
        and actor_selected_rows
        and not actor_only_refresh_rows
    ):
        violations.append(
            "no Actor-only C10 refresh row was identifiable; base RNG isolation "
            "is inconclusive"
        )

    # The 30 Hz episode logger cannot prove the physical publisher's 50 Hz
    # cadence.  The synchronized runtime therefore records fixed-rate
    # publisher evidence in each row.  Accept either the explicit rate or a
    # monotonically increasing publication counter plus its configured rate.
    publisher_rates = {
        _finite_float(
            _field(
                row,
                "command_publisher_hz",
                _field(row, "servo_publish_hz", None),
            ),
            0.0,
        )
        for row in rows
    }
    publisher_rates.discard(0.0)
    publisher_count_samples = [
        (
            row.get("episode_id"),
            int(value),
        )
        for row in rows
        for value in [
            _field(
                row,
                "command_publisher_count",
                _field(row, "servo_publish_count", None),
            )
        ]
        if value is not None
    ]
    publisher_counts = [sample[1] for sample in publisher_count_samples]
    publisher_count_advanced = any(
        current_episode == previous_episode and current_count > previous_count
        for (previous_episode, previous_count), (
            current_episode,
            current_count,
        ) in zip(publisher_count_samples, publisher_count_samples[1:])
    )
    publisher_50hz_evidence = bool(
        any(abs(rate - 50.0) <= 0.5 for rate in publisher_rates)
        and (
            len(publisher_counts) < 2
            or publisher_count_advanced
        )
    )
    if require_50hz_evidence and not publisher_50hz_evidence:
        violations.append(
            "episode has no verifiable 50 Hz publisher evidence "
            "(command_publisher_hz/servo_publish_hz and publication count)"
        )

    boundary_waits = [report["boundary_wait_s"] for report in boundary_reports]
    safety_counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        safety_counts.update(
            str(reason) for reason in _metadata(row).get("safety_reasons", [])
        )
    return {
        "format": "piper_rlt_inference_parity_audit_v1",
        "passed": not violations,
        "violations": violations,
        "rows": len(rows),
        "source_counts": dict(
            collections.Counter(str(row.get("source")) for row in rows)
        ),
        "h50": {
            "boundaries": len(boundary_reports),
            "prefetch_hits": hit_count,
            "prefetch_hit_rate": hit_rate,
            "boundary_wait_s_p50": _percentile(boundary_waits, 50),
            "boundary_wait_s_p95": _percentile(boundary_waits, 95),
            "boundary_wait_s_max": max(boundary_waits, default=None),
            "reports": boundary_reports,
        },
        "actor": {
            "phase_rows": len(phase_rows),
            "selected_rows": len(actor_selected_rows),
            "phase_coverage": actor_coverage,
            "physically_committed_rows": len(actor_physical_rows),
            "physical_phase_coverage": actor_physical_coverage,
            "coverage_by_h50_slice": actor_deciles,
            "full_phase_h50_segments": segment_reports,
            "c10_boundary_count": len(actor_boundary_jumps),
            "residual_boundary_jump_rad_p95": _percentile(
                actor_boundary_jumps, 95
            ),
            "residual_boundary_jump_rad_max": max(
                actor_boundary_jumps, default=None
            ),
            "persistent_carry_boundary_jump_rad_max": max(
                actor_carry_boundary_jumps, default=None
            ),
            "exec_boundary_jump_rad_p95": _percentile(
                actor_exec_boundary_jumps, 95
            ),
            "exec_boundary_jump_rad_max": max(
                actor_exec_boundary_jumps, default=None
            ),
            "legacy_c10_payload_compatible": bool(
                actor_selected_rows
                and accepted_update_rows
                and all(compatibility_rows)
            ),
            "persistent_contract_rows": persistent_contract_rows,
            "legacy_input_contract_rows": legacy_input_contract_rows,
            "persistent_target_update_rows": accepted_update_rows,
            "persistent_hold_rows": persistent_hold_rows,
            "actor_only_refresh_rows": len(actor_only_refresh_rows),
            "actor_only_rng_isolated_rows": actor_only_isolated_rows,
            "checkpoints": sorted(checkpoints),
            "action_schemas": sorted(schemas),
        },
        "publisher": {
            "rates_hz": sorted(publisher_rates),
            "count_first": (
                None if not publisher_counts else publisher_counts[0]
            ),
            "count_last": None if not publisher_counts else publisher_counts[-1],
            "has_50hz_evidence": publisher_50hz_evidence,
        },
        "safety_reason_counts": dict(safety_counts),
    }


def _read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        expanded = path.expanduser()
        with expanded.open("r", encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit pure-inference H50 parity and full-phase Actor coverage"
    )
    parser.add_argument("episode_jsonl", type=Path, nargs="+")
    parser.add_argument("--expected-checkpoint")
    parser.add_argument("--min-h50-prefetch-hit-rate", type=float, default=1.0)
    parser.add_argument("--min-actor-phase-coverage", type=float, default=1.0)
    parser.add_argument("--max-boundary-wait-s", type=float, default=0.04)
    parser.add_argument(
        "--max-actor-boundary-residual-jump-rad", type=float, default=0.001
    )
    parser.add_argument(
        "--allow-missing-50hz-evidence",
        action="store_true",
        help="Use only for historical logs that predate fixed-rate publisher telemetry",
    )
    parser.add_argument(
        "--allow-legacy-zero-endpoint-execution",
        action="store_true",
        help="Historical comparison only; do not require persistent C10 execution metadata",
    )
    parser.add_argument(
        "--allow-unverified-actor-only-rng",
        action="store_true",
        help="Historical comparison only; do not require Actor-only RNG isolation telemetry",
    )
    parser.add_argument(
        "--allow-simulated-actor-execution",
        action="store_true",
        help=(
            "Offline simulation only; do not require per-row proof that the "
            "Actor command was delivered to a real publisher"
        ),
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    report = audit_rlt_inference_parity(
        _read_rows(args.episode_jsonl),
        expected_checkpoint=args.expected_checkpoint,
        min_h50_prefetch_hit_rate=args.min_h50_prefetch_hit_rate,
        min_actor_phase_coverage=args.min_actor_phase_coverage,
        max_boundary_wait_s=args.max_boundary_wait_s,
        max_actor_boundary_residual_jump_rad=(
            args.max_actor_boundary_residual_jump_rad
        ),
        require_50hz_evidence=not args.allow_missing_50hz_evidence,
        require_persistent_actor_contract=(
            not args.allow_legacy_zero_endpoint_execution
        ),
        require_actor_only_rng_isolation=(
            not args.allow_unverified_actor_only_rng
        ),
        require_physical_actor_execution=(
            not args.allow_simulated_actor_execution
        ),
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output_json is not None:
        output = args.output_json.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
