from __future__ import annotations

import dataclasses

import numpy as np

from openpi.rlt.real.config import HUMAN_EXECUTION_PROFILE
from openpi.rlt.real.config import GRIPPER_RESIDUAL_CLOSE_ASSIST
from openpi.rlt.real.config import GRIPPER_RESIDUAL_FROZEN
from openpi.rlt.real.config import PERSISTENT_ACTOR_EXECUTION_PROFILE
from openpi.rlt.real.config import PERSISTENT_ACTION_SCHEMA_FINGERPRINT
from openpi.rlt.real.config import PERSISTENT_EXECUTION_FILTER_PROFILE
from openpi.rlt.real.config import PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT


_PERSISTENT_C10_BLEND = np.asarray(
    [
        0.0,
        0.011532794797541024,
        0.07641111619163743,
        0.2098765432098765,
        0.3966874968246709,
        0.603312503175329,
        0.790123456790123,
        0.9235888838083626,
        0.9884672052024577,
        1.0,
    ],
    dtype=np.float64,
)
_PERSISTENT_PROFILES = {
    PERSISTENT_ACTOR_EXECUTION_PROFILE,
    HUMAN_EXECUTION_PROFILE,
}
_PERSISTENT_SCHEMAS = {
    PERSISTENT_ACTION_SCHEMA_FINGERPRINT,
    PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT,
}
_GRIPPER_CLOSE_ENVELOPE_EXPECTED = {
    "execution_gripper_residual_max_close_m": 0.005,
    "execution_gripper_d1_max_m": 0.0005,
    "execution_gripper_d2_max_m": 0.0003,
    "execution_gripper_boundary_limit_m": 0.0005,
    "execution_gripper_command_min_m": 0.0,
    "execution_gripper_command_max_m": 0.08,
    "execution_gripper_release_reference_m": 0.05,
    "execution_gripper_release_delta_m": 0.002,
}


@dataclasses.dataclass(frozen=True)
class RealStepRecord:
    episode_id: str
    t: int
    z_rl: np.ndarray
    state: np.ndarray
    a_ref: np.ndarray
    a_exec: np.ndarray
    a_human: np.ndarray | None
    a_actor: np.ndarray | None
    source: str
    reward: float
    done: bool
    phase_probability: float = 0.0
    gate_active: bool = False
    global_image: str | None = None
    wrist_image: str | None = None
    timestamp_ns: int | None = None
    replay_include: bool = True
    waiting_for_reward: bool = False
    # Populated by offline enrichment.  ``a_ref`` is the recomputed reference
    # used for training; the logged reference is retained for audit only.
    a_ref_original: np.ndarray | None = None
    reference_recomputed: bool = False
    # True when z_rl was recomputed from the image/state at this exact row.
    # It is intentionally independent of ``reference_recomputed`` because
    # online replay must preserve the stochastic Pi0.5 reference observed at
    # execution time while refreshing a stale plan-level Token per frame.
    token_recomputed: bool = False
    policy_plan_id: str | None = None
    policy_observation_t: int | None = None
    plan_offset: int | None = None
    behavior_actor_checkpoint: str | None = None
    # Persistent-v2 canonical execution evidence.  These values are written
    # after the final hardware safety layer; absent values mean legacy data,
    # never implicit zero carry.
    actor_execution_profile: str | None = None
    actor_execution_plan_id: str | None = None
    actor_execution_plan_offset: int | None = None
    actor_execution_committed: bool | None = None
    actor_canonical_decision: np.ndarray | None = None
    actor_persistent_carry_in: np.ndarray | None = None
    actor_persistent_carry_out: np.ndarray | None = None
    actor_persistent_previous_carry: np.ndarray | None = None
    # The exact governed residual selected for this physical row.  Frozen-v2
    # captures predate this field; close-assist rows require it so the replay
    # can prove that dim 6 was rate-limited once by the governor and was not
    # passed through the joint-only low-pass a second time.
    actor_persistent_planned_residual: np.ndarray | None = None
    actor_execution_boundary_anchor: np.ndarray | None = None
    actor_filtered_base_action: np.ndarray | None = None
    actor_filtered_actual_action: np.ndarray | None = None
    actor_filtered_actual_residual: np.ndarray | None = None
    actor_execution_filter_tau_s: float | None = None
    actor_execution_filter_dt_s: float | None = None
    actor_execution_filter_alpha: float | None = None
    actor_execution_projection_scale: float | None = None
    action_schema_fingerprint: str | None = None
    execution_filter_profile: str | None = None
    safety_reasons: tuple[str, ...] = ()
    actor_execution_residual_max_rad: float | None = None
    actor_execution_d1_max_rad: float | None = None
    actor_execution_d2_max_rad: float | None = None
    actor_execution_direction_cone_deg: float | None = None
    actor_execution_boundary_limit_rad: float | None = None
    actor_execution_projection_scale_steps: int | None = None
    actor_execution_min_projection_scale: float | None = None
    actor_execution_direction_static_threshold_rad: float | None = None
    actor_gripper_residual_mode: str | None = None
    actor_gripper_release_intent: bool | None = None
    execution_gripper_residual_max_close_m: float | None = None
    execution_gripper_d1_max_m: float | None = None
    execution_gripper_d2_max_m: float | None = None
    execution_gripper_boundary_limit_m: float | None = None
    execution_gripper_command_min_m: float | None = None
    execution_gripper_command_max_m: float | None = None
    execution_gripper_release_reference_m: float | None = None
    execution_gripper_release_delta_m: float | None = None
    actor_filtered_actual_gripper_residual_max: float | None = None
    actor_filtered_actual_gripper_residual_d1_max_m: float | None = None
    actor_filtered_actual_gripper_residual_d2_max_m: float | None = None
    actor_filtered_actual_gripper_boundary_jump_max_m: float | None = None


@dataclasses.dataclass(frozen=True)
class RealTransition:
    episode_id: str
    t: int
    z_rl: np.ndarray
    state: np.ndarray
    a_ref: np.ndarray
    a_exec: np.ndarray
    a_human: np.ndarray
    a_actor: np.ndarray
    source: str
    reward: float
    discount: float
    next_z_rl: np.ndarray
    next_state: np.ndarray
    next_a_ref: np.ndarray
    done: bool
    phase_probability: float
    gate_active: bool
    # Per-step provenance. Missing optional actions are represented by the
    # masks, never inferred from an all-zero action vector.
    source_chunk: np.ndarray
    human_mask: np.ndarray
    actor_mask: np.ndarray
    step_mask: np.ndarray
    # Absolute command-space audit arrays. Training arrays above use the
    # coordinate contract documented in the replay manifest.
    a_ref_absolute: np.ndarray
    a_ref_original_absolute: np.ndarray
    a_exec_absolute: np.ndarray
    a_human_absolute: np.ndarray
    a_actor_absolute: np.ndarray
    next_a_ref_absolute: np.ndarray
    # Behavior-policy provenance is attached to each transition anchor.  This
    # makes off-policy replay auditable after Actor promotion/rollback.
    policy_plan_id: str
    policy_observation_t: int
    plan_offset: int
    behavior_actor_checkpoint: str
    actor_execution_profile: str | None = None
    actor_execution_plan_id: str | None = None
    actor_execution_plan_offset: np.ndarray | None = None
    actor_canonical_decision: np.ndarray | None = None
    actor_persistent_carry_in: np.ndarray | None = None
    actor_persistent_carry_out: np.ndarray | None = None
    actor_persistent_previous_carry: np.ndarray | None = None
    actor_execution_boundary_anchor: np.ndarray | None = None
    a_base_filtered: np.ndarray | None = None
    a_filtered_actual: np.ndarray | None = None
    filtered_actual_residual: np.ndarray | None = None
    execution_filter_tau_s: np.ndarray | None = None
    execution_filter_dt_s: np.ndarray | None = None
    execution_filter_alpha: np.ndarray | None = None
    execution_projection_scale: np.ndarray | None = None
    next_a_base_filtered: np.ndarray | None = None
    next_execution_filter_alpha: np.ndarray | None = None
    next_actor_persistent_previous_carry: np.ndarray | None = None
    next_actor_persistent_carry_in: np.ndarray | None = None
    next_actor_execution_boundary_anchor: np.ndarray | None = None
    action_schema_fingerprint: str | None = None
    execution_filter_profile: str | None = None
    terminal_reward_migration_steps: int = 0
    terminal_reward_migration_offset: int = -1
    execution_residual_max_rad: np.ndarray | None = None
    execution_d1_max_rad: np.ndarray | None = None
    execution_d2_max_rad: np.ndarray | None = None
    execution_direction_cone_deg: np.ndarray | None = None
    execution_boundary_limit_rad: np.ndarray | None = None
    execution_projection_scale_steps: np.ndarray | None = None
    execution_min_projection_scale: np.ndarray | None = None
    execution_direction_static_threshold_rad: np.ndarray | None = None
    # Episode-level outcome, deliberately separate from the n-step reward.
    # Early transitions in a successful episode often have reward==0.
    success_mask: bool = False
    actor_persistent_planned_residual: np.ndarray | None = None
    gripper_residual_mode: str | None = None
    actor_gripper_release_intent: np.ndarray | None = None
    execution_gripper_residual_max_close_m: np.ndarray | None = None
    execution_gripper_d1_max_m: np.ndarray | None = None
    execution_gripper_d2_max_m: np.ndarray | None = None
    execution_gripper_boundary_limit_m: np.ndarray | None = None
    execution_gripper_command_min_m: np.ndarray | None = None
    execution_gripper_command_max_m: np.ndarray | None = None
    execution_gripper_release_reference_m: np.ndarray | None = None
    execution_gripper_release_delta_m: np.ndarray | None = None
    actor_filtered_actual_gripper_residual_max: np.ndarray | None = None
    actor_filtered_actual_gripper_residual_d1_max_m: np.ndarray | None = None
    actor_filtered_actual_gripper_residual_d2_max_m: np.ndarray | None = None
    actor_filtered_actual_gripper_boundary_jump_max_m: np.ndarray | None = None


def _as_float32(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _stack_or_pad_step_actions(values: list[np.ndarray], length: int, action_dim: int) -> np.ndarray:
    if not values:
        return np.zeros((length, action_dim), dtype=np.float32)
    arrays = [_as_float32(value).reshape(action_dim) for value in values]
    while len(arrays) < length:
        arrays.append(arrays[-1].copy())
    return np.stack(arrays[:length]).astype(np.float32)


def _chunk_from_first_reference(record: RealStepRecord, length: int, action_dim: int) -> np.ndarray:
    ref = _as_float32(record.a_ref)
    if ref.ndim == 1:
        return np.repeat(ref.reshape(1, action_dim), length, axis=0).astype(np.float32)
    if ref.shape[0] >= length:
        return ref[:length].astype(np.float32)
    pad = [*ref]
    while len(pad) < length:
        pad.append(pad[-1].copy())
    return np.stack(pad[:length]).astype(np.float32)


def _optional_action_chunk(records: list[RealStepRecord], attr: str, length: int, action_dim: int) -> np.ndarray:
    values = []
    for record in records:
        value = getattr(record, attr)
        if value is None:
            values.append(np.zeros(action_dim, dtype=np.float32))
        else:
            arr = _as_float32(value)
            values.append(arr[0] if arr.ndim == 2 else arr)
    return _stack_or_pad_step_actions(values, length, action_dim)


def _optional_action_chunk_and_mask(
    records: list[RealStepRecord], attr: str, length: int, action_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    values: list[np.ndarray] = []
    mask: list[bool] = []
    for record in records:
        value = getattr(record, attr)
        mask.append(value is not None)
        if value is None:
            values.append(np.zeros(action_dim, dtype=np.float32))
        else:
            arr = _as_float32(value)
            values.append(arr[0] if arr.ndim == 2 else arr)
    while len(mask) < length:
        mask.append(False)
    return _stack_or_pad_step_actions(values, length, action_dim), np.asarray(mask[:length], dtype=np.bool_)


def _pad_sources(records: list[RealStepRecord], length: int) -> np.ndarray:
    values = [record.source for record in records]
    values.extend(["pad"] * (length - len(values)))
    width = max(3, *(len(value) for value in values))
    return np.asarray(values[:length], dtype=f"<U{width}")


def _to_training_coordinates(actions_absolute: np.ndarray, start_state: np.ndarray) -> np.ndarray:
    """Convert absolute Piper commands to the real-RLT action contract.

    Joints 0..5 are deltas from the state at the beginning of this chunk;
    gripper index 6 remains an absolute target.
    """

    actions = _as_float32(actions_absolute).copy()
    state = _as_float32(start_state).reshape(-1)
    if actions.shape[-1] != 7 or state.shape[0] < 6:
        raise ValueError("Piper training-coordinate conversion requires 7-D actions and at least 6-D state")
    actions[..., :6] -= state[:6]
    return actions


def _split_contiguous_records(records: list[RealStepRecord]) -> list[list[RealStepRecord]]:
    """Split at episode boundaries, invalid-row gaps, and terminals.

    Enrichment removes invalid rows before calling this function. A timestep
    gap therefore proves an invalid row was removed and chunks must not bridge
    it.
    """

    segments: list[list[RealStepRecord]] = []
    current: list[RealStepRecord] = []
    for record in records:
        split = bool(
            current
            and (
                record.episode_id != current[-1].episode_id
                or record.t != current[-1].t + 1
                or current[-1].done
            )
        )
        if split:
            segments.append(current)
            current = []
        current.append(record)
        if record.done:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def chunk_real_episode(
    records: list[RealStepRecord],
    *,
    chunk_length: int,
    stride: int,
    n_step: int,
    gamma: float,
) -> list[RealTransition]:
    if not records:
        return []
    if any(record.actor_execution_profile in _PERSISTENT_PROFILES for record in records):
        committed_schemas = {
            str(record.action_schema_fingerprint)
            for record in records
            if record.actor_execution_profile in _PERSISTENT_PROFILES
            and record.actor_execution_committed is True
            and record.action_schema_fingerprint is not None
        }
        unsupported = committed_schemas.difference(_PERSISTENT_SCHEMAS)
        if unsupported:
            raise ValueError(
                "unsupported persistent action schema(s): "
                f"{sorted(unsupported)}"
            )
        if len(committed_schemas) > 1:
            raise ValueError(
                "frozen-v2 and gripper-close persistent schemas cannot be "
                f"mixed in one episode: {sorted(committed_schemas)}"
            )
        if chunk_length != 10 or stride != 10 or n_step != 10:
            raise ValueError(
                "persistent-v2 replay requires C=10, stride=10, and n_step=10"
            )
        return _chunk_persistent_execution_episode(records, gamma=gamma)
    transitions: list[RealTransition] = []
    episode_success = any(
        record.done and float(record.reward) > 0.0 for record in records
    )
    for segment in _split_contiguous_records(records):
        transitions.extend(
            _chunk_contiguous_segment(
                segment,
                chunk_length=chunk_length,
                stride=stride,
                n_step=n_step,
                gamma=gamma,
                episode_success=episode_success,
            )
        )
    return transitions


def _complete_execution_plan(
    records: list[RealStepRecord],
    start: int,
) -> list[RealStepRecord] | None:
    block = records[start : start + 10]
    if len(block) != 10:
        return None
    first = block[0]
    profile = first.actor_execution_profile
    plan_id = first.actor_execution_plan_id
    action_schema = first.action_schema_fingerprint
    if profile not in _PERSISTENT_PROFILES or not plan_id:
        return None
    if action_schema not in _PERSISTENT_SCHEMAS:
        raise ValueError(
            "unsupported persistent action schema: "
            f"{action_schema!r}"
        )
    for offset, record in enumerate(block):
        if record.actor_execution_profile != profile:
            return None
        if record.actor_execution_plan_id != plan_id:
            return None
        if record.actor_execution_plan_offset != offset:
            return None
        if record.actor_execution_committed is not True:
            return None
        if record.action_schema_fingerprint != action_schema:
            raise ValueError(
                "persistent action schema changed within one physical C10: "
                f"{record.action_schema_fingerprint!r} != {action_schema!r}"
            )
        if record.execution_filter_profile != PERSISTENT_EXECUTION_FILTER_PROFILE:
            raise ValueError(
                "persistent-v2 execution filter mismatch: "
                f"{record.execution_filter_profile!r}"
            )
        if offset and (
            record.episode_id != block[offset - 1].episode_id
            or record.t != block[offset - 1].t + 1
            or block[offset - 1].done
        ):
            return None
    return block


def _require_plan_vector(record: RealStepRecord, name: str) -> np.ndarray:
    value = getattr(record, name)
    if value is None:
        raise ValueError(f"persistent-v2 row is missing {name}")
    value = _as_float32(value)
    if value.shape != (7,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite with shape (7,), got {value.shape}")
    return value


def _constant_plan_vector(block: list[RealStepRecord], name: str, *, atol: float = 2e-6) -> np.ndarray:
    values = np.stack([_require_plan_vector(record, name) for record in block])
    if not np.allclose(values, values[:1], rtol=0.0, atol=atol):
        raise ValueError(f"{name} must remain constant within one physical C10 plan")
    return values[0].astype(np.float32)


def _plan_filter_array(block: list[RealStepRecord], name: str) -> np.ndarray:
    try:
        values = np.asarray(
            [getattr(record, name) for record in block], dtype=np.float64
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must provide ten finite values") from exc
    if values.shape != (10,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must provide ten finite values")
    return values.astype(np.float32)


def _plan_bool_array(block: list[RealStepRecord], name: str) -> np.ndarray:
    values = [getattr(record, name) for record in block]
    if any(not isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError(f"{name} must provide ten boolean values")
    return np.asarray(values, dtype=np.bool_)


def _validate_persistent_plan_evidence(block: list[RealStepRecord]) -> dict[str, object]:
    profile = str(block[0].actor_execution_profile)
    action_schema = str(block[0].action_schema_fingerprint)
    close_assist = (
        action_schema == PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT
    )
    decision = _constant_plan_vector(block, "actor_canonical_decision")
    carry_in = _constant_plan_vector(block, "actor_persistent_carry_in")
    previous_carry = _constant_plan_vector(block, "actor_persistent_previous_carry")
    boundary_anchor = _constant_plan_vector(block, "actor_execution_boundary_anchor")
    base = np.stack(
        [_require_plan_vector(record, "actor_filtered_base_action") for record in block]
    ).astype(np.float32)
    actual = np.stack(
        [_require_plan_vector(record, "actor_filtered_actual_action") for record in block]
    ).astype(np.float32)
    actual_residual = np.stack(
        [_require_plan_vector(record, "actor_filtered_actual_residual") for record in block]
    ).astype(np.float32)
    carry_out_rows = np.stack(
        [_require_plan_vector(record, "actor_persistent_carry_out") for record in block]
    ).astype(np.float32)
    tau = _plan_filter_array(block, "actor_execution_filter_tau_s")
    dt = _plan_filter_array(block, "actor_execution_filter_dt_s")
    alpha = _plan_filter_array(block, "actor_execution_filter_alpha")
    projection_scale = _plan_filter_array(block, "actor_execution_projection_scale")
    envelope_expected = {
        "actor_execution_residual_max_rad": 0.005,
        "actor_execution_d1_max_rad": 0.0015,
        "actor_execution_d2_max_rad": 0.001,
        "actor_execution_direction_cone_deg": 15.0,
        "actor_execution_boundary_limit_rad": 0.06,
        "actor_execution_projection_scale_steps": 33.0,
        "actor_execution_min_projection_scale": 0.2,
        "actor_execution_direction_static_threshold_rad": 0.001,
    }
    envelope: dict[str, np.ndarray] = {}
    for name, expected in envelope_expected.items():
        values = _plan_filter_array(block, name)
        if not np.allclose(values, expected, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"persistent-v2 runtime envelope drift for {name}: "
                f"{values.tolist()} != {expected}"
            )
        envelope[name] = values
    gripper_envelope: dict[str, np.ndarray] = {}
    planned_residual: np.ndarray | None = None
    release_intent: np.ndarray | None = None
    gripper_certificate: dict[str, np.ndarray] = {}
    if close_assist:
        modes = {
            str(record.actor_gripper_residual_mode)
            for record in block
            if record.actor_gripper_residual_mode is not None
        }
        if modes != {GRIPPER_RESIDUAL_CLOSE_ASSIST}:
            raise ValueError(
                "gripper-close replay requires one explicit residual mode: "
                f"{sorted(modes)}"
            )
        if profile == HUMAN_EXECUTION_PROFILE:
            # Human commands bypass the Actor governor.  Some older logs
            # retained a shadow Actor knot in this field; it was never
            # executed and must not become a behavior target.
            planned_residual = np.zeros((len(block), 7), dtype=np.float32)
        else:
            planned_residual = np.stack(
                [
                    _require_plan_vector(
                        record, "actor_persistent_planned_residual"
                    )
                    for record in block
                ]
            ).astype(np.float32)
        release_intent = (
            np.zeros(len(block), dtype=np.bool_)
            if profile == HUMAN_EXECUTION_PROFILE
            else _plan_bool_array(block, "actor_gripper_release_intent")
        )
        if not np.all(release_intent == release_intent[0]):
            raise ValueError(
                "actor_gripper_release_intent must remain constant within one C10"
            )
        for name, expected in _GRIPPER_CLOSE_ENVELOPE_EXPECTED.items():
            values = _plan_filter_array(block, name)
            if not np.allclose(values, expected, rtol=0.0, atol=1e-9):
                raise ValueError(
                    "gripper-close runtime envelope drift for "
                    f"{name}: {values.tolist()} != {expected}"
                )
            gripper_envelope[name] = values
        for name in (
            "actor_filtered_actual_gripper_residual_max",
            "actor_filtered_actual_gripper_residual_d1_max_m",
            "actor_filtered_actual_gripper_residual_d2_max_m",
            "actor_filtered_actual_gripper_boundary_jump_max_m",
        ):
            raw_values = [getattr(record, name) for record in block]
            if (
                profile == HUMAN_EXECUTION_PROFILE
                and all(value is None for value in raw_values)
            ):
                # Human Pika rows do not pass through the Actor governor and
                # therefore have no execution-time Actor certificate.  The
                # complete physical C10 is sufficient to derive an exact
                # certificate below; do not reject admitted human data.
                continue
            if any(value is None for value in raw_values):
                raise ValueError(
                    f"{name} must be either present for every row or absent "
                    "for every human row in one physical C10 plan"
                )
            values = np.asarray(raw_values, dtype=np.float32)
            if values.shape != (10,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must provide ten finite values")
            if np.any(values < 0.0):
                raise ValueError(f"{name} must be non-negative")
            # Actor certificates are measured for each physical row by
            # certify_filtered_execution(), so they normally vary across the
            # quintic C10.  Human rows that carry a reconstructed certificate
            # retain the historical constant-per-plan representation.
            if (
                profile == HUMAN_EXECUTION_PROFILE
                and not np.allclose(values, values[0], rtol=0.0, atol=1e-7)
            ):
                raise ValueError(
                    f"{name} must remain constant for a human physical C10 plan"
                )
            gripper_certificate[name] = values
    else:
        unexpected_modes = {
            str(record.actor_gripper_residual_mode)
            for record in block
            if record.actor_gripper_residual_mode not in {
                None,
                "",
                GRIPPER_RESIDUAL_FROZEN,
            }
        }
        if unexpected_modes:
            raise ValueError(
                "frozen-v2 rows declare a close-assist gripper mode: "
                f"{sorted(unexpected_modes)}"
            )
    if profile == PERSISTENT_ACTOR_EXECUTION_PROFILE:
        disallowed_safety = sorted(
            {
                reason
                for record in block
                for reason in record.safety_reasons
                if reason not in {"", "model_low_pass"}
            }
        )
        if disallowed_safety:
            raise ValueError(
                "persistent-v2 plan contains nonlinear/unmodelled safety modifications: "
                f"{disallowed_safety}"
            )
    if np.any(tau <= 0.0) or np.any(dt <= 0.0):
        raise ValueError("persistent-v2 filter tau/dt must be positive")
    expected_alpha = 1.0 - np.exp(-dt.astype(np.float64) / tau.astype(np.float64))
    if not np.allclose(alpha, expected_alpha, rtol=2e-5, atol=1e-7):
        raise ValueError("persistent-v2 alpha must equal 1-exp(-dt/tau)")
    if not np.allclose(tau, tau[0], rtol=0.0, atol=1e-9):
        raise ValueError("execution filter tau must remain constant within one C10 plan")
    if np.any(projection_scale < 0.0) or np.any(projection_scale > 1.0):
        raise ValueError("execution projection scale must stay within [0, 1]")
    if not np.allclose(projection_scale, projection_scale[0], rtol=0.0, atol=1e-7):
        raise ValueError("execution projection scale must remain constant within one C10 plan")
    if not close_assist and abs(float(decision[6])) > 1e-7:
        raise ValueError("canonical Actor gripper decision must stay frozen at zero")

    if profile == PERSISTENT_ACTOR_EXECUTION_PROFILE:
        if not np.allclose(actual, base + actual_residual, rtol=0.0, atol=2e-6):
            raise ValueError("persistent actual action must equal filtered base plus residual")
        if not np.allclose(carry_out_rows, actual_residual, rtol=0.0, atol=2e-6):
            raise ValueError("persistent carry_out must equal each committed actual residual")
        target_joint = carry_in[:6] + float(projection_scale[0]) * (
            decision[:6] - carry_in[:6]
        )
        planned_joint = carry_in[None, :6] + _PERSISTENT_C10_BLEND[:, None] * (
            target_joint - carry_in[:6]
        )[None, :]
        if close_assist:
            assert planned_residual is not None
            if not np.allclose(
                planned_residual[:, :6],
                planned_joint,
                rtol=0.0,
                atol=3e-6,
            ):
                raise ValueError(
                    "logged planned joint residual does not match the "
                    "carry-aware quintic governor"
                )
            planned = planned_residual.astype(np.float64)
            _validate_close_assist_gripper_plan(
                profile=profile,
                decision=decision,
                carry_in=carry_in,
                previous_carry=previous_carry,
                planned=planned,
                base=base,
                actual=actual,
                actual_residual=actual_residual,
                release_intent=bool(release_intent[0]),
                certificate=gripper_certificate,
            )
        else:
            planned = np.zeros((10, 7), dtype=np.float64)
            planned[:, :6] = planned_joint
        expected_filtered: list[np.ndarray] = []
        current = carry_in.astype(np.float64)
        for row, row_alpha in zip(planned, alpha):
            current[:6] = (
                (1.0 - float(row_alpha)) * current[:6]
                + float(row_alpha) * row[:6]
            )
            # The native execution low-pass is joint-only.  Close-assist
            # gripper knots are already rate/acceleration limited by the
            # governor and must be committed exactly once.
            current[6] = float(row[6]) if close_assist else 0.0
            expected_filtered.append(current.copy())
        if not np.allclose(
            actual_residual,
            np.asarray(expected_filtered),
            rtol=0.0,
            atol=3e-6,
        ):
            raise ValueError(
                "persistent actual residual does not match carry-aware "
                "quintic plus logged execution low-pass"
            )
    else:
        zeros = (
            decision,
            carry_in,
            previous_carry,
            carry_out_rows,
        )
        if any(not np.allclose(value, 0.0, rtol=0.0, atol=1e-7) for value in zeros):
            raise ValueError("human execution profile requires a reset zero Actor residual channel")
        if not np.allclose(actual, base + actual_residual, rtol=0.0, atol=2e-6):
            raise ValueError("human residual audit must equal actual minus counterfactual base")
        if not np.allclose(
            actual,
            np.stack([_as_float32(record.a_exec) for record in block]),
            rtol=0.0,
            atol=2e-6,
        ):
            raise ValueError("human actual action must match a_exec")
        if close_assist:
            assert planned_residual is not None
            if not np.allclose(
                planned_residual, 0.0, rtol=0.0, atol=1e-7
            ):
                raise ValueError(
                    "human execution requires a zero planned Actor residual"
                )
            _validate_close_assist_gripper_plan(
                profile=profile,
                decision=decision,
                carry_in=carry_in,
                previous_carry=previous_carry,
                planned=planned_residual,
                base=base,
                actual=actual,
                actual_residual=actual_residual,
                release_intent=bool(release_intent[0]),
                certificate=gripper_certificate,
            )

    return {
        "action_schema": action_schema,
        "decision": decision,
        "carry_in": carry_in,
        "carry_out": carry_out_rows[-1],
        "previous_carry": previous_carry,
        "boundary_anchor": boundary_anchor,
        "base": base,
        "actual": actual,
        "actual_residual": actual_residual,
        "tau": tau,
        "dt": dt,
        "alpha": alpha,
        "projection_scale": projection_scale,
        "planned_residual": planned_residual,
        "gripper_residual_mode": (
            GRIPPER_RESIDUAL_CLOSE_ASSIST if close_assist else None
        ),
        "gripper_release_intent": release_intent,
        **envelope,
        **gripper_envelope,
        **gripper_certificate,
    }


def _validate_close_assist_gripper_plan(
    *,
    profile: str,
    decision: np.ndarray,
    carry_in: np.ndarray,
    previous_carry: np.ndarray,
    planned: np.ndarray,
    base: np.ndarray,
    actual: np.ndarray,
    actual_residual: np.ndarray,
    release_intent: bool,
    certificate: dict[str, np.ndarray],
) -> None:
    """Validate the one-sided gripper certificate without filtering it twice."""

    close_limit = _GRIPPER_CLOSE_ENVELOPE_EXPECTED[
        "execution_gripper_residual_max_close_m"
    ]
    d1_limit = _GRIPPER_CLOSE_ENVELOPE_EXPECTED[
        "execution_gripper_d1_max_m"
    ]
    d2_limit = _GRIPPER_CLOSE_ENVELOPE_EXPECTED[
        "execution_gripper_d2_max_m"
    ]
    boundary_limit = _GRIPPER_CLOSE_ENVELOPE_EXPECTED[
        "execution_gripper_boundary_limit_m"
    ]
    command_min = _GRIPPER_CLOSE_ENVELOPE_EXPECTED[
        "execution_gripper_command_min_m"
    ]
    command_max = _GRIPPER_CLOSE_ENVELOPE_EXPECTED[
        "execution_gripper_command_max_m"
    ]
    tolerance = 3e-6
    gripper = np.asarray(planned[:, 6], dtype=np.float64)
    actual_gripper_residual = np.asarray(
        actual_residual[:, 6], dtype=np.float64
    )
    if profile == PERSISTENT_ACTOR_EXECUTION_PROFILE:
        for name, value in (
            ("canonical decision", float(decision[6])),
            ("carry_in", float(carry_in[6])),
            ("previous_carry", float(previous_carry[6])),
        ):
            if value > tolerance or value < -close_limit - tolerance:
                raise ValueError(
                    f"close-assist gripper {name} is outside "
                    f"[-{close_limit}, 0]: {value}"
                )
        if np.any(gripper > tolerance) or np.any(
            gripper < -close_limit - tolerance
        ):
            raise ValueError(
                "close-assist planned gripper residual must stay in "
                f"[-{close_limit}, 0]"
            )
        expected = float(carry_in[6]) + _PERSISTENT_C10_BLEND * (
            float(gripper[-1]) - float(carry_in[6])
        )
        if not np.allclose(gripper, expected, rtol=0.0, atol=tolerance):
            raise ValueError(
                "planned gripper residual does not match its carry-aware "
                "quintic knot"
            )
        desired = (
            0.0
            if release_intent
            else float(np.clip(decision[6], -close_limit, 0.0))
        )
        lower = min(float(carry_in[6]), desired) - tolerance
        upper = max(float(carry_in[6]), desired) + tolerance
        if not lower <= float(gripper[-1]) <= upper:
            raise ValueError(
                "planned gripper target is not a governed interpolation "
                "between carry and the close/release target"
            )
        if not np.allclose(
            actual_gripper_residual, gripper, rtol=0.0, atol=tolerance
        ):
            raise ValueError(
                "close-assist actual gripper residual must equal the planned "
                "knot exactly; the joint low-pass must not be applied twice"
            )

    command = np.asarray(actual[:, 6], dtype=np.float64)
    if np.any(command < command_min - tolerance) or np.any(
        command > command_max + tolerance
    ):
        raise ValueError(
            "executed gripper command leaves the certified Piper range "
            f"[{command_min}, {command_max}]"
        )
    previous_d1 = float(carry_in[6] - previous_carry[6])
    d1_signed = np.diff(
        np.concatenate(
            [
                np.asarray([float(carry_in[6])], dtype=np.float64),
                actual_gripper_residual,
            ]
        )
    )
    d2_signed = d1_signed - np.concatenate(
        [
            np.asarray([previous_d1], dtype=np.float64),
            d1_signed[:-1],
        ]
    )
    measured_by_row = {
        "actor_filtered_actual_gripper_residual_max": np.abs(
            actual_gripper_residual
        ),
        "actor_filtered_actual_gripper_residual_d1_max_m": np.abs(d1_signed),
        "actor_filtered_actual_gripper_residual_d2_max_m": np.abs(d2_signed),
        # The runtime checks the jump from its last committed residual on
        # every row, not only at offset zero.
        "actor_filtered_actual_gripper_boundary_jump_max_m": np.abs(d1_signed),
    }
    measured = {
        name: float(np.max(values))
        for name, values in measured_by_row.items()
    }
    if profile == PERSISTENT_ACTOR_EXECUTION_PROFILE:
        if measured[
            "actor_filtered_actual_gripper_residual_max"
        ] > close_limit + tolerance:
            raise ValueError("close-assist gripper residual limit was exceeded")
        if measured[
            "actor_filtered_actual_gripper_residual_d1_max_m"
        ] > d1_limit + tolerance:
            raise ValueError("close-assist gripper d1 limit was exceeded")
        if measured[
            "actor_filtered_actual_gripper_residual_d2_max_m"
        ] > d2_limit + tolerance:
            raise ValueError("close-assist gripper d2 limit was exceeded")
        if measured[
            "actor_filtered_actual_gripper_boundary_jump_max_m"
        ] > boundary_limit + tolerance:
            raise ValueError(
                "close-assist gripper boundary limit was exceeded"
            )
    for name, expected_max in measured.items():
        values = certificate.get(name)
        if values is None and profile == HUMAN_EXECUTION_PROFILE:
            # This is physical-execution evidence derived from the admitted
            # human C10, not an Actor-governor approval.  Store it as a
            # constant per-plan certificate so the replay schema remains
            # uniform across Actor and human transitions.
            values = np.full(
                actual_gripper_residual.shape,
                expected_max,
                dtype=np.float32,
            )
            certificate[name] = values
        expected = (
            np.full(
                actual_gripper_residual.shape,
                expected_max,
                dtype=np.float64,
            )
            if profile == HUMAN_EXECUTION_PROFILE
            else measured_by_row[name]
        )
        if values is None or not np.allclose(
            values, expected, rtol=0.0, atol=tolerance
        ):
            raise ValueError(
                f"{name} certificate does not match executed evidence: "
                f"{None if values is None else values.tolist()} != "
                f"{expected.tolist()}"
            )


def _chunk_persistent_execution_episode(
    records: list[RealStepRecord],
    *,
    gamma: float,
) -> list[RealTransition]:
    blocks: list[tuple[int, list[RealStepRecord], dict[str, object]]] = []
    index = 0
    while index < len(records):
        block = _complete_execution_plan(records, index)
        if block is None:
            index += 1
            continue
        evidence = _validate_persistent_plan_evidence(block)
        blocks.append((index, block, evidence))
        index += 10

    transitions: list[RealTransition] = []
    if not blocks and any(record.done for record in records):
        raise ValueError(
            "persistent-v2 episode has a terminal reward but no complete committed C10 plan"
        )
    episode_success = any(
        record.done and float(record.reward) > 0.0 for record in records
    )
    for block_index, (start, block, evidence) in enumerate(blocks):
        terminal_offset = next((idx for idx, record in enumerate(block) if record.done), None)
        migrated_terminal_index: int | None = None
        next_block: list[RealStepRecord] | None = None
        next_evidence: dict[str, object] | None = None
        if terminal_offset is None:
            if block_index + 1 >= len(blocks):
                migrated_terminal_index = next(
                    (
                        index
                        for index in range(start + 10, len(records))
                        if records[index].episode_id == block[0].episode_id
                        and records[index].done
                    ),
                    None,
                )
                if migrated_terminal_index is None:
                    continue
                terminal_offset = migrated_terminal_index - start
            else:
                next_start, candidate_next, candidate_evidence = blocks[block_index + 1]
                if next_start != start + 10:
                    continue
                next_block = candidate_next
                next_evidence = candidate_evidence
                current_profile = str(block[0].actor_execution_profile)
                next_profile = str(next_block[0].actor_execution_profile)
                if current_profile == next_profile == PERSISTENT_ACTOR_EXECUTION_PROFILE:
                    if not np.allclose(
                        evidence["carry_out"],
                        next_evidence["carry_in"],
                        rtol=0.0,
                        atol=3e-6,
                    ):
                        raise ValueError("persistent carry_out does not match next-plan carry_in")
                elif not np.allclose(
                    next_evidence["carry_in"], 0.0, rtol=0.0, atol=1e-7
                ):
                    raise ValueError("execution-profile transition must reset next carry_in to zero")

        reward = 0.0
        done = terminal_offset is not None
        reward_records = (
            records[start : migrated_terminal_index + 1]
            if migrated_terminal_index is not None
            else block
        )
        for offset, record in enumerate(reward_records):
            reward += (gamma**offset) * float(record.reward)
            if record.done:
                break
        next_record = (
            records[migrated_terminal_index]
            if migrated_terminal_index is not None
            else (block[-1] if done else next_block[0])
        )
        if next_evidence is None:
            next_evidence = evidence
        start_state = _as_float32(block[0].state)
        ref_abs = _chunk_from_first_reference(block[0], 10, 7)
        original_ref = block[0].a_ref_original
        if original_ref is None:
            original_ref = block[0].a_ref
        original_ref_abs = _chunk_from_first_reference(
            dataclasses.replace(block[0], a_ref=original_ref), 10, 7
        )
        exec_abs = np.stack([_as_float32(record.a_exec) for record in block])
        human_abs, human_mask = _optional_action_chunk_and_mask(block, "a_human", 10, 7)
        actor_abs, actor_mask = _optional_action_chunk_and_mask(block, "a_actor", 10, 7)
        next_ref_abs = _chunk_from_first_reference(next_record, 10, 7)
        human_train = _to_training_coordinates(human_abs, start_state)
        actor_train = _to_training_coordinates(actor_abs, start_state)
        human_train[~human_mask] = 0.0
        actor_train[~actor_mask] = 0.0
        boundary_anchor_train = _to_training_coordinates(
            np.asarray(evidence["boundary_anchor"])[None, :],
            start_state,
        )[0]
        transitions.append(
            RealTransition(
                episode_id=block[0].episode_id,
                t=block[0].t,
                z_rl=_as_float32(block[0].z_rl),
                state=start_state,
                a_ref=_to_training_coordinates(ref_abs, start_state),
                a_exec=_to_training_coordinates(exec_abs, start_state),
                a_human=human_train,
                a_actor=actor_train,
                source=block[0].source,
                reward=float(reward),
                discount=0.0 if done else float(gamma**10),
                next_z_rl=_as_float32(next_record.z_rl),
                next_state=_as_float32(next_record.state),
                next_a_ref=_to_training_coordinates(next_ref_abs, next_record.state),
                done=done,
                phase_probability=float(block[0].phase_probability),
                gate_active=bool(block[0].gate_active),
                source_chunk=_pad_sources(block, 10),
                human_mask=human_mask,
                actor_mask=actor_mask,
                step_mask=np.ones(10, dtype=np.bool_),
                a_ref_absolute=ref_abs,
                a_ref_original_absolute=original_ref_abs,
                a_exec_absolute=exec_abs,
                a_human_absolute=human_abs,
                a_actor_absolute=actor_abs,
                next_a_ref_absolute=next_ref_abs,
                policy_plan_id=str(block[0].policy_plan_id or ""),
                policy_observation_t=(
                    -1
                    if block[0].policy_observation_t is None
                    else int(block[0].policy_observation_t)
                ),
                plan_offset=(
                    -1 if block[0].plan_offset is None else int(block[0].plan_offset)
                ),
                behavior_actor_checkpoint=str(block[0].behavior_actor_checkpoint or "NONE"),
                actor_execution_profile=str(block[0].actor_execution_profile),
                actor_execution_plan_id=str(block[0].actor_execution_plan_id),
                actor_execution_plan_offset=np.arange(10, dtype=np.int64),
                actor_canonical_decision=np.asarray(evidence["decision"], dtype=np.float32),
                actor_persistent_carry_in=np.asarray(evidence["carry_in"], dtype=np.float32),
                actor_persistent_carry_out=np.asarray(evidence["carry_out"], dtype=np.float32),
                actor_persistent_previous_carry=np.asarray(
                    evidence["previous_carry"], dtype=np.float32
                ),
                actor_execution_boundary_anchor=boundary_anchor_train,
                a_base_filtered=_to_training_coordinates(
                    np.asarray(evidence["base"]), start_state
                ),
                a_filtered_actual=_to_training_coordinates(
                    np.asarray(evidence["actual"]), start_state
                ),
                filtered_actual_residual=np.asarray(
                    evidence["actual_residual"], dtype=np.float32
                ),
                execution_filter_tau_s=np.asarray(evidence["tau"], dtype=np.float32),
                execution_filter_dt_s=np.asarray(evidence["dt"], dtype=np.float32),
                execution_filter_alpha=np.asarray(evidence["alpha"], dtype=np.float32),
                execution_projection_scale=np.asarray(
                    evidence["projection_scale"], dtype=np.float32
                ),
                next_a_base_filtered=_to_training_coordinates(
                    np.asarray(next_evidence["base"]), next_record.state
                ),
                next_execution_filter_alpha=np.asarray(
                    next_evidence["alpha"], dtype=np.float32
                ),
                next_actor_persistent_previous_carry=np.asarray(
                    next_evidence["previous_carry"], dtype=np.float32
                ),
                next_actor_persistent_carry_in=np.asarray(
                    next_evidence["carry_in"], dtype=np.float32
                ),
                next_actor_execution_boundary_anchor=_to_training_coordinates(
                    np.asarray(next_evidence["boundary_anchor"])[None, :],
                    next_record.state,
                )[0],
                action_schema_fingerprint=str(evidence["action_schema"]),
                execution_filter_profile=PERSISTENT_EXECUTION_FILTER_PROFILE,
                terminal_reward_migration_steps=(
                    0
                    if migrated_terminal_index is None
                    else migrated_terminal_index - (start + 9)
                ),
                terminal_reward_migration_offset=(
                    -1 if migrated_terminal_index is None else terminal_offset
                ),
                execution_residual_max_rad=np.asarray(
                    evidence["actor_execution_residual_max_rad"], dtype=np.float32
                ),
                execution_d1_max_rad=np.asarray(
                    evidence["actor_execution_d1_max_rad"], dtype=np.float32
                ),
                execution_d2_max_rad=np.asarray(
                    evidence["actor_execution_d2_max_rad"], dtype=np.float32
                ),
                execution_direction_cone_deg=np.asarray(
                    evidence["actor_execution_direction_cone_deg"], dtype=np.float32
                ),
                execution_boundary_limit_rad=np.asarray(
                    evidence["actor_execution_boundary_limit_rad"], dtype=np.float32
                ),
                execution_projection_scale_steps=np.asarray(
                    evidence["actor_execution_projection_scale_steps"], dtype=np.float32
                ),
                execution_min_projection_scale=np.asarray(
                    evidence["actor_execution_min_projection_scale"], dtype=np.float32
                ),
                execution_direction_static_threshold_rad=np.asarray(
                    evidence["actor_execution_direction_static_threshold_rad"],
                    dtype=np.float32,
                ),
                success_mask=episode_success,
                actor_persistent_planned_residual=(
                    None
                    if evidence["planned_residual"] is None
                    else np.asarray(
                        evidence["planned_residual"], dtype=np.float32
                    )
                ),
                gripper_residual_mode=(
                    None
                    if evidence["gripper_residual_mode"] is None
                    else str(evidence["gripper_residual_mode"])
                ),
                actor_gripper_release_intent=(
                    None
                    if evidence["gripper_release_intent"] is None
                    else np.asarray(
                        evidence["gripper_release_intent"], dtype=np.bool_
                    )
                ),
                execution_gripper_residual_max_close_m=(
                    None
                    if evidence.get(
                        "execution_gripper_residual_max_close_m"
                    )
                    is None
                    else np.asarray(
                        evidence[
                            "execution_gripper_residual_max_close_m"
                        ],
                        dtype=np.float32,
                    )
                ),
                execution_gripper_d1_max_m=(
                    None
                    if evidence.get("execution_gripper_d1_max_m") is None
                    else np.asarray(
                        evidence["execution_gripper_d1_max_m"],
                        dtype=np.float32,
                    )
                ),
                execution_gripper_d2_max_m=(
                    None
                    if evidence.get("execution_gripper_d2_max_m") is None
                    else np.asarray(
                        evidence["execution_gripper_d2_max_m"],
                        dtype=np.float32,
                    )
                ),
                execution_gripper_boundary_limit_m=(
                    None
                    if evidence.get(
                        "execution_gripper_boundary_limit_m"
                    )
                    is None
                    else np.asarray(
                        evidence["execution_gripper_boundary_limit_m"],
                        dtype=np.float32,
                    )
                ),
                execution_gripper_command_min_m=(
                    None
                    if evidence.get("execution_gripper_command_min_m")
                    is None
                    else np.asarray(
                        evidence["execution_gripper_command_min_m"],
                        dtype=np.float32,
                    )
                ),
                execution_gripper_command_max_m=(
                    None
                    if evidence.get("execution_gripper_command_max_m")
                    is None
                    else np.asarray(
                        evidence["execution_gripper_command_max_m"],
                        dtype=np.float32,
                    )
                ),
                execution_gripper_release_reference_m=(
                    None
                    if evidence.get(
                        "execution_gripper_release_reference_m"
                    )
                    is None
                    else np.asarray(
                        evidence[
                            "execution_gripper_release_reference_m"
                        ],
                        dtype=np.float32,
                    )
                ),
                execution_gripper_release_delta_m=(
                    None
                    if evidence.get("execution_gripper_release_delta_m")
                    is None
                    else np.asarray(
                        evidence["execution_gripper_release_delta_m"],
                        dtype=np.float32,
                    )
                ),
                actor_filtered_actual_gripper_residual_max=(
                    None
                    if evidence.get(
                        "actor_filtered_actual_gripper_residual_max"
                    )
                    is None
                    else np.asarray(
                        evidence[
                            "actor_filtered_actual_gripper_residual_max"
                        ],
                        dtype=np.float32,
                    )
                ),
                actor_filtered_actual_gripper_residual_d1_max_m=(
                    None
                    if evidence.get(
                        "actor_filtered_actual_gripper_residual_d1_max_m"
                    )
                    is None
                    else np.asarray(
                        evidence[
                            "actor_filtered_actual_gripper_residual_d1_max_m"
                        ],
                        dtype=np.float32,
                    )
                ),
                actor_filtered_actual_gripper_residual_d2_max_m=(
                    None
                    if evidence.get(
                        "actor_filtered_actual_gripper_residual_d2_max_m"
                    )
                    is None
                    else np.asarray(
                        evidence[
                            "actor_filtered_actual_gripper_residual_d2_max_m"
                        ],
                        dtype=np.float32,
                    )
                ),
                actor_filtered_actual_gripper_boundary_jump_max_m=(
                    None
                    if evidence.get(
                        "actor_filtered_actual_gripper_boundary_jump_max_m"
                    )
                    is None
                    else np.asarray(
                        evidence[
                            "actor_filtered_actual_gripper_boundary_jump_max_m"
                        ],
                        dtype=np.float32,
                    )
                ),
            )
        )
    return transitions


def _chunk_contiguous_segment(
    records: list[RealStepRecord],
    *,
    chunk_length: int,
    stride: int,
    n_step: int,
    gamma: float,
    episode_success: bool | None = None,
) -> list[RealTransition]:
    action_dim = int(_as_float32(records[0].a_exec).reshape(-1).shape[0])
    if action_dim != 7:
        raise ValueError(f"real Piper replay requires 7-D actions, got {action_dim}")
    if episode_success is None:
        episode_success = any(
            record.done and float(record.reward) > 0.0 for record in records
        )
    transitions: list[RealTransition] = []
    for start in range(0, len(records), stride):
        chunk = records[start : start + chunk_length]
        if not chunk:
            continue
        terminal_offset = next((idx for idx, record in enumerate(chunk) if record.done), None)
        if len(chunk) < chunk_length and terminal_offset is None:
            continue

        reward = 0.0
        done = False
        for offset, record in enumerate(chunk[:n_step]):
            reward += (gamma**offset) * float(record.reward)
            if record.done:
                done = True
                break

        # Never bootstrap from the last row of a truncated/invalid-bounded
        # segment. A nonterminal n-step sample requires the real state at t+n;
        # otherwise its gamma**n target would silently cross an invalid gap.
        if not done and start + n_step >= len(records):
            continue
        bootstrap_index = min(start + n_step, len(records) - 1)
        next_record = records[bootstrap_index]
        discount = 0.0 if done else gamma**n_step
        start_state = _as_float32(records[start].state)
        ref_abs = _chunk_from_first_reference(records[start], chunk_length, action_dim)
        original_ref = records[start].a_ref_original
        if original_ref is None:
            original_ref = records[start].a_ref
        original_ref_abs = _chunk_from_first_reference(
            dataclasses.replace(records[start], a_ref=original_ref), chunk_length, action_dim
        )
        exec_abs = _stack_or_pad_step_actions([record.a_exec for record in chunk], chunk_length, action_dim)
        human_abs, human_mask = _optional_action_chunk_and_mask(chunk, "a_human", chunk_length, action_dim)
        actor_abs, actor_mask = _optional_action_chunk_and_mask(chunk, "a_actor", chunk_length, action_dim)
        next_ref_abs = _chunk_from_first_reference(next_record, chunk_length, action_dim)
        step_mask = np.zeros(chunk_length, dtype=np.bool_)
        step_mask[: len(chunk)] = True
        human_train = _to_training_coordinates(human_abs, start_state)
        actor_train = _to_training_coordinates(actor_abs, start_state)
        human_train[~human_mask] = 0.0
        actor_train[~actor_mask] = 0.0
        transitions.append(
            RealTransition(
                episode_id=records[start].episode_id,
                t=records[start].t,
                z_rl=_as_float32(records[start].z_rl),
                state=_as_float32(records[start].state),
                a_ref=_to_training_coordinates(ref_abs, start_state),
                a_exec=_to_training_coordinates(exec_abs, start_state),
                a_human=human_train,
                a_actor=actor_train,
                source=records[start].source,
                reward=float(reward),
                discount=float(discount),
                next_z_rl=_as_float32(next_record.z_rl),
                next_state=_as_float32(next_record.state),
                next_a_ref=_to_training_coordinates(next_ref_abs, next_record.state),
                done=done,
                phase_probability=float(records[start].phase_probability),
                gate_active=bool(records[start].gate_active),
                source_chunk=_pad_sources(chunk, chunk_length),
                human_mask=human_mask,
                actor_mask=actor_mask,
                step_mask=step_mask,
                a_ref_absolute=ref_abs,
                a_ref_original_absolute=original_ref_abs,
                a_exec_absolute=exec_abs,
                a_human_absolute=human_abs,
                a_actor_absolute=actor_abs,
                next_a_ref_absolute=next_ref_abs,
                policy_plan_id="" if records[start].policy_plan_id is None else str(records[start].policy_plan_id),
                policy_observation_t=(
                    -1 if records[start].policy_observation_t is None else int(records[start].policy_observation_t)
                ),
                plan_offset=-1 if records[start].plan_offset is None else int(records[start].plan_offset),
                behavior_actor_checkpoint=(
                    "NONE"
                    if records[start].behavior_actor_checkpoint in {None, "", "none"}
                    else str(records[start].behavior_actor_checkpoint)
                ),
                success_mask=episode_success,
            )
        )
    return transitions
