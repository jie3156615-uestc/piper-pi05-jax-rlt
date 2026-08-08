#!/usr/bin/env python3
"""Offline acceptance checks for a real-RLT JAX Actor-Critic checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

_WORKSPACE_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_WORKSPACE_SRC))

if TYPE_CHECKING:
    from openpi.rlt.real.agent_jax import RealRLTLearner


PERSISTENT_ACTOR_EXECUTION_PROFILE = "persistent_c10_filtered_actual_v2"
PERSISTENT_CURRENT_CANDIDATE_FIELDS = (
    "a_base_filtered",
    "actor_persistent_carry_in",
    "actor_persistent_previous_carry",
    "actor_execution_boundary_anchor",
    "execution_filter_alpha",
)
PERSISTENT_NEXT_CANDIDATE_FIELDS = (
    "next_a_base_filtered",
    "next_actor_persistent_carry_in",
    "next_actor_persistent_previous_carry",
    "next_actor_execution_boundary_anchor",
    "next_execution_filter_alpha",
)


def _uses_persistent_candidate(learner: RealRLTLearner) -> bool:
    return (
        getattr(getattr(learner, "config", None), "actor_execution_profile", None)
        == PERSISTENT_ACTOR_EXECUTION_PROFILE
    )


def _candidate_action(
    learner: RealRLTLearner,
    batch: dict[str, np.ndarray],
    *,
    z_rl: np.ndarray | None = None,
    use_target: bool = False,
    next_state: bool = False,
    reference_visible: bool = True,
) -> np.ndarray:
    prefix = "next_" if next_state else ""
    z_key = f"{prefix}z_rl"
    state_key = f"{prefix}state"
    reference_key = f"{prefix}a_ref"
    selected_z = batch[z_key] if z_rl is None else z_rl
    if not _uses_persistent_candidate(learner):
        options: dict[str, bool] = {}
        if use_target:
            options["use_target"] = True
        if not reference_visible:
            options["reference_visible"] = False
        return learner.act(
            selected_z,
            batch[state_key],
            batch[reference_key],
            **options,
        )
    return learner.persistent_candidate_action(
        selected_z,
        batch[state_key],
        batch[reference_key],
        a_base_filtered=batch[f"{prefix}a_base_filtered"],
        carry_in=batch[f"{prefix}actor_persistent_carry_in"],
        previous_carry=batch[
            f"{prefix}actor_persistent_previous_carry"
        ],
        boundary_anchor=batch[f"{prefix}actor_execution_boundary_anchor"],
        filter_alpha=batch[f"{prefix}execution_filter_alpha"],
        use_target=use_target,
        reference_visible=reference_visible,
    )


def _zero_residual_physical_candidate(
    learner: RealRLTLearner,
    batch: dict[str, np.ndarray],
    *,
    next_state: bool = False,
) -> np.ndarray:
    prefix = "next_" if next_state else ""
    if _uses_persistent_candidate(learner):
        return np.asarray(batch[f"{prefix}a_base_filtered"], dtype=np.float32)
    return np.asarray(batch[f"{prefix}a_ref"], dtype=np.float32)


def _physical_candidate_residual_limit(
    learner: RealRLTLearner,
) -> np.ndarray:
    stored = np.asarray(learner.residual_limit, dtype=np.float32)
    if not _uses_persistent_candidate(learner):
        return stored
    limit = np.full_like(
        stored,
        float(learner.config.actor_residual_max_rad),
        dtype=np.float32,
    )
    if learner.config.freeze_gripper_residual:
        limit[..., -1] = 0.0
    else:
        limit[..., -1] = float(
            learner.config.actor_gripper_residual_max_close_m
        )
    return limit


def _nonnegative_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replay-npz", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split", choices=("train", "validation", "test", "all"), default="validation"
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--min-action-sensitivity", type=float, default=1e-7)
    parser.add_argument("--max-joint-residual-limit", type=float, default=0.05)
    parser.add_argument("--max-gripper-residual-limit", type=float, default=0.005)
    parser.add_argument("--required-action-schema-fingerprint")
    parser.add_argument("--max-rank1-fit-error-rad", type=_nonnegative_finite_float)
    parser.add_argument("--max-residual-abs-rad", type=_nonnegative_finite_float)
    parser.add_argument("--max-residual-d1-rad", type=_nonnegative_finite_float)
    parser.add_argument("--max-residual-d2-rad", type=_nonnegative_finite_float)
    parser.add_argument("--max-direction-cone-deg", type=_nonnegative_finite_float)
    parser.add_argument("--max-normalized-residual-step", type=float, default=0.15)
    parser.add_argument(
        "--max-cross-transition-residual-step-p95", type=float, default=0.5
    )
    parser.add_argument("--max-residual-saturation-rate", type=float, default=0.25)
    parser.add_argument(
        "--max-active-normalized-residual-step",
        type=_nonnegative_finite_float,
        help=(
            "Optional gate on the worst active action dimension's mask-aware "
            "chunk-internal normalized residual d1 mean. Frozen dimensions are excluded."
        ),
    )
    parser.add_argument(
        "--max-actor-joint-d1-p95-rad",
        type=_nonnegative_finite_float,
        help="Optional gate on the worst per-joint p95 d1 of the complete Actor command.",
    )
    parser.add_argument(
        "--max-chunk-boundary-normalized-residual-jump-p95",
        type=_nonnegative_finite_float,
        help=(
            "Optional gate on the exact C-step replan-boundary residual jump p95. "
            "Omitting this flag keeps the metric report-only."
        ),
    )
    parser.add_argument(
        "--max-chunk-boundary-actor-command-joint-d1-p95-rad",
        type=_nonnegative_finite_float,
        help=(
            "Optional gate on the worst Piper joint's p95 absolute Actor-command "
            "jump across exact C-step boundaries."
        ),
    )
    parser.add_argument("--incumbent-checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    from openpi.rlt.real.agent_jax import RealRLTLearner
    from openpi.rlt.real.agent_jax import ReplayBatchSampler
    from openpi.rlt.real.agent_jax import ReplaySamplingConfig
    from openpi.rlt.real.agent_jax import filter_replay_by_split
    from openpi.rlt.real.agent_jax import sha256_file

    args = parse_args()
    with np.load(args.replay_npz, allow_pickle=False) as archive:
        replay_all = {name: archive[name] for name in archive.files}
    replay = filter_replay_by_split(replay_all, args.split)
    learner = RealRLTLearner.load_checkpoint(
        args.checkpoint,
        expected_fingerprints={"replay_sha256": sha256_file(args.replay_npz)},
    )
    sampler = ReplayBatchSampler(replay, config=ReplaySamplingConfig(seed=args.seed))
    sample_count = min(args.samples, len(replay["reward"]))
    # Validation must not silently sample the same transition many times.  Use
    # a deterministic without-replacement subset and retain episode/t ordering
    # separately for sequence-level checks below.
    indices = np.random.default_rng(args.seed).permutation(len(replay["reward"]))[
        :sample_count
    ]
    batch = {
        key: np.asarray(replay[key][indices], dtype=np.float32)
        for key in (
            "z_rl",
            "state",
            "a_ref",
            "a_exec",
            "reward",
            "discount",
            "next_z_rl",
            "next_state",
            "next_a_ref",
        )
    }
    if _uses_persistent_candidate(learner):
        missing = sorted(
            set(
                PERSISTENT_CURRENT_CANDIDATE_FIELDS
                + PERSISTENT_NEXT_CANDIDATE_FIELDS
            ).difference(replay)
        )
        if missing:
            raise KeyError(
                "persistent-v2 validation replay is missing physical candidate "
                f"arrays: {missing}"
            )
        for key in (
            PERSISTENT_CURRENT_CANDIDATE_FIELDS
            + PERSISTENT_NEXT_CANDIDATE_FIELDS
        ):
            batch[key] = np.asarray(replay[key][indices], dtype=np.float32)
    batch["success_mask"] = sampler.success_mask[indices]

    action = _candidate_action(learner, batch)
    action_no_ref = _candidate_action(
        learner, batch, reference_visible=False
    )
    shuffled_z = np.roll(batch["z_rl"], shift=1, axis=0)
    action_shuffled_z = _candidate_action(
        learner, batch, z_rl=shuffled_z
    )
    zero_residual_candidate = _zero_residual_physical_candidate(learner, batch)
    stored_residual_limit = np.asarray(learner.residual_limit, dtype=np.float32)
    candidate_residual_limit = _physical_candidate_residual_limit(learner)
    q_exec = learner.q_values(
        batch["z_rl"], batch["state"], batch["a_ref"], batch["a_exec"]
    )
    q_ref = learner.q_values(
        batch["z_rl"],
        batch["state"],
        batch["a_ref"],
        zero_residual_candidate,
    )
    q_actor = learner.q_values(batch["z_rl"], batch["state"], batch["a_ref"], action)
    # Probe every trainable action dimension *inside* the Actor residual
    # envelope. A single out-of-domain joint-0 perturbation can otherwise let
    # a degenerate critic satisfy the sensitivity gate.
    probe_delta = 0.25 * candidate_residual_limit[None]
    probe_plus = zero_residual_candidate + probe_delta
    probe_minus = zero_residual_candidate - probe_delta
    q_probe_plus = learner.q_values(
        batch["z_rl"], batch["state"], batch["a_ref"], probe_plus
    )
    q_probe_minus = learner.q_values(
        batch["z_rl"], batch["state"], batch["a_ref"], probe_minus
    )

    next_action = _candidate_action(learner, batch, use_target=True, next_state=True)
    next_q = learner.q_values(
        batch["next_z_rl"],
        batch["next_state"],
        batch["next_a_ref"],
        next_action,
        use_target=True,
    )
    td_target = batch["reward"] + batch["discount"] * np.minimum(next_q[0], next_q[1])
    q_exec_min = np.minimum(q_exec[0], q_exec[1])
    q_ref_min = np.minimum(q_ref[0], q_ref[1])
    q_actor_min = np.minimum(q_actor[0], q_actor[1])
    actor_q_advantage = q_actor_min - q_ref_min
    success = np.asarray(batch["success_mask"], dtype=bool)
    success_failure_q_gap = None
    if np.any(success) and np.any(~success):
        success_failure_q_gap = float(
            np.mean(q_exec_min[success]) - np.mean(q_exec_min[~success])
        )

    q_action_sensitivity = float(
        0.5
        * (
            np.mean(np.abs(q_probe_plus[0] - q_probe_minus[0]))
            + np.mean(np.abs(q_probe_plus[1] - q_probe_minus[1]))
        )
    )
    residual = action - zero_residual_candidate
    residual_step = np.diff(residual, axis=1)
    residual_second_diff = np.diff(residual_step, axis=1)
    temporal_scale = np.maximum(
        candidate_residual_limit[1:], candidate_residual_limit[:-1]
    )
    normalized_residual_step_abs_mean = float(
        np.mean(np.abs(residual_step) / np.maximum(temporal_scale[None], 1e-8))
    )
    joint_limit_max = float(np.max(stored_residual_limit[..., :6]))
    gripper_limit_max = float(np.max(stored_residual_limit[..., 6]))
    physical_limit_contract = bool(
        joint_limit_max <= args.max_joint_residual_limit + 1e-7
        and gripper_limit_max <= args.max_gripper_residual_limit + 1e-7
    )
    residual_scale = np.maximum(candidate_residual_limit[None], 1e-8)
    saturated = np.abs(residual) >= 0.95 * residual_scale
    saturation_rate = float(np.mean(saturated))
    saturation_rate_per_action = np.mean(saturated, axis=(0, 1))
    gripper_frozen = bool(np.all(np.abs(residual[..., 6]) <= 1e-7))
    close_assist = bool(
        getattr(learner.config, "gripper_residual_mode", "frozen")
        == "close_only_persistent_v1"
        and not learner.config.freeze_gripper_residual
    )
    gripper_residual = np.asarray(residual[..., 6], dtype=np.float64)
    gripper_previous_carry = np.asarray(
        batch.get(
            "actor_persistent_previous_carry",
            np.zeros((len(gripper_residual), 7), dtype=np.float32),
        )[..., 6:7],
        dtype=np.float64,
    )
    gripper_carry = np.asarray(
        batch.get(
            "actor_persistent_carry_in",
            np.zeros((len(gripper_residual), 7), dtype=np.float32),
        )[..., 6:7],
        dtype=np.float64,
    )
    gripper_history = np.concatenate(
        [
            gripper_previous_carry,
            gripper_carry,
            gripper_residual,
            gripper_residual[:, -1:],
        ],
        axis=1,
    )
    gripper_d1_abs_max = float(np.max(np.abs(np.diff(gripper_history, axis=1))))
    gripper_d2_abs_max = float(
        np.max(np.abs(np.diff(gripper_history, n=2, axis=1)))
    )
    gripper_boundary_jump_abs_max = float(
        np.max(
            np.abs(
                gripper_residual[:, 0]
                - gripper_carry[:, 0]
            )
        )
    )
    gripper_target_min_m = float(np.min(action[..., 6]))
    gripper_target_max_m = float(np.max(action[..., 6]))
    gripper_close_only_contract = bool(
        not close_assist
        or (
            np.max(gripper_residual) <= 1.0e-7
            and np.min(gripper_residual)
            >= -float(learner.config.actor_gripper_residual_max_close_m)
            - 1.0e-7
            and gripper_d1_abs_max
            <= float(learner.config.actor_gripper_residual_d1_max_m) + 1.0e-7
            and gripper_d2_abs_max
            <= float(learner.config.actor_gripper_residual_d2_max_m) + 1.0e-7
            and gripper_boundary_jump_abs_max
            <= float(learner.config.actor_gripper_max_boundary_jump_m) + 1.0e-7
            and gripper_target_min_m
            >= float(learner.config.gripper_command_min_m) - 1.0e-7
            and gripper_target_max_m
            <= float(learner.config.gripper_command_max_m) + 1.0e-7
        )
    )
    # Persistent-v2 physically executes a carry-aware, filtered residual which
    # is intentionally no longer a pure rank-one bump.  Keep the rank-one
    # protocol check on the unchanged checkpoint head, while every Q/TD/BC and
    # temporal metric above/below uses the physical candidate.
    checkpoint_protocol_action = (
        learner.act(batch["z_rl"], batch["state"], batch["a_ref"])
        if _uses_persistent_candidate(learner)
        else action
    )
    checkpoint_protocol_residual = checkpoint_protocol_action - batch["a_ref"]
    rank1_metrics = _rank1_directional_contract_metrics(
        batch["a_ref"],
        checkpoint_protocol_residual,
        max_rank1_fit_error_rad=args.max_rank1_fit_error_rad,
        max_residual_abs_rad=args.max_residual_abs_rad,
        max_residual_d1_rad=args.max_residual_d1_rad,
        max_residual_d2_rad=args.max_residual_d2_rad,
        max_direction_cone_deg=args.max_direction_cone_deg,
        gripper_residual_mode=getattr(
            learner.config,
            "gripper_residual_mode",
            "frozen",
        ),
        max_gripper_residual_m=float(
            getattr(
                learner.config,
                "actor_gripper_residual_max_close_m",
                0.0,
            )
        ),
    )
    checkpoint_parameterization_contract = bool(
        learner.config.actor_residual_parameterization == "rank1_bump"
        and (
            learner.config.freeze_gripper_residual
            or close_assist
        )
    )
    action_schema_fingerprint = learner.fingerprints.get("action_schema")
    action_schema_fingerprint_contract = bool(
        args.required_action_schema_fingerprint is None
        or action_schema_fingerprint == args.required_action_schema_fingerprint
    )

    finite_action_inputs = _all_finite(
        batch["a_ref"],
        batch["a_exec"],
        batch["next_a_ref"],
        zero_residual_candidate,
        probe_plus,
        probe_minus,
    )
    finite_action_outputs = _all_finite(
        action, action_no_ref, action_shuffled_z, next_action
    )
    finite_q_values = _all_finite(
        *q_exec,
        *q_ref,
        *q_actor,
        *q_probe_plus,
        *q_probe_minus,
        *next_q,
    )
    finite_td_target = _all_finite(td_target)

    sequence_metrics = _sequence_metrics(
        replay,
        learner,
        max_normalized_step_p95=args.max_cross_transition_residual_step_p95,
        max_active_normalized_residual_step=args.max_active_normalized_residual_step,
        max_actor_joint_d1_p95_rad=args.max_actor_joint_d1_p95_rad,
        max_chunk_boundary_normalized_residual_jump_p95=(
            args.max_chunk_boundary_normalized_residual_jump_p95
        ),
        max_chunk_boundary_actor_command_joint_d1_p95_rad=(
            args.max_chunk_boundary_actor_command_joint_d1_p95_rad
        ),
    )
    incumbent_metrics = None
    if args.incumbent_checkpoint is not None:
        # An incumbent was normally trained on the previous replay snapshot;
        # evaluate it on the current held-out rows without requiring that old
        # replay hash to equal the candidate's refreshed replay hash.
        incumbent = RealRLTLearner.load_checkpoint(args.incumbent_checkpoint)
        incumbent_metrics = _sequence_metrics(
            replay,
            incumbent,
            max_normalized_step_p95=args.max_cross_transition_residual_step_p95,
        )
    optional_promotion_contracts = []
    if args.max_active_normalized_residual_step is not None:
        optional_promotion_contracts.append(
            "active_normalized_residual_temporal_step_contract"
        )
    # Absolute Actor-command d1 includes the Pi0.5 reference velocity.  Keep it
    # in the report for diagnosis, but do not use it to reject a residual Actor:
    # a zero-residual Actor must not fail because the base policy moved faster
    # than an experiment-calibrated threshold.
    if args.max_chunk_boundary_normalized_residual_jump_p95 is not None:
        optional_promotion_contracts.append("chunk_boundary_residual_jump_contract")
    # The offline t -> t+C pairs may cross independent Pi0.5 policy plans, so
    # their absolute-command jump is also report-only.  Promotion continues to
    # require residual-boundary continuity; the runtime separately enforces
    # its configured actual C10 entry-jump envelope before Actor control.
    directional_contract_arguments = (
        args.max_rank1_fit_error_rad,
        args.max_residual_abs_rad,
        args.max_residual_d1_rad,
        args.max_residual_d2_rad,
        args.max_direction_cone_deg,
    )
    if any(value is not None for value in directional_contract_arguments):
        optional_promotion_contracts.extend(
            [
                "rank1_checkpoint_parameterization_contract",
                "rank1_residual_contract",
                "direction_cone_contract",
                (
                    "gripper_close_only_contract"
                    if close_assist
                    else "gripper_residual_frozen"
                ),
            ]
        )
    if args.required_action_schema_fingerprint is not None:
        optional_promotion_contracts.append("action_schema_fingerprint_contract")
    # Standalone callers that omit the enhanced active-dimension gate retain
    # the historical 0.15 promotion behavior.  A caller that explicitly
    # enables the replacement gate keeps the legacy scalar in the report but
    # must not be rejected by both differently defined temporal contracts.
    legacy_normalized_temporal_contract_required = (
        args.max_active_normalized_residual_step is None
    )
    report = {
        "format": "openpi_real_rlt_actor_acceptance",
        "temporal_metrics_semantics_version": 4,
        "absolute_actor_command_metrics_role": "diagnostic_only_reference_inclusive",
        "actor_candidate_semantics": (
            "persistent_filtered_physical_candidate"
            if _uses_persistent_candidate(learner)
            else "legacy_raw_rank1_candidate"
        ),
        "actor_q_baseline_semantics": (
            "zero_residual_a_base_filtered"
            if _uses_persistent_candidate(learner)
            else "nominal_a_ref"
        ),
        "rank1_metrics_action_role": (
            "checkpoint_output_protocol_only"
            if _uses_persistent_candidate(learner)
            else "actor_candidate"
        ),
        "actor_bc_semantics": (
            "filtered_actual_residual_from_a_base_filtered"
            if _uses_persistent_candidate(learner)
            else "raw_candidate_residual_from_a_ref"
        ),
        "critic_behavior_action_semantics": "replay_a_exec",
        "critic_td_target_candidate_semantics": (
            "persistent_filtered_target_actor_candidate"
            if _uses_persistent_candidate(learner)
            else "legacy_smoothed_target_actor_candidate"
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "replay": str(args.replay_npz.resolve()),
        "replay_sha256": sha256_file(args.replay_npz),
        "replay_split": args.split,
        "update_step": learner.update_step,
        "samples": int(len(action)),
        "finite_action": bool(np.all(np.isfinite(action))),
        "finite_action_inputs": finite_action_inputs,
        "finite_action_outputs": (
            finite_action_outputs
            and bool(sequence_metrics["sequence_action_finite"])
            and (
                incumbent_metrics is None
                or bool(incumbent_metrics["sequence_action_finite"])
            )
        ),
        "finite_q_values": finite_q_values,
        "finite_td_target": finite_td_target,
        "residual_abs_mean": float(np.mean(np.abs(residual))),
        "residual_abs_max": float(np.max(np.abs(residual))),
        "residual_within_limit": bool(
            np.all(np.abs(residual) <= candidate_residual_limit[None] + 1e-6)
        ),
        "residual_limit_joint_max_rad": joint_limit_max,
        "residual_limit_gripper_max_m": gripper_limit_max,
        "physical_residual_limit_contract": physical_limit_contract,
        "gripper_residual_frozen": gripper_frozen,
        "gripper_close_assist_enabled": close_assist,
        "gripper_close_only_contract": gripper_close_only_contract,
        "gripper_residual_mean_m": float(np.mean(gripper_residual)),
        "gripper_residual_min_m": float(np.min(gripper_residual)),
        "gripper_residual_max_m": float(np.max(gripper_residual)),
        "gripper_residual_close_fraction": float(
            np.mean(gripper_residual < -1.0e-6)
        ),
        "gripper_residual_open_fraction": float(
            np.mean(gripper_residual > 1.0e-6)
        ),
        "gripper_residual_d1_abs_max_m": gripper_d1_abs_max,
        "gripper_residual_d2_abs_max_m": gripper_d2_abs_max,
        "gripper_boundary_jump_abs_max_m": gripper_boundary_jump_abs_max,
        "gripper_target_min_m": gripper_target_min_m,
        "gripper_target_max_m": gripper_target_max_m,
        "rank1_checkpoint_parameterization_contract": checkpoint_parameterization_contract,
        "action_schema_fingerprint": action_schema_fingerprint,
        "required_action_schema_fingerprint": args.required_action_schema_fingerprint,
        "action_schema_fingerprint_contract": action_schema_fingerprint_contract,
        **rank1_metrics,
        "normalized_residual_temporal_step_abs_mean": normalized_residual_step_abs_mean,
        "normalized_residual_temporal_step_contract": bool(
            normalized_residual_step_abs_mean <= args.max_normalized_residual_step
        ),
        "residual_temporal_step_abs_mean_per_action": np.mean(
            np.abs(residual_step), axis=(0, 1)
        ).tolist(),
        "residual_temporal_second_diff_abs_p95_per_action": np.percentile(
            np.abs(residual_second_diff), 95, axis=(0, 1)
        ).tolist(),
        "residual_saturation_rate": saturation_rate,
        "residual_saturation_rate_per_action": saturation_rate_per_action.tolist(),
        "residual_saturation_contract": bool(
            np.max(saturation_rate_per_action) <= args.max_residual_saturation_rate
        ),
        **sequence_metrics,
        "incumbent_sequence_metrics": incumbent_metrics,
        "optional_promotion_contracts": optional_promotion_contracts,
        "legacy_normalized_residual_temporal_step_promotion_required": (
            legacy_normalized_temporal_contract_required
        ),
        "reference_input_ablation_action_l1": float(
            np.mean(np.abs(action - action_no_ref))
        ),
        "shuffled_z_action_l1": float(np.mean(np.abs(action - action_shuffled_z))),
        "q_action_sensitivity_l1": q_action_sensitivity,
        "q1_q2_gap": float(np.mean(np.abs(q_exec[0] - q_exec[1]))),
        "validation_td_error_abs_mean": float(np.mean(np.abs(q_exec_min - td_target))),
        "validation_actor_bc_mse": float(np.mean(np.square(residual))),
        "actor_q_advantage_over_reference": float(np.mean(actor_q_advantage)),
        "actor_q_advantage_abs_p95": float(
            np.percentile(np.abs(actor_q_advantage), 95)
        ),
        "actor_q_advantage_over_executed": float(np.mean(q_actor_min - q_exec_min)),
        "reward1_reward0_exec_q_gap": success_failure_q_gap,
        # Backward-compatible report alias for older offline dashboards.
        "success_failure_exec_q_gap": success_failure_q_gap,
        "fingerprints": learner.fingerprints,
    }
    report["finite_key_metrics"] = _numeric_values_are_finite(report)
    report["passed"] = _acceptance_passed(
        report,
        min_action_sensitivity=args.min_action_sensitivity,
        optional_contracts=tuple(optional_promotion_contracts),
        require_legacy_normalized_temporal_contract=(
            legacy_normalized_temporal_contract_required
        ),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(2)


def _rank1_directional_contract_metrics(
    a_ref: np.ndarray,
    residual: np.ndarray,
    *,
    max_rank1_fit_error_rad: float | None,
    max_residual_abs_rad: float | None,
    max_residual_d1_rad: float | None,
    max_residual_d2_rad: float | None,
    max_direction_cone_deg: float | None,
    gripper_residual_mode: str = "frozen",
    max_gripper_residual_m: float = 0.0,
    motion_epsilon_rad: float = 1e-3,
) -> dict[str, object]:
    """Validate the exact C10 contract used by the runtime governor.

    All statistics are hard maxima over the validation subset.  Promotion is
    deliberately not based on a percentile: one malformed chunk is enough to
    violate an atomic real-robot Actor contract.
    """

    reference = np.asarray(a_ref, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    if reference.shape != residual.shape or residual.ndim != 3 or residual.shape[1:] != (10, 7):
        raise ValueError(
            "rank1 directional validation requires matching (sample, 10, 7) arrays, "
            f"got {reference.shape} and {residual.shape}"
        )
    window = np.asarray(
        [0.0, 0.2, 0.5, 0.8, 1.0, 1.0, 0.8, 0.5, 0.2, 0.0],
        dtype=np.float64,
    )
    joint_residual = residual[..., :6]
    direction = np.sum(joint_residual * window[None, :, None], axis=1) / np.sum(
        np.square(window)
    )
    reconstruction = window[None, :, None] * direction[:, None, :]
    rank1_fit_error = float(np.max(np.abs(joint_residual - reconstruction)))
    endpoint_abs_max = float(np.max(np.abs(joint_residual[:, (0, -1), :])))
    residual_abs_max = float(np.max(np.abs(joint_residual)))
    padded = np.pad(joint_residual, ((0, 0), (1, 1), (0, 0)), mode="constant")
    residual_d1_abs_max = float(np.max(np.abs(np.diff(padded, axis=1))))
    residual_d2_abs_max = float(np.max(np.abs(np.diff(padded, n=2, axis=1))))
    gripper_residual = residual[..., 6]
    gripper_residual_abs_max = float(np.max(np.abs(gripper_residual)))
    gripper_direction = np.sum(
        gripper_residual * window[None, :],
        axis=1,
    ) / np.sum(np.square(window))
    gripper_reconstruction = window[None, :] * gripper_direction[:, None]
    gripper_rank1_fit_error = float(
        np.max(np.abs(gripper_residual - gripper_reconstruction))
    )
    gripper_endpoint_abs_max = float(
        np.max(np.abs(gripper_residual[:, (0, -1)]))
    )
    if gripper_residual_mode == "close_only_persistent_v1":
        gripper_contract = bool(
            gripper_rank1_fit_error <= 1.0e-5 + 1.0e-9
            and gripper_endpoint_abs_max <= 1.0e-7
            and np.max(gripper_direction) <= 1.0e-7
            and np.min(gripper_direction)
            >= -float(max_gripper_residual_m) - 1.0e-9
        )
    else:
        gripper_contract = gripper_residual_abs_max <= 1.0e-7

    origin = np.zeros_like(reference[:, :1, :6])
    reference_velocity = np.diff(
        np.concatenate([origin, reference[..., :6]], axis=1), axis=1
    )
    actor_velocity = np.diff(
        np.concatenate([origin, reference[..., :6] + joint_residual], axis=1),
        axis=1,
    )
    reference_norm = np.linalg.norm(reference_velocity, axis=-1)
    actor_norm = np.linalg.norm(actor_velocity, axis=-1)
    dot = np.sum(reference_velocity * actor_velocity, axis=-1)
    moving = reference_norm >= motion_epsilon_rad
    cosine = np.full(reference_norm.shape, np.nan, dtype=np.float64)
    valid_cosine = moving & (actor_norm > 1e-12)
    cosine[valid_cosine] = (
        dot[valid_cosine]
        / (reference_norm[valid_cosine] * actor_norm[valid_cosine])
    )
    cone_cosine = None
    direction_violations = np.zeros(reference_norm.shape, dtype=np.bool_)
    if max_direction_cone_deg is not None:
        cone_cosine = math.cos(math.radians(float(max_direction_cone_deg)))
        direction_violations[moving] = (
            (actor_norm[moving] <= 1e-12)
            | (dot[moving] < -1e-9)
            | (cosine[moving] + 1e-9 < cone_cosine)
        )
        direction_violations[~moving] = actor_norm[~moving] > motion_epsilon_rad + 1e-9
    finite_cosines = cosine[np.isfinite(cosine)]

    rank1_contract = bool(
        (max_rank1_fit_error_rad is None or rank1_fit_error <= max_rank1_fit_error_rad + 1e-9)
        and endpoint_abs_max <= 1e-7
        and (max_residual_abs_rad is None or residual_abs_max <= max_residual_abs_rad + 1e-9)
        and (max_residual_d1_rad is None or residual_d1_abs_max <= max_residual_d1_rad + 1e-9)
        and (max_residual_d2_rad is None or residual_d2_abs_max <= max_residual_d2_rad + 1e-9)
        and gripper_contract
    )
    return {
        "rank1_fit_error_abs_max_rad": rank1_fit_error,
        "rank1_fit_error_threshold_rad": max_rank1_fit_error_rad,
        "rank1_endpoint_abs_max_rad": endpoint_abs_max,
        "rank1_joint_residual_abs_max_rad": residual_abs_max,
        "rank1_joint_residual_threshold_rad": max_residual_abs_rad,
        "rank1_joint_residual_d1_abs_max_rad": residual_d1_abs_max,
        "rank1_joint_residual_d1_threshold_rad": max_residual_d1_rad,
        "rank1_joint_residual_d2_abs_max_rad": residual_d2_abs_max,
        "rank1_joint_residual_d2_threshold_rad": max_residual_d2_rad,
        "rank1_gripper_residual_abs_max": gripper_residual_abs_max,
        "rank1_gripper_fit_error_abs_max_m": gripper_rank1_fit_error,
        "rank1_gripper_endpoint_abs_max_m": gripper_endpoint_abs_max,
        "rank1_gripper_direction_min_m": float(np.min(gripper_direction)),
        "rank1_gripper_direction_max_m": float(np.max(gripper_direction)),
        "rank1_gripper_contract": gripper_contract,
        "rank1_residual_contract": rank1_contract,
        "direction_cone_threshold_deg": max_direction_cone_deg,
        "direction_cone_min_cosine": (
            None if finite_cosines.size == 0 else float(np.min(finite_cosines))
        ),
        "direction_cone_required_cosine": cone_cosine,
        "direction_cone_violation_count": int(np.count_nonzero(direction_violations)),
        "direction_cone_contract": bool(
            max_direction_cone_deg is None or not np.any(direction_violations)
        ),
    }


def _sequence_metrics(
    replay: dict[str, np.ndarray],
    learner: RealRLTLearner,
    *,
    max_normalized_step_p95: float,
    max_active_normalized_residual_step: float | None = None,
    max_actor_joint_d1_p95_rad: float | None = None,
    max_chunk_boundary_normalized_residual_jump_p95: float | None = None,
    max_chunk_boundary_actor_command_joint_d1_p95_rad: float | None = None,
) -> dict[str, object]:
    """Evaluate legacy anchor continuity plus mask-aware C-step temporal metrics.

    ``cross_transition_*`` is retained for report and promotion compatibility.
    It compares the first residual at adjacent replay anchors and therefore is
    not an execution-chunk boundary metric when replay stride is smaller than
    the Actor chunk.  ``chunk_boundary_*`` below uses exact ``t -> t + C``
    pairs and the actual ``C - 1 -> 0`` stitch instead.
    """

    batch = {
        "z_rl": np.asarray(replay["z_rl"], dtype=np.float32),
        "state": np.asarray(replay["state"], dtype=np.float32),
        "a_ref": np.asarray(replay["a_ref"], dtype=np.float32),
    }
    if _uses_persistent_candidate(learner):
        missing = sorted(set(PERSISTENT_CURRENT_CANDIDATE_FIELDS).difference(replay))
        if missing:
            raise KeyError(
                "persistent-v2 sequence validation replay is missing physical "
                f"candidate arrays: {missing}"
            )
        for key in PERSISTENT_CURRENT_CANDIDATE_FIELDS:
            batch[key] = np.asarray(replay[key], dtype=np.float32)
    actions = _candidate_action(learner, batch)
    baseline = _zero_residual_physical_candidate(learner, batch)
    residual = actions - baseline
    first_residual = residual[:, 0, :]
    sequence_action_finite = _all_finite(actions, residual, first_residual)
    step_mask = _step_mask(
        replay, sample_count=len(actions), chunk_length=actions.shape[1]
    )
    active_temporal_metrics = _active_normalized_residual_temporal_metrics(
        residual,
        _physical_candidate_residual_limit(learner),
        step_mask=step_mask,
        step_mask_was_stored="step_mask" in replay,
    )
    residual_temporal_metrics = _chunk_joint_temporal_metrics(
        residual,
        step_mask=step_mask,
        prefix="residual",
    )
    actor_temporal_metrics = _chunk_joint_temporal_metrics(
        actions,
        step_mask=step_mask,
        prefix="actor_command",
    )
    active_temporal_value = active_temporal_metrics[
        "active_normalized_residual_temporal_step_abs_mean_action_max"
    ]
    active_temporal_metrics["active_normalized_residual_temporal_step_threshold"] = (
        max_active_normalized_residual_step
    )
    active_temporal_metrics["active_normalized_residual_temporal_step_contract"] = (
        _optional_maximum_contract(
            active_temporal_value, max_active_normalized_residual_step
        )
    )
    actor_d1_p95_max = actor_temporal_metrics[
        "actor_command_temporal_d1_abs_p95_joint_max"
    ]
    actor_temporal_metrics["actor_command_temporal_d1_p95_threshold_rad"] = (
        max_actor_joint_d1_p95_rad
    )
    actor_temporal_metrics["actor_command_temporal_d1_p95_contract"] = (
        _optional_maximum_contract(actor_d1_p95_max, max_actor_joint_d1_p95_rad)
    )
    boundary_metrics = _chunk_boundary_metrics(
        replay,
        actions=actions,
        residual=residual,
        residual_limit=_physical_candidate_residual_limit(learner),
        step_mask=step_mask,
        max_normalized_residual_jump_p95=max_chunk_boundary_normalized_residual_jump_p95,
        max_actor_command_joint_d1_p95_rad=(
            max_chunk_boundary_actor_command_joint_d1_p95_rad
        ),
    )
    episode_ids = np.asarray(
        replay.get("episode_id", np.repeat("unknown", len(actions)))
    ).astype(str)
    timesteps = np.asarray(replay.get("t", np.arange(len(actions))), dtype=np.int64)
    normalized_differences: list[np.ndarray] = []
    physical_differences: list[np.ndarray] = []
    scale = np.maximum(_physical_candidate_residual_limit(learner)[0], 1e-8)
    for episode_id in np.unique(episode_ids):
        episode_indices = np.flatnonzero(episode_ids == episode_id)
        episode_indices = episode_indices[np.argsort(timesteps[episode_indices])]
        if len(episode_indices) < 2:
            continue
        differences = np.diff(first_residual[episode_indices], axis=0)
        physical_differences.append(np.abs(differences))
        normalized_differences.append(np.abs(differences) / scale[None])
    if not normalized_differences:
        normalized_p95 = 0.0
        per_action_p95 = np.zeros(first_residual.shape[-1], dtype=np.float32)
    else:
        normalized = np.concatenate(normalized_differences, axis=0)
        physical = np.concatenate(physical_differences, axis=0)
        normalized_p95 = float(np.percentile(normalized, 95))
        per_action_p95 = np.percentile(physical, 95, axis=0)
    return {
        "sequence_action_finite": sequence_action_finite,
        "cross_transition_residual_step_normalized_p95": normalized_p95,
        "cross_transition_residual_step_abs_p95_per_action": per_action_p95.tolist(),
        "cross_transition_residual_step_contract": bool(
            normalized_p95 <= max_normalized_step_p95
        ),
        **active_temporal_metrics,
        **residual_temporal_metrics,
        **actor_temporal_metrics,
        **boundary_metrics,
    }


def _step_mask(
    replay: dict[str, np.ndarray],
    *,
    sample_count: int,
    chunk_length: int,
) -> np.ndarray:
    if "step_mask" not in replay:
        return np.ones((sample_count, chunk_length), dtype=np.bool_)
    mask = np.asarray(replay["step_mask"], dtype=np.bool_)
    if mask.shape != (sample_count, chunk_length):
        raise ValueError(
            f"step_mask must have shape {(sample_count, chunk_length)}, got {mask.shape}"
        )
    return mask


def _active_normalized_residual_temporal_metrics(
    residual: np.ndarray,
    residual_limit: np.ndarray,
    *,
    step_mask: np.ndarray | None = None,
    step_mask_was_stored: bool = False,
) -> dict[str, object]:
    """Compute chunk d1 only over valid steps and non-frozen action dimensions."""

    residual = np.asarray(residual, dtype=np.float32)
    residual_limit = np.asarray(residual_limit, dtype=np.float32)
    if residual.ndim != 3:
        raise ValueError(
            f"residual must have shape (sample, chunk, action), got {residual.shape}"
        )
    if residual_limit.shape != residual.shape[1:]:
        raise ValueError(
            f"residual_limit must have shape {residual.shape[1:]}, got {residual_limit.shape}"
        )
    if step_mask is None:
        step_mask = np.ones(residual.shape[:2], dtype=np.bool_)
    else:
        step_mask = np.asarray(step_mask, dtype=np.bool_)
    if step_mask.shape != residual.shape[:2]:
        raise ValueError(
            f"step_mask must have shape {residual.shape[:2]}, got {step_mask.shape}"
        )

    residual_d1 = np.abs(np.diff(residual, axis=1))
    scale = np.maximum(residual_limit[1:], residual_limit[:-1])
    active = scale > 1e-8
    valid_step_pairs = step_mask[:, 1:] & step_mask[:, :-1]
    valid = valid_step_pairs[:, :, None] & active[None]
    normalized = np.divide(
        residual_d1,
        scale[None],
        out=np.zeros_like(residual_d1),
        where=active[None],
    )
    values = normalized[valid]
    active_action_dimensions = (
        np.flatnonzero(np.any(active, axis=0)).astype(int).tolist()
    )
    per_action: list[float | None] = []
    for action_index in range(residual.shape[-1]):
        action_valid = valid_step_pairs & active[None, :, action_index]
        action_values = normalized[:, :, action_index][action_valid]
        per_action.append(
            None if action_values.size == 0 else float(np.mean(action_values))
        )
    return {
        "active_normalized_residual_temporal_step_abs_mean": (
            None if values.size == 0 else float(np.mean(values))
        ),
        "active_normalized_residual_temporal_step_value_count": int(values.size),
        "active_normalized_residual_temporal_step_active_action_dimensions": (
            active_action_dimensions
        ),
        "active_normalized_residual_temporal_step_abs_mean_per_action": per_action,
        "active_normalized_residual_temporal_step_abs_mean_action_max": (
            _optional_finite_max(per_action)
        ),
        "active_normalized_residual_temporal_step_valid_pair_count": int(
            np.count_nonzero(valid_step_pairs)
        ),
        "active_normalized_residual_temporal_step_mask_used": bool(
            step_mask_was_stored
        ),
    }


def _chunk_joint_temporal_metrics(
    value: np.ndarray,
    *,
    step_mask: np.ndarray | None,
    prefix: str,
) -> dict[str, object]:
    """Return per-Piper-joint d1/d2 statistics for valid positions in a chunk."""

    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 3:
        raise ValueError(
            f"value must have shape (sample, chunk, action), got {value.shape}"
        )
    if step_mask is None:
        step_mask = np.ones(value.shape[:2], dtype=np.bool_)
    else:
        step_mask = np.asarray(step_mask, dtype=np.bool_)
    if step_mask.shape != value.shape[:2]:
        raise ValueError(
            f"step_mask must have shape {value.shape[:2]}, got {step_mask.shape}"
        )

    d1_mask = step_mask[:, 1:] & step_mask[:, :-1]
    d2_mask = step_mask[:, 2:] & step_mask[:, 1:-1] & step_mask[:, :-2]
    d1 = np.diff(value, axis=1)[d1_mask]
    d2 = np.diff(value, n=2, axis=1)[d2_mask]
    joint_count = min(6, value.shape[-1])
    d1_stats = _absolute_per_joint_statistics(d1, joint_count=joint_count)
    d2_stats = _absolute_per_joint_statistics(d2, joint_count=joint_count)
    return {
        f"{prefix}_temporal_d1_valid_pair_count": int(np.count_nonzero(d1_mask)),
        f"{prefix}_temporal_d2_valid_triplet_count": int(np.count_nonzero(d2_mask)),
        f"{prefix}_temporal_d1_abs_p95_per_joint": d1_stats["p95"],
        f"{prefix}_temporal_d1_abs_p99_per_joint": d1_stats["p99"],
        f"{prefix}_temporal_d1_abs_max_per_joint": d1_stats["max"],
        f"{prefix}_temporal_d1_abs_p95_joint_max": _optional_finite_max(
            d1_stats["p95"]
        ),
        f"{prefix}_temporal_d2_abs_p95_per_joint": d2_stats["p95"],
    }


def _absolute_per_joint_statistics(
    differences: np.ndarray,
    *,
    joint_count: int,
) -> dict[str, list[float | None]]:
    differences = np.asarray(differences, dtype=np.float32)
    if differences.ndim != 2:
        differences = differences.reshape((-1, differences.shape[-1]))
    if len(differences) == 0:
        empty = [None] * joint_count
        return {
            "mean": empty.copy(),
            "p95": empty.copy(),
            "p99": empty.copy(),
            "max": empty.copy(),
        }
    absolute = np.abs(differences[:, :joint_count])
    return {
        "mean": np.mean(absolute, axis=0).astype(float).tolist(),
        "p95": np.percentile(absolute, 95, axis=0).astype(float).tolist(),
        "p99": np.percentile(absolute, 99, axis=0).astype(float).tolist(),
        "max": np.max(absolute, axis=0).astype(float).tolist(),
    }


def _chunk_boundary_metrics(
    replay: dict[str, np.ndarray],
    *,
    actions: np.ndarray,
    residual: np.ndarray,
    residual_limit: np.ndarray,
    step_mask: np.ndarray,
    max_normalized_residual_jump_p95: float | None,
    max_actor_command_joint_d1_p95_rad: float | None = None,
) -> dict[str, object]:
    """Evaluate canonical C-step stitching using exact same-episode ``t + C`` pairs.

    This is an offline canonical boundary, not proof that a particular runtime
    prefetch request used the same observation or policy-plan provenance.
    """

    actions = np.asarray(actions, dtype=np.float32)
    residual = np.asarray(residual, dtype=np.float32)
    residual_limit = np.asarray(residual_limit, dtype=np.float32)
    chunk_length = int(actions.shape[1])
    base: dict[str, object] = {
        "chunk_boundary_available": False,
        "chunk_boundary_gap_steps": chunk_length,
        "chunk_boundary_continuity_rule": (
            "same_episode_exact_t_plus_chunk_length_current_full_chunk_nonterminal_next_first_valid"
        ),
        "chunk_boundary_semantics": "canonical_offline_chunk_stitch_not_runtime_prefetch_provenance",
        "chunk_boundary_pair_count": 0,
        "chunk_boundary_duplicate_episode_t_key_count": 0,
        "chunk_boundary_missing_exact_gap_count": 0,
        "chunk_boundary_tail_without_future_count": 0,
        "chunk_boundary_terminal_start_count": 0,
        "chunk_boundary_invalid_step_mask_count": 0,
        "chunk_boundary_residual_jump_normalized_p95": None,
        "chunk_boundary_residual_jump_normalized_p95_per_action": [None]
        * actions.shape[-1],
        "chunk_boundary_residual_jump_abs_p95_per_action": [None] * actions.shape[-1],
        "chunk_boundary_residual_jump_active_action_dimensions": [],
        "chunk_boundary_residual_jump_threshold": max_normalized_residual_jump_p95,
        "chunk_boundary_residual_jump_contract": (
            None if max_normalized_residual_jump_p95 is None else False
        ),
        "chunk_boundary_actor_command_d1_p95_threshold_rad": (
            max_actor_command_joint_d1_p95_rad
        ),
        "chunk_boundary_actor_command_d1_abs_p95_joint_max": None,
        "chunk_boundary_actor_command_d1_p95_contract": (
            None if max_actor_command_joint_d1_p95_rad is None else False
        ),
    }
    required = {"episode_id", "t", "state", "a_ref", "a_exec"}
    missing = sorted(required.difference(replay))
    if missing:
        base["chunk_boundary_unavailable_reason"] = f"missing replay arrays: {missing}"
        return _add_empty_boundary_action_statistics(base, action_dim=actions.shape[-1])

    episode_ids = np.asarray(replay["episode_id"]).astype(str)
    timesteps = np.asarray(replay["t"], dtype=np.int64)
    done = np.asarray(
        replay.get("done", np.zeros(len(actions), dtype=np.bool_)), dtype=np.bool_
    )
    if episode_ids.shape != (len(actions),) or timesteps.shape != (len(actions),):
        raise ValueError(
            "episode_id and t must be one-dimensional and transition-aligned"
        )
    if done.shape != (len(actions),):
        raise ValueError("done must be one-dimensional and transition-aligned")

    reference_absolute, reference_source = _absolute_action_chunk(
        replay,
        training_key="a_ref",
        absolute_key="a_ref_absolute",
    )
    executed_absolute, executed_source = _absolute_action_chunk(
        replay,
        training_key="a_exec",
        absolute_key="a_exec_absolute",
    )
    actor_absolute = np.asarray(actions, dtype=np.float32).copy()
    state = np.asarray(replay["state"], dtype=np.float32)
    joint_count = min(6, actor_absolute.shape[-1], state.shape[-1])
    actor_absolute[..., :joint_count] += state[:, None, :joint_count]
    base["chunk_boundary_reference_absolute_source"] = reference_source
    base["chunk_boundary_executed_absolute_source"] = executed_source

    pairs: list[tuple[int, int]] = []
    duplicate_count = 0
    missing_count = 0
    tail_count = 0
    terminal_count = 0
    invalid_mask_count = 0
    for episode_id in np.unique(episode_ids):
        episode_indices = np.flatnonzero(episode_ids == episode_id)
        by_t: dict[int, list[int]] = {}
        for index in episode_indices:
            by_t.setdefault(int(timesteps[index]), []).append(int(index))
        duplicate_count += sum(len(indices) > 1 for indices in by_t.values())
        if not by_t:
            continue
        max_t = max(by_t)
        for timestep, current_indices in by_t.items():
            if len(current_indices) != 1:
                continue
            target_t = timestep + chunk_length
            target_indices = by_t.get(target_t)
            if target_indices is None:
                if target_t <= max_t:
                    missing_count += 1
                else:
                    tail_count += 1
                continue
            if len(target_indices) != 1:
                continue
            current = current_indices[0]
            following = target_indices[0]
            if bool(done[current]):
                terminal_count += 1
                continue
            if not bool(np.all(step_mask[current, :chunk_length])) or not bool(
                step_mask[following, 0]
            ):
                invalid_mask_count += 1
                continue
            pairs.append((current, following))

    base.update(
        {
            "chunk_boundary_duplicate_episode_t_key_count": int(duplicate_count),
            "chunk_boundary_missing_exact_gap_count": int(missing_count),
            "chunk_boundary_tail_without_future_count": int(tail_count),
            "chunk_boundary_terminal_start_count": int(terminal_count),
            "chunk_boundary_invalid_step_mask_count": int(invalid_mask_count),
            "chunk_boundary_pair_count": int(len(pairs)),
        }
    )
    if not pairs:
        base["chunk_boundary_unavailable_reason"] = (
            "no valid exact t_plus_chunk_length pairs"
        )
        return _add_empty_boundary_action_statistics(base, action_dim=actions.shape[-1])

    current = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    following = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    residual_jump = residual[following, 0] - residual[current, chunk_length - 1]
    reference_jump = (
        reference_absolute[following, 0] - reference_absolute[current, chunk_length - 1]
    )
    actor_jump = (
        actor_absolute[following, 0] - actor_absolute[current, chunk_length - 1]
    )
    executed_jump = (
        executed_absolute[following, 0] - executed_absolute[current, chunk_length - 1]
    )
    boundary_scale = np.maximum(residual_limit[0], residual_limit[chunk_length - 1])
    active = boundary_scale > 1e-8
    normalized_per_action: list[float | None] = []
    for action_index in range(actions.shape[-1]):
        if not active[action_index]:
            normalized_per_action.append(None)
            continue
        values = np.abs(residual_jump[:, action_index]) / boundary_scale[action_index]
        normalized_per_action.append(float(np.percentile(values, 95)))
    # Gate the worst trainable axis rather than letting quiet/frozen axes hide
    # a single oscillating Piper joint in a grand mean/percentile.
    normalized_p95 = _optional_finite_max(normalized_per_action)
    per_action_p95 = (
        np.percentile(np.abs(residual_jump), 95, axis=0).astype(float).tolist()
    )
    base.update(
        {
            "chunk_boundary_available": True,
            "chunk_boundary_residual_jump_normalized_p95": normalized_p95,
            "chunk_boundary_residual_jump_normalized_p95_per_action": normalized_per_action,
            "chunk_boundary_residual_jump_abs_p95_per_action": per_action_p95,
            "chunk_boundary_residual_jump_active_action_dimensions": (
                np.flatnonzero(active).astype(int).tolist()
            ),
            "chunk_boundary_residual_jump_contract": _optional_maximum_contract(
                normalized_p95,
                max_normalized_residual_jump_p95,
            ),
        }
    )
    for name, differences in (
        ("reference", reference_jump),
        ("actor_command", actor_jump),
        ("executed_command", executed_jump),
    ):
        base.update(_boundary_action_statistics(differences, prefix=name))
    actor_boundary_p95_max = _optional_finite_max(
        base["chunk_boundary_actor_command_d1_abs_p95_per_joint"]
    )
    base["chunk_boundary_actor_command_d1_abs_p95_joint_max"] = (
        actor_boundary_p95_max
    )
    base["chunk_boundary_actor_command_d1_p95_contract"] = (
        _optional_maximum_contract(
            actor_boundary_p95_max,
            max_actor_command_joint_d1_p95_rad,
        )
    )
    return base


def _absolute_action_chunk(
    replay: dict[str, np.ndarray],
    *,
    training_key: str,
    absolute_key: str,
) -> tuple[np.ndarray, str]:
    if absolute_key in replay:
        return np.asarray(replay[absolute_key], dtype=np.float32), "stored"
    training = np.asarray(replay[training_key], dtype=np.float32)
    state = np.asarray(replay["state"], dtype=np.float32)
    if training.ndim != 3 or state.ndim != 2 or training.shape[0] != state.shape[0]:
        raise ValueError(
            f"cannot reconstruct {absolute_key} from non-aligned training actions/state"
        )
    absolute = training.copy()
    joint_count = min(6, training.shape[-1], state.shape[-1])
    absolute[..., :joint_count] += state[:, None, :joint_count]
    return absolute, "reconstructed_from_training_coordinates_and_state"


def _boundary_action_statistics(
    differences: np.ndarray,
    *,
    prefix: str,
) -> dict[str, object]:
    stats = _absolute_per_joint_statistics(
        differences, joint_count=min(6, differences.shape[-1])
    )
    return {
        f"chunk_boundary_{prefix}_d1_abs_mean_per_joint": stats["mean"],
        f"chunk_boundary_{prefix}_d1_abs_p95_per_joint": stats["p95"],
        f"chunk_boundary_{prefix}_d1_abs_p99_per_joint": stats["p99"],
        f"chunk_boundary_{prefix}_d1_abs_max_per_joint": stats["max"],
    }


def _add_empty_boundary_action_statistics(
    metrics: dict[str, object],
    *,
    action_dim: int,
) -> dict[str, object]:
    joint_count = min(6, action_dim)
    for prefix in ("reference", "actor_command", "executed_command"):
        for statistic in ("mean", "p95", "p99", "max"):
            metrics[f"chunk_boundary_{prefix}_d1_abs_{statistic}_per_joint"] = [
                None
            ] * joint_count
    return metrics


def _optional_finite_max(values: list[float | None]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return None if not finite else max(finite)


def _optional_maximum_contract(value: object, threshold: float | None) -> bool | None:
    if threshold is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(math.isfinite(numeric) and numeric <= threshold)


def _all_finite(*values: Any) -> bool:
    try:
        return all(bool(np.all(np.isfinite(np.asarray(value)))) for value in values)
    except (TypeError, ValueError):
        return False


def _numeric_values_are_finite(value: Any) -> bool:
    """Recursively reject NaN/Inf while allowing report metadata and absent optional metrics."""

    if value is None or isinstance(value, (str, bytes, bool, np.bool_)):
        return True
    if isinstance(value, dict):
        return all(_numeric_values_are_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_numeric_values_are_finite(item) for item in value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return math.isfinite(float(value))
    if isinstance(value, np.ndarray):
        return _all_finite(value)
    return False


def _acceptance_passed(
    report: dict[str, Any],
    *,
    min_action_sensitivity: float,
    optional_contracts: tuple[str, ...] = (),
    require_legacy_normalized_temporal_contract: bool = True,
) -> bool:
    try:
        sensitivity = float(report.get("q_action_sensitivity_l1", float("nan")))
        minimum = float(min_action_sensitivity)
        gripper_limit = float(report.get("residual_limit_gripper_max_m", float("nan")))
    except (TypeError, ValueError):
        return False
    if (
        not math.isfinite(sensitivity)
        or not math.isfinite(minimum)
        or minimum < 0.0
        or not math.isfinite(gripper_limit)
    ):
        return False
    required_true = [
        "finite_action",
        "finite_action_inputs",
        "finite_action_outputs",
        "finite_q_values",
        "finite_td_target",
        "finite_key_metrics",
        "residual_within_limit",
        "physical_residual_limit_contract",
        "residual_saturation_contract",
        "cross_transition_residual_step_contract",
    ]
    if require_legacy_normalized_temporal_contract:
        required_true.append("normalized_residual_temporal_step_contract")
    return bool(
        all(report.get(name) is True for name in required_true)
        and all(report.get(name) is True for name in optional_contracts)
        and (gripper_limit > 1e-8 or report.get("gripper_residual_frozen") is True)
        and sensitivity >= minimum
    )


if __name__ == "__main__":
    main()
