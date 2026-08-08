#!/usr/bin/env python3
"""Episode-boundary online RLT update: replay refresh, JAX A-C, validate, promote."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Any

import numpy as np

try:
    from . import persistent_v2_contract as _persistent_v2
except ImportError:  # Direct executable invocation.
    import persistent_v2_contract as _persistent_v2

ACTOR_DIRECTION_STATIC_THRESHOLD_RAD = (
    _persistent_v2.ACTOR_DIRECTION_STATIC_THRESHOLD_RAD
)
ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD = (
    _persistent_v2.ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD
)
ACTOR_MIN_PROJECTION_SCALE = _persistent_v2.ACTOR_MIN_PROJECTION_SCALE
ACTOR_PROJECTION_SCALE_STEPS = _persistent_v2.ACTOR_PROJECTION_SCALE_STEPS
PERSISTENT_V2_ACTION_SCHEMA_FINGERPRINT = (
    _persistent_v2.ACTION_SCHEMA_FINGERPRINT
)
PERSISTENT_V2_ACTOR_EXECUTION_PROFILE = (
    _persistent_v2.ACTOR_EXECUTION_PROFILE
)
PERSISTENT_V2_ACTOR_PROJECTION_PROFILE = (
    _persistent_v2.ACTOR_PROJECTION_PROFILE
)
PERSISTENT_V2_CONTROL_HZ = _persistent_v2.CONTROL_HZ
DEFAULT_MIN_NEW_COMMITTED_EPISODES = (
    _persistent_v2.DEFAULT_MIN_NEW_COMMITTED_EPISODES
)
PERSISTENT_V2_EXECUTION_FILTER_PROFILE = (
    _persistent_v2.EXECUTION_FILTER_PROFILE
)
PERSISTENT_V2_EXECUTION_FILTER_TAU_S = (
    _persistent_v2.EXECUTION_FILTER_TAU_S
)
PERSISTENT_GOVERNOR_FINGERPRINT = (
    _persistent_v2.PERSISTENT_GOVERNOR_FINGERPRINT
)
GRIPPER_RESIDUAL_MODE = _persistent_v2.GRIPPER_RESIDUAL_MODE
GRIPPER_RESIDUAL_MAX_CLOSE_M = _persistent_v2.GRIPPER_RESIDUAL_MAX_CLOSE_M
GRIPPER_RESIDUAL_D1_MAX_M = _persistent_v2.GRIPPER_RESIDUAL_D1_MAX_M
GRIPPER_RESIDUAL_D2_MAX_M = _persistent_v2.GRIPPER_RESIDUAL_D2_MAX_M
GRIPPER_MAX_BOUNDARY_JUMP_M = _persistent_v2.GRIPPER_MAX_BOUNDARY_JUMP_M
GRIPPER_COMMAND_MIN_M = _persistent_v2.GRIPPER_COMMAND_MIN_M
GRIPPER_COMMAND_MAX_M = _persistent_v2.GRIPPER_COMMAND_MAX_M
GRIPPER_RELEASE_REFERENCE_M = _persistent_v2.GRIPPER_RELEASE_REFERENCE_M
GRIPPER_RELEASE_DELTA_M = _persistent_v2.GRIPPER_RELEASE_DELTA_M
HUMAN_GRIPPER_Q_FILTER_MODE = "critic_min_advantage_v1"
HUMAN_GRIPPER_Q_FILTER_MARGIN = 0.0
audit_persistent_v2_episode = _persistent_v2.audit_persistent_v2_episode
LEGACY_STATE_FORMAT = "openpi_piper_online_rlt_state_v2"
PERSISTENT_V2_STATE_FORMAT = "openpi_piper_online_rlt_state_persistent_gripper_v3"
BOOTSTRAP_LINEAGE_MODE = "persistent_gripper_v3_bootstrap_warm_start"
BOOTSTRAP_REPLAY_POLICY = (
    "immutable_migrated_v5_warmup_plus_persistent_v5_online"
)
FRESH_ZERO_LINEAGE_MODE = "persistent_gripper_v3_fresh_zero"
FRESH_ZERO_REPLAY_POLICY = "fresh_persistent_v5_online_only"

try:
    import fcntl
except (
    ImportError
):  # pragma: no cover - production is Linux; keeps pure unit tests portable.
    fcntl = None


BASE_FINGERPRINT = (
    "full20k_step20000_metadata_sha256_"
    "14d9cac129ec7ce91f2e5aab3f5bfac06172c8fb70709f01850fb8e8215870e5"
)
TOKEN_FINGERPRINT = "2f2e1e6bbcae8f08217ec7ba0b88088bfa44627be495035319deb84e06052b49"
PHASE_FINGERPRINT = "8c5b443edd3f399529680ef5e4c5dffcdee4af2ae2f224f6da4152237c9a50dc"
ACTION_SCHEMA_FINGERPRINT = (
    "piper_joint_delta_v5_c10_n10_stride10_behavior_ref50_"
    "rank1_joint_r005_d1_0015_d2_001_cone15_gripper_close_knot_r005"
)
LEGACY_ACTOR_EXECUTION_PROFILE = "rank1_bump_v1"
ACTOR_PROJECTION_PROFILE = (
    "rank1_joint_v1_r005_d1_0015_d2_001_cone15_"
    "scale33_min020_gripper_close_knot_r005"
)
LEARNER_GPU_INDEX = 0
LEARNER_GPU_MEMORY_FRACTION = 0.20
LEARNER_GPU_MIN_FREE_MIB = 7_168
_HISTORICAL_V3_BOOTSTRAP_INCREMENTAL_ONLY_KEYS = frozenset(
    {
        "actor_persistent_planned_residual",
        "actor_gripper_release_intent",
        "actor_filtered_actual_gripper_residual_max",
        "actor_filtered_actual_gripper_residual_d1_max_m",
        "actor_filtered_actual_gripper_residual_d2_max_m",
        "actor_filtered_actual_gripper_boundary_jump_max_m",
    }
)

# These promotion gates were calibrated by comparing the clean30 beta=10 and
# beta=40 candidate sweeps.  They are experiment-specific admission criteria,
# not physical Piper safety limits or mathematical guarantees.
ACTOR_VALIDATION_GATE_PROFILE = "rank1_residual_runtime_boundary_v2"
DEFAULT_MAX_ACTIVE_NORMALIZED_RESIDUAL_STEP = 0.30
DEFAULT_MAX_ACTOR_JOINT_D1_P95_RAD = 0.025
DEFAULT_MAX_CHUNK_BOUNDARY_NORMALIZED_RESIDUAL_JUMP_P95 = 1.75
DEFAULT_MAX_CHUNK_BOUNDARY_ACTOR_COMMAND_JOINT_D1_P95_RAD = math.radians(3.0)
ACTOR_VALIDATION_GATE_SPECS = (
    (
        "max_active_normalized_residual_step",
        "active_normalized_residual_temporal_step_contract",
        "active_normalized_residual_temporal_step_threshold",
    ),
    (
        "max_chunk_boundary_normalized_residual_jump_p95",
        "chunk_boundary_residual_jump_contract",
        "chunk_boundary_residual_jump_threshold",
    ),
)


def _parser() -> argparse.ArgumentParser:
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session-root",
        type=Path,
        default=os.environ.get("RLT_SESSION_ROOT"),
        required=False,
    )
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--warmup-episodes", type=int, default=10)
    parser.add_argument("--min-success", type=int, default=2)
    parser.add_argument("--min-failure", type=int, default=2)
    parser.add_argument(
        "--min-success-human-episodes",
        type=int,
        default=0,
        help=(
            "Deprecated compatibility guard. Gripper-v3 requires this to be "
            "zero so reward-negative admitted-human episodes are not excluded."
        ),
    )
    parser.add_argument(
        "--min-admitted-human-episodes",
        type=int,
        default=0,
        help=(
            "Require this many operator-admitted episodes containing Pika "
            "intervention, independent of terminal reward label."
        ),
    )
    parser.add_argument("--update-every", type=int, default=2)
    parser.add_argument(
        "--min-warmup-transitions",
        type=int,
        default=10,
        help="Minimum audited gate-active trainable transitions before the first update.",
    )
    parser.add_argument(
        "--utd",
        type=float,
        default=1.0,
        help="Learner updates per newly added train transition.",
    )
    parser.add_argument("--min-update-steps", type=int, default=1)
    parser.add_argument("--max-update-steps", type=int, default=1250)
    parser.add_argument("--initial-steps", type=int, default=1250)
    parser.add_argument("--update-steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--beta-bc", type=float, default=10.0)
    parser.add_argument(
        "--beta-human-bc",
        type=float,
        default=0.0,
        help="Additional masked intervention BC weight; original Eq.(5) reference BC remains active.",
    )
    parser.add_argument(
        "--beta-human-gripper-bc",
        type=float,
        default=1.0,
        help=(
            "Q-filtered admitted-human intervention imitation weight for the "
            "absolute gripper dimension; reward 1/0 labels both remain in Critic."
        ),
    )
    parser.add_argument(
        "--human-gripper-bc-scale-m",
        type=float,
        default=GRIPPER_RESIDUAL_MAX_CLOSE_M,
    )
    parser.add_argument(
        "--human-gripper-q-filter-mode",
        default=HUMAN_GRIPPER_Q_FILTER_MODE,
    )
    parser.add_argument(
        "--human-gripper-q-filter-margin",
        type=float,
        default=HUMAN_GRIPPER_Q_FILTER_MARGIN,
    )
    parser.add_argument("--reference-dropout", type=float, default=0.5)
    parser.add_argument("--target-policy-noise-std", type=float, default=0.1)
    parser.add_argument("--target-policy-noise-clip", type=float, default=0.2)
    parser.add_argument("--residual-max", type=float, default=0.005)
    parser.add_argument("--residual-d1-max-rad", type=float, default=0.0015)
    parser.add_argument("--residual-d2-max-rad", type=float, default=0.001)
    parser.add_argument("--direction-cone-deg", type=float, default=15.0)
    parser.add_argument(
        "--action-schema-fingerprint",
        default=ACTION_SCHEMA_FINGERPRINT,
    )
    parser.add_argument(
        "--execution-action-schema-fingerprint",
        help=(
            "Persistent-v2 replay/execution schema. The existing "
            "--action-schema-fingerprint remains the legacy Actor model "
            "response schema."
        ),
    )
    parser.add_argument(
        "--actor-projection-profile",
        default=ACTOR_PROJECTION_PROFILE,
    )
    parser.add_argument(
        "--actor-execution-profile",
        default=LEGACY_ACTOR_EXECUTION_PROFILE,
    )
    parser.add_argument("--execution-filter-profile")
    parser.add_argument("--execution-filter-tau-s", type=float)
    parser.add_argument("--control-hz", type=float)
    parser.add_argument(
        "--actor-live-max-boundary-jump-rad",
        type=float,
        default=ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD,
    )
    parser.add_argument(
        "--actor-projection-scale-steps",
        type=int,
        default=ACTOR_PROJECTION_SCALE_STEPS,
    )
    parser.add_argument(
        "--actor-min-projection-scale",
        type=float,
        default=ACTOR_MIN_PROJECTION_SCALE,
    )
    parser.add_argument(
        "--actor-direction-static-threshold-rad",
        type=float,
        default=ACTOR_DIRECTION_STATIC_THRESHOLD_RAD,
    )
    parser.add_argument(
        "--actor-governor-fingerprint",
        default=PERSISTENT_GOVERNOR_FINGERPRINT,
    )
    parser.add_argument(
        "--warm-start-actor-checkpoint",
        type=Path,
        help=(
            "Persistent-v2 first update only: copy Actor parameters one way; "
            "Critic/targets/optimizers/update/RNG remain fresh."
        ),
    )
    parser.add_argument(
        "--allow-warm-start-objective-migration",
        action="store_true",
        help=(
            "Persistent-v2 first update only: explicitly allow beta_bc/"
            "beta_human_bc to differ from the source Actor checkpoint. The "
            "source and target values remain auditable in the migration report."
        ),
    )
    parser.add_argument(
        "--min-new-persistent-committed-episodes",
        type=int,
        default=DEFAULT_MIN_NEW_COMMITTED_EPISODES,
    )
    parser.add_argument(
        "--gripper-residual-max",
        type=float,
        default=GRIPPER_RESIDUAL_MAX_CLOSE_M,
    )
    parser.add_argument(
        "--gripper-residual-mode",
        default=GRIPPER_RESIDUAL_MODE,
        choices=(GRIPPER_RESIDUAL_MODE,),
    )
    parser.add_argument(
        "--gripper-residual-d1-max-m",
        type=float,
        default=GRIPPER_RESIDUAL_D1_MAX_M,
    )
    parser.add_argument(
        "--gripper-residual-d2-max-m",
        type=float,
        default=GRIPPER_RESIDUAL_D2_MAX_M,
    )
    parser.add_argument(
        "--gripper-max-boundary-jump-m",
        type=float,
        default=GRIPPER_MAX_BOUNDARY_JUMP_M,
    )
    parser.add_argument(
        "--gripper-command-min-m",
        type=float,
        default=GRIPPER_COMMAND_MIN_M,
    )
    parser.add_argument(
        "--gripper-command-max-m",
        type=float,
        default=GRIPPER_COMMAND_MAX_M,
    )
    parser.add_argument(
        "--gripper-release-reference-m",
        type=float,
        default=GRIPPER_RELEASE_REFERENCE_M,
    )
    parser.add_argument(
        "--gripper-release-delta-m",
        type=float,
        default=GRIPPER_RELEASE_DELTA_M,
    )
    parser.add_argument(
        "--freeze-gripper-residual",
        action="store_true",
        help="Unsupported in the gripper-close v3 updater; retained only for fail-closed CLI diagnostics.",
    )
    parser.add_argument("--success-fraction", type=float, default=0.5)
    parser.add_argument("--human-fraction", type=float, default=0.5)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--max-validation-td-error", type=float, default=0.5)
    parser.add_argument("--max-actor-q-advantage", type=float, default=0.5)
    parser.add_argument(
        "--min-reward1-reward0-exec-q-gap",
        "--min-success-failure-exec-q-gap",
        dest="min_reward1_reward0_exec_q_gap",
        type=float,
        default=0.0,
        help=(
            "Require held-out reward-1 execution rows to have no lower mean Q "
            "than reward-0 rows. The success/failure spelling is a legacy alias."
        ),
    )
    parser.add_argument(
        "--max-active-normalized-residual-step",
        type=float,
        default=DEFAULT_MAX_ACTIVE_NORMALIZED_RESIDUAL_STEP,
        help=(
            "Experiment-calibrated maximum worst-action normalized residual d1 mean; "
            "this is not a physical safety theorem."
        ),
    )
    parser.add_argument(
        "--max-actor-joint-d1-p95-rad",
        type=float,
        default=DEFAULT_MAX_ACTOR_JOINT_D1_P95_RAD,
        help=(
            "Experiment-calibrated maximum complete-Actor per-joint d1 p95 in radians; "
            "this is not a physical safety theorem."
        ),
    )
    parser.add_argument(
        "--max-chunk-boundary-normalized-residual-jump-p95",
        type=float,
        default=DEFAULT_MAX_CHUNK_BOUNDARY_NORMALIZED_RESIDUAL_JUMP_P95,
        help=(
            "Experiment-calibrated maximum worst-action C=10 boundary residual jump p95; "
            "this is not a physical safety theorem."
        ),
    )
    parser.add_argument(
        "--max-chunk-boundary-actor-command-joint-d1-p95-rad",
        type=float,
        default=DEFAULT_MAX_CHUNK_BOUNDARY_ACTOR_COMMAND_JOINT_D1_P95_RAD,
        help=(
            "Reject candidates whose worst-joint p95 absolute Actor command jump "
            "across C=10 boundaries exceeds the existing 3-degree execution limit."
        ),
    )
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8001)
    parser.add_argument(
        "--workspace", type=Path, default=home / "openpi_jax_piper_lora_v1_20260707"
    )
    parser.add_argument("--runtime", type=Path, default=home / "piper_jax_inference_v1")
    parser.add_argument(
        "--phase-checkpoint",
        type=Path,
        default=home
        / "rlt_phase_classifiers/greenblock_box_resnet18_v4_manual_intervals/phase_classifier.pt",
    )
    parser.add_argument(
        "--selected-checkpoint-file",
        type=Path,
        default=home / "openpi_rlt/online_current/selected_actor_checkpoint.txt",
    )
    parser.add_argument("--shadow-service", default="openpi-rlt-shadow-policy.service")
    parser.add_argument(
        "--quarantine-registry",
        type=Path,
        help="JSON registry of episode IDs that must never enter replay (defaults under state-root).",
    )
    parser.add_argument(
        "--enrichment-cache",
        type=Path,
        help="Fresh per-gate-row RL-token cache; logged a_ref remains the execution-time reference.",
    )
    parser.add_argument(
        "--allow-logged-token-promotion",
        action="store_true",
        help="Emergency/debug override: permit promotion with temporally repeated logged z_rl values.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.session_root is None:
        raise ValueError("--session-root or RLT_SESSION_ROOT is required")
    args.session_root = args.session_root.expanduser().resolve()
    args.state_root = (
        (args.state_root or args.session_root / ".online_rlt").expanduser().resolve()
    )
    _validate_args(args)
    args.state_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.state_root / "update.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        report = run_update(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def run_update(args: argparse.Namespace) -> dict[str, Any]:
    state_path = args.state_root / "online_state.json"
    state = _load_json(state_path, default={})
    persistent_v2 = _is_persistent_v2(args)
    if persistent_v2:
        _validate_persistent_v2_state(args, state)
    # The v3 lineage owns one immutable, explicitly migrated replay built from
    # the previously audited 30 warm-up episodes.  It has the *new* v5 action
    # schema and remains in every later update so gripper imitation is not
    # forgotten.  This is not the legacy/frozen-schema merge forbidden by v2.
    frozen_base_replay_value = state.get("bootstrap_gripper_replay")
    frozen_base_replay = (
        Path(frozen_base_replay_value).expanduser().resolve()
        if frozen_base_replay_value
        else None
    )
    frozen_base_episode_ids = set(
        _state_episode_ids(state, "bootstrap_gripper_episode_ids") or ()
    )
    bootstrap_lineage = (
        state.get("lineage_mode") == BOOTSTRAP_LINEAGE_MODE
    )
    fresh_zero_lineage = (
        state.get("lineage_mode") == FRESH_ZERO_LINEAGE_MODE
    )
    if persistent_v2 and bootstrap_lineage and (
        frozen_base_replay_value is None or not frozen_base_episode_ids
    ):
        raise ValueError(
            "persistent gripper-v3 requires its immutable migrated warm-up replay"
        )
    if frozen_base_replay is not None:
        if not frozen_base_replay.is_file():
            raise ValueError(f"frozen base replay is missing: {frozen_base_replay}")
        replay_episode_ids = _replay_episode_ids(frozen_base_replay)
        if replay_episode_ids != frozen_base_episode_ids:
            raise ValueError(
                "bootstrap gripper replay episode IDs do not match online_state.json"
            )
        expected_sha = str(state.get("bootstrap_gripper_replay_sha256", ""))
        if not expected_sha or _sha256(frozen_base_replay) != expected_sha:
            raise ValueError("bootstrap gripper replay SHA256 mismatch")

    quarantine_path = (
        getattr(args, "quarantine_registry", None)
        or args.state_root / "episode_quarantine.json"
    )
    quarantine_path = Path(quarantine_path).expanduser().resolve()
    quarantine = _load_quarantine_registry(quarantine_path)
    quarantine_ids = set(quarantine)
    invalid_completed_episodes: list[dict[str, str]] = []
    completed_episode_paths = _completed_episodes(
        args.session_root,
        quarantine_ids=quarantine_ids,
        invalid_reports=invalid_completed_episodes,
    )
    episodes: list[Path] = []
    episode_audit: list[dict[str, Any]] = []
    for path in completed_episode_paths:
        if path.parent.name in frozen_base_episode_ids:
            continue
        try:
            audit = _audit_episode(
                path,
                expected_action_schema=(
                    getattr(args, "execution_action_schema_fingerprint", None)
                    if persistent_v2
                    else getattr(args, "action_schema_fingerprint", None)
                ),
                expected_projection_profile=getattr(args, "actor_projection_profile", None),
                expected_execution_profile=getattr(
                    args, "actor_execution_profile", None
                ),
                expected_filter_profile=getattr(
                    args, "execution_filter_profile", None
                ),
                expected_filter_tau_s=getattr(
                    args, "execution_filter_tau_s", None
                ),
                expected_control_hz=getattr(args, "control_hz", None),
                persistent_v2=persistent_v2,
            )
        except Exception as exc:
            if persistent_v2:
                raise ValueError(
                    "persistent-v2 completed episode failed closed; it cannot "
                    f"be silently mixed/skipped: {path}: {type(exc).__name__}: {exc}"
                ) from exc
            # One corrupt/legacy episode must not permanently wedge every
            # later online update. Exclude it from this replay, keep the raw
            # files untouched, and surface the exact reason for review.
            invalid_completed_episodes.append(
                {
                    "episode_id": path.parent.name,
                    "episode_jsonl": str(path),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        episodes.append(path)
        episode_audit.append(audit)
    incremental_trainable_audit = [
        item for item in episode_audit if item["trainable_transitions"] > 0
    ]
    incremental_trainable_episode_ids = [
        str(item["episode_id"]) for item in incremental_trainable_audit
    ]
    trainable_episode_ids = sorted(
        frozen_base_episode_ids.union(incremental_trainable_episode_ids)
    )
    base_quality = (
        _replay_overall_quality(frozen_base_replay)
        if frozen_base_replay is not None
        else {
            "episodes": 0,
            "successes": 0,
            "failures": 0,
            "human_episodes": 0,
            "success_human_episodes": 0,
            "failure_human_episodes": 0,
            "human_transitions": 0,
            "transitions": 0,
        }
    )
    successes = base_quality["successes"] + sum(
        item["reward"] == 1.0 for item in incremental_trainable_audit
    )
    failures = base_quality["failures"] + sum(
        item["reward"] == 0.0 for item in incremental_trainable_audit
    )
    human_episodes = base_quality["human_episodes"] + sum(
        item["human_eligible_rows"] > 0 for item in incremental_trainable_audit
    )
    success_human_episodes = sum(
        item["reward"] == 1.0 and item["human_trainable_transitions"] > 0
        for item in incremental_trainable_audit
    ) + base_quality["success_human_episodes"]
    failure_human_episodes = base_quality["failure_human_episodes"] + sum(
        item["reward"] == 0.0 and item["human_eligible_rows"] > 0
        for item in incremental_trainable_audit
    )
    audited_trainable_transitions = base_quality["transitions"] + sum(
        int(item["trainable_transitions"]) for item in incremental_trainable_audit
    )
    audited_human_transitions = base_quality["human_transitions"] + sum(
        int(item["human_trainable_transitions"]) for item in incremental_trainable_audit
    )
    latest_checkpoint = state.get("latest_checkpoint")
    warm_start_actor_checkpoint = (
        state.get("initial_actor_warm_start_checkpoint")
        or (
            str(args.warm_start_actor_checkpoint)
            if getattr(args, "warm_start_actor_checkpoint", None)
            else None
        )
    )
    explicit_trained_ids = _state_episode_ids(state, "trained_episode_ids")
    explicit_attempted_ids = _state_episode_ids(state, "last_attempt_episode_ids")
    trained_episode_ids, trained_ids_migrated = _migrate_legacy_episode_ids(
        explicit_trained_ids,
        legacy_count=int(state.get("last_update_episode_count", 0)),
        current_ids=trainable_episode_ids,
    )
    attempted_episode_ids, attempted_ids_migrated = _migrate_legacy_episode_ids(
        explicit_attempted_ids,
        legacy_count=int(
            state.get("last_attempt_episode_count", len(trained_episode_ids))
        ),
        current_ids=trainable_episode_ids,
    )
    new_episode_ids = sorted(
        set(trainable_episode_ids).difference(attempted_episode_ids)
    )
    quarantined_trained_ids = sorted(
        set(explicit_trained_ids or ()).intersection(quarantine_ids)
    )
    legacy_quarantine_ambiguity = bool(
        latest_checkpoint is not None
        and quarantine_ids
        and explicit_trained_ids is None
    )
    min_warmup_transitions = int(getattr(args, "min_warmup_transitions", 1))
    ready_initial = (
        len(trainable_episode_ids) >= args.warmup_episodes
        and audited_trainable_transitions >= min_warmup_transitions
        and successes >= args.min_success
        and failures >= args.min_failure
        and human_episodes
        >= int(getattr(args, "min_admitted_human_episodes", 0))
    )
    if persistent_v2:
        required_persistent_episodes = max(
            int(args.warmup_episodes),
            int(args.min_new_persistent_committed_episodes),
            int(state.get("min_new_persistent_committed_episodes", 0)),
        )
        ready_initial = bool(
            ready_initial
            and len(incremental_trainable_episode_ids)
            >= required_persistent_episodes
        )
    else:
        required_persistent_episodes = 0
    should_update = (
        latest_checkpoint is None
        and ready_initial
        and (not attempted_episode_ids or len(new_episode_ids) >= args.update_every)
    ) or (latest_checkpoint is not None and len(new_episode_ids) >= args.update_every)
    logged_token_alignment_degraded = getattr(args, "enrichment_cache", None) is None
    quality_mix = {
        "reward_positive_episodes": successes,
        "reward_negative_episodes": failures,
        "admitted_human_episodes": human_episodes,
        "reward_positive_human_episodes": success_human_episodes,
        "reward_negative_human_episodes": failure_human_episodes,
        "human_trainable_transitions": audited_human_transitions,
    }
    base_report = {
        "format": "openpi_piper_online_rlt_update_v1",
        "session_root": str(args.session_root),
        "episodes": len(trainable_episode_ids),
        "rewarded_episodes": len(frozen_base_episode_ids) + len(episode_audit),
        "trainable_episode_ids": trainable_episode_ids,
        "new_trainable_episode_ids": new_episode_ids,
        "audited_trainable_transitions": audited_trainable_transitions,
        "successes": successes,
        "failures": failures,
        "human_intervention_episodes": human_episodes,
        "reward_positive_episodes": successes,
        "reward_negative_episodes": failures,
        "admitted_human_episodes": human_episodes,
        "success_human_episodes": success_human_episodes,
        "failure_human_episodes": failure_human_episodes,
        "reward_positive_human_episodes": success_human_episodes,
        "reward_negative_human_episodes": failure_human_episodes,
        "quality_mix": quality_mix,
        "quality_mix_label": (
            f"R+{successes}/R-{failures}/H{human_episodes}"
        ),
        "warmup_episodes": args.warmup_episodes,
        "min_success_human_episodes": int(
            getattr(args, "min_success_human_episodes", 0)
        ),
        "min_admitted_human_episodes": int(
            getattr(args, "min_admitted_human_episodes", 0)
        ),
        "min_warmup_transitions": min_warmup_transitions,
        "trained_episode_ids": sorted(trained_episode_ids),
        "last_attempt_episode_ids": sorted(attempted_episode_ids),
        "legacy_episode_id_state_migrated": trained_ids_migrated
        or attempted_ids_migrated,
        "latest_checkpoint": latest_checkpoint,
        "actor_execution_profile": str(
            getattr(
                args,
                "actor_execution_profile",
                LEGACY_ACTOR_EXECUTION_PROFILE,
            )
        ),
        "execution_filter_profile": getattr(
            args, "execution_filter_profile", None
        ),
        "execution_filter_tau_s": getattr(
            args, "execution_filter_tau_s", None
        ),
        "control_hz": getattr(args, "control_hz", None),
        "actor_live_max_boundary_jump_rad": getattr(
            args, "actor_live_max_boundary_jump_rad", None
        ),
        "actor_projection_scale_steps": getattr(
            args, "actor_projection_scale_steps", None
        ),
        "actor_min_projection_scale": getattr(
            args, "actor_min_projection_scale", None
        ),
        "actor_direction_static_threshold_rad": getattr(
            args, "actor_direction_static_threshold_rad", None
        ),
        "actor_governor_fingerprint": getattr(
            args, "actor_governor_fingerprint", None
        ),
        "persistent_v2": persistent_v2,
        "min_new_persistent_committed_episodes": (
            required_persistent_episodes if persistent_v2 else None
        ),
        "new_persistent_committed_episode_count": (
            len(incremental_trainable_episode_ids) if persistent_v2 else None
        ),
        "warm_start_actor_checkpoint": (
            str(warm_start_actor_checkpoint)
            if warm_start_actor_checkpoint is not None
            else None
        ),
        "legacy_replay_training_rows": 0 if persistent_v2 else None,
        "quarantine_registry": str(quarantine_path),
        "quarantined_episode_ids": sorted(quarantine_ids),
        "quarantine_reasons": quarantine,
        "invalid_completed_episodes": invalid_completed_episodes,
        "logged_token_alignment_degraded": logged_token_alignment_degraded,
        "enrichment_cache": str(args.enrichment_cache)
        if getattr(args, "enrichment_cache", None)
        else None,
        "episode_audit": episode_audit,
        "bootstrap_gripper_replay": str(frozen_base_replay)
        if frozen_base_replay is not None
        else None,
        "bootstrap_gripper_episode_count": len(frozen_base_episode_ids),
        "deprecated_fixed_step_arguments": {
            "initial_steps": getattr(args, "initial_steps", None),
            "update_steps": getattr(args, "update_steps", None),
            "used_for_scheduling": False,
        },
        "learner_objective": {
            "beta_reference_bc": float(getattr(args, "beta_bc", 10.0)),
            "beta_human_bc": float(getattr(args, "beta_human_bc", 0.0)),
            "beta_human_gripper_bc": float(
                getattr(args, "beta_human_gripper_bc", 1.0)
            ),
            "human_gripper_bc_scale_m": float(
                getattr(
                    args,
                    "human_gripper_bc_scale_m",
                    GRIPPER_RESIDUAL_MAX_CLOSE_M,
                )
            ),
            "human_gripper_supervision_scope": (
                "all_admitted_human_reward_labels_preserved"
            ),
            "human_gripper_q_filter_mode": str(
                getattr(
                    args,
                    "human_gripper_q_filter_mode",
                    HUMAN_GRIPPER_Q_FILTER_MODE,
                )
            ),
            "human_gripper_q_filter_margin": float(
                getattr(
                    args,
                    "human_gripper_q_filter_margin",
                    HUMAN_GRIPPER_Q_FILTER_MARGIN,
                )
            ),
            "warm_start_objective_migration_authorized": bool(
                getattr(args, "allow_warm_start_objective_migration", False)
            ),
            "reference_dropout": float(getattr(args, "reference_dropout", 0.5)),
            "target_policy_noise_std": float(
                getattr(args, "target_policy_noise_std", 0.1)
            ),
            "target_policy_noise_clip": float(
                getattr(args, "target_policy_noise_clip", 0.2)
            ),
            "freeze_gripper_residual": bool(
                getattr(args, "freeze_gripper_residual", False)
            ),
            "gripper_residual_mode": str(args.gripper_residual_mode),
            "actor_gripper_residual_max_close_m": float(
                args.gripper_residual_max
            ),
            "actor_gripper_residual_d1_max_m": float(
                args.gripper_residual_d1_max_m
            ),
            "actor_gripper_residual_d2_max_m": float(
                args.gripper_residual_d2_max_m
            ),
            "actor_gripper_max_boundary_jump_m": float(
                args.gripper_max_boundary_jump_m
            ),
            "gripper_command_range_m": [
                float(args.gripper_command_min_m),
                float(args.gripper_command_max_m),
            ],
            "actor_residual_parameterization": "rank1_bump",
            "actor_residual_max_rad": float(getattr(args, "residual_max", 0.005)),
            "actor_residual_d1_max_rad": float(getattr(args, "residual_d1_max_rad", 0.0015)),
            "actor_residual_d2_max_rad": float(getattr(args, "residual_d2_max_rad", 0.001)),
            "actor_direction_cone_deg": float(getattr(args, "direction_cone_deg", 15.0)),
        },
        "actor_model_action_schema_fingerprint": str(
            getattr(args, "action_schema_fingerprint", ACTION_SCHEMA_FINGERPRINT)
        ),
        "execution_action_schema_fingerprint": (
            str(args.execution_action_schema_fingerprint)
            if persistent_v2
            else None
        ),
        "actor_projection_profile": str(
            getattr(args, "actor_projection_profile", ACTOR_PROJECTION_PROFILE)
        ),
        "actor_validation_gates": _actor_validation_gate_report(args),
    }
    if legacy_quarantine_ambiguity or quarantined_trained_ids:
        block_reason = (
            "legacy state has no stable trained_episode_ids, so quarantine contamination cannot be ruled out"
            if legacy_quarantine_ambiguity
            else "the incumbent was trained on one or more now-quarantined episodes"
        )
        incumbent_block = _block_incumbent_for_clean_retrain(
            args=args,
            state=state,
            state_path=state_path,
            reason=block_reason,
            latest_checkpoint=latest_checkpoint,
        )
        report = {
            **base_report,
            "outcome": "incumbent_requires_clean_retrain_after_quarantine",
            "updated": False,
            "quarantined_trained_episode_ids": quarantined_trained_ids,
            "reason": block_reason,
            "incumbent_block": incumbent_block,
        }
        _record_event(args.state_root, report)
        return report
    if not should_update:
        outcome = (
            "waiting_for_quality_warmup"
            if latest_checkpoint is None
            else "waiting_for_update_interval"
        )
        report = {**base_report, "outcome": outcome, "updated": False}
        _record_event(args.state_root, report)
        return report
    if args.dry_run:
        report = {**base_report, "outcome": "dry_run_would_update", "updated": False}
        _record_event(args.state_root, report)
        return report

    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(
        [
            str(args.workspace / "src"),
            str(args.workspace / "packages/openpi-client/src"),
            str(args.runtime),
            env.get("PYTHONPATH", ""),
        ]
    )
    python = args.workspace / ".venv/bin/python"
    trainable_id_set = set(trainable_episode_ids)
    trainable_episodes = [
        path
        for path, audit in zip(episodes, episode_audit)
        if (
            audit["episode_id"] in trainable_id_set
            and audit["episode_id"] not in frozen_base_episode_ids
        )
    ]
    replay_build_episodes = list(trainable_episodes)
    incremental_replay_base: Path | None = None
    if (
        persistent_v2
        and frozen_base_replay is None
        and latest_checkpoint is not None
        and trained_episode_ids
    ):
        previous_replay_value = state.get("latest_replay")
        previous_replay = (
            Path(str(previous_replay_value)).expanduser().resolve()
            if previous_replay_value
            else None
        )
        current_incremental_ids = set(incremental_trainable_episode_ids)
        if (
            previous_replay is not None
            and previous_replay.is_file()
            and trained_episode_ids.issubset(current_incremental_ids)
            and _replay_episode_ids(previous_replay) == trained_episode_ids
            and (
                not state.get("latest_replay_sha256")
                or _sha256(previous_replay)
                == state.get("latest_replay_sha256")
            )
        ):
            delta_ids = current_incremental_ids.difference(
                trained_episode_ids
            )
            if delta_ids:
                incremental_replay_base = previous_replay
                replay_build_episodes = [
                    path
                    for path, audit in zip(episodes, episode_audit)
                    if audit["episode_id"] in delta_ids
                ]
                if {
                    str(audit["episode_id"])
                    for path, audit in zip(episodes, episode_audit)
                    if path in replay_build_episodes
                } != delta_ids:
                    raise ValueError(
                        "incremental replay could not resolve every new "
                        "trainable episode"
                    )
    episode_set_tag = _episode_set_tag(trainable_episode_ids)
    replay_dir = (
        args.state_root
        / "replays"
        / f"episodes_{len(trainable_episodes):04d}_{episode_set_tag}"
    )
    replay_dir.mkdir(parents=True, exist_ok=True)
    prepare_log = replay_dir / "prepare.log"
    replay_source_description = (
        "exact execution-time a_ref plus fresh per-row z_rl enrichment"
        if getattr(args, "enrichment_cache", None)
        else "exact execution-time a_ref plus degraded logged z_rl (promotion will be blocked)"
    )
    print(
        f"[online-rlt] {len(trainable_episodes)} gate-active episodes passed the live replay audit; "
        f"building replay from {replay_source_description}",
        flush=True,
    )
    if incremental_replay_base is not None:
        print(
            "[online-rlt] incremental replay: reusing "
            f"{len(trained_episode_ids)} immutable trained episodes and "
            f"building {len(replay_build_episodes)} new episode(s)",
            flush=True,
        )
    _run(
        _logged_replay_prepare_command(
            episodes=replay_build_episodes,
            args=args,
            python=python,
            replay_dir=replay_dir,
        ),
        env=env,
        log_path=prepare_log,
    )
    if incremental_replay_base is not None:
        incremental_replay_path = replay_dir / "incremental_replay.npz"
        replay_path = replay_dir / "replay.npz"
        replay_path.replace(incremental_replay_path)
        _merge_replays(
            incremental_replay_base,
            incremental_replay_path,
            replay_path,
        )
        print(
            "[online-rlt] merged prior accepted replay with the strict "
            f"new-episode replay: {replay_path}",
            flush=True,
        )
    elif frozen_base_replay is not None:
        incremental_replay_path = replay_dir / "incremental_replay.npz"
        replay_path = replay_dir / "replay.npz"
        replay_path.replace(incremental_replay_path)
        _merge_replays(
            frozen_base_replay,
            incremental_replay_path,
            replay_path,
        )
        print(
            "[online-rlt] merged immutable v5 gripper warm-up replay with "
            f"incremental replay: {replay_path}",
            flush=True,
        )
    print(f"[online-rlt] replay prepared: {replay_dir / 'replay.npz'}", flush=True)
    replay_path = replay_dir / "replay.npz"
    audit_path = replay_dir / "audit.json"
    print(
        "[online-rlt] auditing replay coordinate/action/terminal contracts", flush=True
    )
    _run(
        [
            python,
            args.workspace / "scripts/piper_rlt/tools/audit_real_rlt_replay.py",
            "--replay-npz",
            replay_path,
            "--output-json",
            audit_path,
        ],
        env=env,
        log_path=replay_dir / "audit.stdout.json",
    )

    replay_counts = _replay_transition_counts(replay_path)
    replay_split_quality = _replay_split_quality(replay_path)
    base_report = {**base_report, "replay_split_quality": replay_split_quality}
    train_transition_count = replay_counts["train"]
    last_train_transition_count = _last_train_transition_count(state)
    if train_transition_count < last_train_transition_count:
        block_reason = (
            "current clean replay has fewer train transitions than the incumbent replay"
        )
        incumbent_block = _block_incumbent_for_clean_retrain(
            args=args,
            state=state,
            state_path=state_path,
            reason=block_reason,
            latest_checkpoint=latest_checkpoint,
        )
        report = {
            **base_report,
            "outcome": "clean_retrain_required_after_replay_regression",
            "updated": False,
            "replay": str(replay_path),
            "replay_transition_counts": replay_counts,
            "last_train_transition_count": last_train_transition_count,
            "reason": block_reason,
            "incumbent_block": incumbent_block,
        }
        _record_event(args.state_root, report)
        return report
    train_quality = replay_split_quality["train"]
    train_quality_ready = bool(
        int(train_quality["transitions"]) >= min_warmup_transitions
        and int(train_quality["success_episodes"]) >= args.min_success
        and int(train_quality["failure_episodes"]) >= args.min_failure
        and int(train_quality["human_episodes"])
        >= int(getattr(args, "min_admitted_human_episodes", 0))
    )
    if not train_quality_ready:
        report = {
            **base_report,
            "outcome": "waiting_for_train_split_quality",
            "updated": False,
            "replay": str(replay_path),
            "replay_transition_counts": replay_counts,
            "reason": (
                "the actual train split does not yet satisfy transition/"
                "reward-positive/reward-negative/admitted-human gates"
            ),
        }
        _record_event(args.state_root, report)
        return report
    new_train_transitions = train_transition_count - last_train_transition_count
    if new_train_transitions <= 0:
        report = {
            **base_report,
            "outcome": "no_new_train_transitions",
            "updated": False,
            "replay": str(replay_path),
            "replay_transition_counts": replay_counts,
            "last_train_transition_count": last_train_transition_count,
        }
        _record_event(args.state_root, report)
        return report

    learner_dir = args.state_root / "learner"
    learner_dir.mkdir(parents=True, exist_ok=True)
    training_steps = _compute_training_steps(
        new_train_transitions,
        utd=float(getattr(args, "utd", 1.0)),
        minimum=int(getattr(args, "min_update_steps", 1)),
        maximum=int(getattr(args, "max_update_steps", 1250)),
    )
    try:
        learner_gpu = _learner_gpu_memory_preflight()
    except RuntimeError as exc:
        report = {
            **base_report,
            "outcome": "gpu_memory_guard_blocked_keep_previous_policy",
            "updated": False,
            "replay": str(replay_path),
            "replay_transition_counts": replay_counts,
            "training_steps": training_steps,
            "new_train_transitions": new_train_transitions,
            "reason": str(exc),
        }
        _record_event(args.state_root, report)
        return report
    base_report = {**base_report, "learner_gpu_preflight": learner_gpu}
    print(
        f"[online-rlt] training JAX Actor-Critic for {training_steps} steps "
        f"({new_train_transitions} new train transitions, UTD={float(getattr(args, 'utd', 1.0)):g}); "
        f"CUDA GPU {learner_gpu['index']} free={learner_gpu['free_mib']} MiB, "
        f"learner_pool={LEARNER_GPU_MEMORY_FRACTION:.0%}; "
        f"progress log: {learner_dir / f'train_{episode_set_tag}.log'}",
        flush=True,
    )
    warm_start_report_path: Path | None = None
    training_command: list[Any] = [
        python,
        args.workspace / "scripts/train_real_rlt_jax.py",
        "--replay-npz",
        replay_path,
        "--output-dir",
        learner_dir,
        "--steps",
        str(training_steps),
        "--batch-size",
        str(args.batch_size),
        "--split",
        "train",
        "--success-fraction",
        str(args.success_fraction),
        "--human-fraction",
        str(args.human_fraction),
        "--beta-bc",
        str(args.beta_bc),
        "--beta-human-bc",
        str(getattr(args, "beta_human_bc", 0.0)),
        "--beta-human-gripper-bc",
        str(getattr(args, "beta_human_gripper_bc", 1.0)),
        "--human-gripper-bc-scale-m",
        str(
            getattr(
                args,
                "human_gripper_bc_scale_m",
                GRIPPER_RESIDUAL_MAX_CLOSE_M,
            )
        ),
        "--human-gripper-q-filter-mode",
        str(
            getattr(
                args,
                "human_gripper_q_filter_mode",
                HUMAN_GRIPPER_Q_FILTER_MODE,
            )
        ),
        "--human-gripper-q-filter-margin",
        str(
            getattr(
                args,
                "human_gripper_q_filter_margin",
                HUMAN_GRIPPER_Q_FILTER_MARGIN,
            )
        ),
        "--reference-dropout",
        str(args.reference_dropout),
        "--target-policy-noise-std",
        str(getattr(args, "target_policy_noise_std", 0.1)),
        "--target-policy-noise-clip",
        str(getattr(args, "target_policy_noise_clip", 0.2)),
        "--residual-max",
        str(getattr(args, "residual_max", 0.005)),
        "--actor-residual-parameterization",
        "rank1_bump",
        "--actor-residual-max-rad",
        str(getattr(args, "residual_max", 0.005)),
        "--actor-residual-d1-max-rad",
        str(getattr(args, "residual_d1_max_rad", 0.0015)),
        "--actor-residual-d2-max-rad",
        str(getattr(args, "residual_d2_max_rad", 0.001)),
        "--actor-direction-cone-deg",
        str(getattr(args, "direction_cone_deg", 15.0)),
        "--gripper-residual-max",
        str(args.gripper_residual_max),
        "--no-freeze-gripper-residual",
        "--gripper-residual-mode",
        str(args.gripper_residual_mode),
        "--gripper-residual-d1-max-m",
        str(args.gripper_residual_d1_max_m),
        "--gripper-residual-d2-max-m",
        str(args.gripper_residual_d2_max_m),
        "--gripper-boundary-max-m",
        str(args.gripper_max_boundary_jump_m),
        "--gripper-command-min-m",
        str(args.gripper_command_min_m),
        "--gripper-command-max-m",
        str(args.gripper_command_max_m),
        "--gripper-release-reference-m",
        str(args.gripper_release_reference_m),
        "--gripper-release-delta-m",
        str(args.gripper_release_delta_m),
        "--save-every",
        str(training_steps),
        "--log-every",
        str(max(10, min(100, training_steps))),
        "--base-checkpoint-fingerprint",
        BASE_FINGERPRINT,
        "--rl-token-fingerprint",
        TOKEN_FINGERPRINT,
        "--phase-fingerprint",
        PHASE_FINGERPRINT,
        "--action-schema-fingerprint",
        str(
            getattr(args, "execution_action_schema_fingerprint", None)
            if persistent_v2
            else getattr(
                args, "action_schema_fingerprint", ACTION_SCHEMA_FINGERPRINT
            )
        ),
        "--require-gpu",
    ]
    if getattr(args, "freeze_gripper_residual", False):
        raise ValueError("gripper-close v3 cannot freeze the gripper residual")
    if persistent_v2:
        training_command.extend(
            [
                "--actor-execution-profile",
                PERSISTENT_V2_ACTOR_EXECUTION_PROFILE,
                "--execution-filter-profile",
                PERSISTENT_V2_EXECUTION_FILTER_PROFILE,
                "--execution-filter-tau-s",
                str(PERSISTENT_V2_EXECUTION_FILTER_TAU_S),
                "--actor-max-boundary-jump-rad",
                str(ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD),
                "--actor-direction-static-threshold-rad",
                str(ACTOR_DIRECTION_STATIC_THRESHOLD_RAD),
                "--actor-projection-scale-steps",
                str(ACTOR_PROJECTION_SCALE_STEPS),
                "--actor-min-projection-scale",
                str(ACTOR_MIN_PROJECTION_SCALE),
                "--actor-governor-fingerprint",
                PERSISTENT_GOVERNOR_FINGERPRINT,
            ]
        )
    if fresh_zero_lineage:
        fresh_zero_manifest = args.state_root / "fresh_zero_manifest.json"
        if not fresh_zero_manifest.is_file():
            raise ValueError(
                "fresh-zero lineage manifest is missing before learner training"
            )
        training_command.extend(
            ["--fresh-zero-lineage-manifest", str(fresh_zero_manifest)]
        )
    if latest_checkpoint is not None:
        training_command.extend(
            ["--resume", latest_checkpoint, "--allow-replay-refresh"]
        )
    elif persistent_v2 and warm_start_actor_checkpoint is not None:
        warm_start_report_path = (
            learner_dir / f"warm_start_{episode_set_tag}.json"
        )
        training_command.extend(
            _warm_start_training_arguments(
                checkpoint=Path(str(warm_start_actor_checkpoint)),
                report_path=warm_start_report_path,
                allow_objective_migration=bool(
                    getattr(
                        args,
                        "allow_warm_start_objective_migration",
                        False,
                    )
                ),
            )
        )
    _run(
        training_command,
        env=_training_env(env),
        log_path=learner_dir / f"train_{episode_set_tag}.log",
    )
    warm_start_report = None
    if warm_start_report_path is not None:
        warm_start_report = _validate_actor_only_warm_start_report(
            warm_start_report_path,
            expected_source_checkpoint=Path(
                str(warm_start_actor_checkpoint)
            ).expanduser().resolve(),
            expected_beta_bc=float(args.beta_bc),
            expected_beta_human_bc=float(
                getattr(args, "beta_human_bc", 0.0)
            ),
            allow_objective_migration=bool(
                getattr(args, "allow_warm_start_objective_migration", False)
            ),
        )
    checkpoint = (
        learner_dir / (learner_dir / "latest.txt").read_text(encoding="utf-8").strip()
    )

    validation_path = learner_dir / f"validation_{episode_set_tag}.json"
    print(f"[online-rlt] validating Actor candidate: {checkpoint}", flush=True)
    validation_command = _actor_validation_command(
        python=python,
        args=args,
        checkpoint=checkpoint,
        replay_path=replay_path,
        validation_path=validation_path,
        validation_seed=len(trainable_episodes),
        incumbent_checkpoint=latest_checkpoint,
    )
    _run(
        validation_command,
        env=_validation_env(env),
        log_path=learner_dir / f"validation_{episode_set_tag}.stdout.json",
        accepted_returncodes=(0, 2),
    )
    validation = _load_json(validation_path)
    try:
        _validate_promotion(validation, args)
        if logged_token_alignment_degraded and not getattr(
            args, "allow_logged_token_promotion", False
        ):
            raise RuntimeError(
                "live promotion is blocked because z_rl was not freshly recomputed at every gate-active replay row"
            )
    except RuntimeError as exc:
        rejected_state = {
            **state,
            "format": _state_format(persistent_v2),
            "session_root": str(args.session_root),
            "last_attempt_episode_count": len(trainable_episode_ids),
            "last_attempt_episode_ids": sorted(trainable_episode_ids),
            "last_attempt_train_transition_count": train_transition_count,
            "attempt_index": int(state.get("attempt_index", 0)) + 1,
            "latest_checkpoint": latest_checkpoint,
            "latest_rejected_checkpoint": str(checkpoint),
            "latest_rejected_replay": str(replay_path),
            "latest_rejection_reason": str(exc),
            "actor_validation_gates": _actor_validation_gate_report(args),
            "updated_unix": time.time(),
        }
        _atomic_json(state_path, rejected_state)
        report = {
            **base_report,
            "outcome": "candidate_rejected_keep_previous_policy",
            "updated": False,
            "candidate_checkpoint": str(checkpoint),
            "replay": str(replay_path),
            "validation": validation,
            "rejection_reason": str(exc),
            "training_steps": training_steps,
            "new_train_transitions": new_train_transitions,
            "replay_transition_counts": replay_counts,
        }
        _record_event(args.state_root, report)
        return report

    previous_selected = _read_selected(args.selected_checkpoint_file)
    _write_selected(args.selected_checkpoint_file, str(checkpoint))
    try:
        print(
            "[online-rlt] candidate accepted; restarting shadow policy service and running smoke test",
            flush=True,
        )
        _restart_service(
            args.shadow_service, host=args.policy_host, port=args.policy_port
        )
        smoke_path = args.state_root / f"shadow_acceptance_{episode_set_tag}.json"
        episode_jsonl, timestep = _smoke_observation(trainable_episodes[-1])
        _run(
            [
                python,
                args.runtime / "scripts/smoke_rlt_shadow_policy_client.py",
                "--host",
                args.policy_host,
                "--port",
                str(args.policy_port),
                "--episode-jsonl",
                episode_jsonl,
                "--t",
                str(timestep),
                "--require-actor",
            ],
            env=env,
            log_path=smoke_path,
        )
        smoke = _load_json(smoke_path)
        actor_name = str(smoke.get("shadow", {}).get("actor_name", ""))
        if (
            str(checkpoint) not in actor_name
            or smoke.get("shadow", {}).get("actor_controls_robot") is not False
        ):
            raise RuntimeError(
                "shadow service did not load the promoted checkpoint safely"
            )
        new_state = {
            **state,
            "format": (
                _state_format(persistent_v2)
            ),
            "session_root": str(args.session_root),
            "last_update_episode_count": len(trainable_episode_ids),
            "last_attempt_episode_count": len(trainable_episode_ids),
            "trained_episode_ids": sorted(trainable_episode_ids),
            "last_attempt_episode_ids": sorted(trainable_episode_ids),
            "last_train_transition_count": train_transition_count,
            "last_attempt_train_transition_count": train_transition_count,
            "attempt_index": int(state.get("attempt_index", 0)) + 1,
            "update_index": int(state.get("update_index", 0)) + 1,
            "latest_checkpoint": str(checkpoint),
            "latest_replay": str(replay_path),
            "latest_replay_sha256": _sha256(replay_path),
            "normalization_policy": "frozen_from_initial_warmup",
            "utd": float(getattr(args, "utd", 1.0)),
            "latest_training_steps": training_steps,
            "actor_validation_gates": _actor_validation_gate_report(args),
            "administrative_promotion": state.get("administrative_promotion"),
            "updated_unix": time.time(),
        }
        if persistent_v2 and warm_start_report is not None:
            new_state["warm_start_consumed"] = True
            new_state["warm_start_status"] = "consumed_and_audited"
            if warm_start_report is not None and isinstance(
                state.get("objective_migration"), dict
            ):
                new_state["objective_migration"] = {
                    **state["objective_migration"],
                    "status": "consumed_and_audited",
                    "accepted_checkpoint": str(checkpoint),
                    "warm_start_audit": str(warm_start_report_path),
                }
            new_state["candidate_action_normalization_status"] = (
                "refit_from_persistent_replay"
            )
            new_state["last_warm_start_audit"] = (
                str(warm_start_report_path)
                if warm_start_report_path is not None
                else state.get("last_warm_start_audit")
            )
        elif persistent_v2:
            new_state["candidate_action_normalization_status"] = (
                "fit_from_fresh_zero_replay"
            )
        if frozen_base_replay is not None:
            new_state["bootstrap_gripper_replay"] = str(frozen_base_replay)
            new_state["bootstrap_gripper_replay_sha256"] = _sha256(
                frozen_base_replay
            )
            new_state["bootstrap_gripper_episode_ids"] = sorted(
                frozen_base_episode_ids
            )
        _atomic_json(state_path, new_state)
    except Exception:
        _write_selected(args.selected_checkpoint_file, previous_selected or "NONE")
        _restart_service(
            args.shadow_service, host=args.policy_host, port=args.policy_port
        )
        raise
    report = {
        **base_report,
        "outcome": "updated_and_promoted_at_episode_boundary",
        "updated": True,
        "checkpoint": str(checkpoint),
        "replay": str(replay_path),
        "validation": validation,
        "shadow_acceptance": smoke,
        "training_steps": training_steps,
        "new_train_transitions": new_train_transitions,
        "replay_transition_counts": replay_counts,
        "actor_live_next_episode": True,
        "warm_start_actor_only_audit": warm_start_report,
    }
    _record_event(args.state_root, report)
    print(
        f"[online-rlt] Actor promoted and ready for the next episode: {checkpoint}",
        flush=True,
    )
    return report


def _actor_validation_command(
    *,
    python: Path,
    args: argparse.Namespace,
    checkpoint: Path,
    replay_path: Path,
    validation_path: Path,
    validation_seed: int,
    incumbent_checkpoint: str | Path | None,
) -> list[Any]:
    """Build the explicit, auditable online Actor admission command."""

    command: list[Any] = [
        python,
        args.workspace / "scripts/validate_real_rlt_actor_jax.py",
        "--checkpoint",
        checkpoint,
        "--replay-npz",
        replay_path,
        "--split",
        "validation",
        "--samples",
        "512",
        "--seed",
        str(validation_seed),
        "--output-json",
        validation_path,
        "--max-joint-residual-limit",
        str(getattr(args, "residual_max", 0.005)),
        "--max-gripper-residual-limit",
        str(args.gripper_residual_max),
        "--required-action-schema-fingerprint",
        str(
            getattr(args, "execution_action_schema_fingerprint", None)
            if _is_persistent_v2(args)
            else getattr(
                args, "action_schema_fingerprint", ACTION_SCHEMA_FINGERPRINT
            )
        ),
        "--max-rank1-fit-error-rad",
        "1e-5",
        "--max-residual-abs-rad",
        str(getattr(args, "residual_max", 0.005)),
        "--max-residual-d1-rad",
        str(getattr(args, "residual_d1_max_rad", 0.0015)),
        "--max-residual-d2-rad",
        str(getattr(args, "residual_d2_max_rad", 0.001)),
        "--max-direction-cone-deg",
        str(getattr(args, "direction_cone_deg", 15.0)),
        "--max-active-normalized-residual-step",
        str(args.max_active_normalized_residual_step),
        "--max-actor-joint-d1-p95-rad",
        str(args.max_actor_joint_d1_p95_rad),
        "--max-chunk-boundary-normalized-residual-jump-p95",
        str(args.max_chunk_boundary_normalized_residual_jump_p95),
        "--max-chunk-boundary-actor-command-joint-d1-p95-rad",
        str(args.max_chunk_boundary_actor_command_joint_d1_p95_rad),
    ]
    if incumbent_checkpoint is not None:
        command.extend(["--incumbent-checkpoint", incumbent_checkpoint])
    return command


def _logged_replay_prepare_command(
    *,
    episodes: list[Path],
    args: argparse.Namespace,
    python: Path,
    replay_dir: Path,
) -> list[Any]:
    """Build the online-only replay command from execution-time policy fields.

    Native online episodes contain the exact stochastic Pi0.5 reference chunk
    seen when each command was selected.  That logged ``a_ref`` is immutable.
    Prefer a cache that recomputes a fresh RL Token for every gate row; falling
    back to the temporally repeated online token is audit-only and cannot be
    promoted without an explicit debug override.
    """

    command: list[Any] = [
        python,
        args.workspace / "scripts/piper_rlt/tools/prepare_external_rlt_replay.py",
        "--dataset-root",
        args.session_root,
        "--output",
        replay_dir,
        "--base-fingerprint",
        BASE_FINGERPRINT,
        "--token-fingerprint",
        TOKEN_FINGERPRINT,
        "--phase-fingerprint",
        PHASE_FINGERPRINT,
        "--action-schema-fingerprint",
        str(
            getattr(args, "execution_action_schema_fingerprint", None)
            if _is_persistent_v2(args)
            else getattr(
                args, "action_schema_fingerprint", ACTION_SCHEMA_FINGERPRINT
            )
        ),
        "--actor-projection-profile",
        str(getattr(args, "actor_projection_profile", ACTOR_PROJECTION_PROFILE)),
        "--validation-fraction",
        str(args.validation_fraction),
        "--test-fraction",
        "0",
        "--split-seed",
        "piper-online-rlt-v1",
        "--split-registry",
        args.state_root / "split_registry.json",
    ]
    if _is_persistent_v2(args):
        command.extend(
            [
                "--chunk-length",
                "10",
                "--stride",
                "10",
                "--n-step",
                "10",
                "--actor-execution-profile",
                PERSISTENT_V2_ACTOR_EXECUTION_PROFILE,
                "--execution-filter-profile",
                PERSISTENT_V2_EXECUTION_FILTER_PROFILE,
                "--execution-filter-tau-s",
                str(PERSISTENT_V2_EXECUTION_FILTER_TAU_S),
                "--control-hz",
                str(PERSISTENT_V2_CONTROL_HZ),
            ]
        )
    enrichment_cache = getattr(args, "enrichment_cache", None)
    if enrichment_cache is None:
        command.extend(["--allow-logged-reference", "--allow-logged-phase"])
    else:
        # The enrichment provider recomputes a fresh z_rl for every replay
        # anchor while this flag preserves the exact stochastic Pi0.5 a_ref
        # that was observed and executed online.
        command.extend(
            [
                "--enrichment-cache",
                Path(enrichment_cache).expanduser().resolve(),
                "--preserve-logged-reference",
                "--allow-logged-phase",
            ]
        )
    for episode in episodes:
        command.extend(["--episode-jsonl", episode])
    return command


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "warmup_episodes",
        "min_success",
        "min_failure",
        "min_success_human_episodes",
        "min_admitted_human_episodes",
        "update_every",
        "min_warmup_transitions",
        "min_update_steps",
        "max_update_steps",
        "initial_steps",
        "update_steps",
    ):
        if name in (
            "min_success_human_episodes",
            "min_admitted_human_episodes",
        ):
            if int(getattr(args, name, 0)) < 0:
                raise ValueError(f"{name} must be non-negative")
            continue
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if not np.isfinite(args.utd) or args.utd <= 0:
        raise ValueError("utd must be finite and positive")
    if args.min_update_steps > args.max_update_steps:
        raise ValueError("min_update_steps must not exceed max_update_steps")
    for path in (args.workspace, args.runtime, args.phase_checkpoint):
        if not path.expanduser().exists():
            raise FileNotFoundError(path)
    residual_max = float(getattr(args, "residual_max", 0.005))
    if residual_max <= 0:
        raise ValueError("joint residual limit must be positive")
    for name in ("residual_d1_max_rad", "residual_d2_max_rad"):
        value = float(getattr(args, name, 0.0015 if name == "residual_d1_max_rad" else 0.001))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    direction_cone_deg = float(getattr(args, "direction_cone_deg", 15.0))
    if not np.isfinite(direction_cone_deg) or not 0.0 < direction_cone_deg < 90.0:
        raise ValueError("direction_cone_deg must be finite and in (0, 90)")
    actor_execution_profile = str(
        getattr(
            args,
            "actor_execution_profile",
            LEGACY_ACTOR_EXECUTION_PROFILE,
        )
    )
    action_schema_fingerprint = str(
        getattr(args, "action_schema_fingerprint", ACTION_SCHEMA_FINGERPRINT)
    )
    if actor_execution_profile == PERSISTENT_V2_ACTOR_EXECUTION_PROFILE:
        expected_schema = ACTION_SCHEMA_FINGERPRINT
        expected_projection = PERSISTENT_V2_ACTOR_PROJECTION_PROFILE
    elif actor_execution_profile == LEGACY_ACTOR_EXECUTION_PROFILE:
        expected_schema = ACTION_SCHEMA_FINGERPRINT
        expected_projection = ACTOR_PROJECTION_PROFILE
    else:
        raise ValueError(
            f"unsupported online updater Actor execution profile: "
            f"{actor_execution_profile!r}"
        )
    if action_schema_fingerprint != expected_schema:
        raise ValueError(
            "online updater action schema mismatch: "
            f"{action_schema_fingerprint!r} != {expected_schema!r}"
        )
    actor_projection_profile = str(
        getattr(args, "actor_projection_profile", ACTOR_PROJECTION_PROFILE)
    )
    if actor_projection_profile != expected_projection:
        raise ValueError(
            "online updater projection profile mismatch: "
            f"{actor_projection_profile!r} != {expected_projection!r}"
        )
    if actor_execution_profile == PERSISTENT_V2_ACTOR_EXECUTION_PROFILE:
        if (
            getattr(args, "execution_action_schema_fingerprint", None)
            != PERSISTENT_V2_ACTION_SCHEMA_FINGERPRINT
        ):
            raise ValueError(
                "persistent-v2 execution action schema mismatch"
            )
        if (
            getattr(args, "actor_governor_fingerprint", None)
            != PERSISTENT_GOVERNOR_FINGERPRINT
        ):
            raise ValueError("persistent-v2 governor fingerprint mismatch")
        exact_runtime_contract = (
            (
                "actor_live_max_boundary_jump_rad",
                ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD,
            ),
            ("actor_projection_scale_steps", ACTOR_PROJECTION_SCALE_STEPS),
            ("actor_min_projection_scale", ACTOR_MIN_PROJECTION_SCALE),
            (
                "actor_direction_static_threshold_rad",
                ACTOR_DIRECTION_STATIC_THRESHOLD_RAD,
            ),
        )
        for name, expected in exact_runtime_contract:
            actual = getattr(args, name, None)
            if isinstance(expected, int):
                matched = int(actual) == expected
            else:
                matched = bool(
                    actual is not None
                    and math.isclose(
                        float(actual),
                        expected,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                )
            if not matched:
                raise ValueError(
                    f"persistent-v2 {name} mismatch: {actual!r} != {expected!r}"
                )
        if (
            getattr(args, "execution_filter_profile", None)
            != PERSISTENT_V2_EXECUTION_FILTER_PROFILE
        ):
            raise ValueError("persistent-v2 execution filter profile mismatch")
        for name, expected in (
            (
                "execution_filter_tau_s",
                PERSISTENT_V2_EXECUTION_FILTER_TAU_S,
            ),
            ("control_hz", PERSISTENT_V2_CONTROL_HZ),
        ):
            value = getattr(args, name, None)
            if value is None or not math.isclose(
                float(value), expected, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise ValueError(
                    f"persistent-v2 {name} mismatch: {value!r} != {expected!r}"
                )
        minimum = int(
            getattr(
                args,
                "min_new_persistent_committed_episodes",
                DEFAULT_MIN_NEW_COMMITTED_EPISODES,
            )
        )
        if minimum < 0:
            raise ValueError(
                "persistent gripper-v3 new-episode threshold cannot be negative"
            )
    elif (
        any(
            getattr(args, name, None) is not None
            for name in (
                "execution_filter_profile",
                "execution_filter_tau_s",
                "control_hz",
                "warm_start_actor_checkpoint",
                "execution_action_schema_fingerprint",
            )
        )
        or bool(getattr(args, "allow_warm_start_objective_migration", False))
    ):
        raise ValueError(
            "persistent execution-filter/warm-start arguments cannot be mixed "
            "into a legacy updater"
        )
    if getattr(args, "freeze_gripper_residual", False):
        raise ValueError("gripper-close v3 cannot freeze the gripper residual")
    if int(getattr(args, "min_success_human_episodes", 0)) != 0:
        raise ValueError(
            "gripper-close v3 cannot gate human data on a reward-positive "
            "episode label; use --min-admitted-human-episodes"
        )
    if int(getattr(args, "min_admitted_human_episodes", 0)) < 1:
        raise ValueError(
            "gripper-close v3 requires at least one admitted-human episode"
        )
    exact_gripper_contract = {
        "gripper_residual_max": GRIPPER_RESIDUAL_MAX_CLOSE_M,
        "gripper_residual_d1_max_m": GRIPPER_RESIDUAL_D1_MAX_M,
        "gripper_residual_d2_max_m": GRIPPER_RESIDUAL_D2_MAX_M,
        "gripper_max_boundary_jump_m": GRIPPER_MAX_BOUNDARY_JUMP_M,
        "gripper_command_min_m": GRIPPER_COMMAND_MIN_M,
        "gripper_command_max_m": GRIPPER_COMMAND_MAX_M,
        "gripper_release_reference_m": GRIPPER_RELEASE_REFERENCE_M,
        "gripper_release_delta_m": GRIPPER_RELEASE_DELTA_M,
    }
    if args.gripper_residual_mode != GRIPPER_RESIDUAL_MODE:
        raise ValueError("gripper-close v3 mode mismatch")
    for name, expected in exact_gripper_contract.items():
        actual = float(getattr(args, name))
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                f"gripper-close v3 {name} mismatch: {actual!r} != {expected!r}"
            )
    if not math.isclose(
        float(args.human_gripper_bc_scale_m),
        GRIPPER_RESIDUAL_MAX_CLOSE_M,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("gripper-close v3 human BC scale mismatch")
    if getattr(args, "human_gripper_q_filter_mode", None) != (
        HUMAN_GRIPPER_Q_FILTER_MODE
    ):
        raise ValueError("gripper-close v3 human-gripper Q-filter mode mismatch")
    q_filter_margin = float(
        getattr(
            args,
            "human_gripper_q_filter_margin",
            HUMAN_GRIPPER_Q_FILTER_MARGIN,
        )
    )
    if (
        not np.isfinite(q_filter_margin)
        or q_filter_margin < 0.0
        or not math.isclose(
            q_filter_margin,
            HUMAN_GRIPPER_Q_FILTER_MARGIN,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError("gripper-close v3 human-gripper Q-filter margin mismatch")
    for name in (
        "beta_human_bc",
        "beta_human_gripper_bc",
        "target_policy_noise_std",
        "target_policy_noise_clip",
    ):
        value = float(getattr(args, name, 0.0))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    for name in (
        "max_validation_td_error",
        "max_actor_q_advantage",
        "max_active_normalized_residual_step",
        "max_actor_joint_d1_p95_rad",
        "max_chunk_boundary_normalized_residual_jump_p95",
        "max_chunk_boundary_actor_command_joint_d1_p95_rad",
        "min_reward1_reward0_exec_q_gap",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if (
        args.enrichment_cache is not None
        and not args.enrichment_cache.expanduser().exists()
    ):
        raise FileNotFoundError(args.enrichment_cache)


def _is_persistent_v2(args: argparse.Namespace) -> bool:
    return (
        str(
            getattr(
                args,
                "actor_execution_profile",
                LEGACY_ACTOR_EXECUTION_PROFILE,
            )
        )
        == PERSISTENT_V2_ACTOR_EXECUTION_PROFILE
    )


def _state_format(persistent_v2: bool) -> str:
    return PERSISTENT_V2_STATE_FORMAT if persistent_v2 else LEGACY_STATE_FORMAT


def _validate_persistent_v2_state(
    args: argparse.Namespace, state: dict[str, Any]
) -> None:
    if not state:
        raise ValueError(
            "persistent-v2 requires a pre-created, bound online_state.json; "
            "use the bootstrap fork or fresh-zero initializer first"
        )
    lineage_mode = state.get("lineage_mode")
    if lineage_mode not in {
        BOOTSTRAP_LINEAGE_MODE,
        FRESH_ZERO_LINEAGE_MODE,
    }:
        raise ValueError(
            f"unsupported persistent gripper-v3 lineage mode: {lineage_mode!r}"
        )
    fresh_zero = lineage_mode == FRESH_ZERO_LINEAGE_MODE
    expected_fields: dict[str, Any] = {
        "format": PERSISTENT_V2_STATE_FORMAT,
        "lineage_mode": lineage_mode,
        "actor_execution_profile": PERSISTENT_V2_ACTOR_EXECUTION_PROFILE,
        "actor_model_action_schema_fingerprint": ACTION_SCHEMA_FINGERPRINT,
        "execution_action_schema_fingerprint": (
            PERSISTENT_V2_ACTION_SCHEMA_FINGERPRINT
        ),
        "actor_projection_profile": PERSISTENT_V2_ACTOR_PROJECTION_PROFILE,
        "execution_filter_profile": PERSISTENT_V2_EXECUTION_FILTER_PROFILE,
        "execution_filter_tau_s": PERSISTENT_V2_EXECUTION_FILTER_TAU_S,
        "control_hz": PERSISTENT_V2_CONTROL_HZ,
        "control_dt_s": 1.0 / PERSISTENT_V2_CONTROL_HZ,
        "execution_filter_alpha": 1.0
        - math.exp(
            -(1.0 / PERSISTENT_V2_CONTROL_HZ)
            / PERSISTENT_V2_EXECUTION_FILTER_TAU_S
        ),
        "chunk_length": 10,
        "chunk_stride": 10,
        "actor_live_max_boundary_jump_rad": (
            ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD
        ),
        "actor_projection_scale_steps": ACTOR_PROJECTION_SCALE_STEPS,
        "actor_min_projection_scale": ACTOR_MIN_PROJECTION_SCALE,
        "actor_direction_static_threshold_rad": (
            ACTOR_DIRECTION_STATIC_THRESHOLD_RAD
        ),
        "actor_governor_fingerprint": PERSISTENT_GOVERNOR_FINGERPRINT,
        "gripper_residual_mode": GRIPPER_RESIDUAL_MODE,
        "actor_gripper_residual_max_close_m": GRIPPER_RESIDUAL_MAX_CLOSE_M,
        "actor_gripper_residual_d1_max_m": GRIPPER_RESIDUAL_D1_MAX_M,
        "actor_gripper_residual_d2_max_m": GRIPPER_RESIDUAL_D2_MAX_M,
        "actor_gripper_max_boundary_jump_m": GRIPPER_MAX_BOUNDARY_JUMP_M,
        "gripper_command_min_m": GRIPPER_COMMAND_MIN_M,
        "gripper_command_max_m": GRIPPER_COMMAND_MAX_M,
        "gripper_release_reference_m": GRIPPER_RELEASE_REFERENCE_M,
        "gripper_release_delta_m": GRIPPER_RELEASE_DELTA_M,
        "replay_training_policy": (
            FRESH_ZERO_REPLAY_POLICY
            if fresh_zero
            else BOOTSTRAP_REPLAY_POLICY
        ),
    }
    for key, expected in expected_fields.items():
        actual = state.get(key)
        if isinstance(expected, float):
            matched = bool(
                actual is not None
                and math.isclose(
                    float(actual), expected, rel_tol=0.0, abs_tol=1.0e-12
                )
            )
        else:
            matched = actual == expected
        if not matched:
            raise ValueError(
                f"persistent-v2 state contract mismatch for {key}: "
                f"{actual!r} != {expected!r}"
            )
    if Path(state.get("session_root", "")).resolve() != args.session_root:
        raise ValueError("persistent-v2 state is bound to another session")
    bootstrap_replay_value = state.get("bootstrap_gripper_replay")
    bootstrap_ids = _state_episode_ids(
        state,
        "bootstrap_gripper_episode_ids",
    )
    bootstrap_id_set = set() if bootstrap_ids is None else set(bootstrap_ids)
    if fresh_zero:
        forbidden_source_fields = (
            "bootstrap_gripper_replay",
            "bootstrap_gripper_replay_sha256",
            "bootstrap_gripper_episode_ids",
            "legacy_source_replay",
            "legacy_source_replay_sha256",
            "initial_actor_warm_start_checkpoint",
        )
        present = [
            key for key in forbidden_source_fields if state.get(key) not in (None, [], "")
        ]
        if present:
            raise ValueError(
                "fresh-zero lineage contains forbidden inherited provenance: "
                f"{present}"
            )
    else:
        if not bootstrap_replay_value or not bootstrap_ids:
            raise ValueError(
                "persistent gripper-v3 state lacks its migrated warm-up replay"
            )
        bootstrap_replay = Path(str(bootstrap_replay_value)).expanduser().resolve()
        if not bootstrap_replay.is_file():
            raise ValueError("persistent gripper-v3 bootstrap replay is missing")
        if _replay_episode_ids(bootstrap_replay) != set(bootstrap_ids):
            raise ValueError("persistent gripper-v3 bootstrap episode IDs mismatch")
        if _sha256(bootstrap_replay) != state.get(
            "bootstrap_gripper_replay_sha256"
        ):
            raise ValueError("persistent gripper-v3 bootstrap replay hash mismatch")
        legacy_replay_value = state.get("legacy_source_replay")
        legacy_replay_sha = state.get("legacy_source_replay_sha256")
        if not legacy_replay_value or not legacy_replay_sha:
            raise ValueError("persistent-v2 state lacks immutable source replay provenance")
        legacy_replay = Path(legacy_replay_value).expanduser().resolve()
        if not legacy_replay.is_file() or _sha256(legacy_replay) != legacy_replay_sha:
            raise ValueError("persistent-v2 legacy replay provenance SHA failed")
    floor = int(state.get("episode_index_floor", -1))
    if floor < 0:
        raise ValueError("persistent-v2 state lacks episode_index_floor")
    if fresh_zero and floor != 0:
        raise ValueError("fresh-zero lineage episode_index_floor must be 0")
    for path in args.session_root.glob("episode_[0-9]*"):
        try:
            index = int(path.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if index < floor:
            raise ValueError(
                f"legacy episode appeared inside persistent session: {path.name} "
                f"is below floor {floor}"
            )
    for key in ("trained_episode_ids", "last_attempt_episode_ids"):
        for episode_id in _state_episode_ids(state, key) or ():
            if episode_id in bootstrap_id_set:
                continue
            try:
                index = int(episode_id.split("_", 1)[1])
            except (IndexError, ValueError) as exc:
                raise ValueError(f"invalid persistent episode ID: {episode_id}") from exc
            if index < floor:
                raise ValueError(
                    f"legacy episode ID contaminated persistent state: {episode_id}"
                )
    minimum = int(state.get("min_new_persistent_committed_episodes", 0))
    if minimum < 0:
        raise ValueError("persistent gripper-v3 online warmup threshold is negative")
    if minimum != int(args.min_new_persistent_committed_episodes):
        raise ValueError("persistent-v2 state/CLI warmup threshold mismatch")
    if fresh_zero:
        if minimum != int(args.warmup_episodes):
            raise ValueError(
                "fresh-zero warmup threshold must equal --warmup-episodes"
            )
        if getattr(args, "warm_start_actor_checkpoint", None) is not None:
            raise ValueError("fresh-zero lineage forbids an Actor warm-start")
        if bool(getattr(args, "allow_warm_start_objective_migration", False)):
            raise ValueError(
                "fresh-zero lineage forbids warm-start objective migration"
            )
        latest_checkpoint = state.get("latest_checkpoint")
        if latest_checkpoint:
            latest = Path(str(latest_checkpoint)).expanduser().resolve()
            if (
                args.state_root not in latest.parents
                or not (latest / "learner.msgpack").is_file()
                or not (latest / "metadata.json").is_file()
            ):
                raise ValueError(
                    "fresh-zero latest checkpoint is missing or outside state root"
                )
        return
    warm_start = state.get("initial_actor_warm_start_checkpoint")
    if not warm_start:
        raise ValueError("persistent-v2 state lacks Actor warm-start checkpoint")
    warm_start_path = Path(warm_start).expanduser().resolve()
    if not (warm_start_path / "learner.msgpack").is_file():
        raise ValueError("persistent-v2 Actor warm-start checkpoint is incomplete")
    source_metadata = _load_json(warm_start_path / "metadata.json")
    source_config = source_metadata.get("config")
    if not isinstance(source_config, dict):
        raise ValueError("persistent-v2 Actor warm-start metadata lacks config")
    try:
        source_beta_bc = float(source_config["beta_bc"])
        source_beta_human_bc = float(source_config["beta_human_bc"])
        source_beta_human_gripper_bc = float(
            source_config.get("beta_human_gripper_bc", 0.0)
        )
        target_beta_bc = float(args.beta_bc)
        target_beta_human_bc = float(
            getattr(args, "beta_human_bc", 0.0)
        )
        target_beta_human_gripper_bc = float(
            getattr(args, "beta_human_gripper_bc", 1.0)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "persistent-v2 Actor warm-start objective weights are invalid"
        ) from exc
    objective_changed = not (
        math.isclose(
            source_beta_bc, target_beta_bc, rel_tol=0.0, abs_tol=1.0e-12
        )
        and math.isclose(
            source_beta_human_bc,
            target_beta_human_bc,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            source_beta_human_gripper_bc,
            target_beta_human_gripper_bc,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    )
    migration_authorized = bool(
        getattr(args, "allow_warm_start_objective_migration", False)
    )
    if objective_changed != migration_authorized:
        raise ValueError(
            "persistent-v2 objective migration authorization does not match "
            "the source/target beta weights"
        )
    if objective_changed:
        migration = state.get("objective_migration")
        if not isinstance(migration, dict):
            raise ValueError(
                "persistent-v2 state lacks the explicit objective migration audit"
            )
        expected_migration = {
            "format": "openpi_piper_gripper_v3_migration_v1",
            "source_beta_bc": source_beta_bc,
            "source_beta_human_bc": source_beta_human_bc,
            "source_beta_human_gripper_bc": source_beta_human_gripper_bc,
            "target_beta_bc": target_beta_bc,
            "target_beta_human_bc": target_beta_human_bc,
            "target_beta_human_gripper_bc": target_beta_human_gripper_bc,
            "target_human_gripper_q_filter_mode": (
                HUMAN_GRIPPER_Q_FILTER_MODE
            ),
            "target_human_gripper_q_filter_margin": (
                HUMAN_GRIPPER_Q_FILTER_MARGIN
            ),
            "human_supervision_scope": (
                "all_admitted_human_reward_positive_and_reward_negative"
            ),
            "physical_governor_changed": True,
        }
        for key, expected in expected_migration.items():
            actual = migration.get(key)
            if isinstance(expected, float):
                matched = bool(
                    actual is not None
                    and math.isclose(
                        float(actual),
                        expected,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                )
            else:
                matched = actual == expected
            if not matched:
                raise ValueError(
                    "persistent-v2 objective migration mismatch for "
                    f"{key}: {actual!r} != {expected!r}"
                )
        manifest_value = state.get("objective_migration_manifest")
        if not manifest_value:
            raise ValueError(
                "persistent-v2 state lacks objective_migration_manifest"
            )
        manifest_path = Path(str(manifest_value)).expanduser().resolve()
        if args.state_root not in manifest_path.parents or not manifest_path.is_file():
            raise ValueError(
                "persistent-v2 objective migration manifest is missing or "
                "outside the state root"
            )
        manifest = _load_json(manifest_path)
        if (
            manifest.get("format") != expected_migration["format"]
            or manifest.get("authorization") != migration.get("authorization")
            or manifest.get("source", {}).get("checkpoint") != str(warm_start_path)
            or float(manifest.get("source", {}).get("beta_bc", float("nan")))
            != source_beta_bc
            or float(
                manifest.get("source", {}).get(
                    "beta_human_bc", float("nan")
                )
            )
            != source_beta_human_bc
            or float(
                manifest.get("source", {}).get(
                    "beta_human_gripper_bc", 0.0
                )
            )
            != source_beta_human_gripper_bc
            or float(manifest.get("target", {}).get("beta_bc", float("nan")))
            != target_beta_bc
            or float(
                manifest.get("target", {}).get(
                    "beta_human_bc", float("nan")
                )
            )
            != target_beta_human_bc
            or float(
                manifest.get("target", {}).get(
                    "beta_human_gripper_bc", float("nan")
                )
            )
            != target_beta_human_gripper_bc
            or manifest.get("target", {}).get(
                "human_gripper_q_filter_mode"
            )
            != HUMAN_GRIPPER_Q_FILTER_MODE
            or float(
                manifest.get("target", {}).get(
                    "human_gripper_q_filter_margin", float("nan")
                )
            )
            != HUMAN_GRIPPER_Q_FILTER_MARGIN
            or manifest.get("human_supervision_scope")
            != "all_admitted_human_reward_positive_and_reward_negative"
            or manifest.get("physical_governor_changed") is not True
        ):
            raise ValueError(
                "persistent-v2 objective migration manifest/state mismatch"
            )
    cli_warm_start = getattr(args, "warm_start_actor_checkpoint", None)
    if (
        cli_warm_start is not None
        and cli_warm_start.expanduser().resolve() != warm_start_path
    ):
        raise ValueError("persistent-v2 state/CLI warm-start checkpoint mismatch")
    latest_checkpoint = state.get("latest_checkpoint")
    if latest_checkpoint:
        latest = Path(latest_checkpoint).expanduser().resolve()
        if args.state_root not in latest.parents:
            raise ValueError("persistent-v2 latest checkpoint is outside state root")
        metadata = _load_json(latest / "metadata.json")
        fingerprints = metadata.get("fingerprints") or {}
        latest_config = metadata.get("config") or {}
        for key, expected in (
            ("beta_bc", target_beta_bc),
            ("beta_human_bc", target_beta_human_bc),
            ("beta_human_gripper_bc", target_beta_human_gripper_bc),
        ):
            try:
                actual = float(latest_config[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"persistent-v2 latest checkpoint lacks valid {key}"
                ) from exc
            if not math.isclose(
                actual, expected, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise ValueError(
                    "persistent-v2 latest checkpoint objective mismatch for "
                    f"{key}: {actual} != {expected}"
                )
        expected_latest_config = {
            "freeze_gripper_residual": False,
            "gripper_residual_mode": GRIPPER_RESIDUAL_MODE,
            "actor_gripper_residual_max_close_m": GRIPPER_RESIDUAL_MAX_CLOSE_M,
            "actor_gripper_residual_d1_max_m": GRIPPER_RESIDUAL_D1_MAX_M,
            "actor_gripper_residual_d2_max_m": GRIPPER_RESIDUAL_D2_MAX_M,
            "actor_gripper_max_boundary_jump_m": GRIPPER_MAX_BOUNDARY_JUMP_M,
            "human_gripper_q_filter_mode": HUMAN_GRIPPER_Q_FILTER_MODE,
            "human_gripper_q_filter_margin": HUMAN_GRIPPER_Q_FILTER_MARGIN,
        }
        for key, expected in expected_latest_config.items():
            actual = latest_config.get(key)
            if isinstance(expected, float):
                matched = bool(
                    actual is not None
                    and math.isclose(
                        float(actual),
                        expected,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                )
            else:
                matched = actual == expected
            if not matched:
                raise ValueError(
                    "persistent gripper-v3 checkpoint config mismatch for "
                    f"{key}: {actual!r} != {expected!r}"
                )
        for key, expected in (
            ("action_schema", PERSISTENT_V2_ACTION_SCHEMA_FINGERPRINT),
            (
                "actor_execution_profile",
                PERSISTENT_V2_ACTOR_EXECUTION_PROFILE,
            ),
            (
                "execution_filter_profile",
                PERSISTENT_V2_EXECUTION_FILTER_PROFILE,
            ),
            ("actor_governor", PERSISTENT_GOVERNOR_FINGERPRINT),
        ):
            if fingerprints.get(key) != expected:
                raise ValueError(
                    f"persistent-v2 checkpoint fingerprint mismatch for {key}"
                )


def _validate_actor_only_warm_start_report(
    path: Path,
    *,
    expected_source_checkpoint: Path,
    expected_beta_bc: float,
    expected_beta_human_bc: float,
    allow_objective_migration: bool,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(
            f"Actor-only warm-start audit report was not written: {path}"
        )
    report = _load_json(path)
    if report.get("format") != "openpi_real_rlt_persistent_v2_actor_warm_start":
        raise RuntimeError("unsupported Actor-only warm-start audit format")
    source_value = report.get(
        "source_checkpoint", report.get("warm_start_actor_checkpoint")
    )
    if (
        source_value is None
        or Path(str(source_value)).expanduser().resolve()
        != expected_source_checkpoint
    ):
        raise RuntimeError("warm-start audit source checkpoint mismatch")
    source_metadata_path = expected_source_checkpoint / "metadata.json"
    if not source_metadata_path.is_file():
        raise RuntimeError(
            "warm-start source checkpoint metadata.json is unavailable for "
            "objective-weight verification"
        )
    source_metadata = _load_json(source_metadata_path)
    source_config = source_metadata.get("config")
    if not isinstance(source_config, dict):
        raise RuntimeError(
            "warm-start source checkpoint metadata lacks config"
        )
    required_true = (
        "tree_structure_equal",
        "leaf_shapes_equal",
        "critic_reinitialized",
        "target_critic_reinitialized",
        "optimizer_reinitialized",
        "actor_optimizer_reinitialized",
        "critic_optimizer_reinitialized",
        "rng_reinitialized",
        "target_actor_copied_from_actor",
    )
    for key in required_true:
        if report.get(key) is not True:
            raise RuntimeError(
                f"Actor-only warm-start audit did not prove {key}=true"
            )
    if int(report.get("actor_param_leaf_count", 0)) <= 0:
        raise RuntimeError("warm-start audit has no Actor parameter leaves")
    source_sha = report.get(
        "source_actor_sha256", report.get("source_actor_params_sha256")
    )
    target_sha = report.get(
        "target_actor_sha256", report.get("target_actor_params_sha256")
    )
    target_actor_sha = report.get(
        "target_actor_copy_sha256", target_sha
    )
    if (
        not isinstance(source_sha, str)
        or not source_sha
        or source_sha != target_sha
        or source_sha != target_actor_sha
    ):
        raise RuntimeError(
            "warm-start Actor/target-Actor parameter tree SHA mismatch"
        )
    if int(report.get("update_step", -1)) != 0:
        raise RuntimeError("warm-start update_step is not zero")
    if report.get("old_replay_loaded") is not False:
        raise RuntimeError("warm-start audit did not prove old_replay_loaded=false")
    normalization = report.get("normalization")
    if not isinstance(normalization, dict):
        raise RuntimeError("warm-start audit lacks normalization policy")
    for key in ("z_rl_reused", "state_reused", "a_ref_reused"):
        if normalization.get(key) is not True:
            raise RuntimeError(
                f"warm-start normalization audit did not prove {key}=true"
            )
    if normalization.get("candidate_action_refit_from_new_replay") is not True:
        raise RuntimeError(
            "warm-start candidate-action normalization was not refit from new replay"
        )
    objective = report.get("objective_weights")
    if not isinstance(objective, dict):
        raise RuntimeError("warm-start audit lacks objective-weight policy")
    source_values: dict[str, float] = {}
    for report_key, source_key in (
        ("source_beta_bc", "beta_bc"),
        ("source_beta_human_bc", "beta_human_bc"),
    ):
        try:
            actual = float(objective.get(report_key))
            expected = float(source_config[source_key])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"warm-start audit has invalid {report_key}"
            ) from exc
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
            raise RuntimeError(
                f"warm-start audit {report_key} does not match source "
                f"checkpoint metadata: {actual} != {expected}"
            )
        source_values[source_key] = actual
    target_values: dict[str, float] = {}
    for key, expected in (
        ("beta_bc", expected_beta_bc),
        ("beta_human_bc", expected_beta_human_bc),
    ):
        try:
            actual = float(objective.get(key))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"warm-start audit has invalid target {key}"
            ) from exc
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
            raise RuntimeError(
                f"warm-start audit target {key} mismatch: {actual} != {expected}"
            )
        target_values[key] = actual
    policy = objective.get("policy")
    migration_authorized = objective.get("migration_authorized")
    objective_changed = any(
        not math.isclose(
            source_values[key],
            target_values[key],
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        for key in ("beta_bc", "beta_human_bc")
    )
    if objective_changed and not allow_objective_migration:
        raise RuntimeError(
            "warm-start objective weights changed without authorization"
        )
    if objective_changed:
        if (
            policy != "explicit_target_beta_migration_v1"
            or migration_authorized is not True
        ):
            raise RuntimeError(
                "warm-start objective migration was requested but not audited"
            )
    elif (
        policy != "preserve_source_beta_bc_and_beta_human_bc_exactly_v1"
        or migration_authorized is not False
    ):
        raise RuntimeError(
            "warm-start audit falsely reports an objective migration"
        )
    return report


def _warm_start_training_arguments(
    *,
    checkpoint: Path,
    report_path: Path,
    allow_objective_migration: bool,
) -> list[str]:
    arguments = [
        "--warm-start-actor-checkpoint",
        str(checkpoint),
        "--warm-start-dry-run-report",
        str(report_path),
    ]
    if allow_objective_migration:
        arguments.append("--allow-warm-start-objective-migration")
    return arguments


def _actor_validation_gate_report(args: argparse.Namespace) -> dict[str, Any]:
    """Describe the explicit online promotion profile without claiming hardware safety."""

    return {
        "profile": ACTOR_VALIDATION_GATE_PROFILE,
        "calibration_basis": "clean30 beta10 versus beta40 candidate sweeps",
        "interpretation": "experiment-calibrated promotion criteria; not a physical safety theorem",
        "max_active_normalized_residual_step": float(
            getattr(
                args,
                "max_active_normalized_residual_step",
                DEFAULT_MAX_ACTIVE_NORMALIZED_RESIDUAL_STEP,
            )
        ),
        "max_actor_joint_d1_p95_rad": float(
            getattr(
                args,
                "max_actor_joint_d1_p95_rad",
                DEFAULT_MAX_ACTOR_JOINT_D1_P95_RAD,
            )
        ),
        "max_chunk_boundary_normalized_residual_jump_p95": float(
            getattr(
                args,
                "max_chunk_boundary_normalized_residual_jump_p95",
                DEFAULT_MAX_CHUNK_BOUNDARY_NORMALIZED_RESIDUAL_JUMP_P95,
            )
        ),
        "max_chunk_boundary_actor_command_joint_d1_p95_rad": float(
            getattr(
                args,
                "max_chunk_boundary_actor_command_joint_d1_p95_rad",
                DEFAULT_MAX_CHUNK_BOUNDARY_ACTOR_COMMAND_JOINT_D1_P95_RAD,
            )
        ),
        "min_reward1_reward0_exec_q_gap": float(
            getattr(args, "min_reward1_reward0_exec_q_gap", 0.0)
        ),
    }


def _completed_episodes(
    root: Path,
    *,
    quarantine_ids: set[str] | None = None,
    invalid_reports: list[dict[str, str]] | None = None,
) -> list[Path]:
    quarantine_ids = quarantine_ids or set()
    result = []
    for report_path in sorted(root.glob("*/report.json")):
        try:
            report = _load_json(report_path)
        except Exception as exc:
            if invalid_reports is not None:
                invalid_reports.append(
                    {
                        "episode_id": report_path.parent.name,
                        "report_json": str(report_path),
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
            continue
        episode_id = report_path.parent.name
        if report.get("exclude_from_training") is True or episode_id in quarantine_ids:
            continue
        reward = report.get("terminal_reward")
        episode_jsonl = report_path.parent / "episode.jsonl"
        if (
            report.get("outcome") == "episode_done"
            and reward in (0, 0.0, 1, 1.0)
            and episode_jsonl.is_file()
        ):
            result.append(episode_jsonl)
    return result


def _audit_episode(
    path: Path,
    *,
    expected_action_schema: str | None = None,
    expected_projection_profile: str | None = None,
    expected_execution_profile: str | None = None,
    expected_filter_profile: str | None = None,
    expected_filter_tau_s: float | None = None,
    expected_control_hz: float | None = None,
    persistent_v2: bool = False,
) -> dict[str, Any]:
    persistent_audit = None
    if persistent_v2:
        persistent_audit = audit_persistent_v2_episode(
            path,
            expected_execution_profile=str(expected_execution_profile),
            expected_action_schema=str(expected_action_schema),
            expected_projection_profile=str(expected_projection_profile),
            expected_filter_profile=str(expected_filter_profile),
            expected_filter_tau_s=float(expected_filter_tau_s),
            expected_control_hz=float(expected_control_hz),
        )
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        not rows
        or not rows[-1].get("done")
        or float(rows[-1].get("reward", -1)) not in {0.0, 1.0}
    ):
        raise ValueError(f"episode terminal contract failed: {path}")
    eligible = 0
    true_token = 0
    gate_active = 0
    human_eligible_rows = 0
    sources: dict[str, int] = {}
    trainable_segments: list[list[str]] = []
    current_segment: list[str] = []
    for row in rows:
        metadata = row.get("policy_metadata") or {}
        source = str(row.get("source"))
        include = bool(metadata.get("replay_include", False))
        if include and source in {"pi05", "rlt", "human_pika"}:
            eligible += 1
            row_action_schema = metadata.get(
                "actor_execution_schema_fingerprint"
                if persistent_v2
                else "action_schema_fingerprint"
            )
            if (
                expected_action_schema is not None
                and row_action_schema != expected_action_schema
            ):
                raise ValueError(
                    f"eligible row action schema mismatch: {path} t={row.get('t')} "
                    f"{row_action_schema!r} != {expected_action_schema!r}"
                )
            if (
                expected_projection_profile is not None
                and metadata.get("actor_projection_profile") != expected_projection_profile
            ):
                raise ValueError(
                    f"eligible row Actor projection profile mismatch: {path} t={row.get('t')} "
                    f"{metadata.get('actor_projection_profile')!r} != {expected_projection_profile!r}"
                )
            z_rl = np.asarray(row.get("z_rl"), dtype=np.float32)
            a_ref = np.asarray(row.get("a_ref"), dtype=np.float32)
            if z_rl.shape != (2048,) or not np.all(np.isfinite(z_rl)):
                raise ValueError(
                    f"eligible row lacks true finite z_rl(2048): {path} t={row.get('t')}"
                )
            if (
                a_ref.ndim != 2
                or a_ref.shape[0] < 10
                or a_ref.shape[1] != 7
                or not np.all(np.isfinite(a_ref[:10]))
            ):
                raise ValueError(
                    f"eligible row lacks finite C=10 a_ref: {path} t={row.get('t')}"
                )
            if metadata.get("actor_shadow_z_is_true") is True:
                true_token += 1
            if row.get("gate_active"):
                gate_active += 1
                current_segment.append(source)
            elif current_segment:
                trainable_segments.append(current_segment)
                current_segment = []
            if source == "human_pika":
                human_eligible_rows += 1
            sources[source] = sources.get(source, 0) + 1
        elif current_segment:
            trainable_segments.append(current_segment)
            current_segment = []
    if current_segment:
        trainable_segments.append(current_segment)
    if eligible == 0 or true_token != eligible:
        raise ValueError(
            f"episode has no trustworthy executable Token rows: {path} ({true_token}/{eligible})"
        )
    transition_sources = _transition_anchor_sources(trainable_segments)
    if persistent_audit is not None:
        transition_sources = (
            ["rlt"] * int(persistent_audit["complete_actor_c10_chunks"])
            + ["human_pika"]
            * int(persistent_audit["complete_human_c10_chunks"])
        )
    return {
        "episode": str(path.parent.name),
        "episode_id": str(path.parent.name),
        "reward": float(rows[-1]["reward"]),
        "rows": len(rows),
        "eligible_rows": eligible,
        "gate_active_eligible_rows": gate_active,
        "trainable_transitions": len(transition_sources),
        "human_eligible_rows": human_eligible_rows,
        "human_trainable_transitions": sum(
            source == "human_pika" for source in transition_sources
        ),
        "sources": sources,
        "action_schema_fingerprint": expected_action_schema,
        "actor_projection_profile": expected_projection_profile,
        "actor_execution_profile": expected_execution_profile,
        "persistent_v2_contract": persistent_audit,
    }


def _transition_anchor_sources(
    segments: list[list[str]], *, chunk_length: int = 10, stride: int = 2
) -> list[str]:
    """Mirror C=10/stride=2 transition eligibility for gate-active row segments.

    Replay enrichment moves the sparse terminal marker to the final valid row.
    Therefore the last segment can emit padded terminal chunks, while earlier
    segments require a real state at t+n for nonterminal bootstrapping.
    """

    anchors: list[str] = []
    for segment_index, segment in enumerate(segments):
        length = len(segment)
        if segment_index == len(segments) - 1:
            starts = range(0, length, stride)
        else:
            starts = range(0, max(0, length - chunk_length), stride)
        anchors.extend(segment[start] for start in starts)
    return anchors


def _load_quarantine_registry(path: Path) -> dict[str, str]:
    """Load a small, reviewable registry without coupling it to one JSON layout."""

    if not path.is_file():
        return {}
    payload = _load_json(path)
    entries: Any = payload
    if isinstance(payload, dict):
        entries = payload.get("episodes", payload.get("episode_ids", payload))
    result: dict[str, str] = {}
    if isinstance(entries, list):
        for item in entries:
            if isinstance(item, str):
                result[item] = "quarantined"
            elif isinstance(item, dict) and item.get("episode_id"):
                if item.get("active", True):
                    result[str(item["episode_id"])] = str(
                        item.get("reason", "quarantined")
                    )
    elif isinstance(entries, dict):
        ignored_metadata_keys = {"format", "version", "updated_unix", "notes"}
        for episode_id, detail in entries.items():
            if episode_id in ignored_metadata_keys or detail is False or detail is None:
                continue
            if isinstance(detail, dict):
                if not detail.get("active", True):
                    continue
                reason = str(detail.get("reason", "quarantined"))
            elif detail is True:
                reason = "quarantined"
            else:
                reason = str(detail)
            result[str(episode_id)] = reason
    else:
        raise ValueError(f"unsupported quarantine registry layout: {path}")
    return dict(sorted(result.items()))


def _state_episode_ids(state: dict[str, Any], key: str) -> set[str] | None:
    value = state.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(
            f"online state field {key!r} must be a list of stable episode IDs"
        )
    return set(value)


def _migrate_legacy_episode_ids(
    explicit_ids: set[str] | None,
    *,
    legacy_count: int,
    current_ids: list[str],
) -> tuple[set[str], bool]:
    if explicit_ids is not None:
        return set(explicit_ids), False
    if legacy_count <= 0:
        return set(), False
    # Legacy state only stored a count.  Preserve its scheduling behavior once,
    # then persist concrete IDs after the next accepted/rejected attempt.
    return set(current_ids[: min(legacy_count, len(current_ids))]), True


def _episode_set_tag(episode_ids: list[str]) -> str:
    payload = "\n".join(sorted(episode_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _replay_episode_ids(path: Path) -> set[str]:
    with np.load(path, allow_pickle=False) as replay:
        if "episode_id" not in replay.files:
            raise ValueError("online replay is missing episode_id")
        return set(np.asarray(replay["episode_id"]).astype(str).tolist())


def _replay_overall_quality(path: Path) -> dict[str, int]:
    with np.load(path, allow_pickle=False) as replay:
        episode_ids = np.asarray(replay["episode_id"]).astype(str)
        reward = np.asarray(replay["reward"], dtype=np.float32)
        human_rows = np.asarray(replay["human_mask"], dtype=bool).any(axis=1)
    successes = failures = human_episodes = 0
    success_human_episodes = failure_human_episodes = 0
    for episode_id in np.unique(episode_ids):
        mask = episode_ids == episode_id
        success = bool(np.any(reward[mask] > 0.5))
        human = bool(np.any(human_rows[mask]))
        successes += int(success)
        failures += int(not success)
        human_episodes += int(human)
        success_human_episodes += int(success and human)
        failure_human_episodes += int((not success) and human)
    return {
        "episodes": int(len(np.unique(episode_ids))),
        "successes": successes,
        "failures": failures,
        "human_episodes": human_episodes,
        "success_human_episodes": success_human_episodes,
        "failure_human_episodes": failure_human_episodes,
        "human_transitions": int(np.count_nonzero(human_rows)),
        "transitions": int(episode_ids.shape[0]),
    }


def _merge_replays(base_path: Path, incremental_path: Path, output_path: Path) -> None:
    with np.load(base_path, allow_pickle=False) as base, np.load(
        incremental_path, allow_pickle=False
    ) as incremental:
        base_arrays = {
            key: np.asarray(base[key])
            for key in base.files
        }
        incremental_arrays = {
            key: np.asarray(incremental[key])
            for key in incremental.files
        }
        base_arrays, incremental_arrays = (
            _normalize_gripper_v3_replay_schemas_for_merge(
                base_arrays,
                incremental_arrays,
            )
        )
        if set(base_arrays) != set(incremental_arrays):
            raise ValueError(
                "frozen base and incremental replay schemas do not match: "
                f"{sorted(base_arrays)} != {sorted(incremental_arrays)}"
            )
        base_ids = set(
            np.asarray(base_arrays["episode_id"]).astype(str).tolist()
        )
        incremental_ids = set(
            np.asarray(incremental_arrays["episode_id"]).astype(str).tolist()
        )
        overlap = sorted(base_ids.intersection(incremental_ids))
        if overlap:
            raise ValueError(
                f"incremental replay duplicates frozen episode IDs: {overlap[:5]}"
            )
        merged = {
            key: np.concatenate(
                [base_arrays[key], incremental_arrays[key]], axis=0
            )
            for key in base_arrays
        }
    temporary = output_path.with_name(f".{output_path.name}.merge.tmp.npz")
    np.savez_compressed(temporary, **merged)
    temporary.replace(output_path)


def _normalize_gripper_v3_replay_schemas_for_merge(
    base: dict[str, np.ndarray],
    incremental: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Project only the audited historical-bootstrap telemetry extension.

    The immutable gripper-v3 bootstrap was created from frozen-gripper
    persistent-v2 data before the online replay writer began serializing six
    gripper execution-evidence arrays.  These arrays are required and retained
    in every new incremental archive, but the A-C learner does not consume
    them.  Keep both source archives untouched and omit only this exact
    allowlist from the derived combined learner replay.  Every other mismatch
    remains fail-closed.
    """

    if set(base) == set(incremental):
        return base, incremental

    base_only = set(base).difference(incremental)
    incremental_only = set(incremental).difference(base)
    if (
        base_only
        or incremental_only
        != _HISTORICAL_V3_BOOTSTRAP_INCREMENTAL_ONLY_KEYS
    ):
        return base, incremental

    for label, arrays in (("base", base), ("incremental", incremental)):
        if "action_schema_fingerprint" not in arrays:
            raise ValueError(f"{label} replay lacks its action schema")
        schemas = set(
            np.asarray(arrays["action_schema_fingerprint"]).astype(str).tolist()
        )
        if schemas != {PERSISTENT_V2_ACTION_SCHEMA_FINGERPRINT}:
            raise ValueError(
                f"{label} replay is not the gripper-v3 execution schema: "
                f"{sorted(schemas)}"
            )

    base_rows = int(np.asarray(base["episode_id"]).shape[0])
    incremental_rows = int(np.asarray(incremental["episode_id"]).shape[0])
    expected_optional_shapes = {
        "actor_persistent_planned_residual": (
            incremental_rows,
            10,
            7,
        ),
        "actor_gripper_release_intent": (incremental_rows, 10),
        "actor_filtered_actual_gripper_residual_max": (
            incremental_rows,
            10,
        ),
        "actor_filtered_actual_gripper_residual_d1_max_m": (
            incremental_rows,
            10,
        ),
        "actor_filtered_actual_gripper_residual_d2_max_m": (
            incremental_rows,
            10,
        ),
        "actor_filtered_actual_gripper_boundary_jump_max_m": (
            incremental_rows,
            10,
        ),
    }
    for name, expected_shape in expected_optional_shapes.items():
        actual_shape = np.asarray(incremental[name]).shape
        if actual_shape != expected_shape:
            raise ValueError(
                f"incremental gripper-v3 telemetry {name} has shape "
                f"{actual_shape}, expected {expected_shape}"
            )

    common = set(base).intersection(incremental)
    for name in common:
        base_value = np.asarray(base[name])
        incremental_value = np.asarray(incremental[name])
        if base_value.shape[:1] != (base_rows,):
            raise ValueError(
                f"base replay array {name!r} has inconsistent row count"
            )
        if incremental_value.shape[:1] != (incremental_rows,):
            raise ValueError(
                f"incremental replay array {name!r} has inconsistent row count"
            )
        if base_value.shape[1:] != incremental_value.shape[1:]:
            raise ValueError(
                f"replay array {name!r} trailing shapes do not match: "
                f"{base_value.shape} != {incremental_value.shape}"
            )
        if (
            base_value.dtype != incremental_value.dtype
            and not (
                base_value.dtype.kind in {"U", "S"}
                and incremental_value.dtype.kind
                in {"U", "S"}
            )
        ):
            raise ValueError(
                f"replay array {name!r} dtypes do not match: "
                f"{base_value.dtype} != {incremental_value.dtype}"
            )

    print(
        "[online-rlt] historical gripper-v3 bootstrap projection: "
        "the six newer execution-telemetry arrays remain in "
        "incremental_replay.npz and are omitted only from the combined "
        "learner replay.",
        flush=True,
    )
    return (
        {name: base[name] for name in common},
        {name: incremental[name] for name in common},
    )


def _replay_transition_counts(path: Path) -> dict[str, int]:
    with np.load(path, allow_pickle=False) as replay:
        if "episode_split" not in replay.files:
            raise ValueError("online replay is missing episode_split")
        labels = np.asarray(replay["episode_split"]).astype(str)
    return {
        "all": int(labels.size),
        "train": int(np.count_nonzero(labels == "train")),
        "validation": int(np.count_nonzero(labels == "validation")),
        "test": int(np.count_nonzero(labels == "test")),
    }


def _replay_split_quality(path: Path) -> dict[str, dict[str, int]]:
    """Measure S/F/H coverage on the rows the learner will actually sample."""

    with np.load(path, allow_pickle=False) as archive:
        split = np.asarray(archive["episode_split"]).astype(str)
        episode_ids = np.asarray(archive["episode_id"]).astype(str)
        reward = np.asarray(archive["reward"], dtype=np.float32)
        if "human_mask" in archive.files:
            human_mask = np.asarray(archive["human_mask"], dtype=bool)
            human_rows = (
                human_mask
                if human_mask.ndim == 1
                else np.any(human_mask, axis=tuple(range(1, human_mask.ndim)))
            )
        else:
            human_rows = np.zeros(len(split), dtype=bool)
    result: dict[str, dict[str, int]] = {}
    for split_name in ("train", "validation", "test"):
        split_mask = split == split_name
        success_episodes = 0
        failure_episodes = 0
        human_episodes = 0
        success_human_episodes = 0
        split_episode_ids = np.unique(episode_ids[split_mask])
        for episode_id in split_episode_ids:
            episode_mask = split_mask & (episode_ids == episode_id)
            success = bool(np.any(reward[episode_mask] > 0.5))
            human = bool(np.any(human_rows[episode_mask]))
            success_episodes += int(success)
            failure_episodes += int(not success)
            human_episodes += int(human)
            success_human_episodes += int(success and human)
        result[split_name] = {
            "transitions": int(np.count_nonzero(split_mask)),
            "episodes": int(len(split_episode_ids)),
            "success_episodes": success_episodes,
            "failure_episodes": failure_episodes,
            "human_episodes": human_episodes,
            "success_human_episodes": success_human_episodes,
        }
    return result


def _last_train_transition_count(state: dict[str, Any]) -> int:
    if "last_train_transition_count" in state:
        return int(state["last_train_transition_count"])
    latest_replay = state.get("latest_replay")
    if latest_replay and Path(latest_replay).is_file():
        return _replay_transition_counts(Path(latest_replay))["train"]
    return 0


def _compute_training_steps(
    new_train_transitions: int, *, utd: float, minimum: int, maximum: int
) -> int:
    if new_train_transitions <= 0:
        raise ValueError("new_train_transitions must be positive")
    if not np.isfinite(utd) or utd <= 0:
        raise ValueError("utd must be finite and positive")
    if minimum <= 0 or maximum < minimum:
        raise ValueError("invalid learner step bounds")
    requested = int(math.ceil(new_train_transitions * utd))
    return max(minimum, min(maximum, requested))


def _validate_promotion(report: dict[str, Any], args: argparse.Namespace) -> None:
    max_td_error = float(args.max_validation_td_error)
    max_q_advantage = float(args.max_actor_q_advantage)
    if not math.isfinite(max_td_error) or max_td_error < 0.0:
        raise RuntimeError(
            "validation TD-error promotion threshold must be finite and non-negative"
        )
    if not math.isfinite(max_q_advantage) or max_q_advantage < 0.0:
        raise RuntimeError(
            "Actor Q-advantage promotion threshold must be finite and non-negative"
        )
    if _is_persistent_v2(args):
        try:
            semantics_version = int(
                report.get("temporal_metrics_semantics_version", -1)
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "persistent-v2 Actor validation has an invalid semantics version"
            ) from exc
        expected_semantics = {
            "actor_candidate_semantics": "persistent_filtered_physical_candidate",
            "actor_q_baseline_semantics": "zero_residual_a_base_filtered",
            "rank1_metrics_action_role": "checkpoint_output_protocol_only",
            "actor_bc_semantics": (
                "filtered_actual_residual_from_a_base_filtered"
            ),
            "critic_behavior_action_semantics": "replay_a_exec",
            "critic_td_target_candidate_semantics": (
                "persistent_filtered_target_actor_candidate"
            ),
        }
        if semantics_version != 4:
            raise RuntimeError(
                "persistent-v2 Actor validation did not use unified physical "
                "candidate semantics version 4"
            )
        for field, expected in expected_semantics.items():
            if report.get(field) != expected:
                raise RuntimeError(
                    "persistent-v2 Actor validation semantics mismatch for "
                    f"{field}: {report.get(field)!r} != {expected!r}"
                )
    try:
        td_error = float(report.get("validation_td_error_abs_mean", float("nan")))
        q_advantage = float(report.get("actor_q_advantage_abs_p95", float("nan")))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Actor validation reported non-numeric TD/Q metrics"
        ) from exc
    if not math.isfinite(td_error):
        raise RuntimeError("Actor validation reported a non-finite TD error")
    if not math.isfinite(q_advantage):
        raise RuntimeError("Actor validation reported a non-finite Q advantage")
    enabled_contracts = report.get("optional_promotion_contracts")
    enabled_contracts = (
        set(enabled_contracts) if isinstance(enabled_contracts, list) else set()
    )
    if (
        any(
            getattr(args, argument_name, None) is not None
            for argument_name, _, _ in ACTOR_VALIDATION_GATE_SPECS
        )
        and report.get("legacy_normalized_residual_temporal_step_promotion_required")
        is not False
    ):
        raise RuntimeError(
            "Actor validation did not replace the legacy all-dimension temporal gate "
            "with the calibrated active-dimension gate"
        )
    for argument_name, contract_name, threshold_name in ACTOR_VALIDATION_GATE_SPECS:
        expected_threshold = getattr(args, argument_name, None)
        if expected_threshold is None:
            continue
        if contract_name not in enabled_contracts:
            raise RuntimeError(
                f"Actor validation did not attest required calibrated gate {contract_name}"
            )
        if report.get(contract_name) is not True:
            raise RuntimeError(
                f"Actor validation failed calibrated gate {contract_name}"
            )
        try:
            reported_threshold = float(report.get(threshold_name, float("nan")))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Actor validation reported a non-numeric threshold for {contract_name}"
            ) from exc
        if not math.isfinite(reported_threshold) or not math.isclose(
            reported_threshold,
            float(expected_threshold),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"Actor validation threshold mismatch for {contract_name}: "
                f"expected {expected_threshold}, got {reported_threshold}"
            )
    if (
        report.get("passed") is not True
        or report.get("finite_action") is not True
        or report.get("finite_action_inputs") is not True
        or report.get("finite_action_outputs") is not True
        or report.get("finite_q_values") is not True
        or report.get("finite_td_target") is not True
        or report.get("finite_key_metrics") is not True
    ):
        raise RuntimeError("Actor validation did not pass")
    if td_error > max_td_error:
        raise RuntimeError("validation TD error exceeds the promotion threshold")
    if q_advantage > max_q_advantage:
        raise RuntimeError(
            "Actor Q-advantage p95 suggests extrapolation beyond the promotion threshold"
        )
    min_reward_gap = getattr(
        args, "min_reward1_reward0_exec_q_gap", None
    )
    if min_reward_gap is not None:
        min_reward_gap = float(min_reward_gap)
        if not math.isfinite(min_reward_gap):
            raise RuntimeError(
                "reward1/reward0 Q-gap promotion threshold must be finite"
            )
        try:
            reward_gap = float(
                report.get(
                    "reward1_reward0_exec_q_gap",
                    report.get(
                        "success_failure_exec_q_gap",
                        float("nan"),
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Actor validation reported a non-numeric reward1/reward0 Q gap"
            ) from exc
        if not math.isfinite(reward_gap):
            raise RuntimeError(
                "Actor validation did not contain both reward-1 and reward-0 "
                "held-out Q values"
            )
        if reward_gap < min_reward_gap:
            raise RuntimeError(
                "reward-1 held-out executions have insufficient Q advantage "
                "over reward-0 executions"
            )


def _smoke_observation(path: Path) -> tuple[Path, int]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidates = [
        row
        for row in rows
        if row.get("gate_active")
        and (row.get("policy_metadata") or {}).get("replay_include")
    ]
    row = candidates[-1] if candidates else rows[max(0, len(rows) - 2)]
    return path, int(row["t"])


def _run(
    command: list[Any],
    *,
    env: dict[str, str],
    log_path: Path,
    accepted_returncodes: tuple[int, ...] = (0,),
) -> None:
    rendered = [str(value) for value in command]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            rendered, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    if completed.returncode not in accepted_returncodes:
        tail = "\n".join(
            log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        )
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(rendered)}\n{tail}"
        )


def _validation_env(env: dict[str, str]) -> dict[str, str]:
    """Run the small acceptance check without competing with the policy GPU."""

    validation_env = env.copy()
    validation_env["JAX_PLATFORMS"] = "cpu"
    validation_env["CUDA_VISIBLE_DEVICES"] = ""
    validation_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    return validation_env


def _training_env(env: dict[str, str]) -> dict[str, str]:
    """Require CUDA and reserve a bounded learner pool beside the policy server."""

    training_env = env.copy()
    # Do not inherit a parent shell's CPU-only validation/debug settings. The
    # online learner is required to run on the only physical GPU (RTX 5090).
    training_env["JAX_PLATFORMS"] = "cuda"
    training_env["CUDA_VISIBLE_DEVICES"] = str(LEARNER_GPU_INDEX)
    # The full Pi0.5/RL-token policy stays resident (~9.5 GiB). Give the much
    # smaller A-C learner a deterministic 20% pool (~4.8 GiB on this 24 GiB
    # card), leaving roughly 10 GiB for the policy, CUDA libraries and desktop.
    # A fixed pool is safer than JAX's default 75% reservation or unbounded
    # grow-on-demand allocation when two JAX processes share one GPU.
    training_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    training_env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{LEARNER_GPU_MEMORY_FRACTION:.2f}"
    return training_env


def _learner_gpu_memory_preflight() -> dict[str, int | float | str]:
    """Refuse a new learner process when the shared 5090 lacks safe headroom."""

    command = [
        "nvidia-smi",
        f"--id={LEARNER_GPU_INDEX}",
        "--query-gpu=name,memory.total,memory.free,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"cannot query learner GPU memory safely: {exc}") from exc
    if completed.returncode != 0:
        raise RuntimeError(
            "cannot query learner GPU memory safely: "
            + (completed.stderr.strip() or f"nvidia-smi exit {completed.returncode}")
        )
    fields = [part.strip() for part in completed.stdout.strip().split(",")]
    if len(fields) != 4:
        raise RuntimeError(f"unexpected nvidia-smi learner GPU report: {completed.stdout!r}")
    name = fields[0]
    try:
        total_mib, free_mib, used_mib = (int(value) for value in fields[1:])
    except ValueError as exc:
        raise RuntimeError(f"invalid nvidia-smi memory values: {fields[1:]!r}") from exc
    reserved_mib = int(math.ceil(total_mib * LEARNER_GPU_MEMORY_FRACTION))
    if free_mib < LEARNER_GPU_MIN_FREE_MIB:
        raise RuntimeError(
            f"GPU {LEARNER_GPU_INDEX} free memory {free_mib} MiB is below the safe online learner "
            f"threshold {LEARNER_GPU_MIN_FREE_MIB} MiB; incumbent Actor remains selected"
        )
    if free_mib < reserved_mib + 2_048:
        raise RuntimeError(
            f"GPU {LEARNER_GPU_INDEX} cannot reserve the {reserved_mib} MiB learner pool plus "
            "2048 MiB CUDA headroom; incumbent Actor remains selected"
        )
    return {
        "index": LEARNER_GPU_INDEX,
        "name": name,
        "total_mib": total_mib,
        "free_mib": free_mib,
        "used_mib": used_mib,
        "minimum_free_mib": LEARNER_GPU_MIN_FREE_MIB,
        "pool_fraction": LEARNER_GPU_MEMORY_FRACTION,
        "pool_mib": reserved_mib,
    }


def _restart_service(service: str, *, host: str, port: int) -> None:
    subprocess.run(["systemctl", "--user", "restart", service], check=True)
    deadline = time.monotonic() + 180.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(2.0)
    raise TimeoutError(f"timed out waiting for {service} on {host}:{port}")


def _read_selected(path: Path) -> str | None:
    if not path.is_file():
        return None
    return next(
        (
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ),
        None,
    )


def _write_selected(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(str(value).strip() + "\n", encoding="utf-8")
    temporary.replace(path)


def _block_incumbent_for_clean_retrain(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    state_path: Path,
    reason: str,
    latest_checkpoint: str | None,
) -> dict[str, Any]:
    """Fail closed when replay provenance invalidates the current Actor."""

    if bool(getattr(args, "dry_run", False)):
        return {"performed": False, "dry_run": True, "reason": reason}
    blocked_state = {
        **state,
        "format": _state_format(_is_persistent_v2(args)),
        "session_root": str(args.session_root),
        "latest_checkpoint": None,
        "latest_quarantined_checkpoint": latest_checkpoint,
        "latest_replay": None,
        "trained_episode_ids": [],
        "last_attempt_episode_ids": [],
        "last_update_episode_count": 0,
        "last_attempt_episode_count": 0,
        "last_train_transition_count": 0,
        "last_attempt_train_transition_count": 0,
        "requires_clean_retrain_from_step_zero": True,
        "clean_retrain_reason": reason,
        "updated_unix": time.time(),
    }
    _atomic_json(state_path, blocked_state)
    selected_path = Path(args.selected_checkpoint_file).expanduser().resolve()
    _write_selected(selected_path, "NONE")
    try:
        _restart_service(
            args.shadow_service, host=args.policy_host, port=args.policy_port
        )
    except Exception:
        # If the base-only service cannot be restarted, stopping it is safer
        # than allowing the already-loaded contaminated Actor to remain live.
        subprocess.run(
            ["systemctl", "--user", "stop", args.shadow_service], check=False
        )
        raise
    return {
        "performed": True,
        "selected_checkpoint": "NONE",
        "previous_checkpoint": latest_checkpoint,
        "requires_clean_retrain_from_step_zero": True,
    }


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _record_event(root: Path, value: dict[str, Any]) -> None:
    row = {"timestamp_unix": time.time(), **value}
    with (root / "online_updates.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
