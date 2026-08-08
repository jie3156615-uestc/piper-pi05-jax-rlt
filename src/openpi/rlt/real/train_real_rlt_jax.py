#!/usr/bin/env python3
"""Train the offline JAX residual Actor-Critic on an enriched Piper replay."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

# The 5090 also has an older /home/cwzk/openpi_runtime checkout on
# PYTHONPATH.  Production CLIs must bind to the workspace that contains this
# script, otherwise they can silently train a stale implementation.
_WORKSPACE_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_WORKSPACE_SRC))

from openpi.rlt.real.agent_jax import (
    RealRLTLearner,  # noqa: E402
    ReplayBatchSampler,  # noqa: E402
    ReplaySamplingConfig,  # noqa: E402
    estimate_residual_limit,  # noqa: E402
    filter_replay_by_split,  # noqa: E402
    rank1_direction_limit,  # noqa: E402
    sha256_file,  # noqa: E402
)
from openpi.rlt.real.config import LEGACY_ACTOR_EXECUTION_PROFILE  # noqa: E402
from openpi.rlt.real.config import GRIPPER_RESIDUAL_CLOSE_ASSIST  # noqa: E402
from openpi.rlt.real.config import GRIPPER_RESIDUAL_FROZEN  # noqa: E402
from openpi.rlt.real.config import (  # noqa: E402
    HUMAN_GRIPPER_Q_FILTER_CRITIC_MIN_ADVANTAGE,
)
from openpi.rlt.real.config import (  # noqa: E402
    MIN_GRIPPER_CLOSE_CRITIC_BURN_IN_UPDATES,
)
from openpi.rlt.real.config import PERSISTENT_ACTOR_EXECUTION_PROFILE  # noqa: E402
from openpi.rlt.real.config import PERSISTENT_EXECUTION_FILTER_PROFILE  # noqa: E402
from openpi.rlt.real.config import PERSISTENT_GRIPPER_CLOSE_GOVERNOR_PROFILE  # noqa: E402
from openpi.rlt.real.config import PERSISTENT_GOVERNOR_PROFILE  # noqa: E402
from openpi.rlt.real.config import RealRLTConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="train")
    parser.add_argument("--success-fraction", type=float)
    parser.add_argument("--failure-fraction", type=float)
    parser.add_argument("--human-fraction", type=float)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument(
        "--actor-start-step",
        type=int,
        help=(
            "Absolute learner step after which Actor/Q-filter updates may "
            "start. Close-assist fresh training defaults to 144; checkpoint "
            "resume preserves its stored value."
        ),
    )
    parser.add_argument(
        "--beta-bc",
        type=float,
        help=(
            "Reference BC weight. Fresh training defaults to RealRLTConfig; "
            "an Actor-only warm-start inherits the source checkpoint value."
        ),
    )
    parser.add_argument(
        "--beta-human-bc",
        type=float,
        help="Optional intervention BC weight; defaults to 0 and never replaces the original reference BC",
    )
    parser.add_argument(
        "--beta-human-gripper-bc",
        type=float,
        default=0.0,
        help=(
            "Q-filtered, dim-6 admitted-human imitation weight. Reward 0/1 "
            "both enter the filter; this never changes six-joint human BC."
        ),
    )
    parser.add_argument(
        "--human-gripper-bc-scale-m",
        type=float,
        default=0.005,
        help="Physical normalization scale for Q-filtered gripper BC",
    )
    parser.add_argument(
        "--human-gripper-q-filter-mode",
        choices=(HUMAN_GRIPPER_Q_FILTER_CRITIC_MIN_ADVANTAGE,),
        default=HUMAN_GRIPPER_Q_FILTER_CRITIC_MIN_ADVANTAGE,
        help=(
            "Use clipped-twin Critic advantage to select admitted human "
            "gripper teachers independently of terminal reward."
        ),
    )
    parser.add_argument(
        "--human-gripper-q-filter-margin",
        type=float,
        default=0.0,
        help=(
            "Minimum Q_human-Q_actor required before cloning an admitted "
            "human gripper chunk."
        ),
    )
    parser.add_argument("--reference-dropout", type=float, default=0.5)
    parser.add_argument(
        "--hidden-dim",
        type=int,
        help=(
            "Actor/Critic hidden width. Fresh training defaults to RealRLTConfig; "
            "an Actor-only warm-start inherits the source Actor architecture."
        ),
    )
    parser.add_argument(
        "--projection-dim",
        type=int,
        help=(
            "Actor/Critic projection width. Fresh training defaults to RealRLTConfig; "
            "an Actor-only warm-start inherits the source Actor architecture."
        ),
    )
    parser.add_argument(
        "--target-policy-noise-std",
        type=float,
        help="TD3 target noise std as a fraction of each residual limit (default: config value)",
    )
    parser.add_argument(
        "--target-policy-noise-clip",
        type=float,
        help="TD3 target noise clip as a fraction of each residual limit (default: config value)",
    )
    parser.add_argument(
        "--actor-residual-parameterization",
        choices=("rank1_bump", "legacy_full_chunk"),
        default="rank1_bump",
        help="Training must use rank1_bump; legacy_full_chunk checkpoints are read-only and cannot be resumed",
    )
    parser.add_argument("--actor-residual-max-rad", type=float, default=0.005)
    parser.add_argument("--actor-residual-d1-max-rad", type=float, default=0.0015)
    parser.add_argument("--actor-residual-d2-max-rad", type=float, default=0.001)
    parser.add_argument("--actor-direction-cone-deg", type=float, default=15.0)
    parser.add_argument(
        "--actor-execution-profile",
        choices=(LEGACY_ACTOR_EXECUTION_PROFILE, PERSISTENT_ACTOR_EXECUTION_PROFILE),
        default=LEGACY_ACTOR_EXECUTION_PROFILE,
    )
    parser.add_argument(
        "--execution-filter-profile",
        default=PERSISTENT_EXECUTION_FILTER_PROFILE,
    )
    parser.add_argument("--execution-filter-tau-s", type=float, default=0.05)
    parser.add_argument("--actor-max-boundary-jump-rad", type=float, default=0.06)
    parser.add_argument("--actor-direction-static-threshold-rad", type=float, default=0.001)
    parser.add_argument("--actor-projection-scale-steps", type=int, default=33)
    parser.add_argument("--actor-min-projection-scale", type=float, default=0.2)
    parser.add_argument(
        "--actor-governor-fingerprint",
        "--persistent-governor-profile",
        dest="actor_governor_fingerprint",
        default=None,
    )
    parser.add_argument("--residual-percentile", type=float, default=99.0)
    parser.add_argument("--residual-min", type=float, default=1e-3)
    parser.add_argument("--residual-max", type=float, help="Maximum residual for the six arm joints, in radians")
    parser.add_argument(
        "--gripper-residual-max",
        type=float,
        default=0.005,
        help="Maximum gripper residual in metres (kept separate from joint radians)",
    )
    parser.add_argument(
        "--freeze-gripper-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep Pi0.5 gripper pass-through for the frozen-v2 contract",
    )
    parser.add_argument(
        "--gripper-residual-mode",
        choices=(GRIPPER_RESIDUAL_FROZEN, GRIPPER_RESIDUAL_CLOSE_ASSIST),
        default=GRIPPER_RESIDUAL_FROZEN,
    )
    parser.add_argument("--gripper-residual-d1-max-m", type=float, default=0.0005)
    parser.add_argument("--gripper-residual-d2-max-m", type=float, default=0.0003)
    parser.add_argument("--gripper-boundary-max-m", type=float, default=0.0005)
    parser.add_argument("--gripper-command-min-m", type=float, default=0.0)
    parser.add_argument("--gripper-command-max-m", type=float, default=0.08)
    parser.add_argument("--gripper-release-reference-m", type=float, default=0.05)
    parser.add_argument("--gripper-release-delta-m", type=float, default=0.002)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--warm-start-actor-checkpoint",
        type=Path,
        help=(
            "One-way persistent-v2 migration: restore only the old rank1 Actor "
            "parameters; initialize a fresh Critic, optimizers, RNG, and step 0"
        ),
    )
    parser.add_argument(
        "--fresh-zero-lineage-manifest",
        type=Path,
        help=(
            "Explicit fresh-zero gripper-v3 provenance manifest. The manifest, "
            "bound online state, output directory, and replay location are "
            "validated before its SHA256 is embedded in the checkpoint."
        ),
    )
    parser.add_argument(
        "--warm-start-dry-run-report",
        type=Path,
        help="Write the audited migration report before the first optimizer update",
    )
    parser.add_argument(
        "--allow-warm-start-objective-migration",
        action="store_true",
        help=(
            "Explicitly authorize beta_bc/beta_human_bc to differ from the source "
            "Actor checkpoint during the one-way persistent-v2 warm-start. The "
            "source and target weights are recorded in checkpoint fingerprints "
            "and the migration report."
        ),
    )
    parser.add_argument(
        "--allow-replay-refresh",
        action="store_true",
        help=(
            "Resume optimizer/target/norm state while training on a newly enriched replay snapshot. "
            "Base and RL-Token fingerprints must still match; normalization and residual limits stay frozen."
        ),
    )
    parser.add_argument("--base-checkpoint-fingerprint", required=True)
    parser.add_argument("--rl-token-fingerprint", required=True)
    parser.add_argument("--phase-fingerprint", required=True)
    parser.add_argument("--action-schema-fingerprint", required=True)
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail before replay loading/training unless JAX is using a CUDA GPU backend",
    )
    return parser.parse_args()


def _runtime_device_report(*, require_gpu: bool) -> dict[str, object]:
    """Return auditable JAX placement and reject accidental CPU online training."""

    backend = jax.default_backend()
    devices = [
        {
            "id": int(getattr(device, "id", -1)),
            "platform": str(getattr(device, "platform", "unknown")),
            "device_kind": str(getattr(device, "device_kind", "unknown")),
        }
        for device in jax.devices()
    ]
    report: dict[str, object] = {
        "event": "jax_runtime",
        "backend": backend,
        "devices": devices,
        "require_gpu": bool(require_gpu),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "xla_memory_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
    }
    if require_gpu and (
        backend != "gpu" or not devices or any(item["platform"] != "gpu" for item in devices)
    ):
        raise RuntimeError(
            "online RLT learner requires a CUDA GPU, but JAX placement was "
            f"backend={backend!r}, devices={devices!r}"
        )
    return report


def load_replay(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _residual_cap(
    shape: tuple[int, ...],
    *,
    joint_max: float | None,
    gripper_max: float,
) -> np.ndarray:
    if len(shape) != 2 or shape[1] != 7:
        raise ValueError(f"Piper residual limit must have shape (chunk, 7), got {shape}")
    cap = np.full(shape, np.inf if joint_max is None else float(joint_max), dtype=np.float32)
    cap[:, 6] = float(gripper_max)
    return cap


def expected_actor_updates(
    *,
    start_step: int,
    additional_steps: int,
    actor_start_step: int,
    policy_delay: int,
) -> int:
    """Count delayed Actor updates over an absolute learner-step interval."""

    final_step = start_step + additional_steps
    first_eligible = max(start_step + 1, actor_start_step + 1)
    if first_eligible > final_step:
        return 0
    return (
        final_step // policy_delay
        - (first_eligible - 1) // policy_delay
    )


def _require_resume_config_matches(learner: RealRLTLearner, args: argparse.Namespace) -> None:
    """Reject CLI/checkpoint drift instead of silently training another objective.

    Optimizer learning rates are embedded in the restored learner, and the
    online session contract treats the remaining objective fields as immutable.
    A new experiment is required to change them.
    """

    requested: dict[str, object] = {
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "reference_dropout": args.reference_dropout,
        "freeze_gripper_residual": bool(args.freeze_gripper_residual),
        "gripper_residual_mode": args.gripper_residual_mode,
        "human_gripper_bc_scale_m": args.human_gripper_bc_scale_m,
        "human_gripper_q_filter_mode": args.human_gripper_q_filter_mode,
        "human_gripper_q_filter_margin": args.human_gripper_q_filter_margin,
        "actor_gripper_residual_max_close_m": args.gripper_residual_max,
        "actor_gripper_residual_d1_max_m": args.gripper_residual_d1_max_m,
        "actor_gripper_residual_d2_max_m": args.gripper_residual_d2_max_m,
        "actor_gripper_max_boundary_jump_m": args.gripper_boundary_max_m,
        "gripper_command_min_m": args.gripper_command_min_m,
        "gripper_command_max_m": args.gripper_command_max_m,
        "gripper_release_reference_m": args.gripper_release_reference_m,
        "gripper_release_delta_m": args.gripper_release_delta_m,
        "actor_residual_parameterization": args.actor_residual_parameterization,
        "actor_residual_max_rad": args.actor_residual_max_rad,
        "actor_residual_d1_max_rad": args.actor_residual_d1_max_rad,
        "actor_residual_d2_max_rad": args.actor_residual_d2_max_rad,
        "actor_direction_cone_deg": args.actor_direction_cone_deg,
        "actor_execution_profile": args.actor_execution_profile,
        "execution_filter_profile": args.execution_filter_profile,
        "execution_filter_tau_s": args.execution_filter_tau_s,
        "actor_max_boundary_jump_rad": args.actor_max_boundary_jump_rad,
        "actor_direction_static_threshold_rad": args.actor_direction_static_threshold_rad,
        "actor_projection_scale_steps": args.actor_projection_scale_steps,
        "actor_min_projection_scale": args.actor_min_projection_scale,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    if args.actor_start_step is not None:
        requested["actor_start_step"] = args.actor_start_step
    optional = {
        "beta_bc": args.beta_bc,
        "beta_human_bc": args.beta_human_bc,
        "beta_human_gripper_bc": args.beta_human_gripper_bc,
        "hidden_dim": args.hidden_dim,
        "projection_dim": args.projection_dim,
        "target_policy_noise_std": args.target_policy_noise_std,
        "target_policy_noise_clip": args.target_policy_noise_clip,
    }
    requested.update({name: value for name, value in optional.items() if value is not None})

    mismatches: list[str] = []
    for name, requested_value in requested.items():
        checkpoint_value = getattr(learner.config, name)
        if isinstance(checkpoint_value, float) or isinstance(requested_value, float):
            matches = bool(np.isclose(float(checkpoint_value), float(requested_value), rtol=1e-9, atol=1e-12))
        else:
            matches = checkpoint_value == requested_value
        if not matches:
            mismatches.append(f"{name}: checkpoint={checkpoint_value!r}, requested={requested_value!r}")
    if mismatches:
        raise ValueError(
            "resume configuration differs from the checkpoint; start a new experiment instead: "
            + "; ".join(mismatches)
        )


def _fresh_zero_fingerprints(args: argparse.Namespace) -> dict[str, str]:
    manifest_path = args.fresh_zero_lineage_manifest
    if manifest_path is None:
        return {}
    if (
        args.actor_execution_profile != PERSISTENT_ACTOR_EXECUTION_PROFILE
        or args.gripper_residual_mode != GRIPPER_RESIDUAL_CLOSE_ASSIST
    ):
        raise ValueError(
            "--fresh-zero-lineage-manifest requires persistent gripper-close training"
        )
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(f"fresh-zero lineage manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_format = "openpi_piper_gripper_v3_fresh_zero_lineage_v1"
    if manifest.get("format") != expected_format:
        raise ValueError("fresh-zero lineage manifest format mismatch")
    inherited_fields = (
        "inherited_actor_checkpoints",
        "inherited_critic_checkpoints",
        "inherited_episode_ids",
        "inherited_replays",
    )
    contaminated = [
        name for name in inherited_fields if manifest.get(name) != []
    ]
    if contaminated:
        raise ValueError(
            "fresh-zero lineage manifest contains inherited provenance: "
            f"{contaminated}"
        )
    state_root = Path(str(manifest.get("state_root", ""))).expanduser().resolve()
    session_root = Path(str(manifest.get("session_root", ""))).expanduser().resolve()
    if manifest_path.parent != state_root:
        raise ValueError("fresh-zero manifest is not stored in its bound state root")
    if args.output_dir.expanduser().resolve() != state_root / "learner":
        raise ValueError("fresh-zero learner output is outside its bound state root")
    replay_path = args.replay_npz.expanduser().resolve()
    if state_root not in replay_path.parents or replay_path.parent.parent.name != "replays":
        raise ValueError("fresh-zero replay is outside its bound state root")
    state_path = state_root / "online_state.json"
    if not state_path.is_file():
        raise ValueError("fresh-zero online state is missing")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    expected_state = {
        "lineage_mode": "persistent_gripper_v3_fresh_zero",
        "fresh_zero": True,
        "replay_training_policy": "fresh_persistent_v5_online_only",
        "session_root": str(session_root),
    }
    mismatches = {
        name: (expected, state.get(name))
        for name, expected in expected_state.items()
        if state.get(name) != expected
    }
    if mismatches:
        raise ValueError(
            f"fresh-zero online-state provenance mismatch: {mismatches}"
        )
    forbidden_state_fields = (
        "initial_actor_warm_start_checkpoint",
        "bootstrap_gripper_replay",
        "legacy_source_replay",
    )
    inherited_state = [
        name
        for name in forbidden_state_fields
        if state.get(name) not in (None, "", [])
    ]
    if inherited_state:
        raise ValueError(
            "fresh-zero online state contains forbidden inherited provenance: "
            f"{inherited_state}"
        )
    return {
        "actor_initialization": "fresh_zero_random_v1",
        "fresh_zero_lineage_manifest_format": expected_format,
        "fresh_zero_lineage_manifest_sha256": sha256_file(manifest_path),
    }


def main() -> None:
    args = parse_args()
    print(json.dumps(_runtime_device_report(require_gpu=args.require_gpu), sort_keys=True), flush=True)
    if args.resume is not None and args.warm_start_actor_checkpoint is not None:
        raise ValueError("--resume and --warm-start-actor-checkpoint are mutually exclusive")
    if (
        args.fresh_zero_lineage_manifest is not None
        and args.warm_start_actor_checkpoint is not None
    ):
        raise ValueError(
            "--fresh-zero-lineage-manifest and --warm-start-actor-checkpoint "
            "are mutually exclusive"
        )
    if args.warm_start_dry_run_report is not None and args.warm_start_actor_checkpoint is None:
        raise ValueError("--warm-start-dry-run-report requires --warm-start-actor-checkpoint")
    if (
        args.allow_warm_start_objective_migration
        and args.warm_start_actor_checkpoint is None
    ):
        raise ValueError(
            "--allow-warm-start-objective-migration requires "
            "--warm-start-actor-checkpoint"
        )
    if args.steps < 0 or args.batch_size <= 0:
        raise ValueError("steps must be non-negative and batch-size must be positive")
    if args.actor_start_step is not None and args.actor_start_step < 0:
        raise ValueError("actor-start-step must be non-negative")
    if args.steps == 0 and args.warm_start_dry_run_report is None:
        raise ValueError("steps=0 is allowed only for an explicit warm-start dry run")
    expected_governor = (
        PERSISTENT_GRIPPER_CLOSE_GOVERNOR_PROFILE
        if args.gripper_residual_mode == GRIPPER_RESIDUAL_CLOSE_ASSIST
        else PERSISTENT_GOVERNOR_PROFILE
    )
    if args.actor_governor_fingerprint is None:
        args.actor_governor_fingerprint = expected_governor
    if (
        args.actor_execution_profile == PERSISTENT_ACTOR_EXECUTION_PROFILE
        and args.actor_governor_fingerprint != expected_governor
    ):
        raise ValueError(
            "persistent governor profile mismatch: "
            f"{args.actor_governor_fingerprint!r} != {expected_governor!r}"
        )
    if args.resume is None and args.actor_residual_parameterization != "rank1_bump":
        raise ValueError("new real-robot training must use --actor-residual-parameterization=rank1_bump")
    if args.residual_max is not None and args.residual_max <= 0:
        raise ValueError("residual-max must be positive")
    if args.gripper_residual_max < 0 or (args.gripper_residual_max == 0 and not args.freeze_gripper_residual):
        raise ValueError("gripper-residual-max must be positive unless the gripper residual is frozen")
    for name in (
        "beta_bc",
        "beta_human_bc",
        "beta_human_gripper_bc",
        "target_policy_noise_std",
        "target_policy_noise_clip",
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"{name.replace('_', '-')} must be non-negative")
    for name in (
        "human_gripper_bc_scale_m",
        "gripper_residual_d1_max_m",
        "gripper_residual_d2_max_m",
        "gripper_boundary_max_m",
        "gripper_command_max_m",
        "gripper_release_reference_m",
        "gripper_release_delta_m",
    ):
        if getattr(args, name) <= 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    for name in ("hidden_dim", "projection_dim"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    fresh_zero_fingerprints = _fresh_zero_fingerprints(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    replay_all = load_replay(args.replay_npz)
    replay = filter_replay_by_split(replay_all, args.split)
    fingerprints = {
        "replay_sha256": sha256_file(args.replay_npz),
        "base_checkpoint": args.base_checkpoint_fingerprint,
        "rl_token": args.rl_token_fingerprint,
        "phase_classifier": args.phase_fingerprint,
        "action_schema": args.action_schema_fingerprint,
        "actor_execution_profile": args.actor_execution_profile,
        "execution_filter_profile": args.execution_filter_profile,
        "actor_governor": args.actor_governor_fingerprint,
        **fresh_zero_fingerprints,
    }

    previous_fingerprints: dict[str, str] | None = None
    if args.resume:
        expected = dict(fingerprints)
        if args.actor_execution_profile != PERSISTENT_ACTOR_EXECUTION_PROFILE:
            for name in (
                "actor_execution_profile",
                "execution_filter_profile",
                "actor_governor",
            ):
                expected.pop(name, None)
        if args.allow_replay_refresh:
            expected = {
                "base_checkpoint": fingerprints["base_checkpoint"],
                "rl_token": fingerprints["rl_token"],
                "phase_classifier": fingerprints["phase_classifier"],
                "action_schema": fingerprints["action_schema"],
            }
            if args.actor_execution_profile == PERSISTENT_ACTOR_EXECUTION_PROFILE:
                expected.update(
                    {
                        "actor_execution_profile": fingerprints[
                            "actor_execution_profile"
                        ],
                        "execution_filter_profile": fingerprints[
                            "execution_filter_profile"
                        ],
                        "actor_governor": fingerprints["actor_governor"],
                    }
                )
            expected.update(fresh_zero_fingerprints)
        learner = RealRLTLearner.load_checkpoint(args.resume, expected_fingerprints=expected)
        if learner.config.actor_residual_parameterization != "rank1_bump":
            raise ValueError(
                "legacy_full_chunk Actor checkpoints are read-only and cannot be resumed for training"
            )
        _require_resume_config_matches(learner, args)
        previous_fingerprints = dict(learner.fingerprints)
        if (
            previous_fingerprints.get("actor_initialization")
            == "fresh_zero_random_v1"
            and not fresh_zero_fingerprints
        ):
            raise ValueError(
                "resuming a fresh-zero checkpoint requires its bound "
                "--fresh-zero-lineage-manifest"
            )
        if args.allow_replay_refresh:
            inherited_lineage = {
                key: value
                for key, value in previous_fingerprints.items()
                if key.startswith("warm_start_")
                or key.startswith("fresh_zero_")
                or key == "actor_initialization"
            }
            learner.fingerprints = {
                **inherited_lineage,
                **fingerprints,
            }
        effective_gripper_max = (
            0.0 if learner.config.freeze_gripper_residual else args.gripper_residual_max
        )
        residual_cap = _residual_cap(
            np.asarray(learner.residual_limit).shape,
            joint_max=args.residual_max,
            gripper_max=effective_gripper_max,
        )
        learner.residual_limit = jnp.asarray(
            np.minimum(np.asarray(learner.residual_limit, dtype=np.float32), residual_cap),
            dtype=jnp.float32,
        )
        learner.actor_limit = rank1_direction_limit(learner.residual_limit, learner.config)
        cfg = learner.config
    else:
        default_cfg = RealRLTConfig()
        source_cfg: dict[str, object] | None = None
        if args.warm_start_actor_checkpoint is not None:
            source_metadata_path = args.warm_start_actor_checkpoint / "metadata.json"
            if not source_metadata_path.is_file():
                raise ValueError(
                    "Actor-only warm-start requires checkpoint metadata.json so "
                    "the source objective weights and Actor architecture can be audited"
                )
            source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
            source_cfg = dict(source_metadata.get("config", {}))
            source_cfg.setdefault("beta_human_gripper_bc", 0.0)
            for required_name in (
                "beta_bc",
                "beta_human_bc",
                "hidden_dim",
                "projection_dim",
            ):
                if required_name not in source_cfg:
                    raise ValueError(
                        "Actor-only warm-start source metadata is missing required "
                        f"config field {required_name!r}"
                    )

        def inherited_or_default(name: str, requested: object | None) -> object:
            if requested is not None:
                return requested
            if source_cfg is not None:
                return source_cfg[name]
            return getattr(default_cfg, name)

        cfg = RealRLTConfig(
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            actor_start_step=(
                (
                    MIN_GRIPPER_CLOSE_CRITIC_BURN_IN_UPDATES
                    if args.gripper_residual_mode
                    == GRIPPER_RESIDUAL_CLOSE_ASSIST
                    else default_cfg.actor_start_step
                )
                if args.actor_start_step is None
                else args.actor_start_step
            ),
            beta_bc=float(inherited_or_default("beta_bc", args.beta_bc)),
            beta_human_bc=float(
                inherited_or_default("beta_human_bc", args.beta_human_bc)
            ),
            beta_human_gripper_bc=args.beta_human_gripper_bc,
            human_gripper_bc_scale_m=args.human_gripper_bc_scale_m,
            human_gripper_q_filter_mode=args.human_gripper_q_filter_mode,
            human_gripper_q_filter_margin=args.human_gripper_q_filter_margin,
            reference_dropout=args.reference_dropout,
            target_policy_noise_std=(
                default_cfg.target_policy_noise_std
                if args.target_policy_noise_std is None
                else args.target_policy_noise_std
            ),
            target_policy_noise_clip=(
                default_cfg.target_policy_noise_clip
                if args.target_policy_noise_clip is None
                else args.target_policy_noise_clip
            ),
            actor_residual_parameterization=args.actor_residual_parameterization,
            actor_residual_max_rad=args.actor_residual_max_rad,
            actor_residual_d1_max_rad=args.actor_residual_d1_max_rad,
            actor_residual_d2_max_rad=args.actor_residual_d2_max_rad,
            actor_direction_cone_deg=args.actor_direction_cone_deg,
            actor_execution_profile=args.actor_execution_profile,
            execution_filter_profile=args.execution_filter_profile,
            execution_filter_tau_s=args.execution_filter_tau_s,
            actor_max_boundary_jump_rad=args.actor_max_boundary_jump_rad,
            actor_direction_static_threshold_rad=args.actor_direction_static_threshold_rad,
            actor_projection_scale_steps=args.actor_projection_scale_steps,
            actor_min_projection_scale=args.actor_min_projection_scale,
            chunk_stride=(
                10
                if args.actor_execution_profile == PERSISTENT_ACTOR_EXECUTION_PROFILE
                else default_cfg.chunk_stride
            ),
            freeze_gripper_residual=args.freeze_gripper_residual,
            gripper_residual_mode=args.gripper_residual_mode,
            actor_gripper_residual_max_close_m=args.gripper_residual_max,
            actor_gripper_residual_d1_max_m=args.gripper_residual_d1_max_m,
            actor_gripper_residual_d2_max_m=args.gripper_residual_d2_max_m,
            actor_gripper_max_boundary_jump_m=args.gripper_boundary_max_m,
            gripper_command_min_m=args.gripper_command_min_m,
            gripper_command_max_m=args.gripper_command_max_m,
            gripper_release_reference_m=args.gripper_release_reference_m,
            gripper_release_delta_m=args.gripper_release_delta_m,
            hidden_dim=int(inherited_or_default("hidden_dim", args.hidden_dim)),
            projection_dim=int(
                inherited_or_default("projection_dim", args.projection_dim)
            ),
            batch_size=args.batch_size,
            seed=args.seed,
        )
        effective_gripper_max = 0.0 if cfg.freeze_gripper_residual else args.gripper_residual_max
        residual_cap = _residual_cap(
            (cfg.chunk_length, cfg.action_dim),
            joint_max=args.residual_max,
            gripper_max=effective_gripper_max,
        )
        residual_limit = estimate_residual_limit(
            replay,
            percentile=args.residual_percentile,
            minimum=args.residual_min,
            maximum=residual_cap,
            allow_zero_last_action=cfg.freeze_gripper_residual,
        )
        if args.warm_start_actor_checkpoint is not None:
            if args.actor_execution_profile != PERSISTENT_ACTOR_EXECUTION_PROFILE:
                raise ValueError(
                    "--warm-start-actor-checkpoint requires "
                    f"--actor-execution-profile={PERSISTENT_ACTOR_EXECUTION_PROFILE}"
                )
            expected_source = {
                "base_checkpoint": fingerprints["base_checkpoint"],
                "rl_token": fingerprints["rl_token"],
                "phase_classifier": fingerprints["phase_classifier"],
            }
            learner = RealRLTLearner.warm_start_actor_for_persistent_v2(
                args.warm_start_actor_checkpoint,
                replay,
                config=cfg,
                fingerprints=fingerprints,
                expected_source_fingerprints=expected_source,
                allow_objective_migration=(
                    args.allow_warm_start_objective_migration
                ),
            )
            step_zero = args.output_dir / "step_00000000"
            learner.save_checkpoint(step_zero)
            migration_report = {
                "format": "openpi_real_rlt_persistent_v2_actor_warm_start",
                "source_checkpoint": str(args.warm_start_actor_checkpoint.resolve()),
                "target_checkpoint": str(step_zero.resolve()),
                "tree_structure_equal": True,
                "leaf_shapes_equal": True,
                "actor_param_leaf_count": int(
                    learner.fingerprints["warm_start_actor_param_leaf_count"]
                ),
                "source_actor_sha256": learner.fingerprints[
                    "warm_start_actor_params_sha256"
                ],
                "target_actor_sha256": hashlib.sha256(
                    serialization.msgpack_serialize(
                        serialization.to_state_dict(learner.actor_state.params)
                    )
                ).hexdigest(),
                "target_actor_copied_from_actor": bool(
                    all(
                        np.array_equal(np.asarray(left), np.asarray(right))
                        for left, right in zip(
                            jax.tree_util.tree_leaves(learner.actor_state.params),
                            jax.tree_util.tree_leaves(learner.actor_state.target_params),
                        )
                    )
                ),
                "critic_reinitialized": True,
                "target_critic_reinitialized": True,
                "optimizer_reinitialized": True,
                "actor_optimizer_reinitialized": True,
                "critic_optimizer_reinitialized": True,
                "rng_reinitialized": True,
                "update_step": learner.update_step,
                "old_replay_loaded": False,
                "normalization": {
                    "z_rl_reused": True,
                    "state_reused": True,
                    "a_ref_reused": True,
                    "candidate_action_refit_from_new_replay": True,
                },
                "normalization_policy": learner.fingerprints["warm_start_normalization"],
                "objective_weights": {
                    "source_beta_bc": float(source_cfg["beta_bc"]),
                    "source_beta_human_bc": float(source_cfg["beta_human_bc"]),
                    "source_beta_human_gripper_bc": float(
                        source_cfg["beta_human_gripper_bc"]
                    ),
                    "beta_bc": learner.config.beta_bc,
                    "beta_human_bc": learner.config.beta_human_bc,
                    "beta_human_gripper_bc": (
                        learner.config.beta_human_gripper_bc
                    ),
                    "policy": learner.fingerprints[
                        "warm_start_objective_weights"
                    ],
                    "migration_authorized": (
                        learner.fingerprints[
                            "warm_start_objective_migration_authorized"
                        ]
                        == "true"
                    ),
                },
                "fingerprints": learner.fingerprints,
            }
            if (
                migration_report["source_actor_sha256"]
                != migration_report["target_actor_sha256"]
            ):
                raise RuntimeError("Actor parameter hash changed during exact warm-start")
            report_path = (
                args.warm_start_dry_run_report
                if args.warm_start_dry_run_report is not None
                else args.output_dir / "warm_start_migration_report.json"
            )
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(migration_report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            learner = RealRLTLearner.create(
                replay,
                config=cfg,
                residual_limit=residual_limit,
                fingerprints=fingerprints,
            )

    sampler = ReplayBatchSampler(
        replay,
        config=ReplaySamplingConfig(
            success_fraction=args.success_fraction,
            failure_fraction=args.failure_fraction,
            human_fraction=args.human_fraction,
            seed=args.seed + learner.update_step,
        ),
    )
    run_manifest = {
        "format": "openpi_real_rlt_jax_training_run",
        "replay": str(args.replay_npz.resolve()),
        "replay_split": args.split,
        "replay_transitions_total": int(len(replay_all["reward"])),
        "replay_transitions_selected": int(len(replay["reward"])),
        "config": dataclasses.asdict(cfg),
        "sampling": dataclasses.asdict(sampler.config),
        "fingerprints": fingerprints,
        "resume_checkpoint": None if args.resume is None else str(args.resume.resolve()),
        "warm_start_actor_checkpoint": (
            None
            if args.warm_start_actor_checkpoint is None
            else str(args.warm_start_actor_checkpoint.resolve())
        ),
        "allow_replay_refresh": bool(args.allow_replay_refresh),
        "previous_fingerprints": previous_fingerprints,
        "normalization_policy": (
            "frozen_from_initial_warmup"
            if args.resume
            else (
                "reuse_actor_inputs_refit_candidate_action_v1"
                if args.warm_start_actor_checkpoint is not None
                else "fit_initial_train_split"
            )
        ),
        "residual_limit_min": float(np.min(learner.residual_limit)),
        "residual_limit_max": float(np.max(learner.residual_limit)),
        "residual_limit_per_action_max": np.max(np.asarray(learner.residual_limit), axis=0).tolist(),
        "start_step": learner.update_step,
        "requested_additional_steps": args.steps,
        "actor_start_step": cfg.actor_start_step,
        "actor_updates_expected": expected_actor_updates(
            start_step=learner.update_step,
            additional_steps=args.steps,
            actor_start_step=cfg.actor_start_step,
            policy_delay=cfg.policy_delay,
        ),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics_path = args.output_dir / "metrics.jsonl"
    started = time.monotonic()
    metrics: dict[str, object] = {}
    for local_step in range(1, args.steps + 1):
        metrics = learner.update(sampler.sample(args.batch_size))
        if (
            local_step == 1
            or local_step % args.log_every == 0
            or local_step == args.steps
        ):
            record = {
                "step": learner.update_step,
                "elapsed_s": time.monotonic() - started,
                **metrics,
            }
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
        if learner.update_step % args.save_every == 0:
            learner.save_checkpoint(args.output_dir / f"step_{learner.update_step:08d}")

    final_dir = args.output_dir / f"step_{learner.update_step:08d}"
    learner.save_checkpoint(final_dir)
    (args.output_dir / "latest.txt").write_text(final_dir.name + "\n", encoding="utf-8")
    history_dir = args.output_dir / "run_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    run_manifest["final_step"] = learner.update_step
    run_manifest["checkpoint"] = str(final_dir.resolve())
    run_manifest["elapsed_s"] = time.monotonic() - started
    run_manifest["final_metrics"] = {
        key: float(value) if np.ndim(value) == 0 else np.asarray(value).tolist()
        for key, value in metrics.items()
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "format": "openpi_real_rlt_jax_training_summary_v1",
                "start_step": run_manifest["start_step"],
                "final_step": learner.update_step,
                "additional_steps": args.steps,
                "actor_start_step": cfg.actor_start_step,
                "actor_updates_expected": expected_actor_updates(
                    start_step=run_manifest["start_step"],
                    additional_steps=args.steps,
                    actor_start_step=cfg.actor_start_step,
                    policy_delay=cfg.policy_delay,
                ),
                "critic_updates": args.steps,
                "elapsed_s": run_manifest["elapsed_s"],
                "checkpoint": str(final_dir.resolve()),
                "final_metrics": run_manifest["final_metrics"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (history_dir / f"update_{run_manifest['start_step']:08d}_to_{learner.update_step:08d}.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"outcome": "complete", "checkpoint": str(final_dir), "step": learner.update_step}))


if __name__ == "__main__":
    main()
