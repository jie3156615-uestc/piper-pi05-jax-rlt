#!/usr/bin/env python3
"""One-way audited migration from frozen-gripper persistent-v2 replay.

The source arrays are never edited.  Old Actor rows are a valid zero-gripper
subset of the new close-assist contract, while Human rows retain their actual
absolute gripper actions for Critic learning and success-only gripper BC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from openpi.rlt.real.config import GRIPPER_RESIDUAL_CLOSE_ASSIST
from openpi.rlt.real.config import MIN_GRIPPER_CLOSE_CRITIC_BURN_IN_UPDATES
from openpi.rlt.real.config import PERSISTENT_ACTION_SCHEMA_FINGERPRINT
from openpi.rlt.real.config import PERSISTENT_ACTOR_EXECUTION_PROFILE
from openpi.rlt.real.config import PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT
from openpi.rlt.real.config import PERSISTENT_GRIPPER_CLOSE_GOVERNOR_PROFILE
from openpi.rlt.real.config import RealRLTConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-replay", type=Path, required=True)
    parser.add_argument("--output-replay", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _require_shape(
    arrays: dict[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    if name not in arrays:
        raise ValueError(f"source replay is missing {name!r}")
    value = np.asarray(arrays[name])
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    return value


def migrate(
    source_replay: Path,
    output_replay: Path,
    report_path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    source_replay = source_replay.expanduser().resolve()
    output_replay = output_replay.expanduser().resolve()
    report_path = report_path.expanduser().resolve()
    if not source_replay.is_file():
        raise FileNotFoundError(source_replay)
    for path in (output_replay, report_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite {path}")

    with np.load(source_replay, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    n = int(len(arrays.get("reward", ())))
    if n <= 0:
        raise ValueError("source replay is empty")
    action_shape = (n, 10, 7)
    vector_shape = (n, 7)
    schema = _require_shape(
        arrays,
        "action_schema_fingerprint",
        (n,),
    ).astype(str)
    if set(schema.tolist()) != {PERSISTENT_ACTION_SCHEMA_FINGERPRINT}:
        raise ValueError(
            "source must be the immutable frozen-gripper persistent-v2 schema"
        )
    profile = _require_shape(
        arrays,
        "actor_execution_profile",
        (n,),
    ).astype(str)
    actor_rows = profile == PERSISTENT_ACTOR_EXECUTION_PROFILE
    for name in (
        "actor_canonical_decision",
        "actor_persistent_carry_in",
        "actor_persistent_carry_out",
        "actor_persistent_previous_carry",
    ):
        value = _require_shape(arrays, name, vector_shape).astype(np.float64)
        if np.any(np.abs(value[actor_rows, 6]) > 1e-7):
            raise ValueError(f"{name} contains nonzero frozen-v2 Actor gripper evidence")
    filtered_residual = _require_shape(
        arrays,
        "filtered_actual_residual",
        action_shape,
    ).astype(np.float64)
    if np.any(np.abs(filtered_residual[actor_rows, :, 6]) > 2e-6):
        raise ValueError(
            "frozen-v2 Actor rows contain nonzero executed gripper residual"
        )
    a_base = _require_shape(arrays, "a_base_filtered", action_shape).astype(
        np.float64
    )
    a_exec = _require_shape(arrays, "a_exec", action_shape).astype(np.float64)
    a_human = _require_shape(arrays, "a_human", action_shape).astype(np.float64)
    human_mask = _require_shape(arrays, "human_mask", (n, 10)).astype(bool)
    if not np.all(np.isfinite(a_base)) or not np.all(np.isfinite(a_exec)):
        raise ValueError("gripper migration source contains NaN or inf")
    if np.any(a_exec[..., 6] < -1e-6) or np.any(a_exec[..., 6] > 0.080001):
        raise ValueError("executed gripper actions leave the Piper [0, 0.08] m range")

    episode_ids = _require_shape(arrays, "episode_id", (n,)).astype(str)
    rewards = _require_shape(arrays, "reward", (n,)).astype(np.float32)
    successful_episodes = {
        episode_id
        for episode_id in np.unique(episode_ids)
        if np.any(rewards[episode_ids == episode_id] > 0.0)
    }
    success_mask = np.asarray(
        [episode_id in successful_episodes for episode_id in episode_ids],
        dtype=np.bool_,
    )
    cfg = RealRLTConfig(
        actor_execution_profile=PERSISTENT_ACTOR_EXECUTION_PROFILE,
        chunk_stride=10,
        freeze_gripper_residual=False,
        gripper_residual_mode=GRIPPER_RESIDUAL_CLOSE_ASSIST,
        beta_human_gripper_bc=1.0,
        actor_start_step=MIN_GRIPPER_CLOSE_CRITIC_BURN_IN_UPDATES,
    )
    migrated = dict(arrays)
    migrated["action_schema_fingerprint"] = np.asarray(
        [PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT] * n
    )
    migrated["gripper_residual_mode"] = np.asarray(
        [GRIPPER_RESIDUAL_CLOSE_ASSIST] * n
    )
    migrated["actor_governor_fingerprint"] = np.asarray(
        [PERSISTENT_GRIPPER_CLOSE_GOVERNOR_PROFILE] * n
    )
    migrated["success_mask"] = success_mask
    audit = {
        "execution_gripper_residual_max_close_m": cfg.actor_gripper_residual_max_close_m,
        "execution_gripper_d1_max_m": cfg.actor_gripper_residual_d1_max_m,
        "execution_gripper_d2_max_m": cfg.actor_gripper_residual_d2_max_m,
        "execution_gripper_boundary_limit_m": cfg.actor_gripper_max_boundary_jump_m,
        "execution_gripper_command_min_m": cfg.gripper_command_min_m,
        "execution_gripper_command_max_m": cfg.gripper_command_max_m,
        "execution_gripper_release_reference_m": cfg.gripper_release_reference_m,
        "execution_gripper_release_delta_m": cfg.gripper_release_delta_m,
    }
    for name, value in audit.items():
        migrated[name] = np.full((n, 10), value, dtype=np.float32)

    human_delta = a_human[..., 6] - a_base[..., 6]
    reward1_human = human_mask & success_mask[:, None]
    reward0_human = human_mask & ~success_mask[:, None]

    output_replay.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_replay.with_name(output_replay.name + ".tmp.npz")
    np.savez_compressed(temporary, **migrated)
    os.replace(temporary, output_replay)
    report: dict[str, object] = {
        "format": "openpi_piper_gripper_close_replay_migration_v1",
        "source_replay": str(source_replay),
        "source_sha256": _sha256(source_replay),
        "output_replay": str(output_replay),
        "output_sha256": _sha256(output_replay),
        "source_schema": PERSISTENT_ACTION_SCHEMA_FINGERPRINT,
        "target_schema": PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT,
        "target_governor": PERSISTENT_GRIPPER_CLOSE_GOVERNOR_PROFILE,
        "gripper_residual_mode": GRIPPER_RESIDUAL_CLOSE_ASSIST,
        "transitions": n,
        "episodes": int(len(np.unique(episode_ids))),
        "successful_episodes": int(len(successful_episodes)),
        "actor_rows_verified_zero_gripper": int(np.count_nonzero(actor_rows)),
        "human_steps": int(np.count_nonzero(human_mask)),
        "reward1_human_steps": int(np.count_nonzero(reward1_human)),
        "reward0_human_steps": int(np.count_nonzero(reward0_human)),
        "reward1_human_gripper_delta_median": (
            float(np.median(human_delta[reward1_human]))
            if np.any(reward1_human)
            else None
        ),
        "reward0_human_gripper_delta_median": (
            float(np.median(human_delta[reward0_human]))
            if np.any(reward0_human)
            else None
        ),
        "teacher_policy": (
            "all_admitted_human_dim6_clip_delta_to_[-0.005,0]_"
            "critic_min_advantage_q_filter_reward_independent"
        ),
        "source_mutated": False,
        "new_hardware_data_required": False,
        "contract": audit,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_name(report_path.name + ".tmp")
    temporary_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_report, report_path)
    return report


def main() -> None:
    args = _parse_args()
    report = migrate(
        args.source_replay,
        args.output_replay,
        args.report,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
