#!/usr/bin/env python3
"""Create an isolated close-assist v3 lineage from an audited v2 attempt.

This tool deliberately separates three inputs:

* the immutable source Actor is the accepted step-12627 Actor;
* the rejected frozen-gripper step-144 checkpoint is provenance only;
* its audited 30-episode replay may be migrated to v5 and used as bootstrap.

No source artifact is modified.  Without ``--create`` this command is a
read-only preflight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import stat
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from persistent_v2_contract import (
    ACTION_SCHEMA_FINGERPRINT,
    ACTOR_DIRECTION_STATIC_THRESHOLD_RAD,
    ACTOR_EXECUTION_PROFILE,
    ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD,
    ACTOR_MIN_PROJECTION_SCALE,
    ACTOR_MODEL_ACTION_SCHEMA_FINGERPRINT,
    ACTOR_PROJECTION_PROFILE,
    ACTOR_PROJECTION_SCALE_STEPS,
    CHUNK_LENGTH,
    CHUNK_STRIDE,
    CONTROL_DT_S,
    CONTROL_HZ,
    EXECUTION_FILTER_ALPHA,
    EXECUTION_FILTER_PROFILE,
    EXECUTION_FILTER_TAU_S,
    GRIPPER_COMMAND_MAX_M,
    GRIPPER_COMMAND_MIN_M,
    GRIPPER_MAX_BOUNDARY_JUMP_M,
    GRIPPER_RELEASE_DELTA_M,
    GRIPPER_RELEASE_REFERENCE_M,
    GRIPPER_RESIDUAL_D1_MAX_M,
    GRIPPER_RESIDUAL_D2_MAX_M,
    GRIPPER_RESIDUAL_MAX_CLOSE_M,
    GRIPPER_RESIDUAL_MODE,
    PERSISTENT_GOVERNOR_FINGERPRINT,
)

FORK_FORMAT = "openpi_piper_gripper_close_assist_v3_lineage_fork"
VALIDATION_FORMAT = "openpi_piper_gripper_close_assist_v3_lineage_validation"
STATE_FORMAT = "openpi_piper_online_rlt_state_persistent_gripper_v3"
LINEAGE_MODE = "persistent_gripper_v3_bootstrap_warm_start"
REPLAY_POLICY = "immutable_migrated_v5_warmup_plus_persistent_v5_online"
OBJECTIVE_MIGRATION_FORMAT = "openpi_piper_gripper_v3_migration_v1"
OBJECTIVE_AUTHORIZATION = (
    "user_requested_admitted_human_q_filtered_gripper_imitation_20260727"
)
BOOTSTRAP_MIGRATION_FORMAT = (
    "openpi_piper_gripper_close_replay_migration_v1"
)
BOOTSTRAP_TEACHER_POLICY = (
    "all_admitted_human_dim6_clip_delta_to_[-0.005,0]_"
    "critic_min_advantage_q_filter_reward_independent"
)
HUMAN_GRIPPER_Q_FILTER_MODE = "critic_min_advantage_v1"
HUMAN_GRIPPER_Q_FILTER_MARGIN = 0.0
SOURCE_ACTOR_SCHEMA = (
    "piper_joint_delta_v3_c10_n10_stride2_behavior_ref50_"
    "rank1_bump_r005_d1_0015_d2_001_cone15_"
    "gripper_absolute_frozen_residual"
)
SOURCE_FROZEN_EXECUTION_SCHEMA = (
    "piper_joint_delta_v4_c10_n10_stride10_behavior_ref50_"
    "persistent_filtered_actual_r005_d1_0015_d2_001_cone15_"
    "gripper_absolute_frozen_residual"
)
EXPECTED_SOURCE_ACTOR_STEP = 12627
EXPECTED_REJECTED_STEP = 144
EXPECTED_BOOTSTRAP_EPISODES = 30
EXPECTED_BOOTSTRAP_REWARD_POSITIVE_EPISODES = 24
EXPECTED_BOOTSTRAP_REWARD_NEGATIVE_EPISODES = 6
EXPECTED_BOOTSTRAP_HUMAN_EPISODES = 29
EXPECTED_INITIAL_CRITIC_BURN_IN_STEPS = 144
EXPECTED_INITIAL_CHECKPOINT_STEP = 288
EXPECTED_SOURCE_BETAS = {
    "beta_bc": 40.0,
    "beta_human_bc": 0.0,
    "beta_human_gripper_bc": 0.0,
}
EXPECTED_TARGET_BETAS = {
    "beta_bc": 20.0,
    "beta_human_bc": 0.0,
    "beta_human_gripper_bc": 1.0,
}
EPISODE_PATTERN = re.compile(r"episode_([0-9]+)")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-session-root", type=Path, required=True)
    parser.add_argument("--source-state-root", type=Path, required=True)
    parser.add_argument(
        "--source-actor-checkpoint",
        type=Path,
        help=(
            "Optional exact-path guard.  The checkpoint is otherwise read from "
            "source online_state.json and must still be step 12627."
        ),
    )
    parser.add_argument("--bootstrap-replay", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-migration-report", type=Path, required=True
    )
    parser.add_argument("--initial-v3-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--initial-v3-validation-report",
        type=Path,
        required=True,
        help=(
            "Acceptance report from validate_real_rlt_actor_jax.py; it must "
            "bind the completed checkpoint to the migrated warm-30 replay and "
            "contain passed=true."
        ),
    )
    parser.add_argument("--target-session-root", type=Path, required=True)
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Self-contained gripper-v3 OpenPI workspace written into config.env.",
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        required=True,
        help="Self-contained gripper-v3 Piper runtime written into config.env.",
    )
    parser.add_argument(
        "--target-state-dir",
        default=".online_rlt_persistent_gripper_v3",
    )
    parser.add_argument("--expected-source-latest-episode")
    parser.add_argument(
        "--expected-target-episode-floor",
        type=int,
        help=(
            "Optional guard for the first target episode. The floor is one "
            "greater than the highest source episode directory, including "
            "operator-excluded directories."
        ),
    )
    parser.add_argument(
        "--expected-initial-v3-checkpoint-step",
        type=int,
        help="Optional stale-artifact guard for the completed v3 checkpoint.",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create the target lineage.  Omit for read-only preflight.",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"required JSON is missing: {path}") from None
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _checkpoint_manifest(
    checkpoint: Path,
) -> tuple[list[dict[str, Any]], str]:
    files: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for path in sorted(item for item in checkpoint.rglob("*") if item.is_file()):
        relative = path.relative_to(checkpoint).as_posix()
        file_sha = _sha256(path)
        size = path.stat().st_size
        files.append(
            {"path": relative, "sha256": file_sha, "size_bytes": size}
        )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\n")
    if not files:
        raise ValueError(f"checkpoint contains no files: {checkpoint}")
    required = {"learner.msgpack", "metadata.json"}
    present = {item["path"] for item in files}
    missing = sorted(required.difference(present))
    if missing:
        raise ValueError(
            f"checkpoint is incomplete ({', '.join(missing)}): {checkpoint}"
        )
    return files, digest.hexdigest()


def _manifest_file_sha(
    files: Iterable[dict[str, Any]], relative: str
) -> str:
    for item in files:
        if item.get("path") == relative:
            return str(item["sha256"])
    raise ValueError(f"checkpoint manifest lacks {relative!r}")


def _ensure_inside(path: Path, parent: Path, label: str) -> None:
    if path != parent and parent not in path.parents:
        raise ValueError(f"{label} must be inside {parent}: {path}")


def _canonical_episode_ids(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty JSON list")
    result: list[tuple[int, str]] = []
    for raw in value:
        if not isinstance(raw, str):
            raise ValueError(f"{label} contains a non-string episode ID")
        match = EPISODE_PATTERN.fullmatch(raw)
        if not match:
            raise ValueError(f"{label} contains invalid episode ID {raw!r}")
        index = int(match.group(1))
        canonical = f"episode_{index:06d}"
        if raw != canonical:
            raise ValueError(
                f"{label} contains non-canonical episode ID {raw!r}"
            )
        result.append((index, canonical))
    if len({item[1] for item in result}) != len(result):
        raise ValueError(f"{label} contains duplicate episode IDs")
    return [item[1] for item in sorted(result)]


def _episode_directories(session: Path) -> list[tuple[int, Path]]:
    result: list[tuple[int, Path]] = []
    for path in session.glob("episode_[0-9]*"):
        match = EPISODE_PATTERN.fullmatch(path.name)
        if path.is_dir() and match:
            result.append((int(match.group(1)), path))
    return sorted(result)


def _episode_report(path: Path) -> dict[str, Any]:
    report = _load_json(path / "report.json")
    try:
        reward = float(report["terminal_reward"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{path.name}: report lacks numeric terminal_reward"
        ) from exc
    if not math.isfinite(reward) or reward not in (0.0, 1.0):
        raise ValueError(f"{path.name}: terminal_reward must be exactly 0 or 1")
    if report.get("exclude_from_training") is True:
        raise ValueError(f"{path.name}: episode is excluded from training")
    outcome = report.get("outcome")
    if outcome not in (None, "episode_done"):
        raise ValueError(f"{path.name}: incomplete outcome {outcome!r}")
    try:
        human_commands = int(report["human_passthrough_commands"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{path.name}: report lacks human_passthrough_commands"
        ) from exc
    if human_commands < 0:
        raise ValueError(
            f"{path.name}: human_passthrough_commands is negative"
        )
    return {
        "reward": reward,
        "human_commands": human_commands,
        "report_sha256": _sha256(path / "report.json"),
    }


def _latest_complete_episode(
    source_session: Path,
) -> tuple[int, str, list[str]]:
    directories = _episode_directories(source_session)
    if not directories:
        raise ValueError("source session has no episode directories")
    completed: list[tuple[int, str]] = []
    incomplete: list[str] = []
    for index, path in directories:
        try:
            _episode_report(path)
        except (FileNotFoundError, ValueError):
            incomplete.append(path.name)
            continue
        completed.append((index, path.name))
    if not completed:
        raise ValueError("source session has no completed episode report")
    latest_index, latest_name = completed[-1]
    return latest_index, latest_name, incomplete


def _audit_attempt_quality(
    source_session: Path, episode_ids: list[str]
) -> dict[str, Any]:
    reward_positive = 0
    reward_negative = 0
    human = 0
    episode_reports: list[dict[str, Any]] = []
    for episode_id in episode_ids:
        audit = _episode_report(source_session / episode_id)
        reward_positive += int(audit["reward"] == 1.0)
        reward_negative += int(audit["reward"] == 0.0)
        human += int(audit["human_commands"] > 0)
        episode_reports.append(
            {
                "episode_id": episode_id,
                "reward": audit["reward"],
                "human_passthrough_commands": audit["human_commands"],
                "report_sha256": audit["report_sha256"],
            }
        )
    actual = (len(episode_ids), reward_positive, reward_negative, human)
    expected = (
        EXPECTED_BOOTSTRAP_EPISODES,
        EXPECTED_BOOTSTRAP_REWARD_POSITIVE_EPISODES,
        EXPECTED_BOOTSTRAP_REWARD_NEGATIVE_EPISODES,
        EXPECTED_BOOTSTRAP_HUMAN_EPISODES,
    )
    if actual != expected:
        raise ValueError(
            "source last attempt is not the required admitted "
            "30/R+24/R-6/H29 set: "
            f"actual={actual}, expected={expected}"
        )
    return {
        "episodes": len(episode_ids),
        "reward_positive_episodes": reward_positive,
        "reward_negative_episodes": reward_negative,
        "admitted_human_episodes": human,
        "episode_reports": episode_reports,
    }


def _require_float(
    mapping: dict[str, Any],
    key: str,
    expected: float | None = None,
    *,
    label: str,
) -> float:
    try:
        value = float(mapping[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} lacks valid {key}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} {key} is non-finite")
    if expected is not None and not math.isclose(
        value, expected, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(
            f"{label} {key} mismatch: {value!r} != {expected!r}"
        )
    return value


def _checkpoint_audit(
    checkpoint: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    files, tree_sha = _checkpoint_manifest(checkpoint)
    metadata = _load_json(checkpoint / "metadata.json")
    if metadata.get("format") != "openpi_real_rlt_jax_learner":
        raise ValueError(
            f"unsupported learner checkpoint format: {checkpoint}"
        )
    try:
        step = int(metadata["update_step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint lacks update_step: {checkpoint}") from exc
    if step < 0:
        raise ValueError(f"checkpoint update_step is negative: {checkpoint}")
    return metadata, files, tree_sha


def _audit_source_actor(
    checkpoint: Path,
) -> dict[str, Any]:
    metadata, files, tree_sha = _checkpoint_audit(checkpoint)
    if int(metadata["update_step"]) != EXPECTED_SOURCE_ACTOR_STEP:
        raise ValueError(
            "source Actor must be the accepted step 12627 checkpoint"
        )
    fingerprints = metadata.get("fingerprints")
    config = metadata.get("config")
    if not isinstance(fingerprints, dict) or not isinstance(config, dict):
        raise ValueError("source Actor metadata lacks config/fingerprints")
    if fingerprints.get("action_schema") != SOURCE_ACTOR_SCHEMA:
        raise ValueError("source Actor is not the fixed frozen-gripper v3 Actor")
    if config.get("freeze_gripper_residual") is not True:
        raise ValueError("source Actor gripper residual was not frozen")
    try:
        chunk_stride = int(config["chunk_stride"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("source Actor lacks valid chunk_stride") from exc
    if chunk_stride != 2:
        raise ValueError("source Actor chunk_stride is not the expected 2")
    source_betas = {
        "beta_bc": _require_float(
            config, "beta_bc", label="source Actor config"
        ),
        "beta_human_bc": _require_float(
            config, "beta_human_bc", label="source Actor config"
        ),
        "beta_human_gripper_bc": float(
            config.get("beta_human_gripper_bc", 0.0)
        ),
    }
    if not math.isclose(
        source_betas["beta_human_gripper_bc"],
        0.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("source Actor unexpectedly trained a gripper residual")
    for key, expected in EXPECTED_SOURCE_BETAS.items():
        if not math.isclose(
            source_betas[key], expected, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                f"source Actor objective {key} mismatch: "
                f"{source_betas[key]} != {expected}"
            )
    return {
        "checkpoint": str(checkpoint),
        "step": EXPECTED_SOURCE_ACTOR_STEP,
        "tree_sha256": tree_sha,
        "files": files,
        "learner_msgpack_sha256": _manifest_file_sha(
            files, "learner.msgpack"
        ),
        "actor_schema": SOURCE_ACTOR_SCHEMA,
        "objective_weights": source_betas,
    }


def _audit_rejected_checkpoint(
    checkpoint: Path,
    *,
    attempt_replay_sha256: str,
) -> dict[str, Any]:
    metadata, files, tree_sha = _checkpoint_audit(checkpoint)
    if int(metadata["update_step"]) != EXPECTED_REJECTED_STEP:
        raise ValueError(
            "source latest_rejected_checkpoint is not frozen-v2 step 144"
        )
    fingerprints = metadata.get("fingerprints")
    if not isinstance(fingerprints, dict):
        raise ValueError("rejected checkpoint lacks fingerprints")
    if fingerprints.get("action_schema") != SOURCE_FROZEN_EXECUTION_SCHEMA:
        raise ValueError(
            "source rejected checkpoint is not the frozen-gripper v2 schema"
        )
    if fingerprints.get("replay_sha256") != attempt_replay_sha256:
        raise ValueError(
            "rejected checkpoint was not trained from latest_rejected_replay"
        )
    return {
        "checkpoint": str(checkpoint),
        "step": EXPECTED_REJECTED_STEP,
        "tree_sha256": tree_sha,
        "files": files,
        "learner_msgpack_sha256": _manifest_file_sha(
            files, "learner.msgpack"
        ),
        "action_schema": SOURCE_FROZEN_EXECUTION_SCHEMA,
        "disposition": "rejected_never_warm_start_never_deploy",
    }


def _replay_audit(
    replay_path: Path,
    *,
    expected_episode_ids: list[str],
) -> dict[str, Any]:
    with np.load(replay_path, allow_pickle=False) as replay:
        required = {
            "episode_id",
            "episode_split",
            "action_schema_fingerprint",
            "gripper_residual_mode",
            "actor_governor_fingerprint",
            "success_mask",
            "human_mask",
            "reward",
        }
        missing = sorted(required.difference(replay.files))
        if missing:
            raise ValueError(
                f"bootstrap replay lacks arrays: {', '.join(missing)}"
            )
        episode_id = np.asarray(replay["episode_id"]).astype(str)
        if episode_id.ndim != 1 or len(episode_id) == 0:
            raise ValueError("bootstrap replay episode_id is empty or malformed")
        replay_ids = sorted(set(episode_id.tolist()))
        if replay_ids != expected_episode_ids:
            raise ValueError(
                "bootstrap replay episode IDs do not equal source "
                "last_attempt_episode_ids"
            )
        n = len(episode_id)
        episode_split = np.asarray(replay["episode_split"]).astype(str)
        schema = np.asarray(replay["action_schema_fingerprint"]).astype(str)
        mode = np.asarray(replay["gripper_residual_mode"]).astype(str)
        governor = np.asarray(
            replay["actor_governor_fingerprint"]
        ).astype(str)
        success_mask = np.asarray(replay["success_mask"]).astype(bool)
        human_mask = np.asarray(replay["human_mask"]).astype(bool)
        reward = np.asarray(replay["reward"])
        if episode_split.shape != (n,):
            raise ValueError("bootstrap replay episode_split shape mismatch")
        if schema.shape != (n,) or set(schema.tolist()) != {
            ACTION_SCHEMA_FINGERPRINT
        }:
            raise ValueError("bootstrap replay is not exclusively v5 schema")
        if mode.shape != (n,) or set(mode.tolist()) != {
            GRIPPER_RESIDUAL_MODE
        }:
            raise ValueError("bootstrap replay gripper mode mismatch")
        if governor.shape != (n,) or set(governor.tolist()) != {
            PERSISTENT_GOVERNOR_FINGERPRINT
        }:
            raise ValueError("bootstrap replay governor mismatch")
        if success_mask.shape != (n,):
            raise ValueError("bootstrap replay success_mask shape mismatch")
        if human_mask.shape != (n, CHUNK_LENGTH):
            raise ValueError("bootstrap replay human_mask shape mismatch")
        if reward.shape[0] != n:
            raise ValueError("bootstrap replay reward shape mismatch")
        unknown_splits = sorted(
            set(episode_split.tolist()).difference(
                {"train", "validation", "test"}
            )
        )
        if unknown_splits:
            raise ValueError(
                "bootstrap replay contains unknown episode_split labels: "
                f"{unknown_splits}"
            )
        leaking_episode_ids = sorted(
            episode
            for episode in replay_ids
            if len(set(episode_split[episode_id == episode].tolist())) != 1
        )
        if leaking_episode_ids:
            raise ValueError(
                "bootstrap replay episodes span multiple splits: "
                f"{leaking_episode_ids}"
            )
        train_count = int(np.count_nonzero(episode_split == "train"))
        validation_count = int(
            np.count_nonzero(episode_split == "validation")
        )
        if train_count <= 0:
            raise ValueError("bootstrap replay has no train split transitions")
        if validation_count <= 0:
            raise ValueError(
                "bootstrap replay has no validation split transitions"
            )
        reward_positive_ids = sorted(set(episode_id[success_mask].tolist()))
        reward_negative_ids = sorted(
            set(replay_ids).difference(reward_positive_ids)
        )
        human_rows = np.any(human_mask, axis=1)
        human_ids = sorted(set(episode_id[human_rows].tolist()))
        reward_positive_human_steps = int(
            np.count_nonzero(human_mask & success_mask[:, None])
        )
        reward_negative_human_steps = int(
            np.count_nonzero(human_mask & ~success_mask[:, None])
        )
        quality = (
            len(replay_ids),
            len(reward_positive_ids),
            len(reward_negative_ids),
            len(human_ids),
        )
        expected_quality = (
            EXPECTED_BOOTSTRAP_EPISODES,
            EXPECTED_BOOTSTRAP_REWARD_POSITIVE_EPISODES,
            EXPECTED_BOOTSTRAP_REWARD_NEGATIVE_EPISODES,
            EXPECTED_BOOTSTRAP_HUMAN_EPISODES,
        )
        if quality != expected_quality:
            raise ValueError(
                "bootstrap replay quality mismatch: "
                f"{quality} != {expected_quality}"
            )
        if not np.all(np.isfinite(reward)):
            raise ValueError("bootstrap replay contains non-finite rewards")
        if reward_positive_human_steps <= 0 or reward_negative_human_steps <= 0:
            raise ValueError(
                "bootstrap replay must preserve admitted-human steps carrying "
                "both reward-positive and reward-negative episode labels"
            )
    return {
        "path": str(replay_path),
        "sha256": _sha256(replay_path),
        "transitions": n,
        "train_transitions": train_count,
        "validation_transitions": validation_count,
        "episode_ids": replay_ids,
        "reward_positive_episode_ids": reward_positive_ids,
        "reward_negative_episode_ids": reward_negative_ids,
        "admitted_human_episode_ids": human_ids,
        "admitted_human_steps": (
            reward_positive_human_steps + reward_negative_human_steps
        ),
        "reward_positive_human_steps": reward_positive_human_steps,
        "reward_negative_human_steps": reward_negative_human_steps,
        "quality": {
            "episodes": len(replay_ids),
            "reward_positive_episodes": len(reward_positive_ids),
            "reward_negative_episodes": len(reward_negative_ids),
            "admitted_human_episodes": len(human_ids),
        },
    }


def _audit_bootstrap_migration(
    report_path: Path,
    *,
    source_replay: Path,
    source_replay_sha256: str,
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    report = _load_json(report_path)
    exact = {
        "format": BOOTSTRAP_MIGRATION_FORMAT,
        "source_schema": SOURCE_FROZEN_EXECUTION_SCHEMA,
        "target_schema": ACTION_SCHEMA_FINGERPRINT,
        "target_governor": PERSISTENT_GOVERNOR_FINGERPRINT,
        "gripper_residual_mode": GRIPPER_RESIDUAL_MODE,
        "teacher_policy": BOOTSTRAP_TEACHER_POLICY,
        "source_mutated": False,
    }
    for key, expected in exact.items():
        if report.get(key) != expected:
            raise ValueError(
                f"bootstrap migration report {key} mismatch: "
                f"{report.get(key)!r} != {expected!r}"
            )
    if Path(str(report.get("source_replay", ""))).expanduser().resolve() != (
        source_replay
    ):
        raise ValueError(
            "bootstrap migration report is bound to another source replay"
        )
    if report.get("source_sha256") != source_replay_sha256:
        raise ValueError("bootstrap migration source replay SHA mismatch")
    if Path(str(report.get("output_replay", ""))).expanduser().resolve() != (
        Path(bootstrap["path"]).resolve()
    ):
        raise ValueError(
            "bootstrap migration report is bound to another output replay"
        )
    if report.get("output_sha256") != bootstrap["sha256"]:
        raise ValueError("bootstrap migration output replay SHA mismatch")
    integer_expectations = {
        "transitions": bootstrap["transitions"],
        "episodes": EXPECTED_BOOTSTRAP_EPISODES,
        # The migration-v1 wire field is retained for compatibility; its
        # canonical v3 meaning is terminal reward-positive episodes.
        "successful_episodes": EXPECTED_BOOTSTRAP_REWARD_POSITIVE_EPISODES,
        "reward1_human_steps": bootstrap["reward_positive_human_steps"],
        "reward0_human_steps": bootstrap["reward_negative_human_steps"],
    }
    for key, expected in integer_expectations.items():
        try:
            actual = int(report[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"bootstrap migration report lacks valid {key}"
            ) from exc
        if actual != expected:
            raise ValueError(
                f"bootstrap migration report {key} mismatch: "
                f"{actual} != {expected}"
            )
    contract = report.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("bootstrap migration report lacks contract")
    expected_contract = {
        "execution_gripper_residual_max_close_m": (
            GRIPPER_RESIDUAL_MAX_CLOSE_M
        ),
        "execution_gripper_d1_max_m": GRIPPER_RESIDUAL_D1_MAX_M,
        "execution_gripper_d2_max_m": GRIPPER_RESIDUAL_D2_MAX_M,
        "execution_gripper_boundary_limit_m": (
            GRIPPER_MAX_BOUNDARY_JUMP_M
        ),
        "execution_gripper_command_min_m": GRIPPER_COMMAND_MIN_M,
        "execution_gripper_command_max_m": GRIPPER_COMMAND_MAX_M,
        "execution_gripper_release_reference_m": (
            GRIPPER_RELEASE_REFERENCE_M
        ),
        "execution_gripper_release_delta_m": GRIPPER_RELEASE_DELTA_M,
    }
    for key, expected in expected_contract.items():
        _require_float(
            contract,
            key,
            expected,
            label="bootstrap migration contract",
        )
    return {
        "path": str(report_path),
        "sha256": _sha256(report_path),
        "payload": report,
    }


def _audit_initial_v3_checkpoint(
    checkpoint: Path,
    *,
    bootstrap_replay_sha256: str,
    source_actor: dict[str, Any],
    rejected: dict[str, Any],
    expected_step: int | None,
) -> dict[str, Any]:
    metadata, files, tree_sha = _checkpoint_audit(checkpoint)
    step = int(metadata["update_step"])
    if step <= 0:
        raise ValueError("initial v3 checkpoint has not completed any updates")
    if expected_step is not None and step != expected_step:
        raise ValueError(
            f"initial v3 checkpoint stale guard: {step} != {expected_step}"
        )
    if step != EXPECTED_INITIAL_CHECKPOINT_STEP:
        raise ValueError(
            "initial v3 checkpoint must include the audited Critic burn-in: "
            f"{step} != {EXPECTED_INITIAL_CHECKPOINT_STEP}"
        )
    learner_sha = _manifest_file_sha(files, "learner.msgpack")
    if (
        tree_sha == rejected["tree_sha256"]
        or learner_sha == rejected["learner_msgpack_sha256"]
    ):
        raise ValueError(
            "refusing to use rejected frozen-v2 step 144 as initial v3"
        )
    fingerprints = metadata.get("fingerprints")
    config = metadata.get("config")
    if not isinstance(fingerprints, dict) or not isinstance(config, dict):
        raise ValueError("initial v3 checkpoint lacks config/fingerprints")
    expected_fingerprints = {
        "replay_sha256": bootstrap_replay_sha256,
        "action_schema": ACTION_SCHEMA_FINGERPRINT,
        "actor_execution_profile": ACTOR_EXECUTION_PROFILE,
        "execution_filter_profile": EXECUTION_FILTER_PROFILE,
        "actor_governor": PERSISTENT_GOVERNOR_FINGERPRINT,
        "warm_start_actor_source_schema": SOURCE_ACTOR_SCHEMA,
        "warm_start_source_checkpoint_sha256": source_actor[
            "learner_msgpack_sha256"
        ],
    }
    for key, expected in expected_fingerprints.items():
        if fingerprints.get(key) != expected:
            raise ValueError(
                f"initial v3 checkpoint fingerprint {key} mismatch: "
                f"{fingerprints.get(key)!r} != {expected!r}"
            )
    expected_config: dict[str, Any] = {
        "freeze_gripper_residual": False,
        "gripper_residual_mode": GRIPPER_RESIDUAL_MODE,
        "chunk_stride": CHUNK_STRIDE,
        "actor_gripper_residual_max_close_m": (
            GRIPPER_RESIDUAL_MAX_CLOSE_M
        ),
        "actor_gripper_residual_d1_max_m": GRIPPER_RESIDUAL_D1_MAX_M,
        "actor_gripper_residual_d2_max_m": GRIPPER_RESIDUAL_D2_MAX_M,
        "actor_gripper_max_boundary_jump_m": (
            GRIPPER_MAX_BOUNDARY_JUMP_M
        ),
        "human_gripper_q_filter_mode": HUMAN_GRIPPER_Q_FILTER_MODE,
        "human_gripper_q_filter_margin": HUMAN_GRIPPER_Q_FILTER_MARGIN,
        "actor_start_step": EXPECTED_INITIAL_CRITIC_BURN_IN_STEPS,
    }
    for key, expected in expected_config.items():
        actual = config.get(key)
        if isinstance(expected, float):
            _require_float(
                config, key, expected, label="initial v3 checkpoint config"
            )
        elif actual != expected:
            raise ValueError(
                f"initial v3 checkpoint config {key} mismatch: "
                f"{actual!r} != {expected!r}"
            )
    target_betas = {
        "beta_bc": _require_float(
            config, "beta_bc", label="initial v3 checkpoint config"
        ),
        "beta_human_bc": _require_float(
            config, "beta_human_bc", label="initial v3 checkpoint config"
        ),
        "beta_human_gripper_bc": _require_float(
            config,
            "beta_human_gripper_bc",
            label="initial v3 checkpoint config",
        ),
    }
    if target_betas["beta_human_gripper_bc"] <= 0.0:
        raise ValueError(
            "initial v3 checkpoint does not enable Q-filtered admitted-human "
            "gripper BC"
        )
    for key, expected in EXPECTED_TARGET_BETAS.items():
        if not math.isclose(
            target_betas[key], expected, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                f"initial v3 objective {key} mismatch: "
                f"{target_betas[key]} != {expected}"
            )
    source_betas = source_actor["objective_weights"]
    fingerprint_betas = {
        "warm_start_source_beta_bc": source_betas["beta_bc"],
        "warm_start_source_beta_human_bc": source_betas["beta_human_bc"],
        "warm_start_source_beta_human_gripper_bc": source_betas[
            "beta_human_gripper_bc"
        ],
        "warm_start_target_beta_bc": target_betas["beta_bc"],
        "warm_start_target_beta_human_bc": target_betas["beta_human_bc"],
        "warm_start_target_beta_human_gripper_bc": target_betas[
            "beta_human_gripper_bc"
        ],
    }
    for key, expected in fingerprint_betas.items():
        try:
            actual = float(fingerprints[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"initial v3 checkpoint lacks valid fingerprint {key}"
            ) from exc
        if not math.isclose(
            actual, expected, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                f"initial v3 checkpoint {key} mismatch: "
                f"{actual} != {expected}"
            )
    if fingerprints.get(
        "warm_start_objective_migration_authorized"
    ) != "true":
        raise ValueError(
            "initial v3 checkpoint lacks explicit objective migration authorization"
        )
    return {
        "checkpoint": str(checkpoint),
        "step": step,
        "tree_sha256": tree_sha,
        "files": files,
        "learner_msgpack_sha256": learner_sha,
        "objective_weights": target_betas,
        "fingerprints": {
            key: fingerprints.get(key)
            for key in (
                "base_checkpoint",
                "rl_token",
                "phase_classifier",
                *expected_fingerprints.keys(),
                "warm_start_actor_params_sha256",
                "warm_start_actor_param_shapes_sha256",
                "warm_start_normalization",
                "warm_start_objective_weights",
                "warm_start_objective_migration_authorized",
            )
        },
    }


def _audit_initial_v3_validation(
    report_path: Path,
    *,
    checkpoint: Path,
    checkpoint_step: int,
    bootstrap_replay: Path,
    bootstrap_sha256: str,
) -> dict[str, Any]:
    report = _load_json(report_path)
    if report.get("format") != "openpi_real_rlt_actor_acceptance":
        raise ValueError("unsupported initial v3 validation report format")
    if report.get("passed") is not True:
        raise ValueError("initial v3 checkpoint did not pass Actor acceptance")
    if Path(str(report.get("checkpoint", ""))).expanduser().resolve() != (
        checkpoint
    ):
        raise ValueError(
            "initial v3 validation report is bound to another checkpoint"
        )
    if Path(str(report.get("replay", ""))).expanduser().resolve() != (
        bootstrap_replay
    ):
        raise ValueError(
            "initial v3 validation report is bound to another replay"
        )
    if report.get("replay_sha256") != bootstrap_sha256:
        raise ValueError("initial v3 validation replay SHA mismatch")
    try:
        report_step = int(report["update_step"])
        samples = int(report["samples"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "initial v3 validation lacks update_step/samples"
        ) from exc
    if report_step != checkpoint_step or report_step <= 0:
        raise ValueError(
            "initial v3 validation/checkpoint step mismatch"
        )
    if samples <= 0:
        raise ValueError("initial v3 validation used no held-out samples")
    if report.get("replay_split") != "validation":
        raise ValueError(
            "initial v3 acceptance must use the validation replay split"
        )
    report_fingerprints = report.get("fingerprints")
    if not isinstance(report_fingerprints, dict):
        raise ValueError("initial v3 validation lacks checkpoint fingerprints")
    expected = {
        "replay_sha256": bootstrap_sha256,
        "action_schema": ACTION_SCHEMA_FINGERPRINT,
        "actor_execution_profile": ACTOR_EXECUTION_PROFILE,
        "execution_filter_profile": EXECUTION_FILTER_PROFILE,
        "actor_governor": PERSISTENT_GOVERNOR_FINGERPRINT,
        "warm_start_actor_source_schema": SOURCE_ACTOR_SCHEMA,
    }
    for key, value in expected.items():
        if report_fingerprints.get(key) != value:
            raise ValueError(
                f"initial v3 validation fingerprint {key} mismatch"
            )
    return {
        "path": str(report_path),
        "sha256": _sha256(report_path),
        "format": report["format"],
        "passed": True,
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "replay": str(bootstrap_replay),
        "replay_sha256": bootstrap_sha256,
        "replay_split": "validation",
        "samples": samples,
    }


def _legacy_replay_audit(
    source_state: dict[str, Any], source_state_root: Path
) -> dict[str, Any]:
    raw = source_state.get("legacy_source_replay")
    expected_sha = source_state.get("legacy_source_replay_sha256")
    if not raw:
        raw = source_state.get("latest_replay")
        expected_sha = source_state.get("latest_replay_sha256")
    if not raw or not expected_sha:
        raise ValueError("source state lacks immutable legacy replay provenance")
    path = Path(str(raw)).expanduser().resolve()
    _ensure_inside(path, source_state_root, "source legacy replay")
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = _sha256(path)
    if actual_sha != str(expected_sha):
        raise ValueError("source legacy replay provenance SHA mismatch")
    with np.load(path, allow_pickle=False) as replay:
        if "episode_id" not in replay.files:
            raise ValueError("source legacy replay lacks episode_id")
        ids = sorted(
            set(np.asarray(replay["episode_id"]).astype(str).tolist())
        )
    return {
        "path": str(path),
        "sha256": actual_sha,
        "episode_ids": ids,
        "training_rows_in_target": 0,
        "policy": "immutable_provenance_only_never_merged",
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    source_session = args.source_session_root.expanduser().resolve()
    source_state_root = args.source_state_root.expanduser().resolve()
    bootstrap_replay = args.bootstrap_replay.expanduser().resolve()
    migration_report = args.bootstrap_migration_report.expanduser().resolve()
    initial_v3 = args.initial_v3_checkpoint.expanduser().resolve()
    initial_validation_report = (
        args.initial_v3_validation_report.expanduser().resolve()
    )
    target_session = args.target_session_root.expanduser().resolve()
    workspace = args.workspace.expanduser().resolve()
    runtime = args.runtime.expanduser().resolve()
    if not (workspace / ".venv/bin/python").is_file():
        raise ValueError("gripper-v3 workspace lacks .venv/bin/python")
    if not (workspace / "src/openpi").is_dir():
        raise ValueError("gripper-v3 workspace lacks src/openpi")
    if not (runtime / "piper_runtime").is_dir():
        raise ValueError("gripper-v3 runtime lacks piper_runtime")
    state_dir_name = str(args.target_state_dir)
    if not re.fullmatch(r"[.][A-Za-z0-9._-]+", state_dir_name):
        raise ValueError(
            "--target-state-dir must be one direct hidden directory name"
        )
    target_state = target_session / state_dir_name
    for path, label in (
        (source_session, "source session"),
        (source_state_root, "source state root"),
        (bootstrap_replay, "bootstrap replay"),
        (migration_report, "bootstrap migration report"),
        (initial_v3, "initial v3 checkpoint"),
        (initial_validation_report, "initial v3 validation report"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} is missing: {path}")
    _ensure_inside(source_state_root, source_session, "source state root")
    if target_session == source_session or source_session in target_session.parents:
        raise ValueError("target session must be isolated from source session")
    if target_session.exists() and any(target_session.iterdir()):
        raise FileExistsError(
            f"target session exists and is not empty: {target_session}"
        )

    state_path = source_state_root / "online_state.json"
    source_state = _load_json(state_path)
    if Path(str(source_state.get("session_root", ""))).expanduser().resolve() != (
        source_session
    ):
        raise ValueError("source online_state.json is bound to another session")
    attempt_ids = _canonical_episode_ids(
        source_state.get("last_attempt_episode_ids"),
        "source last_attempt_episode_ids",
    )
    if len(attempt_ids) != EXPECTED_BOOTSTRAP_EPISODES:
        raise ValueError(
            "source last_attempt_episode_ids must contain exactly 30 episodes"
        )
    if int(source_state.get("last_attempt_episode_count", -1)) != len(
        attempt_ids
    ):
        raise ValueError("source last-attempt count/ID mismatch")
    quality = _audit_attempt_quality(source_session, attempt_ids)

    latest_index, latest_episode, incomplete = _latest_complete_episode(
        source_session
    )
    if (
        args.expected_source_latest_episode is not None
        and latest_episode != args.expected_source_latest_episode
    ):
        raise ValueError(
            "source latest-complete episode guard failed: "
            f"{latest_episode} != {args.expected_source_latest_episode}"
        )
    source_episode_directories = _episode_directories(source_session)
    highest_source_index, highest_source_episode_path = (
        source_episode_directories[-1]
    )
    target_episode_floor = highest_source_index + 1
    if (
        args.expected_target_episode_floor is not None
        and target_episode_floor != args.expected_target_episode_floor
    ):
        raise ValueError(
            "target episode floor guard failed: "
            f"{target_episode_floor} != "
            f"{args.expected_target_episode_floor}"
        )

    actor_raw = source_state.get("initial_actor_warm_start_checkpoint")
    if not actor_raw:
        raise ValueError(
            "source state lacks initial_actor_warm_start_checkpoint"
        )
    source_actor_path = Path(str(actor_raw)).expanduser().resolve()
    _ensure_inside(source_actor_path, source_state_root, "source Actor")
    if args.source_actor_checkpoint is not None:
        explicit_actor = (
            args.source_actor_checkpoint.expanduser().resolve()
        )
        if explicit_actor != source_actor_path:
            raise ValueError(
                "explicit source Actor differs from state-bound Actor"
            )
    source_actor = _audit_source_actor(source_actor_path)

    rejected_replay_raw = source_state.get("latest_rejected_replay")
    rejected_checkpoint_raw = source_state.get("latest_rejected_checkpoint")
    if not rejected_replay_raw or not rejected_checkpoint_raw:
        raise ValueError(
            "source state lacks latest rejected replay/checkpoint provenance"
        )
    rejected_replay = Path(str(rejected_replay_raw)).expanduser().resolve()
    rejected_checkpoint = Path(
        str(rejected_checkpoint_raw)
    ).expanduser().resolve()
    _ensure_inside(
        rejected_replay, source_state_root, "source rejected replay"
    )
    _ensure_inside(
        rejected_checkpoint,
        source_state_root,
        "source rejected checkpoint",
    )
    if not rejected_replay.is_file():
        raise FileNotFoundError(rejected_replay)
    rejected_replay_sha = _sha256(rejected_replay)
    rejected = _audit_rejected_checkpoint(
        rejected_checkpoint,
        attempt_replay_sha256=rejected_replay_sha,
    )
    if (
        rejected["tree_sha256"] == source_actor["tree_sha256"]
        or rejected["learner_msgpack_sha256"]
        == source_actor["learner_msgpack_sha256"]
    ):
        raise ValueError("source Actor aliases the rejected step-144 checkpoint")
    deployment_raw = source_state.get("deployment_checkpoint")
    if deployment_raw and Path(
        str(deployment_raw)
    ).expanduser().resolve() != source_actor_path:
        raise ValueError(
            "source deployment checkpoint is not the fixed step-12627 Actor"
        )
    latest_accepted = source_state.get("latest_checkpoint")
    if latest_accepted and Path(
        str(latest_accepted)
    ).expanduser().resolve() == rejected_checkpoint:
        raise ValueError("rejected step 144 is incorrectly marked accepted")

    bootstrap = _replay_audit(
        bootstrap_replay, expected_episode_ids=attempt_ids
    )
    source_attempt_train = int(
        source_state.get("last_attempt_train_transition_count", -1)
    )
    if source_attempt_train != bootstrap["train_transitions"]:
        raise ValueError(
            "source last-attempt train count does not match migrated bootstrap"
        )
    migration = _audit_bootstrap_migration(
        migration_report,
        source_replay=rejected_replay,
        source_replay_sha256=rejected_replay_sha,
        bootstrap=bootstrap,
    )
    initial = _audit_initial_v3_checkpoint(
        initial_v3,
        bootstrap_replay_sha256=bootstrap["sha256"],
        source_actor=source_actor,
        rejected=rejected,
        expected_step=args.expected_initial_v3_checkpoint_step,
    )
    initial_validation = _audit_initial_v3_validation(
        initial_validation_report,
        checkpoint=initial_v3,
        checkpoint_step=initial["step"],
        bootstrap_replay=bootstrap_replay,
        bootstrap_sha256=bootstrap["sha256"],
    )
    legacy = _legacy_replay_audit(source_state, source_state_root)

    return {
        "format": FORK_FORMAT,
        "mode": "create" if args.create else "read_only_preflight",
        "preflight_passed": True,
        "created": False,
        "source": {
            "session_root": str(source_session),
            "state_root": str(source_state_root),
            "online_state": str(state_path),
            "online_state_sha256": _sha256(state_path),
            "last_attempt_episode_ids": attempt_ids,
            "last_attempt_train_transition_count": source_attempt_train,
            "attempt_quality": quality,
            "latest_complete_episode": latest_episode,
            "latest_complete_episode_index": latest_index,
            "highest_source_episode_directory": (
                highest_source_episode_path.name
            ),
            "highest_source_episode_directory_index": highest_source_index,
            "incomplete_episode_directories": incomplete,
            "actor": source_actor,
            "rejected_candidate": rejected,
            "rejected_replay": {
                "path": str(rejected_replay),
                "sha256": rejected_replay_sha,
            },
            "legacy_replay": legacy,
        },
        "bootstrap": {
            **bootstrap,
            "migration_report": {
                "path": migration["path"],
                "sha256": migration["sha256"],
            },
        },
        "initial_v3_checkpoint": initial,
        "initial_v3_validation": initial_validation,
        "target": {
            "session_root": str(target_session),
            "state_root": str(target_state),
            "workspace": str(workspace),
            "runtime": str(runtime),
            "selected_actor_file": str(
                target_state / "selected_actor_checkpoint.txt"
            ),
            "shadow_service": "openpi-rlt-shadow-policy-gripper-v3.service",
            "episode_index_floor": target_episode_floor,
            "first_episode_id": f"episode_{target_episode_floor:06d}",
            "initial_update_index": 1,
            "initial_attempt_index": 1,
        },
        "contract": _contract_payload(),
        "immutability_policy": (
            "create_exclusive_content_addressed_sha256_validate_fail_closed"
        ),
    }


def _contract_payload() -> dict[str, Any]:
    return {
        "state_format": STATE_FORMAT,
        "lineage_mode": LINEAGE_MODE,
        "replay_training_policy": REPLAY_POLICY,
        "actor_execution_profile": ACTOR_EXECUTION_PROFILE,
        "actor_model_action_schema_fingerprint": (
            ACTOR_MODEL_ACTION_SCHEMA_FINGERPRINT
        ),
        "execution_action_schema_fingerprint": ACTION_SCHEMA_FINGERPRINT,
        "actor_projection_profile": ACTOR_PROJECTION_PROFILE,
        "execution_filter_profile": EXECUTION_FILTER_PROFILE,
        "execution_filter_tau_s": EXECUTION_FILTER_TAU_S,
        "control_hz": CONTROL_HZ,
        "control_dt_s": CONTROL_DT_S,
        "execution_filter_alpha": EXECUTION_FILTER_ALPHA,
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
        "gripper_residual_mode": GRIPPER_RESIDUAL_MODE,
        "actor_gripper_residual_max_close_m": (
            GRIPPER_RESIDUAL_MAX_CLOSE_M
        ),
        "actor_gripper_residual_d1_max_m": GRIPPER_RESIDUAL_D1_MAX_M,
        "actor_gripper_residual_d2_max_m": GRIPPER_RESIDUAL_D2_MAX_M,
        "actor_gripper_max_boundary_jump_m": (
            GRIPPER_MAX_BOUNDARY_JUMP_M
        ),
        "gripper_command_min_m": GRIPPER_COMMAND_MIN_M,
        "gripper_command_max_m": GRIPPER_COMMAND_MAX_M,
        "gripper_release_reference_m": GRIPPER_RELEASE_REFERENCE_M,
        "gripper_release_delta_m": GRIPPER_RELEASE_DELTA_M,
        "human_gripper_q_filter_mode": HUMAN_GRIPPER_Q_FILTER_MODE,
        "human_gripper_q_filter_margin": HUMAN_GRIPPER_Q_FILTER_MARGIN,
        "human_gripper_supervision_policy": (
            "all_admitted_human_reward_labels_preserved_"
            "critic_min_advantage_q_filtered"
        ),
        "replay_plan_contract": (
            "complete_same_plan_c10_offsets_0_through_9"
        ),
    }


def _config_env(
    *,
    state_root: Path,
    session_root: Path,
    workspace: Path,
    runtime: Path,
    source_actor: Path,
    floor: int,
    target_betas: dict[str, float],
    initial_fingerprints: dict[str, Any],
) -> str:
    home = Path.home()
    values: dict[str, Any] = {
        "RLT_SESSION_ROOT": str(session_root),
        "RLT_STATE_ROOT": str(state_root),
        "RLT_WORKSPACE": str(workspace),
        "RLT_RUNTIME": str(runtime),
        "RLT_PHASE_CHECKPOINT": str(
            home
            / "rlt_phase_classifiers"
            / "greenblock_box_resnet18_v4_manual_intervals"
            / "phase_classifier.pt"
        ),
        "RLT_SELECTED_ACTOR_FILE": str(
            state_root / "selected_actor_checkpoint.txt"
        ),
        "RLT_SHADOW_SERVICE": (
            "openpi-rlt-shadow-policy-gripper-v3.service"
        ),
        "RLT_POLICY_HOST": "127.0.0.1",
        "RLT_POLICY_PORT": 8001,
        "RLT_LINEAGE_MODE": LINEAGE_MODE,
        "RLT_REPLAY_TRAINING_POLICY": REPLAY_POLICY,
        "RLT_ACTOR_EXECUTION_PROFILE": ACTOR_EXECUTION_PROFILE,
        "RLT_ACTOR_MODEL_ACTION_SCHEMA_FINGERPRINT": (
            ACTOR_MODEL_ACTION_SCHEMA_FINGERPRINT
        ),
        "RLT_EXECUTION_ACTION_SCHEMA_FINGERPRINT": (
            ACTION_SCHEMA_FINGERPRINT
        ),
        "RLT_ACTOR_PROJECTION_PROFILE": ACTOR_PROJECTION_PROFILE,
        "RLT_EXECUTION_FILTER_PROFILE": EXECUTION_FILTER_PROFILE,
        "RLT_EXECUTION_FILTER_TAU_S": EXECUTION_FILTER_TAU_S,
        "RLT_CONTROL_HZ": CONTROL_HZ,
        "RLT_CONTROL_DT_S": CONTROL_DT_S,
        "RLT_EXECUTION_FILTER_ALPHA": EXECUTION_FILTER_ALPHA,
        "RLT_CHUNK_LENGTH": CHUNK_LENGTH,
        "RLT_CHUNK_STRIDE": CHUNK_STRIDE,
        "RLT_REPLAY_STRIDE": CHUNK_STRIDE,
        "RLT_N_STEP": CHUNK_STRIDE,
        "RLT_ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD": (
            ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD
        ),
        "RLT_ACTOR_PROJECTION_SCALE_STEPS": ACTOR_PROJECTION_SCALE_STEPS,
        "RLT_ACTOR_MIN_PROJECTION_SCALE": ACTOR_MIN_PROJECTION_SCALE,
        "RLT_ACTOR_DIRECTION_STATIC_THRESHOLD_RAD": (
            ACTOR_DIRECTION_STATIC_THRESHOLD_RAD
        ),
        "RLT_ACTOR_GOVERNOR_FINGERPRINT": (
            PERSISTENT_GOVERNOR_FINGERPRINT
        ),
        "RLT_GRIPPER_RESIDUAL_MODE": GRIPPER_RESIDUAL_MODE,
        "RLT_GRIPPER_RESIDUAL_MAX": GRIPPER_RESIDUAL_MAX_CLOSE_M,
        "RLT_GRIPPER_RESIDUAL_D1_MAX_M": GRIPPER_RESIDUAL_D1_MAX_M,
        "RLT_GRIPPER_RESIDUAL_D2_MAX_M": GRIPPER_RESIDUAL_D2_MAX_M,
        "RLT_GRIPPER_MAX_BOUNDARY_JUMP_M": (
            GRIPPER_MAX_BOUNDARY_JUMP_M
        ),
        "RLT_GRIPPER_COMMAND_MIN_M": GRIPPER_COMMAND_MIN_M,
        "RLT_GRIPPER_COMMAND_MAX_M": GRIPPER_COMMAND_MAX_M,
        "RLT_GRIPPER_RELEASE_REFERENCE_M": (
            GRIPPER_RELEASE_REFERENCE_M
        ),
        "RLT_GRIPPER_RELEASE_DELTA_M": GRIPPER_RELEASE_DELTA_M,
        "RLT_FREEZE_GRIPPER_RESIDUAL": 0,
        "RLT_BETA_BC": target_betas["beta_bc"],
        "RLT_BETA_HUMAN_BC": target_betas["beta_human_bc"],
        "RLT_BETA_HUMAN_GRIPPER_BC": target_betas[
            "beta_human_gripper_bc"
        ],
        "RLT_HUMAN_GRIPPER_BC_SCALE_M": GRIPPER_RESIDUAL_MAX_CLOSE_M,
        "RLT_HUMAN_GRIPPER_Q_FILTER_MODE": HUMAN_GRIPPER_Q_FILTER_MODE,
        "RLT_HUMAN_GRIPPER_Q_FILTER_MARGIN": HUMAN_GRIPPER_Q_FILTER_MARGIN,
        "RLT_ALLOW_WARM_START_OBJECTIVE_MIGRATION": 1,
        "RLT_WARM_START_ACTOR_CHECKPOINT": str(source_actor),
        "RLT_WARMUP_EPISODES": EXPECTED_BOOTSTRAP_EPISODES,
        "RLT_MIN_NEW_PERSISTENT_COMMITTED_EPISODES": 0,
        "RLT_EPISODE_INDEX_FLOOR": floor,
        "RLT_MIN_SUCCESS": 2,
        "RLT_MIN_FAILURE": 2,
        "RLT_MIN_SUCCESS_HUMAN_EPISODES": 0,
        "RLT_MIN_ADMITTED_HUMAN_EPISODES": 1,
        "RLT_UPDATE_EVERY": 1,
        "RLT_MIN_WARMUP_TRANSITIONS": 30,
        "RLT_UTD": 1.0,
        "RLT_MIN_UPDATE_STEPS": 1,
        "RLT_MAX_UPDATE_STEPS": 1250,
        "RLT_BATCH_SIZE": 256,
        "RLT_REFERENCE_DROPOUT": 0.5,
        "RLT_TARGET_POLICY_NOISE_STD": 0.1,
        "RLT_TARGET_POLICY_NOISE_CLIP": 0.2,
        "RLT_ACTION_HORIZON": 50,
        "RLT_MODEL_EXECUTE_STEPS": 50,
        "RLT_RESIDUAL_MAX": 0.005,
        "RLT_RESIDUAL_D1_MAX_RAD": 0.0015,
        "RLT_RESIDUAL_D2_MAX_RAD": 0.001,
        "RLT_DIRECTION_CONE_DEG": 15.0,
        "RLT_SUCCESS_FRACTION": 0.5,
        "RLT_HUMAN_FRACTION": 0.5,
        "RLT_VALIDATION_FRACTION": 0.15,
        "RLT_MAX_VALIDATION_TD_ERROR": 0.5,
        "RLT_MAX_ACTOR_Q_ADVANTAGE": 0.5,
        "RLT_BASE_FINGERPRINT": initial_fingerprints.get(
            "base_checkpoint", ""
        ),
        "RLT_TOKEN_FINGERPRINT": initial_fingerprints.get("rl_token", ""),
        "RLT_PHASE_FINGERPRINT": initial_fingerprints.get(
            "phase_classifier", ""
        ),
    }
    return "".join(
        f"{key}={shlex.quote(str(value))}\n"
        for key, value in values.items()
    )


def _make_read_only(path: Path) -> None:
    if path.is_file():
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        return
    for item in sorted(path.rglob("*"), reverse=True):
        if item.is_file():
            item.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def create_lineage(
    report: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    target_session = Path(report["target"]["session_root"])
    target_state = Path(report["target"]["state_root"])
    temporary_session = target_session.with_name(
        f".{target_session.name}.creating-{os.getpid()}"
    )
    if target_session.exists() and any(target_session.iterdir()):
        raise FileExistsError(
            f"target session exists and is not empty: {target_session}"
        )
    if temporary_session.exists():
        raise FileExistsError(
            f"temporary target already exists: {temporary_session}"
        )
    temporary_state = temporary_session / target_state.name
    provenance = temporary_state / "provenance"
    bootstrap_dir = provenance / "bootstrap_gripper_v5"
    learner_dir = temporary_state / "learner"
    source_actor_src = Path(report["source"]["actor"]["checkpoint"])
    legacy_src = Path(report["source"]["legacy_replay"]["path"])
    bootstrap_src = Path(report["bootstrap"]["path"])
    bootstrap_report_src = Path(
        report["bootstrap"]["migration_report"]["path"]
    )
    initial_src = Path(report["initial_v3_checkpoint"]["checkpoint"])
    initial_validation_src = Path(
        report["initial_v3_validation"]["path"]
    )
    source_actor_copy = (
        provenance
        / f"source_actor_checkpoint_step_{EXPECTED_SOURCE_ACTOR_STEP:08d}"
    )
    legacy_copy = provenance / "legacy_source_replay_v2.npz"
    bootstrap_copy = bootstrap_dir / "bootstrap_replay_v5.npz"
    bootstrap_report_copy = bootstrap_dir / "migration_report.json"
    initial_copy = (
        learner_dir
        / f"step_{report['initial_v3_checkpoint']['step']:08d}"
    )
    initial_validation_copy = (
        learner_dir / "initial_v3_acceptance.json"
    )
    try:
        provenance.mkdir(parents=True)
        learner_dir.mkdir(parents=True)
        shutil.copytree(
            source_actor_src, source_actor_copy, copy_function=shutil.copy2
        )
        shutil.copy2(legacy_src, legacy_copy)
        bootstrap_dir.mkdir()
        shutil.copy2(bootstrap_src, bootstrap_copy)
        shutil.copy2(bootstrap_report_src, bootstrap_report_copy)
        shutil.copytree(
            initial_src, initial_copy, copy_function=shutil.copy2
        )
        shutil.copy2(initial_validation_src, initial_validation_copy)
        copied_actor_files, copied_actor_tree = _checkpoint_manifest(
            source_actor_copy
        )
        _, copied_initial_tree = _checkpoint_manifest(initial_copy)
        copy_checks = {
            "source_actor_tree": (
                copied_actor_tree,
                report["source"]["actor"]["tree_sha256"],
            ),
            "legacy_replay": (
                _sha256(legacy_copy),
                report["source"]["legacy_replay"]["sha256"],
            ),
            "bootstrap_replay": (
                _sha256(bootstrap_copy),
                report["bootstrap"]["sha256"],
            ),
            "bootstrap_migration_report": (
                _sha256(bootstrap_report_copy),
                report["bootstrap"]["migration_report"]["sha256"],
            ),
            "initial_v3_tree": (
                copied_initial_tree,
                report["initial_v3_checkpoint"]["tree_sha256"],
            ),
            "initial_v3_validation": (
                _sha256(initial_validation_copy),
                report["initial_v3_validation"]["sha256"],
            ),
        }
        failed = {
            key: {"actual": actual, "expected": expected}
            for key, (actual, expected) in copy_checks.items()
            if actual != expected
        }
        if failed:
            raise RuntimeError(f"copied artifact SHA verification failed: {failed}")

        final_state = target_session / target_state.name
        final_source_actor = final_state / source_actor_copy.relative_to(
            temporary_state
        )
        final_legacy = final_state / legacy_copy.relative_to(temporary_state)
        final_bootstrap = final_state / bootstrap_copy.relative_to(
            temporary_state
        )
        final_bootstrap_report = (
            final_state
            / bootstrap_report_copy.relative_to(temporary_state)
        )
        final_initial = final_state / initial_copy.relative_to(
            temporary_state
        )
        final_initial_validation = (
            final_state
            / initial_validation_copy.relative_to(temporary_state)
        )
        created_unix = time.time()
        source_betas = report["source"]["actor"]["objective_weights"]
        target_betas = report["initial_v3_checkpoint"][
            "objective_weights"
        ]
        objective_migration = {
            "format": OBJECTIVE_MIGRATION_FORMAT,
            "authorization": OBJECTIVE_AUTHORIZATION,
            "source_beta_bc": source_betas["beta_bc"],
            "source_beta_human_bc": source_betas["beta_human_bc"],
            "source_beta_human_gripper_bc": source_betas[
                "beta_human_gripper_bc"
            ],
            "target_beta_bc": target_betas["beta_bc"],
            "target_beta_human_bc": target_betas["beta_human_bc"],
            "target_beta_human_gripper_bc": target_betas[
                "beta_human_gripper_bc"
            ],
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
            "status": "consumed_and_audited_in_initial_v3_checkpoint",
            "accepted_checkpoint": str(final_initial),
        }
        objective_manifest = {
            "format": OBJECTIVE_MIGRATION_FORMAT,
            "authorization": OBJECTIVE_AUTHORIZATION,
            "source": {
                "checkpoint": str(final_source_actor),
                "checkpoint_step": EXPECTED_SOURCE_ACTOR_STEP,
                "checkpoint_tree_sha256": report["source"]["actor"][
                    "tree_sha256"
                ],
                "learner_msgpack_sha256": _manifest_file_sha(
                    copied_actor_files, "learner.msgpack"
                ),
                **source_betas,
            },
            "target": {
                "checkpoint": str(final_initial),
                "checkpoint_step": report["initial_v3_checkpoint"]["step"],
                "checkpoint_tree_sha256": report[
                    "initial_v3_checkpoint"
                ]["tree_sha256"],
                **target_betas,
                "human_gripper_q_filter_mode": (
                    HUMAN_GRIPPER_Q_FILTER_MODE
                ),
                "human_gripper_q_filter_margin": (
                    HUMAN_GRIPPER_Q_FILTER_MARGIN
                ),
            },
            "human_supervision_scope": (
                "all_admitted_human_reward_positive_and_reward_negative"
            ),
            "physical_governor_changed": True,
            "bootstrap_replay": str(final_bootstrap),
            "bootstrap_replay_sha256": report["bootstrap"]["sha256"],
            "rejected_source_checkpoint": {
                "step": EXPECTED_REJECTED_STEP,
                "tree_sha256": report["source"]["rejected_candidate"][
                    "tree_sha256"
                ],
                "disposition": "never_warm_start_never_deploy",
            },
            "created_unix": created_unix,
        }
        objective_path = temporary_state / "objective_migration.json"
        _atomic_json(objective_path, objective_manifest)
        objective_sha = _sha256(objective_path)

        manifest = {
            **report,
            "mode": "created",
            "created": True,
            "created_unix": created_unix,
            "target": {
                **report["target"],
                "source_actor_checkpoint_copy": str(final_source_actor),
                "legacy_replay_provenance_copy": str(final_legacy),
                "bootstrap_gripper_replay_copy": str(final_bootstrap),
                "bootstrap_migration_report_copy": str(
                    final_bootstrap_report
                ),
                "initial_v3_checkpoint_copy": str(final_initial),
                "initial_v3_validation_report_copy": str(
                    final_initial_validation
                ),
                "objective_migration_manifest": str(
                    final_state / objective_path.name
                ),
                "objective_migration_manifest_sha256": objective_sha,
            },
        }
        manifest_path = temporary_state / "fork_manifest.json"
        _atomic_json(manifest_path, manifest)
        manifest_sha = _sha256(manifest_path)
        episode_ids = list(report["bootstrap"]["episode_ids"])
        train_count = int(report["bootstrap"]["train_transitions"])
        state = {
            "format": STATE_FORMAT,
            "session_root": str(target_session),
            "workspace": report["target"]["workspace"],
            "runtime": report["target"]["runtime"],
            "selected_checkpoint_file": report["target"][
                "selected_actor_file"
            ],
            "shadow_service": report["target"]["shadow_service"],
            "lineage_mode": LINEAGE_MODE,
            "fork_manifest": str(final_state / manifest_path.name),
            "fork_manifest_sha256": manifest_sha,
            **{
                key: value
                for key, value in report["contract"].items()
                if key
                not in {
                    "state_format",
                    "lineage_mode",
                }
            },
            "bootstrap_gripper_replay": str(final_bootstrap),
            "bootstrap_gripper_replay_sha256": report["bootstrap"][
                "sha256"
            ],
            "bootstrap_gripper_episode_ids": episode_ids,
            "bootstrap_gripper_episode_count": len(episode_ids),
            "bootstrap_gripper_train_transition_count": train_count,
            "bootstrap_gripper_quality": report["bootstrap"]["quality"],
            "bootstrap_gripper_migration_report": str(
                final_bootstrap_report
            ),
            "bootstrap_gripper_migration_report_sha256": report[
                "bootstrap"
            ]["migration_report"]["sha256"],
            "legacy_source_replay": str(final_legacy),
            "legacy_source_replay_sha256": report["source"][
                "legacy_replay"
            ]["sha256"],
            "legacy_source_replay_episode_ids": report["source"][
                "legacy_replay"
            ]["episode_ids"],
            "legacy_replay_training_rows": 0,
            "legacy_replay_policy": (
                "immutable_provenance_only_never_merged"
            ),
            "initial_actor_warm_start_checkpoint": str(
                final_source_actor
            ),
            "initial_actor_warm_start_checkpoint_tree_sha256": report[
                "source"
            ]["actor"]["tree_sha256"],
            "initial_actor_warm_start_learner_msgpack_sha256": report[
                "source"
            ]["actor"]["learner_msgpack_sha256"],
            "source_actor_checkpoint_step": EXPECTED_SOURCE_ACTOR_STEP,
            "source_rejected_checkpoint_step": EXPECTED_REJECTED_STEP,
            "source_rejected_checkpoint_tree_sha256": report["source"][
                "rejected_candidate"
            ]["tree_sha256"],
            "source_rejected_checkpoint_learner_msgpack_sha256": report[
                "source"
            ]["rejected_candidate"]["learner_msgpack_sha256"],
            "source_rejected_checkpoint_disposition": (
                "rejected_never_warm_start_never_deploy"
            ),
            "objective_migration": objective_migration,
            "objective_migration_manifest": str(
                final_state / objective_path.name
            ),
            "objective_migration_manifest_sha256": objective_sha,
            "warm_start_policy": (
                "actor_only_from_step12627_fresh_v3_critic_then_bootstrap_trained"
            ),
            "warm_start_status": (
                "consumed_and_audited_in_initial_v3_checkpoint"
            ),
            "warm_start_consumed": True,
            "candidate_action_normalization_status": (
                "refit_from_bootstrap_gripper_v5_replay"
            ),
            "normalization_policy": (
                "frozen_from_initial_v3_bootstrap_training"
            ),
            "deployment_checkpoint": str(final_initial),
            "latest_checkpoint": str(final_initial),
            "initial_v3_checkpoint": str(final_initial),
            "initial_v3_checkpoint_step": report[
                "initial_v3_checkpoint"
            ]["step"],
            "initial_v3_checkpoint_tree_sha256": report[
                "initial_v3_checkpoint"
            ]["tree_sha256"],
            "initial_v3_checkpoint_learner_msgpack_sha256": report[
                "initial_v3_checkpoint"
            ]["learner_msgpack_sha256"],
            "initial_v3_validation_report": str(
                final_initial_validation
            ),
            "initial_v3_validation_report_sha256": report[
                "initial_v3_validation"
            ]["sha256"],
            "initial_v3_validation_passed": True,
            "initial_v3_validation_samples": report[
                "initial_v3_validation"
            ]["samples"],
            "latest_replay": str(final_bootstrap),
            "latest_replay_sha256": report["bootstrap"]["sha256"],
            "trained_episode_ids": episode_ids,
            "last_attempt_episode_ids": episode_ids,
            "last_update_episode_count": len(episode_ids),
            "last_attempt_episode_count": len(episode_ids),
            "last_train_transition_count": train_count,
            "last_attempt_train_transition_count": train_count,
            "latest_training_steps": report["initial_v3_checkpoint"][
                "step"
            ],
            "attempt_index": report["target"]["initial_attempt_index"],
            "update_index": report["target"]["initial_update_index"],
            "episode_index_floor": report["target"][
                "episode_index_floor"
            ],
            "min_new_persistent_committed_episodes": 0,
            "administrative_promotion": {
                "format": (
                    "openpi_piper_gripper_v3_initial_checkpoint_promotion"
                ),
                "checkpoint": str(final_initial),
                "checkpoint_tree_sha256": report[
                    "initial_v3_checkpoint"
                ]["tree_sha256"],
                "bootstrap_replay_sha256": report["bootstrap"]["sha256"],
                "source_actor_step": EXPECTED_SOURCE_ACTOR_STEP,
                "rejected_source_step": EXPECTED_REJECTED_STEP,
                "authorized": True,
            },
            "created_unix": created_unix,
            "updated_unix": created_unix,
        }
        _atomic_json(temporary_state / "online_state.json", state)
        (temporary_state / "config.env").write_text(
            _config_env(
                state_root=final_state,
                session_root=target_session,
                workspace=Path(report["target"]["workspace"]),
                runtime=Path(report["target"]["runtime"]),
                source_actor=final_source_actor,
                floor=report["target"]["episode_index_floor"],
                target_betas=target_betas,
                initial_fingerprints=report["initial_v3_checkpoint"][
                    "fingerprints"
                ],
            ),
            encoding="utf-8",
        )
        _make_read_only(source_actor_copy)
        _make_read_only(legacy_copy)
        _make_read_only(bootstrap_dir)
        _make_read_only(initial_copy)
        _make_read_only(initial_validation_copy)
        _make_read_only(objective_path)
        _make_read_only(manifest_path)
        temporary_session.rename(target_session)
    except Exception:
        if temporary_session.exists():
            try:
                for item in temporary_session.rglob("*"):
                    if item.is_file():
                        item.chmod(stat.S_IRUSR | stat.S_IWUSR)
                shutil.rmtree(temporary_session)
            except OSError:
                pass
        raise
    return manifest


def main() -> None:
    args = _parser().parse_args()
    report = preflight(args)
    if args.create:
        report = create_lineage(report, args)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
