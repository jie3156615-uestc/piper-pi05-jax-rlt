#!/usr/bin/env python3
"""Preflight or create an isolated persistent-v2 lineage from a legacy Actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import time
from typing import Any

import numpy as np

from persistent_v2_contract import (
    ACTOR_DIRECTION_STATIC_THRESHOLD_RAD,
    ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD,
    ACTOR_MIN_PROJECTION_SCALE,
    ACTOR_PROJECTION_SCALE_STEPS,
    ACTION_SCHEMA_FINGERPRINT,
    ACTOR_EXECUTION_PROFILE,
    ACTOR_PROJECTION_PROFILE,
    CHUNK_LENGTH,
    CHUNK_STRIDE,
    CONTROL_DT_S,
    CONTROL_HZ,
    DEFAULT_MIN_NEW_COMMITTED_EPISODES,
    EXECUTION_FILTER_ALPHA,
    EXECUTION_FILTER_PROFILE,
    EXECUTION_FILTER_TAU_S,
    PERSISTENT_GOVERNOR_FINGERPRINT,
)


FORMAT = "openpi_piper_persistent_v2_lineage_fork"
STATE_FORMAT = "openpi_piper_online_rlt_state_persistent_v2"
SOURCE_ACTION_SCHEMA = (
    "piper_joint_delta_v3_c10_n10_stride2_behavior_ref50_"
    "rank1_bump_r005_d1_0015_d2_001_cone15_"
    "gripper_absolute_frozen_residual"
)
BASE_FINGERPRINT = (
    "full20k_step20000_metadata_sha256_"
    "14d9cac129ec7ce91f2e5aab3f5bfac06172c8fb70709f01850fb8e8215870e5"
)
TOKEN_FINGERPRINT = "2f2e1e6bbcae8f08217ec7ba0b88088bfa44627be495035319deb84e06052b49"
PHASE_FINGERPRINT = "8c5b443edd3f399529680ef5e4c5dffcdee4af2ae2f224f6da4152237c9a50dc"
EPISODE_PATTERN = re.compile(r"episode_([0-9]+)")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-session-root", type=Path, required=True)
    parser.add_argument("--source-state-root", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--target-session-root", type=Path, required=True)
    parser.add_argument(
        "--target-state-dir", default=".online_rlt_persistent_v2"
    )
    parser.add_argument(
        "--min-new-persistent-episodes",
        type=int,
        default=DEFAULT_MIN_NEW_COMMITTED_EPISODES,
    )
    parser.add_argument("--expected-source-latest-episode")
    parser.add_argument("--expected-source-checkpoint-step", type=int)
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create the isolated target. Without this flag the command is read-only.",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _checkpoint_manifest(checkpoint: Path) -> tuple[list[dict[str, Any]], str]:
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
        raise ValueError(f"source checkpoint has no files: {checkpoint}")
    return files, digest.hexdigest()


def _episode_directories(session: Path) -> list[tuple[int, Path]]:
    result: list[tuple[int, Path]] = []
    for path in session.glob("episode_[0-9]*"):
        match = EPISODE_PATTERN.fullmatch(path.name)
        if path.is_dir() and match:
            result.append((int(match.group(1)), path))
    return sorted(result)


def _replay_episode_ids(path: Path) -> list[str]:
    with np.load(path, allow_pickle=False) as replay:
        if "episode_id" not in replay.files:
            raise ValueError(f"source replay lacks episode_id: {path}")
        return sorted(set(np.asarray(replay["episode_id"]).astype(str).tolist()))


def _ensure_inside(path: Path, parent: Path, label: str) -> None:
    if path != parent and parent not in path.parents:
        raise ValueError(f"{label} must be inside {parent}: {path}")


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    source_session = args.source_session_root.expanduser().resolve()
    source_state = args.source_state_root.expanduser().resolve()
    checkpoint = args.source_checkpoint.expanduser().resolve()
    target_session = args.target_session_root.expanduser().resolve()
    state_dir = str(args.target_state_dir)
    if not re.fullmatch(r"[.][A-Za-z0-9._-]+", state_dir):
        raise ValueError(
            "--target-state-dir must be one direct hidden directory name"
        )
    target_state = target_session / state_dir
    if args.min_new_persistent_episodes < DEFAULT_MIN_NEW_COMMITTED_EPISODES:
        raise ValueError(
            "persistent-v2 first-update warmup cannot be weakened below "
            f"{DEFAULT_MIN_NEW_COMMITTED_EPISODES} episodes"
        )
    for path, label in (
        (source_session, "source session"),
        (source_state, "source state"),
        (checkpoint, "source checkpoint"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} is missing: {path}")
    _ensure_inside(source_state, source_session, "source state")
    _ensure_inside(checkpoint, source_state, "source checkpoint")
    if target_session == source_session or source_session in target_session.parents:
        raise ValueError("target session must not be the source or a child of it")
    if target_session == source_state or source_state in target_session.parents:
        raise ValueError("target session must not be inside the legacy state")
    if target_session.exists() and any(target_session.iterdir()):
        raise FileExistsError(
            f"target session exists and is not empty: {target_session}"
        )

    state_path = source_state / "online_state.json"
    source_online_state = _load_json(state_path)
    if Path(source_online_state.get("session_root", "")).resolve() != source_session:
        raise ValueError("source online_state.json is bound to another session")
    if Path(source_online_state.get("latest_checkpoint", "")).resolve() != checkpoint:
        raise ValueError(
            "source checkpoint is not the exact latest_checkpoint in source state"
        )
    latest_replay_raw = source_online_state.get("latest_replay")
    latest_replay_sha = source_online_state.get("latest_replay_sha256")
    if not latest_replay_raw or not latest_replay_sha:
        raise ValueError("source state lacks exact latest replay provenance")
    latest_replay = Path(latest_replay_raw).expanduser().resolve()
    _ensure_inside(latest_replay, source_state, "source latest replay")
    if not latest_replay.is_file():
        raise FileNotFoundError(f"source latest replay is missing: {latest_replay}")
    actual_replay_sha = _sha256(latest_replay)
    if actual_replay_sha != str(latest_replay_sha):
        raise ValueError(
            "source replay SHA does not match source online_state.json"
        )

    metadata_path = checkpoint / "metadata.json"
    learner_path = checkpoint / "learner.msgpack"
    if not metadata_path.is_file() or not learner_path.is_file():
        raise ValueError("source checkpoint is incomplete")
    checkpoint_metadata = _load_json(metadata_path)
    source_step = int(checkpoint_metadata.get("update_step", -1))
    if (
        args.expected_source_checkpoint_step is not None
        and source_step != args.expected_source_checkpoint_step
    ):
        raise ValueError(
            "stale source checkpoint step guard: "
            f"{source_step} != {args.expected_source_checkpoint_step}"
        )
    fingerprints = checkpoint_metadata.get("fingerprints") or {}
    if fingerprints.get("replay_sha256") != actual_replay_sha:
        raise ValueError(
            "checkpoint replay fingerprint does not match exact latest replay"
        )
    if fingerprints.get("action_schema") != SOURCE_ACTION_SCHEMA:
        raise ValueError(
            "source checkpoint is not the expected legacy v3 Actor schema"
        )
    source_config = checkpoint_metadata.get("config")
    if not isinstance(source_config, dict):
        raise ValueError("source checkpoint lacks learner config")
    expected_source_config = {
        "beta_bc": 40.0,
        "beta_human_bc": 0.0,
        "chunk_stride": 2,
        "actor_residual_parameterization": "rank1_bump",
        "freeze_gripper_residual": True,
    }
    source_config_mismatches = {
        key: {"actual": source_config.get(key), "expected": expected}
        for key, expected in expected_source_config.items()
        if source_config.get(key) != expected
    }
    if source_config_mismatches:
        raise ValueError(
            f"source checkpoint is not the expected v8 beta40 Actor: "
            f"{source_config_mismatches}"
        )

    replay_episode_ids = _replay_episode_ids(latest_replay)
    trained_episode_ids = sorted(
        set(source_online_state.get("trained_episode_ids") or [])
    )
    if replay_episode_ids != trained_episode_ids:
        raise ValueError(
            "source latest replay episode IDs differ from trained_episode_ids"
        )

    source_episodes = _episode_directories(source_session)
    if not source_episodes:
        raise ValueError("source session has no episode directories")
    latest_index = source_episodes[-1][0]
    latest_episode = f"episode_{latest_index:06d}"
    if (
        args.expected_source_latest_episode is not None
        and latest_episode != args.expected_source_latest_episode
    ):
        raise ValueError(
            "stale source latest episode guard: "
            f"{latest_episode} != {args.expected_source_latest_episode}"
        )
    all_source_episode_ids = [
        f"episode_{index:06d}" for index, _ in source_episodes
    ]
    post_checkpoint_episode_ids = sorted(
        set(all_source_episode_ids).difference(replay_episode_ids)
    )
    checkpoint_files, checkpoint_tree_sha = _checkpoint_manifest(checkpoint)
    learner_msgpack_sha = next(
        item["sha256"]
        for item in checkpoint_files
        if item["path"] == "learner.msgpack"
    )

    return {
        "format": FORMAT,
        "mode": "create" if args.create else "read_only_preflight",
        "source": {
            "session_root": str(source_session),
            "state_root": str(source_state),
            "checkpoint": str(checkpoint),
            "checkpoint_step": source_step,
            "checkpoint_tree_sha256": checkpoint_tree_sha,
            "checkpoint_files": checkpoint_files,
            "validated_source_config": expected_source_config,
            "learner_msgpack_sha256": learner_msgpack_sha,
            "latest_replay": str(latest_replay),
            "latest_replay_sha256": actual_replay_sha,
            "latest_replay_episode_ids": replay_episode_ids,
            "latest_episode": latest_episode,
            "all_episode_ids": all_source_episode_ids,
            "post_checkpoint_or_untrained_episode_ids": post_checkpoint_episode_ids,
        },
        "target": {
            "session_root": str(target_session),
            "state_root": str(target_state),
            "episode_index_floor": latest_index + 1,
            "first_episode_id": f"episode_{latest_index + 1:06d}",
        },
        "migration": {
            "mode": "actor_only_one_way_warm_start",
            "status": "scheduled_deferred_until_first_30_episode_persistent_replay",
            "actor_params": "not_yet_migrated_at_fork",
            "target_actor": "not_yet_created_at_fork",
            "critic": "fresh_random_initialization",
            "target_critic": "fresh_from_new_critic",
            "actor_optimizer": "fresh",
            "critic_optimizer": "fresh",
            "update_step": 0,
            "rng": "fresh",
            "legacy_replay_training_rows": 0,
            "legacy_replay_policy": "immutable_provenance_only_never_merged",
            "candidate_action_normalization": (
                "deferred_refit_from_first_admitted_persistent_replay"
            ),
            "excluded_source_episode_ids": all_source_episode_ids,
            "explicitly_excluded_ep368_ep369": [
                item
                for item in ("episode_000368", "episode_000369")
                if item in all_source_episode_ids
            ],
        },
        "contract": {
            "actor_execution_profile": ACTOR_EXECUTION_PROFILE,
            "actor_model_action_schema_fingerprint": SOURCE_ACTION_SCHEMA,
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
            "replay_plan_contract": "complete_same_plan_c10_offsets_0_through_9",
            "min_new_persistent_committed_episodes": (
                args.min_new_persistent_episodes
            ),
        },
        "preflight_passed": True,
        "created": False,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _readonly_tree(path: Path) -> None:
    for item in sorted(path.rglob("*"), reverse=True):
        if item.is_file():
            item.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif item.is_dir():
            item.chmod(
                stat.S_IRUSR
                | stat.S_IWUSR
                | stat.S_IXUSR
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH
            )


def _config_env(
    report: dict[str, Any],
    *,
    state_root: Path,
    warm_start_checkpoint: Path,
) -> str:
    target = report["target"]
    contract = report["contract"]
    home = Path.home()
    values: dict[str, Any] = {
        "RLT_SESSION_ROOT": target["session_root"],
        "RLT_STATE_ROOT": str(state_root),
        "RLT_WORKSPACE": str(home / "openpi_jax_piper_lora_v1_20260707"),
        "RLT_RUNTIME": str(home / "piper_jax_inference_v1"),
        "RLT_PHASE_CHECKPOINT": str(
            home
            / "rlt_phase_classifiers/greenblock_box_resnet18_v4_manual_intervals/phase_classifier.pt"
        ),
        "RLT_SELECTED_ACTOR_FILE": str(
            home / "openpi_rlt/online_current/selected_actor_checkpoint.txt"
        ),
        "RLT_SHADOW_SERVICE": "openpi-rlt-shadow-policy.service",
        "RLT_POLICY_HOST": "127.0.0.1",
        "RLT_POLICY_PORT": 8001,
        "RLT_LINEAGE_MODE": "persistent_v2_actor_only_warm_start",
        "RLT_ACTOR_EXECUTION_PROFILE": contract["actor_execution_profile"],
        "RLT_ACTOR_MODEL_ACTION_SCHEMA_FINGERPRINT": contract[
            "actor_model_action_schema_fingerprint"
        ],
        "RLT_EXECUTION_ACTION_SCHEMA_FINGERPRINT": contract[
            "execution_action_schema_fingerprint"
        ],
        "RLT_ACTOR_PROJECTION_PROFILE": contract[
            "actor_projection_profile"
        ],
        "RLT_EXECUTION_FILTER_PROFILE": contract[
            "execution_filter_profile"
        ],
        "RLT_EXECUTION_FILTER_TAU_S": contract["execution_filter_tau_s"],
        "RLT_CONTROL_HZ": contract["control_hz"],
        "RLT_CONTROL_DT_S": contract["control_dt_s"],
        "RLT_EXECUTION_FILTER_ALPHA": contract["execution_filter_alpha"],
        "RLT_CHUNK_LENGTH": contract["chunk_length"],
        "RLT_CHUNK_STRIDE": contract["chunk_stride"],
        "RLT_ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD": contract[
            "actor_live_max_boundary_jump_rad"
        ],
        "RLT_ACTOR_PROJECTION_SCALE_STEPS": contract[
            "actor_projection_scale_steps"
        ],
        "RLT_ACTOR_MIN_PROJECTION_SCALE": contract[
            "actor_min_projection_scale"
        ],
        "RLT_ACTOR_DIRECTION_STATIC_THRESHOLD_RAD": contract[
            "actor_direction_static_threshold_rad"
        ],
        "RLT_ACTOR_GOVERNOR_FINGERPRINT": contract[
            "actor_governor_fingerprint"
        ],
        "RLT_WARMUP_EPISODES": contract[
            "min_new_persistent_committed_episodes"
        ],
        "RLT_MIN_NEW_PERSISTENT_COMMITTED_EPISODES": contract[
            "min_new_persistent_committed_episodes"
        ],
        "RLT_EPISODE_INDEX_FLOOR": target["episode_index_floor"],
        "RLT_WARM_START_ACTOR_CHECKPOINT": str(warm_start_checkpoint),
        "RLT_REPLAY_TRAINING_POLICY": "persistent_only_no_legacy_merge",
        "RLT_MIN_SUCCESS": 2,
        "RLT_MIN_FAILURE": 2,
        "RLT_MIN_SUCCESS_HUMAN_EPISODES": 1,
        "RLT_UPDATE_EVERY": 1,
        "RLT_MIN_WARMUP_TRANSITIONS": 30,
        "RLT_UTD": 1.0,
        "RLT_MIN_UPDATE_STEPS": 1,
        "RLT_MAX_UPDATE_STEPS": 1250,
        "RLT_BATCH_SIZE": 256,
        "RLT_BETA_BC": 40.0,
        "RLT_BETA_HUMAN_BC": 0.0,
        "RLT_REFERENCE_DROPOUT": 0.5,
        "RLT_TARGET_POLICY_NOISE_STD": 0.1,
        "RLT_TARGET_POLICY_NOISE_CLIP": 0.2,
        "RLT_ACTION_HORIZON": 50,
        "RLT_MODEL_EXECUTE_STEPS": 50,
        "RLT_REPLAY_STRIDE": 10,
        "RLT_N_STEP": 10,
        "RLT_RESIDUAL_MAX": 0.005,
        "RLT_RESIDUAL_D1_MAX_RAD": 0.0015,
        "RLT_RESIDUAL_D2_MAX_RAD": 0.001,
        "RLT_DIRECTION_CONE_DEG": 15.0,
        "RLT_GRIPPER_RESIDUAL_MAX": 0.0,
        "RLT_SUCCESS_FRACTION": 0.5,
        "RLT_HUMAN_FRACTION": 0.25,
        "RLT_VALIDATION_FRACTION": 0.15,
        "RLT_MAX_VALIDATION_TD_ERROR": 0.5,
        "RLT_MAX_ACTOR_Q_ADVANTAGE": 0.5,
        "RLT_MAX_ACTIVE_NORMALIZED_RESIDUAL_STEP": 0.30,
        "RLT_MAX_ACTOR_JOINT_D1_P95_RAD": 0.025,
        "RLT_MAX_CHUNK_BOUNDARY_NORMALIZED_RESIDUAL_JUMP_P95": 1.75,
        "RLT_BASE_FINGERPRINT": BASE_FINGERPRINT,
        "RLT_TOKEN_FINGERPRINT": TOKEN_FINGERPRINT,
        "RLT_PHASE_FINGERPRINT": PHASE_FINGERPRINT,
    }
    return "".join(
        f"{key}={shlex.quote(str(value))}\n" for key, value in values.items()
    )


def create_lineage(
    report: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    source = report["source"]
    target_session = Path(report["target"]["session_root"])
    target_state = Path(report["target"]["state_root"])
    temporary_session = target_session.with_name(
        f".{target_session.name}.creating-{os.getpid()}"
    )
    if temporary_session.exists():
        raise FileExistsError(f"temporary target already exists: {temporary_session}")
    temporary_state = temporary_session / target_state.name
    provenance = temporary_state / "provenance"
    source_checkpoint_copy = (
        provenance / f"source_actor_checkpoint_step_{source['checkpoint_step']:08d}"
    )
    source_replay_copy = provenance / "legacy_source_replay.npz"
    try:
        provenance.mkdir(parents=True)
        shutil.copytree(
            Path(source["checkpoint"]),
            source_checkpoint_copy,
            copy_function=shutil.copy2,
        )
        shutil.copy2(Path(source["latest_replay"]), source_replay_copy)
        if _sha256(source_replay_copy) != source["latest_replay_sha256"]:
            raise RuntimeError("copied provenance replay failed SHA verification")
        _, copied_checkpoint_tree_sha = _checkpoint_manifest(
            source_checkpoint_copy
        )
        if copied_checkpoint_tree_sha != source["checkpoint_tree_sha256"]:
            raise RuntimeError("copied source checkpoint failed SHA verification")
        _readonly_tree(provenance)

        final_state_root = target_session / target_state.name
        final_checkpoint = (
            final_state_root
            / "provenance"
            / source_checkpoint_copy.name
        )
        final_replay = final_state_root / "provenance" / source_replay_copy.name
        created_unix = time.time()
        manifest = {
            **report,
            "mode": "created",
            "created": True,
            "created_unix": created_unix,
            "target": {
                **report["target"],
                "source_actor_checkpoint_copy": str(final_checkpoint),
                "legacy_replay_provenance_copy": str(final_replay),
            },
        }
        state = {
            "format": STATE_FORMAT,
            "session_root": str(target_session),
            "lineage_mode": "persistent_v2_actor_only_warm_start",
            "fork_manifest": str(final_state_root / "fork_manifest.json"),
            "actor_execution_profile": ACTOR_EXECUTION_PROFILE,
            "actor_model_action_schema_fingerprint": SOURCE_ACTION_SCHEMA,
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
            "replay_training_policy": "persistent_only_no_legacy_merge",
            "legacy_source_replay": str(final_replay),
            "legacy_source_replay_sha256": source["latest_replay_sha256"],
            "legacy_source_replay_episode_ids": source[
                "latest_replay_episode_ids"
            ],
            "legacy_source_episode_ids_excluded_from_training": source[
                "all_episode_ids"
            ],
            "initial_actor_warm_start_checkpoint": str(final_checkpoint),
            "initial_actor_warm_start_checkpoint_tree_sha256": source[
                "checkpoint_tree_sha256"
            ],
            "initial_actor_warm_start_learner_msgpack_sha256": source[
                "learner_msgpack_sha256"
            ],
            "warm_start_policy": "actor_only_target_actor_equal_fresh_ac",
            "warm_start_status": (
                "scheduled_deferred_until_first_admitted_30_episode_replay"
            ),
            "candidate_action_normalization_status": (
                "deferred_refit_from_first_persistent_replay"
            ),
            "warm_start_consumed": False,
            "deployment_checkpoint": str(final_checkpoint),
            "latest_checkpoint": None,
            "latest_replay": None,
            "trained_episode_ids": [],
            "last_attempt_episode_ids": [],
            "last_update_episode_count": 0,
            "last_attempt_episode_count": 0,
            "last_train_transition_count": 0,
            "last_attempt_train_transition_count": 0,
            "attempt_index": 0,
            "update_index": 0,
            "episode_index_floor": report["target"]["episode_index_floor"],
            "min_new_persistent_committed_episodes": args.min_new_persistent_episodes,
            "created_unix": created_unix,
            "updated_unix": created_unix,
        }
        _atomic_json(temporary_state / "fork_manifest.json", manifest)
        _atomic_json(temporary_state / "online_state.json", state)
        (temporary_state / "config.env").write_text(
            _config_env(
                report,
                state_root=final_state_root,
                warm_start_checkpoint=final_checkpoint,
            ),
            encoding="utf-8",
        )
        temporary_session.rename(target_session)
    except Exception:
        if temporary_session.exists():
            shutil.rmtree(temporary_session)
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
