#!/usr/bin/env python3
"""Read-only, fail-closed validation of a close-assist v3 lineage."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
from pathlib import Path
from typing import Any

import numpy as np
from fork_close_assist_v3_lineage import (
    BOOTSTRAP_MIGRATION_FORMAT,
    BOOTSTRAP_TEACHER_POLICY,
    EXPECTED_BOOTSTRAP_EPISODES,
    EXPECTED_BOOTSTRAP_HUMAN_EPISODES,
    EXPECTED_BOOTSTRAP_REWARD_NEGATIVE_EPISODES,
    EXPECTED_BOOTSTRAP_REWARD_POSITIVE_EPISODES,
    EXPECTED_INITIAL_CHECKPOINT_STEP,
    EXPECTED_INITIAL_CRITIC_BURN_IN_STEPS,
    EXPECTED_REJECTED_STEP,
    EXPECTED_SOURCE_ACTOR_STEP,
    EXPECTED_SOURCE_BETAS,
    EXPECTED_TARGET_BETAS,
    FORK_FORMAT,
    HUMAN_GRIPPER_Q_FILTER_MARGIN,
    HUMAN_GRIPPER_Q_FILTER_MODE,
    LINEAGE_MODE,
    OBJECTIVE_MIGRATION_FORMAT,
    REPLAY_POLICY,
    SOURCE_ACTOR_SCHEMA,
    SOURCE_FROZEN_EXECUTION_SCHEMA,
    STATE_FORMAT,
    VALIDATION_FORMAT,
    _audit_initial_v3_checkpoint,
    _audit_source_actor,
    _canonical_episode_ids,
    _checkpoint_manifest,
    _contract_payload,
    _load_json,
    _manifest_file_sha,
    _replay_audit,
    _require_float,
    _sha256,
)
from persistent_v2_contract import (
    ACTION_SCHEMA_FINGERPRINT,
    ACTOR_DIRECTION_STATIC_THRESHOLD_RAD,
    ACTOR_EXECUTION_PROFILE,
    ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD,
    ACTOR_MIN_PROJECTION_SCALE,
    ACTOR_MODEL_ACTION_SCHEMA_FINGERPRINT,
    ACTOR_PROJECTION_PROFILE,
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

EPISODE_PATTERN = re.compile(r"episode_([0-9]+)")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        help="Defaults to STATE_ROOT/config.env.",
    )
    parser.add_argument(
        "--expected-episode-floor",
        type=int,
        help="Optional stale-lineage guard (the current intended fork is 408).",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        required=True,
        help="Mandatory acknowledgement: this tool never repairs state.",
    )
    return parser


def _parse_config(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: invalid config line")
        key, raw_value = line.split("=", 1)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError(f"{path}:{line_number}: invalid config key")
        values = shlex.split(raw_value, posix=True)
        if len(values) != 1:
            raise ValueError(
                f"{path}:{line_number}: config value must be scalar"
            )
        if key in result:
            raise ValueError(f"{path}:{line_number}: duplicate key {key}")
        result[key] = values[0]
    return result


def _exact_float(actual: Any, expected: float, label: str) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {actual!r}") from exc
    if not math.isfinite(value) or not math.isclose(
        value, expected, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(f"{label} mismatch: {value!r} != {expected!r}")


def _inside(path: Path, parent: Path, label: str) -> None:
    if path != parent and parent not in path.parents:
        raise ValueError(f"{label} is outside state root: {path}")


def _path_from_state(
    state: dict[str, Any], key: str, state_root: Path
) -> Path:
    raw = state.get(key)
    if not raw:
        raise ValueError(f"online state lacks {key}")
    path = Path(str(raw)).expanduser().resolve()
    _inside(path, state_root, key)
    return path


def _verify_tree(
    path: Path,
    expected_tree_sha256: str,
    *,
    label: str,
) -> tuple[list[dict[str, Any]], str]:
    files, tree_sha = _checkpoint_manifest(path)
    if tree_sha != expected_tree_sha256:
        raise ValueError(
            f"{label} tree SHA mismatch: "
            f"{tree_sha} != {expected_tree_sha256}"
        )
    return files, tree_sha


def _validate_bootstrap_report_copy(
    path: Path,
    *,
    expected_sha256: str,
    source_replay_path: Path,
    source_replay_sha256: str,
    original_bootstrap_path: Path,
    bootstrap_sha256: str,
    transitions: int,
    reward_positive_human_steps: int,
    reward_negative_human_steps: int,
) -> dict[str, Any]:
    if _sha256(path) != expected_sha256:
        raise ValueError("bootstrap migration report copy SHA mismatch")
    payload = _load_json(path)
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
        if payload.get(key) != expected:
            raise ValueError(
                f"bootstrap report {key} mismatch: "
                f"{payload.get(key)!r} != {expected!r}"
            )
    if Path(str(payload.get("source_replay", ""))).expanduser().resolve() != (
        source_replay_path
    ):
        raise ValueError("bootstrap report source path mismatch")
    if payload.get("source_sha256") != source_replay_sha256:
        raise ValueError("bootstrap report source SHA mismatch")
    if Path(str(payload.get("output_replay", ""))).expanduser().resolve() != (
        original_bootstrap_path
    ):
        raise ValueError("bootstrap report original output path mismatch")
    if payload.get("output_sha256") != bootstrap_sha256:
        raise ValueError("bootstrap report output SHA mismatch")
    expected_integers = {
        "transitions": transitions,
        "episodes": EXPECTED_BOOTSTRAP_EPISODES,
        "successful_episodes": EXPECTED_BOOTSTRAP_REWARD_POSITIVE_EPISODES,
        "reward1_human_steps": reward_positive_human_steps,
        "reward0_human_steps": reward_negative_human_steps,
    }
    for key, expected in expected_integers.items():
        try:
            actual = int(payload[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"bootstrap report lacks valid {key}") from exc
        if actual != expected:
            raise ValueError(
                f"bootstrap report {key} mismatch: {actual} != {expected}"
            )
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("bootstrap report lacks gripper contract")
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
            contract, key, expected, label="bootstrap report contract"
        )
    return payload


def _validate_v5_replay_generic(
    path: Path,
    *,
    expected_sha256: str,
    expected_episode_ids: list[str],
) -> dict[str, Any]:
    actual_sha = _sha256(path)
    if actual_sha != expected_sha256:
        raise ValueError(f"v5 replay SHA mismatch: {path}")
    with np.load(path, allow_pickle=False) as replay:
        required = {
            "episode_id",
            "episode_split",
            "action_schema_fingerprint",
            "gripper_residual_mode",
            "actor_governor_fingerprint",
        }
        missing = sorted(required.difference(replay.files))
        if missing:
            raise ValueError(
                f"v5 replay lacks arrays {missing}: {path}"
            )
        episode_id = np.asarray(replay["episode_id"]).astype(str)
        episode_split = np.asarray(replay["episode_split"]).astype(str)
        if episode_id.ndim != 1 or episode_split.shape != episode_id.shape:
            raise ValueError(
                f"v5 replay episode_id/episode_split shape mismatch: {path}"
            )
        episode_ids = sorted(set(episode_id.tolist()))
        if episode_ids != expected_episode_ids:
            raise ValueError(
                f"v5 replay/state episode IDs mismatch: {path}"
            )
        unknown_splits = sorted(
            set(episode_split.tolist()).difference(
                {"train", "validation", "test"}
            )
        )
        if unknown_splits:
            raise ValueError(
                "v5 replay contains unknown episode_split labels "
                f"{unknown_splits}: {path}"
            )
        leaking_episode_ids = sorted(
            episode
            for episode in episode_ids
            if len(set(episode_split[episode_id == episode].tolist())) != 1
        )
        if leaking_episode_ids:
            raise ValueError(
                "v5 replay episodes span multiple splits "
                f"{leaking_episode_ids}: {path}"
            )
        n = len(episode_id)
        schema = set(
            np.asarray(replay["action_schema_fingerprint"])
            .astype(str)
            .tolist()
        )
        modes = set(
            np.asarray(replay["gripper_residual_mode"]).astype(str).tolist()
        )
        governors = set(
            np.asarray(replay["actor_governor_fingerprint"])
            .astype(str)
            .tolist()
        )
        if schema != {ACTION_SCHEMA_FINGERPRINT}:
            raise ValueError(f"old/non-v5 schema mixed into replay: {path}")
        if modes != {GRIPPER_RESIDUAL_MODE}:
            raise ValueError(f"wrong gripper mode mixed into replay: {path}")
        if governors != {PERSISTENT_GOVERNOR_FINGERPRINT}:
            raise ValueError(f"wrong governor mixed into replay: {path}")
    return {
        "sha256": actual_sha,
        "transitions": n,
        "episode_ids": episode_ids,
        "split_counts": {
            label: int(np.count_nonzero(episode_split == label))
            for label in ("train", "validation", "test")
        },
    }


def _validate_checkpoint_v3(
    checkpoint: Path,
    *,
    replay_sha256: str,
    target_betas: dict[str, float],
    rejected_tree_sha256: str,
    rejected_learner_sha256: str,
    require_warm_start_lineage: bool = True,
) -> dict[str, Any]:
    files, tree_sha = _checkpoint_manifest(checkpoint)
    learner_sha = _manifest_file_sha(files, "learner.msgpack")
    if (
        tree_sha == rejected_tree_sha256
        or learner_sha == rejected_learner_sha256
    ):
        raise ValueError(
            "rejected frozen-v2 step 144 is selected as a v3 checkpoint"
        )
    metadata = _load_json(checkpoint / "metadata.json")
    fingerprints = metadata.get("fingerprints")
    config = metadata.get("config")
    if (
        metadata.get("format") != "openpi_real_rlt_jax_learner"
        or not isinstance(fingerprints, dict)
        or not isinstance(config, dict)
    ):
        raise ValueError(f"invalid v3 learner metadata: {checkpoint}")
    expected_fingerprints = {
        "replay_sha256": replay_sha256,
        "action_schema": ACTION_SCHEMA_FINGERPRINT,
        "actor_execution_profile": ACTOR_EXECUTION_PROFILE,
        "execution_filter_profile": EXECUTION_FILTER_PROFILE,
        "actor_governor": PERSISTENT_GOVERNOR_FINGERPRINT,
    }
    if require_warm_start_lineage:
        expected_fingerprints[
            "warm_start_actor_source_schema"
        ] = SOURCE_ACTOR_SCHEMA
    for key, expected in expected_fingerprints.items():
        if fingerprints.get(key) != expected:
            raise ValueError(
                f"v3 checkpoint {key} mismatch: "
                f"{fingerprints.get(key)!r} != {expected!r}"
            )
    exact_config: dict[str, Any] = {
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
    for key, expected in exact_config.items():
        actual = config.get(key)
        if isinstance(expected, float):
            _exact_float(actual, expected, f"v3 checkpoint config {key}")
        elif actual != expected:
            raise ValueError(
                f"v3 checkpoint config {key} mismatch: "
                f"{actual!r} != {expected!r}"
            )
    for key, expected in target_betas.items():
        _exact_float(
            config.get(key), expected, f"v3 checkpoint config {key}"
        )
    if int(metadata["update_step"]) < EXPECTED_INITIAL_CHECKPOINT_STEP:
        raise ValueError(
            "v3 checkpoint predates the required Critic burn-in schedule"
        )
    return {
        "path": str(checkpoint),
        "step": int(metadata["update_step"]),
        "tree_sha256": tree_sha,
        "learner_msgpack_sha256": learner_sha,
    }


def _validate_objective_migration(
    *,
    state: dict[str, Any],
    state_root: Path,
    source_actor: Path,
    initial_checkpoint: Path,
    source_betas: dict[str, float],
    target_betas: dict[str, float],
    bootstrap_replay: Path,
    bootstrap_sha256: str,
) -> dict[str, Any]:
    migration = state.get("objective_migration")
    if not isinstance(migration, dict):
        raise ValueError("online state lacks objective_migration")
    exact = {
        "format": OBJECTIVE_MIGRATION_FORMAT,
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
        "target_human_gripper_q_filter_mode": HUMAN_GRIPPER_Q_FILTER_MODE,
        "target_human_gripper_q_filter_margin": (
            HUMAN_GRIPPER_Q_FILTER_MARGIN
        ),
        "human_supervision_scope": (
            "all_admitted_human_reward_positive_and_reward_negative"
        ),
        "physical_governor_changed": True,
    }
    for key, expected in exact.items():
        actual = migration.get(key)
        if isinstance(expected, float):
            _exact_float(actual, expected, f"objective migration {key}")
        elif actual != expected:
            raise ValueError(
                f"objective migration {key} mismatch: "
                f"{actual!r} != {expected!r}"
            )
    authorization = migration.get("authorization")
    if not isinstance(authorization, str) or not authorization:
        raise ValueError("objective migration lacks authorization")
    manifest_path = _path_from_state(
        state, "objective_migration_manifest", state_root
    )
    expected_manifest_sha = state.get(
        "objective_migration_manifest_sha256"
    )
    if not expected_manifest_sha or _sha256(
        manifest_path
    ) != expected_manifest_sha:
        raise ValueError("objective migration manifest SHA mismatch")
    manifest = _load_json(manifest_path)
    if (
        manifest.get("format") != OBJECTIVE_MIGRATION_FORMAT
        or manifest.get("authorization") != authorization
        or manifest.get("physical_governor_changed") is not True
        or manifest.get("human_supervision_scope")
        != "all_admitted_human_reward_positive_and_reward_negative"
    ):
        raise ValueError("objective migration manifest header mismatch")
    source = manifest.get("source")
    target = manifest.get("target")
    if not isinstance(source, dict) or not isinstance(target, dict):
        raise ValueError("objective migration manifest lacks source/target")
    if Path(str(source.get("checkpoint", ""))).resolve() != source_actor:
        raise ValueError("objective migration source checkpoint mismatch")
    if Path(str(target.get("checkpoint", ""))).resolve() != (
        initial_checkpoint
    ):
        raise ValueError("objective migration target checkpoint mismatch")
    for prefix, values, payload in (
        ("source", source_betas, source),
        ("target", target_betas, target),
    ):
        for key, expected in values.items():
            _exact_float(
                payload.get(key),
                expected,
                f"objective manifest {prefix}.{key}",
            )
    if target.get("human_gripper_q_filter_mode") != (
        HUMAN_GRIPPER_Q_FILTER_MODE
    ):
        raise ValueError("objective manifest target Q-filter mode mismatch")
    _exact_float(
        target.get("human_gripper_q_filter_margin"),
        HUMAN_GRIPPER_Q_FILTER_MARGIN,
        "objective manifest target Q-filter margin",
    )
    if Path(str(manifest.get("bootstrap_replay", ""))).resolve() != (
        bootstrap_replay
    ):
        raise ValueError("objective manifest bootstrap path mismatch")
    if manifest.get("bootstrap_replay_sha256") != bootstrap_sha256:
        raise ValueError("objective manifest bootstrap SHA mismatch")
    return {
        "path": str(manifest_path),
        "sha256": expected_manifest_sha,
        "authorization": authorization,
    }


def _validate_config(
    *,
    config_path: Path,
    session: Path,
    state_root: Path,
    state: dict[str, Any],
    source_actor: Path,
    target_betas: dict[str, float],
) -> dict[str, str]:
    config = _parse_config(config_path)
    if Path(config.get("RLT_SESSION_ROOT", "")).expanduser().resolve() != session:
        raise ValueError("config/session binding mismatch")
    if Path(config.get("RLT_STATE_ROOT", "")).expanduser().resolve() != state_root:
        raise ValueError("config/state binding mismatch")
    if Path(
        config.get("RLT_WARM_START_ACTOR_CHECKPOINT", "")
    ).expanduser().resolve() != source_actor:
        raise ValueError("config/source Actor binding mismatch")
    workspace = Path(str(state.get("workspace", ""))).expanduser().resolve()
    runtime = Path(str(state.get("runtime", ""))).expanduser().resolve()
    selected_actor_file = Path(
        str(state.get("selected_checkpoint_file", ""))
    ).expanduser().resolve()
    if Path(config.get("RLT_WORKSPACE", "")).expanduser().resolve() != workspace:
        raise ValueError("config/workspace binding mismatch")
    if Path(config.get("RLT_RUNTIME", "")).expanduser().resolve() != runtime:
        raise ValueError("config/runtime binding mismatch")
    if (
        Path(config.get("RLT_SELECTED_ACTOR_FILE", "")).expanduser().resolve()
        != selected_actor_file
    ):
        raise ValueError("config/selected Actor binding mismatch")
    exact_strings = {
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
        "RLT_ACTOR_GOVERNOR_FINGERPRINT": (
            PERSISTENT_GOVERNOR_FINGERPRINT
        ),
        "RLT_GRIPPER_RESIDUAL_MODE": GRIPPER_RESIDUAL_MODE,
        "RLT_CHUNK_LENGTH": str(CHUNK_LENGTH),
        "RLT_CHUNK_STRIDE": str(CHUNK_STRIDE),
        "RLT_REPLAY_STRIDE": str(CHUNK_STRIDE),
        "RLT_FREEZE_GRIPPER_RESIDUAL": "0",
        "RLT_ALLOW_WARM_START_OBJECTIVE_MIGRATION": "1",
        "RLT_MIN_NEW_PERSISTENT_COMMITTED_EPISODES": "0",
        "RLT_SHADOW_SERVICE": (
            "openpi-rlt-shadow-policy-gripper-v3.service"
        ),
    }
    for key, expected in exact_strings.items():
        if config.get(key) != expected:
            raise ValueError(
                f"config {key} mismatch: "
                f"{config.get(key)!r} != {expected!r}"
            )
    exact_floats = {
        "RLT_EXECUTION_FILTER_TAU_S": EXECUTION_FILTER_TAU_S,
        "RLT_CONTROL_HZ": CONTROL_HZ,
        "RLT_CONTROL_DT_S": CONTROL_DT_S,
        "RLT_EXECUTION_FILTER_ALPHA": EXECUTION_FILTER_ALPHA,
        "RLT_ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD": (
            ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD
        ),
        "RLT_ACTOR_MIN_PROJECTION_SCALE": ACTOR_MIN_PROJECTION_SCALE,
        "RLT_ACTOR_DIRECTION_STATIC_THRESHOLD_RAD": (
            ACTOR_DIRECTION_STATIC_THRESHOLD_RAD
        ),
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
        "RLT_HUMAN_GRIPPER_BC_SCALE_M": (
            GRIPPER_RESIDUAL_MAX_CLOSE_M
        ),
        "RLT_BETA_BC": target_betas["beta_bc"],
        "RLT_BETA_HUMAN_BC": target_betas["beta_human_bc"],
        "RLT_BETA_HUMAN_GRIPPER_BC": target_betas[
            "beta_human_gripper_bc"
        ],
        "RLT_HUMAN_GRIPPER_Q_FILTER_MARGIN": (
            HUMAN_GRIPPER_Q_FILTER_MARGIN
        ),
    }
    for key, expected in exact_floats.items():
        _exact_float(config.get(key), expected, f"config {key}")
    if config.get("RLT_HUMAN_GRIPPER_Q_FILTER_MODE") != (
        HUMAN_GRIPPER_Q_FILTER_MODE
    ):
        raise ValueError("config human-gripper Q-filter mode mismatch")
    if int(config.get("RLT_MIN_ADMITTED_HUMAN_EPISODES", -1)) != 1:
        raise ValueError("config admitted-human warmup guard mismatch")
    if int(config.get("RLT_MIN_SUCCESS_HUMAN_EPISODES", -1)) != 0:
        raise ValueError(
            "config must not gate admitted-human data on reward-positive labels"
        )
    if int(config.get("RLT_EPISODE_INDEX_FLOOR", -1)) != int(
        state["episode_index_floor"]
    ):
        raise ValueError("config/state episode floor mismatch")
    if int(config.get("RLT_WARMUP_EPISODES", -1)) != (
        EXPECTED_BOOTSTRAP_EPISODES
    ):
        raise ValueError("config warmup does not describe the 30 bootstrap episodes")
    return config


def _validate_no_old_v2_mix(
    *,
    state_root: Path,
    legacy_replay: Path,
) -> list[str]:
    audited: list[str] = []
    for replay_path in sorted(state_root.rglob("*.npz")):
        replay_path = replay_path.resolve()
        if replay_path == legacy_replay:
            audited.append(
                f"{replay_path}:legacy_provenance_only_training_rows_zero"
            )
            continue
        with np.load(replay_path, allow_pickle=False) as replay:
            if "action_schema_fingerprint" not in replay.files:
                raise ValueError(
                    f"non-provenance replay lacks schema: {replay_path}"
                )
            schemas = set(
                np.asarray(replay["action_schema_fingerprint"])
                .astype(str)
                .tolist()
            )
        if schemas != {ACTION_SCHEMA_FINGERPRINT}:
            raise ValueError(
                f"old v2 schema mixed outside provenance: "
                f"{replay_path} -> {schemas}"
            )
        audited.append(f"{replay_path}:v5_only")
    return audited


def validate(
    session_root: Path,
    state_root: Path,
    config_path: Path | None = None,
    *,
    expected_episode_floor: int | None = None,
) -> dict[str, Any]:
    session = session_root.expanduser().resolve()
    state_dir = state_root.expanduser().resolve()
    if state_dir.parent != session:
        raise ValueError("state root must be one direct child of session")
    config_file = (
        (state_dir / "config.env")
        if config_path is None
        else config_path.expanduser().resolve()
    )
    state_path = state_dir / "online_state.json"
    state = _load_json(state_path)
    if Path(str(state.get("session_root", ""))).expanduser().resolve() != session:
        raise ValueError("online state/session binding mismatch")

    manifest_path = _path_from_state(state, "fork_manifest", state_dir)
    manifest_sha = _sha256(manifest_path)
    if manifest_sha != state.get("fork_manifest_sha256"):
        raise ValueError("fork manifest SHA mismatch")
    manifest = _load_json(manifest_path)
    if manifest.get("format") != FORK_FORMAT or manifest.get("created") is not True:
        raise ValueError("unsupported or incomplete fork manifest")
    if manifest.get("preflight_passed") is not True:
        raise ValueError("fork manifest did not pass preflight")
    if manifest.get("contract") != _contract_payload():
        raise ValueError("fork manifest v3 contract mismatch")
    target_manifest = manifest.get("target")
    source_manifest = manifest.get("source")
    bootstrap_manifest = manifest.get("bootstrap")
    initial_manifest = manifest.get("initial_v3_checkpoint")
    if not all(
        isinstance(item, dict)
        for item in (
            target_manifest,
            source_manifest,
            bootstrap_manifest,
            initial_manifest,
        )
    ):
        raise ValueError("fork manifest lacks source/bootstrap/target sections")
    if Path(str(target_manifest["session_root"])).resolve() != session:
        raise ValueError("fork manifest/session binding mismatch")
    if Path(str(target_manifest["state_root"])).resolve() != state_dir:
        raise ValueError("fork manifest/state binding mismatch")
    workspace = Path(str(state.get("workspace", ""))).expanduser().resolve()
    runtime = Path(str(state.get("runtime", ""))).expanduser().resolve()
    selected_actor_file = Path(
        str(state.get("selected_checkpoint_file", ""))
    ).expanduser().resolve()
    shadow_service = str(state.get("shadow_service", ""))
    if Path(str(target_manifest.get("workspace", ""))).resolve() != workspace:
        raise ValueError("fork manifest/workspace binding mismatch")
    if Path(str(target_manifest.get("runtime", ""))).resolve() != runtime:
        raise ValueError("fork manifest/runtime binding mismatch")
    if (
        Path(str(target_manifest.get("selected_actor_file", ""))).resolve()
        != selected_actor_file
    ):
        raise ValueError("fork manifest/selected Actor binding mismatch")
    if target_manifest.get("shadow_service") != shadow_service:
        raise ValueError("fork manifest/shadow service binding mismatch")
    if not (workspace / ".venv/bin/python").is_file() or not (
        workspace / "src/openpi"
    ).is_dir():
        raise ValueError("bound gripper-v3 workspace is incomplete")
    if not (runtime / "piper_runtime").is_dir():
        raise ValueError("bound gripper-v3 runtime is incomplete")
    if selected_actor_file != state_dir / "selected_actor_checkpoint.txt":
        raise ValueError("selected Actor file must remain inside the v3 state root")
    if shadow_service != "openpi-rlt-shadow-policy-gripper-v3.service":
        raise ValueError("v3 lineage is bound to the wrong shadow service")

    contract = _contract_payload()
    exact_state = {
        "format": STATE_FORMAT,
        "lineage_mode": LINEAGE_MODE,
        **{
            key: value
            for key, value in contract.items()
            if key not in {"state_format", "lineage_mode"}
        },
    }
    for key, expected in exact_state.items():
        actual = state.get(key)
        if isinstance(expected, float):
            _exact_float(actual, expected, f"state {key}")
        elif actual != expected:
            raise ValueError(
                f"state {key} mismatch: {actual!r} != {expected!r}"
            )
    if state.get("replay_training_policy") != REPLAY_POLICY:
        raise ValueError("state replay training policy mismatch")
    if state.get("frozen_base_replay") is not None or state.get(
        "frozen_base_episode_ids"
    ):
        raise ValueError("legacy frozen-base merge fields are forbidden")
    if int(state.get("legacy_replay_training_rows", -1)) != 0:
        raise ValueError("legacy v2 replay has nonzero target training rows")
    if state.get("legacy_replay_policy") != (
        "immutable_provenance_only_never_merged"
    ):
        raise ValueError("legacy replay policy is not provenance-only")

    floor = int(state.get("episode_index_floor", -1))
    manifest_floor = int(target_manifest["episode_index_floor"])
    source_latest_index = int(
        source_manifest["latest_complete_episode_index"]
    )
    source_highest_directory_index = int(
        source_manifest["highest_source_episode_directory_index"]
    )
    source_highest_directory = str(
        source_manifest["highest_source_episode_directory"]
    )
    if source_highest_directory != (
        f"episode_{source_highest_directory_index:06d}"
    ):
        raise ValueError(
            "fork manifest highest source episode directory/index mismatch"
        )
    if source_latest_index > source_highest_directory_index:
        raise ValueError(
            "source latest-complete episode is newer than its highest directory"
        )
    if (
        floor != manifest_floor
        or floor != source_highest_directory_index + 1
    ):
        raise ValueError(
            "episode floor is not highest source episode directory + 1"
        )
    if target_manifest.get("first_episode_id") != f"episode_{floor:06d}":
        raise ValueError("fork manifest first episode/floor mismatch")
    if expected_episode_floor is not None and floor != expected_episode_floor:
        raise ValueError(
            f"episode floor stale guard: {floor} != {expected_episode_floor}"
        )

    source_actor = _path_from_state(
        state, "initial_actor_warm_start_checkpoint", state_dir
    )
    source_actor_audit = _audit_source_actor(source_actor)
    if (
        source_actor_audit["tree_sha256"]
        != state["initial_actor_warm_start_checkpoint_tree_sha256"]
        or source_actor_audit["learner_msgpack_sha256"]
        != state["initial_actor_warm_start_learner_msgpack_sha256"]
    ):
        raise ValueError("copied source Actor hashes differ from online state")
    if (
        source_actor_audit["tree_sha256"]
        != source_manifest["actor"]["tree_sha256"]
    ):
        raise ValueError("copied source Actor differs from fork manifest")
    if int(state.get("source_actor_checkpoint_step", -1)) != (
        EXPECTED_SOURCE_ACTOR_STEP
    ):
        raise ValueError("online state source Actor is not step 12627")

    rejected = source_manifest.get("rejected_candidate")
    if not isinstance(rejected, dict):
        raise ValueError("fork manifest lacks rejected checkpoint audit")
    if (
        int(rejected.get("step", -1)) != EXPECTED_REJECTED_STEP
        or int(state.get("source_rejected_checkpoint_step", -1))
        != EXPECTED_REJECTED_STEP
        or rejected.get("action_schema")
        != SOURCE_FROZEN_EXECUTION_SCHEMA
        or state.get("source_rejected_checkpoint_disposition")
        != "rejected_never_warm_start_never_deploy"
    ):
        raise ValueError("rejected frozen-v2 step-144 exclusion mismatch")
    if (
        state.get("source_rejected_checkpoint_tree_sha256")
        != rejected.get("tree_sha256")
        or state.get(
            "source_rejected_checkpoint_learner_msgpack_sha256"
        )
        != rejected.get("learner_msgpack_sha256")
    ):
        raise ValueError("rejected checkpoint hash evidence mismatch")
    source_rejected_path = Path(
        str(rejected["checkpoint"])
    ).expanduser().resolve()
    rejected_files, rejected_tree = _checkpoint_manifest(
        source_rejected_path
    )
    if (
        rejected_tree != rejected["tree_sha256"]
        or _manifest_file_sha(rejected_files, "learner.msgpack")
        != rejected["learner_msgpack_sha256"]
    ):
        raise ValueError("source rejected checkpoint changed since fork")

    bootstrap_ids = _canonical_episode_ids(
        state.get("bootstrap_gripper_episode_ids"),
        "state bootstrap_gripper_episode_ids",
    )
    manifest_bootstrap_ids = _canonical_episode_ids(
        bootstrap_manifest.get("episode_ids"),
        "manifest bootstrap episode_ids",
    )
    if bootstrap_ids != manifest_bootstrap_ids:
        raise ValueError("state/manifest bootstrap episode IDs mismatch")
    bootstrap_path = _path_from_state(
        state, "bootstrap_gripper_replay", state_dir
    )
    if Path(str(target_manifest["bootstrap_gripper_replay_copy"])).resolve() != (
        bootstrap_path
    ):
        raise ValueError("fork manifest/bootstrap copy path mismatch")
    bootstrap = _replay_audit(
        bootstrap_path, expected_episode_ids=bootstrap_ids
    )
    if (
        bootstrap["sha256"]
        != state.get("bootstrap_gripper_replay_sha256")
        or bootstrap["sha256"] != bootstrap_manifest["sha256"]
    ):
        raise ValueError("bootstrap replay SHA evidence mismatch")
    if int(state.get("bootstrap_gripper_episode_count", -1)) != len(
        bootstrap_ids
    ):
        raise ValueError("bootstrap episode count mismatch")
    if int(
        state.get("bootstrap_gripper_train_transition_count", -1)
    ) != int(bootstrap["train_transitions"]):
        raise ValueError("bootstrap train transition count mismatch")
    expected_quality = {
        "episodes": EXPECTED_BOOTSTRAP_EPISODES,
        "reward_positive_episodes": EXPECTED_BOOTSTRAP_REWARD_POSITIVE_EPISODES,
        "reward_negative_episodes": EXPECTED_BOOTSTRAP_REWARD_NEGATIVE_EPISODES,
        "admitted_human_episodes": EXPECTED_BOOTSTRAP_HUMAN_EPISODES,
    }
    if state.get("bootstrap_gripper_quality") != expected_quality:
        raise ValueError(
            "state bootstrap quality is not admitted "
            "30/R+24/R-6/H29"
        )
    source_attempt_ids = _canonical_episode_ids(
        source_manifest.get("last_attempt_episode_ids"),
        "manifest source last_attempt_episode_ids",
    )
    if source_attempt_ids != bootstrap_ids:
        raise ValueError(
            "bootstrap IDs are not the exact source last-attempt IDs"
        )

    bootstrap_report_path = _path_from_state(
        state, "bootstrap_gripper_migration_report", state_dir
    )
    source_rejected_replay = Path(
        source_manifest["rejected_replay"]["path"]
    ).expanduser().resolve()
    _validate_bootstrap_report_copy(
        bootstrap_report_path,
        expected_sha256=state[
            "bootstrap_gripper_migration_report_sha256"
        ],
        source_replay_path=source_rejected_replay,
        source_replay_sha256=source_manifest["rejected_replay"]["sha256"],
        original_bootstrap_path=Path(
            bootstrap_manifest["path"]
        ).expanduser().resolve(),
        bootstrap_sha256=bootstrap["sha256"],
        transitions=bootstrap["transitions"],
        reward_positive_human_steps=bootstrap[
            "reward_positive_human_steps"
        ],
        reward_negative_human_steps=bootstrap[
            "reward_negative_human_steps"
        ],
    )

    legacy_path = _path_from_state(
        state, "legacy_source_replay", state_dir
    )
    if _sha256(legacy_path) != state.get("legacy_source_replay_sha256"):
        raise ValueError("legacy replay provenance SHA mismatch")
    if Path(str(target_manifest["legacy_replay_provenance_copy"])).resolve() != (
        legacy_path
    ):
        raise ValueError("legacy replay provenance path mismatch")
    with np.load(legacy_path, allow_pickle=False) as legacy:
        if "episode_id" not in legacy.files:
            raise ValueError("legacy provenance replay lacks episode_id")
        legacy_ids = sorted(
            set(np.asarray(legacy["episode_id"]).astype(str).tolist())
        )
    if legacy_ids != sorted(state["legacy_source_replay_episode_ids"]):
        raise ValueError("legacy provenance episode IDs mismatch")

    initial_checkpoint = _path_from_state(
        state, "initial_v3_checkpoint", state_dir
    )
    if Path(str(target_manifest["initial_v3_checkpoint_copy"])).resolve() != (
        initial_checkpoint
    ):
        raise ValueError("initial v3 checkpoint copy path mismatch")
    initial_audit = _audit_initial_v3_checkpoint(
        initial_checkpoint,
        bootstrap_replay_sha256=bootstrap["sha256"],
        source_actor=source_actor_audit,
        rejected=rejected,
        expected_step=int(state["initial_v3_checkpoint_step"]),
    )
    if (
        initial_audit["tree_sha256"]
        != state["initial_v3_checkpoint_tree_sha256"]
        or initial_audit["tree_sha256"]
        != initial_manifest["tree_sha256"]
        or initial_audit["learner_msgpack_sha256"]
        != state["initial_v3_checkpoint_learner_msgpack_sha256"]
    ):
        raise ValueError("initial v3 checkpoint hash evidence mismatch")
    target_betas = initial_audit["objective_weights"]
    for key, expected in EXPECTED_SOURCE_BETAS.items():
        _exact_float(
            source_actor_audit["objective_weights"].get(key),
            expected,
            f"source Actor objective {key}",
        )
    for key, expected in EXPECTED_TARGET_BETAS.items():
        _exact_float(
            target_betas.get(key),
            expected,
            f"initial v3 objective {key}",
        )
    initial_validation_path = _path_from_state(
        state, "initial_v3_validation_report", state_dir
    )
    if Path(
        str(target_manifest["initial_v3_validation_report_copy"])
    ).resolve() != initial_validation_path:
        raise ValueError("initial v3 validation report path mismatch")
    if (
        _sha256(initial_validation_path)
        != state.get("initial_v3_validation_report_sha256")
    ):
        raise ValueError("initial v3 validation report SHA mismatch")
    initial_validation_manifest = manifest.get("initial_v3_validation")
    if not isinstance(initial_validation_manifest, dict):
        raise ValueError("fork manifest lacks initial v3 validation audit")
    if (
        state.get("initial_v3_validation_passed") is not True
        or int(state.get("initial_v3_validation_samples", 0)) <= 0
    ):
        raise ValueError("initial v3 checkpoint lacks accepted validation state")
    original_initial_path = Path(
        str(initial_manifest["checkpoint"])
    ).expanduser().resolve()
    original_bootstrap_path = Path(
        str(bootstrap_manifest["path"])
    ).expanduser().resolve()
    initial_validation_payload = _load_json(initial_validation_path)
    if Path(
        str(initial_validation_payload.get("checkpoint", ""))
    ).expanduser().resolve() != original_initial_path:
        raise ValueError(
            "copied initial validation is not bound to original checkpoint"
        )
    if Path(
        str(initial_validation_payload.get("replay", ""))
    ).expanduser().resolve() != original_bootstrap_path:
        raise ValueError(
            "copied initial validation is not bound to original bootstrap"
        )
    if (
        initial_validation_payload.get("format")
        != "openpi_real_rlt_actor_acceptance"
        or initial_validation_payload.get("passed") is not True
        or int(initial_validation_payload.get("update_step", -1))
        != initial_audit["step"]
        or initial_validation_payload.get("replay_sha256")
        != bootstrap["sha256"]
        or initial_validation_payload.get("replay_split") != "validation"
        or int(initial_validation_payload.get("samples", 0)) <= 0
    ):
        raise ValueError("initial v3 acceptance report contract mismatch")
    validation_fingerprints = initial_validation_payload.get("fingerprints")
    if not isinstance(validation_fingerprints, dict):
        raise ValueError("initial v3 acceptance report lacks fingerprints")
    for key, expected in (
        ("replay_sha256", bootstrap["sha256"]),
        ("action_schema", ACTION_SCHEMA_FINGERPRINT),
        ("actor_execution_profile", ACTOR_EXECUTION_PROFILE),
        ("execution_filter_profile", EXECUTION_FILTER_PROFILE),
        ("actor_governor", PERSISTENT_GOVERNOR_FINGERPRINT),
        ("warm_start_actor_source_schema", SOURCE_ACTOR_SCHEMA),
    ):
        if validation_fingerprints.get(key) != expected:
            raise ValueError(
                f"initial v3 acceptance fingerprint {key} mismatch"
            )
    migration_audit = _validate_objective_migration(
        state=state,
        state_root=state_dir,
        source_actor=source_actor,
        initial_checkpoint=initial_checkpoint,
        source_betas=source_actor_audit["objective_weights"],
        target_betas=target_betas,
        bootstrap_replay=bootstrap_path,
        bootstrap_sha256=bootstrap["sha256"],
    )

    trained_ids = _canonical_episode_ids(
        state.get("trained_episode_ids"), "state trained_episode_ids"
    )
    attempt_ids = _canonical_episode_ids(
        state.get("last_attempt_episode_ids"),
        "state last_attempt_episode_ids",
    )
    if not set(bootstrap_ids).issubset(trained_ids):
        raise ValueError("trained episode IDs lost bootstrap provenance")
    if not set(bootstrap_ids).issubset(attempt_ids):
        raise ValueError("last-attempt episode IDs lost bootstrap provenance")
    if not set(trained_ids).issubset(attempt_ids):
        raise ValueError(
            "last-attempt episode IDs do not contain every trained episode"
        )
    for label, ids in (
        ("trained", trained_ids),
        ("last-attempt", attempt_ids),
    ):
        for episode_id in sorted(set(ids).difference(bootstrap_ids)):
            index = int(EPISODE_PATTERN.fullmatch(episode_id).group(1))
            if index < floor:
                raise ValueError(
                    f"old v2 episode contaminated {label} IDs: {episode_id}"
                )
    update_index = int(state.get("update_index", -1))
    attempt_index = int(state.get("attempt_index", -1))
    initial_update_index = int(target_manifest["initial_update_index"])
    initial_attempt_index = int(target_manifest["initial_attempt_index"])
    if update_index < initial_update_index:
        raise ValueError("v3 state update_index regressed below its fork")
    if attempt_index < update_index or attempt_index < initial_attempt_index:
        raise ValueError("v3 state attempt/update indices are inconsistent")
    if int(state.get("last_update_episode_count", -1)) != len(trained_ids):
        raise ValueError("v3 state last_update_episode_count mismatch")
    if int(state.get("last_attempt_episode_count", -1)) != len(attempt_ids):
        raise ValueError("v3 state last_attempt_episode_count mismatch")
    last_train_transition_count = int(
        state.get("last_train_transition_count", -1)
    )
    last_attempt_train_transition_count = int(
        state.get("last_attempt_train_transition_count", -1)
    )
    if (
        last_train_transition_count < 0
        or last_attempt_train_transition_count < last_train_transition_count
    ):
        raise ValueError("v3 state train-transition counters are inconsistent")

    initial_state = (
        update_index == initial_update_index
        and attempt_index == initial_attempt_index
        and Path(str(state.get("latest_checkpoint", ""))).resolve()
        == initial_checkpoint
    )
    if initial_state:
        if trained_ids != bootstrap_ids or attempt_ids != bootstrap_ids:
            raise ValueError(
                "initial v3 state IDs are not exactly bootstrap IDs"
            )
        for key in (
            "last_train_transition_count",
            "last_attempt_train_transition_count",
        ):
            if int(state.get(key, -1)) != bootstrap["train_transitions"]:
                raise ValueError(
                    f"initial v3 state {key} is not bootstrap train split"
                )
        if int(state.get("last_update_episode_count", -1)) != len(
            bootstrap_ids
        ):
            raise ValueError("initial v3 update episode count mismatch")

    latest_rejected_attempt_audit: dict[str, Any] | None = None
    if attempt_ids != trained_ids:
        if attempt_index <= update_index:
            raise ValueError(
                "unaccepted v3 attempt IDs require attempt_index > update_index"
            )
        rejected_replay = _path_from_state(
            state, "latest_rejected_replay", state_dir
        )
        rejected_replay_sha = _sha256(rejected_replay)
        rejected_replay_audit = _validate_v5_replay_generic(
            rejected_replay,
            expected_sha256=rejected_replay_sha,
            expected_episode_ids=attempt_ids,
        )
        if (
            rejected_replay_audit["split_counts"]["train"]
            != last_attempt_train_transition_count
        ):
            raise ValueError(
                "latest rejected replay train count does not match state"
            )
        rejected_checkpoint = _path_from_state(
            state, "latest_rejected_checkpoint", state_dir
        )
        accepted_checkpoint = _path_from_state(
            state, "latest_checkpoint", state_dir
        )
        deployment_checkpoint_path = _path_from_state(
            state, "deployment_checkpoint", state_dir
        )
        if rejected_checkpoint in {
            accepted_checkpoint,
            deployment_checkpoint_path,
        }:
            raise ValueError(
                "latest rejected checkpoint is selected or deployed"
            )
        rejected_checkpoint_audit = _validate_checkpoint_v3(
            rejected_checkpoint,
            replay_sha256=rejected_replay_sha,
            target_betas=target_betas,
            rejected_tree_sha256=rejected["tree_sha256"],
            rejected_learner_sha256=rejected["learner_msgpack_sha256"],
            # Historical step_00000299 was rejected before the resume writer
            # preserved inherited warm-start fingerprint fields. It is still
            # checked against its exact replay/core v3 contract and, above,
            # proven neither selected nor deployed. Active/deployment
            # checkpoints always retain the stronger default lineage check.
            require_warm_start_lineage=False,
        )
        rejection_reason = state.get("latest_rejection_reason")
        if not isinstance(rejection_reason, str) or not rejection_reason:
            raise ValueError("latest rejected v3 attempt lacks its reason")
        latest_rejected_attempt_audit = {
            "checkpoint": rejected_checkpoint_audit,
            "replay": rejected_replay_audit,
            "reason": rejection_reason,
        }

    latest_replay = _path_from_state(state, "latest_replay", state_dir)
    latest_replay_sha = str(state.get("latest_replay_sha256", ""))
    latest_replay_audit = _validate_v5_replay_generic(
        latest_replay,
        expected_sha256=latest_replay_sha,
        expected_episode_ids=trained_ids,
    )
    if (
        latest_replay_audit["split_counts"]["train"]
        != last_train_transition_count
    ):
        raise ValueError("latest replay train count does not match state")
    latest_checkpoint = _path_from_state(
        state, "latest_checkpoint", state_dir
    )
    latest_checkpoint_audit = _validate_checkpoint_v3(
        latest_checkpoint,
        replay_sha256=latest_replay_sha,
        target_betas=target_betas,
        rejected_tree_sha256=rejected["tree_sha256"],
        rejected_learner_sha256=rejected["learner_msgpack_sha256"],
    )
    deployment_checkpoint = _path_from_state(
        state, "deployment_checkpoint", state_dir
    )
    deployment_replay_sha = (
        bootstrap["sha256"]
        if deployment_checkpoint == initial_checkpoint
        else latest_replay_sha
    )
    deployment_audit = _validate_checkpoint_v3(
        deployment_checkpoint,
        replay_sha256=deployment_replay_sha,
        target_betas=target_betas,
        rejected_tree_sha256=rejected["tree_sha256"],
        rejected_learner_sha256=rejected["learner_msgpack_sha256"],
    )

    config = _validate_config(
        config_path=config_file,
        session=session,
        state_root=state_dir,
        state=state,
        source_actor=source_actor,
        target_betas=target_betas,
    )
    episode_directories: list[str] = []
    for path in session.glob("episode_[0-9]*"):
        match = EPISODE_PATTERN.fullmatch(path.name)
        if path.is_dir() and match:
            index = int(match.group(1))
            if index < floor:
                raise ValueError(
                    f"old v2 episode directory mixed into target: {path.name}"
                )
            episode_directories.append(path.name)
    replay_schema_audit = _validate_no_old_v2_mix(
        state_root=state_dir, legacy_replay=legacy_path
    )

    return {
        "format": VALIDATION_FORMAT,
        "valid": True,
        "read_only": True,
        "session_root": str(session),
        "state_root": str(state_dir),
        "config": str(config_file),
        "workspace": str(workspace),
        "runtime": str(runtime),
        "selected_actor_file": str(selected_actor_file),
        "shadow_service": shadow_service,
        "fork_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha,
        },
        "episode_index_floor": floor,
        "first_episode_id": f"episode_{floor:06d}",
        "source_latest_complete_episode": source_manifest[
            "latest_complete_episode"
        ],
        "source_actor": {
            "step": EXPECTED_SOURCE_ACTOR_STEP,
            "path": str(source_actor),
            "tree_sha256": source_actor_audit["tree_sha256"],
        },
        "rejected_checkpoint_exclusion": {
            "step": EXPECTED_REJECTED_STEP,
            "path": str(source_rejected_path),
            "tree_sha256": rejected["tree_sha256"],
            "selected_as_warm_start": False,
            "selected_as_initial_v3": False,
            "selected_as_latest": False,
            "selected_as_deployment": False,
        },
        "bootstrap": {
            "path": str(bootstrap_path),
            "sha256": bootstrap["sha256"],
            "episode_ids": bootstrap_ids,
            "quality": expected_quality,
            "transitions": bootstrap["transitions"],
            "train_transitions": bootstrap["train_transitions"],
        },
        "initial_trained_v3_checkpoint": {
            "path": str(initial_checkpoint),
            "step": initial_audit["step"],
            "tree_sha256": initial_audit["tree_sha256"],
            "objective_weights": target_betas,
            "replay_sha256": bootstrap["sha256"],
            "acceptance_report": str(initial_validation_path),
            "acceptance_report_sha256": state[
                "initial_v3_validation_report_sha256"
            ],
            "accepted": True,
        },
        "objective_migration": migration_audit,
        "state_episode_ids": {
            "trained": trained_ids,
            "last_attempt": attempt_ids,
            "bootstrap_subset_preserved": True,
        },
        "latest_rejected_attempt": latest_rejected_attempt_audit,
        "latest_replay": latest_replay_audit,
        "latest_checkpoint": latest_checkpoint_audit,
        "deployment_checkpoint": deployment_audit,
        "target_episode_directories": sorted(episode_directories),
        "old_v2_mix_audit": {
            "legacy_replay_training_rows": 0,
            "replays": replay_schema_audit,
            "v2_outside_provenance": False,
        },
        "config_keys_validated": sorted(config),
    }


def main() -> None:
    args = _parser().parse_args()
    report = validate(
        args.session_root,
        args.state_root,
        args.config,
        expected_episode_floor=args.expected_episode_floor,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
