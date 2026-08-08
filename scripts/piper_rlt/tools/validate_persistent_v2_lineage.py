#!/usr/bin/env python3
"""Read-only validation of a persistent-v2 lineage/state/config binding."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
from typing import Any

import numpy as np

from persistent_v2_contract import (
    ACTION_SCHEMA_FINGERPRINT,
    ACTOR_DIRECTION_STATIC_THRESHOLD_RAD,
    ACTOR_EXECUTION_PROFILE,
    ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD,
    ACTOR_MIN_PROJECTION_SCALE,
    ACTOR_PROJECTION_PROFILE,
    ACTOR_PROJECTION_SCALE_STEPS,
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
from fork_persistent_v2_lineage import SOURCE_ACTION_SCHEMA


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--read-only",
        action="store_true",
        required=True,
        help="Mandatory acknowledgement: this command never repairs state.",
    )
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        values = shlex.split(raw_value, posix=True)
        if len(values) != 1:
            raise ValueError(f"{path}:{line_number}: value must be scalar")
        result[key] = values[0]
    return result


def _exact_float(actual: Any, expected: float, label: str) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {actual!r}") from exc
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"{label} mismatch: {value!r} != {expected!r}")


def validate(
    session_root: Path, state_root: Path, config_path: Path
) -> dict[str, Any]:
    session = session_root.expanduser().resolve()
    state_dir = state_root.expanduser().resolve()
    config_file = config_path.expanduser().resolve()
    if state_dir.parent != session:
        raise ValueError("persistent state must be one direct child of session")
    state = _load_json(state_dir / "online_state.json")
    config = _parse_config(config_file)
    if Path(state.get("session_root", "")).resolve() != session:
        raise ValueError("state/session binding mismatch")
    if Path(config.get("RLT_SESSION_ROOT", "")).resolve() != session:
        raise ValueError("config/session binding mismatch")
    if Path(config.get("RLT_STATE_ROOT", "")).resolve() != state_dir:
        raise ValueError("config/state binding mismatch")

    exact_state = {
        "format": "openpi_piper_online_rlt_state_persistent_v2",
        "lineage_mode": "persistent_v2_actor_only_warm_start",
        "actor_execution_profile": ACTOR_EXECUTION_PROFILE,
        "actor_model_action_schema_fingerprint": SOURCE_ACTION_SCHEMA,
        "execution_action_schema_fingerprint": ACTION_SCHEMA_FINGERPRINT,
        "actor_projection_profile": ACTOR_PROJECTION_PROFILE,
        "execution_filter_profile": EXECUTION_FILTER_PROFILE,
        "chunk_length": CHUNK_LENGTH,
        "chunk_stride": CHUNK_STRIDE,
        "replay_training_policy": "persistent_only_no_legacy_merge",
        "actor_projection_scale_steps": ACTOR_PROJECTION_SCALE_STEPS,
        "actor_governor_fingerprint": PERSISTENT_GOVERNOR_FINGERPRINT,
    }
    for key, expected in exact_state.items():
        if state.get(key) != expected:
            raise ValueError(
                f"state {key} mismatch: {state.get(key)!r} != {expected!r}"
            )
    for key, expected in (
        ("execution_filter_tau_s", EXECUTION_FILTER_TAU_S),
        ("control_hz", CONTROL_HZ),
        ("control_dt_s", CONTROL_DT_S),
        ("execution_filter_alpha", EXECUTION_FILTER_ALPHA),
        (
            "actor_live_max_boundary_jump_rad",
            ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD,
        ),
        ("actor_min_projection_scale", ACTOR_MIN_PROJECTION_SCALE),
        (
            "actor_direction_static_threshold_rad",
            ACTOR_DIRECTION_STATIC_THRESHOLD_RAD,
        ),
    ):
        _exact_float(state.get(key), expected, f"state {key}")

    exact_config = {
        "RLT_LINEAGE_MODE": "persistent_v2_actor_only_warm_start",
        "RLT_ACTOR_EXECUTION_PROFILE": ACTOR_EXECUTION_PROFILE,
        "RLT_ACTOR_MODEL_ACTION_SCHEMA_FINGERPRINT": SOURCE_ACTION_SCHEMA,
        "RLT_EXECUTION_ACTION_SCHEMA_FINGERPRINT": ACTION_SCHEMA_FINGERPRINT,
        "RLT_ACTOR_PROJECTION_PROFILE": ACTOR_PROJECTION_PROFILE,
        "RLT_EXECUTION_FILTER_PROFILE": EXECUTION_FILTER_PROFILE,
        "RLT_REPLAY_TRAINING_POLICY": "persistent_only_no_legacy_merge",
        "RLT_CHUNK_LENGTH": str(CHUNK_LENGTH),
        "RLT_CHUNK_STRIDE": str(CHUNK_STRIDE),
        "RLT_REPLAY_STRIDE": str(CHUNK_STRIDE),
        "RLT_ACTOR_PROJECTION_SCALE_STEPS": str(
            ACTOR_PROJECTION_SCALE_STEPS
        ),
        "RLT_ACTOR_GOVERNOR_FINGERPRINT": (
            PERSISTENT_GOVERNOR_FINGERPRINT
        ),
    }
    for key, expected in exact_config.items():
        if config.get(key) != expected:
            raise ValueError(
                f"config {key} mismatch: {config.get(key)!r} != {expected!r}"
            )
    for key, expected in (
        ("RLT_EXECUTION_FILTER_TAU_S", EXECUTION_FILTER_TAU_S),
        ("RLT_CONTROL_HZ", CONTROL_HZ),
        ("RLT_CONTROL_DT_S", CONTROL_DT_S),
        ("RLT_EXECUTION_FILTER_ALPHA", EXECUTION_FILTER_ALPHA),
        (
            "RLT_ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD",
            ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD,
        ),
        ("RLT_ACTOR_MIN_PROJECTION_SCALE", ACTOR_MIN_PROJECTION_SCALE),
        (
            "RLT_ACTOR_DIRECTION_STATIC_THRESHOLD_RAD",
            ACTOR_DIRECTION_STATIC_THRESHOLD_RAD,
        ),
    ):
        _exact_float(config.get(key), expected, f"config {key}")

    minimum = int(state.get("min_new_persistent_committed_episodes", 0))
    if minimum < DEFAULT_MIN_NEW_COMMITTED_EPISODES:
        raise ValueError("state persistent warmup is below 30")
    if int(config["RLT_MIN_NEW_PERSISTENT_COMMITTED_EPISODES"]) != minimum:
        raise ValueError("state/config persistent warmup mismatch")
    if int(config["RLT_WARMUP_EPISODES"]) < minimum:
        raise ValueError("config warmup is below persistent minimum")
    floor = int(state["episode_index_floor"])
    if int(config["RLT_EPISODE_INDEX_FLOOR"]) != floor:
        raise ValueError("state/config episode floor mismatch")
    if state.get("frozen_base_replay") is not None or state.get(
        "frozen_base_episode_ids"
    ):
        raise ValueError("legacy replay merge fields are forbidden")

    legacy_replay = Path(state["legacy_source_replay"]).expanduser().resolve()
    if not legacy_replay.is_file():
        raise FileNotFoundError(legacy_replay)
    if _sha256(legacy_replay) != state["legacy_source_replay_sha256"]:
        raise ValueError("legacy provenance replay SHA mismatch")
    with np.load(legacy_replay, allow_pickle=False) as replay:
        replay_ids = sorted(
            set(np.asarray(replay["episode_id"]).astype(str).tolist())
        )
    if replay_ids != sorted(state["legacy_source_replay_episode_ids"]):
        raise ValueError("legacy provenance replay episode IDs mismatch")

    warm_start = Path(
        state["initial_actor_warm_start_checkpoint"]
    ).expanduser().resolve()
    if Path(config["RLT_WARM_START_ACTOR_CHECKPOINT"]).resolve() != warm_start:
        raise ValueError("state/config Actor warm-start mismatch")
    if state_dir not in warm_start.parents:
        raise ValueError("Actor warm-start provenance is outside state")
    if not (warm_start / "learner.msgpack").is_file():
        raise FileNotFoundError(warm_start / "learner.msgpack")
    deployment = Path(state["deployment_checkpoint"]).expanduser().resolve()
    if state_dir not in deployment.parents:
        raise ValueError("deployment checkpoint is outside state")
    latest = state.get("latest_checkpoint")
    if latest:
        deployment = Path(latest).expanduser().resolve()
        if state_dir not in deployment.parents:
            raise ValueError("latest checkpoint is outside state")

    episode_names: list[str] = []
    for path in session.glob("episode_[0-9]*"):
        match = re.fullmatch(r"episode_([0-9]+)", path.name)
        if path.is_dir() and match:
            index = int(match.group(1))
            if index < floor:
                raise ValueError(
                    f"legacy episode contaminated persistent session: {path.name}"
                )
            episode_names.append(path.name)
    return {
        "format": "openpi_piper_persistent_v2_lineage_validation",
        "session_root": str(session),
        "state_root": str(state_dir),
        "episode_index_floor": floor,
        "episode_count": len(episode_names),
        "latest_checkpoint": latest,
        "deployment_checkpoint": str(deployment),
        "legacy_replay_training_rows": 0,
        "valid": True,
        "read_only": True,
    }


def main() -> None:
    args = _parser().parse_args()
    report = validate(args.session_root, args.state_root, args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
