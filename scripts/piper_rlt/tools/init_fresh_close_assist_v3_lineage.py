#!/usr/bin/env python3
"""Create or validate an isolated gripper-close v3 RLT lineage from episode zero."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import tempfile
import time
from pathlib import Path


LINEAGE_MODE = "persistent_gripper_v3_fresh_zero"
REPLAY_POLICY = "fresh_persistent_v5_online_only"
STATE_FORMAT = "openpi_piper_online_rlt_state_persistent_gripper_v3"
ACTOR_PROFILE = "persistent_c10_filtered_actual_v2"
RAW_SCHEMA = "piper_joint_delta_v5_c10_n10_stride10_behavior_ref50_rank1_joint_r005_d1_0015_d2_001_cone15_gripper_close_knot_r005"
EXEC_SCHEMA = "piper_joint_delta_v5_c10_n10_stride10_behavior_ref50_persistent_filtered_actual_r005_d1_0015_d2_001_cone15_gripper_close_assist_r005_d1_0005_d2_0003_boundary0005"
PROJECTION = "rank1_joint_v1_r005_d1_0015_d2_001_cone15_scale33_min020_gripper_close_knot_r005"
GOVERNOR = "persistent_governor_v3_joint_r005_d1_0015_d2_001_cone15_boundary060_gripper_close_r005_d1_0005_d2_0003_boundary0005"
GRIPPER_MODE = "close_only_persistent_v1"
FORBIDDEN = (
    "bootstrap_gripper_replay",
    "bootstrap_gripper_replay_sha256",
    "bootstrap_gripper_episode_ids",
    "legacy_source_replay",
    "legacy_source_replay_sha256",
    "initial_actor_warm_start_checkpoint",
)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def _shell(value: object) -> str:
    return shlex.quote(str(value))


def _parse_config(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed = shlex.split(value, posix=True)
        result[key] = parsed[0] if parsed else ""
    return result


def _config(
    session: Path,
    state_root: Path,
    workspace: Path,
    runtime: Path,
    phase_checkpoint: Path,
    warmup: int,
) -> dict[str, object]:
    return {
        "RLT_SESSION_ROOT": session,
        "RLT_STATE_ROOT": state_root,
        "RLT_WORKSPACE": workspace,
        "RLT_RUNTIME": runtime,
        "RLT_PHASE_CHECKPOINT": phase_checkpoint,
        "RLT_SELECTED_ACTOR_FILE": state_root / "selected_actor_checkpoint.txt",
        "RLT_SHADOW_SERVICE": "openpi-rlt-shadow-policy-gripper-v3.service",
        "RLT_POLICY_HOST": "127.0.0.1",
        "RLT_POLICY_PORT": 8001,
        "RLT_LINEAGE_MODE": LINEAGE_MODE,
        "RLT_REPLAY_TRAINING_POLICY": REPLAY_POLICY,
        "RLT_ACTOR_EXECUTION_PROFILE": ACTOR_PROFILE,
        "RLT_ACTOR_MODEL_ACTION_SCHEMA_FINGERPRINT": RAW_SCHEMA,
        "RLT_EXECUTION_ACTION_SCHEMA_FINGERPRINT": EXEC_SCHEMA,
        "RLT_ACTOR_PROJECTION_PROFILE": PROJECTION,
        "RLT_EXECUTION_FILTER_PROFILE": "exp_one_minus_exp_neg_dt_over_tau_v1",
        "RLT_EXECUTION_FILTER_TAU_S": 0.05,
        "RLT_CONTROL_HZ": 30.0,
        "RLT_CONTROL_DT_S": 1.0 / 30.0,
        "RLT_EXECUTION_FILTER_ALPHA": 1.0 - math.exp(-(1.0 / 30.0) / 0.05),
        "RLT_CHUNK_LENGTH": 10,
        "RLT_CHUNK_STRIDE": 10,
        "RLT_REPLAY_STRIDE": 10,
        "RLT_N_STEP": 10,
        "RLT_ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD": 0.06,
        "RLT_ACTOR_PROJECTION_SCALE_STEPS": 33,
        "RLT_ACTOR_MIN_PROJECTION_SCALE": 0.2,
        "RLT_ACTOR_DIRECTION_STATIC_THRESHOLD_RAD": 0.001,
        "RLT_ACTOR_GOVERNOR_FINGERPRINT": GOVERNOR,
        "RLT_GRIPPER_RESIDUAL_MODE": GRIPPER_MODE,
        "RLT_GRIPPER_RESIDUAL_MAX": 0.005,
        "RLT_GRIPPER_RESIDUAL_D1_MAX_M": 0.0005,
        "RLT_GRIPPER_RESIDUAL_D2_MAX_M": 0.0003,
        "RLT_GRIPPER_MAX_BOUNDARY_JUMP_M": 0.0005,
        "RLT_GRIPPER_COMMAND_MIN_M": 0.0,
        "RLT_GRIPPER_COMMAND_MAX_M": 0.08,
        "RLT_GRIPPER_RELEASE_REFERENCE_M": 0.05,
        "RLT_GRIPPER_RELEASE_DELTA_M": 0.002,
        "RLT_FREEZE_GRIPPER_RESIDUAL": 0,
        "RLT_BETA_BC": 20.0,
        "RLT_BETA_HUMAN_BC": 0.0,
        "RLT_BETA_HUMAN_GRIPPER_BC": 1.0,
        "RLT_HUMAN_GRIPPER_BC_SCALE_M": 0.005,
        "RLT_HUMAN_GRIPPER_Q_FILTER_MODE": "critic_min_advantage_v1",
        "RLT_HUMAN_GRIPPER_Q_FILTER_MARGIN": 0.0,
        "RLT_ALLOW_WARM_START_OBJECTIVE_MIGRATION": 0,
        "RLT_WARMUP_EPISODES": warmup,
        "RLT_MIN_NEW_PERSISTENT_COMMITTED_EPISODES": warmup,
        "RLT_EPISODE_INDEX_FLOOR": 0,
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
        "RLT_BASE_FINGERPRINT": "full20k_step20000_metadata_sha256_14d9cac129ec7ce91f2e5aab3f5bfac06172c8fb70709f01850fb8e8215870e5",
        "RLT_TOKEN_FINGERPRINT": "2f2e1e6bbcae8f08217ec7ba0b88088bfa44627be495035319deb84e06052b49",
        "RLT_PHASE_FINGERPRINT": "8c5b443edd3f399529680ef5e4c5dffcdee4af2ae2f224f6da4152237c9a50dc",
    }


def _state(session: Path, warmup: int) -> dict[str, object]:
    return {
        "format": STATE_FORMAT,
        "lineage_mode": LINEAGE_MODE,
        "replay_training_policy": REPLAY_POLICY,
        "session_root": str(session),
        "episode_index_floor": 0,
        "min_new_persistent_committed_episodes": warmup,
        "actor_execution_profile": ACTOR_PROFILE,
        "actor_model_action_schema_fingerprint": RAW_SCHEMA,
        "execution_action_schema_fingerprint": EXEC_SCHEMA,
        "actor_projection_profile": PROJECTION,
        "execution_filter_profile": "exp_one_minus_exp_neg_dt_over_tau_v1",
        "execution_filter_tau_s": 0.05,
        "control_hz": 30.0,
        "control_dt_s": 1.0 / 30.0,
        "execution_filter_alpha": 1.0 - math.exp(-(1.0 / 30.0) / 0.05),
        "chunk_length": 10,
        "chunk_stride": 10,
        "actor_live_max_boundary_jump_rad": 0.06,
        "actor_projection_scale_steps": 33,
        "actor_min_projection_scale": 0.2,
        "actor_direction_static_threshold_rad": 0.001,
        "actor_governor_fingerprint": GOVERNOR,
        "gripper_residual_mode": GRIPPER_MODE,
        "actor_gripper_residual_max_close_m": 0.005,
        "actor_gripper_residual_d1_max_m": 0.0005,
        "actor_gripper_residual_d2_max_m": 0.0003,
        "actor_gripper_max_boundary_jump_m": 0.0005,
        "gripper_command_min_m": 0.0,
        "gripper_command_max_m": 0.08,
        "gripper_release_reference_m": 0.05,
        "gripper_release_delta_m": 0.002,
        "human_gripper_q_filter_mode": "critic_min_advantage_v1",
        "human_gripper_q_filter_margin": 0.0,
        "trained_episode_ids": [],
        "last_attempt_episode_ids": [],
        "last_update_episode_count": 0,
        "last_attempt_episode_count": 0,
        "update_index": 0,
        "attempt_index": 0,
        "fresh_zero": True,
        "created_unix": time.time(),
    }


def validate(session: Path, state_root: Path, config_path: Path) -> dict[str, object]:
    session, state_root, config_path = (
        session.resolve(),
        state_root.resolve(),
        config_path.resolve(),
    )
    if state_root.parent != session:
        raise ValueError("state root must be one direct child of the session root")
    state = json.loads((state_root / "online_state.json").read_text(encoding="utf-8"))
    config = _parse_config(config_path)
    expected = {
        "format": STATE_FORMAT,
        "lineage_mode": LINEAGE_MODE,
        "replay_training_policy": REPLAY_POLICY,
        "session_root": str(session),
        "episode_index_floor": 0,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"fresh-zero state mismatch: {key}")
    if Path(config.get("RLT_SESSION_ROOT", "")).resolve() != session:
        raise ValueError("config is bound to another session root")
    if Path(config.get("RLT_STATE_ROOT", "")).resolve() != state_root:
        raise ValueError("config is bound to another state root")
    if config.get("RLT_LINEAGE_MODE") != LINEAGE_MODE:
        raise ValueError("config lineage mode is not fresh-zero")
    if config.get("RLT_REPLAY_TRAINING_POLICY") != REPLAY_POLICY:
        raise ValueError("config replay policy is not fresh-only")
    warmup = int(config.get("RLT_WARMUP_EPISODES", "0"))
    if warmup <= 0 or int(config.get("RLT_MIN_NEW_PERSISTENT_COMMITTED_EPISODES", "-1")) != warmup:
        raise ValueError("fresh-zero warmup thresholds do not match")
    if int(state.get("min_new_persistent_committed_episodes", -1)) != warmup:
        raise ValueError("state/config warmup threshold mismatch")
    if config.get("RLT_ALLOW_WARM_START_OBJECTIVE_MIGRATION") != "0":
        raise ValueError("fresh-zero objective migration must be disabled")
    if "RLT_WARM_START_ACTOR_CHECKPOINT" in config:
        raise ValueError("fresh-zero config contains an Actor warm-start")
    contaminated = [key for key in FORBIDDEN if state.get(key) not in (None, [], "")]
    if contaminated:
        raise ValueError(f"fresh-zero state contains inherited fields: {contaminated}")
    latest = state.get("latest_checkpoint") or state.get("deployment_checkpoint")
    selector = (state_root / "selected_actor_checkpoint.txt").read_text(encoding="utf-8").strip()
    if latest:
        checkpoint = Path(str(latest)).resolve()
        if state_root not in checkpoint.parents:
            raise ValueError("latest checkpoint escaped this fresh state")
        if not (checkpoint / "learner.msgpack").is_file() or not (checkpoint / "metadata.json").is_file():
            raise ValueError("latest checkpoint is incomplete")
        if selector != str(checkpoint):
            raise ValueError("selected Actor does not match latest checkpoint")
    elif selector != "NONE":
        raise ValueError("untrained fresh-zero selector must be NONE")
    episode_ids = []
    for path in session.glob("episode_[0-9]*"):
        match = re.fullmatch(r"episode_([0-9]{6})", path.name)
        if path.is_dir() and match:
            episode_ids.append(path.name)
    return {
        "valid": True,
        "mode": LINEAGE_MODE,
        "session_root": str(session),
        "state_root": str(state_root),
        "warmup_episodes": warmup,
        "episode_count": len(episode_ids),
        "latest_checkpoint": latest,
        "inherits_prior_training": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--phase-checkpoint", type=Path)
    parser.add_argument("--warmup-episodes", type=int)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", action="store_true")
    mode.add_argument("--validate-existing", action="store_true")
    args = parser.parse_args()

    session = args.session_root.expanduser().resolve()
    state_root = args.state_root.expanduser().resolve()
    config_path = (args.config or state_root / "config.env").expanduser().resolve()
    if args.create:
        if args.warmup_episodes is None or args.warmup_episodes <= 0:
            raise SystemExit("--warmup-episodes must be positive when creating")
        if session.exists():
            raise SystemExit(f"fresh-zero target already exists; refusing to mix or overwrite: {session}")
        if state_root.parent != session:
            raise SystemExit("--state-root must be one direct child of --session-root")
        workspace = (args.workspace or Path.home() / "gripper_close_v3_staging_20260727").expanduser().resolve()
        runtime = (args.runtime or workspace).expanduser().resolve()
        phase = (args.phase_checkpoint or Path.home() / "rlt_phase_classifiers/greenblock_box_resnet18_v4_manual_intervals/phase_classifier.pt").expanduser().resolve()
        for path, label in ((workspace, "workspace"), (runtime, "runtime"), (phase, "phase checkpoint")):
            if not path.exists():
                raise SystemExit(f"{label} does not exist: {path}")
        state_root.mkdir(parents=True)
        config = _config(session, state_root, workspace, runtime, phase, args.warmup_episodes)
        _atomic_text(config_path, "".join(f"{key}={_shell(value)}\n" for key, value in config.items()))
        _atomic_text(state_root / "online_state.json", json.dumps(_state(session, args.warmup_episodes), indent=2, sort_keys=True) + "\n")
        _atomic_text(state_root / "selected_actor_checkpoint.txt", "NONE\n")
        _atomic_text(
            state_root / "fresh_zero_manifest.json",
            json.dumps(
                {
                    "format": "openpi_piper_gripper_v3_fresh_zero_lineage_v1",
                    "created_unix": time.time(),
                    "session_root": str(session),
                    "state_root": str(state_root),
                    "first_episode_id": "episode_000000",
                    "warmup_episodes": args.warmup_episodes,
                    "inherited_episode_ids": [],
                    "inherited_replays": [],
                    "inherited_actor_checkpoints": [],
                    "inherited_critic_checkpoints": [],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
    print(json.dumps(validate(session, state_root, config_path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
