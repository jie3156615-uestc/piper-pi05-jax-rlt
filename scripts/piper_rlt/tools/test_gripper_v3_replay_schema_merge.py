from __future__ import annotations

import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import run_online_rlt_update as updater  # noqa: E402


_OPTIONAL_KEYS = updater._HISTORICAL_V3_BOOTSTRAP_INCREMENTAL_ONLY_KEYS


def test_gripper_v3_training_command_explicitly_unfreezes_gripper() -> None:
    source = inspect.getsource(updater.run_update)
    assert '"--no-freeze-gripper-residual"' in source
    assert source.index('"--no-freeze-gripper-residual"') < source.index(
        '"--gripper-residual-mode"'
    )


def test_promotion_accepts_legacy_reward_gap_report_alias() -> None:
    args = SimpleNamespace(
        actor_execution_profile=updater.LEGACY_ACTOR_EXECUTION_PROFILE,
        max_validation_td_error=0.5,
        max_actor_q_advantage=0.5,
        min_reward1_reward0_exec_q_gap=0.0,
        max_active_normalized_residual_step=None,
        max_chunk_boundary_normalized_residual_jump_p95=None,
    )
    report = {
        "passed": True,
        "finite_action": True,
        "finite_action_inputs": True,
        "finite_action_outputs": True,
        "finite_q_values": True,
        "finite_td_target": True,
        "finite_key_metrics": True,
        "validation_td_error_abs_mean": 0.1,
        "actor_q_advantage_abs_p95": 0.1,
        "success_failure_exec_q_gap": 0.1,
    }
    updater._validate_promotion(report, args)


def test_promotion_reports_real_negative_legacy_gap_not_missing_data() -> None:
    args = SimpleNamespace(
        actor_execution_profile=updater.LEGACY_ACTOR_EXECUTION_PROFILE,
        max_validation_td_error=0.5,
        max_actor_q_advantage=0.5,
        min_reward1_reward0_exec_q_gap=0.0,
        max_active_normalized_residual_step=None,
        max_chunk_boundary_normalized_residual_jump_p95=None,
    )
    report = {
        "passed": True,
        "finite_action": True,
        "finite_action_inputs": True,
        "finite_action_outputs": True,
        "finite_q_values": True,
        "finite_td_target": True,
        "finite_key_metrics": True,
        "validation_td_error_abs_mean": 0.1,
        "actor_q_advantage_abs_p95": 0.1,
        "success_failure_exec_q_gap": -0.1,
    }
    with pytest.raises(
        RuntimeError,
        match="insufficient Q advantage",
    ):
        updater._validate_promotion(report, args)


def _common_arrays(episode_ids: list[str]) -> dict[str, np.ndarray]:
    rows = len(episode_ids)
    return {
        "episode_id": np.asarray(episode_ids),
        "action_schema_fingerprint": np.asarray(
            [updater.PERSISTENT_V2_ACTION_SCHEMA_FINGERPRINT] * rows
        ),
        "reward": np.arange(rows, dtype=np.float32),
        "a_exec": np.arange(rows * 10 * 7, dtype=np.float32).reshape(
            rows, 10, 7
        ),
    }


def _new_gripper_evidence(rows: int) -> dict[str, np.ndarray]:
    scalar = np.arange(rows * 10, dtype=np.float32).reshape(rows, 10)
    return {
        "actor_persistent_planned_residual": np.arange(
            rows * 10 * 7, dtype=np.float32
        ).reshape(rows, 10, 7),
        "actor_gripper_release_intent": np.zeros(
            (rows, 10), dtype=np.bool_
        ),
        "actor_filtered_actual_gripper_residual_max": scalar.copy(),
        "actor_filtered_actual_gripper_residual_d1_max_m": scalar.copy(),
        "actor_filtered_actual_gripper_residual_d2_max_m": scalar.copy(),
        "actor_filtered_actual_gripper_boundary_jump_max_m": scalar.copy(),
    }


def _write(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def test_merge_accepts_exact_historical_incremental_extension(
    tmp_path: Path,
) -> None:
    base = _common_arrays(["bootstrap_0", "bootstrap_1"])
    incremental = {
        **_common_arrays(["episode_2"]),
        **_new_gripper_evidence(1),
    }
    base_path = tmp_path / "base.npz"
    incremental_path = tmp_path / "incremental.npz"
    output_path = tmp_path / "merged.npz"
    _write(base_path, base)
    _write(incremental_path, incremental)

    updater._merge_replays(base_path, incremental_path, output_path)

    with np.load(output_path, allow_pickle=False) as merged:
        assert set(merged.files) == set(base)
        assert _OPTIONAL_KEYS.isdisjoint(merged.files)
        assert merged["episode_id"].astype(str).tolist() == [
            "bootstrap_0",
            "bootstrap_1",
            "episode_2",
        ]
        np.testing.assert_array_equal(
            merged["a_exec"],
            np.concatenate([base["a_exec"], incremental["a_exec"]], axis=0),
        )

    # The derived learner replay omits historical-unavailable evidence, while
    # the immutable incremental source retains every newly logged field.
    with np.load(incremental_path, allow_pickle=False) as source:
        assert _OPTIONAL_KEYS.issubset(source.files)


def test_merge_rejects_unknown_incremental_extension(tmp_path: Path) -> None:
    base = _common_arrays(["bootstrap_0"])
    incremental = {
        **_common_arrays(["episode_1"]),
        **_new_gripper_evidence(1),
        "unknown_incremental_telemetry": np.zeros((1, 10), dtype=np.float32),
    }
    base_path = tmp_path / "base.npz"
    incremental_path = tmp_path / "incremental.npz"
    output_path = tmp_path / "merged.npz"
    _write(base_path, base)
    _write(incremental_path, incremental)

    with pytest.raises(ValueError, match="schemas do not match"):
        updater._merge_replays(base_path, incremental_path, output_path)

    assert not output_path.exists()


def test_merge_rejects_base_only_field(tmp_path: Path) -> None:
    base = {
        **_common_arrays(["bootstrap_0"]),
        "bootstrap_only_field": np.zeros((1, 10), dtype=np.float32),
    }
    incremental = _common_arrays(["episode_1"])
    base_path = tmp_path / "base.npz"
    incremental_path = tmp_path / "incremental.npz"
    output_path = tmp_path / "merged.npz"
    _write(base_path, base)
    _write(incremental_path, incremental)

    with pytest.raises(ValueError, match="schemas do not match"):
        updater._merge_replays(base_path, incremental_path, output_path)

    assert not output_path.exists()


def test_merge_preserves_equal_full_schemas(tmp_path: Path) -> None:
    base = {
        **_common_arrays(["bootstrap_0"]),
        **_new_gripper_evidence(1),
    }
    incremental = {
        **_common_arrays(["episode_1", "episode_2"]),
        **_new_gripper_evidence(2),
    }
    base_path = tmp_path / "base.npz"
    incremental_path = tmp_path / "incremental.npz"
    output_path = tmp_path / "merged.npz"
    _write(base_path, base)
    _write(incremental_path, incremental)

    updater._merge_replays(base_path, incremental_path, output_path)

    with np.load(output_path, allow_pickle=False) as merged:
        assert set(merged.files) == set(base) == set(incremental)
        for name in _OPTIONAL_KEYS:
            np.testing.assert_array_equal(
                merged[name],
                np.concatenate([base[name], incremental[name]], axis=0),
            )
