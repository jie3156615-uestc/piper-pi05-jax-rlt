from __future__ import annotations

import copy
import sys

import numpy as np
import pytest

from scripts import validate_real_rlt_actor_jax as validator


def _passing_report() -> dict[str, object]:
    return {
        "finite_action": True,
        "finite_action_inputs": True,
        "finite_action_outputs": True,
        "finite_q_values": True,
        "finite_td_target": True,
        "finite_key_metrics": True,
        "residual_within_limit": True,
        "physical_residual_limit_contract": True,
        "normalized_residual_temporal_step_contract": True,
        "residual_saturation_contract": True,
        "cross_transition_residual_step_contract": True,
        "residual_limit_gripper_max_m": 0.0,
        "gripper_residual_frozen": True,
        "q_action_sensitivity_l1": 1e-3,
    }


def test_standalone_validator_keeps_enhanced_gates_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate", "--checkpoint", "checkpoint", "--replay-npz", "replay.npz"],
    )

    args = validator.parse_args()

    assert args.max_active_normalized_residual_step is None
    assert args.max_actor_joint_d1_p95_rad is None
    assert args.max_chunk_boundary_normalized_residual_jump_p95 is None
    assert args.max_chunk_boundary_actor_command_joint_d1_p95_rad is None


def test_finiteness_helpers_reject_nan_and_inf_recursively() -> None:
    assert validator._all_finite(np.asarray([0.0, 1.0]))
    assert not validator._all_finite(np.asarray([0.0, np.nan]))
    assert not validator._all_finite(np.asarray([0.0, np.inf]))
    assert validator._numeric_values_are_finite({"metric": [1.0, None], "label": "ok"})
    assert not validator._numeric_values_are_finite({"metric": {"td": float("nan")}})


@pytest.mark.parametrize(
    "flag",
    [
        "finite_action",
        "finite_action_inputs",
        "finite_action_outputs",
        "finite_q_values",
        "finite_td_target",
        "finite_key_metrics",
    ],
)
def test_acceptance_requires_every_finiteness_flag(flag: str) -> None:
    report = copy.deepcopy(_passing_report())
    report[flag] = False

    assert not validator._acceptance_passed(report, min_action_sensitivity=1e-7)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_acceptance_rejects_non_finite_key_metric(value: float) -> None:
    report = _passing_report()
    report["q_action_sensitivity_l1"] = value

    assert not validator._acceptance_passed(report, min_action_sensitivity=1e-7)


def test_active_temporal_metric_excludes_frozen_action_and_masked_padding() -> None:
    residual = np.asarray(
        [
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.2, 100.0],
                [0.2, 0.4, 200.0],
                [0.9, 1.4, 300.0],
            ]
        ],
        dtype=np.float32,
    )
    residual_limit = np.asarray(
        [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    step_mask = np.asarray([[True, True, True, False]])

    metrics = validator._active_normalized_residual_temporal_metrics(
        residual,
        residual_limit,
        step_mask=step_mask,
        step_mask_was_stored=True,
    )

    assert metrics[
        "active_normalized_residual_temporal_step_abs_mean"
    ] == pytest.approx(0.15)
    assert metrics[
        "active_normalized_residual_temporal_step_abs_mean_per_action"
    ] == pytest.approx([0.1, 0.2, None])
    assert metrics[
        "active_normalized_residual_temporal_step_abs_mean_action_max"
    ] == pytest.approx(0.2)
    assert metrics["active_normalized_residual_temporal_step_value_count"] == 4
    assert metrics["active_normalized_residual_temporal_step_valid_pair_count"] == 2
    assert metrics[
        "active_normalized_residual_temporal_step_active_action_dimensions"
    ] == [0, 1]
    assert metrics["active_normalized_residual_temporal_step_mask_used"] is True


def test_chunk_temporal_metrics_report_masked_per_joint_d1_and_d2() -> None:
    commands = np.zeros((1, 4, 7), dtype=np.float32)
    commands[0, :, 0] = [0.0, 1.0, 3.0, 100.0]
    commands[0, :, 1] = [0.0, 2.0, 6.0, 200.0]
    step_mask = np.asarray([[True, True, True, False]])

    metrics = validator._chunk_joint_temporal_metrics(
        commands,
        step_mask=step_mask,
        prefix="actor_command",
    )

    assert metrics["actor_command_temporal_d1_valid_pair_count"] == 2
    assert metrics["actor_command_temporal_d2_valid_triplet_count"] == 1
    assert metrics["actor_command_temporal_d1_abs_p95_per_joint"][:2] == pytest.approx(
        [1.95, 3.9]
    )
    assert metrics["actor_command_temporal_d1_abs_p99_per_joint"][:2] == pytest.approx(
        [1.99, 3.98]
    )
    assert metrics["actor_command_temporal_d1_abs_max_per_joint"][:2] == pytest.approx(
        [2.0, 4.0]
    )
    assert metrics["actor_command_temporal_d2_abs_p95_per_joint"][:2] == pytest.approx(
        [1.0, 2.0]
    )


def test_chunk_boundary_uses_exact_t_plus_c_last_to_first_and_baselines() -> None:
    chunk_length = 10
    action_dim = 7
    timesteps = np.asarray([0, 2, 9, 10, 12], dtype=np.int64)
    sample_count = len(timesteps)
    a_ref = np.zeros((sample_count, chunk_length, action_dim), dtype=np.float32)
    a_exec = np.zeros_like(a_ref)
    a_ref_absolute = np.zeros_like(a_ref)
    a_exec_absolute = np.zeros_like(a_ref)
    residual = np.zeros_like(a_ref)
    # The only valid exact boundary is t=0 horizon 9 -> t=10 horizon 0.
    a_ref_absolute[0, 9, 0] = 1.0
    a_ref_absolute[3, 0, 0] = 3.0
    a_exec_absolute[0, 9, 0] = 2.0
    a_exec_absolute[3, 0, 0] = 5.0
    residual[0, 9, 0] = 0.5
    residual[3, 0, 0] = 0.25
    step_mask = np.ones((sample_count, chunk_length), dtype=np.bool_)
    # t=2 has an exact t=12 partner but its current chunk is not fully valid.
    step_mask[1, -1] = False
    replay = {
        "episode_id": np.repeat("episode", sample_count),
        "t": timesteps,
        "state": np.zeros((sample_count, action_dim), dtype=np.float32),
        "a_ref": a_ref,
        "a_exec": a_exec,
        "a_ref_absolute": a_ref_absolute,
        "a_exec_absolute": a_exec_absolute,
        "done": np.zeros(sample_count, dtype=np.bool_),
    }
    # Candidate actions use replay training coordinates.  State is zero here,
    # so they must agree with the stored absolute reference plus residual.
    actions = a_ref_absolute + residual
    residual_limit = np.ones((chunk_length, action_dim), dtype=np.float32)
    residual_limit[:, -1] = 0.0

    metrics = validator._chunk_boundary_metrics(
        replay,
        actions=actions,
        residual=residual,
        residual_limit=residual_limit,
        step_mask=step_mask,
        max_normalized_residual_jump_p95=0.3,
        max_actor_command_joint_d1_p95_rad=2.0,
    )

    assert metrics["chunk_boundary_gap_steps"] == 10
    assert metrics["chunk_boundary_pair_count"] == 1
    assert metrics["chunk_boundary_invalid_step_mask_count"] == 1
    assert metrics["chunk_boundary_residual_jump_normalized_p95"] == pytest.approx(0.25)
    assert metrics["chunk_boundary_residual_jump_active_action_dimensions"] == list(
        range(6)
    )
    assert metrics["chunk_boundary_residual_jump_contract"] is True
    assert metrics["chunk_boundary_reference_d1_abs_p95_per_joint"][0] == pytest.approx(
        2.0
    )
    assert metrics["chunk_boundary_actor_command_d1_abs_p95_per_joint"][
        0
    ] == pytest.approx(1.75)
    assert metrics["chunk_boundary_actor_command_d1_abs_p95_joint_max"] == pytest.approx(
        1.75
    )
    assert metrics["chunk_boundary_actor_command_d1_p95_contract"] is True
    assert metrics["chunk_boundary_executed_command_d1_abs_p95_per_joint"][
        0
    ] == pytest.approx(3.0)


def test_chunk_boundary_optional_gate_fails_closed_without_valid_pair() -> None:
    replay = {
        "episode_id": np.asarray(["episode", "episode"]),
        "t": np.asarray([0, 9]),
        "state": np.zeros((2, 7), dtype=np.float32),
        "a_ref": np.zeros((2, 10, 7), dtype=np.float32),
        "a_exec": np.zeros((2, 10, 7), dtype=np.float32),
        "done": np.zeros(2, dtype=np.bool_),
    }
    actions = np.zeros((2, 10, 7), dtype=np.float32)
    residual_limit = np.ones((10, 7), dtype=np.float32)
    residual_limit[:, -1] = 0.0

    report_only = validator._chunk_boundary_metrics(
        replay,
        actions=actions,
        residual=actions,
        residual_limit=residual_limit,
        step_mask=np.ones((2, 10), dtype=np.bool_),
        max_normalized_residual_jump_p95=None,
    )
    gated = validator._chunk_boundary_metrics(
        replay,
        actions=actions,
        residual=actions,
        residual_limit=residual_limit,
        step_mask=np.ones((2, 10), dtype=np.bool_),
        max_normalized_residual_jump_p95=0.5,
        max_actor_command_joint_d1_p95_rad=0.05,
    )

    assert report_only["chunk_boundary_pair_count"] == 0
    assert report_only["chunk_boundary_residual_jump_contract"] is None
    assert gated["chunk_boundary_residual_jump_contract"] is False
    assert gated["chunk_boundary_actor_command_d1_p95_contract"] is False


def test_chunk_boundary_actor_command_uses_candidate_not_reference_baseline() -> None:
    replay = {
        "episode_id": np.asarray(["episode", "episode"]),
        "t": np.asarray([0, 10]),
        "state": np.zeros((2, 7), dtype=np.float32),
        "a_ref": np.zeros((2, 10, 7), dtype=np.float32),
        "a_exec": np.zeros((2, 10, 7), dtype=np.float32),
        "a_ref_absolute": np.zeros((2, 10, 7), dtype=np.float32),
        "a_exec_absolute": np.zeros((2, 10, 7), dtype=np.float32),
        "done": np.zeros(2, dtype=np.bool_),
    }
    # Persistent-v2's physical candidate is based on a_base_filtered, not a_ref.
    # Keep residual at zero while making the candidate boundary non-zero so the
    # legacy reconstruction (a_ref_absolute + residual) cannot satisfy this test.
    actions = np.zeros((2, 10, 7), dtype=np.float32)
    actions[0, -1, 0] = 0.4
    actions[1, 0, 0] = 0.1
    residual = np.zeros_like(actions)
    residual_limit = np.ones((10, 7), dtype=np.float32)
    residual_limit[:, -1] = 0.0

    metrics = validator._chunk_boundary_metrics(
        replay,
        actions=actions,
        residual=residual,
        residual_limit=residual_limit,
        step_mask=np.ones((2, 10), dtype=np.bool_),
        max_normalized_residual_jump_p95=None,
    )

    assert metrics["chunk_boundary_actor_command_d1_abs_p95_per_joint"][0] == pytest.approx(
        0.3
    )


def test_rank1_directional_contract_uses_hard_maxima_and_frozen_gripper() -> None:
    reference = np.zeros((2, 10, 7), dtype=np.float32)
    reference[:, :, 0] = np.arange(1, 11, dtype=np.float32)[None] * 0.01
    window = np.asarray([0.0, 0.2, 0.5, 0.8, 1.0, 1.0, 0.8, 0.5, 0.2, 0.0])
    residual = np.zeros_like(reference)
    residual[:, :, 1] = window[None] * 0.002

    metrics = validator._rank1_directional_contract_metrics(
        reference,
        residual,
        max_rank1_fit_error_rad=1e-5,
        max_residual_abs_rad=0.005,
        max_residual_d1_rad=0.0015,
        max_residual_d2_rad=0.001,
        max_direction_cone_deg=15.0,
    )

    assert metrics["rank1_residual_contract"] is True
    assert metrics["direction_cone_contract"] is True
    assert metrics["rank1_fit_error_abs_max_rad"] < 1e-8
    assert metrics["rank1_endpoint_abs_max_rad"] == 0.0


def test_rank1_directional_contract_rejects_one_malformed_chunk() -> None:
    reference = np.zeros((2, 10, 7), dtype=np.float32)
    residual = np.zeros_like(reference)
    residual[1, 3, 2] = 1e-3

    metrics = validator._rank1_directional_contract_metrics(
        reference,
        residual,
        max_rank1_fit_error_rad=1e-5,
        max_residual_abs_rad=0.005,
        max_residual_d1_rad=0.0015,
        max_residual_d2_rad=0.001,
        max_direction_cone_deg=15.0,
    )

    assert metrics["rank1_fit_error_abs_max_rad"] > 1e-5
    assert metrics["rank1_residual_contract"] is False


def test_optional_contracts_do_not_change_default_acceptance() -> None:
    report = _passing_report()
    report["active_normalized_residual_temporal_step_contract"] = False

    assert validator._acceptance_passed(report, min_action_sensitivity=1e-7)
    assert not validator._acceptance_passed(
        report,
        min_action_sensitivity=1e-7,
        optional_contracts=("active_normalized_residual_temporal_step_contract",),
    )


def test_reference_inclusive_absolute_command_metrics_are_diagnostic_only() -> None:
    report = _passing_report()
    report.update(
        {
            "active_normalized_residual_temporal_step_contract": True,
            "chunk_boundary_residual_jump_contract": True,
            # A fast or replanned Pi0.5 reference can make these false even for
            # a zero-residual Actor.  They must not veto residual promotion.
            "actor_command_temporal_d1_p95_contract": False,
            "chunk_boundary_actor_command_d1_p95_contract": False,
        }
    )

    assert validator._acceptance_passed(
        report,
        min_action_sensitivity=1e-7,
        optional_contracts=(
            "active_normalized_residual_temporal_step_contract",
            "chunk_boundary_residual_jump_contract",
        ),
        require_legacy_normalized_temporal_contract=False,
    )


def test_explicit_active_gate_replaces_legacy_temporal_gate() -> None:
    report = _passing_report()
    report["normalized_residual_temporal_step_contract"] = False
    report["active_normalized_residual_temporal_step_contract"] = True

    assert not validator._acceptance_passed(report, min_action_sensitivity=1e-7)
    assert validator._acceptance_passed(
        report,
        min_action_sensitivity=1e-7,
        optional_contracts=("active_normalized_residual_temporal_step_contract",),
        require_legacy_normalized_temporal_contract=False,
    )


def test_sequence_metrics_integrate_new_report_only_metrics_without_new_gate() -> None:
    class FakeLearner:
        def __init__(self, residual: np.ndarray) -> None:
            self.residual = residual
            self.residual_limit = np.full((10, 7), 0.1, dtype=np.float32)
            self.residual_limit[:, -1] = 0.0

        def act(
            self, z_rl: np.ndarray, state: np.ndarray, a_ref: np.ndarray
        ) -> np.ndarray:
            del z_rl, state
            return np.asarray(a_ref, dtype=np.float32) + self.residual[: len(a_ref)]

    residual = np.zeros((2, 10, 7), dtype=np.float32)
    residual[0, :, 0] = np.arange(10, dtype=np.float32) * 0.01
    residual[1, :, 0] = 0.02 + np.arange(10, dtype=np.float32) * 0.01
    replay = {
        "episode_id": np.asarray(["episode", "episode"]),
        "t": np.asarray([0, 10]),
        "z_rl": np.zeros((2, 4), dtype=np.float32),
        "state": np.zeros((2, 7), dtype=np.float32),
        "a_ref": np.zeros((2, 10, 7), dtype=np.float32),
        "a_exec": np.zeros((2, 10, 7), dtype=np.float32),
        "a_ref_absolute": np.zeros((2, 10, 7), dtype=np.float32),
        "a_exec_absolute": np.zeros((2, 10, 7), dtype=np.float32),
        "step_mask": np.ones((2, 10), dtype=np.bool_),
        "done": np.zeros(2, dtype=np.bool_),
    }

    metrics = validator._sequence_metrics(
        replay,
        FakeLearner(residual),
        max_normalized_step_p95=0.5,
    )

    assert "cross_transition_residual_step_contract" in metrics
    assert metrics["active_normalized_residual_temporal_step_contract"] is None
    assert metrics["actor_command_temporal_d1_p95_contract"] is None
    assert metrics["chunk_boundary_residual_jump_contract"] is None
    assert metrics["chunk_boundary_pair_count"] == 1
    assert metrics["active_normalized_residual_temporal_step_mask_used"] is True


def _persistent_candidate_replay(sample_count: int = 2) -> dict[str, np.ndarray]:
    chunk = (sample_count, 10, 7)
    replay = {
        "episode_id": np.repeat("episode", sample_count),
        "t": np.arange(sample_count, dtype=np.int64) * 10,
        "z_rl": np.zeros((sample_count, 4), dtype=np.float32),
        "state": np.zeros((sample_count, 7), dtype=np.float32),
        "a_ref": np.full(chunk, 0.25, dtype=np.float32),
        "a_exec": np.full(chunk, 0.25, dtype=np.float32),
        "a_ref_absolute": np.full(chunk, 0.25, dtype=np.float32),
        "a_exec_absolute": np.full(chunk, 0.25, dtype=np.float32),
        "a_base_filtered": np.full(chunk, 0.20, dtype=np.float32),
        "actor_persistent_carry_in": np.zeros((sample_count, 7), dtype=np.float32),
        "actor_persistent_previous_carry": np.zeros(
            (sample_count, 7), dtype=np.float32
        ),
        "actor_execution_boundary_anchor": np.zeros(
            (sample_count, 7), dtype=np.float32
        ),
        "execution_filter_alpha": np.full(
            (sample_count, 10), 0.5, dtype=np.float32
        ),
        "step_mask": np.ones((sample_count, 10), dtype=np.bool_),
        "done": np.zeros(sample_count, dtype=np.bool_),
    }
    replay["a_base_filtered"][..., -1] = replay["a_ref"][..., -1]
    return replay


def test_persistent_sequence_metrics_use_filtered_physical_candidate() -> None:
    class Config:
        actor_execution_profile = validator.PERSISTENT_ACTOR_EXECUTION_PROFILE
        actor_residual_max_rad = 0.005
        freeze_gripper_residual = True

    class FakeLearner:
        config = Config()
        residual_limit = np.full((10, 7), 0.005, dtype=np.float32)

        def act(self, *args, **kwargs):
            raise AssertionError("raw learner.act must not drive persistent metrics")

        def persistent_candidate_action(
            self,
            z_rl,
            state,
            a_ref,
            *,
            a_base_filtered,
            carry_in,
            previous_carry,
            boundary_anchor,
            filter_alpha,
            **kwargs,
        ):
            del (
                z_rl,
                state,
                a_ref,
                carry_in,
                previous_carry,
                boundary_anchor,
                filter_alpha,
                kwargs,
            )
            residual = np.zeros_like(a_base_filtered)
            residual[..., 0] = 0.001
            return a_base_filtered + residual

    replay = _persistent_candidate_replay()
    metrics = validator._sequence_metrics(
        replay,
        FakeLearner(),
        max_normalized_step_p95=1.0,
    )

    assert metrics["sequence_action_finite"] is True
    assert metrics["residual_temporal_d1_abs_max_per_joint"][0] == pytest.approx(0.0)
    assert metrics["chunk_boundary_pair_count"] == 1


def test_persistent_td_candidate_uses_next_execution_state_and_target_actor() -> None:
    class Config:
        actor_execution_profile = validator.PERSISTENT_ACTOR_EXECUTION_PROFILE
        actor_residual_max_rad = 0.005
        freeze_gripper_residual = True

    captured: dict[str, object] = {}

    class FakeLearner:
        config = Config()

        def act(self, *args, **kwargs):
            raise AssertionError("raw learner.act must not drive persistent TD targets")

        def persistent_candidate_action(
            self,
            z_rl,
            state,
            a_ref,
            *,
            a_base_filtered,
            carry_in,
            previous_carry,
            boundary_anchor,
            filter_alpha,
            use_target,
            reference_visible,
        ):
            captured.update(
                {
                    "z_rl": z_rl,
                    "state": state,
                    "a_ref": a_ref,
                    "base": a_base_filtered,
                    "carry_in": carry_in,
                    "previous_carry": previous_carry,
                    "boundary_anchor": boundary_anchor,
                    "filter_alpha": filter_alpha,
                    "use_target": use_target,
                    "reference_visible": reference_visible,
                }
            )
            return a_base_filtered

    replay = _persistent_candidate_replay()
    sample_count = len(replay["z_rl"])
    replay.update(
        {
            "next_z_rl": np.full_like(replay["z_rl"], 1.0),
            "next_state": np.full_like(replay["state"], 2.0),
            "next_a_ref": np.full_like(replay["a_ref"], 3.0),
            "next_a_base_filtered": np.full_like(
                replay["a_base_filtered"], 4.0
            ),
            "next_actor_persistent_carry_in": np.full(
                (sample_count, 7), 5.0, dtype=np.float32
            ),
            "next_actor_persistent_previous_carry": np.full(
                (sample_count, 7), 6.0, dtype=np.float32
            ),
            "next_actor_execution_boundary_anchor": np.full(
                (sample_count, 7), 7.0, dtype=np.float32
            ),
            "next_execution_filter_alpha": np.full(
                (sample_count, 10), 0.8, dtype=np.float32
            ),
        }
    )

    candidate = validator._candidate_action(
        FakeLearner(), replay, use_target=True, next_state=True
    )

    np.testing.assert_array_equal(candidate, replay["next_a_base_filtered"])
    np.testing.assert_array_equal(captured["z_rl"], replay["next_z_rl"])
    np.testing.assert_array_equal(captured["state"], replay["next_state"])
    np.testing.assert_array_equal(captured["a_ref"], replay["next_a_ref"])
    np.testing.assert_array_equal(
        captured["carry_in"], replay["next_actor_persistent_carry_in"]
    )
    np.testing.assert_array_equal(
        captured["previous_carry"], replay["next_actor_persistent_previous_carry"]
    )
    np.testing.assert_array_equal(
        captured["boundary_anchor"], replay["next_actor_execution_boundary_anchor"]
    )
    np.testing.assert_array_equal(
        captured["filter_alpha"], replay["next_execution_filter_alpha"]
    )
    assert captured["use_target"] is True
    assert captured["reference_visible"] is True


def test_persistent_candidate_fails_closed_when_execution_arrays_are_missing() -> None:
    class Config:
        actor_execution_profile = validator.PERSISTENT_ACTOR_EXECUTION_PROFILE
        actor_residual_max_rad = 0.005
        freeze_gripper_residual = True

    class FakeLearner:
        config = Config()
        residual_limit = np.full((10, 7), 0.005, dtype=np.float32)

    replay = _persistent_candidate_replay()
    replay.pop("execution_filter_alpha")

    with pytest.raises(KeyError, match="physical candidate arrays"):
        validator._sequence_metrics(
            replay,
            FakeLearner(),
            max_normalized_step_p95=1.0,
        )
