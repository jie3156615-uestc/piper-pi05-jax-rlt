from __future__ import annotations

import numpy as np
import pytest

from piper_runtime.rlt_residual_governor import ActorResidualGovernor
from piper_runtime.rlt_residual_governor import ActorResidualGovernorConfig
from piper_runtime.rlt_residual_governor import GRIPPER_RESIDUAL_CLOSE_ASSIST
from piper_runtime.rlt_residual_governor import PersistentActorResidualGovernor
from piper_runtime.rlt_residual_governor import PERSISTENT_C10_EXECUTION_CONTRACT
from piper_runtime.rlt_residual_governor import RANK1_BUMP_WINDOW
from piper_runtime.rlt_residual_governor import govern_actor_plan
from piper_runtime.rlt_residual_governor import govern_persistent_actor_plan


def _moving_reference(step: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    anchor = np.zeros(7, dtype=np.float32)
    reference = np.zeros((10, 7), dtype=np.float32)
    reference[:, 0] = np.arange(1, 11, dtype=np.float32) * step
    reference[:, 6] = 0.04
    return anchor, reference


def _rank1_actor(reference: np.ndarray, direction: np.ndarray) -> np.ndarray:
    actor = reference.copy()
    actor[:, :6] += RANK1_BUMP_WINDOW[:, None] * np.asarray(direction)[None, :]
    return actor


def _govern(reference: np.ndarray, actor: np.ndarray, anchor: np.ndarray):
    return govern_actor_plan(
        plan_id="actor-1",
        behavior_plan_id="behavior-1",
        behavior_start_offset=10,
        behavior_ref=reference,
        raw_actor=actor,
        boundary_anchor=anchor,
    )


def test_exact_rank1_bump_is_cached_with_zero_endpoints_and_frozen_gripper():
    anchor, reference = _moving_reference()
    actor = _rank1_actor(reference, np.array([0.0, 0.002, 0.0, 0.0, 0.0, 0.0]))

    plan = _govern(reference, actor, anchor)

    assert plan.approved is True
    assert plan.projection_scale == 1.0
    np.testing.assert_allclose(plan.safe_residual[0], 0.0, atol=0.0)
    np.testing.assert_allclose(plan.safe_residual[-1], 0.0, atol=0.0)
    np.testing.assert_allclose(plan.safe_actions[:, 6], reference[:, 6], atol=0.0)
    assert plan.safe_residual_max_rad <= 0.005
    assert plan.safe_residual_d1_max_rad <= 0.0015
    assert plan.safe_residual_d2_max_rad <= 0.001
    assert plan.direction_violation_count == 0


def test_one_global_grid_scale_enforces_all_residual_limits():
    anchor, reference = _moving_reference()
    actor = _rank1_actor(reference, np.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0]))

    plan = _govern(reference, actor, anchor)

    assert plan.approved is True
    assert plan.projection_scale == pytest.approx(0.5)
    assert plan.safe_residual_max_rad <= 0.005 + 1e-9
    assert plan.safe_residual_d1_max_rad <= 0.0015 + 1e-9
    assert plan.safe_residual_d2_max_rad <= 0.001 + 1e-9


def test_projection_below_point_two_rejects_complete_chunk():
    anchor, reference = _moving_reference(step=0.001)
    actor = _rank1_actor(reference, np.array([-0.02, 0.0, 0.0, 0.0, 0.0, 0.0]))

    plan = _govern(reference, actor, anchor)

    assert plan.approved is False
    assert plan.rejection_reason == "projection_scale_below_minimum"
    assert plan.projection_scale < 0.2
    np.testing.assert_allclose(plan.safe_actions, reference, atol=0.0)
    np.testing.assert_allclose(plan.safe_residual, 0.0, atol=0.0)


def test_boundary_jump_above_limit_rejects_complete_chunk_even_for_zero_residual():
    anchor, reference = _moving_reference()
    reference[:, 1] = 0.03
    actor = reference.copy()

    plan = _govern(reference, actor, anchor)

    assert plan.approved is False
    assert plan.rejection_reason == "boundary_jump_exceeds_limit"
    assert plan.boundary_jump_max_rad == pytest.approx(0.03)
    assert plan.boundary_jump_limit_rad == pytest.approx(0.02)
    np.testing.assert_allclose(plan.safe_actions, reference, atol=0.0)


def test_non_rank1_legacy_actor_is_rejected_instead_of_fitted_and_executed():
    anchor, reference = _moving_reference()
    actor = _rank1_actor(reference, np.array([0.0, 0.002, 0.0, 0.0, 0.0, 0.0]))
    actor[3, 2] += 0.001

    plan = _govern(reference, actor, anchor)

    assert plan.approved is False
    assert plan.rejection_reason == "rank1_contract_violation"
    np.testing.assert_allclose(plan.safe_actions, reference, atol=0.0)


def test_nonzero_gripper_residual_rejects_complete_chunk():
    anchor, reference = _moving_reference()
    actor = _rank1_actor(reference, np.array([0.0, 0.002, 0.0, 0.0, 0.0, 0.0]))
    actor[:, 6] += 1e-3

    plan = _govern(reference, actor, anchor)

    assert plan.approved is False
    assert plan.rejection_reason == "gripper_residual_not_frozen"
    np.testing.assert_allclose(plan.safe_actions, reference, atol=0.0)


def _close_assist_config() -> ActorResidualGovernorConfig:
    return ActorResidualGovernorConfig(
        max_boundary_jump_rad=0.06,
        gripper_residual_mode=GRIPPER_RESIDUAL_CLOSE_ASSIST,
        gripper_residual_max_close_m=0.005,
        gripper_residual_d1_max_m=0.0005,
        gripper_residual_d2_max_m=0.0003,
        gripper_max_boundary_jump_m=0.0005,
    )


def _close_assist_plan(
    *,
    reference_gripper: np.ndarray | float = 0.02,
    gripper_direction: float = -0.005,
    carry: float = 0.0,
    previous: float = 0.0,
):
    anchor, reference = _moving_reference(step=0.002)
    reference[:, 6] = reference_gripper
    anchor[6] = float(reference[0, 6]) + carry
    raw = _rank1_actor(
        reference,
        np.asarray([0.0, 0.001, 0.0, 0.0, 0.0, 0.0]),
    )
    raw[:, 6] += RANK1_BUMP_WINDOW * gripper_direction
    carry_vector = np.zeros(7, dtype=np.float32)
    carry_vector[6] = carry
    previous_vector = np.zeros(7, dtype=np.float32)
    previous_vector[6] = previous
    return govern_persistent_actor_plan(
        plan_id="gripper-close",
        behavior_plan_id="behavior",
        behavior_start_offset=0,
        behavior_ref=reference,
        raw_actor=raw,
        boundary_anchor=anchor,
        carry_in=carry_vector,
        previous_carry=previous_vector,
        config=_close_assist_config(),
    )


def test_close_assist_accepts_bounded_negative_rank1_gripper_knot() -> None:
    plan = _close_assist_plan()

    assert plan.approved
    assert plan.gripper_residual_mode == GRIPPER_RESIDUAL_CLOSE_ASSIST
    assert plan.safe_residual[0, 6] == pytest.approx(0.0, abs=1e-9)
    assert plan.safe_residual[-1, 6] < 0.0
    assert np.max(plan.safe_residual[:, 6]) <= 1e-9
    assert plan.safe_gripper_residual_max_m <= 0.005 + 1e-9
    assert plan.safe_gripper_d1_max_m <= 0.0005 + 1e-9
    assert plan.safe_gripper_d2_max_m <= 0.0003 + 1e-9
    assert np.min(plan.safe_actions[:, 6]) >= 0.0
    assert np.max(plan.safe_actions[:, 6]) <= 0.08


def test_close_assist_rejects_actor_opening_residual() -> None:
    plan = _close_assist_plan(gripper_direction=0.001)

    assert plan.approved
    assert not plan.target_update_accepted
    assert plan.rejection_reason == "gripper_open_residual_forbidden"
    np.testing.assert_allclose(plan.safe_residual[:, 6], 0.0, atol=0.0)


def test_close_assist_release_reference_monotonically_removes_carry() -> None:
    opening = np.linspace(0.02, 0.065, 10, dtype=np.float32)
    plan = _close_assist_plan(
        reference_gripper=opening,
        gripper_direction=-0.005,
        carry=-0.002,
        previous=-0.002,
    )

    assert plan.approved
    assert plan.gripper_release_intent
    assert plan.safe_residual[0, 6] == pytest.approx(-0.002, abs=1e-8)
    assert plan.safe_residual[-1, 6] > plan.safe_residual[0, 6]
    assert np.all(np.diff(plan.safe_residual[:, 6]) >= -1e-9)


def test_close_assist_absolute_range_projects_near_zero_base() -> None:
    plan = _close_assist_plan(
        reference_gripper=0.002,
        gripper_direction=-0.005,
    )

    assert plan.approved
    assert np.min(plan.safe_actions[:, 6]) >= -1e-9
    assert np.min(plan.safe_residual[:, 6]) >= -0.002 - 1e-9


def test_near_static_reference_uses_actor_step_ball():
    anchor = np.zeros(7, dtype=np.float32)
    reference = np.zeros((10, 7), dtype=np.float32)
    actor = _rank1_actor(reference, np.array([0.005, 0.0, 0.0, 0.0, 0.0, 0.0]))

    plan = _govern(reference, actor, anchor)

    assert plan.approved is True
    assert plan.projection_scale == pytest.approx(0.65625)
    safe_steps = np.diff(
        np.concatenate([anchor[None, :6], plan.safe_actions[:, :6]], axis=0),
        axis=0,
    )
    assert np.max(np.linalg.norm(safe_steps, axis=1)) <= 0.001 + 1e-12


def test_governor_returns_same_cached_plan_and_rejects_plan_id_reuse():
    anchor, reference = _moving_reference()
    actor = _rank1_actor(reference, np.array([0.0, 0.002, 0.0, 0.0, 0.0, 0.0]))
    governor = ActorResidualGovernor()
    kwargs = dict(
        plan_id="actor-1",
        behavior_plan_id="behavior-1",
        behavior_start_offset=10,
        behavior_ref=reference,
        raw_actor=actor,
        boundary_anchor=anchor,
    )

    first = governor.prepare_plan(**kwargs)
    second = governor.prepare_plan(**kwargs)

    assert second is first
    with pytest.raises(ValueError, match="reused"):
        governor.prepare_plan(**{**kwargs, "behavior_start_offset": 20})


def test_persistent_compatibility_layer_keeps_legacy_c10_checkpoint_shape() -> None:
    anchor, reference = _moving_reference()
    actor = _rank1_actor(reference, np.array([0.0, 0.002, 0.0, 0.0, 0.0, 0.0]))
    governor = PersistentActorResidualGovernor()

    plan = governor.prepare_plan(
        plan_id="persistent-1",
        behavior_plan_id="behavior-1",
        behavior_start_offset=0,
        behavior_ref=reference,
        raw_actor=actor,
        boundary_anchor=anchor,
    )

    assert plan.approved is True
    assert plan.target_update_accepted is True
    assert plan.safe_actions.shape == (10, 7)
    assert plan.raw_residual.shape == (10, 7)
    # Unlike the legacy bump execution, the learned direction is retained at
    # the terminal knot instead of being forced back to zero.
    # The execution blend starts exactly at the actually executed carry, so
    # phase entry and each C10 boundary are position-continuous.
    assert plan.safe_residual[0, 1] == 0.0
    assert plan.safe_residual[1, 1] > 0.0
    assert plan.safe_residual[-1, 1] == pytest.approx(0.002)
    assert plan.metadata()["actor_execution_contract"] == PERSISTENT_C10_EXECUTION_CONTRACT
    assert plan.metadata()["actor_governor_legacy_input_contract"] == "rank1_bump_v1"
    np.testing.assert_allclose(plan.safe_actions[:, 6], reference[:, 6], atol=0.0)


def test_persistent_carry_is_committed_only_for_executed_offsets_and_resets() -> None:
    anchor, reference = _moving_reference()
    actor = _rank1_actor(reference, np.array([0.0, 0.002, 0.0, 0.0, 0.0, 0.0]))
    governor = PersistentActorResidualGovernor()
    plan = governor.prepare_plan(
        plan_id="persistent-1",
        behavior_plan_id="behavior-1",
        behavior_start_offset=0,
        behavior_ref=reference,
        raw_actor=actor,
        boundary_anchor=anchor,
    )

    np.testing.assert_allclose(governor.current_residual, 0.0)
    for offset in range(10):
        committed = governor.mark_executed(plan.plan_id, offset)
        np.testing.assert_allclose(committed, plan.safe_residual[offset])
    np.testing.assert_allclose(governor.current_residual, plan.safe_residual[-1])
    np.testing.assert_allclose(governor.previous_residual, plan.safe_residual[-2])

    governor.reset_execution_state("human_takeover")

    np.testing.assert_allclose(governor.current_residual, 0.0)
    np.testing.assert_allclose(governor.previous_residual, 0.0)
    assert governor.last_reset_reason == "human_takeover"


def test_persistent_five_c10_knots_cover_full_h50_without_endpoint_drop() -> None:
    governor = PersistentActorResidualGovernor()
    behavior = np.zeros((50, 7), dtype=np.float32)
    behavior[:, 0] = np.arange(1, 51, dtype=np.float32) * 0.006
    behavior[:, 6] = 0.04
    directions = [0.001, 0.002, 0.003, 0.002, 0.001]
    boundary = np.zeros(7, dtype=np.float32)
    all_residuals: list[np.ndarray] = []
    previous_chunk_last: np.ndarray | None = None

    for chunk_index, direction in enumerate(directions):
        start = chunk_index * 10
        reference = behavior[start : start + 10]
        actor = _rank1_actor(
            reference,
            np.array([0.0, direction, 0.0, 0.0, 0.0, 0.0]),
        )
        plan = governor.prepare_plan(
            plan_id=f"persistent-{chunk_index}",
            behavior_plan_id="behavior-h50",
            behavior_start_offset=start,
            behavior_ref=reference,
            raw_actor=actor,
            boundary_anchor=boundary,
        )
        assert plan.approved is True
        assert plan.target_update_accepted is True
        if chunk_index == 0:
            assert plan.safe_residual[0, 1] == 0.0
            assert np.all(np.abs(plan.safe_residual[1:, 1]) > 0.0)
        else:
            assert np.all(np.abs(plan.safe_residual[:, 1]) > 0.0)
        if previous_chunk_last is not None:
            assert abs(float(plan.safe_residual[0, 1] - previous_chunk_last[1])) < 2e-5
        for offset in range(10):
            governor.mark_executed(plan.plan_id, offset)
        boundary = plan.safe_actions[-1].copy()
        previous_chunk_last = plan.safe_residual[-1].copy()
        all_residuals.append(plan.safe_residual.copy())

    residuals = np.concatenate(all_residuals, axis=0)
    assert residuals.shape == (50, 7)
    assert residuals[0, 1] == 0.0
    assert np.all(np.abs(residuals[1:, 1]) > 0.0)
    assert float(np.max(np.abs(residuals[:, :6]))) <= 0.005 + 1e-9
    padded = np.concatenate(
        [
            np.zeros((1, 6), dtype=np.float32),
            residuals[:, :6],
            residuals[-1:, :6],
        ],
        axis=0,
    )
    d1 = np.diff(padded, axis=0)
    d2 = np.diff(d1, axis=0)
    assert float(np.max(np.abs(d1))) <= 0.0015 + 1e-9
    assert float(np.max(np.abs(d2))) <= 0.001 + 1e-9


def test_bad_new_actor_holds_committed_carry_instead_of_dropping_to_pi05() -> None:
    governor = PersistentActorResidualGovernor()
    anchor, reference = _moving_reference(step=0.006)
    first_actor = _rank1_actor(
        reference,
        np.array([0.0, 0.002, 0.0, 0.0, 0.0, 0.0]),
    )
    first = governor.prepare_plan(
        plan_id="persistent-1",
        behavior_plan_id="behavior-1",
        behavior_start_offset=0,
        behavior_ref=reference,
        raw_actor=first_actor,
        boundary_anchor=anchor,
    )
    for offset in range(10):
        governor.mark_executed(first.plan_id, offset)

    next_reference = reference.copy()
    next_reference[:, 0] += 0.06
    bad_actor = next_reference.copy()
    bad_actor[3, 2] += 0.001  # Not an exact legacy rank1-bump output.
    held = governor.prepare_plan(
        plan_id="persistent-2",
        behavior_plan_id="behavior-1",
        behavior_start_offset=10,
        behavior_ref=next_reference,
        raw_actor=bad_actor,
        boundary_anchor=first.safe_actions[-1],
    )

    assert held.approved is True
    assert held.target_update_accepted is False
    assert held.hold_only is True
    assert held.rejection_reason == "rank1_contract_violation"
    np.testing.assert_allclose(
        held.safe_residual[:, :6],
        np.repeat(first.safe_residual[-1:, :6], 10, axis=0),
    )


def test_persistent_hold_plan_retains_carry_for_actor_only_latency_gap() -> None:
    governor = PersistentActorResidualGovernor()
    anchor, reference = _moving_reference(step=0.006)
    actor = _rank1_actor(reference, np.array([0.0, 0.002, 0.0, 0.0, 0.0, 0.0]))
    first = governor.prepare_plan(
        plan_id="persistent-1",
        behavior_plan_id="behavior-1",
        behavior_start_offset=0,
        behavior_ref=reference,
        raw_actor=actor,
        boundary_anchor=anchor,
    )
    for offset in range(10):
        governor.mark_executed(first.plan_id, offset)
    next_reference = reference.copy()
    next_reference[:, 0] += 0.06

    held = governor.prepare_hold_plan(
        plan_id="hold-1",
        behavior_plan_id="behavior-1",
        behavior_start_offset=10,
        behavior_ref=next_reference,
        boundary_anchor=first.safe_actions[-1],
    )

    assert held.approved is True
    assert held.hold_only is True
    assert held.target_update_accepted is False
    np.testing.assert_allclose(held.safe_residual[-1], governor.current_residual)


def test_filtered_actual_commit_uses_long_lived_base_shadow_and_does_not_decay_carry() -> None:
    governor = PersistentActorResidualGovernor()
    anchor, reference = _moving_reference(step=0.006)
    actor = _rank1_actor(
        reference,
        np.array([0.0, 0.002, 0.0, 0.0, 0.0, 0.0]),
    )
    plan = governor.prepare_plan(
        plan_id="filtered-1",
        behavior_plan_id="behavior-1",
        behavior_start_offset=0,
        behavior_ref=reference,
        raw_actor=actor,
        boundary_anchor=anchor,
    )
    alpha = 1.0 - np.exp(-(1.0 / 30.0) / 0.05)
    base_state = anchor.astype(np.float64)
    actual_state = anchor.astype(np.float64)
    expected_residual = np.zeros(7, dtype=np.float64)
    for offset in range(10):
        base_anchor = base_state.copy()
        actual_anchor = actual_state.copy()
        base_state[:6] += alpha * (
            reference[offset, :6] - base_state[:6]
        )
        actual_state[:6] += alpha * (
            plan.safe_actions[offset, :6] - actual_state[:6]
        )
        base_state[6] = reference[offset, 6]
        actual_state[6] = plan.safe_actions[offset, 6]
        expected_residual = (
            (1.0 - alpha) * expected_residual
            + alpha * plan.safe_residual[offset]
        )
        certificate = governor.certify_filtered_execution(
            plan_id=plan.plan_id,
            offset=offset,
            filtered_base_action=base_state,
            filtered_actual_action=actual_state,
            base_boundary_anchor=base_anchor,
            actual_boundary_anchor=actual_anchor,
        )
        assert certificate.approved is True
        committed = governor.mark_filtered_executed(certificate)
        np.testing.assert_allclose(committed, expected_residual, atol=1e-8)

    # A persistent hold whose target residual equals the physical carry must
    # remain constant. A per-frame same-prestate clone would incorrectly
    # report alpha*carry and fail this assertion.
    next_reference = reference.copy()
    next_reference[:, 0] += 0.06
    hold = governor.prepare_hold_plan(
        plan_id="filtered-hold",
        behavior_plan_id="behavior-1",
        behavior_start_offset=10,
        behavior_ref=next_reference,
        boundary_anchor=actual_state,
    )
    carry_before_hold = governor.current_residual.copy()
    for offset in range(10):
        base_anchor = base_state.copy()
        actual_anchor = actual_state.copy()
        base_state[:6] += alpha * (
            next_reference[offset, :6] - base_state[:6]
        )
        actual_state[:6] += alpha * (
            hold.safe_actions[offset, :6] - actual_state[:6]
        )
        base_state[6] = next_reference[offset, 6]
        actual_state[6] = hold.safe_actions[offset, 6]
        certificate = governor.certify_filtered_execution(
            plan_id=hold.plan_id,
            offset=offset,
            filtered_base_action=base_state,
            filtered_actual_action=actual_state,
            base_boundary_anchor=base_anchor,
            actual_boundary_anchor=actual_anchor,
        )
        assert certificate.approved is True
        governor.mark_filtered_executed(certificate)
        np.testing.assert_allclose(
            governor.current_residual,
            carry_before_hold,
            atol=1e-8,
        )


def test_filtered_actual_certificate_rejects_post_filter_constraint_violation() -> None:
    anchor, reference = _moving_reference(step=0.006)
    actor = _rank1_actor(
        reference,
        np.array([0.0, 0.002, 0.0, 0.0, 0.0, 0.0]),
    )
    governor = PersistentActorResidualGovernor()
    plan = governor.prepare_plan(
        plan_id="filtered-reject",
        behavior_plan_id="behavior-1",
        behavior_start_offset=0,
        behavior_ref=reference,
        raw_actor=actor,
        boundary_anchor=anchor,
    )
    filtered_base = reference[0].copy()
    filtered_actual = filtered_base.copy()
    filtered_actual[1] += 0.01
    certificate = governor.certify_filtered_execution(
        plan_id=plan.plan_id,
        offset=0,
        filtered_base_action=filtered_base,
        filtered_actual_action=filtered_actual,
        base_boundary_anchor=anchor,
        actual_boundary_anchor=anchor,
    )
    assert certificate.approved is False
    assert "residual_max" in certificate.rejection_reason
    with pytest.raises(ValueError, match="rejected"):
        governor.mark_filtered_executed(certificate)
    np.testing.assert_allclose(governor.current_residual, 0.0)
