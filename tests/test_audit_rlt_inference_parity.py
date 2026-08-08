from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_CANDIDATES = (
    ROOT / "remote_piper_runtime" / "scripts" / "audit_rlt_inference_parity.py",
    ROOT / "scripts" / "audit_rlt_inference_parity.py",
)
SCRIPT = next((path for path in SCRIPT_CANDIDATES if path.is_file()), SCRIPT_CANDIDATES[0])
SPEC = importlib.util.spec_from_file_location("audit_rlt_inference_parity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
audit_rlt_inference_parity = MODULE.audit_rlt_inference_parity


def _row(
    *,
    t: int,
    model_index: int,
    actor_offset: int,
    source: str = "rlt",
    phase: bool = True,
    request_reason: str = "replay_enrichment_prefetch",
    prefetch_hit: bool = True,
) -> dict:
    reference = np.zeros((10, 7), dtype=np.float32)
    residual = np.full((10, 7), 0.001, dtype=np.float32)
    residual[:, 6] = 0.0
    actor = reference + residual
    actor_only = request_reason in {
        "actor_prefetch",
        "replay_enrichment",
        "replay_enrichment_prefetch",
        "actor_only",
        "actor_only_prefetch",
    }
    return {
        "t": t,
        "source": source,
        "gate_active": phase,
        "done": False,
        "keyboard": {"mode": "MODEL"},
        "a_ref": reference.tolist(),
        "a_actor": actor.tolist(),
        "a_actor_safe": actor.tolist(),
        "a_exec": actor[actor_offset].tolist(),
        "policy_metadata": {
            "model_action_index": model_index,
            "plan_offset": actor_offset,
            "policy_request_reason": request_reason,
            "policy_prefetch_hit": prefetch_hit,
            "policy_boundary_wait_s": 0.0,
            "h50_prefetch_accepted": True,
            "actor_shadow_ready": True,
            "actor_governor_safe_residual_this_step": residual[
                actor_offset
            ].tolist(),
            "actor_governor_raw_residual_this_step": residual[
                actor_offset
            ].tolist(),
            "actor_execution_contract": "persistent_c10_from_rank1_v1",
            "actor_governor_legacy_input_contract": "rank1_bump_v1",
            "actor_persistent_carry_in": residual[0].tolist(),
            "actor_persistent_carry_out": residual[-1].tolist(),
            "actor_persistent_target_update_accepted": True,
            "actor_persistent_hold": False,
            "actor_physical_execution_committed_this_step": True,
            "actor_only_mode": actor_only,
            "actor_only_protocol": (
                "actor_only_v1" if actor_only else None
            ),
            "actor_only_base_policy_called": not actor_only,
            "actor_only_base_rng_advanced": not actor_only,
            "actor_only_token_encoder_called": not actor_only,
            "actor_only_actor_called": actor_only,
            "behavior_actor_checkpoint": "checkpoint:/legacy/step_00012447",
            "action_schema_fingerprint": "legacy_rank1_c10",
            "command_publisher_hz": 50.0,
            "command_publisher_count": 2 * t,
        },
    }


def _two_h50_rows() -> list[dict]:
    rows = []
    for t in range(100):
        model_index = t % 50
        row = _row(
            t=t,
            model_index=model_index,
            actor_offset=model_index % 10,
            request_reason=(
                "initial"
                if t == 0
                else "behavior_prefetch"
                if t == 50
                else "replay_enrichment_prefetch"
            ),
        )
        rows.append(row)
    return rows


def test_full_h50_actor_coverage_accepts_legacy_c10_checkpoint() -> None:
    report = audit_rlt_inference_parity(
        _two_h50_rows(),
        expected_checkpoint="step_00012447",
    )

    assert report["passed"] is True
    assert report["h50"]["prefetch_hits"] == 1
    assert report["actor"]["phase_coverage"] == 1.0
    assert report["actor"]["legacy_c10_payload_compatible"] is True
    assert report["actor"]["persistent_contract_rows"] == 100
    # The first C10 in each H50 arrives with the base-policy response.  The
    # remaining four C10 slices per H50 use isolated Actor-only refreshes.
    assert report["actor"]["actor_only_rng_isolated_rows"] == 8
    assert [
        item["coverage"] for item in report["actor"]["coverage_by_h50_slice"]
    ] == [1.0] * 5
    assert report["publisher"]["has_50hz_evidence"] is True


def test_boundary_hold_and_missed_prefetch_are_reported() -> None:
    rows = _two_h50_rows()
    hold = copy.deepcopy(rows[49])
    hold["t"] = 50
    hold["source"] = "safety_block"
    hold["policy_metadata"]["model_action_index"] = 50
    rows.insert(50, hold)
    activation = rows[51]
    activation["t"] = 51
    activation["policy_metadata"]["policy_request_reason"] = "boundary_miss"
    activation["policy_metadata"]["policy_prefetch_hit"] = False
    activation["policy_metadata"]["policy_boundary_wait_s"] = 0.18

    report = audit_rlt_inference_parity(
        rows,
        require_50hz_evidence=False,
    )

    assert report["passed"] is False
    assert report["h50"]["prefetch_hits"] == 0
    assert report["h50"]["reports"][0]["preceding_safety_block_t"] == [50]
    assert any("safety_block rows preceded" in item for item in report["violations"])
    assert any("boundary wait" in item for item in report["violations"])


def test_missing_first_and_last_actor_slices_fail_full_coverage() -> None:
    rows = _two_h50_rows()
    for row in rows:
        model_index = row["policy_metadata"]["model_action_index"]
        if model_index < 10 or model_index >= 40:
            row["source"] = "pi05"

    report = audit_rlt_inference_parity(
        rows,
        require_50hz_evidence=False,
    )

    assert report["passed"] is False
    assert report["actor"]["phase_coverage"] == 0.6
    coverage = [
        item["coverage"] for item in report["actor"]["coverage_by_h50_slice"]
    ]
    assert coverage == [0.0, 1.0, 1.0, 1.0, 0.0]


def test_actor_c10_boundary_residual_jump_is_checked() -> None:
    rows = _two_h50_rows()
    current = rows[10]
    safe = np.asarray(current["a_actor_safe"], dtype=np.float32)
    safe[0, 0] += 0.01
    current["a_actor_safe"] = safe.tolist()
    current["policy_metadata"]["actor_governor_safe_residual_this_step"][0] += (
        0.01
    )

    report = audit_rlt_inference_parity(
        rows,
        require_50hz_evidence=False,
        max_actor_boundary_residual_jump_rad=0.001,
    )

    assert report["passed"] is False
    assert report["actor"]["residual_boundary_jump_rad_max"] > 0.009
    assert any(
        "Actor residual boundary jump" in item for item in report["violations"]
    )


def test_persistent_hold_rows_need_no_fresh_legacy_payload() -> None:
    rows = _two_h50_rows()
    for t in range(20, 30):
        row = rows[t]
        row["a_actor"] = None
        row["policy_metadata"]["actor_shadow_ready"] = False
        row["policy_metadata"]["actor_persistent_target_update_accepted"] = False
        row["policy_metadata"]["actor_persistent_hold"] = True

    report = audit_rlt_inference_parity(rows)

    assert report["passed"] is True
    assert report["actor"]["persistent_hold_rows"] == 10
    assert report["actor"]["persistent_target_update_rows"] == 90
    assert report["actor"]["legacy_c10_payload_compatible"] is True


def test_all_hold_episode_is_not_claimed_as_effective_actor_correction() -> None:
    rows = _two_h50_rows()
    for row in rows:
        row["a_actor"] = None
        row["policy_metadata"]["actor_shadow_ready"] = False
        row["policy_metadata"]["actor_persistent_target_update_accepted"] = False
        row["policy_metadata"]["actor_persistent_hold"] = True

    report = audit_rlt_inference_parity(rows)

    assert report["passed"] is False
    assert report["actor"]["persistent_target_update_rows"] == 0
    assert report["actor"]["legacy_c10_payload_compatible"] is False
    assert any(
        "accepted no legacy C10 target update" in item
        for item in report["violations"]
    )


def test_simulated_commit_is_not_physical_actor_coverage() -> None:
    rows = _two_h50_rows()
    for row in rows:
        row["policy_metadata"][
            "actor_physical_execution_committed_this_step"
        ] = False
        row["policy_metadata"]["actor_command_delivery_mode"] = (
            "simulated_no_publish"
        )

    report = audit_rlt_inference_parity(rows)

    assert report["passed"] is False
    assert report["actor"]["phase_coverage"] == 1.0
    assert report["actor"]["physical_phase_coverage"] == 0.0
    assert any(
        "simulated commits are not physical execution evidence" in item
        for item in report["violations"]
    )

    simulated_report = audit_rlt_inference_parity(
        rows,
        require_physical_actor_execution=False,
    )
    assert simulated_report["passed"] is True


def test_fresh_token_enrichment_is_base_rng_isolated() -> None:
    rows = _two_h50_rows()
    for row in rows:
        if row["policy_metadata"]["actor_only_mode"] is True:
            row["policy_metadata"]["actor_only_protocol"] = (
                "actor_enrichment_only_v1"
            )
            row["policy_metadata"]["actor_only_token_encoder_called"] = True

    report = audit_rlt_inference_parity(rows)

    assert report["passed"] is True
    assert report["actor"]["actor_only_rng_isolated_rows"] == 8


def test_30hz_episode_rows_without_publisher_telemetry_are_inconclusive() -> None:
    rows = _two_h50_rows()
    for row in rows:
        row["policy_metadata"].pop("command_publisher_hz")
        row["policy_metadata"].pop("command_publisher_count")

    report = audit_rlt_inference_parity(rows)

    assert report["passed"] is False
    assert report["publisher"]["has_50hz_evidence"] is False
    assert any("no verifiable 50 Hz" in item for item in report["violations"])
