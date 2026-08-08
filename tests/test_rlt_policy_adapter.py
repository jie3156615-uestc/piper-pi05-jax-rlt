from __future__ import annotations

import numpy as np
import pytest

from piper_runtime.rlt_policy_adapter import extract_rlt_policy_output


def test_extract_rlt_policy_output_uses_first_chunk_of_raw_actions() -> None:
    actions = np.arange(50 * 8, dtype=np.float32).reshape(50, 8)
    response = {"actions": actions, "z_rl": [0.1, 0.2, 0.3, 0.4]}

    output = extract_rlt_policy_output(response, chunk_length=10, action_dim=7)

    assert output.a_ref.shape == (10, 7)
    np.testing.assert_allclose(output.a_ref, actions[:10, :7])
    np.testing.assert_allclose(output.z_rl, [0.1, 0.2, 0.3, 0.4])
    assert output.a_actor is None
    assert output.metadata["z_rl_source"] == "z_rl"


def test_extract_rlt_policy_output_falls_back_to_zero_latent_with_metadata() -> None:
    response = {"actions": np.zeros((50, 7), dtype=np.float32)}

    output = extract_rlt_policy_output(response, chunk_length=10, action_dim=7, fallback_z_rl_dim=4)

    np.testing.assert_allclose(output.z_rl, np.zeros(4, dtype=np.float32))
    assert output.metadata["z_rl_source"] == "zeros_missing_from_policy_response"


def test_extract_shadow_actor_converts_delta6_and_absolute_gripper() -> None:
    state = np.array([1, 2, 3, 4, 5, 6, 0.02], dtype=np.float32)
    actor_delta = np.zeros((10, 7), dtype=np.float32)
    actor_delta[:, :6] = 0.25
    actor_delta[:, 6] = 0.07
    response = {
        "actions": np.zeros((10, 7), dtype=np.float32),
        "z_rl": np.ones(4, dtype=np.float32),
        "a_actor": actor_delta,
        "a_actor_action_space": "joint_delta6_gripper_absolute",
    }

    output = extract_rlt_policy_output(
        response,
        chunk_length=10,
        action_dim=7,
        state_snapshot=state,
        read_actor=True,
    )

    np.testing.assert_allclose(output.a_actor[:, :6], np.repeat((state[:6] + 0.25)[None, :], 10, axis=0))
    np.testing.assert_allclose(output.a_actor[:, 6], 0.07)
    assert output.metadata["actor_logged_action_space"] == "joint_absolute_gripper_absolute"


def test_sft_execute50_and_actor_c10_are_parsed_as_separate_horizons() -> None:
    actions = np.arange(50 * 7, dtype=np.float32).reshape(50, 7)
    actor = np.arange(10 * 7, dtype=np.float32).reshape(10, 7)

    output = extract_rlt_policy_output(
        {
            "actions": actions,
            "z_rl": np.ones(8, dtype=np.float32),
            "a_actor": actor,
            "a_actor_action_space": "joint_absolute_gripper_absolute",
        },
        chunk_length=50,
        actor_chunk_length=10,
        action_dim=7,
        read_actor=True,
        state_snapshot=np.zeros(7, dtype=np.float32),
    )

    np.testing.assert_allclose(output.a_ref, actions)
    np.testing.assert_allclose(output.a_actor, actor)
    assert output.metadata["chunk_length"] == 50
    assert output.metadata["actor_chunk_length"] == 10


def test_actor_only_audit_evidence_is_flattened_from_shadow_status() -> None:
    output = extract_rlt_policy_output(
        {
            "actions": np.zeros((10, 7), dtype=np.float32),
            "z_rl": np.ones(2048, dtype=np.float32),
            "a_actor": np.zeros((10, 7), dtype=np.float32),
            "rlt_shadow": {
                "mode": "actor_only",
                "actor_only_protocol": "actor_enrichment_only_v1",
                "base_policy_called": False,
                "base_rng_advanced": False,
                "token_encoder_called": False,
                "actor_called": True,
                "shadow_latency_s": 0.012,
            },
        },
        chunk_length=10,
        actor_chunk_length=10,
        action_dim=7,
        read_actor=True,
        state_snapshot=np.zeros(7, dtype=np.float32),
    )

    assert output.metadata["actor_only_mode"] is True
    assert (
        output.metadata["actor_only_protocol"]
        == "actor_enrichment_only_v1"
    )
    assert output.metadata["actor_only_base_policy_called"] is False
    assert output.metadata["actor_only_base_rng_advanced"] is False
    assert output.metadata["actor_only_token_encoder_called"] is False
    assert output.metadata["actor_only_actor_called"] is True
    assert output.metadata["actor_only_latency_s"] == pytest.approx(0.012)


def test_base_only_audit_evidence_is_flattened_from_shadow_status() -> None:
    output = extract_rlt_policy_output(
        {
            "actions": np.zeros((50, 7), dtype=np.float32),
            "rlt_shadow": {
                "mode": "base_only",
                "base_policy_called": True,
                "base_rng_advanced": True,
                "token_encoder_called": False,
                "actor_called": False,
                "base_policy_latency_s": 0.091,
            },
        },
        chunk_length=50,
        actor_chunk_length=10,
        action_dim=7,
        read_actor=True,
        state_snapshot=np.zeros(7, dtype=np.float32),
    )

    assert output.metadata["base_only_mode"] is True
    assert output.metadata["base_only_base_policy_called"] is True
    assert output.metadata["base_only_base_rng_advanced"] is True
    assert output.metadata["base_only_token_encoder_called"] is False
    assert output.metadata["base_only_actor_called"] is False
    assert output.metadata["base_only_latency_s"] == pytest.approx(0.091)


@pytest.mark.parametrize(
    "bad_actor",
    [
        np.zeros((9, 7), dtype=np.float32),
        np.zeros((10, 6), dtype=np.float32),
        np.full((10, 7), np.nan, dtype=np.float32),
    ],
)
def test_invalid_shadow_actor_is_fail_open_for_base_actions(bad_actor) -> None:
    actions = np.ones((10, 7), dtype=np.float32)
    output = extract_rlt_policy_output(
        {"actions": actions, "z_rl": np.ones(4), "a_actor": bad_actor},
        chunk_length=10,
        action_dim=7,
        read_actor=True,
        state_snapshot=np.zeros(7, dtype=np.float32),
    )

    np.testing.assert_allclose(output.a_ref, actions)
    assert output.a_actor is None
    assert output.metadata["actor_error"]


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"actions": np.zeros((9, 7), dtype=np.float32)},
        {"actions": np.zeros((50, 6), dtype=np.float32)},
        {"actions": np.full((50, 7), np.nan, dtype=np.float32)},
    ],
)
def test_extract_rlt_policy_output_rejects_invalid_actions(response) -> None:
    with pytest.raises(ValueError, match="actions"):
        extract_rlt_policy_output(response, chunk_length=10, action_dim=7)
