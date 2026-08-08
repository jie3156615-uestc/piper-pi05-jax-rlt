from __future__ import annotations

import numpy as np
import pytest

from piper_runtime.rlt_actor_protocol import ACTOR_PROJECTION_PROFILE
from piper_runtime.rlt_actor_protocol import ActorOnlyRequest
from piper_runtime.rlt_actor_protocol import BehaviorReferenceTarget
from piper_runtime.rlt_actor_protocol import add_actor_enrichment_only_request
from piper_runtime.rlt_actor_protocol import add_actor_only_request
from piper_runtime.rlt_actor_protocol import add_base_only_request
from piper_runtime.rlt_actor_protocol import add_token_batch_request
from piper_runtime.rlt_actor_protocol import (
    RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT,
)
from piper_runtime.rlt_actor_protocol import (
    RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE,
)
from piper_runtime.rlt_shadow_policy import ShadowAugmentedPolicy
from piper_runtime.rlt_shadow_policy import ShadowPolicyConfig
from piper_runtime.rlt_shadow_policy_service import _resolve_actor_wire_contract


class FakeBasePolicy:
    metadata = {"base": "full20k"}

    def __init__(self, actions: np.ndarray):
        self.actions = actions
        self.calls = 0

    def infer(self, observation):
        del observation
        self.calls += 1
        return {"actions": self.actions.copy(), "base_only": 123}


class FakeToken:
    def __init__(self):
        self.calls = 0

    def encode(self, observation):
        del observation
        self.calls += 1
        return np.arange(2048, dtype=np.float32)

    def encode_batch(self, observations):
        self.calls += 1
        return np.stack(
            [
                np.arange(2048, dtype=np.float32) + index
                for index, _ in enumerate(observations)
            ]
        )


class ResidualActor:
    def predict(self, *, z_rl, state, a_ref):
        assert z_rl.shape == (2048,)
        assert state.shape == (7,)
        result = a_ref.copy()
        result[:, :6] += 0.25
        return result


class BrokenActor:
    def predict(self, **kwargs):
        del kwargs
        raise RuntimeError("actor exploded")


def _observation(state: np.ndarray) -> dict:
    return {"observation/state": np.asarray(state, dtype=np.float32)}


def test_shadow_wrapper_keeps_pi05_actions_bitwise_unchanged_and_adds_actor() -> None:
    state = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.02], dtype=np.float32)
    actions = np.repeat(state[None, :], 50, axis=0)
    actions[:, 6] = 0.04
    policy = ShadowAugmentedPolicy(
        FakeBasePolicy(actions), token_encoder=FakeToken(), actor=ResidualActor(), actor_name="test"
    )

    output = policy.infer(_observation(state))

    assert np.array_equal(output["actions"], actions)
    assert output["z_rl"].shape == (2048,)
    assert output["a_actor"].shape == (10, 7)
    np.testing.assert_allclose(output["a_actor"][:, :6], actions[:10, :6] + 0.25)
    np.testing.assert_allclose(output["a_actor"][:, 6], actions[:10, 6])
    assert output["a_actor_action_space"] == "joint_absolute_gripper_absolute"
    assert output["rlt_shadow"]["actor_controls_robot"] is False


def test_token_batch_skips_pi05_rng_and_actor() -> None:
    base = FakeBasePolicy(np.ones((50, 7), dtype=np.float32))
    token = FakeToken()

    class SpyActor(ResidualActor):
        def __init__(self):
            self.calls = 0

        def predict(self, **kwargs):
            self.calls += 1
            return super().predict(**kwargs)

    actor = SpyActor()
    policy = ShadowAugmentedPolicy(
        base,
        token_encoder=token,
        actor=actor,
    )
    request = add_token_batch_request(
        [
            _observation(np.zeros(7, dtype=np.float32)),
            _observation(np.ones(7, dtype=np.float32)),
        ]
    )

    output = policy.infer(request)

    assert base.calls == 0
    assert token.calls == 1
    assert actor.calls == 0
    assert output["z_rl_batch"].shape == (2, 2048)
    assert output["rlt_shadow"]["token_status"] == "ok"
    assert output["rlt_shadow"]["base_policy_called"] is False
    assert output["rlt_shadow"]["base_rng_advanced"] is False
    assert output["rlt_shadow"]["actor_called"] is False


def test_close_assist_wire_contract_is_preserved_in_policy_metadata_and_output() -> None:
    state = np.zeros(7, dtype=np.float32)
    actions = np.zeros((50, 7), dtype=np.float32)
    policy = ShadowAugmentedPolicy(
        FakeBasePolicy(actions),
        token_encoder=FakeToken(),
        actor=ResidualActor(),
        config=ShadowPolicyConfig(
            actor_action_schema_fingerprint=(
                RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT
            ),
            actor_projection_profile=(
                RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE
            ),
        ),
    )

    output = policy.infer(_observation(state))

    assert policy.metadata["rlt_action_schema_fingerprint"] == (
        RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT
    )
    assert policy.metadata["rlt_actor_projection_profile"] == (
        RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE
    )
    assert output["a_actor_action_schema_fingerprint"] == (
        RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT
    )
    assert output["a_actor_projection_profile"] == (
        RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE
    )
    assert output["rlt_shadow"]["action_schema_fingerprint"] == (
        RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT
    )
    assert output["rlt_shadow"]["actor_projection_profile"] == (
        RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE
    )


def test_shadow_config_rejects_crossed_schema_projection_pairs() -> None:
    with pytest.raises(ValueError, match="schema/projection mismatch"):
        ShadowPolicyConfig(
            actor_action_schema_fingerprint=(
                RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT
            ),
            actor_projection_profile=ACTOR_PROJECTION_PROFILE,
        ).validate()


def test_service_contract_uses_loaded_actor_and_checks_launcher(
    monkeypatch,
) -> None:
    class CloseActor:
        output_action_schema_fingerprint = (
            RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT
        )
        output_actor_projection_profile = (
            RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE
        )

    monkeypatch.setenv(
        "PIPER_RLT_EXPECTED_RAW_SCHEMA",
        RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT,
    )
    monkeypatch.setenv(
        "PIPER_RLT_EXPECTED_PROJECTION_PROFILE",
        RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE,
    )
    assert _resolve_actor_wire_contract(CloseActor()) == (
        RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT,
        RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE,
    )
    monkeypatch.setenv(
        "PIPER_RLT_EXPECTED_PROJECTION_PROFILE",
        ACTOR_PROJECTION_PROFILE,
    )
    with pytest.raises(ValueError, match="launcher/loaded Actor projection"):
        _resolve_actor_wire_contract(CloseActor())


def test_service_contract_uses_launcher_contract_when_fresh_zero_has_no_actor(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "PIPER_RLT_EXPECTED_RAW_SCHEMA",
        RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT,
    )
    monkeypatch.setenv(
        "PIPER_RLT_EXPECTED_PROJECTION_PROFILE",
        RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE,
    )

    assert _resolve_actor_wire_contract(None) == (
        RANK1_GRIPPER_CLOSE_ACTOR_OUTPUT_SCHEMA_FINGERPRINT,
        RANK1_GRIPPER_CLOSE_ACTOR_PROJECTION_PROFILE,
    )


def test_actor_exception_is_fail_open_and_keeps_token_and_pi05() -> None:
    actions = np.ones((50, 7), dtype=np.float32)
    policy = ShadowAugmentedPolicy(FakeBasePolicy(actions), token_encoder=FakeToken(), actor=BrokenActor())

    output = policy.infer(_observation(np.zeros(7, dtype=np.float32)))

    assert np.array_equal(output["actions"], actions)
    assert output["z_rl"].shape == (2048,)
    assert "a_actor" not in output
    assert output["rlt_shadow"]["actor_status"] == "error"
    assert "actor exploded" in output["rlt_shadow"]["actor_error"]


def test_wrong_token_shape_is_fail_open_for_pi05() -> None:
    class WrongToken:
        def encode(self, observation):
            del observation
            return np.zeros(8, dtype=np.float32)

    actions = np.ones((50, 7), dtype=np.float32)
    output = ShadowAugmentedPolicy(
        FakeBasePolicy(actions), token_encoder=WrongToken(), actor=ResidualActor()
    ).infer(_observation(np.zeros(7, dtype=np.float32)))

    assert np.array_equal(output["actions"], actions)
    assert "z_rl" not in output
    assert "a_actor" not in output
    assert output["rlt_shadow"]["token_status"] == "error"


def test_late_shadow_actor_is_discarded_without_touching_pi05() -> None:
    class Clock:
        def __init__(self):
            self.value = 0.0

        def __call__(self):
            self.value += 0.2
            return self.value

    actions = np.ones((50, 7), dtype=np.float32)
    output = ShadowAugmentedPolicy(
        FakeBasePolicy(actions),
        token_encoder=FakeToken(),
        actor=ResidualActor(),
        config=ShadowPolicyConfig(max_shadow_latency_s=0.1),
        clock=Clock(),
    ).infer(_observation(np.zeros(7, dtype=np.float32)))

    assert np.array_equal(output["actions"], actions)
    assert output["rlt_shadow"]["latency_ok"] is False
    assert output["rlt_shadow"]["actor_status"] == "discarded_late"
    assert "a_actor" not in output


def test_actor_only_request_echoes_exact_reference_without_calling_base_or_token() -> None:
    state = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.02], dtype=np.float32)
    reference = np.repeat(state[None, :], 10, axis=0)
    reference[:, 0] += np.linspace(0.0, 0.02, 10, dtype=np.float32)
    reference[:, 6] = 0.04
    base = FakeBasePolicy(np.full((50, 7), 99.0, dtype=np.float32))
    token = FakeToken()
    policy = ShadowAugmentedPolicy(
        base,
        token_encoder=token,
        actor=ResidualActor(),
        actor_name="test",
    )
    observation = add_actor_only_request(
        _observation(state),
        ActorOnlyRequest(
            target=BehaviorReferenceTarget(
                actions=reference,
                plan_id="behavior-h50-1",
                start_offset=20,
                conditioning_state=state,
            ),
            z_rl=np.arange(2048, dtype=np.float32),
        ),
    )

    output = policy.infer(observation)

    assert base.calls == 0
    assert token.calls == 0
    assert np.array_equal(output["actions"], reference)
    np.testing.assert_allclose(output["a_actor"][:, :6], reference[:, :6] + 0.25)
    np.testing.assert_allclose(output["a_actor"][:, 6], reference[:, 6])
    assert output["rlt_shadow"]["mode"] == "actor_only"
    assert output["rlt_shadow"]["base_policy_called"] is False
    assert output["rlt_shadow"]["base_rng_advanced"] is False
    assert output["rlt_shadow"]["token_encoder_called"] is False
    assert output["rlt_shadow"]["behavior_ref_plan_id"] == "behavior-h50-1"
    assert output["rlt_shadow"]["behavior_ref_start_offset"] == 20


def test_invalid_actor_only_request_fails_closed_without_calling_base() -> None:
    base = FakeBasePolicy(np.ones((50, 7), dtype=np.float32))
    token = FakeToken()
    policy = ShadowAugmentedPolicy(base, token_encoder=token, actor=ResidualActor())

    output = policy.infer(
        {
            **_observation(np.zeros(7, dtype=np.float32)),
            "rlt/actor_only_mode": "actor_only_v1",
            # z_rl and the exact behavior target are deliberately missing.
        }
    )

    assert base.calls == 0
    assert token.calls == 0
    assert "actions" not in output
    assert output["rlt_shadow"]["actor_status"] == "invalid_actor_only_protocol"
    assert output["rlt_shadow"]["base_rng_advanced"] is False


def test_base_only_calls_pi05_once_and_skips_token_and_actor() -> None:
    actions = np.arange(50 * 7, dtype=np.float32).reshape(50, 7)
    base = FakeBasePolicy(actions)
    token = FakeToken()

    class SpyActor(ResidualActor):
        def __init__(self):
            self.calls = 0

        def predict(self, **kwargs):
            self.calls += 1
            return super().predict(**kwargs)

    actor = SpyActor()
    policy = ShadowAugmentedPolicy(base, token_encoder=token, actor=actor)

    output = policy.infer(
        add_base_only_request(_observation(np.zeros(7, dtype=np.float32)))
    )

    assert base.calls == 1
    assert token.calls == 0
    assert actor.calls == 0
    assert np.array_equal(output["actions"], actions)
    assert "z_rl" not in output
    assert "a_actor" not in output
    assert output["rlt_shadow"]["mode"] == "base_only"
    assert output["rlt_shadow"]["base_policy_called"] is True
    assert output["rlt_shadow"]["base_rng_advanced"] is True
    assert output["rlt_shadow"]["token_encoder_called"] is False
    assert output["rlt_shadow"]["actor_called"] is False


def test_fresh_token_enrichment_calls_token_and_actor_but_never_pi05() -> None:
    state = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.02], dtype=np.float32)
    reference = np.repeat(state[None, :], 10, axis=0)
    reference[:, 0] += np.linspace(0.0, 0.02, 10, dtype=np.float32)
    reference[:, 6] = 0.04
    base = FakeBasePolicy(np.full((50, 7), 99.0, dtype=np.float32))
    token = FakeToken()
    policy = ShadowAugmentedPolicy(
        base,
        token_encoder=token,
        actor=ResidualActor(),
        actor_name="test",
    )
    observation = add_actor_enrichment_only_request(
        _observation(state),
        BehaviorReferenceTarget(
            actions=reference,
            plan_id="behavior-h50-2",
            start_offset=30,
            conditioning_state=state,
        ),
    )

    output = policy.infer(observation)

    assert base.calls == 0
    assert token.calls == 1
    assert np.array_equal(output["actions"], reference)
    assert output["z_rl"].shape == (2048,)
    np.testing.assert_allclose(output["a_actor"][:, :6], reference[:, :6] + 0.25)
    assert output["rlt_shadow"]["mode"] == "actor_only"
    assert (
        output["rlt_shadow"]["actor_only_protocol"]
        == "actor_enrichment_only_v1"
    )
    assert output["rlt_shadow"]["base_policy_called"] is False
    assert output["rlt_shadow"]["base_rng_advanced"] is False
    assert output["rlt_shadow"]["token_encoder_called"] is True
    assert output["rlt_shadow"]["actor_called"] is True
