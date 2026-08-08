import numpy as np
import pytest

from piper_runtime.observation import PolicyResponseError, absolute_actions_to_delta, build_observation


def test_build_observation_preserves_training_mapping_and_prompt():
    camera1 = np.full((480, 640, 3), 10, dtype=np.uint8)
    camera2 = np.full((480, 640, 3), 20, dtype=np.uint8)
    state = np.arange(7, dtype=np.float32)
    obs = build_observation({"camera1": camera1, "camera2": camera2}, state)
    assert obs["observation/image"] is camera1
    assert obs["observation/wrist_image"] is camera2
    np.testing.assert_array_equal(obs["observation/state"], state)
    assert obs["prompt"] == 'Put the green block into the box.'


def test_absolute_actions_are_converted_to_joint_delta_once():
    state = np.array([1, 2, 3, 4, 5, 6, 0.04], dtype=np.float32)
    absolute = np.tile(np.arange(32, dtype=np.float32), (50, 1))
    delta = absolute_actions_to_delta({"actions": absolute}, state)
    assert delta.shape == (50, 7)
    np.testing.assert_allclose(delta[0, :6], np.arange(6) - state[:6])
    assert delta[0, 6] == absolute[0, 6]


@pytest.mark.parametrize(
    "actions",
    [np.zeros((49, 32)), np.zeros((50, 6)), np.full((50, 32), np.nan)],
)
def test_invalid_policy_response_is_rejected(actions):
    with pytest.raises(PolicyResponseError):
        absolute_actions_to_delta({"actions": actions}, np.zeros(7, dtype=np.float32))
