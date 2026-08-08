import numpy as np

from piper_runtime.policy_service import make_warmup_observation


def test_warmup_observation_matches_training_schema():
    obs = make_warmup_observation()
    assert obs["observation/image"].shape == (480, 640, 3)
    assert obs["observation/wrist_image"].shape == (480, 640, 3)
    assert obs["observation/image"].dtype == np.uint8
    assert obs["observation/state"].shape == (7,)
    assert obs["prompt"] == "Put the green block into the box."


def test_checkpoint_step_can_be_selected_with_environment(monkeypatch):
    from piper_runtime import policy_service

    monkeypatch.delenv('PIPER_POLICY_CHECKPOINT', raising=False)
    monkeypatch.setenv('PIPER_POLICY_STEP', '20000')

    assert policy_service.resolve_checkpoint().endswith('/20000')


def test_explicit_checkpoint_environment_takes_precedence(monkeypatch):
    from piper_runtime import policy_service

    monkeypatch.setenv('PIPER_POLICY_STEP', '20000')
    monkeypatch.setenv('PIPER_POLICY_CHECKPOINT', '/tmp/specific_checkpoint')

    assert policy_service.resolve_checkpoint() == '/tmp/specific_checkpoint'
