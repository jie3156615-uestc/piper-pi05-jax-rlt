from openpi.training import config


def test_piper_runtime_config_matches_a100_training_model_mode():
    train_config = config.get_config("pi05_piper_takeplaceredcup_jax_delta")
    assert train_config.model.discrete_state_input is False
    assert train_config.model.action_horizon == 50
