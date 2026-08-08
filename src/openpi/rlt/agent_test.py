import torch

from openpi.rlt.agent import RLTAgent
from openpi.rlt.config import RLTActorCriticConfig
from openpi.rlt.networks import RLTBatch


def _batch(batch_size: int = 4) -> RLTBatch:
    action = torch.zeros(batch_size, 2, 1)
    token = torch.zeros(batch_size, 3)
    state = torch.zeros(batch_size, 1)
    return RLTBatch(
        rl_token=token,
        state=state,
        action=action,
        reference_action=action,
        reward=torch.zeros(batch_size),
        discount=torch.full((batch_size,), 0.99),
        next_rl_token=token,
        next_state=state,
        next_reference_action=action,
    )


def test_update_reports_whether_actor_was_updated():
    agent = RLTAgent(
        rl_token_dim=3,
        config=RLTActorCriticConfig(
            state_dim=1,
            action_dim=1,
            chunk_length=2,
            policy_delay=2,
        ),
        device=torch.device("cpu"),
    )

    first = agent.update(_batch())
    second = agent.update(_batch())

    assert first["actor_updated"] is True
    assert "actor_loss" in first
    assert second["actor_updated"] is False
    assert "actor_loss" not in second


def test_absolute_actor_trust_region_clamps_action_delta():
    agent = RLTAgent(
        rl_token_dim=3,
        config=RLTActorCriticConfig(
            state_dim=1,
            action_dim=1,
            chunk_length=2,
            actor_output_mode="absolute",
            absolute_delta_clip=0.05,
        ),
        device=torch.device("cpu"),
    )
    reference = torch.full((4, 2, 1), 0.5)

    action = agent.act(
        rl_token=torch.zeros(4, 3),
        state=torch.zeros(4, 1),
        reference_action=reference,
        deterministic=True,
    )

    assert torch.max(torch.abs(action - reference)).item() <= 0.050001
