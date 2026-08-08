import random

import numpy as np
import torch

from openpi.rlt.config import RLTReplayConfig
from openpi.rlt.replay import RLTReplayBuffer
from openpi.rlt.replay import RLTTransition


def _transition(marker: float, *, reward: float = 0.0) -> RLTTransition:
    scalar = np.asarray([marker], dtype=np.float32)
    action = np.full((2, 1), marker, dtype=np.float32)
    return RLTTransition(
        rl_token=scalar,
        state=scalar,
        action=action,
        reference_action=action,
        reward=reward,
        discount=0.99,
        next_rl_token=scalar,
        next_state=scalar,
        next_reference_action=action,
        done=False,
        source="test",
    )


def test_replay_samples_fixed_demo_and_online_positive_fractions():
    random.seed(0)
    replay = RLTReplayBuffer(
        RLTReplayConfig(
            capacity=16,
            batch_size=8,
            demo_sample_fraction=0.5,
            online_positive_sample_fraction=0.25,
        )
    )
    replay.extend_demo([_transition(10.0) for _ in range(4)])
    replay.extend([_transition(20.0, reward=1.0) for _ in range(2)])
    replay.extend([_transition(30.0) for _ in range(12)])

    batch = replay.sample(torch.device("cpu"))
    markers = batch.state[:, 0].tolist()

    assert markers.count(10.0) == 4
    assert markers.count(20.0) >= 2
    assert len(markers) == 8


def test_replay_stats_separate_demo_from_online_storage():
    replay = RLTReplayBuffer(RLTReplayConfig(capacity=2, batch_size=2))
    replay.extend_demo([_transition(10.0, reward=1.0) for _ in range(3)])
    replay.extend([_transition(30.0) for _ in range(4)])

    stats = replay.stats()

    assert stats["replay_demo_size"] == 3
    assert stats["replay_online_size"] == 2
    assert stats["replay_size"] == 5
    assert stats["replay_reward_positive"] == 3


def test_replay_samples_positive_fraction_from_demo_when_online_has_no_success():
    random.seed(1)
    replay = RLTReplayBuffer(
        RLTReplayConfig(
            capacity=16,
            batch_size=8,
            demo_sample_fraction=0.25,
            online_positive_sample_fraction=0.5,
        )
    )
    replay.extend_demo([_transition(10.0, reward=1.0) for _ in range(2)])
    replay.extend_demo([_transition(11.0) for _ in range(6)])
    replay.extend([_transition(30.0) for _ in range(16)])

    batch = replay.sample(torch.device("cpu"))

    assert int((batch.reward > 0).sum().item()) >= 4
