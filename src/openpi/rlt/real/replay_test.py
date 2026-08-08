import numpy as np

from openpi.rlt.real.config import Source
from openpi.rlt.real.replay import RealStepRecord
from openpi.rlt.real.replay import chunk_real_episode


def _arr(value: float, shape=(7,)):
    return np.full(shape, value, dtype=np.float32)


def _chunk(value: float, length=10):
    return np.full((length, 7), value, dtype=np.float32)


def test_chunk_records_preserve_action_semantics():
    records = []
    for idx in range(12):
        records.append(
            RealStepRecord(
                episode_id="ep0",
                t=idx,
                z_rl=_arr(idx, shape=(4,)),
                state=_arr(idx),
                a_ref=_chunk(1.0),
                a_exec=_arr(2.0 + idx),
                a_human=_arr(9.0) if idx == 3 else None,
                a_actor=_chunk(4.0) if idx >= 5 else None,
                source=Source.HUMAN_PIKA if idx == 3 else Source.PI05,
                reward=0.0,
                done=False,
            )
        )

    transitions = chunk_real_episode(records, chunk_length=10, stride=2, n_step=10, gamma=0.99)

    first = transitions[0]
    assert first.a_ref.shape == (10, 7)
    assert first.a_exec.shape == (10, 7)
    assert first.a_human.shape == (10, 7)
    assert first.a_actor.shape == (10, 7)
    np.testing.assert_allclose(first.a_ref, 1.0)
    np.testing.assert_allclose(first.a_exec[0], 2.0)
    np.testing.assert_allclose(first.a_human[3], 9.0)
    assert first.source == Source.PI05


def test_terminal_reward_is_sparse_and_n_step_discounted():
    records = []
    for idx in range(11):
        records.append(
            RealStepRecord(
                episode_id="success_ep",
                t=idx,
                z_rl=_arr(0.0, shape=(4,)),
                state=_arr(0.0),
                a_ref=_chunk(1.0),
                a_exec=_arr(2.0),
                a_human=None,
                a_actor=None,
                source=Source.RLT,
                reward=1.0 if idx == 9 else 0.0,
                done=idx == 9,
            )
        )

    transitions = chunk_real_episode(records, chunk_length=10, stride=2, n_step=10, gamma=0.99)

    assert transitions[0].done is True
    assert transitions[0].discount == 0.0
    assert transitions[0].reward == 0.99**9
