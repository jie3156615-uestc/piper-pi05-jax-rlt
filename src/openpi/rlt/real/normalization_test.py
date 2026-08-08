import numpy as np

from openpi.rlt.real.normalization import RunningMeanStd
from openpi.rlt.real.normalization import RLTNormalization


def test_running_mean_std_normalizes_state_batch():
    rms = RunningMeanStd(shape=(2,))
    rms.update(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

    normalized = rms.normalize(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

    np.testing.assert_allclose(normalized.mean(axis=0), np.zeros(2), atol=1e-4)
    np.testing.assert_allclose(normalized.std(axis=0), np.ones(2), atol=1e-4)


def test_running_mean_std_clips_extreme_values():
    rms = RunningMeanStd(shape=(1,), epsilon=0.0, clip=2.0)
    rms.update(np.array([[0.0], [1.0]], dtype=np.float32))

    normalized = rms.normalize(np.array([[100.0]], dtype=np.float32))

    assert normalized.item() == 2.0


def test_rlt_normalization_fits_reference_and_executed_actions_separately():
    replay = {
        "z_rl": np.arange(24, dtype=np.float32).reshape(6, 4),
        "state": np.arange(42, dtype=np.float32).reshape(6, 7),
        "a_ref": np.ones((6, 10, 7), dtype=np.float32) * 2.0,
        "a_exec": np.arange(420, dtype=np.float32).reshape(6, 10, 7),
    }

    normalization = RLTNormalization.fit(replay)

    assert normalization.z_rl.mean.shape == (4,)
    assert normalization.state.mean.shape == (7,)
    assert normalization.a_ref.mean.shape == (10, 7)
    assert normalization.candidate_action.mean.shape == (10, 7)
    assert not np.array_equal(normalization.a_ref.mean, normalization.candidate_action.mean)


def test_rlt_normalization_state_dict_roundtrip():
    replay = {
        "z_rl": np.arange(24, dtype=np.float32).reshape(6, 4),
        "state": np.arange(42, dtype=np.float32).reshape(6, 7),
        "a_ref": np.arange(420, dtype=np.float32).reshape(6, 10, 7),
        "a_exec": np.arange(420, dtype=np.float32).reshape(6, 10, 7) * 3.0,
    }
    original = RLTNormalization.fit(replay)

    restored = RLTNormalization.from_state_dict(original.to_state_dict())

    np.testing.assert_array_equal(restored.z_rl.mean, original.z_rl.mean)
    np.testing.assert_array_equal(restored.candidate_action.std, original.candidate_action.std)
