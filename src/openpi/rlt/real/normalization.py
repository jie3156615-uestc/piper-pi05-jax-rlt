from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass
class RunningMeanStd:
    shape: tuple[int, ...]
    epsilon: float = 0.0
    clip: float = 10.0

    def __post_init__(self) -> None:
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self.count = float(self.epsilon)

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64)
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        normalized = (values - self.mean.astype(np.float32)) / np.sqrt(self.var.astype(np.float32) + 1e-8)
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count


@dataclasses.dataclass(frozen=True)
class NormalizationStats:
    """Immutable normalization statistics used by the JAX learner.

    Statistics deliberately keep the original tensor shape (for example
    ``(C, action_dim)`` for action chunks).  This makes accidental mixing of
    state, reference-action, and executed-action scales fail loudly instead of
    silently broadcasting.
    """

    mean: np.ndarray
    std: np.ndarray
    clip: float = 10.0
    count: int = 0

    @classmethod
    def fit(cls, values: np.ndarray, *, clip: float = 10.0, min_std: float = 1e-4) -> "NormalizationStats":
        values = np.asarray(values, dtype=np.float64)
        if values.ndim < 2 or values.shape[0] == 0:
            raise ValueError(f"normalization values must have a non-empty batch dimension, got {values.shape}")
        mean = values.mean(axis=0)
        std = np.maximum(values.std(axis=0), min_std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32), clip=float(clip), count=len(values))

    def normalize_np(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return np.clip((values - self.mean) / self.std, -self.clip, self.clip).astype(np.float32)

    def normalize_jax(self, values: jnp.ndarray) -> jnp.ndarray:
        mean = jnp.asarray(self.mean, dtype=values.dtype)
        std = jnp.asarray(self.std, dtype=values.dtype)
        return jnp.clip((values - mean) / std, -self.clip, self.clip)

    def denormalize_jax(self, values: jnp.ndarray) -> jnp.ndarray:
        mean = jnp.asarray(self.mean, dtype=values.dtype)
        std = jnp.asarray(self.std, dtype=values.dtype)
        return values * std + mean

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "mean": np.asarray(self.mean, dtype=np.float32),
            "std": np.asarray(self.std, dtype=np.float32),
            "clip": float(self.clip),
            "count": int(self.count),
        }

    @classmethod
    def from_state_dict(cls, value: dict[str, Any]) -> "NormalizationStats":
        return cls(
            mean=np.asarray(value["mean"], dtype=np.float32),
            std=np.asarray(value["std"], dtype=np.float32),
            clip=float(value["clip"]),
            count=int(value.get("count", 0)),
        )


@dataclasses.dataclass(frozen=True)
class RLTNormalization:
    """Per-modality statistics for real-robot RLT.

    ``a_ref`` and ``candidate_action`` are intentionally fitted separately.
    The critic therefore cannot obtain a scale cue from one branch solely
    because the human/executed-action distribution is wider than Pi0.5's
    reference distribution.
    """

    z_rl: NormalizationStats
    state: NormalizationStats
    a_ref: NormalizationStats
    candidate_action: NormalizationStats

    @classmethod
    def fit(cls, replay: dict[str, np.ndarray], *, clip: float = 10.0) -> "RLTNormalization":
        required = {"z_rl", "state", "a_ref", "a_exec"}
        missing = required.difference(replay)
        if missing:
            raise KeyError(f"cannot fit normalization; replay is missing {sorted(missing)}")
        return cls(
            z_rl=NormalizationStats.fit(replay["z_rl"], clip=clip),
            state=NormalizationStats.fit(replay["state"], clip=clip),
            a_ref=NormalizationStats.fit(replay["a_ref"], clip=clip),
            candidate_action=NormalizationStats.fit(replay["a_exec"], clip=clip),
        )

    def to_state_dict(self) -> dict[str, dict[str, Any]]:
        return {
            "z_rl": self.z_rl.to_state_dict(),
            "state": self.state.to_state_dict(),
            "a_ref": self.a_ref.to_state_dict(),
            "candidate_action": self.candidate_action.to_state_dict(),
        }

    @classmethod
    def from_state_dict(cls, value: dict[str, dict[str, Any]]) -> "RLTNormalization":
        return cls(**{name: NormalizationStats.from_state_dict(stats) for name, stats in value.items()})

    def normalize_actor_inputs(
        self, z_rl: jnp.ndarray, state: jnp.ndarray, ref_input: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return (
            self.z_rl.normalize_jax(z_rl),
            self.state.normalize_jax(state),
            self.a_ref.normalize_jax(ref_input),
        )

    def normalize_critic_inputs(
        self,
        z_rl: jnp.ndarray,
        state: jnp.ndarray,
        a_ref: jnp.ndarray,
        candidate_action: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return (
            self.z_rl.normalize_jax(z_rl),
            self.state.normalize_jax(state),
            self.a_ref.normalize_jax(a_ref),
            self.candidate_action.normalize_jax(candidate_action),
        )
