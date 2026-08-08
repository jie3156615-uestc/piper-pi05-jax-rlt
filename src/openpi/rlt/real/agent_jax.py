from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training import train_state

from openpi.rlt.real.config import HUMAN_EXECUTION_PROFILE
from openpi.rlt.real.config import GRIPPER_RESIDUAL_CLOSE_ASSIST
from openpi.rlt.real.config import PERSISTENT_ACTION_SCHEMA_FINGERPRINT
from openpi.rlt.real.config import PERSISTENT_ACTOR_EXECUTION_PROFILE
from openpi.rlt.real.config import PERSISTENT_EXECUTION_FILTER_PROFILE
from openpi.rlt.real.config import PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT
from openpi.rlt.real.config import PERSISTENT_GRIPPER_CLOSE_GOVERNOR_PROFILE
from openpi.rlt.real.config import PERSISTENT_GOVERNOR_PROFILE
from openpi.rlt.real.config import RANK1_ACTOR_OUTPUT_SCHEMA_FINGERPRINT
from openpi.rlt.real.config import RealRLTConfig, Source
from openpi.rlt.real.networks_jax import (
    ResidualActor,
    TwinCritic,
    apply_direction_cone_scale,
    rank1_bump_residual,
    rank1_direction_from_residual,
    rank1_direction_limit,
    persistent_filtered_candidate_action,
    residual_temporal_metrics,
)
from openpi.rlt.real.normalization import RLTNormalization

CHECKPOINT_VERSION = 3

REQUIRED_TRAIN_BATCH_KEYS = (
    "z_rl",
    "state",
    "a_ref",
    "a_exec",
    "reward",
    "discount",
    "next_z_rl",
    "next_state",
    "next_a_ref",
)
TRAIN_BATCH_KEYS = REQUIRED_TRAIN_BATCH_KEYS + ("a_human", "human_mask")
PERSISTENT_TRAIN_BATCH_KEYS = (
    "actor_canonical_decision",
    "actor_persistent_carry_in",
    "actor_persistent_carry_out",
    "actor_persistent_previous_carry",
    "actor_execution_boundary_anchor",
    "a_base_filtered",
    "a_filtered_actual",
    "filtered_actual_residual",
    "execution_filter_tau_s",
    "execution_filter_dt_s",
    "execution_filter_alpha",
    "execution_projection_scale",
    "next_a_base_filtered",
    "next_execution_filter_alpha",
    "next_actor_persistent_previous_carry",
    "next_actor_persistent_carry_in",
    "next_actor_execution_boundary_anchor",
)
PERSISTENT_AUDIT_KEYS = (
    "execution_residual_max_rad",
    "execution_d1_max_rad",
    "execution_d2_max_rad",
    "execution_direction_cone_deg",
    "execution_boundary_limit_rad",
    "execution_projection_scale_steps",
    "execution_min_projection_scale",
    "execution_direction_static_threshold_rad",
)
GRIPPER_PERSISTENT_AUDIT_KEYS = (
    "execution_gripper_residual_max_close_m",
    "execution_gripper_d1_max_m",
    "execution_gripper_d2_max_m",
    "execution_gripper_boundary_limit_m",
    "execution_gripper_command_min_m",
    "execution_gripper_command_max_m",
    "execution_gripper_release_reference_m",
    "execution_gripper_release_delta_m",
)


def filter_replay_by_split(
    replay: Mapping[str, np.ndarray], split: str = "train"
) -> dict[str, np.ndarray]:
    """Select a whole-episode replay split without allowing silent leakage.

    Production replay files carry one ``episode_split`` label per transition.
    ``all`` is supported only as an explicit diagnostic choice; training and
    validation callers should select their intended split.
    """

    valid = {"train", "validation", "test", "all"}
    if split not in valid:
        raise ValueError(f"split must be one of {sorted(valid)}, got {split!r}")
    arrays = {name: np.asarray(value) for name, value in replay.items()}
    if split == "all":
        return arrays
    if "episode_split" not in arrays:
        raise KeyError(
            "replay has no episode_split array; regenerate it with whole-episode splits "
            "or explicitly request split='all' for diagnostics"
        )
    labels = arrays["episode_split"].astype(str)
    if labels.ndim != 1:
        raise ValueError(f"episode_split must be one-dimensional, got {labels.shape}")
    size = len(labels)
    unknown = sorted(set(labels.tolist()).difference({"train", "validation", "test"}))
    if unknown:
        raise ValueError(f"replay contains unknown episode_split labels: {unknown}")
    if "episode_id" in arrays:
        episode_ids = arrays["episode_id"].astype(str)
        if episode_ids.shape != labels.shape:
            raise ValueError("episode_id and episode_split arrays must have the same shape")
        leaking = [
            episode_id
            for episode_id in np.unique(episode_ids)
            if len(set(labels[episode_ids == episode_id].tolist())) != 1
        ]
        if leaking:
            raise ValueError(f"episodes span multiple replay splits: {leaking}")
    mask = labels == split
    if not np.any(mask):
        raise ValueError(f"replay split {split!r} is empty")
    selected: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if value.ndim == 0 or len(value) != size:
            raise ValueError(f"replay array {name!r} is not transition-aligned with episode_split")
        selected[name] = value[mask]
    return selected


class TrainState(train_state.TrainState):
    target_params: Any


@dataclasses.dataclass(frozen=True)
class ReplaySamplingConfig:
    """Controls replay composition without changing stored transitions.

    Success/failure and human/non-human are treated as two independent axes.
    Sampling weights are built over their four intersections, so a requested
    human fraction does not accidentally erase the requested success ratio.
    Empty intersections are automatically redistributed over available data.
    """

    success_fraction: float | None = None
    failure_fraction: float | None = None
    human_fraction: float | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("success_fraction", "failure_fraction", "human_fraction"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.success_fraction is not None and self.failure_fraction is not None:
            if not np.isclose(self.success_fraction + self.failure_fraction, 1.0):
                raise ValueError("success_fraction + failure_fraction must equal 1")


class ReplayBatchSampler:
    """Samples numeric learner batches from an enriched replay dictionary."""

    def __init__(
        self,
        replay: Mapping[str, np.ndarray],
        *,
        config: ReplaySamplingConfig | None = None,
        success_mask: np.ndarray | None = None,
        failure_mask: np.ndarray | None = None,
        human_mask: np.ndarray | None = None,
    ) -> None:
        self.replay = {key: np.asarray(value) for key, value in replay.items()}
        self.config = config or ReplaySamplingConfig()
        missing = set(REQUIRED_TRAIN_BATCH_KEYS).difference(self.replay)
        if missing:
            raise KeyError(f"replay is missing learner arrays: {sorted(missing)}")
        self.size = int(len(self.replay["reward"]))
        if self.size == 0:
            raise ValueError("cannot sample an empty replay")
        for key in REQUIRED_TRAIN_BATCH_KEYS:
            if len(self.replay[key]) != self.size:
                raise ValueError(f"replay array {key!r} has inconsistent length")
        self._persistent = "actor_execution_profile" in self.replay
        if self._persistent:
            missing_persistent = set(PERSISTENT_TRAIN_BATCH_KEYS).difference(self.replay)
            if missing_persistent:
                raise KeyError(
                    "persistent-v2 replay is missing learner arrays: "
                    f"{sorted(missing_persistent)}"
                )
            for key in PERSISTENT_TRAIN_BATCH_KEYS:
                if len(self.replay[key]) != self.size:
                    raise ValueError(f"replay array {key!r} has inconsistent length")
        self._add_optional_human_arrays()

        self.success_mask = self._infer_success_mask() if success_mask is None else self._check_mask(success_mask)
        if failure_mask is None:
            self.failure_mask = ~self.success_mask
        else:
            self.failure_mask = self._check_mask(failure_mask)
            if np.any(self.success_mask & self.failure_mask):
                raise ValueError("success_mask and failure_mask overlap")
            uncovered = ~(self.success_mask | self.failure_mask)
            if np.any(uncovered):
                raise ValueError("success_mask and failure_mask must cover every transition")
        self.human_mask = self._infer_human_mask() if human_mask is None else self._check_human_mask(human_mask)
        self._probabilities = self._make_probabilities()
        self._rng = np.random.default_rng(self.config.seed)

    def _add_optional_human_arrays(self) -> None:
        """Normalize optional intervention arrays while accepting old replay files."""

        expected_action = np.asarray(self.replay["a_exec"]).shape
        if "a_human" not in self.replay:
            self.replay["a_human"] = np.zeros(expected_action, dtype=np.float32)
        elif np.asarray(self.replay["a_human"]).shape != expected_action:
            raise ValueError(f"a_human must have shape {expected_action}, got {self.replay['a_human'].shape}")

        expected_mask = expected_action[:2]
        if "human_mask" not in self.replay:
            mask = np.zeros(expected_mask, dtype=np.bool_)
        else:
            mask = np.asarray(self.replay["human_mask"], dtype=np.bool_)
            if mask.shape == (self.size,):
                mask = np.broadcast_to(mask[:, None], expected_mask)
            elif mask.ndim > 2 and mask.shape[:2] == expected_mask:
                mask = np.any(mask, axis=tuple(range(2, mask.ndim)))
            if mask.shape != expected_mask:
                raise ValueError(f"human_mask must have shape {(self.size,)} or {expected_mask}, got {mask.shape}")
        self.replay["human_mask"] = np.asarray(mask, dtype=np.bool_)

    def _check_mask(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.bool_)
        if value.shape != (self.size,):
            raise ValueError(f"mask must have shape {(self.size,)}, got {value.shape}")
        return value

    def _check_human_mask(self, value: np.ndarray) -> np.ndarray:
        """Accepts either a transition mask [N] or per-step mask [N, C]."""

        value = np.asarray(value, dtype=np.bool_)
        if value.ndim > 1:
            value = np.any(value, axis=tuple(range(1, value.ndim)))
        return self._check_mask(value)

    def _infer_success_mask(self) -> np.ndarray:
        for name in ("success_mask", "episode_success", "success"):
            if name in self.replay:
                return self._check_mask(self.replay[name])
        rewards = np.asarray(self.replay["reward"], dtype=np.float32)
        if "episode_id" not in self.replay:
            return rewards > 0.0
        episode_ids = np.asarray(self.replay["episode_id"])
        successful = {
            episode_id
            for episode_id in np.unique(episode_ids)
            if np.any(rewards[episode_ids == episode_id] > 0.0)
        }
        return np.asarray([episode_id in successful for episode_id in episode_ids], dtype=np.bool_)

    def _infer_human_mask(self) -> np.ndarray:
        if "human_mask" in self.replay:
            return self._check_human_mask(self.replay["human_mask"])
        if "source" in self.replay:
            return np.asarray(self.replay["source"]).astype(str) == Source.HUMAN_PIKA
        if "a_human" in self.replay:
            axes = tuple(range(1, np.asarray(self.replay["a_human"]).ndim))
            return np.any(np.abs(self.replay["a_human"]) > 1e-8, axis=axes)
        return np.zeros(self.size, dtype=np.bool_)

    def _make_probabilities(self) -> np.ndarray:
        natural_success = float(self.success_mask.mean())
        success_fraction = self.config.success_fraction
        if success_fraction is None and self.config.failure_fraction is not None:
            success_fraction = 1.0 - self.config.failure_fraction
        if success_fraction is None:
            success_fraction = natural_success
        human_fraction = self.config.human_fraction
        if human_fraction is None:
            human_fraction = float(self.human_mask.mean())

        desired = {
            (True, True): success_fraction * human_fraction,
            (True, False): success_fraction * (1.0 - human_fraction),
            (False, True): (1.0 - success_fraction) * human_fraction,
            (False, False): (1.0 - success_fraction) * (1.0 - human_fraction),
        }
        groups: dict[tuple[bool, bool], np.ndarray] = {}
        for success in (True, False):
            class_mask = self.success_mask if success else self.failure_mask
            for human in (True, False):
                mask = class_mask & (self.human_mask if human else ~self.human_mask)
                groups[(success, human)] = np.flatnonzero(mask)

        available_mass = sum(probability for group, probability in desired.items() if len(groups[group]))
        if available_mass <= 0.0:
            available = [group for group, indices in groups.items() if len(indices)]
            desired = {group: (1.0 / len(available) if group in available else 0.0) for group in groups}
        else:
            desired = {
                group: (probability / available_mass if len(groups[group]) else 0.0)
                for group, probability in desired.items()
            }

        probabilities = np.zeros(self.size, dtype=np.float64)
        for group, indices in groups.items():
            if len(indices):
                probabilities[indices] = desired[group] / len(indices)
        if probabilities.sum() <= 0.0:
            probabilities.fill(1.0 / self.size)
        else:
            probabilities /= probabilities.sum()
        return probabilities

    def sample_indices(self, batch_size: int) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return self._rng.choice(self.size, size=batch_size, replace=True, p=self._probabilities)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        indices = self.sample_indices(batch_size)
        keys = TRAIN_BATCH_KEYS + (PERSISTENT_TRAIN_BATCH_KEYS if self._persistent else ())
        batch = {key: np.asarray(self.replay[key][indices], dtype=np.float32) for key in keys}
        batch["sample_index"] = indices.astype(np.int64)
        batch["success_mask"] = self.success_mask[indices]
        batch["failure_mask"] = self.failure_mask[indices]
        batch["human_transition_mask"] = self.human_mask[indices]
        return batch


def apply_reference_dropout(key: jax.Array, a_ref: jnp.ndarray, *, dropout: float) -> jnp.ndarray:
    """Zeroes whole reference chunks for a random subset of the actor batch."""

    keep_probability = jnp.clip(1.0 - jnp.asarray(dropout, dtype=jnp.float32), 0.0, 1.0)
    keep = jax.random.bernoulli(key, p=keep_probability, shape=(a_ref.shape[0], 1, 1))
    return a_ref * keep.astype(a_ref.dtype)


def critic_batch_view(batch: Mapping[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    """Returns the replay action consumed by critic regression.

    This small explicit adapter protects the central real-RLT invariant: critic
    regression uses what the robot actually executed, never a later actor
    prediction and never the nominal Pi0.5 reference.
    """

    return {
        "a_ref": batch["a_ref"],
        "candidate_action": batch["a_exec"],
    }


def _optimizer(learning_rate: float, cfg: RealRLTConfig) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(learning_rate=learning_rate, weight_decay=cfg.weight_decay),
    )


def _tree_global_norm(tree: Any) -> jnp.ndarray:
    squared = [jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(tree)]
    return jnp.sqrt(jnp.sum(jnp.stack(squared)))


def create_train_state(
    key: jax.Array,
    cfg: RealRLTConfig,
    z_dim: int,
    residual_limit: jnp.ndarray,
) -> tuple[TrainState, TrainState]:
    actor = ResidualActor(cfg)
    critic = TwinCritic(cfg)
    actor_limit = (
        residual_limit
        if cfg.actor_residual_parameterization == "legacy_full_chunk"
        else rank1_direction_limit(residual_limit, cfg)
    )
    actor_key, critic_key = jax.random.split(key)
    actor_variables = actor.init(
        actor_key,
        jnp.zeros((1, z_dim)),
        jnp.zeros((1, cfg.state_dim)),
        jnp.zeros((1, cfg.chunk_length, cfg.action_dim)),
        jnp.zeros((1, cfg.chunk_length, cfg.action_dim)),
        actor_limit,
    )
    critic_variables = critic.init(
        critic_key,
        jnp.zeros((1, z_dim)),
        jnp.zeros((1, cfg.state_dim)),
        jnp.zeros((1, cfg.chunk_length, cfg.action_dim)),
        jnp.zeros((1, cfg.chunk_length, cfg.action_dim)),
    )
    actor_params = actor_variables["params"]
    critic_params = critic_variables["params"]
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor_params,
        tx=_optimizer(cfg.actor_lr, cfg),
        target_params=actor_params,
    )
    critic_state = TrainState.create(
        apply_fn=critic.apply,
        params=critic_params,
        tx=_optimizer(cfg.critic_lr, cfg),
        target_params=critic_params,
    )
    return actor_state, critic_state


def soft_update(target_params: Any, params: Any, tau: float) -> Any:
    return optax.incremental_update(params, target_params, tau)


def _normalization_arrays(normalization: RLTNormalization) -> dict[str, jnp.ndarray]:
    arrays: dict[str, jnp.ndarray] = {}
    for name, stats in normalization.to_state_dict().items():
        arrays[f"{name}_mean"] = jnp.asarray(stats["mean"], dtype=jnp.float32)
        arrays[f"{name}_std"] = jnp.asarray(stats["std"], dtype=jnp.float32)
        arrays[f"{name}_clip"] = jnp.asarray(stats["clip"], dtype=jnp.float32)
    return arrays


def _normalize(value: jnp.ndarray, name: str, norm: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    return jnp.clip(
        (value - norm[f"{name}_mean"]) / norm[f"{name}_std"],
        -norm[f"{name}_clip"],
        norm[f"{name}_clip"],
    )


def _actor_action(
    actor_state: TrainState,
    params: Any,
    z_rl: jnp.ndarray,
    state: jnp.ndarray,
    ref_input_normalized: jnp.ndarray,
    a_ref_original: jnp.ndarray,
    actor_limit: jnp.ndarray,
    norm: Mapping[str, jnp.ndarray],
) -> jnp.ndarray:
    return actor_state.apply_fn(
        {"params": params},
        _normalize(z_rl, "z_rl", norm),
        _normalize(state, "state", norm),
        ref_input_normalized,
        a_ref_original,
        actor_limit,
    )


def _actor_canonical_direction(
    actor_state: TrainState,
    params: Any,
    z_rl: jnp.ndarray,
    state: jnp.ndarray,
    ref_input_normalized: jnp.ndarray,
    a_ref_original: jnp.ndarray,
    actor_limit: jnp.ndarray,
    norm: Mapping[str, jnp.ndarray],
) -> jnp.ndarray:
    """Return the unchanged checkpoint head as one raw seven-D target knot."""

    legacy_rank1_action = _actor_action(
        actor_state,
        params,
        z_rl,
        state,
        ref_input_normalized,
        a_ref_original,
        actor_limit,
        norm,
    )
    return rank1_direction_from_residual(legacy_rank1_action - a_ref_original)


def _persistent_actor_candidate(
    actor_state: TrainState,
    params: Any,
    z_rl: jnp.ndarray,
    state: jnp.ndarray,
    ref_input_normalized: jnp.ndarray,
    a_ref: jnp.ndarray,
    a_base_filtered: jnp.ndarray,
    carry_in: jnp.ndarray,
    previous_carry: jnp.ndarray,
    boundary_anchor: jnp.ndarray,
    filter_alpha: jnp.ndarray,
    actor_limit: jnp.ndarray,
    norm: Mapping[str, jnp.ndarray],
    cfg: RealRLTConfig,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    direction = _actor_canonical_direction(
        actor_state,
        params,
        z_rl,
        state,
        ref_input_normalized,
        a_ref,
        actor_limit,
        norm,
    )
    candidate, execution = persistent_filtered_candidate_action(
        a_base_filtered,
        a_ref,
        direction,
        carry_in,
        previous_carry,
        boundary_anchor,
        filter_alpha,
        cfg,
    )
    execution["canonical_direction"] = direction
    return candidate, execution


def _persistent_temporal_metrics(residual: jnp.ndarray) -> dict[str, jnp.ndarray]:
    joint = residual[..., :6]
    d1 = jnp.diff(joint, axis=1)
    d2 = jnp.diff(joint, n=2, axis=1)
    return {
        "residual_d1_abs_mean": jnp.mean(jnp.abs(d1)),
        "residual_d1_abs_max": jnp.max(jnp.abs(d1)),
        "residual_d2_abs_mean": jnp.mean(jnp.abs(d2)),
        "residual_d2_abs_max": jnp.max(jnp.abs(d2)),
        "residual_endpoint_abs_max": jnp.max(jnp.abs(joint[:, (0, -1), :])),
        "residual_rank1_error_abs_max": jnp.asarray(0.0, dtype=residual.dtype),
    }


def _critic_values(
    critic_state: TrainState,
    params: Any,
    z_rl: jnp.ndarray,
    state: jnp.ndarray,
    a_ref: jnp.ndarray,
    candidate_action: jnp.ndarray,
    norm: Mapping[str, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return critic_state.apply_fn(
        {"params": params},
        _normalize(z_rl, "z_rl", norm),
        _normalize(state, "state", norm),
        _normalize(a_ref, "a_ref", norm),
        _normalize(candidate_action, "candidate_action", norm),
    )


def admitted_human_gripper_q_filter(
    q_human: jnp.ndarray,
    q_actor: jnp.ndarray,
    human_chunk_mask: jnp.ndarray,
    *,
    margin: float | jnp.ndarray = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a stop-gradient Q-filter for admitted human gripper teachers.

    Every admitted human chunk is evaluated, irrespective of its binary
    reward.  The human gripper target is imitated only when the clipped-twin
    Critic values it above the current Actor by more than ``margin``.  Chunks
    valued below the Actor remain in Critic training as negative evidence and
    influence the Actor through its ordinary ``-Q`` objective.
    """

    advantage = jax.lax.stop_gradient(q_human - q_actor)
    admitted = human_chunk_mask.astype(advantage.dtype)
    selected = admitted * (
        advantage > jnp.asarray(margin, dtype=advantage.dtype)
    ).astype(advantage.dtype)
    return selected, advantage


def admitted_human_gripper_candidate(
    actor_direction: jnp.ndarray,
    a_human: jnp.ndarray,
    human_mask: jnp.ndarray,
    a_ref: jnp.ndarray,
    a_base_filtered: jnp.ndarray,
    carry_in: jnp.ndarray,
    previous_carry: jnp.ndarray,
    boundary_anchor: jnp.ndarray,
    filter_alpha: jnp.ndarray,
    cfg: RealRLTConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    """Build a physically executable human-gripper comparison candidate.

    Raw Pika commands can jump and do not obey the Actor's persistent C10
    contract.  Reduce the admitted human close residual to one interpretable
    gripper target knot, keep the Actor's six joint knots unchanged, then run
    the result through the exact carry/rate/boundary/release governor used by
    both training and the robot.
    """

    mask = human_mask.astype(a_base_filtered.dtype)
    observed_delta = jnp.clip(
        a_human[..., 6] - a_base_filtered[..., 6],
        -jnp.asarray(
            cfg.actor_gripper_residual_max_close_m,
            dtype=a_base_filtered.dtype,
        ),
        0.0,
    )
    target_gripper_direction = (
        jnp.sum(observed_delta * mask, axis=1)
        / jnp.maximum(jnp.sum(mask, axis=1), 1.0)
    )
    human_direction = actor_direction.at[..., 6].set(
        target_gripper_direction
    )
    candidate, execution = persistent_filtered_candidate_action(
        a_base_filtered,
        a_ref,
        human_direction,
        carry_in,
        previous_carry,
        boundary_anchor,
        filter_alpha,
        cfg,
    )
    return candidate, human_direction, execution


def q_filtered_human_gripper_bc_loss(
    actor_gripper: jnp.ndarray,
    teacher_gripper: jnp.ndarray,
    human_mask: jnp.ndarray,
    q_human: jnp.ndarray,
    q_actor: jnp.ndarray,
    *,
    margin: float | jnp.ndarray,
    scale_m: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute reward-independent, Critic-gated admitted-human gripper BC."""

    admitted_human_mask = human_mask.astype(actor_gripper.dtype)
    admitted_human_chunk_mask = (
        jnp.max(admitted_human_mask, axis=1) > 0.0
    ).astype(actor_gripper.dtype)
    selected_human_chunk_mask, advantage = (
        admitted_human_gripper_q_filter(
            q_human,
            q_actor,
            admitted_human_chunk_mask,
            margin=margin,
        )
    )
    selected_human_mask = (
        admitted_human_mask * selected_human_chunk_mask[:, None]
    )
    normalized_error = (
        actor_gripper - jax.lax.stop_gradient(teacher_gripper)
    ) / jnp.asarray(scale_m, dtype=actor_gripper.dtype)
    loss = (
        jnp.sum(jnp.square(normalized_error) * selected_human_mask)
        / jnp.maximum(jnp.sum(selected_human_mask), 1.0)
    )
    return loss, selected_human_chunk_mask, selected_human_mask, advantage


def _masked_mean(
    values: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    mask = mask.astype(values.dtype)
    return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def apply_target_policy_smoothing(
    key: jax.Array,
    action: jnp.ndarray,
    a_ref: jnp.ndarray,
    actor_limit: jnp.ndarray,
    *,
    noise_std: jnp.ndarray,
    noise_clip: jnp.ndarray,
    cone_deg: float | jnp.ndarray = 15.0,
) -> jnp.ndarray:
    """Apply TD3 clipped target noise inside the rank1_bump action family.

    For rank1_bump, noise is sampled once per direction dimension, never
    independently per time step.  Re-expansion through the bump and the same
    whole-chunk cone scaling used by the actor preserves all hard constraints.
    A matrix ``actor_limit`` retains the original full-chunk path solely so old
    v1/v2 checkpoints can still be evaluated read-only.
    """

    if actor_limit.ndim == 2:
        limit = actor_limit.reshape((1,) + tuple(actor_limit.shape))
        noise = jax.random.normal(key, shape=action.shape, dtype=action.dtype) * noise_std * limit
        noise = jnp.clip(noise, -noise_clip * limit, noise_clip * limit)
        residual = jnp.clip(action + noise - a_ref, -limit, limit)
        return a_ref + residual
    if actor_limit.ndim != 1:
        raise ValueError(f"actor_limit must have rank 1 or 2, got {actor_limit.shape}")

    limit = actor_limit.reshape((1, actor_limit.shape[-1]))
    direction = rank1_direction_from_residual(action - a_ref)
    noise = jax.random.normal(key, shape=direction.shape, dtype=action.dtype) * noise_std * limit
    noise = jnp.clip(noise, -noise_clip * limit, noise_clip * limit)
    noisy_direction = jnp.clip(direction + noise, -limit, limit)
    residual = rank1_bump_residual(noisy_direction)
    residual, _ = apply_direction_cone_scale(a_ref, residual, cone_deg=cone_deg)
    return a_ref + residual


def apply_target_direction_smoothing(
    key: jax.Array,
    direction: jnp.ndarray,
    actor_limit: jnp.ndarray,
    *,
    noise_std: jnp.ndarray,
    noise_clip: jnp.ndarray,
) -> jnp.ndarray:
    """TD3 target noise in the unchanged seven-D Actor direction space."""

    if direction.ndim != 2 or actor_limit.ndim != 1:
        raise ValueError(
            "persistent-v2 target smoothing requires direction (batch, action) "
            f"and one-D actor_limit, got {direction.shape} and {actor_limit.shape}"
        )
    limit = actor_limit.reshape((1, actor_limit.shape[-1]))
    noise = jax.random.normal(key, shape=direction.shape, dtype=direction.dtype)
    noise = noise * noise_std * limit
    noise = jnp.clip(noise, -noise_clip * limit, noise_clip * limit)
    return jnp.clip(direction + noise, -limit, limit)


@functools.partial(jax.jit, static_argnames=("cfg",))
def _critic_update(
    key: jax.Array,
    actor_state: TrainState,
    critic_state: TrainState,
    batch: Mapping[str, jnp.ndarray],
    norm: Mapping[str, jnp.ndarray],
    actor_limit: jnp.ndarray,
    target_policy_noise_std: jnp.ndarray,
    target_policy_noise_clip: jnp.ndarray,
    cfg: RealRLTConfig,
) -> tuple[TrainState, dict[str, jnp.ndarray]]:
    next_ref_normalized = _normalize(batch["next_a_ref"], "a_ref", norm)
    if cfg.actor_execution_profile == PERSISTENT_ACTOR_EXECUTION_PROFILE:
        next_direction = _actor_canonical_direction(
            actor_state,
            actor_state.target_params,
            batch["next_z_rl"],
            batch["next_state"],
            next_ref_normalized,
            batch["next_a_ref"],
            actor_limit,
            norm,
        )
        next_action_before_smoothing, before_execution = persistent_filtered_candidate_action(
            batch["next_a_base_filtered"],
            batch["next_a_ref"],
            next_direction,
            batch["next_actor_persistent_carry_in"],
            batch["next_actor_persistent_previous_carry"],
            batch["next_actor_execution_boundary_anchor"],
            batch["next_execution_filter_alpha"],
            cfg,
        )
        smoothed_direction = apply_target_direction_smoothing(
            key,
            next_direction,
            actor_limit,
            noise_std=target_policy_noise_std,
            noise_clip=target_policy_noise_clip,
        )
        next_action, target_execution = persistent_filtered_candidate_action(
            batch["next_a_base_filtered"],
            batch["next_a_ref"],
            smoothed_direction,
            batch["next_actor_persistent_carry_in"],
            batch["next_actor_persistent_previous_carry"],
            batch["next_actor_execution_boundary_anchor"],
            batch["next_execution_filter_alpha"],
            cfg,
        )
        target_temporal = _persistent_temporal_metrics(
            target_execution["filtered_residual"]
        )
    else:
        next_action_before_smoothing = _actor_action(
            actor_state,
            actor_state.target_params,
            batch["next_z_rl"],
            batch["next_state"],
            next_ref_normalized,
            batch["next_a_ref"],
            actor_limit,
            norm,
        )
        next_action = apply_target_policy_smoothing(
            key,
            next_action_before_smoothing,
            batch["next_a_ref"],
            actor_limit,
            noise_std=target_policy_noise_std,
            noise_clip=target_policy_noise_clip,
            cone_deg=cfg.actor_direction_cone_deg,
        )
        target_temporal = residual_temporal_metrics(
            next_action - batch["next_a_ref"]
        )
    target_q1, target_q2 = _critic_values(
        critic_state,
        critic_state.target_params,
        batch["next_z_rl"],
        batch["next_state"],
        batch["next_a_ref"],
        next_action,
        norm,
    )
    td_target = jax.lax.stop_gradient(batch["reward"] + batch["discount"] * jnp.minimum(target_q1, target_q2))

    def loss_fn(params: Any) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        # The candidate is a_exec from replay.  Do not replace it with a
        # freshly evaluated actor action here.
        q1, q2 = _critic_values(
            critic_state,
            params,
            batch["z_rl"],
            batch["state"],
            batch["a_ref"],
            batch["a_exec"],
            norm,
        )
        q1_loss = jnp.mean(jnp.square(q1 - td_target))
        q2_loss = jnp.mean(jnp.square(q2 - td_target))
        loss = q1_loss + q2_loss
        reward1_mask = batch["success_mask"].astype(q1.dtype)
        reward0_mask = 1.0 - reward1_mask
        q_min = jnp.minimum(q1, q2)
        reward1_q_mean = _masked_mean(q_min, reward1_mask)
        reward0_q_mean = _masked_mean(q_min, reward0_mask)
        return loss, {
            "critic_loss": loss,
            "critic_q1_loss": q1_loss,
            "critic_q2_loss": q2_loss,
            "q1_mean": jnp.mean(q1),
            "q2_mean": jnp.mean(q2),
            "critic_reward1_q_mean": reward1_q_mean,
            "critic_reward0_q_mean": reward0_q_mean,
            "critic_reward1_reward0_q_gap": reward1_q_mean - reward0_q_mean,
            "td_target_mean": jnp.mean(td_target),
            "td_error_abs": jnp.mean(jnp.abs(jnp.minimum(q1, q2) - td_target)),
            "target_smoothing_abs_mean": jnp.mean(jnp.abs(next_action - next_action_before_smoothing)),
            "target_residual_d1_abs_max": target_temporal["residual_d1_abs_max"],
            "target_residual_d2_abs_max": target_temporal["residual_d2_abs_max"],
            "target_residual_endpoint_abs_max": target_temporal["residual_endpoint_abs_max"],
            "target_residual_rank1_error_abs_max": target_temporal["residual_rank1_error_abs_max"],
        }

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(critic_state.params)
    metrics["critic_grad_norm"] = _tree_global_norm(grads)
    return critic_state.apply_gradients(grads=grads), metrics


@functools.partial(jax.jit, static_argnames=("cfg",))
def _actor_update(
    key: jax.Array,
    actor_state: TrainState,
    critic_state: TrainState,
    batch: Mapping[str, jnp.ndarray],
    norm: Mapping[str, jnp.ndarray],
    actor_limit: jnp.ndarray,
    beta_bc: jnp.ndarray,
    beta_human_bc: jnp.ndarray,
    beta_human_gripper_bc: jnp.ndarray,
    reference_dropout: jnp.ndarray,
    cfg: RealRLTConfig,
) -> tuple[TrainState, dict[str, jnp.ndarray]]:
    # Normalize first, then replace a random subset by exact zeros.  If zeroing
    # happened in physical coordinates, subsequent normalization would leak the
    # reference mean and would not implement the paper's missing-reference input.
    normalized_ref = _normalize(batch["a_ref"], "a_ref", norm)
    dropped_ref = apply_reference_dropout(key, normalized_ref, dropout=reference_dropout)

    def loss_fn(params: Any) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        if cfg.actor_execution_profile == PERSISTENT_ACTOR_EXECUTION_PROFILE:
            a_pred, actor_execution = _persistent_actor_candidate(
                actor_state,
                params,
                batch["z_rl"],
                batch["state"],
                dropped_ref,
                batch["a_ref"],
                batch["a_base_filtered"],
                batch["actor_persistent_carry_in"],
                batch["actor_persistent_previous_carry"],
                batch["actor_execution_boundary_anchor"],
                batch["execution_filter_alpha"],
                actor_limit,
                norm,
                cfg,
            )
            residual = actor_execution["filtered_residual"]
            temporal = _persistent_temporal_metrics(residual)
        else:
            a_pred = _actor_action(
                actor_state,
                params,
                batch["z_rl"],
                batch["state"],
                dropped_ref,
                batch["a_ref"],
                actor_limit,
                norm,
            )
            residual = a_pred - batch["a_ref"]
            temporal = residual_temporal_metrics(residual)
        q1, q2 = _critic_values(
            critic_state,
            critic_state.params,
            batch["z_rl"],
            batch["state"],
            batch["a_ref"],
            a_pred,
            norm,
        )
        squared_residual = jnp.square(residual)
        # Eq. (5) uses the squared L2 norm of the full C-step action chunk,
        # followed by the expectation over the batch.  Averaging over all 70
        # action elements would weaken beta by C * action_dim and lets the
        # offline actor exploit Q extrapolation instead of staying residual.
        bc_loss = jnp.mean(jnp.sum(squared_residual, axis=(-2, -1)))
        bc_mse = jnp.mean(squared_residual)
        # TD3's delayed policy update optimizes Q1.  The clipped minimum is
        # reserved for critic bootstrapping above.
        q_loss = -jnp.mean(q1)
        human_mask = batch["human_mask"].astype(a_pred.dtype)[..., None]
        human_squared_error = jnp.square(a_pred - batch["a_human"])
        human_bc_loss = jnp.mean(jnp.sum(human_squared_error * human_mask, axis=(-2, -1)))
        if cfg.gripper_residual_mode == GRIPPER_RESIDUAL_CLOSE_ASSIST:
            admitted_human_mask = batch["human_mask"].astype(a_pred.dtype)
            admitted_human_chunk_mask = (
                jnp.max(admitted_human_mask, axis=1) > 0.0
            ).astype(a_pred.dtype)
            (
                human_gripper_candidate,
                human_teacher_direction,
                human_teacher_execution,
            ) = admitted_human_gripper_candidate(
                actor_execution["canonical_direction"],
                batch["a_human"],
                admitted_human_mask,
                batch["a_ref"],
                batch["a_base_filtered"],
                batch["actor_persistent_carry_in"],
                batch["actor_persistent_previous_carry"],
                batch["actor_execution_boundary_anchor"],
                batch["execution_filter_alpha"],
                cfg,
            )
            teacher_gripper = human_gripper_candidate[..., 6]
            q_human1, q_human2 = _critic_values(
                critic_state,
                critic_state.params,
                batch["z_rl"],
                batch["state"],
                batch["a_ref"],
                human_gripper_candidate,
                norm,
            )
            actor_q_min = jnp.minimum(q1, q2)
            human_q_min = jnp.minimum(q_human1, q_human2)
            (
                human_gripper_bc_loss,
                selected_human_chunk_mask,
                selected_human_mask,
                human_q_advantage,
            ) = q_filtered_human_gripper_bc_loss(
                a_pred[..., 6],
                teacher_gripper,
                admitted_human_mask,
                human_q_min,
                actor_q_min,
                margin=cfg.human_gripper_q_filter_margin,
                scale_m=cfg.human_gripper_bc_scale_m,
            )
            admitted_human_fraction = jnp.mean(admitted_human_mask)
            admitted_human_chunks = jnp.sum(admitted_human_chunk_mask)
            admitted_human_steps = jnp.sum(admitted_human_mask)
            q_filter_selected_chunks = jnp.sum(selected_human_chunk_mask)
            q_filter_selected_steps = jnp.sum(selected_human_mask)
            q_filter_fraction = (
                q_filter_selected_chunks
                / jnp.maximum(admitted_human_chunks, 1.0)
            )
            human_teacher_gripper_direction_mean = _masked_mean(
                human_teacher_direction[..., 6],
                admitted_human_chunk_mask,
            )
            human_teacher_gripper_residual = human_teacher_execution[
                "filtered_residual"
            ][..., 6]
            human_teacher_gripper_d1_abs_max = jnp.max(
                jnp.abs(
                    jnp.diff(
                        human_teacher_gripper_residual,
                        axis=1,
                    )
                )
            )
            human_teacher_gripper_d2_abs_max = jnp.max(
                jnp.abs(
                    jnp.diff(
                        human_teacher_gripper_residual,
                        n=2,
                        axis=1,
                    )
                )
            )
            q_advantage_mean = _masked_mean(
                human_q_advantage,
                admitted_human_chunk_mask,
            )
            q_advantage_selected_mean = _masked_mean(
                human_q_advantage,
                selected_human_chunk_mask,
            )
            q_human_mean = _masked_mean(
                human_q_min,
                admitted_human_chunk_mask,
            )
            q_actor_mean = _masked_mean(
                actor_q_min,
                admitted_human_chunk_mask,
            )
            reward1_label = batch["success_mask"].astype(a_pred.dtype)
            reward0_label = 1.0 - reward1_label
            reward1_human_mask = admitted_human_chunk_mask * reward1_label
            reward0_human_mask = admitted_human_chunk_mask * reward0_label
            reward1_human_chunks = jnp.sum(
                reward1_human_mask
            )
            reward0_human_chunks = jnp.sum(
                reward0_human_mask
            )
            reward1_q_advantage_mean = _masked_mean(
                human_q_advantage,
                reward1_human_mask,
            )
            reward0_q_advantage_mean = _masked_mean(
                human_q_advantage,
                reward0_human_mask,
            )
            reward1_selected_chunks = jnp.sum(
                selected_human_chunk_mask * reward1_label
            )
            reward0_selected_chunks = jnp.sum(
                selected_human_chunk_mask * reward0_label
            )
            reward1_q_filter_fraction = (
                reward1_selected_chunks
                / jnp.maximum(reward1_human_chunks, 1.0)
            )
            reward0_q_filter_fraction = (
                reward0_selected_chunks
                / jnp.maximum(reward0_human_chunks, 1.0)
            )
            gripper_residual = residual[..., 6]
            gripper_residual_mean = jnp.mean(gripper_residual)
            gripper_residual_min = jnp.min(gripper_residual)
            gripper_residual_max = jnp.max(gripper_residual)
            gripper_close_fraction = jnp.mean(
                (gripper_residual < -1.0e-6).astype(a_pred.dtype)
            )
            gripper_open_fraction = jnp.mean(
                (gripper_residual > 1.0e-6).astype(a_pred.dtype)
            )
            gripper_saturation_fraction = jnp.mean(
                (
                    gripper_residual
                    <= -cfg.actor_gripper_residual_max_close_m + 1.0e-6
                ).astype(a_pred.dtype)
            )
            gripper_target_min = jnp.min(a_pred[..., 6])
            gripper_target_max = jnp.max(a_pred[..., 6])
            gripper_supervised_steps = q_filter_selected_steps
        else:
            human_gripper_bc_loss = jnp.asarray(0.0, dtype=a_pred.dtype)
            admitted_human_fraction = jnp.asarray(0.0, dtype=a_pred.dtype)
            admitted_human_chunks = jnp.asarray(0.0, dtype=a_pred.dtype)
            admitted_human_steps = jnp.asarray(0.0, dtype=a_pred.dtype)
            q_filter_fraction = jnp.asarray(0.0, dtype=a_pred.dtype)
            q_filter_selected_chunks = jnp.asarray(0.0, dtype=a_pred.dtype)
            q_filter_selected_steps = jnp.asarray(0.0, dtype=a_pred.dtype)
            human_teacher_gripper_direction_mean = jnp.asarray(
                0.0, dtype=a_pred.dtype
            )
            human_teacher_gripper_d1_abs_max = jnp.asarray(
                0.0, dtype=a_pred.dtype
            )
            human_teacher_gripper_d2_abs_max = jnp.asarray(
                0.0, dtype=a_pred.dtype
            )
            q_advantage_mean = jnp.asarray(0.0, dtype=a_pred.dtype)
            q_advantage_selected_mean = jnp.asarray(0.0, dtype=a_pred.dtype)
            q_human_mean = jnp.asarray(0.0, dtype=a_pred.dtype)
            q_actor_mean = jnp.asarray(0.0, dtype=a_pred.dtype)
            reward1_human_chunks = jnp.asarray(0.0, dtype=a_pred.dtype)
            reward0_human_chunks = jnp.asarray(0.0, dtype=a_pred.dtype)
            reward1_q_advantage_mean = jnp.asarray(
                0.0, dtype=a_pred.dtype
            )
            reward0_q_advantage_mean = jnp.asarray(
                0.0, dtype=a_pred.dtype
            )
            reward1_selected_chunks = jnp.asarray(0.0, dtype=a_pred.dtype)
            reward0_selected_chunks = jnp.asarray(0.0, dtype=a_pred.dtype)
            reward1_q_filter_fraction = jnp.asarray(
                0.0, dtype=a_pred.dtype
            )
            reward0_q_filter_fraction = jnp.asarray(
                0.0, dtype=a_pred.dtype
            )
            gripper_residual_mean = jnp.asarray(0.0, dtype=a_pred.dtype)
            gripper_residual_min = jnp.asarray(0.0, dtype=a_pred.dtype)
            gripper_residual_max = jnp.asarray(0.0, dtype=a_pred.dtype)
            gripper_close_fraction = jnp.asarray(0.0, dtype=a_pred.dtype)
            gripper_open_fraction = jnp.asarray(0.0, dtype=a_pred.dtype)
            gripper_saturation_fraction = jnp.asarray(0.0, dtype=a_pred.dtype)
            gripper_target_min = jnp.asarray(0.0, dtype=a_pred.dtype)
            gripper_target_max = jnp.asarray(0.0, dtype=a_pred.dtype)
            gripper_supervised_steps = jnp.asarray(0.0, dtype=a_pred.dtype)
        loss = (
            q_loss
            + beta_bc * bc_loss
            + beta_human_bc * human_bc_loss
            + beta_human_gripper_bc * human_gripper_bc_loss
        )
        return loss, {
            "actor_loss": loss,
            "actor_q_loss": q_loss,
            "actor_bc_loss": bc_loss,
            "actor_bc_mse": bc_mse,
            "actor_human_bc_loss": human_bc_loss,
            "actor_human_fraction": jnp.mean(human_mask),
            "actor_human_gripper_bc_loss": human_gripper_bc_loss,
            "actor_admitted_human_gripper_fraction": admitted_human_fraction,
            "actor_admitted_human_gripper_chunks": admitted_human_chunks,
            "actor_admitted_human_gripper_steps": admitted_human_steps,
            "actor_human_gripper_q_filter_fraction": q_filter_fraction,
            "actor_human_gripper_q_filter_selected_chunks": (
                q_filter_selected_chunks
            ),
            "actor_human_gripper_q_filter_selected_steps": (
                q_filter_selected_steps
            ),
            "actor_human_gripper_teacher_direction_mean_m": (
                human_teacher_gripper_direction_mean
            ),
            "actor_human_gripper_teacher_d1_abs_max_m": (
                human_teacher_gripper_d1_abs_max
            ),
            "actor_human_gripper_teacher_d2_abs_max_m": (
                human_teacher_gripper_d2_abs_max
            ),
            "actor_human_gripper_q_advantage_mean": q_advantage_mean,
            "actor_human_gripper_q_advantage_selected_mean": (
                q_advantage_selected_mean
            ),
            "actor_human_gripper_q_human_mean": q_human_mean,
            "actor_human_gripper_q_actor_mean": q_actor_mean,
            "actor_reward1_human_gripper_chunks": reward1_human_chunks,
            "actor_reward0_human_gripper_chunks": reward0_human_chunks,
            "actor_reward1_human_gripper_q_advantage_mean": (
                reward1_q_advantage_mean
            ),
            "actor_reward0_human_gripper_q_advantage_mean": (
                reward0_q_advantage_mean
            ),
            "actor_reward1_human_gripper_q_filter_selected_chunks": (
                reward1_selected_chunks
            ),
            "actor_reward0_human_gripper_q_filter_selected_chunks": (
                reward0_selected_chunks
            ),
            "actor_reward1_human_gripper_q_filter_fraction": (
                reward1_q_filter_fraction
            ),
            "actor_reward0_human_gripper_q_filter_fraction": (
                reward0_q_filter_fraction
            ),
            "actor_human_gripper_supervised_steps": gripper_supervised_steps,
            "actor_gripper_residual_mean_m": gripper_residual_mean,
            "actor_gripper_residual_min_m": gripper_residual_min,
            "actor_gripper_residual_max_m": gripper_residual_max,
            "actor_gripper_close_fraction": gripper_close_fraction,
            "actor_gripper_open_fraction": gripper_open_fraction,
            "actor_gripper_saturation_fraction": (
                gripper_saturation_fraction
            ),
            "actor_gripper_target_min_m": gripper_target_min,
            "actor_gripper_target_max_m": gripper_target_max,
            "actor_q_mean": jnp.mean(q1),
            "actor_residual_abs_mean": jnp.mean(jnp.abs(residual)),
            "actor_residual_d1_abs_mean": temporal["residual_d1_abs_mean"],
            "actor_residual_d1_abs_max": temporal["residual_d1_abs_max"],
            "actor_residual_d2_abs_mean": temporal["residual_d2_abs_mean"],
            "actor_residual_d2_abs_max": temporal["residual_d2_abs_max"],
            "actor_residual_endpoint_abs_max": temporal["residual_endpoint_abs_max"],
            "actor_residual_rank1_error_abs_max": temporal["residual_rank1_error_abs_max"],
        }

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(actor_state.params)
    metrics["actor_grad_norm"] = _tree_global_norm(grads)
    return actor_state.apply_gradients(grads=grads), metrics


class RealRLTLearner:
    """Offline JAX residual Actor-Critic learner for enriched Piper replay."""

    def __init__(
        self,
        *,
        config: RealRLTConfig,
        normalization: RLTNormalization,
        residual_limit: np.ndarray,
        actor_state: TrainState,
        critic_state: TrainState,
        rng: jax.Array,
        update_step: int = 0,
        fingerprints: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        self.normalization = normalization
        residual_limit = np.asarray(residual_limit, dtype=np.float32).copy()
        if config.freeze_gripper_residual:
            residual_limit[..., -1] = 0.0
        self.residual_limit = jnp.asarray(residual_limit, dtype=jnp.float32)
        expected = (config.chunk_length, config.action_dim)
        if self.residual_limit.shape != expected:
            raise ValueError(f"residual_limit must have shape {expected}, got {self.residual_limit.shape}")
        if not np.all(np.isfinite(residual_limit)) or np.any(residual_limit < 0.0):
            raise ValueError("residual_limit must be finite and non-negative")
        if np.any(residual_limit[..., :-1] <= 0.0):
            raise ValueError("joint residual limits must be positive")
        if not config.freeze_gripper_residual and np.any(residual_limit[..., -1] <= 0.0):
            raise ValueError("gripper residual limits must be positive unless freeze_gripper_residual is enabled")
        self.actor_limit = (
            self.residual_limit
            if config.actor_residual_parameterization == "legacy_full_chunk"
            else rank1_direction_limit(self.residual_limit, config)
        )
        self.actor_state = actor_state
        self.critic_state = critic_state
        self.rng = rng
        self.update_step = int(update_step)
        self.fingerprints = dict(fingerprints or {})
        self._norm_arrays = _normalization_arrays(normalization)

    @classmethod
    def create(
        cls,
        replay: Mapping[str, np.ndarray],
        *,
        config: RealRLTConfig | None = None,
        normalization: RLTNormalization | None = None,
        residual_limit: np.ndarray | float | None = None,
        fingerprints: Mapping[str, str] | None = None,
    ) -> "RealRLTLearner":
        config = config or RealRLTConfig()
        _validate_replay_contract(replay, config)
        normalization = normalization or RLTNormalization.fit(dict(replay), clip=config.normalization_clip)
        if residual_limit is None:
            residual_limit = np.full(
                (config.chunk_length, config.action_dim), config.residual_limit_default, dtype=np.float32
            )
        elif np.isscalar(residual_limit):
            residual_limit = np.full(
                (config.chunk_length, config.action_dim), float(residual_limit), dtype=np.float32
            )
        residual_limit = np.asarray(residual_limit, dtype=np.float32)
        z_dim = int(np.asarray(replay["z_rl"]).shape[-1])
        rng = jax.random.PRNGKey(config.seed)
        init_key, learner_key = jax.random.split(rng)
        actor_state, critic_state = create_train_state(init_key, config, z_dim, jnp.asarray(residual_limit))
        return cls(
            config=config,
            normalization=normalization,
            residual_limit=residual_limit,
            actor_state=actor_state,
            critic_state=critic_state,
            rng=learner_key,
            fingerprints=fingerprints,
        )

    @classmethod
    def warm_start_actor_for_persistent_v2(
        cls,
        source_checkpoint: str | Path,
        replay: Mapping[str, np.ndarray],
        *,
        config: RealRLTConfig | None = None,
        fingerprints: Mapping[str, str],
        expected_source_fingerprints: Mapping[str, str] | None = None,
        allow_objective_migration: bool = False,
    ) -> "RealRLTLearner":
        """Create a new persistent lineage from only the old Actor parameters.

        Critic/target critic, both optimizer states, RNG, and update_step are
        freshly initialized.  Old replay is never accepted by the persistent
        config.  The Actor module architecture is unchanged, so every Actor
        leaf must restore exactly; partial or shape-based loading is forbidden.
        """

        source_checkpoint = Path(source_checkpoint).expanduser().resolve()
        source = cls.load_checkpoint(
            source_checkpoint,
            expected_fingerprints=expected_source_fingerprints,
        )
        if source.config.actor_residual_parameterization != "rank1_bump":
            raise ValueError("persistent-v2 warm-start requires a rank1_bump source Actor")
        source_schema = source.fingerprints.get("action_schema")
        if source_schema != RANK1_ACTOR_OUTPUT_SCHEMA_FINGERPRINT:
            raise ValueError(
                "persistent-v2 warm-start source must use the v3 rank1 Actor schema: "
                f"{source_schema!r}"
            )
        if config is None:
            config = dataclasses.replace(
                source.config,
                actor_execution_profile=PERSISTENT_ACTOR_EXECUTION_PROFILE,
                execution_filter_profile=PERSISTENT_EXECUTION_FILTER_PROFILE,
                execution_filter_tau_s=0.05,
                chunk_stride=10,
            )
        if config.actor_execution_profile != PERSISTENT_ACTOR_EXECUTION_PROFILE:
            raise ValueError("warm-start target config must use persistent_c10_filtered_actual_v2")
        objective_mismatches = {
            name: (getattr(source.config, name), getattr(config, name))
            for name in (
                "beta_bc",
                "beta_human_bc",
                "beta_human_gripper_bc",
            )
            if not np.isclose(
                float(getattr(source.config, name)),
                float(getattr(config, name)),
                rtol=0.0,
                atol=0.0,
            )
        }
        if objective_mismatches and not allow_objective_migration:
            raise ValueError(
                "persistent-v2 Actor-only warm-start must preserve the source "
                "objective weights exactly unless an explicit audited migration "
                f"is enabled: {objective_mismatches}"
            )
        new_fingerprints = {str(key): str(value) for key, value in fingerprints.items()}
        close_assist = (
            config.gripper_residual_mode == GRIPPER_RESIDUAL_CLOSE_ASSIST
        )
        required_fingerprints = {
            "action_schema": (
                PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT
                if close_assist
                else PERSISTENT_ACTION_SCHEMA_FINGERPRINT
            ),
            "actor_execution_profile": PERSISTENT_ACTOR_EXECUTION_PROFILE,
            "execution_filter_profile": PERSISTENT_EXECUTION_FILTER_PROFILE,
            "actor_governor": (
                PERSISTENT_GRIPPER_CLOSE_GOVERNOR_PROFILE
                if close_assist
                else PERSISTENT_GOVERNOR_PROFILE
            ),
        }
        mismatches = {
            key: (expected, new_fingerprints.get(key))
            for key, expected in required_fingerprints.items()
            if new_fingerprints.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"persistent-v2 warm-start fingerprint mismatch: {mismatches}")

        fresh_normalization = RLTNormalization.fit(
            dict(replay),
            clip=config.normalization_clip,
        )
        # Actor inputs retain their source normalization exactly; the Critic's
        # candidate-action normalization is fitted only from new-coordinate
        # persistent replay.
        normalization = RLTNormalization(
            z_rl=source.normalization.z_rl,
            state=source.normalization.state,
            a_ref=source.normalization.a_ref,
            candidate_action=fresh_normalization.candidate_action,
        )
        actor_params_bytes = serialization.msgpack_serialize(
            serialization.to_state_dict(source.actor_state.params)
        )
        source_leaves = jax.tree_util.tree_leaves(source.actor_state.params)
        source_shapes = [tuple(np.asarray(leaf).shape) for leaf in source_leaves]
        shape_report_sha256 = hashlib.sha256(
            json.dumps(source_shapes, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        new_fingerprints.update(
            {
                "warm_start_mode": "actor_params_only_v1",
                "warm_start_actor_source_schema": str(source_schema),
                "warm_start_source_checkpoint": str(source_checkpoint),
                "warm_start_source_checkpoint_sha256": sha256_file(
                    source_checkpoint / "learner.msgpack"
                ),
                "warm_start_actor_params_sha256": hashlib.sha256(
                    actor_params_bytes
                ).hexdigest(),
                "warm_start_actor_param_leaf_count": str(len(source_leaves)),
                "warm_start_actor_param_shapes_sha256": shape_report_sha256,
                "warm_start_normalization": (
                    "reuse_z_state_a_ref_refit_candidate_action_v1"
                ),
                "warm_start_objective_weights": (
                    "explicit_target_beta_migration_v1"
                    if objective_mismatches
                    else "preserve_source_beta_bc_and_beta_human_bc_exactly_v1"
                ),
                "warm_start_source_beta_bc": repr(float(source.config.beta_bc)),
                "warm_start_source_beta_human_bc": repr(
                    float(source.config.beta_human_bc)
                ),
                "warm_start_target_beta_bc": repr(float(config.beta_bc)),
                "warm_start_target_beta_human_bc": repr(
                    float(config.beta_human_bc)
                ),
                "warm_start_source_beta_human_gripper_bc": repr(
                    float(source.config.beta_human_gripper_bc)
                ),
                "warm_start_target_beta_human_gripper_bc": repr(
                    float(config.beta_human_gripper_bc)
                ),
                "warm_start_objective_migration_authorized": str(
                    bool(objective_mismatches and allow_objective_migration)
                ).lower(),
            }
        )
        target_residual_limit = np.asarray(
            source.residual_limit,
            dtype=np.float32,
        ).copy()
        if close_assist:
            # The old seven-output head is shape-compatible, but its gripper
            # limit was exactly zero.  A one-way migration must explicitly
            # open only that limit or the seventh output remains gradient-dead.
            target_residual_limit[..., 6] = (
                config.actor_gripper_residual_max_close_m
            )
        target = cls.create(
            replay,
            config=config,
            normalization=normalization,
            residual_limit=target_residual_limit,
            fingerprints=new_fingerprints,
        )
        target_leaves = jax.tree_util.tree_leaves(target.actor_state.params)
        target_shapes = [tuple(np.asarray(leaf).shape) for leaf in target_leaves]
        if jax.tree_util.tree_structure(source.actor_state.params) != jax.tree_util.tree_structure(
            target.actor_state.params
        ) or source_shapes != target_shapes:
            raise ValueError(
                "Actor architecture changed; exact persistent-v2 warm-start is impossible: "
                f"source_shapes={source_shapes}, target_shapes={target_shapes}"
            )
        # target.actor_state.opt_state stays fresh.  Only params and target
        # params are replaced, with target copied exactly from the source Actor.
        target.actor_state = target.actor_state.replace(
            params=source.actor_state.params,
            target_params=source.actor_state.params,
        )
        return target

    def update(self, batch: Mapping[str, np.ndarray]) -> dict[str, float]:
        jax_batch = _prepare_batch(batch, self.config)
        self.rng, critic_noise_key = jax.random.split(self.rng)
        self.critic_state, critic_metrics = _critic_update(
            critic_noise_key,
            self.actor_state,
            self.critic_state,
            jax_batch,
            self._norm_arrays,
            self.actor_limit,
            jnp.asarray(self.config.target_policy_noise_std, dtype=jnp.float32),
            jnp.asarray(self.config.target_policy_noise_clip, dtype=jnp.float32),
            self.config,
        )
        self.update_step += 1
        actor_burn_in_active = (
            self.update_step <= self.config.actor_start_step
        )
        actor_updated = (
            not actor_burn_in_active
            and self.update_step % self.config.policy_delay == 0
        )
        metrics: dict[str, Any] = dict(critic_metrics)
        if actor_updated:
            self.rng, dropout_key = jax.random.split(self.rng)
            self.actor_state, actor_metrics = _actor_update(
                dropout_key,
                self.actor_state,
                self.critic_state,
                jax_batch,
                self._norm_arrays,
                self.actor_limit,
                jnp.asarray(self.config.beta_bc, dtype=jnp.float32),
                jnp.asarray(self.config.beta_human_bc, dtype=jnp.float32),
                jnp.asarray(
                    self.config.beta_human_gripper_bc,
                    dtype=jnp.float32,
                ),
                jnp.asarray(self.config.reference_dropout, dtype=jnp.float32),
                self.config,
            )
            self.actor_state = self.actor_state.replace(
                target_params=soft_update(self.actor_state.target_params, self.actor_state.params, self.config.tau)
            )
            self.critic_state = self.critic_state.replace(
                target_params=soft_update(self.critic_state.target_params, self.critic_state.params, self.config.tau)
            )
            metrics.update(actor_metrics)
        metrics["actor_updated"] = float(actor_updated)
        metrics["actor_burn_in_active"] = float(actor_burn_in_active)
        metrics["actor_start_step"] = float(self.config.actor_start_step)
        metrics["update_step"] = float(self.update_step)
        return {name: float(np.asarray(jax.device_get(value))) for name, value in metrics.items()}

    def act(
        self,
        z_rl: np.ndarray,
        state: np.ndarray,
        a_ref: np.ndarray,
        *,
        use_target: bool = False,
        reference_visible: bool = True,
    ) -> np.ndarray:
        z_rl, state, a_ref = _ensure_batched_inputs(z_rl, state, a_ref)
        normalized_ref = _normalize(jnp.asarray(a_ref), "a_ref", self._norm_arrays)
        if not reference_visible:
            normalized_ref = jnp.zeros_like(normalized_ref)
        params = self.actor_state.target_params if use_target else self.actor_state.params
        action = _actor_action(
            self.actor_state,
            params,
            jnp.asarray(z_rl),
            jnp.asarray(state),
            normalized_ref,
            jnp.asarray(a_ref),
            self.actor_limit,
            self._norm_arrays,
        )
        return np.asarray(jax.device_get(action), dtype=np.float32)

    def q_values(
        self,
        z_rl: np.ndarray,
        state: np.ndarray,
        a_ref: np.ndarray,
        candidate_action: np.ndarray,
        *,
        use_target: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        z_rl, state, a_ref = _ensure_batched_inputs(z_rl, state, a_ref)
        candidate_action = np.asarray(candidate_action, dtype=np.float32)
        if candidate_action.ndim == 2:
            candidate_action = candidate_action[None]
        params = self.critic_state.target_params if use_target else self.critic_state.params
        q1, q2 = _critic_values(
            self.critic_state,
            params,
            jnp.asarray(z_rl),
            jnp.asarray(state),
            jnp.asarray(a_ref),
            jnp.asarray(candidate_action),
            self._norm_arrays,
        )
        return np.asarray(jax.device_get(q1)), np.asarray(jax.device_get(q2))

    def persistent_candidate_action(
        self,
        z_rl: np.ndarray,
        state: np.ndarray,
        a_ref: np.ndarray,
        *,
        a_base_filtered: np.ndarray,
        carry_in: np.ndarray,
        previous_carry: np.ndarray,
        boundary_anchor: np.ndarray,
        filter_alpha: np.ndarray,
        use_target: bool = False,
        reference_visible: bool = True,
    ) -> np.ndarray:
        """Evaluate the Actor through the physical persistent-v2 execution map.

        ``act`` intentionally remains the raw checkpoint-compatible rank1
        action used by the online service.  This method is the counterfactual
        action consumed by Actor-Q and validation.
        """

        if self.config.actor_execution_profile != PERSISTENT_ACTOR_EXECUTION_PROFILE:
            raise ValueError("persistent_candidate_action requires a persistent-v2 learner")
        z_rl, state, a_ref = _ensure_batched_inputs(z_rl, state, a_ref)
        batch_size = len(z_rl)

        def batched(value: np.ndarray, tail: tuple[int, ...], name: str) -> np.ndarray:
            array = np.asarray(value, dtype=np.float32)
            if array.shape == tail:
                array = array[None]
            if array.shape != (batch_size,) + tail:
                raise ValueError(
                    f"{name} must have shape {tail} or {(batch_size,) + tail}, got {array.shape}"
                )
            return array

        a_base_filtered = batched(
            a_base_filtered,
            (self.config.chunk_length, self.config.action_dim),
            "a_base_filtered",
        )
        carry_in = batched(carry_in, (self.config.action_dim,), "carry_in")
        previous_carry = batched(
            previous_carry, (self.config.action_dim,), "previous_carry"
        )
        boundary_anchor = batched(
            boundary_anchor, (self.config.action_dim,), "boundary_anchor"
        )
        filter_alpha = batched(
            filter_alpha, (self.config.chunk_length,), "filter_alpha"
        )
        normalized_ref = _normalize(jnp.asarray(a_ref), "a_ref", self._norm_arrays)
        if not reference_visible:
            normalized_ref = jnp.zeros_like(normalized_ref)
        params = self.actor_state.target_params if use_target else self.actor_state.params
        candidate, _ = _persistent_actor_candidate(
            self.actor_state,
            params,
            jnp.asarray(z_rl),
            jnp.asarray(state),
            normalized_ref,
            jnp.asarray(a_ref),
            jnp.asarray(a_base_filtered),
            jnp.asarray(carry_in),
            jnp.asarray(previous_carry),
            jnp.asarray(boundary_anchor),
            jnp.asarray(filter_alpha),
            self.actor_limit,
            self._norm_arrays,
            self.config,
        )
        return np.asarray(jax.device_get(candidate), dtype=np.float32)

    def save_checkpoint(self, checkpoint_dir: str | Path) -> dict[str, Any]:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "config": dataclasses.asdict(self.config),
            "actor_state": serialization.to_state_dict(self.actor_state),
            "critic_state": serialization.to_state_dict(self.critic_state),
            "normalization": self.normalization.to_state_dict(),
            "residual_limit": np.asarray(self.residual_limit, dtype=np.float32),
            "rng": np.asarray(self.rng, dtype=np.uint32),
            "update_step": self.update_step,
            "fingerprints": self.fingerprints,
        }
        checkpoint_path = checkpoint_dir / "learner.msgpack"
        temporary_path = checkpoint_dir / "learner.msgpack.tmp"
        temporary_path.write_bytes(serialization.msgpack_serialize(payload))
        temporary_path.replace(checkpoint_path)
        metadata = {
            "format": "openpi_real_rlt_jax_learner",
            "version": CHECKPOINT_VERSION,
            "update_step": self.update_step,
            "config": dataclasses.asdict(self.config),
            "fingerprints": self.fingerprints,
            "files": {"learner": checkpoint_path.name},
        }
        metadata_path = checkpoint_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return metadata

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        *,
        expected_fingerprints: Mapping[str, str] | None = None,
    ) -> "RealRLTLearner":
        checkpoint_dir = Path(checkpoint_dir)
        payload = serialization.msgpack_restore((checkpoint_dir / "learner.msgpack").read_bytes())
        checkpoint_version = int(payload.get("checkpoint_version", 1))
        if checkpoint_version not in {1, 2, CHECKPOINT_VERSION}:
            raise ValueError(f"unsupported learner checkpoint version {checkpoint_version}")
        config_payload = dict(payload["config"])
        # V1/V2 checkpoints predate the safe rank1 contract.  Preserve their
        # original 70-output actor solely for read-only reproduction; silently
        # interpreting those weights as the new 7-output actor is impossible.
        if "actor_residual_parameterization" not in config_payload:
            config_payload["actor_residual_parameterization"] = "legacy_full_chunk"
            config_payload.setdefault("freeze_gripper_residual", False)
        config = RealRLTConfig(**config_payload)
        normalization = RLTNormalization.from_state_dict(payload["normalization"])
        residual_limit = np.asarray(payload["residual_limit"], dtype=np.float32)
        z_dim = int(normalization.z_rl.mean.shape[-1])
        actor_state, critic_state = create_train_state(
            jax.random.PRNGKey(config.seed), config, z_dim, jnp.asarray(residual_limit)
        )
        actor_state = serialization.from_state_dict(actor_state, payload["actor_state"])
        critic_state = serialization.from_state_dict(critic_state, payload["critic_state"])
        fingerprints = {str(key): str(value) for key, value in payload.get("fingerprints", {}).items()}
        if expected_fingerprints:
            mismatches = {
                key: (expected, fingerprints.get(key))
                for key, expected in expected_fingerprints.items()
                if fingerprints.get(key) != expected
            }
            if mismatches:
                raise ValueError(f"checkpoint fingerprint mismatch: {mismatches}")
        return cls(
            config=config,
            normalization=normalization,
            residual_limit=residual_limit,
            actor_state=actor_state,
            critic_state=critic_state,
            rng=jnp.asarray(payload["rng"], dtype=jnp.uint32),
            update_step=int(payload["update_step"]),
            fingerprints=fingerprints,
        )


def estimate_residual_limit(
    replay: Mapping[str, np.ndarray],
    *,
    percentile: float = 99.0,
    minimum: float | np.ndarray = 1e-3,
    maximum: float | np.ndarray | None = None,
    allow_zero_last_action: bool = False,
) -> np.ndarray:
    """Estimates per-step/per-joint residual bounds from executed corrections."""

    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in (0, 100]")
    residual = np.abs(np.asarray(replay["a_exec"], dtype=np.float32) - np.asarray(replay["a_ref"], dtype=np.float32))
    limit = np.percentile(residual, percentile, axis=0).astype(np.float32)
    try:
        limit = np.maximum(limit, np.asarray(minimum, dtype=np.float32))
    except ValueError as exc:
        raise ValueError(f"minimum residual limit is not broadcastable to {limit.shape}") from exc
    if maximum is not None:
        maximum_array = np.asarray(maximum, dtype=np.float32)
        try:
            limit = np.minimum(limit, maximum_array)
        except ValueError as exc:
            raise ValueError(
                f"maximum residual limit is not broadcastable to {limit.shape}: {maximum_array.shape}"
            ) from exc
    if not np.all(np.isfinite(limit)) or np.any(limit < 0.0):
        raise ValueError("estimated residual limits must be finite and non-negative")
    if allow_zero_last_action:
        if np.any(limit[..., :-1] <= 0.0):
            raise ValueError("estimated joint residual limits must be positive")
    elif np.any(limit <= 0.0):
        raise ValueError("estimated residual limits must be positive")
    return limit


def sha256_file(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _prepare_batch(batch: Mapping[str, np.ndarray], config: RealRLTConfig) -> dict[str, jnp.ndarray]:
    missing = set(REQUIRED_TRAIN_BATCH_KEYS).difference(batch)
    if missing:
        raise KeyError(f"batch is missing {sorted(missing)}")
    result = {key: jnp.asarray(batch[key], dtype=jnp.float32) for key in REQUIRED_TRAIN_BATCH_KEYS}
    batch_size = result["reward"].shape[0]
    expected_chunk = (batch_size, config.chunk_length, config.action_dim)
    for name in ("a_ref", "a_exec", "next_a_ref"):
        if result[name].shape != expected_chunk:
            raise ValueError(f"{name} must have shape {expected_chunk}, got {result[name].shape}")
    for name in ("reward", "discount"):
        result[name] = result[name].reshape((batch_size,))
    if "a_human" in batch:
        result["a_human"] = jnp.asarray(batch["a_human"], dtype=jnp.float32)
        if result["a_human"].shape != expected_chunk:
            raise ValueError(f"a_human must have shape {expected_chunk}, got {result['a_human'].shape}")
    else:
        result["a_human"] = jnp.zeros_like(result["a_exec"])

    if "human_mask" in batch:
        human_mask = np.asarray(batch["human_mask"], dtype=np.bool_)
        if human_mask.shape == (batch_size,):
            human_mask = np.broadcast_to(human_mask[:, None], (batch_size, config.chunk_length))
        elif human_mask.ndim > 2 and human_mask.shape[:2] == (batch_size, config.chunk_length):
            human_mask = np.any(human_mask, axis=tuple(range(2, human_mask.ndim)))
        if human_mask.shape != (batch_size, config.chunk_length):
            raise ValueError(
                f"human_mask must have shape {(batch_size,)} or "
                f"{(batch_size, config.chunk_length)}, got {human_mask.shape}"
            )
        result["human_mask"] = jnp.asarray(human_mask, dtype=jnp.float32)
    else:
        result["human_mask"] = jnp.zeros((batch_size, config.chunk_length), dtype=jnp.float32)
    if "success_mask" in batch:
        success_mask = np.asarray(batch["success_mask"], dtype=np.bool_)
        if success_mask.shape != (batch_size,):
            raise ValueError(
                f"success_mask must have shape {(batch_size,)}, got {success_mask.shape}"
            )
        result["success_mask"] = jnp.asarray(success_mask, dtype=jnp.float32)
    else:
        result["success_mask"] = jnp.zeros((batch_size,), dtype=jnp.float32)
    if config.actor_execution_profile == PERSISTENT_ACTOR_EXECUTION_PROFILE:
        missing_persistent = set(PERSISTENT_TRAIN_BATCH_KEYS).difference(batch)
        if missing_persistent:
            raise KeyError(
                "persistent-v2 batch is missing "
                f"{sorted(missing_persistent)}"
            )
        vector_names = (
            "actor_canonical_decision",
            "actor_persistent_carry_in",
            "actor_persistent_carry_out",
            "actor_persistent_previous_carry",
            "actor_execution_boundary_anchor",
            "next_actor_persistent_previous_carry",
            "next_actor_persistent_carry_in",
            "next_actor_execution_boundary_anchor",
        )
        for name in vector_names:
            value = jnp.asarray(batch[name], dtype=jnp.float32)
            expected = (batch_size, config.action_dim)
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
            result[name] = value
        chunk_names = (
            "a_base_filtered",
            "a_filtered_actual",
            "filtered_actual_residual",
            "next_a_base_filtered",
        )
        for name in chunk_names:
            value = jnp.asarray(batch[name], dtype=jnp.float32)
            if value.shape != expected_chunk:
                raise ValueError(f"{name} must have shape {expected_chunk}, got {value.shape}")
            result[name] = value
        scalar_chunk_names = (
            "execution_filter_tau_s",
            "execution_filter_dt_s",
            "execution_filter_alpha",
            "execution_projection_scale",
            "next_execution_filter_alpha",
        )
        for name in scalar_chunk_names:
            value = jnp.asarray(batch[name], dtype=jnp.float32)
            expected = (batch_size, config.chunk_length)
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
            result[name] = value
    return result


def _validate_replay_contract(replay: Mapping[str, np.ndarray], config: RealRLTConfig) -> None:
    missing = set(REQUIRED_TRAIN_BATCH_KEYS).difference(replay)
    if missing:
        raise KeyError(f"replay is missing {sorted(missing)}")
    if config.chunk_length != 10 or config.n_step != 10:
        raise ValueError("the first real-Piper RLT reproduction requires C=10 and n_step=10")
    n = len(replay["reward"])
    if n == 0:
        raise ValueError("replay must contain at least one transition")
    expected_chunk = (n, config.chunk_length, config.action_dim)
    for name in ("a_ref", "a_exec", "next_a_ref"):
        if np.asarray(replay[name]).shape != expected_chunk:
            raise ValueError(f"{name} must have shape {expected_chunk}, got {np.asarray(replay[name]).shape}")
    expected_state = (n, config.state_dim)
    for name in ("state", "next_state"):
        if np.asarray(replay[name]).shape != expected_state:
            raise ValueError(f"{name} must have shape {expected_state}, got {np.asarray(replay[name]).shape}")
    z_shape = np.asarray(replay["z_rl"]).shape
    next_z_shape = np.asarray(replay["next_z_rl"]).shape
    if len(z_shape) != 2 or z_shape[0] != n or z_shape[1] <= 0:
        raise ValueError(f"z_rl must have shape (N, Z), got {z_shape}")
    if next_z_shape != z_shape:
        raise ValueError(f"next_z_rl must match z_rl shape {z_shape}, got {next_z_shape}")
    for name in ("reward", "discount"):
        if np.asarray(replay[name]).shape != (n,):
            raise ValueError(f"{name} must have shape ({n},), got {np.asarray(replay[name]).shape}")
    discount = np.asarray(replay["discount"], dtype=np.float32)
    if np.any(discount < 0.0) or np.any(discount > 1.0 + 1e-6):
        raise ValueError("discount must stay within [0, 1]")
    if "done" in replay:
        done = np.asarray(replay["done"], dtype=np.bool_)
        if done.shape != (n,):
            raise ValueError(f"done must have shape ({n},), got {done.shape}")
        if np.any(discount[done] != 0.0):
            raise ValueError("terminal transitions must have zero bootstrap discount")
        expected_discount = config.gamma**config.n_step
        if np.any(~np.isclose(discount[~done], expected_discount, rtol=1e-5, atol=1e-7)):
            raise ValueError(f"nonterminal transitions must use gamma**n_step={expected_discount}")
    if "a_human" in replay and np.asarray(replay["a_human"]).shape != expected_chunk:
        raise ValueError(f"a_human must have shape {expected_chunk}, got {np.asarray(replay['a_human']).shape}")
    if "human_mask" in replay:
        human_shape = np.asarray(replay["human_mask"]).shape
        if human_shape not in ((n,), (n, config.chunk_length)) and not (
            len(human_shape) > 2 and human_shape[:2] == (n, config.chunk_length)
        ):
            raise ValueError(f"human_mask has incompatible shape {human_shape}")
    for name in REQUIRED_TRAIN_BATCH_KEYS + (("a_human",) if "a_human" in replay else ()):
        if not np.all(np.isfinite(np.asarray(replay[name], dtype=np.float32))):
            raise ValueError(f"replay array {name!r} contains NaN or inf")
    has_persistent = "actor_execution_profile" in replay
    expects_persistent = (
        config.actor_execution_profile == PERSISTENT_ACTOR_EXECUTION_PROFILE
    )
    if has_persistent != expects_persistent:
        raise ValueError(
            "legacy and persistent-v2 replay/learner contracts cannot be mixed: "
            f"replay_persistent={has_persistent}, learner_profile={config.actor_execution_profile!r}"
        )
    if not expects_persistent:
        return
    required_persistent_keys = PERSISTENT_TRAIN_BATCH_KEYS + PERSISTENT_AUDIT_KEYS
    if config.gripper_residual_mode == GRIPPER_RESIDUAL_CLOSE_ASSIST:
        required_persistent_keys += GRIPPER_PERSISTENT_AUDIT_KEYS
    missing_persistent = set(required_persistent_keys).difference(replay)
    if missing_persistent:
        raise KeyError(
            f"persistent-v2 replay is missing {sorted(missing_persistent)}"
        )
    profiles = set(np.asarray(replay["actor_execution_profile"]).astype(str).tolist())
    if not profiles or not profiles.issubset(
        {PERSISTENT_ACTOR_EXECUTION_PROFILE, HUMAN_EXECUTION_PROFILE}
    ):
        raise ValueError(f"unsupported persistent-v2 execution profiles: {sorted(profiles)}")
    schemas = set(np.asarray(replay.get("action_schema_fingerprint", [])).astype(str).tolist())
    expected_schema = (
        PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT
        if config.gripper_residual_mode == GRIPPER_RESIDUAL_CLOSE_ASSIST
        else PERSISTENT_ACTION_SCHEMA_FINGERPRINT
    )
    if schemas != {expected_schema}:
        raise ValueError(
            "persistent-v2 action schema mismatch: "
            f"{sorted(schemas)} != {[expected_schema]}"
        )
    filter_profiles = set(
        np.asarray(replay.get("execution_filter_profile", [])).astype(str).tolist()
    )
    if filter_profiles != {PERSISTENT_EXECUTION_FILTER_PROFILE}:
        raise ValueError(
            "persistent-v2 execution filter mismatch: "
            f"{sorted(filter_profiles)} != {[PERSISTENT_EXECUTION_FILTER_PROFILE]}"
        )
    expected_vector = (n, config.action_dim)
    for name in (
        "actor_canonical_decision",
        "actor_persistent_carry_in",
        "actor_persistent_carry_out",
        "actor_persistent_previous_carry",
        "actor_execution_boundary_anchor",
        "next_actor_persistent_previous_carry",
        "next_actor_persistent_carry_in",
        "next_actor_execution_boundary_anchor",
    ):
        if np.asarray(replay[name]).shape != expected_vector:
            raise ValueError(
                f"{name} must have shape {expected_vector}, got {np.asarray(replay[name]).shape}"
            )
    for name in (
        "a_base_filtered",
        "a_filtered_actual",
        "filtered_actual_residual",
        "next_a_base_filtered",
    ):
        if np.asarray(replay[name]).shape != expected_chunk:
            raise ValueError(
                f"{name} must have shape {expected_chunk}, got {np.asarray(replay[name]).shape}"
            )
    expected_scalar_chunk = (n, config.chunk_length)
    for name in (
        "execution_filter_tau_s",
        "execution_filter_dt_s",
        "execution_filter_alpha",
        "execution_projection_scale",
        "next_execution_filter_alpha",
    ):
        values = np.asarray(replay[name], dtype=np.float32)
        if values.shape != expected_scalar_chunk:
            raise ValueError(
                f"{name} must have shape {expected_scalar_chunk}, got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains NaN or inf")
    if not np.allclose(
        np.asarray(replay["a_exec"], dtype=np.float32),
        np.asarray(replay["a_filtered_actual"], dtype=np.float32),
        rtol=0.0,
        atol=2e-6,
    ):
        raise ValueError("critic behavior action a_exec must equal the final filtered actual action")
    alpha = np.asarray(replay["execution_filter_alpha"], dtype=np.float64)
    dt = np.asarray(replay["execution_filter_dt_s"], dtype=np.float64)
    tau = np.asarray(replay["execution_filter_tau_s"], dtype=np.float64)
    if np.any(dt <= 0.0) or np.any(tau <= 0.0):
        raise ValueError("persistent-v2 execution filter dt/tau must be positive")
    if not np.allclose(alpha, 1.0 - np.exp(-dt / tau), rtol=2e-5, atol=1e-7):
        raise ValueError("persistent-v2 execution alpha does not match 1-exp(-dt/tau)")
    if not np.allclose(
        tau,
        config.execution_filter_tau_s,
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError(
            "replay execution filter tau does not match learner config "
            f"{config.execution_filter_tau_s}"
        )
    envelope_expected = {
        "execution_residual_max_rad": config.actor_residual_max_rad,
        "execution_d1_max_rad": config.actor_residual_d1_max_rad,
        "execution_d2_max_rad": config.actor_residual_d2_max_rad,
        "execution_direction_cone_deg": config.actor_direction_cone_deg,
        "execution_boundary_limit_rad": config.actor_max_boundary_jump_rad,
        "execution_projection_scale_steps": float(config.actor_projection_scale_steps),
        "execution_min_projection_scale": config.actor_min_projection_scale,
        "execution_direction_static_threshold_rad": (
            config.actor_direction_static_threshold_rad
        ),
    }
    if config.gripper_residual_mode == GRIPPER_RESIDUAL_CLOSE_ASSIST:
        envelope_expected.update(
            {
                "execution_gripper_residual_max_close_m": (
                    config.actor_gripper_residual_max_close_m
                ),
                "execution_gripper_d1_max_m": (
                    config.actor_gripper_residual_d1_max_m
                ),
                "execution_gripper_d2_max_m": (
                    config.actor_gripper_residual_d2_max_m
                ),
                "execution_gripper_boundary_limit_m": (
                    config.actor_gripper_max_boundary_jump_m
                ),
                "execution_gripper_command_min_m": config.gripper_command_min_m,
                "execution_gripper_command_max_m": config.gripper_command_max_m,
                "execution_gripper_release_reference_m": (
                    config.gripper_release_reference_m
                ),
                "execution_gripper_release_delta_m": (
                    config.gripper_release_delta_m
                ),
            }
        )
    for name, expected in envelope_expected.items():
        values = np.asarray(replay[name], dtype=np.float64)
        if values.shape != (n, config.chunk_length):
            raise ValueError(
                f"{name} must have shape {(n, config.chunk_length)}, got {values.shape}"
            )
        if not np.allclose(values, expected, rtol=0.0, atol=1e-7):
            raise ValueError(
                f"replay runtime envelope {name} does not match learner config {expected}"
            )


def _ensure_batched_inputs(
    z_rl: np.ndarray, state: np.ndarray, a_ref: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_rl = np.asarray(z_rl, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32)
    a_ref = np.asarray(a_ref, dtype=np.float32)
    if z_rl.ndim == 1:
        z_rl = z_rl[None]
    if state.ndim == 1:
        state = state[None]
    if a_ref.ndim == 2:
        a_ref = a_ref[None]
    if not (len(z_rl) == len(state) == len(a_ref)):
        raise ValueError("z_rl, state, and a_ref batch dimensions differ")
    return z_rl, state, a_ref
