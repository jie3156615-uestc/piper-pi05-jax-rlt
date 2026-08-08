from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from openpi.rlt.real.config import Source
from openpi.rlt.real.replay import RealStepRecord


ABSOLUTE_ACTION_SPACE = "joint_absolute_gripper_absolute"
DELTA_ACTION_SPACE = "joint_delta_gripper_absolute"
VALID_EXECUTION_SOURCES = frozenset({Source.PI05, Source.RLT, Source.HUMAN_PIKA})
INVALID_EXECUTION_SOURCES = frozenset({Source.STOP, Source.SAFETY_BLOCK, "wait", "hold"})


@dataclasses.dataclass(frozen=True)
class ReferenceTokenValue:
    """Frozen-base output associated with one observation.

    ``a_ref`` may be absolute or joint-delta according to ``action_space``. It
    is normalized to absolute command space before being attached to a replay
    row. ``z_rl`` is always the frozen RL Token encoder output.
    """

    a_ref: np.ndarray
    z_rl: np.ndarray
    action_space: str = ABSOLUTE_ACTION_SPACE


class ReferenceTokenProvider(Protocol):
    def __call__(self, record: RealStepRecord, episode_root: Path) -> ReferenceTokenValue: ...


class PhaseProbabilityProvider(Protocol):
    def __call__(self, record: RealStepRecord, episode_root: Path) -> float: ...


@dataclasses.dataclass(frozen=True)
class EpisodeEnrichment:
    records: list[RealStepRecord]
    segments: list[list[RealStepRecord]]
    statistics: dict[str, Any]


@dataclasses.dataclass
class SingleLatchGate:
    enter_threshold: float = 0.5
    enter_frames: int = 3

    def __post_init__(self) -> None:
        if not 0.0 <= self.enter_threshold <= 1.0:
            raise ValueError("enter_threshold must be within [0, 1]")
        if self.enter_frames <= 0:
            raise ValueError("enter_frames must be positive")
        self.active = False
        self.high_count = 0
        self.enter_t: int | None = None

    def update(self, probability: float, *, t: int) -> bool:
        if self.active:
            return True
        if probability >= self.enter_threshold:
            self.high_count += 1
        else:
            self.high_count = 0
        if self.high_count >= self.enter_frames:
            self.active = True
            self.enter_t = int(t)
        return self.active


class CachedEnrichmentProvider:
    """JSONL-backed provider usable for reproducible/offline enrichment.

    Each line must contain ``episode_id`` and ``t``. Reference rows additionally
    contain ``a_ref`` (or ``a_ref_absolute``), ``z_rl`` and optional
    ``action_space``. Phase rows contain ``phase_probability``. A single cache
    can implement both provider protocols.
    """

    def __init__(self, values: Mapping[tuple[str, int], Mapping[str, Any]]) -> None:
        self._values = dict(values)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        episode_ids: set[str] | None = None,
    ) -> "CachedEnrichmentProvider":
        values: dict[tuple[str, int], Mapping[str, Any]] = {}
        path = Path(path)
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                if episode_ids is not None:
                    episode_id = _compact_json_episode_id(line)
                    if (
                        episode_id is not None
                        and episode_id not in episode_ids
                    ):
                        continue
                row = json.loads(line)
                try:
                    key = (str(row["episode_id"]), int(row["t"]))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid enrichment cache key on line {line_number}"
                    ) from exc
                if episode_ids is not None and key[0] not in episode_ids:
                    continue
                if key in values:
                    raise ValueError(f"duplicate enrichment cache key: {key}")
                values[key] = row
        return cls(values)

    def __call__(self, record: RealStepRecord, episode_root: Path) -> ReferenceTokenValue:
        del episode_root
        row = self._get(record)
        action = row.get("a_ref_absolute", row.get("a_ref"))
        if action is None or "z_rl" not in row:
            raise KeyError(f"cache entry {(record.episode_id, record.t)} lacks a_ref/z_rl")
        return ReferenceTokenValue(
            a_ref=np.asarray(action, dtype=np.float32),
            z_rl=np.asarray(row["z_rl"], dtype=np.float32),
            action_space=str(row.get("action_space", ABSOLUTE_ACTION_SPACE)),
        )

    def phase_probability(self, record: RealStepRecord, episode_root: Path) -> float:
        del episode_root
        row = self._get(record)
        if "phase_probability" not in row:
            raise KeyError(f"cache entry {(record.episode_id, record.t)} lacks phase_probability")
        return float(row["phase_probability"])

    def _get(self, record: RealStepRecord) -> Mapping[str, Any]:
        key = (record.episode_id, int(record.t))
        try:
            return self._values[key]
        except KeyError as exc:
            raise KeyError(f"enrichment cache has no entry for {key}") from exc


def _compact_json_episode_id(line: str) -> str | None:
    marker = '"episode_id":"'
    start = line.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = line.find('"', start)
    if end < 0:
        return None
    return line[start:end]


def load_callback(spec: str) -> Callable[..., Any]:
    """Load a provider callback from ``package.module:function``."""

    if ":" not in spec:
        raise ValueError("provider callback must use package.module:function syntax")
    module_name, attribute = spec.rsplit(":", 1)
    callback = getattr(importlib.import_module(module_name), attribute)
    if not callable(callback):
        raise TypeError(f"provider is not callable: {spec}")
    return callback


def derive_episode_uid(path: str | Path, *, root: str | Path | None = None) -> str:
    """Derive the stable cache/replay episode key for one ``episode.jsonl``.

    Cache generation and replay preparation must use exactly the same key.  A
    dataset root is strongly recommended because it makes the identifier
    independent of where the dataset is mounted.
    """

    path = Path(path).resolve()
    parent = path.parent
    if root is not None:
        root_path = Path(root).resolve()
        try:
            value = parent.relative_to(root_path).as_posix()
        except ValueError as exc:
            raise ValueError(f"episode is outside dataset_root: {path}") from exc
    else:
        value = "/".join(parent.parts[-2:])
    if not value or value == ".":
        value = parent.name
    return value


def enrich_episode(
    records: Sequence[RealStepRecord],
    *,
    episode_root: str | Path,
    reference_provider: ReferenceTokenProvider | None,
    phase_provider: PhaseProbabilityProvider | None,
    allow_logged_reference: bool = False,
    allow_logged_phase: bool = False,
    preserve_logged_reference: bool = False,
    enter_threshold: float = 0.5,
    enter_frames: int = 3,
) -> EpisodeEnrichment:
    """Recompute frozen-base fields, latch phase, filter and migrate terminal.

    Invalid/wait rows are never returned. They split contiguous segments so
    action chunks cannot silently bridge a hold, safety block, or reward wait.
    Terminal reward/done are copied from the raw terminal row to the last valid
    executed command in the episode.
    """

    if not records:
        raise ValueError("cannot enrich an empty episode")
    if reference_provider is None and not allow_logged_reference:
        raise ValueError("a frozen-base reference provider/cache is required; logged a_ref is audit-only")
    if preserve_logged_reference and reference_provider is None:
        raise ValueError("preserve_logged_reference requires a frozen reference/Token provider or cache")
    if phase_provider is None and not allow_logged_phase:
        raise ValueError("a frozen phase provider/cache is required")

    episode_root = Path(episode_root)
    gate = SingleLatchGate(enter_threshold=enter_threshold, enter_frames=enter_frames)
    filtered_reasons: Counter[str] = Counter()
    enriched_by_index: dict[int, RealStepRecord] = {}
    valid_indices: list[int] = []

    terminal_reward = 0.0
    terminal_seen = False
    for index, record in enumerate(records):
        if record.done:
            terminal_seen = True
            terminal_reward = float(record.reward)

        if phase_provider is not None:
            probability = float(phase_provider(record, episode_root))
        else:
            probability = float(record.phase_probability)
        if not np.isfinite(probability):
            raise ValueError(f"non-finite phase probability at t={record.t}")
        probability = float(np.clip(probability, 0.0, 1.0))
        gate_active = gate.update(probability, t=record.t)

        reason = exclusion_reason(record)
        if reason is None and not gate_active:
            reason = "gate_inactive"
        if reason is not None:
            filtered_reasons[reason] += 1
            continue

        if reference_provider is not None:
            value = reference_provider(record, episode_root)
            z_rl = _require_finite_vector(value.z_rl, "z_rl", t=record.t)
            if preserve_logged_reference:
                # The online Pi0.5 policy is stochastic. Re-running it after an
                # episode is useful for a fresh per-frame Token, but its newly
                # sampled action must not replace the exact reference that the
                # behavior policy saw while the robot executed this row.
                ref_absolute = _require_action_chunk(record.a_ref, "logged a_ref", t=record.t)
                reference_recomputed = False
            else:
                ref_absolute = reference_to_absolute(value, state=record.state)
                reference_recomputed = True
            token_recomputed = True
        else:
            ref_absolute = _require_action_chunk(record.a_ref, "logged a_ref", t=record.t)
            z_rl = _require_finite_vector(record.z_rl, "logged z_rl", t=record.t)
            reference_recomputed = False
            token_recomputed = False

        enriched_by_index[index] = dataclasses.replace(
            record,
            a_ref=ref_absolute,
            z_rl=z_rl,
            reward=0.0,
            done=False,
            phase_probability=probability,
            gate_active=gate_active,
            a_ref_original=np.asarray(record.a_ref, dtype=np.float32),
            reference_recomputed=reference_recomputed,
            token_recomputed=token_recomputed,
        )
        valid_indices.append(index)

    if not terminal_seen:
        raise ValueError("episode has no terminal row; terminal reward cannot be migrated safely")

    if not valid_indices:
        return EpisodeEnrichment(
            records=[],
            segments=[],
            statistics={
                "raw_rows": len(records),
                "valid_action_rows": 0,
                "filtered_rows": int(sum(filtered_reasons.values())),
                "filtered_by_reason": dict(sorted(filtered_reasons.items())),
                "segments": 0,
                "source_steps": {},
                "human_steps": 0,
                "actor_steps": 0,
                "terminal_reward": terminal_reward,
                "gate_enter_t": gate.enter_t,
                "gate_ever_active": gate.active,
                "reference_recomputed": bool(reference_provider is not None and not preserve_logged_reference),
                "token_recomputed": reference_provider is not None,
                "logged_reference_preserved": bool(reference_provider is not None and preserve_logged_reference),
                "skipped_reason": "no_gate_active_valid_action",
            },
        )

    last_index = valid_indices[-1]
    enriched_by_index[last_index] = dataclasses.replace(
        enriched_by_index[last_index], reward=terminal_reward, done=True
    )

    # Preserve invalid gaps as segment boundaries rather than compacting them.
    segments: list[list[RealStepRecord]] = []
    current: list[RealStepRecord] = []
    previous_raw_index: int | None = None
    for raw_index in valid_indices:
        if previous_raw_index is not None and raw_index != previous_raw_index + 1:
            segments.append(current)
            current = []
        current.append(enriched_by_index[raw_index])
        previous_raw_index = raw_index
    if current:
        segments.append(current)

    enriched = [record for segment in segments for record in segment]
    source_counts = Counter(record.source for record in enriched)
    return EpisodeEnrichment(
        records=enriched,
        segments=segments,
        statistics={
            "raw_rows": len(records),
            "valid_action_rows": len(enriched),
            "filtered_rows": int(sum(filtered_reasons.values())),
            "filtered_by_reason": dict(sorted(filtered_reasons.items())),
            "segments": len(segments),
            "source_steps": dict(sorted(source_counts.items())),
            "human_steps": int(sum(record.a_human is not None for record in enriched)),
            "actor_steps": int(sum(record.a_actor is not None for record in enriched)),
            "terminal_reward": terminal_reward,
            "gate_enter_t": gate.enter_t,
            "gate_ever_active": gate.active,
            "reference_recomputed": bool(reference_provider is not None and not preserve_logged_reference),
            "token_recomputed": reference_provider is not None,
            "logged_reference_preserved": bool(reference_provider is not None and preserve_logged_reference),
        },
    )


def exclusion_reason(record: RealStepRecord) -> str | None:
    if record.source in INVALID_EXECUTION_SOURCES:
        return f"source:{record.source}"
    if record.source not in VALID_EXECUTION_SOURCES:
        return f"source:unknown:{record.source}"
    if not record.replay_include:
        return "replay_include_false"
    if record.waiting_for_reward:
        return "waiting_for_reward"
    if not np.all(np.isfinite(np.asarray(record.a_exec, dtype=np.float32))):
        return "nonfinite_a_exec"
    return None


def reference_to_absolute(value: ReferenceTokenValue, *, state: np.ndarray) -> np.ndarray:
    action = _require_action_chunk(value.a_ref, "recomputed a_ref")
    if value.action_space == ABSOLUTE_ACTION_SPACE:
        return action
    if value.action_space == DELTA_ACTION_SPACE:
        result = action.copy()
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (7,):
            raise ValueError(f"state must have shape (7,), got {state.shape}")
        result[..., :6] += state[:6]
        return result
    raise ValueError(f"unsupported reference action space: {value.action_space}")


def assign_episode_splits(
    episode_ids: Sequence[str],
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.0,
    seed: str = "piper-rlt-v1",
) -> dict[str, list[str]]:
    """Deterministically assign whole episodes, never transitions, to splits."""

    unique = sorted(set(episode_ids))
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation/test fractions must be non-negative and sum to less than one")
    ranked = sorted(unique, key=lambda item: hashlib.sha256(f"{seed}:{item}".encode()).hexdigest())
    count = len(ranked)
    n_test = min(count, int(round(count * test_fraction)))
    n_val = min(count - n_test, int(round(count * validation_fraction)))
    return {
        "train": sorted(ranked[n_test + n_val :]),
        "validation": sorted(ranked[n_test : n_test + n_val]),
        "test": sorted(ranked[:n_test]),
    }


def assign_stratified_episode_splits(
    episode_labels: Mapping[str, str],
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.0,
    seed: str = "piper-rlt-v1",
) -> dict[str, list[str]]:
    """Assign whole episodes while preserving success/failure representation.

    Sparse terminal reward makes an all-failure validation set misleading for
    this task.  Each non-empty label group is ranked independently and then
    merged, while every episode still belongs to exactly one split.
    """

    grouped: dict[str, list[str]] = {}
    for episode_id, label in episode_labels.items():
        grouped.setdefault(str(label), []).append(str(episode_id))
    if not grouped:
        return {"train": [], "validation": [], "test": []}

    merged = {"train": [], "validation": [], "test": []}
    for label, episode_ids in sorted(grouped.items()):
        group_validation_fraction = validation_fraction
        if (
            validation_fraction > 0.0
            and len(episode_ids) >= 2
            and (1.0 / len(episode_ids)) + test_fraction < 1.0
        ):
            group_validation_fraction = max(validation_fraction, 1.0 / len(episode_ids))
        group_split = assign_episode_splits(
            episode_ids,
            validation_fraction=group_validation_fraction,
            test_fraction=test_fraction,
            seed=f"{seed}:{label}",
        )
        for split_name in merged:
            merged[split_name].extend(group_split[split_name])
    return {name: sorted(values) for name, values in merged.items()}


def _require_action_chunk(value: Any, name: str, *, t: int | None = None) -> np.ndarray:
    action = np.asarray(value, dtype=np.float32)
    if action.shape == (7,):
        action = action[None, :]
    if action.ndim != 2 or action.shape[0] == 0 or action.shape[1] != 7:
        location = "" if t is None else f" at t={t}"
        raise ValueError(f"{name}{location} must have shape (7,) or (N, 7), got {action.shape}")
    if not np.all(np.isfinite(action)):
        raise ValueError(f"{name} contains non-finite values")
    return action


def _require_finite_vector(value: Any, name: str, *, t: int | None = None) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1 or vector.shape[0] == 0 or not np.all(np.isfinite(vector)):
        location = "" if t is None else f" at t={t}"
        raise ValueError(f"{name}{location} must be a finite non-empty vector")
    return vector
