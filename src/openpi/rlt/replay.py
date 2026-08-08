from __future__ import annotations

import collections
import dataclasses
import random
from collections import deque
from collections.abc import Iterable

import numpy as np
import torch

from openpi.rlt.config import RLTReplayConfig
from openpi.rlt.networks import RLTBatch


@dataclasses.dataclass
class RLTTransition:
    rl_token: np.ndarray
    state: np.ndarray
    action: np.ndarray
    reference_action: np.ndarray
    reward: float
    discount: float
    next_rl_token: np.ndarray
    next_state: np.ndarray
    next_reference_action: np.ndarray
    done: bool = False
    source: str = "rl"


class RLTReplayBuffer:
    def __init__(self, config: RLTReplayConfig):
        self.config = config
        self._storage: deque[RLTTransition] = deque(maxlen=config.capacity)
        self._demo_storage: list[RLTTransition] = []
        self._online_positive_storage: deque[RLTTransition] = deque(maxlen=config.capacity)

    def __len__(self) -> int:
        return len(self._storage) + len(self._demo_storage) + len(self._online_positive_storage)

    def add(self, transition: RLTTransition) -> None:
        if transition.reward > 0.0:
            self._online_positive_storage.append(transition)
        else:
            self._storage.append(transition)

    def extend(self, transitions: Iterable[RLTTransition]) -> None:
        for transition in transitions:
            self.add(transition)

    def add_demo(self, transition: RLTTransition) -> None:
        self._demo_storage.append(transition)

    def extend_demo(self, transitions: Iterable[RLTTransition]) -> None:
        for transition in transitions:
            self.add_demo(transition)

    def state_dict(self) -> dict:
        return {
            "config": dataclasses.asdict(self.config),
            "storage": list(self._storage),
            "demo_storage": list(self._demo_storage),
            "online_positive_storage": list(self._online_positive_storage),
        }

    def load_state_dict(self, state: dict) -> None:
        config = state.get("config")
        if config is not None:
            allowed = {field.name for field in dataclasses.fields(RLTReplayConfig)}
            loaded = RLTReplayConfig(**{key: value for key, value in config.items() if key in allowed})
            self.config = dataclasses.replace(
                loaded,
                demo_sample_fraction=self.config.demo_sample_fraction,
                online_positive_sample_fraction=self.config.online_positive_sample_fraction,
                positive_sample_fraction=self.config.positive_sample_fraction,
                batch_size=self.config.batch_size,
            )
        storage = list(state.get("storage", []))
        has_split_storage = "demo_storage" in state or "online_positive_storage" in state
        self._demo_storage = list(state.get("demo_storage", []))
        positives = state.get("online_positive_storage")
        if positives is None:
            positives = [transition for transition in storage if transition.reward > 0.0]
            storage = [transition for transition in storage if transition.reward <= 0.0]
        self._storage = deque(storage, maxlen=self.config.capacity)
        self._online_positive_storage = deque(positives, maxlen=self.config.capacity)
        if not has_split_storage and self.config.demo_sample_fraction > 0.0:
            self.promote_online_to_demo()

    def promote_online_to_demo(self) -> None:
        self._demo_storage.extend(self._storage)
        self._demo_storage.extend(self._online_positive_storage)
        self._storage.clear()
        self._online_positive_storage.clear()

    def stats(self) -> dict[str, float | int]:
        storage = [*self._demo_storage, *self._storage, *self._online_positive_storage]
        count = len(storage)
        if count == 0:
            return {
                "replay_size": 0,
                "replay_demo_size": 0,
                "replay_online_size": 0,
                "replay_online_positive_size": 0,
                "replay_reward_positive": 0,
                "replay_reward_positive_frac": 0.0,
                "replay_reward_mean": 0.0,
                "replay_terminal": 0,
                "replay_terminal_frac": 0.0,
                "replay_source_rl": 0,
                "replay_source_vla": 0,
            }

        reward_positive = sum(1 for transition in storage if transition.reward > 0.0)
        reward_sum = sum(float(transition.reward) for transition in storage)
        terminal = sum(1 for transition in storage if transition.done)
        source_counts = collections.Counter(transition.source for transition in storage)
        return {
            "replay_size": count,
            "replay_demo_size": len(self._demo_storage),
            "replay_online_size": len(self._storage),
            "replay_online_positive_size": len(self._online_positive_storage),
            "replay_reward_positive": reward_positive,
            "replay_reward_positive_frac": reward_positive / count,
            "replay_reward_mean": reward_sum / count,
            "replay_terminal": terminal,
            "replay_terminal_frac": terminal / count,
            "replay_source_rl": source_counts.get("rl", 0),
            "replay_source_vla": source_counts.get("vla", 0),
        }

    def sample(self, device: torch.device, batch_size: int | None = None) -> RLTBatch:
        batch_size = batch_size or self.config.batch_size
        if len(self) < batch_size:
            raise ValueError(f"Not enough replay samples: have {len(self)}, need {batch_size}")
        samples = self._sample_mixed(batch_size)

        def stack(name: str, dtype=np.float32):
            return torch.as_tensor(np.stack([getattr(sample, name) for sample in samples]), dtype=torch.float32, device=device)

        reward = torch.as_tensor([sample.reward for sample in samples], dtype=torch.float32, device=device)
        discount = torch.as_tensor([sample.discount for sample in samples], dtype=torch.float32, device=device)
        return RLTBatch(
            rl_token=stack("rl_token"),
            state=stack("state"),
            action=stack("action"),
            reference_action=stack("reference_action"),
            reward=reward,
            discount=discount,
            next_rl_token=stack("next_rl_token"),
            next_state=stack("next_state"),
            next_reference_action=stack("next_reference_action"),
        )

    def _sample_mixed(self, batch_size: int) -> list[RLTTransition]:
        demo_fraction = _clamp_fraction(getattr(self.config, "demo_sample_fraction", 0.0))
        online_positive_fraction = _clamp_fraction(getattr(self.config, "online_positive_sample_fraction", 0.0))
        legacy_positive_fraction = _clamp_fraction(getattr(self.config, "positive_sample_fraction", 0.0))
        if online_positive_fraction == 0.0:
            online_positive_fraction = legacy_positive_fraction

        demo_count = int(round(batch_size * demo_fraction)) if self._demo_storage else 0
        positive_storage = self._positive_storage()
        positive_count = int(round(batch_size * online_positive_fraction)) if positive_storage else 0
        if demo_count + positive_count > batch_size:
            overflow = demo_count + positive_count - batch_size
            positive_count = max(0, positive_count - overflow)

        samples: list[RLTTransition] = []
        if demo_count:
            samples.extend(random.choices(self._demo_storage, k=demo_count))
        if positive_count:
            samples.extend(random.choices(positive_storage, k=positive_count))

        remaining_count = batch_size - len(samples)
        all_storage = [*self._storage, *self._online_positive_storage]
        if not all_storage:
            all_storage = [*self._demo_storage]
        if remaining_count:
            if len(all_storage) >= remaining_count:
                samples.extend(random.sample(all_storage, remaining_count))
            else:
                samples.extend(random.choices(all_storage, k=remaining_count))
        random.shuffle(samples)
        return samples

    def _positive_storage(self) -> list[RLTTransition]:
        return [
            *[transition for transition in self._demo_storage if transition.reward > 0.0],
            *self._online_positive_storage,
        ]


def _clamp_fraction(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


@dataclasses.dataclass
class StepRecord:
    rl_token: np.ndarray
    state: np.ndarray
    executed_action: np.ndarray
    reference_action: np.ndarray
    reward: float
    done: bool
    source: str = "rl"
    reference_chunk: np.ndarray | None = None


def _stack_or_pad(values: list[np.ndarray], length: int) -> np.ndarray:
    if not values:
        raise ValueError("Cannot build a fixed-length chunk from an empty list.")
    arrays = [np.asarray(value, dtype=np.float32) for value in values]
    while len(arrays) < length:
        arrays.append(arrays[-1].copy())
    return np.stack(arrays[:length]).astype(np.float32)


def _reference_chunk(records: list[StepRecord], start: int, chunk_length: int) -> np.ndarray:
    reference_chunk = records[start].reference_chunk
    if reference_chunk is not None:
        reference_chunk = np.asarray(reference_chunk, dtype=np.float32)
        if reference_chunk.shape[0] >= chunk_length:
            return reference_chunk[:chunk_length].astype(np.float32)
        return _stack_or_pad([*reference_chunk], chunk_length)
    return _stack_or_pad([record.reference_action for record in records[start : start + chunk_length]], chunk_length)


def chunk_episode_records(
    records: list[StepRecord],
    *,
    chunk_length: int,
    stride: int,
    gamma: float,
) -> list[RLTTransition]:
    """Subsample an episode into overlapping chunk transitions.

    This implements the RLT paper's stride-2 chunk replay idea. Rewards are
    accumulated across the chunk, and the bootstrap state is the observation at
    the next chunk boundary.
    """

    transitions: list[RLTTransition] = []
    if not records:
        return transitions

    for start in range(0, len(records), stride):
        end = start + chunk_length
        chunk = records[start : min(end, len(records))]
        if not chunk:
            continue

        terminal_offset = next((offset for offset, record in enumerate(chunk) if record.done), None)
        if len(chunk) < chunk_length and terminal_offset is None:
            continue
        if end >= len(records) and terminal_offset is None:
            # Non-terminal chunks need the observation at t + C for bootstrapping.
            continue

        next_record = records[end] if end < len(records) else records[-1]
        reward = 0.0
        terminal = False
        for offset, record in enumerate(chunk):
            reward += (gamma**offset) * float(record.reward)
            if record.done:
                terminal = True
                break
        discount = 0.0 if terminal else gamma**chunk_length
        transitions.append(
            RLTTransition(
                rl_token=records[start].rl_token,
                state=records[start].state,
                action=_stack_or_pad([record.executed_action for record in chunk], chunk_length),
                reference_action=_reference_chunk(records, start, chunk_length),
                reward=float(reward),
                discount=float(discount),
                next_rl_token=next_record.rl_token,
                next_state=next_record.state,
                next_reference_action=_reference_chunk(records, min(end, len(records) - 1), chunk_length),
                done=terminal,
                source=records[start].source,
            )
        )
    return transitions
