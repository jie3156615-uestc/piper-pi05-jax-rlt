from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class HysteresisGate:
    enter_threshold: float = 0.7
    exit_threshold: float = 0.4
    enter_frames: int = 3
    exit_frames: int = 3

    def __post_init__(self) -> None:
        self.active = False
        self._enter_count = 0
        self._exit_count = 0

    def update(self, probability: float) -> bool:
        if not self.active:
            if probability >= self.enter_threshold:
                self._enter_count += 1
            else:
                self._enter_count = 0
            if self._enter_count >= self.enter_frames:
                self.active = True
                self._exit_count = 0
        else:
            if probability <= self.exit_threshold:
                self._exit_count += 1
            else:
                self._exit_count = 0
            if self._exit_count >= self.exit_frames:
                self.active = False
                self._enter_count = 0
        return self.active


def event_alignment(
    *,
    predicted_enter_t: int,
    manual_enter_t: int,
    predicted_exit_t: int,
    manual_exit_t: int,
) -> dict[str, int]:
    return {
        "enter_lead_frames": manual_enter_t - predicted_enter_t,
        "exit_lag_frames": predicted_exit_t - manual_exit_t,
    }
