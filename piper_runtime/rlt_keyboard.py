from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class RLTKeyboardSnapshot:
    human_takeover: bool = False
    waiting_for_reward: bool = False
    terminal_reward: float | None = None
    stop_requested: bool = False
    episode_done: bool = False
    last_key: str | None = None
    mode: str = "MODEL"
    takeover_started_s: float | None = None
    takeover_ended_s: float | None = None


@dataclasses.dataclass
class RLTKeyboardStateMachine:
    """Pure keyboard semantics for RLT rollout/warmup.

    The `s` key is an edge-triggered arm signal for this integration phase:

    * first accepted `s`: arm Pika human takeover and hold the robot;
    * a new physical Pika off->on activation edge after arming, followed by a
      fresh command: start the recorded human motion;
    * second accepted `s` or `e`: end the motion phase and wait for a final reward;
    * `e`: end the motion phase and wait for a final reward;
    * `1`/`0`: finalize success/failure only after the motion phase has ended;
    * `q`: immediately stop the episode and mark it as failure.

    Repeated terminal key events within `toggle_debounce_s` are ignored so a
    single physical press cannot both enter and leave takeover.
    """

    toggle_debounce_s: float = 0.25

    def __post_init__(self) -> None:
        self._mode = "MODEL"
        self._last_s_toggle_s: float | None = None
        self._takeover_started_s: float | None = None
        self._takeover_ended_s: float | None = None
        self._terminal_reward: float | None = None
        self._stop_requested = False
        self._last_key: str | None = None

    def press(self, key: str, *, now_s: float) -> RLTKeyboardSnapshot:
        normalized = key.lower()
        self._last_key = normalized
        if normalized == "s":
            self._handle_s(now_s=now_s)
        elif normalized == "1":
            self._finish_with_reward(1.0, now_s=now_s)
        elif normalized == "0":
            self._finish_with_reward(0.0, now_s=now_s)
        elif normalized == "e":
            self.end_motion_phase(now_s=now_s)
        elif normalized == "q":
            self._stop_requested = True
            self._terminal_reward = 0.0
            if self._mode == "HUMAN" and self._takeover_ended_s is None:
                self._takeover_ended_s = now_s
            self._mode = "STOPPED"
        return self.snapshot(now_s=now_s)

    def start_takeover(self, *, now_s: float) -> RLTKeyboardSnapshot:
        if self._mode == "TAKEOVER_ARMED":
            self._mode = "HUMAN"
            if self._takeover_started_s is None:
                self._takeover_started_s = now_s
            self._takeover_ended_s = None
        return self.snapshot(now_s=now_s)

    def end_motion_phase(self, *, now_s: float) -> RLTKeyboardSnapshot:
        if self._mode in {"MODEL", "TAKEOVER_ARMED", "HUMAN"}:
            if self._mode == "HUMAN" and self._takeover_ended_s is None:
                self._takeover_ended_s = now_s
            self._mode = "WAIT_FOR_REWARD"
        return self.snapshot(now_s=now_s)

    def _finish_with_reward(self, reward: float, *, now_s: float) -> None:
        if self._mode == "STOPPED" or self._stop_requested:
            return
        self._terminal_reward = float(reward)
        if self._mode == "HUMAN" and self._takeover_ended_s is None:
            self._takeover_ended_s = now_s
        self._mode = "STOPPED"

    def snapshot(self, *, now_s: float) -> RLTKeyboardSnapshot:
        del now_s
        human_takeover = self._mode in {"TAKEOVER_ARMED", "HUMAN"}
        waiting_for_reward = self._mode == "WAIT_FOR_REWARD"
        episode_done = self._mode == "STOPPED"
        return RLTKeyboardSnapshot(
            human_takeover=human_takeover,
            waiting_for_reward=waiting_for_reward,
            terminal_reward=self._terminal_reward,
            stop_requested=self._stop_requested,
            episode_done=episode_done,
            last_key=self._last_key,
            mode=self._mode,
            takeover_started_s=self._takeover_started_s,
            takeover_ended_s=self._takeover_ended_s,
        )

    def reset_episode(self) -> None:
        self._mode = "MODEL"
        self._last_s_toggle_s = None
        self._takeover_started_s = None
        self._takeover_ended_s = None
        self._terminal_reward = None
        self._stop_requested = False
        self._last_key = None

    def _handle_s(self, *, now_s: float) -> None:
        if self._mode in {"WAIT_FOR_REWARD", "STOPPED"}:
            return
        if self._last_s_toggle_s is not None and now_s - self._last_s_toggle_s < self.toggle_debounce_s:
            return
        self._last_s_toggle_s = now_s
        if self._mode == "MODEL":
            self._mode = "TAKEOVER_ARMED"
            self._takeover_ended_s = None
        elif self._mode in {"TAKEOVER_ARMED", "HUMAN"}:
            self.end_motion_phase(now_s=now_s)
