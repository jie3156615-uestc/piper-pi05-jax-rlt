from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Callable
from typing import Any


@dataclasses.dataclass(frozen=True)
class PolicyWorkerOutput:
    value: Any
    observation_timestamp_s: float
    completed_timestamp_s: float
    error: str | None = None
    policy_plan_id: str | None = None
    policy_observation_t: int | None = None
    worker_lane: str = "shared"


class RLTPolicyWorker:
    """Latest-value background worker for policy inference.

    The control loop calls `submit()` with the newest observation and reads
    `latest()` when it needs a completed policy result. If inference is still
    running, newer submissions replace older pending observations instead of
    forming a backlog.
    """

    planning_mode = "latest_async"

    def __init__(
        self,
        *,
        policy_fn: Callable[[Any], Any],
        worker_lane: str = "shared",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._policy_fn = policy_fn
        self.worker_lane = str(worker_lane).strip() or "shared"
        self._clock = clock
        self._condition = threading.Condition()
        self._running = False
        self._thread: threading.Thread | None = None
        self._pending_observation: Any | None = None
        self._pending_timestamp_s: float | None = None
        self._pending_observation_t: int | None = None
        self._pending_plan_id: str | None = None
        self._next_plan_sequence = 0
        self._latest: PolicyWorkerOutput | None = None

    def start(self) -> None:
        with self._condition:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._run, name="rlt-policy-worker", daemon=True)
            self._thread.start()

    def submit(self, observation: Any, *, timestamp_s: float, observation_t: int | None = None) -> None:
        with self._condition:
            if not self._running:
                raise RuntimeError("policy worker is not running")
            self._next_plan_sequence += 1
            self._pending_observation = observation
            self._pending_timestamp_s = float(timestamp_s)
            self._pending_observation_t = None if observation_t is None else int(observation_t)
            self._pending_plan_id = f"plan_{self._next_plan_sequence:08d}"
            self._condition.notify_all()

    def latest(self) -> PolicyWorkerOutput | None:
        with self._condition:
            return self._latest

    def wait_for_result(
        self,
        *,
        observation_timestamp_s: float,
        timeout_s: float,
    ) -> PolicyWorkerOutput | None:
        """Waits only for the requested (or a newer coalesced) observation."""

        deadline = self._clock() + float(timeout_s)
        with self._condition:
            while self._running:
                if (
                    self._latest is not None
                    and self._latest.observation_timestamp_s >= float(observation_timestamp_s)
                ):
                    return self._latest
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)
            return None

    def stop(self, *, timeout_s: float | None = None) -> None:
        thread: threading.Thread | None
        with self._condition:
            self._running = False
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout_s)

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._running and self._pending_observation is None:
                    self._condition.wait()
                if not self._running:
                    return
                observation = self._pending_observation
                observation_timestamp_s = self._pending_timestamp_s
                policy_observation_t = self._pending_observation_t
                policy_plan_id = self._pending_plan_id
                self._pending_observation = None
                self._pending_timestamp_s = None
                self._pending_observation_t = None
                self._pending_plan_id = None

            assert observation_timestamp_s is not None
            try:
                value = self._policy_fn(observation)
                output = PolicyWorkerOutput(
                    value=value,
                    observation_timestamp_s=observation_timestamp_s,
                    completed_timestamp_s=self._clock(),
                    error=None,
                    policy_plan_id=policy_plan_id,
                    policy_observation_t=policy_observation_t,
                    worker_lane=self.worker_lane,
                )
            except Exception as exc:  # pragma: no cover - defensive path; surfaced through latest()
                output = PolicyWorkerOutput(
                    value=None,
                    observation_timestamp_s=observation_timestamp_s,
                    completed_timestamp_s=self._clock(),
                    error=f"{type(exc).__name__}: {exc}",
                    policy_plan_id=policy_plan_id,
                    policy_observation_t=policy_observation_t,
                    worker_lane=self.worker_lane,
                )

            with self._condition:
                self._latest = output
                self._condition.notify_all()


class SynchronousPolicyWorker:
    """Native rollout-compatible policy adapter.

    ``submit`` blocks until inference completes.  The RLT control loop calls it
    only when the current SFT/Actor chunk is exhausted, matching
    ``policy_hardware_rollout``: observe current state, infer once, then execute
    the selected horizon without accepting a plan generated mid-chunk.
    """

    planning_mode = "native_synchronous"

    def __init__(
        self,
        *,
        policy_fn: Callable[[Any], Any],
        worker_lane: str = "shared",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._policy_fn = policy_fn
        self.worker_lane = str(worker_lane).strip() or "shared"
        self._clock = clock
        self._running = False
        self._latest: PolicyWorkerOutput | None = None
        self._next_plan_sequence = 0

    def start(self) -> None:
        self._running = True

    def submit(self, observation: Any, *, timestamp_s: float, observation_t: int | None = None) -> None:
        if not self._running:
            raise RuntimeError("policy worker is not running")
        self._next_plan_sequence += 1
        policy_plan_id = f"plan_{self._next_plan_sequence:08d}"
        try:
            value = self._policy_fn(observation)
            self._latest = PolicyWorkerOutput(
                value=value,
                observation_timestamp_s=float(timestamp_s),
                completed_timestamp_s=self._clock(),
                error=None,
                policy_plan_id=policy_plan_id,
                policy_observation_t=None if observation_t is None else int(observation_t),
                worker_lane=self.worker_lane,
            )
        except Exception as exc:  # pragma: no cover - defensive path; surfaced through latest()
            self._latest = PolicyWorkerOutput(
                value=None,
                observation_timestamp_s=float(timestamp_s),
                completed_timestamp_s=self._clock(),
                error=f"{type(exc).__name__}: {exc}",
                policy_plan_id=policy_plan_id,
                policy_observation_t=None if observation_t is None else int(observation_t),
                worker_lane=self.worker_lane,
            )

    def latest(self) -> PolicyWorkerOutput | None:
        return self._latest

    def stop(self, *, timeout_s: float | None = None) -> None:
        del timeout_s
        self._running = False
