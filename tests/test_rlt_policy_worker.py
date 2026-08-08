from __future__ import annotations

import threading
import time

from piper_runtime.rlt_policy_worker import RLTPolicyWorker


def test_async_worker_waits_for_the_requested_result() -> None:
    worker = RLTPolicyWorker(policy_fn=lambda observation: {"value": observation})
    worker.start()
    try:
        timestamp = time.monotonic()
        worker.submit("observation", timestamp_s=timestamp)
        result = worker.wait_for_result(observation_timestamp_s=timestamp, timeout_s=1.0)
        assert result is not None
        assert result.value == {"value": "observation"}
        assert result.observation_timestamp_s == timestamp
    finally:
        worker.stop(timeout_s=1.0)


def test_async_worker_wait_timeout_does_not_publish_a_partial_result() -> None:
    release = threading.Event()

    def blocked_policy(observation):
        release.wait(timeout=1.0)
        return observation

    worker = RLTPolicyWorker(policy_fn=blocked_policy)
    worker.start()
    try:
        timestamp = time.monotonic()
        worker.submit("observation", timestamp_s=timestamp)
        assert worker.wait_for_result(observation_timestamp_s=timestamp, timeout_s=0.01) is None
        assert worker.latest() is None
    finally:
        release.set()
        worker.stop(timeout_s=1.0)
