from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = (
    ROOT / "remote_piper_runtime"
    if (ROOT / "remote_piper_runtime").is_dir()
    else ROOT
)
sys.path.insert(0, str(RUNTIME_ROOT))
SCRIPT = RUNTIME_ROOT / "scripts" / "benchmark_rlt_service_protocol.py"
SPEC = importlib.util.spec_from_file_location(
    "benchmark_rlt_service_protocol",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
run_protocol_benchmark = MODULE.run_protocol_benchmark


class FakeBaseClient:
    def infer(self, observation):
        assert observation["rlt/base_only_mode"] == "base_only_v1"
        return {
            "actions": np.zeros((50, 7), dtype=np.float32),
            "rlt_shadow": {
                "mode": "base_only",
                "base_policy_called": True,
                "base_rng_advanced": True,
                "token_encoder_called": False,
                "actor_called": False,
                "base_policy_latency_s": 0.1,
            },
        }


class FakeEnrichmentClient:
    def __init__(self, *, violate_rng: bool = False):
        self.violate_rng = bool(violate_rng)

    def infer(self, observation):
        assert (
            observation["rlt/actor_only_mode"]
            == "actor_enrichment_only_v1"
        )
        reference = np.asarray(
            observation["rlt/behavior_ref"],
            dtype=np.float32,
        )
        return {
            "actions": reference.copy(),
            "z_rl": np.ones(2048, dtype=np.float32),
            "a_actor": reference.copy(),
            "rlt_shadow": {
                "mode": "actor_only",
                "actor_only_protocol": "actor_enrichment_only_v1",
                "base_policy_called": False,
                "base_rng_advanced": self.violate_rng,
                "token_encoder_called": True,
                "actor_called": True,
                "shadow_latency_s": 0.02,
                "token_latency_s": 0.01,
            },
        }


def _observation() -> dict:
    return {
        "observation/image": np.zeros((16, 16, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((16, 16, 3), dtype=np.uint8),
        "observation/state": np.zeros(7, dtype=np.float32),
        "prompt": "test",
    }


def test_split_service_protocol_benchmark_passes_isolated_lanes() -> None:
    report = run_protocol_benchmark(
        base_client=FakeBaseClient(),
        enrichment_client=FakeEnrichmentClient(),
        observation=_observation(),
        requests=3,
    )

    assert report["passed"] is True
    assert report["hardware_commands_published"] == 0
    assert report["base_only"]["latency"]["count"] == 3
    assert report["base_only"]["lead_frame_estimates"]["from_max"] >= 2
    assert report["enrichment_only"]["latency"]["count"] == 3
    assert all(
        sample["checks"]["token_skipped"]
        and sample["checks"]["actor_skipped"]
        for sample in report["base_only"]["samples"]
    )
    assert all(
        sample["checks"]["base_skipped"]
        and sample["checks"]["base_rng_not_advanced"]
        for sample in report["enrichment_only"]["samples"]
    )


def test_split_service_protocol_benchmark_rejects_rng_advance() -> None:
    report = run_protocol_benchmark(
        base_client=FakeBaseClient(),
        enrichment_client=FakeEnrichmentClient(violate_rng=True),
        observation=_observation(),
        requests=2,
    )

    assert report["passed"] is False
    assert any("base_rng_not_advanced" in item for item in report["violations"])
