from __future__ import annotations

"""Read-only latency and isolation benchmark for the split RLT policy service.

The script sends inference requests only.  It does not import ROS, open CAN,
start cameras, or publish a robot command.  Two independent websocket clients
exercise the same split-lane contract used by the rollout:

* ``base_only_v1`` runs Pi0.5 but skips the token encoder and Actor;
* ``actor_enrichment_only_v1`` runs Token+Actor, echoes the supplied behavior
  reference, skips Pi0.5, and must not advance Pi0.5's RNG.
"""

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
from PIL import Image


def _recorded_observation(
    path: Path,
    *,
    timestep: int,
    prompt: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [row for row in rows if int(row["t"]) == int(timestep)]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one t={timestep} row in {path}, found {len(matches)}"
        )
    row = matches[0]
    episode_root = path.parent

    def read_image(key: str) -> np.ndarray:
        relative = Path(str(row[key]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe {key} path: {relative}")
        image_path = episode_root / relative
        with Image.open(image_path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

    observation = {
        "observation/image": read_image("global_image"),
        "observation/wrist_image": read_image("wrist_image"),
        "observation/state": np.asarray(row["state"], dtype=np.float32),
        "prompt": prompt,
    }
    return observation, {
        "episode_jsonl": str(path.resolve()),
        "t": int(timestep),
        "logged_source": row.get("source"),
    }


def _synthetic_observation(
    *,
    seed: int,
    prompt: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    observation = {
        "observation/image": rng.integers(
            0, 256, size=(480, 640, 3), dtype=np.uint8
        ),
        "observation/wrist_image": rng.integers(
            0, 256, size=(480, 640, 3), dtype=np.uint8
        ),
        "observation/state": np.zeros(7, dtype=np.float32),
        "prompt": prompt,
    }
    return observation, {"synthetic_seed": int(seed)}


def _behavior_reference(state: np.ndarray, request_index: int) -> np.ndarray:
    """Create a finite, low-amplitude absolute Piper C10 reference."""

    state = np.asarray(state, dtype=np.float32)
    if state.shape != (7,) or not np.all(np.isfinite(state)):
        raise ValueError("observation/state must be finite with shape (7,)")
    reference = np.repeat(state[None, :], 10, axis=0)
    # Vary the test target without leaving the local neighborhood.  This is
    # protocol input only and is never sent to the robot.
    phase = (int(request_index) % 5) + 1
    direction = np.asarray(
        [1.0, -0.5, 0.25, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    ramp = np.linspace(0.0, phase * 2e-4, 10, dtype=np.float32)
    reference[:, :6] += ramp[:, None] * direction[None, :]
    reference[:, 6] = state[6]
    return reference


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean_s": None if not values else float(np.mean(values)),
        "p50_s": _percentile(values, 50),
        "p95_s": _percentile(values, 95),
        "max_s": None if not values else float(max(values)),
    }


def _finite_actions(
    value: Any,
    *,
    min_rows: int,
    label: str,
) -> np.ndarray:
    try:
        actions = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not an array: {exc}") from exc
    if (
        actions.ndim != 2
        or actions.shape[0] < min_rows
        or actions.shape[1] < 7
        or not np.all(np.isfinite(actions[:min_rows, :7]))
    ):
        raise ValueError(
            f"{label} must contain finite ({min_rows}, 7) actions, got "
            f"{actions.shape}"
        )
    return actions


def _shadow(response: dict[str, Any], *, label: str) -> dict[str, Any]:
    value = response.get("rlt_shadow")
    if not isinstance(value, dict):
        raise ValueError(f"{label} response has no rlt_shadow metadata")
    return value


def run_protocol_benchmark(
    *,
    base_client: Any,
    enrichment_client: Any,
    observation: dict[str, Any],
    requests: int = 20,
    control_hz: float = 30.0,
    lead_safety_frames: int = 2,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run interleaved split-lane requests and return a strict audit report."""

    if requests < 1:
        raise ValueError("requests must be positive")
    if control_hz <= 0.0 or lead_safety_frames < 0:
        raise ValueError("control_hz must be positive and safety frames non-negative")
    state = np.asarray(observation.get("observation/state"), dtype=np.float32)
    if state.shape != (7,) or not np.all(np.isfinite(state)):
        raise ValueError("observation/state must be finite with shape (7,)")

    violations: list[str] = []
    base_samples: list[dict[str, Any]] = []
    enrichment_samples: list[dict[str, Any]] = []

    for index in range(int(requests)):
        base_request = dict(observation)
        base_request["rlt/base_only_mode"] = "base_only_v1"
        started = float(clock())
        base_response = dict(base_client.infer(base_request))
        base_roundtrip_s = max(0.0, float(clock()) - started)
        try:
            base_actions = _finite_actions(
                base_response.get("actions"),
                min_rows=50,
                label=f"base[{index}] actions",
            )
            base_shadow = _shadow(base_response, label=f"base[{index}]")
            checks = {
                "mode_base_only": base_shadow.get("mode") == "base_only",
                "base_called": base_shadow.get("base_policy_called") is True,
                "base_rng_advanced_once": (
                    base_shadow.get("base_rng_advanced") is True
                ),
                "token_skipped": (
                    base_shadow.get("token_encoder_called") is False
                ),
                "actor_skipped": base_shadow.get("actor_called") is False,
                "no_token_payload": base_response.get("z_rl") is None,
                "no_actor_payload": base_response.get("a_actor") is None,
            }
            if not all(checks.values()):
                violations.append(
                    f"base[{index}] split-lane contract failed: "
                    f"{[key for key, value in checks.items() if not value]}"
                )
            base_samples.append(
                {
                    "index": index,
                    "client_roundtrip_s": base_roundtrip_s,
                    "server_latency_s": base_shadow.get(
                        "base_policy_latency_s"
                    ),
                    "actions_shape": list(base_actions.shape),
                    "checks": checks,
                }
            )
        except ValueError as exc:
            violations.append(str(exc))
            base_samples.append(
                {
                    "index": index,
                    "client_roundtrip_s": base_roundtrip_s,
                    "error": str(exc),
                }
            )

        reference = _behavior_reference(state, index)
        # Construct the wire payload explicitly so this benchmark remains an
        # independent protocol check rather than sharing the rollout's request
        # builder implementation.
        enrichment_request = dict(observation)
        enrichment_request.update(
            {
                "rlt/behavior_ref": reference.copy(),
                "rlt/behavior_ref_plan_id": (
                    f"protocol-benchmark-{index:04d}"
                ),
                "rlt/behavior_ref_start_offset": (index % 5) * 10,
                "rlt/behavior_ref_contract": "rank1_bump_v1",
                "rlt/actor_conditioning_state": state.copy(),
                "rlt/actor_only_mode": "actor_enrichment_only_v1",
            }
        )
        started = float(clock())
        enrichment_response = dict(enrichment_client.infer(enrichment_request))
        enrichment_roundtrip_s = max(0.0, float(clock()) - started)
        try:
            echoed = _finite_actions(
                enrichment_response.get("actions"),
                min_rows=10,
                label=f"enrichment[{index}] actions",
            )[:10, :7]
            actor = _finite_actions(
                enrichment_response.get("a_actor"),
                min_rows=10,
                label=f"enrichment[{index}] Actor",
            )[:10, :7]
            z_rl = np.asarray(
                enrichment_response.get("z_rl"),
                dtype=np.float32,
            ).reshape(-1)
            enrichment_shadow = _shadow(
                enrichment_response,
                label=f"enrichment[{index}]",
            )
            checks = {
                "mode_actor_only": (
                    enrichment_shadow.get("mode") == "actor_only"
                ),
                "protocol_enrichment_only": (
                    enrichment_shadow.get("actor_only_protocol")
                    == "actor_enrichment_only_v1"
                ),
                "base_skipped": (
                    enrichment_shadow.get("base_policy_called") is False
                ),
                "base_rng_not_advanced": (
                    enrichment_shadow.get("base_rng_advanced") is False
                ),
                "token_called": (
                    enrichment_shadow.get("token_encoder_called") is True
                ),
                "actor_called": enrichment_shadow.get("actor_called") is True,
                "reference_echo_exact": bool(
                    np.array_equal(echoed, reference)
                ),
                "token_shape_finite": bool(
                    z_rl.shape == (2048,) and np.all(np.isfinite(z_rl))
                ),
                "actor_shape_finite": bool(
                    actor.shape == (10, 7) and np.all(np.isfinite(actor))
                ),
            }
            if not all(checks.values()):
                violations.append(
                    f"enrichment[{index}] split-lane contract failed: "
                    f"{[key for key, value in checks.items() if not value]}"
                )
            enrichment_samples.append(
                {
                    "index": index,
                    "client_roundtrip_s": enrichment_roundtrip_s,
                    "server_latency_s": enrichment_shadow.get(
                        "shadow_latency_s"
                    ),
                    "token_latency_s": enrichment_shadow.get(
                        "token_latency_s"
                    ),
                    "actions_shape": list(echoed.shape),
                    "actor_shape": list(actor.shape),
                    "checks": checks,
                }
            )
        except (TypeError, ValueError) as exc:
            violations.append(str(exc))
            enrichment_samples.append(
                {
                    "index": index,
                    "client_roundtrip_s": enrichment_roundtrip_s,
                    "error": str(exc),
                }
            )

    base_latencies = [
        float(sample["client_roundtrip_s"]) for sample in base_samples
    ]
    enrichment_latencies = [
        float(sample["client_roundtrip_s"])
        for sample in enrichment_samples
    ]
    base_latency_summary = _latency_summary(base_latencies)
    base_steady_summary = _latency_summary(base_latencies[1:])

    def lead_frames(latency_s: float | None) -> int | None:
        if latency_s is None:
            return None
        return int(math.ceil(float(latency_s) * float(control_hz))) + int(
            lead_safety_frames
        )

    report = {
        "format": "rlt_split_service_protocol_benchmark_v1",
        "passed": not violations,
        "violations": violations,
        "requests_per_lane": int(requests),
        "request_order": "interleaved_base_then_enrichment",
        "hardware_commands_published": 0,
        "base_only": {
            "latency": base_latency_summary,
            "steady_state_latency_excluding_first": base_steady_summary,
            "lead_frame_estimates": {
                "control_hz": float(control_hz),
                "safety_frames": int(lead_safety_frames),
                "from_p95": lead_frames(base_latency_summary["p95_s"]),
                "from_max": lead_frames(base_latency_summary["max_s"]),
                "from_steady_state_max": lead_frames(
                    base_steady_summary["max_s"]
                ),
                "note": (
                    "Diagnostic only; keep code and shell defaults identical "
                    "and rerun rollout regressions before changing lead."
                ),
            },
            "samples": base_samples,
        },
        "enrichment_only": {
            "latency": _latency_summary(enrichment_latencies),
            "samples": enrichment_samples,
        },
    }
    # Reject accidental NaN/Infinity before JSON rendering and CI use.
    for latency in base_latencies + enrichment_latencies:
        if not math.isfinite(latency):
            report["violations"].append("non-finite client latency")
            report["passed"] = False
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only 20+20 base/enrichment service isolation and latency benchmark"
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--lead-safety-frames", type=int, default=2)
    parser.add_argument("--episode-jsonl", type=Path)
    parser.add_argument("--t", type=int, default=0)
    parser.add_argument("--prompt", default="Put the green block into the box.")
    parser.add_argument("--synthetic-seed", type=int, default=20260724)
    parser.add_argument("--connect-timeout-s", type=float, default=30.0)
    parser.add_argument("--request-timeout-s", type=float, default=60.0)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if args.episode_jsonl is None:
        observation, observation_source = _synthetic_observation(
            seed=args.synthetic_seed,
            prompt=args.prompt,
        )
    else:
        observation, observation_source = _recorded_observation(
            args.episode_jsonl.expanduser(),
            timestep=args.t,
            prompt=args.prompt,
        )

    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    # Separate websocket clients intentionally mirror rollout base/enrichment
    # lanes and expose any server-side cross-request contamination.
    base_client = WebsocketClientPolicy(
        args.host,
        args.port,
        connect_timeout_s=args.connect_timeout_s,
        request_timeout_s=args.request_timeout_s,
    )
    enrichment_client = WebsocketClientPolicy(
        args.host,
        args.port,
        connect_timeout_s=args.connect_timeout_s,
        request_timeout_s=args.request_timeout_s,
    )
    report = run_protocol_benchmark(
        base_client=base_client,
        enrichment_client=enrichment_client,
        observation=observation,
        requests=args.requests,
        control_hz=args.control_hz,
        lead_safety_frames=args.lead_safety_frames,
    )
    report["observation_source"] = observation_source
    report["service"] = {"host": args.host, "port": args.port}
    rendered = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    if args.output_json is not None:
        output = args.output_json.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
