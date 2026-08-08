"""Build a resumable frozen-policy/Token/phase cache for real Piper replay.

This program is deliberately offline: it reads saved images and joint state,
talks only to the websocket policy service, and imports no ROS/CAN modules.
The resulting JSONL is consumed directly by ``prepare_external_rlt_replay.py``.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

# Production hosts may have an older OpenPI checkout in PYTHONPATH.  Put the
# checkout containing this script (plus the deployed Piper runtime) first so
# enrichment cannot silently import an incompatible replay contract.
_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
_IMPORT_ROOTS = (
    _WORKSPACE_ROOT,
    _WORKSPACE_ROOT / "src",
    _WORKSPACE_ROOT / "packages" / "openpi-client" / "src",
    _WORKSPACE_ROOT / "remote_piper_runtime",
    Path.home() / "piper_jax_inference_v1",
)
sys.path[:0] = [str(path) for path in _IMPORT_ROOTS if path.is_dir()]

import numpy as np
from PIL import Image

from openpi.rlt.real.external_episode import ExternalEpisodeContract
from openpi.rlt.real.external_episode import load_episode_jsonl
from openpi.rlt.real.replay import RealStepRecord
from openpi.rlt.real.replay_enrichment import ABSOLUTE_ACTION_SPACE
from openpi.rlt.real.replay_enrichment import SingleLatchGate
from openpi.rlt.real.replay_enrichment import derive_episode_uid
from openpi.rlt.real.replay_enrichment import exclusion_reason
from piper_runtime.rlt_actor_protocol import BehaviorReferenceTarget
from piper_runtime.rlt_actor_protocol import TOKEN_BATCH_PROTOCOL
from piper_runtime.rlt_actor_protocol import add_actor_enrichment_only_request
from piper_runtime.rlt_actor_protocol import add_token_batch_request


CACHE_SCHEMA = "piper_rlt_external_enrichment_v1"
DEFAULT_PROMPT = "Put the green block into the box."
DEFAULT_BASE_CONFIG = "pi05_piper_greenblock_5090_jax_delta_v1"
DEFAULT_BASE_CHECKPOINT = (
    "/home/cwzk/openpi_checkpoints/pi05_piper_greenblock_5090_jax_delta_v1/"
    "piper_greenblock_5090_delta_sft_30k_20260707/20000"
)
DEFAULT_TOKEN_CHECKPOINT = (
    "/home/cwzk/openpi_rlt/rl_tokens/"
    "pi05_piper_greenblock_5090_full20k_rlt_token_v1_20260709/step_010000_encoder_only"
)


class PhasePredictor(Protocol):
    def predict_probability(self, images: dict[str, np.ndarray]) -> float: ...


class PolicyClient(Protocol):
    def infer(self, observation: dict[str, Any]) -> Mapping[str, Any]: ...

    def get_server_metadata(self) -> Mapping[str, Any]: ...


@dataclasses.dataclass(frozen=True)
class CacheGenerationConfig:
    output: Path
    dataset_root: Path | None
    prompt: str = DEFAULT_PROMPT
    phase_enter_threshold: float = 0.5
    phase_enter_frames: int = 3
    expected_z_dim: int = 2048
    base_fingerprint: str = ""
    token_fingerprint: str = ""
    phase_fingerprint: str = ""
    expected_base_config: str | None = DEFAULT_BASE_CONFIG
    expected_base_checkpoint: str | None = DEFAULT_BASE_CHECKPOINT
    expected_token_checkpoint: str | None = DEFAULT_TOKEN_CHECKPOINT
    skip_invalid_episodes: bool = False
    require_terminal_report: bool = True
    incremental: bool = False
    token_only: bool = False
    token_batch_size: int = 1

    def validate(self) -> "CacheGenerationConfig":
        if not self.base_fingerprint:
            raise ValueError("base_fingerprint is required")
        if not self.token_fingerprint:
            raise ValueError("token_fingerprint is required")
        if not self.phase_fingerprint:
            raise ValueError("phase_fingerprint is required")
        if not 0.0 <= self.phase_enter_threshold <= 1.0:
            raise ValueError("phase_enter_threshold must be within [0, 1]")
        if self.phase_enter_frames < 1:
            raise ValueError("phase_enter_frames must be positive")
        if self.expected_z_dim < 1:
            raise ValueError("expected_z_dim must be positive")
        if not 1 <= int(self.token_batch_size) <= 16:
            raise ValueError("token_batch_size must be within [1, 16]")
        return dataclasses.replace(
            self,
            output=self.output.expanduser().resolve(),
            dataset_root=None if self.dataset_root is None else self.dataset_root.expanduser().resolve(),
            token_batch_size=int(self.token_batch_size),
        )


class _LazyValidatedPolicy:
    def __init__(self, factory: Callable[[], PolicyClient], config: CacheGenerationConfig):
        self._factory = factory
        self._config = config
        self._client: PolicyClient | None = None
        self.metadata: dict[str, Any] | None = None
        self.requests = 0
        self.rows = 0

    def _ensure_client(self) -> PolicyClient:
        if self._client is None:
            self._client = self._factory()
            self.metadata = dict(self._client.get_server_metadata())
            _validate_server_metadata(self.metadata, self._config)
        return self._client

    def infer(self, observation: dict[str, Any]) -> Mapping[str, Any]:
        client = self._ensure_client()
        self.requests += 1
        self.rows += 1
        return client.infer(observation)

    def infer_fresh_tokens(
        self,
        pending: Sequence[
            tuple[dict[str, Any], RealStepRecord, tuple[str, int]]
        ],
    ) -> list[Mapping[str, Any]]:
        """Return one Token response per row without ever invoking Pi0.5."""

        if not pending:
            return []
        client = self._ensure_client()
        batch_supported = (
            self.metadata is not None
            and self.metadata.get("rlt_token_batch_protocol")
            == TOKEN_BATCH_PROTOCOL
        )
        responses: list[Mapping[str, Any]] = []
        batch_size = self._config.token_batch_size if batch_supported else 1
        for start in range(0, len(pending), batch_size):
            batch = list(pending[start : start + batch_size])
            if batch_supported:
                observations = [item[0] for item in batch]
                response = client.infer(add_token_batch_request(observations))
                self.requests += 1
                self.rows += len(batch)
                shadow = response.get("rlt_shadow", {})
                tokens = np.asarray(response.get("z_rl_batch"), dtype=np.float32)
                expected_shape = (len(batch), self._config.expected_z_dim)
                if (
                    not isinstance(shadow, Mapping)
                    or shadow.get("base_policy_called") is not False
                    or shadow.get("base_rng_advanced") is not False
                    or shadow.get("actor_called") is not False
                    or shadow.get("token_status") != "ok"
                    or tokens.shape != expected_shape
                    or not np.all(np.isfinite(tokens))
                ):
                    raise ValueError(
                        f"invalid Token-only batch response: "
                        f"shape={tokens.shape}, shadow={dict(shadow) if isinstance(shadow, Mapping) else shadow}"
                    )
                for token in tokens:
                    responses.append(
                        {
                            "z_rl": token,
                            "rlt_shadow": shadow,
                            "token_batch": True,
                        }
                    )
                continue

            observation, record, key = batch[0]
            reference = np.asarray(record.a_ref, dtype=np.float32)
            if reference.ndim != 2 or reference.shape[0] < 10 or reference.shape[1] < 7:
                raise ValueError(
                    f"logged a_ref for {key} must have at least shape (10, 7), "
                    f"got {reference.shape}"
                )
            target = BehaviorReferenceTarget(
                actions=reference[:10, :7],
                plan_id=f"offline-enrichment/{key[0]}/{key[1]}",
                start_offset=0,
                conditioning_state=np.asarray(record.state, dtype=np.float32),
            )
            response = client.infer(
                add_actor_enrichment_only_request(observation, target)
            )
            self.requests += 1
            self.rows += 1
            shadow = response.get("rlt_shadow", {})
            if (
                not isinstance(shadow, Mapping)
                or shadow.get("base_policy_called") is not False
                or shadow.get("base_rng_advanced") is not False
                or shadow.get("token_status") != "ok"
            ):
                raise ValueError(
                    f"invalid Token-only response for {key}: "
                    f"{dict(shadow) if isinstance(shadow, Mapping) else shadow}"
                )
            responses.append(response)
        return responses


def generate_enrichment_cache(
    episode_paths: Sequence[str | Path],
    *,
    config: CacheGenerationConfig,
    phase_predictor: PhasePredictor,
    policy_client_factory: Callable[[], PolicyClient],
    policy_candidate_indices_by_episode: (
        Mapping[str, set[int]] | None
    ) = None,
) -> dict[str, Any]:
    """Generate one cache row per raw row, resuming safely after interruption.

    Incremental mode treats rewarded/admitted episode directories as immutable.
    Existing episodes are verified by a cheap JSONL/report version fingerprint
    and skipped without re-reading images or parsing their large Token rows.
    """

    config = config.validate()
    paths = sorted({Path(path).expanduser().resolve() for path in episode_paths})
    if not paths:
        raise ValueError("at least one episode.jsonl is required")
    run_identity = _run_identity(config)
    manifest_path = _manifest_path(config.output)
    _validate_existing_manifest(manifest_path, run_identity)
    prior_manifest = _read_json_mapping(manifest_path)
    prior_accepted = {
        str(Path(path).expanduser().resolve())
        for path in prior_manifest.get("episodes", {}).get("accepted", [])
    }
    current_path_strings = {str(path) for path in paths}
    if config.incremental:
        removed = sorted(prior_accepted.difference(current_path_strings))
        if removed:
            raise ValueError(
                "incremental cache input removed previously accepted episodes; "
                f"refusing stale replay reuse: {removed[:3]}"
            )
    prior_fingerprints = dict(prior_manifest.get("episode_fingerprints", {}))
    current_fingerprints = {
        str(path): _episode_input_fingerprint(path)
        for path in paths
        if str(path) in prior_accepted
    }
    changed = [
        path
        for path in paths
        if (
            str(path) in prior_accepted
            and str(path) in prior_fingerprints
            and prior_fingerprints[str(path)] != current_fingerprints[str(path)]
        )
    ]
    if changed:
        raise ValueError(
            "an immutable cached episode changed after enrichment: "
            f"{changed[0]}"
        )
    if config.incremental:
        process_paths = [
            path for path in paths if str(path) not in prior_accepted
        ]
    else:
        process_paths = paths

    journal_path = _journal_path(config.output)
    process_episode_ids = {
        derive_episode_uid(path, root=config.dataset_root)
        for path in process_paths
    }
    if config.incremental:
        (
            entries,
            output_keys,
            output_row_count,
        ) = _load_resume_entries_filtered(
            config.output,
            journal_path,
            run_identity,
            episode_ids=process_episode_ids,
        )
    else:
        entries = _load_resume_entries(
            config.output, journal_path, run_identity
        )
        output_keys = set(entries)
        output_row_count = len(entries)
        _initialize_journal(journal_path, entries)
    lazy_policy = _LazyValidatedPolicy(policy_client_factory, config)
    contract = ExternalEpisodeContract(require_images=True, check_image_exists=True, require_terminal=True)
    accepted: list[str] = sorted(prior_accepted) if config.incremental else []
    skipped: list[dict[str, str]] = (
        list(prior_manifest.get("episodes", {}).get("skipped", []))
        if config.incremental
        else []
    )
    phase_inferences = 0
    prior_counts = dict(prior_manifest.get("counts", {}))
    cache_hits = (
        int(prior_counts.get("rows", output_row_count))
        if config.incremental
        else 0
    )
    policy_cache_hits = (
        int(prior_counts.get("policy_rows", 0))
        if config.incremental
        else 0
    )
    journal_updates: list[dict[str, Any]] = []
    started = time.time()

    for path in process_paths:
        try:
            report_reward = _terminal_report_reward(path.parent) if config.require_terminal_report else None
            records = load_episode_jsonl(path, dataset_root=path.parent, contract=contract)
            if report_reward is not None:
                terminal = records[-1]
                if not terminal.done or float(terminal.reward) != report_reward:
                    raise ValueError(
                        f"report/JSONL terminal mismatch: report={report_reward}, "
                        f"jsonl_done={terminal.done}, jsonl_reward={terminal.reward}"
                    )
        except Exception as exc:
            if not config.skip_invalid_episodes:
                raise
            skipped.append({"file": str(path), "reason": f"{type(exc).__name__}: {exc}"})
            continue

        episode_uid = derive_episode_uid(path, root=config.dataset_root)
        records = [dataclasses.replace(record, episode_id=episode_uid) for record in records]
        current_fingerprints[str(path)] = _episode_input_fingerprint(path)
        candidate_indices = (
            None
            if policy_candidate_indices_by_episode is None
            else set(
                policy_candidate_indices_by_episode.get(episode_uid, set())
            )
        )
        if candidate_indices is not None:
            invalid_indices = sorted(
                index
                for index in candidate_indices
                if index < 0 or index >= len(records)
            )
            if invalid_indices:
                raise ValueError(
                    f"candidate indices outside {episode_uid}: "
                    f"{invalid_indices[:3]}"
                )
        gate = SingleLatchGate(
            enter_threshold=config.phase_enter_threshold,
            enter_frames=config.phase_enter_frames,
        )
        episode_rows: list[dict[str, Any]] = []
        pending: list[
            tuple[dict[str, Any], RealStepRecord, tuple[str, int], int]
        ] = []
        for record_index, record in enumerate(records):
            key = (episode_uid, int(record.t))
            global_path = _resolve_image(path.parent, record.global_image, "global_image")
            wrist_path = _resolve_image(path.parent, record.wrist_image, "wrist_image")
            observation_sha256 = _observation_sha256(
                record,
                global_path=global_path,
                wrist_path=wrist_path,
                prompt=config.prompt,
            )
            cached = entries.get(key)
            images: dict[str, np.ndarray] | None = None
            if cached is not None:
                _validate_cached_identity(cached, run_identity, observation_sha256, key=key)
                probability = _finite_probability(cached.get("phase_probability"), key=key)
                cache_hits += 1
            else:
                images = _read_images(global_path, wrist_path)
                probability = _finite_probability(phase_predictor.predict_probability(images), key=key)
                phase_inferences += 1

            gate_active = gate.update(probability, t=record.t)
            eligible = exclusion_reason(record) is None
            policy_candidate = (
                candidate_indices is None
                or record_index in candidate_indices
            )
            needs_policy = bool(
                gate_active and eligible and policy_candidate
            )
            has_policy = cached is not None and "a_ref_absolute" in cached and "z_rl" in cached
            if needs_policy and has_policy:
                _validated_policy_values(cached, config=config, key=key)
                policy_cache_hits += 1

            row = {
                "cache_schema": CACHE_SCHEMA,
                "episode_id": episode_uid,
                "t": int(record.t),
                "phase_probability": probability,
                "gate_active": bool(gate_active),
                "eligible_execution": bool(eligible),
                "policy_candidate": bool(policy_candidate),
                "policy_requested": bool(needs_policy),
                "observation_sha256": observation_sha256,
                "global_image": str(record.global_image),
                "wrist_image": str(record.wrist_image),
                "base_fingerprint": config.base_fingerprint,
                "token_fingerprint": config.token_fingerprint,
                "phase_fingerprint": config.phase_fingerprint,
            }
            if needs_policy:
                if has_policy:
                    assert cached is not None
                    row.update(
                        {
                            "a_ref_absolute": cached["a_ref_absolute"],
                            "z_rl": cached["z_rl"],
                            "action_space": ABSOLUTE_ACTION_SPACE,
                            "policy_response": cached.get("policy_response", {}),
                        }
                    )
                else:
                    if images is None:
                        images = _read_images(global_path, wrist_path)
                    pending.append(
                        (
                            _build_observation(
                                images, record.state, config.prompt
                            ),
                            record,
                            key,
                            len(episode_rows),
                        )
                    )
            episode_rows.append(row)

        if pending:
            request_items = [
                (observation, record, key)
                for observation, record, key, _ in pending
            ]
            if config.token_only:
                responses = lazy_policy.infer_fresh_tokens(request_items)
            else:
                responses = [
                    lazy_policy.infer(observation)
                    for observation, _, _ in request_items
                ]
            if len(responses) != len(pending):
                raise RuntimeError("policy response count does not match pending rows")
            for response, (_, record, key, row_index) in zip(
                responses, pending
            ):
                if config.token_only:
                    a_ref = np.asarray(record.a_ref, dtype=np.float32)
                    if (
                        a_ref.ndim != 2
                        or a_ref.shape[0] < 10
                        or a_ref.shape[1] < 7
                        or not np.all(np.isfinite(a_ref[:10, :7]))
                    ):
                        raise ValueError(
                            f"logged a_ref for {key} is invalid: {a_ref.shape}"
                        )
                    z_rl = np.asarray(
                        response.get("z_rl"), dtype=np.float32
                    )
                    if (
                        z_rl.shape != (config.expected_z_dim,)
                        or not np.all(np.isfinite(z_rl))
                    ):
                        raise ValueError(
                            f"z_rl for {key} must be finite with shape "
                            f"({config.expected_z_dim},), got {z_rl.shape}"
                        )
                    response_audit = {
                        "actions_source": "logged_execution_time_a_ref",
                        "base_policy_called": False,
                        "base_rng_advanced": False,
                        "token_batch": bool(response.get("token_batch")),
                    }
                    a_ref = a_ref[:10, :7]
                else:
                    a_ref, z_rl, response_audit = _validated_policy_values(
                        response, config=config, key=key
                    )
                episode_rows[row_index].update(
                    {
                        "a_ref_absolute": a_ref.tolist(),
                        "z_rl": z_rl.tolist(),
                        "action_space": ABSOLUTE_ACTION_SPACE,
                        "policy_response": response_audit,
                    }
                )
        for row in episode_rows:
            key = (str(row["episode_id"]), int(row["t"]))
            cached = entries.get(key)
            if cached != row:
                journal_updates.append(row)
                entries[key] = row
        accepted.append(str(path))
        if journal_updates:
            _atomic_write_jsonl(
                journal_path,
                [entries[key] for key in sorted(entries)],
            )
        _write_partial_manifest(
            manifest_path,
            run_identity,
            accepted=accepted,
            skipped=skipped,
            entries=entries,
            policy_requests=lazy_policy.requests,
        )

    ordered_rows = [entries[key] for key in sorted(entries)]
    if config.incremental:
        rows_to_append = [
            row
            for row in ordered_rows
            if (str(row["episode_id"]), int(row["t"])) not in output_keys
        ]
        _atomic_extend_jsonl(config.output, rows_to_append)
        total_rows = output_row_count + len(rows_to_append)
    else:
        _atomic_write_jsonl(config.output, ordered_rows)
        total_rows = len(ordered_rows)
    cache_sha256 = _sha256_file(config.output)
    accepted = sorted(set(accepted))
    incremental_policy_rows = sum(
        bool(row.get("policy_requested")) for row in entries.values()
    )
    total_policy_rows = (
        int(prior_counts.get("policy_rows", 0)) + incremental_policy_rows
        if config.incremental
        else sum(bool(row.get("policy_requested")) for row in ordered_rows)
    )
    manifest = {
        "cache_schema": CACHE_SCHEMA,
        "complete": True,
        "created_unix": time.time(),
        "elapsed_s": time.time() - started,
        "output": str(config.output),
        "cache_sha256": cache_sha256,
        "run_identity": run_identity,
        "server_metadata": (
            lazy_policy.metadata
            if lazy_policy.metadata is not None
            else prior_manifest.get("server_metadata")
        ),
        "episode_fingerprints": current_fingerprints,
        "episodes": {"accepted": accepted, "skipped": skipped},
        "counts": {
            "rows": total_rows,
            "phase_inferences": phase_inferences,
            "phase_cache_hits": cache_hits,
            "policy_requests": lazy_policy.requests,
            "policy_rows_requested": lazy_policy.rows,
            "policy_cache_hits": policy_cache_hits,
            "policy_rows": total_policy_rows,
            "episodes_reused": len(paths) - len(process_paths),
            "episodes_processed": len(process_paths),
        },
        "safety": {
            "ros_imported": False,
            "can_accessed": False,
            "commands_published": False,
            "policy_client_only": True,
        },
        "prepare_external_rlt_replay": {
            "dataset_root": None if config.dataset_root is None else str(config.dataset_root),
            "enrichment_cache": str(config.output),
            "episode_manifest": str(manifest_path),
            "phase_enter_threshold": config.phase_enter_threshold,
            "phase_enter_frames": config.phase_enter_frames,
        },
    }
    _atomic_write_json(manifest_path, manifest)
    journal_path.unlink(missing_ok=True)
    _partial_manifest_path(manifest_path).unlink(missing_ok=True)
    return manifest


def _build_observation(images: dict[str, np.ndarray], state: np.ndarray, prompt: str) -> dict[str, Any]:
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (7,) or not np.all(np.isfinite(state)):
        raise ValueError(f"state must be finite with shape (7,), got {state.shape}")
    return {
        "observation/image": images["camera1"],
        "observation/wrist_image": images["camera2"],
        "observation/state": state.copy(),
        "prompt": prompt,
    }


def _validated_policy_values(
    response: Mapping[str, Any], *, config: CacheGenerationConfig, key: tuple[str, int]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    actions = np.asarray(response.get("a_ref_absolute", response.get("actions")), dtype=np.float32)
    z_rl = np.asarray(response.get("z_rl"), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] < 10 or actions.shape[1] < 7:
        raise ValueError(f"policy actions for {key} must have at least shape (10, 7), got {actions.shape}")
    actions = actions[:10, :7]
    if not np.all(np.isfinite(actions)):
        raise ValueError(f"policy actions for {key} contain non-finite values")
    if z_rl.shape != (config.expected_z_dim,) or not np.all(np.isfinite(z_rl)):
        raise ValueError(f"z_rl for {key} must be finite with shape ({config.expected_z_dim},), got {z_rl.shape}")
    shadow = response.get("rlt_shadow", {})
    if isinstance(shadow, Mapping) and shadow.get("token_status") not in {None, "ok"}:
        raise ValueError(f"policy token failed for {key}: {dict(shadow)}")
    audit = {
        "actions_source": "actions" if "actions" in response else "a_ref_absolute",
        "base_action_rows": int(np.asarray(response.get("actions", actions)).shape[0]),
        "token_status": shadow.get("token_status") if isinstance(shadow, Mapping) else None,
        "actor_controls_robot": shadow.get("actor_controls_robot") if isinstance(shadow, Mapping) else None,
    }
    return actions, z_rl, audit


def _validate_server_metadata(metadata: Mapping[str, Any], config: CacheGenerationConfig) -> None:
    if metadata.get("rlt_actor_controls_robot") is not False:
        raise ValueError("policy service must declare rlt_actor_controls_robot=false")
    checks = {
        "base_config": config.expected_base_config,
        "base_checkpoint": config.expected_base_checkpoint,
        "token_checkpoint": config.expected_token_checkpoint,
    }
    for key, expected in checks.items():
        if expected is None:
            continue
        actual = metadata.get(key)
        if key.endswith("checkpoint"):
            matches = actual is not None and Path(str(actual)).expanduser() == Path(str(expected)).expanduser()
        else:
            matches = str(actual) == str(expected)
        if not matches:
            raise ValueError(f"policy server {key} mismatch: {actual!r} != {expected!r}")


def _run_identity(config: CacheGenerationConfig) -> dict[str, Any]:
    return {
        "cache_schema": CACHE_SCHEMA,
        "prompt": config.prompt,
        "base_fingerprint": config.base_fingerprint,
        "token_fingerprint": config.token_fingerprint,
        "phase_fingerprint": config.phase_fingerprint,
        "expected_base_config": config.expected_base_config,
        "expected_base_checkpoint": config.expected_base_checkpoint,
        "expected_token_checkpoint": config.expected_token_checkpoint,
        "expected_z_dim": config.expected_z_dim,
        "phase_enter_threshold": config.phase_enter_threshold,
        "phase_enter_frames": config.phase_enter_frames,
        "action_space": ABSOLUTE_ACTION_SPACE,
        "require_terminal_report": config.require_terminal_report,
    }


def _validate_existing_manifest(path: Path, identity: Mapping[str, Any]) -> None:
    candidates = [path, _partial_manifest_path(path)]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        value = json.loads(candidate.read_text(encoding="utf-8"))
        if value.get("run_identity") != identity:
            raise ValueError(f"existing cache manifest identity mismatch: {candidate}")


def _validate_cached_identity(
    row: Mapping[str, Any],
    identity: Mapping[str, Any],
    observation_sha256: str,
    *,
    key: tuple[str, int],
) -> None:
    expected = {
        "cache_schema": identity["cache_schema"],
        "base_fingerprint": identity["base_fingerprint"],
        "token_fingerprint": identity["token_fingerprint"],
        "phase_fingerprint": identity["phase_fingerprint"],
        "observation_sha256": observation_sha256,
    }
    mismatches = {name: (value, row.get(name)) for name, value in expected.items() if row.get(name) != value}
    if mismatches:
        raise ValueError(f"stale/incompatible cache entry {key}: {mismatches}")


def _read_images(global_path: Path, wrist_path: Path) -> dict[str, np.ndarray]:
    with Image.open(global_path) as image:
        global_image = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(wrist_path) as image:
        wrist_image = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return {"camera1": global_image, "camera2": wrist_image}


def _resolve_image(episode_root: Path, relative: str | None, label: str) -> Path:
    if not relative:
        raise ValueError(f"missing {label}")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be relative to the episode root: {relative}")
    resolved = (episode_root / path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _terminal_report_reward(episode_root: Path) -> float:
    report_path = episode_root / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"terminal report does not exist: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("exclude_from_training") is True:
        reason = report.get("exclusion_reason", "explicitly excluded")
        raise ValueError(f"episode is quarantined from training: {reason}")
    reward = report.get("terminal_reward")
    if reward is None or isinstance(reward, bool):
        raise ValueError(f"terminal_reward must be 0 or 1 in {report_path}")
    reward = float(reward)
    if reward not in {0.0, 1.0}:
        raise ValueError(f"terminal_reward must be 0 or 1 in {report_path}, got {reward}")
    outcome = report.get("outcome")
    if outcome not in {None, "episode_done"}:
        raise ValueError(f"report is not a rewarded completed episode: outcome={outcome!r}")
    return reward


def _observation_sha256(
    record: RealStepRecord, *, global_path: Path, wrist_path: Path, prompt: str
) -> str:
    digest = hashlib.sha256()
    digest.update(CACHE_SCHEMA.encode())
    digest.update(str(record.episode_id).encode())
    digest.update(str(int(record.t)).encode())
    digest.update(np.asarray(record.state, dtype="<f4").tobytes())
    digest.update(prompt.encode("utf-8"))
    for path in (global_path, wrist_path):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _finite_probability(value: Any, *, key: tuple[str, int]) -> float:
    probability = float(value)
    if not np.isfinite(probability):
        raise ValueError(f"phase probability for {key} is non-finite")
    return float(np.clip(probability, 0.0, 1.0))


def _load_resume_entries(
    output: Path, journal: Path, identity: Mapping[str, Any]
) -> dict[tuple[str, int], dict[str, Any]]:
    entries: dict[tuple[str, int], dict[str, Any]] = {}
    for path in (output, journal):
        if not path.is_file():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["episode_id"]), int(row["t"]))
            # Journal duplicates are intentional updates; final output is
            # compacted and CachedEnrichmentProvider will see unique keys.
            if row.get("cache_schema") != identity["cache_schema"]:
                raise ValueError(f"invalid cache schema on {path}:{line_number}")
            entries[key] = row
    return entries


def _load_resume_entries_filtered(
    output: Path,
    journal: Path,
    identity: Mapping[str, Any],
    *,
    episode_ids: set[str],
) -> tuple[
    dict[tuple[str, int], dict[str, Any]],
    set[tuple[str, int]],
    int,
]:
    """Load only requested episode rows while counting the existing cache.

    The 2,048-D Token makes the JSONL hundreds of MB.  Looking at the compact
    ``episode_id`` prefix before ``json.loads`` keeps incremental refresh O(new
    episodes) in CPU and memory while retaining a full sequential integrity
    read of the existing file.
    """

    entries: dict[tuple[str, int], dict[str, Any]] = {}
    output_keys: set[tuple[str, int]] = set()
    output_rows = 0
    for path in (output, journal):
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                if path == output:
                    output_rows += 1
                episode_id = _compact_json_episode_id(line)
                if episode_id is not None and episode_id not in episode_ids:
                    continue
                row = json.loads(line)
                key = (str(row["episode_id"]), int(row["t"]))
                if key[0] not in episode_ids:
                    continue
                if row.get("cache_schema") != identity["cache_schema"]:
                    raise ValueError(
                        f"invalid cache schema on {path}:{line_number}"
                    )
                entries[key] = row
                if path == output:
                    output_keys.add(key)
    return entries, output_keys, output_rows


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


def _initialize_journal(path: Path, entries: Mapping[tuple[str, int], Mapping[str, Any]]) -> None:
    if path.exists():
        return
    _atomic_write_jsonl(path, [entries[key] for key in sorted(entries)])


def _append_jsonl_fsync(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_extend_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.extend.tmp")
    with temporary.open("wb") as stream:
        if path.is_file():
            with path.open("rb") as source:
                shutil.copyfileobj(source, stream, length=8 * 1024 * 1024)
        for row in rows:
            stream.write(
                (
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
            )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_json_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON mapping expected: {path}")
    return value


def _episode_input_fingerprint(path: Path) -> dict[str, int]:
    """Cheap immutability guard for a terminal episode and its admission."""

    report = path.parent / "report.json"
    if not path.is_file() or not report.is_file():
        raise FileNotFoundError(
            f"episode/report pair is incomplete: {path}, {report}"
        )
    episode_stat = path.stat()
    report_stat = report.stat()
    return {
        "episode_size": int(episode_stat.st_size),
        "episode_mtime_ns": int(episode_stat.st_mtime_ns),
        "report_size": int(report_stat.st_size),
        "report_mtime_ns": int(report_stat.st_mtime_ns),
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_partial_manifest(
    manifest_path: Path,
    identity: Mapping[str, Any],
    *,
    accepted: Sequence[str],
    skipped: Sequence[Mapping[str, str]],
    entries: Mapping[tuple[str, int], Mapping[str, Any]],
    policy_requests: int,
) -> None:
    _atomic_write_json(
        _partial_manifest_path(manifest_path),
        {
            "cache_schema": CACHE_SCHEMA,
            "complete": False,
            "run_identity": identity,
            "episodes": {"accepted": list(accepted), "skipped": list(skipped)},
            "counts": {"rows": len(entries), "policy_requests_this_run": int(policy_requests)},
        },
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _journal_path(output: Path) -> Path:
    return output.with_name(output.name + ".partial")


def _manifest_path(output: Path) -> Path:
    return output.with_name(output.name + ".manifest.json")


def _partial_manifest_path(manifest: Path) -> Path:
    return manifest.with_name(manifest.name + ".partial")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-jsonl", action="append", default=[])
    parser.add_argument("--episode-dir", help="Root searched recursively for episode.jsonl")
    parser.add_argument(
        "--readiness-probe",
        type=Path,
        help=(
            "Authoritative run_online_rlt_update --dry-run JSON. Use exactly "
            "its trainable episodes and complete same-plan C10 row indices."
        ),
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase-checkpoint", type=Path, required=True)
    parser.add_argument("--phase-device", default="cpu")
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8001)
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--phase-enter-threshold", type=float, default=0.5)
    parser.add_argument("--phase-enter-frames", type=int, default=3)
    parser.add_argument("--expected-z-dim", type=int, default=2048)
    parser.add_argument("--base-fingerprint", required=True)
    parser.add_argument("--token-fingerprint", required=True)
    parser.add_argument("--expected-base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--expected-base-checkpoint", default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--expected-token-checkpoint", default=DEFAULT_TOKEN_CHECKPOINT)
    parser.add_argument("--skip-invalid-episodes", action="store_true")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Reuse immutable accepted episodes and append only new cache rows.",
    )
    parser.add_argument(
        "--token-only",
        action="store_true",
        help="Refresh z_rl without invoking Pi0.5 or advancing its RNG.",
    )
    parser.add_argument("--token-batch-size", type=int, default=4)
    parser.add_argument(
        "--allow-missing-terminal-report",
        action="store_true",
        help="Unsafe/debug: include terminal JSONL files without a rewarded report.json.",
    )
    return parser


def _collect_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(path) for path in args.episode_jsonl]
    if args.episode_dir:
        paths.extend(Path(args.episode_dir).expanduser().rglob("episode.jsonl"))
    return sorted({path.expanduser().resolve() for path in paths})


def _load_readiness_candidates(
    path: Path,
    *,
    dataset_root: Path,
) -> tuple[list[Path], dict[str, set[int]]]:
    probe = _read_json_mapping(path.expanduser().resolve())
    if probe.get("outcome") != "dry_run_would_update":
        raise ValueError(
            "readiness probe is not an authoritative update candidate: "
            f"{probe.get('outcome')!r}"
        )
    trainable_ids = probe.get("trainable_episode_ids")
    audits = probe.get("episode_audit")
    if (
        not isinstance(trainable_ids, list)
        or not trainable_ids
        or not isinstance(audits, list)
    ):
        raise ValueError("readiness probe lacks trainable episodes/audits")
    audit_by_id = {
        str(item.get("episode_id")): item
        for item in audits
        if isinstance(item, Mapping) and item.get("episode_id") is not None
    }
    paths: list[Path] = []
    candidates: dict[str, set[int]] = {}
    root = dataset_root.expanduser().resolve()
    for episode_id in sorted({str(value) for value in trainable_ids}):
        audit = audit_by_id.get(episode_id)
        if audit is None:
            raise ValueError(
                f"readiness probe has no audit for {episode_id}"
            )
        contract = audit.get("persistent_v2_contract")
        plans = (
            contract.get("complete_c10_row_indices_by_plan")
            if isinstance(contract, Mapping)
            else None
        )
        if not isinstance(plans, Mapping) or not plans:
            raise ValueError(
                f"readiness probe has no strict C10 candidates for {episode_id}"
            )
        indices: set[int] = set()
        for plan_id, raw_indices in plans.items():
            if not isinstance(raw_indices, list):
                raise ValueError(f"invalid C10 indices for {plan_id}")
            plan_indices = [int(value) for value in raw_indices]
            if (
                len(plan_indices) != 10
                or plan_indices
                != list(range(plan_indices[0], plan_indices[0] + 10))
            ):
                raise ValueError(
                    f"readiness C10 is not exactly contiguous for {plan_id}"
                )
            overlap = indices.intersection(plan_indices)
            if overlap:
                raise ValueError(
                    f"readiness C10 candidates overlap for {episode_id}: "
                    f"{sorted(overlap)[:3]}"
                )
            indices.update(plan_indices)
        episode_path = (root / episode_id / "episode.jsonl").resolve()
        try:
            episode_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"readiness episode escapes dataset root: {episode_id}"
            ) from exc
        if not episode_path.is_file():
            raise FileNotFoundError(episode_path)
        paths.append(episode_path)
        candidates[episode_id] = indices
    return paths, candidates


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parser().parse_args(argv)
    if args.readiness_probe:
        if args.episode_jsonl or args.episode_dir:
            raise ValueError(
                "--readiness-probe is mutually exclusive with episode inputs"
            )
        paths, candidate_indices = _load_readiness_candidates(
            args.readiness_probe,
            dataset_root=args.dataset_root,
        )
    else:
        paths = _collect_paths(args)
        candidate_indices = None
    if not paths:
        raise ValueError(
            "provide --episode-jsonl, --episode-dir or --readiness-probe"
        )
    phase_checkpoint = args.phase_checkpoint.expanduser().resolve()
    if not phase_checkpoint.is_file():
        raise FileNotFoundError(f"phase checkpoint does not exist: {phase_checkpoint}")

    # Reuse the exact live classifier preprocessing.  Neither this class nor
    # the websocket client imports or touches ROS/CAN.
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from piper_runtime.rlt_phase_gate import TorchPhaseClassifier

    phase_predictor = TorchPhaseClassifier(phase_checkpoint, device=args.phase_device)

    def policy_factory() -> PolicyClient:
        return WebsocketClientPolicy(
            args.policy_host,
            args.policy_port,
            connect_timeout_s=args.connect_timeout,
            request_timeout_s=args.request_timeout,
        )

    config = CacheGenerationConfig(
        output=args.output,
        dataset_root=args.dataset_root,
        prompt=args.prompt,
        phase_enter_threshold=args.phase_enter_threshold,
        phase_enter_frames=args.phase_enter_frames,
        expected_z_dim=args.expected_z_dim,
        base_fingerprint=args.base_fingerprint,
        token_fingerprint=args.token_fingerprint,
        phase_fingerprint=_sha256_file(phase_checkpoint),
        expected_base_config=args.expected_base_config or None,
        expected_base_checkpoint=args.expected_base_checkpoint or None,
        expected_token_checkpoint=args.expected_token_checkpoint or None,
        skip_invalid_episodes=args.skip_invalid_episodes,
        require_terminal_report=not args.allow_missing_terminal_report,
        incremental=args.incremental,
        token_only=args.token_only,
        token_batch_size=args.token_batch_size,
    )
    report = generate_enrichment_cache(
        paths,
        config=config,
        phase_predictor=phase_predictor,
        policy_client_factory=policy_factory,
        policy_candidate_indices_by_episode=candidate_indices,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
