from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

_WORKSPACE_SRC = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(_WORKSPACE_SRC))

from openpi.rlt.real.config import RealRLTConfig  # noqa: E402
from openpi.rlt.real.config import Source  # noqa: E402
from openpi.rlt.real.external_episode import ExternalEpisodeContract  # noqa: E402
from openpi.rlt.real.external_episode import load_episode_jsonl  # noqa: E402
from openpi.rlt.real.replay import RealTransition  # noqa: E402
from openpi.rlt.real.replay import chunk_real_episode  # noqa: E402
from openpi.rlt.real.replay_enrichment import ABSOLUTE_ACTION_SPACE  # noqa: E402
from openpi.rlt.real.replay_enrichment import CachedEnrichmentProvider  # noqa: E402
from openpi.rlt.real.replay_enrichment import PhaseProbabilityProvider  # noqa: E402
from openpi.rlt.real.replay_enrichment import ReferenceTokenProvider  # noqa: E402
from openpi.rlt.real.replay_enrichment import ReferenceTokenValue  # noqa: E402
from openpi.rlt.real.replay_enrichment import assign_stratified_episode_splits  # noqa: E402
from openpi.rlt.real.replay_enrichment import derive_episode_uid  # noqa: E402
from openpi.rlt.real.replay_enrichment import enrich_episode  # noqa: E402
from openpi.rlt.real.replay_enrichment import load_callback  # noqa: E402
from openpi.rlt.real.replay_io import write_replay_npz  # noqa: E402
try:  # noqa: E402
    from . import persistent_v2_contract as _persistent_v2
except ImportError:  # Direct executable invocation.
    import persistent_v2_contract as _persistent_v2  # noqa: E402

PERSISTENT_V2_ACTION_SCHEMA = _persistent_v2.ACTION_SCHEMA_FINGERPRINT
PERSISTENT_V2_EXECUTION_PROFILE = _persistent_v2.ACTOR_EXECUTION_PROFILE
PERSISTENT_V2_PROJECTION_PROFILE = _persistent_v2.ACTOR_PROJECTION_PROFILE
PERSISTENT_V2_CONTROL_HZ = _persistent_v2.CONTROL_HZ
PERSISTENT_V2_FILTER_PROFILE = _persistent_v2.EXECUTION_FILTER_PROFILE
PERSISTENT_V2_FILTER_TAU_S = _persistent_v2.EXECUTION_FILTER_TAU_S
audit_persistent_v2_episode = _persistent_v2.audit_persistent_v2_episode


def prepare_external_replay(
    episode_paths: Sequence[str | Path],
    *,
    output_dir: str | Path,
    dataset_root: str | Path | None = None,
    chunk_length: int = RealRLTConfig.chunk_length,
    stride: int = RealRLTConfig.chunk_stride,
    n_step: int = RealRLTConfig.n_step,
    gamma: float = RealRLTConfig.gamma,
    require_images: bool = True,
    check_image_exists: bool = True,
    reference_provider: ReferenceTokenProvider | None = None,
    phase_provider: PhaseProbabilityProvider | None = None,
    allow_logged_reference: bool = False,
    allow_logged_phase: bool = False,
    preserve_logged_reference: bool = False,
    phase_enter_threshold: float = 0.5,
    phase_enter_frames: int = 3,
    base_fingerprint: str | None = None,
    token_fingerprint: str | None = None,
    phase_fingerprint: str | None = None,
    action_schema_fingerprint: str | None = None,
    actor_projection_profile: str | None = None,
    actor_execution_profile: str | None = None,
    execution_filter_profile: str | None = None,
    execution_filter_tau_s: float | None = None,
    control_hz: float | None = None,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.0,
    split_seed: str = "piper-rlt-v1",
    split_registry_path: str | Path | None = None,
) -> dict[str, Any]:
    paths = [Path(path).resolve() for path in episode_paths]
    if not paths:
        raise ValueError("at least one episode JSONL file is required")
    persistent_v2 = actor_execution_profile == PERSISTENT_V2_EXECUTION_PROFILE
    if persistent_v2:
        expected = {
            "action_schema_fingerprint": (
                action_schema_fingerprint,
                PERSISTENT_V2_ACTION_SCHEMA,
            ),
            "actor_projection_profile": (
                actor_projection_profile,
                PERSISTENT_V2_PROJECTION_PROFILE,
            ),
            "execution_filter_profile": (
                execution_filter_profile,
                PERSISTENT_V2_FILTER_PROFILE,
            ),
            "execution_filter_tau_s": (
                execution_filter_tau_s,
                PERSISTENT_V2_FILTER_TAU_S,
            ),
            "control_hz": (control_hz, PERSISTENT_V2_CONTROL_HZ),
            "chunk_length": (chunk_length, 10),
            "stride": (stride, 10),
            "n_step": (n_step, 10),
        }
        mismatches = {
            key: {"actual": actual, "expected": wanted}
            for key, (actual, wanted) in expected.items()
            if actual != wanted
        }
        if mismatches:
            raise ValueError(
                f"persistent-v2 replay contract mismatch: {mismatches}"
            )
    elif chunk_length != 10 or stride != 2 or n_step != 10:
        raise ValueError("real Piper RLT v1 requires C=10, stride=2, n_step=10")

    contract = ExternalEpisodeContract(
        chunk_length=chunk_length,
        require_images=require_images,
        check_image_exists=check_image_exists,
    )
    transitions: list[RealTransition] = []
    episode_reports: dict[str, dict[str, Any]] = {}
    episode_inputs: dict[str, dict[str, str]] = {}
    root = Path(dataset_root).resolve() if dataset_root is not None else None

    for path in paths:
        # Capture logs store image paths relative to the episode directory.
        records = load_episode_jsonl(path, dataset_root=path.parent, contract=contract)
        original_episode_id = records[0].episode_id
        episode_uid = _episode_uid(path, root=root)
        records = [dataclasses.replace(record, episode_id=episode_uid) for record in records]
        if persistent_v2:
            audit = audit_persistent_v2_episode(
                path,
                expected_execution_profile=actor_execution_profile,
                expected_action_schema=action_schema_fingerprint,
                expected_projection_profile=actor_projection_profile,
                expected_filter_profile=execution_filter_profile,
                expected_filter_tau_s=float(execution_filter_tau_s),
                expected_control_hz=float(control_hz),
            )
            persistent_transitions, persistent_statistics = (
                _persistent_v2_transitions(
                    records,
                    audit=audit,
                    episode_root=path.parent,
                    reference_provider=reference_provider,
                    phase_provider=phase_provider,
                    allow_logged_reference=allow_logged_reference,
                    allow_logged_phase=allow_logged_phase,
                    preserve_logged_reference=preserve_logged_reference,
                    phase_enter_threshold=phase_enter_threshold,
                    phase_enter_frames=phase_enter_frames,
                    gamma=gamma,
                )
            )
            transitions.extend(persistent_transitions)
            episode_reports[episode_uid] = persistent_statistics
        else:
            enriched = enrich_episode(
                records,
                episode_root=path.parent,
                reference_provider=reference_provider,
                phase_provider=phase_provider,
                allow_logged_reference=allow_logged_reference,
                allow_logged_phase=allow_logged_phase,
                preserve_logged_reference=preserve_logged_reference,
                enter_threshold=phase_enter_threshold,
                enter_frames=phase_enter_frames,
            )
            for segment in enriched.segments:
                transitions.extend(
                    chunk_real_episode(
                        segment,
                        chunk_length=chunk_length,
                        stride=stride,
                        n_step=n_step,
                        gamma=gamma,
                    )
                )
            episode_reports[episode_uid] = enriched.statistics
        episode_inputs[episode_uid] = {
            "file": str(path),
            "original_episode_id": original_episode_id,
        }

    # Only episodes that actually yielded transitions participate in the
    # learner split.  Including a skipped/too-short episode could otherwise
    # consume the entire validation allocation without providing a val row.
    episode_ids = sorted({transition.episode_id for transition in transitions})
    episode_labels = {
        episode_id: ("success" if float(episode_reports[episode_id]["terminal_reward"]) > 0.0 else "failure")
        for episode_id in episode_ids
    }
    split = assign_stratified_episode_splits(
        episode_labels,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=split_seed,
    )
    split, split_strategy = _apply_split_registry(
        split,
        episode_ids=episode_ids,
        registry_path=split_registry_path,
        seed=split_seed,
    )
    statistics = _aggregate_statistics(episode_reports, transitions)
    metadata = {
        "input_episodes": len(paths),
        "input_episode_files": [str(path) for path in paths],
        "episode_inputs": episode_inputs,
        "chunking": {
            "chunk_length": int(chunk_length),
            "stride": int(stride),
            "n_step": int(n_step),
            "gamma": float(gamma),
        },
        "external_contract": {
            "require_images": bool(require_images),
            "check_image_exists": bool(check_image_exists),
            "state_dim": contract.state_dim,
            "action_dim": contract.action_dim,
        },
        "coordinate_contract": {
            "training_action": {
                "joints_0_to_5": "absolute_command - state_at_chunk_start",
                "gripper_6": "absolute_command",
            },
            "state": "absolute_joint_state_and_absolute_gripper",
            "audit_arrays": "*_absolute preserve published/recomputed absolute targets",
            "logged_reference_audit": "a_ref_original_absolute",
            "action_schema_fingerprint": action_schema_fingerprint,
            "actor_projection_profile": actor_projection_profile,
            "actor_execution_profile": actor_execution_profile,
            "execution_filter_profile": execution_filter_profile,
            "execution_filter_tau_s": execution_filter_tau_s,
            "control_hz": control_hz,
            "persistent_v2_selection": (
                "only audited complete physical same-plan C10 offsets 0..9"
                if persistent_v2
                else None
            ),
        },
        "provenance": {
            "reference_recomputed": bool(reference_provider is not None and not preserve_logged_reference),
            "token_recomputed": reference_provider is not None,
            "logged_reference_used_for_training": bool(
                reference_provider is None or preserve_logged_reference
            ),
            "logged_reference_preserved_with_fresh_token": bool(
                reference_provider is not None and preserve_logged_reference
            ),
            "base_fingerprint": base_fingerprint,
            "rl_token_fingerprint": token_fingerprint,
            "phase_classifier_fingerprint": phase_fingerprint,
            "phase_gate": {
                "type": "single_latch_until_terminal",
                "enter_threshold": float(phase_enter_threshold),
                "enter_frames": int(phase_enter_frames),
            },
        },
        "statistics": statistics,
        "episode_statistics": episode_reports,
        "episode_split": {
            "strategy": split_strategy,
            "seed": split_seed,
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
            **split,
        },
    }
    episode_split_by_id = {
        episode_id: split_name
        for split_name in ("train", "validation", "test")
        for episode_id in split[split_name]
    }
    return write_replay_npz(
        transitions,
        output_dir,
        manifest_metadata=metadata,
        episode_split_by_id=episode_split_by_id,
    )


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = _parser()
    args = parser.parse_args(argv)
    episode_paths = _collect_episode_paths(args)
    if not episode_paths:
        parser.error("provide --episode-jsonl, --episode-dir, or --episode-manifest")

    dataset_root = (
        Path(args.dataset_root).resolve()
        if args.dataset_root is not None
        else None
    )
    cache_episode_ids = {
        _episode_uid(path, root=dataset_root) for path in episode_paths
    }
    cache = (
        CachedEnrichmentProvider.from_jsonl(
            args.enrichment_cache,
            episode_ids=cache_episode_ids,
        )
        if args.enrichment_cache
        else None
    )
    reference_provider: ReferenceTokenProvider | None
    phase_provider: PhaseProbabilityProvider | None
    if args.reference_provider:
        reference_provider = _adapt_reference_callback(load_callback(args.reference_provider))
    else:
        reference_provider = cache
    if args.phase_provider:
        phase_provider = _adapt_phase_callback(load_callback(args.phase_provider))
    elif cache is not None:
        phase_provider = cache.phase_probability
    else:
        phase_provider = None

    try:
        report = prepare_external_replay(
            episode_paths,
            output_dir=args.output,
            dataset_root=args.dataset_root,
            chunk_length=args.chunk_length,
            stride=args.stride,
            n_step=args.n_step,
            gamma=args.gamma,
            require_images=not args.no_require_images,
            check_image_exists=not args.no_check_image_exists,
            reference_provider=reference_provider,
            phase_provider=phase_provider,
            allow_logged_reference=args.allow_logged_reference,
            allow_logged_phase=args.allow_logged_phase,
            preserve_logged_reference=args.preserve_logged_reference,
            phase_enter_threshold=args.phase_enter_threshold,
            phase_enter_frames=args.phase_enter_frames,
            base_fingerprint=args.base_fingerprint,
            token_fingerprint=args.token_fingerprint,
            phase_fingerprint=args.phase_fingerprint,
            action_schema_fingerprint=args.action_schema_fingerprint,
            actor_projection_profile=args.actor_projection_profile,
            actor_execution_profile=args.actor_execution_profile,
            execution_filter_profile=args.execution_filter_profile,
            execution_filter_tau_s=args.execution_filter_tau_s,
            control_hz=args.control_hz,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            split_seed=args.split_seed,
            split_registry_path=args.split_registry,
        )
    except ValueError as exc:
        if argv is None:
            parser.error(str(exc))
        raise
    if argv is None:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def _parser() -> argparse.ArgumentParser:
    cfg = RealRLTConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Enrich Piper/Pika episodes with frozen full-20k references, RL Token and phase gate, "
            "then write a strict OpenPI real-RLT replay."
        )
    )
    parser.add_argument("--episode-jsonl", action="append", default=[], help="One episode.jsonl file. Repeatable.")
    parser.add_argument("--episode-dir", help="Root searched recursively for episode.jsonl files.")
    parser.add_argument(
        "--episode-manifest",
        help="Cache-generator manifest; consumes exactly episodes.accepted and excludes debug/nonterminal episodes.",
    )
    parser.add_argument("--dataset-root", help="Stable root used to derive globally unique episode IDs.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-length", type=int, default=cfg.chunk_length)
    parser.add_argument("--stride", type=int, default=cfg.chunk_stride)
    parser.add_argument("--n-step", type=int, default=cfg.n_step)
    parser.add_argument("--gamma", type=float, default=cfg.gamma)
    parser.add_argument("--enrichment-cache", help="JSONL cache containing a_ref, z_rl and phase_probability per row.")
    parser.add_argument("--reference-provider", help="Python callback package.module:function for full-20k a_ref + z_rl.")
    parser.add_argument("--phase-provider", help="Python callback package.module:function for phase probability.")
    parser.add_argument(
        "--allow-logged-reference",
        action="store_true",
        help="Unsafe/debug only: train from logged a_ref/z_rl instead of recomputing frozen full-20k outputs.",
    )
    parser.add_argument(
        "--preserve-logged-reference",
        action="store_true",
        help=(
            "Use the enrichment provider only to refresh z_rl for the exact saved observation, "
            "while keeping the stochastic Pi0.5 a_ref logged at execution time. Requires a "
            "reference provider or --enrichment-cache."
        ),
    )
    parser.add_argument(
        "--allow-logged-phase", action="store_true", help="Debug only: use phase_probability already in JSONL."
    )
    parser.add_argument("--phase-enter-threshold", type=float, default=0.5)
    parser.add_argument("--phase-enter-frames", type=int, default=3)
    parser.add_argument("--base-fingerprint")
    parser.add_argument("--token-fingerprint")
    parser.add_argument("--phase-fingerprint")
    parser.add_argument("--action-schema-fingerprint")
    parser.add_argument("--actor-projection-profile")
    parser.add_argument("--actor-execution-profile")
    parser.add_argument("--execution-filter-profile")
    parser.add_argument("--execution-filter-tau-s", type=float)
    parser.add_argument("--control-hz", type=float)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument("--split-seed", default="piper-rlt-v1")
    parser.add_argument(
        "--split-registry",
        help=(
            "Persistent online split registry. The first replay records train/validation assignments; "
            "later unseen episodes are added to train without moving held-out episodes."
        ),
    )
    parser.add_argument("--no-require-images", action="store_true")
    parser.add_argument("--no-check-image-exists", action="store_true")
    return parser


def _collect_episode_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(path) for path in args.episode_jsonl]
    if args.episode_dir:
        paths.extend(Path(args.episode_dir).rglob("episode.jsonl"))
    if args.episode_manifest:
        manifest_path = Path(args.episode_manifest).expanduser()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") is not True:
            raise ValueError(f"episode manifest is not complete: {manifest_path}")
        accepted = manifest.get("episodes", {}).get("accepted")
        if not isinstance(accepted, list) or not accepted or not all(isinstance(item, str) for item in accepted):
            raise ValueError("episode manifest must contain non-empty episodes.accepted paths")
        paths.extend(Path(path) for path in accepted)
    return sorted({path.resolve() for path in paths})


def _persistent_v2_transitions(
    records,
    *,
    audit: Mapping[str, Any],
    episode_root: Path,
    reference_provider: ReferenceTokenProvider | None,
    phase_provider: PhaseProbabilityProvider | None,
    allow_logged_reference: bool,
    allow_logged_phase: bool,
    preserve_logged_reference: bool,
    phase_enter_threshold: float,
    phase_enter_frames: int,
    gamma: float,
) -> tuple[list[RealTransition], dict[str, Any]]:
    """Create persistent transitions without inventing terminal C10s.

    Enrich the real episode once, retain only audited complete committed C10
    rows, and let the persistent chunker bootstrap only across a truly
    contiguous next complete C10.  A plan at an invalid/gapped boundary is
    dropped instead of being relabelled as a zero-reward terminal transition.
    The final audited C10 alone receives the episode terminal reward.
    """

    plan_rows = audit["complete_c10_row_indices_by_plan"]
    ordered_plans = sorted(plan_rows.items(), key=lambda item: item[1][0])
    if not ordered_plans:
        raise ValueError("persistent-v2 episode has no audited complete C10")
    terminal_reward = float(records[-1].reward)
    audited_indices: list[int] = []
    for plan_id, indices in ordered_plans:
        if indices != list(range(indices[0], indices[0] + 10)):
            raise ValueError(f"persistent C10 indices are not contiguous: {plan_id}")
        audited_indices.extend(indices)

    audited_index_set = set(audited_indices)
    enrichment_records = [
        (
            record
            if index in audited_index_set
            else dataclasses.replace(record, replay_include=False)
        )
        for index, record in enumerate(records)
    ]
    enriched = enrich_episode(
        enrichment_records,
        episode_root=episode_root,
        reference_provider=reference_provider,
        phase_provider=phase_provider,
        allow_logged_reference=allow_logged_reference,
        allow_logged_phase=allow_logged_phase,
        preserve_logged_reference=preserve_logged_reference,
        enter_threshold=phase_enter_threshold,
        enter_frames=phase_enter_frames,
    )
    enriched_by_t = {int(record.t): record for record in enriched.records}
    selected = []
    selected_plan_ids: list[str] = []
    phase_gate_drop_reasons: dict[str, str] = {}
    for plan_id, indices in ordered_plans:
        enriched_plan = [
            enriched_by_t.get(int(records[index].t)) for index in indices
        ]
        missing_count = sum(record is None for record in enriched_plan)
        if missing_count:
            phase_gate_drop_reasons[plan_id] = (
                "phase_gate_inactive_complete_c10"
                if missing_count == len(indices)
                else "phase_gate_partial_complete_c10"
            )
            continue
        selected_plan_ids.append(plan_id)
        selected.extend(
            dataclasses.replace(
                record,
                replay_include=True,
                waiting_for_reward=False,
                reward=0.0,
                done=False,
                gate_active=True,
            )
            for record in enriched_plan
        )
    if selected:
        selected[-1] = dataclasses.replace(
            selected[-1], reward=terminal_reward, done=True
        )

    selected_segments: list[list[Any]] = []
    current: list[Any] = []
    for record in selected:
        if current and (
            record.episode_id != current[-1].episode_id
            or record.t != current[-1].t + 1
        ):
            selected_segments.append(current)
            current = []
        current.append(record)
    if current:
        selected_segments.append(current)

    result: list[RealTransition] = []
    for segment in selected_segments:
        result.extend(
            chunk_real_episode(
                segment,
                chunk_length=10,
                stride=10,
                n_step=10,
                gamma=gamma,
            )
        )
    result_by_t = {int(transition.t): transition for transition in result}
    if len(result_by_t) != len(result):
        raise ValueError("persistent replay produced duplicate transition anchors")

    per_plan: list[dict[str, Any]] = []
    dropped_no_bootstrap = 0
    for plan_id, indices in ordered_plans:
        start_t = int(records[indices[0]].t)
        transition = result_by_t.get(start_t)
        emitted = transition is not None
        drop_reason = phase_gate_drop_reasons.get(plan_id)
        if not emitted and drop_reason is None:
            drop_reason = "no_contiguous_complete_t_plus_10_bootstrap"
            dropped_no_bootstrap += 1
        per_plan.append(
            {
                "plan_id": plan_id,
                "start_t": start_t,
                "emitted": emitted,
                "terminal": bool(transition.done) if emitted else False,
                "bootstrap_t": (
                    None
                    if not emitted or transition.done
                    else int(transition.t + 10)
                ),
                "drop_reason": (
                    None
                    if emitted
                    else drop_reason
                ),
            }
        )
        if plan_id == (selected_plan_ids[-1] if selected_plan_ids else None) and (
            transition is None or not transition.done
        ):
            raise ValueError("final gate-active audited persistent C10 was not terminalized")

    source_counts = Counter(record.source for record in selected)
    phase_gate_inactive_rows = sum(
        10
        for reason in phase_gate_drop_reasons.values()
        if reason == "phase_gate_inactive_complete_c10"
    )
    phase_gate_partial_rows = sum(
        10
        for reason in phase_gate_drop_reasons.values()
        if reason == "phase_gate_partial_complete_c10"
    )
    unaudited_rows = len(records) - len(audited_indices)
    return result, {
        "raw_rows": len(records),
        "valid_action_rows": len(selected),
        "filtered_rows": len(records) - len(selected),
        "filtered_by_reason": {
            "persistent_v2_not_physically_committed_complete_c10": unaudited_rows,
            "persistent_v2_phase_gate_inactive_complete_c10": (
                phase_gate_inactive_rows
            ),
            "persistent_v2_phase_gate_partial_complete_c10": (
                phase_gate_partial_rows
            ),
        },
        "segments": len(selected_segments),
        "source_steps": dict(sorted(source_counts.items())),
        "human_steps": int(source_counts.get(Source.HUMAN_PIKA, 0)),
        "actor_steps": int(source_counts.get(Source.RLT, 0)),
        "persistent_v2_transition_count": len(result),
        "persistent_v2_dropped_no_bootstrap_c10": dropped_no_bootstrap,
        "persistent_v2_dropped_phase_gate_c10": len(phase_gate_drop_reasons),
        "terminal_reward": terminal_reward,
        "gate_enter_t": enriched.statistics.get(
            "gate_enter_t", int(records[ordered_plans[0][1][0]].t)
        ),
        "gate_ever_active": bool(enriched.statistics.get("gate_ever_active")),
        "reference_recomputed": bool(
            reference_provider is not None and not preserve_logged_reference
        ),
        "token_recomputed": reference_provider is not None,
        "logged_reference_preserved": bool(
            reference_provider is not None and preserve_logged_reference
        ),
        "persistent_v2_audit": dict(audit),
        "persistent_v2_plans": per_plan,
    }


def _episode_uid(path: Path, *, root: Path | None) -> str:
    return derive_episode_uid(path, root=root)


def _adapt_reference_callback(callback: Callable[..., Any]) -> ReferenceTokenProvider:
    def provider(record, episode_root) -> ReferenceTokenValue:
        value = callback(record, episode_root)
        if isinstance(value, ReferenceTokenValue):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("reference provider must return ReferenceTokenValue or a mapping")
        return ReferenceTokenValue(
            a_ref=value.get("a_ref_absolute", value.get("a_ref")),
            z_rl=value["z_rl"],
            action_space=str(value.get("action_space", ABSOLUTE_ACTION_SPACE)),
        )

    return provider


def _adapt_phase_callback(callback: Callable[..., Any]) -> PhaseProbabilityProvider:
    def provider(record, episode_root) -> float:
        value = callback(record, episode_root)
        if isinstance(value, Mapping):
            value = value["phase_probability"]
        return float(value)

    return provider


def _apply_split_registry(
    proposed: Mapping[str, Sequence[str]],
    *,
    episode_ids: Sequence[str],
    registry_path: str | Path | None,
    seed: str,
) -> tuple[dict[str, list[str]], str]:
    strategy = "deterministic_whole_episode_sha256_rank_stratified_by_terminal_reward"
    if registry_path is None:
        return {name: sorted(proposed[name]) for name in ("train", "validation", "test")}, strategy

    path = Path(registry_path).expanduser()
    assignments: dict[str, str]
    if path.is_file():
        registry = json.loads(path.read_text(encoding="utf-8"))
        if registry.get("format") != "openpi_real_rlt_online_split_registry_v1":
            raise ValueError(f"unsupported split registry: {path}")
        assignments = {str(key): str(value) for key, value in registry.get("assignments", {}).items()}
        invalid = sorted(set(assignments.values()).difference({"train", "validation", "test"}))
        if invalid:
            raise ValueError(f"split registry contains invalid labels: {invalid}")
        for episode_id in episode_ids:
            assignments.setdefault(episode_id, "train")
    else:
        assignments = {
            episode_id: split_name
            for split_name in ("train", "validation", "test")
            for episode_id in proposed[split_name]
        }

    split = {
        split_name: sorted(episode_id for episode_id in episode_ids if assignments[episode_id] == split_name)
        for split_name in ("train", "validation", "test")
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "format": "openpi_real_rlt_online_split_registry_v1",
                "seed": seed,
                "policy": "initial_stratified_then_new_episodes_train_only",
                "assignments": dict(sorted(assignments.items())),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return split, "persistent_registry_initial_stratified_then_new_episodes_train_only"


def _aggregate_statistics(
    episode_reports: Mapping[str, Mapping[str, Any]], transitions: Sequence[RealTransition]
) -> dict[str, Any]:
    filtered: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    for report in episode_reports.values():
        filtered.update(report["filtered_by_reason"])
        sources.update(report["source_steps"])
    return {
        "raw_rows": sum(int(report["raw_rows"]) for report in episode_reports.values()),
        "valid_action_rows": sum(int(report["valid_action_rows"]) for report in episode_reports.values()),
        "filtered_rows": sum(int(report["filtered_rows"]) for report in episode_reports.values()),
        "filtered_by_reason": dict(sorted(filtered.items())),
        "segments": sum(int(report["segments"]) for report in episode_reports.values()),
        "source_steps": dict(sorted(sources.items())),
        "human_steps": sum(int(report["human_steps"]) for report in episode_reports.values()),
        "actor_steps": sum(int(report["actor_steps"]) for report in episode_reports.values()),
        "transitions": len(transitions),
        "episodes_with_transitions": sum(int(report["valid_action_rows"]) > 0 for report in episode_reports.values()),
        "skipped_episodes": sum(int(report["valid_action_rows"]) == 0 for report in episode_reports.values()),
        "successful_episodes": sum(float(report["terminal_reward"]) > 0 for report in episode_reports.values()),
        "failed_episodes": sum(float(report["terminal_reward"]) <= 0 for report in episode_reports.values()),
    }


if __name__ == "__main__":
    main()
