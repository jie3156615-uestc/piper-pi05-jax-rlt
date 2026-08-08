#!/usr/bin/env python3
"""Audit one episode before persistent-v2 admission or online learning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from . import persistent_v2_contract as _contract
except ImportError:  # Direct executable invocation.
    import persistent_v2_contract as _contract

ACTION_SCHEMA_FINGERPRINT = _contract.ACTION_SCHEMA_FINGERPRINT
ACTOR_EXECUTION_PROFILE = _contract.ACTOR_EXECUTION_PROFILE
ACTOR_PROJECTION_PROFILE = _contract.ACTOR_PROJECTION_PROFILE
CONTROL_HZ = _contract.CONTROL_HZ
DEFAULT_MIN_PHASE_RLT_PUBLISHED_ROWS = (
    _contract.DEFAULT_MIN_PHASE_RLT_PUBLISHED_ROWS
)
DEFAULT_MIN_PHYSICAL_COMMITTED_ROWS = (
    _contract.DEFAULT_MIN_PHYSICAL_COMMITTED_ROWS
)
EXECUTION_FILTER_PROFILE = _contract.EXECUTION_FILTER_PROFILE
EXECUTION_FILTER_TAU_S = _contract.EXECUTION_FILTER_TAU_S
STRICT_ACTOR_CANARY = _contract.STRICT_ACTOR_CANARY
TRAINING_EPISODE_CONTRACT = _contract.TRAINING_EPISODE_CONTRACT
audit_persistent_v2_episode = _contract.audit_persistent_v2_episode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--mode",
        choices=[STRICT_ACTOR_CANARY, TRAINING_EPISODE_CONTRACT],
        default=TRAINING_EPISODE_CONTRACT,
    )
    parser.add_argument(
        "--actor-execution-profile", default=ACTOR_EXECUTION_PROFILE
    )
    parser.add_argument(
        "--action-schema-fingerprint", default=ACTION_SCHEMA_FINGERPRINT
    )
    parser.add_argument(
        "--actor-projection-profile", default=ACTOR_PROJECTION_PROFILE
    )
    parser.add_argument(
        "--execution-filter-profile", default=EXECUTION_FILTER_PROFILE
    )
    parser.add_argument(
        "--execution-filter-tau-s",
        type=float,
        default=EXECUTION_FILTER_TAU_S,
    )
    parser.add_argument("--control-hz", type=float, default=CONTROL_HZ)
    parser.add_argument(
        "--min-phase-rlt-published-rows",
        type=int,
        default=DEFAULT_MIN_PHASE_RLT_PUBLISHED_ROWS,
    )
    parser.add_argument(
        "--min-physical-committed-rows",
        type=int,
        default=DEFAULT_MIN_PHYSICAL_COMMITTED_ROWS,
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = audit_persistent_v2_episode(
        args.episode_jsonl.expanduser().resolve(),
        expected_execution_profile=args.actor_execution_profile,
        expected_action_schema=args.action_schema_fingerprint,
        expected_projection_profile=args.actor_projection_profile,
        expected_filter_profile=args.execution_filter_profile,
        expected_filter_tau_s=args.execution_filter_tau_s,
        expected_control_hz=args.control_hz,
        mode=args.mode,
        min_phase_rlt_published_rows=args.min_phase_rlt_published_rows,
        min_physical_committed_rows=args.min_physical_committed_rows,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output_json is not None:
        output = args.output_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(payload + "\n", encoding="utf-8")
        temporary.replace(output)
    print(payload)


if __name__ == "__main__":
    main()
