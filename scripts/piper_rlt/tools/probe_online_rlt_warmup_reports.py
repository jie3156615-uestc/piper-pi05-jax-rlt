#!/usr/bin/env python3
"""Cheap report-only upper-bound probe for clean online RLT warmup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def probe(session_root: Path) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    excluded: list[dict[str, str]] = []
    for report_path in sorted(session_root.glob("episode_*/report.json")):
        episode_id = report_path.parent.name
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            excluded.append({"episode_id": episode_id, "reason": f"invalid report: {exc}"})
            continue
        reward = report.get("terminal_reward")
        if report.get("outcome") != "episode_done" or reward not in (0, 0.0, 1, 1.0):
            excluded.append({"episode_id": episode_id, "reason": "not a completed rewarded episode"})
            continue
        if bool(report.get("exclude_from_training", False)):
            excluded.append(
                {"episode_id": episode_id, "reason": str(report.get("exclusion_reason", "excluded"))}
            )
            continue
        if not (report_path.parent / "episode.jsonl").is_file():
            excluded.append({"episode_id": episode_id, "reason": "missing episode.jsonl"})
            continue
        candidates.append({"episode_id": episode_id, "reward": float(reward)})
    return {
        "format": "openpi_piper_warmup_candidate_progress_v1",
        "candidate_episodes": len(candidates),
        "candidate_successes": sum(item["reward"] == 1.0 for item in candidates),
        "candidate_failures": sum(item["reward"] == 0.0 for item in candidates),
        "candidate_episode_ids": [item["episode_id"] for item in candidates],
        "report_exclusions": excluded,
        "note": "report-only prefilter; strict trainable count is produced by run_online_rlt_update.py",
    }


def main() -> None:
    args = parse_args()
    payload = probe(args.session_root.expanduser().resolve())
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


if __name__ == "__main__":
    main()
