#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from piper_runtime.rlt_phase_gate import PhaseGateConfig
from piper_runtime.rlt_phase_gate import SingleLatchPhaseGate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeline-csv", type=Path, required=True)
    parser.add_argument("--enter-threshold", type=float, default=0.5)
    parser.add_argument("--enter-frames", type=int, default=3)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.timeline_csv.open(newline="")))
    by_episode: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_episode[row["episode_id"]].append(row)

    violations: list[tuple[str, int]] = []
    print(f"episodes {len(by_episode)}")
    for episode_id, episode_rows in sorted(by_episode.items()):
        gate = SingleLatchPhaseGate(
            PhaseGateConfig(
                enter_threshold=args.enter_threshold,
                enter_consecutive_frames=args.enter_frames,
            )
        )
        entries = 0
        previous_active = False
        active_after_low = False
        for row in sorted(episode_rows, key=lambda item: int(item["t"])):
            probability = float(row["probability"])
            snapshot = gate.update(probability, t=int(row["t"]))
            if snapshot.active and not previous_active:
                entries += 1
            if snapshot.active and probability < args.enter_threshold:
                active_after_low = True
            previous_active = snapshot.active
        snapshot = gate.snapshot()
        if entries > 1:
            violations.append((episode_id, entries))
        print(
            {
                "episode_id": episode_id,
                "entries": entries,
                "enter_t": snapshot.enter_t,
                "final_state": snapshot.state,
                "active_after_low": active_after_low,
            }
        )

    print(f"violations {violations}")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
