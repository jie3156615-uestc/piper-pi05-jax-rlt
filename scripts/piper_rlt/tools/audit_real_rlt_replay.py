#!/usr/bin/env python3
"""Audit a production Piper real-RLT replay before JAX training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _percentiles(values: np.ndarray) -> dict[str, list[float]]:
    return {
        name: np.percentile(values, percentile, axis=(0, 1)).astype(float).tolist()
        for name, percentile in (("p50", 50.0), ("p90", 90.0), ("p99", 99.0), ("max", 100.0))
    }


def audit(replay_path: Path) -> dict:
    with np.load(replay_path, allow_pickle=False) as archive:
        replay = {name: archive[name] for name in archive.files}

    required = {
        "episode_id",
        "episode_split",
        "z_rl",
        "state",
        "a_ref",
        "a_exec",
        "a_human",
        "a_actor",
        "human_mask",
        "actor_mask",
        "step_mask",
        "a_ref_absolute",
        "a_exec_absolute",
        "reward",
        "discount",
        "done",
    }
    missing = sorted(required.difference(replay))
    if missing:
        raise ValueError(f"missing replay arrays: {missing}")
    n = len(replay["reward"])
    if n == 0:
        raise ValueError("replay is empty")

    numeric_finite = {
        name: bool(np.all(np.isfinite(value)))
        for name, value in replay.items()
        if np.issubdtype(value.dtype, np.number)
    }
    state = np.asarray(replay["state"], dtype=np.float32)
    ref = np.asarray(replay["a_ref"], dtype=np.float32)
    executed = np.asarray(replay["a_exec"], dtype=np.float32)
    ref_absolute = np.asarray(replay["a_ref_absolute"], dtype=np.float32)
    exec_absolute = np.asarray(replay["a_exec_absolute"], dtype=np.float32)
    reconstructed_ref = ref_absolute.copy()
    reconstructed_exec = exec_absolute.copy()
    reconstructed_ref[..., :6] -= state[:, None, :6]
    reconstructed_exec[..., :6] -= state[:, None, :6]

    human_mask = np.asarray(replay["human_mask"], dtype=bool)
    actor_mask = np.asarray(replay["actor_mask"], dtype=bool)
    human = np.asarray(replay["a_human"], dtype=np.float32)
    actor = np.asarray(replay["a_actor"], dtype=np.float32)
    episode_ids = replay["episode_id"].astype(str)
    splits = replay["episode_split"].astype(str)
    rewards = np.asarray(replay["reward"], dtype=np.float32)
    successful_episodes = {
        episode_id for episode_id in np.unique(episode_ids) if np.any(rewards[episode_ids == episode_id] > 0)
    }

    split_report = {}
    for split in ("train", "validation", "test"):
        mask = splits == split
        split_episodes = sorted(set(episode_ids[mask].tolist()))
        split_report[split] = {
            "transitions": int(np.sum(mask)),
            "episodes": len(split_episodes),
            "successful_episodes": sum(episode_id in successful_episodes for episode_id in split_episodes),
            "failed_episodes": sum(episode_id not in successful_episodes for episode_id in split_episodes),
            "human_transitions": int(np.sum(np.any(human_mask[mask], axis=1))) if np.any(mask) else 0,
        }

    residual = np.abs(executed - ref)
    report = {
        "format": "openpi_real_rlt_replay_audit",
        "replay": str(replay_path.resolve()),
        "transitions": n,
        "episodes": int(len(set(episode_ids.tolist()))),
        "z_dim": int(replay["z_rl"].shape[-1]),
        "all_numeric_finite": bool(all(numeric_finite.values())),
        "nonfinite_arrays": sorted(name for name, finite in numeric_finite.items() if not finite),
        "coordinate_max_abs_error": {
            "a_ref": float(np.max(np.abs(ref - reconstructed_ref))),
            "a_exec": float(np.max(np.abs(executed - reconstructed_exec))),
        },
        "optional_action_contract": {
            "human_mask_steps": int(np.sum(human_mask)),
            "actor_mask_steps": int(np.sum(actor_mask)),
            "missing_human_max_abs": float(np.max(np.abs(human[~human_mask]))) if np.any(~human_mask) else 0.0,
            "missing_actor_max_abs": float(np.max(np.abs(actor[~actor_mask]))) if np.any(~actor_mask) else 0.0,
        },
        "split": split_report,
        "terminal": {
            "done_transitions": int(np.sum(replay["done"])),
            "positive_reward_transitions": int(np.sum(rewards > 0)),
            "reward_min": float(np.min(rewards)),
            "reward_max": float(np.max(rewards)),
        },
        "executed_minus_reference_abs": _percentiles(residual),
        "z_rl": {
            "global_mean": float(np.mean(replay["z_rl"])),
            "global_std": float(np.std(replay["z_rl"])),
            "mean_feature_std": float(np.mean(np.std(replay["z_rl"], axis=0))),
        },
    }
    report["passed"] = bool(
        report["all_numeric_finite"]
        and report["z_dim"] == 2048
        and report["coordinate_max_abs_error"]["a_ref"] < 1e-6
        and report["coordinate_max_abs_error"]["a_exec"] < 1e-6
        and report["optional_action_contract"]["missing_human_max_abs"] == 0.0
        and report["optional_action_contract"]["missing_actor_max_abs"] == 0.0
        and split_report["train"]["episodes"] > 0
        and split_report["validation"]["episodes"] > 0
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    report = audit(args.replay_npz)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
