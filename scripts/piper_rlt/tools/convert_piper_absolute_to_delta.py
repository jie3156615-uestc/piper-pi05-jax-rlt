from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ACTION_NAMES = [f"delta_joint_{index}" for index in range(1, 7)] + ["gripper"]


def convert_joint_actions(state: np.ndarray, absolute_action: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    absolute_action = np.asarray(absolute_action, dtype=np.float32)
    if state.shape != absolute_action.shape:
        raise ValueError(f"state and action shapes must match: {state.shape} != {absolute_action.shape}")
    if state.ndim != 2 or state.shape[1] != 7:
        raise ValueError(f"expected [frames, 7] state and action arrays, got {state.shape}")

    converted = absolute_action.copy()
    converted[:, :6] = absolute_action[:, :6] - state[:, :6]
    return converted


def build_feedback_delta(state: np.ndarray, episode_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(state, dtype=np.float32)
    episode_index = np.asarray(episode_index)
    if state.ndim != 2:
        raise ValueError(f"expected a 2-D state array, got {state.shape}")
    if len(state) != len(episode_index):
        raise ValueError(f"state and episode lengths must match: {len(state)} != {len(episode_index)}")

    delta = np.full_like(state, np.nan, dtype=np.float32)
    valid = np.zeros(len(state), dtype=bool)
    if len(state) > 1:
        valid[:-1] = episode_index[:-1] == episode_index[1:]
        delta[:-1][valid[:-1]] = state[1:][valid[:-1]] - state[:-1][valid[:-1]]
    return delta, valid


def compute_stats(values: np.ndarray) -> dict[str, list[Any]]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError(f"expected a non-empty 2-D array, got {values.shape}")
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(len(values))],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_tree_user_writable(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        path.chmod(path.stat().st_mode | stat.S_IWUSR)


def _remove_tree(root: Path) -> None:
    if root.exists():
        _make_tree_user_writable(root)
        shutil.rmtree(root)


def _data_paths(root: Path) -> list[Path]:
    return sorted((root / "data").rglob("*.parquet"))


def _read_arrays(data_paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episodes: list[np.ndarray] = []
    for path in data_paths:
        table = pq.read_table(path, columns=["observation.state", "action", "episode_index"])
        states.append(np.asarray(table["observation.state"].to_pylist(), dtype=np.float32))
        actions.append(np.asarray(table["action"].to_pylist(), dtype=np.float32))
        episodes.append(np.asarray(table["episode_index"].to_pylist(), dtype=np.int64))
    if not states:
        raise ValueError("dataset has no data parquet files")
    return np.concatenate(states), np.concatenate(actions), np.concatenate(episodes)


def _replace_actions(path: Path) -> None:
    table = pq.read_table(path)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    absolute_action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    converted = convert_joint_actions(state, absolute_action)
    action_index = table.schema.get_field_index("action")
    action_array = pa.array(converted.tolist(), type=table.schema.field(action_index).type)
    table = table.set_column(action_index, "action", action_array)
    pq.write_table(table, path)


def _rewrite_episode_stats(root: Path, actions: np.ndarray, episodes: np.ndarray) -> None:
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        if "stats" not in table.column_names:
            continue
        rows = table.to_pylist()
        changed = False
        for row in rows:
            stats = row.get("stats")
            if not isinstance(stats, dict) or "action" not in stats:
                continue
            mask = episodes == int(row["episode_index"])
            stats["action"] = compute_stats(actions[mask])
            changed = True
        if changed:
            pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)


def _write_info(root: Path) -> None:
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["action"]["names"] = ACTION_NAMES
    info_path.write_text(json.dumps(info, indent=4, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_stats(root: Path, actions: np.ndarray) -> None:
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    stats["action"] = compute_stats(actions)
    stats_path.write_text(json.dumps(stats, indent=4, ensure_ascii=True) + "\n", encoding="utf-8")


def _summary(values: np.ndarray) -> dict[str, Any]:
    stats = compute_stats(values)
    return {key: value for key, value in stats.items() if key in {"min", "max", "mean", "std", "q01", "q50", "q99"}}


def verify_dataset(source: Path, output: Path) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    source_paths = _data_paths(source)
    output_paths = _data_paths(output)
    if not source_paths or not output_paths:
        raise ValueError("source and output datasets must contain data parquet files")
    if [path.relative_to(source) for path in source_paths] != [path.relative_to(output) for path in output_paths]:
        raise ValueError("source and output parquet layouts differ")

    state, absolute_action, source_episodes = _read_arrays(source_paths)
    output_state, converted_action, output_episodes = _read_arrays(output_paths)
    if not np.array_equal(state, output_state):
        raise ValueError("observation.state changed during conversion")
    if not np.array_equal(source_episodes, output_episodes):
        raise ValueError("episode_index changed during conversion")

    info = json.loads((output / "meta" / "info.json").read_text(encoding="utf-8"))
    if info["features"]["action"]["names"] != ACTION_NAMES:
        raise ValueError("output action feature names do not describe joint deltas")
    reconstruction_error = np.abs(state[:, :6] + converted_action[:, :6] - absolute_action[:, :6])
    gripper_error = np.abs(converted_action[:, 6] - absolute_action[:, 6])
    if reconstruction_error.max() > 1e-6:
        raise ValueError(f"joint reconstruction error is too large: {reconstruction_error.max()}")
    if gripper_error.max() > 1e-6:
        raise ValueError(f"gripper values changed during conversion: {gripper_error.max()}")

    output_stats = json.loads((output / "meta" / "stats.json").read_text(encoding="utf-8"))["action"]
    expected_stats = compute_stats(converted_action)
    for key in ["min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"]:
        if not np.allclose(output_stats[key], expected_stats[key], atol=1e-8):
            raise ValueError(f"output action stats mismatch for {key}")

    source_videos = sorted(path.relative_to(source) for path in (source / "videos").rglob("*.mp4"))
    output_videos = sorted(path.relative_to(output) for path in (output / "videos").rglob("*.mp4"))
    if source_videos != output_videos:
        raise ValueError("source and output video layouts differ")
    source_hashes = {str(path.relative_to(source)): _sha256(path) for path in source_paths}
    report = json.loads((output / "meta" / "delta_conversion.json").read_text(encoding="utf-8"))
    if report["source_data_sha256"] != source_hashes:
        raise ValueError("source parquet hashes do not match conversion manifest")

    return {
        "ok": True,
        "source": str(source),
        "output": str(output),
        "frames": int(len(state)),
        "episodes": int(len(np.unique(source_episodes))),
        "videos": len(source_videos),
        "reconstruction_mean_abs_error": float(reconstruction_error.mean()),
        "reconstruction_max_abs_error": float(reconstruction_error.max()),
        "gripper_max_abs_error": float(gripper_error.max()),
        "source_data_sha256": source_hashes,
    }


def convert_dataset(source: Path, output: Path) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source dataset does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    source_data_paths = _data_paths(source)
    source_hashes_before = {str(path.relative_to(source)): _sha256(path) for path in source_data_paths}
    state, absolute_action, episodes = _read_arrays(source_data_paths)
    converted_action = convert_joint_actions(state, absolute_action)
    feedback_delta, feedback_valid = build_feedback_delta(state, episodes)
    feedback_joint_delta = feedback_delta[feedback_valid, :6]
    target_joint_delta = converted_action[:, :6]
    reconstruction = state[:, :6] + target_joint_delta
    tracking_error = target_joint_delta[:-1][feedback_valid[:-1]] - feedback_joint_delta

    staging = output.parent / f".{output.name}.tmp-{os.getpid()}"
    if staging.exists():
        _remove_tree(staging)
    try:
        shutil.copytree(source, staging)
        _make_tree_user_writable(staging)
        for path in _data_paths(staging):
            _replace_actions(path)
        _write_info(staging)
        _write_stats(staging, converted_action)
        _rewrite_episode_stats(staging, converted_action, episodes)

        report: dict[str, Any] = {
            "version": 1,
            "source": str(source),
            "output": str(output),
            "frames": int(len(state)),
            "episodes": int(len(np.unique(episodes))),
            "action_semantics": {
                "joints": "target_delta_q[t] = source_absolute_action[t] - observation.state[t]",
                "gripper": "absolute source gripper target preserved unchanged",
                "feedback_audit": "feedback_delta_q[t] = observation.state[t+1] - observation.state[t], within episode only",
            },
            "episode_boundary_audit": {
                "feedback_delta_valid_frames": int(feedback_valid.sum()),
                "feedback_delta_skipped_episode_final_frames": int((~feedback_valid).sum()),
                "cross_episode_differences_used": 0,
            },
            "reconstruction": {
                "mean_abs_error": float(np.abs(reconstruction - absolute_action[:, :6]).mean()),
                "max_abs_error": float(np.abs(reconstruction - absolute_action[:, :6]).max()),
            },
            "target_joint_delta": _summary(target_joint_delta),
            "feedback_joint_delta": _summary(feedback_joint_delta),
            "target_minus_feedback_joint_delta": _summary(tracking_error),
            "source_data_sha256": source_hashes_before,
        }
        (staging / "meta" / "delta_conversion.json").write_text(
            json.dumps(report, indent=4, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        source_hashes_after = {str(path.relative_to(source)): _sha256(path) for path in source_data_paths}
        if source_hashes_before != source_hashes_after:
            raise RuntimeError("source parquet files changed during conversion")
        os.replace(staging, output)
        return report
    except Exception:
        _remove_tree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Piper absolute joint action labels to joint deltas.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    report = verify_dataset(args.source, args.output) if args.verify_only else convert_dataset(args.source, args.output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=4, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
