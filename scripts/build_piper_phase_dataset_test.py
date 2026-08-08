from __future__ import annotations

import csv
import importlib.util
import json
import random
import sys
from pathlib import Path

from PIL import Image


def _load_script(path: str):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), color=color).save(path)


def _write_episode(
    root: Path,
    episode_name: str,
    *,
    terminal_reward: float,
    human_frames: int = 2,
    session_name: str = "session_a",
    row_episode_id: str | None = None,
) -> Path:
    episode_root = root / session_name / episode_name
    rows = []
    for t in range(6):
        _write_image(episode_root / "camera_global" / f"{t:06d}.jpg", (t * 20, 10, 10))
        _write_image(episode_root / "camera_wrist" / f"{t:06d}.jpg", (10, t * 20, 10))
        if 2 <= t < 2 + human_frames:
            source = "human_pika"
            replay_include = True
        else:
            source = "pi05"
            replay_include = True
        rows.append(
            {
                "episode_id": episode_name,
                "t": t,
                "global_image": f"camera_global/{t:06d}.jpg",
                "wrist_image": f"camera_wrist/{t:06d}.jpg",
                "source": source,
                "state": [0, 0, 0, 0, 0, 0, 0.02 if t == 3 else 0.066],
                "a_exec": [0, 0, 0, 0, 0, 0, 0.02 if t == 3 else 0.066],
                "policy_metadata": {"replay_include": replay_include},
                "done": False,
                "reward": None,
            }
        )
        if row_episode_id is not None:
            rows[-1]["episode_id"] = row_episode_id
    rows.append(
        {
            "episode_id": episode_name,
            "t": 6,
            "source": "stop",
            "done": True,
            "reward": terminal_reward,
        }
    )
    episode_root.mkdir(parents=True, exist_ok=True)
    with (episode_root / "episode.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return episode_root


def _read_labels(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_build_phase_dataset_creates_concat_images_labels_and_episode_splits(tmp_path: Path) -> None:
    module = _load_script("scripts/build_piper_phase_dataset.py")
    sessions = tmp_path / "sessions"
    output = tmp_path / "dataset"
    _write_episode(sessions, "episode_success", terminal_reward=1.0, human_frames=2)
    _write_episode(sessions, "episode_failure", terminal_reward=0.0, human_frames=2)

    exit_code = module.main(
        [
            "--sessions",
            str(sessions),
            "--output",
            str(output),
            "--image-mode",
            "concat",
            "--negative-ratio",
            "1.0",
            "--val-ratio",
            "0.5",
            "--seed",
            "3",
        ]
    )

    assert exit_code == 0
    rows = _read_labels(output / "labels.csv")
    assert {row["label"] for row in rows} == {"0", "1"}
    assert sum(row["label"] == "1" for row in rows) == 2
    assert all((output / row["image_path"]).exists() for row in rows)

    sample = Image.open(output / rows[0]["image_path"])
    assert sample.size == (32, 12)

    train_rows = _read_labels(output / "splits" / "train" / "labels.csv")
    val_rows = _read_labels(output / "splits" / "val" / "labels.csv")
    assert train_rows
    assert val_rows
    assert {row["episode_id"] for row in train_rows}.isdisjoint({row["episode_id"] for row in val_rows})

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["positive_frames"] == 2
    assert summary["negative_frames"] == 2
    assert summary["image_mode"] == "concat"


def test_build_phase_dataset_can_use_unsuccessful_human_when_explicitly_enabled(tmp_path: Path) -> None:
    module = _load_script("scripts/build_piper_phase_dataset.py")
    sessions = tmp_path / "sessions"
    output = tmp_path / "dataset"
    _write_episode(sessions, "episode_failure", terminal_reward=0.0, human_frames=3)

    module.main(
        [
            "--sessions",
            str(sessions),
            "--output",
            str(output),
            "--include-unsuccessful-human",
            "--negative-ratio",
            "1.0",
        ]
    )

    rows = _read_labels(output / "labels.csv")
    assert sum(row["label"] == "1" for row in rows) == 3


def test_build_phase_dataset_uses_session_relative_episode_ids_when_json_ids_collide(tmp_path: Path) -> None:
    module = _load_script("scripts/build_piper_phase_dataset.py")
    sessions = tmp_path / "sessions"
    output = tmp_path / "dataset"
    _write_episode(
        sessions,
        "episode_000000",
        session_name="session_a",
        row_episode_id="episode_000000",
        terminal_reward=1.0,
    )
    _write_episode(
        sessions,
        "episode_000000",
        session_name="session_b",
        row_episode_id="episode_000000",
        terminal_reward=1.0,
    )

    module.main(
        [
            "--sessions",
            str(sessions),
            "--output",
            str(output),
            "--negative-ratio",
            "1.0",
        ]
    )

    rows = _read_labels(output / "labels.csv")
    episode_ids = {row["episode_id"] for row in rows}
    assert episode_ids == {"session_a/episode_000000", "session_b/episode_000000"}


def test_build_phase_dataset_splits_positive_episodes_into_train_and_val(tmp_path: Path) -> None:
    module = _load_script("scripts/build_piper_phase_dataset.py")
    sessions = tmp_path / "sessions"
    output = tmp_path / "dataset"
    for index in range(4):
        _write_episode(
            sessions,
            f"episode_success_{index}",
            session_name=f"session_success_{index}",
            terminal_reward=1.0,
        )
    for index in range(4):
        _write_episode(
            sessions,
            f"episode_failure_{index}",
            session_name=f"session_failure_{index}",
            terminal_reward=0.0,
        )

    module.main(
        [
            "--sessions",
            str(sessions),
            "--output",
            str(output),
            "--negative-ratio",
            "1.0",
            "--val-ratio",
            "0.25",
            "--seed",
            "10",
        ]
    )

    train_rows = _read_labels(output / "splits" / "train" / "labels.csv")
    val_rows = _read_labels(output / "splits" / "val" / "labels.csv")
    assert any(row["label"] == "1" for row in train_rows)
    assert any(row["label"] == "1" for row in val_rows)


def test_split_episodes_is_stratified_when_multiple_positive_episodes_exist() -> None:
    module = _load_script("scripts/build_piper_phase_dataset.py")
    rows = []
    for index in range(4):
        rows.append({"episode_id": f"failure_{index}", "label": "0"})
    for index in range(4):
        rows.append({"episode_id": f"success_{index}", "label": "1"})

    train_ids, val_ids = module._split_episodes(rows, val_ratio=0.25, rng=random.Random(10))

    assert any(episode_id.startswith("success_") for episode_id in train_ids)
    assert any(episode_id.startswith("success_") for episode_id in val_ids)


def test_build_phase_dataset_can_restrict_positive_human_frames_by_gripper_value(tmp_path: Path) -> None:
    module = _load_script("scripts/build_piper_phase_dataset.py")
    sessions = tmp_path / "sessions"
    output = tmp_path / "dataset"
    _write_episode(sessions, "episode_success", terminal_reward=1.0, human_frames=3)

    module.main(
        [
            "--sessions",
            str(sessions),
            "--output",
            str(output),
            "--positive-gripper-max",
            "0.04",
            "--human-outside-positive-as-negative",
            "--negative-ratio",
            "0",
        ]
    )

    rows = _read_labels(output / "labels.csv")
    human_rows = [row for row in rows if row["source"] == "human_pika"]
    assert sum(row["label"] == "1" for row in human_rows) == 1
    assert sum(row["label"] == "0" for row in human_rows) == 2
