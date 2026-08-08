from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LABEL_FIELDS = [
    "image_path",
    "label",
    "episode_id",
    "t",
    "source",
    "terminal_reward",
    "gripper_value",
    "episode_root",
    "global_image",
    "wrist_image",
]


@dataclass(frozen=True)
class CandidateFrame:
    episode_id: str
    episode_root: Path
    t: int
    source: str
    terminal_reward: float | None
    global_image: str
    wrist_image: str
    label: int
    gripper_value: float | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Piper precision-phase ResNet dataset from RLT episode JSONL logs.")
    parser.add_argument(
        "--sessions",
        required=True,
        action="append",
        type=Path,
        help="RLT sessions root. May be passed multiple times.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output dataset directory.")
    parser.add_argument(
        "--image-mode",
        choices=("global", "wrist", "concat"),
        default="concat",
        help="Which image stream to use. concat writes global+wrist side-by-side.",
    )
    parser.add_argument(
        "--include-unsuccessful-human",
        action="store_true",
        help="Use human_pika rows as positives even when the terminal reward is not 1.",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=2.0,
        help="Maximum number of negative frames per positive frame. <=0 keeps all negatives.",
    )
    parser.add_argument(
        "--positive-gripper-max",
        type=float,
        help="If set, human_pika frames are positive only when the gripper value is <= this threshold.",
    )
    parser.add_argument(
        "--human-outside-positive-as-negative",
        action="store_true",
        help="When positive filters reject successful human_pika frames, keep them as negative frames.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.25, help="Episode-level validation ratio.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory before writing. Without this flag, an existing output directory fails.",
    )
    return parser


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "episode"


def _iter_episode_jsonl(session_roots: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in session_roots:
        paths.extend(sorted(root.expanduser().glob("**/episode.jsonl")))
    return sorted(set(paths))


def _parse_jsonl(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    rows.append(json.loads(text))
                except json.JSONDecodeError as exc:
                    return [], f"{path}:{line_number}: {exc}"
    except OSError as exc:
        return [], f"{path}: {exc}"
    return rows, None


def _terminal_reward(rows: list[dict[str, Any]]) -> float | None:
    for row in reversed(rows):
        if row.get("done") and row.get("reward") is not None:
            try:
                return float(row["reward"])
            except (TypeError, ValueError):
                return None
    return None


def _replay_included(row: dict[str, Any]) -> bool:
    metadata = row.get("policy_metadata")
    if isinstance(metadata, dict) and "replay_include" in metadata:
        return bool(metadata["replay_include"])
    return True


def _to_int_t(row: dict[str, Any]) -> int:
    value = row.get("t", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _last_numeric(row: dict[str, Any], key: str) -> float | None:
    values = row.get(key)
    if not isinstance(values, list) or not values:
        return None
    try:
        return float(values[-1])
    except (TypeError, ValueError):
        return None


def _gripper_value(row: dict[str, Any]) -> float | None:
    state_value = _last_numeric(row, "state")
    if state_value is not None:
        return state_value
    return _last_numeric(row, "a_exec")


def _episode_id(path: Path) -> str:
    """Return an episode id that remains unique across repeated session-local episode names."""

    return f"{path.parent.parent.name}/{path.parent.name}"


def _frame_candidates(
    path: Path,
    *,
    include_unsuccessful_human: bool,
    positive_gripper_max: float | None,
    human_outside_positive_as_negative: bool,
) -> tuple[list[CandidateFrame], str | None]:
    rows, error = _parse_jsonl(path)
    if error:
        return [], error
    if not rows:
        return [], None
    episode_root = path.parent
    episode_id = _episode_id(path)
    reward = _terminal_reward(rows)
    candidates: list[CandidateFrame] = []
    for row in rows:
        source = str(row.get("source") or "")
        if source not in {"human_pika", "pi05"}:
            continue
        if not _replay_included(row):
            continue
        global_image = row.get("global_image")
        wrist_image = row.get("wrist_image")
        if not isinstance(global_image, str) or not isinstance(wrist_image, str):
            continue
        gripper_value = _gripper_value(row)
        if source == "human_pika":
            if reward != 1.0 and not include_unsuccessful_human:
                continue
            is_positive = True
            if positive_gripper_max is not None:
                is_positive = gripper_value is not None and gripper_value <= positive_gripper_max
            if is_positive:
                label = 1
            elif human_outside_positive_as_negative:
                label = 0
            else:
                continue
        else:
            label = 0
        if not (episode_root / global_image).exists() or not (episode_root / wrist_image).exists():
            continue
        candidates.append(
            CandidateFrame(
                episode_id=episode_id,
                episode_root=episode_root,
                t=_to_int_t(row),
                source=source,
                terminal_reward=reward,
                global_image=global_image,
                wrist_image=wrist_image,
                label=label,
                gripper_value=gripper_value,
            )
        )
    return candidates, None


def _sample_negatives(candidates: list[CandidateFrame], *, negative_ratio: float, rng: random.Random) -> list[CandidateFrame]:
    positives = [item for item in candidates if item.label == 1]
    negatives = [item for item in candidates if item.label == 0]
    if negative_ratio <= 0 or not positives:
        return positives + negatives
    max_negatives = int(math.ceil(len(positives) * negative_ratio))
    if len(negatives) <= max_negatives:
        sampled_negatives = negatives
    else:
        sampled_negatives = rng.sample(negatives, max_negatives)
    return positives + sampled_negatives


def _copy_or_compose_image(candidate: CandidateFrame, *, output_path: Path, image_mode: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if image_mode != "concat":
        source_relative = candidate.global_image if image_mode == "global" else candidate.wrist_image
        shutil.copy2(candidate.episode_root / source_relative, output_path)
        return

    from PIL import Image

    global_image = Image.open(candidate.episode_root / candidate.global_image).convert("RGB")
    wrist_image = Image.open(candidate.episode_root / candidate.wrist_image).convert("RGB")
    if global_image.height != wrist_image.height:
        target_height = max(global_image.height, wrist_image.height)
        global_width = round(global_image.width * target_height / global_image.height)
        wrist_width = round(wrist_image.width * target_height / wrist_image.height)
        global_image = global_image.resize((global_width, target_height))
        wrist_image = wrist_image.resize((wrist_width, target_height))
    canvas = Image.new("RGB", (global_image.width + wrist_image.width, global_image.height))
    canvas.paste(global_image, (0, 0))
    canvas.paste(wrist_image, (global_image.width, 0))
    canvas.save(output_path, quality=95)


def _row_for_candidate(candidate: CandidateFrame, *, image_path: str) -> dict[str, str]:
    return {
        "image_path": image_path,
        "label": str(candidate.label),
        "episode_id": candidate.episode_id,
        "t": str(candidate.t),
        "source": candidate.source,
        "terminal_reward": "" if candidate.terminal_reward is None else str(candidate.terminal_reward),
        "gripper_value": "" if candidate.gripper_value is None else str(candidate.gripper_value),
        "episode_root": str(candidate.episode_root),
        "global_image": candidate.global_image,
        "wrist_image": candidate.wrist_image,
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _split_group(episode_ids: list[str], *, val_ratio: float, rng: random.Random) -> tuple[list[str], list[str]]:
    shuffled = sorted(set(episode_ids))
    rng.shuffle(shuffled)
    if len(shuffled) <= 1:
        return shuffled, []
    val_count = max(1, round(len(shuffled) * val_ratio))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def _split_episodes(rows: list[dict[str, str]], *, val_ratio: float, rng: random.Random) -> tuple[list[str], list[str]]:
    positive_episodes = sorted({row["episode_id"] for row in rows if float(row["label"]) >= 0.5})
    all_episodes = sorted({row["episode_id"] for row in rows})
    negative_only_episodes = sorted(set(all_episodes) - set(positive_episodes))

    train_positive, val_positive = _split_group(positive_episodes, val_ratio=val_ratio, rng=rng)
    train_negative, val_negative = _split_group(negative_only_episodes, val_ratio=val_ratio, rng=rng)
    train_ids = sorted(set(train_positive + train_negative))
    val_ids = sorted(set(val_positive + val_negative))

    if not val_ids and len(train_ids) > 1:
        val_ids = [train_ids.pop()]
    return train_ids, val_ids


def _rewrite_for_split(row: dict[str, str]) -> dict[str, str]:
    split_row = dict(row)
    split_row["image_path"] = f"../../{row['image_path']}"
    return split_row


def build_dataset(
    *,
    sessions: list[Path],
    output: Path,
    image_mode: str,
    include_unsuccessful_human: bool,
    negative_ratio: float,
    positive_gripper_max: float | None,
    human_outside_positive_as_negative: bool,
    val_ratio: float,
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    rng = random.Random(seed)
    output = output.expanduser()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists. Use --overwrite to replace it: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    parse_errors: list[str] = []
    all_candidates: list[CandidateFrame] = []
    for jsonl_path in _iter_episode_jsonl(sessions):
        candidates, error = _frame_candidates(
            jsonl_path,
            include_unsuccessful_human=include_unsuccessful_human,
            positive_gripper_max=positive_gripper_max,
            human_outside_positive_as_negative=human_outside_positive_as_negative,
        )
        if error:
            parse_errors.append(error)
            continue
        all_candidates.extend(candidates)

    selected = _sample_negatives(all_candidates, negative_ratio=negative_ratio, rng=rng)
    selected.sort(key=lambda item: (item.episode_id, item.t, item.source))

    rows: list[dict[str, str]] = []
    for index, candidate in enumerate(selected):
        episode_name = _safe_name(candidate.episode_id)
        image_path = f"frames/{episode_name}_{candidate.t:06d}_{index:06d}_{candidate.label}.jpg"
        _copy_or_compose_image(candidate, output_path=output / image_path, image_mode=image_mode)
        rows.append(_row_for_candidate(candidate, image_path=image_path))

    _write_csv(output / "labels.csv", rows)

    train_ids, val_ids = _split_episodes(rows, val_ratio=val_ratio, rng=rng)
    train_set = set(train_ids)
    val_set = set(val_ids)
    train_rows = [_rewrite_for_split(row) for row in rows if row["episode_id"] in train_set]
    val_rows = [_rewrite_for_split(row) for row in rows if row["episode_id"] in val_set]
    _write_csv(output / "splits" / "train" / "labels.csv", train_rows)
    _write_csv(output / "splits" / "val" / "labels.csv", val_rows)

    summary = {
        "sessions": [str(path.expanduser()) for path in sessions],
        "output": str(output),
        "image_mode": image_mode,
        "include_unsuccessful_human": include_unsuccessful_human,
        "negative_ratio": negative_ratio,
        "positive_gripper_max": positive_gripper_max,
        "human_outside_positive_as_negative": human_outside_positive_as_negative,
        "val_ratio": val_ratio,
        "seed": seed,
        "jsonl_files": len(_iter_episode_jsonl(sessions)),
        "parse_errors": parse_errors,
        "selected_frames": len(rows),
        "positive_frames": sum(row["label"] == "1" for row in rows),
        "negative_frames": sum(row["label"] == "0" for row in rows),
        "train_frames": len(train_rows),
        "val_frames": len(val_rows),
        "train_episodes": train_ids,
        "val_episodes": val_ids,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_dataset(
        sessions=args.sessions,
        output=args.output,
        image_mode=args.image_mode,
        include_unsuccessful_human=args.include_unsuccessful_human,
        negative_ratio=args.negative_ratio,
        positive_gripper_max=args.positive_gripper_max,
        human_outside_positive_as_negative=args.human_outside_positive_as_negative,
        val_ratio=args.val_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
