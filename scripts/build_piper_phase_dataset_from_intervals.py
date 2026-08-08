from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import random
import re
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


LABEL_FIELDS = [
    "image_path",
    "label",
    "episode_id",
    "t",
    "source_type",
    "episode_path",
    "phase_start_t",
    "phase_end_t",
]


@dataclass(frozen=True)
class FrameCandidate:
    episode_id: str
    episode_path: Path
    source_type: str
    t: int
    label: int
    phase_start_t: int
    phase_end_t: int
    row_index: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Piper phase classifier dataset from manually annotated intervals.")
    parser.add_argument("--annotations", required=True, type=Path, help="CSV with episode_path,phase_start_t,phase_end_t.")
    parser.add_argument("--output", required=True, type=Path, help="Output dataset directory.")
    parser.add_argument(
        "--lerobot-root",
        default=Path("/home/cwzk/lerobot_datasets/piper_takeplaceredcup_5090_v2"),
        type=Path,
        help="LeRobot dataset root containing concatenated camera videos for OpenPI parquet episodes.",
    )
    parser.add_argument("--frame-stride", type=int, default=2, help="Keep one frame every N frames before sampling negatives.")
    parser.add_argument("--negative-ratio", type=float, default=2.0, help="Max negatives per positive. <=0 keeps all negatives.")
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--skip-invalid", action="store_true", help="Skip invalid annotation rows and record them in summary.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "episode"


def parquet_episode_index(path: Path) -> int:
    match = re.search(r"episode_(\d+)\.parquet$", path.name)
    if not match:
        raise ValueError(f"cannot parse episode index: {path}")
    return int(match.group(1))


def episode_id_for_path(path: Path) -> str:
    if path.suffix == ".parquet":
        return f"openpi/{path.stem}"
    if path.suffix == ".jsonl":
        return f"{path.parent.parent.name}/{path.parent.name}"
    return safe_name(str(path))


def label_for_t(t: int, start_t: int, end_t: int) -> int:
    if start_t == -1 and end_t == -1:
        return 0
    return int(start_t <= t <= end_t)


def _read_parquet_frame_indices(path: Path) -> list[int]:
    import polars as pl

    df = pl.read_parquet(path, columns=["frame_index"])
    return [int(value) for value in df["frame_index"].to_list()]


def _read_rlt_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row.get("global_image"), str) and isinstance(row.get("wrist_image"), str):
                rows.append(row)
    rows.sort(key=lambda row: int(row.get("t", 0)))
    return rows


def validate_and_collect_candidates(
    *,
    annotations: Path,
    frame_stride: int,
    skip_invalid: bool,
) -> tuple[list[FrameCandidate], list[dict[str, Any]]]:
    candidates: list[FrameCandidate] = []
    skipped: list[dict[str, Any]] = []
    with annotations.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row_index, row in enumerate(rows, start=1):
        episode_path = Path(row["episode_path"])
        raw_start = str(row.get("phase_start_t", "")).strip()
        raw_end = str(row.get("phase_end_t", "")).strip()
        try:
            start_t = int(raw_start)
            end_t = int(raw_end)
        except (TypeError, ValueError):
            error = {"row_index": row_index, "episode_path": str(episode_path), "reason": "bad integer", "start": raw_start, "end": raw_end}
            if skip_invalid:
                skipped.append(error)
                continue
            raise ValueError(error)
        if not episode_path.exists():
            error = {"row_index": row_index, "episode_path": str(episode_path), "reason": "episode path missing"}
            if skip_invalid:
                skipped.append(error)
                continue
            raise FileNotFoundError(error)
        if episode_path.suffix == ".parquet":
            frame_indices = _read_parquet_frame_indices(episode_path)
            source_type = "openpi_parquet"
        elif episode_path.suffix == ".jsonl":
            rlt_rows = _read_rlt_rows(episode_path)
            frame_indices = [int(item.get("t", 0)) for item in rlt_rows]
            source_type = "rlt_jsonl"
        else:
            error = {"row_index": row_index, "episode_path": str(episode_path), "reason": "unsupported suffix"}
            if skip_invalid:
                skipped.append(error)
                continue
            raise ValueError(error)
        if not frame_indices:
            error = {"row_index": row_index, "episode_path": str(episode_path), "reason": "no frames"}
            if skip_invalid:
                skipped.append(error)
                continue
            raise ValueError(error)
        min_t = min(frame_indices)
        max_t = max(frame_indices)
        if (start_t, end_t) != (-1, -1) and (start_t < min_t or end_t > max_t or start_t > end_t):
            error = {
                "row_index": row_index,
                "episode_path": str(episode_path),
                "reason": "range outside episode",
                "start": start_t,
                "end": end_t,
                "min_t": min_t,
                "max_t": max_t,
            }
            if skip_invalid:
                skipped.append(error)
                continue
            raise ValueError(error)
        episode_id = episode_id_for_path(episode_path)
        for t in frame_indices:
            if t % frame_stride != 0:
                continue
            candidates.append(
                FrameCandidate(
                    episode_id=episode_id,
                    episode_path=episode_path,
                    source_type=source_type,
                    t=t,
                    label=label_for_t(t, start_t, end_t),
                    phase_start_t=start_t,
                    phase_end_t=end_t,
                    row_index=row_index,
                )
            )
    return candidates, skipped


def sample_candidates(candidates: list[FrameCandidate], *, negative_ratio: float, rng: random.Random) -> list[FrameCandidate]:
    positives = [item for item in candidates if item.label == 1]
    negatives = [item for item in candidates if item.label == 0]
    if negative_ratio <= 0 or not positives:
        sampled_negatives = negatives
    else:
        max_negatives = int(math.ceil(len(positives) * negative_ratio))
        sampled_negatives = negatives if len(negatives) <= max_negatives else rng.sample(negatives, max_negatives)
    selected = positives + sampled_negatives
    selected.sort(key=lambda item: (item.episode_id, item.t, item.label))
    return selected


def split_episodes(rows: list[dict[str, str]], *, val_ratio: float, rng: random.Random) -> tuple[list[str], list[str]]:
    positive_eps = sorted({row["episode_id"] for row in rows if row["label"] == "1"})
    all_eps = sorted({row["episode_id"] for row in rows})
    negative_only_eps = sorted(set(all_eps) - set(positive_eps))

    def split_group(items: list[str]) -> tuple[list[str], list[str]]:
        items = list(items)
        rng.shuffle(items)
        if len(items) <= 1:
            return items, []
        val_count = max(1, round(len(items) * val_ratio))
        val_count = min(val_count, len(items) - 1)
        return items[val_count:], items[:val_count]

    train_pos, val_pos = split_group(positive_eps)
    train_neg, val_neg = split_group(negative_only_eps)
    return sorted(set(train_pos + train_neg)), sorted(set(val_pos + val_neg))


def compose_image(left: Image.Image, right: Image.Image) -> Image.Image:
    left = left.convert("RGB").resize((320, 240))
    right = right.convert("RGB").resize((320, 240))
    canvas = Image.new("RGB", (640, 240), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (320, 0))
    return canvas


def pil_from_av_frame(frame) -> Image.Image:
    return frame.to_image().convert("RGB")


def save_parquet_images(
    *,
    selected: list[FrameCandidate],
    output: Path,
    lerobot_root: Path,
    image_paths: dict[FrameCandidate, str],
) -> None:
    import av
    import polars as pl

    video1 = lerobot_root / "videos/observation.images.camera1/chunk-000/file-000.mp4"
    video2 = lerobot_root / "videos/observation.images.camera2/chunk-000/file-000.mp4"
    by_episode: dict[Path, list[FrameCandidate]] = {}
    for item in selected:
        if item.source_type == "openpi_parquet":
            by_episode.setdefault(item.episode_path, []).append(item)
    for episode_path, items in by_episode.items():
        wanted = {item.t: item for item in items}
        df = pl.read_parquet(episode_path, columns=["frame_index", "index"])
        frame_indices = [int(value) for value in df["frame_index"].to_list()]
        global_indices = [int(value) for value in df["index"].to_list()]
        container1 = av.open(str(video1))
        container2 = av.open(str(video2))
        stream1 = container1.streams.video[0]
        stream2 = container2.streams.video[0]
        start_time = float(global_indices[0]) / float(stream1.average_rate)
        container1.seek(int(start_time / float(stream1.time_base)), stream=stream1, any_frame=False, backward=True)
        container2.seek(int(start_time / float(stream2.time_base)), stream=stream2, any_frame=False, backward=True)
        frames1 = container1.decode(stream1)
        frames2 = container2.decode(stream2)
        for local_i, (frame1, frame2) in enumerate(zip(frames1, frames2)):
            if local_i >= len(frame_indices):
                break
            t = frame_indices[local_i]
            item = wanted.get(t)
            if item is None:
                continue
            image_path = output / image_paths[item]
            image_path.parent.mkdir(parents=True, exist_ok=True)
            compose_image(pil_from_av_frame(frame1), pil_from_av_frame(frame2)).save(image_path, quality=92)
        container1.close()
        container2.close()


def save_rlt_images(*, selected: list[FrameCandidate], output: Path, image_paths: dict[FrameCandidate, str]) -> None:
    by_episode: dict[Path, list[FrameCandidate]] = {}
    for item in selected:
        if item.source_type == "rlt_jsonl":
            by_episode.setdefault(item.episode_path, []).append(item)
    for episode_path, items in by_episode.items():
        wanted = {item.t: item for item in items}
        for row in _read_rlt_rows(episode_path):
            t = int(row.get("t", 0))
            item = wanted.get(t)
            if item is None:
                continue
            left_path = episode_path.parent / row["global_image"]
            right_path = episode_path.parent / row["wrist_image"]
            if not left_path.exists() or not right_path.exists():
                continue
            image_path = output / image_paths[item]
            image_path.parent.mkdir(parents=True, exist_ok=True)
            compose_image(Image.open(left_path), Image.open(right_path)).save(image_path, quality=92)


def row_for_candidate(item: FrameCandidate, image_path: str) -> dict[str, str]:
    return {
        "image_path": image_path,
        "label": str(item.label),
        "episode_id": item.episode_id,
        "t": str(item.t),
        "source_type": item.source_type,
        "episode_path": str(item.episode_path),
        "phase_start_t": str(item.phase_start_t),
        "phase_end_t": str(item.phase_end_t),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def split_row(row: dict[str, str]) -> dict[str, str]:
    copied = dict(row)
    copied["image_path"] = f"../../{row['image_path']}"
    return copied


def build_dataset(
    *,
    annotations: Path,
    output: Path,
    lerobot_root: Path,
    frame_stride: int,
    negative_ratio: float,
    val_ratio: float,
    seed: int,
    skip_invalid: bool,
    overwrite: bool,
) -> dict[str, Any]:
    rng = random.Random(seed)
    output = output.expanduser()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists. Use --overwrite: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    candidates, skipped = validate_and_collect_candidates(annotations=annotations, frame_stride=frame_stride, skip_invalid=skip_invalid)
    selected = sample_candidates(candidates, negative_ratio=negative_ratio, rng=rng)
    image_paths: dict[FrameCandidate, str] = {}
    for idx, item in enumerate(selected):
        image_paths[item] = f"frames/{safe_name(item.episode_id)}_{item.t:06d}_{idx:06d}_{item.label}.jpg"
    save_parquet_images(selected=selected, output=output, lerobot_root=lerobot_root, image_paths=image_paths)
    save_rlt_images(selected=selected, output=output, image_paths=image_paths)
    rows = [row_for_candidate(item, image_paths[item]) for item in selected if (output / image_paths[item]).exists()]
    write_csv(output / "labels.csv", rows)
    train_eps, val_eps = split_episodes(rows, val_ratio=val_ratio, rng=rng)
    train_rows = [split_row(row) for row in rows if row["episode_id"] in set(train_eps)]
    val_rows = [split_row(row) for row in rows if row["episode_id"] in set(val_eps)]
    write_csv(output / "splits" / "train" / "labels.csv", train_rows)
    write_csv(output / "splits" / "val" / "labels.csv", val_rows)
    summary = {
        "annotations": str(annotations),
        "output": str(output),
        "frame_stride": frame_stride,
        "negative_ratio": negative_ratio,
        "val_ratio": val_ratio,
        "seed": seed,
        "skipped_rows": skipped,
        "candidate_frames": len(candidates),
        "selected_frames": len(rows),
        "positive_frames": sum(row["label"] == "1" for row in rows),
        "negative_frames": sum(row["label"] == "0" for row in rows),
        "train_frames": len(train_rows),
        "val_frames": len(val_rows),
        "train_episodes": train_eps,
        "val_episodes": val_eps,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_dataset(
        annotations=args.annotations,
        output=args.output,
        lerobot_root=args.lerobot_root,
        frame_stride=args.frame_stride,
        negative_ratio=args.negative_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        skip_invalid=args.skip_invalid,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
