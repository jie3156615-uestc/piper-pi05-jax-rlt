from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
import av
import polars as pl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export phase-annotation visual assets from OpenPI parquet and RLT JSONL episodes.")
    parser.add_argument("--template", required=True, type=Path, help="CSV with episode_path,phase_start_t,phase_end_t.")
    parser.add_argument("--output", required=True, type=Path, help="Output asset directory.")
    parser.add_argument(
        "--lerobot-root",
        default=Path("/home/cwzk/lerobot_datasets/piper_takeplaceredcup_5090_v2"),
        type=Path,
        help="LeRobot dataset root that owns the concatenated camera videos.",
    )
    parser.add_argument("--sheet-stride", default=10, type=int)
    parser.add_argument("--video-fps", default=15, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def pil_from_cv_bgr(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def pil_from_av_frame(frame) -> Image.Image:
    return frame.to_image().convert("RGB")


def cv_bgr_from_pil(image: Image.Image):
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def compose_pil(left: Image.Image, right: Image.Image, label: str, *, cam_w: int = 320, cam_h: int = 240) -> Image.Image:
    left = left.convert("RGB").resize((cam_w, cam_h))
    right = right.convert("RGB").resize((cam_w, cam_h))
    canvas = Image.new("RGB", (cam_w * 2, cam_h + 28), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (cam_w, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, cam_h, cam_w * 2, cam_h + 28), fill=(245, 245, 245))
    draw.text((8, cam_h + 6), label, fill=(0, 0, 0))
    return canvas


def make_sheets(sampled: list[tuple[int, Image.Image]], out_dir: Path, prefix: str, *, sheet_stride: int) -> list[Path]:
    cols = 4
    rows = 5
    thumb_w = 320
    thumb_h = 180
    caption_h = 32
    out_dir.mkdir(parents=True, exist_ok=True)
    per_page = cols * rows
    paths = []
    for page_idx in range(math.ceil(len(sampled) / per_page)):
        chunk = sampled[page_idx * per_page : (page_idx + 1) * per_page]
        if not chunk:
            continue
        sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + caption_h) + 26), "white")
        draw = ImageDraw.Draw(sheet)
        draw.rectangle((0, 0, sheet.width, 26), fill=(230, 230, 230))
        draw.text((8, 6), f"{prefix} page={page_idx + 1} stride={sheet_stride}", fill=(0, 0, 0))
        for i, (t, image) in enumerate(chunk):
            thumb = image.copy()
            thumb.thumbnail((thumb_w, thumb_h))
            x = (i % cols) * thumb_w
            y = 26 + (i // cols) * (thumb_h + caption_h)
            sheet.paste(thumb, (x, y))
            draw.text((x + 4, y + thumb_h + 4), f"t={t}", fill=(0, 0, 0))
        path = out_dir / f"{prefix}_sheet_{page_idx + 1:02d}.jpg"
        sheet.save(path, quality=92)
        paths.append(path)
    return paths


def parquet_episode_index(path: Path) -> int:
    match = re.search(r"episode_(\d+)\.parquet$", path.name)
    if not match:
        raise ValueError(f"cannot parse episode index: {path}")
    return int(match.group(1))


def convert_parquet_episode(
    *,
    entry_idx: int,
    episode_path: Path,
    out_ep: Path,
    video1: Path,
    video2: Path,
    sheet_stride: int,
    video_fps: int,
) -> dict[str, str | int]:
    df = pl.read_parquet(episode_path)
    frame_indices = df["frame_index"].to_list()
    global_indices = df["index"].to_list()
    ep = parquet_episode_index(episode_path)
    container1 = av.open(str(video1))
    container2 = av.open(str(video2))
    stream1 = container1.streams.video[0]
    stream2 = container2.streams.video[0]
    start_time = float(global_indices[0]) / float(stream1.average_rate)
    container1.seek(int(start_time / float(stream1.time_base)), stream=stream1, any_frame=False, backward=True)
    container2.seek(int(start_time / float(stream2.time_base)), stream=stream2, any_frame=False, backward=True)
    video_path = out_ep / f"{entry_idx:02d}_openpi_episode_{ep:06d}.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (640, 268))
    sampled = []
    frames1 = container1.decode(stream1)
    frames2 = container2.decode(stream2)
    for local_i, (t, frame1, frame2) in enumerate(zip(frame_indices, frames1, frames2)):
        if local_i >= len(frame_indices):
            break
        label = f"openpi episode={ep:06d} t={t} global={global_indices[local_i]}"
        composed = compose_pil(pil_from_av_frame(frame1), pil_from_av_frame(frame2), label)
        writer.write(cv_bgr_from_pil(composed))
        if int(t) % sheet_stride == 0:
            sampled.append((int(t), composed))
    writer.release()
    container1.close()
    container2.close()
    sheet_paths = make_sheets(sampled, out_ep, f"{entry_idx:02d}_openpi_episode_{ep:06d}", sheet_stride=sheet_stride)
    return {
        "source_type": "openpi_parquet",
        "episode_path": str(episode_path),
        "num_frames": len(frame_indices),
        "preview_mp4": str(video_path),
        "sheet_paths": ";".join(str(path) for path in sheet_paths),
    }


def load_rlt_rows(path: Path) -> list[dict]:
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


def convert_rlt_episode(
    *,
    entry_idx: int,
    episode_path: Path,
    out_ep: Path,
    sheet_stride: int,
    video_fps: int,
) -> dict[str, str | int]:
    rows = load_rlt_rows(episode_path)
    prefix = f"{entry_idx:02d}_{safe_name(episode_path.parent.parent.name)}_{episode_path.parent.name}"
    video_path = out_ep / f"{prefix}.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (640, 268))
    sampled = []
    for row in rows:
        t = int(row.get("t", 0))
        left_path = episode_path.parent / row["global_image"]
        right_path = episode_path.parent / row["wrist_image"]
        if not left_path.exists() or not right_path.exists():
            continue
        label = f"rlt {episode_path.parent.parent.name}/{episode_path.parent.name} t={t} src={row.get('source')}"
        composed = compose_pil(Image.open(left_path), Image.open(right_path), label)
        writer.write(cv_bgr_from_pil(composed))
        if t % sheet_stride == 0:
            sampled.append((t, composed))
    writer.release()
    sheet_paths = make_sheets(sampled, out_ep, prefix, sheet_stride=sheet_stride)
    return {
        "source_type": "rlt_jsonl",
        "episode_path": str(episode_path),
        "num_frames": len(rows),
        "preview_mp4": str(video_path),
        "sheet_paths": ";".join(str(path) for path in sheet_paths),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists, pass --overwrite: {output}")
        shutil.rmtree(output)
    (output / "episodes").mkdir(parents=True)
    video1 = args.lerobot_root / "videos/observation.images.camera1/chunk-000/file-000.mp4"
    video2 = args.lerobot_root / "videos/observation.images.camera2/chunk-000/file-000.mp4"
    with args.template.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    manifest = []
    for entry_idx, row in enumerate(rows, start=1):
        episode_path = Path(row["episode_path"])
        stem = episode_path.parent.name if episode_path.suffix == ".jsonl" else episode_path.stem
        out_ep = output / "episodes" / f"{entry_idx:02d}_{safe_name(stem)}"
        out_ep.mkdir(parents=True, exist_ok=True)
        print(f"[{entry_idx:02d}/{len(rows)}] converting {episode_path}", flush=True)
        if episode_path.suffix == ".parquet":
            info = convert_parquet_episode(
                entry_idx=entry_idx,
                episode_path=episode_path,
                out_ep=out_ep,
                video1=video1,
                video2=video2,
                sheet_stride=args.sheet_stride,
                video_fps=args.video_fps,
            )
        elif episode_path.suffix == ".jsonl":
            info = convert_rlt_episode(
                entry_idx=entry_idx,
                episode_path=episode_path,
                out_ep=out_ep,
                sheet_stride=args.sheet_stride,
                video_fps=args.video_fps,
            )
        else:
            raise ValueError(f"unsupported episode path: {episode_path}")
        info["entry_index"] = entry_idx
        info["asset_dir"] = str(out_ep)
        manifest.append(info)
    manifest_path = output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["entry_index", "source_type", "episode_path", "num_frames", "asset_dir", "preview_mp4", "sheet_paths"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)
    print(f"WROTE {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
