#!/usr/bin/env python3
"""Create a compact QA grid for a phase-classifier validation timeline CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _get(row: dict[str, str], *names: str, default: str = "") -> str:
    for name in names:
        if name in row:
            return row[name]
    return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeline-csv", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--samples-per-group", type=int, default=8)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.timeline_csv.open(newline="")))
    if not rows:
        raise RuntimeError(f"empty timeline CSV: {args.timeline_csv}")

    image_by_key: dict[tuple[str, str], str] = {}
    if args.labels_csv is not None:
        labels_root = args.labels_csv.parent
        for label_row in csv.DictReader(args.labels_csv.open(newline="")):
            episode_id = label_row.get("episode_id", "")
            t = label_row.get("t", "")
            image_path = label_row.get("image_path", "")
            if image_path:
                path = Path(image_path)
                if not path.is_absolute():
                    path = (labels_root / path).resolve()
                image_by_key[(episode_id, t)] = str(path)

    for row in rows:
        row["_p"] = float(_get(row, "probability", "prob", "p", default="0") or 0)
        row["_y"] = int(float(_get(row, "label", "y", "manual_label", default="0") or 0))
        row["_pred"] = int(
            float(
                _get(
                    row,
                    "prediction",
                    "pred",
                    "is_active",
                    default="1" if row["_p"] >= args.threshold else "0",
                )
                or 0
            )
        )
        row["_ep"] = _get(row, "episode_id", "episode", default="")
        row["_t"] = _get(row, "t", "frame_index", "step", default="")
        row["_img"] = _get(row, "image_path", "path", "image", default="")
        if not row["_img"]:
            row["_img"] = image_by_key.get((row["_ep"], row["_t"]), "")

    groups = []
    for title, predicate, key in [
        ("TP: y=1 pred=1 high p", lambda r: r["_y"] == 1 and r["_pred"] == 1, lambda r: -r["_p"]),
        ("FP: y=0 pred=1 high p", lambda r: r["_y"] == 0 and r["_pred"] == 1, lambda r: -r["_p"]),
        ("FN: y=1 pred=0 low p", lambda r: r["_y"] == 1 and r["_pred"] == 0, lambda r: r["_p"]),
        ("TN: y=0 pred=0 low p", lambda r: r["_y"] == 0 and r["_pred"] == 0, lambda r: r["_p"]),
    ]:
        items = [row for row in rows if predicate(row)]
        groups.append((title, items, key))

    thumb_w, thumb_h = 240, 120
    pad = 12
    label_h = 38
    cols = 4
    section_h = label_h + 2 * (thumb_h + label_h + pad) + pad
    canvas_w = cols * (thumb_w + pad) + pad
    canvas_h = len(groups) * section_h + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
        font_b = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = font_b = None

    y0 = pad
    for title, items, key in groups:
        total = len(items)
        items = sorted(items, key=key)[: args.samples_per_group]
        draw.text((pad, y0), f"{title}  shown={len(items)} total={total}", fill="black", font=font_b)
        yy = y0 + label_h
        for idx, row in enumerate(items):
            x = pad + (idx % cols) * (thumb_w + pad)
            if idx and idx % cols == 0:
                yy += thumb_h + label_h + pad
            img_path = Path(row["_img"])
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((thumb_w, thumb_h))
            tile = Image.new("RGB", (thumb_w, thumb_h), (245, 245, 245))
            tile.paste(img, ((thumb_w - img.width) // 2, (thumb_h - img.height) // 2))
            canvas.paste(tile, (x, yy))
            border = "green" if row["_y"] == row["_pred"] else "red"
            draw.rectangle([x, yy, x + thumb_w - 1, yy + thumb_h - 1], outline=border, width=3)
            episode = row["_ep"]
            parts = episode.split("/")
            ep_short = "/".join(parts[-2:]) if len(parts) >= 2 else episode
            caption = f"p={row['_p']:.2f} y={row['_y']} pred={row['_pred']} t={row['_t']}\n{ep_short[:34]}"
            draw.text((x, yy + thumb_h + 2), caption, fill="black", font=font)
        y0 += section_h

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=92)
    print(args.output)


if __name__ == "__main__":
    main()
