from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from openpi.rlt.real.phase_classifier import PhaseFrameDataset
from openpi.rlt.real.phase_classifier import load_phase_classifier_checkpoint
from openpi.rlt.real.phase_gate import HysteresisGate
from openpi.rlt.real.phase_gate import event_alignment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate frozen Piper RLT phase classifier.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--heldout-dataset", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--enter-threshold", type=float, default=0.7)
    parser.add_argument("--exit-threshold", type=float, default=0.4)
    parser.add_argument("--enter-frames", type=int, default=3)
    parser.add_argument("--exit-frames", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5, help="Frame-level probability threshold for metrics.")
    parser.add_argument("--timeline-csv", type=Path, help="Optional per-frame probability CSV output path.")
    return parser


def _as_python_list(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _classification_metrics(rows: list[dict], *, threshold: float) -> dict:
    true_positive = false_positive = true_negative = false_negative = 0
    for row in rows:
        label = float(row["label"]) >= 0.5
        predicted = float(row["probability"]) >= threshold
        if label and predicted:
            true_positive += 1
        elif not label and predicted:
            false_positive += 1
        elif not label and not predicted:
            true_negative += 1
        else:
            false_negative += 1
    total = len(rows)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": threshold,
        "accuracy": (true_positive + true_negative) / max(total, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive / max(false_positive + true_negative, 1),
        "false_negative_rate": false_negative / max(false_negative + true_positive, 1),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def _write_timeline_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode_id", "t", "label", "probability"])
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    import torch
    from torch.utils.data import DataLoader

    args = build_parser().parse_args(argv)
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Classifier checkpoint does not exist: {args.checkpoint}")
    if not args.heldout_dataset.exists():
        raise FileNotFoundError(f"Held-out dataset does not exist: {args.heldout_dataset}")
    dataset = PhaseFrameDataset(args.heldout_dataset)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _metadata = load_phase_classifier_checkpoint(args.checkpoint, device=device)

    rows = []
    with torch.no_grad():
        for images, labels, metadata in loader:
            probabilities = torch.sigmoid(model(images.to(device)).squeeze(-1)).cpu().tolist()
            episode_ids = _as_python_list(metadata["episode_id"])
            timesteps = _as_python_list(metadata["t"])
            label_values = _as_python_list(labels)
            for probability, label, episode_id, timestep in zip(probabilities, label_values, episode_ids, timesteps):
                rows.append(
                    {
                        "episode_id": str(episode_id),
                        "t": int(timestep),
                        "label": float(label),
                        "probability": float(probability),
                    }
                )

    by_episode: dict[str, list[dict]] = {}
    for row in rows:
        by_episode.setdefault(row["episode_id"], []).append(row)

    episode_reports = []
    false_positive_frames = 0
    negative_frames = 0
    for episode_id, episode_rows in by_episode.items():
        episode_rows.sort(key=lambda item: item["t"])
        gate = HysteresisGate(
            enter_threshold=args.enter_threshold,
            exit_threshold=args.exit_threshold,
            enter_frames=args.enter_frames,
            exit_frames=args.exit_frames,
        )
        active_times = []
        manual_positive_times = []
        for row in episode_rows:
            active = gate.update(row["probability"])
            if active:
                active_times.append(row["t"])
            if row["label"] >= 0.5:
                manual_positive_times.append(row["t"])
            else:
                negative_frames += 1
                if active:
                    false_positive_frames += 1
        if active_times and manual_positive_times:
            alignment = event_alignment(
                predicted_enter_t=active_times[0],
                manual_enter_t=manual_positive_times[0],
                predicted_exit_t=active_times[-1],
                manual_exit_t=manual_positive_times[-1],
            )
        else:
            alignment = {"enter_lead_frames": 0, "exit_lag_frames": 0}
        episode_reports.append(
            {
                "episode_id": episode_id,
                "num_frames": len(episode_rows),
                "predicted_active_frames": len(active_times),
                "manual_positive_frames": len(manual_positive_times),
                **alignment,
            }
        )

    report = {
        "checkpoint": str(args.checkpoint),
        "heldout_dataset": str(args.heldout_dataset),
        "metrics": _classification_metrics(rows, threshold=args.threshold),
        "num_episodes": len(by_episode),
        "num_frames": len(rows),
        "false_positive_rate_on_negative_frames": false_positive_frames / max(negative_frames, 1),
        "episodes": episode_reports,
        "timeline_rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.timeline_csv:
        _write_timeline_csv(args.timeline_csv, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
