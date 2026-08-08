from __future__ import annotations

import argparse
import json
from pathlib import Path

from openpi.rlt.real.phase_classifier import PhaseFrameDataset
from openpi.rlt.real.phase_classifier import build_resnet18_binary_classifier
from openpi.rlt.real.phase_classifier import save_phase_classifier_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train frozen Piper RLT phase classifier.")
    parser.add_argument("--dataset", required=True, type=Path, help="Directory with labeled episode frames and labels.csv.")
    parser.add_argument("--output", required=True, type=Path, help="Output .pt checkpoint path.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not initialize ResNet-18 from ImageNet weights.")
    parser.add_argument("--report", type=Path, help="Optional JSON training report path.")
    parser.add_argument(
        "--pos-weight",
        default="none",
        help="Positive class weight for BCEWithLogitsLoss. Use 'auto', 'none', or a numeric value.",
    )
    return parser


def _label_counts(dataset: PhaseFrameDataset) -> tuple[int, int]:
    positives = sum(1 for row in dataset.rows if float(row["label"]) >= 0.5)
    negatives = len(dataset.rows) - positives
    return positives, negatives


def _resolve_pos_weight(value: str, *, positives: int, negatives: int, device):
    import torch

    normalized = value.strip().lower()
    if normalized in {"none", "off", "false", "0"}:
        return None, None
    if normalized == "auto":
        weight = negatives / max(positives, 1)
    else:
        weight = float(value)
    return torch.tensor([weight], dtype=torch.float32, device=device), weight


def main(argv: list[str] | None = None) -> int:
    import torch
    from torch.utils.data import DataLoader

    args = build_parser().parse_args(argv)
    dataset = PhaseFrameDataset(args.dataset)
    positives, negatives = _label_counts(dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18_binary_classifier(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    pos_weight_tensor, pos_weight_value = _resolve_pos_weight(args.pos_weight, positives=positives, negatives=negatives, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    model.train()
    epoch_reports = []
    for epoch in range(args.epochs):
        total_loss = 0.0
        total = 0
        for images, labels, _metadata in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images).squeeze(-1)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * images.shape[0]
            total += images.shape[0]
        average_loss = total_loss / max(total, 1)
        epoch_reports.append({"epoch": epoch + 1, "loss": average_loss})
        print(f"epoch={epoch + 1} loss={average_loss:.6f}")

    save_phase_classifier_checkpoint(
        model,
        args.output,
        arch="resnet18",
        extra={
            "epochs": args.epochs,
            "dataset": str(args.dataset),
            "pretrained": not args.no_pretrained,
            "positive_frames": positives,
            "negative_frames": negatives,
            "pos_weight": pos_weight_value,
        },
    )
    if args.report:
        report = {
            "dataset": str(args.dataset),
            "output": str(args.output),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "pretrained": not args.no_pretrained,
            "positive_frames": positives,
            "negative_frames": negatives,
            "pos_weight": pos_weight_value,
            "device": str(device),
            "epochs_report": epoch_reports,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
