from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


def _torch():
    import torch

    return torch


def _torchvision_models_transforms():
    from torchvision import models
    from torchvision import transforms

    return models, transforms


def default_image_transform():
    _, transforms = _torchvision_models_transforms()
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class PhaseFrameDataset:
    """Dataset for frame-level precision-phase labels.

    Expected CSV columns: image_path,label,episode_id,t.
    """

    def __init__(self, dataset_dir: str | Path, transform: Any | None = None):
        self.dataset_dir = Path(dataset_dir)
        labels_path = self.dataset_dir / "labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing labels.csv: {labels_path}")
        with labels_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"image_path", "label", "episode_id", "t"}
            if not required.issubset(reader.fieldnames or set()):
                raise ValueError(f"labels.csv must contain columns: {sorted(required)}")
            self.rows = list(reader)
        if not self.rows:
            raise ValueError(f"No rows found in {labels_path}")
        self.transform = transform or default_image_transform()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        torch = _torch()
        from PIL import Image

        row = self.rows[index]
        image_path = self.dataset_dir / row["image_path"]
        image = Image.open(image_path).convert("RGB")
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        metadata = {
            "episode_id": str(row["episode_id"]),
            "t": int(row["t"]),
            "image_path": str(row["image_path"]),
        }
        return self.transform(image), label, metadata


def build_resnet18_binary_classifier(*, pretrained: bool = True):
    torch = _torch()
    models, _ = _torchvision_models_transforms()
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    return model


def save_phase_classifier_checkpoint(model, path: str | Path, *, arch: str = "resnet18", extra: dict[str, Any] | None = None) -> None:
    torch = _torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "arch": arch,
        "labels": {"negative": 0, "precision_phase": 1},
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_phase_classifier_checkpoint(path: str | Path, *, device=None):
    torch = _torch()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(Path(path), map_location=device)
    arch = checkpoint.get("arch", "resnet18")
    if arch != "resnet18":
        raise ValueError(f"unsupported phase classifier architecture: {arch}")
    model = build_resnet18_binary_classifier(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    metadata = {key: value for key, value in checkpoint.items() if key != "model"}
    return model, metadata


def predict_phase_probabilities(model, images: Iterable[Any], *, device=None, batch_size: int = 64) -> list[float]:
    torch = _torch()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    image_list = list(images)
    probabilities: list[float] = []
    with torch.no_grad():
        for start in range(0, len(image_list), batch_size):
            batch = torch.stack(image_list[start : start + batch_size]).to(device)
            logits = model(batch).squeeze(-1)
            probabilities.extend(float(value) for value in torch.sigmoid(logits).detach().cpu().tolist())
    return probabilities
