from __future__ import annotations

import csv
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from openpi.rlt.real.phase_classifier import PhaseFrameDataset
from openpi.rlt.real.phase_classifier import build_resnet18_binary_classifier
from openpi.rlt.real.phase_classifier import load_phase_classifier_checkpoint
from openpi.rlt.real.phase_classifier import predict_phase_probabilities
from openpi.rlt.real.phase_classifier import save_phase_classifier_checkpoint


def _write_dataset(root: Path) -> None:
    (root / "frames").mkdir(parents=True)
    for index, color in enumerate([(255, 0, 0), (0, 255, 0)]):
        image = Image.new("RGB", (32, 32), color=color)
        image.save(root / "frames" / f"{index:06d}.jpg")
    with (root / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "label", "episode_id", "t"])
        writer.writeheader()
        writer.writerow({"image_path": "frames/000000.jpg", "label": "0", "episode_id": "ep0", "t": "0"})
        writer.writerow({"image_path": "frames/000001.jpg", "label": "1", "episode_id": "ep0", "t": "1"})


def test_phase_frame_dataset_loads_images_and_metadata(tmp_path: Path) -> None:
    _write_dataset(tmp_path)

    dataset = PhaseFrameDataset(tmp_path)
    image, label, meta = dataset[1]

    assert len(dataset) == 2
    assert tuple(image.shape) == (3, 224, 224)
    assert float(label) == 1.0
    assert meta == {"episode_id": "ep0", "t": 1, "image_path": "frames/000001.jpg"}


def test_resnet_binary_classifier_predicts_one_probability_per_image(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    dataset = PhaseFrameDataset(tmp_path)
    model = build_resnet18_binary_classifier(pretrained=False)

    probabilities = predict_phase_probabilities(model, [dataset[0][0], dataset[1][0]], device=torch.device("cpu"))

    assert len(probabilities) == 2
    assert all(0.0 <= probability <= 1.0 for probability in probabilities)


def test_phase_classifier_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = build_resnet18_binary_classifier(pretrained=False)
    checkpoint = tmp_path / "phase_classifier.pt"

    save_phase_classifier_checkpoint(model, checkpoint, arch="resnet18", extra={"epochs": 0})
    loaded_model, metadata = load_phase_classifier_checkpoint(checkpoint, device=torch.device("cpu"))

    assert metadata["arch"] == "resnet18"
    assert metadata["labels"] == {"negative": 0, "precision_phase": 1}
    assert metadata["epochs"] == 0
    assert type(loaded_model).__name__ == type(model).__name__
