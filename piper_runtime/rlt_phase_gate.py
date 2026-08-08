from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class PhaseGateConfig:
    """Configuration for single-latch RLT precision-phase entry."""

    enter_threshold: float = 0.5
    enter_consecutive_frames: int = 3
    classifier_period: int = 1

    def validate(self) -> "PhaseGateConfig":
        if not 0.0 <= float(self.enter_threshold) <= 1.0:
            raise ValueError("enter_threshold must be in [0, 1]")
        if int(self.enter_consecutive_frames) < 1:
            raise ValueError("enter_consecutive_frames must be >= 1")
        if int(self.classifier_period) < 1:
            raise ValueError("classifier_period must be >= 1")
        return dataclasses.replace(
            self,
            enter_threshold=float(self.enter_threshold),
            enter_consecutive_frames=int(self.enter_consecutive_frames),
            classifier_period=int(self.classifier_period),
        )


@dataclasses.dataclass(frozen=True)
class PhaseGateSnapshot:
    probability: float | None
    active: bool
    state: str
    enter_t: int | None
    exit_t: int | None
    reason: str
    high_count: int


class SingleLatchPhaseGate:
    """Enter once on consecutive positive predictions and stay active until terminal.

    This intentionally avoids probability-based exit hysteresis. For this robot
    task, early/late boundary frame errors are acceptable; repeated in/out RLT
    transitions during one placement are not.
    """

    def __init__(self, config: PhaseGateConfig | None = None):
        self.config = (config or PhaseGateConfig()).validate()
        self._state = "IDLE"
        self._enter_t: int | None = None
        self._exit_t: int | None = None
        self._high_count = 0
        self._last_probability: float | None = None
        self._reason = "waiting_for_phase"

    def update(
        self,
        probability: float | None,
        *,
        t: int,
        terminal: bool = False,
        terminal_reason: str = "terminal",
    ) -> PhaseGateSnapshot:
        if probability is not None:
            probability = float(probability)
            if not np.isfinite(probability):
                raise ValueError("phase probability must be finite")
            probability = min(1.0, max(0.0, probability))
            self._last_probability = probability

        if terminal:
            if self._state == "ACTIVE" and self._exit_t is None:
                self._exit_t = int(t)
                self._reason = str(terminal_reason)
            elif self._state != "ACTIVE":
                self._reason = str(terminal_reason)
            self._state = "EXITED"
            return self.snapshot()

        if self._state == "ACTIVE":
            self._reason = "locked_until_terminal"
            return self.snapshot()
        if self._state == "EXITED":
            return self.snapshot()
        if probability is None:
            self._reason = "no_prediction"
            return self.snapshot()

        if probability >= self.config.enter_threshold:
            self._high_count += 1
            self._reason = "above_enter_threshold"
        else:
            self._high_count = 0
            self._reason = "below_enter_threshold"

        if self._high_count >= self.config.enter_consecutive_frames:
            self._state = "ACTIVE"
            if self._enter_t is None:
                self._enter_t = int(t)
            self._reason = "entered_single_latch"
        return self.snapshot()

    def snapshot(self) -> PhaseGateSnapshot:
        return PhaseGateSnapshot(
            probability=self._last_probability,
            active=self._state == "ACTIVE",
            state=self._state,
            enter_t=self._enter_t,
            exit_t=self._exit_t,
            reason=self._reason,
            high_count=self._high_count,
        )


class TorchPhaseClassifier:
    """Small runtime wrapper for the trained ResNet-18 phase classifier."""

    def __init__(self, checkpoint: str | Path, *, device: str = "cpu"):
        self.checkpoint = Path(checkpoint).expanduser()
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"phase classifier checkpoint does not exist: {self.checkpoint}")
        self.device_name = str(device)
        self._torch = None
        self._model = None
        self._transform = None
        self._device = None

    def predict_probability(self, images: dict[str, np.ndarray]) -> float:
        self._ensure_loaded()
        assert self._torch is not None
        assert self._model is not None
        assert self._transform is not None
        assert self._device is not None
        pil_image = _compose_dual_camera_image(images)
        tensor = self._transform(pil_image).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            logits = self._model(tensor).squeeze(-1)
            probability = self._torch.sigmoid(logits)[0].detach().cpu().item()
        return float(probability)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from torchvision import models
        from torchvision import transforms

        try:
            checkpoint = torch.load(self.checkpoint, map_location=self.device_name, weights_only=True)
        except TypeError:
            checkpoint = torch.load(self.checkpoint, map_location=self.device_name)
        arch = checkpoint.get("arch", "resnet18")
        if arch != "resnet18":
            raise ValueError(f"unsupported phase classifier architecture: {arch}")
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
        model.load_state_dict(checkpoint["model"])
        device = torch.device(self.device_name)
        model = model.to(device)
        model.eval()
        self._torch = torch
        self._model = model
        self._device = device
        self._transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )


def should_run_classifier(*, t: int, period: int) -> bool:
    if period < 1:
        raise ValueError("period must be >= 1")
    return int(t) % int(period) == 0


def _compose_dual_camera_image(images: dict[str, np.ndarray]) -> Any:
    from PIL import Image

    try:
        left = Image.fromarray(_as_uint8_rgb(images["camera1"]))
        right = Image.fromarray(_as_uint8_rgb(images["camera2"]))
    except KeyError as exc:
        raise KeyError("phase classifier expects images with 'camera1' and 'camera2'") from exc
    left = left.convert("RGB").resize((320, 240))
    right = right.convert("RGB").resize((320, 240))
    canvas = Image.new("RGB", (640, 240), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (320, 0))
    return canvas


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected RGB image shape (H, W, 3), got {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array
