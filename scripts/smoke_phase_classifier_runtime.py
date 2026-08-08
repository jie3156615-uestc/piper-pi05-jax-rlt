#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np

from piper_runtime.rlt_phase_gate import TorchPhaseClassifier


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    classifier = TorchPhaseClassifier(args.checkpoint, device=args.device)
    probability = classifier.predict_probability(
        {
            "camera1": np.zeros((480, 640, 3), dtype=np.uint8),
            "camera2": np.zeros((480, 640, 3), dtype=np.uint8),
        }
    )
    print(f"probability {probability:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
