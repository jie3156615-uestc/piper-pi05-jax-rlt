#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import numpy as np

from piper_runtime.rlt_phase_gate import TorchPhaseClassifier


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    classifier = TorchPhaseClassifier(args.checkpoint, device=args.device)
    images = {
        "camera1": np.zeros((480, 640, 3), dtype=np.uint8),
        "camera2": np.zeros((480, 640, 3), dtype=np.uint8),
    }
    classifier.predict_probability(images)
    start = time.perf_counter()
    probabilities = [classifier.predict_probability(images) for _ in range(args.iters)]
    elapsed = time.perf_counter() - start
    print(f"iters {args.iters}")
    print(f"avg_ms {elapsed * 1000.0 / max(args.iters, 1):.3f}")
    print(f"last_probability {probabilities[-1]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
