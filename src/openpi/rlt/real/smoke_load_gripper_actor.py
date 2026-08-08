#!/usr/bin/env python3
"""Load a gripper-close checkpoint through the production Actor adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openpi.rlt.real.actor_runtime import JaxCheckpointShadowActor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    actor = JaxCheckpointShadowActor(args.checkpoint)
    z_dim = int(actor.learner.normalization.z_rl.mean.shape[0])
    action = actor.predict(
        z_rl=np.zeros(z_dim, dtype=np.float32),
        state=np.zeros(7, dtype=np.float32),
        a_ref=np.zeros((10, 7), dtype=np.float32),
    )
    residual = action
    report = {
        **actor.provenance_metadata(),
        "shape": list(action.shape),
        "gripper_residual_min_m": float(np.min(residual[..., 6])),
        "gripper_residual_max_m": float(np.max(residual[..., 6])),
    }
    if np.any(residual[..., 6] > 1e-8):
        raise RuntimeError("close-only Actor emitted an opening residual")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
