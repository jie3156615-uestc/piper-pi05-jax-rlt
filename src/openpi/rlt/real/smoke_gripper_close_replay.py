#!/usr/bin/env python3
"""CPU-safe one-update smoke test for migrated gripper-close replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openpi.rlt.real.agent_jax import RealRLTLearner
from openpi.rlt.real.agent_jax import ReplayBatchSampler
from openpi.rlt.real.agent_jax import ReplaySamplingConfig
from openpi.rlt.real.config import GRIPPER_RESIDUAL_CLOSE_ASSIST
from openpi.rlt.real.config import MIN_GRIPPER_CLOSE_CRITIC_BURN_IN_UPDATES
from openpi.rlt.real.config import PERSISTENT_ACTOR_EXECUTION_PROFILE
from openpi.rlt.real.config import PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT
from openpi.rlt.real.config import PERSISTENT_GRIPPER_CLOSE_GOVERNOR_PROFILE
from openpi.rlt.real.config import RealRLTConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.replay, allow_pickle=False) as loaded:
        replay = {name: np.asarray(loaded[name]) for name in loaded.files}
    cfg = RealRLTConfig(
        actor_execution_profile=PERSISTENT_ACTOR_EXECUTION_PROFILE,
        chunk_stride=10,
        hidden_dim=32,
        projection_dim=16,
        batch_size=32,
        policy_delay=1,
        beta_bc=20.0,
        beta_human_bc=0.0,
        beta_human_gripper_bc=1.0,
        human_gripper_bc_scale_m=0.005,
        freeze_gripper_residual=False,
        gripper_residual_mode=GRIPPER_RESIDUAL_CLOSE_ASSIST,
        actor_start_step=MIN_GRIPPER_CLOSE_CRITIC_BURN_IN_UPDATES,
    )
    residual_limit = np.full((10, 7), 0.005, dtype=np.float32)
    learner = RealRLTLearner.create(
        replay,
        config=cfg,
        residual_limit=residual_limit,
        fingerprints={
            "action_schema": PERSISTENT_GRIPPER_CLOSE_ACTION_SCHEMA_FINGERPRINT,
            "actor_execution_profile": PERSISTENT_ACTOR_EXECUTION_PROFILE,
            "actor_governor": PERSISTENT_GRIPPER_CLOSE_GOVERNOR_PROFILE,
        },
    )
    # This smoke isolates the first post-burn-in Actor update.  Production
    # reaches the same absolute step through 144 real Critic-only updates.
    learner.update_step = cfg.actor_start_step
    sampler = ReplayBatchSampler(
        replay,
        config=ReplaySamplingConfig(
            success_fraction=0.8,
            failure_fraction=0.2,
            human_fraction=0.25,
            seed=27,
        ),
    )
    batch = sampler.sample(cfg.batch_size)
    metrics = learner.update(batch)
    action = learner.act(
        batch["z_rl"],
        batch["state"],
        batch["a_ref"],
    )
    raw_gripper_residual = action[..., 6] - batch["a_ref"][..., 6]
    report = {
        "replay_transitions": int(len(replay["reward"])),
        "actor_updated": metrics["actor_updated"],
        "critic_loss": metrics["critic_loss"],
        "actor_loss": metrics["actor_loss"],
        "actor_human_gripper_bc_loss": metrics[
            "actor_human_gripper_bc_loss"
        ],
        "actor_admitted_human_gripper_fraction": metrics[
            "actor_admitted_human_gripper_fraction"
        ],
        "actor_human_gripper_q_filter_fraction": metrics[
            "actor_human_gripper_q_filter_fraction"
        ],
        "actor_reward1_human_gripper_chunks": metrics[
            "actor_reward1_human_gripper_chunks"
        ],
        "actor_reward0_human_gripper_chunks": metrics[
            "actor_reward0_human_gripper_chunks"
        ],
        "raw_gripper_residual_min_m": float(np.min(raw_gripper_residual)),
        "raw_gripper_residual_max_m": float(np.max(raw_gripper_residual)),
        "raw_gripper_positive_count": int(
            np.count_nonzero(raw_gripper_residual > 1e-8)
        ),
        "joint_human_bc_weight": cfg.beta_human_bc,
        "gripper_human_bc_weight": cfg.beta_human_gripper_bc,
    }
    if not np.all(np.isfinite(np.asarray(list(metrics.values())))):
        raise RuntimeError("learner update produced non-finite metrics")
    if report["raw_gripper_positive_count"] != 0:
        raise RuntimeError("close-only Actor emitted a positive/open residual")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
