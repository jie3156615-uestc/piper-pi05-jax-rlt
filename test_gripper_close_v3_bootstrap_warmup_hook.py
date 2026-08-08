#!/usr/bin/env python3
"""Regression tests for cumulative v3 bootstrap warm-up accounting.

The gripper-close v3 lineage starts with 30 immutable, already-audited
bootstrap episodes.  The after-episode hook must add those episodes to its
cheap report probe before deciding whether the strict updater is worth
invoking.  It must not treat legacy ``frozen_base_episode_ids`` as equivalent
bootstrap provenance.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


HOOK = Path(
    os.environ.get(
        "GRIPPER_V3_UPDATE_HOOK",
        Path(__file__).with_name(
            "run_rlt_online_update_hook_gripper_close_v3.sh"
        ),
    )
).resolve()
POSIX_ONLY = pytest.mark.skipif(
    os.name != "posix", reason="production hook targets Ubuntu Bash"
)

RAW_SCHEMA = (
    "piper_joint_delta_v5_c10_n10_stride10_behavior_ref50_"
    "rank1_joint_r005_d1_0015_d2_001_cone15_"
    "gripper_close_knot_r005"
)
EXECUTION_SCHEMA = (
    "piper_joint_delta_v5_c10_n10_stride10_behavior_ref50_"
    "persistent_filtered_actual_r005_d1_0015_d2_001_cone15_"
    "gripper_close_assist_r005_d1_0005_d2_0003_boundary0005"
)
PROJECTION = (
    "rank1_joint_v1_r005_d1_0015_d2_001_cone15_"
    "scale33_min020_gripper_close_knot_r005"
)
GOVERNOR = (
    "persistent_governor_v3_joint_r005_d1_0015_d2_001_cone15_"
    "boundary060_gripper_close_r005_d1_0005_d2_0003_boundary0005"
)


def _write_online_candidates(session_root: Path, count: int) -> None:
    for index in range(count):
        episode = session_root / f"episode_{408 + index:06d}"
        episode.mkdir(parents=True)
        (episode / "report.json").write_text(
            json.dumps(
                {
                    "outcome": "episode_done",
                    "terminal_reward": float(index % 2),
                    "exclude_from_training": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (episode / "episode.jsonl").write_text("{}\n", encoding="utf-8")


def _make_fake_workspace(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    python = workspace / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(Path(sys.executable))
    tools = workspace / "scripts/piper_rlt/tools"
    tools.mkdir(parents=True)
    marker = tmp_path / "calls.jsonl"

    (tools / "probe_online_rlt_warmup_reports.py").write_text(
        """#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--session-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
rewards = []
for report_path in args.session_root.glob("episode_*/report.json"):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("outcome") == "episode_done"
        and report.get("terminal_reward") in (0, 0.0, 1, 1.0)
        and report.get("exclude_from_training") is not True
        and (report_path.parent / "episode.jsonl").is_file()
    ):
        rewards.append(float(report["terminal_reward"]))
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(
    json.dumps(
        {
            "candidate_episodes": len(rewards),
            "candidate_successes": sum(value == 1.0 for value in rewards),
            "candidate_failures": sum(value == 0.0 for value in rewards),
        }
    )
    + "\\n",
    encoding="utf-8",
)
""",
        encoding="utf-8",
    )
    (tools / "print_json_fields.py").write_text(
        """#!/usr/bin/env python3
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--field", action="append", required=True)
args = parser.parse_args()
with open(args.input, encoding="utf-8") as stream:
    payload = json.load(stream)
for field in args.field:
    print(payload.get(field, ""))
""",
        encoding="utf-8",
    )
    (tools / "run_online_rlt_update.py").write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

marker = Path(os.environ["FAKE_MARKER"])
kind = "strict-dry-run" if "--dry-run" in sys.argv else "formal-update"
with marker.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({"kind": kind, "argv": sys.argv[1:]}) + "\\n")
if kind == "strict-dry-run":
    print(
        json.dumps(
            {
                "outcome": "dry_run_would_update",
                "episodes": 31,
                "warmup_episodes": 30,
                "successes": 24,
                "failures": 7,
                "human_intervention_episodes": 30,
                "audited_trainable_transitions": 145,
            }
        )
    )
else:
    print(json.dumps({"outcome": "updated_and_promoted_at_episode_boundary"}))
""",
        encoding="utf-8",
    )
    (tools / "generate_external_rlt_enrichment_cache.py").write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

marker = Path(os.environ["FAKE_MARKER"])
with marker.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({"kind": "enrichment"}) + "\\n")
output = Path(sys.argv[sys.argv.index("--output") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("{}\\n", encoding="utf-8")
""",
        encoding="utf-8",
    )
    return workspace, marker


def _write_config(
    path: Path,
    *,
    session_root: Path,
    state_root: Path,
    workspace: Path,
    runtime: Path,
) -> None:
    values: dict[str, object] = {
        "RLT_SESSION_ROOT": session_root,
        "RLT_STATE_ROOT": state_root,
        "RLT_WORKSPACE": workspace,
        "RLT_RUNTIME": runtime,
        "RLT_LINEAGE_MODE": "persistent_gripper_v3_bootstrap_warm_start",
        "RLT_REPLAY_TRAINING_POLICY": (
            "immutable_migrated_v5_warmup_plus_persistent_v5_online"
        ),
        "RLT_PHASE_CHECKPOINT": runtime / "phase.pt",
        "RLT_SELECTED_ACTOR_FILE": runtime / "selected_actor.txt",
        "RLT_SHADOW_SERVICE": "openpi-rlt-shadow-policy-gripper-v3.service",
        "RLT_POLICY_HOST": "127.0.0.1",
        "RLT_POLICY_PORT": "8001",
        "RLT_WARMUP_EPISODES": "30",
        "RLT_MIN_SUCCESS": "0",
        "RLT_MIN_FAILURE": "0",
        "RLT_MIN_SUCCESS_HUMAN_EPISODES": "0",
        "RLT_MIN_ADMITTED_HUMAN_EPISODES": "1",
        "RLT_UPDATE_EVERY": "1",
        "RLT_MIN_WARMUP_TRANSITIONS": "1",
        "RLT_UTD": "1.0",
        "RLT_MIN_UPDATE_STEPS": "1",
        "RLT_MAX_UPDATE_STEPS": "1250",
        "RLT_BATCH_SIZE": "256",
        "RLT_BETA_BC": "20.0",
        "RLT_BETA_HUMAN_BC": "0.0",
        "RLT_BETA_HUMAN_GRIPPER_BC": "1.0",
        "RLT_HUMAN_GRIPPER_BC_SCALE_M": "0.005",
        "RLT_HUMAN_GRIPPER_Q_FILTER_MODE": "critic_min_advantage_v1",
        "RLT_HUMAN_GRIPPER_Q_FILTER_MARGIN": "0.0",
        "RLT_REFERENCE_DROPOUT": "0.5",
        "RLT_TARGET_POLICY_NOISE_STD": "0.1",
        "RLT_TARGET_POLICY_NOISE_CLIP": "0.2",
        "RLT_RESIDUAL_MAX": "0.005",
        "RLT_RESIDUAL_D1_MAX_RAD": "0.0015",
        "RLT_RESIDUAL_D2_MAX_RAD": "0.001",
        "RLT_DIRECTION_CONE_DEG": "15.0",
        "RLT_ACTOR_MODEL_ACTION_SCHEMA_FINGERPRINT": RAW_SCHEMA,
        "RLT_EXECUTION_ACTION_SCHEMA_FINGERPRINT": EXECUTION_SCHEMA,
        "RLT_ACTOR_PROJECTION_PROFILE": PROJECTION,
        "RLT_ACTOR_EXECUTION_PROFILE": "persistent_c10_filtered_actual_v2",
        "RLT_EXECUTION_FILTER_PROFILE": (
            "exp_one_minus_exp_neg_dt_over_tau_v1"
        ),
        "RLT_EXECUTION_FILTER_TAU_S": "0.05",
        "RLT_CONTROL_HZ": "30.0",
        "RLT_ACTOR_LIVE_MAX_BOUNDARY_JUMP_RAD": "0.06",
        "RLT_ACTOR_PROJECTION_SCALE_STEPS": "33",
        "RLT_ACTOR_MIN_PROJECTION_SCALE": "0.2",
        "RLT_ACTOR_DIRECTION_STATIC_THRESHOLD_RAD": "0.001",
        "RLT_ACTOR_GOVERNOR_FINGERPRINT": GOVERNOR,
        "RLT_MIN_NEW_PERSISTENT_COMMITTED_EPISODES": "0",
        "RLT_GRIPPER_RESIDUAL_MODE": "close_only_persistent_v1",
        "RLT_GRIPPER_RESIDUAL_MAX": "0.005",
        "RLT_GRIPPER_RESIDUAL_D1_MAX_M": "0.0005",
        "RLT_GRIPPER_RESIDUAL_D2_MAX_M": "0.0003",
        "RLT_GRIPPER_MAX_BOUNDARY_JUMP_M": "0.0005",
        "RLT_GRIPPER_COMMAND_MIN_M": "0.0",
        "RLT_GRIPPER_COMMAND_MAX_M": "0.08",
        "RLT_GRIPPER_RELEASE_REFERENCE_M": "0.05",
        "RLT_GRIPPER_RELEASE_DELTA_M": "0.002",
        "RLT_FREEZE_GRIPPER_RESIDUAL": "0",
        "RLT_SUCCESS_FRACTION": "0.5",
        "RLT_HUMAN_FRACTION": "0.25",
        "RLT_VALIDATION_FRACTION": "0.15",
        "RLT_MAX_VALIDATION_TD_ERROR": "0.5",
        "RLT_MAX_ACTOR_Q_ADVANTAGE": "0.5",
        "RLT_ALLOW_WARM_START_OBJECTIVE_MIGRATION": "1",
        "RLT_WARM_START_ACTOR_CHECKPOINT": runtime / "step_00000288",
        "RLT_BASE_FINGERPRINT": "base",
        "RLT_TOKEN_FINGERPRINT": "token",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            f"{key}={shlex.quote(str(value))}\n"
            for key, value in values.items()
        ),
        encoding="utf-8",
    )
    (runtime / "selected_actor.txt").write_text(
        f"{runtime / 'step_00000288'}\n", encoding="utf-8"
    )


def _valid_bootstrap_state() -> dict[str, object]:
    episode_ids = [f"episode_{index:06d}" for index in range(30)]
    return {
        "lineage_mode": "persistent_gripper_v3_bootstrap_warm_start",
        "bootstrap_gripper_episode_ids": episode_ids,
        "bootstrap_gripper_episode_count": 30,
        "bootstrap_gripper_quality": {
            "episodes": 30,
            "reward_positive_episodes": 24,
            "reward_negative_episodes": 6,
        },
    }


def _run_hook(
    tmp_path: Path,
    *,
    state: dict[str, object] | None,
    online_candidates: int = 1,
) -> tuple[subprocess.CompletedProcess[str], list[dict[str, object]]]:
    session_root = tmp_path / "session"
    state_root = session_root / ".online_rlt_persistent_gripper_v3"
    runtime = tmp_path / "runtime"
    runtime.mkdir(parents=True)
    workspace, marker = _make_fake_workspace(tmp_path)
    _write_online_candidates(session_root, online_candidates)
    if state is not None:
        state_root.mkdir(parents=True, exist_ok=True)
        (state_root / "online_state.json").write_text(
            json.dumps(state) + "\n", encoding="utf-8"
        )
    config = state_root / "config.env"
    _write_config(
        config,
        session_root=session_root,
        state_root=state_root,
        workspace=workspace,
        runtime=runtime,
    )
    env = {
        **os.environ,
        "RLT_ONLINE_CONFIG": str(config),
        "RLT_V3_WORKSPACE_OVERRIDE": str(workspace),
        "RLT_V3_RUNTIME_OVERRIDE": str(runtime),
        "FAKE_MARKER": str(marker),
    }
    completed = subprocess.run(
        ["bash", str(HOOK)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    calls: list[dict[str, object]] = []
    if marker.is_file():
        calls = [
            json.loads(line)
            for line in marker.read_text(encoding="utf-8").splitlines()
        ]
    return completed, calls


def test_hook_source_counts_only_validated_v3_bootstrap_provenance() -> None:
    """Cross-platform structural guard for the cheap cumulative gate."""

    text = HOOK.read_text(encoding="utf-8")
    for needle in (
        'state.get("bootstrap_gripper_episode_ids")',
        'state.get("bootstrap_gripper_episode_count", len(bootstrap_ids))',
        'state.get("bootstrap_gripper_quality")',
        'bootstrap_quality.get("episodes", len(bootstrap_ids))',
        "bootstrap_count != len(bootstrap_ids)",
        "bootstrap_quality_count != len(bootstrap_ids)",
        "bootstrap_successes + bootstrap_failures != len(bootstrap_ids)",
        "INCREMENTAL_CANDIDATE_EPISODES",
        "CANDIDATE_EPISODES + FROZEN_BASE_EPISODES + BOOTSTRAP_EPISODES",
        "bootstrap=${BOOTSTRAP_EPISODES} + "
        "newly-admitted=${INCREMENTAL_CANDIDATE_EPISODES}",
    ):
        assert needle in text

    legacy_reject = text.index(
        "persistent-v2 state contains forbidden frozen-base episodes"
    )
    cumulative_add = text.index(
        "CANDIDATE_EPISODES + FROZEN_BASE_EPISODES + BOOTSTRAP_EPISODES"
    )
    assert legacy_reject < cumulative_add


@POSIX_ONLY
def test_bootstrap30_plus_first_admitted_episode_runs_strict_updater(
    tmp_path: Path,
) -> None:
    completed, calls = _run_hook(
        tmp_path,
        state=_valid_bootstrap_state(),
        online_candidates=1,
    )

    assert completed.returncode == 0, completed.stderr
    assert "1/30" not in completed.stdout
    assert "learner gate is ready at 31 admitted episodes" in completed.stdout
    assert [call["kind"] for call in calls] == [
        "strict-dry-run",
        "enrichment",
        "formal-update",
    ]
    dry_run_argv = calls[0]["argv"]
    assert "--warmup-episodes" in dry_run_argv
    assert dry_run_argv[dry_run_argv.index("--warmup-episodes") + 1] == "30"


@POSIX_ONLY
def test_missing_bootstrap_provenance_still_waits_at_one_of_thirty(
    tmp_path: Path,
) -> None:
    completed, calls = _run_hook(
        tmp_path,
        state={"lineage_mode": "legacy_without_v3_bootstrap"},
        online_candidates=1,
    )

    assert completed.returncode == 0, completed.stderr
    assert (
        "cumulative learner readiness: bootstrap=0 + "
        "newly-admitted=1 = 1/30"
    ) in completed.stdout
    assert calls == []


@POSIX_ONLY
def test_legacy_frozen_base_cannot_satisfy_v3_bootstrap_gate(
    tmp_path: Path,
) -> None:
    completed, calls = _run_hook(
        tmp_path,
        state={
            "lineage_mode": "persistent_v2_actor_only_warm_start",
            "frozen_base_episode_ids": [
                f"episode_{index:06d}" for index in range(30)
            ],
        },
        online_candidates=1,
    )

    assert completed.returncode == 2
    assert (
        "persistent-v2 state contains forbidden frozen-base episodes"
        in completed.stderr
    )
    assert calls == []
