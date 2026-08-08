from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np

runtime_root = Path(__file__).parents[1]
scripts_dir = runtime_root / "remote_piper_runtime" / "scripts"
if not scripts_dir.is_dir():
    scripts_dir = runtime_root / "scripts"
sys.path.insert(0, str(scripts_dir))

from audit_rlt_live_canary import audit_canary_rows  # noqa: E402


CHECKPOINT = "/tmp/actor/step_00001250"


def _rows() -> list[dict]:
    rows = []
    for t in range(12):
        is_rlt = t < 10
        ref = np.zeros((10, 7), dtype=np.float32)
        actor = ref.copy()
        actor[:, :6] = 0.01
        rows.append(
            {
                "t": t,
                "source": "rlt" if is_rlt else "pi05",
                "gate_active": True,
                "gate_enter_t": 0,
                "policy_observation_t": 0,
                "policy_plan_id": "plan-1" if is_rlt else "plan-2",
                "plan_offset": t if is_rlt else t - 10,
                "behavior_actor_checkpoint": f"checkpoint:{CHECKPOINT}",
                "a_ref": ref.tolist(),
                "a_actor": actor.tolist(),
                "a_exec": (np.full(7, t * 0.001, dtype=np.float32)).tolist(),
                "policy_metadata": {
                    "actor_shadow_ready": True,
                    "actor_live_chunks_completed": 1 if t >= 9 else 0,
                    "policy_inference_latency_s": 0.1,
                    "safety_reasons": ["model_low_pass"],
                },
            }
        )
    return rows


def test_complete_one_chunk_canary_passes() -> None:
    report = audit_canary_rows(_rows(), expected_checkpoint=CHECKPOINT)
    assert report["passed"] is True
    assert report["rlt_rows"] == 10
    assert report["actor_plan_offsets"] == list(range(10))


def test_mid_chunk_plan_reset_fails() -> None:
    rows = _rows()
    rows[5]["policy_plan_id"] = "plan-2"
    rows[5]["plan_offset"] = 0
    report = audit_canary_rows(rows, expected_checkpoint=CHECKPOINT)
    assert report["passed"] is False
    assert any("plan changed" in value or "offsets=" in value for value in report["violations"])


def test_residual_and_filtered_step_limits_fail_closed() -> None:
    rows = _rows()
    rows[3]["a_actor"][0][2] = 0.03
    rows[4]["a_exec"][1] = math.radians(2.0)
    report = audit_canary_rows(rows, expected_checkpoint=CHECKPOINT)
    assert report["passed"] is False
    assert any("residual exceeds" in value for value in report["violations"])
    assert any("filtered command step" in value for value in report["violations"])


def test_no_actor_rows_is_inconclusive() -> None:
    rows = copy.deepcopy(_rows())
    for row in rows:
        row["source"] = "pi05"
    report = audit_canary_rows(rows, expected_checkpoint=CHECKPOINT)
    assert report["passed"] is False
    assert report["rlt_rows"] == 0
