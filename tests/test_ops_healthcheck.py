from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "ops" / "piper_rlt_healthcheck.py"
SPEC = importlib.util.spec_from_file_location("piper_rlt_healthcheck", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
health = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = health
SPEC.loader.exec_module(health)


def test_latest_validation_and_episode_metrics(tmp_path: Path, monkeypatch) -> None:
    session = tmp_path / "session"
    state = session / ".online_rlt_test"
    learner = state / "learner"
    learner.mkdir(parents=True)
    (state / "selected_actor_checkpoint.txt").write_text(
        "/tmp/learner/step_00000821\n", encoding="utf-8"
    )
    for index, reward in enumerate((1.0, 0.0, 1.0)):
        episode = session / f"episode_{index:06d}"
        episode.mkdir()
        (episode / "report.json").write_text(
            json.dumps({"outcome": "episode_done", "terminal_reward": reward}), encoding="utf-8"
        )
    (learner / "validation_test.json").write_text(
        json.dumps(
            {
                "validation_td_error_abs_mean": 0.1,
                "actor_q_advantage_abs_p95": 0.2,
                "reward1_reward0_exec_q_gap": 0.3,
            }
        ),
        encoding="utf-8",
    )
    (learner / "training_summary.json").write_text(
        json.dumps(
            {
                "final_step": 821,
                "additional_steps": 5,
                "final_metrics": {"q1_mean": 0.7, "q2_mean": 0.65, "td_error_abs": 0.08},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(health, "_check_gpu", lambda: health.Check("gpu", "ok", "test", {}))
    report = health.collect("offline", session, min_disk_gib=0.0, min_disk_percent=0.0)
    checks = {item["name"]: item for item in report["checks"]}
    assert checks["selected_actor"]["details"]["step"] == 821
    assert checks["episode_metrics"]["details"]["success_rate"] == 2 / 3
    assert checks["learner_validation"]["details"]["validation_td_error_abs_mean"] == 0.1
    assert checks["learner_training"]["details"]["q1_mean"] == 0.7
    assert report["overall"] == "ok"


def test_missing_learning_artifacts_are_warnings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(health, "_check_gpu", lambda: health.Check("gpu", "ok", "test", {}))
    report = health.collect("offline", tmp_path / "missing", min_disk_gib=0.0, min_disk_percent=0.0)
    assert report["overall"] == "warn"
    assert report["counts"]["fail"] == 0


def test_find_key_walks_nested_reports() -> None:
    value = {"validation": [{"validation_td_error_abs_mean": 0.25}]}
    assert health._find_key(value, "validation_td_error_abs_mean") == 0.25
