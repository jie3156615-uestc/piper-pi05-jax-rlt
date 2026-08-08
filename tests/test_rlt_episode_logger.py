from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from piper_runtime.rlt_episode_logger import RLTEpisodeLogger
from piper_runtime.rlt_keyboard import RLTKeyboardStateMachine


def test_episode_logger_writes_external_replay_compatible_jsonl(tmp_path: Path) -> None:
    keyboard = RLTKeyboardStateMachine()
    output = tmp_path / "episode.jsonl"

    with RLTEpisodeLogger(output, episode_id="ep_log") as logger:
        logger.append_step(
            t=0,
            global_image="camera_global/000000.jpg",
            wrist_image="camera_wrist/000000.jpg",
            z_rl=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            state=np.zeros(7, dtype=np.float32),
            a_ref=np.zeros((10, 7), dtype=np.float32),
            a_exec=np.ones(7, dtype=np.float32),
            a_human=np.ones(7, dtype=np.float32),
            a_actor=None,
            source="human_pika",
            phase_probability=0.8,
            gate_active=True,
            keyboard=keyboard.press("s", now_s=1.0),
            timestamp_ns=1000,
            takeover_active=True,
            takeover_started_t=0,
            takeover_ended_t=None,
            policy_metadata={"z_rl_source": "zeros_missing_from_policy_response"},
        )
        keyboard.press("s", now_s=1.5)
        logger.append_step(
            t=1,
            global_image="camera_global/000001.jpg",
            wrist_image="camera_wrist/000001.jpg",
            z_rl=[0.2, 0.3, 0.4, 0.5],
            state=[0.0] * 7,
            a_ref=[[0.0] * 7 for _ in range(10)],
            a_exec=[0.0] * 7,
            a_human=None,
            a_actor=None,
            source="pi05",
            phase_probability=0.1,
            gate_active=False,
            keyboard=keyboard.press("1", now_s=2.0),
            timestamp_ns=2000,
            takeover_active=False,
            takeover_started_t=0,
            takeover_ended_t=1,
            policy_metadata={"z_rl_source": "zeros_missing_from_policy_response"},
        )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["episode_id"] == "ep_log"
    assert rows[0]["t"] == 0
    assert rows[0]["reward"] == 0.0
    assert rows[0]["done"] is False
    assert rows[0]["keyboard"]["human_takeover"] is True
    assert rows[0]["takeover_active"] is True
    assert rows[0]["takeover_started_t"] == 0
    assert rows[0]["takeover_ended_t"] is None
    assert rows[0]["policy_metadata"]["z_rl_source"] == "zeros_missing_from_policy_response"
    assert rows[0]["a_exec"] == [1.0] * 7
    assert rows[1]["reward"] == 1.0
    assert rows[1]["done"] is True
    assert rows[1]["keyboard"]["terminal_reward"] == 1.0
    assert rows[1]["takeover_active"] is False
    assert rows[1]["takeover_ended_t"] == 1


def test_episode_logger_rejects_steps_after_terminal_reward(tmp_path: Path) -> None:
    keyboard = RLTKeyboardStateMachine(toggle_debounce_s=0.01)
    output = tmp_path / "episode.jsonl"

    with RLTEpisodeLogger(output, episode_id="ep_done") as logger:
        keyboard.press("s", now_s=0.8)
        keyboard.press("s", now_s=0.9)
        logger.append_step(
            t=0,
            global_image="camera_global/000000.jpg",
            wrist_image="camera_wrist/000000.jpg",
            z_rl=[0.0],
            state=[0.0] * 7,
            a_ref=[[0.0] * 7],
            a_exec=[0.0] * 7,
            a_human=None,
            a_actor=None,
            source="pi05",
            phase_probability=0.0,
            gate_active=False,
            keyboard=keyboard.press("0", now_s=1.0),
            timestamp_ns=1000,
        )

        try:
            logger.append_step(
                t=1,
                global_image="camera_global/000001.jpg",
                wrist_image="camera_wrist/000001.jpg",
                z_rl=[0.0],
                state=[0.0] * 7,
                a_ref=[[0.0] * 7],
                a_exec=[0.0] * 7,
                a_human=None,
                a_actor=None,
                source="pi05",
                phase_probability=0.0,
                gate_active=False,
                keyboard=keyboard.snapshot(now_s=1.1),
                timestamp_ns=1100,
            )
        except RuntimeError as exc:
            assert "terminal" in str(exc)
        else:
            raise AssertionError("expected RuntimeError for appending after terminal row")
