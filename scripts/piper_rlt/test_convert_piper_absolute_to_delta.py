from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TOOLS_DIR = Path(__file__).resolve().parent / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from convert_piper_absolute_to_delta import (  # noqa: E402
    build_feedback_delta,
    convert_dataset,
    convert_joint_actions,
    verify_dataset,
)


def test_convert_joint_actions_uses_six_joint_deltas_and_absolute_gripper() -> None:
    state = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.01],
            [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 0.02],
        ],
        dtype=np.float32,
    )
    absolute_action = np.array(
        [
            [1.1, 1.8, 3.3, 4.0, 4.5, 6.6, 0.07],
            [1.4, 2.9, 3.5, 4.2, 5.8, 6.1, 0.08],
        ],
        dtype=np.float32,
    )

    converted = convert_joint_actions(state, absolute_action)

    np.testing.assert_allclose(
        converted,
        [
            [0.1, -0.2, 0.3, 0.0, -0.5, 0.6, 0.07],
            [-0.1, 0.4, 0.0, -0.3, 0.3, -0.4, 0.08],
        ],
        atol=1e-6,
    )


def test_feedback_delta_never_crosses_episode_boundaries() -> None:
    state = np.array(
        [
            [1.0, 0.01],
            [1.2, 0.02],
            [9.0, 0.03],
            [9.5, 0.04],
        ],
        dtype=np.float32,
    )
    episodes = np.array([0, 0, 1, 1], dtype=np.int64)

    delta, valid = build_feedback_delta(state, episodes)

    np.testing.assert_array_equal(valid, [True, False, True, False])
    np.testing.assert_allclose(delta[valid], [[0.2, 0.01], [0.5, 0.01]], atol=1e-6)
    assert np.isnan(delta[~valid]).all()


def test_convert_dataset_rewrites_actions_and_preserves_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    (source / "data" / "chunk-000").mkdir(parents=True)
    (source / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (source / "videos" / "observation.images.camera1" / "chunk-000").mkdir(parents=True)

    absolute_action = [[1.1, 1.8, 3.3, 4.0, 4.5, 6.6, 0.07], [1.4, 2.9, 3.5, 4.2, 5.8, 6.1, 0.08]]
    state = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.01], [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 0.02]]
    data_table = pa.table(
        {
            "action": pa.array(absolute_action, type=pa.list_(pa.float32(), 7)),
            "observation.state": pa.array(state, type=pa.list_(pa.float32(), 7)),
            "episode_index": pa.array([0, 0], type=pa.int64()),
            "frame_index": pa.array([0, 1], type=pa.int64()),
        }
    )
    pq.write_table(data_table, source / "data" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.table({"episode_index": [0]}), source / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    (source / "videos" / "observation.images.camera1" / "chunk-000" / "file-000.mp4").write_bytes(b"video")
    info = {
        "total_episodes": 1,
        "total_frames": 2,
        "features": {
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"],
            }
        },
    }
    (source / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (source / "meta" / "stats.json").write_text(json.dumps({"action": {"mean": [0.0] * 7}}), encoding="utf-8")
    for path in source.rglob("*"):
        if path.is_file():
            path.chmod(0o444)

    report = convert_dataset(source, output)

    source_actions = pq.read_table(source / "data" / "chunk-000" / "file-000.parquet")["action"].to_pylist()
    converted_actions = pq.read_table(output / "data" / "chunk-000" / "file-000.parquet")["action"].to_pylist()
    np.testing.assert_allclose(source_actions, absolute_action, atol=1e-6)
    np.testing.assert_allclose(
        converted_actions,
        [[0.1, -0.2, 0.3, 0.0, -0.5, 0.6, 0.07], [-0.1, 0.4, 0.0, -0.3, 0.3, -0.4, 0.08]],
        atol=1e-6,
    )
    output_info = json.loads((output / "meta" / "info.json").read_text(encoding="utf-8"))
    assert output_info["features"]["action"]["names"] == [
        "delta_joint_1",
        "delta_joint_2",
        "delta_joint_3",
        "delta_joint_4",
        "delta_joint_5",
        "delta_joint_6",
        "gripper",
    ]
    assert (output / "videos" / "observation.images.camera1" / "chunk-000" / "file-000.mp4").read_bytes() == b"video"
    assert report["frames"] == 2
    assert report["episodes"] == 1
    assert report["reconstruction"]["max_abs_error"] < 1e-6
    verification = verify_dataset(source, output)
    assert verification["ok"] is True
    assert verification["frames"] == 2
    assert verification["episodes"] == 1
