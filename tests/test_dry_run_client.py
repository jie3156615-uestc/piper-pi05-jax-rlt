from pathlib import Path

import numpy as np

from piper_runtime.dry_run_client import execute_dry_run, make_mock_profile


class FakePolicy:
    def __init__(self, state):
        self.state = state

    def infer(self, observation):
        actions = np.zeros((50, 32), dtype=np.float32)
        actions[:, :6] = self.state[:6]
        actions[:, 6] = 0.02
        return {"actions": actions}


def test_execute_dry_run_writes_audit_without_hardware_emitter(tmp_path):
    images = {
        "camera1": np.full((480, 640, 3), 10, dtype=np.uint8),
        "camera2": np.full((480, 640, 3), 20, dtype=np.uint8),
    }
    state = np.zeros(7, dtype=np.float32)
    audit_path = tmp_path / "audit.jsonl"
    result = execute_dry_run(
        policy=FakePolicy(state),
        images=images,
        state_snapshot=state,
        profile=make_mock_profile(),
        audit_path=audit_path,
        now=100.0,
    )
    assert result.stop_reason is None
    assert len(result.emitted) == 50
    assert len(audit_path.read_text().splitlines()) == 50
    source = Path(__import__("piper_runtime.dry_run_client").dry_run_client.__file__).read_text()
    assert "PiperSDKEmitter" not in source
    assert "JointCtrl" not in source
    assert "GripperCtrl" not in source
