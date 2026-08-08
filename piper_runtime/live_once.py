"""One real-input, hardware-read-only Piper policy inference."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from piper_runtime.cameras import DualRealSenseReader
from piper_runtime.dry_run_client import execute_dry_run, make_mock_profile
from piper_runtime.piper_feedback import PiperFeedbackReader


def connect_read_only(piper) -> None:
    piper.ConnectPort(False, False, True)


def read_can_tx_packets() -> int:
    return int(Path("/sys/class/net/can0/statistics/tx_packets").read_text().strip())


class RecordingPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.response = None
        self.elapsed_s = None

    def infer(self, observation):
        started = time.monotonic()
        self.response = self.policy.infer(observation)
        self.elapsed_s = time.monotonic() - started
        return self.response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from piper_sdk import C_PiperInterface_V2

    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    tx_before = read_can_tx_packets()
    piper = C_PiperInterface_V2("can0")
    connect_read_only(piper)
    try:
        feedback = PiperFeedbackReader(piper)
        feedback.wait_until_healthy(timeout_s=5.0)
        with DualRealSenseReader() as cameras:
            images = cameras.read(timeout_ms=5000, warmup_frames=60)
            state = feedback.read()
        policy = RecordingPolicy(WebsocketClientPolicy("127.0.0.1", 8000))
        result = execute_dry_run(
            policy=policy,
            images=images,
            state_snapshot=state,
            profile=make_mock_profile(),
            audit_path=args.audit,
        )
    finally:
        piper.DisconnectPort()
    tx_after = read_can_tx_packets()
    actions = np.asarray(policy.response["actions"])
    report = {
        "camera": {
            name: {
                "shape": list(image.shape),
                "dtype": str(image.dtype),
                "mean": float(image.mean()),
                "min": int(image.min()),
                "max": int(image.max()),
            }
            for name, image in images.items()
        },
        "state": state.astype(float).tolist(),
        "action_shape": list(actions.shape),
        "action_finite": bool(np.all(np.isfinite(actions[:, :7]))),
        "inference_s": policy.elapsed_s,
        "server_metadata": policy.policy.get_server_metadata(),
        "dry_run_emitted_count": len(result.emitted),
        "dry_run_stop_reason": result.stop_reason,
        "audit_path": str(args.audit),
        "can_tx_before": tx_before,
        "can_tx_after": tx_after,
        "can_tx_delta": tx_after - tx_before,
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, default=str))
    if report["can_tx_delta"] != 0:
        raise RuntimeError("read-only client changed CAN TX counter")


if __name__ == "__main__":
    main()
