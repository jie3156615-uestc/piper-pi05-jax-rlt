from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image


def _recorded_observation(path: Path, timestep: int, prompt: str) -> tuple[dict, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    matches = [row for row in rows if int(row["t"]) == timestep]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one t={timestep} row in {path}, found {len(matches)}")
    row = matches[0]
    episode_root = path.parent

    def read_image(key: str) -> np.ndarray:
        relative = Path(str(row[key]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe {key} path: {relative}")
        with Image.open(episode_root / relative) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

    observation = {
        "observation/image": read_image("global_image"),
        "observation/wrist_image": read_image("wrist_image"),
        "observation/state": np.asarray(row["state"], dtype=np.float32),
        "prompt": prompt,
    }
    return observation, {"episode_jsonl": str(path.resolve()), "t": timestep, "logged_source": row.get("source")}


def main() -> None:
    parser = argparse.ArgumentParser(description="One read-only request to an RLT actor-shadow policy service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--episode-jsonl", type=Path)
    parser.add_argument("--t", type=int, default=0)
    parser.add_argument("--prompt", default="Put the green block into the box.")
    parser.add_argument("--require-actor", action="store_true")
    args = parser.parse_args()

    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    if args.episode_jsonl:
        observation, observation_source = _recorded_observation(
            args.episode_jsonl.expanduser(), args.t, args.prompt
        )
    else:
        rng = np.random.default_rng(20260710)
        observation = {
            "observation/image": rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8),
            "observation/wrist_image": rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8),
            "observation/state": np.zeros(7, dtype=np.float32),
            "prompt": args.prompt,
        }
        observation_source = {"synthetic_seed": 20260710}
    client = WebsocketClientPolicy(args.host, args.port, connect_timeout_s=30.0, request_timeout_s=30.0)
    started = time.monotonic()
    output = client.infer(observation)
    client_roundtrip_s = time.monotonic() - started
    actions = np.asarray(output.get("actions"), dtype=np.float32)
    z_rl = np.asarray(output.get("z_rl"), dtype=np.float32)
    a_actor = np.asarray(output.get("a_actor"), dtype=np.float32) if output.get("a_actor") is not None else None
    actor_minus_reference = None if a_actor is None else a_actor - actions[:10, :7]
    report = {
        "observation_source": observation_source,
        "client_roundtrip_s": client_roundtrip_s,
        "actions_shape": list(actions.shape),
        "actions_finite": bool(np.all(np.isfinite(actions))),
        "z_rl_shape": list(z_rl.shape),
        "z_rl_finite": bool(np.all(np.isfinite(z_rl))),
        "z_rl_std": float(np.std(z_rl)),
        "a_actor_shape": None if a_actor is None else list(a_actor.shape),
        "a_actor_finite": None if a_actor is None else bool(np.all(np.isfinite(a_actor))),
        "actor_minus_reference_abs_mean": (
            None if actor_minus_reference is None else float(np.mean(np.abs(actor_minus_reference)))
        ),
        "actor_minus_reference_abs_max": (
            None if actor_minus_reference is None else float(np.max(np.abs(actor_minus_reference)))
        ),
        "shadow": output.get("rlt_shadow", {}),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if actions.shape != (50, 7) or not report["actions_finite"]:
        raise RuntimeError("invalid Pi0.5 actions")
    if z_rl.shape != (2048,) or not report["z_rl_finite"]:
        raise RuntimeError("invalid real RL token")
    if report["shadow"].get("actor_controls_robot") is not False:
        raise RuntimeError("shadow service did not preserve actor_controls_robot=false")
    if args.require_actor and (a_actor is None or a_actor.shape != (10, 7) or not np.all(np.isfinite(a_actor))):
        raise RuntimeError("valid Actor shadow output was required but not returned")


if __name__ == "__main__":
    main()
