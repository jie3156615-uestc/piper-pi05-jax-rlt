"""Load, warm, and serve the Piper JAX policy on localhost."""

import logging
import os
import time

import numpy as np


DEFAULT_CONFIG_NAME = 'pi05_piper_greenblock_5090_lora_delta'
DEFAULT_CHECKPOINT_RUN_DIR = '/home/cwzk/openpi_jax_piper_lora_v1_20260707/checkpoints/pi05_piper_greenblock_5090_lora_delta/piper_greenblock_5090_lora_delta_30k_official_pi05_base_20260707'
DEFAULT_CHECKPOINT_STEP = '49999'
DEFAULT_PROMPT = 'Put the green block into the box.'


def resolve_config_name() -> str:
    return os.environ.get('PIPER_POLICY_CONFIG', DEFAULT_CONFIG_NAME)


def resolve_checkpoint() -> str:
    explicit = os.environ.get('PIPER_POLICY_CHECKPOINT')
    if explicit:
        return explicit
    step = os.environ.get('PIPER_POLICY_STEP', DEFAULT_CHECKPOINT_STEP)
    return os.path.join(DEFAULT_CHECKPOINT_RUN_DIR, str(step))


def resolve_prompt() -> str:
    return os.environ.get('PIPER_POLICY_PROMPT', DEFAULT_PROMPT)


CONFIG_NAME = resolve_config_name()
CHECKPOINT = resolve_checkpoint()
PROMPT = resolve_prompt()


def make_warmup_observation() -> dict:
    rng = np.random.default_rng(0)
    return {
        "observation/image": rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8),
        "observation/state": np.zeros(7, dtype=np.float32),
        "prompt": PROMPT,
    }


def make_server_metadata(policy_metadata: dict) -> dict:
    metadata = dict(policy_metadata)
    metadata["prompt"] = PROMPT
    return metadata


def main() -> None:
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config

    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("Loading config=%s checkpoint=%s", CONFIG_NAME, CHECKPOINT)
    policy = policy_config.create_trained_policy(config.get_config(CONFIG_NAME), CHECKPOINT, default_prompt=PROMPT)
    started = time.monotonic()
    output = policy.infer(make_warmup_observation())
    actions = np.asarray(output["actions"])
    if actions.shape[0] != 50 or actions.shape[1] < 7 or not np.all(np.isfinite(actions[:, :7])):
        raise RuntimeError("warmup returned invalid actions: shape=%r" % (actions.shape,))
    logging.info("Policy warmup complete in %.3fs action_shape=%s", time.monotonic() - started, actions.shape)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="127.0.0.1",
        port=8000,
        metadata=make_server_metadata(policy.metadata),
    ).serve_forever()


if __name__ == "__main__":
    main()
