"""Serve full-20k Pi0.5 with fail-open RL-token/Actor shadow fields.

This service has no ROS or hardware dependency.  It intentionally uses port
8001 by default so it can be verified beside the existing Pi0.5 service before
an operator chooses to point a rollout at it.
"""

from __future__ import annotations

import importlib
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from piper_runtime.rlt_actor_protocol import ACTION_SCHEMA_FINGERPRINT
from piper_runtime.rlt_actor_protocol import ACTOR_PROJECTION_PROFILE
from piper_runtime.rlt_actor_protocol import (
    ACTOR_PROJECTION_PROFILE_BY_ACTION_SCHEMA,
)
from piper_runtime.rlt_shadow_policy import PassThroughShadowActor
from piper_runtime.rlt_shadow_policy import ShadowAugmentedPolicy
from piper_runtime.rlt_shadow_policy import ShadowPolicyConfig
from piper_runtime.rlt_token_runtime import JaxRLTokenEncoder


DEFAULT_CONFIG_NAME = "pi05_piper_greenblock_5090_jax_delta_v1"
DEFAULT_CHECKPOINT = (
    "/home/cwzk/openpi_checkpoints/pi05_piper_greenblock_5090_jax_delta_v1/"
    "piper_greenblock_5090_delta_sft_30k_20260707/20000"
)
DEFAULT_TOKEN_CHECKPOINT = (
    "/home/cwzk/openpi_rlt/rl_tokens/"
    "pi05_piper_greenblock_5090_full20k_rlt_token_v1_20260709/step_010000"
)
DEFAULT_PROMPT = "Put the green block into the box."


def main() -> None:
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as training_config

    logging.basicConfig(level=logging.INFO, force=True)
    config_name = os.environ.get("PIPER_POLICY_CONFIG", DEFAULT_CONFIG_NAME)
    checkpoint = os.environ.get("PIPER_POLICY_CHECKPOINT", DEFAULT_CHECKPOINT)
    token_checkpoint = os.environ.get("PIPER_RL_TOKEN_CHECKPOINT", DEFAULT_TOKEN_CHECKPOINT)
    prompt = os.environ.get("PIPER_POLICY_PROMPT", DEFAULT_PROMPT)
    host = os.environ.get("PIPER_RLT_SHADOW_HOST", "127.0.0.1")
    port = int(os.environ.get("PIPER_RLT_SHADOW_PORT", "8001"))
    actor_mode = os.environ.get("PIPER_RLT_ACTOR_MODE", "none").strip().lower()

    logging.info("Loading frozen full-20k config=%s checkpoint=%s", config_name, checkpoint)
    base_policy = policy_config.create_trained_policy(
        training_config.get_config(config_name), checkpoint, default_prompt=prompt
    )
    token_encoder = JaxRLTokenEncoder(base_policy, token_checkpoint)
    _validate_token_identity(token_encoder, config_name=config_name, checkpoint=checkpoint)
    actor, actor_name = _load_actor(actor_mode)
    actor_action_schema, actor_projection_profile = _resolve_actor_wire_contract(
        actor
    )
    policy = ShadowAugmentedPolicy(
        base_policy,
        token_encoder=token_encoder,
        actor=actor,
        config=ShadowPolicyConfig(
            actor_action_schema_fingerprint=actor_action_schema,
            actor_projection_profile=actor_projection_profile,
        ),
        actor_name=actor_name,
    )

    started = time.monotonic()
    output = policy.infer(_make_warmup_observation(prompt))
    actions = np.asarray(output["actions"])
    if actions.shape[0] < 10 or actions.shape[1] < 7 or not np.all(np.isfinite(actions[:10, :7])):
        raise RuntimeError(f"warmup returned invalid base actions: {actions.shape}")
    z_rl = np.asarray(output.get("z_rl"))
    if z_rl.shape != (2048,) or not np.all(np.isfinite(z_rl)):
        raise RuntimeError(f"warmup did not return a valid real z_rl: {z_rl.shape}")
    logging.info(
        "Shadow warmup complete in %.3fs base=%s token=%s actor=%s",
        time.monotonic() - started,
        actions.shape,
        z_rl.shape,
        output["rlt_shadow"].get("actor_status"),
    )
    metadata = dict(policy.metadata)
    metadata.update(
        {
            "prompt": prompt,
            "base_config": config_name,
            "base_checkpoint": checkpoint,
            "token_checkpoint": token_checkpoint,
        }
    )
    if actor is not None and callable(
        provenance_metadata := getattr(actor, "provenance_metadata", None)
    ):
        metadata.update(provenance_metadata())
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=host,
        port=port,
        metadata=metadata,
    ).serve_forever()


def _load_actor(mode: str) -> tuple[Any | None, str]:
    if mode in {"", "none", "token-only"}:
        return None, "none"
    if mode in {"mock", "dry", "passthrough"}:
        return PassThroughShadowActor(), "mock_passthrough"
    if mode != "checkpoint":
        raise ValueError("PIPER_RLT_ACTOR_MODE must be none, mock, or checkpoint")
    checkpoint = os.environ.get("PIPER_RLT_ACTOR_CHECKPOINT")
    factory_spec = os.environ.get(
        "PIPER_RLT_ACTOR_FACTORY", "openpi.rlt.real.actor_runtime:create_shadow_actor"
    )
    if not checkpoint or not Path(checkpoint).expanduser().exists():
        raise FileNotFoundError("checkpoint actor mode requires an existing PIPER_RLT_ACTOR_CHECKPOINT")
    if ":" not in factory_spec:
        raise ValueError("PIPER_RLT_ACTOR_FACTORY must use module:function syntax")
    module_name, function_name = factory_spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    return factory(Path(checkpoint).expanduser()), f"checkpoint:{checkpoint}"


def _resolve_actor_wire_contract(actor: Any | None) -> tuple[str, str]:
    """Bind service metadata to the loaded Actor and launcher contract."""

    launcher_schema = os.environ.get("PIPER_RLT_EXPECTED_RAW_SCHEMA", "").strip()
    launcher_projection = os.environ.get(
        "PIPER_RLT_EXPECTED_PROJECTION_PROFILE", ""
    ).strip()
    # A fresh-zero warmup intentionally has no Actor checkpoint.  In that
    # state there is no loaded Actor whose class attributes can define the wire
    # contract, so bind the token-only service to the already-audited launcher
    # contract.  This does not enable an Actor; it only keeps service metadata
    # compatible with the gripper-v3 rollout that will consume Pi0.5 + z_rl.
    if actor is None and (launcher_schema or launcher_projection):
        if not launcher_schema or not launcher_projection:
            raise ValueError(
                "Actor-none mode requires both launcher wire schema and projection"
            )
        expected_projection = ACTOR_PROJECTION_PROFILE_BY_ACTION_SCHEMA.get(
            launcher_schema
        )
        if expected_projection is None:
            raise ValueError(
                f"launcher advertises unsupported wire schema {launcher_schema!r}"
            )
        if launcher_projection != expected_projection:
            raise ValueError(
                "launcher wire schema/projection mismatch: "
                f"{launcher_projection!r} != {expected_projection!r}"
            )
        return launcher_schema, launcher_projection

    actor_schema = str(
        getattr(actor, "output_action_schema_fingerprint", ACTION_SCHEMA_FINGERPRINT)
    )
    actor_projection = str(
        getattr(
            actor,
            "output_actor_projection_profile",
            ACTOR_PROJECTION_PROFILE_BY_ACTION_SCHEMA.get(
                actor_schema, ACTOR_PROJECTION_PROFILE
            ),
        )
    )
    expected_projection = ACTOR_PROJECTION_PROFILE_BY_ACTION_SCHEMA.get(
        actor_schema
    )
    if expected_projection is None:
        raise ValueError(
            f"loaded Actor advertises unsupported wire schema {actor_schema!r}"
        )
    if actor_projection != expected_projection:
        raise ValueError(
            "loaded Actor wire schema/projection mismatch: "
            f"{actor_projection!r} != {expected_projection!r}"
        )

    if launcher_schema and launcher_schema != actor_schema:
        raise ValueError(
            "launcher/loaded Actor wire schema mismatch: "
            f"{launcher_schema!r} != {actor_schema!r}"
        )
    if launcher_projection and launcher_projection != actor_projection:
        raise ValueError(
            "launcher/loaded Actor projection mismatch: "
            f"{launcher_projection!r} != {actor_projection!r}"
        )
    return actor_schema, actor_projection


def _validate_token_identity(token_encoder: JaxRLTokenEncoder, *, config_name: str, checkpoint: str) -> None:
    identity = token_encoder.identity
    if identity.config_name and identity.config_name != config_name:
        raise ValueError(f"RL-token config mismatch: {identity.config_name!r} != {config_name!r}")
    token_base = Path(identity.checkpoint_dir).name if identity.checkpoint_dir else None
    if token_base and token_base != Path(checkpoint).name:
        raise ValueError(f"RL-token base checkpoint mismatch: step {token_base!r} != {Path(checkpoint).name!r}")


def _make_warmup_observation(prompt: str) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    return {
        "observation/image": rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8),
        "observation/state": np.zeros(7, dtype=np.float32),
        "prompt": prompt,
    }


if __name__ == "__main__":
    main()
