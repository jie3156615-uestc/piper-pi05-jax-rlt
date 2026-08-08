from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np


_ENCODER_PARAM_NAMES = {"rl_token", "encoder_pos", "z_norm"}


@dataclasses.dataclass(frozen=True)
class TokenCheckpointIdentity:
    format: str
    config_name: str | None
    checkpoint_dir: str | None
    step: int | None
    embedding_dim: int


class JaxRLTokenEncoder:
    """Encoder-only runtime for a frozen JAX RL-token checkpoint.

    Decoder parameters are ignored even when loading the original autoencoder
    checkpoint.  ``export_encoder_only_checkpoint`` can persist the filtered
    tree so the policy service never has to hold decoder weights.
    """

    def __init__(self, base_policy: Any, checkpoint_dir: str | Path) -> None:
        import jax
        import jax.numpy as jnp
        from flax import linen as nn
        from flax import serialization
        from flax.core import freeze
        from openpi.models import model as model_lib
        from openpi.rlt.jax_token import RLTTokenJaxConfig
        from openpi.rlt.jax_token import _EncoderBlock
        from openpi.shared import nnx_utils

        self.checkpoint_dir = Path(checkpoint_dir).expanduser()
        self.identity, cfg = read_token_checkpoint_identity(self.checkpoint_dir)
        params_path = self.checkpoint_dir / "params.msgpack"
        if not params_path.is_file():
            raise FileNotFoundError(f"missing RL-token params: {params_path}")
        raw_params = serialization.msgpack_restore(params_path.read_bytes())
        if "params" in raw_params and isinstance(raw_params["params"], dict):
            raw_params = raw_params["params"]
        encoder_params = _filter_encoder_params(raw_params)

        class _EncoderOnly(nn.Module):
            config: RLTTokenJaxConfig

            @nn.compact
            def __call__(self, embeddings, valid_mask):
                token_cfg = self.config
                batch_size, seq_len, dim = embeddings.shape
                if dim != token_cfg.embedding_dim:
                    raise ValueError(f"expected embedding_dim={token_cfg.embedding_dim}, got {dim}")
                if seq_len > token_cfg.max_seq_len:
                    raise ValueError(f"sequence length {seq_len} exceeds {token_cfg.max_seq_len}")
                token = self.param("rl_token", nn.initializers.normal(0.02), (1, 1, dim))
                encoder_pos = self.param(
                    "encoder_pos", nn.initializers.normal(0.02), (1, token_cfg.max_seq_len + 1, dim)
                )
                x = jnp.concatenate([embeddings, jnp.broadcast_to(token, (batch_size, 1, dim))], axis=1)
                x = x + encoder_pos[:, : seq_len + 1]
                encoder_valid = jnp.concatenate(
                    [valid_mask, jnp.ones((batch_size, 1), dtype=bool)], axis=1
                )
                for layer in range(token_cfg.num_encoder_layers):
                    x = _EncoderBlock(token_cfg, name=f"encoder_{layer}")(x, encoder_valid, train=False)
                return nn.LayerNorm(name="z_norm")(x[:, -1])

        token_cfg = RLTTokenJaxConfig(**cfg)
        encoder_model = _EncoderOnly(token_cfg)
        frozen_params = freeze(encoder_params)
        self._token_apply = jax.jit(lambda embeddings, valid: encoder_model.apply({"params": frozen_params}, embeddings, valid))
        model = getattr(base_policy, "_model", None)
        transform = getattr(base_policy, "_input_transform", None)
        if model is None or transform is None or not hasattr(model, "encode_prefix_tokens"):
            raise TypeError("base policy must expose JAX _model.encode_prefix_tokens and _input_transform")
        self._input_transform = transform
        self._observation_type = model_lib.Observation
        self._extract_prefix = nnx_utils.module_jit(model.encode_prefix_tokens)
        self._jax = jax
        self._jnp = jnp

    def encode(self, observation: dict[str, Any]) -> np.ndarray:
        inputs = self._jax.tree.map(lambda x: x, observation)
        inputs = self._input_transform(inputs)
        inputs = self._jax.tree.map(
            lambda x: self._jnp.asarray(x)[np.newaxis, ...],
            inputs,
        )
        model_observation = self._observation_type.from_dict(inputs)
        embeddings, valid = self._extract_prefix(model_observation)
        z_rl = self._token_apply(embeddings, valid)
        z_rl = np.asarray(self._jax.device_get(z_rl), dtype=np.float32)
        if z_rl.shape != (1, self.identity.embedding_dim):
            raise RuntimeError(
                f"unexpected token encoder output shape: {z_rl.shape}"
            )
        return z_rl[0]

    def encode_batch(
        self,
        observations: list[dict[str, Any]],
    ) -> np.ndarray:
        """Encode one wire batch while preserving the established token values.

        A larger leading JAX batch changes GPU reduction order and measurably
        perturbs the frozen encoder output.  Online replay must remain in the
        same feature distribution as the existing cache, so requests are
        batched over one WebSocket exchange but each item retains the exact
        batch-size-one encoder path.
        """

        if not observations:
            raise ValueError("RL-token batch must not be empty")
        z_rl = np.stack(
            [self.encode(observation) for observation in observations],
            axis=0,
        ).astype(np.float32, copy=False)
        expected_shape = (len(observations), self.identity.embedding_dim)
        if z_rl.shape != expected_shape:
            raise RuntimeError(f"unexpected token encoder output shape: {z_rl.shape}")
        return z_rl


def read_token_checkpoint_identity(checkpoint_dir: str | Path) -> tuple[TokenCheckpointIdentity, dict[str, Any]]:
    checkpoint_dir = Path(checkpoint_dir).expanduser()
    metadata_path = checkpoint_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    token_cfg = dict(metadata["token_config"])
    args = metadata.get("args", {})
    identity = TokenCheckpointIdentity(
        format=str(metadata.get("format", "")),
        config_name=args.get("config_name"),
        checkpoint_dir=args.get("checkpoint_dir"),
        step=None if metadata.get("step") is None else int(metadata["step"]),
        embedding_dim=int(token_cfg["embedding_dim"]),
    )
    if identity.format not in {"jax_flax_rl_token_autoencoder_v1", "jax_flax_rl_token_encoder_v1"}:
        raise ValueError(f"unsupported RL-token checkpoint format: {identity.format!r}")
    return identity, token_cfg


def export_encoder_only_checkpoint(source_dir: str | Path, output_dir: str | Path) -> Path:
    from flax import serialization

    source_dir = Path(source_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    identity, _ = read_token_checkpoint_identity(source_dir)
    params = serialization.msgpack_restore((source_dir / "params.msgpack").read_bytes())
    wrapped = "params" in params and isinstance(params["params"], dict)
    raw = params["params"] if wrapped else params
    filtered = _filter_encoder_params(raw)
    payload = {"params": filtered} if wrapped else filtered
    metadata = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata.update(
        {
            "format": "jax_flax_rl_token_encoder_v1",
            "source_format": identity.format,
            "source_checkpoint": str(source_dir),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "params.msgpack").write_bytes(serialization.msgpack_serialize(payload))
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_dir


def _filter_encoder_params(params: dict[str, Any]) -> dict[str, Any]:
    filtered = {
        key: value
        for key, value in params.items()
        if key in _ENCODER_PARAM_NAMES or str(key).startswith("encoder_")
    }
    required = {"rl_token", "encoder_pos", "z_norm"}
    missing = sorted(required - set(filtered))
    if missing or not any(str(key).startswith("encoder_") for key in filtered):
        raise ValueError(f"RL-token checkpoint is missing encoder parameters: {missing}")
    return filtered
