from __future__ import annotations

import dataclasses

from flax import linen as nn
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class RLTTokenJaxConfig:
    """JAX RL-token bottleneck config used by Piper RLT."""

    embedding_dim: int = 2048
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_seq_len: int = 1024


class _EncoderBlock(nn.Module):
    config: RLTTokenJaxConfig

    @nn.compact
    def __call__(self, x: jax.Array, valid_mask: jax.Array, *, train: bool) -> jax.Array:
        cfg = self.config
        attn_mask = valid_mask[:, None, None, :]
        y = nn.LayerNorm()(x)
        y = nn.SelfAttention(
            num_heads=cfg.num_heads,
            qkv_features=cfg.embedding_dim,
            out_features=cfg.embedding_dim,
            dropout_rate=cfg.dropout,
            deterministic=not train,
        )(y, mask=attn_mask)
        x = x + y
        y = nn.LayerNorm()(x)
        y = nn.Dense(int(cfg.embedding_dim * cfg.mlp_ratio))(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=cfg.dropout, deterministic=not train)(y)
        y = nn.Dense(cfg.embedding_dim)(y)
        return x + y


class _DecoderBlock(nn.Module):
    config: RLTTokenJaxConfig

    @nn.compact
    def __call__(self, x: jax.Array, z: jax.Array, valid_mask: jax.Array, *, train: bool) -> jax.Array:
        cfg = self.config
        seq_len = x.shape[1]
        causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        self_mask = causal[None, None, :, :] & valid_mask[:, None, None, :]
        y = nn.LayerNorm()(x)
        y = nn.SelfAttention(
            num_heads=cfg.num_heads,
            qkv_features=cfg.embedding_dim,
            out_features=cfg.embedding_dim,
            dropout_rate=cfg.dropout,
            deterministic=not train,
        )(y, mask=self_mask)
        x = x + y
        y = nn.LayerNorm()(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=cfg.num_heads,
            qkv_features=cfg.embedding_dim,
            out_features=cfg.embedding_dim,
            dropout_rate=cfg.dropout,
            deterministic=not train,
        )(y, z[:, None, :])
        x = x + y
        y = nn.LayerNorm()(x)
        y = nn.Dense(int(cfg.embedding_dim * cfg.mlp_ratio))(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=cfg.dropout, deterministic=not train)(y)
        y = nn.Dense(cfg.embedding_dim)(y)
        return x + y


class RLTokenAutoencoder(nn.Module):
    """Encoder-decoder bottleneck trained on frozen JAX OpenPI prefix tokens."""

    config: RLTTokenJaxConfig

    @nn.compact
    def __call__(self, embeddings: jax.Array, valid_mask: jax.Array, *, train: bool = False):
        cfg = self.config
        batch_size, seq_len, dim = embeddings.shape
        if dim != cfg.embedding_dim:
            raise ValueError(f"Expected embedding_dim={cfg.embedding_dim}, got {dim}")
        if seq_len > cfg.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={cfg.max_seq_len}")

        rl_token = self.param("rl_token", nn.initializers.normal(0.02), (1, 1, dim))
        encoder_pos = self.param("encoder_pos", nn.initializers.normal(0.02), (1, cfg.max_seq_len + 1, dim))
        decoder_pos = self.param("decoder_pos", nn.initializers.normal(0.02), (1, cfg.max_seq_len, dim))

        token = jnp.broadcast_to(rl_token, (batch_size, 1, dim))
        x = jnp.concatenate([embeddings, token], axis=1)
        x = x + encoder_pos[:, : seq_len + 1]
        encoder_valid = jnp.concatenate([valid_mask, jnp.ones((batch_size, 1), dtype=bool)], axis=1)
        for layer in range(cfg.num_encoder_layers):
            x = _EncoderBlock(cfg, name=f"encoder_{layer}")(x, encoder_valid, train=train)
        z_rl = nn.LayerNorm(name="z_norm")(x[:, -1])

        start = jnp.zeros((batch_size, 1, dim), dtype=embeddings.dtype)
        decoder_in = jnp.concatenate([start, jax.lax.stop_gradient(embeddings[:, :-1])], axis=1)
        y = decoder_in + decoder_pos[:, :seq_len]
        for layer in range(cfg.num_decoder_layers):
            y = _DecoderBlock(cfg, name=f"decoder_{layer}")(y, z_rl, valid_mask, train=train)
        reconstruction = nn.Dense(dim, name="output")(nn.LayerNorm(name="decoder_out_norm")(y))
        return z_rl, reconstruction


def reconstruction_loss(reconstruction: jax.Array, target: jax.Array, valid_mask: jax.Array) -> jax.Array:
    per_token = jnp.mean(jnp.square(reconstruction - jax.lax.stop_gradient(target)), axis=-1)
    valid = valid_mask.astype(per_token.dtype)
    return jnp.sum(per_token * valid) / jnp.maximum(jnp.sum(valid), 1.0)
