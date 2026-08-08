from __future__ import annotations

import dataclasses

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.rlt.config import RLTActorCriticConfig
from openpi.rlt.config import RLTTokenConfig


def _make_mlp(input_dim: int, output_dim: int, hidden_dim: int, num_hidden_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for _ in range(num_hidden_layers):
        layers += [nn.Linear(last_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()]
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class RLTokenModule(nn.Module):
    """Encoder-decoder bottleneck used by RLT.

    The encoder appends a learned RL token to frozen VLA token embeddings and
    uses the final special-token output as the compact RL state. The decoder is
    trained to reconstruct the original embeddings autoregressively from that
    bottleneck, matching Eq. (1)-(2) in the RLT paper.
    """

    def __init__(self, config: RLTTokenConfig):
        super().__init__()
        self.config = config
        dim = config.embedding_dim
        ff_dim = int(dim * config.mlp_ratio)
        self.rl_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.encoder_pos = nn.Parameter(torch.zeros(1, config.max_seq_len + 1, dim))
        self.decoder_pos = nn.Parameter(torch.zeros(1, config.max_seq_len + 1, dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=config.num_heads,
            dim_feedforward=ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=config.num_heads,
            dim_feedforward=ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=config.num_encoder_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=config.num_decoder_layers)
        self.output = nn.Linear(dim, dim)
        nn.init.normal_(self.rl_token, std=0.02)
        nn.init.normal_(self.encoder_pos, std=0.02)
        nn.init.normal_(self.decoder_pos, std=0.02)

    def encode(self, embeddings: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, dim = embeddings.shape
        if dim != self.config.embedding_dim:
            raise ValueError(f"Expected embedding dim {self.config.embedding_dim}, got {dim}")
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.config.max_seq_len}")

        rl_token = self.rl_token.expand(batch_size, -1, -1)
        x = torch.cat([embeddings, rl_token], dim=1)
        x = x + self.encoder_pos[:, : seq_len + 1].to(dtype=x.dtype, device=x.device)
        if padding_mask is not None:
            rl_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=padding_mask.device)
            padding_mask = torch.cat([padding_mask, rl_mask], dim=1)
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        return encoded[:, -1]

    def reconstruct(self, rl_token: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = embeddings.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.config.max_seq_len}")

        memory = rl_token[:, None, :]
        start = torch.zeros(batch_size, 1, embeddings.shape[-1], dtype=embeddings.dtype, device=embeddings.device)
        decoder_in = torch.cat([start, embeddings[:, :-1].detach()], dim=1)
        decoder_in = decoder_in + self.decoder_pos[:, :seq_len].to(dtype=decoder_in.dtype, device=decoder_in.device)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=embeddings.device),
            diagonal=1,
        )
        decoded = self.decoder(decoder_in, memory, tgt_mask=causal_mask)
        return self.output(decoded)

    def forward(
        self, embeddings: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rl_token = self.encode(embeddings, padding_mask=padding_mask)
        reconstruction = self.reconstruct(rl_token, embeddings)
        return rl_token, reconstruction

    def reconstruction_loss(
        self, embeddings: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rl_token, reconstruction = self(embeddings, padding_mask=padding_mask)
        per_token = F.mse_loss(reconstruction, embeddings.detach(), reduction="none").mean(dim=-1)
        if padding_mask is not None:
            valid = (~padding_mask).to(per_token.dtype)
            loss = (per_token * valid).sum() / valid.sum().clamp_min(1.0)
        else:
            loss = per_token.mean()
        return loss, rl_token


class GaussianChunkActor(nn.Module):
    def __init__(self, rl_token_dim: int, config: RLTActorCriticConfig):
        super().__init__()
        self.config = config
        action_flat_dim = config.chunk_length * config.action_dim
        input_dim = rl_token_dim + config.state_dim + action_flat_dim
        self.net = _make_mlp(input_dim, action_flat_dim, config.hidden_dim, config.num_hidden_layers)
        self.register_buffer("std", torch.tensor(config.fixed_std, dtype=torch.float32))

    def mean(self, rl_token: torch.Tensor, state: torch.Tensor, reference_action: torch.Tensor) -> torch.Tensor:
        reference_flat = reference_action.reshape(reference_action.shape[0], -1)
        x = torch.cat([rl_token, state, reference_flat], dim=-1)
        raw = self.net(x).reshape(reference_action.shape)
        if getattr(self.config, "actor_output_mode", "absolute") == "residual":
            delta = torch.tanh(raw) * getattr(self.config, "residual_scale", 0.05)
            return reference_action + delta
        delta_clip = float(getattr(self.config, "absolute_delta_clip", 0.0))
        if delta_clip > 0.0:
            return reference_action + (raw - reference_action).clamp(-delta_clip, delta_clip)
        return raw

    def forward(
        self,
        rl_token: torch.Tensor,
        state: torch.Tensor,
        reference_action: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean(rl_token, state, reference_action)
        if deterministic:
            return mean, torch.zeros(mean.shape[0], dtype=mean.dtype, device=mean.device)
        noise = torch.randn_like(mean)
        action = mean + noise * self.std.to(dtype=mean.dtype, device=mean.device)
        log_prob = -0.5 * ((action - mean) / self.std).pow(2).flatten(1).sum(dim=-1)
        return action, log_prob


class TwinQCritic(nn.Module):
    def __init__(self, rl_token_dim: int, config: RLTActorCriticConfig):
        super().__init__()
        action_flat_dim = config.chunk_length * config.action_dim
        input_dim = rl_token_dim + config.state_dim + action_flat_dim
        self.q1 = _make_mlp(input_dim, 1, config.hidden_dim, config.num_hidden_layers)
        self.q2 = _make_mlp(input_dim, 1, config.hidden_dim, config.num_hidden_layers)

    def forward(self, rl_token: torch.Tensor, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action_flat = action.reshape(action.shape[0], -1)
        x = torch.cat([rl_token, state, action_flat], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def min_q(self, rl_token: torch.Tensor, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(rl_token, state, action)
        return torch.minimum(q1, q2)


@dataclasses.dataclass
class RLTBatch:
    rl_token: torch.Tensor
    state: torch.Tensor
    action: torch.Tensor
    reference_action: torch.Tensor
    reward: torch.Tensor
    discount: torch.Tensor
    next_rl_token: torch.Tensor
    next_state: torch.Tensor
    next_reference_action: torch.Tensor
