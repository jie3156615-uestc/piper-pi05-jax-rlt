from __future__ import annotations

import copy
import dataclasses
import math

import torch
import torch.nn.functional as F  # noqa: N812

from openpi.rlt.config import RLTActorCriticConfig
from openpi.rlt.networks import GaussianChunkActor
from openpi.rlt.networks import RLTBatch
from openpi.rlt.networks import TwinQCritic


class RLTAgent:
    """TD3-style actor-critic used for RLT online training."""

    def __init__(self, rl_token_dim: int, config: RLTActorCriticConfig, device: torch.device):
        config = _normalize_actor_critic_config(config)
        self.config = config
        self.device = device
        self.actor = GaussianChunkActor(rl_token_dim, config).to(device)
        self.critic = TwinQCritic(rl_token_dim, config).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=config.critic_lr)
        self.num_updates = 0

    @torch.no_grad()
    def act(
        self,
        rl_token: torch.Tensor,
        state: torch.Tensor,
        reference_action: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> torch.Tensor:
        self.actor.eval()
        action, _ = self.actor(rl_token, state, reference_action, deterministic=deterministic)
        self.actor.train()
        return action

    def update(self, batch: RLTBatch) -> dict[str, float]:
        cfg = self.config
        with torch.no_grad():
            next_ref = batch.next_reference_action
            next_action, _ = self.actor_target(batch.next_rl_token, batch.next_state, next_ref)
            if cfg.target_policy_noise > 0:
                noise = torch.randn_like(next_action) * cfg.target_policy_noise
                noise = noise.clamp(-cfg.target_noise_clip, cfg.target_noise_clip)
                next_action = next_action + noise
            bootstrap_q = self.critic_target.min_q(batch.next_rl_token, batch.next_state, next_action)
            target = batch.reward + batch.discount * bootstrap_q

        q1, q2 = self.critic(batch.rl_token, batch.state, batch.action)
        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)
        critic_loss = q1_loss + q2_loss
        critic_finite = bool(torch.isfinite(critic_loss).detach().cpu())
        if critic_finite:
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip_norm)
            self.critic_optimizer.step()

        with torch.no_grad():
            action_delta = batch.action - batch.reference_action
            td_error = 0.5 * ((q1 - target).abs() + (q2 - target).abs())
        info = {
            "critic_updated": critic_finite,
            "actor_updated": False,
            "critic_loss": float(critic_loss.detach().cpu()),
            "critic_q1_loss": float(q1_loss.detach().cpu()),
            "critic_q2_loss": float(q2_loss.detach().cpu()),
            "q1": float(q1.mean().detach().cpu()),
            "q2": float(q2.mean().detach().cpu()),
            "q_min": float(torch.minimum(q1, q2).mean().detach().cpu()),
            "q_gap": float((q1 - q2).abs().mean().detach().cpu()),
            "bootstrap_q": float(bootstrap_q.mean().detach().cpu()),
            "target_mean": float(target.mean().detach().cpu()),
            "target_max": float(target.max().detach().cpu()),
            "td_error_abs": float(td_error.mean().detach().cpu()),
            "batch_reward_mean": float(batch.reward.mean().detach().cpu()),
            "batch_reward_positive_frac": float((batch.reward > 0).float().mean().detach().cpu()),
            "batch_discount_mean": float(batch.discount.mean().detach().cpu()),
            "batch_action_delta_l2": float(action_delta.pow(2).sum(dim=(-1, -2)).sqrt().mean().detach().cpu()),
            "batch_action_delta_abs_max": float(action_delta.abs().amax().detach().cpu()),
        }

        if self.num_updates % cfg.policy_delay == 0:
            reference_action = batch.reference_action
            if cfg.reference_dropout > 0:
                keep = torch.rand(reference_action.shape[0], 1, 1, device=reference_action.device) > cfg.reference_dropout
                actor_reference = reference_action * keep.to(reference_action.dtype)
            else:
                actor_reference = reference_action
            actor_action, _ = self.actor(batch.rl_token, batch.state, actor_reference)
            actor_q = self.critic.min_q(batch.rl_token, batch.state, actor_action)
            bc_loss = F.mse_loss(actor_action, reference_action)
            actor_rl_loss = -actor_q.mean()
            actor_bc_loss = cfg.beta_bc * bc_loss
            actor_loss = actor_rl_loss + actor_bc_loss
            actor_finite = bool(torch.isfinite(actor_loss).detach().cpu())
            if actor_finite:
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip_norm)
                self.actor_optimizer.step()
                self._soft_update(self.actor_target, self.actor)
                self._soft_update(self.critic_target, self.critic)
            with torch.no_grad():
                actor_delta = actor_action - reference_action
            info.update(
                {
                    "actor_updated": actor_finite,
                    "actor_loss": float(actor_loss.detach().cpu()),
                    "actor_rl_loss": float(actor_rl_loss.detach().cpu()),
                    "actor_bc_weighted_loss": float(actor_bc_loss.detach().cpu()),
                    "actor_q": float(actor_q.mean().detach().cpu()),
                    "bc_loss": float(bc_loss.detach().cpu()),
                    "actor_action_delta_l2": float(actor_delta.pow(2).sum(dim=(-1, -2)).sqrt().mean().detach().cpu()),
                    "actor_action_delta_abs_max": float(actor_delta.abs().amax().detach().cpu()),
                }
            )

        if any(isinstance(value, float) and not math.isfinite(value) for value in info.values()):
            info["nonfinite_detected"] = True
        else:
            info["nonfinite_detected"] = False
        self.num_updates += 1
        return info

    def _soft_update(self, target: torch.nn.Module, source: torch.nn.Module) -> None:
        tau = self.config.tau
        with torch.no_grad():
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "num_updates": self.num_updates,
            "config": self.config,
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_target.load_state_dict(state.get("actor_target", state["actor"]))
        self.critic_target.load_state_dict(state.get("critic_target", state["critic"]))
        if "actor_optimizer" in state:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        if "critic_optimizer" in state:
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.num_updates = int(state.get("num_updates", 0))


def _normalize_actor_critic_config(config: RLTActorCriticConfig | dict) -> RLTActorCriticConfig:
    if isinstance(config, dict):
        allowed = {field.name for field in dataclasses.fields(RLTActorCriticConfig)}
        return RLTActorCriticConfig(**{key: value for key, value in config.items() if key in allowed})
    for field in dataclasses.fields(RLTActorCriticConfig):
        if not hasattr(config, field.name):
            object.__setattr__(config, field.name, field.default)
    return config
