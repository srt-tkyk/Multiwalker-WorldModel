"""Shared-policy Dreamer-style agent for Multiwalker."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoders import ObservationEncoder
from src.models.decoders import ObservationDecoder, RewardDecoder, TerminationDecoder
from src.models.heads import ActorHead, ValueHead
from src.models.rssm import RSSM, RSSMState
from src.utils.replay_buffer import ReplayBuffer


@dataclass
class DreamerParams:
    deter_size: int
    stoch_size: int
    hidden_size: int
    obs_encoder: Tuple[int, ...]
    obs_decoder: Tuple[int, ...]
    reward_decoder: Tuple[int, ...]
    actor_head: Tuple[int, ...]
    value_head: Tuple[int, ...]
    world_lr: float
    actor_lr: float
    value_lr: float
    grad_clip: float
    entropy_scale: float
    discount: float
    lambda_return: float
    kl_scale: float


class DreamerSharedAgent:
    """Minimal Dreamer-style agent with shared actor across homogeneous walkers."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        params: DreamerParams,
        device: torch.device,
    ) -> None:
        self.device = device
        self.params = params
        embed_dim = params.hidden_size
        self.encoder = ObservationEncoder(obs_dim, params.obs_encoder, embed_dim).to(device)
        self.rssm = RSSM(
            deter_size=params.deter_size,
            stoch_size=params.stoch_size,
            hidden_size=params.hidden_size,
            action_dim=action_dim,
            embed_dim=embed_dim,
        ).to(device)
        latent_dim = params.deter_size + params.stoch_size
        self.obs_decoder = ObservationDecoder(latent_dim, params.obs_decoder, obs_dim).to(device)
        self.reward_decoder = RewardDecoder(latent_dim, params.reward_decoder).to(device)
        self.termination_decoder = TerminationDecoder(latent_dim, params.reward_decoder).to(device)
        self.actor = ActorHead(latent_dim, params.actor_head, action_dim, min_std=0.1, max_std=1.0).to(device)
        self.critic = ValueHead(latent_dim, params.value_head).to(device)
        self.world_optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.rssm.parameters())
            + list(self.obs_decoder.parameters())
            + list(self.reward_decoder.parameters())
            + list(self.termination_decoder.parameters()),
            lr=params.world_lr,
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=params.actor_lr)
        self.value_optimizer = torch.optim.Adam(self.critic.parameters(), lr=params.value_lr)
        self._prev_action = None
        self._state = None
        self._last_features: torch.Tensor | None = None

    def policy(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        obs = obs.to(self.device)
        emb = self.encoder(obs)
        if self._state is None:
            batch_size = emb.shape[0]
            self._state = self.rssm.initial_state(batch_size, self.device)
        if self._prev_action is None:
            self._prev_action = torch.zeros(obs.shape[0], self.actor._action_dim, device=self.device)
        prior_state = self.rssm.img_step(self._state, self._prev_action)
        post_state = self.rssm.obs_step(prior_state, emb)
        features = self.rssm.get_features(post_state)
        dist = self.actor(features)
        action = dist.mean if deterministic else dist.sample()
        self._prev_action = action.detach()
        self._state = post_state.detach()
        self._last_features = features.detach()
        return action.tanh()

    def reset_state(self) -> None:
        self._state = None
        self._prev_action = None
        self._last_features = None

    def last_features(self) -> torch.Tensor | None:
        return self._last_features

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"].to(self.device)  # [B, T, obs_dim]
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        discounts = batch["discounts"].to(self.device)
        batch_size, horizon, _ = obs.shape

        embed = self.encoder(obs.view(-1, obs.shape[-1])).view(batch_size, horizon, -1)
        embed = embed.permute(1, 0, 2)
        actions_time = actions.permute(1, 0, 2)
        state0 = self.rssm.initial_state(batch_size, self.device)
        prior, posterior = self.rssm.observe(embed, actions_time, state0)
        features = torch.cat([posterior.deter, posterior.stoch], dim=-1)
        flat_features = features.reshape(-1, features.shape[-1])
        obs_target = obs.permute(1, 0, 2).reshape(-1, obs.shape[-1])
        recon = self.obs_decoder(flat_features)
        recon_loss = F.mse_loss(recon, obs_target)
        reward_pred = self.reward_decoder(flat_features).view(horizon, batch_size)
        reward_loss = F.mse_loss(reward_pred, rewards.permute(1, 0))
        term_pred = self.termination_decoder(flat_features).view(horizon, batch_size)
        term_target = 1.0 - discounts.permute(1, 0)
        termination_loss = F.binary_cross_entropy(term_pred, term_target)
        kl = self.rssm.kl_divergence(posterior, prior).mean()
        world_loss = recon_loss + reward_loss + termination_loss + self.params.kl_scale * kl
        self.world_optimizer.zero_grad()
        world_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.rssm.parameters())
            + list(self.obs_decoder.parameters())
            + list(self.reward_decoder.parameters())
            + list(self.termination_decoder.parameters()),
            self.params.grad_clip,
        )
        self.world_optimizer.step()

        actor_features = flat_features.detach()

        with torch.no_grad():
            values = self.critic(actor_features).view(horizon, batch_size)
            next_values = torch.cat([values[1:], torch.zeros_like(values[:1])], dim=0)
            targets = rewards.permute(1, 0) + discounts.permute(1, 0) * next_values
            advantages = targets - values

        dist = self.actor(actor_features)
        log_probs = dist.log_prob(actions.permute(1, 0, 2).reshape(-1, actions.shape[-1]))
        entropy = dist.entropy()
        actor_loss = -(log_probs.view(horizon, batch_size) * advantages.detach()).mean()
        actor_loss -= self.params.entropy_scale * entropy.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.params.grad_clip)
        self.actor_optimizer.step()

        critic_features = actor_features.detach()
        values = self.critic(critic_features)
        value_loss = F.mse_loss(values.view(horizon, batch_size), targets)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.params.grad_clip)
        self.value_optimizer.step()

        stats = {
            "world_loss": world_loss.item(),
            "recon_loss": recon_loss.item(),
            "reward_loss": reward_loss.item(),
            "termination_loss": termination_loss.item(),
            "kl_loss": kl.item(),
            "actor_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
        }
        return stats
