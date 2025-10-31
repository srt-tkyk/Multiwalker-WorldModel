"""Recurrent state-space model (RSSM) implementation for Dreamer-style agents."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RSSMState:
    """Container for latent state tuples."""

    mean: torch.Tensor
    std: torch.Tensor
    stoch: torch.Tensor
    deter: torch.Tensor

    def detach(self) -> "RSSMState":
        return RSSMState(
            mean=self.mean.detach(),
            std=self.std.detach(),
            stoch=self.stoch.detach(),
            deter=self.deter.detach(),
        )


class RSSM(nn.Module):
    """Dreamer-style recurrent state-space model with Gaussian latents."""

    def __init__(
        self,
        deter_size: int,
        stoch_size: int,
        hidden_size: int,
        action_dim: int,
        embed_dim: int,
        min_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.hidden_size = hidden_size
        self.min_std = min_std
        self._gru = nn.GRUCell(stoch_size + action_dim, deter_size)
        self._prior = nn.Sequential(
            nn.Linear(deter_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 2 * stoch_size),
        )
        self._posterior = nn.Sequential(
            nn.Linear(deter_size + embed_dim, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 2 * stoch_size),
        )
        self._layer_norm = nn.LayerNorm(deter_size)

    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        zeros = torch.zeros(batch_size, self.stoch_size, device=device)
        deter = torch.zeros(batch_size, self.deter_size, device=device)
        return RSSMState(mean=zeros, std=torch.ones_like(zeros), stoch=zeros, deter=deter)

    def observe(
        self,
        embeds: torch.Tensor,
        actions: torch.Tensor,
        prev_state: RSSMState,
    ) -> Tuple[RSSMState, RSSMState]:
        priors = []
        posteriors = []
        state = prev_state
        for embed, action in zip(embeds.unbind(dim=0), actions.unbind(dim=0)):
            state = self.img_step(state, action)
            priors.append(state)
            post_state = self.obs_step(state, embed)
            posteriors.append(post_state)
            state = post_state
        prior = RSSMState(
            mean=torch.stack([s.mean for s in priors]),
            std=torch.stack([s.std for s in priors]),
            stoch=torch.stack([s.stoch for s in priors]),
            deter=torch.stack([s.deter for s in priors]),
        )
        posterior = RSSMState(
            mean=torch.stack([s.mean for s in posteriors]),
            std=torch.stack([s.std for s in posteriors]),
            stoch=torch.stack([s.stoch for s in posteriors]),
            deter=torch.stack([s.deter for s in posteriors]),
        )
        return prior, posterior

    def imagine(
        self,
        actions: torch.Tensor,
        prev_state: RSSMState,
    ) -> RSSMState:
        actions = actions.unbind(dim=0)
        state = prev_state
        priors = []
        for action in actions:
            state = self.img_step(state, action)
            priors.append(state)
        return RSSMState(
            mean=torch.stack([s.mean for s in priors]),
            std=torch.stack([s.std for s in priors]),
            stoch=torch.stack([s.stoch for s in priors]),
            deter=torch.stack([s.deter for s in priors]),
        )

    def get_features(self, state: RSSMState) -> torch.Tensor:
        return torch.cat([state.deter, state.stoch], dim=-1)

    def obs_step(self, state: RSSMState, embed: torch.Tensor) -> RSSMState:
        x = torch.cat([state.deter, embed], dim=-1)
        stats = self._posterior(x)
        mean, std = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(std) + self.min_std
        stoch = self._reparameterise(mean, std)
        deter = state.deter
        return RSSMState(mean=mean, std=std, stoch=stoch, deter=deter)

    def img_step(self, state: RSSMState, action: torch.Tensor) -> RSSMState:
        gru_input = torch.cat([state.stoch, action], dim=-1)
        deter = self._gru(gru_input, state.deter)
        deter = self._layer_norm(deter)
        deter = torch.tanh(deter)
        stats = self._prior(deter)
        mean, std = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(std) + self.min_std
        stoch = self._reparameterise(mean, std)
        return RSSMState(mean=mean, std=std, stoch=stoch, deter=deter)

    @staticmethod
    def _reparameterise(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(std)
        return mean + std * eps

    @staticmethod
    def kl_divergence(post: RSSMState, prior: RSSMState) -> torch.Tensor:
        mean_diff = prior.mean - post.mean
        log_std_diff = torch.log(prior.std) - torch.log(post.std)
        numerator = post.std.pow(2) + mean_diff.pow(2)
        denom = 2 * prior.std.pow(2)
        return (log_std_diff + (numerator / denom) - 0.5).sum(dim=-1)
