"""Policy and value heads operating on latent features."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    layers = []
    last = input_dim
    for hidden in hidden_dims:
        layers.append(nn.Linear(last, hidden))
        layers.append(nn.ELU())
        last = hidden
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


@dataclass
class ActionDistribution:
    mean: torch.Tensor
    std: torch.Tensor

    def sample(self) -> torch.Tensor:
        eps = torch.randn_like(self.std)
        return self.mean + self.std * eps

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        var = self.std.pow(2)
        log_std = torch.log(self.std)
        constant = math.log(2 * math.pi)
        return -0.5 * (((value - self.mean) ** 2) / var + 2 * log_std + constant).sum(-1)

    def entropy(self) -> torch.Tensor:
        log_var = 2 * torch.log(self.std)
        constant = 1.0 + math.log(2 * math.pi)
        return 0.5 * (log_var + constant).sum(-1)


class ActorHead(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: Sequence[int], action_dim: int, min_std: float = 0.1, max_std: float = 1.0) -> None:
        super().__init__()
        self._trunk = build_mlp(latent_dim, hidden_dims, action_dim * 2)
        self._min_std = min_std
        self._max_std = max_std
        self._action_dim = action_dim

    def forward(self, latents: torch.Tensor) -> ActionDistribution:
        params = self._trunk(latents)
        mean, std = torch.chunk(params, 2, dim=-1)
        std = torch.sigmoid(std)
        std = self._min_std + (self._max_std - self._min_std) * std
        return ActionDistribution(mean=torch.tanh(mean), std=std)


class ValueHead(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.model = build_mlp(latent_dim, hidden_dims, 1)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.model(latents).squeeze(-1)
