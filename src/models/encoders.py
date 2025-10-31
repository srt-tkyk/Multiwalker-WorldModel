"""Observation encoders for Multiwalker agents."""
from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn


def mlp(input_dim: int, hidden_dims: Sequence[int], activation: nn.Module = nn.ELU()) -> nn.Sequential:
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation)
        last_dim = hidden_dim
    return nn.Sequential(*layers)


class ObservationEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> None:
        super().__init__()
        self.net = mlp(input_dim, hidden_dims)
        self.proj = nn.Linear(hidden_dims[-1] if hidden_dims else input_dim, output_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs
        if len(self.net) > 0:
            x = self.net(obs)
        return torch.tanh(self.proj(x))
