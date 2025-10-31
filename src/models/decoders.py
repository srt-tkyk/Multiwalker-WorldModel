"""Decoders projecting latent features back to observations and auxiliary targets."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def build_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    layers = []
    last = input_dim
    for hidden in hidden_dims:
        layers.append(nn.Linear(last, hidden))
        layers.append(nn.ELU())
        last = hidden
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class ObservationDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: Sequence[int], obs_dim: int) -> None:
        super().__init__()
        self.model = build_mlp(latent_dim, hidden_dims, obs_dim)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.model(latents)


class RewardDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.model = build_mlp(latent_dim, hidden_dims, 1)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.model(latents).squeeze(-1)


class TerminationDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.model = build_mlp(latent_dim, hidden_dims, 1)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(latents).squeeze(-1))
