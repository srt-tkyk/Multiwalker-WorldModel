"""Simple replay buffer for Dreamer-style sequence sampling."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch


@dataclass
class Transition:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    discount: float


class ReplayBuffer:
    """Stores fixed-length trajectories for latent imagination training."""

    def __init__(
        self,
        capacity: int,
        sequence_length: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        num_agents: int,
    ) -> None:
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.num_agents = num_agents
        self._storage: List[List[Transition]] = []
        self._pending: List[List[Transition]] = [[] for _ in range(num_agents)]

    def add_step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        discount: float = 0.99,
    ) -> None:
        obs = np.asarray(obs)
        action = np.asarray(action)
        for idx in range(self.num_agents):
            transition = Transition(
                obs=obs[idx].copy(),
                action=action[idx].copy(),
                reward=float(reward),
                discount=float(discount),
            )
            self._pending[idx].append(transition)
        if done:
            self._commit_episode()

    def _commit_episode(self) -> None:
        for idx in range(self.num_agents):
            episode = self._pending[idx]
            if not episode:
                continue
            self._storage.append(episode.copy())
            self._pending[idx] = []
        if len(self._storage) > self.capacity:
            overflow = len(self._storage) - self.capacity
            del self._storage[0:overflow]

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if len(self._storage) == 0:
            raise ValueError("ReplayBuffer is empty.")
        episodes = random.choices(self._storage, k=batch_size)
        obs_batch = []
        action_batch = []
        reward_batch = []
        discount_batch = []
        for ep in episodes:
            if len(ep) < self.sequence_length:
                start = 0
            else:
                start = random.randint(0, len(ep) - self.sequence_length)
            seq = ep[start : start + self.sequence_length]
            obs_batch.append(np.stack([tr.obs for tr in seq], axis=0))
            action_batch.append(np.stack([tr.action for tr in seq], axis=0))
            reward_batch.append([tr.reward for tr in seq])
            discount_batch.append([tr.discount for tr in seq])
        obs_tensor = torch.tensor(np.stack(obs_batch), dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(np.stack(action_batch), dtype=torch.float32, device=self.device)
        reward_tensor = torch.tensor(np.array(reward_batch), dtype=torch.float32, device=self.device)
        discount_tensor = torch.tensor(np.array(discount_batch), dtype=torch.float32, device=self.device)
        return {
            "obs": obs_tensor,
            "actions": action_tensor,
            "rewards": reward_tensor,
            "discounts": discount_tensor,
        }

    def __len__(self) -> int:
        return sum(len(ep) for ep in self._storage)

    def flush(self) -> None:
        self._commit_episode()
