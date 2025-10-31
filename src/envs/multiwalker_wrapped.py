"""Utility wrappers for PettingZoo Multiwalker to enable shared-policy training."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from pettingzoo.sisl import multiwalker_v9

try:
    import supersuit as ss
except ImportError:  # pragma: no cover - optional import
    ss = None


@dataclass
class EnvConfig:
    """Configuration parameters required to build the environment."""

    max_episode_steps: int = 500
    frame_skip: int = 1
    normalize_obs: bool = True
    stack_observations: int = 1
    add_agent_id: bool = True
    vector_envs: int = 1


class RunningNorm:
    """Tracks running mean and variance for observation normalisation."""

    def __init__(self, shape: Tuple[int, ...], clip: float = 5.0) -> None:
        self._shape = shape
        self._clip = clip
        self._count = 1e-4
        self._mean = np.zeros(shape, dtype=np.float64)
        self._var = np.ones(shape, dtype=np.float64)

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch[None, :]
        count = batch.shape[0]
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        delta = batch_mean - self._mean
        tot_count = self._count + count
        new_mean = self._mean + delta * count / tot_count
        m_a = self._var * self._count
        m_b = batch_var * count
        M2 = m_a + m_b + np.square(delta) * self._count * count / tot_count
        new_var = M2 / tot_count
        self._mean = new_mean
        self._var = new_var
        self._count = tot_count

    def normalise(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        std = np.sqrt(self._var) + 1e-8
        normalised = (obs - self._mean) / std
        if self._clip is not None:
            normalised = np.clip(normalised, -self._clip, self._clip)
        return normalised.astype(np.float32)


class MultiwalkerSharedEnv(gym.Env):
    """Parallel PettingZoo Multiwalker adapted for shared-policy learners."""

    metadata = {"render_modes": ["human", "rgb_array"], "name": "MultiwalkerSharedEnv"}

    def __init__(
        self,
        config: EnvConfig,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._render_mode = render_mode
        self._seed = seed
        self._raw_env = multiwalker_v9.parallel_env(render_mode=render_mode)
        self._agents: List[str] = list(self._raw_env.possible_agents)
        self.num_agents = len(self._agents)
        self._agent_id_vectors = np.eye(self.num_agents, dtype=np.float32)
        self._obs_dim = self._raw_env.observation_space(self._agents[0]).shape[0]
        self._action_space = self._raw_env.action_space(self._agents[0])
        obs_dim = self._obs_dim + (self.num_agents if config.add_agent_id else 0)
        if config.stack_observations > 1:
            obs_dim *= config.stack_observations
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_agents, obs_dim),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.broadcast_to(self._action_space.low, (self.num_agents, self._action_space.shape[0])),
            high=np.broadcast_to(self._action_space.high, (self.num_agents, self._action_space.shape[0])),
            dtype=np.float32,
        )
        self._running_norm = RunningNorm((obs_dim,), clip=5.0) if config.normalize_obs else None
        self._frame_skip = max(1, config.frame_skip)
        self._stack = config.stack_observations
        self._obs_buffers = [np.zeros((self._stack, obs_dim), dtype=np.float32) for _ in range(self.num_agents)]
        if seed is not None:
            self.reset(seed=seed)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Dict]]:
        if seed is not None:
            self._seed = seed
        obs, info = self._raw_env.reset(seed=self._seed, options=options)
        obs_batch = self._format_observations(obs)
        self._clear_stacks()
        self._push_obs(obs_batch)
        return self._get_stacked_obs(), {"raw": info}

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Dict]]:
        action_dict = self._to_action_dict(actions)
        total_reward = 0.0
        terminated = False
        truncated = False
        info_accum: Dict[str, Dict] = {}
        obs = {}
        for _ in range(self._frame_skip):
            obs, rewards, terms, truncs, infos = self._raw_env.step(action_dict)
            total_reward += float(np.mean(list(rewards.values())))
            terminated = terminated or any(terms.values())
            truncated = truncated or any(truncs.values())
            if terminated or truncated:
                break
        obs_batch = self._format_observations(obs)
        self._push_obs(obs_batch)
        info_accum.update(infos)
        return self._get_stacked_obs(), total_reward, terminated, truncated, {"raw": info_accum}

    def render(self):  # pragma: no cover - passthrough
        return self._raw_env.render()

    def close(self):  # pragma: no cover - passthrough
        self._raw_env.close()

    def _format_observations(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        obs_batch = []
        for idx, agent in enumerate(self._agents):
            agent_obs = obs[agent].astype(np.float32)
            if self._config.add_agent_id:
                agent_obs = np.concatenate([agent_obs, self._agent_id_vectors[idx]], axis=0)
            obs_batch.append(agent_obs)
        obs_batch_np = np.stack(obs_batch, axis=0)
        if self._running_norm is not None:
            self._running_norm.update(obs_batch_np.reshape(-1, obs_batch_np.shape[-1]))
            obs_batch_np = self._running_norm.normalise(obs_batch_np)
        return obs_batch_np

    def _to_action_dict(self, actions: np.ndarray) -> Dict[str, np.ndarray]:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape == self._action_space.shape:
            actions = actions[None, :]
        return {agent: actions[idx] for idx, agent in enumerate(self._agents)}

    def _clear_stacks(self) -> None:
        for buf in self._obs_buffers:
            buf[...] = 0.0

    def _push_obs(self, obs_batch: np.ndarray) -> None:
        for idx, buf in enumerate(self._obs_buffers):
            buf[:-1] = buf[1:]
            buf[-1] = obs_batch[idx]

    def _get_stacked_obs(self) -> np.ndarray:
        stacked = [buf.reshape(-1) for buf in self._obs_buffers]
        return np.stack(stacked, axis=0)


def make_env(config: EnvConfig, seed: Optional[int] = None, render_mode: Optional[str] = None) -> MultiwalkerSharedEnv:
    """Factory helper to construct the shared-policy Multiwalker environment."""

    return MultiwalkerSharedEnv(config=config, seed=seed, render_mode=render_mode)


def make_vector_env(config: EnvConfig, base_seed: int = 42) -> gym.vector.VectorEnv:
    """Create a vectorized environment as required by online collection workers."""

    env_fns = []
    for idx in range(config.vector_envs):
        env_seed = None if base_seed is None else base_seed + idx

        def _thunk(local_seed=env_seed):
            return MultiwalkerSharedEnv(config=config, seed=local_seed)

        env_fns.append(_thunk)
    return gym.vector.AsyncVectorEnv(env_fns)
