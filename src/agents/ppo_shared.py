"""Shared-policy PPO baseline leveraging Stable-Baselines3."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.envs.multiwalker_wrapped import EnvConfig, make_env


class FlattenedMultiwalkerEnv(gym.Env):
    """Flattens shared-policy environment for SB3 compatibility."""

    def __init__(self, config: EnvConfig, seed: Optional[int] = None) -> None:
        super().__init__()
        self._inner = make_env(config, seed=seed)
        obs_sample, _ = self._inner.reset()
        self.num_agents, obs_dim = obs_sample.shape
        self.action_dim = self._inner.action_space.shape[-1]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_agents * obs_dim,),
            dtype=np.float32,
        )
        low = self._inner.action_space.low.reshape(-1)
        high = self._inner.action_space.high.reshape(-1)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self._inner.reset(seed=seed, options=options)
        return obs.reshape(-1), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self.num_agents, self.action_dim)
        obs, reward, terminated, truncated, info = self._inner.step(action)
        return obs.reshape(-1), reward, terminated, truncated, info

    def close(self):  # pragma: no cover
        self._inner.close()


@dataclass
class PPOParams:
    total_timesteps: int
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    target_kl: Optional[float]


def train_shared_ppo(config: EnvConfig, params: PPOParams, seed: int = 42, log_dir: str = "runs/exp_ppo_shared") -> PPO:
    from pathlib import Path

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    env_fn = lambda: FlattenedMultiwalkerEnv(config, seed=seed)
    vec_env = DummyVecEnv([env_fn])
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=params.learning_rate,
        n_steps=params.n_steps,
        batch_size=params.batch_size,
        n_epochs=params.n_epochs,
        gamma=params.gamma,
        gae_lambda=params.gae_lambda,
        clip_range=params.clip_range,
        ent_coef=params.ent_coef,
        vf_coef=params.vf_coef,
        max_grad_norm=params.max_grad_norm,
        target_kl=params.target_kl,
        tensorboard_log=log_dir,
        verbose=1,
    )
    model.learn(total_timesteps=params.total_timesteps, progress_bar=True)
    model.save(f"{log_dir}/ppo_shared")
    vec_env.close()
    return model
