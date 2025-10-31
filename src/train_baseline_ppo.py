"""Training script for the shared-policy PPO baseline."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import yaml

from src.agents.ppo_shared import PPOParams, train_shared_ppo
from src.envs.multiwalker_wrapped import EnvConfig


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO shared policy baseline.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--run_dir", type=str, default="runs/exp_ppo_shared")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg_dict = cfg["environment"]
    env_cfg = EnvConfig(
        max_episode_steps=env_cfg_dict["max_episode_steps"],
        frame_skip=env_cfg_dict.get("frame_skip", 1),
        normalize_obs=env_cfg_dict.get("normalize_obs", True),
        stack_observations=env_cfg_dict.get("stack_observations", 1),
        add_agent_id=env_cfg_dict.get("add_agent_id", True),
        vector_envs=1,
    )
    ppo_cfg = cfg["ppo"]
    params = PPOParams(
        total_timesteps=ppo_cfg["total_timesteps"],
        learning_rate=ppo_cfg["learning_rate"],
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_range"],
        ent_coef=ppo_cfg["ent_coef"],
        vf_coef=ppo_cfg["vf_coef"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        target_kl=ppo_cfg.get("target_kl", None),
    )
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    train_shared_ppo(env_cfg, params, seed=cfg["experiment"].get("seed", 42), log_dir=args.run_dir)


if __name__ == "__main__":
    main()
