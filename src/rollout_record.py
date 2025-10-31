"""Generate rollout videos using a trained Dreamer agent."""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import numpy as np
import torch
import yaml

from src.agents.dreamer_shared import DreamerParams, DreamerSharedAgent
from src.envs.multiwalker_wrapped import EnvConfig, make_env


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record Dreamer policy rollout.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--steps", type=int, default=500)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(env_cfg, seed=cfg["experiment"].get("seed", 42), render_mode="rgb_array")
    obs, _ = env.reset()
    obs_dim = obs.shape[-1]
    action_dim = env.action_space.shape[-1]
    rssm_cfg = cfg["rssm"]
    world_cfg = cfg["world_model"]
    params = DreamerParams(
        deter_size=rssm_cfg["deter_size"],
        stoch_size=rssm_cfg["stoch_size"],
        hidden_size=rssm_cfg["hidden_size"],
        obs_encoder=tuple(world_cfg["obs_encoder"]),
        obs_decoder=tuple(world_cfg["obs_decoder"]),
        reward_decoder=tuple(world_cfg["reward_decoder"]),
        actor_head=tuple(world_cfg["actor_head"]),
        value_head=tuple(world_cfg["value_head"]),
        world_lr=world_cfg["lr"],
        actor_lr=world_cfg.get("actor_lr", world_cfg["lr"]),
        value_lr=world_cfg.get("value_lr", world_cfg["lr"]),
        grad_clip=world_cfg.get("grad_clip", 100.0),
        entropy_scale=world_cfg.get("entropy_scale", 1e-3),
        discount=world_cfg.get("discount", 0.99),
        lambda_return=world_cfg.get("lambda_return", 0.95),
        kl_scale=rssm_cfg.get("kl_scale", 1.0),
    )
    agent = DreamerSharedAgent(obs_dim, action_dim, params, device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    agent.encoder.load_state_dict(checkpoint["encoder"])
    agent.rssm.load_state_dict(checkpoint["rssm"])
    agent.obs_decoder.load_state_dict(checkpoint["obs_decoder"])
    agent.reward_decoder.load_state_dict(checkpoint["reward_decoder"])
    agent.termination_decoder.load_state_dict(checkpoint["termination_decoder"])
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.critic.load_state_dict(checkpoint["critic"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path, fps=cfg.get("video", {}).get("fps", 30))

    agent.reset_state()
    total_reward = 0.0
    for _ in range(args.steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            action_tensor = agent.policy(obs_tensor, deterministic=True)
        action_np = action_tensor.cpu().numpy()
        obs, reward, terminated, truncated, _ = env.step(action_np)
        total_reward += reward
        frame = env.render()
        if frame is not None:
            writer.append_data(frame)
        if terminated or truncated:
            break
    writer.close()
    env.close()
    print({"total_reward": total_reward})


if __name__ == "__main__":
    main()
