"""Evaluation utilities for Dreamer and PPO checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import imageio
import numpy as np
import torch
import yaml

from src.agents.dreamer_shared import DreamerParams, DreamerSharedAgent
from src.envs.multiwalker_wrapped import EnvConfig, make_env


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_agent(cfg: Dict, obs_dim: int, action_dim: int, device: torch.device) -> DreamerSharedAgent:
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
    return agent


def evaluate(cfg: Dict, ckpt_path: str, episodes: int, save_video: bool, out_path: str | None, dump_latents: bool, latents_out: str | None) -> Dict:
    exp_cfg = cfg["experiment"]
    env_cfg_dict = cfg["environment"]
    render_mode = "rgb_array" if save_video else None
    env_cfg = EnvConfig(
        max_episode_steps=env_cfg_dict["max_episode_steps"],
        frame_skip=env_cfg_dict.get("frame_skip", 1),
        normalize_obs=env_cfg_dict.get("normalize_obs", True),
        stack_observations=env_cfg_dict.get("stack_observations", 1),
        add_agent_id=env_cfg_dict.get("add_agent_id", True),
        vector_envs=1,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(env_cfg, seed=exp_cfg.get("seed", 42), render_mode=render_mode)
    obs, _ = env.reset()
    num_agents, obs_dim = obs.shape
    action_dim = env.action_space.shape[-1]
    agent = build_agent(cfg, obs_dim, action_dim, device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    agent.encoder.load_state_dict(checkpoint["encoder"])
    agent.rssm.load_state_dict(checkpoint["rssm"])
    agent.obs_decoder.load_state_dict(checkpoint["obs_decoder"])
    agent.reward_decoder.load_state_dict(checkpoint["reward_decoder"])
    agent.termination_decoder.load_state_dict(checkpoint["termination_decoder"])
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.critic.load_state_dict(checkpoint["critic"])
    agent.encoder.eval()
    agent.rssm.eval()
    agent.obs_decoder.eval()
    agent.reward_decoder.eval()
    agent.termination_decoder.eval()
    agent.actor.eval()
    agent.critic.eval()

    latents: List[np.ndarray] = []
    returns = []
    lengths = []
    writer = None
    if save_video and out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(out, fps=cfg.get("video", {}).get("fps", 30))

    for _ in range(episodes):
        obs, _ = env.reset()
        agent.reset_state()
        ep_return = 0.0
        ep_length = 0
        done = False
        while not done and ep_length < env_cfg.max_episode_steps:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action_tensor = agent.policy(obs_tensor, deterministic=True)
            action_np = action_tensor.cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(action_np)
            done = bool(terminated or truncated)
            ep_return += reward
            ep_length += 1
            if writer is not None:
                frame = env.render()
                if frame is not None:
                    writer.append_data(frame)
            feats = agent.last_features()
            if dump_latents and feats is not None:
                latents.append(feats.cpu().numpy())
        returns.append(ep_return)
        lengths.append(ep_length)

    if writer is not None:
        writer.close()
    env.close()

    if dump_latents and latents_out is not None and latents:
        latents_arr = np.concatenate(latents, axis=0)
        out = Path(latents_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, latents=latents_arr)

    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "length_mean": float(np.mean(lengths)),
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Dreamer checkpoints.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--dump_latents", action="store_true")
    parser.add_argument("--latents_out", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    metrics = evaluate(
        cfg,
        ckpt_path=args.ckpt,
        episodes=args.episodes,
        save_video=args.save_video,
        out_path=args.out,
        dump_latents=args.dump_latents,
        latents_out=args.latents_out,
    )
    print(metrics)


if __name__ == "__main__":
    main()
