"""Training entry point for Dreamer-style shared policy on Multiwalker."""
from __future__ import annotations

import argparse
import csv
import random
from collections import deque
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

from src.agents.dreamer_shared import DreamerParams, DreamerSharedAgent
from src.envs.multiwalker_wrapped import EnvConfig, make_env
from src.utils.replay_buffer import ReplayBuffer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(pref: str) -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Dreamer shared policy on Multiwalker.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--run_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_cfg = cfg["experiment"]
    set_seed(exp_cfg.get("seed", 42))
    device = select_device(exp_cfg.get("device", "auto"))
    env_cfg_dict = cfg["environment"]
    env_cfg = EnvConfig(
        max_episode_steps=env_cfg_dict["max_episode_steps"],
        frame_skip=env_cfg_dict.get("frame_skip", 1),
        normalize_obs=env_cfg_dict.get("normalize_obs", True),
        stack_observations=env_cfg_dict.get("stack_observations", 1),
        add_agent_id=env_cfg_dict.get("add_agent_id", True),
        vector_envs=env_cfg_dict.get("vector_envs", 1),
    )
    root_dir = Path(exp_cfg.get("root_dir", "runs"))
    run_name = exp_cfg.get("name", "exp_dreamer_shared")
    if args.run_dir is not None:
        run_name = args.run_dir
    run_dir = root_dir / run_name
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    for directory in (checkpoints_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    env = make_env(env_cfg, seed=exp_cfg.get("seed", 42))
    obs, _ = env.reset()
    num_agents, obs_dim = obs.shape
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
    buffer_cfg = cfg["replay_buffer"]
    horizon = rssm_cfg.get("horizon", 15)
    try:
        agent = DreamerSharedAgent(obs_dim, action_dim, params, device)
    except RuntimeError as err:
        if "out of memory" in str(err).lower() and device.type == "cuda":
            print("CUDAメモリ不足が発生したため、CPUへフォールバックします。")
            torch.cuda.empty_cache()
            device = torch.device("cpu")
            agent = DreamerSharedAgent(obs_dim, action_dim, params, device)
        else:
            raise

    replay = ReplayBuffer(
        capacity=buffer_cfg["capacity"],
        sequence_length=horizon,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        num_agents=num_agents,
    )

    opt_cfg = cfg["optimization"]
    total_steps = opt_cfg.get("train_steps", 1_000_000)
    init_random_steps = env_cfg_dict.get("init_random_steps", 0)
    batch_size = buffer_cfg.get("batch_size", 32)
    updates_per_step = max(1, int(opt_cfg.get("updates_per_env_step", 1)))
    log_interval = opt_cfg.get("log_interval", 1000)
    save_interval = opt_cfg.get("save_interval", 100000)

    metrics_path = logs_dir / "metrics.csv"
    action_cols = [f"action_{i}" for i in range(num_agents * action_dim)]

    with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "step",
            "episode",
            "episode_return",
            "episode_length",
            "world_loss",
            "actor_loss",
            "value_loss",
            "recon_loss",
            "reward_loss",
            "termination_loss",
            "kl_loss",
            "entropy",
        ] + action_cols
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        episode_return = 0.0
        episode_length = 0
        episode_idx = 0
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        stats_window = deque(maxlen=log_interval)
        action_history: List[np.ndarray] = []

        for step in range(1, total_steps + 1):
            if step <= init_random_steps:
                action_np = np.random.uniform(
                    low=env.action_space.low,
                    high=env.action_space.high,
                    size=env.action_space.shape,
                ).astype(np.float32)
            else:
                with torch.no_grad():
                    action_tensor = agent.policy(obs_tensor)
                action_np = action_tensor.cpu().numpy()
            action_history.append(action_np.reshape(-1))
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = bool(terminated or truncated)
            discount = params.discount if not done else 0.0
            replay.add_step(obs, action_np, reward, done, discount)
            episode_return += reward
            episode_length += 1
            obs = next_obs
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

            if step > init_random_steps and len(replay) >= batch_size * horizon:
                for _ in range(updates_per_step):
                    batch = replay.sample(batch_size)
                    stats = agent.update(batch)
                    stats_window.append(stats)

            if done:
                episode_idx += 1
                averaged = {
                    key: float(np.mean([s[key] for s in stats_window])) if stats_window else 0.0
                    for key in [
                        "world_loss",
                        "actor_loss",
                        "value_loss",
                        "recon_loss",
                        "reward_loss",
                        "termination_loss",
                        "kl_loss",
                        "entropy",
                    ]
                }
                action_means = {}
                if action_history:
                    stacked_actions = np.stack(action_history, axis=0)
                    for idx in range(stacked_actions.shape[1]):
                        action_means[f"action_{idx}"] = float(stacked_actions[:, idx].mean())
                writer.writerow(
                    {
                        "step": step,
                        "episode": episode_idx,
                        "episode_return": episode_return,
                        "episode_length": episode_length,
                        **averaged,
                        **action_means,
                    }
                )
                csv_file.flush()
                obs, _ = env.reset()
                agent.reset_state()
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                episode_return = 0.0
                episode_length = 0
                stats_window.clear()
                action_history.clear()

            if step % save_interval == 0:
                checkpoint_path = checkpoints_dir / f"step_{step}.pt"
                torch.save({
                    "encoder": agent.encoder.state_dict(),
                    "rssm": agent.rssm.state_dict(),
                    "obs_decoder": agent.obs_decoder.state_dict(),
                    "reward_decoder": agent.reward_decoder.state_dict(),
                    "termination_decoder": agent.termination_decoder.state_dict(),
                    "actor": agent.actor.state_dict(),
                    "critic": agent.critic.state_dict(),
                }, checkpoint_path)

    env.close()


if __name__ == "__main__":
    main()
