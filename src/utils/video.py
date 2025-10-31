"""Video rendering utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import imageio
import numpy as np


def record_episode(env, policy_fn: Callable[[np.ndarray], np.ndarray], out_path: Path, fps: int = 30, num_steps: int = 500) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    obs, _ = env.reset()
    for _ in range(num_steps):
        action = policy_fn(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if terminated or truncated:
            break
    if not frames:
        raise RuntimeError("環境からレンダリングフレームを取得できませんでした。")
    with imageio.get_writer(out_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
