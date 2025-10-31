"""t-SNE visualisation utilities for latent trajectories."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.manifold import TSNE


def run_tsne(latents_path: str, output_path: Optional[str] = None, perplexity: int = 30, n_iter: int = 1000) -> np.ndarray:
    latents = np.load(latents_path)["latents"]
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, learning_rate="auto")
    embedding = tsne.fit_transform(latents)
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, embedding=embedding)
    return embedding
