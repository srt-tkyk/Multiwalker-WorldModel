"""Plotting helpers for training diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_metrics(run_path: Path) -> pd.DataFrame:
    csv_path = run_path / "logs" / "metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"メトリクスファイルが見つかりません: {csv_path}")
    return pd.read_csv(csv_path)


def plot_action_distribution(run_dir: str) -> Path:
    run_path = Path(run_dir)
    data = _load_metrics(run_path)
    actions = data.filter(regex=r"action_.*").values
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(actions.flatten(), bins=50, alpha=0.7)
    ax.set_title("Action Distribution")
    ax.set_xlabel("Action Value")
    ax.set_ylabel("Frequency")
    out_path = run_path / "figs" / "action_hist.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_value_curves(run_dir: str) -> Path:
    run_path = Path(run_dir)
    data = _load_metrics(run_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    if "value" not in data.columns:
        raise KeyError("value列がCSVに存在しません")
    ax.plot(data["timestep"], data["value"], label="Value Estimate")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Value")
    ax.set_title("Value Curve")
    ax.legend()
    out_path = run_path / "figs" / "value_curve.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
