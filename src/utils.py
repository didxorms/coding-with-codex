from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_metrics_csv(path: Path, rewards: Iterable[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "reward"])
        for idx, reward in enumerate(rewards, start=1):
            writer.writerow([idx, reward])


def plot_rewards(path: Path, rewards: Iterable[float], window: int = 10) -> None:
    rewards_array = np.array(list(rewards), dtype=np.float32)
    if rewards_array.size == 0:
        return

    rolling = np.convolve(
        rewards_array,
        np.ones(window) / window,
        mode="valid",
    )

    plt.figure(figsize=(8, 4))
    plt.plot(rewards_array, label="reward", alpha=0.6)
    if rolling.size > 0:
        plt.plot(
            np.arange(window - 1, window - 1 + rolling.size),
            rolling,
            label=f"{window}-episode avg",
            linewidth=2,
        )
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
