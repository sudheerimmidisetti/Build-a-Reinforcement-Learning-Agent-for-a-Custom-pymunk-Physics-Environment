"""Matplotlib-based visualization for training and evaluation results.

Generates four plot types:
1. Training reward curve (from TensorBoard/monitor logs)
2. Episode length curve
3. Evaluation reward distribution (box plot)
4. Pole angle trajectory over a single episode
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from utils.helpers import ensure_dir, get_project_root
from utils.logger import setup_logger

logger = setup_logger(__name__)

# ── Style configuration ──────────────────────────────────────────────────────

plt.style.use("dark_background")
COLOURS = {
    "reward": "#4a90d9",
    "length": "#38c9b1",
    "pole1": "#e8833a",
    "pole2": "#38c9b1",
    "cart": "#4a90d9",
    "mean_line": "#e94560",
    "fill": "#4a90d960",
    "grid": "#333355",
}


def generate_plots(args: Any) -> None:
    """Generate all available plots from training/evaluation data.

    Args:
        args: Parsed CLI arguments with ``log_dir`` and ``output_dir``.
    """
    log_dir = getattr(args, "log_dir", str(get_project_root() / "logs"))
    output_dir = getattr(args, "output_dir", str(get_project_root() / "plots"))
    ensure_dir(output_dir)

    # ── Training reward curve ─────────────────────────────────────────────
    evaluations_path = os.path.join(log_dir, "evaluations.npz")
    if os.path.exists(evaluations_path):
        plot_training_curve(evaluations_path, output_dir)
        plot_episode_length_curve(evaluations_path, output_dir)
    else:
        logger.warning(f"No evaluations.npz found in {log_dir} — skipping training curves.")

    # ── Evaluation metrics ────────────────────────────────────────────────
    metrics_path = os.path.join(log_dir, "evaluation_metrics.json")
    if os.path.exists(metrics_path):
        plot_eval_distribution(metrics_path, output_dir)
    else:
        logger.warning(f"No evaluation_metrics.json found — skipping distribution plot.")

    logger.info(f"Plots saved to {output_dir}")


def plot_training_curve(evaluations_path: str, output_dir: str) -> None:
    """Plot the training reward curve from SB3's EvalCallback output.

    Args:
        evaluations_path: Path to ``evaluations.npz``.
        output_dir: Directory to save the plot.
    """
    data = np.load(evaluations_path)
    timesteps = data["timesteps"]
    results = data["results"]  # shape: (n_evals, n_episodes)

    mean_rewards = np.mean(results, axis=1)
    std_rewards = np.std(results, axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(timesteps, mean_rewards, color=COLOURS["reward"], linewidth=2, label="Mean Reward")
    ax.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.3,
        color=COLOURS["reward"],
        label="±1 Std Dev",
    )

    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Evaluation Reward", fontsize=12)
    ax.set_title("Training Reward Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, color=COLOURS["grid"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_path = os.path.join(output_dir, "training_reward_curve.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_episode_length_curve(evaluations_path: str, output_dir: str) -> None:
    """Plot the episode length curve over training.

    Args:
        evaluations_path: Path to ``evaluations.npz``.
        output_dir: Directory to save the plot.
    """
    data = np.load(evaluations_path)
    timesteps = data["timesteps"]
    ep_lengths = data["ep_lengths"]  # shape: (n_evals, n_episodes)

    mean_lengths = np.mean(ep_lengths, axis=1)
    std_lengths = np.std(ep_lengths, axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(timesteps, mean_lengths, color=COLOURS["length"], linewidth=2, label="Mean Episode Length")
    ax.fill_between(
        timesteps,
        mean_lengths - std_lengths,
        mean_lengths + std_lengths,
        alpha=0.3,
        color=COLOURS["length"],
        label="±1 Std Dev",
    )

    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Episode Length (steps)", fontsize=12)
    ax.set_title("Episode Length Over Training", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, color=COLOURS["grid"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_path = os.path.join(output_dir, "episode_length_curve.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_eval_distribution(metrics_path: str, output_dir: str) -> None:
    """Plot a box plot of evaluation episode rewards.

    Args:
        metrics_path: Path to ``evaluation_metrics.json``.
        output_dir: Directory to save the plot.
    """
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    rewards = metrics.get("episode_rewards", [])
    if not rewards:
        logger.warning("No episode rewards in metrics — skipping box plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    bp = ax.boxplot(
        rewards,
        patch_artist=True,
        boxprops=dict(facecolor=COLOURS["reward"], alpha=0.7),
        medianprops=dict(color=COLOURS["mean_line"], linewidth=2),
        whiskerprops=dict(color="white"),
        capprops=dict(color="white"),
        flierprops=dict(marker="o", markerfacecolor=COLOURS["pole1"], markersize=5),
    )

    ax.set_ylabel("Evaluation Reward", fontsize=12)
    ax.set_title("Evaluation Reward Distribution", fontsize=14, fontweight="bold")
    ax.set_xticklabels(["Agent"])
    ax.grid(True, axis="y", alpha=0.3, color=COLOURS["grid"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add mean marker
    mean_reward = np.mean(rewards)
    ax.axhline(y=mean_reward, color=COLOURS["mean_line"], linestyle="--", alpha=0.7, label=f"Mean: {mean_reward:.1f}")
    ax.legend(fontsize=10)

    fig.tight_layout()
    save_path = os.path.join(output_dir, "eval_reward_distribution.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_angle_trajectory(
    angles_1: List[float],
    angles_2: List[float],
    cart_positions: Optional[List[float]] = None,
    output_dir: Optional[str] = None,
) -> None:
    """Plot pole angle trajectories over a single episode.

    Args:
        angles_1: List of pole 1 angles (radians) per timestep.
        angles_2: List of pole 2 angles (radians) per timestep.
        cart_positions: Optional list of cart positions per timestep.
        output_dir: Directory to save the plot. If None, just shows.
    """
    steps = np.arange(len(angles_1))
    a1_deg = np.degrees(angles_1)
    a2_deg = np.degrees(angles_2)

    n_subplots = 3 if cart_positions is not None else 2
    fig, axes = plt.subplots(n_subplots, 1, figsize=(12, 4 * n_subplots), sharex=True)

    # Pole 1 angle
    axes[0].plot(steps, a1_deg, color=COLOURS["pole1"], linewidth=1.5, label="θ₁")
    axes[0].axhline(y=0, color="white", linestyle=":", alpha=0.3)
    axes[0].set_ylabel("Pole 1 Angle (°)", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, color=COLOURS["grid"])

    # Pole 2 angle
    axes[1].plot(steps, a2_deg, color=COLOURS["pole2"], linewidth=1.5, label="θ₂")
    axes[1].axhline(y=0, color="white", linestyle=":", alpha=0.3)
    axes[1].set_ylabel("Pole 2 Angle (°)", fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, color=COLOURS["grid"])

    # Cart position (optional)
    if cart_positions is not None:
        axes[2].plot(steps, cart_positions, color=COLOURS["cart"], linewidth=1.5, label="Cart x")
        axes[2].axhline(y=0, color="white", linestyle=":", alpha=0.3)
        axes[2].set_ylabel("Cart Position (m)", fontsize=11)
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3, color=COLOURS["grid"])

    axes[-1].set_xlabel("Timestep", fontsize=12)
    fig.suptitle("Episode Trajectory", fontsize=14, fontweight="bold")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()

    if output_dir:
        ensure_dir(output_dir)
        save_path = os.path.join(output_dir, "angle_trajectory.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {save_path}")
    else:
        plt.show()
