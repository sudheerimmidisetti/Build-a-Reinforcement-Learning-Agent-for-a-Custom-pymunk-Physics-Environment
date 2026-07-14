"""Custom Stable-Baselines3 callbacks for the training pipeline."""

import os
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)

from utils.helpers import ensure_dir
from utils.logger import setup_logger

logger = setup_logger(__name__)


class TrainingMetricsCallback(BaseCallback):
    """Logs custom per-step metrics to TensorBoard.

    Records pole angles, cart position, and reward components for
    detailed analysis beyond SB3's default logging.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Log info from the first environment (index 0)
        infos = self.locals.get("infos", [])
        if infos and len(infos) > 0:
            info = infos[0]
            if "pole1_angle" in info:
                self.logger.record(
                    "custom/pole1_angle_deg",
                    np.degrees(info["pole1_angle"]),
                )
                self.logger.record(
                    "custom/pole2_angle_deg",
                    np.degrees(info["pole2_angle"]),
                )
                self.logger.record(
                    "custom/cart_position",
                    info["cart_position"],
                )
        return True


class GracefulExitCallback(BaseCallback):
    """Saves a checkpoint when training is interrupted via Ctrl+C.

    Wraps the training loop so that KeyboardInterrupt saves the model
    before exiting, preventing loss of a long training run.
    """

    def __init__(self, save_path: str, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        """Called at the end of training (including interrupts if caught)."""
        save_file = os.path.join(self.save_path, "final_model")
        self.model.save(save_file)
        if self.verbose > 0:
            logger.info(f"Final model saved to {save_file}")


def build_callbacks(
    train_config: dict,
    eval_env,
    model_dir: str,
    log_dir: str,
) -> list:
    """Build the full list of SB3 callbacks from training config.

    Args:
        train_config: Training configuration dictionary.
        eval_env: Vectorised evaluation environment.
        model_dir: Directory for saving model checkpoints.
        log_dir: Directory for logs.

    Returns:
        List of SB3 callback instances.
    """
    cb_config = train_config.get("callbacks", {})

    ensure_dir(model_dir)
    ensure_dir(log_dir)
    best_model_dir = os.path.join(model_dir, "best")
    ensure_dir(best_model_dir)

    callbacks = []

    # ── Evaluation callback ───────────────────────────────────────────────
    eval_freq = cb_config.get("eval_freq", 10_000)
    eval_episodes = cb_config.get("eval_episodes", 10)
    early_stop_reward = cb_config.get("early_stop_reward", None)
    early_stop_patience = cb_config.get("early_stop_patience", 5)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=log_dir,
        eval_freq=max(eval_freq // train_config.get("n_envs", 1), 1),
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )
    callbacks.append(eval_callback)

    # ── Checkpoint callback ───────────────────────────────────────────────
    checkpoint_freq = cb_config.get("checkpoint_freq", 50_000)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(checkpoint_freq // train_config.get("n_envs", 1), 1),
        save_path=model_dir,
        name_prefix="checkpoint",
        verbose=1,
    )
    callbacks.append(checkpoint_callback)

    # ── Custom metrics callback ───────────────────────────────────────────
    metrics_callback = TrainingMetricsCallback()
    callbacks.append(metrics_callback)

    # ── Graceful exit callback ────────────────────────────────────────────
    graceful_callback = GracefulExitCallback(save_path=model_dir)
    callbacks.append(graceful_callback)

    return callbacks
