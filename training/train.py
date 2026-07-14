"""Training pipeline for the Double Inverted Pendulum RL agent.

Orchestrates environment creation, vectorisation, normalisation,
algorithm instantiation, callback wiring, and model training/saving.
"""

import os
import sys
from typing import Any, Dict

import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import envs  # noqa: F401  — triggers gymnasium.register
from training.callbacks import build_callbacks
from training.hyperparams import get_algo_params
from utils.helpers import ensure_dir, get_project_root, timestamp_string
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Map algorithm names to SB3 classes
ALGO_MAP = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
}


def run_training(
    env_config: Dict[str, Any],
    train_config: Dict[str, Any],
    args: Any,
) -> None:
    """Run the full training pipeline.

    Args:
        env_config: Environment configuration dict.
        train_config: Training configuration dict.
        args: Parsed CLI arguments (may contain overrides).
    """
    project_root = get_project_root()

    # ── Resolve parameters ────────────────────────────────────────────────
    algo_name = train_config.get("algorithm", "PPO").upper()
    seed = args.seed if args.seed is not None else train_config.get("seed", 42)
    total_timesteps = (
        args.timesteps
        if args.timesteps is not None
        else train_config.get("total_timesteps", 1_000_000)
    )
    n_envs = train_config.get("n_envs", 4)
    use_vec_normalize = train_config.get("use_vec_normalize", True)

    # ── Paths ─────────────────────────────────────────────────────────────
    paths = train_config.get("paths", {})
    model_dir = str(project_root / paths.get("model_dir", "models"))
    log_dir = str(project_root / paths.get("log_dir", "logs"))
    tb_dir = str(project_root / paths.get("tensorboard_dir", "logs/tensorboard"))

    ensure_dir(model_dir)
    ensure_dir(log_dir)
    ensure_dir(tb_dir)

    run_name = f"{algo_name}_{timestamp_string()}"
    logger.info(f"Starting training run: {run_name}")
    logger.info(f"Algorithm: {algo_name} | Envs: {n_envs} | Steps: {total_timesteps}")

    # ── Create vectorised training environment ────────────────────────────
    def make_env():
        from envs.double_pendulum_env import DoublePendulumEnv

        return DoublePendulumEnv(render_mode=None, env_config=env_config)

    train_env = DummyVecEnv([make_env for _ in range(n_envs)])

    if use_vec_normalize:
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )
        logger.info("VecNormalize enabled (obs + reward)")

    # ── Create evaluation environment ─────────────────────────────────────
    eval_env = DummyVecEnv([make_env])
    if use_vec_normalize:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,  # Don't normalise reward for evaluation
            clip_obs=10.0,
            training=False,  # Don't update running stats during eval
        )

    # ── Get algorithm hyperparameters ─────────────────────────────────────
    algo_params = get_algo_params(train_config)
    logger.info(f"Hyperparameters: {algo_params}")

    # ── Create model ──────────────────────────────────────────────────────
    AlgoClass = ALGO_MAP.get(algo_name)
    if AlgoClass is None:
        logger.error(f"Unknown algorithm: {algo_name}")
        sys.exit(1)

    model = AlgoClass(
        policy="MlpPolicy",
        env=train_env,
        seed=seed,
        verbose=1,
        tensorboard_log=tb_dir,
        **algo_params,
    )
    logger.info(f"Model created: {AlgoClass.__name__} with MlpPolicy")

    # ── Build callbacks ───────────────────────────────────────────────────
    callbacks = build_callbacks(train_config, eval_env, model_dir, log_dir)

    # ── Train ─────────────────────────────────────────────────────────────
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=train_config.get("callbacks", {}).get("log_interval", 1),
            tb_log_name=run_name,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
    finally:
        # ── Save final model ──────────────────────────────────────────────
        final_path = os.path.join(model_dir, "final_model")
        model.save(final_path)
        logger.info(f"Final model saved to {final_path}.zip")

        # ── Save VecNormalize stats ───────────────────────────────────────
        if use_vec_normalize:
            norm_path = os.path.join(model_dir, "vecnormalize.pkl")
            train_env.save(norm_path)
            logger.info(f"VecNormalize stats saved to {norm_path}")

        train_env.close()
        eval_env.close()

    logger.info("Training complete.")
