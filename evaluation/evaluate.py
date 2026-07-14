"""Evaluation and rendering of trained RL agents.

Loads a saved model (and VecNormalize stats), runs deterministic episodes,
computes aggregate metrics, and supports real-time rendering.
"""

import json
import os
import time
from typing import Any, Dict

import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import envs  # noqa: F401
from utils.helpers import ensure_dir, get_project_root
from utils.logger import setup_logger

logger = setup_logger(__name__)

ALGO_MAP = {"PPO": PPO, "SAC": SAC, "TD3": TD3}


def _load_model_and_env(
    model_path: str,
    env_config: Dict[str, Any],
    render_mode: str = None,
    deterministic: bool = True,
):
    """Load a saved model and create a matching environment.

    Automatically loads VecNormalize stats if they exist alongside the model.

    Args:
        model_path: Path to the saved ``.zip`` model file.
        env_config: Environment configuration dict.
        render_mode: ``"human"``, ``"rgb_array"``, or ``None``.
        deterministic: Whether to use deterministic actions.

    Returns:
        Tuple of ``(model, vec_env)``.
    """
    from envs.double_pendulum_env import DoublePendulumEnv

    # Try to determine the algorithm from the model path or default to PPO
    model_dir = os.path.dirname(model_path)

    # Try each algorithm until one succeeds
    model = None
    for algo_name, AlgoClass in ALGO_MAP.items():
        try:
            model = AlgoClass.load(model_path)
            logger.info(f"Loaded model as {algo_name} from {model_path}")
            break
        except Exception:
            continue

    if model is None:
        raise RuntimeError(f"Could not load model from {model_path}")

    # Create environment
    def make_env():
        return DoublePendulumEnv(render_mode=render_mode, env_config=env_config)

    vec_env = DummyVecEnv([make_env])

    # Load VecNormalize stats if they exist
    norm_path = os.path.join(model_dir, "vecnormalize.pkl")
    if os.path.exists(norm_path):
        vec_env = VecNormalize.load(norm_path, vec_env)
        vec_env.training = False  # Don't update stats during evaluation
        vec_env.norm_reward = False  # Don't normalise rewards for eval
        logger.info(f"Loaded VecNormalize stats from {norm_path}")
    else:
        logger.warning(
            "No VecNormalize stats found — using raw observations. "
            "Results may be unreliable if training used VecNormalize."
        )

    model.set_env(vec_env)
    return model, vec_env


def run_evaluation(env_config: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """Run evaluation episodes and report aggregate metrics.

    Args:
        env_config: Environment configuration dict.
        args: Parsed CLI arguments with ``model_path`` and ``episodes``.

    Returns:
        Dictionary of evaluation metrics.
    """
    model, vec_env = _load_model_and_env(
        args.model_path, env_config, render_mode=None
    )

    n_episodes = getattr(args, "episodes", 10)
    episode_rewards = []
    episode_lengths = []
    max_angles_1 = []
    max_angles_2 = []

    logger.info(f"Running {n_episodes} evaluation episodes...")

    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0
        max_a1 = 0.0
        max_a2 = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env.step(action)
            ep_reward += reward[0]
            ep_length += 1

            info = infos[0]
            max_a1 = max(max_a1, abs(info.get("pole1_angle", 0)))
            max_a2 = max(max_a2, abs(info.get("pole2_angle", 0)))

            done = dones[0]

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        max_angles_1.append(np.degrees(max_a1))
        max_angles_2.append(np.degrees(max_a2))

        logger.info(
            f"  Episode {ep + 1:>3d}: reward={ep_reward:>8.2f}  "
            f"length={ep_length:>5d}  "
            f"max|θ₁|={np.degrees(max_a1):>6.2f}°  "
            f"max|θ₂|={np.degrees(max_a2):>6.2f}°"
        )

    vec_env.close()

    # ── Aggregate metrics ─────────────────────────────────────────────────
    metrics = {
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "mean_max_angle1_deg": float(np.mean(max_angles_1)),
        "mean_max_angle2_deg": float(np.mean(max_angles_2)),
        "episode_rewards": [float(r) for r in episode_rewards],
        "episode_lengths": [int(l) for l in episode_lengths],
    }

    # Print summary
    logger.info("=" * 60)
    logger.info(f"  Mean reward:   {metrics['mean_reward']:>8.2f} ± {metrics['std_reward']:.2f}")
    logger.info(f"  Mean length:   {metrics['mean_length']:>8.1f} ± {metrics['std_length']:.1f}")
    logger.info(f"  Reward range:  [{metrics['min_reward']:.2f}, {metrics['max_reward']:.2f}]")
    logger.info(f"  Mean max |θ₁|: {metrics['mean_max_angle1_deg']:.2f}°")
    logger.info(f"  Mean max |θ₂|: {metrics['mean_max_angle2_deg']:.2f}°")
    logger.info("=" * 60)

    # Save metrics to JSON
    project_root = get_project_root()
    eval_dir = str(project_root / "logs")
    ensure_dir(eval_dir)
    metrics_path = os.path.join(eval_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    return metrics


def run_render(env_config: Dict[str, Any], args: Any) -> None:
    """Render a trained agent in real-time with a Pygame window.

    Args:
        env_config: Environment configuration dict.
        args: Parsed CLI arguments with ``model_path``.
    """
    model, vec_env = _load_model_and_env(
        args.model_path, env_config, render_mode="human"
    )

    logger.info("Rendering agent — close the Pygame window or press Ctrl+C to stop.")

    try:
        obs = vec_env.reset()
        episode = 0
        ep_reward = 0.0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env.step(action)
            ep_reward += reward[0]

            # Render is called automatically by the env if render_mode="human"
            # But we also call it explicitly to ensure it happens
            vec_env.render()

            if dones[0]:
                episode += 1
                logger.info(f"Episode {episode}: reward={ep_reward:.2f}")
                ep_reward = 0.0
                obs = vec_env.reset()

    except KeyboardInterrupt:
        logger.info("Rendering stopped by user.")
    finally:
        vec_env.close()
