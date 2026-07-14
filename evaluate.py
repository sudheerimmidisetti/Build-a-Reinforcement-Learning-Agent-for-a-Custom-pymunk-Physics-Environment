"""Evaluate a trained PPO agent and export gameplay as GIF."""

import argparse
import os
from typing import List

import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.environment import DoublePendulumEnv


# ── helpers ───────────────────────────────────────────────────────────────

def make_env(reward_type: str = "shaped", render_mode: str = None):
    def _init():
        return DoublePendulumEnv(render_mode=render_mode, reward_type=reward_type)
    return _init


def load_model(model_path: str):
    model_dir = os.path.dirname(model_path)

    env = DummyVecEnv([make_env(render_mode="rgb_array")])

    norm_path = os.path.join(model_dir, "vecnormalize.pkl")
    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False
        print(f"Loaded VecNormalize stats from {norm_path}")

    model = PPO.load(model_path, env=env)
    print(f"Loaded model from {model_path}")
    return model, env


def capture_episode(model, env) -> tuple:
    """Run one deterministic episode, collect rgb frames and total reward."""
    frames: List[np.ndarray] = []
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        frame = env.render()
        if frame is not None:
            if isinstance(frame, list):
                frame = frame[0]
            frames.append(frame)
        total_reward += float(reward[0])
        steps += 1
        done = dones[0]

    return frames, total_reward, steps


def save_gif(frames: List[np.ndarray], path: str, fps: int = 30) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    duration = int(1000 / fps)
    images = [Image.fromarray(f) for f in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"GIF saved to {path}  ({len(frames)} frames, {size_mb:.1f} MB)")


# ── evaluation ────────────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> None:
    model, env = load_model(args.model_path)

    all_rewards = []
    all_lengths = []
    best_frames = []
    best_reward = -float("inf")

    print(f"\nRunning {args.episodes} deterministic episodes...\n")

    for ep in range(args.episodes):
        frames, reward, steps = capture_episode(model, env)
        all_rewards.append(reward)
        all_lengths.append(steps)

        if reward > best_reward:
            best_reward = reward
            best_frames = frames

        print(f"  Episode {ep + 1:>3d}:  reward={reward:>8.2f}  steps={steps:>5d}")

    env.close()

    # ── summary ───────────────────────────────────────────────────────
    rewards = np.array(all_rewards)
    lengths = np.array(all_lengths)
    print(f"\n{'=' * 50}")
    print(f"  Episodes:     {args.episodes}")
    print(f"  Mean reward:  {rewards.mean():.2f} ± {rewards.std():.2f}")
    print(f"  Min / Max:    {rewards.min():.2f} / {rewards.max():.2f}")
    print(f"  Mean length:  {lengths.mean():.1f} ± {lengths.std():.1f}")
    print(f"{'=' * 50}\n")

    # ── save gif from best episode ────────────────────────────────────
    if best_frames:
        save_gif(best_frames, args.gif_path, fps=30)


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO agent on DoublePendulum-v0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path", type=str, default="models/final_model.zip",
        help="Path to saved PPO model (.zip)",
    )
    parser.add_argument(
        "--gif_path", type=str, default="media/agent_final.gif",
        help="Output path for the GIF recording",
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Number of evaluation episodes",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
