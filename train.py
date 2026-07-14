"""Train a PPO agent on the Double Inverted Pendulum environment."""

import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.environment import DoublePendulumEnv


# ── helpers ───────────────────────────────────────────────────────────────

def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def make_env(reward_type: str, render_mode=None):
    def _init():
        return DoublePendulumEnv(render_mode=render_mode, reward_type=reward_type)
    return _init


# ── callbacks ─────────────────────────────────────────────────────────────

class MetricsCallback(BaseCallback):
    """Log custom per-step metrics to TensorBoard."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos:
            info = infos[0]
            self.logger.record("custom/pole1_angle_deg", np.degrees(info.get("pole1_angle", 0)))
            self.logger.record("custom/pole2_angle_deg", np.degrees(info.get("pole2_angle", 0)))
            self.logger.record("custom/cart_position", info.get("cart_position", 0))
        return True


class GracefulSaveCallback(BaseCallback):
    """Save model on training end (including keyboard interrupts caught by the caller)."""

    def __init__(self, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        path = os.path.join(self.save_path, "final_model")
        self.model.save(path)
        if self.verbose:
            print(f"[callback] Final model saved to {path}.zip")


# ── training ──────────────────────────────────────────────────────────────

def build_callbacks(
    eval_env: VecNormalize,
    save_path: str,
    log_dir: str,
    n_envs: int,
) -> list:
    best_dir = os.path.join(save_path, "best")
    ensure_dirs(best_dir)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=log_dir,
        eval_freq=max(10_000 // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path=save_path,
        name_prefix="checkpoint",
        verbose=1,
    )

    return [eval_cb, checkpoint_cb, MetricsCallback(), GracefulSaveCallback(save_path)]


def train(args: argparse.Namespace) -> None:
    n_envs = 4
    seed = 42

    ensure_dirs(args.save_path, args.log_dir)
    tb_dir = os.path.join(args.log_dir, "tensorboard")
    ensure_dirs(tb_dir)

    # ── environments ──────────────────────────────────────────────────
    train_env = DummyVecEnv([make_env(args.reward_type) for _ in range(n_envs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env(args.reward_type)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms

    # ── model ─────────────────────────────────────────────────────────
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        seed=seed,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=1,
        tensorboard_log=tb_dir,
    )

    # ── CSV + TensorBoard logger ──────────────────────────────────────
    logger = configure(args.log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)

    # ── callbacks ─────────────────────────────────────────────────────
    callbacks = build_callbacks(eval_env, args.save_path, args.log_dir, n_envs)

    # ── train ─────────────────────────────────────────────────────────
    print(f"Training PPO | timesteps={args.timesteps} | reward_type={args.reward_type}")
    print(f"  save_path={args.save_path}")
    print(f"  log_dir={args.log_dir}")
    print(f"  n_envs={n_envs} | seed={seed}")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            log_interval=1,
            tb_log_name="PPO",
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # save model
        final_path = os.path.join(args.save_path, "final_model")
        model.save(final_path)
        print(f"Model saved to {final_path}.zip")

        # save normalisation stats (critical for evaluation)
        norm_path = os.path.join(args.save_path, "vecnormalize.pkl")
        train_env.save(norm_path)
        print(f"VecNormalize stats saved to {norm_path}")

        train_env.close()
        eval_env.close()

    print("Training complete.")


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO on DoublePendulum-v0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--timesteps", type=int, default=500_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--reward_type", type=str, default="shaped", choices=["baseline", "shaped"],
        help="Reward function variant",
    )
    parser.add_argument(
        "--save_path", type=str, default="models",
        help="Directory to save trained models and checkpoints",
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs",
        help="Directory for CSV and TensorBoard logs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
