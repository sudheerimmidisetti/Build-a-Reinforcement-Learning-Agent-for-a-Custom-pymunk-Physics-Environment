"""Smoke tests for the training pipeline (Phase 5 gate)."""

import os
import pytest
from pathlib import Path

from utils.helpers import load_config, get_project_root

CONFIG_PATH = str(get_project_root() / "configs" / "env_config.yaml")
TRAIN_CONFIG_PATH = str(get_project_root() / "configs" / "train_config.yaml")


@pytest.fixture
def env_config():
    return load_config(CONFIG_PATH)


@pytest.fixture
def train_config():
    return load_config(TRAIN_CONFIG_PATH)


class TestHyperparams:
    """Test hyperparameter loading."""

    def test_get_ppo_params(self, train_config):
        from training.hyperparams import get_algo_params

        params = get_algo_params(train_config, algo_override="PPO")
        assert "learning_rate" in params
        assert "n_steps" in params
        assert "batch_size" in params

    def test_get_sac_params(self, train_config):
        from training.hyperparams import get_algo_params

        params = get_algo_params(train_config, algo_override="SAC")
        assert "learning_rate" in params
        assert "buffer_size" in params

    def test_unknown_algo_raises(self, train_config):
        from training.hyperparams import get_algo_params

        with pytest.raises(ValueError, match="Unsupported algorithm"):
            get_algo_params(train_config, algo_override="UNKNOWN")


class TestTrainingSmokeTest:
    """Quick smoke test: train for very few steps to verify the pipeline."""

    @pytest.mark.slow
    def test_short_training_run(self, env_config, train_config, tmp_path):
        """Train for 512 steps and verify model is saved."""
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        except ImportError:
            pytest.skip("stable-baselines3 not installed")

        import envs  # noqa: F401
        from envs.double_pendulum_env import DoublePendulumEnv

        # Create a single env
        def make_env():
            return DoublePendulumEnv(render_mode=None, env_config=env_config)

        vec_env = DummyVecEnv([make_env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

        # Train for minimal steps
        model = PPO("MlpPolicy", vec_env, n_steps=64, batch_size=32, verbose=0)
        model.learn(total_timesteps=512)

        # Save model
        model_path = str(tmp_path / "test_model")
        model.save(model_path)
        assert os.path.exists(model_path + ".zip")

        # Save VecNormalize stats
        norm_path = str(tmp_path / "vecnormalize.pkl")
        vec_env.save(norm_path)
        assert os.path.exists(norm_path)

        # Reload model
        loaded_model = PPO.load(model_path)
        assert loaded_model is not None

        # Predict with loaded model
        obs = vec_env.reset()
        action, _ = loaded_model.predict(obs, deterministic=True)
        assert action.shape == (1, 1)

        vec_env.close()
