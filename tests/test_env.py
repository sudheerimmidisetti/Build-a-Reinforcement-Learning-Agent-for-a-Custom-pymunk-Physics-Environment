"""Tests for the DoublePendulumEnv Gymnasium environment (Phase 3 gate)."""

import math

import gymnasium
import numpy as np
import pytest

# Importing envs triggers registration of DoublePendulum-v0
import envs  # noqa: F401
from envs.double_pendulum_env import DoublePendulumEnv
from utils.helpers import load_config, get_project_root

CONFIG_PATH = str(get_project_root() / "configs" / "env_config.yaml")


@pytest.fixture
def config():
    return load_config(CONFIG_PATH)


@pytest.fixture
def env(config):
    """Create an environment with no rendering for tests."""
    e = DoublePendulumEnv(render_mode=None, env_config=config)
    yield e
    e.close()


class TestEnvCreation:
    """Test that the environment can be created correctly."""

    def test_direct_instantiation(self, config):
        env = DoublePendulumEnv(render_mode=None, env_config=config)
        assert env is not None
        env.close()

    def test_gymnasium_make(self):
        env = gymnasium.make("DoublePendulum-v0", render_mode=None)
        assert env is not None
        env.close()


class TestSpaces:
    """Test observation and action space definitions."""

    def test_observation_space_shape(self, env):
        assert env.observation_space.shape == (6,)

    def test_observation_space_dtype(self, env):
        assert env.observation_space.dtype == np.float32

    def test_action_space_shape(self, env):
        assert env.action_space.shape == (1,)

    def test_action_space_dtype(self, env):
        assert env.action_space.dtype == np.float32

    def test_action_space_bounds(self, env):
        assert env.action_space.low[0] == pytest.approx(-1.0)
        assert env.action_space.high[0] == pytest.approx(1.0)


class TestReset:
    """Test the reset method."""

    def test_returns_tuple(self, env):
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_obs_shape(self, env):
        obs, info = env.reset()
        assert obs.shape == (6,)

    def test_obs_dtype(self, env):
        obs, info = env.reset()
        assert obs.dtype == np.float32

    def test_obs_in_space(self, env):
        obs, info = env.reset()
        assert env.observation_space.contains(obs), (
            f"Observation {obs} not in space"
        )

    def test_info_is_dict(self, env):
        _, info = env.reset()
        assert isinstance(info, dict)

    def test_info_has_expected_keys(self, env):
        _, info = env.reset()
        expected_keys = {
            "cart_position",
            "cart_velocity",
            "pole1_angle",
            "pole1_angular_velocity",
            "pole2_angle",
            "pole2_angular_velocity",
            "step",
        }
        assert expected_keys.issubset(info.keys())

    def test_step_counter_resets(self, env):
        env.reset()
        # Step a few times
        for _ in range(10):
            env.step(env.action_space.sample())
        # Reset again
        _, info = env.reset()
        assert info["step"] == 0

    def test_seeded_reset_is_deterministic(self, config):
        env1 = DoublePendulumEnv(render_mode=None, env_config=config)
        env2 = DoublePendulumEnv(render_mode=None, env_config=config)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        np.testing.assert_array_almost_equal(obs1, obs2)

        env1.close()
        env2.close()


class TestStep:
    """Test the step method."""

    def test_returns_5_tuple(self, env):
        env.reset(seed=42)
        result = env.step(env.action_space.sample())
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_obs_shape_after_step(self, env):
        env.reset(seed=42)
        obs, _, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
        assert obs.shape == (6,)

    def test_obs_dtype_after_step(self, env):
        env.reset(seed=42)
        obs, _, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
        assert obs.dtype == np.float32

    def test_reward_is_float(self, env):
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
        assert isinstance(reward, float)

    def test_reward_is_finite(self, env):
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
        assert math.isfinite(reward)

    def test_terminated_is_bool(self, env):
        env.reset(seed=42)
        _, _, terminated, _, _ = env.step(env.action_space.sample())
        assert isinstance(terminated, bool)

    def test_truncated_is_bool(self, env):
        env.reset(seed=42)
        _, _, _, truncated, _ = env.step(env.action_space.sample())
        assert isinstance(truncated, bool)

    def test_info_is_dict_after_step(self, env):
        env.reset(seed=42)
        _, _, _, _, info = env.step(env.action_space.sample())
        assert isinstance(info, dict)

    def test_action_clipping(self, env):
        """Actions outside [-1, 1] should be clipped, not crash."""
        env.reset(seed=42)
        # Extreme actions
        obs, r, d, t, info = env.step(np.array([100.0], dtype=np.float32))
        assert np.all(np.isfinite(obs))
        obs, r, d, t, info = env.step(np.array([-100.0], dtype=np.float32))
        assert np.all(np.isfinite(obs))


class TestTermination:
    """Test episode termination logic."""

    def test_terminates_when_pole_falls(self, env):
        """Running with zero force should eventually terminate (poles fall)."""
        env.reset(seed=42)
        terminated = False
        for _ in range(500):
            _, _, terminated, _, _ = env.step(np.array([0.0], dtype=np.float32))
            if terminated:
                break
        assert terminated, "Episode should terminate when poles fall"

    def test_truncation_at_max_steps(self, env):
        """If we somehow keep it alive, truncation happens at max_steps."""
        env.reset(seed=42)
        truncated = False
        terminated = False
        for i in range(env.max_steps + 10):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                break
        assert terminated or truncated, "Episode should end eventually"


class TestReward:
    """Test the reward function."""

    def test_reward_at_perfect_balance(self, env):
        """When perfectly balanced, reward should be close to alive_bonus."""
        env.reset(seed=42)
        # The first step from near-zero state should have reward ≈ alive_bonus
        obs, reward, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
        # Allow some tolerance since there's initial noise
        assert reward > 0, "Reward at near-balance should be positive"
        assert reward <= env.alive_bonus + 0.01, "Reward should not exceed alive_bonus"

    def test_reward_decreases_as_poles_fall(self, env):
        """Reward should decrease as the pendulum falls further."""
        env.reset(seed=42)
        rewards = []
        for _ in range(20):
            _, reward, terminated, _, _ = env.step(np.array([0.0], dtype=np.float32))
            rewards.append(reward)
            if terminated:
                break
        # The first reward should generally be higher than later ones
        if len(rewards) > 5:
            assert rewards[0] > rewards[-1], (
                "Reward should decrease as poles deviate"
            )


class TestRandomActionLoop:
    """Stress test: run many steps with random actions."""

    def test_1000_random_steps(self, env):
        """Run 1000 steps with random actions — should not crash."""
        env.reset(seed=42)
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            assert np.all(np.isfinite(obs)), f"Non-finite obs: {obs}"
            assert math.isfinite(reward), f"Non-finite reward: {reward}"

            if terminated or truncated:
                obs, info = env.reset()


class TestSB3Compatibility:
    """Test compatibility with Stable-Baselines3 utilities."""

    def test_check_env(self, config):
        """SB3's check_env should pass without warnings/errors."""
        try:
            from stable_baselines3.common.env_checker import check_env
        except ImportError:
            pytest.skip("stable-baselines3 not installed")

        env = DoublePendulumEnv(render_mode=None, env_config=config)
        # check_env raises or warns if there's a problem
        check_env(env, warn=True)
        env.close()
