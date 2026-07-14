"""envs package — registers the custom Gymnasium environment."""

import gymnasium

gymnasium.register(
    id="DoublePendulum-v0",
    entry_point="envs.environment:DoublePendulumEnv",
    max_episode_steps=1000,
)
