"""Custom Gymnasium environment for the double inverted pendulum.

This module subclasses ``gymnasium.Env`` and wires together the
``PhysicsEngine`` (Pymunk simulation) and ``Renderer`` (Pygame display)
to produce a fully SB3-compatible continuous-control environment.

Registration happens in ``envs/__init__.py`` under the id
``DoublePendulum-v0``.
"""

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

from envs.physics_engine import PhysicsEngine
from envs.renderer import Renderer
from utils.helpers import get_project_root, load_config


class DoublePendulumEnv(gymnasium.Env):
    """A cart with two chained pendulum poles — balance them both upright.

    **Observation** (6-D continuous):
        ``[cart_x, cart_vx, θ₁, ω₁, θ₂, ω₂]``

    **Action** (1-D continuous):
        Horizontal force on the cart, normalised to ``[-1, 1]``.

    **Reward** (dense, shaped):
        ``alive_bonus − α(θ₁² + θ₂²) − β·x² − γ(ω₁² + ω₂²)``

    **Termination**:
        Any pole angle exceeds threshold or cart leaves the rail.

    **Truncation**:
        Episode reaches ``max_steps``.

    Args:
        render_mode: ``"human"``, ``"rgb_array"``, or ``None``.
        env_config: Optional config dict.  If ``None``, loads
            ``configs/env_config.yaml`` from the project root.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        env_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        # ── Load config ───────────────────────────────────────────────────
        if env_config is None:
            config_path = get_project_root() / "configs" / "env_config.yaml"
            env_config = load_config(str(config_path))
        self.config = env_config

        # ── Termination thresholds ────────────────────────────────────────
        term = env_config["termination"]
        self.max_pole_angle: float = term["max_pole_angle"]    # rad
        self.max_cart_disp: float = term["max_cart_displacement"]  # m

        # ── Episode limits ────────────────────────────────────────────────
        self.max_steps: int = env_config["episode"]["max_steps"]

        # ── Reward coefficients ───────────────────────────────────────────
        rew = env_config["reward"]
        self.alive_bonus: float = rew["alive_bonus"]
        self.angle_coeff: float = rew["angle_penalty_coeff"]
        self.cart_coeff: float = rew["cart_penalty_coeff"]
        self.vel_coeff: float = rew["velocity_penalty_coeff"]

        # ── Observation space ─────────────────────────────────────────────
        # [x, vx, θ₁, ω₁, θ₂, ω₂]
        high = np.array(
            [
                self.max_cart_disp * 2,  # x can slightly exceed rail before termination
                np.inf,                  # vx unbounded
                math.pi,                # θ₁
                np.inf,                  # ω₁
                math.pi,                # θ₂
                np.inf,                  # ω₂
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # ── Action space ─────────────────────────────────────────────────
        # Normalised force in [-1, 1]; scaled inside step()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # ── Physics engine ────────────────────────────────────────────────
        self.physics = PhysicsEngine(env_config)

        # ── Renderer ──────────────────────────────────────────────────────
        self.render_mode = render_mode
        self.renderer = Renderer(env_config, render_mode)

        # ── Internal state ────────────────────────────────────────────────
        self.step_count: int = 0
        self.last_reward: float = 0.0

    # ── Gymnasium API ─────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to a fresh episode.

        Args:
            seed: Optional RNG seed for reproducibility.
            options: Unused; present for Gymnasium API compliance.

        Returns:
            Tuple of ``(observation, info)`` where observation has shape
            ``(6,)`` and info is a dict with diagnostic values.
        """
        super().reset(seed=seed)

        self.step_count = 0
        self.last_reward = 0.0

        # PhysicsEngine rebuilds the entire Pymunk space from scratch
        state = self.physics.reset(self.np_random)

        obs = state.astype(np.float32)
        info = self._build_info(state)

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: Array of shape ``(1,)`` with values in ``[-1, 1]``.

        Returns:
            ``(observation, reward, terminated, truncated, info)``
        """
        # 1. Clip and scale action → force in Newtons
        clipped = np.clip(action, -1.0, 1.0)
        force = float(clipped[0]) * self.physics.force_limit

        # 2. Apply force and advance simulation
        self.physics.apply_force(force)
        self.physics.step()

        # 3. Read new state
        state = self.physics.get_state()
        obs = state.astype(np.float32)

        # 4. Compute reward
        reward = self._compute_reward(state)
        self.last_reward = reward

        # 5. Termination & truncation
        terminated = self._check_termination(state)
        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        # 6. Info
        info = self._build_info(state)

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Render the current state of the environment.

        Returns:
            ``None`` for ``human`` mode (draws to window).
            ``np.ndarray`` of shape ``(H, W, 3)`` for ``rgb_array`` mode.
        """
        if self.render_mode is None:
            return None

        endpoints = self.physics.get_pole_endpoints()
        state = self.physics.get_state()

        return self.renderer.draw(
            endpoints=endpoints,
            step_count=self.step_count,
            reward=self.last_reward,
            theta1=float(state[2]),
            theta2=float(state[4]),
        )

    def close(self) -> None:
        """Clean up Pygame and Pymunk resources."""
        self.renderer.close()

    # ── Private helpers ───────────────────────────────────────────────────

    def _compute_reward(self, state: np.ndarray) -> float:
        """Compute the dense shaped reward for the current state.

        Formula:
            r = alive_bonus
              − α · (θ₁² + θ₂²)
              − β · x²
              − γ · (ω₁² + ω₂²)

        Args:
            state: Array ``[x, vx, θ₁, ω₁, θ₂, ω₂]``.

        Returns:
            Scalar reward.
        """
        x, vx, theta1, omega1, theta2, omega2 = state

        angle_penalty = self.angle_coeff * (theta1 ** 2 + theta2 ** 2)
        cart_penalty = self.cart_coeff * (x ** 2)
        vel_penalty = self.vel_coeff * (omega1 ** 2 + omega2 ** 2)

        reward = self.alive_bonus - angle_penalty - cart_penalty - vel_penalty

        return float(reward)

    def _check_termination(self, state: np.ndarray) -> bool:
        """Check whether the episode should terminate (failure).

        Conditions:
            - |θ₁| > max_pole_angle
            - |θ₂| > max_pole_angle
            - |x|  > max_cart_displacement

        Args:
            state: Array ``[x, vx, θ₁, ω₁, θ₂, ω₂]``.

        Returns:
            ``True`` if any termination condition is met.
        """
        x = state[0]
        theta1 = state[2]
        theta2 = state[4]

        return bool(
            abs(theta1) > self.max_pole_angle
            or abs(theta2) > self.max_pole_angle
            or abs(x) > self.max_cart_disp
        )

    def _build_info(self, state: np.ndarray) -> Dict[str, Any]:
        """Build the info dict returned by reset/step.

        Args:
            state: Array ``[x, vx, θ₁, ω₁, θ₂, ω₂]``.

        Returns:
            Dictionary with diagnostic values.
        """
        return {
            "cart_position": float(state[0]),
            "cart_velocity": float(state[1]),
            "pole1_angle": float(state[2]),
            "pole1_angular_velocity": float(state[3]),
            "pole2_angle": float(state[4]),
            "pole2_angular_velocity": float(state[5]),
            "step": self.step_count,
        }
