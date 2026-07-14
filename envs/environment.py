"""Gymnasium environment for a double inverted pendulum on a cart.

Physics simulated with Pymunk. Rendering via Pygame.
Observation: [cart_x, cart_vx, theta1, omega1, theta2, omega2]  shape (6,)
Action:      [force]  continuous in [-1, 1]                     shape (1,)
"""

import math
from typing import Any, Dict, Optional, Tuple

import gymnasium
import numpy as np
import pymunk
from gymnasium import spaces

try:
    import pygame
except ImportError:
    pygame = None


class DoublePendulumEnv(gymnasium.Env):
    """Cart–double-pendulum balancing task.

    A cart slides along a horizontal rail. Two poles are chained above it
    via revolute joints. The agent applies a horizontal force to the cart
    and must keep both poles upright simultaneously.

    Args:
        render_mode: ``"human"`` | ``"rgb_array"`` | ``None``
        reward_type: ``"baseline"`` (constant +1 alive) or
                     ``"shaped"`` (alive bonus minus angle/cart/velocity penalties).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # ── construction ──────────────────────────────────────────────────────

    def __init__(
        self,
        render_mode: Optional[str] = None,
        reward_type: str = "shaped",
    ) -> None:
        super().__init__()

        # physics constants
        self.gravity = 9.81          # m/s²
        self.dt = 0.02               # gym timestep  (50 Hz)
        self.substeps = 5            # pymunk sub-steps per gym step
        self.scale = 150.0           # pixels per metre

        # cart
        self.cart_mass = 1.0         # kg
        self.cart_w = 0.4            # m
        self.cart_h = 0.2            # m
        self.rail_limit = 2.4        # m each side
        self.force_mag = 10.0        # N  (action is scaled by this)

        # poles
        self.p1_mass = 0.5;  self.p1_len = 0.5   # kg, m
        self.p2_mass = 0.5;  self.p2_len = 0.5

        # termination
        self.angle_limit = 0.418     # rad ≈ 24°
        self.cart_limit = 2.4        # m
        self.max_steps = 1000

        # reward
        if reward_type not in ("baseline", "shaped"):
            raise ValueError(f"reward_type must be 'baseline' or 'shaped', got '{reward_type}'")
        self.reward_type = reward_type

        # spaces
        obs_high = np.array(
            [self.cart_limit * 2, np.inf, math.pi, np.inf, math.pi, np.inf],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        # rendering state (lazily initialised)
        self.render_mode = render_mode
        self.screen_w = 800
        self.screen_h = 600
        self._screen: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._font = None

        # pymunk handles (set by _build_space)
        self.space: Optional[pymunk.Space] = None
        self.cart_body: Optional[pymunk.Body] = None
        self.pole1_body: Optional[pymunk.Body] = None
        self.pole2_body: Optional[pymunk.Body] = None

        self._step_count = 0
        self._last_reward = 0.0
        self._last_obs = np.zeros(6, dtype=np.float32)

    # ── gymnasium API ─────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        self._last_reward = 0.0

        cx = float(self.np_random.uniform(-0.05, 0.05))
        t1 = float(self.np_random.uniform(-0.05, 0.05))
        t2 = float(self.np_random.uniform(-0.05, 0.05))
        self._build_space(cx, t1, t2)

        obs = self._get_obs()
        self._last_obs = obs
        return obs, self._make_info(obs)

    def step(
        self, action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.space is None:
            raise RuntimeError("Call reset() before step()")

        force = float(np.clip(action[0], -1.0, 1.0)) * self.force_mag
        self.cart_body.apply_force_at_local_point((force * self.scale, 0), (0, 0))

        sub_dt = self.dt / self.substeps
        for _ in range(self.substeps):
            self.space.step(sub_dt)

        obs = self._get_obs()
        self._last_obs = obs
        reward = self._compute_reward(obs)
        self._last_reward = reward
        terminated = self._is_terminated(obs)
        self._step_count += 1
        truncated = self._step_count >= self.max_steps

        return obs, reward, terminated, truncated, self._make_info(obs)

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None
        if pygame is None:
            raise ImportError("pygame is required for rendering")

        if self._screen is None:
            self._init_render()

        self._draw_frame()

        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.render_mode = None
                    self.close()
            return None

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self._screen)), (1, 0, 2)
        ).copy()

    def close(self) -> None:
        if self._screen is not None:
            pygame.quit()
            self._screen = None
            self._clock = None
            self._font = None

    # ── pymunk construction ───────────────────────────────────────────────

    def _build_space(self, cart_x: float, theta1: float, theta2: float) -> None:
        s = self.scale

        space = pymunk.Space()
        space.gravity = (0, -self.gravity * s)
        space.iterations = 30

        static = space.static_body

        # cart (infinite moment → no rotation)
        cart_body = pymunk.Body(self.cart_mass, float('inf'))
        cart_body.position = (cart_x * s, 0)
        hw = self.cart_w * s / 2
        hh = self.cart_h * s / 2
        cart_shape = pymunk.Poly(cart_body, [
            (-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh),
        ])
        cart_shape.filter = pymunk.ShapeFilter(group=1)
        space.add(cart_body, cart_shape)

        # groove joint constrains cart to horizontal rail
        groove = pymunk.GrooveJoint(
            static, cart_body,
            (-self.rail_limit * s, 0), (self.rail_limit * s, 0),
            (0, 0),
        )
        space.add(groove)

        # pivot 1: top-centre of cart
        pivot1 = (cart_body.position.x, hh)

        # pole 1 (lower)
        pole1_body = self._make_pole(space, self.p1_mass, self.p1_len, theta1, pivot1)
        space.add(pymunk.PivotJoint(cart_body, pole1_body, pivot1))

        # pivot 2: top end of pole 1
        half_L1 = self.p1_len * s / 2
        pivot2 = (
            pole1_body.position.x - half_L1 * math.sin(theta1),
            pole1_body.position.y + half_L1 * math.cos(theta1),
        )

        # pole 2 (upper)
        pole2_body = self._make_pole(space, self.p2_mass, self.p2_len, theta2, pivot2)
        space.add(pymunk.PivotJoint(pole1_body, pole2_body, pivot2))

        self.space = space
        self.cart_body = cart_body
        self.pole1_body = pole1_body
        self.pole2_body = pole2_body

    def _make_pole(
        self,
        space: pymunk.Space,
        mass: float,
        length: float,
        angle: float,
        pivot: Tuple[float, float],
    ) -> pymunk.Body:
        s = self.scale
        half_L = length * s / 2

        moment = pymunk.moment_for_segment(mass, (0, -half_L), (0, half_L), 1)
        body = pymunk.Body(mass, moment)
        body.angle = angle
        body.position = (
            pivot[0] - half_L * math.sin(angle),
            pivot[1] + half_L * math.cos(angle),
        )

        shape = pymunk.Segment(body, (0, -half_L), (0, half_L), radius=2)
        shape.filter = pymunk.ShapeFilter(group=1)
        space.add(body, shape)
        return body

    # ── state helpers ─────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        s = self.scale
        return np.array([
            self.cart_body.position.x / s,
            self.cart_body.velocity.x / s,
            _wrap_angle(self.pole1_body.angle),
            self.pole1_body.angular_velocity,
            _wrap_angle(self.pole2_body.angle),
            self.pole2_body.angular_velocity,
        ], dtype=np.float32)

    def _compute_reward(self, obs: np.ndarray) -> float:
        if self.reward_type == "baseline":
            return 1.0

        x, _, t1, w1, t2, w2 = obs
        alive_bonus = 1.0
        angle_penalty = 1.0 * (t1 ** 2 + t2 ** 2)
        cart_penalty = 0.5 * (x ** 2)
        velocity_penalty = 0.01 * (w1 ** 2 + w2 ** 2)
        return float(alive_bonus - angle_penalty - cart_penalty - velocity_penalty)

    def _is_terminated(self, obs: np.ndarray) -> bool:
        x, _, t1, _, t2, _ = obs
        return bool(
            abs(t1) > self.angle_limit
            or abs(t2) > self.angle_limit
            or abs(x) > self.cart_limit
        )

    def _make_info(self, obs: np.ndarray) -> Dict[str, Any]:
        return {
            "cart_position": float(obs[0]),
            "cart_velocity": float(obs[1]),
            "pole1_angle": float(obs[2]),
            "pole1_angular_velocity": float(obs[3]),
            "pole2_angle": float(obs[4]),
            "pole2_angular_velocity": float(obs[5]),
            "step": self._step_count,
        }

    # ── rendering ─────────────────────────────────────────────────────────

    def _pole_endpoints(self) -> Tuple[
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
    ]:
        s = self.scale
        half_L1 = self.p1_len * s / 2
        half_L2 = self.p2_len * s / 2

        cart = tuple(self.cart_body.position)
        j1 = tuple(self.pole1_body.local_to_world((0, -half_L1)))
        j2 = tuple(self.pole1_body.local_to_world((0, half_L1)))
        tip = tuple(self.pole2_body.local_to_world((0, half_L2)))
        return cart, j1, j2, tip

    def _to_screen(self, wx: float, wy: float) -> Tuple[int, int]:
        return (
            int(wx + self.screen_w / 2),
            int(self.screen_h / 2 - wy),
        )

    def _init_render(self) -> None:
        pygame.init()
        if self.render_mode == "human":
            self._screen = pygame.display.set_mode((self.screen_w, self.screen_h))
            pygame.display.set_caption("Double Inverted Pendulum")
        else:
            self._screen = pygame.Surface((self.screen_w, self.screen_h))
        self._clock = pygame.time.Clock()
        pygame.font.init()
        self._font = pygame.font.SysFont("consolas", 16)

    def _draw_frame(self) -> None:
        BG       = (26,  26,  46)
        RAIL     = (64,  64,  96)
        TICK     = (80,  80, 120)
        CART_CLR = (74, 144, 217)
        CART_BRD = (100, 180, 255)
        POLE1    = (232, 131,  58)
        POLE2    = (56,  201, 177)
        JOINT    = (255, 255, 255)
        TIP_CLR  = (255, 230,  80)
        HUD      = (200, 200, 220)

        self._screen.fill(BG)

        # rail
        rl = self._to_screen(-self.rail_limit * self.scale, 0)
        rr = self._to_screen(self.rail_limit * self.scale, 0)
        pygame.draw.line(self._screen, RAIL, rl, rr, 3)

        for m in np.arange(-self.rail_limit, self.rail_limit + 0.01, 0.5):
            tp = self._to_screen(m * self.scale, 0)
            pygame.draw.line(self._screen, TICK, (tp[0], tp[1] - 5), (tp[0], tp[1] + 5), 1)

        cart, j1, j2, tip = self._pole_endpoints()
        cs  = self._to_screen(*cart)
        j1s = self._to_screen(*j1)
        j2s = self._to_screen(*j2)
        ts  = self._to_screen(*tip)

        # cart body
        cw = int(self.cart_w * self.scale)
        ch = int(self.cart_h * self.scale)
        rect = pygame.Rect(cs[0] - cw // 2, cs[1] - ch // 2, cw, ch)
        pygame.draw.rect(self._screen, CART_CLR, rect, border_radius=4)
        pygame.draw.rect(self._screen, CART_BRD, rect, 2, border_radius=4)

        # poles
        pygame.draw.line(self._screen, POLE1, j1s, j2s, 6)
        pygame.draw.line(self._screen, POLE2, j2s, ts, 6)

        # joints & tip
        pygame.draw.circle(self._screen, JOINT, j1s, 5)
        pygame.draw.circle(self._screen, JOINT, j2s, 5)
        pygame.draw.circle(self._screen, TIP_CLR, ts, 4)

        # hud — reuse cached obs instead of re-querying pymunk
        obs = self._last_obs
        lines = [
            f"Step: {self._step_count:>5d}",
            f"Reward: {self._last_reward:>+8.3f}",
            f"\u03b81: {math.degrees(obs[2]):>+7.2f}\u00b0",
            f"\u03b82: {math.degrees(obs[4]):>+7.2f}\u00b0",
            f"Mode: {self.reward_type}",
        ]
        for i, line in enumerate(lines):
            surf = self._font.render(line, True, HUD)
            self._screen.blit(surf, (10, 10 + i * 22))


def _wrap_angle(angle: float) -> float:
    """Normalise an angle to [-π, π]."""
    return math.atan2(math.sin(angle), math.cos(angle))
