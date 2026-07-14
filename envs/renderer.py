"""Pygame-based renderer for the double inverted pendulum environment.

Handles all visual output: coordinate transforms, drawing the cart/poles/
joints/rail/HUD, and supporting both ``human`` (live window) and
``rgb_array`` (off-screen NumPy frame) render modes.

Coordinate convention:
    Pymunk:  Y ↑  (origin at rail centre)
    Pygame:  Y ↓  (origin at top-left)

    screen_x =  world_x + screen_width  / 2
    screen_y = -world_y + screen_height / 2
"""

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import pygame
    import pygame.freetype
except ImportError:
    pygame = None  # Allow import without pygame (e.g., for headless training)


# ── Colour palette ────────────────────────────────────────────────────────

COLORS = {
    "background": (26, 26, 46),       # #1a1a2e
    "rail": (64, 64, 96),             # #404060
    "cart": (74, 144, 217),           # #4a90d9
    "pole1": (232, 131, 58),          # #e8833a
    "pole2": (56, 201, 177),          # #38c9b1
    "joint": (255, 255, 255),         # white
    "pole2_tip": (255, 230, 80),      # yellow
    "hud_text": (200, 200, 220),      # light grey
    "rail_mark": (80, 80, 120),       # tick marks on rail
}


class Renderer:
    """Pygame renderer for the double inverted pendulum.

    Args:
        config: Dictionary loaded from ``configs/env_config.yaml``.
        render_mode: ``"human"`` for a live window, ``"rgb_array"`` for
            off-screen rendering, or ``None`` to skip initialisation.
    """

    def __init__(self, config: Dict[str, Any], render_mode: Optional[str]) -> None:
        self.render_mode = render_mode

        rendering = config.get("rendering", {})
        self.screen_width: int = rendering.get("screen_width", 800)
        self.screen_height: int = rendering.get("screen_height", 600)
        self.fps: int = rendering.get("fps", 50)
        self.scale: float = rendering.get("scale", 100.0)

        # Physics dimensions for drawing
        cart_cfg = config.get("cart", {})
        self.cart_width_px: float = cart_cfg.get("width", 0.4) * self.scale
        self.cart_height_px: float = cart_cfg.get("height", 0.2) * self.scale
        self.rail_limit: float = cart_cfg.get("rail_limit", 2.4)

        # Pole visual thickness (pixels)
        self.pole_thickness: int = 6
        self.joint_radius: int = 5
        self.tip_radius: int = 4

        # Pygame state (lazily initialised)
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font: Optional[pygame.font.Font] = None
        self._initialised: bool = False

    # ── Public API ────────────────────────────────────────────────────────

    def draw(
        self,
        endpoints: Dict[str, Tuple[float, float]],
        step_count: int,
        reward: float,
        theta1: float,
        theta2: float,
    ) -> Optional[np.ndarray]:
        """Render one frame of the environment.

        Args:
            endpoints: Dictionary from ``PhysicsEngine.get_pole_endpoints()``
                with keys ``cart``, ``joint1``, ``joint2``, ``pole2_tip``
                in Pymunk pixel coordinates.
            step_count: Current step number in the episode.
            reward: Current step reward.
            theta1: Pole 1 angle (radians, for HUD).
            theta2: Pole 2 angle (radians, for HUD).

        Returns:
            ``None`` if ``render_mode == "human"`` (draws to screen).
            ``np.ndarray`` of shape ``(H, W, 3)`` if ``render_mode == "rgb_array"``.
        """
        if self.render_mode is None:
            return None

        if not self._initialised:
            self._init_pygame()

        surface = self.screen

        # ── Background ────────────────────────────────────────────────────
        surface.fill(COLORS["background"])

        # ── Rail ──────────────────────────────────────────────────────────
        self._draw_rail(surface)

        # ── Cart ──────────────────────────────────────────────────────────
        cart_screen = self._world_to_screen(*endpoints["cart"])
        self._draw_cart(surface, cart_screen)

        # ── Poles ─────────────────────────────────────────────────────────
        j1_screen = self._world_to_screen(*endpoints["joint1"])
        j2_screen = self._world_to_screen(*endpoints["joint2"])
        tip_screen = self._world_to_screen(*endpoints["pole2_tip"])

        # Pole 1: from joint1 to joint2
        self._draw_pole(surface, j1_screen, j2_screen, COLORS["pole1"])
        # Pole 2: from joint2 to tip
        self._draw_pole(surface, j2_screen, tip_screen, COLORS["pole2"])

        # ── Joints and tip ────────────────────────────────────────────────
        pygame.draw.circle(surface, COLORS["joint"], j1_screen, self.joint_radius)
        pygame.draw.circle(surface, COLORS["joint"], j2_screen, self.joint_radius)
        pygame.draw.circle(surface, COLORS["pole2_tip"], tip_screen, self.tip_radius)

        # ── HUD ───────────────────────────────────────────────────────────
        self._draw_hud(surface, step_count, reward, theta1, theta2)

        # ── Output ────────────────────────────────────────────────────────
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.fps)
            # Pump events to keep the window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
            return None

        elif self.render_mode == "rgb_array":
            # Return pixels as NumPy array (H, W, 3)
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
            ).copy()

        return None

    def close(self) -> None:
        """Destroy the Pygame window and clean up resources."""
        if self._initialised:
            pygame.quit()
            self._initialised = False
            self.screen = None

    # ── Private helpers ───────────────────────────────────────────────────

    def _init_pygame(self) -> None:
        """Lazily initialise Pygame display and resources."""
        if pygame is None:
            raise ImportError("pygame is required for rendering but is not installed.")

        pygame.init()

        if self.render_mode == "human":
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
            pygame.display.set_caption("Double Inverted Pendulum")
        else:
            # Off-screen surface for rgb_array mode
            self.screen = pygame.Surface((self.screen_width, self.screen_height))

        self.clock = pygame.time.Clock()

        # Use the default system font at a reasonable size
        pygame.font.init()
        self.font = pygame.font.SysFont("consolas", 16)

        self._initialised = True

    def _world_to_screen(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert Pymunk world coordinates (pixels) to Pygame screen coordinates.

        Args:
            wx: X position in Pymunk pixels (origin at rail centre).
            wy: Y position in Pymunk pixels (Y-up).

        Returns:
            ``(screen_x, screen_y)`` with Y flipped for Pygame.
        """
        sx = int(wx + self.screen_width / 2)
        sy = int(self.screen_height / 2 - wy)
        return (sx, sy)

    def _draw_rail(self, surface: pygame.Surface) -> None:
        """Draw the horizontal rail with tick marks."""
        rail_left = self._world_to_screen(-self.rail_limit * self.scale, 0)
        rail_right = self._world_to_screen(+self.rail_limit * self.scale, 0)
        pygame.draw.line(surface, COLORS["rail"], rail_left, rail_right, 3)

        # Tick marks every 0.5 m
        for offset_m in np.arange(-self.rail_limit, self.rail_limit + 0.01, 0.5):
            tick_pos = self._world_to_screen(offset_m * self.scale, 0)
            tick_top = (tick_pos[0], tick_pos[1] - 5)
            tick_bot = (tick_pos[0], tick_pos[1] + 5)
            pygame.draw.line(surface, COLORS["rail_mark"], tick_top, tick_bot, 1)

    def _draw_cart(
        self, surface: pygame.Surface, screen_pos: Tuple[int, int]
    ) -> None:
        """Draw the cart as a filled rectangle centred on ``screen_pos``."""
        w = int(self.cart_width_px)
        h = int(self.cart_height_px)
        rect = pygame.Rect(
            screen_pos[0] - w // 2,
            screen_pos[1] - h // 2,
            w,
            h,
        )
        pygame.draw.rect(surface, COLORS["cart"], rect, border_radius=4)
        # Subtle border
        pygame.draw.rect(surface, (100, 180, 255), rect, width=2, border_radius=4)

    def _draw_pole(
        self,
        surface: pygame.Surface,
        start: Tuple[int, int],
        end: Tuple[int, int],
        colour: Tuple[int, int, int],
    ) -> None:
        """Draw a pole as a thick anti-aliased line."""
        pygame.draw.line(surface, colour, start, end, self.pole_thickness)

    def _draw_hud(
        self,
        surface: pygame.Surface,
        step: int,
        reward: float,
        theta1: float,
        theta2: float,
    ) -> None:
        """Render the heads-up display with metrics."""
        lines = [
            f"Step: {step:>5d}",
            f"Reward: {reward:>+8.3f}",
            f"θ₁: {math.degrees(theta1):>+7.2f}°",
            f"θ₂: {math.degrees(theta2):>+7.2f}°",
        ]
        y_offset = 10
        for line in lines:
            text_surface = self.font.render(line, True, COLORS["hud_text"])
            surface.blit(text_surface, (10, y_offset))
            y_offset += 22
