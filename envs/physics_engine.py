"""Pymunk-based physics engine for the double inverted pendulum.

Encapsulates all Pymunk logic: space creation, body/shape/joint setup,
force application, simulation stepping, and state extraction.

Coordinate convention:
    - Y-axis points UP (Pymunk default).
    - θ = 0 means pole is vertical (upright).
    - Positive θ = counter-clockwise.
    - 1 metre = ``scale`` pixels (default 100).
"""

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pymunk


class PhysicsEngine:
    """2-D rigid-body simulation of a cart with two chained pendulum poles.

    The cart slides along a horizontal rail (``GrooveJoint``).
    Pole 1 is attached to the cart via a ``PivotJoint``.
    Pole 2 is attached to the top of pole 1 via a second ``PivotJoint``.

    Args:
        config: Dictionary loaded from ``configs/env_config.yaml``.
    """

    # ── Construction ──────────────────────────────────────────────────────

    def __init__(self, config: Dict[str, Any]) -> None:
        # Unpack physics parameters
        phys = config["physics"]
        self.gravity: float = phys["gravity"]  # m/s²
        self.dt: float = phys["dt"]  # Gym timestep (s)
        self.substeps: int = phys["substeps"]
        self.pymunk_dt: float = self.dt / self.substeps

        # Cart parameters
        cart_cfg = config["cart"]
        self.cart_mass: float = cart_cfg["mass"]
        self.cart_width: float = cart_cfg["width"]
        self.cart_height: float = cart_cfg["height"]
        self.rail_limit: float = cart_cfg["rail_limit"]  # m
        self.force_limit: float = cart_cfg["force_limit"]  # N

        # Pole parameters
        p1 = config["pole1"]
        self.pole1_mass: float = p1["mass"]
        self.pole1_length: float = p1["length"]

        p2 = config["pole2"]
        self.pole2_mass: float = p2["mass"]
        self.pole2_length: float = p2["length"]

        # Joint damping
        self.joint_damping: float = config["joints"]["damping"]

        # Scale: pixels per metre (for internal Pymunk coords)
        self.scale: float = config.get("rendering", {}).get("scale", 100.0)

        # Will be populated by reset()
        self.space: Optional[pymunk.Space] = None
        self.cart_body: Optional[pymunk.Body] = None
        self.pole1_body: Optional[pymunk.Body] = None
        self.pole2_body: Optional[pymunk.Body] = None

    # ── Public API ────────────────────────────────────────────────────────

    def reset(self, rng: np.random.Generator) -> np.ndarray:
        """Tear down any existing space and create a fresh simulation.

        Applies small random perturbations to initial pole angles so that
        every episode starts slightly differently.

        Args:
            rng: NumPy random generator for reproducible perturbations.

        Returns:
            Initial state vector of shape ``(6,)`` as float32.
        """
        # Small random perturbations
        cart_x_noise = rng.uniform(-0.05, 0.05)  # metres
        theta1_noise = rng.uniform(-0.05, 0.05)  # radians (~3°)
        theta2_noise = rng.uniform(-0.05, 0.05)

        self._build_space(cart_x_noise, theta1_noise, theta2_noise)

        return self.get_state()

    def apply_force(self, force: float) -> None:
        """Apply a horizontal force to the cart body.

        Args:
            force: Force in Newtons (positive = rightward).
                   Will be scaled to Pymunk units internally.
        """
        # Convert N → Pymunk force units (1 m = scale px)
        pymunk_force = force * self.scale
        self.cart_body.apply_force_at_local_point((pymunk_force, 0), (0, 0))

    def step(self) -> None:
        """Advance the Pymunk simulation by one Gym timestep.

        Internally performs ``self.substeps`` iterations at the finer
        physics dt for joint stability.
        """
        for _ in range(self.substeps):
            self.space.step(self.pymunk_dt)

    def get_state(self) -> np.ndarray:
        """Read the current state from Pymunk bodies.

        Returns:
            State vector ``[x, vx, θ₁, ω₁, θ₂, ω₂]`` as float32.
        """
        s = self.scale

        x = self.cart_body.position.x / s
        vx = self.cart_body.velocity.x / s

        theta1 = self._normalise_angle(self.pole1_body.angle)
        omega1 = self.pole1_body.angular_velocity

        theta2 = self._normalise_angle(self.pole2_body.angle)
        omega2 = self.pole2_body.angular_velocity

        return np.array([x, vx, theta1, omega1, theta2, omega2], dtype=np.float32)

    def get_pole_endpoints(self) -> Dict[str, Tuple[float, float]]:
        """Return world-space positions (in pixels) of key points for rendering.

        Returns:
            Dictionary with keys ``cart``, ``joint1``, ``joint2``, ``pole2_tip``.
            Each value is an ``(x, y)`` tuple in Pymunk pixel coordinates.
        """
        cart_pos = tuple(self.cart_body.position)
        s = self.scale

        # Joint 1: top of cart
        j1 = (cart_pos[0], cart_pos[1] + (self.cart_height / 2) * s)

        # Joint 2: top of pole 1
        # Pole 1 bottom is at j1; top is at j1 + pole1_length in pole direction
        p1_angle = self.pole1_body.angle
        j2_x = j1[0] + self.pole1_length * s * math.sin(p1_angle)
        j2_y = j1[1] + self.pole1_length * s * math.cos(p1_angle)
        j2 = (j2_x, j2_y)

        # Pole 2 tip: top of pole 2
        p2_angle = self.pole2_body.angle
        tip_x = j2[0] + self.pole2_length * s * math.sin(p2_angle)
        tip_y = j2[1] + self.pole2_length * s * math.cos(p2_angle)
        tip = (tip_x, tip_y)

        return {
            "cart": cart_pos,
            "joint1": j1,
            "joint2": j2,
            "pole2_tip": tip,
        }

    # ── Private helpers ───────────────────────────────────────────────────

    def _build_space(
        self, cart_x: float, theta1: float, theta2: float
    ) -> None:
        """Create the Pymunk space and all bodies/shapes/joints from scratch.

        Args:
            cart_x:  Initial cart offset in metres.
            theta1:  Initial angle of pole 1 in radians (from vertical).
            theta2:  Initial angle of pole 2 in radians (from vertical).
        """
        s = self.scale

        # ── Space ─────────────────────────────────────────────────────────
        space = pymunk.Space()
        space.gravity = (0, -self.gravity * s)  # downward
        space.iterations = 30  # higher for joint stability

        # ── Rail (static ground) ──────────────────────────────────────────
        rail_y = 0.0  # rail sits at y = 0 in world coords
        rail_body = space.static_body

        # ── Cart ──────────────────────────────────────────────────────────
        cart_body, cart_shape = self._create_cart(cart_x * s, rail_y, s)
        space.add(cart_body, cart_shape)

        # Groove joint: constrains cart to horizontal motion along the rail
        groove_a = (-self.rail_limit * s, rail_y)
        groove_b = (+self.rail_limit * s, rail_y)
        groove = pymunk.GrooveJoint(
            rail_body, cart_body, groove_a, groove_b, (0, 0)
        )
        space.add(groove)

        # ── Pole 1 (lower) ───────────────────────────────────────────────
        # Pivot point: top-centre of cart
        pivot1_world = (cart_body.position.x, rail_y + (self.cart_height / 2) * s)
        pole1_body, pole1_shape = self._create_pole(
            self.pole1_mass,
            self.pole1_length,
            theta1,
            pivot1_world,
            s,
        )
        space.add(pole1_body, pole1_shape)

        joint1 = pymunk.PivotJoint(cart_body, pole1_body, pivot1_world)
        space.add(joint1)

        # Optional damping on joint 1
        if self.joint_damping > 0:
            spring1 = pymunk.DampedRotarySpring(
                cart_body, pole1_body, 0, 0, self.joint_damping
            )
            space.add(spring1)

        # ── Pole 2 (upper) ───────────────────────────────────────────────
        # Pivot point: top end of pole 1
        p1_top_x = pivot1_world[0] + self.pole1_length * s * math.sin(theta1)
        p1_top_y = pivot1_world[1] + self.pole1_length * s * math.cos(theta1)
        pivot2_world = (p1_top_x, p1_top_y)

        pole2_body, pole2_shape = self._create_pole(
            self.pole2_mass,
            self.pole2_length,
            theta2,
            pivot2_world,
            s,
        )
        space.add(pole2_body, pole2_shape)

        joint2 = pymunk.PivotJoint(pole1_body, pole2_body, pivot2_world)
        space.add(joint2)

        # Optional damping on joint 2
        if self.joint_damping > 0:
            spring2 = pymunk.DampedRotarySpring(
                pole1_body, pole2_body, 0, 0, self.joint_damping
            )
            space.add(spring2)

        # ── Store references ──────────────────────────────────────────────
        self.space = space
        self.cart_body = cart_body
        self.pole1_body = pole1_body
        self.pole2_body = pole2_body

    def _create_cart(
        self, x: float, rail_y: float, s: float
    ) -> Tuple[pymunk.Body, pymunk.Shape]:
        """Create the cart body and shape.

        Args:
            x: Horizontal position in Pymunk pixels.
            rail_y: Vertical position of the rail in Pymunk pixels.
            s: Scale factor (pixels per metre).

        Returns:
            Tuple of (cart_body, cart_shape).
        """
        mass = self.cart_mass
        w = self.cart_width * s
        h = self.cart_height * s

        # Infinite moment → cart cannot rotate
        cart_body = pymunk.Body(mass, float('inf'))
        cart_body.position = (x, rail_y)

        # Rectangular shape centred on body
        half_w, half_h = w / 2, h / 2
        vertices = [
            (-half_w, -half_h),
            (+half_w, -half_h),
            (+half_w, +half_h),
            (-half_w, +half_h),
        ]
        cart_shape = pymunk.Poly(cart_body, vertices)
        cart_shape.friction = 0.0
        # Use a collision category that doesn't collide with poles
        cart_shape.filter = pymunk.ShapeFilter(group=1)

        return cart_body, cart_shape

    def _create_pole(
        self,
        mass: float,
        length: float,
        angle: float,
        pivot_world: Tuple[float, float],
        s: float,
    ) -> Tuple[pymunk.Body, pymunk.Shape]:
        """Create a pole body and segment shape at a given angle.

        The pole's local origin is at its **bottom** (pivot end). The segment
        extends from ``(0, 0)`` to ``(0, length * s)`` in local coordinates.
        The body is positioned so that its local origin aligns with
        ``pivot_world`` after rotation.

        Args:
            mass: Pole mass in kg.
            length: Pole length in metres.
            angle: Initial angle from vertical in radians.
            pivot_world: World-space position of the pivot point (px).
            s: Scale factor (pixels per metre).

        Returns:
            Tuple of (pole_body, pole_shape).
        """
        L = length * s  # length in pixels

        # Moment of inertia for a thin rod about one end
        moment = pymunk.moment_for_segment(mass, (0, 0), (0, L), 1)

        pole_body = pymunk.Body(mass, moment)
        pole_body.angle = angle

        # Position body so that local (0,0) — the pivot end — maps to pivot_world.
        # After rotation by `angle`, the centre of mass (at L/2 along the rod)
        # shifts. We need the pivot end (local origin) at pivot_world.
        # Pymunk's body.position is the position of the body's origin in world space.
        # If the body's local origin is the pivot end, then:
        #   body.position = pivot_world
        # But Pymunk positions the body at its centre of mass by default for
        # moment_for_segment. We use a segment from (0,0) to (0,L) and the
        # CoG will be at (0, L/2) in local coords.
        # So body.position should be such that local(0,0) = pivot_world.
        # body.position = pivot_world + body.local_to_world((0,0)) offset
        # Actually, body.position IS the origin in world space; the segment
        # geometry is relative to this origin. So we just set:
        pole_body.position = pivot_world

        pole_shape = pymunk.Segment(pole_body, (0, 0), (0, L), radius=2)
        pole_shape.friction = 0.0
        pole_shape.filter = pymunk.ShapeFilter(group=1)  # no self-collision

        return pole_body, pole_shape

    @staticmethod
    def _normalise_angle(angle: float) -> float:
        """Wrap an angle to the range ``[-π, π]``.

        Args:
            angle: Angle in radians.

        Returns:
            Equivalent angle in ``[-π, π]``.
        """
        return math.atan2(math.sin(angle), math.cos(angle))
