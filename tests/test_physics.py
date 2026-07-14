"""Tests for the Pymunk physics engine (Phase 2 gate)."""

import math

import numpy as np
import pytest

from utils.helpers import load_config, get_project_root

# Load the default env config
CONFIG_PATH = str(get_project_root() / "configs" / "env_config.yaml")


@pytest.fixture
def config():
    return load_config(CONFIG_PATH)


@pytest.fixture
def engine(config):
    from envs.physics_engine import PhysicsEngine

    return PhysicsEngine(config)


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


class TestPhysicsEngineInit:
    """Test that the engine initialises with correct parameters."""

    def test_creates_engine(self, engine):
        assert engine is not None

    def test_gravity_loaded(self, engine):
        assert engine.gravity == pytest.approx(9.81)

    def test_cart_mass(self, engine):
        assert engine.cart_mass == pytest.approx(1.0)

    def test_pole_lengths(self, engine):
        assert engine.pole1_length == pytest.approx(0.5)
        assert engine.pole2_length == pytest.approx(0.5)

    def test_pole_masses(self, engine):
        assert engine.pole1_mass == pytest.approx(0.5)
        assert engine.pole2_mass == pytest.approx(0.5)


class TestPhysicsReset:
    """Test that reset creates a valid initial state."""

    def test_reset_returns_6d_state(self, engine, rng):
        state = engine.reset(rng)
        assert state.shape == (6,)

    def test_reset_dtype_is_float32(self, engine, rng):
        state = engine.reset(rng)
        assert state.dtype == np.float32

    def test_cart_near_zero_after_reset(self, engine, rng):
        state = engine.reset(rng)
        assert abs(state[0]) < 0.1  # cart x within noise range

    def test_angles_near_zero_after_reset(self, engine, rng):
        state = engine.reset(rng)
        assert abs(state[2]) < 0.1  # θ₁ within noise range
        assert abs(state[4]) < 0.1  # θ₂ within noise range

    def test_reset_creates_pymunk_space(self, engine, rng):
        engine.reset(rng)
        assert engine.space is not None
        assert engine.cart_body is not None
        assert engine.pole1_body is not None
        assert engine.pole2_body is not None

    def test_reset_is_deterministic_with_same_seed(self, engine):
        rng1 = np.random.default_rng(seed=123)
        state1 = engine.reset(rng1)

        rng2 = np.random.default_rng(seed=123)
        state2 = engine.reset(rng2)

        np.testing.assert_array_almost_equal(state1, state2)

    def test_multiple_resets_do_not_leak(self, engine, rng):
        """Calling reset multiple times should not accumulate bodies."""
        for _ in range(10):
            engine.reset(rng)
        # Should still work correctly
        state = engine.get_state()
        assert state.shape == (6,)
        assert np.all(np.isfinite(state))


class TestPhysicsStep:
    """Test that stepping the simulation produces correct dynamics."""

    def test_poles_fall_under_gravity(self, engine, rng):
        """With no force, poles should fall — angles should increase."""
        state0 = engine.reset(rng)
        # Apply no force and step many times
        for _ in range(100):
            engine.step()
        state1 = engine.get_state()
        # At least one angle should have increased significantly
        angle_change = abs(state1[2]) + abs(state1[4])
        assert angle_change > 0.1, (
            f"Poles did not fall: angle change = {angle_change}"
        )

    def test_positive_force_moves_cart_right(self, engine, rng):
        engine.reset(rng)
        initial_x = engine.get_state()[0]
        # Apply rightward force for several steps
        for _ in range(50):
            engine.apply_force(5.0)
            engine.step()
        final_x = engine.get_state()[0]
        assert final_x > initial_x, "Positive force should move cart right"

    def test_negative_force_moves_cart_left(self, engine, rng):
        engine.reset(rng)
        initial_x = engine.get_state()[0]
        for _ in range(50):
            engine.apply_force(-5.0)
            engine.step()
        final_x = engine.get_state()[0]
        assert final_x < initial_x, "Negative force should move cart left"

    def test_state_is_finite(self, engine, rng):
        """State should never contain NaN or Inf."""
        engine.reset(rng)
        for _ in range(200):
            engine.apply_force(rng.uniform(-10, 10))
            engine.step()
            state = engine.get_state()
            assert np.all(np.isfinite(state)), f"Non-finite state: {state}"

    def test_substeps_count(self, engine, rng):
        """Verify that substeps parameter is respected."""
        engine.reset(rng)
        assert engine.substeps == 5
        assert engine.pymunk_dt == pytest.approx(engine.dt / engine.substeps)


class TestPoleEndpoints:
    """Test the endpoint extraction used for rendering."""

    def test_returns_all_keys(self, engine, rng):
        engine.reset(rng)
        endpoints = engine.get_pole_endpoints()
        assert "cart" in endpoints
        assert "joint1" in endpoints
        assert "joint2" in endpoints
        assert "pole2_tip" in endpoints

    def test_endpoints_are_tuples(self, engine, rng):
        engine.reset(rng)
        endpoints = engine.get_pole_endpoints()
        for key, val in endpoints.items():
            assert isinstance(val, tuple), f"{key} is not a tuple"
            assert len(val) == 2, f"{key} should have 2 elements"


class TestAngleNormalisation:
    """Test the static angle normalisation helper."""

    def test_zero(self):
        from envs.physics_engine import PhysicsEngine

        assert PhysicsEngine._normalise_angle(0) == pytest.approx(0)

    def test_pi(self):
        from envs.physics_engine import PhysicsEngine

        result = PhysicsEngine._normalise_angle(math.pi)
        assert abs(result) == pytest.approx(math.pi)

    def test_large_positive(self):
        from envs.physics_engine import PhysicsEngine

        result = PhysicsEngine._normalise_angle(3 * math.pi)
        assert -math.pi <= result <= math.pi

    def test_large_negative(self):
        from envs.physics_engine import PhysicsEngine

        result = PhysicsEngine._normalise_angle(-5 * math.pi)
        assert -math.pi <= result <= math.pi
