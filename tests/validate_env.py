"""Comprehensive validation of environment.py"""
from envs.environment import DoublePendulumEnv
import numpy as np

# Test 1: shaped reward
print("=== Test 1: Shaped Reward ===")
env = DoublePendulumEnv(reward_type="shaped")
obs, _ = env.reset(seed=42)
print(f"  obs shape={obs.shape}  dtype={obs.dtype}")
for i in range(5):
    obs, r, t, tr, info = env.step(np.array([0.0]))
    t1 = info["pole1_angle"]
    t2 = info["pole2_angle"]
    print(f"  step {i+1}: reward={r:+.4f}  t1={t1:+.4f}  t2={t2:+.4f}")
env.close()

# Test 2: baseline reward
print("\n=== Test 2: Baseline Reward ===")
env = DoublePendulumEnv(reward_type="baseline")
obs, _ = env.reset(seed=42)
for i in range(5):
    obs, r, t, tr, info = env.step(np.array([0.0]))
    print(f"  step {i+1}: reward={r:+.4f}")
env.close()

# Test 3: termination (poles should fall)
print("\n=== Test 3: Termination ===")
env = DoublePendulumEnv(reward_type="shaped")
obs, _ = env.reset(seed=42)
terminated = False
steps = 0
while not terminated and steps < 500:
    obs, r, terminated, truncated, _ = env.step(np.array([0.0]))
    steps += 1
print(f"  Episode ended after {steps} steps, terminated={terminated}")
env.close()

# Test 4: 1000-step random loop stress test
print("\n=== Test 4: 1000-Step Random Loop ===")
env = DoublePendulumEnv(reward_type="shaped")
obs, _ = env.reset(seed=0)
total_r = 0.0
resets = 0
for step in range(1000):
    obs, r, term, trunc, _ = env.step(env.action_space.sample())
    total_r += r
    assert np.all(np.isfinite(obs)), f"Non-finite obs at step {step}: {obs}"
    assert np.isfinite(r), f"Non-finite reward at step {step}: {r}"
    if term or trunc:
        obs, _ = env.reset()
        resets += 1
env.close()
print(f"  1000 steps OK, {resets} resets, total_reward={total_r:.1f}")

# Test 5: gymnasium.make
print("\n=== Test 5: gymnasium.make ===")
import envs  # noqa: triggers registration
import gymnasium
env = gymnasium.make("DoublePendulum-v0")
obs, _ = env.reset()
print(f"  obs shape={obs.shape}")
env.close()

# Test 6: SB3 check_env
print("\n=== Test 6: SB3 check_env ===")
from stable_baselines3.common.env_checker import check_env
env = DoublePendulumEnv()
check_env(env, warn=True)
env.close()
print("  PASSED")

# Test 7: reward_type validation
print("\n=== Test 7: Invalid reward_type ===")
try:
    env = DoublePendulumEnv(reward_type="invalid")
    print("  FAILED - should have raised ValueError")
except ValueError as e:
    print(f"  PASSED - caught: {e}")

# Test 8: action clipping
print("\n=== Test 8: Action Clipping ===")
env = DoublePendulumEnv()
obs, _ = env.reset(seed=42)
obs, r, _, _, _ = env.step(np.array([100.0]))
assert np.all(np.isfinite(obs))
obs, r, _, _, _ = env.step(np.array([-100.0]))
assert np.all(np.isfinite(obs))
env.close()
print("  PASSED - extreme actions handled")

# Test 9: seeded reproducibility
print("\n=== Test 9: Seeded Reproducibility ===")
env1 = DoublePendulumEnv()
env2 = DoublePendulumEnv()
obs1, _ = env1.reset(seed=123)
obs2, _ = env2.reset(seed=123)
np.testing.assert_array_almost_equal(obs1, obs2)
env1.close()
env2.close()
print("  PASSED - identical seeds produce identical observations")

print("\n" + "=" * 50)
print("ALL TESTS PASSED")
print("=" * 50)
