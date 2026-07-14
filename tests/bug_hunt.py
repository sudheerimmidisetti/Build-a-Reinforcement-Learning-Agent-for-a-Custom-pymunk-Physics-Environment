"""Targeted diagnostics to surface hidden bugs in environment.py"""
import math
import numpy as np
from envs.environment import DoublePendulumEnv

print("=== BUG HUNT ===\n")

# 1. Does render() after close() crash?
print("1. render() after close()...")
env = DoublePendulumEnv(render_mode="rgb_array")
env.reset(seed=0)
env.step(np.array([0.0]))
env.render()
env.close()
try:
    env.render()  # screen is None, should re-init or handle gracefully
    print("   OK - no crash")
except Exception as e:
    print(f"   BUG: {type(e).__name__}: {e}")

# 2. Does step() before reset() crash?
print("\n2. step() before reset()...")
env = DoublePendulumEnv()
try:
    env.step(np.array([0.0]))
    print("   BUG - should have crashed (no space created)")
except AttributeError as e:
    print(f"   Expected crash: {e}")
except Exception as e:
    print(f"   Unexpected: {type(e).__name__}: {e}")
env.close()

# 3. Force accumulation between steps - does force persist?
print("\n3. Force persistence check...")
env = DoublePendulumEnv()
env.reset(seed=42)
# Apply force only on step 1, then nothing
env.step(np.array([1.0]))
obs1 = env._get_obs()
env.step(np.array([0.0]))
obs2 = env._get_obs()
env.step(np.array([0.0]))
obs3 = env._get_obs()
# Cart should decelerate after force removed (assuming no friction, it coasts)
# Velocity at step 2 and 3 should be roughly equal (no new force)
vel_diff = abs(obs3[1] - obs2[1])
print(f"   vx after force: {obs1[1]:.4f}")
print(f"   vx step after:  {obs2[1]:.4f}")
print(f"   vx 2 steps after: {obs3[1]:.4f}")
print(f"   Velocity change without force: {vel_diff:.6f}")
# With pendulum dynamics the cart velocity WILL change (pendulum reaction),
# but it should not get the same force impulse again
env.close()

# 4. Angle observation matches actual body angle?
print("\n4. Angle consistency...")
env = DoublePendulumEnv()
env.reset(seed=42)
for _ in range(10):
    env.step(np.array([0.5]))
obs = env._get_obs()
raw_angle1 = env.pole1_body.angle
wrapped = math.atan2(math.sin(raw_angle1), math.cos(raw_angle1))
print(f"   raw body.angle:  {raw_angle1:.6f}")
print(f"   wrapped:         {wrapped:.6f}")
print(f"   obs[2]:          {obs[2]:.6f}")
assert abs(wrapped - obs[2]) < 1e-5, "Angle mismatch!"
print("   OK - angles consistent")
env.close()

# 5. Type annotation on _screen - does it crash when pygame not used?
print("\n5. Type hint crash test (pygame=None scenario)...")
# We can't actually set pygame to None without monkey-patching,
# but check that render_mode=None never touches pygame
env = DoublePendulumEnv(render_mode=None)
env.reset(seed=0)
result = env.render()
assert result is None
env.close()
print("   OK - render_mode=None skips pygame entirely")

# 6. Observation within declared space bounds?
print("\n6. Observation bounds check (100 random episodes)...")
env = DoublePendulumEnv()
violations = 0
for ep in range(100):
    obs, _ = env.reset(seed=ep)
    for _ in range(50):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        if not env.observation_space.contains(obs):
            violations += 1
        if term or trunc:
            break
env.close()
print(f"   {violations} violations out of ~5000 steps")
if violations > 0:
    print("   BUG: observations sometimes exceed declared bounds")
else:
    print("   OK")

# 7. Does step_count increment correctly (off-by-one)?
print("\n7. Step count off-by-one check...")
env = DoublePendulumEnv()
env.reset(seed=42)
assert env._step_count == 0, f"After reset: {env._step_count}"
env.step(np.array([0.0]))
assert env._step_count == 1, f"After 1 step: {env._step_count}"
# Check truncation triggers at exactly max_steps
env._step_count = env.max_steps - 1
_, _, _, truncated, _ = env.step(np.array([0.0]))
assert truncated, "Should truncate at max_steps"
assert env._step_count == env.max_steps
env.close()
print(f"   OK - truncation at step {env.max_steps}")

# 8. Pivot2 computation - does it use local_to_world or manual trig?
print("\n8. Pivot2 computed correctly at init?")
env = DoublePendulumEnv()
env.reset(seed=42)
half_L1 = env.p1_len * env.scale / 2
# The top of pole1 via local_to_world
pole1_top = env.pole1_body.local_to_world((0, half_L1))
# The bottom of pole2 via local_to_world
half_L2 = env.p2_len * env.scale / 2
pole2_bottom = env.pole2_body.local_to_world((0, -half_L2))
dist = math.sqrt((pole1_top.x - pole2_bottom.x)**2 + (pole1_top.y - pole2_bottom.y)**2)
print(f"   pole1 top:    ({pole1_top.x:.2f}, {pole1_top.y:.2f})")
print(f"   pole2 bottom: ({pole2_bottom.x:.2f}, {pole2_bottom.y:.2f})")
print(f"   gap: {dist:.4f} px")
if dist > 1.0:
    print("   WARNING: joint gap > 1px - PivotJoint should close this")
else:
    print("   OK - poles properly connected")
env.close()

# 9. Does the cart actually stay on the rail (GrooveJoint)?
print("\n9. Cart stays on rail (y=0)?")
env = DoublePendulumEnv()
env.reset(seed=42)
for _ in range(200):
    env.step(env.action_space.sample())
cart_y = env.cart_body.position.y
print(f"   Cart y after 200 steps: {cart_y:.6f}")
if abs(cart_y) > 0.1:
    print("   BUG: cart drifting off rail")
else:
    print("   OK - groove joint holding")
env.close()

# 10. Memory: does repeated reset accumulate pymunk objects?
print("\n10. Memory leak check (1000 resets)...")
import sys
env = DoublePendulumEnv()
for i in range(1000):
    env.reset(seed=i)
# After 1000 resets, the space should only have current objects
n_bodies = len(env.space.bodies)
n_shapes = len(env.space.shapes)
n_constraints = len(env.space.constraints)
print(f"   Bodies: {n_bodies}, Shapes: {n_shapes}, Constraints: {n_constraints}")
if n_bodies > 5 or n_shapes > 5 or n_constraints > 5:
    print("   BUG: objects accumulating in space")
else:
    print("   OK - no leaks")
env.close()

print("\n=== DONE ===")
