"""Automated evaluator — checks every contract specification."""
import os
import sys
import traceback

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

PASS = "PASS"
FAIL = "FAIL"
results = []

def check(name, condition, reason=""):
    status = PASS if condition else FAIL
    results.append((name, status, reason))
    mark = "[PASS]" if condition else "[FAIL]"
    print(f"  {mark} {name}" + (f" -- {reason}" if reason and not condition else ""))
    return condition

# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 1: FILE STRUCTURE")
print("=" * 70)

required_files = [
    "envs/environment.py", "envs/__init__.py",
    "train.py", "evaluate.py", "plot_results.py",
    "Dockerfile", "docker-compose.yml",
    "requirements.txt", "README.md",
    ".env.example", ".gitignore", ".dockerignore",
]
for f in required_files:
    check(f"File exists: {f}", os.path.exists(f), f"Missing: {f}")

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2: GYMNASIUM API")
print("=" * 70)

from envs.environment import DoublePendulumEnv

# 2.1 Class exists and inherits from gymnasium.Env
import gymnasium
check("DoublePendulumEnv inherits gymnasium.Env",
      issubclass(DoublePendulumEnv, gymnasium.Env))

# 2.2 Observation space
env = DoublePendulumEnv()
check("observation_space is Box", isinstance(env.observation_space, gymnasium.spaces.Box))
check("observation_space shape == (6,)", env.observation_space.shape == (6,))
check("observation_space dtype == float32", env.observation_space.dtype == np.float32)

# 2.3 Action space
check("action_space is Box", isinstance(env.action_space, gymnasium.spaces.Box))
check("action_space shape == (1,)", env.action_space.shape == (1,))
check("action_space dtype == float32", env.action_space.dtype == np.float32)
check("action_space low == -1", float(env.action_space.low[0]) == -1.0)
check("action_space high == 1", float(env.action_space.high[0]) == 1.0)

# 2.4 reset() returns (obs, info)
obs, info = env.reset(seed=42)
check("reset() returns tuple of length 2", isinstance(obs, np.ndarray) and isinstance(info, dict))
check("reset() obs shape == (6,)", obs.shape == (6,))
check("reset() obs dtype == float32", obs.dtype == np.float32)
check("reset() obs within bounds", env.observation_space.contains(obs))

# 2.5 step() returns 5-tuple
action = np.array([0.0], dtype=np.float32)
result = env.step(action)
check("step() returns tuple of length 5", len(result) == 5)
obs2, reward, terminated, truncated, info2 = result
check("step() obs shape == (6,)", obs2.shape == (6,))
check("step() reward is float", isinstance(reward, (float, np.floating)))
check("step() terminated is bool", isinstance(terminated, (bool, np.bool_)))
check("step() truncated is bool", isinstance(truncated, (bool, np.bool_)))
check("step() info is dict", isinstance(info2, dict))

# 2.6 render() exists
check("render() method exists", hasattr(env, 'render'))
check("close() method exists", hasattr(env, 'close'))
env.close()

# 2.7 metadata
check("metadata has render_modes", "render_modes" in env.metadata)
check("metadata has render_fps", "render_fps" in env.metadata)

# 2.8 gymnasium.make works
import envs
gym_env = gymnasium.make("DoublePendulum-v0")
obs3, _ = gym_env.reset()
check("gymnasium.make('DoublePendulum-v0') works", obs3.shape == (6,))
gym_env.close()

# 2.9 SB3 check_env
from stable_baselines3.common.env_checker import check_env
try:
    test_env = DoublePendulumEnv()
    check_env(test_env, warn=True)
    test_env.close()
    check("SB3 check_env() passes", True)
except Exception as e:
    check("SB3 check_env() passes", False, str(e))

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3: REWARD FUNCTIONS")
print("=" * 70)

# 3.1 Baseline
env_b = DoublePendulumEnv(reward_type="baseline")
env_b.reset(seed=42)
_, r_b, _, _, _ = env_b.step(np.array([0.0]))
check("reward_type='baseline' returns 1.0", r_b == 1.0)
env_b.close()

# 3.2 Shaped
env_s = DoublePendulumEnv(reward_type="shaped")
env_s.reset(seed=42)
_, r_s, _, _, _ = env_s.step(np.array([0.0]))
check("reward_type='shaped' returns float != 1.0", isinstance(r_s, float) and r_s != 1.0)
check("shaped reward < 1.0 (penalties applied)", r_s < 1.0)
env_s.close()

# 3.3 Invalid reward_type
try:
    DoublePendulumEnv(reward_type="invalid")
    check("Invalid reward_type raises ValueError", False, "No exception raised")
except ValueError:
    check("Invalid reward_type raises ValueError", True)

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4: PHYSICS (PYMUNK)")
print("=" * 70)

env = DoublePendulumEnv()
env.reset(seed=42)
check("pymunk.Space created", env.space is not None)
check("space.gravity is set", env.space.gravity[1] < 0)
check("cart_body exists", env.cart_body is not None)
check("pole1_body exists", env.pole1_body is not None)
check("pole2_body exists", env.pole2_body is not None)

# Stability: 500 random steps
stable = True
for _ in range(500):
    obs, _, term, trunc, _ = env.step(env.action_space.sample())
    if not np.all(np.isfinite(obs)):
        stable = False
        break
    if term or trunc:
        env.reset()
check("500 random steps: no NaN/Inf", stable)
env.close()

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5: EPISODE TERMINATION")
print("=" * 70)

env = DoublePendulumEnv()
env.reset(seed=42)
terminated = False
steps = 0
while not terminated and steps < 500:
    _, _, terminated, _, _ = env.step(np.array([0.0]))
    steps += 1
check("Episode terminates (poles fall with zero force)", terminated)
check("Termination before 500 steps", steps < 500)

# Truncation
env.reset(seed=42)
env._step_count = env.max_steps - 1
_, _, _, truncated, _ = env.step(np.array([0.0]))
check("Truncation at max_steps", truncated)
env.close()

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 6: TRAINING PIPELINE")
print("=" * 70)

# 6.1 Imports
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    check("SB3 PPO imports", True)
except ImportError as e:
    check("SB3 PPO imports", False, str(e))

# 6.2 argparse
import train as train_module
check("train.py has parse_args()", hasattr(train_module, 'parse_args'))
check("train.py has train()", hasattr(train_module, 'train'))

# 6.3 Short training run
try:
    def mk():
        return DoublePendulumEnv(render_mode=None, reward_type="shaped")
    vec = DummyVecEnv([mk])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=True)
    model = PPO("MlpPolicy", vec, n_steps=64, batch_size=32, verbose=0)
    model.learn(total_timesteps=128)
    model.save("models/_eval_test")
    check("PPO trains for 128 steps", True)
    check("Model saves to .zip", os.path.exists("models/_eval_test.zip"))
    vec.save("models/_eval_test_norm.pkl")
    check("VecNormalize saves to .pkl", os.path.exists("models/_eval_test_norm.pkl"))
    loaded = PPO.load("models/_eval_test")
    check("Model reloads successfully", loaded is not None)
    vec.close()
except Exception as e:
    check("PPO training smoke test", False, str(e))

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 7: EVALUATION & GIF")
print("=" * 70)

# 7.1 evaluate.py imports and functions
import evaluate as eval_module
check("evaluate.py has parse_args()", hasattr(eval_module, 'parse_args'))
check("evaluate.py has evaluate()", hasattr(eval_module, 'evaluate'))
check("evaluate.py has save_gif()", hasattr(eval_module, 'save_gif'))
check("evaluate.py has capture_episode()", hasattr(eval_module, 'capture_episode'))

# 7.2 GIF generation
try:
    from PIL import Image
    frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(5)]
    eval_module.save_gif(frames, "media/_eval_test.gif", fps=10)
    check("GIF generation works", os.path.exists("media/_eval_test.gif"))
    os.remove("media/_eval_test.gif")
except Exception as e:
    check("GIF generation works", False, str(e))

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 8: PLOTTING")
print("=" * 70)

import plot_results as plot_module
check("plot_results.py has parse_args()", hasattr(plot_module, 'parse_args'))
check("plot_results.py has run()", hasattr(plot_module, 'run'))
check("plot_results.py has load_evaluations()", hasattr(plot_module, 'load_evaluations'))
check("plot_results.py has plot_comparison()", hasattr(plot_module, 'plot_comparison'))

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 9: LOGGING")
print("=" * 70)

check("CSV log exists (logs/progress.csv)", os.path.exists("logs/progress.csv"))
check("TensorBoard dir exists", os.path.exists("logs/tensorboard") or any("tfevents" in f for f in os.listdir("logs")))
check("Evaluations NPZ exists", os.path.exists("logs/baseline/evaluations.npz") or os.path.exists("logs/shaped/evaluations.npz"))

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 10: DOCKER")
print("=" * 70)

with open("Dockerfile") as f:
    dockerfile = f.read()
check("Dockerfile uses python:3.11-slim", "python:3.11-slim" in dockerfile)
check("Dockerfile has WORKDIR /app", "WORKDIR /app" in dockerfile)
check("Dockerfile copies requirements.txt first", "COPY requirements.txt" in dockerfile)
check("Dockerfile installs deps", "pip install" in dockerfile)
check("Dockerfile copies project", "COPY . ." in dockerfile)
check("Dockerfile has ENTRYPOINT", "ENTRYPOINT" in dockerfile)
check("Dockerfile has non-root USER", "USER agent" in dockerfile)

with open("docker-compose.yml") as f:
    compose = f.read()
check("Compose has train service", "train:" in compose)
check("Compose has evaluate service", "evaluate:" in compose)
check("Compose has plot service", "plot:" in compose)
check("Compose mounts volumes", "volumes:" in compose)
check("Compose has SDL env vars", "SDL_VIDEODRIVER" in compose)

check(".dockerignore exists", os.path.exists(".dockerignore"))

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 11: DOCUMENTATION")
print("=" * 70)

with open("README.md", encoding="utf-8") as f:
    readme = f.read().lower()
sections = ["overview", "features", "environment design", "observation space",
            "action space", "reward", "training", "evaluation", "docker",
            "repository structure", "results", "future"]
for s in sections:
    check(f"README has '{s}' section", s in readme, f"Missing section: {s}")

check("README has pip install command", "pip install" in readme)
check("README has docker compose command", "docker compose" in readme)
check("README has train.py command", "train.py" in readme)
check("README has evaluate.py command", "evaluate.py" in readme)

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 12: .env.example")
print("=" * 70)

with open(".env.example") as f:
    envex = f.read()
for var in ["MODEL_PATH", "TRAIN_TIMESTEPS", "LOG_DIR", "REWARD_TYPE", "FPS", "GIF_OUTPUT"]:
    check(f".env.example has {var}", var in envex, f"Missing: {var}")

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 13: REQUIREMENTS")
print("=" * 70)

with open("requirements.txt") as f:
    reqs = f.read().lower()
for pkg in ["gymnasium", "stable-baselines3", "torch", "pymunk", "pygame", "matplotlib", "numpy", "pillow"]:
    check(f"requirements.txt has {pkg}", pkg in reqs)

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 14: SEEDED REPRODUCIBILITY")
print("=" * 70)

env1 = DoublePendulumEnv()
env2 = DoublePendulumEnv()
o1, _ = env1.reset(seed=999)
o2, _ = env2.reset(seed=999)
check("Seeded reproducibility", np.array_equal(o1, o2))
env1.close(); env2.close()

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 15: RENDER (rgb_array)")
print("=" * 70)

env = DoublePendulumEnv(render_mode="rgb_array")
env.reset(seed=42)
env.step(np.array([0.0]))
frame = env.render()
check("render() returns ndarray", isinstance(frame, np.ndarray))
check("render() frame has 3 dims (H,W,C)", frame.ndim == 3)
check("render() frame channel == 3 (RGB)", frame.shape[2] == 3)
check("render() frame dtype == uint8", frame.dtype == np.uint8)
env.close()

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)

passed = sum(1 for _, s, _ in results if s == PASS)
failed = sum(1 for _, s, _ in results if s == FAIL)
total = len(results)

if failed > 0:
    print(f"\nFAILED CHECKS ({failed}):")
    for name, status, reason in results:
        if status == FAIL:
            print(f"  [FAIL] {name}: {reason}")

print(f"\n  Total checks:  {total}")
print(f"  Passed:        {passed}")
print(f"  Failed:        {failed}")
print(f"  Score:         {passed}/{total} ({100*passed//total}%)")
print("=" * 70)

# Cleanup
for f in ["models/_eval_test.zip", "models/_eval_test_norm.pkl"]:
    if os.path.exists(f):
        os.remove(f)
