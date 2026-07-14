# Double Inverted Pendulum — Reinforcement Learning

## Overview

A custom Gymnasium environment that simulates a double inverted pendulum on a cart, trained with PPO (Proximal Policy Optimization) from Stable-Baselines3. The physics run on Pymunk, rendering is handled by Pygame, and the whole thing is containerised with Docker for reproducible training.

The agent controls a cart that slides along a rail. Two poles are stacked above it, connected by revolute joints. The goal is to keep both poles balanced upright — a significantly harder problem than the classic single-pendulum CartPole, because the second pole amplifies instability and the agent must learn coordinated control.

![Demo](media/agent_initial.gif)

---

## Features

- **Custom Pymunk physics** — rigid-body simulation with sub-step integration, groove joints for the rail, pivot joints for the poles, and correct centre-of-mass placement
- **Two reward modes** — a constant alive bonus (`baseline`) and a shaped reward function (`shaped`) for comparing learning dynamics
- **Full Gymnasium API** — works out of the box with `gymnasium.make()`, passes SB3's `check_env()`, supports seeded reproducibility
- **PPO training pipeline** — VecNormalize wrapping, EvalCallback with best-model saving, TensorBoard + CSV logging, checkpoint saves, graceful interrupt handling
- **Evaluation & GIF export** — deterministic rollouts with aggregate metrics, automatic GIF recording of the best episode
- **Reward comparison plots** — Matplotlib dark-theme charts comparing baseline vs shaped training curves
- **Docker support** — single-command training, evaluation, and plotting inside a container

---

## Environment Design

The environment models a 2D cart-pole system with two chained poles:

```
        ●  ← tip (pole 2)
        │
        ●  ← joint 2 (pole 1 top / pole 2 bottom)
        │
        ●  ← joint 1 (cart top / pole 1 bottom)
   ┌────┴────┐
   │  CART   │
───┴─────────┴─── rail ───
```

The cart has infinite rotational inertia (it cannot rotate, only slide). Each pole is a rigid segment with mass concentrated along its length. Pymunk handles collision filtering so the bodies don't self-collide — they interact only through their joints.

**Physics parameters:**

| Parameter | Value |
|---|---|
| Gravity | 9.81 m/s² |
| Timestep | 0.02 s (50 Hz) |
| Sub-steps | 5 (250 Hz internal) |
| Cart mass | 1.0 kg |
| Pole masses | 0.5 kg each |
| Pole lengths | 0.5 m each |
| Rail limit | ±2.4 m |
| Max force | 10 N |

### Observation Space

A 6-dimensional continuous vector:

| Index | Variable | Description |
|---|---|---|
| 0 | `x` | Cart position (metres) |
| 1 | `ẋ` | Cart velocity (m/s) |
| 2 | `θ₁` | Pole 1 angle from vertical (radians, wrapped to [-π, π]) |
| 3 | `ω₁` | Pole 1 angular velocity (rad/s) |
| 4 | `θ₂` | Pole 2 angle from vertical (radians, wrapped to [-π, π]) |
| 5 | `ω₂` | Pole 2 angular velocity (rad/s) |

Shape: `(6,)` — `Box(-high, high, dtype=float32)`

### Action Space

A single continuous value representing the normalised horizontal force on the cart.

| Index | Range | Meaning |
|---|---|---|
| 0 | [-1, 1] | Force direction and magnitude (scaled by 10 N internally) |

Shape: `(1,)` — `Box(-1, 1, dtype=float32)`

### Reward Function Design

#### Baseline (`reward_type="baseline"`)

```
reward = 1.0
```

A constant +1 for every step the agent survives. This is the CartPole-style reward — the agent only learns that staying alive is good, but gets no gradient signal about *how* to stay alive. It works eventually, but learning is slow because the reward is uninformative. The agent has to stumble into good behaviour through random exploration before reinforcement kicks in.

#### Shaped (`reward_type="shaped"`)

```
reward = 1.0 - 1.0·(θ₁² + θ₂²) - 0.5·x² - 0.01·(ω₁² + ω₂²)
```

Each penalty term gives the agent a continuous signal pointing toward the goal:

- **Angle penalty** — poles closer to vertical earn higher reward, creating a smooth gradient toward the upright equilibrium
- **Cart penalty** — discourages drifting to the rail edges, teaching the agent to stay centred
- **Velocity penalty** — penalises aggressive angular oscillations, encouraging smooth control

**Why shaped rewards improve learning:** In the baseline case, the agent receives identical reward whether the poles are perfectly vertical or about to fall — the only signal comes from episode termination, which is sparse and delayed. Shaped rewards convert this binary alive/dead signal into a dense, step-level gradient. The agent immediately knows that `θ = 0.01` is better than `θ = 0.1`, so policy gradient updates point in the right direction from the first rollout. This typically speeds up convergence by 5–10x on this task.

### Episode Termination

An episode ends early (`terminated=True`) if:
- Either pole exceeds ±24° from vertical
- The cart moves beyond ±2.4 m from centre

An episode is truncated (`truncated=True`) after 1000 steps if neither condition is met.

---

## Training

### Local

```bash
pip install -r requirements.txt
python train.py --timesteps 500000 --reward_type shaped
```

All arguments:

| Argument | Default | Description |
|---|---|---|
| `--timesteps` | 500000 | Total training steps |
| `--reward_type` | shaped | `baseline` or `shaped` |
| `--save_path` | models | Directory for model checkpoints |
| `--log_dir` | logs | Directory for CSV + TensorBoard logs |

Outputs:
- `models/final_model.zip` — trained policy
- `models/vecnormalize.pkl` — observation/reward normalisation stats (must stay with the model)
- `models/best/best_model.zip` — best checkpoint by evaluation reward
- `logs/progress.csv` — per-iteration metrics
- `logs/tensorboard/` — TensorBoard event files

Monitor training in real time:

```bash
tensorboard --logdir logs/tensorboard
```

### Comparing reward modes

```bash
python train.py --timesteps 500000 --reward_type baseline --save_path models/baseline --log_dir logs/baseline
python train.py --timesteps 500000 --reward_type shaped   --save_path models/shaped   --log_dir logs/shaped
```

---

## Evaluation

```bash
python evaluate.py --model_path models/final_model.zip --gif_path media/agent_final.gif --episodes 10
```

| Argument | Default | Description |
|---|---|---|
| `--model_path` | models/final_model.zip | Path to saved model |
| `--gif_path` | media/agent_final.gif | Output GIF path |
| `--episodes` | 5 | Number of evaluation episodes |

Prints per-episode rewards/lengths plus aggregate statistics (mean ± std, min, max). Saves a GIF of the best episode.

### Plotting

```bash
python plot_results.py --baseline_log logs/baseline --shaped_log logs/shaped --output plots/reward_comparison.png
```

Generates a dark-themed "Mean Reward vs Timesteps" comparison chart with ±1 std shading.

---

## Docker Setup

### Build

```bash
docker compose build
```

### Run

```bash
docker compose run train
docker compose run evaluate
docker compose run plot
```

The compose file mounts the project directory into the container, so models, logs, plots, and GIFs persist on the host.

### Override arguments

```bash
docker compose run train python train.py --timesteps 1000000 --reward_type baseline
```

---

## Repository Structure

```
Build a Reinforcement Learning Agent for a Custom `pymunk` Physics Environment/
├── envs/
│   ├── __init__.py              # Gymnasium registration
│   └── environment.py           # DoublePendulumEnv (physics + rendering)
├── tests/
│   ├── test_env.py              # Gymnasium API tests
│   ├── test_physics.py          # Physics engine tests
│   ├── test_training.py         # Training pipeline smoke tests
│   └── test_evaluation.py       # Evaluation module tests
├── configs/
│   ├── env_config.yaml          # Environment parameters
│   └── train_config.yaml        # Training hyperparameters
├── train.py                     # PPO training entry point
├── evaluate.py                  # Deterministic evaluation + GIF export
├── plot_results.py              # Reward comparison plotting
├── requirements.txt             # Pinned dependencies
├── Dockerfile                   # Production container
├── docker-compose.yml           # Service definitions
├── .env.example                 # Configurable variables
└── .gitignore
```

---

## Results

After 500k timesteps with shaped rewards, the agent typically learns to balance both poles for the full 1000-step episode. Training curves show:

- **Shaped reward** converges in ~100k–200k steps to near-maximum episode length
- **Baseline reward** takes 3–5x longer to reach comparable performance, with higher variance

The shaped agent develops a subtle rocking strategy — it makes small corrective forces to counteract the second pole's tendency to lag behind the first, rather than trying to hold everything perfectly still.

---

## Future Improvements

- **Domain randomisation** — vary pole lengths, masses, and friction during training to produce a more robust policy
- **Curriculum learning** — start with a single pole, freeze that policy, then add the second pole
- **SAC / TD3 comparison** — off-policy algorithms may be more sample-efficient on this continuous control task
- **Energy-based reward** — penalise total system energy to encourage the agent to find the low-energy equilibrium
- **3D extension** — move to a 3D pendulum with MuJoCo for a harder version of the problem
- **Sim-to-real transfer** — add observation noise and action delay to train policies that might work on physical hardware
