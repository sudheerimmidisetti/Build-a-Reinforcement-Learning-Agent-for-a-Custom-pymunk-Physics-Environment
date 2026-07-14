"""Compare baseline vs shaped reward training runs and plot results."""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── style ─────────────────────────────────────────────────────────────────

plt.style.use("dark_background")
COLORS = {"baseline": "#e94560", "shaped": "#4a90d9"}
GRID_COLOR = "#333355"


# ── data loading ──────────────────────────────────────────────────────────

def load_evaluations(log_dir: str) -> pd.DataFrame | None:
    """Load EvalCallback npz file (timesteps, mean reward, std reward)."""
    path = os.path.join(log_dir, "evaluations.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    timesteps = data["timesteps"]
    results = data["results"]
    return pd.DataFrame({
        "timesteps": timesteps,
        "mean_reward": results.mean(axis=1),
        "std_reward": results.std(axis=1),
    })


def load_monitor_csvs(log_dir: str) -> pd.DataFrame | None:
    """Load SB3 Monitor CSVs and compute rolling mean reward vs timestep."""
    pattern = os.path.join(log_dir, "**", "monitor.csv")
    files = glob.glob(pattern, recursive=True)
    if not files:
        pattern = os.path.join(log_dir, "**", "*.monitor.csv")
        files = glob.glob(pattern, recursive=True)
    if not files:
        return None

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, skiprows=1)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("t").reset_index(drop=True)
    combined["timesteps"] = combined["l"].cumsum()
    combined["mean_reward"] = combined["r"].rolling(window=20, min_periods=1).mean()
    combined["std_reward"] = combined["r"].rolling(window=20, min_periods=1).std().fillna(0)
    return combined[["timesteps", "mean_reward", "std_reward"]]


def load_progress_csv(log_dir: str) -> pd.DataFrame | None:
    """Load SB3 logger progress.csv (fallback when no eval/monitor data)."""
    path = os.path.join(log_dir, "progress.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    ts_col = next((c for c in df.columns if "timesteps" in c.lower()), None)
    rew_col = next((c for c in df.columns if "reward" in c.lower() or "ep_rew" in c.lower()), None)
    if ts_col is None:
        return None
    if rew_col is None:
        return None
    result = pd.DataFrame({
        "timesteps": df[ts_col],
        "mean_reward": df[rew_col],
        "std_reward": 0.0,
    })
    return result.dropna()


def load_run(log_dir: str) -> pd.DataFrame | None:
    """Try each data source in priority order."""
    for loader in [load_evaluations, load_monitor_csvs, load_progress_csv]:
        df = loader(log_dir)
        if df is not None and len(df) > 0:
            return df
    return None


# ── plotting ──────────────────────────────────────────────────────────────

def plot_comparison(
    baseline_df: pd.DataFrame | None,
    shaped_df: pd.DataFrame | None,
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    for df, label, color in [
        (baseline_df, "Baseline Reward", COLORS["baseline"]),
        (shaped_df, "Shaped Reward", COLORS["shaped"]),
    ]:
        if df is None:
            continue
        ts = df["timesteps"].values
        mean = df["mean_reward"].values
        std = df["std_reward"].values

        ax.plot(ts, mean, color=color, linewidth=2, label=label)
        ax.fill_between(ts, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_title("Mean Reward vs Timesteps", fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("Timesteps", fontsize=13)
    ax.set_ylabel("Mean Reward", fontsize=13)
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(True, alpha=0.3, color=GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output_path}")


def plot_single(df: pd.DataFrame, label: str, color: str, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    ts = df["timesteps"].values
    mean = df["mean_reward"].values
    std = df["std_reward"].values

    ax.plot(ts, mean, color=color, linewidth=2, label=label)
    ax.fill_between(ts, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_title("Mean Reward vs Timesteps", fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("Timesteps", fontsize=13)
    ax.set_ylabel("Mean Reward", fontsize=13)
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(True, alpha=0.3, color=GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output_path}")


# ── main ──────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    baseline_df = None
    shaped_df = None

    if args.baseline_log:
        baseline_df = load_run(args.baseline_log)
        if baseline_df is not None:
            print(f"Loaded baseline data: {len(baseline_df)} points from {args.baseline_log}")
        else:
            print(f"Warning: no plottable data found in {args.baseline_log}")

    if args.shaped_log:
        shaped_df = load_run(args.shaped_log)
        if shaped_df is not None:
            print(f"Loaded shaped data:   {len(shaped_df)} points from {args.shaped_log}")
        else:
            print(f"Warning: no plottable data found in {args.shaped_log}")

    if baseline_df is None and shaped_df is None:
        print("Error: no data to plot. Run training first or check log directories.")
        return

    if baseline_df is not None and shaped_df is not None:
        plot_comparison(baseline_df, shaped_df, args.output)
    elif shaped_df is not None:
        plot_single(shaped_df, "Shaped Reward", COLORS["shaped"], args.output)
    else:
        plot_single(baseline_df, "Baseline Reward", COLORS["baseline"], args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot reward comparison: baseline vs shaped",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--baseline_log", type=str, default=None,
        help="Log directory for baseline reward training run",
    )
    parser.add_argument(
        "--shaped_log", type=str, default=None,
        help="Log directory for shaped reward training run",
    )
    parser.add_argument(
        "--output", type=str, default="plots/reward_comparison.png",
        help="Output path for the comparison plot",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
