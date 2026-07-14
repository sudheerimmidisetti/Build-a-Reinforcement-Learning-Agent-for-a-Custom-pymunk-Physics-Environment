"""CLI entry point for the Double Inverted Pendulum RL project.

Usage:
    python main.py train       # Train the RL agent
    python main.py evaluate    # Evaluate a trained agent
    python main.py render      # Render the agent in real-time
    python main.py plot        # Generate training/evaluation plots
"""

import argparse
import sys
from pathlib import Path

from utils.helpers import get_project_root, load_config, set_global_seed
from utils.logger import setup_logger

# Resolve project root so relative paths in configs work regardless of cwd
PROJECT_ROOT = get_project_root()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Double Inverted Pendulum — RL Agent",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── train ─────────────────────────────────────────────────────────────
    train_parser = subparsers.add_parser("train", help="Train the RL agent")
    train_parser.add_argument(
        "--env-config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "env_config.yaml"),
        help="Path to environment config YAML",
    )
    train_parser.add_argument(
        "--train-config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "train_config.yaml"),
        help="Path to training config YAML",
    )
    train_parser.add_argument("--seed", type=int, default=None, help="Override seed")
    train_parser.add_argument(
        "--timesteps", type=int, default=None, help="Override total_timesteps"
    )

    # ── evaluate ──────────────────────────────────────────────────────────
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained agent")
    eval_parser.add_argument(
        "--model-path",
        type=str,
        default=str(PROJECT_ROOT / "models" / "best_model.zip"),
        help="Path to saved model .zip",
    )
    eval_parser.add_argument(
        "--episodes", type=int, default=10, help="Number of evaluation episodes"
    )
    eval_parser.add_argument(
        "--env-config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "env_config.yaml"),
    )

    # ── render ────────────────────────────────────────────────────────────
    render_parser = subparsers.add_parser(
        "render", help="Render the agent in real-time"
    )
    render_parser.add_argument(
        "--model-path",
        type=str,
        default=str(PROJECT_ROOT / "models" / "best_model.zip"),
    )
    render_parser.add_argument(
        "--env-config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "env_config.yaml"),
    )

    # ── plot ──────────────────────────────────────────────────────────────
    plot_parser = subparsers.add_parser(
        "plot", help="Generate training/evaluation plots"
    )
    plot_parser.add_argument(
        "--log-dir",
        type=str,
        default=str(PROJECT_ROOT / "logs"),
        help="Directory containing training logs",
    )
    plot_parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "plots"),
        help="Directory to save plots",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point — dispatches to the appropriate subcommand."""
    args = parse_args()
    logger = setup_logger()

    if args.command is None:
        logger.error("No command specified. Use --help for usage information.")
        sys.exit(1)

    logger.info(f"Running command: {args.command}")

    if args.command == "train":
        env_config = load_config(args.env_config)
        train_config = load_config(args.train_config)
        seed = args.seed if args.seed is not None else train_config.get("seed", 42)
        set_global_seed(seed)
        logger.info(f"Seed set to {seed}")

        from training.train import run_training

        run_training(env_config, train_config, args)

    elif args.command == "evaluate":
        env_config = load_config(args.env_config)

        from evaluation.evaluate import run_evaluation

        run_evaluation(env_config, args)

    elif args.command == "render":
        env_config = load_config(args.env_config)

        from evaluation.evaluate import run_render

        run_render(env_config, args)

    elif args.command == "plot":
        from evaluation.visualize import generate_plots

        generate_plots(args)

    logger.info("Done.")


if __name__ == "__main__":
    main()
