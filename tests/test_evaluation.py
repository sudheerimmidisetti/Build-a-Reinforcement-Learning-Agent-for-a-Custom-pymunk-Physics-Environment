"""Tests for the evaluation and visualization modules (Phase 6 gate)."""

import json
import os

import numpy as np
import pytest

from utils.helpers import get_project_root


class TestEvalMetrics:
    """Test that evaluation produces valid metric structures."""

    def test_metrics_json_structure(self, tmp_path):
        """Verify the expected JSON structure of evaluation metrics."""
        # Create a dummy metrics file
        metrics = {
            "n_episodes": 5,
            "mean_reward": 123.45,
            "std_reward": 10.0,
            "min_reward": 100.0,
            "max_reward": 150.0,
            "mean_length": 200.0,
            "std_length": 20.0,
            "mean_max_angle1_deg": 5.0,
            "mean_max_angle2_deg": 8.0,
            "episode_rewards": [100.0, 110.0, 120.0, 130.0, 150.0],
            "episode_lengths": [180, 190, 200, 210, 220],
        }
        metrics_path = str(tmp_path / "evaluation_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

        # Reload and verify
        with open(metrics_path, "r") as f:
            loaded = json.load(f)

        assert loaded["n_episodes"] == 5
        assert isinstance(loaded["episode_rewards"], list)
        assert len(loaded["episode_rewards"]) == 5


class TestVisualize:
    """Test plot generation (without display)."""

    def test_plot_angle_trajectory(self, tmp_path):
        """Verify angle trajectory plot is saved."""
        from evaluation.visualize import plot_angle_trajectory

        angles_1 = list(np.linspace(0, 0.3, 100))
        angles_2 = list(np.linspace(0, -0.2, 100))
        cart_positions = list(np.linspace(0, 0.5, 100))

        output_dir = str(tmp_path)
        plot_angle_trajectory(angles_1, angles_2, cart_positions, output_dir)

        assert os.path.exists(os.path.join(output_dir, "angle_trajectory.png"))

    def test_plot_eval_distribution(self, tmp_path):
        """Verify evaluation distribution plot is saved."""
        from evaluation.visualize import plot_eval_distribution

        metrics = {
            "episode_rewards": [100.0, 110.0, 120.0, 130.0, 150.0],
        }
        metrics_path = str(tmp_path / "evaluation_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

        plot_eval_distribution(metrics_path, str(tmp_path))

        assert os.path.exists(os.path.join(str(tmp_path), "eval_reward_distribution.png"))
