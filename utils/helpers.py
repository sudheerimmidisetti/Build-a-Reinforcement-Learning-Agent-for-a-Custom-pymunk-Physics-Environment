"""Utility helper functions for the Double Inverted Pendulum project."""

import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file and return it as a dictionary.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary of configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def ensure_dir(path: str) -> Path:
    """Create a directory (and parents) if it does not already exist.

    Args:
        path: Directory path to create.

    Returns:
        The Path object for the created directory.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def timestamp_string() -> str:
    """Return a filesystem-safe timestamp string for naming artefacts.

    Returns:
        String in format 'YYYYMMDD_HHMMSS'.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_project_root() -> Path:
    """Return the root directory of the project.

    Walks up from this file until it finds setup.py.

    Returns:
        Path to the project root.
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "setup.py").exists():
            return current
        current = current.parent
    # Fallback: return parent of utils/
    return Path(__file__).resolve().parent.parent
