"""Logging configuration for the Double Inverted Pendulum project."""

import logging
import sys
from pathlib import Path
from typing import Optional

from utils.helpers import ensure_dir


def setup_logger(
    name: str = "double_pendulum",
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a logger with console and optional file output.

    Args:
        name: Logger name.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to a log file. If None, logs only to console.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times when called repeatedly
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # ── Formatter ──────────────────────────────────────────────────────────
    formatter = logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ───────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File handler (optional) ───────────────────────────────────────────
    if log_file is not None:
        log_path = Path(log_file)
        ensure_dir(str(log_path.parent))
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
