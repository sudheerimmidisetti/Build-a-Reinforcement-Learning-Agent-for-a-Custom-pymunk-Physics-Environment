"""Setup script for Double Inverted Pendulum RL project."""

from setuptools import setup, find_packages

setup(
    name="double_inverted_pendulum",
    version="0.1.0",
    description="RL Agent for a Custom Double Inverted Pendulum",
    author="GPP",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.1.0",
        "torch>=2.0.0",
        "pymunk>=6.5.0",
        "pygame>=2.5.0",
        "matplotlib>=3.7.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tensorboard>=2.14.0",
    ],
    entry_points={
        "console_scripts": [
            "double-pendulum=main:main",
        ],
    },
)
