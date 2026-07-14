"""Hyperparameter definitions for RL algorithms.

Loads from ``configs/train_config.yaml`` and provides algorithm-specific
parameter dictionaries compatible with Stable-Baselines3 constructors.
"""

from typing import Any, Dict, Optional


def get_algo_params(
    train_config: Dict[str, Any], algo_override: Optional[str] = None
) -> Dict[str, Any]:
    """Extract algorithm-specific hyperparameters from the training config.

    Args:
        train_config: Full training config dict loaded from YAML.
        algo_override: Optional algorithm name override (e.g. ``"SAC"``).
            If ``None``, uses ``train_config["algorithm"]``.

    Returns:
        Dictionary of kwargs suitable for passing to the SB3 algorithm
        constructor (e.g. ``PPO(**params)``).
    """
    algo = (algo_override or train_config["algorithm"]).upper()

    if algo == "PPO":
        return _get_ppo_params(train_config)
    elif algo == "SAC":
        return _get_sac_params(train_config)
    elif algo == "TD3":
        return _get_td3_params(train_config)
    else:
        raise ValueError(
            f"Unsupported algorithm: {algo}. Choose from PPO, SAC, TD3."
        )


def _get_ppo_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build PPO constructor kwargs from config."""
    ppo = config.get("ppo", {})

    params = {
        "learning_rate": ppo.get("learning_rate", 3e-4),
        "n_steps": ppo.get("n_steps", 2048),
        "batch_size": ppo.get("batch_size", 64),
        "n_epochs": ppo.get("n_epochs", 10),
        "gamma": ppo.get("gamma", 0.99),
        "gae_lambda": ppo.get("gae_lambda", 0.95),
        "clip_range": ppo.get("clip_range", 0.2),
        "ent_coef": ppo.get("ent_coef", 0.0),
        "vf_coef": ppo.get("vf_coef", 0.5),
        "max_grad_norm": ppo.get("max_grad_norm", 0.5),
    }

    # Policy kwargs (network architecture)
    policy_kwargs = ppo.get("policy_kwargs", {})
    if policy_kwargs:
        net_arch = policy_kwargs.get("net_arch", None)
        if net_arch:
            params["policy_kwargs"] = {"net_arch": net_arch}

    return params


def _get_sac_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build SAC constructor kwargs from config."""
    sac = config.get("sac", {})

    params = {
        "learning_rate": sac.get("learning_rate", 3e-4),
        "buffer_size": sac.get("buffer_size", 1_000_000),
        "learning_starts": sac.get("learning_starts", 10_000),
        "batch_size": sac.get("batch_size", 256),
        "tau": sac.get("tau", 0.005),
        "gamma": sac.get("gamma", 0.99),
        "ent_coef": sac.get("ent_coef", "auto"),
    }

    policy_kwargs = sac.get("policy_kwargs", {})
    if policy_kwargs:
        net_arch = policy_kwargs.get("net_arch", None)
        if net_arch:
            params["policy_kwargs"] = {"net_arch": net_arch}

    return params


def _get_td3_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build TD3 constructor kwargs from config (uses SAC section as base)."""
    td3 = config.get("td3", config.get("sac", {}))

    return {
        "learning_rate": td3.get("learning_rate", 3e-4),
        "buffer_size": td3.get("buffer_size", 1_000_000),
        "learning_starts": td3.get("learning_starts", 10_000),
        "batch_size": td3.get("batch_size", 256),
        "tau": td3.get("tau", 0.005),
        "gamma": td3.get("gamma", 0.99),
    }
