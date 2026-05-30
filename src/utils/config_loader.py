"""YAML configuration loader for Guardian Ear.

Provides a cached ``load_config`` function so that every module in the
project reads the same configuration dict without repeated disk I/O.

Usage::

    from src.utils.config_loader import load_config

    cfg = load_config()
    sample_rate = cfg["audio"]["sample_rate"]
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ModuleNotFoundError:
    # Self-heal: Streamlit may run with a subprocess that excludes the venv
    # from sys.path. Find guardian_env relative to this file and inject it.
    import sys as _sys
    from pathlib import Path as _Path
    _project_root = _Path(__file__).resolve().parents[2]
    _venv_site = _project_root / 'guardian_env' / 'lib' / 'site-packages'
    if _venv_site.exists() and str(_venv_site) not in _sys.path:
        _sys.path.insert(0, str(_venv_site))
    import yaml  # retry after path fix — will raise if still missing

from src.utils.logger import get_logger

logger = get_logger("config_loader")

# Default config path relative to the project root
_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


@functools.lru_cache(maxsize=4)
def load_config(path: str | None = None) -> Dict[str, Any]:
    """Load and return the YAML configuration as a dictionary.

    The result is cached via ``functools.lru_cache`` so subsequent calls
    with the same *path* return the identical dict without re-reading
    the file.

    Args:
        path: Absolute or relative path to a YAML config file.
            Defaults to ``configs/config.yaml`` at the project root.

    Returns:
        A nested dictionary mirroring the YAML structure.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML content is malformed.

    Examples:
        >>> cfg = load_config()
        >>> cfg["audio"]["sample_rate"]
        22050
        >>> cfg["class_names"][6]
        'gun_shot'
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            f"Ensure 'configs/config.yaml' exists at the project root."
        )

    logger.info("Loading configuration from %s", config_path)

    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            config: Dict[str, Any] = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        logger.error("Failed to parse YAML config: %s", exc)
        raise

    if config is None:
        logger.warning("Config file is empty, returning empty dict")
        return {}

    logger.debug(
        "Configuration loaded — top-level keys: %s",
        list(config.keys()),
    )
    return config


def reload_config(path: str | None = None) -> Dict[str, Any]:
    """Clear the cache and reload the configuration from disk.

    Useful after programmatic edits to the config file or during
    testing when you need a fresh read.

    Args:
        path: Same as :func:`load_config`.

    Returns:
        A freshly-loaded configuration dictionary.
    """
    load_config.cache_clear()
    logger.info("Configuration cache cleared — reloading")
    return load_config(path)
