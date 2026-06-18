"""YAML configuration loader for Guardian Ear.

Provides a cached ``load_config`` function so that every module in the
project reads the same configuration dict without repeated disk I/O.
Returns a **deep copy** of the cached dict so callers can mutate their
local copy (e.g., override ``inference.location``) without corrupting
the shared singleton.

Usage::

    from src.utils.config_loader import load_config

    cfg = load_config()
    sample_rate = cfg["audio"]["sample_rate"]
"""

from __future__ import annotations

import copy
import functools
from pathlib import Path
from typing import Any, Dict, Optional

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
def _load_config_cached(path: str) -> Dict[str, Any]:
    """Internal cached loader — returns the ORIGINAL dict (do not mutate)."""
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            f"Ensure 'configs/config.yaml' exists at the project root."
        )

    logger.info("Loading configuration from %s", config_path)

    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            config: Dict[str, Any] = yaml.safe_load(fh) or {}
            
        # ── SECURITY: Load and merge local overrides ────────────────────────
        local_path = config_path.parent / "config.local.yaml"
        if local_path.exists():
            logger.info("Found local override config at %s", local_path)
            with open(local_path, "r", encoding="utf-8") as fh:
                local_config = yaml.safe_load(fh) or {}
                
            # Simple recursive merge
            def _merge(target, source):
                for k, v in source.items():
                    if isinstance(v, dict) and k in target and isinstance(target[k], dict):
                        _merge(target[k], v)
                    else:
                        target[k] = v
            _merge(config, local_config)
            
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


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load and return the YAML configuration as a **deep-copied** dictionary.

    The raw config is cached internally to avoid repeated disk I/O.
    Every call returns an independent deep copy so callers can safely
    mutate their local instance (e.g., ``cfg['inference']['location'] = loc``)
    without corrupting the shared singleton.

    Args:
        path: Absolute or relative path to a YAML config file.
            Defaults to ``configs/config.yaml`` at the project root.

    Returns:
        A nested dictionary mirroring the YAML structure (deep copy).

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
    resolved = str(Path(path).resolve() if path else _DEFAULT_CONFIG)
    return copy.deepcopy(_load_config_cached(resolved))


def get_config_value(key_path: str, default: Any = None, path: Optional[str] = None) -> Any:
    """Safely retrieve a nested config value using dot notation.

    Args:
        key_path: Dot-separated key path, e.g. ``'audio.sample_rate'``.
        default: Value to return if the key is not found.
        path: Optional config file path override.

    Returns:
        The value at the given key path, or ``default``.

    Examples:
        >>> get_config_value('audio.sample_rate')
        22050
        >>> get_config_value('audio.nonexistent', default=16000)
        16000
    """
    cfg = load_config(path)
    keys = key_path.split('.')
    node: Any = cfg
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, None)
        if node is None:
            return default
    return node


def reload_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Clear the cache and reload the configuration from disk.

    Useful after programmatic edits to the config file or during
    testing when you need a fresh read.

    Args:
        path: Same as :func:`load_config`.

    Returns:
        A freshly-loaded configuration dictionary (deep copy).
    """
    _load_config_cached.cache_clear()
    logger.info("Configuration cache cleared — reloading")
    return load_config(path)
