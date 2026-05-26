"""Production-grade logging module for Guardian Ear.

Provides a ``get_logger`` factory that returns consistently configured
loggers under the ``GuardianEar`` namespace.  All loggers share:

* A **console handler** with ANSI colour-coded log levels.
* A **rotating file handler** that writes to ``logs/guardian_ear.log``
  (max 5 MB per file, 3 backups).

Set the environment variable ``GUARDIAN_DEBUG=1`` to switch every logger
to DEBUG level.

Usage::

    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Model loaded successfully")
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


# ─── Constants ────────────────────────────────────────────────────────────────
_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"  # <project>/logs
_LOG_FILE = _LOG_DIR / "guardian_ear.log"
_MAX_BYTES = 5 * 1024 * 1024   # 5 MB per log file
_BACKUP_COUNT = 3

_DEBUG_ENV_VAR = "GUARDIAN_DEBUG"

# ANSI colour codes for console output
_ANSI_COLORS: dict[str, str] = {
    "DEBUG": "\033[36m",       # Cyan
    "INFO": "\033[32m",        # Green
    "WARNING": "\033[33m",     # Yellow
    "ERROR": "\033[31m",       # Red
    "CRITICAL": "\033[1;31m",  # Bold Red
}
_ANSI_RESET = "\033[0m"

# Module-level flag to prevent duplicate initialisation
_initialized: bool = False


# ─── Custom Formatter ─────────────────────────────────────────────────────────
class _ColorFormatter(logging.Formatter):
    """Logging formatter that prepends ANSI colour codes on terminals.

    Falls back to plain text when stdout is not a TTY (e.g. piped to a
    file or running inside CI).
    """

    def __init__(self, fmt: str, datefmt: str) -> None:
        super().__init__(fmt, datefmt=datefmt)
        self._use_color = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        if self._use_color:
            colour = _ANSI_COLORS.get(record.levelname, "")
            return f"{colour}{formatted}{_ANSI_RESET}"
        return formatted


# ─── Internal Setup ───────────────────────────────────────────────────────────
def _resolve_level() -> int:
    """Return DEBUG if ``GUARDIAN_DEBUG=1``, otherwise INFO."""
    return logging.DEBUG if os.environ.get(_DEBUG_ENV_VAR) == "1" else logging.INFO


def _setup_root_logger() -> None:
    """Configure the root ``GuardianEar`` logger exactly once.

    Adds both a coloured console handler and a rotating file handler.
    """
    global _initialized
    if _initialized:
        return

    level = _resolve_level()
    root = logging.getLogger("GuardianEar")
    root.setLevel(level)

    # Prevent log messages from propagating to the default root logger
    root.propagate = False

    # ── Console handler (coloured) ──────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(_ColorFormatter(_LOG_FORMAT, _DATE_FORMAT))
    root.addHandler(console_handler)

    # ── Rotating file handler (plain text) ──────────────────────────
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=str(_LOG_FILE),
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(file_handler)
    except OSError as exc:
        # If we can't write to disk (e.g. read-only FS), log a warning
        # to the console handler that's already attached and continue.
        root.warning("Failed to create file log handler: %s", exc)

    _initialized = True


# ─── Public API ───────────────────────────────────────────────────────────────
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger under the ``GuardianEar`` namespace.

    Args:
        name: Dot-separated child name appended to ``GuardianEar``.
            Pass ``__name__`` for automatic module-scoped naming.
            If *None*, the root ``GuardianEar`` logger is returned.

    Returns:
        A ``logging.Logger`` instance with console and file handlers.

    Examples:
        >>> logger = get_logger("feature_extraction")
        >>> logger.info("Processing %d files", 100)
        [2026-05-21 22:50:00] [INFO] [GuardianEar.feature_extraction] Processing 100 files
    """
    _setup_root_logger()
    full_name = f"GuardianEar.{name}" if name else "GuardianEar"
    return logging.getLogger(full_name)
