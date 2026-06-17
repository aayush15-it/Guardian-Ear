"""
src/notifications/desktop_notifier.py
──────────────────────────────────────
Desktop Notification helper for Guardian Ear.

Provides a three-tier fallback chain so the system never crashes due
to a missing notification backend:

  1. plyer.notification.notify()   — cross-platform, preferred
  2. win10toast.ToastNotifier()    — Windows-native fallback
  3. logger.warning()              — last-resort console fallback (returns False)

Usage::

    from src.notifications.desktop_notifier import DesktopNotifier

    notifier = DesktopNotifier()
    notifier.notify_emergency(
        sound_class='gun_shot',
        confidence=92.5,
        threat_score=88,
        threat_level='CRITICAL',
        location='parking_lot',
    )
"""

from __future__ import annotations

import logging

# ─── Logger setup (graceful fallback if src package isn't on sys.path) ────────
try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)


class DesktopNotifier:
    """Cross-platform desktop notification sender with automatic fallbacks.

    Attributes:
        _backend (str): Name of the active notification backend
            (``'plyer'``, ``'win10toast'``, or ``'none'``).
    """

    def __init__(self) -> None:
        self._backend: str = self._detect_backend()
        logger.debug("DesktopNotifier initialised — backend: %s", self._backend)

    # ─── Backend detection ────────────────────────────────────────────────────

    @staticmethod
    def _detect_backend() -> str:
        """Return the name of the first importable notification backend."""
        try:
            import plyer  # noqa: F401
            return "plyer"
        except ImportError:
            pass
        try:
            import win10toast  # noqa: F401
            return "win10toast"
        except ImportError:
            pass
        return "none"

    # ─── Public API ───────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return ``True`` if either plyer or win10toast is importable."""
        return self._backend in ("plyer", "win10toast")

    def notify(self, title: str, message: str, timeout: int = 10) -> bool:
        """Send a desktop notification.

        Tries plyer first, then win10toast, then falls back to logging.

        Args:
            title:   Notification title string.
            message: Notification body text.
            timeout: Display duration in seconds (best-effort; platform dependent).

        Returns:
            ``True`` if a visual notification was dispatched, ``False`` otherwise.
        """
        # ── Attempt 1: plyer ──────────────────────────────────────────────────
        try:
            from plyer import notification as plyer_notification  # type: ignore
            plyer_notification.notify(
                title=title,
                message=message,
                app_name="Guardian Ear",
                timeout=timeout,
            )
            logger.debug("Desktop notification sent via plyer: %s", title)
            return True
        except Exception as exc:
            logger.debug("plyer notification failed: %s", exc)

        # ── Attempt 2: win10toast ─────────────────────────────────────────────
        try:
            from win10toast import ToastNotifier  # type: ignore
            toaster = ToastNotifier()
            toaster.show_toast(
                title,
                message,
                duration=timeout,
                threaded=True,
            )
            logger.debug("Desktop notification sent via win10toast: %s", title)
            return True
        except Exception as exc:
            logger.debug("win10toast notification failed: %s", exc)

        # ── Attempt 3: console fallback ───────────────────────────────────────
        logger.warning(
            "No desktop notification backend available — alert suppressed to log only.\n"
            "  Title  : %s\n"
            "  Message: %s",
            title,
            message,
        )
        return False

    def notify_emergency(
        self,
        sound_class: str,
        confidence: float,
        threat_score: float,
        threat_level: str,
        location: str,
    ) -> bool:
        """Send a pre-formatted Guardian Ear emergency desktop notification.

        Args:
            sound_class:  Detected sound label (e.g. ``'gun_shot'``).
            confidence:   Model confidence in percent (0-100).
            threat_score: Computed threat score (0-100).
            threat_level: Human-readable level string (e.g. ``'CRITICAL'``).
            location:     Deployment location string.

        Returns:
            ``True`` if a visual notification was dispatched successfully.
        """
        title = "GUARDIAN EAR ALERT"
        message = (
            f"Sound: {sound_class}\n"
            f"Confidence: {confidence:.1f}%\n"
            f"Threat: {threat_level} ({threat_score:.0f}/100)\n"
            f"Location: {location}\n"
            "\nImmediate Attention Required"
        )
        logger.info(
            "Dispatching emergency desktop notification — %s @ %s (score=%s)",
            sound_class,
            location,
            threat_score,
        )
        return self.notify(title=title, message=message, timeout=10)
