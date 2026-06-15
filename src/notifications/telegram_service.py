"""
src/notifications/telegram_service.py
──────────────────────────────────────
Telegram Alert Service for Guardian Ear.

Sends real-time push notifications to a Telegram chat whenever the threat
engine raises an alert.  Deliberately uses only the Python standard library
(``urllib.request``) so no third-party packages are required.

Configuration precedence (highest → lowest):
    1. Arguments passed to ``__init__``.
    2. Environment variables ``GUARDIAN_TELEGRAM_TOKEN`` and
       ``GUARDIAN_TELEGRAM_CHAT_ID``.
    3. Empty strings (service disabled).

Typical usage
─────────────
    from src.notifications.telegram_service import TelegramAlertService

    svc = TelegramAlertService()          # reads env vars automatically
    if svc.is_configured():
        svc.send_alert(
            sound_class="gun_shot",
            confidence=0.94,
            threat_level="CRITICAL",
            threat_score=87,
            location="parking_lot",
            action_type="EMERGENCY_CALL",
        )
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
_TELEGRAM_API_BASE: str = "https://api.telegram.org/bot{token}/sendMessage"
_ENV_TOKEN: str = "GUARDIAN_TELEGRAM_TOKEN"
_ENV_CHAT_ID: str = "GUARDIAN_TELEGRAM_CHAT_ID"

# Retry schedule for _send_with_retry: delays in seconds between attempts.
_RETRY_DELAYS: tuple[float, ...] = (0.5, 1.0, 2.0)

# HTTP request timeout in seconds.
_REQUEST_TIMEOUT: float = 10.0


class TelegramAlertService:
    """Push real-time Guardian Ear threat alerts to a Telegram chat.

    Parameters
    ----------
    token : str, optional
        Telegram Bot API token.  Falls back to ``GUARDIAN_TELEGRAM_TOKEN``
        environment variable when not supplied or empty.
    chat_id : str, optional
        Target Telegram chat or channel ID.  Falls back to
        ``GUARDIAN_TELEGRAM_CHAT_ID`` environment variable when not supplied
        or empty.

    Attributes
    ----------
    _token : str
        Resolved bot token (may be empty if neither arg nor env var set).
    _chat_id : str
        Resolved chat ID (may be empty if neither arg nor env var set).
    """

    def __init__(
        self,
        token: str = "",
        chat_id: str = "",
    ) -> None:
        self._token: str = token or os.environ.get(_ENV_TOKEN, "").strip()
        self._chat_id: str = chat_id or os.environ.get(_ENV_CHAT_ID, "").strip()

        if self.is_configured():
            logger.info(
                "TelegramAlertService initialised | chat_id=%s | token=***%s",
                self._chat_id,
                self._token[-4:] if len(self._token) >= 4 else "****",
            )
        else:
            logger.warning(
                "TelegramAlertService: token or chat_id not set — "
                "alerts will be suppressed.  Set %s and %s environment "
                "variables or pass credentials to __init__.",
                _ENV_TOKEN,
                _ENV_CHAT_ID,
            )

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def is_configured(self) -> bool:
        """Return ``True`` only when both bot token and chat ID are non-empty.

        Returns
        -------
        bool
            ``True`` if the service can send messages; ``False`` otherwise.
        """
        return bool(self._token) and bool(self._chat_id)

    def send_alert(
        self,
        sound_class: str,
        confidence: float,
        threat_level: str,
        threat_score: int | float,
        location: str,
        action_type: str,
    ) -> bool:
        """Send a formatted threat-alert message to the configured Telegram chat.

        Parameters
        ----------
        sound_class : str
            Detected UrbanSound8K class name (e.g. ``"gun_shot"``).
        confidence : float
            Model confidence in the range [0, 1].  Displayed as a percentage.
        threat_level : str
            Human-readable severity label (e.g. ``"CRITICAL"``, ``"HIGH"``).
        threat_score : int or float
            Numeric threat score in the range [0, 100].
        location : str
            Deployment location identifier (e.g. ``"parking_lot"``).
        action_type : str
            Action taken by the action engine (e.g. ``"EMERGENCY_CALL"``).

        Returns
        -------
        bool
            ``True`` if the message was delivered successfully, ``False`` if
            the service is not configured or all retry attempts failed.
        """
        if not self.is_configured():
            logger.debug(
                "send_alert skipped — TelegramAlertService not configured."
            )
            return False

        message: str = self._format_message(
            sound_class=sound_class,
            confidence=confidence,
            threat_level=threat_level,
            threat_score=threat_score,
            location=location,
            action_type=action_type,
        )

        logger.info(
            "Sending Telegram alert | class=%s | score=%s | level=%s",
            sound_class,
            threat_score,
            threat_level,
        )

        success: bool = self._send_with_retry(message)
        if success:
            logger.info("Telegram alert delivered successfully.")
        else:
            logger.error(
                "Telegram alert delivery FAILED after %d attempts.",
                len(_RETRY_DELAYS) + 1,
            )
        return success

    def send_test_alert(self) -> bool:
        """Send a connectivity test message to verify the configuration.

        Returns
        -------
        bool
            ``True`` if the test message was delivered, ``False`` otherwise.
        """
        if not self.is_configured():
            logger.warning(
                "send_test_alert skipped — TelegramAlertService not configured."
            )
            return False

        test_message: str = (
            "✅ *Guardian Ear Connected\\. System ready\\.*\n"
            "_This is an automated connectivity test message\\._"
        )

        logger.info("Sending Telegram test alert.")
        success: bool = self._send_with_retry(test_message)

        if success:
            logger.info("Telegram test alert delivered successfully.")
        else:
            logger.error("Telegram test alert delivery FAILED.")
        return success

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _format_message(
        self,
        sound_class: str,
        confidence: float,
        threat_level: str,
        threat_score: int | float,
        location: str,
        action_type: str,
    ) -> str:
        """Build the Telegram MarkdownV2 alert message string.

        Parameters
        ----------
        sound_class : str
            Detected sound class name.
        confidence : float
            Model confidence [0, 1]; displayed as a rounded percentage.
        threat_level : str
            Severity label string.
        threat_score : int or float
            Numeric threat score out of 100.
        location : str
            Deployment location label.
        action_type : str
            Action engine response label.

        Returns
        -------
        str
            Fully-formatted MarkdownV2 message ready for the Telegram API.
        """
        confidence_pct: float = round(confidence * 100, 1)
        timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Escape characters that are special in Telegram MarkdownV2.
        def _esc(text: str) -> str:
            """Escape special MarkdownV2 characters."""
            special_chars = r"\_*[]()~`>#+-=|{}.!"
            for ch in special_chars:
                text = text.replace(ch, f"\\{ch}")
            return text

        message: str = (
            "🚨 *GUARDIAN EAR ALERT*\n"
            "\n"
            f"🔊 *Sound:* {_esc(sound_class.replace('_', ' ').title())}\n"
            f"📊 *Confidence:* {_esc(str(confidence_pct))}%\n"
            f"🎯 *Threat Score:* {_esc(str(int(threat_score)))}/100\n"
            f"⚠️ *Threat Level:* {_esc(threat_level)}\n"
            f"📍 *Location:* {_esc(location.replace('_', ' ').title())}\n"
            f"⏰ *Time:* {_esc(timestamp)}\n"
            "\n"
            f"_Action: {_esc(action_type)}_"
        )

        return message

    def _send_with_retry(
        self,
        message: str,
        max_retries: int = 3,
    ) -> bool:
        """Attempt to send *message* with exponential backoff on failure.

        Makes an initial attempt followed by up to *max_retries* − 1 retries
        with delays of 0.5 s, 1.0 s, and 2.0 s between attempts.

        Parameters
        ----------
        message : str
            Pre-formatted message body to send.
        max_retries : int, optional
            Total number of attempts (initial + retries).  Defaults to 3.

        Returns
        -------
        bool
            ``True`` if any attempt succeeds, ``False`` if all fail.
        """
        delays: tuple[float, ...] = _RETRY_DELAYS[: max(0, max_retries - 1)]

        for attempt in range(1, max_retries + 1):
            try:
                success: bool = self._send_message_once(message)
                if success:
                    logger.debug("Message delivered on attempt %d.", attempt)
                    return True
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Telegram send attempt %d/%d failed: %s",
                    attempt,
                    max_retries,
                    exc,
                )

            if attempt < max_retries:
                delay: float = delays[attempt - 1] if attempt - 1 < len(delays) else 2.0
                logger.debug(
                    "Retrying in %.1f s (attempt %d/%d) …",
                    delay,
                    attempt,
                    max_retries,
                )
                time.sleep(delay)

        logger.error("All %d send attempts exhausted.", max_retries)
        return False

    def _send_message_once(self, message: str) -> bool:
        """Send *message* to the Telegram Bot API (single attempt).

        Uses ``urllib.request.urlopen`` with ``parse_mode=MarkdownV2`` and a
        configurable timeout.  Reads and logs the JSON response body on both
        success and failure.

        Parameters
        ----------
        message : str
            Formatted message body (MarkdownV2).

        Returns
        -------
        bool
            ``True`` if the API returns ``{"ok": true}``, ``False`` otherwise.

        Raises
        ------
        urllib.error.URLError
            On network-level errors (DNS, timeout, refused connection).
        urllib.error.HTTPError
            On HTTP 4xx / 5xx responses from the Telegram API.
        """
        url: str = _TELEGRAM_API_BASE.format(token=self._token)

        payload: dict[str, Any] = {
            "chat_id": self._chat_id,
            "text": message,
            "parse_mode": "MarkdownV2",
            "disable_web_page_preview": True,
        }

        data: bytes = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "GuardianEar/1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT) as response:
                raw_body: bytes = response.read()
                body: dict[str, Any] = json.loads(raw_body.decode("utf-8"))

                ok: bool = body.get("ok", False)
                if ok:
                    logger.debug("Telegram API response: ok=True")
                else:
                    logger.warning(
                        "Telegram API returned ok=False: %s",
                        body.get("description", "no description"),
                    )
                return ok

        except urllib.error.HTTPError as exc:
            raw_body = exc.read()
            logger.error(
                "Telegram HTTP error %d | body=%s",
                exc.code,
                raw_body.decode("utf-8", errors="replace"),
            )
            raise

        except urllib.error.URLError as exc:
            logger.error("Telegram network error: %s", exc.reason)
            raise

    # ──────────────────────────────────────────────────────────────────────
    # Dunder helpers
    # ──────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:  # pragma: no cover
        configured: str = "configured" if self.is_configured() else "NOT configured"
        return (
            f"TelegramAlertService("
            f"chat_id={self._chat_id!r}, "
            f"status={configured!r})"
        )
