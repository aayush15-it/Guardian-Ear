"""
src/notifications/emergency_engine.py
─────────────────────────────────────
Emergency Notification Engine for Guardian Ear.

Orchestrates all alert channels when a critical threat is detected:
  Desktop Notification → Email Alert → Emergency Contact Log → Dashboard Event

All channels are optional and gracefully degrade — if desktop notifications
are unavailable, email still fires. If email is unconfigured, contact log
still records. The system never crashes due to a missing channel.

Configuration is read from ``configs/config.yaml`` under the ``emergency`` key.
If that key is absent, safe defaults are applied so the engine always boots.

Typical usage::

    from src.notifications.emergency_engine import from_config

    engine = from_config()
    record = engine.dispatch(
        sound_class='gun_shot',
        confidence=93.5,
        threat_score=88,
        threat_level='CRITICAL',
        location='parking_lot',
    )
    print(record['channels_fired'])
"""

from __future__ import annotations

import logging
import time
import csv
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

# ─── Logger (graceful fallback) ───────────────────────────────────────────────
try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)

# ─── Internal notification modules ────────────────────────────────────────────
from .desktop_notifier import DesktopNotifier
from .email_service import EmailAlertService


# ─────────────────────────────────────────────────────────────────────────────
class EmergencyNotificationEngine:
    """Central orchestrator that fires all notification channels on a threat.

    Channels fired in order:
        1. Desktop notification (plyer → win10toast → log fallback)
        2. Email alert (SMTP, optional — requires configuration)
        3. Emergency contact escalation log (simulation + future SMS hook)

    All channels are independent; failure in one does not block others.

    Args:
        config: Optional configuration dictionary. If ``None``, an attempt is
            made to load ``configs/config.yaml`` automatically.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._cfg = self._resolve_config(config)
        self._desktop = DesktopNotifier()
        self._email = self._build_email_service()
        self._last_email_time: Dict[str, float] = {}
        self._email_cooldown = self._cfg.get("dispatch", {}).get("cooldown_seconds", 60)
        
        logger.info(
            "EmergencyNotificationEngine ready — "
            "desktop=%s | email=%s | contact=%s",
            self._desktop.is_available(),
            self._email.is_configured(),
            self.is_contact_configured(),
        )

    # ─── Config helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _resolve_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load config from YAML if not provided; always return the *emergency* sub-dict."""
        if config is not None:
            # Caller may pass either the full config dict or just the emergency sub-dict
            return config.get("emergency", config)

        try:
            from src.utils.config_loader import load_config
            full_cfg = load_config()
            return full_cfg.get("emergency", {})
        except Exception as exc:
            logger.warning(
                "Could not load config.yaml — using default emergency settings. (%s)", exc
            )
            return {}

    def _build_email_service(self) -> EmailAlertService:
        """Instantiate EmailAlertService from config values."""
        email_cfg = self._cfg.get("email", {})
        return EmailAlertService(
            sender_email=email_cfg.get("sender_email", ""),
            sender_password=email_cfg.get("sender_password", ""),
            recipient_email=email_cfg.get("recipient_email", ""),
            smtp_host=email_cfg.get("smtp_host", "smtp.gmail.com"),
            smtp_port=int(email_cfg.get("smtp_port", 587)),
            enabled=bool(email_cfg.get("enabled", True)),
        )

    # ─── Public status checks ─────────────────────────────────────────────────

    def is_email_configured(self) -> bool:
        """Return ``True`` if SMTP credentials and addresses are all set."""
        return self._email.is_configured()

    def is_contact_configured(self) -> bool:
        """Return ``True`` if at least one emergency contact is configured."""
        contacts = self._cfg.get("contacts", [])
        return bool(contacts)

    # ─── Main dispatch method ─────────────────────────────────────────────────

    def dispatch(
        self,
        sound_class: str,
        confidence: float,
        threat_score: float,
        threat_level: str,
        location: str,
        open_set_status: str = "KNOWN",
        vote_count: int = 1,
    ) -> Dict[str, Any]:
        """Fire all notification channels and return a consolidated record.

        Args:
            sound_class:  Detected sound label (e.g. ``'gun_shot'``).
            confidence:   Model confidence in percent (0-100).
            threat_score: Computed threat score (0-100).
            threat_level: Human-readable level string (e.g. ``'CRITICAL'``).
            location:     Deployment location string.
            open_set_status: Result of open-set rejection ('KNOWN' or 'UNKNOWN').
            vote_count:   Number of consecutive temporal votes.

        Returns:
            A ``notification_record`` dict containing status of every channel.
        """
        now = datetime.now()
        timestamp_str = now.strftime("%H:%M:%S")
        date_str = now.strftime("%Y-%m-%d")

        channels_fired: List[str] = []

        logger.info(
            "=== EMERGENCY DISPATCH === %s | %s | confidence=%.1f%% | score=%.0f | %s",
            sound_class.upper(),
            threat_level,
            confidence,
            threat_score,
            location,
        )

        # ── Channel 1: Desktop notification ───────────────────────────────────
        desktop_sent = self._fire_desktop(
            sound_class=sound_class,
            confidence=confidence,
            threat_score=threat_score,
            threat_level=threat_level,
            location=location,
        )
        if desktop_sent:
            channels_fired.append("desktop")

        # ── Channel 2: Email alert ────────────────────────────────────────────
        email_sent = self._fire_email(
            sound_class=sound_class,
            confidence=confidence,
            threat_score=threat_score,
            threat_level=threat_level,
            location=location,
        )
        if email_sent:
            channels_fired.append("email")

        # ── Channel 3: Emergency contact escalation ───────────────────────────
        contact_record = self._fire_contact_escalation(
            sound_class=sound_class,
            confidence=confidence,
            threat_score=threat_score,
            threat_level=threat_level,
        )
        if contact_record.get("contact_status") == "NOTIFIED":
            channels_fired.append("contact")

        # ── Build consolidated notification record ────────────────────────────
        notification_record: Dict[str, Any] = {
            "timestamp": timestamp_str,
            "date": date_str,
            "sound_class": sound_class,
            "confidence": round(float(confidence), 2),
            "threat_score": round(float(threat_score), 2),
            "threat_level": threat_level,
            "location": location,
            "desktop_sent": desktop_sent,
            "email_sent": email_sent,
            "contact_name": contact_record.get("contact_name", ""),
            "contact_phone": contact_record.get("contact_phone", ""),
            "contact_status": contact_record.get("contact_status", "NOT_CONFIGURED"),
            "channels_fired": channels_fired,
            "notification_type": "EMERGENCY",
            "open_set_status": open_set_status,
            "vote_count": vote_count,
            "delivery_status": "SUCCESS" if channels_fired else "FAILED",
            "recipient": contact_record.get("contact_email", "") or contact_record.get("contact_phone", "None"),
        }

        self._log_to_csv(notification_record)

        logger.info(
            "Dispatch complete — channels fired: %s",
            channels_fired if channels_fired else ["none"],
        )
        return notification_record

    # ─── Individual channel methods ───────────────────────────────────────────

    def _fire_desktop(
        self,
        sound_class: str,
        confidence: float,
        threat_score: float,
        threat_level: str,
        location: str,
    ) -> bool:
        """Attempt to send a desktop notification.

        Falls back gracefully through plyer → win10toast → log-only.

        Returns:
            ``True`` if a visual notification was successfully displayed.
        """
        try:
            return self._desktop.notify_emergency(
                sound_class=sound_class,
                confidence=confidence,
                threat_score=threat_score,
                threat_level=threat_level,
                location=location,
            )
        except Exception as exc:
            logger.error("Unexpected error in desktop notification channel: %s", exc)
            return False

    def _fire_email(
        self,
        sound_class: str,
        confidence: float,
        threat_score: float,
        threat_level: str,
        location: str,
    ) -> bool:
        """Attempt to send an SMTP email alert.

        Silently skips if email is not configured or if rate limited.

        Returns:
            ``True`` if email was delivered, ``False`` otherwise.
        """
        if not self._email.is_configured():
            logger.debug("Email channel skipped — not configured.")
            return False

        # Apply rate limiting to prevent alert storms
        now = time.time()
        last_time = self._last_email_time.get(sound_class, 0)
        if now - last_time < self._email_cooldown:
            logger.warning("Email rate limited for %s (cooldown active)", sound_class)
            return False

        subject = (
            f"[GUARDIAN EAR] {threat_level} ALERT: "
            f"{sound_class.replace('_', ' ').title()} detected at {location}"
        )
        timestamp_str = datetime.now().strftime("%H:%M:%S")

        try:
            success = self._email.send_alert(
                subject=subject,
                sound_class=sound_class,
                confidence=confidence,
                threat_score=threat_score,
                threat_level=threat_level,
                location=location,
                timestamp=timestamp_str,
            )
            if success:
                self._last_email_time[sound_class] = now
            return success
        except Exception as exc:
            logger.error("Unexpected error in email notification channel: %s", exc)
            return False

    def _fire_contact_escalation(
        self,
        sound_class: str,
        confidence: float,
        threat_score: float,
        threat_level: str,
    ) -> Dict[str, Any]:
        """Log an emergency contact escalation event.

        Currently simulated — a future SMS/call integration can replace the
        ``# TODO: SMS hook`` section below with a real provider (e.g. Twilio).

        Returns:
            A dict with keys ``contact_name``, ``contact_phone``,
            ``contact_status`` (``'NOTIFIED'`` or ``'NOT_CONFIGURED'``).
        """
        contacts: List[Dict[str, str]] = self._cfg.get("contacts", [])

        if not contacts:
            # Fallback to older emergency_contact structure if contacts list isn't present
            legacy = self._cfg.get("emergency_contact", {})
            if legacy.get("name") or legacy.get("phone"):
                contacts = [legacy]
            else:
                logger.debug("Contact escalation skipped — no contacts configured.")
                return {
                    "contact_name": "",
                    "contact_phone": "",
                    "contact_email": "",
                    "contact_status": "NOT_CONFIGURED",
                }

        # Use the first configured contact as the primary escalation target
        primary = contacts[0]
        name = primary.get("name", "Emergency Contact")
        phone = primary.get("phone", "")
        email_addr = primary.get("email", "")

        logger.warning(
            "CONTACT ESCALATION → %s (%s) | %s | %.1f%% confidence | score=%.0f",
            name,
            phone or "no phone",
            sound_class,
            confidence,
            threat_score,
        )

        # TODO: SMS hook — replace this block with Twilio/AWS SNS/other provider
        # e.g.:
        #   from twilio.rest import Client
        #   client = Client(account_sid, auth_token)
        #   client.messages.create(
        #       body=f"[Guardian Ear] {threat_level}: {sound_class} detected!",
        #       from_=twilio_number,
        #       to=phone,
        #   )

        return {
            "contact_name": name,
            "contact_phone": phone,
            "contact_email": email_addr,
            "contact_status": "NOTIFIED",
        }

    # ─── Test utilities ───────────────────────────────────────────────────────

    def send_test_alert(self) -> Dict[str, Any]:
        """Fire a test alert across all channels to verify configuration.

        Returns:
            A results dict with per-channel success flags.
        """
        logger.info("Sending Guardian Ear test alert across all channels …")

        desktop_ok = self._fire_desktop(
            sound_class="TEST_SOUND",
            confidence=99.9,
            threat_score=0.0,
            threat_level="TEST",
            location="System Diagnostics",
        )

        email_ok: bool = False
        if self._email.is_configured():
            try:
                email_ok = self._email.send_test_email()
            except Exception as exc:
                logger.error("Test email failed: %s", exc)
        else:
            logger.info("Test email skipped — email not configured.")

        contact_record = self._fire_contact_escalation(
            sound_class="TEST_SOUND",
            confidence=99.9,
            threat_score=0.0,
            threat_level="TEST",
        )

        results = {
            "desktop": desktop_ok,
            "email": email_ok,
            "contact": contact_record.get("contact_status") == "NOTIFIED",
            "email_configured": self._email.is_configured(),
            "contact_configured": self.is_contact_configured(),
            "desktop_available": self._desktop.is_available(),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }

        logger.info("Test alert results: %s", results)
        return results

    # ─── Enterprise Expansion Stubs ───────────────────────────────────────────

    def _fire_teams_webhook(self, alert_data: Dict[str, Any]) -> bool:
        """Future Scope: Send adaptive card to Microsoft Teams."""
        # TODO: Implement requests.post(webhook_url, json=card_payload)
        return False

    def _fire_slack_webhook(self, alert_data: Dict[str, Any]) -> bool:
        """Future Scope: Send block kit message to Slack."""
        # TODO: Implement slack_sdk.WebClient.chat_postMessage
        return False

    def _fire_sms_gateway(self, alert_data: Dict[str, Any], phone: str) -> bool:
        """Future Scope: Send SMS via Twilio or MSG91."""
        # TODO: Implement Twilio Client
        return False

    def _fire_voice_call(self, alert_data: Dict[str, Any], phone: str) -> bool:
        """Future Scope: Trigger automated voice call via Google Voice / Twilio API."""
        # TODO: Implement TwiML voice synthesis
        return False

    # ─── Audit Logging ────────────────────────────────────────────────────────
    
    def _log_to_csv(self, record: Dict[str, Any]) -> None:
        """Save a persistent audit trail of the notification dispatch."""
        try:
            path = Path("alerts/notification_logs.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare row matching Enterprise Requirements
            row = {
                "timestamp": f"{record.get('date')} {record.get('timestamp')}",
                "sound_class": record.get("sound_class"),
                "confidence": record.get("confidence"),
                "threat_level": record.get("threat_level"),
                "location": record.get("location"),
                "notification_type": "|".join(record.get("channels_fired", [])),
                "recipient": record.get("recipient"),
                "delivery_status": record.get("delivery_status"),
                "open_set_status": record.get("open_set_status"),
                "vote_count": record.get("vote_count"),
            }
            
            file_exists = path.exists()
            with open(path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as exc:
            logger.error("Failed to write to notification_logs.csv: %s", exc)


# ─── Module-level factory ─────────────────────────────────────────────────────

def from_config(config: Optional[Dict[str, Any]] = None) -> EmergencyNotificationEngine:
    """Factory function that constructs an :class:`EmergencyNotificationEngine`.

    Reads ``configs/config.yaml`` automatically when *config* is ``None``.

    Args:
        config: Optional config dict (full project config or just the
            ``emergency`` sub-dict). When ``None``, the YAML is loaded
            from disk.

    Returns:
        A fully initialised :class:`EmergencyNotificationEngine`.

    Example::

        from src.notifications.emergency_engine import from_config
        engine = from_config()
        record = engine.dispatch('gun_shot', 93.5, 88.0, 'CRITICAL', 'parking_lot')
    """
    return EmergencyNotificationEngine(config=config)
