"""
src/notifications/email_service.py
────────────────────────────────────
Email Alert Service for Guardian Ear.

Sends formatted HTML emergency alert emails via SMTP (Gmail by default).
Uses only Python stdlib: smtplib + email.mime.  No external dependencies.

Features:
  - STARTTLS encryption
  - Up to 3 automatic retries with 1-second back-off between attempts
  - Graceful failure — logs errors without raising exceptions to callers
  - HTML + plain-text alternative body (RFC 2046 multipart/alternative)

Usage::

    from src.notifications.email_service import EmailAlertService

    svc = EmailAlertService(
        sender_email='guardian@gmail.com',
        sender_password='app-password-here',
        recipient_email='admin@example.com',
    )
    if svc.is_configured():
        svc.send_alert(
            subject='CRITICAL: gun_shot detected',
            sound_class='gun_shot',
            confidence=94.2,
            threat_score=91,
            threat_level='CRITICAL',
            location='parking_lot',
            timestamp='22:15:03',
        )
"""

from __future__ import annotations

import logging
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

# ─── Logger ──────────────────────────────────────────────────────────────────
try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)


class EmailAlertService:
    """SMTP-based email alert sender for Guardian Ear emergencies.

    Args:
        sender_email:     Gmail (or other SMTP) address used to send alerts.
        sender_password:  App password or account password for the sender.
        recipient_email:  Address that receives the alert.
        smtp_host:        SMTP server hostname. Default: ``smtp.gmail.com``.
        smtp_port:        SMTP server port for STARTTLS. Default: ``587``.
        enabled:          Set ``False`` to silently skip sending (useful for testing).
    """

    # Maximum number of send attempts before giving up
    _MAX_RETRIES: int = 3
    # Seconds to wait between retry attempts
    _RETRY_BACKOFF: float = 1.0

    def __init__(
        self,
        sender_email: str = "",
        sender_password: str = "",
        recipient_email: str = "",
        smtp_host: str = "smtp.gmail.com",
        smtp_port: int = 587,
        enabled: bool = True,
    ) -> None:
        self.sender_email = sender_email.strip()
        self.sender_password = sender_password
        self.recipient_email = recipient_email.strip()
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.enabled = enabled

        logger.debug(
            "EmailAlertService initialised — sender=%s, recipient=%s, host=%s:%s, enabled=%s",
            self.sender_email or "(not set)",
            self.recipient_email or "(not set)",
            self.smtp_host,
            self.smtp_port,
            self.enabled,
        )

    # ─── Public API ───────────────────────────────────────────────────────────

    def is_configured(self) -> bool:
        """Return ``True`` only if all three address/credential fields are non-empty."""
        return bool(self.sender_email and self.sender_password and self.recipient_email)

    def send_alert(
        self,
        subject: str,
        sound_class: str,
        confidence: float,
        threat_score: float,
        threat_level: str,
        location: str,
        timestamp: str,
    ) -> bool:
        """Send a formatted Guardian Ear emergency alert email.

        Args:
            subject:      Email subject line.
            sound_class:  Detected sound label (e.g. ``'gun_shot'``).
            confidence:   Model confidence in percent (0-100).
            threat_score: Computed threat score (0-100).
            threat_level: Human-readable threat level (e.g. ``'CRITICAL'``).
            location:     Deployment location string.
            timestamp:    Detection time string (``HH:MM:SS`` format).

        Returns:
            ``True`` if the email was delivered successfully, ``False`` otherwise.
        """
        if not self.enabled:
            logger.info("EmailAlertService is disabled — skipping send.")
            return False

        if not self.is_configured():
            logger.warning(
                "Email alert skipped — sender/recipient credentials are not configured."
            )
            return False

        msg = self._build_message(
            subject=subject,
            sound_class=sound_class,
            confidence=confidence,
            threat_score=threat_score,
            threat_level=threat_level,
            location=location,
            timestamp=timestamp,
        )

        return self._send_with_retry(msg)

    def send_test_email(self) -> bool:
        """Send a test email to verify SMTP credentials and connectivity.

        Returns:
            ``True`` if the test email was delivered successfully.
        """
        logger.info("Sending Guardian Ear test email to %s …", self.recipient_email)
        return self.send_alert(
            subject="[TEST] Guardian Ear Email Alert System — OK",
            sound_class="TEST_SOUND",
            confidence=99.9,
            threat_score=0.0,
            threat_level="TEST",
            location="System Diagnostics",
            timestamp=time.strftime("%H:%M:%S"),
        )

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _build_message(
        self,
        subject: str,
        sound_class: str,
        confidence: float,
        threat_score: float,
        threat_level: str,
        location: str,
        timestamp: str,
    ) -> MIMEMultipart:
        """Construct a multipart/alternative MIME email message.

        Returns a ``MIMEMultipart`` object ready for delivery.
        """
        confidence_pct = f"{confidence:.1f}%"
        score_str = f"{threat_score:.0f}/100"

        # ── Plain-text body ───────────────────────────────────────────────────
        plain_body = (
            "Guardian Ear Emergency Alert\n"
            "============================\n\n"
            f"Sound Detected:   {sound_class}\n"
            f"Confidence:       {confidence_pct}\n"
            f"Threat Level:     {threat_level}\n"
            f"Threat Score:     {score_str}\n"
            f"Location:         {location}\n"
            f"Timestamp:        {timestamp}\n\n"
            "Immediate attention is recommended.\n\n"
            "--\n"
            "Guardian Ear - AI-Based Assistive Environmental Awareness System"
        )

        # ── HTML body ─────────────────────────────────────────────────────────
        level_color = {
            "CRITICAL": "#d32f2f",
            "HIGH": "#f57c00",
            "MEDIUM": "#fbc02d",
            "LOW": "#388e3c",
            "TEST": "#1976d2",
        }.get(threat_level.upper(), "#424242")

        html_body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Guardian Ear Alert</title>
</head>
<body style="margin:0;padding:0;font-family:Arial,Helvetica,sans-serif;background:#f5f5f5;">
  <table width="100%" cellpadding="0" cellspacing="0" style="padding:30px 0;">
    <tr>
      <td align="center">
        <table width="600" cellpadding="0" cellspacing="0"
               style="background:#ffffff;border-radius:8px;overflow:hidden;
                      box-shadow:0 2px 8px rgba(0,0,0,.15);">

          <!-- Header -->
          <tr>
            <td style="background:{level_color};padding:24px 32px;text-align:center;">
              <h1 style="margin:0;color:#ffffff;font-size:22px;letter-spacing:1px;">
                &#128680; Guardian Ear Emergency Alert
              </h1>
            </td>
          </tr>

          <!-- Body -->
          <tr>
            <td style="padding:32px;">
              <table width="100%" cellpadding="8" cellspacing="0"
                     style="border-collapse:collapse;font-size:15px;">
                <tr style="border-bottom:1px solid #e0e0e0;">
                  <td style="color:#757575;width:45%;">Sound Detected</td>
                  <td style="font-weight:bold;color:#212121;">{sound_class}</td>
                </tr>
                <tr style="border-bottom:1px solid #e0e0e0;">
                  <td style="color:#757575;">Confidence</td>
                  <td style="font-weight:bold;color:#212121;">{confidence_pct}</td>
                </tr>
                <tr style="border-bottom:1px solid #e0e0e0;">
                  <td style="color:#757575;">Threat Level</td>
                  <td style="font-weight:bold;color:{level_color};">{threat_level}</td>
                </tr>
                <tr style="border-bottom:1px solid #e0e0e0;">
                  <td style="color:#757575;">Threat Score</td>
                  <td style="font-weight:bold;color:#212121;">{score_str}</td>
                </tr>
                <tr style="border-bottom:1px solid #e0e0e0;">
                  <td style="color:#757575;">Location</td>
                  <td style="font-weight:bold;color:#212121;">{location}</td>
                </tr>
                <tr>
                  <td style="color:#757575;">Timestamp</td>
                  <td style="font-weight:bold;color:#212121;">{timestamp}</td>
                </tr>
              </table>

              <p style="margin:24px 0 0;padding:16px;background:#fff3e0;
                         border-left:4px solid {level_color};border-radius:4px;
                         font-size:14px;color:#e65100;">
                <strong>Immediate attention is recommended.</strong>
              </p>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="background:#fafafa;padding:16px 32px;
                        border-top:1px solid #e0e0e0;text-align:center;
                        font-size:12px;color:#9e9e9e;">
              Guardian Ear &mdash; AI-Based Assistive Environmental Awareness System
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""

        # ── Assemble MIME message ─────────────────────────────────────────────
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"Guardian Ear <{self.sender_email}>"
        msg["To"] = self.recipient_email

        msg.attach(MIMEText(plain_body, "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html", "utf-8"))
        return msg

    def _send_with_retry(self, msg: MIMEMultipart) -> bool:
        """Attempt SMTP delivery up to ``_MAX_RETRIES`` times.

        Args:
            msg: Pre-built MIME message to deliver.

        Returns:
            ``True`` on first successful delivery; ``False`` if all attempts fail.
        """
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                logger.info(
                    "SMTP delivery attempt %d/%d to %s via %s:%s …",
                    attempt,
                    self._MAX_RETRIES,
                    self.recipient_email,
                    self.smtp_host,
                    self.smtp_port,
                )
                with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as server:
                    server.ehlo()
                    server.starttls()
                    server.ehlo()
                    server.login(self.sender_email, self.sender_password)
                    server.sendmail(
                        self.sender_email,
                        self.recipient_email,
                        msg.as_string(),
                    )
                logger.info(
                    "Email alert delivered successfully to %s (attempt %d).",
                    self.recipient_email,
                    attempt,
                )
                return True

            except smtplib.SMTPAuthenticationError as exc:
                # Wrong credentials — no point retrying
                logger.error(
                    "SMTP authentication failed — check sender credentials. "
                    "For Gmail, use an App Password: %s",
                    exc,
                )
                return False

            except Exception as exc:
                logger.warning(
                    "Email delivery attempt %d/%d failed: %s",
                    attempt,
                    self._MAX_RETRIES,
                    exc,
                )
                if attempt < self._MAX_RETRIES:
                    time.sleep(self._RETRY_BACKOFF)

        logger.error(
            "Email alert to %s failed after %d attempts — giving up.",
            self.recipient_email,
            self._MAX_RETRIES,
        )
        return False
