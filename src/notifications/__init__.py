"""
Guardian Ear notification services.

Provides the Emergency Notification Engine that orchestrates
desktop notifications, email alerts, and emergency contact escalation.

Telegram is archived (kept in telegram_service.py for reference) but
no longer the primary notification mechanism.
"""
from .emergency_engine import EmergencyNotificationEngine, from_config
from .email_service import EmailAlertService
from .desktop_notifier import DesktopNotifier

__all__ = [
    'EmergencyNotificationEngine',
    'EmailAlertService',
    'DesktopNotifier',
    'from_config',
]
