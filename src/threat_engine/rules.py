"""
Guardian Ear — Threat Assessment Rules Engine.

Provides deterministic threat scoring, alert classification, and
O(1) append-only CSV logging. All configuration is injected via
a config dictionary—no hardcoded magic numbers.

Author: Guardian Ear Team
"""

import csv
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

import pandas as pd

try:
    from src.utils.logger import get_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

from src.threat_engine.tracker import TemporalPatternTracker, PatternSummary

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────
@dataclass
class AlertRecord:
    """Complete alert record ready for logging and UI display."""
    timestamp: str
    date: str
    time: str
    sound_class: str
    confidence: float
    location: str
    sound_mode: str
    description: str
    pattern_label: str
    pattern_score: float
    detection_count: int
    duration_seconds: float
    should_escalate: bool
    threat_score: float
    threat_level: str
    threat_color: str
    alert_raised: bool


# ─────────────────────────────────────────────────────────────
# Sound mode helpers
# ─────────────────────────────────────────────────────────────
_MODE_DESCRIPTIONS: Dict[str, str] = {
    'gun_shot': 'Potential weapon discharge detected — immediate response required',
    'siren': 'Emergency siren detected — verify source and respond',
    'jackhammer': 'Heavy impact sound detected — check for forced entry',
    'dog_bark': 'Dog barking detected — may indicate intruder or distress',
    'children_playing': 'Children activity detected — monitor if after hours',
    'air_conditioner': 'HVAC sound detected — check if running abnormally long',
    'street_music': 'Music or crowd noise detected — monitor noise levels',
    'engine_idling': 'Vehicle idling detected — check for suspicious activity',
    'car_horn': 'Vehicle horn detected — possible road incident',
    'drilling': 'Drilling sound detected — verify authorized work',
}


def get_sound_mode(
    sound_class: str,
    alert_sounds: Optional[list] = None,
    assistive_sounds: Optional[list] = None,
) -> str:
    """Classify a sound into ALERT / ASSISTIVE / NEUTRAL.

    Args:
        sound_class: Detected class name.
        alert_sounds: Override list of alert-class names.
        assistive_sounds: Override list of assistive-class names.

    Returns:
        Mode string.
    """
    alert_sounds = alert_sounds or ['gun_shot', 'siren', 'jackhammer']
    assistive_sounds = assistive_sounds or [
        'dog_bark', 'children_playing', 'air_conditioner',
        'street_music', 'engine_idling', 'car_horn', 'drilling',
    ]
    if sound_class in alert_sounds:
        return 'ALERT'
    if sound_class in assistive_sounds:
        return 'ASSISTIVE'
    return 'NEUTRAL'


def get_mode_description(sound_class: str) -> str:
    """Return a human-readable description for a detected sound."""
    return _MODE_DESCRIPTIONS.get(
        sound_class, 'Unknown sound detected — monitor situation',
    )


# ─────────────────────────────────────────────────────────────
# Threat Assessor
# ─────────────────────────────────────────────────────────────
class ThreatAssessor:
    """Computes threat scores, classifies levels, and logs alerts.

    Attributes:
        threshold: Minimum threat score to raise an alert.
        alerts_path: Directory for the alert CSV log.
    """

    THREAT_COLORS: Dict[str, str] = {
        'CRITICAL': '#FF0000',
        'HIGH': '#FF6600',
        'MEDIUM': '#FFAA00',
        'LOW': '#88CC00',
        'SAFE': '#00AA00',
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise from a config dictionary.

        Args:
            config: Dict loaded from configs/config.yaml.
        """
        te = (config or {}).get('threat_engine', {})
        self.threshold: int = te.get('threshold', 60)
        self.alerts_path: str = (config or {}).get('paths', {}).get('alerts', 'alerts')

        # Weights
        self.sound_weights: Dict[str, float] = te.get('sound_weights', {
            'gun_shot': 1.0, 'siren': 0.85, 'jackhammer': 0.60,
            'drilling': 0.55, 'engine_idling': 0.40, 'car_horn': 0.50,
            'children_playing': 0.15, 'dog_bark': 0.20,
            'street_music': 0.10, 'air_conditioner': 0.05,
        })
        self.location_weights: Dict[str, float] = te.get('location_weights', {
            'parking_lot': 1.0, 'hostel': 0.9, 'corridor': 0.85,
            'entrance': 0.8, 'library': 0.7, 'classroom': 0.6,
            'office': 0.55, 'cafeteria': 0.5, 'garden': 0.4,
            'unknown': 0.5,
        })

        # Formula coefficients
        formula = te.get('formula_weights', {})
        self.alpha: float = formula.get('confidence', 0.30)
        self.beta: float = formula.get('time', 0.20)
        self.gamma: float = formula.get('location', 0.15)
        self.delta: float = formula.get('sound', 0.15)
        self.epsilon: float = formula.get('pattern', 0.20)

        self.alert_sounds: list = te.get('alert_sounds', ['gun_shot', 'siren', 'jackhammer'])
        self.assistive_sounds: list = te.get('assistive_sounds', [
            'dog_bark', 'children_playing', 'air_conditioner',
            'street_music', 'engine_idling', 'car_horn', 'drilling',
        ])

        logger.info("ThreatAssessor initialised — threshold=%d", self.threshold)

    # ── Time weight ─────────────────────────────────────────
    @staticmethod
    def get_time_weight(hour: Optional[int] = None) -> float:
        """Return a risk multiplier based on the hour of day.

        Late night (0-5) → 1.0, night (21-23) → 0.8,
        morning (6-8) → 0.6, evening (17-20) → 0.5,
        daytime (9-16) → 0.3.
        """
        if hour is None:
            hour = datetime.now().hour
        if 0 <= hour < 6:
            return 1.0
        elif 6 <= hour < 9:
            return 0.6
        elif 9 <= hour < 17:
            return 0.3
        elif 17 <= hour < 21:
            return 0.5
        return 0.8

    # ── Threat score calculation ────────────────────────────
    def calculate_threat_score(
        self,
        sound_class: str,
        confidence: float,
        location: str = 'unknown',
        hour: Optional[int] = None,
        pattern_score: float = 0.0,
    ) -> float:
        """Compute a threat score in [0, 100].

        Formula (v2):
            TS = α·Confidence + β·TimeWeight + γ·LocationWeight
               + δ·SoundWeight + ε·PatternScore

        Args:
            sound_class: Detected class name.
            confidence: Model confidence in [0, 1].
            location: Deployment location key.
            hour: Hour of day (0-23). Defaults to current hour.
            pattern_score: Temporal pattern intensity in [0, 1].

        Returns:
            Threat score scaled to [0, 100].
        """
        sw = self.sound_weights.get(sound_class, 0.5)
        lw = self.location_weights.get(location, 0.5)
        tw = self.get_time_weight(hour)

        raw = (
            self.alpha * confidence
            + self.beta * tw
            + self.gamma * lw
            + self.delta * sw
            + self.epsilon * pattern_score
        )
        return round(raw * 100, 1)

    # ── Threat level ────────────────────────────────────────
    @staticmethod
    def get_threat_level(score: float) -> str:
        """Map a numeric threat score to a categorical level."""
        if score >= 80:
            return 'CRITICAL'
        if score >= 60:
            return 'HIGH'
        if score >= 40:
            return 'MEDIUM'
        if score >= 20:
            return 'LOW'
        return 'SAFE'

    @classmethod
    def get_threat_color(cls, level: str) -> str:
        """Return a hex colour code for a threat level."""
        return cls.THREAT_COLORS.get(level, '#888888')

    # ── Alert generation ────────────────────────────────────
    def generate_alert(
        self,
        sound_class: str,
        confidence: float,
        location: str = 'unknown',
        tracker: Optional[TemporalPatternTracker] = None,
        device_id: str = 'default',
        hour: Optional[int] = None,
    ) -> AlertRecord:
        """Evaluate a detection and produce an alert record.

        Steps:
            1. Record detection in tracker.
            2. Compute pattern analysis.
            3. Calculate threat score with escalation.
            4. Build and optionally persist the alert.

        Args:
            sound_class: Detected class name.
            confidence: Model confidence in [0, 1].
            location: Deployment location key.
            tracker: Optional TemporalPatternTracker instance.
            device_id: Sensor identifier.
            hour: Override hour of day.

        Returns:
            Populated AlertRecord dataclass.
        """
        # Pattern analysis
        if tracker is not None:
            tracker.add_detection(sound_class, device_id)
            pattern = tracker.get_pattern_summary(sound_class, device_id)
        else:
            pattern = PatternSummary(
                pattern_label='BRIEF_NORMAL',
                pattern_score=0.0,
                detection_count=1,
                duration_seconds=0.0,
                should_escalate=False,
            )

        mode = get_sound_mode(sound_class, self.alert_sounds, self.assistive_sounds)
        description = get_mode_description(sound_class)

        # Threat score
        threat_score = self.calculate_threat_score(
            sound_class, confidence, location, hour, pattern.pattern_score,
        )

        # Escalation bonus
        if pattern.should_escalate and threat_score < 80:
            threat_score = min(threat_score + 10, 100.0)

        threat_level = self.get_threat_level(threat_score)
        threat_color = self.get_threat_color(threat_level)
        alert_raised = threat_score >= self.threshold

        now = datetime.now()
        alert = AlertRecord(
            timestamp=now.strftime('%Y-%m-%d %H:%M:%S'),
            date=now.strftime('%Y-%m-%d'),
            time=now.strftime('%H:%M:%S'),
            sound_class=sound_class,
            confidence=round(confidence * 100, 1),
            location=location,
            sound_mode=mode,
            description=description,
            pattern_label=pattern.pattern_label,
            pattern_score=pattern.pattern_score,
            detection_count=pattern.detection_count,
            duration_seconds=pattern.duration_seconds,
            should_escalate=pattern.should_escalate,
            threat_score=threat_score,
            threat_level=threat_level,
            threat_color=threat_color,
            alert_raised=alert_raised,
        )

        if alert_raised:
            logger.warning(
                "ALERT RAISED — %s (%.1f%%) at %s → score=%.1f [%s]",
                sound_class, confidence * 100, location,
                threat_score, threat_level,
            )
            self.save_alert(alert)
        else:
            logger.info(
                "Detection — %s (%.1f%%) → score=%.1f [%s]",
                sound_class, confidence * 100, threat_score, threat_level,
            )

        return alert

    # ── O(1) CSV logging ────────────────────────────────────
    def save_alert(self, alert: AlertRecord) -> None:
        """Append an alert record to the CSV log in O(1).

        Args:
            alert: The AlertRecord to persist.
        """
        path = Path(self.alerts_path) / 'alert_log.csv'
        path.parent.mkdir(parents=True, exist_ok=True)

        alert_dict = asdict(alert)
        alert_dict.pop('threat_color', None)  # UI-only field

        file_exists = path.exists()
        try:
            with open(path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=alert_dict.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(alert_dict)
            logger.debug("Alert appended to %s", path)
        except OSError as exc:
            logger.error("Failed to write alert CSV: %s", exc)

    # ── History loading ─────────────────────────────────────
    def load_alert_history(self) -> pd.DataFrame:
        """Load the full alert history from CSV.

        Returns:
            DataFrame of past alerts, or empty DataFrame if none exist.
        """
        csv_path = os.path.join(self.alerts_path, 'alert_log.csv')
        if os.path.exists(csv_path):
            try:
                return pd.read_csv(csv_path)
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
                logger.warning("Corrupt alert log: %s", exc)
                return pd.DataFrame()
        return pd.DataFrame()
