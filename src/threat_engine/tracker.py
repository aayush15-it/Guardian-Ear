"""
Guardian Ear — Thread-Safe Temporal Pattern Tracker.

Monitors sound detection history over sliding time windows to identify
sustained, repetitive, or abnormal acoustic patterns. Supports
multi-device deployments and exponential decay weighting.

Author: Guardian Ear Team
"""

import threading
import time
import math
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from src.utils.logger import get_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────
@dataclass
class PatternSummary:
    """Immutable snapshot of a temporal pattern analysis."""
    pattern_label: str
    pattern_score: float
    detection_count: int
    duration_seconds: float
    should_escalate: bool


# ─────────────────────────────────────────────────────────────
# Thread-Safe Temporal Pattern Tracker
# ─────────────────────────────────────────────────────────────
class TemporalPatternTracker:
    """Tracks detection history per (sound_class, device_id) pair.

    All public methods are protected by a reentrant lock so the tracker
    can be shared across inference threads safely.

    Attributes:
        window_size: Sliding window in seconds (default 600 = 10 min).
        max_history: Upper bound on per-key deque length to cap memory.
    """

    # Default thresholds (can be overridden via config)
    DEFAULT_DURATION_THRESHOLDS: Dict[str, int] = {
        'dog_bark': 180, 'engine_idling': 600, 'drilling': 900,
        'children_playing': 1800, 'street_music': 600,
        'air_conditioner': 1800, 'car_horn': 60, 'siren': 120,
        'gun_shot': 0, 'jackhammer': 300,
    }
    DEFAULT_REPETITION_THRESHOLDS: Dict[str, int] = {
        'dog_bark': 8, 'car_horn': 5, 'gun_shot': 2,
        'siren': 3, 'children_playing': 20, 'drilling': 10,
        'jackhammer': 8,
    }

    def __init__(
        self,
        window_size: int = 600,
        max_history: int = 500,
        duration_thresholds: Optional[Dict[str, int]] = None,
        repetition_thresholds: Optional[Dict[str, int]] = None,
        decay_half_life: float = 120.0,
    ) -> None:
        """Initialise the tracker.

        Args:
            window_size: Sliding window in seconds.
            max_history: Maximum entries kept per (class, device).
            duration_thresholds: Override default duration thresholds.
            repetition_thresholds: Override default repetition thresholds.
            decay_half_life: Half-life in seconds for exponential decay
                            weighting of older detections.
        """
        self._lock = threading.RLock()
        self.window_size = window_size
        self.max_history = max_history
        self.decay_half_life = decay_half_life

        self.duration_thresholds = (
            duration_thresholds or self.DEFAULT_DURATION_THRESHOLDS.copy()
        )
        self.repetition_thresholds = (
            repetition_thresholds or self.DEFAULT_REPETITION_THRESHOLDS.copy()
        )

        # key = (sound_class, device_id), value = deque of timestamps
        self._history: Dict[Tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=max_history)
        )

    # ── Recording ───────────────────────────────────────────
    def add_detection(
        self,
        sound_class: str,
        device_id: str = 'default',
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a new detection event.

        Args:
            sound_class: Detected class name.
            device_id: Identifier for the audio source / sensor.
            timestamp: Unix epoch; defaults to *time.time()*.
        """
        ts = timestamp or time.time()
        key = (sound_class, device_id)
        with self._lock:
            self._history[key].append(ts)
            self._evict_stale(key)

    # ── Queries ─────────────────────────────────────────────
    def get_detection_count(
        self, sound_class: str, device_id: str = 'default',
    ) -> int:
        """Return the number of detections within the window."""
        key = (sound_class, device_id)
        with self._lock:
            self._evict_stale(key)
            return len(self._history[key])

    def get_duration(
        self, sound_class: str, device_id: str = 'default',
    ) -> float:
        """Return seconds elapsed since first detection in the window."""
        key = (sound_class, device_id)
        with self._lock:
            self._evict_stale(key)
            history = self._history[key]
            if not history:
                return 0.0
            return time.time() - history[0]

    def get_pattern_score(
        self, sound_class: str, device_id: str = 'default',
    ) -> float:
        """Compute a pattern intensity score in [0, 1].

        Combines repetition and duration analysis with exponential
        decay weighting of older events.
        """
        key = (sound_class, device_id)
        with self._lock:
            self._evict_stale(key)
            history = self._history[key]
            if not history:
                return 0.0

            now = time.time()

            # Repetition score (decay-weighted count)
            weighted_count = sum(
                math.exp(-0.693 * (now - t) / max(self.decay_half_life, 1))
                for t in history
            )
            rep_thresh = self.repetition_thresholds.get(sound_class, 10)
            rep_score = min(weighted_count / rep_thresh, 1.0)

            # Duration score
            duration = now - history[0]
            dur_thresh = self.duration_thresholds.get(sound_class, 300)
            dur_score = 1.0 if dur_thresh == 0 else min(duration / dur_thresh, 1.0)

            return round(max(rep_score, dur_score), 3)

    def get_pattern_summary(
        self, sound_class: str, device_id: str = 'default',
    ) -> PatternSummary:
        """Return a full pattern analysis snapshot."""
        score = self.get_pattern_score(sound_class, device_id)
        count = self.get_detection_count(sound_class, device_id)
        duration = self.get_duration(sound_class, device_id)

        if score >= 0.8:
            label = 'SUSTAINED_DISTRESS'
        elif score >= 0.5:
            label = 'PROLONGED_ABNORMAL'
        elif score >= 0.2:
            label = 'REPETITIVE_PATTERN'
        else:
            label = 'BRIEF_NORMAL'

        should_escalate = label in ('SUSTAINED_DISTRESS', 'PROLONGED_ABNORMAL')

        return PatternSummary(
            pattern_label=label,
            pattern_score=score,
            detection_count=count,
            duration_seconds=round(duration, 1),
            should_escalate=should_escalate,
        )

    def get_all_active_patterns(
        self, device_id: str = 'default',
    ) -> Dict[str, PatternSummary]:
        """Return pattern summaries for all classes with active detections."""
        result: Dict[str, PatternSummary] = {}
        with self._lock:
            for (cls, dev), history in list(self._history.items()):
                if dev != device_id or not history:
                    continue
                result[cls] = self.get_pattern_summary(cls, dev)
        return result

    # ── Maintenance ─────────────────────────────────────────
    def clear_stale_patterns(self, max_age_seconds: float = 1800.0) -> int:
        """Remove all entries older than *max_age_seconds*.

        Returns:
            Number of keys fully cleared.
        """
        cutoff = time.time() - max_age_seconds
        cleared = 0
        with self._lock:
            for key in list(self._history.keys()):
                self._history[key] = deque(
                    (t for t in self._history[key] if t > cutoff),
                    maxlen=self.max_history,
                )
                if not self._history[key]:
                    del self._history[key]
                    cleared += 1
        logger.debug("Cleared %d stale pattern keys", cleared)
        return cleared

    def reset(
        self,
        sound_class: Optional[str] = None,
        device_id: str = 'default',
    ) -> None:
        """Reset history for one class or all classes on a device."""
        with self._lock:
            if sound_class:
                key = (sound_class, device_id)
                if key in self._history:
                    self._history[key].clear()
            else:
                keys_to_remove = [
                    k for k in self._history if k[1] == device_id
                ]
                for k in keys_to_remove:
                    del self._history[k]

    # ── Internal ────────────────────────────────────────────
    def _evict_stale(self, key: Tuple[str, str]) -> None:
        """Remove entries outside the sliding window (caller holds lock)."""
        cutoff = time.time() - self.window_size
        dq = self._history[key]
        while dq and dq[0] < cutoff:
            dq.popleft()
