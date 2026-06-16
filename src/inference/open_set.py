"""
src/inference/open_set.py
─────────────────────────
Open-Set Classification for Guardian Ear.

Standard softmax classifiers assign every input to one of the known classes
even when the sound was never in the training set (e.g. tap water, microwave
beep, white noise).  This module adds a three-gate rejection layer that
inspects the full probability distribution and marks uncertain inputs as
'unknown' — displayed as MONITORING mode in the dashboard — so the downstream
threat engine never fires on them.

Architecture position:
    AudioFeatureExtractor → CRNN Model (softmax) → OpenSetClassifier → ThreatEngine

Three rejection gates (any failure = MONITORING):
    1. Confidence gate   — max(probs) must exceed confidence_threshold
    2. Margin gate       — gap between top-1 and top-2 must exceed margin_threshold
    3. Entropy gate      — Shannon entropy (nats) must not exceed entropy_threshold

All thresholds are configurable from configs/config.yaml under [open_set]:
    open_set:
      confidence_threshold: 0.70
      margin_threshold: 0.20
      entropy_threshold: 1.80
      enabled: true

Real-world example — tap water through a dog_bark model:
    Model output: dog_bark=0.57, air_conditioner=0.19, engine_idling=0.14 ...
    Gate 1: 0.57 < 0.70 → FAIL → MONITORING (no false dog_bark alert fired)

Classes:
    OpenSetClassifier  – wraps a softmax output vector and applies rejection.
    from_config()      – factory that reads thresholds from configs/config.yaml.

Constants:
    CLASS_NAMES  – ordered list of UrbanSound8K class labels (classID 0–9).
    UNKNOWN_MODE – sound_mode string shown for rejected predictions ("MONITORING").
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Class labels (UrbanSound8K, ordered by classID 0 → 9)
# Must match the label ordering used during model training.
# ──────────────────────────────────────────────────────────────────────────────
CLASS_NAMES: list[str] = [
    "air_conditioner",   # classID 0
    "car_horn",          # classID 1
    "children_playing",  # classID 2
    "dog_bark",          # classID 3
    "drilling",          # classID 4
    "engine_idling",     # classID 5
    "gun_shot",          # classID 6
    "jackhammer",        # classID 7
    "siren",             # classID 8
    "street_music",      # classID 9
]

_NUM_CLASSES: int = len(CLASS_NAMES)

# The sound_mode string shown in the dashboard for rejected predictions.
# "MONITORING" is more professional than "UNKNOWN" during demos and clearly
# communicates that the system is watching but not falsely alarming.
UNKNOWN_MODE: str = "MONITORING"

# ──────────────────────────────────────────────────────────────────────────────
# Hard-coded defaults (overridden by config.yaml or constructor args)
# ──────────────────────────────────────────────────────────────────────────────
_DEFAULT_CONFIDENCE: float = 0.70
_DEFAULT_MARGIN: float = 0.20
# Maximum entropy for a 10-class *uniform* distribution: log(10) ≈ 2.303.
# 1.80 means we reject anything with entropy above ~78% of the maximum.
_DEFAULT_ENTROPY: float = 1.80


# ──────────────────────────────────────────────────────────────────────────────
# Factory — reads from config.yaml automatically
# ──────────────────────────────────────────────────────────────────────────────
def from_config() -> "OpenSetClassifier":
    """Create an OpenSetClassifier using thresholds from configs/config.yaml.

    Falls back to built-in defaults if the config file is unavailable.

    Returns:
        OpenSetClassifier with thresholds read from the ``open_set`` section.

    Example:
        >>> clf = from_config()  # reads open_set.confidence_threshold etc.
    """
    try:
        from src.utils.config_loader import load_config
        cfg = load_config().get("open_set", {})
        conf = float(cfg.get("confidence_threshold", _DEFAULT_CONFIDENCE))
        margin = float(cfg.get("margin_threshold", _DEFAULT_MARGIN))
        entropy = float(cfg.get("entropy_threshold", _DEFAULT_ENTROPY))
        enabled = bool(cfg.get("enabled", True))
    except Exception as exc:
        logger.warning("Could not load open_set config — using defaults: %s", exc)
        conf, margin, entropy, enabled = _DEFAULT_CONFIDENCE, _DEFAULT_MARGIN, _DEFAULT_ENTROPY, True

    logger.info(
        "OpenSetClassifier from_config | conf=%.2f | margin=%.2f | entropy=%.3f | enabled=%s",
        conf, margin, entropy, enabled,
    )
    return OpenSetClassifier(
        confidence_threshold=conf,
        margin_threshold=margin,
        entropy_threshold=entropy,
        enabled=enabled,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main classifier
# ──────────────────────────────────────────────────────────────────────────────
class OpenSetClassifier:
    """Gates the output of a closed-set softmax classifier.

    Applies three sequential rejection gates.  Any gate failure causes the
    sample to be reported as *unknown* / MONITORING rather than being assigned
    to the nearest (but wrong) known class.

    Gates
    -----
    1. **Confidence gate** — ``max(probs) >= confidence_threshold``
       Low peak = model is not sure which class it is.
       *Example*: tap water → dog_bark 57 % → REJECT (< 70 %)

    2. **Margin gate** — ``top1 - top2 >= margin_threshold``
       Small margin = two classes are almost equally likely = ambiguous.
       *Example*: dog_bark 41 %, engine_idling 38 % → REJECT (margin 3 %)

    3. **Entropy gate** — ``entropy(probs) <= entropy_threshold``
       High entropy = probability mass spread across many classes = confused.
       *Example*: flat distribution [0.11 × 9, 0.01] → REJECT

    Parameters
    ----------
    confidence_threshold : float
        Minimum ``max(probs)`` to accept.  Default 0.70.
    margin_threshold : float
        Minimum ``top1 − top2`` to accept.  Default 0.20.
    entropy_threshold : float
        Maximum Shannon entropy (nats) to accept.  Default 1.80.
    enabled : bool
        If ``False``, all inputs pass through without rejection (bypass mode).

    Examples
    --------
    >>> import numpy as np
    >>> clf = OpenSetClassifier()
    >>> # High-confidence gun_shot
    >>> p = np.zeros(10); p[6] = 0.95; p[8] = 0.03; p[0] = 0.02
    >>> r = clf.classify(p)
    >>> r['class_name'], r['is_known'], r['sound_mode']
    ('gun_shot', True, 'ALERT')
    >>> # Low-confidence ambiguous tap water
    >>> p2 = np.full(10, 0.06); p2[3] = 0.57; p2[5] = 0.37
    >>> r2 = clf.classify(p2)
    >>> r2['class_name'], r2['is_known'], r2['sound_mode']
    ('unknown', False, 'MONITORING')
    """

    def __init__(
        self,
        confidence_threshold: float = _DEFAULT_CONFIDENCE,
        margin_threshold: float = _DEFAULT_MARGIN,
        entropy_threshold: float = _DEFAULT_ENTROPY,
        enabled: bool = True,
    ) -> None:
        if not 0.0 < confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold must be in (0, 1], got {confidence_threshold}")
        if not 0.0 <= margin_threshold < 1.0:
            raise ValueError(f"margin_threshold must be in [0, 1), got {margin_threshold}")
        if entropy_threshold <= 0.0:
            raise ValueError(f"entropy_threshold must be positive, got {entropy_threshold}")

        self._confidence_threshold = confidence_threshold
        self._margin_threshold = margin_threshold
        self._entropy_threshold = entropy_threshold
        self._enabled = enabled

        logger.info(
            "OpenSetClassifier ready | conf_thresh=%.2f | margin_thresh=%.2f | "
            "entropy_thresh=%.3f | enabled=%s",
            confidence_threshold, margin_threshold, entropy_threshold, enabled,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def classify(
        self,
        probs: np.ndarray,
        confidence_threshold: Optional[float] = None,
        margin_threshold: Optional[float] = None,
        entropy_threshold: Optional[float] = None,
    ) -> dict:
        """Classify a softmax probability vector with open-set rejection.

        Parameters
        ----------
        probs : np.ndarray
            1-D array of length 10 with softmax probabilities (must sum ≈ 1).
        confidence_threshold / margin_threshold / entropy_threshold : float, optional
            Per-call overrides for the instance-level thresholds.

        Returns
        -------
        dict with keys:

            ``class_name``       str   — Predicted label or ``'unknown'``
            ``class_id``         int   — 0–9 or ``-1`` when unknown
            ``confidence``       float — max(probs), rounded to 4 dp
            ``top2_margin``      float — top1 − top2, rounded to 4 dp
            ``entropy``          float — Shannon entropy (nats), rounded to 4 dp
            ``is_known``         bool  — True if all gates passed
            ``sound_mode``       str   — mode string for dashboard display
            ``rejection_reason`` str|None — human-readable gate failure
        """
        probs = self._validate_probs(probs)

        # Per-call threshold overrides
        c_thresh = confidence_threshold if confidence_threshold is not None else self._confidence_threshold
        m_thresh = margin_threshold if margin_threshold is not None else self._margin_threshold
        e_thresh = entropy_threshold if entropy_threshold is not None else self._entropy_threshold

        # Raw statistics
        sorted_probs = np.sort(probs)[::-1]
        confidence: float = float(sorted_probs[0])
        top2_margin: float = float(sorted_probs[0] - sorted_probs[1])
        entropy: float = self._shannon_entropy(probs)
        predicted_idx: int = int(np.argmax(probs))
        predicted_class: str = CLASS_NAMES[predicted_idx]

        # Gate evaluation (if disabled, skip all gates)
        rejection_reasons: list[str] = []
        if self._enabled:
            # Gate 1 — confidence
            if confidence < c_thresh:
                rejection_reasons.append(
                    f"low confidence {confidence*100:.1f}% < {c_thresh*100:.0f}%"
                )
            # Gate 2 — margin
            if top2_margin < m_thresh:
                rejection_reasons.append(
                    f"ambiguous margin {top2_margin*100:.1f}% < {m_thresh*100:.0f}%"
                )
            # Gate 3 — entropy
            if entropy > e_thresh:
                rejection_reasons.append(
                    f"high entropy {entropy:.3f} > {e_thresh:.3f}"
                )

        is_known: bool = len(rejection_reasons) == 0
        rejection_reason: Optional[str] = " | ".join(rejection_reasons) if rejection_reasons else None

        # Determine sound_mode for dashboard display
        if is_known:
            from src.threat_engine.rules import get_sound_mode
            try:
                sound_mode = get_sound_mode(predicted_class)
            except Exception:
                sound_mode = "NEUTRAL"
        else:
            sound_mode = UNKNOWN_MODE  # "MONITORING"

        result = {
            "class_name": predicted_class if is_known else "unknown",
            "class_id": predicted_idx if is_known else -1,
            "confidence": round(confidence, 4),
            "top2_margin": round(top2_margin, 4),
            "entropy": round(entropy, 4),
            "is_known": is_known,
            "sound_mode": sound_mode,
            "rejection_reason": rejection_reason,
        }

        if is_known:
            logger.debug(
                "OpenSet ACCEPT | class=%s | conf=%.3f | margin=%.3f | entropy=%.3f",
                predicted_class, confidence, top2_margin, entropy,
            )
        else:
            logger.warning(
                "OpenSet REJECT | candidate=%s | conf=%.3f | reason=[%s]",
                predicted_class, confidence, rejection_reason,
            )

        return result

    def get_class_centroids_distance(
        self,
        probs: np.ndarray,
        centroids: np.ndarray,
    ) -> float:
        """Stub for future Mahalanobis-based open-set detection.

        When implemented, computes the minimum Mahalanobis distance between
        *probs* and each class centroid in *centroids*.  Samples above a
        fitted threshold will be flagged as unknown.

        Currently always returns 0.0 (not yet implemented).
        """
        logger.debug("get_class_centroids_distance (stub) called")
        return 0.0

    # ──────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, value: float) -> None:
        if not 0.0 < value <= 1.0:
            raise ValueError(f"confidence_threshold must be in (0, 1], got {value}")
        self._confidence_threshold = value

    @property
    def margin_threshold(self) -> float:
        return self._margin_threshold

    @margin_threshold.setter
    def margin_threshold(self, value: float) -> None:
        if not 0.0 <= value < 1.0:
            raise ValueError(f"margin_threshold must be in [0, 1), got {value}")
        self._margin_threshold = value

    @property
    def entropy_threshold(self) -> float:
        return self._entropy_threshold

    @entropy_threshold.setter
    def entropy_threshold(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError(f"entropy_threshold must be positive, got {value}")
        self._entropy_threshold = value

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = bool(value)
        logger.info("OpenSetClassifier enabled=%s", self._enabled)

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _shannon_entropy(probs: np.ndarray) -> float:
        """Shannon entropy H = -Σ p·log(p + ε) in nats."""
        return float(-np.sum(probs * np.log(probs + 1e-9)))

    @staticmethod
    def _validate_probs(probs: np.ndarray) -> np.ndarray:
        """Validate and return a flattened float32 probability vector."""
        probs = np.asarray(probs, dtype=np.float32).ravel()
        if probs.ndim != 1 or probs.shape[0] != _NUM_CLASSES:
            raise ValueError(
                f"probs must be a 1-D array with {_NUM_CLASSES} elements, got shape {probs.shape}"
            )
        if np.any(probs < 0.0):
            raise ValueError("probs must contain non-negative values only.")
        return probs

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"OpenSetClassifier("
            f"confidence_threshold={self._confidence_threshold}, "
            f"margin_threshold={self._margin_threshold}, "
            f"entropy_threshold={self._entropy_threshold}, "
            f"enabled={self._enabled})"
        )
