"""
src/inference/open_set.py
─────────────────────────
Open-Set Classification for Guardian Ear.

Standard softmax classifiers assign every input to one of the known classes
with a spuriously high confidence.  This module adds a gating layer that
inspects the full probability distribution and rejects inputs whose
confidence is too low OR whose entropy is too high.  Such inputs are labelled
'unknown' so the downstream threat engine does not act on them.

Architecture position:
    AudioFeatureExtractor → CNN Model (softmax) → OpenSetClassifier → ThreatEngine

Classes:
    OpenSetClassifier  – wraps a softmax output vector and applies rejection.

Constants:
    CLASS_NAMES  – ordered list of UrbanSound8K class labels (classID 0–9).
"""

from __future__ import annotations

import logging
import math
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

# ──────────────────────────────────────────────────────────────────────────────
# Default gate thresholds (can be overridden per-call)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.60
# Maximum entropy for a 10-class *uniform* distribution: log(10) ≈ 2.303.
# A threshold of 1.8 means we accept up to ~79 % of maximum possible entropy.
DEFAULT_ENTROPY_THRESHOLD: float = 1.8


class OpenSetClassifier:
    """Gates the output of a closed-set softmax classifier.

    The classifier operates in two stages:

    1. **Entropy check** – Shannon entropy is computed over the full
       probability vector.  A high-entropy (flat) distribution indicates the
       model is uncertain; the sample is rejected as *unknown*.

    2. **Confidence check** – The maximum probability (i.e. the predicted
       class score) must exceed a minimum threshold.  A low peak also leads
       to rejection.

    Either check failing causes the sample to be reported as unknown.

    Parameters
    ----------
    confidence_threshold : float, optional
        Minimum value of ``max(probs)`` required to accept a prediction.
        Defaults to :data:`DEFAULT_CONFIDENCE_THRESHOLD` (0.60).
    entropy_threshold : float, optional
        Maximum Shannon entropy (nats) allowed for a prediction to be
        accepted.  Defaults to :data:`DEFAULT_ENTROPY_THRESHOLD` (1.8).

    Examples
    --------
    >>> import numpy as np
    >>> clf = OpenSetClassifier()
    >>> probs = np.array([0.01]*9 + [0.91])   # high confidence → siren
    >>> result = clf.classify(probs)
    >>> result['class_name']
    'siren'
    >>> result['is_known']
    True
    """

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
    ) -> None:
        if not 0.0 < confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in (0, 1], got {confidence_threshold}"
            )
        if entropy_threshold <= 0.0:
            raise ValueError(
                f"entropy_threshold must be positive, got {entropy_threshold}"
            )

        self._confidence_threshold = confidence_threshold
        self._entropy_threshold = entropy_threshold

        logger.info(
            "OpenSetClassifier initialised | confidence_threshold=%.2f | "
            "entropy_threshold=%.3f",
            confidence_threshold,
            entropy_threshold,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def classify(
        self,
        probs: np.ndarray,
        confidence_threshold: Optional[float] = None,
        entropy_threshold: Optional[float] = None,
    ) -> dict:
        """Classify a softmax probability vector with open-set rejection.

        Parameters
        ----------
        probs : np.ndarray
            1-D array of length 10 containing softmax probabilities.  Values
            must be non-negative and approximately sum to 1.
        confidence_threshold : float, optional
            Per-call override for the instance-level confidence threshold.
        entropy_threshold : float, optional
            Per-call override for the instance-level entropy threshold.

        Returns
        -------
        dict
            A result dictionary with the following keys:

            ``class_name`` : str
                Predicted UrbanSound8K class label, or ``'unknown'`` if
                rejected.
            ``class_id`` : int
                Index of the winning class (0–9), or ``-1`` when unknown.
            ``confidence`` : float
                Value of ``max(probs)``, rounded to 4 decimal places.
            ``is_known`` : bool
                ``True`` if the prediction passes both gates.
            ``entropy`` : float
                Shannon entropy (nats) of the distribution, rounded to 4 dp.
            ``rejection_reason`` : str or None
                Human-readable rejection reason when ``is_known`` is
                ``False``, otherwise ``None``.

        Raises
        ------
        ValueError
            If *probs* is not a 1-D array with exactly 10 elements.
        """
        probs = self._validate_probs(probs)

        conf_thresh = (
            confidence_threshold
            if confidence_threshold is not None
            else self._confidence_threshold
        )
        ent_thresh = (
            entropy_threshold
            if entropy_threshold is not None
            else self._entropy_threshold
        )

        entropy: float = self._shannon_entropy(probs)
        confidence: float = float(np.max(probs))
        predicted_idx: int = int(np.argmax(probs))
        predicted_class: str = CLASS_NAMES[predicted_idx]

        rejection_reason: Optional[str] = None

        # Gate 1 – confidence check
        if confidence < conf_thresh:
            rejection_reason = (
                f"confidence {confidence:.4f} < threshold {conf_thresh:.4f}"
            )

        # Gate 2 – entropy check (evaluated independently; both can fail)
        if entropy > ent_thresh:
            ent_msg = f"entropy {entropy:.4f} > threshold {ent_thresh:.4f}"
            rejection_reason = (
                f"{rejection_reason} | {ent_msg}" if rejection_reason else ent_msg
            )

        is_known: bool = rejection_reason is None

        result: dict = {
            "class_name": predicted_class if is_known else "unknown",
            "class_id": predicted_idx if is_known else -1,
            "confidence": round(confidence, 4),
            "is_known": is_known,
            "entropy": round(entropy, 4),
            "rejection_reason": rejection_reason,
        }

        if is_known:
            logger.debug(
                "OpenSet ACCEPT | class=%s | confidence=%.4f | entropy=%.4f",
                result["class_name"],
                confidence,
                entropy,
            )
        else:
            logger.warning(
                "OpenSet REJECT | candidate=%s | reason=%s",
                predicted_class,
                rejection_reason,
            )

        return result

    def get_class_centroids_distance(
        self,
        probs: np.ndarray,
        centroids: np.ndarray,
    ) -> float:
        """Compute Mahalanobis distance from a probability vector to class centroids.

        .. note::
            This method is a **stub** reserved for a future Mahalanobis-based
            open-set detection layer.  When implemented, it will compute the
            minimum Mahalanobis distance between *probs* and each of the
            provided class centroids using the per-class inverse covariance
            matrices.  Samples whose minimum distance exceeds a fitted
            threshold will be flagged as *unknown*.

        Parameters
        ----------
        probs : np.ndarray
            1-D softmax probability vector of shape ``(num_classes,)``.
        centroids : np.ndarray
            2-D array of shape ``(num_classes, num_classes)`` where each row
            is the mean probability vector for a training-set class cluster.

        Returns
        -------
        float
            Minimum Mahalanobis distance to any centroid.  Currently always
            returns ``0.0`` (stub implementation).
        """
        logger.debug(
            "get_class_centroids_distance called (stub) | probs.shape=%s | "
            "centroids.shape=%s",
            getattr(probs, "shape", "?"),
            getattr(centroids, "shape", "?"),
        )
        return 0.0

    # ──────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────

    @property
    def confidence_threshold(self) -> float:
        """The instance-level confidence gate threshold."""
        return self._confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, value: float) -> None:
        if not 0.0 < value <= 1.0:
            raise ValueError(f"confidence_threshold must be in (0, 1], got {value}")
        self._confidence_threshold = value
        logger.info("confidence_threshold updated to %.4f", value)

    @property
    def entropy_threshold(self) -> float:
        """The instance-level entropy gate threshold."""
        return self._entropy_threshold

    @entropy_threshold.setter
    def entropy_threshold(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError(f"entropy_threshold must be positive, got {value}")
        self._entropy_threshold = value
        logger.info("entropy_threshold updated to %.4f", value)

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _shannon_entropy(probs: np.ndarray) -> float:
        """Compute Shannon entropy (nats) of a probability vector.

        Uses the numerically stable form ``H = -Σ p·log(p + ε)`` where
        ``ε = 1e-9`` prevents ``log(0)`` for zero-probability bins.

        Parameters
        ----------
        probs : np.ndarray
            Non-negative 1-D probability array.

        Returns
        -------
        float
            Shannon entropy in nats.
        """
        eps: float = 1e-9
        entropy: float = -float(np.sum(probs * np.log(probs + eps)))
        return entropy

    @staticmethod
    def _validate_probs(probs: np.ndarray) -> np.ndarray:
        """Validate and normalise a softmax probability vector.

        Parameters
        ----------
        probs : np.ndarray
            Input array.  Must be 1-D with exactly ``_NUM_CLASSES`` elements
            and non-negative values.

        Returns
        -------
        np.ndarray
            The input cast to ``float32`` and flattened.

        Raises
        ------
        ValueError
            On shape or value violations.
        """
        probs = np.asarray(probs, dtype=np.float32).ravel()
        if probs.ndim != 1 or probs.shape[0] != _NUM_CLASSES:
            raise ValueError(
                f"probs must be a 1-D array with {_NUM_CLASSES} elements, "
                f"got shape {probs.shape}"
            )
        if np.any(probs < 0.0):
            raise ValueError("probs must contain non-negative values only.")
        return probs

    # ──────────────────────────────────────────────────────────────────────
    # Dunder helpers
    # ──────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"OpenSetClassifier("
            f"confidence_threshold={self._confidence_threshold}, "
            f"entropy_threshold={self._entropy_threshold})"
        )
