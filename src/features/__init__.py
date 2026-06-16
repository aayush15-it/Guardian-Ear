"""Feature extraction sub-package for GuardianEar.

Exposes :class:`AudioFeatureExtractor` as the primary public API.
"""

from src.features.audio_pipeline import AudioFeatureExtractor

__all__ = ["AudioFeatureExtractor"]
