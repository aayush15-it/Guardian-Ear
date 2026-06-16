"""
Guardian Ear — Production-Grade Async Real-Time Audio Detection Engine (v3).

Improvements over v2:
  - Adaptive gain normalization before feature extraction
  - Dynamic noise floor estimation (rolling percentile)
  - Dynamic RMS threshold (noise_floor * multiplier)
  - Confidence smoothing via rolling average window
  - Temporal voting — N consecutive detections before escalation
  - Detection history ring buffer (last 20 events) for dashboard timeline
  - Per-class persistence tracking for the event engine
  - False-positive reduction: single frames don't escalate
  - Speaker-to-microphone resilience (very low silence_rms_threshold)

Author: Guardian Ear Team
"""

import logging
import os
import json
import threading
import time
from collections import deque
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import librosa
import sounddevice as sd

try:
    from src.utils.logger import get_logger
    from src.features.audio_pipeline import AudioFeatureExtractor
    from src.threat_engine.tracker import TemporalPatternTracker
    from src.threat_engine.rules import ThreatAssessor, get_sound_mode, get_mode_description
except ImportError:
    logging.basicConfig(level=logging.INFO)
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Detection Event — for timeline and history
# ─────────────────────────────────────────────────────────────
class DetectionEvent:
    """Lightweight record of a single detection frame."""
    __slots__ = ('timestamp', 'class_name', 'confidence', 'threat_score',
                 'threat_level', 'mode', 'rms')

    def __init__(self, class_name: str, confidence: float, threat_score: float,
                 threat_level: str, mode: str, rms: float) -> None:
        self.timestamp: str = datetime.now().strftime('%H:%M:%S')
        self.class_name = class_name
        self.confidence = confidence
        self.threat_score = threat_score
        self.threat_level = threat_level
        self.mode = mode
        self.rms = rms

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'threat_score': self.threat_score,
            'threat_level': self.threat_level,
            'mode': self.mode,
            'rms': self.rms,
        }


# ─────────────────────────────────────────────────────────────
# Real-Time Detector v3
# ─────────────────────────────────────────────────────────────
class RealTimeDetector:
    """Async real-time audio classifier with ring-buffer capture (v3).

    The detector runs two threads:
        1. **Capture thread**: Continuously records audio into a ring buffer.
        2. **Inference thread**: Every *step_seconds*, extracts the latest
           *duration* seconds from the ring buffer and runs the CRNN model.

    New in v3:
        - Adaptive gain normalization per chunk
        - Dynamic noise floor + dynamic RMS silence threshold
        - Confidence smoothing (rolling average)
        - Temporal voting (require N consecutive hits before escalation)
        - Detection history (last 20 events) for dashboard timeline
        - Per-class consecutive counter for persistence tracking

    Attributes:
        model: Loaded TensorFlow/Keras model.
        extractor: AudioFeatureExtractor for DSP.
        assessor: ThreatAssessor for threat scoring.
        tracker: TemporalPatternTracker for temporal analysis.
    """

    CLASS_NAMES: List[str] = [
        'air_conditioner', 'car_horn', 'children_playing',
        'dog_bark', 'drilling', 'engine_idling',
        'gun_shot', 'jackhammer', 'siren', 'street_music',
    ]

    # ── Tuning constants ────────────────────────────────────
    # Silence: RMS must exceed noise_floor * this multiplier to run inference
    _SILENCE_SNR_MULTIPLIER: float = 2.0
    # Absolute minimum RMS floor — catches dead-mic silence
    _ABS_SILENCE_FLOOR: float = 0.0005
    # Noise floor estimation: rolling window of N frames
    _NOISE_FLOOR_WINDOW: int = 30
    # Confidence smoothing: average last N predictions
    _SMOOTH_WINDOW: int = 3
    # Temporal voting: must detect same class N times consecutively to escalate
    _VOTE_WINDOW: int = 3
    # Detection history: last N events exposed to dashboard
    _HISTORY_MAXLEN: int = 20

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model=None,
    ) -> None:
        """Initialise the real-time detector.

        Args:
            config: Dict loaded from configs/config.yaml.
            model: Pre-loaded Keras model (if None, loads from config path).
        """
        cfg = config or {}
        audio_cfg = cfg.get('audio', {})
        inf_cfg = cfg.get('inference', {})
        paths_cfg = cfg.get('paths', {})

        self.sample_rate: int = audio_cfg.get('sample_rate', 22050)
        self.duration: int = audio_cfg.get('duration', 3)
        self.n_samples: int = self.sample_rate * self.duration
        self.confidence_threshold: float = inf_cfg.get('confidence_threshold', 0.60)
        self.location: str = inf_cfg.get('location', 'unknown')
        self.step_seconds: float = inf_cfg.get('step_seconds', 1.0)

        # ── Ring buffer ─────────────────────────────────────
        buffer_seconds = max(self.duration * 2, 10)
        self._ring_buffer = deque(maxlen=self.sample_rate * buffer_seconds)
        self._buffer_lock = threading.Lock()   # guards _ring_buffer
        self._pred_lock = threading.Lock()     # guards latest_prediction + history

        # ── Noise floor estimation ───────────────────────────
        # Rolling window of recent RMS values to estimate the background noise floor.
        self._noise_floor: float = self._ABS_SILENCE_FLOOR
        self._noise_window = deque(maxlen=self._NOISE_FLOOR_WINDOW)

        # ── Open-Set Classifier ────────────────────────────
        try:
            from src.inference.open_set import from_config as _osc_from_config
            self.osc = _osc_from_config()
        except Exception:
            self.osc = None

        # The dynamic silence threshold is noise_floor * _SILENCE_SNR_MULTIPLIER.
        self._rms_history: deque = deque(maxlen=self._NOISE_FLOOR_WINDOW)
        self._noise_floor: float = self._ABS_SILENCE_FLOOR
        self.current_rms: float = 0.0

        # ── Confidence smoothing ─────────────────────────────
        # Rolling window of (class_name, confidence, all_probs) for averaging.
        self._pred_window: deque = deque(maxlen=self._SMOOTH_WINDOW)

        # ── Temporal voting ──────────────────────────────────
        # Consecutive detection counter per class for false-positive reduction.
        self._consecutive: Dict[str, int] = {}

        # ── Detection history (dashboard timeline) ───────────
        self._detection_history: deque = deque(maxlen=self._HISTORY_MAXLEN)

        # ── Components ───────────────────────────────────────
        self.extractor = AudioFeatureExtractor(cfg)
        self.tracker = TemporalPatternTracker()
        self.assessor = ThreatAssessor(cfg)
        self.latest_prediction: Optional[Dict[str, Any]] = None

        # ── Normalization values ─────────────────────────────
        self.X_min: Optional[float] = None
        self.X_max: Optional[float] = None
        self._load_normalization(paths_cfg.get('model_dir', 'model'))

        # ── Model ────────────────────────────────────────────
        self.model = model
        if self.model is None:
            self._load_model(paths_cfg.get('model_dir', 'model'))

        # ── Threading ────────────────────────────────────────
        self._stop_event = threading.Event()
        self._inference_thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.InputStream] = None

        self._chunk_count: int = 0

        logger.info(
            "RealTimeDetector v3 ready — sr=%d, dur=%ds, step=%.1fs, "
            "threshold=%.0f%%, location=%s",
            self.sample_rate, self.duration, self.step_seconds,
            self.confidence_threshold * 100, self.location,
        )

    # ── Model / normalization loading ───────────────────────
    def _load_model(self, model_dir: str) -> None:
        """Load the Keras model from disk."""
        model_path = os.path.join(model_dir, 'guardian_ear_model.h5')
        if not os.path.exists(model_path):
            fallback_path = os.path.join(model_dir, 'best_model.h5')
            if os.path.exists(fallback_path):
                model_path = fallback_path

        if not os.path.exists(model_path):
            logger.error("Model not found at %s", os.path.join(model_dir, 'guardian_ear_model.h5'))
            return
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path)
            logger.info("Model loaded: %s", model_path)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)

    def _load_normalization(self, model_dir: str) -> None:
        """Load min/max normalization values from training."""
        x_min_path = os.path.join(model_dir, 'X_min.npy')
        x_max_path = os.path.join(model_dir, 'X_max.npy')
        if os.path.exists(x_min_path) and os.path.exists(x_max_path):
            self.X_min = float(np.load(x_min_path)[0])
            self.X_max = float(np.load(x_max_path)[0])
            logger.info("Normalization loaded: min=%.2f, max=%.2f", self.X_min, self.X_max)

    # ── Audio capture ───────────────────────────────────────
    def _audio_callback(self, indata: np.ndarray, frames: int,
                        time_info: Any, status: Any) -> None:
        """SoundDevice callback — pushes samples into the ring buffer."""
        if status:
            logger.debug("Audio status: %s", status)
        with self._buffer_lock:
            self._ring_buffer.extend(indata[:, 0])

    def _get_latest_audio(self) -> Optional[np.ndarray]:
        """Extract the latest *duration* seconds from the ring buffer."""
        with self._buffer_lock:
            if len(self._ring_buffer) < self.n_samples:
                return None
            audio = np.array(list(self._ring_buffer))[-self.n_samples:]
        return audio.astype(np.float32)

    def get_latest_audio_samples(self, n_samples: int) -> np.ndarray:
        """Thread-safe access to raw ring buffer samples for visualization."""
        with self._buffer_lock:
            if len(self._ring_buffer) == 0:
                return np.zeros(n_samples, dtype=np.float32)
            samples = list(self._ring_buffer)[-n_samples:]
            if len(samples) < n_samples:
                return np.pad(np.array(samples, dtype=np.float32), (n_samples - len(samples), 0))
            return np.array(samples, dtype=np.float32)

    def get_latest_prediction(self) -> Optional[Dict[str, Any]]:
        """Thread-safe access to the latest inference result."""
        with self._pred_lock:
            return self.latest_prediction

    def get_detection_history(self) -> List[Dict[str, Any]]:
        """Return the last N detection events as a list of dicts (newest first)."""
        with self._pred_lock:
            return [e.to_dict() for e in reversed(self._detection_history)]

    def get_system_health(self) -> Dict[str, Any]:
        """Return real-time system health metrics for the dashboard."""
        return {
            'chunk_count': self._chunk_count,
            'noise_floor': round(self._noise_floor, 5),
            'current_rms': round(self.current_rms, 5),
            'dynamic_threshold': round(
                max(self._noise_floor * self._SILENCE_SNR_MULTIPLIER, self._ABS_SILENCE_FLOOR), 5
            ),
            'buffer_fill': len(self._ring_buffer),
            'buffer_max': self._ring_buffer.maxlen,
            'stream_active': self._stream is not None and self._stream.active,
        }

    # ── Adaptive gain normalization ─────────────────────────
    @staticmethod
    def _adaptive_gain(audio: np.ndarray, target_rms: float = 0.05) -> np.ndarray:
        """Normalize audio to a target RMS level.

        This compensates for room acoustics, microphone distance, and
        speaker-to-microphone capture volume differences. The model was
        trained on studio-quality recordings; real microphone audio is
        typically 10–40 dB quieter. Adaptive gain brings it to a
        comparable amplitude before feature extraction.

        Args:
            audio: Raw waveform.
            target_rms: Desired RMS level (0.05 ≈ typical dataset RMS).

        Returns:
            Gain-adjusted waveform, clipped to [-1, 1].
        """
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-9:
            return audio
        gain = target_rms / rms
        # Cap gain at 20× to avoid amplifying pure noise into false detections
        gain = min(gain, 20.0)
        return np.clip(audio * gain, -1.0, 1.0)

    # ── Feature extraction + prediction ─────────────────────
    def _extract_and_normalize(self, audio: np.ndarray) -> np.ndarray:
        """Extract features and normalize for model input."""
        features = self.extractor.extract_features(audio, self.sample_rate)
        if self.X_min is not None and self.X_max is not None:
            features = (features - self.X_min) / (self.X_max - self.X_min + 1e-8)
        else:
            f_min, f_max = features.min(), features.max()
            if f_max - f_min > 0:
                features = (features - f_min) / (f_max - f_min)
        return features[np.newaxis, ..., np.newaxis]  # (1, 180, 130, 1)

    def predict(self, audio: np.ndarray) -> Tuple[str, float, np.ndarray, str]:
        """Run model inference on an audio clip.

        Args:
            audio: 1-D waveform array.

        Returns:
            Tuple of (class_name, confidence, all_probabilities, sound_mode).
        """
        features = self._extract_and_normalize(audio)
        preds = self.model.predict(features, verbose=0)[0]
        class_id = int(np.argmax(preds))
        confidence = float(preds[class_id])
        class_name = self.CLASS_NAMES[class_id]
        mode = get_sound_mode(class_name)
        return class_name, confidence, preds, mode

    # ── Confidence smoothing ─────────────────────────────────
    def _smooth_prediction(
        self, class_name: str, confidence: float, all_probs: np.ndarray
    ) -> Tuple[str, float, np.ndarray]:
        """Average predictions over a rolling window to reduce noise.

        If all recent frames agree on the same class, confidence is
        boosted by their average. If frames disagree, the result is
        the mode class with averaged probabilities.

        Returns:
            Smoothed (class_name, confidence, all_probs).
        """
        self._pred_window.append((class_name, confidence, all_probs.copy()))

        if len(self._pred_window) < 2:
            return class_name, confidence, all_probs

        # Average probability vectors
        avg_probs = np.mean([p for _, _, p in self._pred_window], axis=0)
        best_id = int(np.argmax(avg_probs))
        best_conf = float(avg_probs[best_id])
        best_class = self.CLASS_NAMES[best_id]

        return best_class, best_conf, avg_probs

    # ── Temporal voting ──────────────────────────────────────
    def _update_vote(self, class_name: str) -> int:
        """Increment consecutive count for class_name; reset all others.

        Returns:
            Current consecutive count for class_name.
        """
        for k in list(self._consecutive.keys()):
            if k != class_name:
                self._consecutive[k] = 0
        self._consecutive[class_name] = self._consecutive.get(class_name, 0) + 1
        return self._consecutive[class_name]

    # ── Inference loop ──────────────────────────────────────
    def _inference_loop(self) -> None:
        """Continuously process audio from the ring buffer."""
        logger.info("Inference thread started — listening (v3)...")
        while not self._stop_event.is_set():
            try:
                audio = self._get_latest_audio()
                if audio is None:
                    time.sleep(0.1)
                    continue

                self._chunk_count += 1

                # ── Compute RMS and update noise floor ──────
                rms = float(np.sqrt(np.mean(audio ** 2)))
                self.current_rms = rms
                self._rms_history.append(rms)

                # Noise floor = 20th percentile of recent RMS values
                if len(self._rms_history) >= 5:
                    self._noise_floor = float(np.percentile(list(self._rms_history), 20))
                    self._noise_floor = max(self._noise_floor, self._ABS_SILENCE_FLOOR)

                dynamic_threshold = max(
                    self._noise_floor * self._SILENCE_SNR_MULTIPLIER,
                    self._ABS_SILENCE_FLOOR,
                )
                logger.debug(
                    "Chunk #%d: RMS=%.5f, noise_floor=%.5f, threshold=%.5f",
                    self._chunk_count, rms, self._noise_floor, dynamic_threshold,
                )

                # ── Silence detection ────────────────────────
                if rms < dynamic_threshold:
                    self._pred_window.clear()  # reset smoothing on silence
                    with self._pred_lock:
                        self.latest_prediction = {
                            'class_name': 'silence',
                            'confidence': 1.0,
                            'all_probs': np.zeros(len(self.CLASS_NAMES)),
                            'mode': 'NEUTRAL',
                            'timestamp': time.time(),
                            'alert': None,
                            'voted': False,
                            'consecutive': 0,
                            'rms': rms,
                            'noise_floor': self._noise_floor,
                        }
                    time.sleep(self.step_seconds)
                    continue

                # ── Adaptive gain ─────────────────────────────
                audio_normalized = self._adaptive_gain(audio)

                # ── Model inference ───────────────────────────
                class_name, confidence, all_probs, mode = self.predict(audio_normalized)

                # ── Open-Set Rejection ────────────────────────
                osc_result = None
                if self.osc:
                    osc_result = self.osc.classify(all_probs)
                    if not osc_result['is_known']:
                        class_name = 'unknown'
                        confidence = osc_result['confidence']
                        mode = osc_result['sound_mode']

                # ── Confidence smoothing ──────────────────────
                class_name, confidence, all_probs = self._smooth_prediction(
                    class_name, confidence, all_probs
                )


                # ── Temporal voting ───────────────────────────
                consecutive = self._update_vote(class_name)
                voted = consecutive >= self._VOTE_WINDOW

                # ── Alert generation (only when voted + above threshold) ─
                alert = None
                if confidence >= self.confidence_threshold and voted:
                    alert = self.assessor.generate_alert(
                        sound_class=class_name,
                        confidence=confidence,
                        location=self.location,
                        tracker=self.tracker,
                    )
                    self._display_result(class_name, confidence, all_probs, mode, alert)
                elif confidence >= self.confidence_threshold and not voted:
                    logger.info(
                        "Chunk #%d: %s (%.1f%%) — waiting for votes (%d/%d)",
                        self._chunk_count, class_name, confidence * 100,
                        consecutive, self._VOTE_WINDOW,
                    )
                else:
                    logger.info(
                        "Chunk #%d: %s (%.1f%%) — below threshold (%.0f%%)",
                        self._chunk_count, class_name, confidence * 100,
                        self.confidence_threshold * 100,
                    )

                # ── Compute threat score for dashboard display ─
                threat_score = 0.0
                threat_level = 'SAFE'
                if class_name != 'silence':
                    threat_score = self.assessor.calculate_threat_score(
                        class_name, confidence, self.location
                    )
                    threat_level = self.assessor.get_threat_level(threat_score)

                # ── Build detection event for history ─────────
                event = DetectionEvent(
                    class_name=class_name,
                    confidence=round(confidence * 100, 1),
                    threat_score=threat_score,
                    threat_level=threat_level,
                    mode=mode,
                    rms=round(rms, 5),
                )

                # ── Update shared state ────────────────────────
                with self._pred_lock:
                    self.latest_prediction = {
                        'class_name': class_name,
                        'confidence': confidence,
                        'all_probs': all_probs,
                        'mode': mode,
                        'timestamp': time.time(),
                        'alert': alert,
                        'voted': voted,
                        'consecutive': consecutive,
                        'rms': rms,
                        'noise_floor': self._noise_floor,
                        'threat_score': threat_score,
                        'threat_level': threat_level,
                        'osc_result': osc_result,
                    }
                    self._detection_history.append(event)


                time.sleep(self.step_seconds)

            except KeyboardInterrupt:
                break
            except Exception as exc:
                logger.error("Inference error: %s", exc, exc_info=True)
                time.sleep(1.0)

    def _display_result(
        self, class_name: str, confidence: float,
        all_probs: np.ndarray, mode: str, alert=None,
    ) -> None:
        """Print formatted detection result to console."""
        desc = get_mode_description(class_name)
        icon = '🚨' if mode == 'ALERT' else 'ℹ️' if mode == 'ASSISTIVE' else '👁'

        logger.info(
            "%s [%s] %s — %.1f%% | Threat: %.1f/100 [%s]",
            icon, mode, class_name, confidence * 100,
            alert.threat_score if alert else 0,
            alert.threat_level if alert else 'N/A',
        )

        top3 = np.argsort(all_probs)[::-1][:3]
        for idx in top3:
            bar = '█' * int(all_probs[idx] * 15)
            logger.info(
                "    %s %5.1f%% %s",
                self.CLASS_NAMES[idx].ljust(22),
                all_probs[idx] * 100, bar,
            )

    # ── Lifecycle ───────────────────────────────────────────
    def start(self) -> None:
        """Start microphone capture and inference threads."""
        if self.model is None:
            logger.error("Cannot start — model not loaded")
            return

        self._stop_event.clear()

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=1024,
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info("Microphone stream started (blocksize=1024)")

        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True,
        )
        self._inference_thread.start()
        logger.info("Inference thread started")

    def stop(self) -> None:
        """Gracefully stop capture and inference."""
        logger.info("Stopping real-time detector...")
        self._stop_event.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=5.0)
        logger.info("Detector stopped. Processed %d chunks.", self._chunk_count)

    # ── File simulation ─────────────────────────────────────
    def simulate_on_file(self, audio_path: str, location: Optional[str] = None):
        """Run detection on a saved audio file."""
        if not os.path.exists(audio_path):
            logger.error("File not found: %s", audio_path)
            return None

        loc = location or self.location
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
        logger.info("Simulating on %s (%.2fs @ %dHz)", audio_path, len(audio) / sr, sr)

        class_name, confidence, all_probs, mode = self.predict(audio)
        alert = self.assessor.generate_alert(
            sound_class=class_name,
            confidence=confidence,
            location=loc,
            tracker=self.tracker,
        )
        self._display_result(class_name, confidence, all_probs, mode, alert)
        return alert


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    try:
        import yaml
    except ModuleNotFoundError:
        import sys
        from pathlib import Path
        _root = Path(__file__).resolve().parents[2]
        _site = _root / 'guardian_env' / 'lib' / 'site-packages'
        if _site.exists():
            sys.path.insert(0, str(_site))
        import yaml

    parser = argparse.ArgumentParser(description='Guardian Ear — Real-Time Detection v3')
    parser.add_argument('--config', default='configs/config.yaml', help='Config file')
    parser.add_argument('--mode', choices=['realtime', 'file'], default='realtime')
    parser.add_argument('--file', type=str, help='Audio file for simulation mode')
    parser.add_argument('--location', type=str, default=None)
    args = parser.parse_args()

    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f) or {}

    detector = RealTimeDetector(config)

    if args.mode == 'file' and args.file:
        detector.simulate_on_file(args.file, args.location)
    else:
        try:
            detector.start()
            print("\n🎧 Guardian Ear v3 — Listening... Press Ctrl+C to stop\n")
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            detector.stop()
