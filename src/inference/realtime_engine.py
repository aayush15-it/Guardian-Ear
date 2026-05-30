"""
Guardian Ear — Async Real-Time Audio Detection Engine.

Uses a ring buffer for non-blocking audio capture and overlapping
inference windows for low-latency classification. Decouples
microphone I/O from model inference via threading.

Author: Guardian Ear Team
"""

import logging
import os
import json
import threading
import time
from collections import deque
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
# Real-Time Detector
# ─────────────────────────────────────────────────────────────
class RealTimeDetector:
    """Async real-time audio classifier with ring-buffer capture.

    The detector runs two threads:
        1. **Capture thread**: Continuously records audio into a ring buffer.
        2. **Inference thread**: Every *step_seconds*, extracts the latest
           *duration* seconds from the ring buffer and runs the CRNN model.

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
        self.confidence_threshold: float = inf_cfg.get('confidence_threshold', 0.70)
        self.location: str = inf_cfg.get('location', 'unknown')
        self.step_seconds: float = inf_cfg.get('step_seconds', 1.0)

        # Ring buffer: holds latest N seconds of audio
        buffer_seconds = max(self.duration * 2, 10)
        self._ring_buffer = deque(maxlen=self.sample_rate * buffer_seconds)
        self._buffer_lock = threading.Lock()   # guards _ring_buffer
        self._pred_lock = threading.Lock()     # guards latest_prediction (separate to avoid contention)

        # Silence threshold: deliberately low so microphone-captured system
        # audio (played through speakers) is not incorrectly filtered as silence.
        self.silence_rms_threshold: float = 0.001

        # Components
        self.extractor = AudioFeatureExtractor(cfg)
        self.tracker = TemporalPatternTracker()
        self.assessor = ThreatAssessor(cfg)
        self.latest_prediction: Optional[Dict[str, Any]] = None

        # Normalization values
        self.X_min: Optional[float] = None
        self.X_max: Optional[float] = None
        self._load_normalization(paths_cfg.get('model_dir', 'model'))

        # Model
        self.model = model
        if self.model is None:
            self._load_model(paths_cfg.get('model_dir', 'model'))

        # Threading
        self._stop_event = threading.Event()
        self._inference_thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.InputStream] = None

        self._chunk_count: int = 0

        logger.info(
            "RealTimeDetector ready — sr=%d, dur=%ds, step=%.1fs, "
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
            # Take the last n_samples
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

    # ── Inference loop ──────────────────────────────────────
    def _inference_loop(self) -> None:
        """Continuously process audio from the ring buffer."""
        logger.info("Inference thread started — listening...")
        while not self._stop_event.is_set():
            try:
                audio = self._get_latest_audio()
                if audio is None:
                    time.sleep(0.1)
                    continue

                self._chunk_count += 1

                # Check for silence — deliberately low threshold (0.001) so that
                # audio played through speakers and captured via microphone is not
                # incorrectly silenced (system audio is quieter than direct load).
                rms = np.sqrt(np.mean(audio ** 2))
                logger.debug("Chunk #%d RMS=%.5f", self._chunk_count, rms)

                if rms < self.silence_rms_threshold:
                    with self._pred_lock:
                        self.latest_prediction = {
                            'class_name': 'silence',
                            'confidence': 1.0,
                            'all_probs': np.zeros(len(self.CLASS_NAMES)),
                            'mode': 'NEUTRAL',
                            'timestamp': time.time(),
                            'alert': None
                        }
                    time.sleep(self.step_seconds)
                    continue

                class_name, confidence, all_probs, mode = self.predict(audio)
                alert = None

                if confidence >= self.confidence_threshold:
                    alert = self.assessor.generate_alert(
                        sound_class=class_name,
                        confidence=confidence,
                        location=self.location,
                        tracker=self.tracker,
                    )
                    self._display_result(class_name, confidence, all_probs, mode, alert)
                else:
                    # Still log below-threshold detections at info level
                    logger.info(
                        "Chunk #%d: %s (%.1f%%) — below threshold (%.0f%%)",
                        self._chunk_count, class_name, confidence * 100,
                        self.confidence_threshold * 100,
                    )

                # ALWAYS update latest_prediction so the dashboard always
                # has fresh data to render, even for low-confidence results.
                with self._pred_lock:
                    self.latest_prediction = {
                        'class_name': class_name,
                        'confidence': confidence,
                        'all_probs': all_probs,
                        'mode': mode,
                        'timestamp': time.time(),
                        'alert': alert
                    }

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

        # Top-3 predictions
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

        # Start audio input stream
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=1024,
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info("Microphone stream started (blocksize=1024)")

        # Start inference thread
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True,
        )
        self._inference_thread.start()

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
        """Run detection on a saved audio file.

        Args:
            audio_path: Path to a .wav file.
            location: Override deployment location.

        Returns:
            AlertRecord or None.
        """
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
    import yaml

    parser = argparse.ArgumentParser(description='Guardian Ear — Real-Time Detection')
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
            print("\n🎧 Guardian Ear — Listening... Press Ctrl+C to stop\n")
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            detector.stop()
