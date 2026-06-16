"""
Guardian Ear — Production-Grade Audio Feature Extraction Pipeline.

Extracts fused audio features (Mel Spectrogram + MFCC + Chroma STFT)
with SpecAugment, silence removal, and chunked HDF5 persistence to
avoid OOM on large datasets.

Author: Guardian Ear Team
"""

import os
import sys
import logging
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────
# Allow running both as a module (from src.utils...) and standalone
try:
    from src.utils.logger import get_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Audio Feature Extractor
# ─────────────────────────────────────────────────────────────
class AudioFeatureExtractor:
    """Production-grade audio feature extraction with augmentation.

    Attributes:
        sample_rate: Target sampling rate in Hz.
        duration: Target clip duration in seconds.
        n_mels: Number of Mel filter-bank bins.
        n_mfcc: Number of MFCC coefficients.
        target_time_frames: Fixed time-axis length after resize.
        n_samples: Total samples per clip (sample_rate * duration).
    """

    # ── Sound categories ────────────────────────────────────
    CLASS_NAMES: List[str] = [
        'air_conditioner', 'car_horn', 'children_playing',
        'dog_bark', 'drilling', 'engine_idling',
        'gun_shot', 'jackhammer', 'siren', 'street_music',
    ]
    ALERT_SOUNDS: List[str] = ['gun_shot', 'siren', 'jackhammer']
    ASSISTIVE_SOUNDS: List[str] = [
        'dog_bark', 'children_playing', 'air_conditioner',
        'street_music', 'engine_idling', 'car_horn', 'drilling',
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the extractor from a config dictionary.

        Args:
            config: Optional dict loaded from configs/config.yaml.
                    Falls back to sane defaults when *None*.
        """
        audio_cfg = (config or {}).get('audio', {})
        self.sample_rate: int = audio_cfg.get('sample_rate', 22050)
        self.duration: int = audio_cfg.get('duration', 3)
        self.n_mels: int = audio_cfg.get('n_mels', 128)
        self.n_mfcc: int = audio_cfg.get('n_mfcc', 40)
        self.target_time_frames: int = audio_cfg.get(
            'target_time_frames',
            audio_cfg.get('target_time_steps', 130),
        )
        self.n_samples: int = self.sample_rate * self.duration

        # SpecAugment parameters
        aug_cfg = (config or {}).get('augmentation', {})
        self.freq_mask_param: int = aug_cfg.get('freq_mask_param', 15)
        self.time_mask_param: int = aug_cfg.get('time_mask_param', 20)

        # Paths
        paths_cfg = (config or {}).get('paths', {})
        self.dataset_path: str = paths_cfg.get('dataset', 'dataset/UrbanSound8K/audio')
        self.metadata_path: str = paths_cfg.get('metadata', 'dataset/UrbanSound8K/metadata/UrbanSound8K.csv')
        self.features_path: str = paths_cfg.get('features', 'features')

        logger.info(
            "AudioFeatureExtractor initialised — sr=%d, dur=%ds, "
            "feature_shape=(%d, %d)",
            self.sample_rate, self.duration,
            self.n_mels + self.n_mfcc + 12,   # 128+40+12 = 180
            self.target_time_frames,
        )

    # ── Audio I/O ───────────────────────────────────────────
    def load_audio(self, file_path: str) -> Optional[Tuple[np.ndarray, int]]:
        """Load and normalise an audio file to a fixed-length array.

        Applies silence trimming, zero-padding or truncation, and
        peak-normalisation.

        Args:
            file_path: Absolute or relative path to a .wav file.

        Returns:
            Tuple of (audio_array, sample_rate) or *None* on failure.
        """
        try:
            audio, sr = librosa.load(
                file_path, sr=self.sample_rate, duration=self.duration,
            )
            # Trim leading/trailing silence
            audio, _ = librosa.effects.trim(audio, top_db=25)

            # Pad or truncate to fixed length
            if len(audio) < self.n_samples:
                audio = np.pad(audio, (0, self.n_samples - len(audio)))
            else:
                audio = audio[: self.n_samples]

            return audio, sr
        except FileNotFoundError:
            logger.warning("File not found: %s", file_path)
            return None
        except Exception as exc:
            logger.error("Failed to load %s: %s", file_path, exc)
            return None

    # ── Feature extraction ──────────────────────────────────
    @staticmethod
    def resize_feature(feature: np.ndarray, target_len: int) -> np.ndarray:
        """Resize a 2-D feature matrix to a fixed time dimension.

        Args:
            feature: Shape (freq_bins, time_frames).
            target_len: Desired number of time frames.

        Returns:
            Feature matrix with shape (freq_bins, target_len).
        """
        if feature.shape[1] > target_len:
            return feature[:, :target_len]
        elif feature.shape[1] < target_len:
            pad_width = target_len - feature.shape[1]
            return np.pad(feature, ((0, 0), (0, pad_width)))
        return feature

    def extract_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract and fuse Mel + MFCC + Chroma into a single tensor.

        Args:
            audio: 1-D waveform array of length *n_samples*.
            sr: Sampling rate.

        Returns:
            Fused feature array of shape (180, target_time_frames).
        """
        # Mel Spectrogram → (128, T)
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=self.n_mels, fmax=8000,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # MFCC → (40, T)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)

        # Chroma STFT → (12, T)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

        # Resize to fixed time axis
        tl = self.target_time_frames
        mel_db = self.resize_feature(mel_db, tl)
        mfcc = self.resize_feature(mfcc, tl)
        chroma = self.resize_feature(chroma, tl)

        # Stack → (180, 130)
        return np.vstack([mel_db, mfcc, chroma])

    # ── SpecAugment ─────────────────────────────────────────
    def spec_augment(self, features: np.ndarray) -> np.ndarray:
        """Apply SpecAugment (frequency + time masking) to a feature tensor.

        Args:
            features: 2-D array of shape (freq_bins, time_frames).

        Returns:
            Augmented feature array (same shape).
        """
        augmented = features.copy()
        freq_bins, time_frames = augmented.shape

        # Frequency masking
        max_freq_mask = min(self.freq_mask_param, freq_bins)
        if max_freq_mask > 0:
            f = np.random.randint(0, max_freq_mask)
            f0 = np.random.randint(0, max(freq_bins - f, 1))
            augmented[f0: f0 + f, :] = 0.0

        # Time masking
        max_time_mask = min(self.time_mask_param, time_frames)
        if max_time_mask > 0:
            t = np.random.randint(0, max_time_mask)
            t0 = np.random.randint(0, max(time_frames - t, 1))
            augmented[:, t0: t0 + t] = 0.0

        return augmented

    # ── Augmentation pipeline ───────────────────────────────
    def augment_audio(self, audio: np.ndarray, sr: int) -> List[np.ndarray]:
        """Generate multiple augmented versions of an audio clip.

        Produces 9 variants: original, 2× noise, 2× time-stretch,
        2× pitch-shift, volume-quiet, volume-loud.

        Args:
            audio: 1-D waveform array.
            sr: Sampling rate.

        Returns:
            List of augmented waveforms.
        """
        augmented: List[np.ndarray] = [audio]  # 1. original

        # 2-3. Gaussian noise (mild & heavy)
        augmented.append(audio + 0.005 * np.random.randn(len(audio)))
        augmented.append(audio + 0.015 * np.random.randn(len(audio)))

        # 4-5. Time stretch (slow & fast)
        for rate in (0.9, 1.1):
            try:
                stretched = librosa.effects.time_stretch(audio, rate=rate)
                stretched = librosa.util.fix_length(stretched, size=self.n_samples)
                augmented.append(stretched)
            except Exception:
                augmented.append(audio)

        # 6-7. Pitch shift (up & down)
        for n_steps in (2, -2):
            try:
                pitched = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
                augmented.append(pitched)
            except Exception:
                augmented.append(audio)

        # 8. Volume reduction
        augmented.append(audio * 0.4)

        # 9. Volume boost
        augmented.append(np.clip(audio * 1.5, -1.0, 1.0))

        return augmented

    # ── Sound mode helper ───────────────────────────────────
    @classmethod
    def get_sound_mode(cls, sound_class: str) -> str:
        """Return the operational mode for a detected sound class.

        Args:
            sound_class: One of CLASS_NAMES.

        Returns:
            'ALERT', 'ASSISTIVE', or 'NEUTRAL'.
        """
        if sound_class in cls.ALERT_SOUNDS:
            return 'ALERT'
        if sound_class in cls.ASSISTIVE_SOUNDS:
            return 'ASSISTIVE'
        return 'NEUTRAL'

    # ── Dataset processing ──────────────────────────────────
    def process_dataset(
        self,
        metadata_path: Optional[str] = None,
        audio_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        chunk_size: int = 500,
        use_hdf5: bool = True,
    ) -> None:
        """Process the entire UrbanSound8K dataset with augmentation.

        Features are saved in chunks to avoid OOM. Supports both .npy
        (legacy) and HDF5 output formats.

        Args:
            metadata_path: Path to UrbanSound8K.csv.
            audio_dir: Root audio directory (containing fold1/ … fold10/).
            output_dir: Where to write extracted features.
            chunk_size: Number of raw clips to process per memory batch.
            use_hdf5: If True, save to HDF5 (recommended); else .npy.
        """
        metadata_path = metadata_path or self.metadata_path
        audio_dir = audio_dir or self.dataset_path
        output_dir = output_dir or self.features_path
        os.makedirs(output_dir, exist_ok=True)

        df = pd.read_csv(metadata_path)
        logger.info(
            "Dataset loaded — %d samples, %d classes",
            len(df), df['class'].nunique(),
        )

        all_features: List[np.ndarray] = []
        all_labels: List[int] = []
        all_modes: List[str] = []
        skipped = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
            fold = f"fold{row['fold']}"
            file_path = os.path.join(audio_dir, fold, row['slice_file_name'])

            if not os.path.exists(file_path):
                skipped += 1
                continue

            result = self.load_audio(file_path)
            if result is None:
                skipped += 1
                continue

            audio, sr = result
            sound_mode = self.get_sound_mode(row['class'])

            # Augment
            augmented_clips = self.augment_audio(audio, sr)
            for clip in augmented_clips:
                features = self.extract_features(clip, sr)
                # Optionally apply SpecAugment (skip for original)
                if clip is not audio:
                    features = self.spec_augment(features)
                all_features.append(features)
                all_labels.append(row['classID'])
                all_modes.append(sound_mode)

            # Flush to disk in chunks to control memory
            if len(all_features) >= chunk_size * 9:
                self._flush_chunk(
                    all_features, all_labels, all_modes,
                    output_dir, use_hdf5,
                )
                all_features.clear()
                all_labels.clear()
                all_modes.clear()

        # Final flush
        if all_features:
            self._flush_chunk(
                all_features, all_labels, all_modes,
                output_dir, use_hdf5,
            )

        logger.info(
            "Feature extraction complete — skipped %d files", skipped,
        )

    # ── Chunk persistence ───────────────────────────────────
    def _flush_chunk(
        self,
        features: List[np.ndarray],
        labels: List[int],
        modes: List[str],
        output_dir: str,
        use_hdf5: bool,
    ) -> None:
        """Persist a batch of features to disk.

        For HDF5, creates or appends to *features.h5*.
        For .npy, creates or appends to *X_features.npy*.
        """
        X = np.array(features)
        y = np.array(labels)
        m = np.array(modes)

        if use_hdf5:
            try:
                import h5py
                h5_path = os.path.join(output_dir, 'features.h5')
                with h5py.File(h5_path, 'a') as f:
                    for name, data in [('X', X), ('y', y)]:
                        if name in f:
                            f[name].resize(f[name].shape[0] + data.shape[0], axis=0)
                            f[name][-data.shape[0]:] = data
                        else:
                            maxshape = (None,) + data.shape[1:]
                            f.create_dataset(
                                name, data=data, maxshape=maxshape,
                                chunks=True, compression='gzip',
                            )
                    # Modes stored as variable-length strings
                    if 'modes' in f:
                        old_len = f['modes'].shape[0]
                        f['modes'].resize(old_len + len(m), axis=0)
                        f['modes'][-len(m):] = m
                    else:
                        dt = h5py.string_dtype()
                        f.create_dataset(
                            'modes', data=m, maxshape=(None,),
                            chunks=True, dtype=dt,
                        )
                logger.info("Flushed %d samples to HDF5", len(X))
            except ImportError:
                logger.warning("h5py not installed — falling back to .npy")
                self._flush_npy(X, y, m, output_dir)
        else:
            self._flush_npy(X, y, m, output_dir)

    @staticmethod
    def _flush_npy(
        X: np.ndarray, y: np.ndarray, m: np.ndarray,
        output_dir: str,
    ) -> None:
        """Append-save to .npy files (legacy fallback)."""
        x_path = os.path.join(output_dir, 'X_features.npy')
        y_path = os.path.join(output_dir, 'y_labels.npy')
        m_path = os.path.join(output_dir, 'y_modes.npy')

        if os.path.exists(x_path):
            X_old = np.load(x_path)
            X = np.concatenate([X_old, X], axis=0)
        if os.path.exists(y_path):
            y_old = np.load(y_path)
            y = np.concatenate([y_old, y], axis=0)
        if os.path.exists(m_path):
            m_old = np.load(m_path, allow_pickle=True)
            m = np.concatenate([m_old, m], axis=0)

        np.save(x_path, X)
        np.save(y_path, y)
        np.save(m_path, m)
        logger.info("Flushed %d samples to .npy", X.shape[0])


# ─────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import yaml

    config_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'configs', 'config.yaml',
    )
    config: Dict[str, Any] = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}

    extractor = AudioFeatureExtractor(config)
    extractor.process_dataset()
