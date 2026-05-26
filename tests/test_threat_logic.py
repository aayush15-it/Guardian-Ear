"""
Tests for the Guardian Ear threat engine.

Covers:
    - TemporalPatternTracker thread safety and scoring
    - ThreatAssessor scoring formula and alert generation
    - O(1) CSV append logging
"""

import os
import sys
import time
import threading
import tempfile
from pathlib import Path
from unittest import TestCase, main

# Resolve imports
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.threat_engine.tracker import TemporalPatternTracker, PatternSummary
from src.threat_engine.rules import ThreatAssessor, get_sound_mode, get_mode_description


class TestSoundMode(TestCase):
    """Test sound mode classification."""

    def test_alert_sounds(self):
        for s in ('gun_shot', 'siren', 'jackhammer'):
            self.assertEqual(get_sound_mode(s), 'ALERT')

    def test_assistive_sounds(self):
        for s in ('dog_bark', 'children_playing', 'car_horn'):
            self.assertEqual(get_sound_mode(s), 'ASSISTIVE')

    def test_unknown_sound(self):
        self.assertEqual(get_sound_mode('unknown_sound'), 'NEUTRAL')

    def test_descriptions(self):
        desc = get_mode_description('gun_shot')
        self.assertIn('weapon', desc.lower())
        desc_unknown = get_mode_description('xyz')
        self.assertIn('Unknown', desc_unknown)


class TestTemporalPatternTracker(TestCase):
    """Test the thread-safe temporal pattern tracker."""

    def setUp(self):
        self.tracker = TemporalPatternTracker(window_size=60)

    def test_add_and_count(self):
        self.tracker.add_detection('gun_shot')
        self.tracker.add_detection('gun_shot')
        self.assertEqual(self.tracker.get_detection_count('gun_shot'), 2)

    def test_empty_pattern(self):
        summary = self.tracker.get_pattern_summary('dog_bark')
        self.assertEqual(summary.pattern_label, 'BRIEF_NORMAL')
        self.assertEqual(summary.pattern_score, 0.0)
        self.assertFalse(summary.should_escalate)

    def test_pattern_score_increases(self):
        for _ in range(10):
            self.tracker.add_detection('dog_bark')
        score = self.tracker.get_pattern_score('dog_bark')
        self.assertGreater(score, 0.0)

    def test_multi_device(self):
        self.tracker.add_detection('siren', device_id='cam_01')
        self.tracker.add_detection('siren', device_id='cam_02')
        self.assertEqual(self.tracker.get_detection_count('siren', 'cam_01'), 1)
        self.assertEqual(self.tracker.get_detection_count('siren', 'cam_02'), 1)

    def test_reset(self):
        self.tracker.add_detection('gun_shot')
        self.tracker.reset('gun_shot')
        self.assertEqual(self.tracker.get_detection_count('gun_shot'), 0)

    def test_thread_safety(self):
        """Ensure concurrent access doesn't crash."""
        errors = []

        def worker(sound, n):
            try:
                for _ in range(n):
                    self.tracker.add_detection(sound)
                    self.tracker.get_pattern_summary(sound)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=('gun_shot', 50)),
            threading.Thread(target=worker, args=('siren', 50)),
            threading.Thread(target=worker, args=('gun_shot', 50)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)

    def test_get_all_active_patterns(self):
        self.tracker.add_detection('gun_shot')
        self.tracker.add_detection('siren')
        patterns = self.tracker.get_all_active_patterns()
        self.assertIn('gun_shot', patterns)
        self.assertIn('siren', patterns)
        self.assertIsInstance(patterns['gun_shot'], PatternSummary)


class TestThreatAssessor(TestCase):
    """Test the threat scoring engine."""

    def setUp(self):
        self.assessor = ThreatAssessor()

    def test_gun_shot_high_confidence_night(self):
        score = self.assessor.calculate_threat_score(
            'gun_shot', confidence=0.95, location='parking_lot',
            hour=2, pattern_score=0.5,
        )
        self.assertGreaterEqual(score, 70)

    def test_children_playing_daytime(self):
        score = self.assessor.calculate_threat_score(
            'children_playing', confidence=0.85, location='garden',
            hour=12, pattern_score=0.0,
        )
        self.assertLess(score, 40)

    def test_threat_levels(self):
        self.assertEqual(self.assessor.get_threat_level(85), 'CRITICAL')
        self.assertEqual(self.assessor.get_threat_level(65), 'HIGH')
        self.assertEqual(self.assessor.get_threat_level(45), 'MEDIUM')
        self.assertEqual(self.assessor.get_threat_level(25), 'LOW')
        self.assertEqual(self.assessor.get_threat_level(10), 'SAFE')

    def test_time_weights(self):
        self.assertEqual(self.assessor.get_time_weight(2), 1.0)
        self.assertEqual(self.assessor.get_time_weight(12), 0.3)

    def test_generate_alert(self):
        tracker = TemporalPatternTracker()
        alert = self.assessor.generate_alert(
            sound_class='gun_shot',
            confidence=0.92,
            location='parking_lot',
            tracker=tracker,
            hour=2,
        )
        self.assertEqual(alert.sound_class, 'gun_shot')
        self.assertEqual(alert.sound_mode, 'ALERT')
        self.assertGreater(alert.threat_score, 0)

    def test_csv_append_logging(self):
        """Test O(1) CSV logging doesn't crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assessor = ThreatAssessor({'paths': {'alerts': tmpdir}})
            tracker = TemporalPatternTracker()

            for _ in range(3):
                assessor.generate_alert(
                    'gun_shot', 0.95, 'parking_lot',
                    tracker=tracker, hour=2,
                )

            csv_path = os.path.join(tmpdir, 'alert_log.csv')
            self.assertTrue(os.path.exists(csv_path))

            import pandas as pd
            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 3)


class TestFeatureExtractor(TestCase):
    """Test the audio feature extractor."""

    def test_resize_feature(self):
        from src.features.audio_pipeline import AudioFeatureExtractor
        import numpy as np

        extractor = AudioFeatureExtractor()

        # Test padding
        short = np.random.randn(128, 100)
        resized = extractor.resize_feature(short, 130)
        self.assertEqual(resized.shape, (128, 130))

        # Test trimming
        long = np.random.randn(128, 200)
        resized = extractor.resize_feature(long, 130)
        self.assertEqual(resized.shape, (128, 130))

        # Test exact
        exact = np.random.randn(128, 130)
        resized = extractor.resize_feature(exact, 130)
        self.assertEqual(resized.shape, (128, 130))

    def test_spec_augment(self):
        from src.features.audio_pipeline import AudioFeatureExtractor
        import numpy as np

        extractor = AudioFeatureExtractor()
        features = np.random.randn(180, 130)
        augmented = extractor.spec_augment(features)
        self.assertEqual(augmented.shape, features.shape)
        # SpecAugment should zero out some regions
        self.assertLess(np.count_nonzero(augmented), np.count_nonzero(features))


if __name__ == '__main__':
    main()
