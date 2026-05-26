import json
import os
import time
from datetime import datetime
from collections import defaultdict
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
ALERTS_PATH = 'alerts'
THRESHOLD   = 60    # Threat score threshold

# ─────────────────────────────────────────
# DUAL MODE CLASSIFICATION
# ─────────────────────────────────────────
ALERT_SOUNDS = [
    'gun_shot',
    'siren',
    'jackhammer'
]

ASSISTIVE_SOUNDS = [
    'dog_bark',
    'children_playing',
    'air_conditioner',
    'street_music',
    'engine_idling',
    'car_horn',
    'drilling'
]

def get_sound_mode(sound_class):
    """
    Classifies sound into one of three modes:

    ALERT     → immediate security response needed
    ASSISTIVE → informational feedback only
    NEUTRAL   → monitor and observe
    """
    if sound_class in ALERT_SOUNDS:
        return 'ALERT'
    elif sound_class in ASSISTIVE_SOUNDS:
        return 'ASSISTIVE'
    else:
        return 'NEUTRAL'

def get_mode_description(sound_class):
    """
    Returns human-readable description
    for each sound in context.
    """
    descriptions = {
        # ALERT sounds
        'gun_shot'        : 'Potential weapon discharge detected — immediate response required',
        'siren'           : 'Emergency siren detected — verify source and respond',
        'jackhammer'      : 'Heavy impact sound detected — check for forced entry',

        # ASSISTIVE sounds
        'dog_bark'        : 'Dog barking detected — may indicate intruder or distress',
        'children_playing': 'Children activity detected — monitor if after hours',
        'air_conditioner' : 'HVAC sound detected — check if running abnormally long',
        'street_music'    : 'Music or crowd noise detected — monitor noise levels',
        'engine_idling'   : 'Vehicle idling detected — check for suspicious activity',
        'car_horn'        : 'Vehicle horn detected — possible road incident',
        'drilling'        : 'Drilling sound detected — verify authorized work',
    }
    return descriptions.get(
        sound_class,
        'Unknown sound detected — monitor situation'
    )

# ─────────────────────────────────────────
# SOUND RISK WEIGHTS (0 to 1)
# How inherently dangerous each sound is
# ─────────────────────────────────────────
SOUND_WEIGHTS = {
    'gun_shot'        : 1.00,
    'siren'           : 0.85,
    'jackhammer'      : 0.60,
    'drilling'        : 0.55,
    'engine_idling'   : 0.40,
    'car_horn'        : 0.50,
    'children_playing': 0.15,
    'dog_bark'        : 0.20,
    'street_music'    : 0.10,
    'air_conditioner' : 0.05
}

# ─────────────────────────────────────────
# LOCATION RISK WEIGHTS (0 to 1)
# How sensitive each location is
# ─────────────────────────────────────────
LOCATION_WEIGHTS = {
    'parking_lot' : 1.00,
    'hostel'      : 0.90,
    'corridor'    : 0.85,
    'entrance'    : 0.80,
    'library'     : 0.70,
    'classroom'   : 0.60,
    'office'      : 0.55,
    'cafeteria'   : 0.50,
    'garden'      : 0.40,
    'unknown'     : 0.50
}

# ─────────────────────────────────────────
# TIME RISK WEIGHTS
# Night hours are higher risk
# ─────────────────────────────────────────
def get_time_weight(hour=None):
    """
    Returns risk weight based on time of day.

    Late night  → 1.00 (highest risk)
    Night       → 0.80
    Evening     → 0.50
    Morning     → 0.60
    Daytime     → 0.30 (lowest risk)
    """
    if hour is None:
        hour = datetime.now().hour

    if 0 <= hour < 6:       # Late night
        return 1.00
    elif 6 <= hour < 9:     # Early morning
        return 0.60
    elif 9 <= hour < 17:    # Daytime
        return 0.30
    elif 17 <= hour < 21:   # Evening
        return 0.50
    else:                   # Night
        return 0.80

# ─────────────────────────────────────────
# TEMPORAL PATTERN TRACKER
# Tracks duration and repetition of sounds
# to detect sustained/abnormal patterns
# ─────────────────────────────────────────
class TemporalPatternTracker:
    def __init__(self):
        # Stores timestamps of detections
        self.detection_history = defaultdict(list)

        # How long before considered abnormal
        self.duration_thresholds = {
            'dog_bark'        : 180,   # 3 min
            'engine_idling'   : 600,   # 10 min
            'drilling'        : 900,   # 15 min
            'children_playing': 1800,  # 30 min
            'street_music'    : 600,   # 10 min
            'air_conditioner' : 1800,  # 30 min
            'car_horn'        : 60,    # 1 min
            'siren'           : 120,   # 2 min
            'gun_shot'        : 0,     # instant
            'jackhammer'      : 300,   # 5 min
        }

        # How many times before considered abnormal
        self.repetition_thresholds = {
            'dog_bark'        : 8,
            'car_horn'        : 5,
            'gun_shot'        : 2,
            'siren'           : 3,
            'children_playing': 20,
            'drilling'        : 10,
            'jackhammer'      : 8,
        }

        # Analysis window — 10 minutes
        self.window_size = 600

    def add_detection(self, sound_class,
                      timestamp=None):
        """Records a new detection timestamp."""
        if timestamp is None:
            timestamp = time.time()
        self.detection_history[sound_class].append(
            timestamp
        )
        self._clean_old_detections(sound_class)

    def _clean_old_detections(self, sound_class):
        """Removes detections outside time window."""
        cutoff = time.time() - self.window_size
        self.detection_history[sound_class] = [
            t for t in
            self.detection_history[sound_class]
            if t > cutoff
        ]

    def get_detection_count(self, sound_class):
        """Returns count of detections in window."""
        return len(
            self.detection_history[sound_class]
        )

    def get_duration(self, sound_class):
        """
        Returns how long sound has been
        detected in seconds.
        """
        history = self.detection_history[sound_class]
        if not history:
            return 0
        return time.time() - history[0]

    def get_pattern_score(self, sound_class):
        """
        Returns pattern intensity score (0-1).
        Based on duration and repetition analysis.

        0.0 → brief/normal occurrence
        1.0 → sustained/abnormal pattern
        """
        history = self.detection_history[sound_class]
        if not history:
            return 0.0

        # Repetition score
        count = len(history)
        rep_thresh = self.repetition_thresholds.get(
            sound_class, 10
        )
        rep_score = min(count / rep_thresh, 1.0)

        # Duration score
        duration  = self.get_duration(sound_class)
        dur_thresh = self.duration_thresholds.get(
            sound_class, 300
        )
        if dur_thresh == 0:
            dur_score = 1.0
        else:
            dur_score = min(
                duration / dur_thresh, 1.0
            )

        # Take the maximum of both scores
        pattern_score = max(rep_score, dur_score)

        return round(pattern_score, 2)

    def get_pattern_label(self, sound_class):
        """
        Returns human-readable pattern label
        and score for the detected sound.
        """
        score = self.get_pattern_score(sound_class)
        count = self.get_detection_count(sound_class)
        duration = self.get_duration(sound_class)

        if score >= 0.8:
            label = 'SUSTAINED_DISTRESS'
        elif score >= 0.5:
            label = 'PROLONGED_ABNORMAL'
        elif score >= 0.2:
            label = 'REPETITIVE_PATTERN'
        else:
            label = 'BRIEF_NORMAL'

        return label, score, count, duration

    def should_escalate(self, sound_class):
        """
        Returns True if pattern suggests
        escalation to higher alert level.
        """
        label, score, _, _ = self.get_pattern_label(
            sound_class
        )
        return label in [
            'SUSTAINED_DISTRESS',
            'PROLONGED_ABNORMAL'
        ]

    def get_pattern_summary(self, sound_class):
        """
        Returns full pattern analysis summary
        as a dictionary.
        """
        label, score, count, duration = \
            self.get_pattern_label(sound_class)

        return {
            'pattern_label'   : label,
            'pattern_score'   : score,
            'detection_count' : count,
            'duration_seconds': round(duration, 1),
            'should_escalate' : self.should_escalate(
                sound_class
            )
        }

    def reset(self, sound_class=None):
        """Resets tracker for one or all sounds."""
        if sound_class:
            self.detection_history[sound_class] = []
        else:
            self.detection_history = defaultdict(list)


# Create global tracker instance
tracker = TemporalPatternTracker()

# ─────────────────────────────────────────
# THREAT SCORE FORMULA v2
# Includes temporal pattern score
# ─────────────────────────────────────────
def calculate_threat_score(
    sound_class,
    confidence,
    location='unknown',
    hour=None,
    pattern_score=0.0
):
    """
    Calculates threat score from 0 to 100.

    Updated Formula (v2):
    TS = (α × Confidence)    → 30%
       + (β × Time_Weight)   → 20%
       + (γ × Location)      → 15%
       + (δ × Sound_Weight)  → 15%
       + (ε × Pattern_Score) → 20%

    All weights sum to 1.0
    Output range: 0 to 100
    """
    sound_w    = SOUND_WEIGHTS.get(sound_class, 0.5)
    location_w = LOCATION_WEIGHTS.get(location, 0.5)
    time_w     = get_time_weight(hour)

    # Formula weights
    alpha   = 0.30   # model confidence
    beta    = 0.20   # time of day
    gamma   = 0.15   # location risk
    delta   = 0.15   # sound type risk
    epsilon = 0.20   # temporal pattern

    raw_score = (
        (alpha   * confidence)    +
        (beta    * time_w)        +
        (gamma   * location_w)    +
        (delta   * sound_w)       +
        (epsilon * pattern_score)
    )

    threat_score = round(raw_score * 100, 1)
    return threat_score

# ─────────────────────────────────────────
# THREAT LEVEL CLASSIFICATION
# ─────────────────────────────────────────
def get_threat_level(threat_score):
    """
    Converts numerical score to threat level.

    80-100 → CRITICAL
    60-79  → HIGH
    40-59  → MEDIUM
    20-39  → LOW
    0-19   → SAFE
    """
    if threat_score >= 80:
        return 'CRITICAL'
    elif threat_score >= 60:
        return 'HIGH'
    elif threat_score >= 40:
        return 'MEDIUM'
    elif threat_score >= 20:
        return 'LOW'
    else:
        return 'SAFE'

def get_threat_color(threat_level):
    """Returns hex color for each threat level."""
    colors = {
        'CRITICAL': '#FF0000',
        'HIGH'    : '#FF6600',
        'MEDIUM'  : '#FFAA00',
        'LOW'     : '#88CC00',
        'SAFE'    : '#00AA00'
    }
    return colors.get(threat_level, '#888888')

# ─────────────────────────────────────────
# MAIN ALERT GENERATION FUNCTION
# ─────────────────────────────────────────
def generate_alert(
    sound_class,
    confidence,
    location='unknown',
    hour=None
):
    """
    Main function — evaluates detection and
    generates intelligent alert.

    Steps:
    1. Add to temporal tracker
    2. Get pattern analysis
    3. Get sound mode (ALERT/ASSISTIVE)
    4. Calculate threat score
    5. Classify threat level
    6. Build alert object
    7. Print + save if threshold exceeded
    """

    # Step 1 — Add to tracker
    tracker.add_detection(sound_class)

    # Step 2 — Get pattern analysis
    pattern = tracker.get_pattern_summary(
        sound_class
    )

    # Step 3 — Get sound mode
    sound_mode   = get_sound_mode(sound_class)
    description  = get_mode_description(sound_class)

    # Step 4 — Calculate threat score
    threat_score = calculate_threat_score(
        sound_class,
        confidence,
        location,
        hour,
        pattern['pattern_score']
    )

    # Step 5 — Classify threat level
    threat_level = get_threat_level(threat_score)
    threat_color = get_threat_color(threat_level)

    # Step 6 — Check escalation
    if pattern['should_escalate']:
        if threat_score < 80:
            threat_score = min(
                threat_score + 10, 100
            )
            threat_level = get_threat_level(
                threat_score
            )

    # Step 7 — Build alert object
    now = datetime.now()
    alert = {
        'timestamp'       : now.strftime(
            '%Y-%m-%d %H:%M:%S'
        ),
        'date'            : now.strftime('%Y-%m-%d'),
        'time'            : now.strftime('%H:%M:%S'),
        'sound_class'     : sound_class,
        'confidence'      : round(confidence * 100, 1),
        'location'        : location,
        'sound_mode'      : sound_mode,
        'description'     : description,
        'pattern_label'   : pattern['pattern_label'],
        'pattern_score'   : pattern['pattern_score'],
        'detection_count' : pattern['detection_count'],
        'duration_seconds': pattern['duration_seconds'],
        'should_escalate' : pattern['should_escalate'],
        'threat_score'    : threat_score,
        'threat_level'    : threat_level,
        'threat_color'    : threat_color,
        'alert_raised'    : threat_score >= THRESHOLD
    }

    # Print alert
    print_alert(alert)

    # Save if threshold exceeded
    if alert['alert_raised']:
        save_alert(alert)

    return alert

# ─────────────────────────────────────────
# PRINT ALERT TO CONSOLE
# ─────────────────────────────────────────
def print_alert(alert):
    print("\n" + "=" * 55)
    if alert['alert_raised']:
        print("   ⚠  GUARDIAN EAR — ALERT RAISED")
    else:
        print("   Guardian Ear — Detection Log")
    print("=" * 55)
    print(f"  Timestamp      : {alert['timestamp']}")
    print(f"  Sound Type     : {alert['sound_class']}")
    print(f"  Mode           : {alert['sound_mode']}")
    print(f"  Confidence     : {alert['confidence']}%")
    print(f"  Location       : {alert['location']}")
    print(f"  Description    : {alert['description']}")
    print(f"  Pattern        : {alert['pattern_label']}")
    print(f"  Pattern Score  : {alert['pattern_score']}")
    print(f"  Detection Count: {alert['detection_count']}")
    print(f"  Duration       : {alert['duration_seconds']}s")
    print(f"  Escalation     : {alert['should_escalate']}")
    print(f"  Threat Score   : {alert['threat_score']}/100")
    print(f"  Threat Level   : {alert['threat_level']}")
    print(f"  Alert Raised   : {alert['alert_raised']}")
    print("=" * 55 + "\n")

# ─────────────────────────────────────────
# SAVE ALERT TO CSV LOG
# ─────────────────────────────────────────
def save_alert(alert):
    """Saves alert to persistent CSV log file."""
    os.makedirs(ALERTS_PATH, exist_ok=True)
    csv_path = os.path.join(
        ALERTS_PATH, 'alert_log.csv'
    )

    # Remove color field before saving
    alert_data = {
        k: v for k, v in alert.items()
        if k != 'threat_color'
    }

    df_new = pd.DataFrame([alert_data])

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat(
            [df_existing, df_new],
            ignore_index=True
        )
    else:
        df_combined = df_new

    df_combined.to_csv(csv_path, index=False)
    print(f"Alert saved to {csv_path}")

# ─────────────────────────────────────────
# LOAD ALERT HISTORY
# ─────────────────────────────────────────
def load_alert_history():
    """Loads saved alert log from CSV."""
    csv_path = os.path.join(
        ALERTS_PATH, 'alert_log.csv'
    )
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        return pd.DataFrame()

# ─────────────────────────────────────────
# MAIN — TEST ALL FUNCTIONALITY
# ─────────────────────────────────────────
if __name__ == "__main__":

    print("Testing Guardian Ear Threat Logic v2\n")

    # Test 1 — Critical alert scenario
    print("TEST 1: Gun shot at 2AM in parking lot")
    generate_alert(
        sound_class='gun_shot',
        confidence=0.92,
        location='parking_lot',
        hour=2
    )

    # Test 2 — Low risk scenario
    print("TEST 2: Children playing at noon cafeteria")
    generate_alert(
        sound_class='children_playing',
        confidence=0.85,
        location='cafeteria',
        hour=12
    )

    # Test 3 — Assistive mode scenario
    print("TEST 3: Dog barking at evening garden")
    generate_alert(
        sound_class='dog_bark',
        confidence=0.78,
        location='garden',
        hour=19
    )

    # Test 4 — Pattern escalation test
    print("TEST 4: Dog barking repeatedly (8 times)")
    for i in range(8):
        tracker.add_detection('dog_bark')
    generate_alert(
        sound_class='dog_bark',
        confidence=0.80,
        location='parking_lot',
        hour=23
    )

    # Test 5 — Medium risk scenario
    print("TEST 5: Siren at corridor evening")
    generate_alert(
        sound_class='siren',
        confidence=0.75,
        location='corridor',
        hour=20
    )

    # Show alert history
    print("\nAlert History Log:")
    history = load_alert_history()
    if not history.empty:
        cols = [
            'timestamp', 'sound_class',
            'sound_mode', 'pattern_label',
            'threat_score', 'threat_level'
        ]
        print(
            history[cols].to_string(index=False)
        )
    else:
        print("No alerts saved yet.")