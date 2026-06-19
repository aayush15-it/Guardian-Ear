"""
Guardian Ear — Production-Grade Streamlit Dashboard.

Refactored from 05_dashboard.py with proper session_state management,
modular imports from src/, and improved UI/UX with dark mode support.

Launch with::

    streamlit run ui/dashboard.py
"""

import os
import sys
import json
import time
import warnings
import re
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── Resolve project root and ensure venv is in sys.path ────────────────────────
# ROOT CAUSE FIX: When Streamlit launches without the virtual environment
# activated (no 'activate' call), its subprocess may not include guardian_env
# in sys.path. This causes 'import yaml' in config_loader.py to raise
# ModuleNotFoundError which then surfaces as 'Could not open microphone'.
# We self-heal by injecting BOTH the project root AND the venv site-packages.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Inject venv site-packages so yaml/librosa/sounddevice are always importable
_VENV_SITE_PACKAGES = _PROJECT_ROOT / 'guardian_env' / 'lib' / 'site-packages'
if _VENV_SITE_PACKAGES.exists() and str(_VENV_SITE_PACKAGES) not in sys.path:
    sys.path.insert(1, str(_VENV_SITE_PACKAGES))

from src.threat_engine.rules import (
    ThreatAssessor, get_sound_mode, get_mode_description,
)
from src.threat_engine.tracker import TemporalPatternTracker
from src.features.audio_pipeline import AudioFeatureExtractor
from src.inference.realtime_engine import RealTimeDetector

# ─────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title='Guardian Ear — AI Acoustic Intelligence',
    page_icon='🎧',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ─────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────
MODEL_PATH = 'model/guardian_ear_model.h5'
INFO_PATH = 'model/class_info.json'
SAMPLE_RATE = 22050
DURATION = 3
SAMPLES = SAMPLE_RATE * DURATION
NUM_CLASSES = 10

CLASS_NAMES = [
    'air_conditioner', 'car_horn', 'children_playing',
    'dog_bark', 'drilling', 'engine_idling',
    'gun_shot', 'jackhammer', 'siren', 'street_music',
]

LOCATIONS = [
    'parking_lot', 'corridor', 'hostel',
    'library', 'cafeteria', 'classroom',
    'entrance', 'garden', 'office',
]


# ─────────────────────────────────────────────────
# CUSTOM CSS — Dark mode professional theme
# ─────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  .main-title {
    font-family: 'Inter', sans-serif;
    font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; padding: 10px 0 4px;
  }
  .subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 1.05rem; color: #94a3b8;
    text-align: center; margin-bottom: 24px;
  }
  .mode-alert {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border-left: 5px solid #ef4444; padding: 14px 18px;
    border-radius: 10px; margin: 10px 0;
    font-weight: 600; color: #fca5a5;
  }
  .mode-assistive {
    background: linear-gradient(135deg, #0c1e3a, #1e3a5f);
    border-left: 5px solid #3b82f6; padding: 14px 18px;
    border-radius: 10px; margin: 10px 0;
    font-weight: 600; color: #93c5fd;
  }
  .mode-neutral {
    background: linear-gradient(135deg, #1a1a00, #3d3d00);
    border-left: 5px solid #eab308; padding: 14px 18px;
    border-radius: 10px; margin: 10px 0;
    font-weight: 600; color: #fde047;
  }
  .alert-critical {
    background: linear-gradient(135deg, #450a0a, #991b1b);
    border-left: 6px solid #ef4444; padding: 16px 18px;
    border-radius: 12px; margin: 10px 0; color: #fecaca;
  }
  .alert-high {
    background: linear-gradient(135deg, #431407, #9a3412);
    border-left: 6px solid #f97316; padding: 16px 18px;
    border-radius: 12px; margin: 10px 0; color: #fed7aa;
  }
  .alert-medium {
    background: linear-gradient(135deg, #1c1917, #78350f);
    border-left: 6px solid #eab308; padding: 16px 18px;
    border-radius: 12px; margin: 10px 0; color: #fef08a;
  }
  .alert-low {
    background: linear-gradient(135deg, #052e16, #166534);
    border-left: 6px solid #22c55e; padding: 16px 18px;
    border-radius: 12px; margin: 10px 0; color: #bbf7d0;
  }
  .escalation-warn {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 2px solid #ef4444; border-radius: 12px;
    padding: 14px; margin: 10px 0; color: #fca5a5;
    font-weight: 700; text-align: center; font-size: 1.1rem;
  }
  .stat-card {
    background: linear-gradient(135deg, #1e1b4b, #312e81);
    border-radius: 14px; padding: 16px; margin: 6px 0;
    text-align: center; border: 1px solid #4338ca;
  }
  .stat-card h3 { color: #a5b4fc; font-size: 0.85rem; margin: 0; }
  .stat-card p { color: #e0e7ff; font-size: 1.8rem; font-weight: 800; margin: 4px 0 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────
# SESSION STATE — prevents duplicate alerts on rerun
# ─────────────────────────────────────────────────
if 'tracker' not in st.session_state:
    st.session_state.tracker = TemporalPatternTracker()
if 'assessor' not in st.session_state:
    st.session_state.assessor = ThreatAssessor()
if 'extractor' not in st.session_state:
    st.session_state.extractor = AudioFeatureExtractor()
if 'live_monitoring' not in st.session_state:
    st.session_state.live_monitoring = False
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'triggered_actions' not in st.session_state:
    st.session_state.triggered_actions = []
if 'session_start_time' not in st.session_state:
    st.session_state.session_start_time = time.time()
if 'session_detection_count' not in st.session_state:
    st.session_state.session_detection_count = 0
if 'session_peak_threat' not in st.session_state:
    st.session_state.session_peak_threat = 0
if 'session_peak_source' not in st.session_state:
    st.session_state.session_peak_source = "None"
if 'session_peak_time' not in st.session_state:
    st.session_state.session_peak_time = "--:--:--"
if 'session_status' not in st.session_state:
    st.session_state.session_status = "IDLE"
if 'last_detected_class' not in st.session_state:
    st.session_state.last_detected_class = None
if 'last_detected_time' not in st.session_state:
    st.session_state.last_detected_time = 0.0
if 'emergency_test_status' not in st.session_state:
    st.session_state.emergency_test_status = None
if 'notification_log' not in st.session_state:
    st.session_state.notification_log = []  # list of notification_record dicts
if 'emergency_engine' not in st.session_state:
    st.session_state.emergency_engine = None  # lazy-loaded on first use
# Open-set classifier — created once per session, reads from config.yaml
if 'open_set_clf' not in st.session_state:
    try:
        from src.inference.open_set import from_config as _osc_from_config
        st.session_state.open_set_clf = _osc_from_config()
    except Exception:
        st.session_state.open_set_clf = None

# ── Hydrate SMTP + Contact settings from config.yaml on first load ────────────
if 'smtp_sender' not in st.session_state:
    try:
        from src.utils.config_loader import load_config
        _full = load_config()
        _em = _full.get("emergency", {})
        _ec = _em.get("contacts", [{}])[0] if _em.get("contacts") else {}
        _email_cfg = _em.get("email", {})
        st.session_state.smtp_sender = _email_cfg.get("sender_email", "")
        st.session_state.smtp_password = _email_cfg.get("sender_password", "")
        st.session_state.smtp_host = _email_cfg.get("smtp_host", "smtp.gmail.com")
        st.session_state.smtp_port = int(_email_cfg.get("smtp_port", 587))
        st.session_state.ec_name = _ec.get("name", "")
        st.session_state.ec_phone = _ec.get("phone", "")
        st.session_state.ec_email = _ec.get("email", "")
    except Exception:
        st.session_state.smtp_sender = ""
        st.session_state.smtp_password = ""
        st.session_state.smtp_host = "smtp.gmail.com"
        st.session_state.smtp_port = 587


def _get_emergency_engine():
    """Lazy-load the EmergencyNotificationEngine (once per session)."""
    if st.session_state.emergency_engine is None:
        try:
            from src.notifications.emergency_engine import from_config as _eng_from_config
            st.session_state.emergency_engine = _eng_from_config()
        except Exception:
            pass
    return st.session_state.emergency_engine

def _save_emergency_config(
    name: str, phone: str, email: str,
    smtp_sender: str = "", smtp_password: str = "",
    smtp_host: str = "smtp.gmail.com", smtp_port: int = 587,
    smtp_enabled: bool = False,
) -> tuple:
    """Save emergency contact AND SMTP credentials to configs/config.yaml.
    
    Returns:
        (success: bool, message: str)
    """
    config_path = Path("configs/config.local.yaml")
    
    # We always write to local.yaml now, read existing if any
    try:
        import yaml
        cfg = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        
        # Ensure emergency dict exists
        if "emergency" not in cfg:
            cfg["emergency"] = {}
        
        # ── Save emergency contact ────────────────────────────────────────
        if "contacts" not in cfg["emergency"] or not cfg["emergency"]["contacts"]:
            cfg["emergency"]["contacts"] = [{}]
        cfg["emergency"]["contacts"][0]["name"] = name
        cfg["emergency"]["contacts"][0]["phone"] = phone
        cfg["emergency"]["contacts"][0]["email"] = email
        
        # ── Save SMTP email credentials ───────────────────────────────────
        if "email" not in cfg["emergency"]:
            cfg["emergency"]["email"] = {}
        cfg["emergency"]["email"]["sender_email"] = smtp_sender.strip()
        cfg["emergency"]["email"]["sender_password"] = smtp_password
        cfg["emergency"]["email"]["recipient_email"] = email.strip()  # alert goes to emergency contact
        cfg["emergency"]["email"]["smtp_host"] = smtp_host
        cfg["emergency"]["email"]["smtp_port"] = smtp_port
        # Only auto-enable if all three required fields are present
        auto_enable = bool(smtp_sender.strip() and smtp_password and email.strip())
        cfg["emergency"]["email"]["enabled"] = auto_enable
        
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        
        # Force re-initialisation of engine with new credentials
        from src.notifications.emergency_engine import from_config
        st.session_state.emergency_engine = from_config(cfg)
        
        if auto_enable:
            return True, "Configuration saved. Email alerts ENABLED."
        else:
            missing = []
            if not smtp_sender.strip(): missing.append("Sender Email")
            if not smtp_password: missing.append("App Password")
            if not email.strip(): missing.append("Recipient Email")
            return True, f"Saved (email disabled — missing: {', '.join(missing)})."
    except Exception as e:
        return False, f"Error saving config: {e}"

# ─────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the Keras model (cached across reruns)."""
    path = MODEL_PATH
    if not os.path.exists(path):
        fallback_path = 'model/best_model.h5'
        if os.path.exists(fallback_path):
            path = fallback_path
        else:
            return None
    import tensorflow as tf
    return tf.keras.models.load_model(path)

@st.cache_data
def load_class_info():
    """Load class metadata JSON."""
    if os.path.exists(INFO_PATH):
        with open(INFO_PATH, 'r') as f:
            return json.load(f)
    return None

def load_normalization() -> Tuple[Optional[float], Optional[float]]:
    """Load training normalization bounds."""
    x_min_p = 'model/X_min.npy'
    x_max_p = 'model/X_max.npy'
    if os.path.exists(x_min_p) and os.path.exists(x_max_p):
        return float(np.load(x_min_p)[0]), float(np.load(x_max_p)[0])
    return None, None


# ─────────────────────────────────────────────────
# PRIORITY-BASED ACTION ENGINE & MOBILE DISPATCH
# ─────────────────────────────────────────────────
def process_priority_actions(sound_class: str, confidence: float, threat_score: float, threat_level: str, open_set_status: str = "KNOWN", vote_count: int = 1) -> Optional[Dict[str, Any]]:
    """
    Decides and triggers emergency response actions based on threat priority tiers.

    Tiers:
      TIER 5 (CRITICAL): gun_shot, siren, jackhammer, explosion, glass_breaking, fire_alarm
          Action: Desktop notification + Email + Emergency Contact Escalation
      TIER 4 (GUARDIAN): children_playing, child_crying, elderly_distress
          Action: Emergency contact notification + Desktop alert
      TIER 3 (HOME SAFETY): water_flow, drilling, gas_alarm
          Action: Dashboard warning + Desktop notification
      TIER 2 (LOW ALERT): dog_bark, engine_idling, car_horn
          Action: Dashboard notice only
      TIER 1 (AMBIENT): street_music, air_conditioner
          Action: Log only
    """
    if sound_class in ('silence', '', None, 'unknown'):
        return None

    # ── Live Metrics Tracking ──────────────────────────────────────────────────
    current_time = time.time()
    last_class = st.session_state.get('last_detected_class')
    last_time = st.session_state.get('last_detected_time', 0.0)

    # Detection Debouncing: Increment if new class OR if same class but > 3s cooldown
    if sound_class != last_class or (current_time - last_time > 3.0):
        st.session_state.session_detection_count = st.session_state.get('session_detection_count', 0) + 1

    st.session_state.last_detected_class = sound_class
    st.session_state.last_detected_time = current_time

    # Peak Threat Tracking
    if threat_score > st.session_state.get('session_peak_threat', 0):
        st.session_state.session_peak_threat = threat_score
        st.session_state.session_peak_source = sound_class.replace('_', ' ').title()
        st.session_state.session_peak_time = time.strftime('%H:%M:%S')
    # ───────────────────────────────────────────────────────────────────────────

    action_type = "Log Only"
    details = ""
    triggered = False
    fire_emergency_engine = False

    # Load emergency contact info from config
    try:
        from src.utils.config_loader import load_config
        config = load_config()
        ec_cfg = config.get('emergency_contact', {})
        contact_name = ec_cfg.get('name', 'Emergency Contact') or 'Emergency Contact'
        contact_phone = ec_cfg.get('phone', 'Not configured') or 'Not configured'
    except Exception:
        contact_name = 'Emergency Contact'
        contact_phone = 'Not configured'

    location = st.session_state.get('location_override', 'Hostel').replace('_', ' ').title()

    # ── Action routing ─────────────────────────────────────────────────────────
    if sound_class in ('gun_shot', 'siren', 'jackhammer', 'explosion', 'glass_breaking', 'fire_alarm'):
        action_type = "🚨 EMERGENCY DISPATCHED"
        details = (
            f"CRITICAL threat: {sound_class.replace('_',' ').title()} "
            f"detected at {confidence*100:.1f}% confidence. "
            f"Emergency contact notified. All alert channels fired."
        )
        triggered = True
        fire_emergency_engine = True

    elif sound_class in ('children_playing', 'child_crying', 'baby_crying', 'elderly_distress'):
        action_type = "📞 GUARDIAN CONTACT NOTIFIED"
        details = (
            f"Distress pattern: {sound_class.replace('_',' ').title()} detected. "
            f"Emergency contact {contact_name} alerted."
        )
        triggered = True
        fire_emergency_engine = True

    elif sound_class in ('water_flow', 'drilling', 'gas_alarm', 'water_leakage'):
        action_type = "🚰 SMART HOME WARNING"
        details = (
            f"Home safety alert: {sound_class.replace('_',' ').title()} detected. "
            f"Verify source and take corrective action."
        )
        triggered = True
        fire_emergency_engine = (threat_score >= 60)

    elif sound_class in ('dog_bark', 'engine_idling', 'car_horn'):
        action_type = "🔔 DASHBOARD NOTICE"
        details = f"Low-threat event: {sound_class.replace('_', ' ').title()} in vicinity."
        triggered = True
        fire_emergency_engine = False

    elif sound_class in ('street_music', 'air_conditioner'):
        action_type = "📋 AMBIENT LOG"
        details = f"Ambient sound logged: {sound_class.replace('_', ' ').title()}."
        triggered = False

    if triggered:
        action_record = {
            'timestamp': time.strftime('%H:%M:%S'),
            'date': time.strftime('%Y-%m-%d'),
            'sound_class': sound_class,
            'threat_level': threat_level,
            'threat_score': threat_score,
            'action_type': action_type,
            'details': details,
            'location': location,
            'desktop_sent': False,
            'email_sent': False,
            'contact_name': contact_name,
            'contact_phone': contact_phone,
            'contact_status': 'NOT_CONFIGURED',
            'notification_type': 'EMERGENCY' if fire_emergency_engine else 'INFO',
        }

        if 'triggered_actions' not in st.session_state:
            st.session_state.triggered_actions = []

        # Rate-limit: don't re-trigger same class within 5 seconds
        last_action_time = st.session_state.get('last_action_time', 0)
        last_class = st.session_state.triggered_actions[0]['sound_class'] if st.session_state.triggered_actions else None
        cooldown_ok = (last_class != sound_class) or (time.time() - last_action_time > 5)

        if cooldown_ok:
            # ── Fire Emergency Notification Engine in background thread ────────
            if fire_emergency_engine:
                import threading
                def _fire_engine():
                    try:
                        eng = _get_emergency_engine()
                        if eng:
                            rec = eng.dispatch(
                                sound_class=sound_class,
                                confidence=confidence,
                                threat_score=threat_score,
                                threat_level=threat_level,
                                location=location,
                                open_set_status=open_set_status,
                                vote_count=vote_count,
                            )
                            # Update action record with actual delivery status
                            action_record.update({
                                'desktop_sent': rec.get('desktop_sent', False),
                                'email_sent': rec.get('email_sent', False),
                                'contact_status': rec.get('contact_status', 'NOTIFIED'),
                                'channels_fired': rec.get('channels_fired', []),
                            })
                            # Store in notification_log for Emergency Response Center
                            st.session_state.notification_log.insert(0, rec)
                            st.session_state.notification_log = st.session_state.notification_log[:100]
                    except Exception:
                        pass
                threading.Thread(target=_fire_engine, daemon=True).start()

            st.session_state.triggered_actions.insert(0, action_record)
            st.session_state.triggered_actions = st.session_state.triggered_actions[:50]
            st.session_state.last_action_time = time.time()

            return action_record
    return None




# ─────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────
def extract_features_for_dashboard(audio, sr, X_min, X_max):
    """Extract features and return raw components for visualization.

    NORMALIZATION NOTE:
    Must use the global X_min / X_max values saved at training time
    (model/X_min.npy, model/X_max.npy). These encode the global feature
    range across all 78k+ training samples.

    Per-sample normalization (computing min/max of just this 3-second clip)
    destroys the relative feature distribution the model learned during
    training and causes systematic misclassification (e.g. tap water → dog_bark).
    """
    import librosa

    if len(audio) < SAMPLES:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))
    else:
        audio = audio[:SAMPLES]

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

    target_len = 130
    def resize(f):
        if f.shape[1] > target_len:
            return f[:, :target_len]
        elif f.shape[1] < target_len:
            return np.pad(f, ((0, 0), (0, target_len - f.shape[1])))
        return f

    mel_db = resize(mel_db)
    mfcc = resize(mfcc)
    chroma = resize(chroma)

    features = np.vstack([mel_db, mfcc, chroma])

    if X_min is None or X_max is None:
        # Hard failure: per-sample fallback is NOT acceptable here.
        # Return unnormalized features and surface a visible error.
        st.error(
            "⛔ **Normalization files missing** — `model/X_min.npy` and/or "
            "`model/X_max.npy` not found.\n\n"
            "The model will produce **wrong predictions** without these files. "
            "Re-run `02_train_model.py` to regenerate them, then restart the dashboard."
        )
        return features, mel_db, mfcc, chroma

    features = (features - X_min) / (X_max - X_min + 1e-8)
    return features, mel_db, mfcc, chroma


# ─────────────────────────────────────────────────
# GRAD-CAM VISUALIZATION
# ─────────────────────────────────────────────────
def generate_gradcam(model, features, class_idx):
    """Generate Grad-CAM heatmap for explainability."""
    try:
        import tensorflow as tf

        last_conv = None
        for layer in model.layers:
            if 'conv2d' in layer.name:
                last_conv = layer.name
        if last_conv is None:
            return None

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv).output, model.output],
        )

        inp = tf.cast(features[np.newaxis, ..., np.newaxis], tf.float32)
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(inp)
            loss = preds[:, class_idx]

        grads = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception:
        return None


# ─────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────
def plot_features(mel_db, mfcc, chroma, audio, sr):
    """Create a 2×2 feature visualization figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor('#0e1117')

    for ax in axes.flat:
        ax.set_facecolor('#0e1117')
        ax.tick_params(colors='#94a3b8')
        for spine in ax.spines.values():
            spine.set_color('#334155')

    # Waveform
    t = np.linspace(0, DURATION, len(audio))
    axes[0, 0].plot(t, audio, color='#818cf8', linewidth=0.8)
    axes[0, 0].set_title('Audio Waveform', color='#e2e8f0', fontweight='bold')
    axes[0, 0].set_xlabel('Time (s)', color='#94a3b8')
    axes[0, 0].set_ylabel('Amplitude', color='#94a3b8')
    axes[0, 0].grid(True, alpha=0.15, color='#475569')

    # Mel Spectrogram
    im1 = axes[0, 1].imshow(mel_db, aspect='auto', origin='lower', cmap='magma')
    axes[0, 1].set_title('Mel Spectrogram', color='#e2e8f0', fontweight='bold')
    axes[0, 1].set_xlabel('Time Frames', color='#94a3b8')
    axes[0, 1].set_ylabel('Mel Bins', color='#94a3b8')
    plt.colorbar(im1, ax=axes[0, 1])

    # MFCC
    im2 = axes[1, 0].imshow(mfcc, aspect='auto', origin='lower', cmap='coolwarm')
    axes[1, 0].set_title('MFCC Features', color='#e2e8f0', fontweight='bold')
    axes[1, 0].set_xlabel('Time Frames', color='#94a3b8')
    axes[1, 0].set_ylabel('Coefficients', color='#94a3b8')
    plt.colorbar(im2, ax=axes[1, 0])

    # Chroma STFT
    im3 = axes[1, 1].imshow(chroma, aspect='auto', origin='lower', cmap='YlOrRd')
    axes[1, 1].set_title('Chroma STFT', color='#e2e8f0', fontweight='bold')
    axes[1, 1].set_xlabel('Time Frames', color='#94a3b8')
    axes[1, 1].set_ylabel('Pitch Classes', color='#94a3b8')
    plt.colorbar(im3, ax=axes[1, 1])

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────
def render_sidebar_metrics(placeholder=None):
    """Render Live Session Statistics dynamically."""
    if placeholder is None:
        return
        
    if st.session_state.live_monitoring:
        elapsed = int(time.time() - st.session_state.get('session_start_time', time.time()))
        st.session_state.session_elapsed = elapsed
    else:
        elapsed = st.session_state.get('session_elapsed', 0)

    h_e, rem_e = divmod(elapsed, 3600)
    m_e, s_e = divmod(rem_e, 60)
    
    _pt = st.session_state.get('session_peak_threat', 0)
    _src = st.session_state.get('session_peak_source', 'None')
    _time = st.session_state.get('session_peak_time', '--:--:--')
    
    status = st.session_state.get('session_status', 'IDLE')
    status_icon = "🟢" if status == "ACTIVE" else "🟡" if status == "MONITORING" else "🔴" if status == "ERROR" else "⚪"
    
    with placeholder.container():
        st.markdown(f"**\U0001f552 Session:** `{h_e:02d}:{m_e:02d}:{s_e:02d}`")
        st.markdown(f"**\U0001f50a Detections:** `{st.session_state.get('session_detection_count', 0)}`")
        
        if _pt > 0:
            st.markdown(f"**\U0001f6a8 Peak Threat:** `{_pt}/100`")
            st.caption(f"↳ Source: `{_src}` at `{_time}`")
        else:
            st.markdown(f"**\U0001f6a8 Peak Threat:** `0/100`")

        st.markdown(f"**Status:** {status_icon} {status}")
        st.markdown("**Version:** 3.0 Production")



def render_sidebar():
    """Render the sidebar with navigation and settings."""
    st.sidebar.markdown("## 🎧 Guardian Ear")
    st.sidebar.markdown("*AI Acoustic Intelligence System*")
    st.sidebar.divider()

    st.sidebar.subheader("⚙️ Settings")
    location = st.sidebar.selectbox("📍 Location", LOCATIONS)
    threshold = st.sidebar.slider("🎯 Alert Threshold", 0, 100, 50)

    st.sidebar.divider()
    st.sidebar.subheader("🔊 Sound Modes")
    st.sidebar.markdown("**🚨 Alert:** gun_shot, siren, jackhammer")
    st.sidebar.markdown("**ℹ️ Assistive:** dog_bark, children_playing, drilling, engine_idling, car_horn, street_music, air_conditioner")

    st.sidebar.divider()
    st.sidebar.subheader("🗂️ Navigation")
    page = st.sidebar.radio("Go to", [
        "🏠 Live Detection",
        "♿ Assistive Hearing Mode",
        "🚨 Emergency Response Center",
        "📋 Alert History",
        "📊 Feature Visualization",
        "🧠 System Info",
    ])

    sim_sound = "None"
    if page in ["🏠 Live Detection", "♿ Assistive Hearing Mode"]:
        st.sidebar.divider()
        st.sidebar.subheader("🎮 Live Demo Simulator")
        st.sidebar.info("Simulate audio events to test alerts and actions instantly:")
        sim_sound = st.sidebar.selectbox("Inject Sound Event", [
            "None",
            "Gun Shot 🚨",
            "Siren 🚨",
            "Child Crying 👶",
            "Running Tap Water 🚰",
            "Dog Bark ℹ️",
            "Car Horn ℹ️"
        ])

    st.sidebar.divider()
    st.sidebar.subheader("\U0001f6a8 Emergency Notification System")

    # ── Emergency Contact ──────────────────────────────────────────────
    with st.sidebar.expander("\U0001f4de Emergency Contact", expanded=True):
        ec_name = st.text_input(
            "Contact Name",
            value=st.session_state.get('ec_name', ''),
            placeholder="e.g. Parent / Guardian",
            key="ec_name_input"
        )
        ec_phone = st.text_input(
            "Phone Number",
            value=st.session_state.get('ec_phone', ''),
            placeholder="+91XXXXXXXXXX",
            key="ec_phone_input"
        )
        ec_email_val = st.text_input(
            "Recipient Email (Alerts sent here)",
            value=st.session_state.get('ec_email', ''),
            placeholder="guardian@example.com",
            key="ec_email_input"
        )
        if ec_name != st.session_state.get('ec_name', ''):
            st.session_state.ec_name = ec_name
        if ec_phone != st.session_state.get('ec_phone', ''):
            st.session_state.ec_phone = ec_phone
        if ec_email_val != st.session_state.get('ec_email', ''):
            st.session_state.ec_email = ec_email_val

    # ── SMTP Email Server ──────────────────────────────────────────────
    with st.sidebar.expander("\U0001f4e7 Email Server (SMTP)", expanded=True):
        smtp_sender = st.text_input(
            "Sender Email (Gmail)",
            value=st.session_state.get('smtp_sender', ''),
            placeholder="guardianear@gmail.com",
            key="smtp_sender_input"
        )
        smtp_password = st.text_input(
            "Gmail App Password",
            value=st.session_state.get('smtp_password', ''),
            placeholder="xxxx xxxx xxxx xxxx",
            type="password",
            key="smtp_password_input",
            help="NOT your Gmail password. Go to Gmail → Security → 2FA → App Passwords"
        )
        smtp_host = st.text_input(
            "SMTP Server",
            value=st.session_state.get('smtp_host', 'smtp.gmail.com'),
            key="smtp_host_input"
        )
        smtp_port = st.number_input(
            "SMTP Port",
            value=int(st.session_state.get('smtp_port', 587)),
            min_value=1, max_value=65535,
            key="smtp_port_input"
        )
        # Update session state
        st.session_state.smtp_sender = smtp_sender
        st.session_state.smtp_password = smtp_password
        st.session_state.smtp_host = smtp_host
        st.session_state.smtp_port = smtp_port

        if st.button("\U0001f4be Save Configuration", use_container_width=True, key="save_config_btn"):
            # ── Validation ──
            errors = []
            if not ec_phone.strip():
                errors.append("Phone Number cannot be empty.")
            elif not re.match(r"^\d{10}$", ec_phone.strip()):
                errors.append("Phone number must contain exactly 10 digits.")
            
            if ec_email_val.strip() and not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", ec_email_val.strip()):
                errors.append("Enter a valid recipient email address.")
            
            if smtp_sender.strip() and not smtp_sender.strip().lower().endswith("@gmail.com"):
                errors.append("Sender Email must be a valid Gmail address.")
            
            try:
                p = int(smtp_port)
                if not (1 <= p <= 65535):
                    errors.append("SMTP Port must be between 1 and 65535.")
            except ValueError:
                errors.append("SMTP Port must be an integer.")

            if errors:
                for err in errors:
                    st.error(f"❌ {err}")
            else:
                ok, msg = _save_emergency_config(
                    name=ec_name, phone=ec_phone, email=ec_email_val,
                    smtp_sender=smtp_sender, smtp_password=smtp_password,
                    smtp_host=smtp_host, smtp_port=int(smtp_port),
                )
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    # ── Alert Channel Status ───────────────────────────────────────────
    with st.sidebar.expander("\U0001f514 Alert Channels", expanded=False):
        desktop_on = st.checkbox("\U0001f5a5\ufe0f Desktop Notifications", value=True, key="chk_desktop")
        _email_ready = bool(st.session_state.get('smtp_sender') and st.session_state.get('smtp_password'))
        _contact_ok = bool(st.session_state.get('ec_name') or st.session_state.get('ec_phone'))
        st.markdown(
            f"{'🟢' if desktop_on else '🔴'} Desktop  |  "
            f"{'🟢' if _email_ready else '🔴'} Email  |  "
            f"{'🟢' if _contact_ok else '🟡'} Contact"
        )
        if not st.session_state.get('smtp_sender'):
            st.caption("❌ Missing: Sender Email")
        if not st.session_state.get('smtp_password'):
            st.caption("❌ Missing: Gmail App Password")
        if not st.session_state.get('ec_email'):
            st.caption("❌ Missing: Recipient Email")

    # ── Test Buttons ────────────────────────────────────────────────────
    _test_result = st.session_state.get('emergency_test_status')
    if st.sidebar.button("\U0001f9ea Send Test Emergency Alert", key="emergency_test_btn", type="primary", use_container_width=True):
        try:
            eng = _get_emergency_engine()
            if eng:
                result = eng.send_test_alert()
                st.session_state.emergency_test_status = result
            else:
                st.session_state.emergency_test_status = {'error': 'Engine not available'}
        except Exception as _err:
            st.session_state.emergency_test_status = {'error': str(_err)}

    if _test_result:
        if isinstance(_test_result, dict) and 'error' not in _test_result:
            _desktop = _test_result.get('desktop', False)
            st.sidebar.success(f"\u2705 Test fired! Desktop: {'✓' if _desktop else '✗'}")
        elif isinstance(_test_result, dict):
            st.sidebar.error(f"\u274c {_test_result.get('error', 'Unknown error')}")
        st.session_state.emergency_test_status = None

    if st.sidebar.button("\U0001f4e7 Send Test Email", key="email_test_btn", use_container_width=True):
        st.session_state.email_success_count = st.session_state.get('email_success_count', 0)
        st.session_state.email_failure_count = st.session_state.get('email_failure_count', 0)
        # Validate fields before attempting SMTP
        _s = st.session_state.get('smtp_sender', '').strip()
        _p = st.session_state.get('smtp_password', '')
        _r = st.session_state.get('ec_email', '').strip()
        if not _s:
            st.sidebar.error("\u274c Missing Sender Email — fill in Email Server section above.")
        elif not _p:
            st.sidebar.error("\u274c Missing App Password — generate one in Gmail \u2192 Security \u2192 App Passwords.")
        elif not _r:
            st.sidebar.error("\u274c Missing Recipient Email — fill in Emergency Contact section above.")
        else:
            try:
                eng = _get_emergency_engine()
                if eng and eng._email.is_configured():
                    success = eng._email.send_test_email()
                    if success:
                        st.session_state.email_success_count += 1
                        st.session_state.last_email_sent = time.strftime("%H:%M:%S")
                        st.sidebar.success("\u2705 Test email delivered to " + _r)
                    else:
                        st.session_state.email_failure_count += 1
                        st.sidebar.error("\u274c SMTP Authentication Failed — check App Password is correct.")
                else:
                    st.sidebar.error("\u274c Engine not ready. Click Save Configuration first.")
            except Exception as _err:
                st.session_state.email_failure_count += 1
                st.sidebar.error(f"\u274c Error: {str(_err)}")

    st.sidebar.divider()
    # Live session statistics
    sb_metrics_placeholder = st.sidebar.empty()
    render_sidebar_metrics(sb_metrics_placeholder)

    # Set location override in state for the background Notification Engine
    st.session_state.location_override = location

    return location, threshold, page, sim_sound, sb_metrics_placeholder



# ─────────────────────────────────────────────────
# PAGE 1 — LIVE DETECTION
# ─────────────────────────────────────────────────
def page_live_detection(model, location, threshold, X_min, X_max, sim_sound="None", sb_metrics_placeholder=None):
    """Render the main live detection page."""
    st.markdown("<div class='main-title'>🎧 Guardian Ear</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>"
        "AI-Based Acoustic Anomaly Detection · Dual-Mode Intelligence System"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Metrics
    assessor = st.session_state.assessor
    history = assessor.load_alert_history()
    c1, c2, c3, c4, c5 = st.columns(5)

    total = len(history) if not history.empty else 0
    critical = len(history[history['threat_level'] == 'CRITICAL']) if not history.empty else 0
    alert_n = len(history[history['sound_mode'] == 'ALERT']) if not history.empty and 'sound_mode' in history.columns else 0
    assist_n = len(history[history['sound_mode'] == 'ASSISTIVE']) if not history.empty and 'sound_mode' in history.columns else 0

    c1.metric("📊 Total Alerts", total)
    c2.metric("🔴 Critical Tiers", critical)
    c3.metric("🚨 Alert Mode", alert_n)
    c4.metric("ℹ️ Assistive Mode", assist_n)
    c5.metric("📍 Active Loc", location.replace('_', ' ').title())

    st.divider()

    # Operations mode selection
    mode_option = st.radio(
        "Select Operation Mode",
        ["📁 Demo / Upload Mode", "🎙️ Live Surveillance Mode (Microphone)"],
        horizontal=True
    )

    if mode_option == "📁 Demo / Upload Mode":
        # Upload
        st.subheader("🎙️ Upload Audio for Detection")
        uploaded = st.file_uploader("Upload a .wav audio file", type=['wav'])

        if uploaded is not None:
            import librosa

            temp_path = os.path.join(str(_PROJECT_ROOT), '.tmp', 'dashboard_audio.wav')
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            with open(temp_path, 'wb') as f:
                f.write(uploaded.read())

            audio, sr = librosa.load(temp_path, sr=SAMPLE_RATE, duration=DURATION)
            st.audio(temp_path)

            with st.spinner("🔬 Extracting features..."):
                features, mel_db, mfcc, chroma = extract_features_for_dashboard(
                    audio, sr, X_min, X_max,
                )

            if model is not None:
                with st.spinner("\U0001f9e0 Running CRNN inference..."):
                    inp = features[np.newaxis, ..., np.newaxis]
                    preds = model.predict(inp, verbose=0)[0]
                    class_id = int(np.argmax(preds))
                    confidence = float(preds[class_id])
                    class_name = CLASS_NAMES[class_id]

                # ── Open-Set Rejection ─────────────────────────────────
                osc = st.session_state.get('open_set_clf')
                osc_result = osc.classify(preds) if osc else None
                is_known = osc_result['is_known'] if osc_result else True

                if not is_known:
                    st.divider()
                    st.subheader("\U0001f9ff Open-Set Rejection — MONITORING")
                    st.markdown(
                        "<div style='background:linear-gradient(135deg,#1e1b4b,#312e81);"
                        "border-left:5px solid #818cf8;padding:16px 20px;"
                        "border-radius:12px;margin:10px 0;'>"
                        "<b style='color:#a5b4fc;font-size:1.1rem;'>\U0001f441 MONITORING MODE</b><br/>"
                        "<span style='color:#c7d2fe;'>This sound does not confidently match any trained class. "
                        "Guardian Ear is watching but will NOT raise a false alert.</span>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    g1, g2, g3, g4 = st.columns(4)
                    g1.metric("Nearest Class", class_name.replace('_',' ').title())
                    g2.metric("Confidence", f"{osc_result['confidence']*100:.1f}%")
                    g3.metric("Top-2 Margin", f"{osc_result['top2_margin']*100:.1f}%")
                    g4.metric("Entropy", f"{osc_result['entropy']:.3f} nats")
                    st.info(
                        f"\U0001f6ab **Rejection reason:** {osc_result['rejection_reason']}\n\n"
                        "The model's confidence is too low or the prediction is too ambiguous "
                        "to classify reliably. This is the Open-Set Rejection working correctly — "
                        "it prevented a potential **false positive**."
                    )
                    st.caption(
                        "To adjust sensitivity, edit `open_set.confidence_threshold` in `configs/config.yaml`."
                    )
                    # Show probability chart so user can see the distribution
                    st.subheader("\U0001f4ca Class Probabilities (for reference)")
                    prob_df = pd.DataFrame({
                        'Sound Class': [c.replace('_', ' ').title() for c in CLASS_NAMES],
                        'Probability (%)': preds * 100,
                    }).sort_values('Probability (%)', ascending=False)
                    st.bar_chart(prob_df.set_index('Sound Class')['Probability (%)'])

                else:
                    # ── Known sound — continue normal threat pipeline ──
                    tracker = st.session_state.tracker
                    sound_mode = osc_result.get('sound_mode', get_sound_mode(class_name)) if osc_result else get_sound_mode(class_name)
                    description = get_mode_description(class_name)

                    tracker.add_detection(class_name)
                    pattern = tracker.get_pattern_summary(class_name)

                    threat_score = assessor.calculate_threat_score(
                        class_name, confidence, location,
                        pattern_score=pattern.pattern_score,
                    )
                    threat_level = assessor.get_threat_level(threat_score)
                    threat_color = assessor.get_threat_color(threat_level)

                    # Process Actions
                    process_priority_actions(class_name, confidence, threat_score, threat_level)

                    st.divider()

                    # Result
                    st.subheader("🎯 Detection Result")
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Sound", class_name.replace('_', ' ').title())
                    r2.metric("Confidence", f"{confidence * 100:.1f}%")
                    r3.metric("Threat", f"{threat_score}/100")
                    r4.metric("Level", threat_level)

                    # Mode
                    st.subheader("🔊 Sound Mode")
                    mode_class = f"mode-{sound_mode.lower()}" if sound_mode != 'NEUTRAL' else 'mode-neutral'
                    icon = '🚨' if sound_mode == 'ALERT' else 'ℹ️' if sound_mode == 'ASSISTIVE' else '👁'
                    st.markdown(
                        f"<div class='{mode_class}'>{icon} {sound_mode} MODE — {description}</div>",
                        unsafe_allow_html=True,
                    )

                    # Pattern
                    st.subheader("⏱️ Temporal Pattern Analysis")
                    p1, p2, p3, p4 = st.columns(4)
                    p1.metric("Pattern", pattern.pattern_label)
                    p2.metric("Score", f"{pattern.pattern_score:.2f}")
                    p3.metric("Detections", pattern.detection_count)
                    p4.metric("Duration", f"{pattern.duration_seconds}s")
                    st.progress(float(pattern.pattern_score))

                    if pattern.should_escalate:
                        st.markdown(
                            "<div class='escalation-warn'>"
                            "⚠️ ESCALATION — Sound pattern abnormally sustained!"
                            "</div>",
                            unsafe_allow_html=True,
                        )

                    # Threat box
                    st.subheader("🚨 Threat Assessment")
                    level_lower = threat_level.lower()
                    if level_lower in ('critical', 'high', 'medium', 'low'):
                        st.markdown(
                            f"<div class='alert-{level_lower}'>"
                            f"<b>{threat_level} ALERT</b><br>"
                            f"Sound: {class_name.replace('_',' ').title()}"
                            f" | Mode: {sound_mode}"
                            f" | Confidence: {confidence*100:.1f}%"
                            f" | Location: {location.replace('_',' ').title()}"
                            f" | Score: {threat_score}/100"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    # Action Engine log
                    st.subheader("🚨 Triggered Action")
                    if 'triggered_actions' in st.session_state and st.session_state.triggered_actions:
                        latest_action = st.session_state.triggered_actions[0]
                        st.info(f"**{latest_action['action_type']}** — {latest_action['details']}")
                    else:
                        st.success("No emergency action required.")

                    # Probability chart
                    st.subheader("📊 Class Probabilities")
                    prob_df = pd.DataFrame({
                        'Sound Class': [c.replace('_', ' ').title() for c in CLASS_NAMES],
                        'Probability (%)': preds * 100,
                    }).sort_values('Probability (%)', ascending=False)
                    st.bar_chart(prob_df.set_index('Sound Class')['Probability (%)'])

                # Grad-CAM
                st.subheader("🔍 Grad-CAM Explainability")
                with st.spinner("Generating heatmap..."):
                    heatmap = generate_gradcam(model, features, class_id)
                if heatmap is not None:
                    import tensorflow as tf
                    fig_gc, axes_gc = plt.subplots(1, 2, figsize=(14, 4))
                    fig_gc.patch.set_facecolor('#0e1117')
                    for ax in axes_gc:
                        ax.set_facecolor('#0e1117')

                    axes_gc[0].imshow(mel_db, aspect='auto', origin='lower', cmap='magma')
                    axes_gc[0].set_title('Mel Spectrogram', color='#e2e8f0')
                    axes_gc[1].imshow(mel_db, aspect='auto', origin='lower', cmap='magma')
                    hm = np.array(tf.image.resize(
                        heatmap[..., np.newaxis], [mel_db.shape[0], mel_db.shape[1]],
                    )).squeeze()
                    axes_gc[1].imshow(hm, aspect='auto', origin='lower', cmap='jet', alpha=0.5)
                    axes_gc[1].set_title('Grad-CAM Heatmap', color='#e2e8f0')
                    plt.tight_layout()
                    st.pyplot(fig_gc)
                    st.caption("🔴 Red regions = frequency areas that triggered detection")

                # Feature viz
                st.subheader("📈 Feature Visualization")
                fig = plot_features(mel_db, mfcc, chroma, audio, sr)
                st.pyplot(fig)

                # Actions
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Save Alert"):
                        assessor.generate_alert(
                            sound_class=class_name,
                            confidence=confidence,
                            location=location,
                            tracker=tracker,
                        )
                        st.success("Alert saved!")
                with col2:
                    if st.button("🔄 Reset Tracker"):
                        tracker.reset(class_name)
                        st.success("Tracker reset!")
            else:
                st.warning("⚠️ Model not found! Run training first.")

            if os.path.exists(temp_path):
                os.remove(temp_path)

    else:
        # LIVE SURVEILLANCE MODE
        st.subheader("🎙️ Live Acoustic Surveillance")
        st.markdown(
            "Continuous local microphone capture. Uses overlapping windows "
            "to perform inference every 1 second."
        )

        if model is None:
            st.warning("⚠️ Model not found! Cannot start surveillance.")
            return

        # Start/Stop Button triggers
        if not st.session_state.live_monitoring:
            if st.button("🟢 START LIVE SURVEILLANCE", use_container_width=True):
                st.session_state.live_monitoring = True
                st.session_state.session_start_time = time.time()
                st.session_state.session_status = "ACTIVE"
                st.session_state.session_detection_count = 0
                st.session_state.session_peak_threat = 0
                st.session_state.session_peak_source = "None"
                st.session_state.session_peak_time = "--:--:--"
                st.rerun()
        else:
            if st.button("🔴 STOP LIVE SURVEILLANCE", use_container_width=True):
                st.session_state.live_monitoring = False
                st.session_state.session_status = "STOPPED"
                if st.session_state.detector:
                    st.session_state.detector.stop()
                    st.session_state.detector = None
                st.rerun()

        # Display Placeholders
        wave_title = st.empty()
        wave_placeholder = st.empty()
        cols_placeholder = st.empty()
        status_placeholder = st.empty()
        actions_placeholder = st.empty()

        # Surveillance Loop
        if st.session_state.live_monitoring:
            # Start detector thread if not alive
            if st.session_state.detector is None:
                with st.spinner("Initializing Audio Input..."):
                    try:
                        import traceback as _tb
                        from src.utils.config_loader import load_config
                        cfg = load_config()
                        # Override default config options with live UI values
                        cfg['inference']['location'] = location
                        cfg['inference']['confidence_threshold'] = threshold / 100.0
                        
                        st.session_state.detector = RealTimeDetector(cfg, model=model)
                        st.session_state.detector.start()
                    except Exception as e:
                        st.session_state.session_status = "ERROR"
                        _full_tb = _tb.format_exc()
                        # Log full traceback to console/log file for debugging
                        import logging as _logging
                        _logging.getLogger('GuardianEar.dashboard').error(
                            'Live surveillance init failed:\n%s', _full_tb
                        )
                        # Show the REAL error (not a misleading 'microphone' message)
                        st.error(
                            f'**Live Surveillance startup failed.**\n\n'
                            f'**Error:** `{type(e).__name__}: {e}`\n\n'
                            f'**Full traceback** (expand to debug):\n```\n{_full_tb}\n```'
                        )
                        st.warning(
                            'If you see **\'No module named yaml\'**, run: '
                            '`guardian_env\\Scripts\\activate` then restart Streamlit.'
                        )
                        st.session_state.detector = None

            # Placeholder for diagnostics so it updates in-place without adding new elements
            diag_placeholder = st.empty()

            # Rerun loop
            while st.session_state.live_monitoring:
                # 1. Fetch data
                if sim_sound != "None":
                    st.session_state.session_status = "ACTIVE"
                    sim_map = {
                        "Gun Shot 🚨": "gun_shot",
                        "Siren \U0001f6a8": "siren",
                        "Child Crying \U0001f476": "child_crying",
                        "Running Tap Water \U0001f6b0": "water_flow",
                        "Dog Bark \u2139\ufe0f": "dog_bark",
                        "Car Horn \u2139\ufe0f": "car_horn"
                    }
                    mapped_class = sim_map[sim_sound]
                    preds = np.zeros(len(CLASS_NAMES))
                    if mapped_class in CLASS_NAMES:
                        preds[CLASS_NAMES.index(mapped_class)] = 0.99
                    elif mapped_class == 'child_crying':
                        preds[CLASS_NAMES.index('children_playing')] = 0.99
                    elif mapped_class == 'water_flow':
                        preds[CLASS_NAMES.index('drilling')] = 0.99
                    best_idx = int(np.argmax(preds))
                    latest = {
                        'class_name': CLASS_NAMES[best_idx],
                        'confidence': float(preds[best_idx]),
                        'all_probs': preds,
                        'mode': get_sound_mode(CLASS_NAMES[best_idx]),
                        'timestamp': time.time(),
                        'alert': True,
                        'voted': True,
                        'consecutive': 3,
                        'rms': 0.05,
                        'noise_floor': 0.001,
                        'threat_score': 0,
                        'threat_level': 'SAFE',
                    }
                    t_vals = np.linspace(0, 2, 8820)
                    raw_audio = 0.15 * np.random.normal(0, 0.1, 8820) + 0.3 * np.sin(2 * np.pi * 120 * t_vals)
                else:
                    if st.session_state.detector:
                        latest = st.session_state.detector.get_latest_prediction()
                        raw_audio = st.session_state.detector.get_latest_audio_samples(8820)
                    else:
                        latest = None
                        raw_audio = np.zeros(8820)

                # 2. Live Waveform
                wave_title.subheader("📈 Live Audio Stream")
                wave_placeholder.line_chart(raw_audio[::40], height=140)

                # 2.5 Diagnostics Panel
                if st.session_state.detector:
                    h = st.session_state.detector.get_system_health()
                    rms_val = h.get('current_rms', 0.0)
                    dyn_thresh = h.get('dynamic_threshold', 0.001)
                    
                    # Reconstruct gain logic for display
                    calc_gain = min(0.05 / max(rms_val, 1e-9), 5.0)
                    post_gain_rms = rms_val * calc_gain
                    
                    silence_triggered = rms_val < dyn_thresh
                    inf_enabled = not silence_triggered and post_gain_rms >= 0.0003
                    
                    # We can render this in a temporary st.info or metrics block inside a dedicated placeholder
                    diag_placeholder.markdown(
                        f"""
                        <div style='background-color: #1e1e1e; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 5px solid {"#ff4b4b" if silence_triggered else "#4CAF50"};'>
                            <b style='color: white;'>Live Surveillance Diagnostics (Temporary)</b><br/>
                            <code style='color: #a5b4fc;'>Raw RMS:</code> {rms_val:.5f} &nbsp;|&nbsp; 
                            <code style='color: #a5b4fc;'>Noise Floor:</code> {h.get('noise_floor', 0.0):.5f} &nbsp;|&nbsp; 
                            <code style='color: #a5b4fc;'>Dynamic Gate:</code> {dyn_thresh:.5f} <br/>
                            <code style='color: #a5b4fc;'>Applied Gain:</code> {calc_gain:.2f}x &nbsp;|&nbsp; 
                            <code style='color: #a5b4fc;'>Post-Gain RMS:</code> {post_gain_rms:.5f} <br/>
                            <code style='color: #a5b4fc;'>Silence Gate Triggered:</code> <b>{'YES (Ignored)' if silence_triggered else 'NO (Processing)'}</b> &nbsp;|&nbsp;
                            <code style='color: #a5b4fc;'>Inference Enabled:</code> <b>{'YES' if inf_enabled else 'NO'}</b>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

                # 3. System Health Monitor Row
                if st.session_state.detector:
                    health = st.session_state.detector.get_system_health()
                    with cols_placeholder.container():
                        if latest:
                            c_name = latest['class_name']
                            conf = latest['confidence']
                            s_mode = latest['mode']
                            conf_threshold_val = threshold / 100.0
                            rms_val = latest.get('rms', 0.0)
                            noise_fl = latest.get('noise_floor', 0.0)
                            voted = latest.get('voted', False)
                            consecutive = latest.get('consecutive', 0)

                            # Dynamically update status based on what is heard
                            if c_name == 'unknown':
                                st.session_state.session_status = "MONITORING"
                            elif c_name != 'silence':
                                st.session_state.session_status = "ACTIVE"

                            # Always use threat score from engine if available
                            if c_name == 'silence':
                                threat_score = 0
                                threat_level = "SAFE"
                                display_name = "Silence"
                            elif c_name == 'child_crying':
                                threat_score = 85
                                threat_level = "HIGH"
                                display_name = "Child Crying (Distress)"
                                s_mode = "ASSISTIVE"
                            elif c_name == 'water_flow':
                                threat_score = 65
                                threat_level = "MEDIUM"
                                display_name = "Running Tap Water"
                                s_mode = "ASSISTIVE"
                            else:
                                display_name = c_name.replace('_', ' ').title()
                                threat_score = latest.get('threat_score',
                                    assessor.calculate_threat_score(c_name, conf, location))
                                threat_level = latest.get('threat_level',
                                    assessor.get_threat_level(threat_score))

                            # Only trigger actions when voted + above threshold
                            if c_name not in ('silence', 'child_crying', 'water_flow', 'unknown') and \
                               conf >= conf_threshold_val and voted:
                                osc_res = latest.get('osc_result', {})
                                is_known = osc_res.get('is_known', True) if isinstance(osc_res, dict) else True
                                o_status = "KNOWN" if is_known else "UNKNOWN"
                                process_priority_actions(
                                    c_name, conf, threat_score, threat_level,
                                    open_set_status=o_status, vote_count=consecutive
                                )

                            # ── Open-Set Rejection UI (Live Mode) ─────────────────
                            osc_res = latest.get('osc_result')
                            if osc_res and not osc_res.get('is_known', True):
                                st.markdown(
                                    "<div style='background:linear-gradient(135deg,#1e1b4b,#312e81);"
                                    "border-left:5px solid #818cf8;padding:12px 16px;"
                                    "border-radius:10px;margin-bottom:12px;'>"
                                    "<b style='color:#a5b4fc;font-size:1.0rem;'>\U0001f441 MONITORING MODE: "
                                    f"Rejected '{osc_res['class_name'].replace('_', ' ').title()}'</b><br/>"
                                    f"<span style='color:#c7d2fe;font-size:0.9rem;'>{osc_res['rejection_reason']}</span>"
                                    "</div>",
                                    unsafe_allow_html=True,
                                )

                            # ── Detection metrics row ─────────────────────────
                            r1, r2, r3, r4 = st.columns(4)
                            r1.metric("\U0001f50a Live Sound", display_name)
                            r2.metric("Confidence", f"{conf * 100:.1f}%" if c_name not in ('silence', 'unknown') else "\u2014")
                            r3.metric("Threat Score", f"{threat_score}/100")
                            r4.metric("Threat Level", threat_level)

                            # ── System health row ─────────────────────────────
                            h1, h2, h3, h4 = st.columns(4)
                            h1.metric("\U0001f3a4 Mic RMS", f"{rms_val:.4f}")
                            h2.metric("Noise Floor", f"{noise_fl:.4f}")
                            h3.metric("Chunks", health['chunk_count'])
                            h4.metric("Vote Status",
                                      f"{consecutive}/{st.session_state.detector._VOTE_WINDOW}"
                                      if c_name != 'silence' else "\u2014")

                            # ── Noise level bar ───────────────────────────────
                            dyn_thresh = health['dynamic_threshold']
                            rms_clamped = min(rms_val / max(dyn_thresh * 5, 0.01), 1.0)
                            noise_color = (
                                "#ef4444" if rms_clamped > 0.7 else
                                "#f59e0b" if rms_clamped > 0.3 else "#22c55e"
                            )
                            st.markdown(
                                f"<div style='margin: 4px 0 8px; font-size: 0.82rem; color:#94a3b8;'>"
                                f"\U0001f50a Noise Level"
                                f"<div style='background:#1e293b; border-radius:6px; height:10px; margin-top:4px;'>"
                                f"<div style='width:{rms_clamped*100:.0f}%; background:{noise_color}; "
                                f"height:100%; border-radius:6px;'></div></div></div>",
                                unsafe_allow_html=True,
                            )

                            if c_name not in ('silence',) and conf < conf_threshold_val:
                                st.info(
                                    f"\u2139\ufe0f Model detected **{display_name}** at **{conf*100:.1f}%** "
                                    f"\u2014 below the **{threshold}%** alert threshold. Monitoring active."
                                )
                            if c_name not in ('silence',) and not voted:
                                st.caption(
                                    f"\u23f3 Temporal voting: {consecutive}/{st.session_state.detector._VOTE_WINDOW} "
                                    f"consecutive detections needed before escalation."
                                )
                        else:
                            st.info("\U0001f399\ufe0f Calibrating live audio stream... (~3 seconds for first chunk)")

                    # ── Active threat status ───────────────────────────────
                    if latest and latest.get('class_name') not in ('silence', None):
                        threat_level_now = latest.get('threat_level', 'SAFE')
                        with status_placeholder.container():
                            st.subheader("\U0001f4e2 Active Monitoring Status")
                            s_mode_now = latest.get('mode', 'NEUTRAL')
                            icon = '\U0001f6a8' if s_mode_now == 'ALERT' else '\u2139\ufe0f' if s_mode_now == 'ASSISTIVE' else '\U0001f441'
                            mode_class = f"mode-{s_mode_now.lower()}" if s_mode_now != 'NEUTRAL' else 'mode-neutral'
                            disp_name = latest['class_name'].replace('_', ' ').title()
                            st.markdown(
                                f"<div class='{mode_class}'>{icon} {s_mode_now} MODE \u2014 {disp_name} detected.</div>",
                                unsafe_allow_html=True,
                            )
                            if threat_level_now != "SAFE" and latest.get('voted', False):
                                st.markdown(
                                    f"<div class='alert-{threat_level_now.lower()}'>"
                                    f"<b>{threat_level_now} THREAT</b><br>"
                                    f"Sound: {disp_name} | Location: {location.replace('_',' ').title()} "
                                    f"| Score: {latest.get('threat_score', 0)}/100"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                    # ── Live Event Timeline (last 20) ──────────────────────
                    history_list = st.session_state.detector.get_detection_history()
                    if history_list:
                        with actions_placeholder.container():
                            col_tl, col_act = st.columns([1, 1])
                            with col_tl:
                                st.subheader("\U0001f4cb Live Event Timeline")
                                for ev in history_list[:10]:
                                    level_col = (
                                        "#ef4444" if ev['threat_level'] == 'CRITICAL' else
                                        "#f97316" if ev['threat_level'] == 'HIGH' else
                                        "#eab308" if ev['threat_level'] == 'MEDIUM' else
                                        "#22c55e" if ev['threat_level'] == 'LOW' else
                                        "#475569"
                                    )
                                    icon_ev = '\U0001f6a8' if ev['mode'] == 'ALERT' else '\u2139\ufe0f' if ev['mode'] == 'ASSISTIVE' else '\U0001f4a4'
                                    st.markdown(
                                        f"<div style='border-left:3px solid {level_col}; "
                                        f"padding: 4px 10px; margin: 3px 0; font-size:0.85rem;'>"
                                        f"{icon_ev} <b>{ev['class_name'].replace('_',' ').title()}</b> "
                                        f"<span style='color:#94a3b8'>{ev['confidence']:.0f}% | {ev['timestamp']}</span>"
                                        f"</div>",
                                        unsafe_allow_html=True,
                                    )

                            with col_act:
                                st.subheader("\U0001f6a8 Live Action Dispatch Log")
                                if st.session_state.triggered_actions:
                                    for action in st.session_state.triggered_actions[:4]:
                                        card_border = (
                                            "#ef4444" if "\U0001f6a8" in action['action_type'] else
                                            "#f59e0b" if "\U0001f4e7" in action['action_type'] else
                                            "#3b82f6"
                                        )
                                        st.markdown(
                                            f"<div style='border-left:5px solid {card_border}; "
                                            f"padding-left:12px; margin:8px 0;'>"
                                            f"<b>{action['action_type']}</b> \u2014 <small>{action['timestamp']}</small><br>"
                                            f"<i>{action['details']}</i><br>"
                                            f"<small>Threat: {action['threat_level']} ({action['threat_score']}/100)</small>"
                                            f"</div>",
                                            unsafe_allow_html=True,
                                        )
                                else:
                                    st.info("Continuous monitoring active. No actions triggered yet.")

                            # Confidence trend chart
                            if len(history_list) >= 3:
                                trend_data = [
                                    ev['confidence'] for ev in reversed(history_list[:15])
                                    if ev['class_name'] != 'silence'
                                ]
                                if trend_data:
                                    st.subheader("\U0001f4c8 Confidence Trend")
                                    st.line_chart(
                                        pd.DataFrame({'Confidence (%)': trend_data}),
                                        height=120,
                                    )
                    else:
                        with actions_placeholder.container():
                            st.info("Continuous monitoring active. No security or safety actions triggered yet.")
                else:
                    # Detector not yet initialised
                    with cols_placeholder.container():
                        st.info("\U0001f399\ufe0f Calibrating live audio stream... (~3 seconds for first chunk)")

                render_sidebar_metrics(sb_metrics_placeholder)
                time.sleep(1.0)


# ─────────────────────────────────────────────────
# PAGE 1.5 — ♿ ASSISTIVE HEARING MODE
# ─────────────────────────────────────────────────
def page_assistive_hearing_mode(model, location, threshold, X_min, X_max, sim_sound="None", sb_metrics_placeholder=None):
    """Render the accessibility page for the hearing-impaired."""
    # CRITICAL FIX: bind assessor from session_state — was missing, causing NameError
    assessor = st.session_state.assessor

    st.markdown("<div class='main-title'>&#9851; Assistive Hearing Mode</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>"
        "Designed to assist hearing-impaired users by translating critical environmental sounds into clear visual and haptic feedback."
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Enable simulation message
    if sim_sound == "None" and not st.session_state.live_monitoring:
        st.info("💡 Tip: Use the 'Live Demo Simulator' in the sidebar to inject sounds and test visual alarms instantly!")

    # Start/Stop Button triggers
    if not st.session_state.live_monitoring:
        if st.button("🟢 ACTIVATE ASSISTIVE STREAM", use_container_width=True):
            st.session_state.live_monitoring = True
            st.session_state.session_start_time = time.time()
            st.session_state.session_status = "ACTIVE"
            st.session_state.session_detection_count = 0
            st.session_state.session_peak_threat = 0
            st.session_state.session_peak_source = "None"
            st.session_state.session_peak_time = "--:--:--"
            st.rerun()
    else:
        if st.button("🔴 DEACTIVATE ASSISTIVE STREAM", use_container_width=True):
            st.session_state.live_monitoring = False
            st.session_state.session_status = "STOPPED"
            if st.session_state.detector:
                st.session_state.detector.stop()
                st.session_state.detector = None
            st.rerun()

    # Visual Placeholders
    flash_placeholder = st.empty()
    alert_detail_placeholder = st.empty()
    vibe_placeholder = st.empty()

    if st.session_state.live_monitoring:
        # Start detector thread if not alive
        if st.session_state.detector is None:
            with st.spinner("Initializing Audio Input..."):
                try:
                    import traceback as _tb
                    from src.utils.config_loader import load_config
                    cfg = load_config()
                    cfg['inference']['location'] = location
                    cfg['inference']['confidence_threshold'] = threshold / 100.0
                    
                    st.session_state.detector = RealTimeDetector(cfg, model=model)
                    st.session_state.detector.start()
                except Exception as e:
                    st.session_state.session_status = "ERROR"
                    _full_tb = _tb.format_exc()
                    import logging as _logging
                    _logging.getLogger('GuardianEar.dashboard').error(
                        'Assistive stream init failed:\n%s', _full_tb
                    )
                    st.error(
                        f'**Assistive Stream startup failed.**\n\n'
                        f'**Error:** `{type(e).__name__}: {e}`\n\n'
                        f'**Full traceback:**\n```\n{_full_tb}\n```'
                    )
                    st.warning(
                        'If you see **\'No module named yaml\'**, run: '
                        '`guardian_env\\Scripts\\activate` then restart Streamlit.'
                    )
                    st.session_state.detector = None

        # Surveillance Loop
        while st.session_state.live_monitoring:
            # 1. Fetch data
            if sim_sound != "None":
                st.session_state.session_status = "ACTIVE"
                sim_map = {
                    "Gun Shot 🚨": "gun_shot",
                    "Siren 🚨": "siren",
                    "Child Crying 👶": "child_crying",
                    "Running Tap Water 🚰": "water_flow",
                    "Dog Bark ℹ️": "dog_bark",
                    "Car Horn ℹ️": "car_horn"
                }
                mapped_class = sim_map[sim_sound]
                preds = np.zeros(10)
                if mapped_class in CLASS_NAMES:
                    mapped_idx = CLASS_NAMES.index(mapped_class)
                    preds[mapped_idx] = 0.99
                elif mapped_class == 'child_crying':
                    mapped_idx = CLASS_NAMES.index('children_playing')
                    preds[mapped_idx] = 0.99
                elif mapped_class == 'water_flow':
                    mapped_idx = CLASS_NAMES.index('drilling')
                    preds[mapped_idx] = 0.99
                
                latest = {
                    'class_name': mapped_class,
                    'confidence': 0.99,
                    'all_probs': preds,
                    'mode': get_sound_mode(mapped_class) if mapped_class in CLASS_NAMES else 'ASSISTIVE',
                    'timestamp': time.time(),
                    'alert': True
                }
            else:
                if st.session_state.detector:
                    latest = st.session_state.detector.get_latest_prediction()
                else:
                    latest = None

            # 2. Process Prediction
            if latest:
                c_name = latest['class_name']
                conf = latest['confidence']
                s_mode = latest['mode']
                
                # ———————————————————————————————————————————
                # Classify prediction and compute threat score
                # SAFETY: silence must ALWAYS short-circuit before calculate_threat_score
                # ———————————————————————————————————————————
                # Dynamically update status based on what is heard
                if c_name == 'unknown':
                    st.session_state.session_status = "MONITORING"
                elif c_name != 'silence':
                    st.session_state.session_status = "ACTIVE"

                # Default safe values — always defined so display never crashes
                threat_score = 0
                threat_level = "SAFE"
                display_name = c_name.replace('_', ' ').title()
                caption = ""
                bg_color = "linear-gradient(135deg, #111827, #1f2937)"
                text_color = "#9ca3af"

                if c_name == 'silence':
                    # Silence: zero threat, no alerts, never reaches threat engine
                    threat_score = 0
                    threat_level = "SAFE"
                    display_name = "Silence"
                    caption = "Your environment is quiet."
                    bg_color = "linear-gradient(135deg, #111827, #1f2937)"
                    text_color = "#9ca3af"
                elif c_name == 'child_crying':
                    threat_score = 85
                    threat_level = "HIGH"
                    s_mode = "ASSISTIVE"
                    display_name = "Child Crying (Distress)"
                    caption = "&#128118; WARNING: A baby crying distress signature has been detected. Check immediately!"
                    bg_color = "linear-gradient(135deg, #1e3a8a, #3b82f6)"
                    text_color = "#bfdbfe"
                elif c_name == 'water_flow':
                    threat_score = 65
                    threat_level = "MEDIUM"
                    s_mode = "ASSISTIVE"
                    display_name = "Running Tap Water (Wastage)"
                    caption = "&#128688; NOTICE: Continuous running water detected. Verify if a tap is left open."
                    bg_color = "linear-gradient(135deg, #0c4a6e, #0ea5e9)"
                    text_color = "#e0f2fe"
                else:
                    # Real sound class — safe to call calculate_threat_score
                    display_name = c_name.replace('_', ' ').title()
                    tracker = st.session_state.tracker
                    tracker.add_detection(c_name)
                    pattern = tracker.get_pattern_summary(c_name)
                    threat_score = assessor.calculate_threat_score(
                        c_name, conf, location,
                        pattern_score=pattern.pattern_score,
                    )
                    threat_level = assessor.get_threat_level(threat_score)

                    if s_mode == "ALERT":
                        caption = f"&#128680; URGENT: Emergency sound '{display_name}' detected! Seek safety immediately."
                        bg_color = "linear-gradient(135deg, #7f1d1d, #ef4444)"
                        text_color = "#fca5a5"
                    else:
                        caption = f"&#8505;&#65039; INFO: Assistive sound '{display_name}' detected nearby."
                        bg_color = "linear-gradient(135deg, #1e293b, #334155)"
                        text_color = "#cbd5e1"
                
                # Trigger actions
                if c_name != 'silence':
                    process_priority_actions(c_name, conf, threat_score, threat_level)

                # Render Flashing Banner
                with flash_placeholder.container():
                    st.markdown(
                        f"<div style='background: {bg_color}; color: {text_color}; "
                        f"padding: 30px; border-radius: 16px; text-align: center; "
                        f"box-shadow: 0 4px 20px rgba(0,0,0,0.5); border: 2px solid {text_color}80;'>"
                        f"<h1 style='font-size: 3.5rem; margin: 0; color: white;'>{display_name.upper()}</h1>"
                        f"<p style='font-size: 1.4rem; margin-top: 15px; font-weight: bold;'>{caption}</p>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # Render details and vibration trigger
                with alert_detail_placeholder.container():
                    col_info, col_act = st.columns(2)
                    with col_info:
                        st.subheader("📋 Sound Details")
                        st.markdown(f"**Sound Class:** `{display_name}`")
                        st.markdown(f"**Inference Confidence:** `{conf * 100:.1f}%`" if c_name != 'silence' else "**Confidence:** `100%`")
                        st.markdown(f"**Assessed Threat level:** `{threat_level}` (`{threat_score}/100`)")
                    
                    with col_act:
                        st.subheader("⚡ Automated Responses")
                        if c_name != 'silence':
                            latest_act = st.session_state.triggered_actions[0] if st.session_state.triggered_actions else None
                            if latest_act:
                                st.markdown(f"**Action:** `{latest_act['action_type']}`")
                                st.markdown(f"**Details:** *{latest_act['details']}*")
                            else:
                                st.write("Monitoring environmental baselines.")
                        else:
                            st.write("Listening for anomalies...")

                # HTML5 Vibration trigger using javascript injection (runs on companion browser client)
                if threat_level in ["HIGH", "CRITICAL"]:
                    with vibe_placeholder.container():
                        st.markdown(
                            """
                            <script>
                            if (navigator.vibrate) {
                                // Vibrate: 400ms on, 200ms off, 400ms on
                                navigator.vibrate([400, 200, 400]);
                            }
                            console.log('♿ Guardian Ear: Haptic alert vibration triggered!');
                            </script>
                            """,
                            unsafe_allow_html=True
                        )
            else:
                with flash_placeholder.container():
                    st.info("Connecting stream... Adjust simulator or make noise to test.")

            render_sidebar_metrics(sb_metrics_placeholder)
            time.sleep(1.0)


# ─────────────────────────────────────────────────
# PAGE 2 — ALERT HISTORY
# ─────────────────────────────────────────────────
def page_alert_history():
    """Render the alert history page with filtering."""
    st.title("📋 Alert History")

    history = st.session_state.assessor.load_alert_history()
    if history.empty:
        st.info("No alerts recorded yet.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", len(history))
    c2.metric("Critical", len(history[history['threat_level'] == 'CRITICAL']))
    c3.metric("High", len(history[history['threat_level'] == 'HIGH']))
    c4.metric("Avg Score", f"{history['threat_score'].mean():.1f}")

    st.divider()

    # Filters
    f1, f2 = st.columns(2)
    with f1:
        levels = st.multiselect("Threat Level", ['CRITICAL','HIGH','MEDIUM','LOW','SAFE'], default=['CRITICAL','HIGH'])
    with f2:
        classes = st.multiselect("Sound Class", CLASS_NAMES, default=CLASS_NAMES)

    filtered = history[
        history['threat_level'].isin(levels) &
        history['sound_class'].isin(classes)
    ]

    st.subheader(f"Showing {len(filtered)} alerts")
    display_cols = ['timestamp', 'sound_class', 'threat_score', 'threat_level']
    if 'sound_mode' in filtered.columns:
        display_cols.insert(2, 'sound_mode')
    if 'confidence' in filtered.columns:
        display_cols.append('confidence')

    st.dataframe(
        filtered[display_cols].sort_values('timestamp', ascending=False),
        use_container_width=True,
    )

    if len(history) > 1:
        st.subheader("📈 Threat Score Trend")
        st.line_chart(history.set_index('timestamp')['threat_score'])

    csv = filtered.to_csv(index=False)
    st.download_button("⬇️ Download CSV", csv, "guardian_ear_alerts.csv", "text/csv")


# ─────────────────────────────────────────────────
# PAGE 3 — FEATURE VISUALIZATION
# ─────────────────────────────────────────────────
def page_feature_visualization():
    """Render the standalone feature visualization page."""
    st.title("📊 Feature Visualization")
    st.info("Upload an audio file to visualize extracted features.")

    uploaded = st.file_uploader("Upload .wav file", type=['wav'], key='viz_upload')
    if uploaded:
        import librosa

        temp_path = os.path.join(str(_PROJECT_ROOT), '.tmp', 'viz_audio.wav')
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        with open(temp_path, 'wb') as f:
            f.write(uploaded.read())

        audio, sr = librosa.load(temp_path, sr=SAMPLE_RATE, duration=DURATION)
        st.audio(temp_path)

        X_min, X_max = load_normalization()
        features, mel_db, mfcc, chroma = extract_features_for_dashboard(audio, sr, X_min, X_max)

        fig = plot_features(mel_db, mfcc, chroma, audio, sr)
        st.pyplot(fig)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Mel Shape", f"{mel_db.shape[0]}×{mel_db.shape[1]}")
        s2.metric("MFCC Shape", f"{mfcc.shape[0]}×{mfcc.shape[1]}")
        s3.metric("Chroma Shape", f"{chroma.shape[0]}×{chroma.shape[1]}")
        s4.metric("Fused", "180×130")

        if os.path.exists(temp_path):
            os.remove(temp_path)

# ─────────────────────────────────────────────────
# PAGE — SYSTEM INFO (PROJECT OVERVIEW)
# ─────────────────────────────────────────────────
def page_system_info(model):
    """Render the simplified, presentation-safe Project Overview page."""
    st.markdown("<div class='main-title'>\u2139\ufe0f Project Overview</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Guardian Ear \u00b7 AI-Based Environmental Awareness & Emergency Response System</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Row 1: Core Project Metrics
    st.subheader("\U0001f4cc Project Identity")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Project", "Guardian Ear")
    col2.metric("Model", "Attention-CRNN v2")
    col3.metric("Dataset", "UrbanSound8K")
    col4.metric("Sound Classes", "10")
    col5.metric("Input Shape", "(180 x 130 x 1)")

    st.divider()

    # Row 2: Feature Extraction & Detection Modes
    r2a, r2b = st.columns(2)

    with r2a:
        st.subheader("\U0001f52c Feature Extraction Methods")
        st.markdown(
            "| Feature | Description |\n"
            "|---|---|\n"
            "| **MFCC** | 40 Mel-Frequency Cepstral Coefficients |\n"
            "| **Mel Spectrogram** | 128 Mel-scale frequency bins |\n"
            "| **Chroma STFT** | 12 pitch class features |\n"
        )

    with r2b:
        st.subheader("\U0001f39b\ufe0f Detection Modes")
        st.markdown(
            "| Mode | Purpose |\n"
            "|---|---|\n"
            "| **Live Detection** | Real-time microphone surveillance with threat scoring |\n"
            "| **Assistive Hearing** | Aids deaf/hard-of-hearing users with contextual alerts |\n"
            "| **Demo Upload Mode** | Classify pre-recorded WAV/MP3 files for testing |\n"
        )

    st.divider()

    # Row 3: Advanced AI Features
    st.subheader("\U0001f916 Advanced AI & Safety Features")
    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown(
            "**Open-Set Recognition**\n\n"
            "Prevents unknown sounds from generating false alerts. "
            "Uses entropy thresholding + top-2 margin gating.\n\n"
            "**Temporal Voting**\n\n"
            "Requires consistent predictions across 3 of 5 consecutive frames before raising an alert."
        )
    with f2:
        st.markdown(
            "**Context-Aware Threat Scoring**\n\n"
            "Multi-factor threat score (0-100) based on model confidence, time of day, "
            "location sensitivity, sound type risk, and temporal pattern.\n\n"
            "**Grad-CAM Explainability**\n\n"
            "Heatmaps over the Mel Spectrogram show which frequency bands drove the decision."
        )
    with f3:
        st.markdown(
            "**Emergency Response Center**\n\n"
            "Real-time notification dispatch via desktop alerts and email. "
            "Rate-limited with 60-second cooldown protection.\n\n"
            "**CSV Audit Trail**\n\n"
            "Every dispatch logged to `alerts/notification_logs.csv` with open-set status and vote count."
        )

    st.divider()

    # Row 4: Deployment Readiness
    st.subheader("\U0001f680 Deployment Readiness")
    d1, d2 = st.columns(2)

    with d1:
        st.markdown(
            "| Deployment Target | Status |\n"
            "|---|---|\n"
            "| Desktop (Windows/Linux/Mac) | Ready |\n"
            "| Docker Container | Dockerfile included |\n"
            "| FastAPI REST API | Integrated |\n"
            "| Edge Device (Raspberry Pi 4) | TFLite Ready |\n"
            "| TensorFlow Lite (INT8) | Model exported |\n"
            "| ONNX Runtime | Model exported |\n"
        )
    with d2:
        st.markdown(
            "| Future Integration | Path |\n"
            "|---|---|\n"
            "| SMS Alerts | Twilio / MSG91 |\n"
            "| Voice Calls | Twilio TwiML |\n"
            "| Microsoft Teams | Adaptive Card Webhook |\n"
            "| Slack | Block Kit Webhook |\n"
            "| Mobile Push | Firebase FCM |\n"
        )

    st.divider()

    st.info(
        "\U0001f512 **Privacy & Data Security:** Guardian Ear runs 100% locally on-device. "
        "No audio is transmitted over the internet. "
        "Microphone input is held in a volatile in-memory buffer and overwritten every 3 seconds."
    )

# ─────────────────────────────────────────────────
# EMERGENCY RESPONSE CENTER PAGE
# ─────────────────────────────────────────────────
def page_emergency_response_center():
    """Emergency Response Center — notification channels, escalation log, contact status."""
    st.markdown("<div class='main-title'>\U0001f6a8 Emergency Response Center</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Notification Engine \u00b7 Escalation Log \u00b7 Contact Management</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Channel Status Row ──────────────────────────────────────────────────
    st.subheader("\U0001f514 Alert Channel Status")
    eng = _get_emergency_engine()

    c1, c2, c3, c4 = st.columns(4)
    st.session_state.email_success_count = st.session_state.get('email_success_count', 0)
    st.session_state.email_failure_count = st.session_state.get('email_failure_count', 0)
    st.session_state.last_email_sent = st.session_state.get('last_email_sent', 'Never')
    _ec_name = st.session_state.get('ec_name', '')
    _ec_phone = st.session_state.get('ec_phone', '')
    _ec_email = st.session_state.get('ec_email', '')

    desktop_available = False
    email_configured = False
    contact_configured = bool(_ec_name or _ec_phone)

    try:
        from src.notifications.desktop_notifier import DesktopNotifier
        desktop_available = DesktopNotifier().is_available()
    except Exception:
        pass

    if eng:
        email_configured = eng.is_email_configured()
        contact_configured = eng.is_contact_configured() or contact_configured

    c1.metric("\U0001f5a5\ufe0f Desktop", "ACTIVE" if desktop_available else "FALLBACK")
    c2.metric("\U0001f4e7 Email", "CONFIGURED" if email_configured else "NOT SET")
    c3.metric("\U0001f4de Contact", "READY" if contact_configured else "NOT SET")
    c4.metric("\U0001f4e8 Notifications Sent", len(st.session_state.get('notification_log', [])))

    st.divider()
    
    st.subheader("\U0001f4ca Email Delivery Metrics")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Status", "ACTIVE" if email_configured else "DISABLED")
    e2.metric("Last Sent", st.session_state.get('last_email_sent', 'Never'))
    e3.metric("Success Count", st.session_state.get('email_success_count', 0))
    e4.metric("Failure Count", st.session_state.get('email_failure_count', 0))

    st.divider()

    # ── Emergency Contact Card ──────────────────────────────────────────────
    st.subheader("\U0001f464 Emergency Contact Information")
    col_a, col_b = st.columns(2)
    with col_a:
        if _ec_name or _ec_phone or _ec_email:
            st.markdown(
                f"<div style='background:linear-gradient(135deg,#0f3460,#16213e);"
                f"border-left:4px solid #e94560;padding:16px 20px;"
                f"border-radius:12px;margin-bottom:16px;'>"
                f"<b style='color:#e94560;font-size:1.1rem;'>\U0001f4de {_ec_name or 'Contact'}</b><br/>"
                f"<span style='color:#a8b2d8;'>\U0001f4f1 {_ec_phone or 'No phone'}</span><br/>"
                f"<span style='color:#a8b2d8;'>\U0001f4e7 {_ec_email or 'No email'}</span><br/>"
                f"<span style='color:#4ade80;font-size:0.9rem;'>\U0001f7e2 Escalation Ready</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("No emergency contact configured. Add contact details in the sidebar.")
    with col_b:
        st.markdown("**\U0001f6a8 Future Integration Path**")
        st.markdown(
            "- \U0001f4f1 **SMS** via Twilio / MSG91\n"
            "- \U0001f4de **Voice Call** via Google Voice API\n"
            "- \U0001f4ac **WhatsApp** via Meta Business API\n"
            "- \U0001f4f2 **Mobile Push** via Firebase FCM"
        )

    st.divider()

    # ── Live Test Panel ─────────────────────────────────────────────────────
    st.subheader("\U0001f9ea Live Test Emergency Alert")
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        test_sound = st.selectbox("Simulate Sound", [
            "gun_shot", "siren", "jackhammer", "children_playing", "dog_bark"
        ], key="erc_test_sound")
        test_conf = st.slider("Confidence", 50, 100, 95, key="erc_test_conf") / 100.0
        test_loc = st.selectbox("Location", LOCATIONS, key="erc_test_loc")
    with col_t2:
        st.markdown("<br/>", unsafe_allow_html=True)
        if st.button("\U0001f9ea Fire Test Alert", key="erc_fire_btn", type="primary"):
            import threading
            result_holder = {}
            def _test_dispatch():
                try:
                    e = _get_emergency_engine()
                    if e:
                        rec = e.dispatch(
                            sound_class=test_sound,
                            confidence=test_conf,
                            threat_score=85.0,
                            threat_level="CRITICAL",
                            location=test_loc.replace('_',' ').title(),
                            open_set_status="KNOWN",
                            vote_count=1,
                        )
                        st.session_state.notification_log.insert(0, rec)
                        st.session_state.notification_log = st.session_state.notification_log[:100]
                except Exception:
                    pass
            threading.Thread(target=_test_dispatch, daemon=True).start()
            st.success("\u2705 Test alert dispatched! Desktop notification should appear momentarily.")

    st.divider()

    # ── Notification Log ───────────────────────────────────────────────────
    st.subheader("\U0001f4cb Emergency Escalation Log")
    notif_log = st.session_state.get('notification_log', [])

    if notif_log:
        for rec in notif_log[:20]:
            _sc = rec.get('sound_class', '?').replace('_',' ').title()
            _tl = rec.get('threat_level', 'UNKNOWN')
            _ts = rec.get('timestamp', '?')
            _cname = rec.get('contact_name', 'N/A')
            _cstatus = rec.get('contact_status', 'N/A')
            _desktop = rec.get('desktop_sent', False)
            _email = rec.get('email_sent', False)
            _color = '#ef4444' if _tl == 'CRITICAL' else '#f59e0b' if _tl == 'HIGH' else '#3b82f6'
            _desktop_icon = "✓" if _desktop else "✗"
            _email_icon = "✓" if _email else "✗"
            _html_card = (
                f"<div style='background:#0f172a;border-left:4px solid {_color};"
                f"padding:10px 16px;border-radius:8px;margin-bottom:8px;'>"
                f"<b style='color:{_color};'>{_sc}</b> "
                f"<span style='color:#94a3b8;font-size:0.85rem;'>{_tl} \u00b7 {_ts}</span><br/>"
                f"<span style='color:#64748b;font-size:0.82rem;'>"
                f"\U0001f464 {_cname} \u2014 {_cstatus}&nbsp;&nbsp;"
                f"\U0001f5a5\ufe0f {_desktop_icon} Desktop&nbsp;"
                f"\U0001f4e7 {_email_icon} Email"
                "</span></div>"
            )
            st.markdown(_html_card, unsafe_allow_html=True)

    else:
        st.info("\U0001f4ec No emergency escalations yet. Alerts will appear here when a critical threat is detected.")

    # ── Also show triggered_actions as overview ─────────────────────────────
    triggered = st.session_state.get('triggered_actions', [])
    if triggered:
        st.divider()
        st.subheader("\U0001f4cb Action Dispatch Log (Current Session)")
        action_df = pd.DataFrame(triggered)[[
            'timestamp', 'sound_class', 'threat_level', 'threat_score', 'action_type', 'notification_type'
        ] if 'notification_type' in triggered[0] else [
            'timestamp', 'sound_class', 'threat_level', 'threat_score', 'action_type'
        ]]
        st.dataframe(action_df, use_container_width=True)


# ─────────────────────────────────────────────────
def main():
    """Main application entry point."""
    model = load_model()
    X_min, X_max = load_normalization()
    location, threshold, page, sim_sound, sb_metrics_placeholder = render_sidebar()

    if page == "🏠 Live Detection":
        page_live_detection(model, location, threshold, X_min, X_max, sim_sound, sb_metrics_placeholder)
    elif page == "♿ Assistive Hearing Mode":
        page_assistive_hearing_mode(model, location, threshold, X_min, X_max, sim_sound, sb_metrics_placeholder)
    elif page == "🚨 Emergency Response Center":
        page_emergency_response_center()
    elif page == "📋 Alert History":
        page_alert_history()
    elif page == "📊 Feature Visualization":
        page_feature_visualization()
    elif page == "🧠 System Info":
        page_system_info(model)


if __name__ == '__main__':
    main()
