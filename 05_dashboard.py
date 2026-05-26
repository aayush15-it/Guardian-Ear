import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import time
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from threat_logic import (
    generate_alert,
    get_sound_mode,
    get_mode_description,
    get_threat_level,
    get_threat_color,
    calculate_threat_score,
    load_alert_history,
    tracker,
    ALERT_SOUNDS,
    ASSISTIVE_SOUNDS,
    SOUND_WEIGHTS,
    LOCATION_WEIGHTS
)

# ─────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────
st.set_page_config(
    page_title='Guardian Ear',
    page_icon='🎧',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
MODEL_PATH  = 'model/guardian_ear_model.h5'
INFO_PATH   = 'model/class_info.json'
SAMPLE_RATE = 22050
DURATION    = 3
SAMPLES     = SAMPLE_RATE * DURATION
NUM_CLASSES = 10

CLASS_NAMES = [
    'air_conditioner', 'car_horn',
    'children_playing', 'dog_bark',
    'drilling', 'engine_idling',
    'gun_shot', 'jackhammer',
    'siren', 'street_music'
]

LOCATIONS = [
    'parking_lot', 'corridor', 'hostel',
    'library', 'cafeteria', 'classroom',
    'entrance', 'garden', 'office'
]

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
  .main-title {
    font-size: 2.4rem; font-weight: 800;
    color: #1a1a2e; text-align: center;
    padding: 10px 0 4px;
  }
  .subtitle {
    font-size: 1rem; color: #666;
    text-align: center; margin-bottom: 20px;
  }
  .mode-alert {
    background: #FFF5F5;
    border-left: 5px solid #E53E3E;
    padding: 12px 16px; border-radius: 8px;
    margin: 8px 0; font-weight: 600;
    color: #C53030;
  }
  .mode-assistive {
    background: #EBF8FF;
    border-left: 5px solid #3182CE;
    padding: 12px 16px; border-radius: 8px;
    margin: 8px 0; font-weight: 600;
    color: #2B6CB0;
  }
  .mode-neutral {
    background: #FFFFF0;
    border-left: 5px solid #D69E2E;
    padding: 12px 16px; border-radius: 8px;
    margin: 8px 0; font-weight: 600;
    color: #975A16;
  }
  .alert-critical {
    background: #FFF5F5;
    border-left: 5px solid #E53E3E;
    padding: 14px 16px; border-radius: 8px;
    margin: 8px 0;
  }
  .alert-high {
    background: #FFFAF0;
    border-left: 5px solid #DD6B20;
    padding: 14px 16px; border-radius: 8px;
    margin: 8px 0;
  }
  .alert-medium {
    background: #FFFFF0;
    border-left: 5px solid #D69E2E;
    padding: 14px 16px; border-radius: 8px;
    margin: 8px 0;
  }
  .alert-low {
    background: #F0FFF4;
    border-left: 5px solid #38A169;
    padding: 14px 16px; border-radius: 8px;
    margin: 8px 0;
  }
  .pattern-card {
    background: #F7FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px; padding: 14px;
    margin: 8px 0;
  }
  .escalation-warn {
    background: #FFF5F5;
    border: 2px solid #E53E3E;
    border-radius: 10px; padding: 12px;
    margin: 8px 0; color: #C53030;
    font-weight: 600; text-align: center;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(
            MODEL_PATH
        )
        return model
    return None

@st.cache_data
def load_class_info():
    if os.path.exists(INFO_PATH):
        with open(INFO_PATH, 'r') as f:
            return json.load(f)
    return None

def load_normalization():
    x_min_path = 'model/X_min.npy'
    x_max_path = 'model/X_max.npy'
    if os.path.exists(x_min_path) and \
       os.path.exists(x_max_path):
        return (
            np.load(x_min_path)[0],
            np.load(x_max_path)[0]
        )
    return None, None

# ─────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────
def extract_features(
    audio, sr=SAMPLE_RATE,
    X_min=None, X_max=None
):
    if len(audio) < SAMPLES:
        audio = np.pad(
            audio, (0, SAMPLES - len(audio))
        )
    else:
        audio = audio[:SAMPLES]

    mel    = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_mels=128, fmax=8000
    )
    mel_db = librosa.power_to_db(
        mel, ref=np.max
    )
    mfcc   = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=40
    )
    chroma = librosa.feature.chroma_stft(
        y=audio, sr=sr
    )

    target_len = 130
    def resize(f):
        if f.shape[1] > target_len:
            return f[:, :target_len]
        elif f.shape[1] < target_len:
            return np.pad(
                f,
                ((0, 0),
                 (0, target_len - f.shape[1]))
            )
        return f

    mel_db = resize(mel_db)
    mfcc   = resize(mfcc)
    chroma = resize(chroma)

    features = np.vstack([mel_db, mfcc, chroma])

    if X_min is not None and X_max is not None:
        features = (features - X_min) / (
            X_max - X_min + 1e-8
        )
    else:
        f_min = features.min()
        f_max = features.max()
        if f_max - f_min > 0:
            features = (
                (features - f_min) /
                (f_max  - f_min)
            )

    return features, mel_db, mfcc, chroma

# ─────────────────────────────────────────
# GRAD-CAM VISUALIZATION
# ─────────────────────────────────────────
def generate_gradcam(model, features, class_idx):
    try:
        last_conv = None
        for layer in model.layers:
            if 'conv2d' in layer.name:
                last_conv = layer.name

        if last_conv is None:
            return None

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(last_conv).output,
                model.output
            ]
        )

        input_tensor = tf.cast(
            features[np.newaxis, ..., np.newaxis],
            tf.float32
        )

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(
                input_tensor
            )
            loss = preds[:, class_idx]

        grads   = tape.gradient(loss, conv_out)
        pooled  = tf.reduce_mean(
            grads, axis=(0, 1, 2)
        )
        heatmap = conv_out[0] @ \
                  pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (
            tf.math.reduce_max(heatmap) + 1e-8
        )
        return heatmap.numpy()

    except Exception as e:
        return None

# ─────────────────────────────────────────
# PLOT FEATURES
# ─────────────────────────────────────────
def plot_features(
    mel_db, mfcc, chroma, audio, sr
):
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 8)
    )

    # Waveform
    axes[0, 0].plot(
        np.linspace(0, DURATION, len(audio)),
        audio,
        color='steelblue', linewidth=0.8
    )
    axes[0, 0].set_title('Audio Waveform')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)

    # Mel Spectrogram
    im1 = axes[0, 1].imshow(
        mel_db, aspect='auto',
        origin='lower', cmap='magma'
    )
    axes[0, 1].set_title('Mel Spectrogram')
    axes[0, 1].set_xlabel('Time Frames')
    axes[0, 1].set_ylabel('Mel Bins')
    plt.colorbar(im1, ax=axes[0, 1], label='dB')

    # MFCC
    im2 = axes[1, 0].imshow(
        mfcc, aspect='auto',
        origin='lower', cmap='coolwarm'
    )
    axes[1, 0].set_title('MFCC Features')
    axes[1, 0].set_xlabel('Time Frames')
    axes[1, 0].set_ylabel('Coefficients')
    plt.colorbar(im2, ax=axes[1, 0])

    # Chroma STFT
    im3 = axes[1, 1].imshow(
        chroma, aspect='auto',
        origin='lower', cmap='YlOrRd'
    )
    axes[1, 1].set_title('Chroma STFT')
    axes[1, 1].set_xlabel('Time Frames')
    axes[1, 1].set_ylabel('Pitch Classes')
    plt.colorbar(im3, ax=axes[1, 1])

    plt.tight_layout()
    return fig

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
def render_sidebar():
    st.sidebar.image(
        "https://img.icons8.com/color/96/headphones.png",
        width=70
    )
    st.sidebar.title("Guardian Ear")
    st.sidebar.markdown(
        "*AI Acoustic Safety System*"
    )
    st.sidebar.divider()

    st.sidebar.subheader("⚙️ Settings")

    location = st.sidebar.selectbox(
        "📍 Current Location", LOCATIONS
    )
    threshold = st.sidebar.slider(
        "🎯 Alert Threshold",
        0, 100, 60,
        help="Minimum threat score to raise alert"
    )

    st.sidebar.divider()

    st.sidebar.subheader("🔊 Sound Modes")
    st.sidebar.markdown("**🚨 Alert Sounds:**")
    for s in ALERT_SOUNDS:
        st.sidebar.markdown(f"  • {s}")
    st.sidebar.markdown("**ℹ️ Assistive Sounds:**")
    for s in ASSISTIVE_SOUNDS:
        st.sidebar.markdown(f"  • {s}")

    st.sidebar.divider()

    st.sidebar.subheader("🗂️ Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "🏠 Live Detection",
            "📋 Alert History",
            "📊 Feature Visualization",
            "🧠 System Info"
        ]
    )

    st.sidebar.divider()
    st.sidebar.markdown(
        "**Status:** 🟢 ACTIVE"
    )

    return location, threshold, page

# ─────────────────────────────────────────
# PAGE 1 — LIVE DETECTION
# ─────────────────────────────────────────
def page_live_detection(
    model, location,
    threshold, X_min, X_max
):
    st.markdown(
        "<div class='main-title'>"
        "🎧 Guardian Ear"
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='subtitle'>"
        "AI-Based Acoustic Anomaly Detection "
        "· Dual-Mode Intelligence System"
        "</div>",
        unsafe_allow_html=True
    )
    st.divider()

    # Top metrics
    history = load_alert_history()
    c1, c2, c3, c4, c5 = st.columns(5)

    total = len(history) if not history.empty \
        else 0
    critical = len(
        history[
            history['threat_level'] == 'CRITICAL'
        ]
    ) if not history.empty else 0
    alert_count = len(
        history[
            history['sound_mode'] == 'ALERT'
        ]
    ) if not history.empty \
      and 'sound_mode' in history.columns \
      else 0
    assistive_count = len(
        history[
            history['sound_mode'] == 'ASSISTIVE'
        ]
    ) if not history.empty \
      and 'sound_mode' in history.columns \
      else 0

    c1.metric("📊 Total Detections", total)
    c2.metric("🔴 Critical Alerts", critical)
    c3.metric("🚨 Alert Mode", alert_count)
    c4.metric("ℹ️ Assistive Mode", assistive_count)
    c5.metric(
        "📍 Location",
        location.replace('_', ' ').title()
    )

    st.divider()

    # Upload audio
    st.subheader("🎙️ Upload Audio for Detection")
    uploaded = st.file_uploader(
        "Upload a .wav audio file",
        type=['wav']
    )

    if uploaded is not None:
        temp_path = 'temp_audio.wav'
        with open(temp_path, 'wb') as f:
            f.write(uploaded.read())

        # Load audio
        audio, sr = librosa.load(
            temp_path,
            sr=SAMPLE_RATE,
            duration=DURATION
        )
        st.audio(temp_path)

        # Extract features
        with st.spinner(
            "Extracting features..."
        ):
            features, mel_db, mfcc, chroma = \
                extract_features(
                    audio, sr, X_min, X_max
                )

        if model is not None:
            with st.spinner(
                "Running CRNN model..."
            ):
                inp = features[
                    np.newaxis, ..., np.newaxis
                ]
                preds      = model.predict(
                    inp, verbose=0
                )[0]
                class_id   = np.argmax(preds)
                confidence = float(preds[class_id])
                class_name = CLASS_NAMES[class_id]

            # Get sound mode
            sound_mode  = get_sound_mode(class_name)
            description = get_mode_description(
                class_name
            )

            # Pattern analysis
            tracker.add_detection(class_name)
            pattern = tracker.get_pattern_summary(
                class_name
            )

            # Threat score
            threat_score = calculate_threat_score(
                class_name, confidence,
                location,
                pattern_score=pattern[
                    'pattern_score'
                ]
            )
            threat_level = get_threat_level(
                threat_score
            )
            threat_color = get_threat_color(
                threat_level
            )

            st.divider()

            # ── DETECTION RESULT ──
            st.subheader("🎯 Detection Result")

            r1, r2, r3, r4 = st.columns(4)
            r1.metric(
                "Sound Detected",
                class_name.replace(
                    '_', ' '
                ).title()
            )
            r2.metric(
                "Confidence",
                f"{confidence*100:.1f}%"
            )
            r3.metric(
                "Threat Score",
                f"{threat_score}/100"
            )
            r4.metric(
                "Threat Level",
                threat_level
            )

            # Mode display
            st.subheader("🔊 Sound Mode")
            if sound_mode == 'ALERT':
                st.markdown(
                    f"<div class='mode-alert'>"
                    f"🚨 ALERT MODE — "
                    f"{description}"
                    f"</div>",
                    unsafe_allow_html=True
                )
            elif sound_mode == 'ASSISTIVE':
                st.markdown(
                    f"<div class='mode-assistive'>"
                    f"ℹ️ ASSISTIVE MODE — "
                    f"{description}"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='mode-neutral'>"
                    f"👁 NEUTRAL MODE — "
                    f"{description}"
                    f"</div>",
                    unsafe_allow_html=True
                )

            # ── TEMPORAL PATTERN ANALYSIS ──
            st.subheader(
                "⏱️ Temporal Pattern Analysis"
            )

            p1, p2, p3, p4 = st.columns(4)
            p1.metric(
                "Pattern",
                pattern['pattern_label']
            )
            p2.metric(
                "Pattern Score",
                f"{pattern['pattern_score']:.2f}"
            )
            p3.metric(
                "Detection Count",
                pattern['detection_count']
            )
            p4.metric(
                "Duration",
                f"{pattern['duration_seconds']}s"
            )

            st.write("Pattern Intensity:")
            st.progress(
                float(pattern['pattern_score'])
            )

            if pattern['should_escalate']:
                st.markdown(
                    "<div class='escalation-warn'>"
                    "⚠️ ESCALATION DETECTED — "
                    "This sound has been occurring "
                    "abnormally long or frequently!"
                    "</div>",
                    unsafe_allow_html=True
                )

            # ── THREAT ALERT BOX ──
            st.subheader("🚨 Threat Assessment")
            level_lower = threat_level.lower()
            if level_lower in [
                'critical', 'high',
                'medium', 'low'
            ]:
                st.markdown(
                    f"<div class='alert-"
                    f"{level_lower}'>"
                    f"<b>{threat_level} ALERT</b>"
                    f"<br>"
                    f"Sound: "
                    f"{class_name.replace('_',' ').title()}"
                    f" | Mode: {sound_mode}"
                    f" | Confidence: "
                    f"{confidence*100:.1f}%"
                    f" | Location: "
                    f"{location.replace('_',' ').title()}"
                    f" | Score: {threat_score}/100"
                    f"</div>",
                    unsafe_allow_html=True
                )

            # ── PROBABILITY CHART ──
            st.subheader(
                "📊 Class Probabilities"
            )
            prob_df = pd.DataFrame({
                'Sound Class': [
                    c.replace('_', ' ').title()
                    for c in CLASS_NAMES
                ],
                'Probability (%)': preds * 100,
                'Mode': [
                    get_sound_mode(c)
                    for c in CLASS_NAMES
                ]
            }).sort_values(
                'Probability (%)',
                ascending=False
            )
            st.bar_chart(
                prob_df.set_index(
                    'Sound Class'
                )['Probability (%)']
            )

            # ── GRAD-CAM ──
            st.subheader(
                "🔍 Grad-CAM Explainability"
            )
            st.caption(
                "Highlights which part of the "
                "spectrogram triggered detection"
            )

            with st.spinner(
                "Generating Grad-CAM heatmap..."
            ):
                heatmap = generate_gradcam(
                    model, features, class_id
                )

            if heatmap is not None:
                fig_gc, axes_gc = plt.subplots(
                    1, 2, figsize=(12, 4)
                )

                axes_gc[0].imshow(
                    mel_db, aspect='auto',
                    origin='lower', cmap='magma'
                )
                axes_gc[0].set_title(
                    'Mel Spectrogram'
                )
                axes_gc[0].set_xlabel(
                    'Time Frames'
                )
                axes_gc[0].set_ylabel(
                    'Mel Frequency Bins'
                )

                axes_gc[1].imshow(
                    mel_db, aspect='auto',
                    origin='lower', cmap='magma'
                )
                hm_resized = np.array(
                    tf.image.resize(
                        heatmap[..., np.newaxis],
                        [
                            mel_db.shape[0],
                            mel_db.shape[1]
                        ]
                    )
                ).squeeze()
                axes_gc[1].imshow(
                    hm_resized, aspect='auto',
                    origin='lower', cmap='jet',
                    alpha=0.5
                )
                axes_gc[1].set_title(
                    'Grad-CAM Heatmap'
                )
                axes_gc[1].set_xlabel(
                    'Time Frames'
                )
                axes_gc[1].set_ylabel(
                    'Mel Frequency Bins'
                )

                plt.tight_layout()
                st.pyplot(fig_gc)
                st.caption(
                    "🔴 Red regions = frequency "
                    "areas that triggered detection"
                )
            else:
                st.info(
                    "Grad-CAM not available — "
                    "run with trained model."
                )

            # ── FEATURE VISUALIZATION ──
            st.subheader(
                "📈 Feature Visualization"
            )
            fig = plot_features(
                mel_db, mfcc, chroma, audio, sr
            )
            st.pyplot(fig)

            # ── SAVE ALERT BUTTON ──
            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "💾 Save Alert to Log"
                ):
                    generate_alert(
                        sound_class=class_name,
                        confidence=confidence,
                        location=location
                    )
                    st.success(
                        "Alert saved to log!"
                    )
            with col2:
                if st.button(
                    "🔄 Reset Pattern Tracker"
                ):
                    tracker.reset(class_name)
                    st.success(
                        "Pattern tracker reset!"
                    )

        else:
            st.warning(
                "⚠️ Model not found! "
                "Please run 02_train_model.py"
            )

        if os.path.exists(temp_path):
            os.remove(temp_path)

# ─────────────────────────────────────────
# PAGE 2 — ALERT HISTORY
# ─────────────────────────────────────────
def page_alert_history():
    st.title("📋 Alert History")

    history = load_alert_history()

    if history.empty:
        st.info(
            "No alerts recorded yet. "
            "Run a detection first!"
        )
        return

    # Summary metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total", len(history))
    c2.metric(
        "Critical",
        len(history[
            history['threat_level'] == 'CRITICAL'
        ])
    )
    c3.metric(
        "High",
        len(history[
            history['threat_level'] == 'HIGH'
        ])
    )
    c4.metric(
        "Avg Score",
        f"{history['threat_score'].mean():.1f}"
    )
    if 'sound_mode' in history.columns:
        c5.metric(
            "Alert Mode",
            len(history[
                history['sound_mode'] == 'ALERT'
            ])
        )

    st.divider()

    # Filters
    st.subheader("🔍 Filter Alerts")
    f1, f2, f3 = st.columns(3)

    with f1:
        level_filter = st.multiselect(
            "Threat Level",
            ['CRITICAL','HIGH','MEDIUM',
             'LOW','SAFE'],
            default=['CRITICAL', 'HIGH']
        )
    with f2:
        class_filter = st.multiselect(
            "Sound Class",
            CLASS_NAMES,
            default=CLASS_NAMES
        )
    with f3:
        if 'sound_mode' in history.columns:
            mode_filter = st.selectbox(
                "Sound Mode",
                ['All', 'ALERT',
                 'ASSISTIVE', 'NEUTRAL']
            )
        else:
            mode_filter = 'All'

    # Apply filters
    filtered = history[
        (history['threat_level'].isin(
            level_filter
        )) &
        (history['sound_class'].isin(
            class_filter
        ))
    ]

    if mode_filter != 'All' and \
       'sound_mode' in filtered.columns:
        filtered = filtered[
            filtered['sound_mode'] == mode_filter
        ]

    st.subheader(
        f"Showing {len(filtered)} alerts"
    )

    # Display columns
    display_cols = [
        'timestamp', 'sound_class',
        'threat_score', 'threat_level'
    ]
    if 'sound_mode' in filtered.columns:
        display_cols.insert(2, 'sound_mode')
    if 'pattern_label' in filtered.columns:
        display_cols.insert(3, 'pattern_label')
    if 'confidence' in filtered.columns:
        display_cols.append('confidence')
    if 'location' in filtered.columns:
        display_cols.append('location')

    st.dataframe(
        filtered[display_cols].sort_values(
            'timestamp', ascending=False
        ),
        use_container_width=True
    )

    # Threat score trend
    if len(history) > 1:
        st.subheader("📈 Threat Score Trend")
        chart_data = history.set_index(
            'timestamp'
        )['threat_score']
        st.line_chart(chart_data)

    # Mode distribution chart
    if 'sound_mode' in history.columns:
        st.subheader("🔊 Mode Distribution")
        mode_counts = history[
            'sound_mode'
        ].value_counts()
        st.bar_chart(mode_counts)

    # Download
    csv = filtered.to_csv(index=False)
    st.download_button(
        "⬇️ Download Alert Log as CSV",
        csv,
        "guardian_ear_alerts.csv",
        "text/csv"
    )

# ─────────────────────────────────────────
# PAGE 3 — FEATURE VISUALIZATION
# ─────────────────────────────────────────
def page_feature_visualization():
    st.title("📊 Feature Visualization")
    st.info(
        "Upload an audio file to visualize "
        "all extracted features."
    )

    uploaded = st.file_uploader(
        "Upload .wav file", type=['wav']
    )

    if uploaded:
        temp_path = 'temp_viz.wav'
        with open(temp_path, 'wb') as f:
            f.write(uploaded.read())

        audio, sr = librosa.load(
            temp_path,
            sr=SAMPLE_RATE,
            duration=DURATION
        )
        st.audio(temp_path)

        X_min, X_max = load_normalization()
        features, mel_db, mfcc, chroma = \
            extract_features(
                audio, sr, X_min, X_max
            )

        st.subheader(
            "All Extracted Features"
        )
        fig = plot_features(
            mel_db, mfcc, chroma, audio, sr
        )
        st.pyplot(fig)

        # Feature stats
        st.subheader("Feature Statistics")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric(
            "Mel Shape",
            f"{mel_db.shape[0]}×{mel_db.shape[1]}"
        )
        s2.metric(
            "MFCC Shape",
            f"{mfcc.shape[0]}×{mfcc.shape[1]}"
        )
        s3.metric(
            "Chroma Shape",
            f"{chroma.shape[0]}×{chroma.shape[1]}"
        )
        s4.metric(
            "Fused Shape",
            f"180×130"
        )

        if os.path.exists(temp_path):
            os.remove(temp_path)

# ─────────────────────────────────────────
# PAGE 4 — SYSTEM INFO
# ─────────────────────────────────────────
def page_system_info(model):
    st.title("🧠 System Information")

    # Model info
    st.subheader("Model Details")
    if model is not None:
        i1, i2, i3 = st.columns(3)
        i1.metric("Model Type", "CRNN v2")
        i2.metric(
            "Input Shape",
            str(model.input_shape[1:])
        )
        i3.metric(
            "Output Classes",
            str(NUM_CLASSES)
        )

        # Model layers table
        st.write("Model Architecture:")
        layer_info = []
        for layer in model.layers:
            layer_info.append({
                'Layer'       : layer.name,
                'Type'        : layer.__class__.__name__,
                'Output Shape': str(
                    layer.output_shape
                )
            })
        st.dataframe(
            pd.DataFrame(layer_info),
            use_container_width=True
        )
    else:
        st.warning(
            "Model not loaded. "
            "Run 02_train_model.py first."
        )

    st.divider()

    # Research gaps
    st.subheader("✅ Research Gaps Covered")
    gaps = {
        "1": (
            "Poor noise performance",
            "Multi-environment augmentation"
        ),
        "2": (
            "Cloud dependency",
            "Edge processing with TFLite"
        ),
        "3": (
            "Single feature limitation",
            "Mel + MFCC + Chroma fusion"
        ),
        "4": (
            "High false alarms",
            "Context-aware threat logic"
        ),
        "5": (
            "Slow detection",
            "Real-time pipeline"
        ),
        "6": (
            "No temporal modeling",
            "CRNN — CNN + LSTM"
        ),
        "7": (
            "No severity quantification",
            "Threat Score (0-100)"
        ),
        "8": (
            "Poor generalization",
            "Multi-environment augmentation"
        ),
        "9": (
            "No explainability",
            "Grad-CAM visualization"
        ),
        "10": (
            "No pattern intelligence",
            "Temporal Pattern Tracker"
        ),
    }

    for num, (gap, solution) in gaps.items():
        st.success(
            f"Gap {num}: {gap} → {solution}"
        )

    st.divider()

    # Dual mode table
    st.subheader("🔊 Dual Mode Classification")
    mode_data = []
    for cls in CLASS_NAMES:
        mode = get_sound_mode(cls)
        desc = get_mode_description(cls)
        mode_data.append({
            'Sound Class': cls,
            'Mode'       : mode,
            'Description': desc
        })
    st.dataframe(
        pd.DataFrame(mode_data),
        use_container_width=True
    )

    st.divider()

    # Tech stack
    st.subheader("🛠️ Technology Stack")
    tech = {
        'Component': [
            'Audio Processing',
            'Deep Learning',
            'Dashboard',
            'Edge Deployment',
            'Dataset',
            'Explainability'
        ],
        'Technology': [
            'Librosa 0.10',
            'TensorFlow/Keras 2.13',
            'Streamlit',
            'TensorFlow Lite',
            'UrbanSound8K',
            'Grad-CAM'
        ]
    }
    st.table(pd.DataFrame(tech))

# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────
def main():
    # Load resources
    model      = load_model()
    class_info = load_class_info()
    X_min, X_max = load_normalization()

    # Render sidebar
    location, threshold, page = render_sidebar()

    # Route to selected page
    if page == "🏠 Live Detection":
        page_live_detection(
            model, location,
            threshold, X_min, X_max
        )
    elif page == "📋 Alert History":
        page_alert_history()
    elif page == "📊 Feature Visualization":
        page_feature_visualization()
    elif page == "🧠 System Info":
        page_system_info(model)

if __name__ == "__main__":
    main()