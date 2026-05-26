import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
import threading
import queue
import time
import os
import json
import warnings
warnings.filterwarnings('ignore')

from threat_logic import (
    generate_alert,
    get_sound_mode,
    get_mode_description,
    get_threat_level,
    get_threat_color,
    tracker,
    ALERT_SOUNDS,
    ASSISTIVE_SOUNDS
)

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
MODEL_PATH  = 'model/guardian_ear_model.h5'
INFO_PATH   = 'model/class_info.json'
SAMPLE_RATE = 22050
DURATION    = 3
SAMPLES     = SAMPLE_RATE * DURATION
THRESHOLD   = 0.70     # minimum confidence

CLASS_NAMES = [
    'air_conditioner', 'car_horn',
    'children_playing', 'dog_bark',
    'drilling', 'engine_idling',
    'gun_shot', 'jackhammer',
    'siren', 'street_music'
]

# Change this based on deployment location
CURRENT_LOCATION = 'parking_lot'

# ─────────────────────────────────────────
# STEP 1 — LOAD MODEL & CLASS INFO
# ─────────────────────────────────────────
def load_model():
    print("=" * 55)
    print("   Guardian Ear — Real-Time Detection v2")
    print("=" * 55)

    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: Model not found!")
        print(f"Path: {MODEL_PATH}")
        print(f"Please run 02_train_model.py first!")
        return None, None

    print("\nLoading CRNN model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
    print(f"  Input shape  : {model.input_shape}")
    print(f"  Output shape : {model.output_shape}")

    # Load class info if available
    class_info = None
    if os.path.exists(INFO_PATH):
        with open(INFO_PATH, 'r') as f:
            class_info = json.load(f)
        print(f"\nClass info loaded!")
        print(
            f"  Classes      : "
            f"{class_info['num_classes']}"
        )
        print(
            f"  Alert sounds : "
            f"{class_info['alert_sounds']}"
        )

    return model, class_info

# ─────────────────────────────────────────
# STEP 2 — LOAD NORMALIZATION VALUES
# ─────────────────────────────────────────
def load_normalization():
    """
    Loads min/max values used during training
    for consistent normalization at inference.
    """
    x_min_path = 'model/X_min.npy'
    x_max_path = 'model/X_max.npy'

    if os.path.exists(x_min_path) and \
       os.path.exists(x_max_path):
        X_min = np.load(x_min_path)[0]
        X_max = np.load(x_max_path)[0]
        print(
            f"Normalization loaded: "
            f"min={X_min:.2f}, max={X_max:.2f}"
        )
        return X_min, X_max
    else:
        print(
            "Normalization values not found — "
            "using default normalization"
        )
        return None, None

# ─────────────────────────────────────────
# STEP 3 — FEATURE EXTRACTION
# Must match exactly with training pipeline
# ─────────────────────────────────────────
def extract_features(audio, sr=SAMPLE_RATE,
                     X_min=None, X_max=None):
    """
    Extracts triple fused features from audio.
    Must match 01_feature_extraction.py exactly.
    """
    # Pad or trim to fixed length
    if len(audio) < SAMPLES:
        audio = np.pad(
            audio,
            (0, SAMPLES - len(audio))
        )
    else:
        audio = audio[:SAMPLES]

    # 1. Mel Spectrogram
    mel    = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_mels=128, fmax=8000
    )
    mel_db = librosa.power_to_db(
        mel, ref=np.max
    )

    # 2. MFCC
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=40
    )

    # 3. Chroma STFT
    chroma = librosa.feature.chroma_stft(
        y=audio, sr=sr
    )

    # Resize all to fixed time dimension
    target_len = 130

    def resize(f):
        if f.shape[1] > target_len:
            return f[:, :target_len]
        elif f.shape[1] < target_len:
            pad = target_len - f.shape[1]
            return np.pad(
                f, ((0, 0), (0, pad))
            )
        return f

    mel_db = resize(mel_db)
    mfcc   = resize(mfcc)
    chroma = resize(chroma)

    # Stack features → (180, 130)
    features = np.vstack([mel_db, mfcc, chroma])

    # Normalize using training values
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
                (f_max - f_min)
            )

    # Add batch + channel → (1, 180, 130, 1)
    features = features[np.newaxis, ..., np.newaxis]

    return features

# ─────────────────────────────────────────
# STEP 4 — PREDICT SOUND CLASS
# ─────────────────────────────────────────
def predict_sound(model, features):
    """
    Runs CRNN model inference and returns:
    - Detected class name
    - Confidence score
    - All class probabilities
    - Sound mode (ALERT/ASSISTIVE/NEUTRAL)
    """
    predictions = model.predict(
        features, verbose=0
    )
    class_id   = np.argmax(predictions[0])
    confidence = float(predictions[0][class_id])
    class_name = CLASS_NAMES[class_id]
    sound_mode = get_sound_mode(class_name)

    return (
        class_name,
        confidence,
        predictions[0],
        sound_mode
    )

# ─────────────────────────────────────────
# STEP 5 — DISPLAY PREDICTION RESULT
# ─────────────────────────────────────────
def display_result(
    class_name, confidence,
    all_probs, sound_mode,
    alert=None
):
    """
    Displays detection result in
    formatted console output.
    """
    print("\n" + "─" * 55)

    # Mode indicator
    if sound_mode == 'ALERT':
        print(f"  🚨 MODE: ALERT")
    elif sound_mode == 'ASSISTIVE':
        print(f"  ℹ️  MODE: ASSISTIVE")
    else:
        print(f"  👁  MODE: NEUTRAL")

    print(f"  Sound    : {class_name}")
    print(f"  Conf     : {confidence*100:.1f}%")

    # Description
    desc = get_mode_description(class_name)
    print(f"  Info     : {desc}")

    # Pattern info
    pattern = tracker.get_pattern_summary(
        class_name
    )
    print(
        f"  Pattern  : {pattern['pattern_label']}"
    )
    print(
        f"  Count    : "
        f"{pattern['detection_count']} detections"
    )
    print(
        f"  Duration : "
        f"{pattern['duration_seconds']}s"
    )

    # Alert info if generated
    if alert:
        print(
            f"  Threat   : "
            f"{alert['threat_score']}/100 "
            f"— {alert['threat_level']}"
        )
        if alert['should_escalate']:
            print(
                f"\n  ⚠ ESCALATION — Pattern "
                f"abnormally sustained!"
            )

    # Top 3 predictions
    top3 = np.argsort(all_probs)[::-1][:3]
    print(f"\n  Top predictions:")
    for idx in top3:
        bar = '█' * int(all_probs[idx] * 15)
        mode = get_sound_mode(CLASS_NAMES[idx])
        print(
            f"    {CLASS_NAMES[idx]:<22} "
            f"{all_probs[idx]*100:5.1f}% "
            f"{bar} [{mode}]"
        )
    print("─" * 55)

# ─────────────────────────────────────────
# STEP 6 — AUDIO RECORDING SETUP
# ─────────────────────────────────────────
audio_queue = queue.Queue()

def audio_callback(indata, frames,
                   time_info, status):
    """
    Called automatically by sounddevice
    for each recorded audio chunk.
    """
    if status:
        print(f"Audio status: {status}")
    audio_queue.put(indata.copy())

def start_recording():
    """Creates and returns audio input stream."""
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        blocksize=SAMPLES,
        callback=audio_callback
    )
    return stream

# ─────────────────────────────────────────
# STEP 7 — REAL-TIME PROCESSING LOOP
# ─────────────────────────────────────────
def processing_loop(
    model, location,
    stop_event,
    X_min=None, X_max=None
):
    """
    Continuously processes audio chunks
    from the microphone queue.
    """
    print(f"\nLocation     : {location}")
    print(f"Threshold    : {THRESHOLD*100:.0f}%")
    print(f"Sample rate  : {SAMPLE_RATE} Hz")
    print(f"Chunk size   : {DURATION} seconds")
    print(f"\nListening... Press Ctrl+C to stop\n")

    chunk_count = 0

    while not stop_event.is_set():
        try:
            # Get audio chunk from queue
            audio_chunk = audio_queue.get(timeout=1)
            audio_flat  = audio_chunk.flatten()
            chunk_count += 1

            print(
                f"\nChunk #{chunk_count} — "
                f"Processing..."
            )

            # Extract features
            features = extract_features(
                audio_flat,
                X_min=X_min,
                X_max=X_max
            )

            # Predict
            (
                class_name,
                confidence,
                all_probs,
                sound_mode
            ) = predict_sound(model, features)

            # Only process if above threshold
            if confidence >= THRESHOLD:
                alert = generate_alert(
                    sound_class=class_name,
                    confidence=confidence,
                    location=location
                )
                display_result(
                    class_name, confidence,
                    all_probs, sound_mode,
                    alert
                )
            else:
                print(
                    f"  Low confidence "
                    f"({confidence*100:.1f}%) "
                    f"— skipping"
                )

        except queue.Empty:
            continue
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Processing error: {e}")
            continue

# ─────────────────────────────────────────
# STEP 8 — SIMULATE ON AUDIO FILE
# For testing without microphone
# ─────────────────────────────────────────
def simulate_detection(
    model, audio_file,
    location='unknown',
    X_min=None, X_max=None
):
    """
    Runs detection on a saved audio file.
    Useful for testing and demonstration.
    """
    print(f"\nSimulating on: {audio_file}")
    print(f"Location     : {location}\n")

    if not os.path.exists(audio_file):
        print(f"File not found: {audio_file}")
        return None

    # Load audio
    audio, sr = librosa.load(
        audio_file,
        sr=SAMPLE_RATE,
        duration=DURATION
    )

    print(
        f"Audio loaded: "
        f"{len(audio)/sr:.2f}s @ {sr}Hz"
    )

    # Extract features
    features = extract_features(
        audio, sr,
        X_min=X_min,
        X_max=X_max
    )

    # Predict
    (
        class_name,
        confidence,
        all_probs,
        sound_mode
    ) = predict_sound(model, features)

    # Generate alert
    alert = generate_alert(
        sound_class=class_name,
        confidence=confidence,
        location=location
    )

    # Display result
    display_result(
        class_name, confidence,
        all_probs, sound_mode,
        alert
    )

    # Show all probabilities
    print("\nAll class probabilities:")
    sorted_idx = np.argsort(all_probs)[::-1]
    for idx in sorted_idx:
        bar  = '█' * int(all_probs[idx] * 20)
        mode = get_sound_mode(CLASS_NAMES[idx])
        print(
            f"  {CLASS_NAMES[idx]:<22} "
            f"{all_probs[idx]*100:5.1f}% "
            f"{bar} [{mode}]"
        )

    return alert

# ─────────────────────────────────────────
# STEP 9 — SHOW SYSTEM STATUS
# ─────────────────────────────────────────
def show_system_status():
    """
    Displays current system configuration
    and mode classification.
    """
    print("\n" + "=" * 55)
    print("   Guardian Ear — System Status")
    print("=" * 55)
    print(f"  Model path   : {MODEL_PATH}")
    print(f"  Location     : {CURRENT_LOCATION}")
    print(f"  Sample rate  : {SAMPLE_RATE} Hz")
    print(f"  Duration     : {DURATION} seconds")
    print(f"  Threshold    : {THRESHOLD*100:.0f}%")
    print(f"\n  Sound Modes:")
    for cls in CLASS_NAMES:
        mode = get_sound_mode(cls)
        icon = (
            '🚨' if mode == 'ALERT'
            else 'ℹ️' if mode == 'ASSISTIVE'
            else '👁'
        )
        print(f"    {icon} {cls:<25} {mode}")
    print("=" * 55 + "\n")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":

    # Load model
    model, class_info = load_model()
    if model is None:
        exit()

    # Load normalization values
    X_min, X_max = load_normalization()

    # Show system status
    show_system_status()

    # Select mode
    print("Select detection mode:")
    print("  1 — Real-time microphone detection")
    print("  2 — Simulate on audio file")
    print("  3 — Show system status only")
    mode = input(
        "\nEnter choice (1/2/3): "
    ).strip()

    if mode == '1':
        # Real-time microphone mode
        print(
            f"\nStarting real-time detection..."
        )
        print(
            f"Location: {CURRENT_LOCATION}"
        )

        stop_event  = threading.Event()
        proc_thread = threading.Thread(
            target=processing_loop,
            args=(
                model,
                CURRENT_LOCATION,
                stop_event,
                X_min, X_max
            )
        )
        proc_thread.start()

        try:
            with start_recording():
                print(
                    "Microphone active — "
                    "Listening...\n"
                )
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping detection...")
            stop_event.set()
            proc_thread.join()
            print("Guardian Ear stopped.")

    elif mode == '2':
        # File simulation mode
        audio_file = input(
            "\nEnter path to .wav file: "
        ).strip()

        location = input(
            "Enter location (or press Enter "
            f"for '{CURRENT_LOCATION}'): "
        ).strip()

        if not location:
            location = CURRENT_LOCATION

        simulate_detection(
            model=model,
            audio_file=audio_file,
            location=location,
            X_min=X_min,
            X_max=X_max
        )

    elif mode == '3':
        show_system_status()

    else:
        print("Invalid choice!")