import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
DATASET_PATH  = 'dataset/UrbanSound8K/audio'
METADATA_PATH = 'dataset/UrbanSound8K/metadata/UrbanSound8K.csv'
FEATURES_PATH = 'features'
SAMPLE_RATE   = 22050
DURATION      = 3
SAMPLES       = SAMPLE_RATE * DURATION

# ─────────────────────────────────────────
# CLASS CONFIGURATION
# ─────────────────────────────────────────
CLASS_NAMES = [
    'air_conditioner', 'car_horn', 'children_playing',
    'dog_bark', 'drilling', 'engine_idling',
    'gun_shot', 'jackhammer', 'siren', 'street_music'
]

# Dual Mode Classification
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
    Returns the mode of a detected sound.
    ALERT    → immediate security alert needed
    ASSISTIVE → informational feedback only
    NEUTRAL  → monitor only
    """
    if sound_class in ALERT_SOUNDS:
        return 'ALERT'
    elif sound_class in ASSISTIVE_SOUNDS:
        return 'ASSISTIVE'
    else:
        return 'NEUTRAL'

# ─────────────────────────────────────────
# DURATION THRESHOLDS (seconds)
# How long a sound must persist to be
# considered abnormal
# ─────────────────────────────────────────
DURATION_THRESHOLDS = {
    'dog_bark'        : 180,    # 3 minutes
    'engine_idling'   : 600,    # 10 minutes
    'drilling'        : 900,    # 15 minutes
    'children_playing': 1800,   # 30 minutes
    'street_music'    : 600,    # 10 minutes
    'air_conditioner' : 1800,   # 30 minutes
    'car_horn'        : 60,     # 1 minute
    'siren'           : 120,    # 2 minutes
    'gun_shot'        : 0,      # instant alert
    'jackhammer'      : 300,    # 5 minutes
}

# ─────────────────────────────────────────
# REPETITION THRESHOLDS (count per window)
# How many times in 10 min window
# before considered abnormal
# ─────────────────────────────────────────
REPETITION_THRESHOLDS = {
    'dog_bark'        : 8,
    'car_horn'        : 5,
    'gun_shot'        : 2,
    'siren'           : 3,
    'children_playing': 20,
    'drilling'        : 10,
}

# ─────────────────────────────────────────
# STEP 1 — LOAD AUDIO
# ─────────────────────────────────────────
def load_audio(file_path):
    """
    Loads audio file and pads/trims to
    fixed DURATION length.
    """
    try:
        audio, sr = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            duration=DURATION
        )
        if len(audio) < SAMPLES:
            audio = np.pad(
                audio,
                (0, SAMPLES - len(audio))
            )
        else:
            audio = audio[:SAMPLES]
        return audio, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

# ─────────────────────────────────────────
# STEP 2 — DATA AUGMENTATION
# Multi-environment augmentation for
# better real-world generalization
# ─────────────────────────────────────────
def augment_audio(audio, sr):
    """
    Creates multiple augmented versions
    of each audio clip to simulate
    different real-world environments.
    """
    augmented = []

    # 1. Original — clean recording
    augmented.append(audio)

    # 2. Gaussian noise — mild background noise
    noise_mild = audio + 0.005 * np.random.randn(
        len(audio)
    )
    augmented.append(noise_mild)

    # 3. Heavy noise — crowded/noisy environment
    noise_heavy = audio + 0.015 * np.random.randn(
        len(audio)
    )
    augmented.append(noise_heavy)

    # 4. Time stretch slower — distant sound
    try:
        stretched = librosa.effects.time_stretch(
            audio, rate=0.9
        )
        stretched = librosa.util.fix_length(
            stretched, size=SAMPLES
        )
        augmented.append(stretched)
    except:
        augmented.append(audio)

    # 5. Time stretch faster — closer sound
    try:
        stretched_fast = librosa.effects.time_stretch(
            audio, rate=1.1
        )
        stretched_fast = librosa.util.fix_length(
            stretched_fast, size=SAMPLES
        )
        augmented.append(stretched_fast)
    except:
        augmented.append(audio)

    # 6. Pitch shift up — higher frequency
    try:
        pitched_up = librosa.effects.pitch_shift(
            audio, sr=sr, n_steps=2
        )
        augmented.append(pitched_up)
    except:
        augmented.append(audio)

    # 7. Pitch shift down — lower frequency
    try:
        pitched_down = librosa.effects.pitch_shift(
            audio, sr=sr, n_steps=-2
        )
        augmented.append(pitched_down)
    except:
        augmented.append(audio)

    # 8. Volume reduction — far/muffled sound
    quiet = audio * 0.4
    augmented.append(quiet)

    # 9. Volume boost — close/loud sound
    loud = np.clip(audio * 1.5, -1.0, 1.0)
    augmented.append(loud)

    return augmented

# ─────────────────────────────────────────
# STEP 3 — FEATURE EXTRACTION
# Triple feature fusion:
# Mel Spectrogram + MFCC + Chroma STFT
# ─────────────────────────────────────────
def extract_features(audio, sr):
    """
    Extracts and fuses three complementary
    audio features into single representation.

    Mel Spectrogram → frequency vs time map
    MFCC            → texture and timbre
    Chroma STFT     → tonal and pitch info
    """

    # 1. Mel Spectrogram (128 mel bins)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128,
        fmax=8000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # 2. MFCC (40 coefficients)
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    # 3. Chroma STFT (12 pitch classes)
    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr
    )

    # Resize all to same time dimension
    target_len = 130
    mel_db = resize_feature(mel_db, target_len)  # (128, 130)
    mfcc   = resize_feature(mfcc,   target_len)  # (40,  130)
    chroma = resize_feature(chroma, target_len)  # (12,  130)

    # Stack → fused feature (180, 130)
    features = np.vstack([mel_db, mfcc, chroma])

    return features

def resize_feature(feature, target_len):
    """
    Resizes feature to fixed time dimension.
    Pads if too short, trims if too long.
    """
    if feature.shape[1] > target_len:
        return feature[:, :target_len]
    elif feature.shape[1] < target_len:
        pad = target_len - feature.shape[1]
        return np.pad(feature, ((0, 0), (0, pad)))
    return feature

# ─────────────────────────────────────────
# STEP 4 — PROCESS ENTIRE DATASET
# ─────────────────────────────────────────
def process_dataset():
    print("=" * 55)
    print("   Guardian Ear — Feature Extraction v2")
    print("=" * 55)

    # Load metadata CSV
    df = pd.read_csv(METADATA_PATH)
    print(f"\nDataset loaded successfully!")
    print(f"Total samples  : {len(df)}")
    print(f"Total classes  : {df['class'].nunique()}")
    print(f"\nClass distribution:")
    print(df['class'].value_counts().to_string())

    # Show mode classification
    print(f"\nSound Mode Classification:")
    for cls in CLASS_NAMES:
        mode = get_sound_mode(cls)
        print(f"  {cls:<25} → {mode}")

    all_features = []
    all_labels   = []
    all_modes    = []
    skipped      = 0

    print(f"\nStarting feature extraction...")
    print(f"Augmentations per clip : 9")
    print(f"Expected total samples : ~{len(df) * 9}\n")

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Extracting"
    ):
        fold      = f"fold{row['fold']}"
        file_name = row['slice_file_name']
        label     = row['classID']
        cls_name  = row['class']
        file_path = os.path.join(
            DATASET_PATH, fold, file_name
        )

        if not os.path.exists(file_path):
            skipped += 1
            continue

        # Load audio
        audio, sr = load_audio(file_path)
        if audio is None:
            skipped += 1
            continue

        # Get sound mode
        sound_mode = get_sound_mode(cls_name)

        # Augment — 9 versions per clip
        augmented_clips = augment_audio(audio, sr)

        # Extract features from each clip
        for clip in augmented_clips:
            features = extract_features(clip, sr)
            all_features.append(features)
            all_labels.append(label)
            all_modes.append(sound_mode)

    # Convert to numpy arrays
    X = np.array(all_features)  # (samples, 180, 130)
    y = np.array(all_labels)    # (samples,)
    m = np.array(all_modes)     # (samples,) — NEW

    print(f"\nFeature extraction complete!")
    print(f"X shape  : {X.shape}")
    print(f"y shape  : {y.shape}")
    print(f"m shape  : {m.shape}")
    print(f"Skipped  : {skipped} files")

    # Mode distribution
    unique, counts = np.unique(m, return_counts=True)
    print(f"\nMode distribution:")
    for u, c in zip(unique, counts):
        print(f"  {u:<12} : {c} samples")

    # Save features
    os.makedirs(FEATURES_PATH, exist_ok=True)
    np.save(
        os.path.join(FEATURES_PATH, 'X_features.npy'), X
    )
    np.save(
        os.path.join(FEATURES_PATH, 'y_labels.npy'), y
    )
    np.save(
        os.path.join(FEATURES_PATH, 'y_modes.npy'), m
    )

    print(f"\nFiles saved to '{FEATURES_PATH}/' folder:")
    print(f"  X_features.npy  → feature arrays")
    print(f"  y_labels.npy    → class labels")
    print(f"  y_modes.npy     → sound modes")
    print(f"\nReady for model training!")

# ─────────────────────────────────────────
# STEP 5 — VERIFY SAVED FEATURES
# ─────────────────────────────────────────
def verify_features():
    """
    Quick verification that saved features
    are correct and ready for training.
    """
    print("\nVerifying saved features...")

    X_path = os.path.join(FEATURES_PATH, 'X_features.npy')
    y_path = os.path.join(FEATURES_PATH, 'y_labels.npy')
    m_path = os.path.join(FEATURES_PATH, 'y_modes.npy')

    if not os.path.exists(X_path):
        print("X_features.npy not found!")
        return False

    X = np.load(X_path)
    y = np.load(y_path)
    m = np.load(m_path)

    print(f"X shape        : {X.shape}")
    print(f"y shape        : {y.shape}")
    print(f"m shape        : {m.shape}")
    print(f"X min value    : {X.min():.4f}")
    print(f"X max value    : {X.max():.4f}")
    print(f"Unique classes : {np.unique(y)}")
    print(f"Unique modes   : {np.unique(m)}")
    print(f"\nVerification passed!")

    return True

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("Guardian Ear — Feature Extraction")
    print("Starting dataset processing...\n")

    # Process dataset
    process_dataset()

    # Verify output
    verify_features()