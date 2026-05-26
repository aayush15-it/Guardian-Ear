import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization,
    MaxPooling2D, Dropout, Reshape,
    LSTM, Dense
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau
)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
FEATURES_PATH = 'features'
MODEL_PATH    = 'model'
NUM_CLASSES   = 10
EPOCHS        = 50
BATCH_SIZE    = 32
LEARNING_RATE = 0.001

CLASS_NAMES = [
    'air_conditioner', 'car_horn',
    'children_playing', 'dog_bark',
    'drilling', 'engine_idling',
    'gun_shot', 'jackhammer',
    'siren', 'street_music'
]

# Dual mode classification
ALERT_SOUNDS = [
    'gun_shot', 'siren', 'jackhammer'
]
ASSISTIVE_SOUNDS = [
    'dog_bark', 'children_playing',
    'air_conditioner', 'street_music',
    'engine_idling', 'car_horn', 'drilling'
]

# ─────────────────────────────────────────
# STEP 1 — LOAD FEATURES
# ─────────────────────────────────────────
def load_features():
    print("=" * 55)
    print("   Guardian Ear — CRNN Model Training v2")
    print("=" * 55)

    X_path = os.path.join(
        FEATURES_PATH, 'X_features.npy'
    )
    y_path = os.path.join(
        FEATURES_PATH, 'y_labels.npy'
    )
    m_path = os.path.join(
        FEATURES_PATH, 'y_modes.npy'
    )

    if not os.path.exists(X_path):
        print("ERROR: X_features.npy not found!")
        print("Please run 01_feature_extraction.py first!")
        return None, None, None

    X = np.load(X_path)
    y = np.load(y_path)

    # Load modes if available
    if os.path.exists(m_path):
        m = np.load(m_path)
    else:
        m = None

    print(f"\nFeatures loaded successfully!")
    print(f"X shape        : {X.shape}")
    print(f"y shape        : {y.shape}")
    print(f"Total samples  : {len(X)}")
    print(f"Total classes  : {NUM_CLASSES}")

    if m is not None:
        unique, counts = np.unique(m, return_counts=True)
        print(f"\nMode distribution:")
        for u, c in zip(unique, counts):
            print(f"  {u:<12} : {c} samples")

    return X, y, m

# ─────────────────────────────────────────
# STEP 2 — PREPARE DATA
# ─────────────────────────────────────────
def prepare_data(X, y):
    print("\nPreparing data...")

    # Normalize features to 0-1 range
    X_min = X.min()
    X_max = X.max()
    X = (X - X_min) / (X_max - X_min + 1e-8)

    # Save normalization values for inference
    os.makedirs(MODEL_PATH, exist_ok=True)
    np.save(
        os.path.join(MODEL_PATH, 'X_min.npy'),
        np.array([X_min])
    )
    np.save(
        os.path.join(MODEL_PATH, 'X_max.npy'),
        np.array([X_max])
    )
    print(f"Normalization values saved!")

    # Add channel dimension → (samples, 180, 130, 1)
    X = X[..., np.newaxis]

    # One-hot encode labels
    lb = LabelBinarizer()
    y_encoded = lb.fit_transform(y)

    # Train / Val / Test split → 70 / 15 / 15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded,
        test_size=0.30,
        random_state=42,
        stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    print(f"\nData split (70/15/15):")
    print(f"  Train samples : {X_train.shape[0]}")
    print(f"  Val samples   : {X_val.shape[0]}")
    print(f"  Test samples  : {X_test.shape[0]}")
    print(f"  Input shape   : {X_train.shape[1:]}")

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        lb
    )

# ─────────────────────────────────────────
# STEP 3 — BUILD CRNN MODEL
# Improved architecture with 4 CNN blocks
# ─────────────────────────────────────────
def build_crnn_model(input_shape):
    print(f"\nBuilding CRNN model...")
    print(f"Input shape : {input_shape}")

    inputs = Input(shape=input_shape)

    # ── CNN Block 1 — Low-level features ──
    x = Conv2D(
        32, (3, 3),
        activation='relu',
        padding='same'
    )(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # ── CNN Block 2 — Mid-level features ──
    x = Conv2D(
        64, (3, 3),
        activation='relu',
        padding='same'
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # ── CNN Block 3 — High-level features ──
    x = Conv2D(
        128, (3, 3),
        activation='relu',
        padding='same'
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # ── CNN Block 4 — Deep features (NEW) ──
    x = Conv2D(
        256, (3, 3),
        activation='relu',
        padding='same'
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # ── Reshape for LSTM ──
    shape = x.shape
    x = Reshape(
        (shape[1], shape[2] * shape[3])
    )(x)

    # ── LSTM Block — Temporal modeling ──
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.3)(x)

    # ── Dense Classification Head ──
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(
        NUM_CLASSES,
        activation='softmax'
    )(x)

    model = Model(
        inputs, outputs,
        name='GuardianEar_CRNN_v2'
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            LEARNING_RATE
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ─────────────────────────────────────────
# STEP 4 — TRAIN MODEL
# ─────────────────────────────────────────
def train_model(
    model, X_train, X_val,
    y_train, y_val
):
    os.makedirs(MODEL_PATH, exist_ok=True)

    callbacks = [
        # Stop early if no improvement
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Save best model automatically
        ModelCheckpoint(
            filepath=os.path.join(
                MODEL_PATH, 'best_model.h5'
            ),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Reduce LR when stuck
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print(f"\nStarting model training...")
    print(f"Epochs        : {EPOCHS}")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"Learning rate : {LEARNING_RATE}")
    print(f"Classes       : {NUM_CLASSES}\n")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    return history

# ─────────────────────────────────────────
# STEP 5 — EVALUATE MODEL
# ─────────────────────────────────────────
def evaluate_model(model, X_test, y_test, lb):
    print("\nEvaluating on test set...")

    loss, accuracy = model.evaluate(
        X_test, y_test, verbose=0
    )
    print(f"\nTest Results:")
    print(f"  Accuracy : {accuracy * 100:.2f}%")
    print(f"  Loss     : {loss:.4f}")

    # Predictions
    y_pred         = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Per-class accuracy
    print(f"\nPer-class accuracy:")
    for i, cls in enumerate(CLASS_NAMES):
        mask = y_true_classes == i
        if mask.sum() > 0:
            cls_acc = (
                y_pred_classes[mask] == i
            ).mean() * 100
            mode = (
                'ALERT' if cls in ALERT_SOUNDS
                else 'ASSISTIVE'
            )
            print(
                f"  {cls:<25} "
                f"{cls_acc:.1f}%  [{mode}]"
            )

    # Full classification report
    print(f"\nClassification Report:")
    print(classification_report(
        y_true_classes,
        y_pred_classes,
        target_names=CLASS_NAMES
    ))

    return y_pred_classes, y_true_classes

# ─────────────────────────────────────────
# STEP 6 — PLOT RESULTS
# ─────────────────────────────────────────
def plot_results(history, y_pred, y_true):
    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        history.history['accuracy'],
        label='Train Accuracy',
        linewidth=2
    )
    axes[0].plot(
        history.history['val_accuracy'],
        label='Val Accuracy',
        linewidth=2
    )
    axes[0].set_title(
        'Guardian Ear — Model Accuracy'
    )
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        history.history['loss'],
        label='Train Loss',
        linewidth=2
    )
    axes[1].plot(
        history.history['val_loss'],
        label='Val Loss',
        linewidth=2
    )
    axes[1].set_title(
        'Guardian Ear — Model Loss'
    )
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(MODEL_PATH, 'training_curves.png'),
        dpi=150, bbox_inches='tight'
    )
    plt.show()
    print("Training curves saved!")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    plt.imshow(
        cm, interpolation='nearest',
        cmap=plt.cm.Blues
    )
    plt.title(
        'Guardian Ear — Confusion Matrix',
        fontsize=14
    )
    plt.colorbar()
    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(
        tick_marks, CLASS_NAMES,
        rotation=45, ha='right'
    )
    plt.yticks(tick_marks, CLASS_NAMES)

    # Add numbers inside cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i,j] > thresh
                else "black",
                fontsize=8
            )

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            MODEL_PATH, 'confusion_matrix.png'
        ),
        dpi=150, bbox_inches='tight'
    )
    plt.show()
    print("Confusion matrix saved!")

# ─────────────────────────────────────────
# STEP 7 — SAVE MODEL + CLASS INFO
# ─────────────────────────────────────────
def save_and_convert(model):
    # Save full Keras model
    model_path = os.path.join(
        MODEL_PATH, 'guardian_ear_model.h5'
    )
    model.save(model_path)
    print(f"\nFull model saved!")

    # Save class information JSON
    class_info = {
        'class_names'      : CLASS_NAMES,
        'num_classes'      : NUM_CLASSES,
        'alert_sounds'     : ALERT_SOUNDS,
        'assistive_sounds' : ASSISTIVE_SOUNDS,
        'sample_rate'      : 22050,
        'duration'         : 3,
        'n_mels'           : 128,
        'n_mfcc'           : 40,
        'feature_shape'    : [180, 130]
    }

    json_path = os.path.join(
        MODEL_PATH, 'class_info.json'
    )
    with open(json_path, 'w') as f:
        json.dump(class_info, f, indent=2)
    print(f"Class info saved as class_info.json")

    # Convert to TensorFlow Lite
    print(f"\nConverting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(
        model
    )
    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]
    tflite_model = converter.convert()

    tflite_path = os.path.join(
        MODEL_PATH, 'guardian_ear.tflite'
    )
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(tflite_path) / (1024*1024)
    print(f"TFLite model saved!")
    print(f"TFLite size : {size_mb:.2f} MB")

    print(f"\nAll model files saved to '{MODEL_PATH}/':")
    print(f"  best_model.h5")
    print(f"  guardian_ear_model.h5")
    print(f"  guardian_ear.tflite")
    print(f"  class_info.json")
    print(f"  training_curves.png")
    print(f"  confusion_matrix.png")
    print(f"  X_min.npy + X_max.npy")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":

    # Load features
    X, y, m = load_features()
    if X is None:
        exit()

    # Prepare data
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        lb
    ) = prepare_data(X, y)

    # Build model
    input_shape = X_train.shape[1:]
    model = build_crnn_model(input_shape)

    # Print model summary
    print("\nCRNN Model Architecture:")
    model.summary()

    # Train
    history = train_model(
        model,
        X_train, X_val,
        y_train, y_val
    )

    # Evaluate
    y_pred, y_true = evaluate_model(
        model, X_test, y_test, lb
    )

    # Plot results
    plot_results(history, y_pred, y_true)

    # Save + convert
    save_and_convert(model)

    print("\nTraining pipeline complete!")
    print("Guardian Ear model is ready!")