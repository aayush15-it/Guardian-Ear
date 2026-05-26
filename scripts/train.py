#!/usr/bin/env python
"""Guardian Ear — Refactored CRNN Training Pipeline.

Replaces the legacy ``02_train_model.py`` with critical improvements:

* **No data leakage** — normalisation stats are computed on the training
  split only and then applied to validation / test sets.
* **Class-weighted loss** — handles natural class imbalance via
  ``sklearn.utils.class_weight``.
* **Cosine-decay-with-restarts LR schedule** *plus* ``ReduceLROnPlateau``
  safety net.
* **Comprehensive evaluation** — per-class P/R/F1, confusion matrix image,
  ROC curves, AUC scores, and training curves.
* **Multi-format export** — ``.h5``, TFLite (INT8 quantised), and ONNX.
* **CLI** via ``argparse`` with optional YAML config override.
* **HDF5 / .npy feature loading** support.

Usage::

    python scripts/train.py --config configs/config.yaml --epochs 50 --batch-size 32
    python scripts/train.py  # uses built-in defaults

Requires: tensorflow>=2.13, scikit-learn, matplotlib, pyyaml, tf2onnx
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Resolve project root so ``src.*`` imports work regardless of CWD ──
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.logger import get_logger  # noqa: E402

log = get_logger("train")

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Built-in defaults (overridden by YAML / CLI)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULTS: Dict[str, Any] = {
    # Paths
    "features_dir": "features",
    "output_dir": "model",
    # Data
    "num_classes": 10,
    "class_names": [
        "air_conditioner", "car_horn", "children_playing",
        "dog_bark", "drilling", "engine_idling",
        "gun_shot", "jackhammer", "siren", "street_music",
    ],
    "alert_sounds": ["gun_shot", "siren", "jackhammer"],
    "assistive_sounds": [
        "dog_bark", "children_playing", "air_conditioner",
        "street_music", "engine_idling", "car_horn", "drilling",
    ],
    # Training
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "val_split": 0.15,
    "test_split": 0.15,
    "random_seed": 42,
    # Model
    "model_type": "attention",  # "attention" | "legacy"
    "use_mixed_precision": False,
    # Feature format
    "feature_format": "npy",  # "npy" | "hdf5"
    # Export
    "export_tflite": True,
    "export_onnx": True,
    "tflite_int8": True,
}


# ──────────────────────────────────────────────────────────────────────────────
# Configuration helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed dictionary.
    """
    import yaml  # deferred so yaml is optional at import time

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge DEFAULTS ← YAML ← CLI into a single config dict.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Fully resolved configuration dictionary.
    """
    cfg: Dict[str, Any] = dict(DEFAULTS)

    # Layer 2 – YAML overrides
    if args.config and os.path.isfile(args.config):
        log.info("Loading config from %s", args.config)
        yaml_cfg = _load_yaml_config(args.config)
        cfg.update(yaml_cfg)

    # Layer 3 – CLI overrides (only if explicitly set)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        cfg["learning_rate"] = args.learning_rate
    if args.features_dir is not None:
        cfg["features_dir"] = args.features_dir
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.model_type is not None:
        cfg["model_type"] = args.model_type

    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_features_npy(features_dir: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load features from ``.npy`` files.

    Args:
        features_dir: Directory containing ``X_features.npy``,
            ``y_labels.npy``, and optionally ``y_modes.npy``.

    Returns:
        Tuple of ``(X, y, modes)`` where *modes* may be ``None``.

    Raises:
        FileNotFoundError: If required feature files are missing.
    """
    x_path = os.path.join(features_dir, "X_features.npy")
    y_path = os.path.join(features_dir, "y_labels.npy")
    m_path = os.path.join(features_dir, "y_modes.npy")

    if not os.path.exists(x_path):
        raise FileNotFoundError(
            f"X_features.npy not found in {features_dir}. "
            "Run 01_feature_extraction.py first."
        )

    X = np.load(x_path)
    y = np.load(y_path)
    modes = np.load(m_path) if os.path.exists(m_path) else None

    log.info("Loaded .npy features — X: %s, y: %s", X.shape, y.shape)
    return X, y, modes


def load_features_hdf5(features_dir: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load features from an HDF5 archive.

    Expects a file ``features.h5`` inside *features_dir* with datasets
    ``X``, ``y``, and optionally ``modes``.

    Args:
        features_dir: Directory containing ``features.h5``.

    Returns:
        Tuple of ``(X, y, modes)``.

    Raises:
        FileNotFoundError: If the HDF5 file is missing.
    """
    import h5py

    h5_path = os.path.join(features_dir, "features.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"features.h5 not found in {features_dir}")

    with h5py.File(h5_path, "r") as hf:
        X = hf["X"][:]
        y = hf["y"][:]
        modes = hf["modes"][:] if "modes" in hf else None

    log.info("Loaded HDF5 features — X: %s, y: %s", X.shape, y.shape)
    return X, y, modes


def load_features(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Dispatch to the correct loader based on ``cfg['feature_format']``.

    Args:
        cfg: Full configuration dictionary.

    Returns:
        Tuple of ``(X, y, modes)``.
    """
    fmt = cfg.get("feature_format", "npy")
    features_dir = cfg["features_dir"]

    if fmt == "hdf5":
        return load_features_hdf5(features_dir)
    return load_features_npy(features_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Data preparation — FIXES DATA LEAKAGE
# ──────────────────────────────────────────────────────────────────────────────

def prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Split, normalise, and encode data **without leaking test information**.

    The critical fix: normalisation statistics (min / max) are computed
    **only on the training set** and then applied identically to validation
    and test sets.

    Args:
        X: Raw feature array of shape ``(N, H, W)``.
        y: Integer label array of shape ``(N,)``.
        cfg: Configuration dict with split ratios and paths.

    Returns:
        Dictionary with keys ``X_train``, ``X_val``, ``X_test``,
        ``y_train``, ``y_val``, ``y_test``, ``y_train_int``,
        ``y_val_int``, ``y_test_int``, ``label_binarizer``,
        ``norm_min``, ``norm_max``.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelBinarizer

    seed = cfg["random_seed"]
    val_ratio = cfg["val_split"]
    test_ratio = cfg["test_split"]
    output_dir = cfg["output_dir"]

    # ── 1. Add channel dim ──
    X = X[..., np.newaxis]  # (N, H, W, 1)

    # ── 2. Encode labels ──
    lb = LabelBinarizer()
    y_encoded = lb.fit_transform(y)

    # ── 3. Split FIRST — before any normalisation ──
    temp_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded,
        test_size=temp_ratio,
        random_state=seed,
        stratify=y_encoded,
    )
    relative_test = test_ratio / temp_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test,
        random_state=seed,
        stratify=y_temp,
    )

    log.info(
        "Data split — train: %d, val: %d, test: %d",
        len(X_train), len(X_val), len(X_test),
    )

    # ── 4. Compute normalisation from TRAINING SET ONLY ──
    x_min = float(X_train.min())
    x_max = float(X_train.max())
    denom = x_max - x_min + 1e-8

    X_train = (X_train - x_min) / denom
    X_val = (X_val - x_min) / denom
    X_test = (X_test - x_min) / denom

    # Persist normalisation params for inference
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X_min.npy"), np.array([x_min]))
    np.save(os.path.join(output_dir, "X_max.npy"), np.array([x_max]))
    log.info("Normalisation stats saved (min=%.4f, max=%.4f)", x_min, x_max)

    # Integer labels for metrics
    y_train_int = np.argmax(y_train, axis=1)
    y_val_int = np.argmax(y_val, axis=1)
    y_test_int = np.argmax(y_test, axis=1)

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "y_train_int": y_train_int, "y_val_int": y_val_int, "y_test_int": y_test_int,
        "label_binarizer": lb,
        "norm_min": x_min, "norm_max": x_max,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Class weights
# ──────────────────────────────────────────────────────────────────────────────

def compute_class_weights(y_train_int: np.ndarray, num_classes: int) -> Dict[int, float]:
    """Compute per-class weights to counter class imbalance.

    Args:
        y_train_int: Integer training labels.
        num_classes: Total number of classes.

    Returns:
        Dictionary mapping class index to weight.
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.arange(num_classes)
    weights = compute_class_weight("balanced", classes=classes, y=y_train_int)
    cw = {i: float(w) for i, w in enumerate(weights)}
    log.info("Class weights: %s", cw)
    return cw


# ──────────────────────────────────────────────────────────────────────────────
# Model building
# ──────────────────────────────────────────────────────────────────────────────

def build_model(input_shape: Tuple[int, ...], cfg: Dict[str, Any]) -> "tf.keras.Model":
    """Construct and compile the requested model variant.

    Args:
        input_shape: Shape of a single input sample (H, W, C).
        cfg: Configuration dictionary.

    Returns:
        Compiled Keras model.
    """
    import tensorflow as tf
    from src.models.crnn import build_attention_crnn, build_legacy_crnn

    num_classes = cfg["num_classes"]
    lr = cfg["learning_rate"]
    model_type = cfg.get("model_type", "attention")

    if model_type == "legacy":
        log.info("Building legacy CRNN model.")
        model = build_legacy_crnn(input_shape, num_classes)
    else:
        log.info("Building Attention-CRNN model.")
        model = build_attention_crnn(input_shape, num_classes, config=cfg)

    # ── Learning-rate schedule ──
    steps_per_epoch = max(1, cfg.get("_steps_per_epoch", 100))
    total_steps = steps_per_epoch * cfg["epochs"]

    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=lr,
        first_decay_steps=steps_per_epoch * 10,  # restart every 10 epochs
        t_mul=1.5,
        m_mul=0.9,
        alpha=1e-6,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    log.info("Model compiled — LR schedule: CosineDecayRestarts (initial=%.1e)", lr)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_model(
    model: "tf.keras.Model",
    data: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    class_weights: Dict[int, float],
) -> "tf.keras.callbacks.History":
    """Run the training loop with callbacks.

    Args:
        model: Compiled Keras model.
        data: Dictionary from ``prepare_data``.
        cfg: Configuration dictionary.
        class_weights: Per-class loss weights.

    Returns:
        Keras ``History`` object.
    """
    import tensorflow as tf

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, "best_model.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    log.info(
        "Training — epochs: %d, batch: %d, classes: %d",
        cfg["epochs"], cfg["batch_size"], cfg["num_classes"],
    )

    history = model.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    return history


# ──────────────────────────────────────────────────────────────────────────────
# Comprehensive evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model: "tf.keras.Model",
    data: Dict[str, np.ndarray],
    history: "tf.keras.callbacks.History",
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Run full evaluation and save all artefacts.

    Produces:
      * Per-class precision, recall, F1
      * Overall accuracy
      * Confusion matrix image
      * ROC curves per class image
      * AUC scores per class
      * Training curves image

    Args:
        model: Trained Keras model.
        data: Dictionary from ``prepare_data``.
        history: Training history object.
        cfg: Configuration dictionary.

    Returns:
        Dictionary of evaluation metrics.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        accuracy_score,
        auc,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_curve,
    )

    output_dir = cfg["output_dir"]
    class_names: List[str] = cfg["class_names"]
    num_classes: int = cfg["num_classes"]
    alert_sounds: List[str] = cfg.get("alert_sounds", [])

    X_test = data["X_test"]
    y_test = data["y_test"]
    y_true = data["y_test_int"]

    # ── Predictions ──
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # ── Overall accuracy ──
    acc = accuracy_score(y_true, y_pred)
    loss, keras_acc = model.evaluate(X_test, y_test, verbose=0)
    log.info("Test accuracy: %.2f%%  |  Test loss: %.4f", acc * 100, loss)

    # ── Per-class metrics ──
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(num_classes), zero_division=0,
    )

    log.info("Per-class results:")
    for i, name in enumerate(class_names):
        mode = "ALERT" if name in alert_sounds else "ASSISTIVE"
        log.info(
            "  %-25s P=%.3f  R=%.3f  F1=%.3f  n=%d  [%s]",
            name, precision[i], recall[i], f1[i], support[i], mode,
        )

    report_str = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0,
    )
    log.info("\n%s", report_str)

    # ── Confusion matrix ──
    cm = confusion_matrix(y_true, y_pred)
    _plot_confusion_matrix(cm, class_names, output_dir)

    # ── ROC curves + AUC ──
    auc_scores = _plot_roc_curves(y_test, y_pred_proba, class_names, output_dir)

    # ── Training curves ──
    _plot_training_curves(history, output_dir)

    metrics = {
        "test_accuracy": float(acc),
        "test_loss": float(loss),
        "per_class_precision": {class_names[i]: float(precision[i]) for i in range(num_classes)},
        "per_class_recall": {class_names[i]: float(recall[i]) for i in range(num_classes)},
        "per_class_f1": {class_names[i]: float(f1[i]) for i in range(num_classes)},
        "per_class_auc": auc_scores,
    }

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    log.info("Evaluation metrics saved to %s", metrics_path)

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_dir: str,
) -> None:
    """Save a labelled confusion-matrix heatmap.

    Args:
        cm: Confusion matrix array.
        class_names: List of class display names.
        output_dir: Directory to save the image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Guardian Ear — Confusion Matrix", fontsize=14)
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()

    path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Confusion matrix saved to %s", path)


def _plot_roc_curves(
    y_test_onehot: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: List[str],
    output_dir: str,
) -> Dict[str, float]:
    """Plot per-class ROC curves and return AUC scores.

    Args:
        y_test_onehot: One-hot encoded true labels.
        y_pred_proba: Predicted probabilities.
        class_names: List of class names.
        output_dir: Directory to save the image.

    Returns:
        Dictionary mapping class name to AUC score.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    n_classes = len(class_names)
    auc_scores: Dict[str, float] = {}

    fig, ax = plt.subplots(figsize=(12, 8))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_onehot[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[class_names[i]] = float(roc_auc)
        ax.plot(fpr, tpr, linewidth=1.5, label=f"{class_names[i]} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Guardian Ear — ROC Curves (per class)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "roc_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("ROC curves saved to %s", path)

    return auc_scores


def _plot_training_curves(
    history: "tf.keras.callbacks.History",
    output_dir: str,
) -> None:
    """Save accuracy and loss training curves.

    Args:
        history: Keras training history.
        output_dir: Directory to save the image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
    axes[0].set_title("Guardian Ear — Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"], label="Train Loss", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    axes[1].set_title("Guardian Ear — Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Training curves saved to %s", path)


# ──────────────────────────────────────────────────────────────────────────────
# Model export
# ──────────────────────────────────────────────────────────────────────────────

def save_and_export(
    model: "tf.keras.Model",
    data: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
) -> None:
    """Save the model in multiple formats and persist metadata.

    Exports:
      * ``.h5`` (full Keras model)
      * TFLite with optional INT8 quantisation
      * ONNX (via ``tf2onnx``)
      * ``class_info.json`` with all metadata

    Args:
        model: Trained Keras model.
        data: Data dictionary (used for representative dataset in INT8 quant).
        cfg: Configuration dictionary.
    """
    import tensorflow as tf

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Keras H5 ──
    h5_path = os.path.join(output_dir, "guardian_ear_model.h5")
    model.save(h5_path)
    log.info("Keras model saved to %s", h5_path)

    # ── 2. class_info.json ──
    class_info = {
        "class_names": cfg["class_names"],
        "num_classes": cfg["num_classes"],
        "alert_sounds": cfg.get("alert_sounds", []),
        "assistive_sounds": cfg.get("assistive_sounds", []),
        "model_type": cfg.get("model_type", "attention"),
        "sample_rate": cfg.get("sample_rate", 22050),
        "duration": cfg.get("duration", 3),
        "n_mels": cfg.get("n_mels", 128),
        "n_mfcc": cfg.get("n_mfcc", 40),
        "feature_shape": list(cfg.get("feature_shape", [180, 130])),
        "norm_min": data.get("norm_min"),
        "norm_max": data.get("norm_max"),
        "epochs_trained": cfg["epochs"],
        "batch_size": cfg["batch_size"],
    }
    info_path = os.path.join(output_dir, "class_info.json")
    with open(info_path, "w", encoding="utf-8") as fh:
        json.dump(class_info, fh, indent=2)
    log.info("class_info.json saved to %s", info_path)

    # ── 3. TFLite ──
    if cfg.get("export_tflite", True):
        _export_tflite(model, data, cfg)

    # ── 4. ONNX ──
    if cfg.get("export_onnx", True):
        _export_onnx(model, cfg)

    log.info("All model artefacts saved to '%s/'", output_dir)


def _export_tflite(
    model: "tf.keras.Model",
    data: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
) -> None:
    """Convert model to TFLite with optional INT8 quantisation.

    Args:
        model: Trained Keras model.
        data: Data dictionary (``X_train`` used for representative dataset).
        cfg: Configuration dictionary.
    """
    import tensorflow as tf

    output_dir = cfg["output_dir"]

    log.info("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if cfg.get("tflite_int8", True):
        log.info("Applying INT8 full-integer quantisation.")

        X_train = data["X_train"]
        # Use a subset for the representative dataset (max 200 samples)
        num_repr = min(200, len(X_train))

        def representative_dataset():
            """Yield representative samples for INT8 calibration."""
            indices = np.random.choice(len(X_train), num_repr, replace=False)
            for idx in indices:
                sample = X_train[idx : idx + 1].astype(np.float32)
                yield [sample]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,  # fallback for unsupported ops
        ]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    tflite_path = os.path.join(output_dir, "guardian_ear.tflite")
    with open(tflite_path, "wb") as fh:
        fh.write(tflite_model)

    size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
    log.info("TFLite model saved to %s (%.2f MB)", tflite_path, size_mb)


def _export_onnx(model: "tf.keras.Model", cfg: Dict[str, Any]) -> None:
    """Export model to ONNX format via ``tf2onnx``.

    Args:
        model: Trained Keras model.
        cfg: Configuration dictionary.
    """
    output_dir = cfg["output_dir"]
    onnx_path = os.path.join(output_dir, "guardian_ear.onnx")

    try:
        import tf2onnx
        import tensorflow as tf

        # tf2onnx expects a concrete function or SavedModel
        # Save as SavedModel first, then convert
        saved_model_dir = os.path.join(output_dir, "_temp_saved_model")
        model.save(saved_model_dir, save_format="tf")

        import subprocess
        result = subprocess.run(
            [
                sys.executable, "-m", "tf2onnx.convert",
                "--saved-model", saved_model_dir,
                "--output", onnx_path,
                "--opset", "13",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            log.info("ONNX model saved to %s (%.2f MB)", onnx_path, size_mb)
        else:
            log.warning("ONNX export failed: %s", result.stderr)

        # Clean up temp SavedModel
        import shutil
        if os.path.isdir(saved_model_dir):
            shutil.rmtree(saved_model_dir, ignore_errors=True)

    except ImportError:
        log.warning(
            "tf2onnx not installed — skipping ONNX export. "
            "Install with: pip install tf2onnx"
        )
    except Exception as exc:
        log.warning("ONNX export failed: %s", exc)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Guardian Ear — Train CRNN sound classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (overrides built-in defaults).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, dest="batch_size", help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=None, dest="learning_rate", help="Initial learning rate.")
    parser.add_argument("--features-dir", type=str, default=None, dest="features_dir", help="Feature directory path.")
    parser.add_argument("--output-dir", type=str, default=None, dest="output_dir", help="Output / model directory path.")
    parser.add_argument(
        "--model-type", type=str, default=None, dest="model_type",
        choices=["attention", "legacy"],
        help="Model architecture: 'attention' (upgraded) or 'legacy'.",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """End-to-end training pipeline."""
    args = parse_args()
    cfg = build_config(args)

    log.info("=" * 60)
    log.info("   Guardian Ear — CRNN Training Pipeline v3")
    log.info("=" * 60)

    # ── 1. Load features ──
    X, y, modes = load_features(cfg)
    log.info("Total samples: %d  |  Feature shape: %s", len(X), X.shape[1:])

    if modes is not None:
        unique, counts = np.unique(modes, return_counts=True)
        for u, c in zip(unique, counts):
            log.info("  Mode %-12s : %d samples", u, c)

    # ── 2. Prepare data (leak-free) ──
    data = prepare_data(X, y, cfg)

    # ── 3. Class weights ──
    class_weights = compute_class_weights(data["y_train_int"], cfg["num_classes"])

    # Inject steps_per_epoch for LR schedule
    cfg["_steps_per_epoch"] = max(1, len(data["X_train"]) // cfg["batch_size"])

    # ── 4. Build model ──
    input_shape = data["X_train"].shape[1:]
    model = build_model(input_shape, cfg)

    # ── 5. Train ──
    history = train_model(model, data, cfg, class_weights)

    # ── 6. Evaluate ──
    metrics = evaluate_model(model, data, history, cfg)

    # ── 7. Save & export ──
    # Attach norm stats to data dict for export
    save_and_export(model, data, cfg)

    log.info("=" * 60)
    log.info("Training pipeline complete!")
    log.info("Test accuracy: %.2f%%", metrics["test_accuracy"] * 100)
    log.info("Guardian Ear model is ready.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
