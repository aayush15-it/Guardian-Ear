"""CRNN model architectures for GuardianEar sound classification.

This module provides two model builders:
  - ``build_attention_crnn``  – upgraded architecture with residual blocks,
    bidirectional LSTMs, and temporal attention.
  - ``build_legacy_crnn``     – original v2 architecture kept for backward
    compatibility and checkpoint loading.

Typical usage::

    from src.models.crnn import build_attention_crnn

    model = build_attention_crnn(
        input_shape=(180, 130, 1),
        num_classes=10,
        config={"use_mixed_precision": False},
    )
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Bidirectional,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LSTM,
    Lambda,
    MaxPooling2D,
    Multiply,
    Permute,
    RepeatVector,
    Reshape,
)
from tensorflow.keras.models import Model

from src.utils.logger import get_logger

log = get_logger("models.crnn")

# ──────────────────────────────────────────────────────────────────────────────
# Default hyper-parameter map (can be overridden through *config* dict)
# ──────────────────────────────────────────────────────────────────────────────
_DEFAULTS: Dict[str, Any] = {
    # CNN block filters
    "cnn_filters": [32, 64, 128, 256],
    # Dropout after each CNN block
    "cnn_dropout": 0.25,
    # LSTM units for the two stacked BiLSTM layers
    "lstm_units": [128, 64],
    # Dropout after each BiLSTM
    "lstm_dropout": 0.3,
    # Dense head
    "dense_units": [256, 128],
    "dense_dropouts": [0.4, 0.3],
    # Mixed-precision training
    "use_mixed_precision": False,
}


def _get(config: Optional[Dict[str, Any]], key: str) -> Any:
    """Retrieve *key* from *config*, falling back to ``_DEFAULTS``."""
    if config and key in config:
        return config[key]
    return _DEFAULTS[key]


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _conv_bn_relu(x: tf.Tensor, filters: int, kernel_size: Tuple[int, int] = (3, 3)) -> tf.Tensor:
    """Conv2D → BatchNormalization → ReLU (no inline activation for BN order)."""
    x = Conv2D(filters, kernel_size, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def _cnn_block(
    x: tf.Tensor,
    filters: int,
    dropout_rate: float,
    use_residual: bool = False,
) -> tf.Tensor:
    """Dual-conv CNN block with optional residual skip connection.

    Architecture per block:
        Conv2D → BN → ReLU → Conv2D → BN → ReLU → (+ residual) → MaxPool → Dropout

    Args:
        x: Input tensor.
        filters: Number of convolution filters.
        dropout_rate: Spatial dropout probability after pooling.
        use_residual: If *True*, add a 1×1 conv skip connection around the
            two convolutional layers (before pooling).

    Returns:
        Output tensor after the block.
    """
    shortcut = x

    x = _conv_bn_relu(x, filters)
    x = _conv_bn_relu(x, filters)

    if use_residual:
        # 1×1 projection to match channel dimension
        shortcut = Conv2D(filters, (1, 1), padding="same", use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)
        x = Add()([x, shortcut])
        x = Activation("relu")(x)

    x = MaxPooling2D((2, 2))(x)
    x = Dropout(dropout_rate)(x)
    return x


def _temporal_attention(lstm_out: tf.Tensor, features_dim: int) -> tf.Tensor:
    """Soft temporal attention that learns which timesteps matter most.

    Args:
        lstm_out: Tensor of shape ``(batch, timesteps, features_dim)``.
        features_dim: Last dimension of *lstm_out*.

    Returns:
        Context vector of shape ``(batch, features_dim)``.
    """
    attention_weights = Dense(1, activation="tanh")(lstm_out)
    attention_weights = Flatten()(attention_weights)
    attention_weights = Activation("softmax")(attention_weights)
    attention_weights = RepeatVector(features_dim)(attention_weights)
    attention_weights = Permute([2, 1])(attention_weights)

    attended = Multiply()([lstm_out, attention_weights])
    context = Lambda(lambda t: K.sum(t, axis=1), name="attention_context")(attended)
    return context


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC API – Upgraded Attention-CRNN
# ──────────────────────────────────────────────────────────────────────────────

def build_attention_crnn(
    input_shape: Tuple[int, ...],
    num_classes: int,
    config: Optional[Dict[str, Any]] = None,
) -> Model:
    """Build the upgraded Attention-BiLSTM-CRNN model.

    Improvements over the legacy architecture:
      * 4 CNN blocks with dual convolutions (deeper feature extraction).
      * Residual skip connections on blocks 2 and 4.
      * Two stacked Bidirectional LSTMs.
      * Soft temporal attention over LSTM outputs.
      * Larger classification head with heavier regularisation.
      * Optional mixed-precision (float16) support.

    Args:
        input_shape: Shape of a single sample **with** channel dimension,
            e.g. ``(180, 130, 1)``.
        num_classes: Number of target sound classes.
        config: Optional dictionary overriding default hyper-parameters.
            Recognised keys match ``_DEFAULTS``.

    Returns:
        A compiled ``tf.keras.Model`` instance named
        ``GuardianEar_AttentionCRNN``.
    """
    # ── Optional mixed-precision ──
    if _get(config, "use_mixed_precision"):
        log.info("Enabling mixed_float16 global policy.")
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    cnn_filters = _get(config, "cnn_filters")
    cnn_dropout = _get(config, "cnn_dropout")
    lstm_units = _get(config, "lstm_units")
    lstm_dropout = _get(config, "lstm_dropout")
    dense_units = _get(config, "dense_units")
    dense_dropouts = _get(config, "dense_dropouts")

    # ── Input ──
    inputs = Input(shape=input_shape, name="spectrogram_input")

    # ── CNN Blocks ──
    # Block 1 – no residual
    x = _cnn_block(inputs, cnn_filters[0], cnn_dropout, use_residual=False)
    # Block 2 – with residual
    x = _cnn_block(x, cnn_filters[1], cnn_dropout, use_residual=True)
    # Block 3 – no residual
    x = _cnn_block(x, cnn_filters[2], cnn_dropout, use_residual=False)
    # Block 4 – with residual
    x = _cnn_block(x, cnn_filters[3], cnn_dropout, use_residual=True)

    # ── Reshape for RNN: (timesteps, features) ──
    cnn_shape = x.shape  # (batch, H, W, C)
    timesteps = cnn_shape[1]
    features = cnn_shape[2] * cnn_shape[3]
    x = Reshape((timesteps, features), name="reshape_for_rnn")(x)

    # ── Stacked Bidirectional LSTM ──
    x = Bidirectional(
        LSTM(lstm_units[0], return_sequences=True), name="bilstm_1"
    )(x)
    x = Dropout(lstm_dropout)(x)

    x = Bidirectional(
        LSTM(lstm_units[1], return_sequences=True), name="bilstm_2"
    )(x)
    x = Dropout(lstm_dropout)(x)

    # ── Temporal Attention ──
    # features_dim = 2 * lstm_units[1] (bidirectional doubles the units)
    features_dim = 2 * lstm_units[1]
    x = _temporal_attention(x, features_dim)

    # ── Classification Head ──
    for units, drop in zip(dense_units, dense_dropouts):
        x = Dense(units, activation="relu")(x)
        x = Dropout(drop)(x)

    # Final output – explicit float32 for mixed-precision safety
    outputs = Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)

    model = Model(inputs, outputs, name="GuardianEar_AttentionCRNN")

    # Log summary
    log.info("Attention-CRNN model built successfully.")
    model.summary(print_fn=log.info)

    return model


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC API – Legacy CRNN (backward-compatible with 02_train_model.py)
# ──────────────────────────────────────────────────────────────────────────────

def build_legacy_crnn(
    input_shape: Tuple[int, ...],
    num_classes: int,
) -> Model:
    """Re-create the original CRNN architecture from ``02_train_model.py``.

    This is provided for **backward compatibility** so that existing
    ``best_model.h5`` checkpoints can be loaded without architecture
    mismatch errors.

    Args:
        input_shape: Input tensor shape, e.g. ``(180, 130, 1)``.
        num_classes: Number of output classes.

    Returns:
        An **uncompiled** ``tf.keras.Model`` named ``GuardianEar_CRNN_v2``.
    """
    from tensorflow.keras.layers import LSTM as _LSTM  # noqa: already imported

    inputs = Input(shape=input_shape)

    # ── CNN Block 1 ──
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # ── CNN Block 2 ──
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # ── CNN Block 3 ──
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # ── CNN Block 4 ──
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # ── Reshape for LSTM ──
    shape = x.shape
    x = Reshape((shape[1], shape[2] * shape[3]))(x)

    # ── LSTM ──
    x = _LSTM(128, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = _LSTM(64, return_sequences=False)(x)
    x = Dropout(0.3)(x)

    # ── Dense Head ──
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="GuardianEar_CRNN_v2")

    log.info("Legacy CRNN model built (uncompiled).")
    model.summary(print_fn=log.info)

    return model
