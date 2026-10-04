"""
models.py — Model factory for Brain Tumor MRI Classification.

Provides:
  - build_custom_cnn()    : Custom CNN built from scratch
  - build_mobilenetv2()   : MobileNetV2 transfer-learning model
  - build_efficientnetb0(): EfficientNetB0 transfer-learning model
  - get_callbacks()       : EarlyStopping + ModelCheckpoint + ReduceLROnPlateau
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from src.config import (
    IMG_SIZE, IMG_CHANNELS, NUM_CLASSES,
    LR_INITIAL, LR_FINETUNE,
    EPOCHS_FROZEN, EPOCHS_FINETUNE,
    ES_PATIENCE, ES_MIN_DELTA,
    MOBILENET_UNFREEZE_LAYERS,
    EFFICIENTNET_UNFREEZE_LAYERS,
    MODELS_DIR,
)


# ===========================================================================
# 1.  CUSTOM CNN
# ===========================================================================

def build_custom_cnn(
    input_shape: tuple = (*IMG_SIZE, IMG_CHANNELS),
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LR_INITIAL,
) -> tf.keras.Model:
    """
    Custom CNN built from scratch.

    Architecture:
        Input (224×224×3)
        → [Conv2D-32 → BN → ReLU → MaxPool] × 1
        → [Conv2D-64 → BN → ReLU → MaxPool] × 1
        → [Conv2D-128 → BN → ReLU → MaxPool] × 1
        → [Conv2D-256 → BN → ReLU → MaxPool] × 1
        → GlobalAveragePooling
        → Dense-256 → Dropout(0.5)
        → Dense-128 → Dropout(0.3)
        → Dense(num_classes, softmax)

    Designed to be appropriately sized for ~1,700 training images.
    L2 regularisation on dense layers reduces overfitting.
    """
    inputs = tf.keras.Input(shape=input_shape, name="input")

    # ----- Block 1 -----
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # ----- Block 2 -----
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # ----- Block 3 -----
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # ----- Block 4 -----
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

    # ----- Head -----
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(
        256, activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense1",
    )(x)
    x = layers.Dropout(0.5, name="dropout1")(x)
    x = layers.Dense(
        128, activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense2",
    )(x)
    x = layers.Dropout(0.3, name="dropout2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = tf.keras.Model(inputs, outputs, name="CustomCNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ===========================================================================
# 2.  MOBILENETV2
# ===========================================================================

def build_mobilenetv2(
    input_shape: tuple = (*IMG_SIZE, IMG_CHANNELS),
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LR_INITIAL,
) -> tf.keras.Model:
    """
    MobileNetV2 transfer-learning model.

    Phase 1 (frozen): Only classification head is trained.
    Phase 2 (fine-tuning): Top `MOBILENET_UNFREEZE_LAYERS` layers unfrozen
    with a reduced learning rate — call fine_tune_mobilenetv2() for phase 2.

    Preprocessing: expects inputs preprocessed with
    tf.keras.applications.mobilenet_v2.preprocess_input (range [-1, 1]).
    """
    base = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False   # Freeze all base layers for phase 1

    inputs = tf.keras.Input(shape=input_shape, name="input")
    x = base(inputs, training=False)           # training=False → BN uses inference stats
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(
        256, activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense1",
    )(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = tf.keras.Model(inputs, outputs, name="MobileNetV2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def fine_tune_mobilenetv2(
    model: tf.keras.Model,
    n_unfreeze: int = MOBILENET_UNFREEZE_LAYERS,
    learning_rate: float = LR_FINETUNE,
) -> tf.keras.Model:
    """
    Prepare MobileNetV2 for fine-tuning:
    Unfreeze the last `n_unfreeze` layers of the base and recompile
    with a smaller learning rate to avoid catastrophic forgetting.
    """
    base = model.layers[1]   # MobileNetV2 base is the 2nd layer (index 1)
    base.trainable = True

    # Freeze all layers except the last n_unfreeze
    for layer in base.layers[:-n_unfreeze]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ===========================================================================
# 3.  EFFICIENTNETB0
# ===========================================================================

def build_efficientnetb0(
    input_shape: tuple = (*IMG_SIZE, IMG_CHANNELS),
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LR_INITIAL,
) -> tf.keras.Model:
    """
    EfficientNetB0 transfer-learning model.

    Preprocessing: expects inputs preprocessed with
    tf.keras.applications.efficientnet.preprocess_input.
    """
    base = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False   # Freeze all base layers for phase 1

    inputs = tf.keras.Input(shape=input_shape, name="input")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(
        256, activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense1",
    )(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = tf.keras.Model(inputs, outputs, name="EfficientNetB0")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def fine_tune_efficientnetb0(
    model: tf.keras.Model,
    n_unfreeze: int = EFFICIENTNET_UNFREEZE_LAYERS,
    learning_rate: float = LR_FINETUNE,
) -> tf.keras.Model:
    """
    Prepare EfficientNetB0 for fine-tuning.
    Unfreeze the last `n_unfreeze` base layers with a smaller LR.
    """
    base = model.layers[1]
    base.trainable = True

    for layer in base.layers[:-n_unfreeze]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ===========================================================================
# 4.  TRAINING CALLBACKS
# ===========================================================================

def get_callbacks(
    checkpoint_path: str,
    patience: int = ES_PATIENCE,
    min_delta: float = ES_MIN_DELTA,
    monitor: str = "val_loss",
) -> list:
    """
    Return a list of standard Keras callbacks:
      - ModelCheckpoint  : saves best model weights
      - EarlyStopping    : stops when val_loss stops improving
      - ReduceLROnPlateau: halves LR if no improvement after 5 epochs
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor,
        save_best_only=True,
        save_weights_only=False,
        mode="min" if "loss" in monitor else "max",
        verbose=1,
    )

    early_stop = EarlyStopping(
        monitor=monitor,
        patience=patience,
        min_delta=min_delta,
        mode="min" if "loss" in monitor else "max",
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor=monitor,
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1,
    )

    return [checkpoint, early_stop, reduce_lr]
