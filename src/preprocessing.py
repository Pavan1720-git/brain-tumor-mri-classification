"""
preprocessing.py — Image preprocessing and data generators.

Provides:
  - Architecture-specific preprocessing functions
  - Keras ImageDataGenerator builders for train / valid / test
  - A standalone preprocess_image() for use in the Streamlit app
"""

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import (
    TRAIN_DIR, VALID_DIR, TEST_DIR,
    IMG_SIZE, BATCH_SIZE, CLASS_NAMES,
    AUG_ROTATION_RANGE, AUG_WIDTH_SHIFT, AUG_HEIGHT_SHIFT,
    AUG_ZOOM_RANGE, AUG_BRIGHTNESS_RANGE,
    AUG_HORIZONTAL_FLIP, AUG_VERTICAL_FLIP, AUG_FILL_MODE,
    RANDOM_SEED,
)


# ---------------------------------------------------------------------------
# Architecture-specific preprocessing lookup
# ---------------------------------------------------------------------------

def get_preprocess_fn(model_name: str):
    """
    Return the correct Keras preprocessing function for a given architecture.
    Different pretrained models expect different pixel value ranges.

    model_name options:
      - "custom"        → simple /255 normalisation (scale to [0, 1])
      - "mobilenetv2"   → tf.keras.applications.mobilenet_v2.preprocess_input  ([-1, 1])
      - "efficientnetb0"→ tf.keras.applications.efficientnet.preprocess_input  ([-1, 1] for B0)
      - "resnet50"      → tf.keras.applications.resnet.preprocess_input
    """
    name = model_name.lower()
    if name == "custom":
        return lambda x: x / 255.0
    elif name == "mobilenetv2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        return preprocess_input
    elif name in ("efficientnetb0", "efficientnet"):
        from tensorflow.keras.applications.efficientnet import preprocess_input
        return preprocess_input
    elif name == "resnet50":
        from tensorflow.keras.applications.resnet import preprocess_input
        return preprocess_input
    else:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            "Choose from: custom, mobilenetv2, efficientnetb0, resnet50"
        )


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def build_generators(model_name: str = "custom"):
    """
    Build and return (train_gen, valid_gen, test_gen) using ImageDataGenerator.

    Augmentation is applied ONLY to the training generator.
    Validation and test generators only apply the architecture-specific
    preprocessing — no random transforms.

    Args:
        model_name: Architecture name (determines preprocessing function).

    Returns:
        (train_generator, valid_generator, test_generator)
    """
    preprocess_fn = get_preprocess_fn(model_name)

    # ------------------------------------------------------------------
    # TRAIN generator — with conservative augmentation
    # Augmentation parameters are read from config.py
    # ------------------------------------------------------------------
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=AUG_ROTATION_RANGE,
        width_shift_range=AUG_WIDTH_SHIFT,
        height_shift_range=AUG_HEIGHT_SHIFT,
        zoom_range=AUG_ZOOM_RANGE,
        brightness_range=AUG_BRIGHTNESS_RANGE,
        horizontal_flip=AUG_HORIZONTAL_FLIP,  # False — anatomical orientation matters
        vertical_flip=AUG_VERTICAL_FLIP,       # False
        fill_mode=AUG_FILL_MODE,
    )

    # ------------------------------------------------------------------
    # VALID / TEST generators — preprocessing only, no augmentation
    # ------------------------------------------------------------------
    eval_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
    )

    # Shared flow parameters
    flow_kwargs = dict(
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,   # fix alphabetical order: glioma, meningioma, no_tumor, pituitary
        seed=RANDOM_SEED,
    )

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        shuffle=True,
        **flow_kwargs,
    )

    valid_gen = eval_datagen.flow_from_directory(
        VALID_DIR,
        shuffle=False,
        **flow_kwargs,
    )

    test_gen = eval_datagen.flow_from_directory(
        TEST_DIR,
        shuffle=False,
        **flow_kwargs,
    )

    return train_gen, valid_gen, test_gen


# ---------------------------------------------------------------------------
# Standalone image preprocessor (used by Streamlit app and manual testing)
# ---------------------------------------------------------------------------

def preprocess_image(image_input, model_name: str = "custom") -> np.ndarray:
    """
    Preprocess a single image for model inference.

    Args:
        image_input: A file path (str) OR a PIL.Image object OR a numpy array.
        model_name:  Architecture name to select the correct preprocessing.

    Returns:
        np.ndarray of shape (1, IMG_SIZE[0], IMG_SIZE[1], 3)  — batch of 1.
    """
    # 1. Load image
    if isinstance(image_input, str):
        img = Image.open(image_input)
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input.astype("uint8"))
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    # 2. Convert to RGB (handles grayscale, RGBA, etc.)
    img = img.convert("RGB")

    # 3. Resize to model's expected input size
    img = img.resize((IMG_SIZE[1], IMG_SIZE[0]), Image.LANCZOS)

    # 4. Convert to float32 numpy array
    arr = np.array(img, dtype=np.float32)

    # 5. Apply architecture-specific preprocessing
    preprocess_fn = get_preprocess_fn(model_name)
    arr = preprocess_fn(arr)

    # 6. Add batch dimension → (1, H, W, 3)
    return np.expand_dims(arr, axis=0)
