"""
config.py — Central configuration for Brain Tumor MRI Classification Project.

All hyperparameters, paths, and settings are defined here so they can be
changed in one place and automatically propagate through the entire project.
"""

import os

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Project root — always resolved relative to this file so no absolute paths
# are hard-coded.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR   = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR  = os.path.join(DATA_DIR, "train")
VALID_DIR  = os.path.join(DATA_DIR, "valid")
TEST_DIR   = os.path.join(DATA_DIR, "test")

# ---------------------------------------------------------------------------
# Class names — derived from actual folder names (alphabetical order, which
# is also what Keras ImageDataGenerator uses internally).
# ---------------------------------------------------------------------------
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
NUM_CLASSES  = len(CLASS_NAMES)

# Human-readable display labels (used in plots, reports, Streamlit)
DISPLAY_NAMES = {
    "glioma":      "Glioma",
    "meningioma":  "Meningioma",
    "no_tumor":    "No Tumor",
    "pituitary":   "Pituitary",
}

# ---------------------------------------------------------------------------
# Image settings
# ---------------------------------------------------------------------------
IMG_SIZE    = (224, 224)   # (height, width) — standard for most pretrained CNNs
IMG_CHANNELS = 3           # RGB

# ---------------------------------------------------------------------------
# Training hyperparameters (optimized for CPU execution with EarlyStopping)
# ---------------------------------------------------------------------------
BATCH_SIZE      = 32
EPOCHS_FROZEN   = 15   # epochs while base is frozen (transfer learning phase-1)
EPOCHS_FINETUNE = 12   # additional epochs after unfreezing top layers
EPOCHS_CUSTOM   = 25   # maximum epochs for custom CNN

# ---------------------------------------------------------------------------
# Optimiser settings
# ---------------------------------------------------------------------------
LR_INITIAL  = 1e-3   # learning rate for initial / custom CNN training
LR_FINETUNE = 1e-5   # reduced learning rate for fine-tuning

# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------
ES_PATIENCE     = 6    # stop if val_loss doesn't improve for 6 epochs
ES_MIN_DELTA    = 1e-4 # minimum improvement considered meaningful

# ---------------------------------------------------------------------------
# Model & results output paths
# ---------------------------------------------------------------------------
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
REPORTS_DIR = os.path.join(RESULTS_DIR, "classification_reports")
MISC_DIR    = os.path.join(RESULTS_DIR, "misclassified_samples")

# Individual model checkpoint paths
CUSTOM_CNN_PATH     = os.path.join(MODELS_DIR, "custom_cnn_best.keras")
MOBILENET_PATH      = os.path.join(MODELS_DIR, "mobilenetv2_best.keras")
EFFICIENTNET_PATH   = os.path.join(MODELS_DIR, "efficientnetb0_best.keras")
BEST_MODEL_PATH     = os.path.join(MODELS_DIR, "best_brain_tumor_model.keras")
CLASS_NAMES_JSON    = os.path.join(MODELS_DIR, "class_names.json")

# ---------------------------------------------------------------------------
# Augmentation parameters (conservative — medical imaging)
# ---------------------------------------------------------------------------
AUG_ROTATION_RANGE     = 10    # degrees — small rotation only
AUG_WIDTH_SHIFT        = 0.05  # 5% horizontal shift
AUG_HEIGHT_SHIFT       = 0.05  # 5% vertical shift
AUG_ZOOM_RANGE         = 0.10  # 10% zoom
AUG_BRIGHTNESS_RANGE   = [0.9, 1.1]   # subtle brightness variation
# Note: horizontal/vertical flips are NOT used because MRI orientation carries
# anatomical meaning and flipping could produce unrealistic images.
AUG_HORIZONTAL_FLIP    = False
AUG_VERTICAL_FLIP      = False
AUG_FILL_MODE          = "nearest"

# ---------------------------------------------------------------------------
# Fine-tuning: how many top layers of the base to unfreeze
# ---------------------------------------------------------------------------
MOBILENET_UNFREEZE_LAYERS    = 30   # unfreeze last 30 layers of MobileNetV2
EFFICIENTNET_UNFREEZE_LAYERS = 20   # unfreeze last 20 layers of EfficientNetB0

# ---------------------------------------------------------------------------
# Ensure output directories exist when this module is imported
# ---------------------------------------------------------------------------
for _dir in [MODELS_DIR, RESULTS_DIR, REPORTS_DIR, MISC_DIR]:
    os.makedirs(_dir, exist_ok=True)
