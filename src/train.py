"""
train.py - End-to-end training script for Brain Tumor MRI Classification.

Usage (from project root):
    python -m src.train
"""

import os
import sys
import json
import random
import shutil
import numpy as np

# Force UTF-8 encoding on standard streams to avoid Windows charmap errors
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

# Suppress TF info/warning logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    EPOCHS_CUSTOM, EPOCHS_FROZEN, EPOCHS_FINETUNE,
    CUSTOM_CNN_PATH, MOBILENET_PATH, EFFICIENTNET_PATH,
    BEST_MODEL_PATH, CLASS_NAMES_JSON,
    CLASS_NAMES, DISPLAY_NAMES,
    RESULTS_DIR,
)
from src.preprocessing import build_generators
from src.models import (
    build_custom_cnn,
    build_mobilenetv2, fine_tune_mobilenetv2,
    build_efficientnetb0, fine_tune_efficientnetb0,
    get_callbacks,
)
from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_training_history,
    save_classification_report,
    compare_models,
    find_misclassified,
)
from src.data_utils import compute_class_weights, save_class_names


def print_gpu_info() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            print(f"  GPU found: {g.name}")
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except RuntimeError:
                pass
    else:
        print("  No GPU detected - training on CPU.")
    print(f"  TensorFlow version: {tf.__version__}")


def merge_histories(h1, h2) -> dict:
    """Concatenate two Keras history dicts."""
    merged = {}
    h1_dict = h1.history if hasattr(h1, "history") else h1
    h2_dict = h2.history if hasattr(h2, "history") else h2
    for key in h1_dict:
        merged[key] = list(h1_dict[key]) + list(h2_dict.get(key, []))
    return merged


def main() -> None:
    print("\n" + "=" * 65)
    print("  BRAIN TUMOR MRI - TRAINING PIPELINE")
    print("=" * 65)

    print("\n[INFO] Hardware:")
    print_gpu_info()

    print("\n[INFO] Computing class weights from training split...")
    cw = compute_class_weights(
        train_dir=os.path.join(os.path.dirname(__file__), "..", "data", "train"),
        classes=CLASS_NAMES,
    )
    print(f"  Class weights: {cw}")

    all_results = []

    # ======================================================================
    # 1. CUSTOM CNN
    # ======================================================================
    print("\n\n" + "=" * 65)
    print("  STEP 1 / 3 - CUSTOM CNN")
    print("=" * 65)

    train_gen, valid_gen, test_gen = build_generators("custom")

    cnn_model = build_custom_cnn()
    cnn_model.summary()

    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    out = cnn_model(dummy, training=False)
    assert out.shape == (1, 4), f"Expected (1,4) got {out.shape}"
    print(f"\n[SANITY] Output shape OK: {out.shape}")

    cnn_history = cnn_model.fit(
        train_gen,
        epochs=EPOCHS_CUSTOM,
        validation_data=valid_gen,
        callbacks=get_callbacks(CUSTOM_CNN_PATH),
        class_weight=cw,
        verbose=1,
    )

    # Reload best weights
    cnn_model = tf.keras.models.load_model(CUSTOM_CNN_PATH)

    # Evaluate
    cnn_result = evaluate_model(cnn_model, test_gen, "Custom CNN")
    all_results.append(cnn_result)

    plot_training_history(cnn_history, "Custom CNN")
    plot_confusion_matrix(cnn_result["y_true"], cnn_result["y_pred"], "Custom CNN")
    save_classification_report(cnn_result["report"], "Custom CNN")
    find_misclassified(cnn_model, test_gen, "Custom CNN")

    # ======================================================================
    # 2. MOBILENETV2
    # ======================================================================
    print("\n\n" + "=" * 65)
    print("  STEP 2 / 3 - MOBILENETV2 (frozen -> fine-tune)")
    print("=" * 65)

    train_gen_mv2, valid_gen_mv2, test_gen_mv2 = build_generators("mobilenetv2")

    mv2_model = build_mobilenetv2()

    print("\n[MobileNetV2] Phase 1 - training head only (base frozen)...")
    mv2_h1 = mv2_model.fit(
        train_gen_mv2,
        epochs=EPOCHS_FROZEN,
        validation_data=valid_gen_mv2,
        callbacks=get_callbacks(MOBILENET_PATH),
        class_weight=cw,
        verbose=1,
    )

    print("\n[MobileNetV2] Phase 2 - fine-tuning top layers...")
    mv2_model = tf.keras.models.load_model(MOBILENET_PATH)
    mv2_model = fine_tune_mobilenetv2(mv2_model)

    mv2_h2 = mv2_model.fit(
        train_gen_mv2,
        epochs=EPOCHS_FINETUNE,
        validation_data=valid_gen_mv2,
        callbacks=get_callbacks(MOBILENET_PATH),
        class_weight=cw,
        verbose=1,
    )

    mv2_model = tf.keras.models.load_model(MOBILENET_PATH)

    mv2_result = evaluate_model(mv2_model, test_gen_mv2, "MobileNetV2")
    all_results.append(mv2_result)

    merged_mv2 = merge_histories(mv2_h1, mv2_h2)
    plot_training_history(merged_mv2, "MobileNetV2")
    plot_confusion_matrix(mv2_result["y_true"], mv2_result["y_pred"], "MobileNetV2")
    save_classification_report(mv2_result["report"], "MobileNetV2")
    find_misclassified(mv2_model, test_gen_mv2, "MobileNetV2")

    # ======================================================================
    # 3. EFFICIENTNETB0
    # ======================================================================
    print("\n\n" + "=" * 65)
    print("  STEP 3 / 3 - EFFICIENTNETB0 (frozen -> fine-tune)")
    print("=" * 65)

    train_gen_eff, valid_gen_eff, test_gen_eff = build_generators("efficientnetb0")

    eff_model = build_efficientnetb0()

    print("\n[EfficientNetB0] Phase 1 - training head only (base frozen)...")
    eff_h1 = eff_model.fit(
        train_gen_eff,
        epochs=EPOCHS_FROZEN,
        validation_data=valid_gen_eff,
        callbacks=get_callbacks(EFFICIENTNET_PATH),
        class_weight=cw,
        verbose=1,
    )

    print("\n[EfficientNetB0] Phase 2 - fine-tuning top layers...")
    eff_model = tf.keras.models.load_model(EFFICIENTNET_PATH)
    eff_model = fine_tune_efficientnetb0(eff_model)

    eff_h2 = eff_model.fit(
        train_gen_eff,
        epochs=EPOCHS_FINETUNE,
        validation_data=valid_gen_eff,
        callbacks=get_callbacks(EFFICIENTNET_PATH),
        class_weight=cw,
        verbose=1,
    )

    eff_model = tf.keras.models.load_model(EFFICIENTNET_PATH)

    eff_result = evaluate_model(eff_model, test_gen_eff, "EfficientNetB0")
    all_results.append(eff_result)

    merged_eff = merge_histories(eff_h1, eff_h2)
    plot_training_history(merged_eff, "EfficientNetB0")
    plot_confusion_matrix(eff_result["y_true"], eff_result["y_pred"], "EfficientNetB0")
    save_classification_report(eff_result["report"], "EfficientNetB0")
    find_misclassified(eff_model, test_gen_eff, "EfficientNetB0")

    # ======================================================================
    # 4. MODEL COMPARISON & SELECTION
    # ======================================================================
    print("\n\n" + "=" * 65)
    print("  MODEL COMPARISON & DEPLOYMENT MODEL SELECTION")
    print("=" * 65)

    df = compare_models(all_results)

    best_row = df.loc[df["F1-score"].idxmax()]
    best_name = best_row["Model"]
    print(f"\n[SELECTION] Best model by F1-score: {best_name}")

    model_map = {
        "Custom CNN": (cnn_model, "custom"),
        "MobileNetV2": (mv2_model, "mobilenetv2"),
        "EfficientNetB0": (eff_model, "efficientnetb0"),
    }
    best_model, best_preproc = model_map[best_name]

    src_path = {
        "Custom CNN": CUSTOM_CNN_PATH,
        "MobileNetV2": MOBILENET_PATH,
        "EfficientNetB0": EFFICIENTNET_PATH,
    }[best_name]
    shutil.copy2(src_path, BEST_MODEL_PATH)
    print(f"[SELECTION] Best model saved -> {BEST_MODEL_PATH}")

    save_class_names(CLASS_NAMES)

    meta = {
        "best_model_name": best_name,
        "preprocessing": best_preproc,
        "img_size": [224, 224],
        "num_classes": 4,
        "class_names": {str(i): DISPLAY_NAMES[c] for i, c in enumerate(CLASS_NAMES)},
        "test_f1": float(best_row["F1-score"]),
        "test_accuracy": float(best_row["Accuracy"]),
    }
    meta_path = os.path.join(
        os.path.dirname(BEST_MODEL_PATH), "deployment_meta.json"
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[SELECTION] Deployment metadata saved -> {meta_path}")

    print("\n" + "=" * 65)
    print("  TRAINING PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  Best model : {best_name}")
    print(f"  F1-score   : {best_row['F1-score']:.4f}")
    print(f"  Accuracy   : {best_row['Accuracy']:.4f}")
    print(f"\n  Run the app: streamlit run app.py")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
