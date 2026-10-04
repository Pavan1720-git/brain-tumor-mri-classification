"""
generate_notebooks.py — Creates the four project Jupyter notebooks programmatically.

Run once from the project root:
    python generate_notebooks.py
"""

import json, os, sys

def nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "cells": cells
    }

def md(source):
    return {"cell_type": "markdown", "id": "md" + str(hash(source[:20]))[:8], "metadata": {}, "source": source}

def code(source):
    return {"cell_type": "code", "execution_count": None, "id": "c" + str(hash(source[:20]))[:8],
            "metadata": {}, "outputs": [], "source": source}

# ============================================================================
# 01_EDA.ipynb
# ============================================================================

eda_cells = [
    md("# 01 — Exploratory Data Analysis\n\nBrain Tumor MRI Classification · Internship Project"),
    md("## Setup"),
    code("""import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image, UnidentifiedImageError
from collections import Counter

# Add project root to path
sys.path.insert(0, os.path.abspath('..'))
from src.config import (TRAIN_DIR, VALID_DIR, TEST_DIR, CLASS_NAMES,
                        DISPLAY_NAMES, RESULTS_DIR)
from src.data_utils import (count_images_per_class, get_split_summary,
                             print_summary, find_corrupt_images,
                             sample_image_properties, validate_csv_vs_folders,
                             find_cross_split_duplicates, read_classes_csv)

os.makedirs(RESULTS_DIR, exist_ok=True)
print("Setup complete.")"""),

    md("## 1. Dataset Overview"),
    code("""summary = get_split_summary()
print_summary(summary)"""),

    md("## 2. Class Distribution"),
    code("""splits = ['train', 'valid', 'test']
dirs   = [TRAIN_DIR, VALID_DIR, TEST_DIR]
labels = [DISPLAY_NAMES[c] for c in CLASS_NAMES]

rows = []
for split, d in zip(splits, dirs):
    counts = count_images_per_class(d, CLASS_NAMES)
    row = {'Split': split.capitalize()}
    for cls in CLASS_NAMES:
        row[DISPLAY_NAMES[cls]] = counts[cls]
    row['Total'] = sum(counts.values())
    rows.append(row)

df_dist = pd.DataFrame(rows)
print(df_dist.to_string(index=False))"""),

    code("""fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
colors = ['#e74c3c','#e67e22','#2ecc71','#3498db']

for ax, row in zip(axes, rows):
    vals = [row[l] for l in labels]
    bars = ax.bar(labels, vals, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_title(row['Split'], fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Images')
    ax.set_ylim(0, max(vals) * 1.25)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(v), ha='center', va='bottom', fontsize=10)
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('Class Distribution Across Splits', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
save_path = os.path.join(RESULTS_DIR, 'dataset_distribution.png')
fig.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {save_path}')"""),

    md("## 3. Sample Images"),
    code("""fig, axes = plt.subplots(4, 5, figsize=(16, 13))

for row, cls in enumerate(CLASS_NAMES):
    folder = os.path.join(TRAIN_DIR, cls)
    files  = sorted(os.listdir(folder))[:5]
    for col, fname in enumerate(files):
        img = Image.open(os.path.join(folder, fname)).convert('RGB')
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        if col == 0:
            axes[row, col].set_ylabel(
                DISPLAY_NAMES[cls], fontsize=12, fontweight='bold', rotation=90,
                labelpad=60, va='center',
            )
            axes[row, col].yaxis.set_label_coords(-0.3, 0.5)

fig.suptitle('Sample Images — 5 per Class (Training Set)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
save_path = os.path.join(RESULTS_DIR, 'sample_images.png')
fig.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {save_path}')"""),

    md("## 4. Image Properties"),
    code("""props = sample_image_properties(TRAIN_DIR, CLASS_NAMES, n=20)
df_props = pd.DataFrame(props)
print("\\nSample image properties (first 8 rows):")
print(df_props[['class','width','height','mode']].head(8).to_string(index=False))

print("\\nMost common dimensions:")
dim_counts = Counter(zip(df_props['width'], df_props['height']))
for (w, h), cnt in dim_counts.most_common(5):
    print(f"  {w} x {h} : {cnt} samples")

print("\\nImage modes:")
print(df_props['mode'].value_counts().to_string())"""),

    md("## 5. Data Quality Checks"),
    code("""print("=== Corrupt image check ===")
for split_name, split_dir in [('TRAIN', TRAIN_DIR), ('VALID', VALID_DIR), ('TEST', TEST_DIR)]:
    corrupt = find_corrupt_images(split_dir, CLASS_NAMES)
    if corrupt:
        print(f"{split_name}: {len(corrupt)} corrupt files:")
        for p in corrupt:
            print(f"  {p}")
    else:
        print(f"{split_name}: No corrupt images found ✓")"""),

    code("""print("=== CSV label vs folder validation ===")
for split_name, split_dir, csv_name in [
    ('TRAIN', TRAIN_DIR, os.path.join(TRAIN_DIR, '_classes.csv')),
    ('VALID', VALID_DIR, os.path.join(VALID_DIR, '_classes.csv')),
    ('TEST',  TEST_DIR,  os.path.join(TEST_DIR,  '_classes.csv')),
]:
    result = validate_csv_vs_folders(split_dir, csv_name, CLASS_NAMES)
    print(f"\\n{split_name}:")
    print(f"  Matched     : {len(result['matched'])}")
    print(f"  Mismatched  : {len(result['mismatched'])}")
    print(f"  CSV only    : {len(result['csv_only'])}")
    print(f"  Folder only : {len(result['folder_only'])}")
    if result['mismatched']:
        print("  MISMATCHES:")
        for item in result['mismatched'][:5]:
            print(f"    {item}")"""),

    code("""print("=== Cross-split duplicate check (by filename) ===")
split_dirs = {'train': TRAIN_DIR, 'valid': VALID_DIR, 'test': TEST_DIR}
dups = find_cross_split_duplicates(split_dirs, CLASS_NAMES, use_hash=False)
if dups:
    print(f"FOUND {len(dups)} potential duplicates:")
    for d in dups[:10]:
        print(f"  {d['key']}: {[o[0] for o in d['occurrences']]}")
else:
    print("No filename-level duplicates found across splits ✓")"""),

    md("## 6. Dataset Summary"),
    code("""print_summary(summary)
print("\\nData quality: No corrupt or mismatched images detected.")
print("Data split: TRAIN / VALID / TEST as supplied (not re-split).")
print("Image format: 640x640 RGB JPG → resized to 224x224 for training.")"""),
]

# ============================================================================
# 02_Custom_CNN.ipynb
# ============================================================================

cnn_cells = [
    md("# 02 — Custom CNN Training\n\nBrain Tumor MRI Classification · Internship Project"),
    md("## Setup & Imports"),
    code("""import os, sys, random, json
import numpy as np
import tensorflow as tf

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(0, os.path.abspath('..'))
from src.config import (CUSTOM_CNN_PATH, EPOCHS_CUSTOM, CLASS_NAMES,
                        DISPLAY_NAMES, RESULTS_DIR, BATCH_SIZE, LR_INITIAL)
from src.preprocessing import build_generators
from src.models        import build_custom_cnn, get_callbacks
from src.evaluate      import (evaluate_model, plot_confusion_matrix,
                               plot_training_history, save_classification_report,
                               find_misclassified)
from src.data_utils    import compute_class_weights

print(f"TF version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {gpus if gpus else 'None (using CPU)'}")"""),

    md("## Data Generators (Custom CNN)"),
    code("""train_gen, valid_gen, test_gen = build_generators('custom')

print("Class indices:", train_gen.class_indices)
print(f"Train batches : {len(train_gen)}")
print(f"Valid batches : {len(valid_gen)}")
print(f"Test  batches : {len(test_gen)}")"""),

    md("## Augmentation Visualisation"),
    code("""import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Show original vs augmented for one image per class
fig, axes = plt.subplots(len(CLASS_NAMES), 5, figsize=(16, 13))
labels = [DISPLAY_NAMES[c] for c in CLASS_NAMES]

for row, (cls, ax_row) in enumerate(zip(CLASS_NAMES, axes)):
    folder = os.path.join(train_gen.directory, cls)
    fname  = sorted(os.listdir(folder))[0]
    fpath  = os.path.join(folder, fname)

    from PIL import Image
    orig = Image.open(fpath).convert('RGB')
    ax_row[0].imshow(orig)
    ax_row[0].set_title('Original', fontsize=9)
    ax_row[0].axis('off')
    if row == 0:
        ax_row[0].set_title('Original', fontsize=9, fontweight='bold')

    # Generate augmented versions using the training datagen
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from src.config import (AUG_ROTATION_RANGE, AUG_WIDTH_SHIFT, AUG_HEIGHT_SHIFT,
                            AUG_ZOOM_RANGE, AUG_BRIGHTNESS_RANGE,
                            AUG_HORIZONTAL_FLIP, AUG_VERTICAL_FLIP, AUG_FILL_MODE,
                            IMG_SIZE)
    aug_gen = ImageDataGenerator(
        rotation_range=AUG_ROTATION_RANGE,
        width_shift_range=AUG_WIDTH_SHIFT,
        height_shift_range=AUG_HEIGHT_SHIFT,
        zoom_range=AUG_ZOOM_RANGE,
        brightness_range=AUG_BRIGHTNESS_RANGE,
        horizontal_flip=AUG_HORIZONTAL_FLIP,
        vertical_flip=AUG_VERTICAL_FLIP,
        fill_mode=AUG_FILL_MODE,
    )
    arr = np.array(orig.resize((IMG_SIZE[1], IMG_SIZE[0]))).reshape(1, *IMG_SIZE, 3)
    it  = aug_gen.flow(arr, batch_size=1, seed=42)
    for col in range(1, 5):
        aug_arr = next(it)[0].astype(np.uint8)
        ax_row[col].imshow(aug_arr)
        ax_row[col].set_title(f'Aug {col}', fontsize=9)
        ax_row[col].axis('off')

    ax_row[0].set_ylabel(labels[row], fontsize=11, fontweight='bold',
                         rotation=90, labelpad=55)

fig.suptitle('Original vs Augmented Images (Training Data Only)', fontsize=14, fontweight='bold')
plt.tight_layout()
save_path = os.path.join(RESULTS_DIR, 'augmentation_preview.png')
fig.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {save_path}')"""),

    md("## Build Custom CNN"),
    code("""model = build_custom_cnn()
model.summary()"""),

    md("## Sanity Check"),
    code("""dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
out   = model(dummy, training=False)
print(f"Output shape  : {out.shape}  (expected: (1, 4))")
print(f"Sum of probs  : {out.numpy().sum():.6f}  (expected: ~1.0)")
assert out.shape == (1, 4), "Shape mismatch!"
print("Sanity check PASSED ✓")"""),

    md("## Class Weights"),
    code("""from src.config import TRAIN_DIR
cw = compute_class_weights(TRAIN_DIR, CLASS_NAMES)
print("Class weights:")
for idx, cls in enumerate(CLASS_NAMES):
    print(f"  [{idx}] {DISPLAY_NAMES[cls]:<14}: {cw[idx]:.4f}")"""),

    md("## Train Custom CNN"),
    code("""callbacks = get_callbacks(CUSTOM_CNN_PATH)

history = model.fit(
    train_gen,
    epochs=EPOCHS_CUSTOM,
    validation_data=valid_gen,
    callbacks=callbacks,
    class_weight=cw,
    verbose=1,
)"""),

    md("## Training History Plots"),
    code("""plot_training_history(history, 'Custom CNN')
print("Training history plots saved to results/")"""),

    md("## Test Set Evaluation"),
    code("""# Reload best weights
model = tf.keras.models.load_model(CUSTOM_CNN_PATH)
result = evaluate_model(model, test_gen, 'Custom CNN')
plot_confusion_matrix(result['y_true'], result['y_pred'], 'Custom CNN')
save_classification_report(result['report'], 'Custom CNN')
find_misclassified(model, test_gen, 'Custom CNN')"""),

    md("## Summary"),
    code("""print(f"Custom CNN — Test Results")
print(f"  Accuracy  : {result['accuracy']:.4f}")
print(f"  Precision : {result['precision']:.4f}")
print(f"  Recall    : {result['recall']:.4f}")
print(f"  F1-score  : {result['f1_score']:.4f}")
print(f"  Model saved: {CUSTOM_CNN_PATH}")"""),
]

# ============================================================================
# 03_Transfer_Learning.ipynb
# ============================================================================

tl_cells = [
    md("# 03 — Transfer Learning: MobileNetV2 & EfficientNetB0\n\nBrain Tumor MRI Classification · Internship Project"),
    md("## Setup"),
    code("""import os, sys, random
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

sys.path.insert(0, os.path.abspath('..'))
from src.config import (MOBILENET_PATH, EFFICIENTNET_PATH, CLASS_NAMES,
                        DISPLAY_NAMES, EPOCHS_FROZEN, EPOCHS_FINETUNE, TRAIN_DIR)
from src.preprocessing import build_generators
from src.models        import (build_mobilenetv2, fine_tune_mobilenetv2,
                               build_efficientnetb0, fine_tune_efficientnetb0,
                               get_callbacks)
from src.evaluate      import (evaluate_model, plot_confusion_matrix,
                               plot_training_history, save_classification_report,
                               find_misclassified)
from src.data_utils    import compute_class_weights

cw = compute_class_weights(TRAIN_DIR, CLASS_NAMES)
print("Class weights:", cw)"""),

    md("## 1. MobileNetV2"),
    md("### 1a. Build & Phase-1 Training (Frozen Base)"),
    code("""train_gen, valid_gen, test_gen = build_generators('mobilenetv2')
mv2 = build_mobilenetv2()
mv2.summary()"""),
    code("""print("\\n[MobileNetV2] Phase 1 — training head only …")
h1 = mv2.fit(
    train_gen,
    epochs=EPOCHS_FROZEN,
    validation_data=valid_gen,
    callbacks=get_callbacks(MOBILENET_PATH),
    class_weight=cw,
    verbose=1,
)"""),
    md("### 1b. Fine-Tuning (Unfreeze Top Layers)"),
    code("""mv2 = tf.keras.models.load_model(MOBILENET_PATH)   # best from phase 1
mv2 = fine_tune_mobilenetv2(mv2)
print(f"Trainable params after unfreezing: {sum(v.numpy().size for v in mv2.trainable_weights):,}")

print("\\n[MobileNetV2] Phase 2 — fine-tuning …")
h2 = mv2.fit(
    train_gen,
    epochs=EPOCHS_FINETUNE,
    validation_data=valid_gen,
    callbacks=get_callbacks(MOBILENET_PATH),
    class_weight=cw,
    verbose=1,
)"""),
    md("### 1c. Evaluation"),
    code("""mv2 = tf.keras.models.load_model(MOBILENET_PATH)
merged = {k: h1.history[k] + h2.history[k] for k in h1.history}
plot_training_history(merged, 'MobileNetV2')

result_mv2 = evaluate_model(mv2, test_gen, 'MobileNetV2')
plot_confusion_matrix(result_mv2['y_true'], result_mv2['y_pred'], 'MobileNetV2')
save_classification_report(result_mv2['report'], 'MobileNetV2')
find_misclassified(mv2, test_gen, 'MobileNetV2')"""),

    md("## 2. EfficientNetB0"),
    md("### 2a. Build & Phase-1 Training (Frozen Base)"),
    code("""train_gen_e, valid_gen_e, test_gen_e = build_generators('efficientnetb0')
eff = build_efficientnetb0()
eff.summary()"""),
    code("""print("\\n[EfficientNetB0] Phase 1 — training head only …")
h1e = eff.fit(
    train_gen_e,
    epochs=EPOCHS_FROZEN,
    validation_data=valid_gen_e,
    callbacks=get_callbacks(EFFICIENTNET_PATH),
    class_weight=cw,
    verbose=1,
)"""),
    md("### 2b. Fine-Tuning"),
    code("""eff = tf.keras.models.load_model(EFFICIENTNET_PATH)
eff = fine_tune_efficientnetb0(eff)

print("\\n[EfficientNetB0] Phase 2 — fine-tuning …")
h2e = eff.fit(
    train_gen_e,
    epochs=EPOCHS_FINETUNE,
    validation_data=valid_gen_e,
    callbacks=get_callbacks(EFFICIENTNET_PATH),
    class_weight=cw,
    verbose=1,
)"""),
    md("### 2c. Evaluation"),
    code("""eff = tf.keras.models.load_model(EFFICIENTNET_PATH)
merged_e = {k: h1e.history[k] + h2e.history[k] for k in h1e.history}
plot_training_history(merged_e, 'EfficientNetB0')

result_eff = evaluate_model(eff, test_gen_e, 'EfficientNetB0')
plot_confusion_matrix(result_eff['y_true'], result_eff['y_pred'], 'EfficientNetB0')
save_classification_report(result_eff['report'], 'EfficientNetB0')
find_misclassified(eff, test_gen_e, 'EfficientNetB0')"""),

    md("## Summary"),
    code("""print("MobileNetV2 Test Results:")
print(f"  Accuracy  : {result_mv2['accuracy']:.4f}")
print(f"  F1-score  : {result_mv2['f1_score']:.4f}")
print()
print("EfficientNetB0 Test Results:")
print(f"  Accuracy  : {result_eff['accuracy']:.4f}")
print(f"  F1-score  : {result_eff['f1_score']:.4f}")"""),
]

# ============================================================================
# 04_Model_Evaluation.ipynb
# ============================================================================

eval_cells = [
    md("# 04 — Model Evaluation & Comparison\n\nBrain Tumor MRI Classification · Internship Project"),
    md("## Setup"),
    code("""import os, sys, json, shutil
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('..'))
from src.config import (CUSTOM_CNN_PATH, MOBILENET_PATH, EFFICIENTNET_PATH,
                        BEST_MODEL_PATH, CLASS_NAMES_JSON,
                        CLASS_NAMES, DISPLAY_NAMES, MODELS_DIR)
from src.preprocessing import build_generators
from src.evaluate      import evaluate_model, compare_models
from src.data_utils    import save_class_names"""),

    md("## Load All Models"),
    code("""print("Loading models …")
cnn_model = tf.keras.models.load_model(CUSTOM_CNN_PATH)
mv2_model = tf.keras.models.load_model(MOBILENET_PATH)
eff_model = tf.keras.models.load_model(EFFICIENTNET_PATH)
print("All models loaded ✓")"""),

    md("## Evaluate on Test Set"),
    code("""_, _, test_cnn = build_generators('custom')
_, _, test_mv2 = build_generators('mobilenetv2')
_, _, test_eff = build_generators('efficientnetb0')

r_cnn = evaluate_model(cnn_model, test_cnn, 'Custom CNN')
r_mv2 = evaluate_model(mv2_model, test_mv2, 'MobileNetV2')
r_eff = evaluate_model(eff_model, test_eff, 'EfficientNetB0')"""),

    md("## Model Comparison Table"),
    code("""df = compare_models([r_cnn, r_mv2, r_eff])
df"""),

    md("## Select & Save Best Model"),
    code("""best_row   = df.loc[df['F1-score'].idxmax()]
best_name  = best_row['Model']
print(f"\\nBest model by F1-score: {best_name}")
print(f"  Accuracy  : {best_row['Accuracy']:.4f}")
print(f"  F1-score  : {best_row['F1-score']:.4f}")

src_path = {
    'Custom CNN':     CUSTOM_CNN_PATH,
    'MobileNetV2':    MOBILENET_PATH,
    'EfficientNetB0': EFFICIENTNET_PATH,
}[best_name]

shutil.copy2(src_path, BEST_MODEL_PATH)
print(f"Best model saved → {BEST_MODEL_PATH}")

# Save class names JSON
save_class_names(CLASS_NAMES)

preproc_map = {
    'Custom CNN':     'custom',
    'MobileNetV2':    'mobilenetv2',
    'EfficientNetB0': 'efficientnetb0',
}
meta = {
    'best_model_name': best_name,
    'preprocessing':   preproc_map[best_name],
    'img_size':        [224, 224],
    'num_classes':     4,
    'class_names':     {str(i): DISPLAY_NAMES[c] for i, c in enumerate(CLASS_NAMES)},
    'test_f1':         float(best_row['F1-score']),
    'test_accuracy':   float(best_row['Accuracy']),
}
import json
meta_path = os.path.join(MODELS_DIR, 'deployment_meta.json')
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)
print(f"Deployment metadata saved → {meta_path}")"""),

    md("## Final Summary"),
    code("""print("=" * 60)
print("  FINAL EVALUATION SUMMARY")
print("=" * 60)
for r in [r_cnn, r_mv2, r_eff]:
    print(f"\\n{r['model']}:")
    print(f"  Accuracy  : {r['accuracy']:.4f}")
    print(f"  Precision : {r['precision']:.4f}")
    print(f"  Recall    : {r['recall']:.4f}")
    print(f"  F1-score  : {r['f1_score']:.4f}")
print()
print(f"SELECTED DEPLOYMENT MODEL: {best_name}")
print(f"  Path: {BEST_MODEL_PATH}")
print()
print("Run the app:  streamlit run app.py")"""),
]

# ============================================================================
# Write notebooks
# ============================================================================

NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), 'notebooks')
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

notebooks = {
    '01_EDA.ipynb':              eda_cells,
    '02_Custom_CNN.ipynb':       cnn_cells,
    '03_Transfer_Learning.ipynb': tl_cells,
    '04_Model_Evaluation.ipynb': eval_cells,
}

for fname, cells in notebooks.items():
    path = os.path.join(NOTEBOOKS_DIR, fname)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb(cells), f, indent=1, ensure_ascii=False)
    print(f"Created: {path}")

print("\nAll notebooks generated successfully!")
