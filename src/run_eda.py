"""
run_eda.py - Generate EDA plots and dataset distribution figures.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import (
    TRAIN_DIR, VALID_DIR, TEST_DIR, CLASS_NAMES, DISPLAY_NAMES, RESULTS_DIR
)
from src.data_utils import (
    count_images_per_class, get_split_summary, print_summary,
    find_corrupt_images, sample_image_properties, validate_csv_vs_folders,
    find_cross_split_duplicates
)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = get_split_summary()
    print_summary(summary)

    # 1. Class Distribution Bar Chart
    splits = ["train", "valid", "test"]
    dirs = [TRAIN_DIR, VALID_DIR, TEST_DIR]
    labels = [DISPLAY_NAMES[c] for c in CLASS_NAMES]
    colors = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db"]

    rows = []
    for split, d in zip(splits, dirs):
        counts = count_images_per_class(d, CLASS_NAMES)
        row = {"Split": split.capitalize()}
        for cls in CLASS_NAMES:
            row[DISPLAY_NAMES[cls]] = counts[cls]
        row["Total"] = sum(counts.values())
        rows.append(row)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, row in zip(axes, rows):
        vals = [row[l] for l in labels]
        bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(f"{row['Split']} Set (N={row['Total']})", fontsize=12, fontweight="bold")
        ax.set_ylabel("Number of Images")
        ax.set_ylim(0, max(vals) * 1.25)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    str(v), ha="center", va="bottom", fontsize=10)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Brain Tumor MRI Dataset - Class Distribution Across Splits", fontsize=14, fontweight="bold")
    plt.tight_layout()
    dist_path = os.path.join(RESULTS_DIR, "dataset_distribution.png")
    fig.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Dataset distribution plot saved -> {dist_path}")

    # 2. Sample Images Grid (5 per class from train set)
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    for row, cls in enumerate(CLASS_NAMES):
        folder = os.path.join(TRAIN_DIR, cls)
        files = sorted(os.listdir(folder))[:5]
        for col, fname in enumerate(files):
            img = Image.open(os.path.join(folder, fname)).convert("RGB")
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(
                    DISPLAY_NAMES[cls], fontsize=12, fontweight="bold",
                    rotation=90, labelpad=60, va="center"
                )
                axes[row, col].yaxis.set_label_coords(-0.25, 0.5)

    fig.suptitle("Sample Brain MRI Scans - 5 Images Per Category (Training Set)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    samples_path = os.path.join(RESULTS_DIR, "sample_images.png")
    fig.savefig(samples_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Sample images grid saved -> {samples_path}")

    # 3. Data Quality Checks
    print("\n--- Running Data Quality Verification ---")
    for name, path in [("TRAIN", TRAIN_DIR), ("VALID", VALID_DIR), ("TEST", TEST_DIR)]:
        corrupt = find_corrupt_images(path, CLASS_NAMES)
        print(f"  {name} corrupt images: {len(corrupt)}")

    print("\nEDA and initial visualizations successfully generated.")

if __name__ == "__main__":
    main()
