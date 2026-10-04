"""
data_utils.py — Dataset utilities for Brain Tumor MRI Classification.

Provides:
  - Dataset inspection helpers
  - Class-count tables
  - Class-weight computation
  - Cross-split duplicate detection
  - Summary printing
"""

import os
import sys
import json
import csv
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.config import (
    TRAIN_DIR, VALID_DIR, TEST_DIR,
    CLASS_NAMES, DISPLAY_NAMES,
    CLASS_NAMES_JSON, MODELS_DIR,
)


# ---------------------------------------------------------------------------
# 1. Image counting helpers
# ---------------------------------------------------------------------------

def count_images_per_class(split_dir: str, classes: list[str]) -> dict[str, int]:
    """Return a {class_name: image_count} dict for a given split directory."""
    counts = {}
    for cls in classes:
        folder = os.path.join(split_dir, cls)
        if os.path.isdir(folder):
            counts[cls] = len([
                f for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
        else:
            counts[cls] = 0
    return counts


def get_split_summary() -> dict:
    """Return a complete summary dict for train / valid / test splits."""
    summary = {}
    for name, path in [("train", TRAIN_DIR), ("valid", VALID_DIR), ("test", TEST_DIR)]:
        counts = count_images_per_class(path, CLASS_NAMES)
        summary[name] = {
            "counts": counts,
            "total":  sum(counts.values()),
            "path":   path,
        }
    summary["grand_total"] = sum(s["total"] for s in summary.values() if isinstance(s, dict) and "total" in s)
    return summary


def print_summary(summary: dict) -> None:
    """Print a formatted dataset summary to stdout."""
    print("\n" + "=" * 60)
    print("  BRAIN TUMOR MRI - DATASET SUMMARY")
    print("=" * 60)
    for split in ["train", "valid", "test"]:
        s = summary[split]
        print(f"\n{split.upper()}  ({s['total']} images total)")
        for cls in CLASS_NAMES:
            disp = DISPLAY_NAMES.get(cls, cls)
            pct  = 100 * s["counts"][cls] / s["total"] if s["total"] else 0
            print(f"  {disp:<14} {s['counts'][cls]:>5}  ({pct:.1f}%)")
    print(f"\n  GRAND TOTAL : {summary['grand_total']} images")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# 2. Image quality inspection
# ---------------------------------------------------------------------------

def find_corrupt_images(split_dir: str, classes: list[str]) -> list[str]:
    """
    Attempt to open every image; return a list of paths that fail.
    Uses PIL.Image.verify() which is faster than full decode.
    """
    corrupt = []
    for cls in classes:
        folder = os.path.join(split_dir, cls)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = os.path.join(folder, fname)
            try:
                with Image.open(fpath) as img:
                    img.verify()          # verify header / trailer only
            except (UnidentifiedImageError, Exception):
                corrupt.append(fpath)
    return corrupt


def sample_image_properties(split_dir: str, classes: list[str], n: int = 5) -> list[dict]:
    """
    Sample up to `n` images per class and return their properties.
    Returns a list of dicts with keys: path, class, width, height, mode.
    """
    records = []
    for cls in classes:
        folder = os.path.join(split_dir, cls)
        if not os.path.isdir(folder):
            continue
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for fname in files[:n]:
            fpath = os.path.join(folder, fname)
            try:
                with Image.open(fpath) as img:
                    records.append({
                        "path":   fpath,
                        "class":  cls,
                        "width":  img.width,
                        "height": img.height,
                        "mode":   img.mode,
                    })
            except Exception:
                pass
    return records


# ---------------------------------------------------------------------------
# 3. CSV label validation
# ---------------------------------------------------------------------------

def read_classes_csv(csv_path: str) -> dict[str, str]:
    """
    Parse `_classes.csv` (one-hot format) and return {filename: class_name}.
    CSV header example: filename, Glioma, Meningioma, No Tumor, Pituitary
    """
    # Map CSV column headers → folder names
    header_to_folder = {
        "Glioma":     "glioma",
        "Meningioma": "meningioma",
        "No Tumor":   "no_tumor",
        "Pituitary":  "pituitary",
    }
    mapping = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"].strip()
            for col, folder in header_to_folder.items():
                if row.get(col, "0").strip() == "1":
                    mapping[fname] = folder
                    break
    return mapping


def validate_csv_vs_folders(split_dir: str, csv_path: str, classes: list[str]) -> dict:
    """
    Cross-check CSV labels against folder placement.
    Returns a dict with keys:
      - matched   : list of (filename, class) that agree
      - mismatched: list of (filename, csv_class, folder_class)
      - csv_only  : files in CSV but not found in any folder
      - folder_only: files in folders but not in CSV
    """
    csv_labels = read_classes_csv(csv_path)

    # Build folder → set-of-filenames index
    folder_index: dict[str, set] = defaultdict(set)
    for cls in classes:
        folder = os.path.join(split_dir, cls)
        if os.path.isdir(folder):
            for f in os.listdir(folder):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    folder_index[cls].add(f)

    all_folder_files = {f: cls for cls, files in folder_index.items() for f in files}

    matched, mismatched, csv_only = [], [], []
    for fname, csv_cls in csv_labels.items():
        folder_cls = all_folder_files.get(fname)
        if folder_cls is None:
            csv_only.append(fname)
        elif folder_cls == csv_cls:
            matched.append((fname, csv_cls))
        else:
            mismatched.append((fname, csv_cls, folder_cls))

    folder_only = [f for f in all_folder_files if f not in csv_labels]

    return {
        "matched":     matched,
        "mismatched":  mismatched,
        "csv_only":    csv_only,
        "folder_only": folder_only,
    }


# ---------------------------------------------------------------------------
# 4. Cross-split duplicate detection
# ---------------------------------------------------------------------------

def _file_hash(path: str, block: int = 65536) -> str:
    """Return MD5 hash of file contents."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def find_cross_split_duplicates(
    split_dirs: dict[str, str],
    classes: list[str],
    use_hash: bool = False,
) -> list[dict]:
    """
    Detect images that appear in more than one split.
    By default compares filenames only (fast).
    If use_hash=True, compares MD5 checksums (slow but definitive).
    Returns a list of collision records.
    """
    # {identifier: [(split, full_path), ...]}
    index: dict[str, list] = defaultdict(list)

    for split_name, split_dir in split_dirs.items():
        for cls in classes:
            folder = os.path.join(split_dir, cls)
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                fpath = os.path.join(folder, fname)
                key = _file_hash(fpath) if use_hash else fname
                index[key].append((split_name, fpath))

    duplicates = [
        {"key": k, "occurrences": v}
        for k, v in index.items()
        if len(v) > 1
    ]
    return duplicates


# ---------------------------------------------------------------------------
# 5. Class weight computation
# ---------------------------------------------------------------------------

def compute_class_weights(train_dir: str, classes: list[str]) -> dict[int, float]:
    """
    Compute balanced class weights from training data ONLY.
    Uses sklearn's 'balanced' strategy:
        weight_i = total / (n_classes * count_i)
    Returns {class_index: weight} dict compatible with Keras.
    """
    counts = count_images_per_class(train_dir, classes)
    total  = sum(counts.values())
    n_cls  = len(classes)
    weights = {}
    for idx, cls in enumerate(classes):
        c = counts[cls]
        weights[idx] = total / (n_cls * c) if c > 0 else 1.0
    return weights


# ---------------------------------------------------------------------------
# 6. Save / load class names JSON
# ---------------------------------------------------------------------------

def save_class_names(classes: list[str], path: str = CLASS_NAMES_JSON) -> None:
    """Save {index: display_name} JSON for use by the Streamlit app."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mapping = {str(i): DISPLAY_NAMES.get(c, c) for i, c in enumerate(classes)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"Class names saved -> {path}")


def load_class_names(path: str = CLASS_NAMES_JSON) -> dict[str, str]:
    """Load {index: display_name} JSON. Returns dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
