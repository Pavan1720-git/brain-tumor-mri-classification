"""
evaluate.py - Evaluation utilities for Brain Tumor MRI Classification.

Provides:
  - evaluate_model()         : compute metrics on test generator
  - plot_confusion_matrix()  : save confusion matrix figure
  - plot_training_history()  : save accuracy/loss training curves
  - save_classification_report()
  - compare_models()         : build comparison table and bar chart
  - find_misclassified()     : locate and save misclassified sample images
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

import tensorflow as tf
from PIL import Image

from src.config import (
    CLASS_NAMES, DISPLAY_NAMES,
    RESULTS_DIR, REPORTS_DIR, MISC_DIR,
    NUM_CLASSES,
)

LABELS = [DISPLAY_NAMES.get(c, c) for c in CLASS_NAMES]


def evaluate_model(
    model: tf.keras.Model,
    test_gen,
    model_name: str,
) -> dict:
    """Run model on test_gen and return a metrics dict."""
    test_gen.reset()

    y_prob = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_gen.classes

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(
        y_true, y_pred,
        target_names=LABELS,
        digits=4,
        zero_division=0,
    )

    print(f"\n{'='*60}")
    print(f"  {model_name} - TEST SET EVALUATION")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"\n{report}")

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "report": report,
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    labels: list[str] = LABELS,
    save_path: str = None,
) -> None:
    """Plot and save a seaborn heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path is None:
        fname = f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        save_path = os.path.join(RESULTS_DIR, fname)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved -> {save_path}")


def plot_training_history(
    history,
    model_name: str,
    metrics: list[str] = None,
) -> None:
    """Plot training / validation accuracy and loss curves and save to results/."""
    if metrics is None:
        metrics = ["accuracy", "loss"]

    hist = history.history if hasattr(history, "history") else history

    for metric in metrics:
        val_metric = f"val_{metric}"
        if metric not in hist:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        epochs = range(1, len(hist[metric]) + 1)
        ax.plot(epochs, hist[metric], label=f"Train {metric.capitalize()}", linewidth=2)
        if val_metric in hist:
            ax.plot(epochs, hist[val_metric], label=f"Val {metric.capitalize()}",
                    linestyle="--", linewidth=2)
        ax.set_title(f"{model_name} - {metric.capitalize()} over Epochs",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(metric.capitalize(), fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        fname = f"{model_name.lower().replace(' ', '_')}_{metric}.png"
        save_path = os.path.join(RESULTS_DIR, fname)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Training {metric} plot saved -> {save_path}")


def save_classification_report(report: str, model_name: str) -> None:
    """Save sklearn classification report text to results/classification_reports/."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    fname = f"{model_name.lower().replace(' ', '_')}_classification_report.txt"
    path = os.path.join(REPORTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write("=" * 60 + "\n")
        f.write(report)
    print(f"Classification report saved -> {path}")


def compare_models(
    results: list[dict],
    save_csv: bool = True,
    save_plot: bool = True,
) -> pd.DataFrame:
    """Build a comparison DataFrame from a list of evaluate_model() result dicts."""
    rows = []
    for r in results:
        rows.append({
            "Model": r["model"],
            "Accuracy": round(r["accuracy"], 4),
            "Precision": round(r["precision"], 4),
            "Recall": round(r["recall"], 4),
            "F1-score": round(r["f1_score"], 4),
        })
    df = pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print("  MODEL COMPARISON TABLE")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60 + "\n")

    if save_csv:
        csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
        df.to_csv(csv_path, index=False)
        print(f"Model comparison CSV saved -> {csv_path}")

    if save_plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(df))
        width = 0.2
        metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
        colors = ["steelblue", "coral", "mediumseagreen", "mediumpurple"]

        for i, (met, col) in enumerate(zip(metrics, colors)):
            bars = ax.bar(x + i * width, df[met], width, label=met, color=col, alpha=0.85)
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(df["Model"], fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title("Model Comparison - Test Set Metrics", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(RESULTS_DIR, "model_comparison.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Model comparison chart saved -> {plot_path}")

    return df


def find_misclassified(
    model: tf.keras.Model,
    test_gen,
    model_name: str,
    n_samples: int = 12,
) -> None:
    """Find misclassified images and save a grid visualisation."""
    test_gen.reset()
    y_prob = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_gen.classes

    file_paths = [
        os.path.join(test_gen.directory, fname)
        for fname in test_gen.filenames
    ]

    wrong_idx = np.where(y_true != y_pred)[0]
    print(f"\n{model_name}: {len(wrong_idx)} / {len(y_true)} misclassified on test set.")

    if len(wrong_idx) == 0:
        print("No misclassified samples - perfect test accuracy!")
        return

    os.makedirs(MISC_DIR, exist_ok=True)
    csv_rows = []
    for idx in wrong_idx:
        csv_rows.append({
            "file": file_paths[idx],
            "true_label": LABELS[y_true[idx]],
            "pred_label": LABELS[y_pred[idx]],
            "confidence": round(float(y_prob[idx][y_pred[idx]]), 4),
        })
    pd.DataFrame(csv_rows).to_csv(
        os.path.join(MISC_DIR, f"{model_name.lower().replace(' ', '_')}_misclassified.csv"),
        index=False,
    )

    sample_idx = wrong_idx[:n_samples]
    ncols = 4
    nrows = int(np.ceil(len(sample_idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()

    for i, idx in enumerate(sample_idx):
        img = Image.open(file_paths[idx]).convert("RGB")
        axes[i].imshow(img)
        axes[i].set_title(
            f"True: {LABELS[y_true[idx]]}\nPred: {LABELS[y_pred[idx]]}"
            f"\n({y_prob[idx][y_pred[idx]]*100:.1f}%)",
            fontsize=8, color="red",
        )
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Misclassified Samples - {model_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(MISC_DIR, f"{model_name.lower().replace(' ', '_')}_misclassified.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Misclassified grid saved -> {out_path}")
