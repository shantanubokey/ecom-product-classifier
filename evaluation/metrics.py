"""
Evaluation metrics — accuracy, precision, recall, F1, confusion matrix.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List


def evaluate(y_true: List[int], y_pred: List[int],
             label_names: List[str] = None) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"{'='*50}")

    if label_names:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def plot_confusion_matrix(y_true, y_pred, label_names: List[str],
                          title: str = "Confusion Matrix", save_path: str = None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.show()


def compare_models(results: dict, save_path: str = None):
    """Bar chart comparing accuracy/F1 across models."""
    models = list(results.keys())
    accs   = [results[m]["accuracy"] for m in models]
    f1s    = [results[m]["f1"] for m in models]

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, accs, 0.4, label="Accuracy", color="#3498db", alpha=0.85)
    ax.bar(x + 0.2, f1s,  0.4, label="F1 Score",  color="#e74c3c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Accuracy & F1")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - 0.2, a + 0.02, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + 0.2, f + 0.02, f"{f:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.show()
