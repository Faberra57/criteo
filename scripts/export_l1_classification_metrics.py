#!/usr/bin/env python3
"""Export Level 1 classification metrics and a normalized confusion matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "models"
    / "embedding_runs"
    / "20260503_091523__enriched-triplet-finetune__model-baai-bge-small-en-v1-5__label-level-1-name__bs-128__sampler-tempered__ep-4__lr-2e-04__seq-256__seed-42"
)
FIGURE_DIR = REPO_ROOT / "latex" / "rapport_final" / "figures"
DEFAULT_METRICS_CSV = FIGURE_DIR / "l1_classification_metrics.csv"
DEFAULT_PER_CLASS_CSV = FIGURE_DIR / "l1_classification_per_class_metrics.csv"
DEFAULT_CONFUSION_PDF = FIGURE_DIR / "l1_confusion_matrix_normalized.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export standard classification metrics for Level 1 retrieval predictions."
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=DEFAULT_RUN_DIR / "retrieval_predictions.csv",
        help="Path to retrieval_predictions.csv.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=DEFAULT_METRICS_CSV,
        help="Output CSV with global metrics.",
    )
    parser.add_argument(
        "--per-class-csv",
        type=Path,
        default=DEFAULT_PER_CLASS_CSV,
        help="Output CSV with per-class precision, recall and F1.",
    )
    parser.add_argument(
        "--confusion-pdf",
        type=Path,
        default=DEFAULT_CONFUSION_PDF,
        help="Output PDF with row-normalized confusion matrix.",
    )
    return parser.parse_args()


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def read_predictions(predictions_csv: Path) -> tuple[list[str], list[str], list[list[str]]]:
    y_true: list[str] = []
    y_pred: list[str] = []
    top_k: list[list[str]] = []

    with predictions_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"label_key", "predicted_node_key", "top_k_predictions_json"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(
                f"{predictions_csv} must contain columns: {sorted(required)}"
            )

        for row in reader:
            label = row["label_key"]
            pred = row["predicted_node_key"]
            y_true.append(label)
            y_pred.append(pred)
            try:
                parsed = json.loads(row["top_k_predictions_json"])
            except json.JSONDecodeError:
                parsed = []
            top_k.append([str(item.get("node_key", "")) for item in parsed])

    if not y_true:
        raise ValueError(f"No predictions found in {predictions_csv}")
    return y_true, y_pred, top_k


def compute_metrics(
    y_true: list[str], y_pred: list[str], top_k: list[list[str]]
) -> tuple[list[dict[str, str | int | float]], dict[str, float | int]]:
    labels = sorted(set(y_true) | set(y_pred))
    support = Counter(y_true)
    predicted = Counter(y_pred)
    tp = Counter(label for label, pred in zip(y_true, y_pred, strict=True) if label == pred)

    per_class: list[dict[str, str | int | float]] = []
    for label in labels:
        label_tp = tp[label]
        label_support = support[label]
        label_predicted = predicted[label]
        precision = safe_div(label_tp, label_predicted)
        recall = safe_div(label_tp, label_support)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_class.append(
            {
                "category": label,
                "support": label_support,
                "predicted": label_predicted,
                "true_positive": label_tp,
                "false_positive": label_predicted - label_tp,
                "false_negative": label_support - label_tp,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    n = len(y_true)
    correct = sum(1 for label, pred in zip(y_true, y_pred, strict=True) if label == pred)
    top3 = sum(label in preds[:3] for label, preds in zip(y_true, top_k, strict=True))
    top5 = sum(label in preds[:5] for label, preds in zip(y_true, top_k, strict=True))

    macro_precision = sum(float(row["precision"]) for row in per_class) / len(per_class)
    macro_recall = sum(float(row["recall"]) for row in per_class) / len(per_class)
    macro_f1 = sum(float(row["f1"]) for row in per_class) / len(per_class)
    weighted_precision = sum(
        float(row["precision"]) * int(row["support"]) for row in per_class
    ) / n
    weighted_recall = sum(float(row["recall"]) * int(row["support"]) for row in per_class) / n
    weighted_f1 = sum(float(row["f1"]) * int(row["support"]) for row in per_class) / n

    accuracy = correct / n
    global_metrics: dict[str, float | int] = {
        "n_samples": n,
        "n_classes": len(labels),
        "top1_accuracy": accuracy,
        "top3_accuracy": top3 / n,
        "top5_accuracy": top5 / n,
        "micro_precision": accuracy,
        "micro_recall": accuracy,
        "micro_f1": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": macro_recall,
    }

    per_class.sort(key=lambda row: (float(row["f1"]), int(row["support"])))
    return per_class, global_metrics


def write_global_metrics(metrics: dict[str, float | int], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({"metric": key, "value": value})


def write_per_class_metrics(
    per_class: list[dict[str, str | int | float]], output_csv: Path
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "category",
        "support",
        "predicted",
        "true_positive",
        "false_positive",
        "false_negative",
        "precision",
        "recall",
        "f1",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_class)


def export_confusion_matrix(y_true: list[str], y_pred: list[str], output_pdf: Path) -> None:
    cache_root = Path(tempfile.gettempdir()) / "criteo-plot-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

    import matplotlib.pyplot as plt
    import numpy as np

    labels = sorted(set(y_true) | set(y_pred), key=lambda label: (-y_true.count(label), label))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=float)
    for label, pred in zip(y_true, y_pred, strict=True):
        matrix[label_to_idx[label], label_to_idx[pred]] += 1.0

    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(12.5, 10.8))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)

    ax.set_title("Matrice de confusion normalisee par categorie vraie", fontsize=14, pad=14)
    ax.set_xlabel("Categorie predite")
    ax.set_ylabel("Categorie vraie")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = normalized[i, j]
            if value >= 0.08 or i == j:
                ax.text(
                    j,
                    i,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if value > 0.5 else "#1f2937",
                )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Part des produits de la categorie vraie")
    fig.tight_layout()
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    y_true, y_pred, top_k = read_predictions(args.predictions_csv)
    per_class, metrics = compute_metrics(y_true, y_pred, top_k)
    write_global_metrics(metrics, args.metrics_csv)
    write_per_class_metrics(per_class, args.per_class_csv)
    export_confusion_matrix(y_true, y_pred, args.confusion_pdf)
    print(f"Exported global metrics to {args.metrics_csv}")
    print(f"Exported per-class metrics to {args.per_class_csv}")
    print(f"Exported confusion matrix to {args.confusion_pdf}")


if __name__ == "__main__":
    main()
