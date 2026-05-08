#!/usr/bin/env python3
"""Export per-category Level 1 retrieval accuracy diagnostics."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS_CSV = (
    REPO_ROOT
    / "models"
    / "embedding_runs"
    / "20260502_214051__enriched-triplet-finetune__model-baai-bge-small-en-v1-5__label-level-1-name__bs-64__sampler-balanced__ep-4__lr-2e-04__seq-256__seed-42"
    / "retrieval_predictions.csv"
)
DEFAULT_OUTPUT_PDF = (
    REPO_ROOT
    / "latex"
    / "rapport_final"
    / "figures"
    / "l1_category_accuracy_balanced_sampling.pdf"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "latex"
    / "rapport_final"
    / "figures"
    / "l1_category_accuracy_balanced_sampling.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a bar plot of top-1 retrieval accuracy by Level 1 category."
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=DEFAULT_PREDICTIONS_CSV,
        help="Path to retrieval_predictions.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PDF,
        help="Output PDF path.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV containing per-category metrics.",
    )
    return parser.parse_args()


def read_category_metrics(predictions_csv: Path) -> list[dict[str, float | int | str]]:
    support: Counter[str] = Counter()
    correct: Counter[str] = Counter()

    with predictions_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"label_key", "predicted_node_key"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(
                f"{predictions_csv} must contain columns: {sorted(required)}"
            )

        for row in reader:
            label = row["label_key"]
            prediction = row["predicted_node_key"]
            support[label] += 1
            correct[label] += int(label == prediction)

    if not support:
        raise ValueError(f"No predictions found in {predictions_csv}")

    metrics: list[dict[str, float | int | str]] = []
    for category in support:
        category_support = support[category]
        category_correct = correct[category]
        metrics.append(
            {
                "category": category,
                "support": category_support,
                "correct": category_correct,
                "accuracy": category_correct / category_support,
            }
        )

    return sorted(
        metrics, key=lambda row: (float(row["accuracy"]), int(row["support"]))
    )


def write_summary_csv(
    metrics: list[dict[str, float | int | str]], output_csv: Path
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["category", "support", "correct", "accuracy"]
        )
        writer.writeheader()
        writer.writerows(metrics)


def export_plot(metrics: list[dict[str, float | int | str]], output_pdf: Path) -> None:
    cache_root = Path(tempfile.gettempdir()) / "criteo-plot-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

    import matplotlib.pyplot as plt

    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    categories = [str(row["category"]) for row in metrics]
    accuracies = [float(row["accuracy"]) for row in metrics]
    supports = [int(row["support"]) for row in metrics]
    macro_accuracy = sum(accuracies) / len(accuracies)
    micro_accuracy = sum(int(row["correct"]) for row in metrics) / sum(supports)

    colors = ["#b4473f" if acc < macro_accuracy else "#315f7d" for acc in accuracies]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig_height = max(5.5, 0.32 * len(categories))
    fig, ax = plt.subplots(figsize=(9.2, fig_height))

    bars = ax.barh(categories, accuracies, color=colors, alpha=0.92)
    ax.axvline(
        macro_accuracy,
        color="#2d2d2d",
        linestyle="--",
        linewidth=1.2,
        label=f"Accuracy moyenne par categorie: {macro_accuracy:.1%}",
    )
    ax.axvline(
        micro_accuracy,
        color="red",
        linestyle=":",
        linewidth=1.3,
        label=f"Accuracy globale: {micro_accuracy:.1%}",
    )

    for bar, accuracy, support in zip(bars, accuracies, supports, strict=True):
        ax.text(
            min(accuracy + 0.015, 0.99),
            bar.get_y() + bar.get_height() / 2,
            f"{accuracy:.0%} (n={support})",
            va="center",
            ha="left" if accuracy < 0.92 else "right",
            fontsize=8,
            color="#222222",
        )

    ax.set_title("Accuracy top-1 par categorie de Niveau 1", fontsize=13, pad=12)
    ax.set_xlabel("Accuracy top-1")
    ax.set_xlim(0, 1.04)
    ax.legend(loc="lower right", frameon=True)
    ax.margins(y=0.01)

    fig.tight_layout()
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    metrics = read_category_metrics(args.predictions_csv)
    write_summary_csv(metrics, args.summary_csv)
    export_plot(metrics, args.output)
    print(f"Exported category accuracy plot to {args.output}")
    print(f"Exported category accuracy summary to {args.summary_csv}")


if __name__ == "__main__":
    main()
