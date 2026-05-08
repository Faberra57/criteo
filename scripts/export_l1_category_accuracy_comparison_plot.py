#!/usr/bin/env python3
"""Export grouped per-category Level 1 accuracy comparison across fine-tuning runs."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = REPO_ROOT / "models" / "embedding_runs"
FIGURE_ROOT = REPO_ROOT / "latex" / "rapport_final" / "figures"

DEFAULT_RUNS: list[tuple[str, Path]] = [
    (
        "Sans sampling",
        RUN_ROOT
        / "20260502_120635__enriched-triplet-finetune__model-baai-bge-small-en-v1-5__label-level-1-name__bs-64__ep-3__lr-2e-04__seq-256__seed-42",
    ),
    (
        "Balanced",
        RUN_ROOT
        / "20260502_214051__enriched-triplet-finetune__model-baai-bge-small-en-v1-5__label-level-1-name__bs-64__sampler-balanced__ep-4__lr-2e-04__seq-256__seed-42",
    ),
    (
        "Tempered alpha=0.5",
        RUN_ROOT
        / "20260503_091523__enriched-triplet-finetune__model-baai-bge-small-en-v1-5__label-level-1-name__bs-128__sampler-tempered__ep-4__lr-2e-04__seq-256__seed-42",
    ),
    (
        "Tempered alpha=0.3",
        RUN_ROOT
        / "20260503_092243__enriched-triplet-finetune__model-baai-bge-small-en-v1-5__label-level-1-name__bs-128__sampler-tempered__ep-4__lr-2e-04__seq-256__seed-42",
    ),
    (
        "Tempered alpha=0.8",
        RUN_ROOT
        / "20260503_131422__enriched-triplet-finetune__model-baai-bge-small-en-v1-5__label-level-1-name__bs-128__sampler-tempered__ep-4__lr-2e-04__seq-256__seed-42",
    ),
]

DEFAULT_OUTPUT_PDF = FIGURE_ROOT / "l1_category_accuracy_sampling_comparison.pdf"
DEFAULT_OUTPUT_CSV = FIGURE_ROOT / "l1_category_accuracy_sampling_comparison.csv"
DEFAULT_METRICS_CSV = FIGURE_ROOT / "l1_sampling_strategy_metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a grouped bar plot comparing Level 1 category accuracy "
            "across sampling strategies."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "RUN_DIR"),
        help=(
            "Run label and run directory. Can be repeated. Defaults to the four "
            "known L1 runs."
        ),
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
        help="Wide CSV with per-category accuracies for all runs.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=DEFAULT_METRICS_CSV,
        help="CSV with macro/micro accuracy per run.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["baseline", "support", "macro_gain", "alpha03"],
        default="baseline",
        help="Category ordering in the plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = parse_runs(args.run)
    per_run = {
        label: read_category_metrics(run_dir / "retrieval_predictions.csv")
        for label, run_dir in runs
    }
    rows = build_comparison_rows(per_run, sort_by=args.sort_by)
    metrics = build_strategy_metrics(per_run)
    write_csv(rows, args.summary_csv)
    write_csv(metrics, args.metrics_csv)
    export_plot(rows, metrics, [label for label, _ in runs], args.output)
    print(f"Exported comparison plot to {args.output}")
    print(f"Exported per-category comparison to {args.summary_csv}")
    print(f"Exported strategy metrics to {args.metrics_csv}")


def parse_runs(raw_runs: list[list[str]] | None) -> list[tuple[str, Path]]:
    if not raw_runs:
        runs = DEFAULT_RUNS
    else:
        runs = [(label, Path(path)) for label, path in raw_runs]
    missing = [
        str(run_dir / "retrieval_predictions.csv")
        for _, run_dir in runs
        if not (run_dir / "retrieval_predictions.csv").exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing retrieval_predictions.csv for these runs:\n" + "\n".join(missing)
        )
    return runs


def read_category_metrics(
    predictions_csv: Path,
) -> dict[str, dict[str, float | int | str]]:
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
            label = str(row["label_key"]).strip()
            prediction = str(row["predicted_node_key"]).strip()
            support[label] += 1
            correct[label] += int(label == prediction)

    if not support:
        raise ValueError(f"No predictions found in {predictions_csv}")

    return {
        category: {
            "category": category,
            "support": int(support[category]),
            "correct": int(correct[category]),
            "accuracy": correct[category] / support[category],
        }
        for category in support
    }


def build_comparison_rows(
    per_run: dict[str, dict[str, dict[str, float | int | str]]],
    *,
    sort_by: str,
) -> list[dict[str, float | int | str]]:
    run_labels = list(per_run)
    categories = sorted(
        {category for metrics in per_run.values() for category in metrics}
    )
    baseline_label = run_labels[0]
    alpha03_label = run_labels[-1]

    rows: list[dict[str, float | int | str]] = []
    for category in categories:
        row: dict[str, float | int | str] = {"category": category}
        support_values = []
        for label in run_labels:
            metrics = per_run[label].get(category)
            if metrics is None:
                row[f"{label}_support"] = 0
                row[f"{label}_correct"] = 0
                row[f"{label}_accuracy"] = 0.0
                continue
            row[f"{label}_support"] = int(metrics["support"])
            row[f"{label}_correct"] = int(metrics["correct"])
            row[f"{label}_accuracy"] = float(metrics["accuracy"])
            support_values.append(int(metrics["support"]))
        row["support"] = max(support_values) if support_values else 0
        row["delta_alpha03_vs_baseline"] = float(
            row.get(f"{alpha03_label}_accuracy", 0.0)
        ) - float(row.get(f"{baseline_label}_accuracy", 0.0))
        rows.append(row)

    if sort_by == "support":
        return sorted(rows, key=lambda row: (int(row["support"]), str(row["category"])))
    if sort_by == "macro_gain":
        return sorted(
            rows,
            key=lambda row: (
                float(row["delta_alpha03_vs_baseline"]),
                str(row["category"]),
            ),
        )
    if sort_by == "alpha03":
        return sorted(
            rows,
            key=lambda row: (
                float(row.get(f"{alpha03_label}_accuracy", 0.0)),
                int(row["support"]),
            ),
        )
    return sorted(
        rows,
        key=lambda row: (
            float(row.get(f"{baseline_label}_accuracy", 0.0)),
            int(row["support"]),
        ),
    )


def build_strategy_metrics(
    per_run: dict[str, dict[str, dict[str, float | int | str]]],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for label, metrics in per_run.items():
        supports = [int(row["support"]) for row in metrics.values()]
        corrects = [int(row["correct"]) for row in metrics.values()]
        accuracies = [float(row["accuracy"]) for row in metrics.values()]
        rows.append(
            {
                "model": label,
                "n_categories": len(metrics),
                "support": sum(supports),
                "correct": sum(corrects),
                "micro_accuracy": sum(corrects) / sum(supports),
                "macro_accuracy": sum(accuracies) / len(accuracies),
            }
        )
    return rows


def write_csv(rows: list[dict[str, float | int | str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_plot(
    rows: list[dict[str, float | int | str]],
    metrics: list[dict[str, float | int | str]],
    run_labels: list[str],
    output_pdf: Path,
) -> None:
    cache_root = Path(tempfile.gettempdir()) / "criteo-plot-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

    import matplotlib.pyplot as plt
    import numpy as np

    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    categories = [str(row["category"]) for row in rows]
    y = np.arange(len(categories))
    bar_height = min(0.18, 0.78 / max(1, len(run_labels)))
    offsets = (np.arange(len(run_labels)) - (len(run_labels) - 1) / 2) * (
        bar_height * 1.15
    )
    colors = ["#4b6f8f", "#c65f46", "#6a9f58", "#9a6fb0"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig_height = max(6.0, 0.42 * len(categories))
    fig, ax = plt.subplots(figsize=(11.0, fig_height))

    for index, label in enumerate(run_labels):
        values = [float(row.get(f"{label}_accuracy", 0.0)) for row in rows]
        ax.barh(
            y + offsets[index],
            values,
            height=bar_height,
            label=label,
            color=colors[index % len(colors)],
            alpha=0.9,
        )

    baseline_label = run_labels[0]
    alpha03_label = run_labels[-1]
    for row_idx, row in enumerate(rows):
        support = int(row["support"])
        baseline = float(row.get(f"{baseline_label}_accuracy", 0.0))
        alpha03 = float(row.get(f"{alpha03_label}_accuracy", 0.0))
        delta = alpha03 - baseline
        ax.text(
            1.01,
            row_idx,
            f"n={support}  Δ={delta:+.0%}",
            va="center",
            ha="left",
            fontsize=7.5,
            color="#333333",
        )

    for metric in metrics:
        label = str(metric["model"])
        ax.axvline(
            float(metric["macro_accuracy"]),
            linestyle="--",
            linewidth=0.9,
            alpha=0.45,
            color=colors[run_labels.index(label) % len(colors)],
        )

    metric_text = " | ".join(
        f"{row['model']}: micro {float(row['micro_accuracy']):.1%}, macro {float(row['macro_accuracy']):.1%}"
        for row in metrics
    )
    ax.text(
        0.0,
        1.01,
        metric_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#333333",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Accuracy top-1")
    ax.set_title(
        "Comparaison des accuracies L1 par categorie selon la strategie de sampling",
        fontsize=13,
        pad=18,
    )
    ax.legend(loc="lower right", frameon=True)
    ax.margins(y=0.01)
    fig.tight_layout()
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
