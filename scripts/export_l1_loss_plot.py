#!/usr/bin/env python3
"""Export the Level 1 fine-tuning loss curve as a PDF figure."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY_CSV = (
    REPO_ROOT
    / "models"
    / "embedding_runs"
    / "20260503_091523__enriched-triplet-finetune__model-baai-bge-small-en-v1-5__label-level-1-name__bs-128__sampler-tempered__ep-4__lr-2e-04__seq-256__seed-42"
    / "train_step_history.csv"
)
DEFAULT_OUTPUT_PDF = (
    REPO_ROOT / "latex" / "rapport_final" / "figures" / "l1_triplet_alpha05.pdf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a Matplotlib PDF plot of the triplet fine-tuning loss."
    )
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=DEFAULT_HISTORY_CSV,
        help="Path to train_step_history.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PDF,
        help="Output PDF path.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Moving average window, in training steps.",
    )
    return parser.parse_args()


def read_history(history_csv: Path) -> tuple[list[int], list[float], list[int]]:
    steps: list[int] = []
    losses: list[float] = []
    epochs: list[int] = []

    with history_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("split") != "train":
                continue
            steps.append(int(row["global_step"]))
            losses.append(float(row["loss"]))
            epochs.append(int(row["epoch"]))

    if not steps:
        raise ValueError(f"No training rows found in {history_csv}")

    return steps, losses, epochs


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values

    averaged: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        averaged.append(mean(values[start : index + 1]))
    return averaged


def export_plot(
    steps: list[int],
    losses: list[float],
    epochs: list[int],
    output_pdf: Path,
    window: int,
) -> None:
    cache_root = Path(tempfile.gettempdir()) / "criteo-plot-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

    import matplotlib.pyplot as plt

    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    smoothed_losses = moving_average(losses, window)
    first_step_by_epoch: dict[int, int] = {}
    for step, epoch in zip(steps, epochs, strict=True):
        first_step_by_epoch.setdefault(epoch, step)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    ax.plot(
        steps, losses, color="#8fb3ff", linewidth=0.55, alpha=0.32, label="Loss brute"
    )
    ax.plot(
        steps,
        smoothed_losses,
        color="#183a6b",
        linewidth=2.0,
        label=f"Moyenne glissante ({window} steps)",
    )

    for epoch, first_step in sorted(first_step_by_epoch.items()):
        if epoch == min(first_step_by_epoch):
            continue
        ax.axvline(
            first_step, color="#9a9a9a", linewidth=0.8, linestyle="--", alpha=0.55
        )
        ax.text(
            first_step,
            max(losses),
            f"Epoch {epoch}",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color="#666666",
        )

    ax.set_title("Evolution de la loss pendant le fine-tuning L1", fontsize=13, pad=12)
    ax.set_xlabel("Etape globale")
    ax.set_ylabel("Batch Hard Triplet Loss")
    ax.legend(frameon=True, loc="best")
    ax.margins(x=0)

    fig.tight_layout()
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    steps, losses, epochs = read_history(args.history_csv)
    export_plot(steps, losses, epochs, args.output, args.window)
    print(f"Exported loss plot to {args.output}")


if __name__ == "__main__":
    main()
