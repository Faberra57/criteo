#!/usr/bin/env python3
"""Export annex figures for L2-L7 family-3 agreement matrices and divergence curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_ROOT = ROOT / "latex/rapport_final/figures"
SOURCES = {
    "qwen": FIG_ROOT / "l2_l7_family3_qwen",
    "jasper": FIG_ROOT / "l2_l7_family3_jasper",
}
OUT = FIG_ROOT / "l2_l7_family3_annex"
PIPELINE_LABELS = {
    "beam": "Beam",
    "beam_reranker": "Beam + rerank",
    "clustering_l1_hungarian": "Clustering",
    "global_path": "Global path",
    "greedy": "Greedy",
}
ORDER = ["greedy", "beam", "beam_reranker", "global_path", "clustering_l1_hungarian"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for model, base in SOURCES.items():
        export_level_heatmaps(model, base)
    export_divergence_curve()


def export_level_heatmaps(model: str, base: Path) -> None:
    for level in range(2, 8):
        matrix = read_matrix(base / f"family3_level_{level}_agreement_matrix.csv")
        plot_heatmap(
            matrix,
            title=f"{model.upper()} - accord exact au niveau L{level}",
            output=OUT / f"family3_{model}_level_{level}_agreement_heatmap.pdf",
        )


def read_matrix(path: Path) -> pd.DataFrame:
    matrix = pd.read_csv(path, index_col=0)
    present = [p for p in ORDER if p in matrix.index and p in matrix.columns]
    matrix = matrix.loc[present, present]
    matrix.index = [PIPELINE_LABELS.get(x, x) for x in matrix.index]
    matrix.columns = [PIPELINE_LABELS.get(x, x) for x in matrix.columns]
    return matrix


def plot_heatmap(matrix: pd.DataFrame, *, title: str, output: Path) -> None:
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(values, vmin=0, vmax=1, cmap="YlGnBu")
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]), matrix.index)
    ax.tick_params(axis="both", labelsize=8)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isfinite(value):
                color = "white" if value > 0.58 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=color)
            else:
                ax.text(j, i, "NA", ha="center", va="center", fontsize=7, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Taux d'accord", fontsize=9)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def export_divergence_curve() -> None:
    levels = list(range(2, 8))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, (model, base) in zip(axes, SOURCES.items(), strict=False):
        pairwise = pd.read_csv(base / "family3_pairwise_metrics.csv")
        agreement_cols = [f"level_{level}_agreement" for level in levels]
        pair_labels = [format_pair(row["pipeline_left"], row["pipeline_right"]) for _, row in pairwise.iterrows()]
        divergences = 1.0 - pairwise[agreement_cols].astype(float)
        for idx, row in divergences.iterrows():
            ax.plot(levels, row.values, color="#8a8f98", linewidth=0.9, alpha=0.42)
        mean_divergence = divergences.mean(axis=0, skipna=True)
        ax.plot(levels, mean_divergence.values, color="#0b3d91", marker="o", linewidth=2.8, label="Divergence moyenne")
        # Highlight the two most operational comparisons.
        highlight_pairs = {
            ("greedy", "beam"): "Greedy vs Beam",
            ("beam", "global_path"): "Beam vs Global path",
        }
        for pair, label in highlight_pairs.items():
            mask = pairwise.apply(
                lambda r: {r["pipeline_left"], r["pipeline_right"]} == set(pair), axis=1
            )
            if mask.any():
                y = divergences.loc[mask, :].iloc[0].values
                color = "#d1495b" if pair == ("greedy", "beam") else "#00798c"
                ax.plot(levels, y, marker="s", linewidth=2.0, label=label, color=color)
        ax.set_title(model.upper())
        ax.set_xlabel("Niveau taxonomique")
        ax.set_xticks(levels)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")
        ax.text(
            0.99,
            0.03,
            f"{len(pair_labels)} paires de pipelines",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#444444",
        )
    axes[0].set_ylabel("Divergence = 1 - accord exact")
    fig.suptitle("Vitesse de divergence entre pipelines selon la profondeur", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "family3_divergence_by_depth.pdf", bbox_inches="tight")
    plt.close(fig)


def format_pair(left: str, right: str) -> str:
    return f"{PIPELINE_LABELS.get(left, left)} -- {PIPELINE_LABELS.get(right, right)}"


if __name__ == "__main__":
    main()
