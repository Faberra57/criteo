#!/usr/bin/env python3
"""Create report figures/tables from full L2-L7 NLP audit outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = {
    "Qwen3": ROOT / "data/l2_l7_nlp_audit_qwen",
    "Jasper": ROOT / "data/l2_l7_nlp_audit_jasper",
}
OUT = ROOT / "latex/rapport_final/figures/l2_l7_nlp_audit_full"
PIPELINE_LABELS = {
    "greedy": "Greedy",
    "beam": "Beam",
    "beam_reranker": "Beam + rerank",
    "clustering_l1_hungarian": "Clustering",
    "global_path": "Global path",
}
ORDER = ["greedy", "beam", "beam_reranker", "clustering_l1_hungarian", "global_path"]
COLORS = {
    "Greedy": "#4c78a8",
    "Beam": "#f58518",
    "Beam + rerank": "#e45756",
    "Clustering": "#72b7b2",
    "Global path": "#54a24b",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    combined = load_summaries()
    combined.to_csv(OUT / "l2_l7_nlp_audit_full_summary.csv", index=False)
    runtime = load_runtime()
    runtime.to_csv(OUT / "l2_l7_nlp_audit_full_runtime.csv", index=False)
    plot_win_rates(combined)
    plot_mean_scores(combined)
    plot_runtime(runtime)


def load_summaries() -> pd.DataFrame:
    rows = []
    for embedder, base in DATA.items():
        for scorer in ["bi_encoder", "cross_encoder"]:
            df = pd.read_csv(base / f"{scorer}_summary.csv")
            df["embedder"] = embedder
            df["scorer"] = "Bi-encoder" if scorer == "bi_encoder" else "Cross-encoder"
            df["pipeline_label"] = df["pipeline"].map(PIPELINE_LABELS).fillna(df["pipeline"])
            rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["pipeline_order"] = out["pipeline"].map({p: i for i, p in enumerate(ORDER)})
    return out.sort_values(["embedder", "scorer", "pipeline_order"])


def load_runtime() -> pd.DataFrame:
    rows = []
    for embedder, base in DATA.items():
        with open(base / "audit_nlp_metrics.json", encoding="utf-8") as f:
            metrics = json.load(f)
        for scorer, values in metrics["scorers"].items():
            rows.append(
                {
                    "embedder": embedder,
                    "scorer": "Bi-encoder" if scorer == "bi-encoder" else "Cross-encoder",
                    "n_products": metrics["n_products_selected"],
                    "n_pairs": metrics["n_pairs_total"],
                    "score_seconds": values.get("score_seconds"),
                    "wall_seconds": values.get("total_scorer_wall_seconds"),
                    "pairs_per_second": values.get("pairs_per_second"),
                    "cuda_peak_allocated_mb": values.get("cuda_peak_allocated_mb"),
                    "cuda_peak_reserved_mb": values.get("cuda_peak_reserved_mb"),
                    "model": values.get("model"),
                }
            )
    return pd.DataFrame(rows)


def plot_win_rates(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
    for ax, scorer in zip(axes, ["Bi-encoder", "Cross-encoder"], strict=True):
        sub = df[df["scorer"] == scorer]
        labels = [PIPELINE_LABELS[p] for p in ORDER]
        x = np.arange(len(labels))
        width = 0.36
        for offset, embedder in [(-width / 2, "Qwen3"), (width / 2, "Jasper")]:
            vals = (
                sub[sub["embedder"] == embedder]
                .set_index("pipeline")
                .reindex(ORDER)["win_rate"]
                .to_numpy()
                * 100
            )
            ax.bar(x + offset, vals, width=width, label=embedder)
        ax.set_title(scorer)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylabel("Win rate (%)" if scorer == "Bi-encoder" else "")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.suptitle("Audit NLP complet : taux de rang 1 par pipeline", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "l2_l7_nlp_audit_win_rates.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_mean_scores(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    for ax, scorer in zip(axes, ["Bi-encoder", "Cross-encoder"], strict=True):
        sub = df[df["scorer"] == scorer]
        labels = [PIPELINE_LABELS[p] for p in ORDER]
        x = np.arange(len(labels))
        width = 0.36
        for offset, embedder in [(-width / 2, "Qwen3"), (width / 2, "Jasper")]:
            vals = (
                sub[sub["embedder"] == embedder]
                .set_index("pipeline")
                .reindex(ORDER)["mean_nlp_score"]
                .to_numpy()
            )
            ax.bar(x + offset, vals, width=width, label=embedder)
        ax.set_title(scorer)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylabel("Score NLP moyen")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.suptitle("Audit NLP complet : score moyen produit--chemin", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "l2_l7_nlp_audit_mean_scores.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_runtime(runtime: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    runtime = runtime.sort_values(["scorer", "embedder"])
    labels = runtime["embedder"] + "\n" + runtime["scorer"]
    vals = runtime["wall_seconds"] / 60.0
    bars = ax.bar(labels, vals, color=["#4c78a8", "#54a24b", "#f58518", "#e45756"])
    ax.set_ylabel("Temps total de scoring (minutes)")
    ax.set_title("Coût de l'audit NLP complet sur 641 265 paires")
    ax.grid(axis="y", alpha=0.25)
    for bar, pps in zip(bars, runtime["pairs_per_second"], strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{pps:.0f} p/s", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "l2_l7_nlp_audit_runtime.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
