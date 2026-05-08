#!/usr/bin/env python3
"""Export descriptive statistics plots for the report."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "dataset" / "preprocessed_lv2.parquet"
DEFAULT_TAXONOMY_PATH = PROJECT_ROOT / "taxonomy.txt"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "latex" / "rapport_final" / "figures"


PALETTE = {
    "blue": "#2F5D8C",
    "orange": "#D9822B",
    "green": "#4B7F52",
    "red": "#A44A3F",
    "gray": "#5B6470",
    "light_gray": "#D9DEE7",
    "background": "#FBFAF7",
}


def _clean_label(label: str, max_len: int = 36) -> str:
    label = str(label).strip()
    if len(label) <= max_len:
        return label
    return label[: max_len - 1] + "…"


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(PALETTE["background"])
    ax.grid(axis="x", alpha=0.25, color="#9CA3AF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")


def load_taxonomy(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            node_id, path_text = line.split("\t", 1)
            parts = [part.strip() for part in path_text.split(">")]
            rows.append(
                {
                    "node_id": node_id.strip(),
                    "path": path_text.strip(),
                    "depth": len(parts),
                    "level1": parts[0] if parts else "",
                    "level2": parts[1] if len(parts) > 1 else "",
                    "label": parts[-1] if parts else "",
                }
            )
    return pd.DataFrame(rows)


def export_l1_distribution(df: pd.DataFrame, output_path: Path) -> None:
    counts = df["level_1_name"].value_counts().sort_values()
    fig_height = max(6.5, 0.34 * len(counts))
    fig, ax = plt.subplots(figsize=(10.5, fig_height), facecolor=PALETTE["background"])
    colors = [
        PALETTE["red"] if value < 200 else PALETTE["orange"] if value < 1000 else PALETTE["blue"]
        for value in counts.values
    ]
    ax.barh([_clean_label(x) for x in counts.index], counts.values, color=colors)
    _style_axes(ax)
    ax.set_title("Distribution des produits annotés au niveau 1", fontsize=14, weight="bold")
    ax.set_xlabel("Nombre de produits")
    ax.set_ylabel("")
    for idx, value in enumerate(counts.values):
        ax.text(value + max(counts.values) * 0.01, idx, f"{value:,}".replace(",", " "), va="center", fontsize=8)
    ax.set_xlim(0, max(counts.values) * 1.16)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def export_taxonomy_depth_distribution(taxonomy: pd.DataFrame, output_path: Path) -> None:
    counts = taxonomy["depth"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=PALETTE["background"])
    bars = ax.bar(counts.index.astype(str), counts.values, color=PALETTE["green"])
    _style_axes(ax)
    ax.grid(axis="y", alpha=0.25, color="#9CA3AF")
    ax.grid(axis="x", visible=False)
    ax.set_title("Nombre de catégories par profondeur taxonomique", fontsize=14, weight="bold")
    ax.set_xlabel("Niveau de profondeur")
    ax.set_ylabel("Nombre de nœuds")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + max(counts.values) * 0.015, f"{int(height)}", ha="center", fontsize=9)
    ax.set_ylim(0, max(counts.values) * 1.15)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def export_taxonomy_l1_branch_sizes(taxonomy: pd.DataFrame, output_path: Path) -> None:
    branch_sizes = taxonomy.groupby("level1").size().sort_values()
    fig_height = max(8.0, 0.28 * len(branch_sizes))
    fig, ax = plt.subplots(figsize=(10.5, fig_height), facecolor=PALETTE["background"])
    ax.barh([_clean_label(x) for x in branch_sizes.index], branch_sizes.values, color=PALETTE["orange"])
    _style_axes(ax)
    ax.set_title("Taille des sous-arbres de niveau 1 dans la taxonomie", fontsize=14, weight="bold")
    ax.set_xlabel("Nombre de catégories dans le sous-arbre")
    ax.set_ylabel("")
    for idx, value in enumerate(branch_sizes.values):
        ax.text(value + max(branch_sizes.values) * 0.01, idx, str(value), va="center", fontsize=8)
    ax.set_xlim(0, max(branch_sizes.values) * 1.15)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def export_l1_median_price(df: pd.DataFrame, output_path: Path) -> None:
    price_stats = (
        df.groupby("level_1_name")["sale_price"]
        .median()
        .sort_values()
    )
    fig_height = max(6.5, 0.34 * len(price_stats))
    fig, ax = plt.subplots(figsize=(10.5, fig_height), facecolor=PALETTE["background"])
    ax.barh([_clean_label(x) for x in price_stats.index], price_stats.values, color=PALETTE["gray"])
    _style_axes(ax)
    ax.set_title("Prix médian par catégorie de niveau 1", fontsize=14, weight="bold")
    ax.set_xlabel("Prix médian")
    ax.set_ylabel("")
    for idx, value in enumerate(price_stats.values):
        ax.text(value + max(price_stats.values) * 0.01, idx, f"{value:.0f}", va="center", fontsize=8)
    ax.set_xlim(0, max(price_stats.values) * 1.15)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def export_taxonomy_tree(taxonomy: pd.DataFrame, output_path: Path, top_l1: int = 10, max_l2: int = 5) -> None:
    branch_sizes = taxonomy.groupby("level1").size().sort_values(ascending=False)
    selected_l1 = branch_sizes.head(top_l1).index.tolist()

    l2_sizes = (
        taxonomy[taxonomy["level2"] != ""]
        .groupby(["level1", "level2"])
        .size()
        .sort_values(ascending=False)
    )

    fig_height = 1.15 * len(selected_l1) + 2.2
    fig, ax = plt.subplots(figsize=(14.5, fig_height), facecolor=PALETTE["background"])
    ax.set_facecolor(PALETTE["background"])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    root_x, l1_x, l2_x = 0.06, 0.33, 0.68
    root_y = 0.5
    ax.text(root_x, root_y, "Taxonomie\nGoogle", ha="center", va="center", fontsize=12, weight="bold",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFFFFF", edgecolor=PALETTE["blue"], linewidth=1.4))

    y_positions = list(reversed([0.08 + i * (0.84 / max(1, len(selected_l1) - 1)) for i in range(len(selected_l1))]))
    for l1_name, y in zip(selected_l1, y_positions):
        size = int(branch_sizes.loc[l1_name])
        ax.plot([root_x + 0.055, l1_x - 0.02], [root_y, y], color=PALETTE["light_gray"], linewidth=1.0)
        ax.text(l1_x, y, f"{_clean_label(l1_name, 28)}\n({size} nœuds)", ha="center", va="center", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFFF", edgecolor=PALETTE["orange"], linewidth=1.0))

        children = []
        if l1_name in l2_sizes.index.get_level_values(0):
            children = l2_sizes.loc[l1_name].sort_values(ascending=False).head(max_l2)
        if len(children) == 0:
            continue

        child_offsets = [0] if len(children) == 1 else [
            -0.06 + i * (0.12 / max(1, len(children) - 1)) for i in range(len(children))
        ]
        for (child_name, child_size), offset in zip(children.items(), child_offsets):
            cy = min(0.96, max(0.04, y + offset))
            ax.plot([l1_x + 0.07, l2_x - 0.02], [y, cy], color=PALETTE["light_gray"], linewidth=0.9)
            ax.text(l2_x, cy, f"{_clean_label(child_name, 34)} ({int(child_size)})",
                    ha="left", va="center", fontsize=7.4,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=0.8))

    ax.text(0.02, 0.98, "Vue partielle : 10 plus grands sous-arbres L1 et 5 principaux enfants L2",
            ha="left", va="top", fontsize=10, color=PALETTE["gray"])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def export_summary(df: pd.DataFrame, taxonomy: pd.DataFrame, output_path: Path) -> None:
    word_count = df["text"].fillna("").str.split().str.len()
    l1_counts = df["level_1_name"].value_counts()
    rows = [
        ("n_products", len(df)),
        ("n_l1_labels", df["level_1_name"].nunique()),
        ("largest_l1_count", int(l1_counts.max())),
        ("smallest_l1_count", int(l1_counts.min())),
        ("median_l1_count", float(l1_counts.median())),
        ("price_median", float(df["sale_price"].median())),
        ("price_p95", float(df["sale_price"].quantile(0.95))),
        ("word_count_median", float(word_count.median())),
        ("word_count_p95", float(word_count.quantile(0.95))),
        ("taxonomy_nodes", len(taxonomy)),
        ("taxonomy_l1_nodes", taxonomy["level1"].nunique()),
        ("taxonomy_max_depth", int(taxonomy["depth"].max())),
    ]
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--taxonomy-path", type=Path, default=DEFAULT_TAXONOMY_PATH)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--top-l1-tree", type=int, default=10)
    parser.add_argument("--max-l2-tree", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.dataset_path)
    taxonomy = load_taxonomy(args.taxonomy_path)

    export_l1_distribution(df, args.figures_dir / "dataset_l1_distribution.pdf")
    export_taxonomy_depth_distribution(taxonomy, args.figures_dir / "taxonomy_depth_distribution.pdf")
    export_taxonomy_l1_branch_sizes(taxonomy, args.figures_dir / "taxonomy_l1_branch_sizes.pdf")
    export_taxonomy_tree(taxonomy, args.figures_dir / "taxonomy_l1_l2_tree.pdf", args.top_l1_tree, args.max_l2_tree)
    export_l1_median_price(df, args.figures_dir / "dataset_l1_median_price.pdf")
    export_summary(df, taxonomy, args.figures_dir / "dataset_descriptive_summary.csv")

    print(f"Exported descriptive figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
