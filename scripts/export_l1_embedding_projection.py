#!/usr/bin/env python3
"""Project product embeddings to 2D and export a category-colored scatter plot."""

from __future__ import annotations

import argparse
import csv
import os
import random
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
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "latex"
    / "rapport_final"
    / "figures"
    / "l1_embedding_projection_pca.pdf"
)
DEFAULT_COORDS_CSV = (
    REPO_ROOT
    / "latex"
    / "rapport_final"
    / "figures"
    / "l1_embedding_projection_pca.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a fine-tuned SentenceTransformer, embed products from "
            "retrieval_predictions.csv, reduce embeddings to 2D, and export a plot."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Model directory. Defaults to <run-dir>/best_model, then <run-dir>/final_model.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="CSV with text and label columns. Defaults to <run-dir>/retrieval_predictions.csv.",
    )
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label_key")
    parser.add_argument(
        "--color-by",
        choices=["label_key", "predicted_node_key"],
        default="label_key",
        help="Column used for point colors when present.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--coords-csv", type=Path, default=DEFAULT_COORDS_CSV)
    parser.add_argument(
        "--method",
        choices=["pca", "tsne", "umap"],
        default="pca",
        help="2D reduction method. PCA is fastest and deterministic.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Optional number of products to sample. 0 means use all products.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default=None, help="Optional device: cuda, mps, cpu."
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable embedding normalization.",
    )
    parser.add_argument(
        "--cache-embeddings",
        type=Path,
        default=None,
        help="Optional .npz cache path to avoid recomputing embeddings.",
    )
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--point-size", type=float, default=7.0)
    parser.add_argument("--point-alpha", type=float, default=0.42)
    parser.add_argument(
        "--label-centroids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Annotate category centroids on the plot.",
    )
    parser.add_argument(
        "--ellipses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw robust covariance ellipses around categories with enough points.",
    )
    parser.add_argument(
        "--min-ellipse-points",
        type=int,
        default=30,
        help="Minimum category support required to draw an ellipse.",
    )
    return parser.parse_args()


def configure_plot_runtime() -> None:
    cache_root = Path(tempfile.gettempdir()) / "criteo-plot-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))


def default_model_dir(run_dir: Path) -> Path:
    best_model = run_dir / "best_model"
    final_model = run_dir / "final_model"
    if best_model.exists():
        return best_model
    if final_model.exists():
        return final_model
    raise FileNotFoundError(f"No best_model or final_model found in {run_dir}")


def read_products(
    input_csv: Path,
    *,
    text_col: str,
    label_col: str,
    color_by: str,
    sample_size: int,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    rows: list[dict[str, str]] = []
    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {text_col, label_col}
        missing = sorted(required.difference(fieldnames))
        if missing:
            raise ValueError(f"Missing columns in {input_csv}: {missing}")
        color_col = color_by if color_by in fieldnames else label_col
        for row in reader:
            text = (row.get(text_col) or "").strip()
            label = (row.get(label_col) or "").strip()
            color_label = (row.get(color_col) or label).strip()
            if text and label:
                rows.append({text_col: text, label_col: label, color_col: color_label})

    if not rows:
        raise ValueError(f"No usable rows found in {input_csv}")

    if sample_size > 0 and sample_size < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, sample_size)

    texts = [row[text_col] for row in rows]
    labels = [row[label_col] for row in rows]
    color_labels = [
        row[color_by] if color_by in row else row[label_col] for row in rows
    ]
    return texts, labels, color_labels


def load_or_compute_embeddings(
    *,
    texts: list[str],
    labels: list[str],
    color_labels: list[str],
    model_dir: Path,
    batch_size: int,
    max_seq_length: int,
    normalize: bool,
    device: str | None,
    cache_path: Path | None,
):
    import numpy as np

    if cache_path and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        return (
            cached["embeddings"],
            cached["labels"].tolist(),
            cached["color_labels"].tolist(),
        )

    from sentence_transformers import SentenceTransformer

    model_kwargs = {"trust_remote_code": True}
    if device:
        model_kwargs["device"] = device
    model = SentenceTransformer(str(model_dir), **model_kwargs)
    model.max_seq_length = max_seq_length

    encode_kwargs = {
        "batch_size": batch_size,
        "show_progress_bar": True,
        "convert_to_numpy": True,
        "normalize_embeddings": normalize,
    }
    if device:
        encode_kwargs["device"] = device

    try:
        embeddings = model.encode(texts, task="retrieval", **encode_kwargs)
    except TypeError:
        embeddings = model.encode(texts, **encode_kwargs)

    embeddings = embeddings.astype(np.float32)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            embeddings=embeddings,
            labels=np.asarray(labels, dtype=object),
            color_labels=np.asarray(color_labels, dtype=object),
        )
    return embeddings, labels, color_labels


def reduce_to_2d(embeddings, *, method: str, seed: int, args: argparse.Namespace):
    if method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2, random_state=seed)
        return reducer.fit_transform(embeddings), "PCA"

    if method == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(
            n_components=2,
            perplexity=args.tsne_perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        )
        return reducer.fit_transform(embeddings), "t-SNE"

    try:
        import umap
    except ImportError as exc:
        raise ImportError("UMAP requires installing umap-learn.") from exc

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(embeddings), "UMAP"


def write_coords_csv(
    coords, labels: list[str], color_labels: list[str], output_csv: Path
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "label_key", "color_label"])
        for point, label, color_label in zip(coords, labels, color_labels, strict=True):
            writer.writerow([float(point[0]), float(point[1]), label, color_label])


def export_plot(
    coords,
    *,
    color_labels: list[str],
    output_path: Path,
    title: str,
    method_label: str,
    point_size: float,
    point_alpha: float,
    label_centroids: bool,
    ellipses: bool,
    min_ellipse_points: int,
) -> None:
    configure_plot_runtime()
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse

    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter(color_labels)
    categories = [category for category, _ in counts.most_common()]
    palette = list(plt.get_cmap("tab20").colors)
    if len(categories) > len(palette):
        palette.extend(plt.get_cmap("tab20b").colors)
        palette.extend(plt.get_cmap("tab20c").colors)
    category_to_color = {
        category: palette[index % len(palette)]
        for index, category in enumerate(categories)
    }

    fig, ax = plt.subplots(figsize=(13.5, 8.5))

    # Plot larger classes first so rare classes remain visible on top.
    for category in reversed(categories):
        mask = np.asarray([label == category for label in color_labels])
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            alpha=point_alpha,
            color=category_to_color[category],
            linewidths=0,
            rasterized=True,
        )

    centroid_rows = []
    for category in categories:
        mask = np.asarray([label == category for label in color_labels])
        points = coords[mask]
        if len(points) == 0:
            continue
        centroid = points.mean(axis=0)
        centroid_rows.append((category, len(points), centroid, points))

    if ellipses:
        for category, support, centroid, points in centroid_rows:
            if support < min_ellipse_points:
                continue
            covariance = np.cov(points[:, 0], points[:, 1])
            if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
                continue
            values, vectors = np.linalg.eigh(covariance)
            values = np.maximum(values, 0)
            order = values.argsort()[::-1]
            values = values[order]
            vectors = vectors[:, order]
            angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
            width, height = 2.0 * np.sqrt(values)
            ellipse = Ellipse(
                xy=centroid,
                width=width,
                height=height,
                angle=angle,
                facecolor=category_to_color[category],
                edgecolor=category_to_color[category],
                alpha=0.075,
                linewidth=1.0,
            )
            ax.add_patch(ellipse)

    for category, support, centroid, _ in centroid_rows:
        ax.scatter(
            centroid[0],
            centroid[1],
            marker="X",
            s=55,
            color=category_to_color[category],
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )

    if label_centroids:
        for category, support, centroid, _ in centroid_rows:
            label = category.replace(" & ", " &\n")
            ax.annotate(
                label,
                xy=centroid,
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.5,
                color="#222222",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": category_to_color[category],
                    "linewidth": 0.6,
                    "alpha": 0.72,
                },
                zorder=6,
            )

    handles = []
    for category in categories:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=5,
                markerfacecolor=category_to_color[category],
                markeredgecolor="none",
                label=f"{category} (n={counts[category]})",
            )
        )

    total_points = len(color_labels)
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel(f"{method_label} 1")
    ax.set_ylabel(f"{method_label} 2")
    ax.text(
        0.01,
        0.01,
        f"{total_points:,} produits | couleur = categorie vraie",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#dddddd",
            "alpha": 0.85,
        },
    )
    ax.grid(True, alpha=0.16)
    ax.set_axisbelow(True)
    ax.legend(
        handles=handles,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=True,
        fontsize=7.4,
        ncol=2 if len(categories) > 14 else 1,
        borderaxespad=0.0,
    )
    fig.tight_layout()
    if output_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        fig.savefig(output_path, bbox_inches="tight", dpi=220)
    else:
        fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv or (args.run_dir / "retrieval_predictions.csv")
    model_dir = args.model_dir or default_model_dir(args.run_dir)
    if args.method != "pca":
        if args.output == DEFAULT_OUTPUT:
            args.output = args.output.with_name(
                f"l1_embedding_projection_{args.method}.pdf"
            )
        if args.coords_csv == DEFAULT_COORDS_CSV:
            args.coords_csv = args.coords_csv.with_name(
                f"l1_embedding_projection_{args.method}.csv"
            )

    texts, labels, color_labels = read_products(
        input_csv,
        text_col=args.text_col,
        label_col=args.label_col,
        color_by=args.color_by,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    embeddings, labels, color_labels = load_or_compute_embeddings(
        texts=texts,
        labels=labels,
        color_labels=color_labels,
        model_dir=model_dir,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        normalize=not args.no_normalize,
        device=args.device,
        cache_path=args.cache_embeddings,
    )
    coords, method_label = reduce_to_2d(
        embeddings,
        method=args.method,
        seed=args.seed,
        args=args,
    )
    write_coords_csv(coords, labels, color_labels, args.coords_csv)
    export_plot(
        coords,
        color_labels=color_labels,
        output_path=args.output,
        method_label=method_label,
        title=f"Projection 2D des embeddings produits L1 ({method_label})",
        point_size=args.point_size,
        point_alpha=args.point_alpha,
        label_centroids=args.label_centroids,
        ellipses=args.ellipses,
        min_ellipse_points=args.min_ellipse_points,
    )

    print(f"Model: {model_dir}")
    print(f"Input products: {len(labels)}")
    print(f"Exported plot: {args.output}")
    print(f"Exported coordinates: {args.coords_csv}")


if __name__ == "__main__":
    main()
