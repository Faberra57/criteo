#!/usr/bin/env python3
"""Export family-3 metrics for unsupervised L2-L7 pipeline comparison.

Family 3 metrics compare the predictions produced by several pipelines without
using L2-L7 ground truth:
- exact agreement on the full predicted taxonomy path;
- exact agreement level by level from L2 to L7;
- Cramer's V association for discrete predicted labels;
- product-level disagreement summaries.

By default the script runs on all common products across the selected pipelines.
Use --selection disagreements or --sample-size only for audits, not for the main
global comparison in the report.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

QWEN_PRESET = {
    "greedy": ROOT / "data/qwen/pipeline1_hierarchical_predictions_greedy.csv",
    "beam": ROOT / "data/qwen/pipeline1_hierarchical_predictions_beam.csv",
    "beam_reranker": ROOT / "data/qwen/pipeline1_hierarchical_predictions_beam-rerank.csv",
    "clustering_l1_hungarian": ROOT / "data/qwen/pipeline2_hierarchical_clustering_predictions_l1routed_hungarian.csv",
    "global_path": ROOT / "data/qwen/pipeline3_global_path_predictions.csv",
}

JASPER_PRESET = {
    "greedy": ROOT / "data/jasper/pipeline1_hierarchical_predictions_jasper_greedy.csv",
    "beam": ROOT / "data/jasper/pipeline1_hierarchical_predictions_jasper_beam.csv",
    "beam_reranker": ROOT / "data/jasper/pipeline1_hierarchical_predictions_beam-rerank_jasper.csv",
    "clustering_l1_hungarian": ROOT / "data/jasper/pipeline2_hierarchical_clustering_predictions_jasper_l1routed_hungarian.csv",
    "global_path": ROOT / "data/jasper/pipeline3_global_path_predictions_jasper.csv",
}


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_paths = resolve_pipeline_paths(args)
    if len(pipeline_paths) < 2:
        raise SystemExit("At least two existing pipeline prediction files are required.")

    frames = {label: load_prediction_frame(path, label, args) for label, path in pipeline_paths.items()}
    selected_ids = select_product_ids(frames, args)
    if not selected_ids:
        raise SystemExit("No product id remains after the selected filtering.")

    wide = build_wide_predictions(frames, selected_ids, args)
    pairwise = compute_pairwise_metrics(wide, args)
    product_summary = compute_product_disagreement_summary(wide, args)
    global_summary = compute_global_summary(wide, product_summary, pairwise, args)

    pairwise.to_csv(output_dir / "family3_pairwise_metrics.csv", index=False)
    product_summary.to_csv(output_dir / "family3_product_disagreement_summary.csv", index=False)
    global_summary.to_csv(output_dir / "family3_global_summary.csv", index=False)
    write_metric_matrices(pairwise, output_dir)

    heatmap_outputs = []
    if not args.no_plots:
        heatmap_outputs = write_heatmaps(pairwise, output_dir)

    metadata = {
        "args": vars(args),
        "pipeline_paths": {label: str(path) for label, path in pipeline_paths.items()},
        "pipelines": list(pipeline_paths),
        "n_pipelines": len(pipeline_paths),
        "n_common_products_before_filter": int(common_id_count(frames, args.id_col)),
        "n_products_selected": int(len(selected_ids)),
        "selection": args.selection,
        "sample_size": args.sample_size,
        "elapsed_seconds": float(time.perf_counter() - started),
        "outputs": {
            "pairwise_metrics": str(output_dir / "family3_pairwise_metrics.csv"),
            "product_disagreement_summary": str(output_dir / "family3_product_disagreement_summary.csv"),
            "global_summary": str(output_dir / "family3_global_summary.csv"),
            "heatmaps": heatmap_outputs,
        },
    }
    (output_dir / "family3_metrics.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Saved family-3 metrics to {output_dir}")
    print(f"Products compared: {len(selected_ids):,}")
    print(f"Pipelines: {', '.join(pipeline_paths)}")
    print("\nPairwise metrics:")
    print(pairwise.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["qwen", "jasper", "none"], default="qwen")
    parser.add_argument(
        "--pipeline",
        action="append",
        help="Prediction file as label=path. Can be repeated. Extends or overrides the preset.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: latex/rapport_final/figures/l2_l7_family3_<preset>.",
    )
    parser.add_argument("--id-col", default="hashed_external_id")
    parser.add_argument("--path-col", default="predicted_taxonomy_path")
    parser.add_argument("--level-prefix", default="predicted_level_")
    parser.add_argument("--level-suffix", default="_name")
    parser.add_argument("--min-level", type=int, default=2)
    parser.add_argument("--max-level", type=int, default=7)
    parser.add_argument(
        "--selection",
        choices=["all", "disagreements"],
        default="all",
        help="Use all common products for global comparison, or only products where at least two paths differ.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Optional random sample after selection. 0 means no sampling.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true", help="Do not export heatmap PDFs.")
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        path = Path(args.output_dir).expanduser()
    else:
        suffix = args.preset if args.preset != "none" else "custom"
        path = ROOT / "latex/rapport_final/figures" / f"l2_l7_family3_{suffix}"
    if not path.is_absolute():
        path = ROOT / path
    return path


def resolve_pipeline_paths(args: argparse.Namespace) -> dict[str, Path]:
    if args.preset == "qwen":
        paths = dict(QWEN_PRESET)
    elif args.preset == "jasper":
        paths = dict(JASPER_PRESET)
    else:
        paths = {}

    for item in args.pipeline or []:
        if "=" not in item:
            path = resolve_path(item)
            paths[path.stem] = path
            continue
        label, path_str = item.split("=", 1)
        paths[label.strip()] = resolve_path(path_str.strip())

    resolved = {}
    for label, path in paths.items():
        path = resolve_path(path)
        if path.exists():
            resolved[label] = path
        else:
            print(f"WARNING missing prediction file for {label}: {path}")
    return resolved


def resolve_path(path: str | Path) -> Path:
    out = Path(path).expanduser()
    if not out.is_absolute():
        out = ROOT / out
    return out


def load_prediction_frame(path: Path, label: str, args: argparse.Namespace) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    required = {args.id_col, args.path_col}
    missing = required.difference(header.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    usecols = [args.id_col, args.path_col]
    for level in range(args.min_level, args.max_level + 1):
        col = level_col(args, level)
        if col in header.columns:
            usecols.append(col)
        else:
            print(f"WARNING {path.name} has no column {col}; level {level} will be empty for {label}.")

    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df = df.drop_duplicates(args.id_col).copy()
    df[args.id_col] = df[args.id_col].astype(str)
    df[args.path_col] = normalize_label_series(df[args.path_col])
    for level in range(args.min_level, args.max_level + 1):
        col = level_col(args, level)
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = normalize_label_series(df[col])
    df["pipeline"] = label
    return df


def normalize_label_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def level_col(args: argparse.Namespace, level: int) -> str:
    return f"{args.level_prefix}{level}{args.level_suffix}"


def common_id_count(frames: dict[str, pd.DataFrame], id_col: str) -> int:
    id_sets = [set(df[id_col]) for df in frames.values()]
    return len(set.intersection(*id_sets)) if id_sets else 0


def select_product_ids(frames: dict[str, pd.DataFrame], args: argparse.Namespace) -> list[str]:
    id_sets = [set(df[args.id_col]) for df in frames.values()]
    common_ids = sorted(set.intersection(*id_sets))
    if args.selection == "disagreements":
        path_wide = build_single_wide(frames, common_ids, args.id_col, args.path_col)
        common_ids = path_wide.index[path_wide.nunique(axis=1) > 1].astype(str).tolist()
    if args.sample_size and args.sample_size < len(common_ids):
        rng = np.random.default_rng(args.random_seed)
        common_ids = sorted(rng.choice(common_ids, size=args.sample_size, replace=False).astype(str).tolist())
    return common_ids


def build_single_wide(
    frames: dict[str, pd.DataFrame],
    selected_ids: list[str],
    id_col: str,
    value_col: str,
) -> pd.DataFrame:
    selected = set(selected_ids)
    cols = []
    for label, df in frames.items():
        tmp = df[df[id_col].isin(selected)][[id_col, value_col]].copy()
        tmp = tmp.drop_duplicates(id_col).set_index(id_col)[value_col]
        cols.append(tmp.rename(label))
    return pd.concat(cols, axis=1, join="inner").fillna("").astype(str)


def build_wide_predictions(
    frames: dict[str, pd.DataFrame],
    selected_ids: list[str],
    args: argparse.Namespace,
) -> dict[str, pd.DataFrame]:
    wide = {"path": build_single_wide(frames, selected_ids, args.id_col, args.path_col)}
    for level in range(args.min_level, args.max_level + 1):
        col = level_col(args, level)
        wide[f"level_{level}"] = build_single_wide(frames, selected_ids, args.id_col, col)
    return wide


def compute_pairwise_metrics(wide: dict[str, pd.DataFrame], args: argparse.Namespace) -> pd.DataFrame:
    pipelines = list(wide["path"].columns)
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(pipelines):
        for right in pipelines[i + 1 :]:
            path_pair = wide["path"][[left, right]].dropna()
            row: dict[str, Any] = {
                "pipeline_left": left,
                "pipeline_right": right,
                "n_overlap": int(len(path_pair)),
                "exact_path_agreement": exact_agreement(path_pair[left], path_pair[right]),
                "path_cramers_v": cramers_v(path_pair[left], path_pair[right]),
            }
            shared_prefix_depths = []
            for level in range(args.min_level, args.max_level + 1):
                key = f"level_{level}"
                level_pair = wide[key][[left, right]].fillna("").astype(str)
                non_empty = (level_pair[left] != "") | (level_pair[right] != "")
                level_pair = level_pair[non_empty]
                row[f"level_{level}_n_overlap"] = int(len(level_pair))
                if len(level_pair) == 0:
                    row[f"level_{level}_agreement"] = math.nan
                    row[f"level_{level}_cramers_v"] = math.nan
                    continue
                row[f"level_{level}_agreement"] = exact_agreement(level_pair[left], level_pair[right])
                row[f"level_{level}_cramers_v"] = cramers_v(level_pair[left], level_pair[right])

            for product_id in path_pair.index:
                depth = 1
                for level in range(args.min_level, args.max_level + 1):
                    values = wide[f"level_{level}"].loc[product_id, [left, right]].fillna("").astype(str)
                    if values.iloc[0] == values.iloc[1] and values.iloc[0] != "":
                        depth = level
                    else:
                        break
                shared_prefix_depths.append(depth)
            row["mean_shared_prefix_depth"] = float(np.mean(shared_prefix_depths)) if shared_prefix_depths else math.nan
            row["median_shared_prefix_depth"] = float(np.median(shared_prefix_depths)) if shared_prefix_depths else math.nan
            rows.append(row)
    return pd.DataFrame(rows)


def exact_agreement(left: pd.Series, right: pd.Series) -> float:
    left_values = left.fillna("").astype(str)
    right_values = right.fillna("").astype(str)
    if len(left_values) == 0:
        return math.nan
    return float((left_values == right_values).mean())


def cramers_v(left: pd.Series, right: pd.Series) -> float:
    left_values = left.fillna("").astype(str)
    right_values = right.fillna("").astype(str)
    table = pd.crosstab(left_values, right_values)
    n = table.to_numpy().sum()
    if n == 0 or table.shape[0] < 2 or table.shape[1] < 2:
        return math.nan
    observed = table.to_numpy(dtype=float)
    row_sum = observed.sum(axis=1, keepdims=True)
    col_sum = observed.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / n
    mask = expected > 0
    chi2 = float(((observed[mask] - expected[mask]) ** 2 / expected[mask]).sum())
    phi2 = chi2 / n
    denominator = max(1, min(table.shape[0] - 1, table.shape[1] - 1))
    return float(math.sqrt(phi2 / denominator))


def compute_product_disagreement_summary(wide: dict[str, pd.DataFrame], args: argparse.Namespace) -> pd.DataFrame:
    path_wide = wide["path"]
    rows = []
    for product_id, values in path_wide.iterrows():
        paths = values.fillna("").astype(str)
        row: dict[str, Any] = {
            args.id_col: product_id,
            "n_unique_paths": int(paths.nunique()),
            "all_pipelines_same_path": bool(paths.nunique() == 1),
            "predicted_paths_json": json.dumps(paths.to_dict(), ensure_ascii=False),
        }
        first_disagreement = math.nan
        last_common_level = 1
        for level in range(args.min_level, args.max_level + 1):
            level_values = wide[f"level_{level}"].loc[product_id].fillna("").astype(str)
            non_empty = level_values[level_values != ""]
            unique_count = int(non_empty.nunique()) if len(non_empty) else 0
            row[f"n_unique_level_{level}"] = unique_count
            row[f"all_pipelines_same_level_{level}"] = bool(unique_count <= 1) if len(non_empty) else False
            if unique_count <= 1 and len(non_empty):
                last_common_level = level
            elif math.isnan(first_disagreement):
                first_disagreement = level
        row["first_disagreement_level"] = first_disagreement
        row["last_common_level_before_disagreement"] = last_common_level
        rows.append(row)
    return pd.DataFrame(rows)


def compute_global_summary(
    wide: dict[str, pd.DataFrame],
    product_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    row: dict[str, Any] = {
        "n_products": int(len(product_summary)),
        "n_pipelines": int(len(wide["path"].columns)),
        "all_pipelines_same_path_rate": float(product_summary["all_pipelines_same_path"].mean()),
        "mean_unique_paths_per_product": float(product_summary["n_unique_paths"].mean()),
        "mean_pairwise_exact_path_agreement": float(pairwise["exact_path_agreement"].mean()) if not pairwise.empty else math.nan,
        "mean_pairwise_path_cramers_v": float(pairwise["path_cramers_v"].mean()) if not pairwise.empty else math.nan,
        "mean_last_common_level_before_disagreement": float(
            pd.to_numeric(product_summary["last_common_level_before_disagreement"], errors="coerce").mean()
        ),
    }
    for level in range(args.min_level, args.max_level + 1):
        col = f"all_pipelines_same_level_{level}"
        row[f"all_pipelines_same_level_{level}_rate"] = float(product_summary[col].mean())
        pair_col = f"level_{level}_agreement"
        cramer_col = f"level_{level}_cramers_v"
        row[f"mean_pairwise_level_{level}_agreement"] = float(pairwise[pair_col].mean()) if pair_col in pairwise else math.nan
        row[f"mean_pairwise_level_{level}_cramers_v"] = float(pairwise[cramer_col].mean()) if cramer_col in pairwise else math.nan
    rows.append(row)
    return pd.DataFrame(rows)


def write_metric_matrices(pairwise: pd.DataFrame, output_dir: Path) -> None:
    metrics = ["exact_path_agreement", "path_cramers_v", "mean_shared_prefix_depth", "median_shared_prefix_depth"]
    metrics.extend(col for col in pairwise.columns if col.startswith("level_") and col.endswith(("_agreement", "_cramers_v")))
    for metric in metrics:
        if metric not in pairwise.columns:
            continue
        matrix = pairwise_metric_matrix(pairwise, metric)
        matrix.to_csv(output_dir / f"family3_{metric}_matrix.csv")


def pairwise_metric_matrix(pairwise: pd.DataFrame, metric: str) -> pd.DataFrame:
    pipelines = sorted(set(pairwise["pipeline_left"]).union(pairwise["pipeline_right"]))
    matrix = pd.DataFrame(np.nan, index=pipelines, columns=pipelines, dtype=float)
    np.fill_diagonal(matrix.values, 1.0)
    for _, row in pairwise.iterrows():
        left = row["pipeline_left"]
        right = row["pipeline_right"]
        value = row.get(metric, math.nan)
        matrix.loc[left, right] = value
        matrix.loc[right, left] = value
    return matrix


def write_heatmaps(pairwise: pd.DataFrame, output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        print(f"WARNING could not import matplotlib, skipping heatmaps: {exc}")
        return []

    outputs = []
    heatmap_specs = [
        ("exact_path_agreement", "Exact path agreement"),
        ("path_cramers_v", "Cramer's V on full path"),
        ("level_2_agreement", "L2 agreement"),
        ("level_3_agreement", "L3 agreement"),
        ("level_4_agreement", "L4 agreement"),
    ]
    for metric, title in heatmap_specs:
        if metric not in pairwise.columns:
            continue
        matrix = pairwise_metric_matrix(pairwise, metric)
        fig, ax = plt.subplots(figsize=(7.5, 6.2))
        im = ax.imshow(matrix.values, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(matrix.index)), matrix.index)
        ax.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                if np.isfinite(value):
                    color = "white" if value < 0.55 else "black"
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out = output_dir / f"family3_{metric}_heatmap.pdf"
        fig.savefig(out)
        plt.close(fig)
        outputs.append(str(out))
    return outputs


if __name__ == "__main__":
    main()
