#!/usr/bin/env python3
"""Evaluate unsupervised L2-L7 pipeline predictions with agreement and reranking proxies.

The script does not require L2-L7 ground truth. It compares pipelines through:
- agreement/disagreement between predicted paths;
- categorical association between pipeline predictions with Cramer's V;
- NLP scores between product text and predicted taxonomy path;
- per-product ranking of pipeline predictions by that NLP score;
- score and win-rate breakdowns by predicted L1 category.
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_paths = resolve_pipeline_paths(args)
    if len(pipeline_paths) < 2:
        raise SystemExit("At least two prediction files are required.")

    prototypes = load_category_prototypes(args.category_prototypes_path, args.max_prototypes_per_category)
    frames = {
        label: load_prediction_frame(path, label=label, args=args, prototypes=prototypes)
        for label, path in pipeline_paths.items()
    }

    selected_ids = select_product_ids(frames, args=args)
    long_df = build_long_frame(frames, selected_ids=selected_ids, args=args)
    long_df["candidate_text"] = build_candidate_texts(long_df)

    agreement = compute_product_agreement(long_df, args=args)
    agreement.to_csv(output_dir / "l2_l7_reranking_product_agreement.csv", index=False)

    pairwise_agreement = compute_pairwise_agreement(long_df, args=args)
    pairwise_agreement.to_csv(output_dir / "l2_l7_reranking_pairwise_agreement.csv", index=False)
    write_pairwise_matrices(pairwise_agreement, output_dir=output_dir)

    if args.skip_scorer:
        long_df["nlp_score"] = np.nan
        ranking = pd.DataFrame()
        summary = summarize_without_scores(long_df, pairwise_agreement)
        by_l1 = pd.DataFrame()
        pairwise_scores = pd.DataFrame()
    else:
        scores, score_metrics = score_predictions(long_df, args=args)
        long_df["nlp_score"] = scores
        if not np.isfinite(scores).any():
            raise RuntimeError(
                "The selected scorer returned only non-finite scores. "
                "Try --scorer bi-encoder or another model."
            )
        ranking = rank_pipelines_per_product(long_df, args=args)
        summary = summarize_scores(long_df, ranking, pairwise_agreement, score_metrics)
        by_l1 = summarize_by_l1(long_df, ranking, min_products=args.min_products_per_l1)
        pairwise_scores = compute_pairwise_score_preferences(long_df, args=args)
        ranking.to_csv(output_dir / "l2_l7_reranking_product_ranks.csv", index=False)
        by_l1.to_csv(output_dir / "l2_l7_reranking_by_l1.csv", index=False)
        pairwise_scores.to_csv(output_dir / "l2_l7_reranking_pairwise_scores.csv", index=False)

    long_export_cols = [
        args.id_col,
        "pipeline",
        "predicted_taxonomy_path",
        "predicted_level_1_name",
        "resolved_depth",
        "nlp_score",
        "candidate_text",
    ]
    long_df[[col for col in long_export_cols if col in long_df.columns]].to_csv(
        output_dir / "l2_l7_reranking_long_scores.csv", index=False
    )
    summary.to_csv(output_dir / "l2_l7_reranking_summary.csv", index=False)

    metrics = {
        "args": vars(args),
        "pipeline_paths": {label: str(path) for label, path in pipeline_paths.items()},
        "n_pipelines": len(pipeline_paths),
        "n_products_selected": int(len(selected_ids)),
        "n_pairs_scored": int(len(long_df) if not args.skip_scorer else 0),
        "elapsed_seconds": float(time.perf_counter() - started),
        "output_dir": str(output_dir),
    }
    (output_dir / "l2_l7_reranking_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Saved analysis to {output_dir}")
    print(summary.to_string(index=False))
    if not by_l1.empty:
        print("\nTop L1 breakdown rows:")
        print(by_l1.head(20).to_string(index=False))


def resolve_pipeline_paths(args: argparse.Namespace) -> dict[str, Path]:
    if args.preset == "qwen":
        paths = dict(QWEN_PRESET)
    elif args.preset == "jasper":
        paths = dict(JASPER_PRESET)
    else:
        paths = {}

    for item in args.pipeline or []:
        if "=" not in item:
            path = Path(item)
            paths[path.stem] = path
            continue
        label, path_str = item.split("=", 1)
        paths[label.strip()] = Path(path_str)

    resolved = {}
    for label, path in paths.items():
        path = path.expanduser()
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            resolved[label] = path
        else:
            print(f"WARNING missing prediction file for {label}: {path}")
    return resolved


def load_category_prototypes(path_str: str, max_prototypes: int) -> dict[str, str]:
    if not path_str:
        return {}
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        print(f"WARNING missing category prototypes file: {path}")
        return {}
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    required = {"node_key", "prototype_text"}
    if not required.issubset(df.columns):
        print(f"WARNING category prototypes file must contain {required}: {path}")
        return {}
    out: dict[str, str] = {}
    for node_key, group in df.dropna(subset=["node_key", "prototype_text"]).groupby("node_key", sort=False):
        texts = (
            group["prototype_text"]
            .astype(str)
            .drop_duplicates()
            .head(max_prototypes)
            .tolist()
        )
        out[str(node_key)] = "; ".join(texts)
    return out


def load_prediction_frame(
    path: Path,
    *,
    label: str,
    args: argparse.Namespace,
    prototypes: dict[str, str],
) -> pd.DataFrame:
    usecols = lambda col: col in columns_to_read(args)
    df = pd.read_csv(path, usecols=usecols)
    if args.id_col not in df.columns:
        raise ValueError(f"{path} does not contain id column {args.id_col!r}")
    if args.path_col not in df.columns:
        raise ValueError(f"{path} does not contain path column {args.path_col!r}")
    df = df.drop_duplicates(args.id_col).copy()
    df["pipeline"] = label
    df["source_file"] = str(path)
    df["final_category_key"] = deepest_value(df, prefix="predicted_level_", suffix="_key")
    df["final_category_name"] = deepest_value(df, prefix="predicted_level_", suffix="_name")
    if prototypes:
        df["final_category_enrichment"] = df["final_category_key"].map(prototypes).fillna("")
    else:
        df["final_category_enrichment"] = ""
    return df


def columns_to_read(args: argparse.Namespace) -> set[str]:
    cols = {
        args.id_col,
        args.text_col,
        args.path_col,
        args.key_path_col,
        args.l1_col,
        args.score_col,
        args.depth_col,
        "pipeline_mode",
    }
    for depth in range(1, 8):
        cols.add(f"predicted_level_{depth}_name")
        cols.add(f"predicted_level_{depth}_key")
    return cols


def deepest_value(df: pd.DataFrame, *, prefix: str, suffix: str) -> pd.Series:
    values = pd.Series("", index=df.index, dtype=object)
    for depth in range(1, 8):
        col = f"{prefix}{depth}{suffix}"
        if col in df.columns:
            current = df[col].fillna("").astype(str)
            values = values.mask(current != "", current)
    return values


def select_product_ids(
    frames: dict[str, pd.DataFrame],
    *,
    args: argparse.Namespace,
) -> list[str]:
    id_sets = [set(df[args.id_col].astype(str)) for df in frames.values()]
    common_ids = sorted(set.intersection(*id_sets))
    if not common_ids:
        raise ValueError("No common product id across all pipeline outputs.")

    if args.selection == "disagreements":
        path_by_pipeline = []
        for df in frames.values():
            tmp = df[[args.id_col, args.path_col]].copy()
            tmp[args.id_col] = tmp[args.id_col].astype(str)
            path_by_pipeline.append(tmp.set_index(args.id_col)[args.path_col].fillna("").astype(str))
        paths = pd.concat(path_by_pipeline, axis=1, join="inner")
        common_ids = paths.index[paths.nunique(axis=1) > 1].astype(str).tolist()
    elif args.selection == "low-margin-disagreements":
        path_by_pipeline = []
        for df in frames.values():
            tmp = df[[args.id_col, args.path_col]].copy()
            tmp[args.id_col] = tmp[args.id_col].astype(str)
            path_by_pipeline.append(tmp.set_index(args.id_col)[args.path_col].fillna("").astype(str))
        paths = pd.concat(path_by_pipeline, axis=1, join="inner")
        common_ids = paths.index[paths.nunique(axis=1) > 1].astype(str).tolist()

    if args.sample_size and args.sample_size < len(common_ids):
        rng = np.random.default_rng(args.random_seed)
        common_ids = sorted(rng.choice(common_ids, size=args.sample_size, replace=False).astype(str).tolist())
    return common_ids


def build_long_frame(
    frames: dict[str, pd.DataFrame],
    *,
    selected_ids: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    selected = set(selected_ids)
    rows = []
    for label, df in frames.items():
        work = df[df[args.id_col].astype(str).isin(selected)].copy()
        rows.append(work)
    out = pd.concat(rows, ignore_index=True)
    out[args.id_col] = out[args.id_col].astype(str)
    return out


def build_candidate_texts(df: pd.DataFrame) -> pd.Series:
    path = df["predicted_taxonomy_path"].fillna("").astype(str)
    final_name = df["final_category_name"].fillna("").astype(str)
    enrichment = df["final_category_enrichment"].fillna("").astype(str)
    candidate = "Taxonomy path: " + path
    candidate += np.where(final_name != "", ". Final category: " + final_name, "")
    candidate += np.where(enrichment != "", ". Enriched category terms: " + enrichment, "")
    return pd.Series(candidate, index=df.index)


def compute_product_agreement(df: pd.DataFrame, *, args: argparse.Namespace) -> pd.DataFrame:
    grouped = df.groupby(args.id_col, sort=False)
    rows = []
    for product_id, group in grouped:
        paths = group["predicted_taxonomy_path"].fillna("").astype(str)
        row: dict[str, Any] = {
            args.id_col: product_id,
            "n_pipelines": int(group["pipeline"].nunique()),
            "n_unique_paths": int(paths.nunique()),
            "all_pipelines_same_path": bool(paths.nunique() == 1),
            "predicted_paths_json": json.dumps(
                dict(zip(group["pipeline"], paths, strict=False)), ensure_ascii=False
            ),
        }
        for level in range(2, 8):
            col = f"predicted_level_{level}_name"
            if col not in group.columns:
                continue
            values = group[col].fillna("").astype(str)
            non_empty = values[values != ""]
            row[f"n_unique_level_{level}"] = int(non_empty.nunique()) if len(non_empty) else 0
            row[f"all_pipelines_same_level_{level}"] = bool(non_empty.nunique() <= 1) if len(non_empty) else False
        rows.append(
            row
        )
    return pd.DataFrame(rows)


def compute_pairwise_agreement(df: pd.DataFrame, *, args: argparse.Namespace) -> pd.DataFrame:
    path_wide = df.pivot_table(
        index=args.id_col,
        columns="pipeline",
        values="predicted_taxonomy_path",
        aggfunc="first",
    )
    rows = []
    pipelines = list(path_wide.columns)
    for i, left in enumerate(pipelines):
        for right in pipelines[i + 1 :]:
            pair = path_wide[[left, right]].dropna()
            if pair.empty:
                continue
            row: dict[str, Any] = {
                "pipeline_left": left,
                "pipeline_right": right,
                "n_overlap": int(len(pair)),
                "exact_path_agreement": exact_agreement(pair[left], pair[right]),
                "path_cramers_v": cramers_v(pair[left], pair[right]),
            }
            for level in range(2, 8):
                level_col = f"predicted_level_{level}_name"
                if level_col not in df.columns:
                    continue
                level_wide = df.pivot_table(
                    index=args.id_col,
                    columns="pipeline",
                    values=level_col,
                    aggfunc="first",
                )
                if left not in level_wide.columns or right not in level_wide.columns:
                    continue
                level_pair = level_wide[[left, right]].fillna("").astype(str)
                non_empty = (level_pair[left] != "") | (level_pair[right] != "")
                level_pair = level_pair[non_empty]
                if level_pair.empty:
                    row[f"level_{level}_agreement"] = math.nan
                    row[f"level_{level}_cramers_v"] = math.nan
                    continue
                row[f"level_{level}_agreement"] = exact_agreement(level_pair[left], level_pair[right])
                row[f"level_{level}_cramers_v"] = cramers_v(level_pair[left], level_pair[right])
            rows.append(row)
    return pd.DataFrame(rows)


def exact_agreement(left: pd.Series, right: pd.Series) -> float:
    left_values = left.fillna("").astype(str)
    right_values = right.fillna("").astype(str)
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
    return float(math.sqrt(phi2 / max(1, min(table.shape[0] - 1, table.shape[1] - 1))))


def score_predictions(df: pd.DataFrame, *, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    if args.scorer == "cross-encoder":
        return score_with_cross_encoder(df, args=args)
    if args.scorer == "bi-encoder":
        return score_with_bi_encoder(df, args=args)
    raise ValueError(f"Unsupported scorer: {args.scorer}")


def score_with_cross_encoder(df: pd.DataFrame, *, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    from sentence_transformers import CrossEncoder

    model_name = resolve_local_model_path(args.scorer_model)
    started = time.perf_counter()
    model = CrossEncoder(model_name, device=args.device, trust_remote_code=True)
    load_seconds = time.perf_counter() - started
    pairs = list(zip(df[args.text_col].fillna("").astype(str), df["candidate_text"].astype(str), strict=False))
    score_started = time.perf_counter()
    scores = np.asarray(
        model.predict(pairs, batch_size=args.batch_size, show_progress_bar=True),
        dtype=float,
    ).reshape(-1)
    score_seconds = time.perf_counter() - score_started
    metrics = {
        "scorer": "cross-encoder",
        "scorer_model": str(model_name),
        "scorer_load_seconds": float(load_seconds),
        "scorer_score_seconds": float(score_seconds),
        "scorer_pairs_scored": int(len(pairs)),
        "scorer_pairs_per_second": float(len(pairs) / score_seconds) if score_seconds > 0 else math.nan,
    }
    return scores, metrics


def score_with_bi_encoder(df: pd.DataFrame, *, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    from sentence_transformers import SentenceTransformer

    model_name = resolve_local_model_path(args.scorer_model)
    started = time.perf_counter()
    model = SentenceTransformer(model_name, device=args.device, trust_remote_code=True)
    load_seconds = time.perf_counter() - started
    score_started = time.perf_counter()

    product_texts = df[args.text_col].fillna("").astype(str).tolist()
    candidate_texts = df["candidate_text"].fillna("").astype(str).tolist()
    product_embeddings = model.encode(
        product_texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    candidate_embeddings = model.encode(
        candidate_texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    scores = np.sum(np.asarray(product_embeddings) * np.asarray(candidate_embeddings), axis=1)
    score_seconds = time.perf_counter() - score_started
    metrics = {
        "scorer": "bi-encoder",
        "scorer_model": str(model_name),
        "scorer_load_seconds": float(load_seconds),
        "scorer_score_seconds": float(score_seconds),
        "scorer_pairs_scored": int(len(df)),
        "scorer_pairs_per_second": float(len(df) / score_seconds) if score_seconds > 0 else math.nan,
    }
    return scores.astype(float), metrics


def resolve_local_model_path(model_name: str) -> str:
    """Use a cached Hugging Face snapshot when available to avoid network calls."""
    path = Path(model_name).expanduser()
    if path.exists():
        return str(path)

    cache_root = Path.home() / ".cache/huggingface/hub"
    repo_dir = cache_root / ("models--" + model_name.replace("/", "--"))
    snapshots_dir = repo_dir / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted([p for p in snapshots_dir.iterdir() if p.is_dir()])
        if snapshots:
            return str(snapshots[-1])
    return model_name


def rank_pipelines_per_product(df: pd.DataFrame, *, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for product_id, group in df.groupby(args.id_col, sort=False):
        ranked = group.sort_values("nlp_score", ascending=False).reset_index(drop=True)
        top_score = float(ranked.loc[0, "nlp_score"])
        second_score = float(ranked.loc[1, "nlp_score"]) if len(ranked) > 1 else math.nan
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            rows.append(
                {
                    args.id_col: product_id,
                    "pipeline": row["pipeline"],
                    "rank": rank,
                    "nlp_score": float(row["nlp_score"]),
                    "winner": bool(rank == 1),
                    "winner_margin": float(top_score - second_score) if rank == 1 and not math.isnan(second_score) else math.nan,
                    "predicted_level_1_name": row.get("predicted_level_1_name", ""),
                    "predicted_taxonomy_path": row.get("predicted_taxonomy_path", ""),
                }
            )
    return pd.DataFrame(rows)


def summarize_scores(
    df: pd.DataFrame,
    ranking: pd.DataFrame,
    pairwise_agreement: pd.DataFrame,
    score_metrics: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    rank_stats = ranking.groupby("pipeline").agg(
        win_rate=("winner", "mean"),
        mean_rank=("rank", "mean"),
        median_rank=("rank", "median"),
    )
    for pipeline, group in df.groupby("pipeline", sort=False):
        rows.append(
            {
                "pipeline": pipeline,
                "n_products": int(group.shape[0]),
                "mean_nlp_score": float(group["nlp_score"].mean()),
                "median_nlp_score": float(group["nlp_score"].median()),
                "std_nlp_score": float(group["nlp_score"].std()),
                "mean_resolved_depth": float(pd.to_numeric(group.get("resolved_depth"), errors="coerce").mean()),
                "n_unique_paths": int(group["predicted_taxonomy_path"].fillna("").astype(str).nunique()),
                "win_rate": float(rank_stats.loc[pipeline, "win_rate"]) if pipeline in rank_stats.index else math.nan,
                "mean_rank": float(rank_stats.loc[pipeline, "mean_rank"]) if pipeline in rank_stats.index else math.nan,
                "median_rank": float(rank_stats.loc[pipeline, "median_rank"]) if pipeline in rank_stats.index else math.nan,
                **score_metrics,
            }
        )
    if not pairwise_agreement.empty:
        global_agreement = {
            "mean_pairwise_exact_path_agreement": float(pairwise_agreement["exact_path_agreement"].mean())
        }
        for row in rows:
            row.update(global_agreement)
    return pd.DataFrame(rows)


def summarize_without_scores(df: pd.DataFrame, pairwise_agreement: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pipeline, group in df.groupby("pipeline", sort=False):
        rows.append(
            {
                "pipeline": pipeline,
                "n_products": int(group.shape[0]),
                "mean_resolved_depth": float(pd.to_numeric(group.get("resolved_depth"), errors="coerce").mean()),
                "n_unique_paths": int(group["predicted_taxonomy_path"].fillna("").astype(str).nunique()),
                "mean_pairwise_exact_path_agreement": float(pairwise_agreement["exact_path_agreement"].mean())
                if not pairwise_agreement.empty
                else math.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_by_l1(df: pd.DataFrame, ranking: pd.DataFrame, *, min_products: int) -> pd.DataFrame:
    rows = []
    merged = df.merge(
        ranking[[df.columns[0], "pipeline", "rank", "winner"]],
        on=[df.columns[0], "pipeline"],
        how="left",
    )
    for (l1, pipeline), group in merged.groupby(["predicted_level_1_name", "pipeline"], dropna=False):
        if len(group) < min_products:
            continue
        rows.append(
            {
                "predicted_level_1_name": l1,
                "pipeline": pipeline,
                "n_products": int(len(group)),
                "mean_nlp_score": float(group["nlp_score"].mean()),
                "median_nlp_score": float(group["nlp_score"].median()),
                "win_rate": float(group["winner"].mean()),
                "mean_rank": float(group["rank"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["predicted_level_1_name", "mean_nlp_score"], ascending=[True, False]
    )


def compute_pairwise_score_preferences(df: pd.DataFrame, *, args: argparse.Namespace) -> pd.DataFrame:
    scores = df.pivot_table(
        index=args.id_col,
        columns="pipeline",
        values="nlp_score",
        aggfunc="first",
    )
    rows = []
    pipelines = list(scores.columns)
    for i, left in enumerate(pipelines):
        for right in pipelines[i + 1 :]:
            pair = scores[[left, right]].dropna()
            if pair.empty:
                continue
            diff = pair[left] - pair[right]
            rows.append(
                {
                    "pipeline_left": left,
                    "pipeline_right": right,
                    "n_overlap": int(len(pair)),
                    "left_mean_score": float(pair[left].mean()),
                    "right_mean_score": float(pair[right].mean()),
                    "mean_score_diff_left_minus_right": float(diff.mean()),
                    "left_preferred_share": float((diff > 0).mean()),
                    "right_preferred_share": float((diff < 0).mean()),
                    "tie_share": float((diff == 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def write_pairwise_matrices(pairwise: pd.DataFrame, *, output_dir: Path) -> None:
    if pairwise.empty:
        return
    metrics = ["exact_path_agreement", "path_cramers_v"]
    metrics.extend([f"level_{level}_agreement" for level in range(2, 8)])
    metrics.extend([f"level_{level}_cramers_v" for level in range(2, 8)])
    for metric in metrics:
        if metric not in pairwise.columns:
            continue
        matrix = pairwise_metric_matrix(pairwise, metric)
        matrix.to_csv(output_dir / f"l2_l7_{metric}_matrix.csv")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["qwen", "jasper", "none"], default="qwen")
    parser.add_argument(
        "--pipeline",
        action="append",
        help="Pipeline prediction file as label=path. Can be repeated. Overrides or extends presets.",
    )
    parser.add_argument("--output-dir", default=str(ROOT / "latex/rapport_final/figures/l2_l7_reranking_qwen"))
    parser.add_argument(
        "--category-prototypes-path",
        default=str(ROOT / "dataset/shared_l1_l7_embeddings_Qwen3-Embedding-0.6B/category_prototypes.parquet"),
    )
    parser.add_argument("--id-col", default="hashed_external_id")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--path-col", default="predicted_taxonomy_path")
    parser.add_argument("--key-path-col", default="predicted_taxonomy_key_path")
    parser.add_argument("--l1-col", default="predicted_level_1_name")
    parser.add_argument("--score-col", default="predicted_path_score")
    parser.add_argument("--depth-col", default="resolved_depth")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--selection", choices=["random", "disagreements", "low-margin-disagreements"], default="disagreements")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--skip-scorer", action="store_true")
    parser.add_argument("--skip-cross-encoder", action="store_true", help="Deprecated alias for --skip-scorer.")
    parser.add_argument("--scorer", choices=["bi-encoder", "cross-encoder"], default="bi-encoder")
    parser.add_argument("--scorer-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--cross-encoder-model", default=None, help="Deprecated alias for --scorer-model.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-prototypes-per-category", type=int, default=6)
    parser.add_argument("--min-products-per-l1", type=int, default=25)
    args = parser.parse_args()
    if args.skip_cross_encoder:
        args.skip_scorer = True
    if args.cross_encoder_model:
        args.scorer = "cross-encoder"
        args.scorer_model = args.cross_encoder_model
    return args


if __name__ == "__main__":
    main()
