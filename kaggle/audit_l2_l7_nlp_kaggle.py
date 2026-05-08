#!/usr/bin/env python3
"""Run NLP audits for unsupervised L2-L7 predictions on Kaggle.

The script scores, for each product and each pipeline prediction, the pair:
    (product text, predicted enriched taxonomy path)

It supports:
- bi-encoder scoring with cosine similarity;
- cross-encoder scoring with direct pair scoring;
- scoring all common products by default, not only a sample;
- optional restriction to products where pipelines disagree;
- summary tables, per-product rankings, pairwise preferences and runtime metrics.

Typical Kaggle usage:
python /kaggle/input/<dataset>/audit_l2_l7_nlp_kaggle.py \
  --predictions-dir /kaggle/input/criteo-l2-l7-qwen \
  --category-prototypes-path /kaggle/input/shared-embeddings/category_prototypes.parquet \
  --scorer both \
  --device cuda \
  --output-dir /kaggle/working/l2_l7_nlp_audit_qwen
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


QWEN_FILES = {
    "greedy": "pipeline1_hierarchical_predictions_greedy.parquet",
    "beam": "pipeline1_hierarchical_predictions_beam.parquet",
    "beam_reranker": "pipeline1_hierarchical_predictions_beam-rerank.parquet",
    "clustering_l1_hungarian": "pipeline2_hierarchical_clustering_predictions_l1routed_hungarian.parquet",
    "global_path": "pipeline3_global_path_predictions.parquet",
}

JASPER_FILES = {
    "greedy": "pipeline1_hierarchical_predictions_jasper_greedy.parquet",
    "beam": "pipeline1_hierarchical_predictions_jasper_beam.parquet",
    "beam_reranker": "pipeline1_hierarchical_predictions_beam-rerank_jasper.parquet",
    "clustering_l1_hungarian": "pipeline2_hierarchical_clustering_predictions_jasper_l1routed_hungarian.parquet",
    "global_path": "pipeline3_global_path_predictions_jasper.parquet",
}


def main() -> None:
    args = parse_args()
    configure_cache(args.hf_cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    pipeline_paths = resolve_pipeline_paths(args)
    if len(pipeline_paths) < 2:
        raise SystemExit("At least two prediction files are required. Use --pipeline label=/path/file.csv.")

    print("Pipeline files:")
    for label, path in pipeline_paths.items():
        print(f"  - {label}: {path}")

    prototypes = load_category_prototypes(args.category_prototypes_path, args.max_prototypes_per_category)
    frames = {
        label: load_prediction_frame(path, label=label, args=args, prototypes=prototypes)
        for label, path in pipeline_paths.items()
    }
    selected_ids = select_product_ids(frames, args=args)
    if args.max_products and len(selected_ids) > args.max_products:
        rng = np.random.default_rng(args.random_seed)
        selected_ids = sorted(rng.choice(selected_ids, size=args.max_products, replace=False).astype(str).tolist())

    long_df = build_long_frame(frames, selected_ids=selected_ids, args=args)
    long_df["candidate_text"] = build_candidate_texts(long_df, args=args)
    long_df.to_csv(output_dir / "audit_input_long.csv", index=False)

    print(f"Products selected: {len(selected_ids):,}")
    print(f"Pairs to score: {len(long_df):,}")

    scorers = ["bi-encoder", "cross-encoder"] if args.scorer == "both" else [args.scorer]
    run_metrics: dict[str, Any] = {
        "args": vars(args),
        "pipeline_paths": {k: str(v) for k, v in pipeline_paths.items()},
        "n_products_selected": int(len(selected_ids)),
        "n_pairs_total": int(len(long_df)),
        "scorers": {},
    }

    for scorer in scorers:
        print(f"\n=== Running {scorer} audit ===")
        reset_cuda_peak_memory()
        scorer_started = time.perf_counter()
        if scorer == "bi-encoder":
            scores, scorer_metrics = score_with_bi_encoder(long_df, args=args)
        elif scorer == "cross-encoder":
            scores, scorer_metrics = score_with_cross_encoder(long_df, args=args)
        else:
            raise ValueError(f"Unsupported scorer: {scorer}")
        scorer_metrics["total_scorer_wall_seconds"] = float(time.perf_counter() - scorer_started)
        scorer_metrics.update(cuda_memory_metrics())
        run_metrics["scorers"][scorer] = scorer_metrics

        scored = long_df.copy()
        scored["nlp_score"] = np.asarray(scores, dtype=float)
        prefix = scorer.replace("-", "_")
        export_scored_outputs(scored, prefix=prefix, output_dir=output_dir, args=args)
        del scored, scores
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run_metrics["elapsed_seconds"] = float(time.perf_counter() - started)
    (output_dir / "audit_nlp_metrics.json").write_text(
        json.dumps(run_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved NLP audit outputs to {output_dir}")
    print(json.dumps(run_metrics, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["qwen", "jasper", "none"], default="qwen")
    parser.add_argument(
        "--predictions-dir",
        default="/kaggle/input/criteo-l2-l7-predictions",
        help="Directory containing prediction Parquet or CSV files with the expected preset filenames.",
    )
    parser.add_argument(
        "--pipeline",
        action="append",
        help="Custom prediction file as label=/path/file.parquet or label=/path/file.csv. Can be repeated and overrides preset labels.",
    )
    parser.add_argument("--output-dir", default="/kaggle/working/l2_l7_nlp_audit")
    parser.add_argument("--category-prototypes-path", default="")
    parser.add_argument("--max-prototypes-per-category", type=int, default=6)
    parser.add_argument("--id-col", default="hashed_external_id")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--path-col", default="predicted_taxonomy_path")
    parser.add_argument("--key-path-col", default="predicted_taxonomy_key_path")
    parser.add_argument("--depth-col", default="resolved_depth")
    parser.add_argument("--l1-col", default="predicted_level_1_name")
    parser.add_argument("--selection", choices=["all", "disagreements"], default="all")
    parser.add_argument(
        "--max-products",
        type=int,
        default=0,
        help="Optional random cap for debugging. 0 means all selected products.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--scorer", choices=["bi-encoder", "cross-encoder", "both"], default="both")
    parser.add_argument("--bi-encoder-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--device", default=None, help="cuda, cuda:0, cpu, or empty for auto.")
    parser.add_argument("--bi-batch-size", type=int, default=256)
    parser.add_argument("--cross-batch-size", type=int, default=64)
    parser.add_argument("--cross-chunk-size", type=int, default=20000)
    parser.add_argument("--hf-cache-dir", default="/kaggle/working/huggingface_cache")
    parser.add_argument("--min-products-per-l1", type=int, default=25)
    parser.add_argument("--no-long-export", action="store_true", help="Do not export full long score CSVs.")
    return parser.parse_args()


def configure_cache(cache_dir: str) -> None:
    if not cache_dir:
        return
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", str(Path(cache_dir) / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(cache_dir) / "transformers"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(Path(cache_dir) / "sentence_transformers"))


def resolve_pipeline_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    pred_dir = Path(args.predictions_dir)
    if args.preset == "qwen":
        paths.update({label: pred_dir / filename for label, filename in QWEN_FILES.items()})
    elif args.preset == "jasper":
        paths.update({label: pred_dir / filename for label, filename in JASPER_FILES.items()})

    for item in args.pipeline or []:
        if "=" not in item:
            path = Path(item)
            paths[path.stem] = path
        else:
            label, path_str = item.split("=", 1)
            paths[label.strip()] = Path(path_str.strip())

    resolved: dict[str, Path] = {}
    for label, path in paths.items():
        path = path.expanduser()
        path = resolve_prediction_file(path)
        if path is not None:
            resolved[label] = path
        else:
            print(f"WARNING missing prediction file for {label}: {paths[label]}")
    return resolved


def resolve_prediction_file(path: Path) -> Path | None:
    if path.exists():
        return path
    alternatives = []
    if path.suffix == ".parquet":
        alternatives.append(path.with_suffix(".csv"))
    elif path.suffix == ".csv":
        alternatives.append(path.with_suffix(".parquet"))
    else:
        alternatives.extend([path.with_suffix(".parquet"), path.with_suffix(".csv")])
    for candidate in alternatives:
        if candidate.exists():
            return candidate
    return None


def load_category_prototypes(path_str: str, max_prototypes: int) -> dict[str, str]:
    if not path_str:
        print("No category prototype file provided; candidate text will use only predicted paths.")
        return {}
    path = Path(path_str).expanduser()
    if not path.exists():
        print(f"WARNING missing category prototype file: {path}")
        return {}
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if not {"node_key", "prototype_text"}.issubset(df.columns):
        print(f"WARNING {path} must contain node_key and prototype_text columns.")
        return {}
    out: dict[str, str] = {}
    for node_key, group in df.dropna(subset=["node_key", "prototype_text"]).groupby("node_key", sort=False):
        texts = group["prototype_text"].astype(str).drop_duplicates().head(max_prototypes).tolist()
        out[str(node_key)] = "; ".join(texts)
    print(f"Loaded prototype enrichments for {len(out):,} category nodes.")
    return out


def load_prediction_frame(path: Path, *, label: str, args: argparse.Namespace, prototypes: dict[str, str]) -> pd.DataFrame:
    columns = read_columns(path)
    usecols = columns_to_read(args).intersection(columns)
    missing = {args.id_col, args.text_col, args.path_col}.difference(usecols)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df = read_prediction_table(path, sorted(usecols))
    df = df.drop_duplicates(args.id_col).copy()
    df[args.id_col] = df[args.id_col].astype(str)
    df["pipeline"] = label
    df["source_file"] = str(path)
    df["final_category_key"] = deepest_value(df, prefix="predicted_level_", suffix="_key")
    df["final_category_name"] = deepest_value(df, prefix="predicted_level_", suffix="_name")
    df["final_category_enrichment"] = df["final_category_key"].map(prototypes).fillna("") if prototypes else ""
    print(f"Loaded {label}: {len(df):,} rows from {path.name}")
    return df


def read_columns(path: Path) -> set[str]:
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            return set(pq.ParquetFile(path).schema_arrow.names)
        except Exception:
            return set(pd.read_parquet(path).columns)
    return set(pd.read_csv(path, nrows=0).columns)


def read_prediction_table(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns, low_memory=False)


def columns_to_read(args: argparse.Namespace) -> set[str]:
    cols = {
        args.id_col,
        args.text_col,
        args.path_col,
        args.key_path_col,
        args.depth_col,
        args.l1_col,
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


def select_product_ids(frames: dict[str, pd.DataFrame], *, args: argparse.Namespace) -> list[str]:
    id_sets = [set(df[args.id_col].astype(str)) for df in frames.values()]
    common_ids = sorted(set.intersection(*id_sets))
    if not common_ids:
        raise ValueError("No common product id across pipeline outputs.")

    if args.selection == "disagreements":
        path_cols = []
        for label, df in frames.items():
            tmp = df[[args.id_col, args.path_col]].copy()
            tmp[args.path_col] = tmp[args.path_col].fillna("").astype(str)
            path_cols.append(tmp.set_index(args.id_col)[args.path_col].rename(label))
        wide = pd.concat(path_cols, axis=1, join="inner")
        common_ids = wide.index[wide.nunique(axis=1) > 1].astype(str).tolist()
    return common_ids


def build_long_frame(frames: dict[str, pd.DataFrame], *, selected_ids: list[str], args: argparse.Namespace) -> pd.DataFrame:
    selected = set(selected_ids)
    rows = []
    for label, df in frames.items():
        part = df[df[args.id_col].isin(selected)].copy()
        rows.append(part)
    long_df = pd.concat(rows, axis=0, ignore_index=True)
    long_df[args.path_col] = long_df[args.path_col].fillna("").astype(str)
    long_df[args.text_col] = long_df[args.text_col].fillna("").astype(str)
    return long_df


def build_candidate_texts(df: pd.DataFrame, *, args: argparse.Namespace) -> pd.Series:
    path = df[args.path_col].fillna("").astype(str)
    final_name = df.get("final_category_name", pd.Series("", index=df.index)).fillna("").astype(str)
    enrichment = df.get("final_category_enrichment", pd.Series("", index=df.index)).fillna("").astype(str)
    return (
        "Taxonomy path: " + path
        + ". Final category: " + final_name
        + np.where(enrichment != "", ". Enriched semantic prototypes: " + enrichment, "")
    )


def resolve_device(device: str | None) -> str | None:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_local_model_path(model_name: str) -> str:
    path = Path(model_name).expanduser()
    if path.exists():
        return str(path)
    cache_roots = [
        Path(os.environ.get("HF_HUB_CACHE", "")),
        Path(os.environ.get("HF_HOME", "")) / "hub",
        Path.home() / ".cache/huggingface/hub",
    ]
    for cache_root in cache_roots:
        if not str(cache_root):
            continue
        repo_dir = cache_root / ("models--" + model_name.replace("/", "--"))
        snapshots_dir = repo_dir / "snapshots"
        if snapshots_dir.exists():
            snapshots = sorted([p for p in snapshots_dir.iterdir() if p.is_dir()])
            if snapshots:
                return str(snapshots[-1])
    return model_name


def score_with_bi_encoder(df: pd.DataFrame, *, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    from sentence_transformers import SentenceTransformer

    device = resolve_device(args.device)
    model_name = resolve_local_model_path(args.bi_encoder_model)
    print(f"Loading bi-encoder on {device}: {model_name}")
    load_started = time.perf_counter()
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    load_seconds = time.perf_counter() - load_started

    products = df[[args.id_col, args.text_col]].drop_duplicates(args.id_col).reset_index(drop=True)
    candidates = df[["candidate_text"]].drop_duplicates("candidate_text").reset_index(drop=True)
    product_to_idx = {pid: idx for idx, pid in enumerate(products[args.id_col].astype(str))}
    candidate_to_idx = {txt: idx for idx, txt in enumerate(candidates["candidate_text"].astype(str))}

    score_started = time.perf_counter()
    product_embeddings = model.encode(
        products[args.text_col].fillna("").astype(str).tolist(),
        batch_size=args.bi_batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    candidate_embeddings = model.encode(
        candidates["candidate_text"].fillna("").astype(str).tolist(),
        batch_size=args.bi_batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    product_indices = df[args.id_col].astype(str).map(product_to_idx).to_numpy(dtype=np.int64)
    candidate_indices = df["candidate_text"].astype(str).map(candidate_to_idx).to_numpy(dtype=np.int64)
    scores = np.sum(product_embeddings[product_indices] * candidate_embeddings[candidate_indices], axis=1)
    score_seconds = time.perf_counter() - score_started

    metrics = {
        "scorer": "bi-encoder",
        "model": str(model_name),
        "device": str(device),
        "load_seconds": float(load_seconds),
        "score_seconds": float(score_seconds),
        "n_pairs_scored": int(len(df)),
        "n_unique_products_encoded": int(len(products)),
        "n_unique_candidate_texts_encoded": int(len(candidates)),
        "pairs_per_second": float(len(df) / score_seconds) if score_seconds > 0 else math.nan,
        "product_embedding_shape": list(product_embeddings.shape),
        "candidate_embedding_shape": list(candidate_embeddings.shape),
    }
    del model, product_embeddings, candidate_embeddings
    return scores.astype(float), metrics


def score_with_cross_encoder(df: pd.DataFrame, *, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    from sentence_transformers import CrossEncoder

    device = resolve_device(args.device)
    model_name = resolve_local_model_path(args.cross_encoder_model)
    print(f"Loading cross-encoder on {device}: {model_name}")
    load_started = time.perf_counter()
    model = CrossEncoder(model_name, device=device, trust_remote_code=True)
    load_seconds = time.perf_counter() - load_started

    product_texts = df[args.text_col].fillna("").astype(str).tolist()
    candidate_texts = df["candidate_text"].fillna("").astype(str).tolist()
    scores = np.empty(len(df), dtype=np.float32)
    score_started = time.perf_counter()
    for start in range(0, len(df), args.cross_chunk_size):
        end = min(start + args.cross_chunk_size, len(df))
        pairs = list(zip(product_texts[start:end], candidate_texts[start:end], strict=False))
        chunk_scores = model.predict(
            pairs,
            batch_size=args.cross_batch_size,
            show_progress_bar=True,
        )
        scores[start:end] = np.asarray(chunk_scores, dtype=np.float32).reshape(-1)
        print(f"Cross-encoder scored {end:,}/{len(df):,} pairs")
    score_seconds = time.perf_counter() - score_started

    metrics = {
        "scorer": "cross-encoder",
        "model": str(model_name),
        "device": str(device),
        "load_seconds": float(load_seconds),
        "score_seconds": float(score_seconds),
        "n_pairs_scored": int(len(df)),
        "pairs_per_second": float(len(df) / score_seconds) if score_seconds > 0 else math.nan,
        "cross_batch_size": int(args.cross_batch_size),
        "cross_chunk_size": int(args.cross_chunk_size),
    }
    del model
    return scores.astype(float), metrics


def export_scored_outputs(scored: pd.DataFrame, *, prefix: str, output_dir: Path, args: argparse.Namespace) -> None:
    ranking = rank_pipelines_per_product(scored, args=args)
    summary = summarize_scores(scored, ranking, args=args)
    by_l1 = summarize_by_l1(scored, ranking, args=args)
    pairwise = compute_pairwise_score_preferences(scored, args=args)
    product_winners = summarize_product_winners(scored, ranking, args=args)

    if not args.no_long_export:
        export_cols = [
            args.id_col,
            "pipeline",
            args.text_col,
            args.path_col,
            "candidate_text",
            args.l1_col,
            args.depth_col,
            "nlp_score",
        ]
        export_cols = [c for c in export_cols if c in scored.columns]
        scored[export_cols].to_csv(output_dir / f"{prefix}_long_scores.csv", index=False)
    ranking.to_csv(output_dir / f"{prefix}_product_ranks.csv", index=False)
    summary.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
    by_l1.to_csv(output_dir / f"{prefix}_by_l1.csv", index=False)
    pairwise.to_csv(output_dir / f"{prefix}_pairwise_score_preferences.csv", index=False)
    product_winners.to_csv(output_dir / f"{prefix}_product_winners.csv", index=False)
    print(f"\n{prefix} summary:")
    print(summary.to_string(index=False))


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
                    args.l1_col: row.get(args.l1_col, ""),
                    args.path_col: row.get(args.path_col, ""),
                }
            )
    return pd.DataFrame(rows)


def summarize_scores(df: pd.DataFrame, ranking: pd.DataFrame, *, args: argparse.Namespace) -> pd.DataFrame:
    rank_stats = ranking.groupby("pipeline").agg(
        win_rate=("winner", "mean"),
        mean_rank=("rank", "mean"),
        median_rank=("rank", "median"),
    )
    rows = []
    for pipeline, group in df.groupby("pipeline", sort=False):
        rows.append(
            {
                "pipeline": pipeline,
                "n_products": int(len(group)),
                "mean_nlp_score": float(group["nlp_score"].mean()),
                "median_nlp_score": float(group["nlp_score"].median()),
                "std_nlp_score": float(group["nlp_score"].std()),
                "win_rate": float(rank_stats.loc[pipeline, "win_rate"]),
                "mean_rank": float(rank_stats.loc[pipeline, "mean_rank"]),
                "median_rank": float(rank_stats.loc[pipeline, "median_rank"]),
                "mean_resolved_depth": float(pd.to_numeric(group.get(args.depth_col), errors="coerce").mean()) if args.depth_col in group else math.nan,
                "n_unique_paths": int(group[args.path_col].fillna("").astype(str).nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_nlp_score", ascending=False)


def summarize_by_l1(df: pd.DataFrame, ranking: pd.DataFrame, *, args: argparse.Namespace) -> pd.DataFrame:
    if args.l1_col not in df.columns:
        return pd.DataFrame()
    merged = df.merge(
        ranking[[args.id_col, "pipeline", "rank", "winner"]],
        on=[args.id_col, "pipeline"],
        how="left",
    )
    rows = []
    for (l1, pipeline), group in merged.groupby([args.l1_col, "pipeline"], dropna=False):
        if len(group) < args.min_products_per_l1:
            continue
        rows.append(
            {
                args.l1_col: l1,
                "pipeline": pipeline,
                "n_products": int(len(group)),
                "mean_nlp_score": float(group["nlp_score"].mean()),
                "median_nlp_score": float(group["nlp_score"].median()),
                "win_rate": float(group["winner"].mean()),
                "mean_rank": float(group["rank"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values([args.l1_col, "mean_nlp_score"], ascending=[True, False]) if rows else pd.DataFrame()


def compute_pairwise_score_preferences(df: pd.DataFrame, *, args: argparse.Namespace) -> pd.DataFrame:
    scores = df.pivot_table(index=args.id_col, columns="pipeline", values="nlp_score", aggfunc="first")
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


def summarize_product_winners(df: pd.DataFrame, ranking: pd.DataFrame, *, args: argparse.Namespace) -> pd.DataFrame:
    winners = ranking[ranking["rank"] == 1][[args.id_col, "pipeline", "nlp_score", "winner_margin"]].copy()
    winners = winners.rename(columns={"pipeline": "winner_pipeline", "nlp_score": "winner_score"})
    paths = df.pivot_table(index=args.id_col, columns="pipeline", values=args.path_col, aggfunc="first")
    scores = df.pivot_table(index=args.id_col, columns="pipeline", values="nlp_score", aggfunc="first")
    out = winners.merge(paths.reset_index(), on=args.id_col, how="left")
    score_cols = scores.add_prefix("score_").reset_index()
    out = out.merge(score_cols, on=args.id_col, how="left")
    return out


def reset_cuda_peak_memory() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def cuda_memory_metrics() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    device = torch.cuda.current_device()
    return {
        "cuda_available": True,
        "cuda_device": torch.cuda.get_device_name(device),
        "cuda_peak_allocated_mb": float(torch.cuda.max_memory_allocated(device) / (1024**2)),
        "cuda_peak_reserved_mb": float(torch.cuda.max_memory_reserved(device) / (1024**2)),
    }


if __name__ == "__main__":
    main()
