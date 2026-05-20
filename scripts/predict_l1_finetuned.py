#!/usr/bin/env python3
"""Predict Level 1 categories with the retained fine-tuned BGE-small model."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = REPO_ROOT / "dataset" / "preprocessed_lv2.parquet"
DEFAULT_ENRICHMENT_DIR = REPO_ROOT / "dataset" / "category_enrichment"
DEFAULT_L1_RUN_DIR = (
    REPO_ROOT
    / "models"
    / "embedding_runs"
    / "20260503_131422__enriched-triplet-finetune__model-baai-bge-small-en-v1-5__label-level-1-name__bs-128__sampler-tempered__ep-4__lr-2e-04__seq-256__seed-42"
)
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "l1_finetuned_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--input-format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--enrichment-dir", type=Path, default=DEFAULT_ENRICHMENT_DIR)
    parser.add_argument("--l1-run-dir", type=Path, default=DEFAULT_L1_RUN_DIR)
    parser.add_argument("--l1-model-subdir", default="best_model")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--id-col", default="hashed_external_id")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="level_1_name")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--score-aggregation", choices=["max", "mean"], default="max")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
    reset_accelerator_peak_memory()

    started = time.perf_counter()
    products = load_products(args)
    prototypes = load_l1_prototypes(args.enrichment_dir)
    predictions, metrics = predict_l1(products, prototypes, args)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_path, index=False)
    metrics_path = args.output_path.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "args": serializable_args(args),
                "metrics": metrics | {"elapsed_seconds_wall": time.perf_counter() - started},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Predictions written to {args.output_path}")
    print(f"Metrics written to {metrics_path}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def load_products(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_parquet(args.input_path) if args.input_format == "parquet" else pd.read_csv(args.input_path)
    required = {args.id_col, args.text_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    keep = [args.id_col, args.text_col]
    if args.label_col in df.columns:
        keep.append(args.label_col)
    products = df[keep].copy()
    products[args.text_col] = products[args.text_col].fillna("").astype(str)
    products = products[products[args.text_col].str.len() > 0].reset_index(drop=True)
    if args.sample_size is not None and len(products) > args.sample_size:
        products = (
            products.sample(args.sample_size, random_state=args.random_seed)
            .sort_index()
            .reset_index(drop=True)
        )
    return products


def load_l1_prototypes(enrichment_dir: Path) -> pd.DataFrame:
    path = enrichment_dir / "level_1_reference_texts.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Level 1 reference texts: {path}")
    df = pd.read_csv(path)
    required = {"node_key", "category_name", "prototype_type", "prototype_text"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df[df["prototype_type"].fillna("") != "descendant_names_text"].copy()
    df["prototype_text"] = df["prototype_text"].fillna("").astype(str)
    df = df[df["prototype_text"].str.len() > 0]
    return df.drop_duplicates(
        subset=["node_key", "category_name", "prototype_type", "prototype_text"]
    ).reset_index(drop=True)


def predict_l1(
    products: pd.DataFrame, prototypes: pd.DataFrame, args: argparse.Namespace
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from sentence_transformers import SentenceTransformer

    base_path = Path(args.l1_run_dir)
    candidate = base_path / args.l1_model_subdir if args.l1_model_subdir else base_path
    model_path = candidate if candidate.exists() else base_path
    if not model_path.exists():
        raise FileNotFoundError(f"L1 model directory not found: {model_path}")

    load_started = time.perf_counter()
    model = SentenceTransformer(str(model_path), device=args.device)
    model_load_seconds = time.perf_counter() - load_started

    proto_started = time.perf_counter()
    prototype_embeddings = encode_texts(
        model,
        prototypes["prototype_text"].tolist(),
        batch_size=args.batch_size,
        device=args.device,
    )
    prototype_encode_seconds = time.perf_counter() - proto_started

    product_started = time.perf_counter()
    product_embeddings = encode_texts(
        model,
        products[args.text_col].tolist(),
        batch_size=args.batch_size,
        device=args.device,
    )
    product_encode_seconds = time.perf_counter() - product_started

    node_to_indices = (
        prototypes.reset_index().groupby("node_key")["index"].apply(list).to_dict()
    )
    node_names = prototypes.groupby("node_key")["category_name"].first().to_dict()
    node_keys = sorted(node_to_indices)

    score_started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    n_dot_products = 0
    for row, embedding in zip(products.to_dict("records"), product_embeddings, strict=True):
        scores = []
        for node_key in node_keys:
            indices = np.asarray(node_to_indices[node_key], dtype=int)
            sims = prototype_embeddings[indices] @ embedding
            n_dot_products += len(indices)
            scores.append(
                {
                    "node_key": str(node_key),
                    "node_name": str(node_names.get(node_key, node_key)),
                    "score": aggregate(sims, args.score_aggregation),
                    "n_prototypes": int(len(indices)),
                }
            )
        scores.sort(key=lambda item: float(item["score"]), reverse=True)
        add_rank_and_margins(scores)
        top_candidates = scores[: max(1, args.top_k)]
        top1 = top_candidates[0]
        top2 = top_candidates[1] if len(top_candidates) > 1 else {}
        rows.append(
            {
                args.id_col: row[args.id_col],
                "text": row[args.text_col],
                "predicted_level_1_key": str(top1.get("node_key", "")),
                "predicted_level_1_name": str(top1.get("node_name", "")),
                "top1_score": float(top1.get("score", np.nan)),
                "top2_score": float(top2.get("score", np.nan)),
                "top1_top2_margin": float(top1.get("score", np.nan) - top2.get("score", np.nan))
                if top2
                else np.nan,
                "top_candidates_json": json.dumps(top_candidates, ensure_ascii=False),
            }
        )
    score_seconds = time.perf_counter() - score_started
    out = pd.DataFrame(rows)

    metrics: dict[str, Any] = {
        "model_path": str(model_path),
        "n_products": int(len(products)),
        "n_categories": int(len(node_keys)),
        "n_prototypes": int(len(prototypes)),
        "embedding_dim": int(product_embeddings.shape[1]) if len(product_embeddings) else 0,
        "model_load_seconds": float(model_load_seconds),
        "prototype_encode_seconds": float(prototype_encode_seconds),
        "product_encode_seconds": float(product_encode_seconds),
        "score_seconds": float(score_seconds),
        "total_seconds": float(
            model_load_seconds + prototype_encode_seconds + product_encode_seconds + score_seconds
        ),
        "milliseconds_per_product": float(
            1000.0
            * (model_load_seconds + prototype_encode_seconds + product_encode_seconds + score_seconds)
            / max(1, len(products))
        ),
        "throughput_products_per_second": float(
            len(products)
            / max(
                1e-9,
                model_load_seconds + prototype_encode_seconds + product_encode_seconds + score_seconds,
            )
        ),
        "similarity_dot_products": int(n_dot_products),
        "similarity_dot_products_per_product": float(n_dot_products / max(1, len(products))),
        "approx_million_scalar_multiply_adds": float(
            n_dot_products * (int(product_embeddings.shape[1]) if len(product_embeddings) else 0) / 1_000_000
        ),
        "median_margin": float(pd.to_numeric(out["top1_top2_margin"], errors="coerce").median()),
        "mean_margin": float(pd.to_numeric(out["top1_top2_margin"], errors="coerce").mean()),
        "share_margin_lt_0_03": float(
            (pd.to_numeric(out["top1_top2_margin"], errors="coerce") < 0.03).mean()
        ),
        "resource_metrics_end": process_resource_metrics(),
        **accelerator_memory_metrics(),
    }
    if args.label_col in products.columns:
        labels = products[args.label_col].astype(str)
        preds = out["predicted_level_1_key"].astype(str)
        metrics["accuracy_if_label_available"] = float((labels == preds).mean())
    return out, metrics


def encode_texts(model: Any, texts: list[str], *, batch_size: int, device: str | None) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=device,
    ).astype(np.float32)


def aggregate(similarities: np.ndarray, mode: str) -> float:
    if similarities.size == 0:
        return float("-inf")
    if mode == "mean":
        return float(np.mean(similarities))
    return float(np.max(similarities))


def add_rank_and_margins(rows: list[dict[str, Any]]) -> None:
    previous_score: float | None = None
    for idx, row in enumerate(rows):
        row["rank"] = idx + 1
        row["margin_to_next"] = float(row["score"] - rows[idx + 1]["score"]) if idx + 1 < len(rows) else np.nan
        row["margin_to_prev"] = np.nan if previous_score is None else float(previous_score - row["score"])
        previous_score = float(row["score"])


def process_resource_metrics() -> dict[str, Any]:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        peak_ram_mb = rss / (1024 * 1024)
    else:
        peak_ram_mb = rss / 1024
    return {
        "platform": platform.system(),
        "peak_ram_mb": float(peak_ram_mb),
    }


def reset_accelerator_peak_memory() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(idx)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def accelerator_memory_metrics() -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    try:
        import torch
    except ImportError:
        return metrics
    if torch.cuda.is_available():
        peak_allocated = 0
        peak_reserved = 0
        for idx in range(torch.cuda.device_count()):
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(idx))
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(idx))
        metrics["peak_cuda_allocated_mb"] = float(peak_allocated / (1024 ** 2))
        metrics["peak_cuda_reserved_mb"] = float(peak_reserved / (1024 ** 2))
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        current = getattr(torch.mps, "current_allocated_memory", lambda: 0)()
        driver = getattr(torch.mps, "driver_allocated_memory", lambda: 0)()
        metrics["mps_current_allocated_mb"] = float(current / (1024 ** 2))
        metrics["mps_driver_allocated_mb"] = float(driver / (1024 ** 2))
    return metrics


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


if __name__ == "__main__":
    main()
