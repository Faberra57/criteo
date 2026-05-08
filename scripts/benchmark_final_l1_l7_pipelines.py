#!/usr/bin/env python3
"""Benchmark the two final L1-L7 classification pipelines.

The script evaluates end-to-end prediction cost on X products:
1. L1 is predicted with the fine-tuned BGE-small model.
2. L2-L7 is predicted with Qwen embeddings computed at runtime using:
    - greedy hierarchical decoding;
    - hybrid decoding: beam search + global path check + selective reranking.

Category enrichment prototypes with prototype_type == descendant_names_text are
filtered out, so the benchmark uses category names, paths and lexical expansions
but not descendant summaries.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import resource
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = REPO_ROOT / "dataset" / "preprocessed_lv2.parquet"
DEFAULT_ENRICHMENT_DIR = REPO_ROOT / "dataset" / "category_enrichment"
DEFAULT_QWEN_EMBEDDINGS_DIR = REPO_ROOT / "dataset" / "shared_l1_l7_embeddings_qwen"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_TAXONOMY_PATH = REPO_ROOT / "taxonomy.txt"
DEFAULT_L1_RUN_DIR = (
    REPO_ROOT
    / "models"
    / "embedding_runs"
    / "20260503_131422__enriched-triplet-finetune__model-baai-bge-small-en-v1-5__label-level-1-name__bs-128__sampler-tempered__ep-4__lr-2e-04__seq-256__seed-42"
    / "best_model"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "final_l1_l7_pipeline_benchmark"


def main() -> None:
    args = parse_args()
    configure_runtime(args)
    reset_accelerator_peak_memory()
    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_profile: dict[str, Any] = {
        "args": serializable_args(args),
        "system": system_info(),
        "started_at_unix": time.time(),
    }

    load_started = time.perf_counter()
    products = load_products(args)
    qwen = generate_qwen_embeddings(Path(args.qwen_embeddings_dir), products, args)
    l1_prototypes = load_l1_prototypes(Path(args.enrichment_dir))
    load_seconds = time.perf_counter() - load_started

    l1_started = time.perf_counter()
    l1_predictions, l1_metrics = predict_l1(products, l1_prototypes, args)
    l1_seconds = time.perf_counter() - l1_started
    products = products.join(l1_predictions)

    index_started = time.perf_counter()
    qwen_index = QwenPrototypeIndex(
        nodes=qwen["nodes"],
        prototypes=qwen["prototypes"],
        prototype_embeddings=qwen["prototype_embeddings"],
        aggregation=args.score_aggregation,
    )
    global_index = GlobalPathIndex(
        global_paths=qwen["global_paths"],
        global_path_embeddings=qwen["global_path_embeddings"],
    )
    index_seconds = time.perf_counter() - index_started

    results: dict[str, Any] = {
        "load_inputs_seconds": load_seconds,
        "l1_total_seconds": l1_seconds,
        "index_build_seconds": index_seconds,
        "l1_metrics": l1_metrics,
        "pipelines": {},
    }

    if args.pipeline in {"greedy", "both"}:
        greedy = run_greedy_pipeline(
            products, qwen["product_embeddings"], qwen_index, args
        )
        write_pipeline_outputs(greedy, output_dir, "greedy")
        results["pipelines"]["greedy"] = greedy["metrics"]

    if args.pipeline in {"hybrid", "both"}:
        hybrid = run_hybrid_pipeline(
            products,
            qwen["product_embeddings"],
            qwen_index,
            global_index,
            args,
        )
        write_pipeline_outputs(hybrid, output_dir, "hybrid")
        results["pipelines"]["hybrid"] = hybrid["metrics"]

    results.update(
        {
            "elapsed_seconds": time.perf_counter() - started,
            "n_products": int(len(products)),
            "qwen_embedding_metrics": qwen.get("metrics", {}),
            "resource_metrics_end": process_resource_metrics(),
            **accelerator_memory_metrics(),
        }
    )
    write_json(
        run_profile | {"results": results}, output_dir / "benchmark_metrics.json"
    )
    write_summary_csv(results, output_dir / "benchmark_summary.csv")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Benchmark outputs written to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        "--products-path",
        dest="input_path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Product catalog input file.",
    )
    parser.add_argument("--input-format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument(
        "--qwen-embeddings-dir",
        "--categories-dir",
        dest="qwen_embeddings_dir",
        type=Path,
        default=DEFAULT_QWEN_EMBEDDINGS_DIR,
        help=(
            "Directory with category prototypes/global paths and optional precomputed embeddings."
        ),
    )
    parser.add_argument(
        "--qwen-model",
        "--l2l7-embedding-model",
        dest="qwen_model",
        default=DEFAULT_QWEN_MODEL,
    )
    parser.add_argument("--qwen-device", default=None)
    parser.add_argument("--qwen-batch-size", type=int, default=64)
    parser.add_argument(
        "--qwen-max-seq-length",
        type=int,
        default=None,
        help="Optional override for the Qwen embedding model max sequence length.",
    )
    parser.add_argument(
        "--taxonomy-path",
        type=Path,
        default=DEFAULT_TAXONOMY_PATH,
        help="Taxonomy file with one path per line (e.g. level1 > level2 > ...).",
    )
    parser.add_argument("--enrichment-dir", type=Path, default=DEFAULT_ENRICHMENT_DIR)
    parser.add_argument("--l1-run-dir", type=Path, default=DEFAULT_L1_RUN_DIR)
    parser.add_argument("--l1-model-subdir", default="best_model")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--id-col", default="hashed_external_id")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-l1-col", default="level_1_name")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--pipeline", choices=["greedy", "hybrid", "both"], default="both"
    )
    parser.add_argument("--l1-device", default=None)
    parser.add_argument("--l1-batch-size", type=int, default=128)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--score-aggregation", choices=["max", "mean"], default="max")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--children-top-k", type=int, default=8)
    parser.add_argument("--output-top-paths", type=int, default=10)
    parser.add_argument("--global-top-k", type=int, default=20)
    parser.add_argument(
        "--rerank-if-disagree-at-or-before-level",
        type=int,
        default=3,
        help="Trigger reranking if beam and global path first disagree at or before this level.",
    )
    parser.add_argument(
        "--beam-margin-threshold",
        type=float,
        default=None,
        help="Optional reranking trigger when beam top1-top2 margin is below this threshold.",
    )
    parser.add_argument(
        "--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    parser.add_argument("--reranker-device", default=None)
    parser.add_argument("--reranker-batch-size", type=int, default=64)
    parser.add_argument(
        "--keep-prediction-text",
        action="store_true",
        help="Keep product text in prediction CSVs. Disabled by default to reduce output size.",
    )
    return parser.parse_args()


def configure_runtime(args: argparse.Namespace) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.random_seed is not None:
        np.random.seed(args.random_seed)


def load_products(args: argparse.Namespace) -> pd.DataFrame:
    df = (
        pd.read_parquet(args.input_path)
        if args.input_format == "parquet"
        else pd.read_csv(args.input_path)
    )
    required = {args.id_col, args.text_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing product columns in {args.input_path}: {sorted(missing)}"
        )
    keep = [args.id_col, args.text_col]
    if args.label_l1_col in df.columns:
        keep.append(args.label_l1_col)
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
        raise FileNotFoundError(f"Missing L1 reference texts: {path}")
    df = pd.read_csv(path)
    required = {"node_key", "category_name", "prototype_type", "prototype_text"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df[df["prototype_type"].fillna("") != "descendant_names_text"].copy()
    df["prototype_text"] = df["prototype_text"].fillna("").astype(str)
    df = df[df["prototype_text"].str.len() > 0]
    return df.drop_duplicates(
        ["node_key", "prototype_type", "prototype_text"]
    ).reset_index(drop=True)


def load_taxonomy_nodes(taxonomy_path: Path) -> pd.DataFrame:
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"Missing taxonomy file: {taxonomy_path}")
    nodes: dict[str, dict[str, Any]] = {}
    lines = taxonomy_path.read_text(encoding="utf-8").splitlines()
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(" > ") if part.strip()]
        if not parts:
            continue
        for idx in range(len(parts)):
            node_key = " > ".join(parts[: idx + 1])
            if node_key in nodes:
                continue
            parent_key = " > ".join(parts[:idx]) if idx > 0 else ""
            nodes[node_key] = {
                "node_key": node_key,
                "taxonomy_path": node_key,
                "depth": idx + 1,
                "category_name": parts[idx],
                "parent_key": parent_key,
                "level_1_name": parts[0],
            }
    for node in nodes.values():
        parent_key = node.get("parent_key", "")
        node["parent_name"] = (
            nodes[parent_key]["category_name"] if parent_key in nodes else ""
        )
    df = pd.DataFrame(nodes.values())
    if not df.empty:
        df = df.sort_values(["depth", "taxonomy_path"]).reset_index(drop=True)
    return df


def build_global_candidate_text(row: pd.Series) -> str:
    taxonomy_path = str(row.get("taxonomy_path", "")).strip()
    category_name = str(row.get("category_name", "")).strip()
    depth = row.get("depth", "")
    return (
        f"Taxonomy path: {taxonomy_path}. "
        f"Category name: {category_name}. "
        f"Level: {depth}."
    )


def generate_qwen_embeddings(
    embeddings_dir: Path, products: pd.DataFrame, args: argparse.Namespace
) -> dict[str, Any]:
    required_files = [
        "category_prototypes.parquet",
        "global_paths.parquet",
    ]
    missing = [name for name in required_files if not (embeddings_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing Qwen shared inputs in {embeddings_dir}: {missing}"
        )

    nodes = load_taxonomy_nodes(Path(args.taxonomy_path))
    prototypes = pd.read_parquet(
        embeddings_dir / "category_prototypes.parquet"
    ).reset_index(drop=True)
    required = {"node_key", "category_name", "prototype_type", "prototype_text"}
    missing_cols = required.difference(prototypes.columns)
    if missing_cols:
        raise ValueError(
            f"category_prototypes.parquet missing columns: {sorted(missing_cols)}"
        )
    prototypes["prototype_text"] = prototypes["prototype_text"].fillna("").astype(str)
    prototypes = prototypes[prototypes["prototype_text"].str.len() > 0]
    keep_mask = prototypes["prototype_type"].fillna("") != "descendant_names_text"
    prototypes = prototypes[keep_mask].reset_index(drop=True)

    global_paths = pd.read_parquet(embeddings_dir / "global_paths.parquet").reset_index(
        drop=True
    )
    if "candidate_text" not in global_paths.columns:
        global_paths["candidate_text"] = global_paths.apply(
            build_global_candidate_text, axis=1
        )
    global_paths["candidate_text"] = (
        global_paths["candidate_text"].fillna("").astype(str)
    )
    global_keep_mask = global_paths["candidate_text"].str.len() > 0
    global_paths = global_paths[global_keep_mask].reset_index(drop=True)

    from sentence_transformers import SentenceTransformer

    load_started = time.perf_counter()
    model = SentenceTransformer(
        args.qwen_model, device=args.qwen_device, trust_remote_code=True
    )
    if args.qwen_max_seq_length:
        model.max_seq_length = args.qwen_max_seq_length
    model_load_seconds = time.perf_counter() - load_started

    proto_embed_path = embeddings_dir / "category_prototype_embeddings.npy"
    if proto_embed_path.exists():
        prototype_embeddings = np.load(proto_embed_path).astype(np.float32)
        prototype_embeddings = prototype_embeddings[
            np.flatnonzero(keep_mask.to_numpy())
        ]
        prototype_encode_seconds = 0.0
        prototype_source = "precomputed"
    else:
        proto_started = time.perf_counter()
        prototype_embeddings = encode_embedding_texts(
            model,
            prototypes["prototype_text"].tolist(),
            batch_size=args.qwen_batch_size,
            device=args.qwen_device,
            task="retrieval",
        )
        prototype_encode_seconds = time.perf_counter() - proto_started
        prototype_source = "runtime"

    product_started = time.perf_counter()
    product_embeddings = encode_embedding_texts(
        model,
        products[args.text_col].fillna("").astype(str).tolist(),
        batch_size=args.qwen_batch_size,
        device=args.qwen_device,
        task="retrieval",
    )
    product_encode_seconds = time.perf_counter() - product_started

    global_embed_path = embeddings_dir / "global_path_embeddings.npy"
    if global_embed_path.exists():
        global_path_embeddings = np.load(global_embed_path).astype(np.float32)
        global_path_embeddings = global_path_embeddings[
            np.flatnonzero(global_keep_mask.to_numpy())
        ]
        global_path_encode_seconds = 0.0
        global_path_source = "precomputed"
    else:
        global_started = time.perf_counter()
        global_path_embeddings = encode_embedding_texts(
            model,
            global_paths["candidate_text"].tolist(),
            batch_size=args.qwen_batch_size,
            device=args.qwen_device,
            task="retrieval",
        )
        global_path_encode_seconds = time.perf_counter() - global_started
        global_path_source = "runtime"

    total_seconds = (
        model_load_seconds
        + prototype_encode_seconds
        + product_encode_seconds
        + global_path_encode_seconds
    )
    metrics = {
        "model_name": str(args.qwen_model),
        "device": args.qwen_device,
        "batch_size": int(args.qwen_batch_size),
        "max_seq_length": int(args.qwen_max_seq_length)
        if args.qwen_max_seq_length
        else None,
        "n_products": int(len(products)),
        "n_prototypes": int(len(prototypes)),
        "n_global_paths": int(len(global_paths)),
        "embedding_dim": int(product_embeddings.shape[1])
        if len(product_embeddings)
        else 0,
        "model_load_seconds": float(model_load_seconds),
        "prototype_encode_seconds": float(prototype_encode_seconds),
        "product_encode_seconds": float(product_encode_seconds),
        "global_path_encode_seconds": float(global_path_encode_seconds),
        "prototype_embeddings_source": prototype_source,
        "global_path_embeddings_source": global_path_source,
        "total_seconds": float(total_seconds),
        "seconds_per_product_including_global": float(
            total_seconds / max(1, len(products))
        ),
    }

    return {
        "product_embeddings": product_embeddings,
        "nodes": nodes,
        "prototypes": prototypes,
        "prototype_embeddings": prototype_embeddings,
        "global_paths": global_paths,
        "global_path_embeddings": global_path_embeddings,
        "metrics": metrics,
    }


def encode_embedding_texts(
    model: Any,
    texts: list[str],
    *,
    batch_size: int,
    device: str | None,
    task: str | None = None,
) -> np.ndarray:
    encode_kwargs = {
        "batch_size": batch_size,
        "show_progress_bar": True,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
    }
    if device:
        encode_kwargs["device"] = device
    if task:
        try:
            return model.encode(texts, task=task, **encode_kwargs).astype(np.float32)
        except TypeError:
            pass
    return model.encode(texts, **encode_kwargs).astype(np.float32)


def predict_l1(
    products: pd.DataFrame, prototypes: pd.DataFrame, args: argparse.Namespace
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from sentence_transformers import SentenceTransformer

    base_path = Path(args.l1_run_dir)
    candidate = base_path / args.l1_model_subdir if args.l1_model_subdir else base_path
    model_path = candidate if candidate.exists() else base_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"L1 model directory not found: {model_path} (base: {base_path})"
        )

    load_started = time.perf_counter()
    model = SentenceTransformer(str(model_path), device=args.l1_device)
    model_load_seconds = time.perf_counter() - load_started

    proto_started = time.perf_counter()
    prototype_embeddings = encode_sentence_transformer(
        model,
        prototypes["prototype_text"].tolist(),
        batch_size=args.l1_batch_size,
        device=args.l1_device,
    )
    prototype_encode_seconds = time.perf_counter() - proto_started

    product_started = time.perf_counter()
    product_embeddings = encode_sentence_transformer(
        model,
        products[args.text_col].tolist(),
        batch_size=args.l1_batch_size,
        device=args.l1_device,
    )
    product_encode_seconds = time.perf_counter() - product_started

    score_started = time.perf_counter()
    node_to_indices = (
        prototypes.reset_index().groupby("node_key")["index"].apply(list).to_dict()
    )
    node_names = prototypes.groupby("node_key")["category_name"].first().to_dict()
    node_keys = sorted(node_to_indices)
    rows = []
    n_dot_products = 0
    for product_idx, query in enumerate(product_embeddings):
        scores = []
        for node_key in node_keys:
            indices = np.asarray(node_to_indices[node_key], dtype=int)
            sims = prototype_embeddings[indices] @ query
            n_dot_products += len(indices)
            scores.append(
                {
                    "node_key": str(node_key),
                    "node_name": str(node_names.get(node_key, node_key)),
                    "score": aggregate(sims, args.score_aggregation),
                    "n_prototypes": int(len(indices)),
                }
            )
        scores.sort(key=lambda row: float(row["score"]), reverse=True)
        add_rank_and_margins(scores)
        top1 = scores[0] if scores else {}
        top2 = scores[1] if len(scores) > 1 else {}
        rows.append(
            {
                "l1_predicted_key": str(top1.get("node_key", "")),
                "l1_predicted_name": str(top1.get("node_name", "")),
                "l1_top1_score": float(top1.get("score", np.nan)),
                "l1_top2_score": float(top2.get("score", np.nan)),
                "l1_top1_top2_margin": float(
                    top1.get("score", np.nan) - top2.get("score", np.nan)
                )
                if top2
                else np.nan,
                "l1_top_candidates_json": json.dumps(scores[:5], ensure_ascii=False),
            }
        )
    score_seconds = time.perf_counter() - score_started

    out = pd.DataFrame(rows)
    metrics = {
        "model_path": str(model_path),
        "n_products": int(len(products)),
        "n_l1_categories": int(len(node_keys)),
        "n_l1_prototypes": int(len(prototypes)),
        "embedding_dim": int(product_embeddings.shape[1])
        if len(product_embeddings)
        else 0,
        "model_load_seconds": float(model_load_seconds),
        "prototype_encode_seconds": float(prototype_encode_seconds),
        "product_encode_seconds": float(product_encode_seconds),
        "score_seconds": float(score_seconds),
        "total_seconds": float(
            model_load_seconds
            + prototype_encode_seconds
            + product_encode_seconds
            + score_seconds
        ),
        "seconds_per_product_including_l1_encoding": float(
            (
                model_load_seconds
                + prototype_encode_seconds
                + product_encode_seconds
                + score_seconds
            )
            / max(1, len(products))
        ),
        "l1_similarity_dot_products": int(n_dot_products),
        "l1_similarity_dot_products_per_product": float(
            n_dot_products / max(1, len(products))
        ),
        "l1_approx_million_scalar_multiply_adds": float(
            n_dot_products
            * (int(product_embeddings.shape[1]) if len(product_embeddings) else 0)
            / 1_000_000
        ),
        "median_l1_margin": float(
            pd.to_numeric(out["l1_top1_top2_margin"], errors="coerce").median()
        ),
        "share_l1_margin_lt_0_03": float(
            (pd.to_numeric(out["l1_top1_top2_margin"], errors="coerce") < 0.03).mean()
        ),
        "resource_after_l1": process_resource_metrics(),
        **accelerator_memory_metrics(prefix="l1_"),
    }
    if args.label_l1_col in products.columns:
        metrics["l1_accuracy_if_label_available"] = float(
            (
                products[args.label_l1_col].astype(str)
                == out["l1_predicted_key"].astype(str)
            ).mean()
        )
    return out, metrics


def encode_sentence_transformer(
    model: Any, texts: list[str], *, batch_size: int, device: str | None
) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=device,
    ).astype(np.float32)


class QwenPrototypeIndex:
    def __init__(
        self,
        *,
        nodes: pd.DataFrame,
        prototypes: pd.DataFrame,
        prototype_embeddings: np.ndarray,
        aggregation: str,
    ) -> None:
        self.nodes = nodes.copy().reset_index(drop=True)
        self.prototypes = prototypes.copy().reset_index(drop=True)
        self.prototype_embeddings = prototype_embeddings.astype(np.float32)
        self.aggregation = aggregation
        self.node_meta = self.nodes.set_index("node_key").to_dict("index")
        self.parent_to_children = build_parent_to_children(self.nodes)
        self.node_to_indices = (
            self.prototypes.reset_index()
            .groupby("node_key")["index"]
            .apply(list)
            .to_dict()
        )
        self.node_to_centroid = self._build_centroids()

    def _build_centroids(self) -> dict[str, np.ndarray]:
        centroids = {}
        for node_key, indices in self.node_to_indices.items():
            matrix = self.prototype_embeddings[np.asarray(indices, dtype=int)]
            centroid = matrix.mean(axis=0, keepdims=True)
            centroids[node_key] = normalize_rows(centroid)[0]
        return centroids

    def children(self, node_key: str) -> list[str]:
        return self.parent_to_children.get(node_key, [])

    def score_nodes(
        self, query: np.ndarray, node_keys: list[str]
    ) -> list[dict[str, Any]]:
        rows = []
        for node_key in node_keys:
            indices = self.node_to_indices.get(node_key, [])
            if not indices:
                continue
            sims = self.prototype_embeddings[np.asarray(indices, dtype=int)] @ query
            meta = self.node_meta.get(node_key, {})
            rows.append(
                {
                    "node_key": str(node_key),
                    "node_name": str(meta.get("category_name", node_key)),
                    "depth": int(meta.get("depth", 0)),
                    "score": aggregate(sims, self.aggregation),
                    "n_prototypes": int(len(indices)),
                }
            )
        rows.sort(key=lambda item: float(item["score"]), reverse=True)
        add_rank_and_margins(rows)
        return rows

    def prototype_count(self, node_keys: list[str]) -> int:
        return int(sum(len(self.node_to_indices.get(key, [])) for key in node_keys))

    def path_internal_coherence(self, path_keys: list[str]) -> dict[str, float]:
        vectors = [
            self.node_to_centroid[key]
            for key in path_keys
            if key in self.node_to_centroid
        ]
        if len(vectors) < 2:
            return {
                "path_mean_adjacent_similarity": np.nan,
                "path_min_adjacent_similarity": np.nan,
                "final_to_ancestors_mean_similarity": np.nan,
            }
        adjacent = [float(vectors[i] @ vectors[i + 1]) for i in range(len(vectors) - 1)]
        final_to_ancestors = [
            float(vectors[-1] @ ancestor) for ancestor in vectors[:-1]
        ]
        return {
            "path_mean_adjacent_similarity": float(np.mean(adjacent)),
            "path_min_adjacent_similarity": float(np.min(adjacent)),
            "final_to_ancestors_mean_similarity": float(np.mean(final_to_ancestors)),
        }


class GlobalPathIndex:
    def __init__(
        self, *, global_paths: pd.DataFrame, global_path_embeddings: np.ndarray
    ) -> None:
        self.global_paths = global_paths.copy().reset_index(drop=True)
        self.global_path_embeddings = global_path_embeddings.astype(np.float32)
        self.by_l1 = self._build_by_l1()

    def _build_by_l1(self) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
        out = {}
        for l1_name, group in self.global_paths.groupby("level_1_name", sort=False):
            indices = group.index.to_numpy(dtype=int)
            out[str(l1_name)] = (
                group.reset_index(drop=True),
                self.global_path_embeddings[indices],
            )
        return out

    def top_paths(
        self, query: np.ndarray, l1_key: str, top_k: int
    ) -> tuple[list[dict[str, Any]], int]:
        candidates_df, candidates_embeddings = self.by_l1.get(str(l1_key), (None, None))
        if (
            candidates_df is None
            or candidates_embeddings is None
            or len(candidates_df) == 0
        ):
            return [], 0
        scores = candidates_embeddings @ query
        top_idx = topk_indices(scores, min(top_k, len(scores)))
        rows = candidates_df.iloc[top_idx].copy().reset_index(drop=True)
        rows["embedding_score"] = scores[top_idx]
        candidates = rows.to_dict("records")
        candidates.sort(key=lambda item: float(item["embedding_score"]), reverse=True)
        return candidates, int(len(scores))


def run_greedy_pipeline(
    products: pd.DataFrame,
    product_embeddings: np.ndarray,
    index: QwenPrototypeIndex,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    resource_before = process_resource_metrics()
    rows = []
    n_dot_products = 0
    n_node_score_calls = 0
    for row, embedding in zip(
        products.to_dict("records"), product_embeddings, strict=True
    ):
        paths, node_calls, dots = greedy_decode(
            index, embedding, str(row["l1_predicted_key"]), max_depth=args.max_depth
        )
        best_path = paths[0] if paths else empty_path(str(row["l1_predicted_key"]))
        n_node_score_calls += node_calls
        n_dot_products += dots
        rows.append(
            build_prediction_row(
                row,
                args=args,
                pipeline_mode="greedy",
                best_path=best_path,
                top_paths=paths[:1],
                node_score_calls=node_calls,
                similarity_dot_products=dots,
                rerank_pair_count=0,
                global_path=None,
                rerank_triggered=False,
                rerank_reason="",
                path_internal_coherence=index.path_internal_coherence(
                    best_path.get("path_keys", [])
                ),
            )
        )
    output = pd.DataFrame(rows)
    elapsed = time.perf_counter() - started
    metrics = build_pipeline_metrics(
        output,
        mode="greedy",
        elapsed=elapsed,
        embedding_dim=int(product_embeddings.shape[1]),
        n_node_score_calls=n_node_score_calls,
        n_similarity_dot_products=n_dot_products,
        n_global_path_dot_products=0,
        n_rerank_pairs=0,
        resource_before=resource_before,
    )
    return {"predictions": output, "metrics": metrics}


def run_hybrid_pipeline(
    products: pd.DataFrame,
    product_embeddings: np.ndarray,
    index: QwenPrototypeIndex,
    global_index: GlobalPathIndex,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    resource_before = process_resource_metrics()
    reranker = None
    reranker_load_seconds = 0.0
    rerank_seconds = 0.0
    n_rerank_pairs = 0
    n_dot_products = 0
    n_global_dot_products = 0
    n_node_score_calls = 0
    rows = []

    for row, embedding in zip(
        products.to_dict("records"), product_embeddings, strict=True
    ):
        beam_paths, node_calls, dots = beam_decode(
            index,
            embedding,
            str(row["l1_predicted_key"]),
            max_depth=args.max_depth,
            beam_width=args.beam_width,
            children_top_k=args.children_top_k,
            output_top_paths=args.output_top_paths,
        )
        global_candidates, global_dots = global_index.top_paths(
            embedding, str(row["l1_predicted_key"]), args.global_top_k
        )
        n_node_score_calls += node_calls
        n_dot_products += dots
        n_global_dot_products += global_dots

        best_beam = (
            beam_paths[0] if beam_paths else empty_path(str(row["l1_predicted_key"]))
        )
        best_global = (
            global_candidate_to_path(global_candidates[0])
            if global_candidates
            else None
        )
        triggered, reason = should_rerank(best_beam, best_global, beam_paths, args)

        final_paths = beam_paths
        if triggered and beam_paths:
            if reranker is None:
                load_started = time.perf_counter()
                reranker = load_cross_encoder(args.reranker_model, args.reranker_device)
                reranker_load_seconds += time.perf_counter() - load_started
            rerank_started = time.perf_counter()
            final_paths = rerank_paths(
                reranker,
                product_text=str(row.get(args.text_col, "")),
                paths=beam_paths,
                batch_size=args.reranker_batch_size,
            )
            rerank_seconds += time.perf_counter() - rerank_started
            n_rerank_pairs += len(beam_paths)

        best_path = final_paths[0] if final_paths else best_beam
        rows.append(
            build_prediction_row(
                row,
                args=args,
                pipeline_mode="hybrid_beam_global_selective_rerank",
                best_path=best_path,
                top_paths=final_paths[: args.output_top_paths],
                node_score_calls=node_calls,
                similarity_dot_products=dots + global_dots,
                rerank_pair_count=len(beam_paths) if triggered else 0,
                global_path=best_global,
                rerank_triggered=triggered,
                rerank_reason=reason,
                path_internal_coherence=index.path_internal_coherence(
                    best_path.get("path_keys", [])
                ),
            )
        )

    output = pd.DataFrame(rows)
    elapsed = time.perf_counter() - started
    metrics = build_pipeline_metrics(
        output,
        mode="hybrid_beam_global_selective_rerank",
        elapsed=elapsed,
        embedding_dim=int(product_embeddings.shape[1]),
        n_node_score_calls=n_node_score_calls,
        n_similarity_dot_products=n_dot_products + n_global_dot_products,
        n_global_path_dot_products=n_global_dot_products,
        n_rerank_pairs=n_rerank_pairs,
        resource_before=resource_before,
        extra={
            "beam_width": int(args.beam_width),
            "children_top_k": int(args.children_top_k),
            "global_top_k": int(args.global_top_k),
            "reranker_model": args.reranker_model,
            "reranker_load_seconds": float(reranker_load_seconds),
            "rerank_seconds": float(rerank_seconds),
            "rerank_trigger_rate": float(output["rerank_triggered"].mean())
            if len(output)
            else np.nan,
            "n_reranked_products": int(output["rerank_triggered"].sum())
            if len(output)
            else 0,
        },
    )
    return {"predictions": output, "metrics": metrics}


def greedy_decode(
    index: QwenPrototypeIndex, query: np.ndarray, start_l1: str, *, max_depth: int
) -> tuple[list[dict[str, Any]], int, int]:
    path = initialize_path(index, start_l1)
    n_node_calls = 0
    n_dot_products = 0
    while path["resolved_depth"] < max_depth:
        child_keys = index.children(path["path_keys"][-1]) if path["path_keys"] else []
        if not child_keys:
            break
        scores = index.score_nodes(query, child_keys)
        n_node_calls += len(child_keys)
        n_dot_products += index.prototype_count(child_keys)
        if not scores:
            break
        append_child(path, scores[0])
    return [path], n_node_calls, n_dot_products


def beam_decode(
    index: QwenPrototypeIndex,
    query: np.ndarray,
    start_l1: str,
    *,
    max_depth: int,
    beam_width: int,
    children_top_k: int,
    output_top_paths: int,
) -> tuple[list[dict[str, Any]], int, int]:
    frontier = [initialize_path(index, start_l1)]
    completed = []
    n_node_calls = 0
    n_dot_products = 0
    while frontier:
        expanded = []
        for state in frontier:
            if state["resolved_depth"] >= max_depth:
                completed.append(state)
                continue
            child_keys = (
                index.children(state["path_keys"][-1]) if state["path_keys"] else []
            )
            if not child_keys:
                completed.append(state)
                continue
            scores = index.score_nodes(query, child_keys)[:children_top_k]
            n_node_calls += len(child_keys)
            n_dot_products += index.prototype_count(child_keys)
            for score in scores:
                candidate = copy_jsonable(state)
                append_child(candidate, score)
                expanded.append(candidate)
        if not expanded:
            break
        expanded.sort(key=lambda item: float(item["cumulative_score"]), reverse=True)
        frontier = expanded[:beam_width]
    paths = deduplicate_paths(completed + frontier)
    return paths[:output_top_paths], n_node_calls, n_dot_products


def rerank_paths(
    reranker: Any, *, product_text: str, paths: list[dict[str, Any]], batch_size: int
) -> list[dict[str, Any]]:
    pairs = [
        (product_text, "Taxonomy path: " + " > ".join(path.get("path_names", [])))
        for path in paths
    ]
    scores = np.asarray(
        reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False),
        dtype=float,
    ).reshape(-1)
    reranked = []
    for path, score in zip(paths, scores, strict=True):
        item = copy_jsonable(path)
        item["rerank_score"] = float(score)
        reranked.append(item)
    reranked.sort(key=lambda item: float(item["rerank_score"]), reverse=True)
    return reranked


def initialize_path(index: QwenPrototypeIndex, start_l1: str) -> dict[str, Any]:
    meta = index.node_meta.get(start_l1, {})
    return {
        "path_keys": [start_l1] if start_l1 else [],
        "path_names": [str(meta.get("category_name", start_l1))] if start_l1 else [],
        "score_trace": [],
        "margin_trace": [],
        "cumulative_score": 0.0,
        "resolved_depth": 1 if start_l1 else 0,
    }


def append_child(path: dict[str, Any], child_score: dict[str, Any]) -> None:
    path["path_keys"].append(str(child_score["node_key"]))
    path["path_names"].append(str(child_score["node_name"]))
    path["score_trace"].append(float(child_score["score"]))
    path["margin_trace"].append(float(child_score.get("sibling_margin", np.nan)))
    path["resolved_depth"] = int(child_score["depth"])
    path["cumulative_score"] = (
        float(np.mean(path["score_trace"])) if path["score_trace"] else 0.0
    )


def empty_path(start_l1: str) -> dict[str, Any]:
    return {
        "path_keys": [start_l1] if start_l1 else [],
        "path_names": [start_l1] if start_l1 else [],
        "score_trace": [],
        "margin_trace": [],
        "cumulative_score": 0.0,
        "resolved_depth": 1 if start_l1 else 0,
    }


def global_candidate_to_path(candidate: dict[str, Any]) -> dict[str, Any]:
    taxonomy_path = str(candidate.get("taxonomy_path", ""))
    parts = [part.strip() for part in taxonomy_path.split(" > ") if part.strip()]
    return {
        "path_keys": [" > ".join(parts[: i + 1]) for i in range(len(parts))],
        "path_names": parts,
        "score_trace": [float(candidate.get("embedding_score", np.nan))],
        "margin_trace": [],
        "cumulative_score": float(candidate.get("embedding_score", np.nan)),
        "resolved_depth": int(candidate.get("depth", len(parts))),
        "global_path_score": float(candidate.get("embedding_score", np.nan)),
    }


def should_rerank(
    beam_path: dict[str, Any],
    global_path: dict[str, Any] | None,
    beam_paths: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[bool, str]:
    reasons = []
    if global_path is not None:
        disagreement_level = first_disagreement_level(
            beam_path.get("path_keys", []), global_path.get("path_keys", [])
        )
        if (
            disagreement_level is not None
            and disagreement_level <= args.rerank_if_disagree_at_or_before_level
        ):
            reasons.append(f"beam_global_disagree_at_L{disagreement_level}")
    if args.beam_margin_threshold is not None and len(beam_paths) >= 2:
        scores = path_candidate_scores(beam_paths)
        if len(scores) >= 2 and scores[0] - scores[1] < args.beam_margin_threshold:
            reasons.append(f"beam_margin_lt_{args.beam_margin_threshold}")
    return bool(reasons), ";".join(reasons)


def first_disagreement_level(path_a: list[str], path_b: list[str]) -> int | None:
    max_len = max(len(path_a), len(path_b))
    for idx in range(max_len):
        a = path_a[idx] if idx < len(path_a) else ""
        b = path_b[idx] if idx < len(path_b) else ""
        if a != b:
            return idx + 1
    return None


def build_prediction_row(
    row: dict[str, Any],
    *,
    args: argparse.Namespace,
    pipeline_mode: str,
    best_path: dict[str, Any],
    top_paths: list[dict[str, Any]],
    node_score_calls: int,
    similarity_dot_products: int,
    rerank_pair_count: int,
    global_path: dict[str, Any] | None,
    rerank_triggered: bool,
    rerank_reason: str,
    path_internal_coherence: dict[str, float],
) -> dict[str, Any]:
    path_names = best_path.get("path_names", [])
    path_keys = best_path.get("path_keys", [])
    score_trace = [
        float(x) for x in best_path.get("score_trace", []) if is_finite_number(x)
    ]
    margin_trace = [
        float(x) for x in best_path.get("margin_trace", []) if is_finite_number(x)
    ]
    candidate_scores = path_candidate_scores(top_paths)
    top1 = candidate_scores[0] if len(candidate_scores) >= 1 else np.nan
    top2 = candidate_scores[1] if len(candidate_scores) >= 2 else np.nan
    global_keys = global_path.get("path_keys", []) if global_path else []
    out = {
        args.id_col: row.get(args.id_col, ""),
        "pipeline_mode": pipeline_mode,
        "l1_predicted_key": row.get("l1_predicted_key", ""),
        "l1_predicted_name": row.get("l1_predicted_name", ""),
        "l1_top1_score": row.get("l1_top1_score", np.nan),
        "l1_top1_top2_margin": row.get("l1_top1_top2_margin", np.nan),
        "input_level_1_name": row.get(args.label_l1_col, ""),
        "predicted_taxonomy_path": " > ".join(path_names),
        "predicted_taxonomy_key_path": " || ".join(path_keys),
        "predicted_path_score": float(
            best_path.get("rerank_score", best_path.get("cumulative_score", np.nan))
        ),
        "embedding_path_score": float(best_path.get("cumulative_score", np.nan)),
        "top1_score": float(top1),
        "top2_score": float(top2),
        "top1_top2_margin": float(top1 - top2)
        if np.isfinite(top1) and np.isfinite(top2)
        else np.nan,
        "final_category_score": float(score_trace[-1]) if score_trace else np.nan,
        "mean_local_score": float(np.mean(score_trace)) if score_trace else np.nan,
        "mean_local_margin": float(np.mean(margin_trace)) if margin_trace else np.nan,
        "min_local_margin": float(np.min(margin_trace)) if margin_trace else np.nan,
        "resolved_depth": int(best_path.get("resolved_depth", 0)),
        "node_score_calls": int(node_score_calls),
        "similarity_dot_products": int(similarity_dot_products),
        "rerank_pair_count": int(rerank_pair_count),
        "rerank_triggered": bool(rerank_triggered),
        "rerank_reason": rerank_reason,
        "global_path": " > ".join(global_path.get("path_names", []))
        if global_path
        else "",
        "global_path_score": float(global_path.get("global_path_score", np.nan))
        if global_path
        else np.nan,
        "beam_global_first_disagreement_level": first_disagreement_level(
            path_keys, global_keys
        )
        if global_path
        else np.nan,
        "path_mean_adjacent_similarity": path_internal_coherence[
            "path_mean_adjacent_similarity"
        ],
        "path_min_adjacent_similarity": path_internal_coherence[
            "path_min_adjacent_similarity"
        ],
        "final_to_ancestors_mean_similarity": path_internal_coherence[
            "final_to_ancestors_mean_similarity"
        ],
        "path_score_trace_json": json.dumps(score_trace, ensure_ascii=False),
        "local_margin_trace_json": json.dumps(margin_trace, ensure_ascii=False),
        "top_paths_json": json.dumps(make_json_safe(top_paths), ensure_ascii=False),
    }
    if args.keep_prediction_text:
        out[args.text_col] = row.get(args.text_col, "")
    for depth in range(1, 8):
        out[f"predicted_level_{depth}_name"] = (
            path_names[depth - 1] if len(path_names) >= depth else ""
        )
        out[f"predicted_level_{depth}_key"] = (
            path_keys[depth - 1] if len(path_keys) >= depth else ""
        )
    return out


def build_pipeline_metrics(
    output: pd.DataFrame,
    *,
    mode: str,
    elapsed: float,
    embedding_dim: int,
    n_node_score_calls: int,
    n_similarity_dot_products: int,
    n_global_path_dot_products: int,
    n_rerank_pairs: int,
    resource_before: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    n_products = int(len(output))
    approx_ops = int(n_similarity_dot_products * embedding_dim)
    metrics = {
        "mode": mode,
        "n_products": n_products,
        "elapsed_seconds": float(elapsed),
        "seconds_per_product": float(elapsed / max(1, n_products)),
        "milliseconds_per_product": float(1000 * elapsed / max(1, n_products)),
        "throughput_products_per_second": float(n_products / elapsed)
        if elapsed > 0
        else np.nan,
        "embedding_dim": int(embedding_dim),
        "cost_unit": "similarity dot products and scalar multiply-adds, excluding transformer forward FLOPs",
        "node_score_calls": int(n_node_score_calls),
        "node_score_calls_per_product": float(n_node_score_calls / max(1, n_products)),
        "similarity_dot_products": int(n_similarity_dot_products),
        "similarity_dot_products_per_product": float(
            n_similarity_dot_products / max(1, n_products)
        ),
        "global_path_similarity_dot_products": int(n_global_path_dot_products),
        "global_path_similarity_dot_products_per_product": float(
            n_global_path_dot_products / max(1, n_products)
        ),
        "approx_scalar_multiply_adds": approx_ops,
        "approx_million_scalar_multiply_adds": float(approx_ops / 1_000_000),
        "approx_million_scalar_multiply_adds_per_product": float(
            approx_ops / max(1, n_products) / 1_000_000
        ),
        "rerank_pair_count": int(n_rerank_pairs),
        "rerank_pairs_per_product": float(n_rerank_pairs / max(1, n_products)),
        "mean_resolved_depth": numeric_mean(output, "resolved_depth"),
        "median_path_score": numeric_median(output, "predicted_path_score"),
        "median_top1_top2_margin": numeric_median(output, "top1_top2_margin"),
        "mean_top1_top2_margin": numeric_mean(output, "top1_top2_margin"),
        "median_mean_local_margin": numeric_median(output, "mean_local_margin"),
        "mean_path_mean_adjacent_similarity": numeric_mean(
            output, "path_mean_adjacent_similarity"
        ),
        "mean_final_to_ancestors_similarity": numeric_mean(
            output, "final_to_ancestors_mean_similarity"
        ),
        "n_unique_paths": int(output["predicted_taxonomy_path"].fillna("").nunique()),
        "top_path_share": top_share(output["predicted_taxonomy_path"]),
        "path_entropy": entropy(output["predicted_taxonomy_path"]),
        "resource_before": resource_before,
        "resource_after": process_resource_metrics(),
        **accelerator_memory_metrics(),
    }
    if "input_level_1_name" in output.columns:
        mask = output["input_level_1_name"].fillna("").astype(str) != ""
        if mask.any():
            metrics["l1_accuracy_if_label_available"] = float(
                (
                    output.loc[mask, "input_level_1_name"].astype(str)
                    == output.loc[mask, "l1_predicted_key"].astype(str)
                ).mean()
            )
    for col in ["top1_top2_margin", "mean_local_margin", "min_local_margin"]:
        values = pd.to_numeric(output.get(col, pd.Series(dtype=float)), errors="coerce")
        for threshold in [0.01, 0.03, 0.05]:
            metrics[f"share_{col}_lt_{threshold}"] = float((values < threshold).mean())
    for level in range(1, 8):
        col = f"predicted_level_{level}_name"
        if col in output.columns:
            non_empty = output[col].fillna("").astype(str)
            non_empty = non_empty[non_empty != ""]
            if len(non_empty):
                metrics[f"level_{level}_n_unique"] = int(non_empty.nunique())
                metrics[f"level_{level}_top_share"] = top_share(non_empty)
    if extra:
        metrics.update(extra)
    return metrics


def write_pipeline_outputs(result: dict[str, Any], output_dir: Path, name: str) -> None:
    predictions_path = output_dir / f"{name}_predictions.csv"
    metrics_path = output_dir / f"{name}_metrics.json"
    result["predictions"].to_csv(predictions_path, index=False)
    write_json(result["metrics"], metrics_path)


def write_summary_csv(results: dict[str, Any], output_path: Path) -> None:
    rows = []
    for name, metrics in results.get("pipelines", {}).items():
        rows.append(
            {
                "pipeline": name,
                "n_products": metrics.get("n_products"),
                "milliseconds_per_product": metrics.get("milliseconds_per_product"),
                "throughput_products_per_second": metrics.get(
                    "throughput_products_per_second"
                ),
                "approx_million_scalar_multiply_adds_per_product": metrics.get(
                    "approx_million_scalar_multiply_adds_per_product"
                ),
                "rerank_pairs_per_product": metrics.get("rerank_pairs_per_product"),
                "mean_resolved_depth": metrics.get("mean_resolved_depth"),
                "median_top1_top2_margin": metrics.get("median_top1_top2_margin"),
                "n_unique_paths": metrics.get("n_unique_paths"),
                "top_path_share": metrics.get("top_path_share"),
                "path_entropy": metrics.get("path_entropy"),
                "peak_gpu_memory_allocated_mb": metrics.get(
                    "peak_gpu_memory_allocated_mb"
                ),
                "rss_mb_after": metrics.get("resource_after", {}).get("rss_mb"),
                "max_rss_mb_after": metrics.get("resource_after", {}).get("max_rss_mb"),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def build_parent_to_children(nodes: pd.DataFrame) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = defaultdict(list)
    for row in nodes.to_dict("records"):
        node_key = str(row["node_key"])
        parent_key = str(row.get("parent_key", "") or "")
        if parent_key:
            mapping[parent_key].append(node_key)
    return {key: sorted(set(values)) for key, values in mapping.items()}


def aggregate(values: np.ndarray, mode: str) -> float:
    if len(values) == 0:
        return float("-inf")
    return float(np.mean(values) if mode == "mean" else np.max(values))


def add_rank_and_margins(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    top1 = float(rows[0]["score"])
    top2 = float(rows[1]["score"]) if len(rows) >= 2 else np.nan
    margin = top1 - top2 if np.isfinite(top2) else np.nan
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["sibling_margin"] = margin
        row["score_gap_to_best"] = top1 - float(row["score"])


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= len(scores):
        return np.argsort(scores)[::-1]
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]


def path_candidate_scores(paths: list[dict[str, Any]]) -> list[float]:
    scores = [
        float(path.get("rerank_score", path.get("cumulative_score", np.nan)))
        for path in paths
    ]
    return sorted([score for score in scores if np.isfinite(score)], reverse=True)


def deduplicate_paths(paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for path in sorted(
        paths, key=lambda item: float(item["cumulative_score"]), reverse=True
    ):
        key = tuple(path["path_keys"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def load_cross_encoder(model_name: str, device: str | None) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device=device, trust_remote_code=True)


def process_resource_metrics() -> dict[str, Any]:
    rss_mb = np.nan
    try:
        import psutil

        process = psutil.Process(os.getpid())
        rss_mb = process.memory_info().rss / (1024**2)
    except Exception:
        pass
    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss = float(usage.ru_maxrss)
    if platform.system() == "Darwin":
        max_rss_mb = max_rss / (1024**2)
    else:
        max_rss_mb = max_rss / 1024
    return {
        "rss_mb": float(rss_mb),
        "max_rss_mb": float(max_rss_mb),
        "user_cpu_seconds": float(usage.ru_utime),
        "system_cpu_seconds": float(usage.ru_stime),
    }


def reset_accelerator_peak_memory() -> None:
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            try:
                with torch.cuda.device(idx):
                    torch.cuda.reset_peak_memory_stats()
            except Exception as exc:
                print(f"WARNING: cannot reset CUDA peak memory for cuda:{idx}: {exc}")
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def accelerator_memory_metrics(prefix: str = "") -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {
            f"{prefix}cuda_available": False,
            f"{prefix}mps_available": False,
            f"{prefix}peak_gpu_memory_allocated_mb": np.nan,
            f"{prefix}peak_gpu_memory_reserved_mb": np.nan,
        }
    metrics: dict[str, Any] = {
        f"{prefix}cuda_available": bool(torch.cuda.is_available()),
        f"{prefix}cuda_device_count": int(torch.cuda.device_count())
        if torch.cuda.is_available()
        else 0,
        f"{prefix}mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }
    if torch.cuda.is_available():
        allocated = []
        reserved = []
        by_device = {}
        for idx in range(torch.cuda.device_count()):
            try:
                with torch.cuda.device(idx):
                    alloc_mb = torch.cuda.max_memory_allocated() / (1024**2)
                    reserv_mb = torch.cuda.max_memory_reserved() / (1024**2)
            except Exception as exc:
                by_device[f"cuda:{idx}"] = {"error": str(exc)}
                continue
            allocated.append(alloc_mb)
            reserved.append(reserv_mb)
            by_device[f"cuda:{idx}"] = {
                "peak_allocated_mb": float(alloc_mb),
                "peak_reserved_mb": float(reserv_mb),
            }
        metrics[f"{prefix}peak_gpu_memory_allocated_mb"] = (
            float(max(allocated)) if allocated else np.nan
        )
        metrics[f"{prefix}peak_gpu_memory_reserved_mb"] = (
            float(max(reserved)) if reserved else np.nan
        )
        metrics[f"{prefix}gpu_memory_by_device"] = by_device
    else:
        metrics[f"{prefix}peak_gpu_memory_allocated_mb"] = np.nan
        metrics[f"{prefix}peak_gpu_memory_reserved_mb"] = np.nan
    if metrics[f"{prefix}mps_available"]:
        try:
            metrics[f"{prefix}mps_current_allocated_mb"] = float(
                torch.mps.current_allocated_memory() / (1024**2)
            )
            metrics[f"{prefix}mps_driver_allocated_mb"] = float(
                torch.mps.driver_allocated_memory() / (1024**2)
            )
        except Exception:
            pass
    return metrics


def numeric_mean(df: pd.DataFrame, col: str) -> float:
    return (
        float(pd.to_numeric(df[col], errors="coerce").mean())
        if col in df.columns
        else np.nan
    )


def numeric_median(df: pd.DataFrame, col: str) -> float:
    return (
        float(pd.to_numeric(df[col], errors="coerce").median())
        if col in df.columns
        else np.nan
    )


def top_share(values: pd.Series) -> float:
    counts = values.fillna("").astype(str).value_counts(normalize=True)
    return float(counts.iloc[0]) if len(counts) else np.nan


def entropy(values: pd.Series) -> float:
    counts = values.fillna("").astype(str).value_counts(normalize=True)
    return float(-(counts * np.log(counts + 1e-12)).sum()) if len(counts) else np.nan


def is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def make_json_safe(value: Any) -> Any:
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


def copy_jsonable(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(make_json_safe(value), ensure_ascii=False))


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(make_json_safe(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def system_info() -> dict[str, Any]:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = (
            int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        )
        info["mps_available"] = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except Exception:
        pass
    return info


if __name__ == "__main__":
    try:
        main()
    finally:
        gc.collect()
