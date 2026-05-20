#!/usr/bin/env python3
"""Alternative unsupervised pipelines for taxonomy prediction below L1.

Subcommands:
- rerank-beam: rerank the top beam-search paths with a cross-encoder.
- global-path: retrieve global taxonomy paths, then optionally rerank or use a local LLM selector.
- cluster-l2: cluster products inside each L1 group and map clusters to L2 categories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENRICHMENT_DIR = ROOT / "dataset" / "category_enrichment"
DEFAULT_BEAM_INPUT = ROOT / "data" / "l1_to_l7_beam_predictions.csv"
DEFAULT_PRODUCT_INPUT = ROOT / "dataset" / "preprocessed_lv2.parquet"
DEFAULT_CACHE_DIR = ROOT / "data" / "alternative_pipeline_cache"


@dataclass(slots=True)
class RerankBeamConfig:
    input_path: str = str(DEFAULT_BEAM_INPUT)
    output_path: str = str(ROOT / "data" / "l1_to_l7_beam_reranked.csv")
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    text_col: str = "text"
    id_col: str = "hashed_external_id"
    beam_json_col: str = "beam_paths_json"
    top_k: int = 10
    chunksize: int = 512
    batch_size: int = 64
    device: str | None = None
    sample_size: int | None = None
    random_seed: int = 42
    dry_run: bool = False


@dataclass(slots=True)
class GlobalPathConfig:
    input_path: str = str(DEFAULT_BEAM_INPUT)
    input_format: str = "csv"
    output_path: str = str(ROOT / "data" / "l1_to_l7_global_path_predictions.csv")
    enrichment_dir: str = str(DEFAULT_ENRICHMENT_DIR)
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "Qwen/Qwen2.5-3B-Instruct"
    selector: str = "embedding"
    text_col: str = "text"
    id_col: str = "hashed_external_id"
    l1_col: str = "predicted_level_1_name"
    output_top_k: int = 20
    retrieve_top_k: int = 50
    min_depth: int = 2
    max_depth: int = 7
    batch_size: int = 64
    reranker_batch_size: int = 64
    device: str | None = None
    reranker_device: str | None = None
    llm_device: str | None = None
    llm_max_new_tokens: int = 8
    cache_dir: str = str(DEFAULT_CACHE_DIR)
    overwrite_cache: bool = False
    sample_size: int | None = None
    random_seed: int = 42
    dry_run: bool = False


@dataclass(slots=True)
class ClusterL2Config:
    input_path: str = str(DEFAULT_PRODUCT_INPUT)
    input_format: str = "parquet"
    output_path: str = str(ROOT / "data" / "l2_cluster_predictions.csv")
    cluster_summary_output: str = str(ROOT / "data" / "l2_cluster_summary.csv")
    enrichment_dir: str = str(DEFAULT_ENRICHMENT_DIR)
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    text_col: str = "text"
    id_col: str = "hashed_external_id"
    l1_col: str = "level_1_name"
    k_mode: str = "l2_children"
    max_clusters_per_l1: int = 60
    min_products_per_l1: int = 10
    batch_size: int = 64
    device: str | None = None
    cache_dir: str = str(DEFAULT_CACHE_DIR)
    overwrite_cache: bool = False
    sample_size: int | None = None
    random_seed: int = 42
    dry_run: bool = False


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "rerank-beam":
        run_rerank_beam(RerankBeamConfig(**vars_without_command(args)))
    elif args.command == "global-path":
        run_global_path(GlobalPathConfig(**vars_without_command(args)))
    elif args.command == "cluster-l2":
        run_cluster_l2(ClusterL2Config(**vars_without_command(args)))
    else:
        parser.error("Unknown command.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    rerank = subparsers.add_parser(
        "rerank-beam",
        help="Pipeline 2: rerank existing beam-search candidate paths with a cross-encoder.",
    )
    add_dataclass_args(rerank, RerankBeamConfig())

    global_path = subparsers.add_parser(
        "global-path",
        help="Pipeline 3: retrieve global taxonomy paths and select by embedding, reranker, or local LLM.",
    )
    add_dataclass_args(global_path, GlobalPathConfig())
    global_path.add_argument(
        "--selector",
        choices=["embedding", "reranker", "llm"],
        default=GlobalPathConfig.selector,
        help="How to choose among retrieved top-k paths.",
    )
    global_path.add_argument(
        "--input-format",
        choices=["csv", "parquet"],
        default=GlobalPathConfig.input_format,
    )

    cluster = subparsers.add_parser(
        "cluster-l2",
        help="Pipeline 5: cluster products inside each L1 group and map clusters to L2 categories.",
    )
    add_dataclass_args(cluster, ClusterL2Config())
    cluster.add_argument(
        "--input-format",
        choices=["csv", "parquet"],
        default=ClusterL2Config.input_format,
    )
    cluster.add_argument(
        "--k-mode",
        choices=["l2_children", "sqrt", "auto"],
        default=ClusterL2Config.k_mode,
        help="How to choose the number of clusters inside each L1 group.",
    )
    return parser


def add_dataclass_args(parser: argparse.ArgumentParser, cfg: object) -> None:
    for field_name, value in asdict(cfg).items():
        if field_name in {"selector", "input_format", "k_mode"}:
            continue
        arg = "--" + field_name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                parser.add_argument(f"--no-{field_name.replace('_', '-')}", action="store_false", dest=field_name)
            else:
                parser.add_argument(arg, action="store_true", dest=field_name)
        elif isinstance(value, int) or value is None and field_name in {
            "top_k",
            "chunksize",
            "batch_size",
            "sample_size",
            "random_seed",
            "output_top_k",
            "retrieve_top_k",
            "min_depth",
            "max_depth",
            "reranker_batch_size",
            "llm_max_new_tokens",
            "max_clusters_per_l1",
            "min_products_per_l1",
        }:
            parser.add_argument(arg, type=int, default=value)
        else:
            parser.add_argument(arg, default=value)


def vars_without_command(args: argparse.Namespace) -> dict[str, Any]:
    payload = vars(args).copy()
    payload.pop("command", None)
    return payload


# ---------------------------------------------------------------------------
# Pipeline 2: beam candidates reranking
# ---------------------------------------------------------------------------


def run_rerank_beam(config: RerankBeamConfig) -> None:
    input_path = Path(config.input_path)
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = pd.read_csv(input_path, nrows=0).columns.tolist()
    required = {config.text_col, config.beam_json_col}
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {sorted(missing)}")

    if config.dry_run:
        preview = pd.read_csv(input_path, nrows=3)
        print(f"Input columns: {columns}")
        print(preview[[config.id_col, config.text_col, config.beam_json_col]].head().to_string(index=False))
        return

    reranker = load_cross_encoder(config.reranker_model, device=config.device)
    rng = np.random.default_rng(config.random_seed)
    rows_seen = 0
    rows_written = 0
    wrote_header = False

    for chunk in pd.read_csv(input_path, chunksize=config.chunksize):
        if config.sample_size is not None:
            remaining = max(0, config.sample_size - rows_seen)
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.sample(remaining, random_state=int(rng.integers(0, 1_000_000)))

        output_rows = []
        for row in chunk.to_dict("records"):
            product_text = clean_text(row.get(config.text_col, ""))
            paths = parse_beam_paths(row.get(config.beam_json_col, ""))[: config.top_k]
            candidates = [beam_path_to_candidate(path) for path in paths]
            if not candidates:
                output_rows.append({**row, "reranked_taxonomy_path": "", "reranked_score": np.nan, "reranked_candidates_json": "[]"})
                continue

            pairs = [(product_text, candidate["candidate_text"]) for candidate in candidates]
            scores = predict_cross_encoder(reranker, pairs, batch_size=config.batch_size)
            ranked = rank_candidates(candidates, scores)
            best = ranked[0]
            output_rows.append(
                {
                    **row,
                    "reranked_taxonomy_path": best["taxonomy_path"],
                    "reranked_taxonomy_key_path": best["taxonomy_key_path"],
                    "reranked_score": best["rerank_score"],
                    "reranked_resolved_depth": best["resolved_depth"],
                    "reranked_candidates_json": json.dumps(ranked, ensure_ascii=False),
                }
            )

        out = pd.DataFrame(output_rows)
        out.to_csv(output_path, index=False, mode="a", header=not wrote_header)
        wrote_header = True
        rows_seen += len(chunk)
        rows_written += len(out)
        print(f"Reranked {rows_written:,} rows...", flush=True)

    save_json(asdict(config), output_path.with_suffix(".config.json"))
    print(f"Reranked predictions saved to {output_path}")


# ---------------------------------------------------------------------------
# Pipeline 3: global path retrieval plus reranker / LLM selection
# ---------------------------------------------------------------------------


def run_global_path(config: GlobalPathConfig) -> None:
    products = load_products(
        Path(config.input_path),
        input_format=config.input_format,
        text_col=config.text_col,
        sample_size=config.sample_size,
        random_seed=config.random_seed,
    )
    ensure_columns(products, [config.id_col, config.text_col, config.l1_col])
    paths = load_taxonomy_path_candidates(
        Path(config.enrichment_dir),
        min_depth=config.min_depth,
        max_depth=config.max_depth,
    )
    print(f"Products: {len(products):,}")
    print(f"Path candidates L{config.min_depth}-L{config.max_depth}: {len(paths):,}")
    if config.dry_run:
        print(paths.head(10).to_string(index=False))
        return

    embedder = SentenceTransformer(config.embedding_model, device=config.device, trust_remote_code=True)
    path_embeddings = load_or_encode_texts(
        texts=paths["candidate_text"].tolist(),
        model=embedder,
        model_name=config.embedding_model,
        cache_dir=Path(config.cache_dir),
        cache_name="global_path_candidates",
        cache_fingerprint=frame_fingerprint(paths[["node_key", "candidate_text"]]),
        batch_size=config.batch_size,
        device=config.device,
        overwrite=config.overwrite_cache,
    )
    product_embeddings = encode_texts(
        embedder,
        products[config.text_col].fillna("").astype(str).tolist(),
        batch_size=config.batch_size,
        device=config.device,
    )

    reranker = None
    llm = None
    if config.selector == "reranker":
        reranker = load_cross_encoder(config.reranker_model, device=config.reranker_device)
    elif config.selector == "llm":
        llm = load_local_llm(config.llm_model, device=config.llm_device)

    grouped = build_l1_candidate_slices(paths, path_embeddings)
    output_rows: list[dict[str, Any]] = []
    for product, query_embedding in zip(products.to_dict("records"), product_embeddings, strict=True):
        l1 = clean_text(product.get(config.l1_col, ""))
        candidate_frame, candidate_embeddings = grouped.get(l1, (None, None))
        if candidate_frame is None or candidate_embeddings is None or len(candidate_frame) == 0:
            output_rows.append(empty_global_output(product, config, reason="no_l1_candidates"))
            continue

        sims = candidate_embeddings @ query_embedding
        top_indices = topk_indices(sims, min(config.retrieve_top_k, len(sims)))
        retrieved = candidate_frame.iloc[top_indices].copy().reset_index(drop=True)
        retrieved["embedding_score"] = sims[top_indices]
        retrieved_candidates = retrieved.to_dict("records")

        if config.selector == "embedding":
            selected = retrieved_candidates[0]
            ranked = retrieved_candidates[: config.output_top_k]
        elif config.selector == "reranker":
            assert reranker is not None
            pairs = [(str(product[config.text_col]), str(c["candidate_text"])) for c in retrieved_candidates]
            scores = predict_cross_encoder(reranker, pairs, batch_size=config.reranker_batch_size)
            ranked = rank_candidates(retrieved_candidates, scores)[: config.output_top_k]
            selected = ranked[0]
        else:
            assert llm is not None
            llm_top = retrieved_candidates[: min(config.output_top_k, len(retrieved_candidates))]
            selected_idx = select_with_llm(llm, str(product[config.text_col]), llm_top, max_new_tokens=config.llm_max_new_tokens)
            ranked = llm_top
            selected = llm_top[selected_idx]
            selected["llm_selected_rank"] = selected_idx + 1

        output_rows.append(build_global_output(product, config, selected, ranked))

    out = pd.DataFrame(output_rows)
    Path(config.output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.output_path, index=False)
    save_json(asdict(config), Path(config.output_path).with_suffix(".config.json"))
    print(f"Global path predictions saved to {config.output_path}")


# ---------------------------------------------------------------------------
# Pipeline 5: product clustering inside each L1, then L2 mapping
# ---------------------------------------------------------------------------


def run_cluster_l2(config: ClusterL2Config) -> None:
    products = load_products(
        Path(config.input_path),
        input_format=config.input_format,
        text_col=config.text_col,
        sample_size=config.sample_size,
        random_seed=config.random_seed,
    )
    ensure_columns(products, [config.id_col, config.text_col, config.l1_col])
    level2_prototypes = load_level2_prototypes(Path(config.enrichment_dir))
    print(f"Products: {len(products):,}")
    print(f"L2 prototype rows: {len(level2_prototypes):,}")
    print(f"L1 groups in products: {products[config.l1_col].nunique():,}")
    if config.dry_run:
        print(level2_prototypes.head(10).to_string(index=False))
        return

    embedder = SentenceTransformer(config.embedding_model, device=config.device, trust_remote_code=True)
    product_embeddings = load_or_encode_texts(
        texts=products[config.text_col].fillna("").astype(str).tolist(),
        model=embedder,
        model_name=config.embedding_model,
        cache_dir=Path(config.cache_dir),
        cache_name="cluster_l2_products",
        cache_fingerprint=series_fingerprint(products[config.id_col].astype(str) + products[config.text_col].astype(str)),
        batch_size=config.batch_size,
        device=config.device,
        overwrite=config.overwrite_cache,
    )
    proto_embeddings = load_or_encode_texts(
        texts=level2_prototypes["prototype_text"].tolist(),
        model=embedder,
        model_name=config.embedding_model,
        cache_dir=Path(config.cache_dir),
        cache_name="cluster_l2_prototypes",
        cache_fingerprint=frame_fingerprint(level2_prototypes[["node_key", "prototype_text"]]),
        batch_size=config.batch_size,
        device=config.device,
        overwrite=config.overwrite_cache,
    )
    l2_centroids = build_l2_centroids(level2_prototypes, proto_embeddings)

    prediction_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    products_work = products.reset_index(drop=True).copy()
    products_work["_row_idx"] = np.arange(len(products_work))

    for l1_name, group in products_work.groupby(config.l1_col, sort=True):
        l1_key = clean_text(l1_name)
        l2_info = l2_centroids.get(l1_key)
        if l2_info is None:
            for row in group.to_dict("records"):
                prediction_rows.append(build_cluster_output(row, config, reason="no_l2_children"))
            continue

        row_indices = group["_row_idx"].to_numpy(dtype=int)
        group_embeddings = product_embeddings[row_indices]
        n_products = len(group)
        if n_products < config.min_products_per_l1:
            assigned = assign_products_directly_to_l2(group_embeddings, l2_info)
            for row, assignment in zip(group.to_dict("records"), assigned, strict=True):
                prediction_rows.append(build_cluster_output(row, config, assignment=assignment, reason="direct_small_l1"))
            continue

        n_clusters = choose_cluster_count(
            n_products=n_products,
            n_l2_children=len(l2_info["node_keys"]),
            mode=config.k_mode,
            max_clusters=config.max_clusters_per_l1,
        )
        if n_clusters <= 1:
            assigned = assign_products_directly_to_l2(group_embeddings, l2_info)
            for row, assignment in zip(group.to_dict("records"), assigned, strict=True):
                prediction_rows.append(build_cluster_output(row, config, assignment=assignment, reason="direct_single_cluster"))
            continue

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=config.random_seed,
            batch_size=max(256, min(4096, n_products)),
            n_init="auto",
        )
        cluster_labels = kmeans.fit_predict(group_embeddings)
        cluster_centers = normalize_rows(kmeans.cluster_centers_.astype(np.float32))
        cluster_assignments = assign_products_directly_to_l2(cluster_centers, l2_info)

        for cluster_id, assignment in enumerate(cluster_assignments):
            cluster_size = int(np.sum(cluster_labels == cluster_id))
            cluster_rows.append(
                {
                    "level_1_name": l1_key,
                    "cluster_id": cluster_id,
                    "cluster_size": cluster_size,
                    "predicted_level_2_name": assignment["level_2_name"],
                    "predicted_level_2_key": assignment["level_2_key"],
                    "cluster_l2_score": assignment["score"],
                    "cluster_l2_margin": assignment["margin"],
                    "n_clusters_l1": n_clusters,
                    "n_l2_children": len(l2_info["node_keys"]),
                }
            )

        group_records = group.to_dict("records")
        for row, cluster_id in zip(group_records, cluster_labels, strict=True):
            assignment = dict(cluster_assignments[int(cluster_id)])
            assignment["cluster_id"] = int(cluster_id)
            assignment["n_clusters_l1"] = int(n_clusters)
            prediction_rows.append(build_cluster_output(row, config, assignment=assignment, reason="cluster"))

        print(f"L1={l1_key}: products={n_products:,}, L2 children={len(l2_info['node_keys'])}, clusters={n_clusters}", flush=True)

    out = pd.DataFrame(prediction_rows).drop(columns=["_row_idx"], errors="ignore")
    summary = pd.DataFrame(cluster_rows)
    Path(config.output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.output_path, index=False)
    summary.to_csv(config.cluster_summary_output, index=False)
    save_json(asdict(config), Path(config.output_path).with_suffix(".config.json"))
    print(f"L2 cluster predictions saved to {config.output_path}")
    print(f"L2 cluster summary saved to {config.cluster_summary_output}")


# ---------------------------------------------------------------------------
# Shared data loading and scoring helpers
# ---------------------------------------------------------------------------


def load_products(
    path: Path,
    *,
    input_format: str,
    text_col: str,
    sample_size: int | None,
    random_seed: int,
) -> pd.DataFrame:
    if input_format == "parquet":
        df = pd.read_parquet(path)
    else:
        # With a sample, read only the first rows to avoid loading the full
        # 1GB+ beam prediction file during smoke tests.
        df = pd.read_csv(path, nrows=sample_size, low_memory=False)
    if text_col not in df.columns:
        raise ValueError(f"Missing text column '{text_col}' in {path}.")
    work = df.copy()
    work[text_col] = work[text_col].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    work = work[work[text_col] != ""].reset_index(drop=True)
    if input_format != "csv" and sample_size is not None and len(work) > sample_size:
        work = work.sample(sample_size, random_state=random_seed).reset_index(drop=True)
    return work


def load_taxonomy_path_candidates(enrichment_dir: Path, *, min_depth: int, max_depth: int) -> pd.DataFrame:
    frames = []
    for depth in range(min_depth, max_depth + 1):
        path = enrichment_dir / f"level_{depth}_categories.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No level category files found in {enrichment_dir}.")
    df = pd.concat(frames, ignore_index=True)
    required = {"node_key", "depth", "category_name", "taxonomy_path", "parent_key"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Category files missing columns: {sorted(missing)}")
    for col in ["node_key", "category_name", "taxonomy_path", "parent_key"]:
        df[col] = df[col].fillna("").astype(str).map(clean_text)
    df["level_1_name"] = df["taxonomy_path"].map(lambda x: x.split(" > ")[0] if x else "")
    df["candidate_text"] = df.apply(build_path_candidate_text, axis=1)
    return df[
        ["node_key", "depth", "category_name", "taxonomy_path", "parent_key", "level_1_name", "candidate_text"]
    ].drop_duplicates("node_key").reset_index(drop=True)


def build_path_candidate_text(row: pd.Series) -> str:
    parts = [
        f"Taxonomy path: {clean_text(row.get('taxonomy_path', ''))}.",
        f"Category name: {clean_text(row.get('category_name', ''))}.",
    ]
    for col, label in [
        ("children_names", "Children"),
        ("descendant_names", "Descendants"),
        ("enriched_description", "Description"),
        ("children_summary", "Children summary"),
        ("descendants_summary", "Descendants summary"),
    ]:
        value = clean_text(row.get(col, ""))
        if value:
            parts.append(f"{label}: {value}.")
    return " ".join(parts)


def load_level2_prototypes(enrichment_dir: Path) -> pd.DataFrame:
    path = enrichment_dir / "level_2_reference_texts.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"node_key", "category_name", "prototype_text"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"L2 reference text file missing columns: {sorted(missing)}")
    for col in ["node_key", "category_name", "prototype_text"]:
        df[col] = df[col].fillna("").astype(str).map(clean_text)
    df = df[df["prototype_text"] != ""].drop_duplicates(["node_key", "prototype_text"])
    df["level_1_name"] = df["node_key"].map(lambda x: x.split(" > ")[0] if x else "")
    return df.reset_index(drop=True)


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    *,
    batch_size: int,
    device: str | None,
) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=device,
    ).astype(np.float32)


def load_or_encode_texts(
    *,
    texts: list[str],
    model: SentenceTransformer,
    model_name: str,
    cache_dir: Path,
    cache_name: str,
    cache_fingerprint: str,
    batch_size: int,
    device: str | None,
    overwrite: bool,
) -> np.ndarray:
    key = slugify(model_name) + "__" + cache_name + "__" + cache_fingerprint[:12]
    path = cache_dir / key / "embeddings.npy"
    if path.exists() and not overwrite:
        print(f"Loading cached embeddings from {path}")
        return np.load(path).astype(np.float32)
    embeddings = encode_texts(model, texts, batch_size=batch_size, device=device)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    return embeddings


def build_l1_candidate_slices(paths: pd.DataFrame, embeddings: np.ndarray) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    grouped: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    for l1_name, group in paths.groupby("level_1_name", sort=False):
        indices = group.index.to_numpy(dtype=int)
        grouped[str(l1_name)] = (group.reset_index(drop=True), embeddings[indices])
    return grouped


def build_l2_centroids(prototypes: pd.DataFrame, embeddings: np.ndarray) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    work = prototypes.reset_index(drop=True).copy()
    work["_idx"] = np.arange(len(work))
    for l1_name, l1_group in work.groupby("level_1_name"):
        node_keys: list[str] = []
        node_names: list[str] = []
        centroids: list[np.ndarray] = []
        for node_key, group in l1_group.groupby("node_key"):
            indices = group["_idx"].to_numpy(dtype=int)
            centroid = normalize_rows(embeddings[indices].mean(axis=0, keepdims=True))[0]
            node_keys.append(str(node_key))
            node_names.append(str(group["category_name"].iloc[0]))
            centroids.append(centroid)
        if centroids:
            output[str(l1_name)] = {
                "node_keys": node_keys,
                "node_names": node_names,
                "embeddings": np.vstack(centroids).astype(np.float32),
            }
    return output


def assign_products_directly_to_l2(query_embeddings: np.ndarray, l2_info: dict[str, Any]) -> list[dict[str, Any]]:
    sims = query_embeddings @ l2_info["embeddings"].T
    best_indices = np.argmax(sims, axis=1)
    assignments = []
    for row_idx, best_idx in enumerate(best_indices):
        scores = sims[row_idx]
        best_score = float(scores[best_idx])
        if len(scores) > 1:
            second = float(np.partition(scores, -2)[-2])
        else:
            second = float("nan")
        assignments.append(
            {
                "level_2_key": l2_info["node_keys"][int(best_idx)],
                "level_2_name": l2_info["node_names"][int(best_idx)],
                "score": best_score,
                "margin": best_score - second if not math.isnan(second) else float("nan"),
            }
        )
    return assignments


def choose_cluster_count(*, n_products: int, n_l2_children: int, mode: str, max_clusters: int) -> int:
    if n_products <= 1 or n_l2_children <= 1:
        return 1
    sqrt_k = max(2, int(round(math.sqrt(n_products))))
    if mode == "l2_children":
        k = n_l2_children
    elif mode == "sqrt":
        k = sqrt_k
    else:
        k = min(n_l2_children, sqrt_k)
    return max(1, min(k, n_products, max_clusters))


def build_cluster_output(
    row: dict[str, Any],
    config: ClusterL2Config,
    *,
    assignment: dict[str, Any] | None = None,
    reason: str,
) -> dict[str, Any]:
    assignment = assignment or {}
    return {
        config.id_col: row.get(config.id_col, ""),
        "text": row.get(config.text_col, ""),
        "level_1_name": row.get(config.l1_col, ""),
        "predicted_level_2_name": assignment.get("level_2_name", ""),
        "predicted_level_2_key": assignment.get("level_2_key", ""),
        "cluster_id": assignment.get("cluster_id", ""),
        "n_clusters_l1": assignment.get("n_clusters_l1", ""),
        "l2_cluster_score": assignment.get("score", np.nan),
        "l2_cluster_margin": assignment.get("margin", np.nan),
        "prediction_mode": reason,
    }


def empty_global_output(product: dict[str, Any], config: GlobalPathConfig, *, reason: str) -> dict[str, Any]:
    return {
        config.id_col: product.get(config.id_col, ""),
        config.text_col: product.get(config.text_col, ""),
        "input_l1_for_global_path": product.get(config.l1_col, ""),
        "global_path_prediction": "",
        "global_path_key": "",
        "global_path_depth": "",
        "global_path_score": np.nan,
        "global_path_selector": config.selector,
        "global_path_reason": reason,
        "global_path_top_candidates_json": "[]",
    }


def build_global_output(
    product: dict[str, Any],
    config: GlobalPathConfig,
    selected: dict[str, Any],
    ranked: list[dict[str, Any]],
) -> dict[str, Any]:
    score = selected.get("rerank_score", selected.get("embedding_score", np.nan))
    return {
        config.id_col: product.get(config.id_col, ""),
        config.text_col: product.get(config.text_col, ""),
        "input_l1_for_global_path": product.get(config.l1_col, ""),
        "global_path_prediction": selected.get("taxonomy_path", ""),
        "global_path_key": selected.get("node_key", ""),
        "global_path_depth": selected.get("depth", ""),
        "global_path_score": float(score) if score != "" and not pd.isna(score) else np.nan,
        "global_path_selector": config.selector,
        "global_path_reason": "ok",
        "global_path_top_candidates_json": json.dumps(make_json_safe(ranked), ensure_ascii=False),
    }


def parse_beam_paths(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, str) or not value:
        return []
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def beam_path_to_candidate(path: dict[str, Any]) -> dict[str, Any]:
    names = [str(x) for x in path.get("path_names", [])]
    keys = [str(x) for x in path.get("path_keys", [])]
    trace = path.get("score_trace", [])
    taxonomy_path = " > ".join(names)
    return {
        "taxonomy_path": taxonomy_path,
        "taxonomy_key_path": " || ".join(keys),
        "resolved_depth": int(path.get("resolved_depth", len(names))),
        "embedding_path_score": float(path.get("cumulative_score", np.nan)),
        "score_trace": trace if isinstance(trace, list) else [],
        "candidate_text": build_rerank_candidate_text(taxonomy_path, trace),
    }


def build_rerank_candidate_text(taxonomy_path: str, score_trace: object) -> str:
    return f"Taxonomy path: {taxonomy_path}. Candidate category hierarchy for an ecommerce product."


def rank_candidates(candidates: list[dict[str, Any]], scores: Iterable[float]) -> list[dict[str, Any]]:
    ranked = []
    for candidate, score in zip(candidates, scores, strict=True):
        item = dict(candidate)
        item["rerank_score"] = float(score)
        ranked.append(item)
    ranked.sort(key=lambda item: float(item["rerank_score"]), reverse=True)
    return ranked


def load_cross_encoder(model_name: str, *, device: str | None):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError("sentence-transformers CrossEncoder is required for reranking.") from exc
    return CrossEncoder(model_name, device=device, trust_remote_code=True)


def predict_cross_encoder(model: Any, pairs: list[tuple[str, str]], *, batch_size: int) -> list[float]:
    if not pairs:
        return []
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    return np.asarray(scores, dtype=float).reshape(-1).tolist()


def load_local_llm(model_name: str, *, device: str | None) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers and torch are required for --selector llm.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if device:
        kwargs["device_map"] = {"": device}
    else:
        kwargs["device_map"] = "auto"
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return {"tokenizer": tokenizer, "model": model}


def select_with_llm(llm: dict[str, Any], product_text: str, candidates: list[dict[str, Any]], *, max_new_tokens: int) -> int:
    tokenizer = llm["tokenizer"]
    model = llm["model"]
    options = "\n".join(
        f"{idx + 1}. {candidate.get('taxonomy_path', '')}"
        for idx, candidate in enumerate(candidates)
    )
    prompt = (
        "You are selecting an ecommerce taxonomy path. "
        "Choose exactly one candidate number. Do not explain.\n\n"
        f"Product:\n{product_text[:1600]}\n\n"
        f"Candidates:\n{options}\n\n"
        "Best candidate number:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    match = re.search(r"\d+", generated)
    if not match:
        return 0
    idx = int(match.group(0)) - 1
    return max(0, min(idx, len(candidates) - 1))


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= len(scores):
        return np.argsort(scores)[::-1]
    indices = np.argpartition(scores, -k)[-k:]
    return indices[np.argsort(scores[indices])[::-1]]


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "model"


def frame_fingerprint(df: pd.DataFrame) -> str:
    digest = hashlib.sha1()
    digest.update(pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).values.tobytes())
    return digest.hexdigest()


def series_fingerprint(series: pd.Series) -> str:
    digest = hashlib.sha1()
    digest.update(pd.util.hash_pandas_object(series.reset_index(drop=True), index=False).values.tobytes())
    return digest.hexdigest()


def make_json_safe(value: Any) -> Any:
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): make_json_safe(val) for key, val in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def save_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
