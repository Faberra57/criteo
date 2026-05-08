#!/usr/bin/env python3
"""Pipeline 1: hierarchical L2-L7 decoding from shared embeddings.

Modes:
- greedy: one best child at each level;
- beam: keep top beam paths;
- beam-rerank: generate beam paths, then rerank them with a cross-encoder.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_EMBEDDINGS_DIR = "dataset/shared_l1_l7_embeddings_Jasper-Token-Compression-600M"
DEFAULT_OUTPUT_PATH = "data/pipeline1_hierarchical_predictions_jasper"


def main() -> None:
    args = parse_args()
    reset_cuda_peak_memory()
    started = time.perf_counter()
    load_started = time.perf_counter()
    data = load_shared_embeddings(Path(args.embeddings_dir))
    load_seconds = time.perf_counter() - load_started
    if args.dry_run:
        print_summary(data)
        return

    reranker_load_seconds = 0.0
    reranker = None
    if args.mode == "beam-rerank":
        reranker_started = time.perf_counter()
        reranker = load_cross_encoder(args.reranker_model, args.reranker_device)
        reranker_load_seconds = time.perf_counter() - reranker_started

    index_started = time.perf_counter()
    index = PrototypeIndex(
        nodes=data["nodes"],
        prototypes=data["prototypes"],
        prototype_embeddings=data["prototype_embeddings"],
        aggregation=args.score_aggregation,
    )
    index_build_seconds = time.perf_counter() - index_started
    products = data["products"]
    product_embeddings = data["product_embeddings"]
    if args.sample_size is not None and len(products) > args.sample_size:
        products = products.sample(
            args.sample_size, random_state=args.random_seed
        ).sort_index()
        product_embeddings = product_embeddings[products.index.to_numpy()]
        products = products.reset_index(drop=True)

    output_rows: list[dict[str, Any]] = []
    n_node_score_calls = 0
    n_similarity_dot_products = 0
    n_rerank_pairs = 0
    decode_started = time.perf_counter()
    for row, embedding in zip(
        products.to_dict("records"), product_embeddings, strict=True
    ):
        l1_scores = index.score_nodes(embedding, index.l1_node_keys)
        product_node_calls = len(index.l1_node_keys)
        product_dot_products = index.prototype_count(index.l1_node_keys)
        start_l1 = l1_scores[0]["node_key"] if l1_scores else ""

        if args.mode == "greedy":
            paths, scored_nodes, scored_dots = greedy_decode(
                index, embedding, start_l1, max_depth=args.max_depth
            )
        else:
            paths, scored_nodes, scored_dots = beam_decode(
                index,
                embedding,
                start_l1,
                max_depth=args.max_depth,
                beam_width=args.beam_width,
                children_top_k=args.children_top_k,
                output_top_paths=args.output_top_paths,
            )
        product_node_calls += scored_nodes
        product_dot_products += scored_dots
        n_node_score_calls += product_node_calls
        n_similarity_dot_products += product_dot_products

        if args.mode == "beam-rerank" and paths:
            n_rerank_pairs += len(paths)
            paths = rerank_paths(
                reranker,
                product_text=str(row.get(args.text_col, "")),
                paths=paths,
                batch_size=args.reranker_batch_size,
            )

        best_path = paths[0] if paths else empty_path(start_l1)
        output_rows.append(
            build_output_row(
                row,
                id_col=args.id_col,
                text_col=args.text_col,
                original_l1_col=args.original_l1_col,
                mode=args.mode,
                l1_scores=l1_scores[: args.l1_top_k],
                best_path=best_path,
                top_paths=paths[: args.output_top_paths],
                node_score_calls=product_node_calls,
                similarity_dot_products=product_dot_products,
                rerank_pair_count=len(paths) if args.mode == "beam-rerank" else 0,
                path_internal_coherence=index.path_internal_coherence(
                    path_keys=best_path.get("path_keys", [])
                ),
            )
        )
    decode_seconds = time.perf_counter() - decode_started

    output = pd.DataFrame(output_rows)
    output_path = Path(args.output_path + f"_{args.mode}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_started = time.perf_counter()
    output.to_csv(output_path, index=False)
    write_seconds = time.perf_counter() - write_started
    metrics = build_metrics(
        output,
        mode=args.mode,
        elapsed=time.perf_counter() - started,
        n_products=len(output),
        n_node_score_calls=n_node_score_calls,
        n_similarity_dot_products=n_similarity_dot_products,
        n_rerank_pairs=n_rerank_pairs,
        embedding_dim=int(product_embeddings.shape[1]),
        phase_seconds={
            "load_embeddings_seconds": load_seconds,
            "index_build_seconds": index_build_seconds,
            "reranker_load_seconds": reranker_load_seconds,
            "decode_seconds": decode_seconds,
            "write_outputs_seconds": write_seconds,
        },
        resource_metrics=embedding_resource_metrics(
            data, product_embeddings=product_embeddings
        ),
    )
    write_json(
        {"args": vars(args), "metrics": metrics},
        output_path.with_suffix(".metrics.json"),
    )
    print(f"Predictions saved to {output_path}")
    print(json.dumps(metrics, indent=2))


class PrototypeIndex:
    def __init__(
        self,
        *,
        nodes: pd.DataFrame,
        prototypes: pd.DataFrame,
        prototype_embeddings: np.ndarray,
        aggregation: str,
    ) -> None:
        self.nodes = nodes.reset_index(drop=True).copy()
        self.prototypes = prototypes.reset_index(drop=True).copy()
        self.prototype_embeddings = prototype_embeddings.astype(np.float32)
        self.aggregation = aggregation
        self.node_meta = self.nodes.set_index("node_key").to_dict("index")
        self.parent_to_children = build_parent_to_children(self.nodes)
        self.l1_node_keys = sorted(
            self.nodes.loc[self.nodes["depth"] == 1, "node_key"].tolist()
        )
        self.node_to_indices = (
            self.prototypes.reset_index()
            .groupby("node_key")["index"]
            .apply(list)
            .to_dict()
        )
        self.node_to_centroid = self._build_centroids()

    def _build_centroids(self) -> dict[str, np.ndarray]:
        centroids: dict[str, np.ndarray] = {}
        for node_key, indices in self.node_to_indices.items():
            matrix = self.prototype_embeddings[np.asarray(indices, dtype=int)]
            centroid = matrix.mean(axis=0, keepdims=True)
            centroids[node_key] = normalize_rows(centroid)[0]
        return centroids

    def score_nodes(
        self, query: np.ndarray, node_keys: list[str]
    ) -> list[dict[str, Any]]:
        rows = []
        for node_key in node_keys:
            indices = self.node_to_indices.get(node_key, [])
            if not indices:
                continue
            sims = self.prototype_embeddings[np.asarray(indices)] @ query
            score = aggregate(sims, self.aggregation)
            meta = self.node_meta.get(node_key, {})
            rows.append(
                {
                    "node_key": node_key,
                    "node_name": meta.get("category_name", node_key),
                    "depth": int(meta.get("depth", 0)),
                    "score": score,
                    "n_prototypes": len(indices),
                }
            )
        rows.sort(key=lambda x: float(x["score"]), reverse=True)
        add_rank_and_margins(rows)
        return rows

    def children(self, parent_key: str) -> list[str]:
        return self.parent_to_children.get(parent_key, [])

    def prototype_count(self, node_keys: list[str]) -> int:
        return int(
            sum(len(self.node_to_indices.get(node_key, [])) for node_key in node_keys)
        )

    def path_internal_coherence(self, path_keys: list[str]) -> dict[str, float]:
        vectors = [
            self.node_to_centroid[key]
            for key in path_keys
            if key in self.node_to_centroid
        ]
        if len(vectors) < 2:
            return {
                "path_mean_adjacent_similarity": float("nan"),
                "path_min_adjacent_similarity": float("nan"),
                "final_to_ancestors_mean_similarity": float("nan"),
            }
        adjacent = [
            float(vectors[idx] @ vectors[idx + 1]) for idx in range(len(vectors) - 1)
        ]
        final = vectors[-1]
        ancestors = vectors[:-1]
        final_to_ancestors = [float(final @ ancestor) for ancestor in ancestors]
        return {
            "path_mean_adjacent_similarity": float(np.mean(adjacent)),
            "path_min_adjacent_similarity": float(np.min(adjacent)),
            "final_to_ancestors_mean_similarity": float(np.mean(final_to_ancestors)),
        }


def greedy_decode(
    index: PrototypeIndex, query: np.ndarray, start_l1: str, *, max_depth: int
) -> tuple[list[dict[str, Any]], int, int]:
    path = initialize_path(index, start_l1)
    n_node_calls = 0
    n_dot_products = 0
    while path["resolved_depth"] < max_depth:
        child_keys = index.children(path["path_keys"][-1])
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
    index: PrototypeIndex,
    query: np.ndarray,
    start_l1: str,
    *,
    max_depth: int,
    beam_width: int,
    children_top_k: int,
    output_top_paths: int,
) -> tuple[list[dict[str, Any]], int, int]:
    frontier = [initialize_path(index, start_l1)]
    completed: list[dict[str, Any]] = []
    n_node_calls = 0
    n_dot_products = 0
    while frontier:
        expanded = []
        for state in frontier:
            if state["resolved_depth"] >= max_depth:
                completed.append(state)
                continue
            child_keys = index.children(state["path_keys"][-1])
            if not child_keys:
                completed.append(state)
                continue
            scores = index.score_nodes(query, child_keys)[:children_top_k]
            n_node_calls += len(child_keys)
            n_dot_products += index.prototype_count(child_keys)
            for score in scores:
                candidate = copy_path(state)
                append_child(candidate, score)
                expanded.append(candidate)
        if not expanded:
            break
        expanded.sort(key=lambda x: float(x["cumulative_score"]), reverse=True)
        frontier = expanded[:beam_width]
    paths = deduplicate_paths(completed + frontier)
    return paths[:output_top_paths], n_node_calls, n_dot_products


def rerank_paths(
    reranker: Any, *, product_text: str, paths: list[dict[str, Any]], batch_size: int
) -> list[dict[str, Any]]:
    pairs = [
        (product_text, "Taxonomy path: " + " > ".join(path["path_names"]))
        for path in paths
    ]
    scores = np.asarray(
        reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False),
        dtype=float,
    ).reshape(-1)
    reranked = []
    for path, score in zip(paths, scores, strict=True):
        item = copy_path(path)
        item["rerank_score"] = float(score)
        reranked.append(item)
    reranked.sort(key=lambda x: float(x["rerank_score"]), reverse=True)
    return reranked


def initialize_path(index: PrototypeIndex, start_l1: str) -> dict[str, Any]:
    meta = index.node_meta.get(start_l1, {})
    return {
        "path_keys": [start_l1] if start_l1 else [],
        "path_names": [str(meta.get("category_name", start_l1))] if start_l1 else [],
        "score_trace": [],
        "margin_trace": [],
        "prototype_trace": [],
        "cumulative_score": 0.0,
        "resolved_depth": 1 if start_l1 else 0,
    }


def append_child(path: dict[str, Any], child_score: dict[str, Any]) -> None:
    path["path_keys"].append(str(child_score["node_key"]))
    path["path_names"].append(str(child_score["node_name"]))
    path["score_trace"].append(float(child_score["score"]))
    path["margin_trace"].append(float(child_score.get("sibling_margin", np.nan)))
    path["prototype_trace"].append(child_score)
    path["resolved_depth"] = int(child_score["depth"])
    path["cumulative_score"] = (
        float(np.mean(path["score_trace"])) if path["score_trace"] else 0.0
    )


def copy_path(path: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(path, ensure_ascii=False))


def empty_path(start_l1: str) -> dict[str, Any]:
    return {
        "path_keys": [start_l1] if start_l1 else [],
        "path_names": [start_l1] if start_l1 else [],
        "score_trace": [],
        "margin_trace": [],
        "prototype_trace": [],
        "cumulative_score": 0.0,
        "resolved_depth": 1 if start_l1 else 0,
    }


def deduplicate_paths(paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for path in sorted(paths, key=lambda x: float(x["cumulative_score"]), reverse=True):
        key = tuple(path["path_keys"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def build_output_row(
    row: dict[str, Any],
    *,
    id_col: str,
    text_col: str,
    original_l1_col: str,
    mode: str,
    l1_scores: list[dict[str, Any]],
    best_path: dict[str, Any],
    top_paths: list[dict[str, Any]],
    node_score_calls: int,
    similarity_dot_products: int,
    rerank_pair_count: int,
    path_internal_coherence: dict[str, float],
) -> dict[str, Any]:
    path_names = best_path.get("path_names", [])
    path_keys = best_path.get("path_keys", [])
    score_trace = [float(x) for x in best_path.get("score_trace", [])]
    margin_trace = [float(x) for x in best_path.get("margin_trace", [])]
    finite_margins = [x for x in margin_trace if np.isfinite(x)]
    candidate_scores = path_candidate_scores(top_paths)
    top1_score = candidate_scores[0] if len(candidate_scores) >= 1 else np.nan
    top2_score = candidate_scores[1] if len(candidate_scores) >= 2 else np.nan
    out = {
        id_col: row.get(id_col, ""),
        text_col: row.get(text_col, ""),
        "input_level_1_name": row.get(original_l1_col, ""),
        "pipeline_mode": mode,
        "predicted_taxonomy_path": " > ".join(path_names),
        "predicted_taxonomy_key_path": " || ".join(path_keys),
        "predicted_path_score": float(
            best_path.get("rerank_score", best_path.get("cumulative_score", 0.0))
        ),
        "embedding_path_score": float(best_path.get("cumulative_score", 0.0)),
        "top1_score": float(top1_score),
        "top2_score": float(top2_score),
        "top1_top2_margin": float(top1_score - top2_score)
        if np.isfinite(top1_score) and np.isfinite(top2_score)
        else np.nan,
        "l1_top1_score": float(l1_scores[0]["score"]) if l1_scores else np.nan,
        "l1_top2_score": float(l1_scores[1]["score"])
        if len(l1_scores) >= 2
        else np.nan,
        "l1_top1_top2_margin": float(l1_scores[0]["score"] - l1_scores[1]["score"])
        if len(l1_scores) >= 2
        else np.nan,
        "final_category_score": float(score_trace[-1]) if score_trace else np.nan,
        "mean_local_score": float(np.mean(score_trace)) if score_trace else np.nan,
        "mean_local_margin": float(np.mean(finite_margins))
        if finite_margins
        else np.nan,
        "min_local_margin": float(np.min(finite_margins)) if finite_margins else np.nan,
        "path_score_trace_json": json.dumps(score_trace, ensure_ascii=False),
        "local_margin_trace_json": json.dumps(margin_trace, ensure_ascii=False),
        "node_score_calls": int(node_score_calls),
        "similarity_dot_products": int(similarity_dot_products),
        "rerank_pair_count": int(rerank_pair_count),
        "path_mean_adjacent_similarity": path_internal_coherence[
            "path_mean_adjacent_similarity"
        ],
        "path_min_adjacent_similarity": path_internal_coherence[
            "path_min_adjacent_similarity"
        ],
        "final_to_ancestors_mean_similarity": path_internal_coherence[
            "final_to_ancestors_mean_similarity"
        ],
        "resolved_depth": int(best_path.get("resolved_depth", 0)),
        "l1_candidates_json": json.dumps(l1_scores, ensure_ascii=False),
        "top_paths_json": json.dumps(top_paths, ensure_ascii=False),
    }
    for depth in range(1, 8):
        out[f"predicted_level_{depth}_name"] = (
            path_names[depth - 1] if len(path_names) >= depth else ""
        )
        out[f"predicted_level_{depth}_key"] = (
            path_keys[depth - 1] if len(path_keys) >= depth else ""
        )
        out[f"level_{depth}_local_score"] = (
            score_trace[depth - 2]
            if depth >= 2 and len(score_trace) >= depth - 1
            else ""
        )
        out[f"level_{depth}_local_margin"] = (
            margin_trace[depth - 2]
            if depth >= 2 and len(margin_trace) >= depth - 1
            else ""
        )
    return out


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
        row["sibling_top_score"] = top1
        row["sibling_second_score"] = top2
        row["sibling_margin"] = margin
        row["score_gap_to_best"] = top1 - float(row["score"])


def path_candidate_scores(paths: list[dict[str, Any]]) -> list[float]:
    scores = [
        float(path.get("rerank_score", path.get("cumulative_score", np.nan)))
        for path in paths
    ]
    return sorted([score for score in scores if np.isfinite(score)], reverse=True)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def load_shared_embeddings(path: Path) -> dict[str, Any]:
    return {
        "products": pd.read_parquet(path / "products.parquet"),
        "product_embeddings": np.load(path / "product_embeddings.npy").astype(
            np.float32
        ),
        "nodes": pd.read_parquet(path / "category_nodes.parquet"),
        "prototypes": pd.read_parquet(path / "category_prototypes.parquet"),
        "prototype_embeddings": np.load(
            path / "category_prototype_embeddings.npy"
        ).astype(np.float32),
    }


def print_summary(data: dict[str, Any]) -> None:
    print(
        f"Products: {len(data['products']):,} / embeddings {data['product_embeddings'].shape}"
    )
    print(f"Nodes: {len(data['nodes']):,}")
    print(
        f"Prototypes: {len(data['prototypes']):,} / embeddings {data['prototype_embeddings'].shape}"
    )


def build_metrics(
    output: pd.DataFrame,
    *,
    mode: str,
    elapsed: float,
    n_products: int,
    n_node_score_calls: int,
    n_similarity_dot_products: int,
    n_rerank_pairs: int,
    embedding_dim: int,
    phase_seconds: dict[str, float],
    resource_metrics: dict[str, Any],
) -> dict[str, Any]:
    l1_agreement = np.nan
    if "input_level_1_name" in output.columns:
        mask = output["input_level_1_name"].fillna("") != ""
        if mask.any():
            l1_agreement = float(
                (
                    output.loc[mask, "input_level_1_name"]
                    == output.loc[mask, "predicted_level_1_name"]
                ).mean()
            )
    return {
        "mode": mode,
        "n_products": int(n_products),
        "elapsed_seconds": float(elapsed),
        "seconds_per_product": float(elapsed / max(1, n_products)),
        "throughput_products_per_second": float(n_products / elapsed)
        if elapsed > 0
        else np.nan,
        **{key: float(value) for key, value in phase_seconds.items()},
        "inference_cost_unit": "M scalar multiply-adds for embedding dot products",
        "embedding_dim": int(embedding_dim),
        "node_score_calls": int(n_node_score_calls),
        "node_score_calls_per_product": float(n_node_score_calls / max(1, n_products)),
        "similarity_dot_products": int(n_similarity_dot_products),
        "similarity_dot_products_per_product": float(
            n_similarity_dot_products / max(1, n_products)
        ),
        "approx_scalar_multiply_adds": int(n_similarity_dot_products * embedding_dim),
        "approx_million_scalar_multiply_adds": float(
            (n_similarity_dot_products * embedding_dim) / 1_000_000
        ),
        "approx_million_scalar_multiply_adds_per_product": float(
            (n_similarity_dot_products * embedding_dim) / max(1, n_products) / 1_000_000
        ),
        "rerank_pair_count": int(n_rerank_pairs),
        "rerank_pairs_per_product": float(n_rerank_pairs / max(1, n_products)),
        "mean_resolved_depth": float(
            pd.to_numeric(output["resolved_depth"], errors="coerce").mean()
        ),
        "median_path_score": float(
            pd.to_numeric(output["predicted_path_score"], errors="coerce").median()
        ),
        "l1_agreement_if_available": l1_agreement,
        **resource_metrics,
        **prediction_diagnostics(output),
        **cuda_memory_metrics(),
    }


def embedding_resource_metrics(
    data: dict[str, Any],
    *,
    product_embeddings: np.ndarray,
) -> dict[str, Any]:
    product_matrix = product_embeddings
    prototype_matrix = data["prototype_embeddings"]
    return {
        "n_product_embeddings_loaded": int(len(product_matrix)),
        "n_category_prototype_embeddings_loaded": int(len(prototype_matrix)),
        "product_embeddings_mb_used_for_run": float(product_matrix.nbytes / (1024**2)),
        "category_prototype_embeddings_mb_loaded": float(
            prototype_matrix.nbytes / (1024**2)
        ),
        "total_embedding_matrix_mb_used_for_run": float(
            (product_matrix.nbytes + prototype_matrix.nbytes) / (1024**2)
        ),
    }


def prediction_diagnostics(output: pd.DataFrame) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    diagnostics.update(depth_diagnostics(output))
    diagnostics.update(diversity_diagnostics(output))
    for col in [
        "top1_top2_margin",
        "mean_local_margin",
        "min_local_margin",
        "l1_top1_top2_margin",
    ]:
        if col in output.columns:
            values = pd.to_numeric(output[col], errors="coerce")
            diagnostics[f"median_{col}"] = float(values.median())
            diagnostics[f"mean_{col}"] = float(values.mean())
            for threshold in [0.01, 0.03, 0.05]:
                diagnostics[f"share_{col}_lt_{threshold}"] = float(
                    (values < threshold).mean()
                )
    for col in [
        "path_mean_adjacent_similarity",
        "path_min_adjacent_similarity",
        "final_to_ancestors_mean_similarity",
    ]:
        if col in output.columns:
            diagnostics[f"mean_{col}"] = float(
                pd.to_numeric(output[col], errors="coerce").mean()
            )
    return diagnostics


def depth_diagnostics(output: pd.DataFrame) -> dict[str, Any]:
    depth = pd.to_numeric(output["resolved_depth"], errors="coerce")
    diagnostics = {
        "mean_resolved_depth_output": float(depth.mean()),
        "median_resolved_depth_output": float(depth.median()),
        "share_depth_le_2": float((depth <= 2).mean()),
        "share_depth_ge_4": float((depth >= 4).mean()),
        "share_depth_ge_6": float((depth >= 6).mean()),
    }
    counts = depth.value_counts(normalize=True).sort_index()
    for level, share in counts.items():
        if pd.notna(level):
            diagnostics[f"share_resolved_depth_{int(level)}"] = float(share)
    return diagnostics


def diversity_diagnostics(output: pd.DataFrame) -> dict[str, Any]:
    path = output["predicted_taxonomy_path"].fillna("").astype(str)
    counts = path.value_counts()
    probabilities = counts / max(1, len(path))
    diagnostics = {
        "n_unique_paths": int(counts.size),
        "top_path_share": float(probabilities.iloc[0])
        if len(probabilities)
        else np.nan,
        "path_entropy": float(-(probabilities * np.log(probabilities + 1e-12)).sum()),
        "path_normalized_entropy": float(
            (-(probabilities * np.log(probabilities + 1e-12)).sum())
            / np.log(max(2, counts.size))
        ),
    }
    for depth in range(1, 8):
        col = f"predicted_level_{depth}_name"
        if col in output.columns:
            values = output[col].fillna("").astype(str)
            non_empty = values[values != ""]
            if len(non_empty):
                level_counts = non_empty.value_counts(normalize=True)
                diagnostics[f"level_{depth}_n_unique"] = int(non_empty.nunique())
                diagnostics[f"level_{depth}_top_share"] = float(level_counts.iloc[0])
    return diagnostics


def reset_cuda_peak_memory() -> None:
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    for idx in range(torch.cuda.device_count()):
        try:
            with torch.cuda.device(idx):
                torch.cuda.reset_peak_memory_stats()
        except Exception as exc:
            print(f"WARNING: cannot reset CUDA peak memory for cuda:{idx}: {exc}")


def cuda_memory_metrics() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {
            "cuda_available": False,
            "cuda_device_count": 0,
            "peak_gpu_memory_allocated_mb": np.nan,
            "peak_gpu_memory_reserved_mb": np.nan,
        }
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "cuda_device_count": 0,
            "peak_gpu_memory_allocated_mb": np.nan,
            "peak_gpu_memory_reserved_mb": np.nan,
        }
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
    return {
        "cuda_available": True,
        "cuda_device_count": int(torch.cuda.device_count()),
        "peak_gpu_memory_allocated_mb": float(max(allocated)) if allocated else np.nan,
        "peak_gpu_memory_reserved_mb": float(max(reserved)) if reserved else np.nan,
        "gpu_memory_by_device": by_device,
    }


def cuda_memory_metrics() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {
            "cuda_available": False,
            "peak_gpu_memory_allocated_mb": np.nan,
            "peak_gpu_memory_reserved_mb": np.nan,
        }
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "peak_gpu_memory_allocated_mb": np.nan,
            "peak_gpu_memory_reserved_mb": np.nan,
        }
    allocated = []
    reserved = []
    by_device = {}
    for idx in range(torch.cuda.device_count()):
        alloc_mb = torch.cuda.max_memory_allocated(idx) / (1024**2)
        reserv_mb = torch.cuda.max_memory_reserved(idx) / (1024**2)
        allocated.append(alloc_mb)
        reserved.append(reserv_mb)
        by_device[f"cuda:{idx}"] = {
            "peak_allocated_mb": float(alloc_mb),
            "peak_reserved_mb": float(reserv_mb),
        }
    return {
        "cuda_available": True,
        "peak_gpu_memory_allocated_mb": float(max(allocated) if allocated else 0.0),
        "peak_gpu_memory_reserved_mb": float(max(reserved) if reserved else 0.0),
        "gpu_memory_by_device": by_device,
    }


def load_cross_encoder(model_name: str, device: str | None) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device=device, trust_remote_code=True)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--mode", choices=["greedy", "beam", "beam-rerank"], default="beam"
    )
    parser.add_argument("--id-col", default="hashed_external_id")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--original-l1-col", default="level_1_name")
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--l1-top-k", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--children-top-k", type=int, default=8)
    parser.add_argument("--output-top-paths", type=int, default=10)
    parser.add_argument("--score-aggregation", choices=["max", "mean"], default="max")
    parser.add_argument(
        "--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    parser.add_argument("--reranker-device", default="mps")
    parser.add_argument("--reranker-batch-size", type=int, default=64)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
