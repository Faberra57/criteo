#!/usr/bin/env python3
"""Pipeline 3: global path retrieval from L2 to L7.

The product is first routed to L1 with the fine-tuned L1 embedding space.
Then the script retrieves top-k complete taxonomy paths starting at L2 inside
the predicted L1 subtree. Optionally, a cross-encoder reranks the retrieved
paths.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_EMBEDDINGS_DIR = "dataset/shared_l1_l7_embeddings_Jasper-Token-Compression-600M"
DEFAULT_OUTPUT_PATH = "data/pipeline3_global_path_predictions_jasper.csv"


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

    index_started = time.perf_counter()
    products = data["products"].copy().reset_index(drop=True)
    product_embeddings = data["product_embeddings"]
    if args.sample_size is not None and len(products) > args.sample_size:
        products = products.sample(
            args.sample_size, random_state=args.random_seed
        ).sort_index()
        product_embeddings = product_embeddings[products.index.to_numpy()]
        products = products.reset_index(drop=True)

    l1_index = L1PrototypeIndex(
        data["nodes"],
        data["prototypes"],
        data["prototype_embeddings"],
        args.score_aggregation,
    )
    category_index = CategoryCentroidIndex(
        data["nodes"], data["prototypes"], data["prototype_embeddings"]
    )
    global_paths = data["global_paths"]
    global_embeddings = data["global_path_embeddings"]
    l1_to_global = build_l1_global_slices(global_paths, global_embeddings)
    index_build_seconds = time.perf_counter() - index_started

    reranker_load_seconds = 0.0
    reranker = None
    if args.selector == "reranker":
        reranker_started = time.perf_counter()
        reranker = load_cross_encoder(args.reranker_model, args.reranker_device)
        reranker_load_seconds = time.perf_counter() - reranker_started

    output_rows: list[dict[str, Any]] = []
    n_global_path_dot_products = 0
    n_l1_dot_products = 0
    n_rerank_pairs = 0
    decode_started = time.perf_counter()
    for row, embedding in zip(
        products.to_dict("records"), product_embeddings, strict=True
    ):
        l1_scores = l1_index.score(embedding)
        n_l1_dot_products += l1_index.prototype_count()
        predicted_l1 = l1_scores[0]["node_key"] if l1_scores else ""
        candidates_df, candidates_embeddings = l1_to_global.get(
            predicted_l1, (None, None)
        )
        if (
            candidates_df is None
            or candidates_embeddings is None
            or len(candidates_df) == 0
        ):
            output_rows.append(empty_output(row, args, predicted_l1, l1_scores))
            continue

        scores = candidates_embeddings @ embedding
        n_global_path_dot_products += len(scores)
        top_idx = topk_indices(scores, min(args.retrieve_top_k, len(scores)))
        retrieved = candidates_df.iloc[top_idx].copy().reset_index(drop=True)
        retrieved["embedding_score"] = scores[top_idx]
        candidates = retrieved.to_dict("records")

        if args.selector == "reranker":
            n_rerank_pairs += len(candidates)
            pairs = [
                (
                    str(row.get(args.text_col, "")),
                    "Taxonomy path: " + str(c["taxonomy_path"]),
                )
                for c in candidates
            ]
            rerank_scores = np.asarray(
                reranker.predict(
                    pairs, batch_size=args.reranker_batch_size, show_progress_bar=False
                ),
                dtype=float,
            ).reshape(-1)
            for candidate, score in zip(candidates, rerank_scores, strict=True):
                candidate["rerank_score"] = float(score)
            candidates.sort(key=lambda item: float(item["rerank_score"]), reverse=True)
        else:
            candidates.sort(
                key=lambda item: float(item["embedding_score"]), reverse=True
            )

        best = candidates[0]
        output_rows.append(
            build_output(
                row,
                args,
                predicted_l1,
                l1_scores,
                best,
                candidates[: args.output_top_k],
                category_index=category_index,
            )
        )
    decode_seconds = time.perf_counter() - decode_started

    output = pd.DataFrame(output_rows)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_started = time.perf_counter()
    output.to_csv(output_path, index=False)
    write_seconds = time.perf_counter() - write_started
    metrics = build_metrics(
        output,
        time.perf_counter() - started,
        n_global_path_dot_products,
        args.selector,
        n_l1_dot_products=n_l1_dot_products,
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


class L1PrototypeIndex:
    def __init__(
        self,
        nodes: pd.DataFrame,
        prototypes: pd.DataFrame,
        embeddings: np.ndarray,
        aggregation: str,
    ) -> None:
        self.nodes = nodes.reset_index(drop=True).copy()
        self.prototypes = prototypes.reset_index(drop=True).copy()
        self.embeddings = embeddings.astype(np.float32)
        self.aggregation = aggregation
        self.node_meta = self.nodes.set_index("node_key").to_dict("index")
        self.l1_keys = sorted(
            self.nodes.loc[self.nodes["depth"] == 1, "node_key"].tolist()
        )
        self.node_to_indices = (
            self.prototypes.reset_index()
            .groupby("node_key")["index"]
            .apply(list)
            .to_dict()
        )

    def score(self, query: np.ndarray) -> list[dict[str, Any]]:
        rows = []
        for node_key in self.l1_keys:
            indices = self.node_to_indices.get(node_key, [])
            if not indices:
                continue
            sims = self.embeddings[np.asarray(indices)] @ query
            score = float(np.mean(sims) if self.aggregation == "mean" else np.max(sims))
            rows.append(
                {
                    "node_key": node_key,
                    "node_name": self.node_meta.get(node_key, {}).get(
                        "category_name", node_key
                    ),
                    "score": score,
                    "n_prototypes": len(indices),
                }
            )
        rows.sort(key=lambda item: float(item["score"]), reverse=True)
        add_rank_and_margins(rows)
        return rows

    def prototype_count(self) -> int:
        return int(
            sum(
                len(self.node_to_indices.get(node_key, [])) for node_key in self.l1_keys
            )
        )


class CategoryCentroidIndex:
    def __init__(
        self, nodes: pd.DataFrame, prototypes: pd.DataFrame, embeddings: np.ndarray
    ) -> None:
        self.nodes = nodes.reset_index(drop=True).copy()
        self.prototypes = prototypes.reset_index(drop=True).copy()
        self.embeddings = embeddings.astype(np.float32)
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
            matrix = self.embeddings[np.asarray(indices, dtype=int)]
            centroids[node_key] = normalize_rows(matrix.mean(axis=0, keepdims=True))[0]
        return centroids


def build_l1_global_slices(
    paths: pd.DataFrame, embeddings: np.ndarray
) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    mapping = {}
    for l1_name, group in paths.groupby("level_1_name", sort=False):
        indices = group.index.to_numpy(dtype=int)
        mapping[str(l1_name)] = (group.reset_index(drop=True), embeddings[indices])
    return mapping


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= len(scores):
        return np.argsort(scores)[::-1]
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]


def build_output(
    row: dict[str, Any],
    args: argparse.Namespace,
    predicted_l1: str,
    l1_scores: list[dict[str, Any]],
    best: dict[str, Any],
    top_candidates: list[dict[str, Any]],
    *,
    category_index: CategoryCentroidIndex,
) -> dict[str, Any]:
    score = best.get("rerank_score", best.get("embedding_score", np.nan))
    taxonomy_path = str(best.get("taxonomy_path", ""))
    path_names = taxonomy_path.split(" > ") if taxonomy_path else []
    path_keys = cumulative_path_keys(taxonomy_path)
    selection_scores = candidate_selection_scores(
        top_candidates, selector=args.selector
    )
    embedding_scores = sorted(
        [
            float(candidate.get("embedding_score", np.nan))
            for candidate in top_candidates
            if np.isfinite(float(candidate.get("embedding_score", np.nan)))
        ],
        reverse=True,
    )
    rerank_scores = sorted(
        [
            float(candidate.get("rerank_score", np.nan))
            for candidate in top_candidates
            if np.isfinite(float(candidate.get("rerank_score", np.nan)))
        ],
        reverse=True,
    )
    coherence = path_internal_coherence(category_index, path_keys)
    out = {
        args.id_col: row.get(args.id_col, ""),
        args.text_col: row.get(args.text_col, ""),
        "input_level_1_name": row.get(args.original_l1_col, ""),
        "pipeline_mode": f"global_path_{args.selector}",
        "predicted_taxonomy_path": taxonomy_path,
        "predicted_taxonomy_key_path": str(best.get("node_key", "")),
        "predicted_path_score": float(score),
        "embedding_path_score": float(best.get("embedding_score", np.nan)),
        "product_path_similarity": float(best.get("embedding_score", np.nan)),
        "top1_score": selection_scores[0] if len(selection_scores) >= 1 else np.nan,
        "top2_score": selection_scores[1] if len(selection_scores) >= 2 else np.nan,
        "top1_top2_margin": selection_scores[0] - selection_scores[1]
        if len(selection_scores) >= 2
        else np.nan,
        "embedding_top1_top2_margin": embedding_scores[0] - embedding_scores[1]
        if len(embedding_scores) >= 2
        else np.nan,
        "rerank_top1_top2_margin": rerank_scores[0] - rerank_scores[1]
        if len(rerank_scores) >= 2
        else np.nan,
        "l1_top1_score": float(l1_scores[0]["score"]) if l1_scores else np.nan,
        "l1_top2_score": float(l1_scores[1]["score"])
        if len(l1_scores) >= 2
        else np.nan,
        "l1_top1_top2_margin": float(l1_scores[0]["score"] - l1_scores[1]["score"])
        if len(l1_scores) >= 2
        else np.nan,
        "final_category_score": float(best.get("embedding_score", np.nan)),
        "mean_local_score": float(best.get("embedding_score", np.nan)),
        "mean_local_margin": embedding_scores[0] - embedding_scores[1]
        if len(embedding_scores) >= 2
        else np.nan,
        "min_local_margin": embedding_scores[0] - embedding_scores[1]
        if len(embedding_scores) >= 2
        else np.nan,
        "path_score_trace_json": json.dumps(
            [float(best.get("embedding_score", np.nan))], ensure_ascii=False
        ),
        "local_margin_trace_json": json.dumps(
            [embedding_scores[0] - embedding_scores[1]]
            if len(embedding_scores) >= 2
            else [],
            ensure_ascii=False,
        ),
        "path_mean_adjacent_similarity": coherence["path_mean_adjacent_similarity"],
        "path_min_adjacent_similarity": coherence["path_min_adjacent_similarity"],
        "final_to_ancestors_mean_similarity": coherence[
            "final_to_ancestors_mean_similarity"
        ],
        "resolved_depth": int(best.get("depth", 0)),
        "l1_candidates_json": json.dumps(
            l1_scores[: args.l1_top_k], ensure_ascii=False
        ),
        "top_global_paths_json": json.dumps(
            make_json_safe(top_candidates), ensure_ascii=False
        ),
    }
    for depth in range(1, 8):
        out[f"predicted_level_{depth}_name"] = (
            path_names[depth - 1] if len(path_names) >= depth else ""
        )
        out[f"predicted_level_{depth}_key"] = (
            path_keys[depth - 1] if len(path_keys) >= depth else ""
        )
        out[f"level_{depth}_local_score"] = (
            float(best.get("embedding_score", np.nan))
            if depth == int(best.get("depth", 0))
            else ""
        )
        out[f"level_{depth}_local_margin"] = (
            embedding_scores[0] - embedding_scores[1]
            if depth == int(best.get("depth", 0)) and len(embedding_scores) >= 2
            else ""
        )
    return out


def empty_output(
    row: dict[str, Any],
    args: argparse.Namespace,
    predicted_l1: str,
    l1_scores: list[dict[str, Any]],
) -> dict[str, Any]:
    out = {
        args.id_col: row.get(args.id_col, ""),
        args.text_col: row.get(args.text_col, ""),
        "input_level_1_name": row.get(args.original_l1_col, ""),
        "pipeline_mode": f"global_path_{args.selector}",
        "predicted_taxonomy_path": predicted_l1,
        "predicted_taxonomy_key_path": predicted_l1,
        "predicted_path_score": 0.0,
        "embedding_path_score": 0.0,
        "product_path_similarity": 0.0,
        "top1_score": np.nan,
        "top2_score": np.nan,
        "top1_top2_margin": np.nan,
        "embedding_top1_top2_margin": np.nan,
        "rerank_top1_top2_margin": np.nan,
        "l1_top1_score": float(l1_scores[0]["score"]) if l1_scores else np.nan,
        "l1_top2_score": float(l1_scores[1]["score"])
        if len(l1_scores) >= 2
        else np.nan,
        "l1_top1_top2_margin": float(l1_scores[0]["score"] - l1_scores[1]["score"])
        if len(l1_scores) >= 2
        else np.nan,
        "final_category_score": np.nan,
        "mean_local_score": np.nan,
        "mean_local_margin": np.nan,
        "min_local_margin": np.nan,
        "path_score_trace_json": "[]",
        "local_margin_trace_json": "[]",
        "path_mean_adjacent_similarity": np.nan,
        "path_min_adjacent_similarity": np.nan,
        "final_to_ancestors_mean_similarity": np.nan,
        "resolved_depth": 1 if predicted_l1 else 0,
        "l1_candidates_json": json.dumps(
            l1_scores[: args.l1_top_k], ensure_ascii=False
        ),
        "top_global_paths_json": "[]",
    }
    for depth in range(1, 8):
        out[f"predicted_level_{depth}_name"] = predicted_l1 if depth == 1 else ""
        out[f"predicted_level_{depth}_key"] = predicted_l1 if depth == 1 else ""
        out[f"level_{depth}_local_score"] = ""
        out[f"level_{depth}_local_margin"] = ""
    return out


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
        "global_paths": pd.read_parquet(path / "global_paths.parquet"),
        "global_path_embeddings": np.load(path / "global_path_embeddings.npy").astype(
            np.float32
        ),
    }


def print_summary(data: dict[str, Any]) -> None:
    print(
        f"Products: {len(data['products']):,} / embeddings {data['product_embeddings'].shape}"
    )
    print(
        f"Global paths: {len(data['global_paths']):,} / embeddings {data['global_path_embeddings'].shape}"
    )
    print(f"Category prototypes: {len(data['prototypes']):,}")


def build_metrics(
    output: pd.DataFrame,
    elapsed: float,
    n_global_path_dot_products: int,
    selector: str,
    *,
    n_l1_dot_products: int,
    n_rerank_pairs: int,
    embedding_dim: int,
    phase_seconds: dict[str, float],
    resource_metrics: dict[str, Any],
) -> dict[str, Any]:
    l1_agreement = np.nan
    mask = output["input_level_1_name"].fillna("") != ""
    if mask.any():
        l1_agreement = float(
            (
                output.loc[mask, "input_level_1_name"]
                == output.loc[mask, "predicted_level_1_name"]
            ).mean()
        )
    total_dot_products = int(n_l1_dot_products + n_global_path_dot_products)
    return {
        "mode": f"global_path_{selector}",
        "n_products": int(len(output)),
        "elapsed_seconds": float(elapsed),
        "seconds_per_product": float(elapsed / max(1, len(output))),
        "throughput_products_per_second": float(len(output) / elapsed)
        if elapsed > 0
        else np.nan,
        **{key: float(value) for key, value in phase_seconds.items()},
        "inference_cost_unit": "M scalar multiply-adds for embedding dot products",
        "embedding_dim": int(embedding_dim),
        "l1_similarity_dot_products": int(n_l1_dot_products),
        "global_path_similarity_dot_products": int(n_global_path_dot_products),
        "similarity_dot_products": total_dot_products,
        "similarity_dot_products_per_product": float(
            total_dot_products / max(1, len(output))
        ),
        "approx_global_path_score_calls": int(n_global_path_dot_products),
        "approx_global_path_score_calls_per_product": float(
            n_global_path_dot_products / max(1, len(output))
        ),
        "approx_scalar_multiply_adds": int(total_dot_products * embedding_dim),
        "approx_million_scalar_multiply_adds": float(
            (total_dot_products * embedding_dim) / 1_000_000
        ),
        "approx_million_scalar_multiply_adds_per_product": float(
            (total_dot_products * embedding_dim) / max(1, len(output)) / 1_000_000
        ),
        "rerank_pair_count": int(n_rerank_pairs),
        "rerank_pairs_per_product": float(n_rerank_pairs / max(1, len(output))),
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
    global_path_matrix = data["global_path_embeddings"]
    return {
        "n_product_embeddings_loaded": int(len(product_matrix)),
        "n_category_prototype_embeddings_loaded": int(len(prototype_matrix)),
        "n_global_path_embeddings_loaded": int(len(global_path_matrix)),
        "product_embeddings_mb_used_for_run": float(product_matrix.nbytes / (1024**2)),
        "category_prototype_embeddings_mb_loaded": float(
            prototype_matrix.nbytes / (1024**2)
        ),
        "global_path_embeddings_mb_loaded": float(
            global_path_matrix.nbytes / (1024**2)
        ),
        "total_embedding_matrix_mb_used_for_run": float(
            (
                product_matrix.nbytes
                + prototype_matrix.nbytes
                + global_path_matrix.nbytes
            )
            / (1024**2)
        ),
    }


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


def cumulative_path_keys(taxonomy_path: str) -> list[str]:
    parts = [part.strip() for part in taxonomy_path.split(" > ") if part.strip()]
    return [" > ".join(parts[: idx + 1]) for idx in range(len(parts))]


def candidate_selection_scores(
    top_candidates: list[dict[str, Any]], *, selector: str
) -> list[float]:
    score_col = "rerank_score" if selector == "reranker" else "embedding_score"
    scores = [float(candidate.get(score_col, np.nan)) for candidate in top_candidates]
    return sorted([score for score in scores if np.isfinite(score)], reverse=True)


def path_internal_coherence(
    index: CategoryCentroidIndex, path_keys: list[str]
) -> dict[str, float]:
    vectors = [
        index.node_to_centroid[key]
        for key in path_keys
        if key in index.node_to_centroid
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
    final_to_ancestors = [float(final @ ancestor) for ancestor in vectors[:-1]]
    return {
        "path_mean_adjacent_similarity": float(np.mean(adjacent)),
        "path_min_adjacent_similarity": float(np.min(adjacent)),
        "final_to_ancestors_mean_similarity": float(np.mean(final_to_ancestors)),
    }


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def prediction_diagnostics(output: pd.DataFrame) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    diagnostics.update(depth_diagnostics(output))
    diagnostics.update(diversity_diagnostics(output))
    for col in [
        "top1_top2_margin",
        "embedding_top1_top2_margin",
        "rerank_top1_top2_margin",
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
        "product_path_similarity",
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
    entropy = float(-(probabilities * np.log(probabilities + 1e-12)).sum())
    diagnostics = {
        "n_unique_paths": int(counts.size),
        "top_path_share": float(probabilities.iloc[0])
        if len(probabilities)
        else np.nan,
        "path_entropy": entropy,
        "path_normalized_entropy": float(entropy / np.log(max(2, counts.size))),
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


def make_json_safe(value: Any) -> Any:
    if isinstance(value, list):
        return [make_json_safe(x) for x in value]
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--selector", choices=["embedding", "reranker"], default="embedding"
    )
    parser.add_argument("--id-col", default="hashed_external_id")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--original-l1-col", default="level_1_name")
    parser.add_argument("--l1-top-k", type=int, default=5)
    parser.add_argument("--retrieve-top-k", type=int, default=50)
    parser.add_argument("--output-top-k", type=int, default=10)
    parser.add_argument("--score-aggregation", choices=["max", "mean"], default="max")
    parser.add_argument(
        "--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    parser.add_argument("--reranker-device", default=None)
    parser.add_argument("--reranker-batch-size", type=int, default=64)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
