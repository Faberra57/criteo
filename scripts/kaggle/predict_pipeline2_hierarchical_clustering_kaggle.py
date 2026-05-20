#!/usr/bin/env python3
"""Pipeline 2: recursive clustering inside each predicted parent category.

The script consumes embeddings exported by export_shared_embeddings_kaggle.py.
It predicts L1 by prototype similarity, then recursively:
1. groups products assigned to a parent node;
2. clusters their product embeddings;
3. maps each cluster centroid to one enriched child category centroid;
4. repeats for the next level.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - scipy is normally available with sklearn.
    linear_sum_assignment = None

DEFAULT_EMBEDDINGS_DIR = "dataset/shared_l1_l7_embeddings_Jasper-Token-Compression-600M"
DEFAULT_OUTPUT_PATH = "data/pipeline2_hierarchical_clustering_predictions_jasper.csv"
DEFAULT_SUMMARY_PATH = "data/pipeline2_hierarchical_clustering_summary_jasper.csv"


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

    index = CategoryCentroidIndex(
        nodes=data["nodes"],
        prototypes=data["prototypes"],
        prototype_embeddings=data["prototype_embeddings"],
    )
    index_build_seconds = time.perf_counter() - index_started
    n = len(products)
    paths: list[dict[str, Any]] = [empty_path() for _ in range(n)]
    active_parent: dict[str, list[int]] = defaultdict(list)
    n_child_scores = 0
    n_kmeans_distance_ops = 0

    decode_started = time.perf_counter()
    if args.l1_predictions_path:
        active_parent, fallback_indices = initialize_l1_from_predictions(
            products,
            paths,
            index,
            predictions_path=Path(args.l1_predictions_path),
            id_col=args.id_col,
            l1_key_col=args.l1_prediction_key_col,
        )
        if fallback_indices:
            n_child_scores += route_l1_by_centroid(
                product_embeddings,
                paths,
                index,
                active_parent,
                product_indices=fallback_indices,
            )
    else:
        n_child_scores += route_l1_by_centroid(
            product_embeddings,
            paths,
            index,
            active_parent,
            product_indices=list(range(n)),
        )

    cluster_rows: list[dict[str, Any]] = []
    for depth in range(1, args.max_depth):
        next_active: dict[str, list[int]] = defaultdict(list)
        for parent_key, product_indices in active_parent.items():
            child_keys = index.children(parent_key)
            if not child_keys:
                continue
            if (
                len(product_indices) < args.min_products_per_parent
                or len(child_keys) == 1
            ):
                assignments, scored = direct_assign(
                    product_embeddings[product_indices], index, child_keys
                )
                n_child_scores += scored
                for local_idx, assignment in enumerate(assignments):
                    product_idx = product_indices[local_idx]
                    append_node(
                        paths[product_idx],
                        index,
                        assignment["node_key"],
                        score=assignment["score"],
                        margin=assignment["margin"],
                    )
                    next_active[assignment["node_key"]].append(product_idx)
                continue

            k = choose_k(
                n_products=len(product_indices),
                n_children=len(child_keys),
                mode=args.k_mode,
                fixed_k=args.fixed_k,
                max_clusters=args.max_clusters_per_parent,
            )
            group_embeddings = product_embeddings[product_indices]
            if k <= 1:
                assignments, scored = direct_assign(group_embeddings, index, child_keys)
                n_child_scores += scored
                for local_idx, assignment in enumerate(assignments):
                    product_idx = product_indices[local_idx]
                    append_node(
                        paths[product_idx],
                        index,
                        assignment["node_key"],
                        score=assignment["score"],
                        margin=assignment["margin"],
                    )
                    next_active[assignment["node_key"]].append(product_idx)
                continue

            kmeans = MiniBatchKMeans(
                n_clusters=k,
                random_state=args.random_seed,
                batch_size=max(256, min(4096, len(product_indices))),
                n_init="auto",
            )
            labels = kmeans.fit_predict(group_embeddings)
            n_iter = int(getattr(kmeans, "n_iter_", 1) or 1)
            n_kmeans_distance_ops += int(len(product_indices) * k * n_iter)
            centers = normalize_rows(kmeans.cluster_centers_.astype(np.float32))
            cluster_assignments, scored = assign_clusters_to_categories(
                centers,
                index,
                child_keys,
                strategy=args.cluster_category_assignment,
            )
            n_child_scores += scored

            for cluster_id, assignment in enumerate(cluster_assignments):
                members = np.where(labels == cluster_id)[0]
                cluster_rows.append(
                    {
                        "parent_key": parent_key,
                        "parent_depth": depth,
                        "cluster_id": cluster_id,
                        "cluster_size": int(len(members)),
                        "assigned_child_key": assignment["node_key"],
                        "assigned_child_name": assignment["node_name"],
                        "assignment_score": assignment["score"],
                        "assignment_margin": assignment["margin"],
                        "assignment_strategy": assignment["assignment_strategy"],
                        "n_children": len(child_keys),
                        "n_clusters": k,
                    }
                )
                for local_idx in members:
                    product_idx = product_indices[int(local_idx)]
                    append_node(
                        paths[product_idx],
                        index,
                        assignment["node_key"],
                        score=assignment["score"],
                        margin=assignment["margin"],
                    )
                    next_active[assignment["node_key"]].append(product_idx)

        active_parent = next_active
        if not active_parent:
            break
    decode_seconds = time.perf_counter() - decode_started

    output = build_output(
        products,
        paths,
        index=index,
        id_col=args.id_col,
        text_col=args.text_col,
        original_l1_col=args.original_l1_col,
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_started = time.perf_counter()
    output.to_csv(output_path, index=False)
    pd.DataFrame(cluster_rows).to_csv(args.summary_path, index=False)
    write_seconds = time.perf_counter() - write_started
    metrics = build_metrics(
        output,
        time.perf_counter() - started,
        n_child_scores,
        n_kmeans_distance_ops=n_kmeans_distance_ops,
        embedding_dim=int(product_embeddings.shape[1]),
        phase_seconds={
            "load_embeddings_seconds": load_seconds,
            "index_build_seconds": index_build_seconds,
            "decode_seconds": decode_seconds,
            "write_outputs_seconds": write_seconds,
        },
        resource_metrics=embedding_resource_metrics(
            data, product_embeddings=product_embeddings
        ),
        cluster_rows=cluster_rows,
    )
    write_json(
        {"args": vars(args), "metrics": metrics},
        output_path.with_suffix(".metrics.json"),
    )
    print(f"Predictions saved to {output_path}")
    print(f"Cluster summary saved to {args.summary_path}")
    print(json.dumps(metrics, indent=2))


class CategoryCentroidIndex:
    def __init__(
        self,
        *,
        nodes: pd.DataFrame,
        prototypes: pd.DataFrame,
        prototype_embeddings: np.ndarray,
    ) -> None:
        self.nodes = nodes.reset_index(drop=True).copy()
        self.prototypes = prototypes.reset_index(drop=True).copy()
        self.prototype_embeddings = prototype_embeddings.astype(np.float32)
        self.node_meta = self.nodes.set_index("node_key").to_dict("index")
        self.parent_to_children = build_parent_to_children(self.nodes)
        self.l1_node_keys = sorted(
            self.nodes.loc[self.nodes["depth"] == 1, "node_key"].tolist()
        )
        self.node_to_centroid = self._build_centroids()

    def _build_centroids(self) -> dict[str, np.ndarray]:
        work = self.prototypes.reset_index().rename(columns={"index": "_idx"})
        centroids = {}
        for node_key, group in work.groupby("node_key"):
            indices = group["_idx"].to_numpy(dtype=int)
            centroids[node_key] = normalize_rows(
                self.prototype_embeddings[indices].mean(axis=0, keepdims=True)
            )[0]
        return centroids

    def centroid_matrix(self, node_keys: list[str]) -> np.ndarray:
        return np.vstack([self.node_to_centroid[key] for key in node_keys]).astype(
            np.float32
        )

    def children(self, parent_key: str) -> list[str]:
        return [
            key
            for key in self.parent_to_children.get(parent_key, [])
            if key in self.node_to_centroid
        ]

    def name(self, node_key: str) -> str:
        return str(self.node_meta.get(node_key, {}).get("category_name", node_key))

    def depth(self, node_key: str) -> int:
        return int(self.node_meta.get(node_key, {}).get("depth", 0))


def initialize_l1_from_predictions(
    products: pd.DataFrame,
    paths: list[dict[str, Any]],
    index: CategoryCentroidIndex,
    *,
    predictions_path: Path,
    id_col: str,
    l1_key_col: str,
) -> tuple[dict[str, list[int]], list[int]]:
    predictions = pd.read_csv(predictions_path, usecols=lambda col: col in {id_col, l1_key_col})
    if id_col not in predictions.columns or l1_key_col not in predictions.columns:
        raise ValueError(
            f"{predictions_path} must contain {id_col!r} and {l1_key_col!r}."
        )
    id_to_l1 = (
        predictions[[id_col, l1_key_col]]
        .dropna()
        .drop_duplicates(id_col)
        .set_index(id_col)[l1_key_col]
        .astype(str)
        .to_dict()
    )
    active_parent: dict[str, list[int]] = defaultdict(list)
    fallback_indices: list[int] = []
    for idx, product in enumerate(products.to_dict("records")):
        product_id = product.get(id_col, "")
        node_key = str(id_to_l1.get(product_id, "")).strip()
        if node_key in index.l1_node_keys:
            append_node(
                paths[idx],
                index,
                node_key,
                score=np.nan,
                margin=np.nan,
            )
            active_parent[node_key].append(idx)
        else:
            fallback_indices.append(idx)
    print(
        "L1 initialization from external predictions: "
        f"{len(products) - len(fallback_indices):,} matched, "
        f"{len(fallback_indices):,} fallback to centroid routing."
    )
    return active_parent, fallback_indices


def route_l1_by_centroid(
    product_embeddings: np.ndarray,
    paths: list[dict[str, Any]],
    index: CategoryCentroidIndex,
    active_parent: dict[str, list[int]],
    *,
    product_indices: list[int],
) -> int:
    if not product_indices:
        return 0
    l1_scores = product_embeddings[product_indices] @ index.centroid_matrix(index.l1_node_keys).T
    best_l1 = np.argmax(l1_scores, axis=1)
    for local_idx, best_idx in enumerate(best_l1):
        product_idx = product_indices[local_idx]
        node_key = index.l1_node_keys[int(best_idx)]
        score = float(l1_scores[local_idx, best_idx])
        append_node(
            paths[product_idx],
            index,
            node_key,
            score=score,
            margin=margin_from_scores(l1_scores[local_idx]),
        )
        active_parent[node_key].append(product_idx)
    return int(len(product_indices) * len(index.l1_node_keys))


def direct_assign(
    query_embeddings: np.ndarray, index: CategoryCentroidIndex, child_keys: list[str]
) -> tuple[list[dict[str, Any]], int]:
    child_matrix = index.centroid_matrix(child_keys)
    scores = query_embeddings @ child_matrix.T
    best = np.argmax(scores, axis=1)
    assignments = []
    for row_idx, child_idx in enumerate(best):
        child_key = child_keys[int(child_idx)]
        assignments.append(
            {
                "node_key": child_key,
                "node_name": index.name(child_key),
                "score": float(scores[row_idx, child_idx]),
                "margin": margin_from_scores(scores[row_idx]),
                "assignment_strategy": "greedy_nearest",
            }
        )
    return assignments, int(len(query_embeddings) * len(child_keys))


def assign_clusters_to_categories(
    centers: np.ndarray,
    index: CategoryCentroidIndex,
    child_keys: list[str],
    *,
    strategy: str,
) -> tuple[list[dict[str, Any]], int]:
    if strategy == "greedy" or linear_sum_assignment is None:
        assignments, scored = direct_assign(centers, index, child_keys)
        for assignment in assignments:
            assignment["assignment_strategy"] = "greedy_nearest"
        return assignments, scored

    child_matrix = index.centroid_matrix(child_keys)
    scores = centers @ child_matrix.T
    assignments: list[dict[str, Any] | None] = [None] * len(centers)
    row_idx, col_idx = linear_sum_assignment(-scores)
    for row, col in zip(row_idx, col_idx, strict=True):
        child_key = child_keys[int(col)]
        assignments[int(row)] = {
            "node_key": child_key,
            "node_name": index.name(child_key),
            "score": float(scores[row, col]),
            "margin": margin_from_scores(scores[row]),
            "assignment_strategy": "hungarian_max_similarity",
        }

    # Rectangular cases can leave clusters unassigned when there are more clusters
    # than children. Assign the remaining clusters to their nearest child.
    for row, assignment in enumerate(assignments):
        if assignment is not None:
            continue
        col = int(np.argmax(scores[row]))
        child_key = child_keys[col]
        assignments[row] = {
            "node_key": child_key,
            "node_name": index.name(child_key),
            "score": float(scores[row, col]),
            "margin": margin_from_scores(scores[row]),
            "assignment_strategy": "hungarian_fallback_nearest",
        }
    return [assignment for assignment in assignments if assignment is not None], int(len(centers) * len(child_keys))


def choose_k(
    *,
    n_products: int,
    n_children: int,
    mode: str,
    fixed_k: int | None,
    max_clusters: int,
) -> int:
    if fixed_k is not None:
        return max(1, min(fixed_k, n_products, max_clusters))
    if mode == "children":
        k = n_children
    elif mode == "sqrt":
        k = int(round(math.sqrt(n_products)))
    else:
        k = min(n_children, int(round(math.sqrt(n_products))))
    return max(1, min(k, n_products, max_clusters))


def append_node(
    path: dict[str, Any],
    index: CategoryCentroidIndex,
    node_key: str,
    *,
    score: float,
    margin: float,
) -> None:
    path["path_keys"].append(node_key)
    path["path_names"].append(index.name(node_key))
    path["score_trace"].append(float(score))
    path["margin_trace"].append(float(margin))
    path["resolved_depth"] = index.depth(node_key)


def empty_path() -> dict[str, Any]:
    return {
        "path_keys": [],
        "path_names": [],
        "score_trace": [],
        "margin_trace": [],
        "resolved_depth": 0,
    }


def build_output(
    products: pd.DataFrame,
    paths: list[dict[str, Any]],
    *,
    index: CategoryCentroidIndex,
    id_col: str,
    text_col: str,
    original_l1_col: str,
) -> pd.DataFrame:
    rows = []
    for product, path in zip(products.to_dict("records"), paths, strict=True):
        score_trace = [float(x) for x in path["score_trace"]]
        margin_trace = [float(x) for x in path["margin_trace"]]
        finite_scores = [x for x in score_trace if np.isfinite(x)]
        finite_margins = [x for x in margin_trace if np.isfinite(x)]
        coherence = path_internal_coherence(index, path["path_keys"])
        row = {
            id_col: product.get(id_col, ""),
            text_col: product.get(text_col, ""),
            "input_level_1_name": product.get(original_l1_col, ""),
            "pipeline_mode": "hierarchical_clustering",
            "predicted_taxonomy_path": " > ".join(path["path_names"]),
            "predicted_taxonomy_key_path": " || ".join(path["path_keys"]),
            "predicted_path_score": float(np.mean(finite_scores)) if finite_scores else 0.0,
            "embedding_path_score": float(np.mean(finite_scores)) if finite_scores else 0.0,
            "top1_score": float(np.mean(finite_scores)) if finite_scores else np.nan,
            "top2_score": np.nan,
            "top1_top2_margin": float(np.mean(finite_margins))
            if finite_margins
            else np.nan,
            "l1_top1_score": score_trace[0] if score_trace else np.nan,
            "l1_top2_score": np.nan,
            "l1_top1_top2_margin": margin_trace[0] if margin_trace else np.nan,
            "final_category_score": next((x for x in reversed(score_trace) if np.isfinite(x)), np.nan),
            "mean_local_score": float(np.mean(finite_scores)) if finite_scores else np.nan,
            "mean_assignment_margin": float(np.mean(finite_margins))
            if finite_margins
            else np.nan,
            "mean_local_margin": float(np.mean(finite_margins))
            if finite_margins
            else np.nan,
            "min_local_margin": float(np.min(finite_margins))
            if finite_margins
            else np.nan,
            "path_score_trace_json": json.dumps(score_trace, ensure_ascii=False),
            "local_margin_trace_json": json.dumps(margin_trace, ensure_ascii=False),
            "path_mean_adjacent_similarity": coherence["path_mean_adjacent_similarity"],
            "path_min_adjacent_similarity": coherence["path_min_adjacent_similarity"],
            "final_to_ancestors_mean_similarity": coherence[
                "final_to_ancestors_mean_similarity"
            ],
            "resolved_depth": int(path["resolved_depth"]),
            "path_debug_json": json.dumps(path, ensure_ascii=False),
        }
        for depth in range(1, 8):
            row[f"predicted_level_{depth}_name"] = (
                path["path_names"][depth - 1]
                if len(path["path_names"]) >= depth
                else ""
            )
            row[f"predicted_level_{depth}_key"] = (
                path["path_keys"][depth - 1] if len(path["path_keys"]) >= depth else ""
            )
            row[f"level_{depth}_local_score"] = (
                score_trace[depth - 1] if len(score_trace) >= depth else ""
            )
            row[f"level_{depth}_local_margin"] = (
                margin_trace[depth - 1] if len(margin_trace) >= depth else ""
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_parent_to_children(nodes: pd.DataFrame) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = defaultdict(list)
    for row in nodes.to_dict("records"):
        parent = str(row.get("parent_key", "") or "")
        if parent:
            mapping[parent].append(str(row["node_key"]))
    return {key: sorted(set(values)) for key, values in mapping.items()}


def margin_from_scores(scores: np.ndarray) -> float:
    if len(scores) < 2:
        return float("nan")
    top2 = np.partition(scores, -2)[-2:]
    return float(top2[-1] - top2[-2])


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


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
    elapsed: float,
    n_child_scores: int,
    *,
    n_kmeans_distance_ops: int,
    embedding_dim: int,
    phase_seconds: dict[str, float],
    resource_metrics: dict[str, Any],
    cluster_rows: list[dict[str, Any]],
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
    return {
        "mode": "hierarchical_clustering",
        "n_products": int(len(output)),
        "elapsed_seconds": float(elapsed),
        "seconds_per_product": float(elapsed / max(1, len(output))),
        "throughput_products_per_second": float(len(output) / elapsed)
        if elapsed > 0
        else np.nan,
        **{key: float(value) for key, value in phase_seconds.items()},
        "inference_cost_unit": "M scalar multiply-adds for embedding dot products",
        "embedding_dim": int(embedding_dim),
        "approx_child_score_calls": int(n_child_scores),
        "approx_child_score_calls_per_product": float(
            n_child_scores / max(1, len(output))
        ),
        "similarity_dot_products": int(n_child_scores),
        "similarity_dot_products_per_product": float(
            n_child_scores / max(1, len(output))
        ),
        "approx_scalar_multiply_adds": int(n_child_scores * embedding_dim),
        "approx_million_scalar_multiply_adds": float(
            (n_child_scores * embedding_dim) / 1_000_000
        ),
        "approx_million_scalar_multiply_adds_per_product": float(
            (n_child_scores * embedding_dim) / max(1, len(output)) / 1_000_000
        ),
        "kmeans_distance_ops": int(n_kmeans_distance_ops),
        "kmeans_distance_ops_per_product": float(
            n_kmeans_distance_ops / max(1, len(output))
        ),
        "approx_kmeans_scalar_multiply_adds": int(
            n_kmeans_distance_ops * embedding_dim
        ),
        "approx_kmeans_million_scalar_multiply_adds": float(
            (n_kmeans_distance_ops * embedding_dim) / 1_000_000
        ),
        "n_clusters_fit": int(len(cluster_rows)),
        "mean_cluster_size": float(
            np.mean([row["cluster_size"] for row in cluster_rows])
        )
        if cluster_rows
        else np.nan,
        "median_cluster_size": float(
            np.median([row["cluster_size"] for row in cluster_rows])
        )
        if cluster_rows
        else np.nan,
        "mean_resolved_depth": float(
            pd.to_numeric(output["resolved_depth"], errors="coerce").mean()
        ),
        "median_path_score": float(
            pd.to_numeric(output["predicted_path_score"], errors="coerce").median()
        ),
        "median_assignment_margin": float(
            pd.to_numeric(output["mean_assignment_margin"], errors="coerce").median()
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

def write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--id-col", default="hashed_external_id")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--original-l1-col", default="level_1_name")
    parser.add_argument(
        "--l1-predictions-path",
        default=None,
        help=(
            "Optional predictions CSV produced by the fine-tuned L1 pipeline. "
            "When provided, pipeline 2 starts from this L1 instead of routing "
            "L1 with L2-L7 category centroids."
        ),
    )
    parser.add_argument("--l1-prediction-key-col", default="predicted_level_1_key")
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument(
        "--k-mode", choices=["children", "sqrt", "auto"], default="children"
    )
    parser.add_argument("--fixed-k", type=int, default=None)
    parser.add_argument(
        "--cluster-category-assignment",
        choices=["hungarian", "greedy"],
        default="hungarian",
        help=(
            "How to map cluster centroids to taxonomy children. 'hungarian' "
            "maximizes the global centroid/category similarity assignment."
        ),
    )
    parser.add_argument("--max-clusters-per-parent", type=int, default=64)
    parser.add_argument("--min-products-per-parent", type=int, default=10)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
