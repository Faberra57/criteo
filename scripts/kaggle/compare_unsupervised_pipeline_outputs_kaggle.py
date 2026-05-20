#!/usr/bin/env python3
"""Compare unsupervised L2-L7 pipeline outputs.

The comparison remains proxy-based because L2-L7 ground truth is absent.
It reports runtime metrics saved by each predictor plus output diagnostics:
depth, score, L1 agreement if available, path diversity and top-path share.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def main() -> None:
    args = parse_args()
    rows = []
    frames: list[tuple[str, pd.DataFrame]] = []
    for path_str in args.prediction_paths:
        path = Path(path_str)
        if not path.exists():
            print(f"WARNING missing file: {path}")
            continue
        df = pd.read_csv(path)
        metrics = load_metrics(path.with_suffix(".metrics.json"))
        label = str(metrics.get("mode") or path.stem)
        rows.append({**metrics, **diagnose_predictions(df), "prediction_path": str(path)})
        frames.append((label, df))
    out = pd.DataFrame(rows)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Comparison saved to {output_path}")
    print(out.to_string(index=False))
    if len(frames) >= 2:
        pairwise_path = Path(args.pairwise_output_path) if args.pairwise_output_path else output_path.with_name(output_path.stem + "_pairwise.csv")
        pairwise = pairwise_agreements(frames, id_col=args.id_col, low_margin_threshold=args.low_margin_threshold)
        pairwise.to_csv(pairwise_path, index=False)
        print(f"Pairwise agreement saved to {pairwise_path}")
        print(pairwise.to_string(index=False))


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", payload)
    return metrics if isinstance(metrics, dict) else {}


def diagnose_predictions(df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    path_col = "predicted_taxonomy_path"
    score_col = "predicted_path_score"
    depth_col = "resolved_depth"
    path_counts = df[path_col].fillna("").astype(str).value_counts() if path_col in df else pd.Series(dtype=float)
    path_probs = path_counts / max(1, n)
    entropy = float(-(path_probs * np.log(path_probs + 1e-12)).sum()) if len(path_probs) else np.nan
    diagnostics: dict[str, Any] = {
        "n_rows_output": n,
        "n_unique_paths_output": int(path_counts.size) if path_col in df else np.nan,
        "top_path_share_output": float(path_probs.iloc[0]) if len(path_probs) else np.nan,
        "path_entropy_output": entropy,
        "path_normalized_entropy_output": float(entropy / np.log(max(2, path_counts.size))) if len(path_probs) else np.nan,
        "mean_resolved_depth_output": float(pd.to_numeric(df[depth_col], errors="coerce").mean()) if depth_col in df else np.nan,
        "median_resolved_depth_output": float(pd.to_numeric(df[depth_col], errors="coerce").median()) if depth_col in df else np.nan,
        "median_score_output": float(pd.to_numeric(df[score_col], errors="coerce").median()) if score_col in df else np.nan,
        "mean_score_output": float(pd.to_numeric(df[score_col], errors="coerce").mean()) if score_col in df else np.nan,
    }
    if depth_col in df:
        depth = pd.to_numeric(df[depth_col], errors="coerce")
        diagnostics["share_depth_le_2_output"] = float((depth <= 2).mean())
        diagnostics["share_depth_ge_4_output"] = float((depth >= 4).mean())
        diagnostics["share_depth_ge_6_output"] = float((depth >= 6).mean())
        for level, share in depth.value_counts(normalize=True).sort_index().items():
            if pd.notna(level):
                diagnostics[f"share_resolved_depth_{int(level)}_output"] = float(share)
    for col in [
        "top1_top2_margin",
        "mean_local_margin",
        "min_local_margin",
        "embedding_top1_top2_margin",
        "rerank_top1_top2_margin",
        "l1_top1_top2_margin",
    ]:
        if col in df:
            values = pd.to_numeric(df[col], errors="coerce")
            diagnostics[f"median_{col}_output"] = float(values.median())
            diagnostics[f"mean_{col}_output"] = float(values.mean())
            for threshold in [0.01, 0.03, 0.05]:
                diagnostics[f"share_{col}_lt_{threshold}_output"] = float((values < threshold).mean())
    for col in [
        "product_path_similarity",
        "final_category_score",
        "path_mean_adjacent_similarity",
        "path_min_adjacent_similarity",
        "final_to_ancestors_mean_similarity",
    ]:
        if col in df:
            diagnostics[f"mean_{col}_output"] = float(pd.to_numeric(df[col], errors="coerce").mean())
    for depth in range(1, 8):
        col = f"predicted_level_{depth}_name"
        if col in df:
            values = df[col].fillna("").astype(str)
            non_empty = values[values != ""]
            if len(non_empty):
                level_counts = non_empty.value_counts(normalize=True)
                diagnostics[f"level_{depth}_n_unique_output"] = int(non_empty.nunique())
                diagnostics[f"level_{depth}_top_share_output"] = float(level_counts.iloc[0])
    if {"input_level_1_name", "predicted_level_1_name"}.issubset(df.columns):
        mask = df["input_level_1_name"].fillna("") != ""
        diagnostics["l1_agreement_output"] = (
            float((df.loc[mask, "input_level_1_name"] == df.loc[mask, "predicted_level_1_name"]).mean())
            if mask.any()
            else np.nan
        )
    return diagnostics


def pairwise_agreements(
    frames: list[tuple[str, pd.DataFrame]],
    *,
    id_col: str,
    low_margin_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prepared = [(label, prepare_for_pairwise(df, id_col=id_col)) for label, df in frames]
    for left_idx in range(len(prepared)):
        for right_idx in range(left_idx + 1, len(prepared)):
            left_label, left = prepared[left_idx]
            right_label, right = prepared[right_idx]
            merged = left.merge(right, on="_join_id", suffixes=("_left", "_right"))
            row: dict[str, Any] = {
                "method_left": left_label,
                "method_right": right_label,
                "n_overlap": int(len(merged)),
            }
            if len(merged) == 0:
                rows.append(row)
                continue
            exact = merged["predicted_taxonomy_path_left"].fillna("").astype(str) == merged["predicted_taxonomy_path_right"].fillna("").astype(str)
            row["exact_path_agreement"] = float(exact.mean())
            for depth in range(1, 8):
                col_left = f"predicted_level_{depth}_name_left"
                col_right = f"predicted_level_{depth}_name_right"
                if col_left in merged and col_right in merged:
                    left_values = merged[col_left].fillna("").astype(str)
                    right_values = merged[col_right].fillna("").astype(str)
                    non_empty = (left_values != "") | (right_values != "")
                    row[f"level_{depth}_agreement"] = float((left_values[non_empty] == right_values[non_empty]).mean()) if non_empty.any() else np.nan
            margin_left = best_margin_series(merged, suffix="_left")
            margin_right = best_margin_series(merged, suffix="_right")
            min_margin = pd.concat([margin_left, margin_right], axis=1).min(axis=1)
            disagreement = ~exact
            row["disagreement_share"] = float(disagreement.mean())
            row["low_margin_disagreement_share"] = float((disagreement & (min_margin < low_margin_threshold)).mean())
            row["median_min_margin_on_disagreement"] = float(min_margin[disagreement].median()) if disagreement.any() else np.nan
            if {"predicted_level_1_name_left", "predicted_level_1_name_right"}.issubset(merged.columns):
                row["l1_pair_agreement"] = float(
                    (
                        merged["predicted_level_1_name_left"].fillna("").astype(str)
                        == merged["predicted_level_1_name_right"].fillna("").astype(str)
                    ).mean()
                )
            rows.append(row)
    return pd.DataFrame(rows)


def prepare_for_pairwise(df: pd.DataFrame, *, id_col: str) -> pd.DataFrame:
    work = df.copy()
    if id_col in work.columns:
        work["_join_id"] = work[id_col].astype(str)
    else:
        work["_join_id"] = np.arange(len(work)).astype(str)
    keep = ["_join_id", "predicted_taxonomy_path"]
    keep.extend([f"predicted_level_{depth}_name" for depth in range(1, 8) if f"predicted_level_{depth}_name" in work.columns])
    keep.extend([col for col in ["top1_top2_margin", "mean_local_margin", "min_local_margin", "embedding_top1_top2_margin", "rerank_top1_top2_margin"] if col in work.columns])
    return work[[col for col in keep if col in work.columns]].drop_duplicates("_join_id")


def best_margin_series(df: pd.DataFrame, *, suffix: str) -> pd.Series:
    candidates = [
        f"top1_top2_margin{suffix}",
        f"mean_local_margin{suffix}",
        f"embedding_top1_top2_margin{suffix}",
        f"rerank_top1_top2_margin{suffix}",
        f"min_local_margin{suffix}",
    ]
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_paths", nargs="+")
    parser.add_argument("--output-path", default="/kaggle/working/pipeline_comparison_summary.csv")
    parser.add_argument("--pairwise-output-path", default=None)
    parser.add_argument("--id-col", default="hashed_external_id")
    parser.add_argument("--low-margin-threshold", type=float, default=0.03)
    return parser.parse_args()


if __name__ == "__main__":
    main()
