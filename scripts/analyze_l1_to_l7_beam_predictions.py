#!/usr/bin/env python3
"""Analyze L1-to-L7 zero-shot beam-search predictions without deep ground truth."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "l1_to_l7_beam_predictions.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "latex" / "rapport_final" / "figures"

LEVELS = range(1, 8)
PALETTE = {
    "blue": "#2F5D8C",
    "orange": "#D9822B",
    "green": "#4B7F52",
    "red": "#A44A3F",
    "gray": "#5B6470",
    "light_gray": "#D9DEE7",
    "background": "#FBFAF7",
}


def clean_label(label: object, max_len: int = 42) -> str:
    text = str(label)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.set_facecolor(PALETTE["background"])
    ax.grid(axis=grid_axis, alpha=0.25, color="#9CA3AF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")


def parse_beam_paths(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def cumulative_score(score_trace: list[float]) -> float:
    if not score_trace:
        return 0.0
    return float(sum(score_trace) / len(score_trace))


def approximate_greedy_from_saved_beams(paths: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Approximate greedy decoding using only the retained beam alternatives.

    The original run does not store all children scored at every expansion. This
    function therefore reconstructs the locally greedy path from the final paths
    saved in beam_paths_json. It is a diagnostic proxy, not an exact replay.
    """
    candidates: list[dict[str, Any]] = []
    for path in paths:
        names = path.get("path_names") or []
        scores = path.get("score_trace") or []
        if isinstance(names, list) and isinstance(scores, list) and names:
            candidates.append({"names": names, "scores": [float(x) for x in scores]})
    if not candidates:
        return None

    prefix = [candidates[0]["names"][0]]
    score_trace: list[float] = []

    while True:
        next_options: dict[str, float] = {}
        for candidate in candidates:
            names = candidate["names"]
            scores = candidate["scores"]
            pos = len(prefix)
            if len(names) <= pos:
                continue
            if names[:pos] != prefix:
                continue
            score_idx = pos - 1
            if score_idx >= len(scores):
                continue
            node_name = str(names[pos])
            next_options[node_name] = max(next_options.get(node_name, -math.inf), float(scores[score_idx]))

        if not next_options:
            break

        next_node, next_score = max(next_options.items(), key=lambda item: item[1])
        prefix.append(next_node)
        score_trace.append(next_score)

        if len(prefix) >= 7:
            break

    return {
        "path_names": prefix,
        "score_trace": score_trace,
        "cumulative_score": cumulative_score(score_trace),
        "resolved_depth": len(prefix),
    }


def first_divergence_depth(path_a: list[str], path_b: list[str]) -> int:
    for idx, (a, b) in enumerate(zip(path_a, path_b), start=1):
        if a != b:
            return idx
    if len(path_a) != len(path_b):
        return min(len(path_a), len(path_b)) + 1
    return 0


def update_numeric(values: dict[str, list[float]], key: str, series: pd.Series) -> None:
    data = pd.to_numeric(series, errors="coerce").dropna()
    if not data.empty:
        values[key].extend(data.astype(float).tolist())


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": math.nan, "median": math.nan, "p10": math.nan, "p25": math.nan, "p75": math.nan, "p90": math.nan}
    s = pd.Series(values, dtype=float)
    return {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p10": float(s.quantile(0.10)),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
    }


def export_depth_plot(depth_counts: Counter[int], output_path: Path) -> None:
    depths = list(LEVELS)
    counts = [depth_counts.get(depth, 0) for depth in depths]
    total = max(1, sum(counts))
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=PALETTE["background"])
    bars = ax.bar([str(d) for d in depths], [100 * c / total for c in counts], color=PALETTE["blue"])
    style_axes(ax, grid_axis="y")
    ax.set_title("Profondeur finale des chemins prédits L1--L7", fontsize=14, weight="bold")
    ax.set_xlabel("Profondeur résolue")
    ax.set_ylabel("Part des produits (%)")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4, f"{100 * count / total:.1f}%", ha="center", fontsize=8)
    ax.set_ylim(0, max([100 * c / total for c in counts] + [1]) * 1.18)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def export_score_boxplot(score_values: dict[str, list[float]], output_path: Path) -> None:
    labels: list[str] = []
    data: list[list[float]] = []
    for level in LEVELS:
        key = "predicted_path_score" if level == 1 else f"level_{level}_local_score"
        values = score_values.get(key, [])
        if values:
            labels.append("Path" if level == 1 else f"L{level}")
            data.append(values)
    fig, ax = plt.subplots(figsize=(9.5, 5.2), facecolor=PALETTE["background"])
    ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False, medianprops={"color": PALETTE["red"]})
    for patch in ax.artists:
        patch.set_facecolor(PALETTE["light_gray"])
    style_axes(ax, grid_axis="y")
    ax.set_title("Distribution des scores cosinus par niveau", fontsize=14, weight="bold")
    ax.set_ylabel("Score cosinus")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def export_margin_hist(margins: list[float], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8), facecolor=PALETTE["background"])
    ax.hist(margins, bins=40, color=PALETTE["orange"], edgecolor="white")
    style_axes(ax, grid_axis="y")
    ax.set_title("Marge de confiance top-1 / top-2 du beam", fontsize=14, weight="bold")
    ax.set_xlabel("Différence de score cumulatif")
    ax.set_ylabel("Nombre de produits")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def export_beam_gain_hist(gains: list[float], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8), facecolor=PALETTE["background"])
    ax.hist(gains, bins=50, color=PALETTE["green"], edgecolor="white")
    ax.axvline(0, color=PALETTE["red"], linewidth=1.2)
    style_axes(ax, grid_axis="y")
    ax.set_title("Gain du beam search face au greedy approximé", fontsize=14, weight="bold")
    ax.set_xlabel("Score beam top-1 - score greedy approximé")
    ax.set_ylabel("Nombre de produits")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def export_l1_agreement_plot(l1_rows: list[dict[str, object]], output_path: Path) -> None:
    df = pd.DataFrame(l1_rows).sort_values("agreement_rate")
    fig_height = max(6.0, 0.34 * len(df))
    fig, ax = plt.subplots(figsize=(10.5, fig_height), facecolor=PALETTE["background"])
    ax.barh([clean_label(x, 34) for x in df["input_level_1_name"]], df["agreement_rate"] * 100, color=PALETTE["gray"])
    style_axes(ax, grid_axis="x")
    ax.set_title("Accord entre le label L1 d'entrée et le L1 prédit par le pipeline", fontsize=14, weight="bold")
    ax.set_xlabel("Accord top-1 L1 (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def export_top_paths_plot(path_counts: Counter[str], output_path: Path, top_n: int = 20) -> None:
    rows = path_counts.most_common(top_n)
    labels = [clean_label(path, 58) for path, _ in rows][::-1]
    counts = [count for _, count in rows][::-1]
    fig_height = max(6.0, 0.34 * len(rows))
    fig, ax = plt.subplots(figsize=(12.0, fig_height), facecolor=PALETTE["background"])
    ax.barh(labels, counts, color=PALETTE["blue"])
    style_axes(ax, grid_axis="x")
    ax.set_title(f"{top_n} chemins taxonomiques les plus prédits", fontsize=14, weight="bold")
    ax.set_xlabel("Nombre de produits")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunksize", type=int, default=5000)
    parser.add_argument("--top-n", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    usecols = [
        "hashed_external_id",
        "input_level_1_name",
        "predicted_taxonomy_path",
        "predicted_path_score",
        "resolved_depth",
        "beam_paths_json",
        "predicted_level_1_name",
    ]
    for level in range(2, 8):
        usecols.extend([f"predicted_level_{level}_name", f"level_{level}_local_score"])

    total_rows = 0
    depth_counts: Counter[int] = Counter()
    input_l1_counts: Counter[str] = Counter()
    predicted_l1_counts: Counter[str] = Counter()
    path_counts: Counter[str] = Counter()
    level_node_counts: dict[int, Counter[str]] = {level: Counter() for level in LEVELS}
    score_values: dict[str, list[float]] = defaultdict(list)
    l1_agreement_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    margins: list[float] = []
    beam_gains: list[float] = []
    greedy_diff_flags: list[int] = []
    divergence_depths: Counter[int] = Counter()
    alt_path_counts: list[int] = []

    for chunk in pd.read_csv(args.input, usecols=usecols, chunksize=args.chunksize):
        total_rows += len(chunk)
        depth_counts.update(pd.to_numeric(chunk["resolved_depth"], errors="coerce").dropna().astype(int).tolist())
        input_l1_counts.update(chunk["input_level_1_name"].dropna().astype(str).tolist())
        predicted_l1_counts.update(chunk["predicted_level_1_name"].dropna().astype(str).tolist())
        path_counts.update(chunk["predicted_taxonomy_path"].dropna().astype(str).tolist())
        update_numeric(score_values, "predicted_path_score", chunk["predicted_path_score"])

        for level in LEVELS:
            name_col = f"predicted_level_{level}_name"
            if name_col in chunk.columns:
                level_node_counts[level].update(chunk[name_col].dropna().astype(str).tolist())
            score_col = f"level_{level}_local_score"
            if score_col in chunk.columns:
                update_numeric(score_values, score_col, chunk[score_col])

        agreements = chunk["input_level_1_name"].astype(str) == chunk["predicted_level_1_name"].astype(str)
        for label, ok in zip(chunk["input_level_1_name"].astype(str), agreements):
            l1_agreement_counts[label][0] += int(bool(ok))
            l1_agreement_counts[label][1] += 1

        for _, row in chunk[["beam_paths_json", "predicted_path_score"]].iterrows():
            paths = parse_beam_paths(row["beam_paths_json"])
            alt_path_counts.append(len(paths))
            if len(paths) >= 2:
                top1 = float(paths[0].get("cumulative_score", row["predicted_path_score"]) or 0.0)
                top2 = float(paths[1].get("cumulative_score", 0.0) or 0.0)
                margins.append(top1 - top2)

            greedy = approximate_greedy_from_saved_beams(paths)
            if greedy is None or not paths:
                continue
            beam_names = [str(x) for x in (paths[0].get("path_names") or [])]
            greedy_names = [str(x) for x in greedy["path_names"]]
            greedy_diff_flags.append(int(beam_names != greedy_names))
            divergence_depths.update([first_divergence_depth(beam_names, greedy_names)])
            beam_score = float(paths[0].get("cumulative_score", row["predicted_path_score"]) or 0.0)
            beam_gains.append(beam_score - float(greedy["cumulative_score"]))

    l1_rows = []
    for label, (ok, count) in l1_agreement_counts.items():
        l1_rows.append({"input_level_1_name": label, "n_products": count, "agreement": ok, "agreement_rate": ok / count if count else math.nan})
    l1_df = pd.DataFrame(l1_rows).sort_values(["agreement_rate", "n_products"], ascending=[True, False])

    level_rows = []
    for level in LEVELS:
        score_key = "predicted_path_score" if level == 1 else f"level_{level}_local_score"
        stats = summarize(score_values.get(score_key, []))
        level_rows.append(
            {
                "level": level,
                "n_predicted_nodes": sum(level_node_counts[level].values()),
                "n_unique_nodes": len(level_node_counts[level]),
                **{f"score_{k}": v for k, v in stats.items()},
            }
        )
    level_df = pd.DataFrame(level_rows)

    summary_rows = [
        {"metric": "n_products", "value": total_rows},
        {"metric": "mean_resolved_depth", "value": sum(depth * count for depth, count in depth_counts.items()) / max(1, total_rows)},
        {"metric": "share_depth_1", "value": depth_counts.get(1, 0) / max(1, total_rows)},
        {"metric": "share_depth_2_or_more", "value": sum(count for depth, count in depth_counts.items() if depth >= 2) / max(1, total_rows)},
        {"metric": "share_depth_4_or_more", "value": sum(count for depth, count in depth_counts.items() if depth >= 4) / max(1, total_rows)},
        {"metric": "n_unique_predicted_paths", "value": len(path_counts)},
        {"metric": "top1_path_share", "value": path_counts.most_common(1)[0][1] / max(1, total_rows) if path_counts else math.nan},
        {"metric": "l1_agreement_rate", "value": sum(v[0] for v in l1_agreement_counts.values()) / max(1, sum(v[1] for v in l1_agreement_counts.values()))},
        {"metric": "median_predicted_path_score", "value": summarize(score_values["predicted_path_score"])["median"]},
        {"metric": "median_beam_margin_top1_top2", "value": summarize(margins)["median"]},
        {"metric": "share_beam_differs_from_greedy_proxy", "value": sum(greedy_diff_flags) / max(1, len(greedy_diff_flags))},
        {"metric": "median_beam_score_gain_vs_greedy_proxy", "value": summarize(beam_gains)["median"]},
        {"metric": "mean_saved_alternative_paths", "value": sum(alt_path_counts) / max(1, len(alt_path_counts))},
    ]
    summary_df = pd.DataFrame(summary_rows)

    depth_df = pd.DataFrame([{"resolved_depth": depth, "n_products": count, "share": count / max(1, total_rows)} for depth, count in sorted(depth_counts.items())])
    top_paths_df = pd.DataFrame(
        [{"predicted_taxonomy_path": path, "n_products": count, "share": count / max(1, total_rows)} for path, count in path_counts.most_common(args.top_n)]
    )
    top_nodes_rows = []
    for level, counter in level_node_counts.items():
        for node, count in counter.most_common(args.top_n):
            top_nodes_rows.append({"level": level, "node": node, "n_products": count, "share": count / max(1, total_rows)})
    top_nodes_df = pd.DataFrame(top_nodes_rows)
    divergence_df = pd.DataFrame(
        [{"divergence_depth": depth, "n_products": count, "share": count / max(1, len(greedy_diff_flags))} for depth, count in sorted(divergence_depths.items())]
    )

    prefix = args.output_dir / "l1_to_l7"
    summary_df.to_csv(f"{prefix}_beam_analysis_summary.csv", index=False)
    depth_df.to_csv(f"{prefix}_resolved_depth.csv", index=False)
    level_df.to_csv(f"{prefix}_level_score_summary.csv", index=False)
    l1_df.to_csv(f"{prefix}_l1_agreement.csv", index=False)
    top_paths_df.to_csv(f"{prefix}_top_predicted_paths.csv", index=False)
    top_nodes_df.to_csv(f"{prefix}_top_predicted_nodes.csv", index=False)
    pd.DataFrame({"top1_top2_margin": margins}).to_csv(f"{prefix}_beam_margins.csv", index=False)
    pd.DataFrame({"beam_gain_vs_greedy_proxy": beam_gains}).to_csv(f"{prefix}_beam_vs_greedy.csv", index=False)
    divergence_df.to_csv(f"{prefix}_beam_greedy_divergence.csv", index=False)

    export_depth_plot(depth_counts, args.output_dir / "l1_to_l7_resolved_depth.pdf")
    export_score_boxplot(score_values, args.output_dir / "l1_to_l7_score_by_level.pdf")
    export_margin_hist(margins, args.output_dir / "l1_to_l7_beam_margin.pdf")
    export_beam_gain_hist(beam_gains, args.output_dir / "l1_to_l7_beam_vs_greedy_gain.pdf")
    export_l1_agreement_plot(l1_rows, args.output_dir / "l1_to_l7_l1_agreement.pdf")
    export_top_paths_plot(path_counts, args.output_dir / "l1_to_l7_top_predicted_paths.pdf", top_n=20)

    print(summary_df.to_string(index=False))
    print(f"Exported analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
