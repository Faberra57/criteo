from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hierarchical_prototype_search import (  # noqa: E402
    BeamSearchConfig,
    PrototypeIndex,
    predict_dataframe,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict hierarchical categories with multi-prototype retrieval and beam search."
    )
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--input-format", choices=["csv", "parquet"], default="parquet")
    parser.add_argument("--index-dir", default="artifacts/hierarchical_prototype_index")
    parser.add_argument("--output-path", default="data/hierarchical_predictions.csv")
    parser.add_argument("--level1-col", default="level_1_name")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--title-col", default="title")
    parser.add_argument("--description-col", default="description")
    parser.add_argument("--brand-col", default="brand")
    parser.add_argument("--price-col", default="sale_price")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--ambiguity-margin", type=float, default=0.03)
    parser.add_argument("--resolution-margin", type=float, default=0.05)
    parser.add_argument("--score-aggregation", choices=["max", "mean_top_k"], default="mean_top_k")
    parser.add_argument("--top-k-prototypes", type=int, default=3)
    return parser


def load_frame(path: str, file_format: str) -> pd.DataFrame:
    if file_format == "csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def main() -> None:
    args = build_parser().parse_args()
    df = load_frame(args.input_path, args.input_format)
    index = PrototypeIndex.load(args.index_dir)
    search_config = BeamSearchConfig(
        max_depth=args.max_depth,
        beam_width=args.beam_width,
        ambiguity_margin=args.ambiguity_margin,
        resolution_margin=args.resolution_margin,
        score_aggregation=args.score_aggregation,
        top_k_prototypes=args.top_k_prototypes,
    )

    predictions_df = predict_dataframe(
        df,
        index=index,
        level1_col=args.level1_col,
        text_col=args.text_col,
        title_col=args.title_col,
        description_col=args.description_col,
        brand_col=args.brand_col,
        price_col=args.price_col,
        batch_size=args.batch_size,
        device=args.device,
        search_config=search_config,
    )
    predictions_df.to_csv(args.output_path, index=False)
    print(f"Predictions saved to {args.output_path}")
    print(predictions_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
