from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lib.preprocessing_core import (  # noqa: E402
    PreprocessingConfig,
    load_catalog_and_ground_truth,
    merge_catalog_with_ground_truth,
    preprocess_for_nlp,
    save_preprocessed_frame,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess catalog data for LV2 experiments."
    )
    parser.add_argument(
        "--catalog-path", default="dataset/ensae_export_without_l1.parquet"
    )
    parser.add_argument(
        "--ground-truth-path", default="dataset/ground_truth_level_1.parquet"
    )
    parser.add_argument("--output-path", default="dataset/preprocessed_lv2.parquet")
    parser.add_argument("--title-col", default="title")
    parser.add_argument("--desc-col", default="description")
    parser.add_argument("--brand-col", default="brand")
    parser.add_argument("--id-col", default="hashed_external_id")
    parser.add_argument("--price-col", default="sale_price")
    parser.add_argument("--category-col", default="level_1_name")
    parser.add_argument("--max-description-chars", type=int, default=800)
    parser.add_argument("--currency", default="USD")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    catalog_df, ground_truth_df = load_catalog_and_ground_truth(
        catalog_path=args.catalog_path,
        ground_truth_path=args.ground_truth_path,
    )
    merged_df = merge_catalog_with_ground_truth(
        catalog_df=catalog_df,
        ground_truth_df=ground_truth_df,
        id_col=args.id_col,
        category_col=args.category_col,
    )

    processed_df, _ = preprocess_for_nlp(
        merged_df,
        config=PreprocessingConfig(
            title_col=args.title_col,
            desc_col=args.desc_col,
            brand_col=args.brand_col,
            id_col=args.id_col,
            price_col=args.price_col,
            category_col=args.category_col,
            max_description_chars=args.max_description_chars,
            currency=args.currency,
        ),
    )
    save_preprocessed_frame(processed_df, args.output_path)
    print(f"Saved preprocessed data to {args.output_path}")
    print(processed_df.head())


if __name__ == "__main__":
    main()
