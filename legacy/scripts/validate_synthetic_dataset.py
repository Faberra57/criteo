from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synthetic_validation import (  # noqa: E402
    SyntheticValidationConfig,
    evaluate_real_vs_synthetic,
    evaluate_sibling_classification,
    evaluate_synthetic_to_real_transfer,
    format_validation_summary,
    save_validation_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a synthetic L2 dataset against real product data."
    )
    parser.add_argument("--real-path", required=True)
    parser.add_argument("--synthetic-path", required=True)
    parser.add_argument("--real-format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--synthetic-format", choices=["parquet", "csv"], default="csv")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--level1-col", default="level_1_name")
    parser.add_argument("--level2-col", default="level_2_name")
    parser.add_argument("--title-col", default="title")
    parser.add_argument("--description-col", default="description")
    parser.add_argument("--brand-col", default="brand")
    parser.add_argument("--price-col", default="sale_price")
    parser.add_argument("--hash-col", default="hashed_external_id")
    parser.add_argument("--parent-value", default=None)
    parser.add_argument("--min-examples-per-class", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-report", default=None)
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["real-vs-synth", "sibling-clf", "synth-to-real"],
        default=["real-vs-synth", "sibling-clf", "synth-to-real"],
    )
    return parser


def load_frame(path: str, file_format: str) -> pd.DataFrame:
    if file_format == "csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def main() -> None:
    args = build_parser().parse_args()
    real_df = load_frame(args.real_path, args.real_format)
    synthetic_df = load_frame(args.synthetic_path, args.synthetic_format)

    config = SyntheticValidationConfig(
        text_col=args.text_col,
        level1_col=args.level1_col,
        level2_col=args.level2_col,
        title_col=args.title_col,
        description_col=args.description_col,
        brand_col=args.brand_col,
        price_col=args.price_col,
        hash_col=args.hash_col,
        parent_value=args.parent_value,
        min_examples_per_class=args.min_examples_per_class,
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        cv_folds=args.cv_folds,
        random_seed=args.random_seed,
    )

    results: dict[str, object] = {"config": vars(args), "metrics": {}}

    if "real-vs-synth" in args.steps:
        results["metrics"]["real_vs_synthetic"] = evaluate_real_vs_synthetic(
            real_df, synthetic_df, config
        )

    if "sibling-clf" in args.steps:
        results["metrics"]["sibling_classification"] = evaluate_sibling_classification(
            synthetic_df, config
        )

    if "synth-to-real" in args.steps:
        results["metrics"]["synthetic_to_real_transfer"] = evaluate_synthetic_to_real_transfer(
            synthetic_df, real_df, config
        )

    for name, metrics in results["metrics"].items():
        print(f"\n=== {name} ===")
        print(format_validation_summary(metrics))

    if args.output_report:
        save_validation_report(results, args.output_report)
        print(f"\nSaved report to {args.output_report}")


if __name__ == "__main__":
    main()
