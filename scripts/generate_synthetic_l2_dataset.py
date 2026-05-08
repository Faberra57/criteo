from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synthetic_generation import (  # noqa: E402
    SyntheticGenerationConfig,
    generate_bootstrap_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a bootstrap synthetic L2 dataset and a real annotation sample."
    )
    parser.add_argument("--taxonomy-path", default="taxonomy.txt")
    parser.add_argument("--category-keywords-path", default="categories_level_2.csv")
    parser.add_argument("--catalog-path", default="data/ensae_export_without_l1.parquet")
    parser.add_argument("--ground-truth-path", default="data/ground_truth_level_1.parquet")
    parser.add_argument("--output-synthetic-path", default="data/synthetic_l2_seed.csv")
    parser.add_argument(
        "--output-annotation-path",
        default="data/real_l2_annotation_sample.csv",
    )
    parser.add_argument(
        "--output-summary-path",
        default="data/synthetic_l2_generation_summary.json",
    )
    parser.add_argument("--examples-per-l2", type=int, default=8)
    parser.add_argument("--annotation-samples-per-l1", type=int, default=15)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SyntheticGenerationConfig(
        taxonomy_path=args.taxonomy_path,
        category_keywords_path=args.category_keywords_path,
        catalog_path=args.catalog_path,
        ground_truth_path=args.ground_truth_path,
        output_synthetic_path=args.output_synthetic_path,
        output_annotation_path=args.output_annotation_path,
        output_summary_path=args.output_summary_path,
        examples_per_l2=args.examples_per_l2,
        annotation_samples_per_l1=args.annotation_samples_per_l1,
        random_seed=args.random_seed,
    )

    synthetic_df, annotation_df, summary = generate_bootstrap_artifacts(config)
    print(f"Synthetic dataset saved to {config.output_synthetic_path}")
    print(f"Annotation sample saved to {config.output_annotation_path}")
    print(f"Summary saved to {config.output_summary_path}")
    print(f"Synthetic rows: {len(synthetic_df)}")
    print(f"Annotation rows: {len(annotation_df)}")
    print(f"Level-2 nodes covered: {summary['n_level2_nodes']}")
    print("\nSynthetic preview:")
    print(synthetic_df.head(8).to_string(index=False))
    print("\nAnnotation preview:")
    print(annotation_df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
