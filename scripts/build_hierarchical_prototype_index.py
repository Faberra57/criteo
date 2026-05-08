from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hierarchical_prototype_search import (  # noqa: E402
    PrototypeBuildConfig,
    build_and_save_prototype_index,
    build_prototype_records,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a hierarchical multi-prototype index from taxonomy and synthetic data."
    )
    parser.add_argument("--taxonomy-path", default="taxonomy.txt")
    parser.add_argument("--category-keywords-path", default="categories_level_2.csv")
    parser.add_argument("--synthetic-dataset-path", default="data/synthetic_l2_llm_targeted_50.csv")
    parser.add_argument(
        "--model-name",
        default="level2",
        help="Explicit model id or one of: level1, level2, default.",
    )
    parser.add_argument("--output-dir", default="artifacts/hierarchical_prototype_index")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-descendants-per-node", type=int, default=8)
    parser.add_argument("--max-synthetic-examples-per-node", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--texts-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PrototypeBuildConfig(
        taxonomy_path=args.taxonomy_path,
        category_keywords_path=args.category_keywords_path,
        synthetic_dataset_path=args.synthetic_dataset_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        device=args.device,
        max_descendants_per_node=args.max_descendants_per_node,
        max_synthetic_examples_per_node=args.max_synthetic_examples_per_node,
        batch_size=args.batch_size,
    )

    if args.texts_only:
        prototypes_df, _, _ = build_prototype_records(config)
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "prototype_texts.parquet"
        prototypes_df.to_parquet(output_path, index=False)
        print(f"Model configured for build: {config.model_name}")
        print(f"Prototype texts saved to {output_path}")
        print(f"Rows: {len(prototypes_df)}")
        print(prototypes_df.head(12).to_string(index=False))
        return

    index = build_and_save_prototype_index(config)
    print(f"Prototype index saved to {config.output_dir}")
    print(f"Model: {index.model_name}")
    print(f"Prototype rows: {len(index.prototypes_df)}")
    print(index.prototypes_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
