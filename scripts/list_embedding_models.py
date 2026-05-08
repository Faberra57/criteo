from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.embedding_model_registry import (  # noqa: E402
    EmbeddingModelConfig,
    find_catalog_row,
    load_embedding_model_catalog,
    load_embedding_model_config,
)


DEFAULT_COLUMNS = [
    "Rank (Borda)",
    "model_id",
    "model_label",
    "Mean (Task)",
    "Classification",
    "Retrieval",
    "STS",
    "Memory Usage (MB)",
    "Embedding Dimensions",
    "Max Tokens",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List embedding model candidates from the project catalog."
    )
    parser.add_argument("--catalog-path", default=None)
    parser.add_argument("--config-path", default="data/embedding_models.json")
    parser.add_argument("--sort-by", default="Mean (Task)")
    parser.add_argument("--top-k", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_embedding_model_config(args.config_path)
    catalog_path = args.catalog_path or config.catalog_path
    catalog = load_embedding_model_catalog(catalog_path)

    if args.sort_by not in catalog.columns:
        raise ValueError(f"Unknown sort column: {args.sort_by}")

    ranked = catalog.sort_values(args.sort_by, ascending=False, na_position="last")
    columns = [col for col in DEFAULT_COLUMNS if col in ranked.columns]

    print("Active embedding models")
    print(f"  level1: {config.level1_model}")
    print(f"  level2: {config.level2_model}")
    print("")

    for task_name, model_name in (
        ("level1", config.level1_model),
        ("level2", config.level2_model),
    ):
        row = find_catalog_row(model_name, catalog_path=catalog_path)
        if row is None:
            print(f"{task_name}: {model_name} not found in catalog")
            continue
        print(
            f"{task_name}: rank={row.get('Rank (Borda)')} "
            f"mean={row.get('Mean (Task)')} "
            f"classif={row.get('Classification')} "
            f"retrieval={row.get('Retrieval')}"
        )
    print("")

    print(f"Top {args.top_k} models sorted by '{args.sort_by}'")
    print(ranked[columns].head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()
