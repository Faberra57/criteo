from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.embedding_model_registry import (  # noqa: E402
    find_catalog_row,
    load_embedding_model_config,
    save_embedding_model_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Set the active embedding models used by level1 and level2 pipelines."
    )
    parser.add_argument("--config-path", default="data/embedding_models.json")
    parser.add_argument("--catalog-path", default=None)
    parser.add_argument("--level1-model", default=None)
    parser.add_argument("--level2-model", default=None)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow model ids that are not present in the catalog.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_embedding_model_config(args.config_path)
    catalog_path = args.catalog_path or config.catalog_path

    if args.catalog_path:
        config.catalog_path = args.catalog_path

    if args.level1_model:
        _validate_model(args.level1_model, catalog_path, args.allow_missing, "level1")
        config.level1_model = args.level1_model.strip()
    if args.level2_model:
        _validate_model(args.level2_model, catalog_path, args.allow_missing, "level2")
        config.level2_model = args.level2_model.strip()

    save_embedding_model_config(config, args.config_path)

    print("Saved embedding configuration")
    print(f"  config: {args.config_path}")
    print(f"  level1: {config.level1_model}")
    print(f"  level2: {config.level2_model}")
    print(f"  catalog: {config.catalog_path}")


def _validate_model(
    model_name: str,
    catalog_path: str,
    allow_missing: bool,
    task_name: str,
) -> None:
    row = find_catalog_row(model_name, catalog_path=catalog_path)
    if row is not None:
        return
    if allow_missing:
        return
    raise ValueError(
        f"{task_name} model '{model_name}' not found in catalog '{catalog_path}'. "
        "Use --allow-missing to bypass this check."
    )


if __name__ == "__main__":
    main()
