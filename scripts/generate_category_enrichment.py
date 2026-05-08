from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.category_enrichment import (  # noqa: E402
    DEFAULT_MLX_LOCAL_MODEL_DIR,
    DEFAULT_MLX_MODEL,
    DEFAULT_GITHUB_MODELS_BASE_URL,
    CategoryEnrichmentConfig,
    generate_category_enrichment,
    prepare_local_model_reference,
)


DEFAULT_PRIORITY_CATEGORY_NAMES = [
    "hardware",
    "software",
    "office supplies",
    "vehicles & parts",
    "arts & entertainment",
    "business & industrial",
    "religious & ceremonial",
    "mature",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate LLM-enriched taxonomy reference texts per category level."
    )
    parser.add_argument("--taxonomy-path", default="dataset/taxonomy.txt")
    parser.add_argument("--output-dir", default="dataset/category_enrichment_local")
    parser.add_argument(
        "--provider",
        choices=["template", "local", "huggingface-local", "github-models"],
        default="local",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MLX_MODEL,
        help=(
            "Model name or local path used for generation. Defaults to "
            f"'{DEFAULT_MLX_MODEL}'."
        ),
    )
    parser.add_argument("--api-base-url", default=DEFAULT_GITHUB_MODELS_BASE_URL)
    parser.add_argument("--api-key-env", default="GITHUB_TOKEN")
    parser.add_argument(
        "--local-model-dir",
        default=DEFAULT_MLX_LOCAL_MODEL_DIR,
        help=(
            "Local directory used to store the downloaded MLX model. If it already exists, "
            "the model is loaded from there and not downloaded again."
        ),
    )
    parser.add_argument(
        "--mlx-backend",
        choices=["auto", "lm", "vlm"],
        default="auto",
        help="MLX backend for local generation. 'auto' inspects the local model config.",
    )
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download the model into --local-model-dir and exit without generating outputs.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=None,
        help="Levels to generate, for example: --levels 1 2 3",
    )
    parser.add_argument("--max-children", type=int, default=5)
    parser.add_argument("--max-descendants", type=int, default=5)
    parser.add_argument("--min-lexical-variants", type=int, default=5)
    parser.add_argument(
        "--priority-category-names",
        nargs="*",
        default=DEFAULT_PRIORITY_CATEGORY_NAMES,
        help=(
            "Category names that receive more lexical variants. Defaults to the weak "
            "Level 1 classes observed in retrieval validation. Pass no values after the "
            "flag to disable this list."
        ),
    )
    parser.add_argument(
        "--priority-node-keys",
        nargs="*",
        default=None,
        help="Exact taxonomy node keys that receive more lexical variants.",
    )
    parser.add_argument(
        "--priority-min-lexical-variants",
        type=int,
        default=24,
        help="Minimum lexical variants generated for priority categories.",
    )
    parser.add_argument("--request-max-tokens", type=int, default=220)
    parser.add_argument(
        "--request-batch-size",
        type=int,
        default=4,
        help="Local MLX backend uses true batch generation when batch size > 1.",
    )
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--retry-backoff-seconds", type=float, default=65.0)
    parser.add_argument("--max-retry-delay-seconds", type=float, default=120.0)
    parser.add_argument("--network-max-retries", type=int, default=3)
    parser.add_argument("--network-retry-backoff-seconds", type=float, default=5.0)
    parser.add_argument("--save-every-n", type=int, default=50)
    parser.add_argument("--limit-per-level", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = CategoryEnrichmentConfig(
        taxonomy_path=args.taxonomy_path,
        output_dir=args.output_dir,
        provider=args.provider,
        model=args.model,
        api_base_url=args.api_base_url,
        api_key_env=args.api_key_env or "GITHUB_TOKEN",
        local_model_dir=args.local_model_dir,
        model_revision=args.model_revision,
        mlx_backend=args.mlx_backend,
        trust_remote_code=args.trust_remote_code,
        temperature=args.temperature,
        levels=args.levels,
        max_children=args.max_children,
        max_descendants=args.max_descendants,
        min_lexical_variants=args.min_lexical_variants,
        priority_category_names=args.priority_category_names,
        priority_node_keys=args.priority_node_keys,
        priority_min_lexical_variants=args.priority_min_lexical_variants,
        request_max_tokens=args.request_max_tokens,
        request_batch_size=args.request_batch_size,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        max_retry_delay_seconds=args.max_retry_delay_seconds,
        network_max_retries=args.network_max_retries,
        network_retry_backoff_seconds=args.network_retry_backoff_seconds,
        save_every_n=args.save_every_n,
        limit_per_level=args.limit_per_level,
        overwrite=args.overwrite,
        sleep_seconds=args.sleep_seconds,
    )

    if args.download_only:
        model_ref = prepare_local_model_reference(config)
        print("MLX model download complete")
        print(f"Model: {config.model}")
        print(f"Local model path: {model_ref}")
        print(f"MLX backend: {config.mlx_backend}")
        return

    outputs = generate_category_enrichment(config)

    print("Category enrichment generation complete")
    print(f"Provider: {config.provider}")
    display_model = "<template>" if config.provider == "template" else (config.model or "<unknown>")
    print(f"Model: {display_model}")
    if config.provider in {"local", "huggingface-local"}:
        print(f"Local model path: {config.local_model_dir}")
        print(f"MLX backend: {config.mlx_backend}")
    print(f"Output dir: {config.output_dir}")
    for level, paths in sorted(outputs.items()):
        print(f"Level {level}:")
        print(f"  categories: {paths['categories_csv']}")
        print(f"  reference_texts: {paths['reference_texts_csv']}")


if __name__ == "__main__":
    main()
