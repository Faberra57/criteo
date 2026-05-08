from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DEFAULT_INPUT_PATH = "/kaggle/input/criteo-finetuning/preprocessed_lv2.parquet"
DEFAULT_ENRICHMENT_DIR = "/kaggle/input/criteo-finetuning/category_enrichment"
DEFAULT_TAXONOMY_PATH = "/kaggle/input/criteo-finetuning/taxonomy.txt"
DEFAULT_L1_MODEL_PATH = "/kaggle/input/criteo-l1-bge-small-finetuned/best_model"
DEFAULT_L1_RUN_DIR = "/kaggle/input/criteo-l1-bge-small-finetuned"
DEFAULT_OUTPUT_PATH = "/kaggle/working/l1_to_l7_beam_predictions.csv"
DEFAULT_CACHE_DIR = "/kaggle/working/prototype_embedding_cache"
DEFAULT_HF_CACHE = "/kaggle/working/huggingface_cache"


@dataclass(slots=True)
class PipelineConfig:
    input_path: str = DEFAULT_INPUT_PATH
    input_format: str = "parquet"
    output_path: str = DEFAULT_OUTPUT_PATH
    enrichment_dir: str = DEFAULT_ENRICHMENT_DIR
    taxonomy_path: Optional[str] = DEFAULT_TAXONOMY_PATH
    l1_model_path: str = DEFAULT_L1_MODEL_PATH
    l1_run_dir: str = DEFAULT_L1_RUN_DIR
    deep_model_name: str = "BAAI/bge-base-en-v1.5"
    prototype_cache_dir: Optional[str] = DEFAULT_CACHE_DIR
    hf_cache_dir: str = DEFAULT_HF_CACHE
    overwrite_cache: bool = False
    text_col: str = "text"
    id_col: str = "hashed_external_id"
    original_l1_col: str = "level_1_name"
    title_col: str = "title"
    description_col: str = "description"
    brand_col: str = "brand"
    price_col: str = "sale_price"
    sample_size: Optional[int] = None
    random_seed: int = 42
    l1_top_k: int = 5
    beam_width: int = 5
    children_top_k: int = 8
    max_depth: int = 7
    output_top_paths: int = 10
    score_aggregation: str = "max"
    batch_size: int = 128
    device_l1: Optional[str] = None
    device_deep: Optional[str] = None
    normalize_embeddings: bool = True


@dataclass(slots=True)
class PrototypeIndex:
    prototypes: pd.DataFrame
    embeddings: np.ndarray
    node_metadata: dict[str, dict[str, object]]
    parent_to_children: dict[str, list[str]]
    node_to_indices: dict[str, list[int]] | None = None
    parent_to_proto_indices: dict[str, np.ndarray] | None = None
    parent_to_proto_child_keys: dict[str, list[str]] | None = None

    def __post_init__(self) -> None:
        self.prototypes = self.prototypes.reset_index(drop=True).copy()
        self.prototypes["prototype_idx"] = np.arange(len(self.prototypes))
        self.embeddings = self.embeddings.astype(np.float32)
        self.node_to_indices = (
            self.prototypes.groupby("node_key")["prototype_idx"].apply(list).to_dict()
        )
        self.parent_to_proto_indices = {}
        self.parent_to_proto_child_keys = {}
        for parent_key, child_keys in self.parent_to_children.items():
            indices: list[int] = []
            prototype_child_keys: list[str] = []
            for child_key in child_keys:
                child_indices = self.node_to_indices.get(child_key, [])
                indices.extend(child_indices)
                prototype_child_keys.extend([child_key] * len(child_indices))
            if indices:
                self.parent_to_proto_indices[parent_key] = np.asarray(
                    indices, dtype=np.int32
                )
                self.parent_to_proto_child_keys[parent_key] = prototype_child_keys

    def score_nodes(
        self,
        query_embedding: np.ndarray,
        node_keys: list[str],
        *,
        aggregation: str,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for node_key in node_keys:
            indices = self.node_to_indices.get(node_key, [])
            if not indices:
                continue
            sims = self.embeddings[np.asarray(indices, dtype=np.int32)] @ query_embedding
            score = aggregate_scores(sims, aggregation=aggregation)
            metadata = self.node_metadata.get(node_key, {})
            rows.append(
                {
                    "node_key": node_key,
                    "node_name": str(metadata.get("category_name", node_key)),
                    "depth": int(metadata.get("depth", 0)),
                    "score": score,
                    "n_prototypes": len(indices),
                }
            )
        rows.sort(key=lambda item: float(item["score"]), reverse=True)
        return rows

    def score_children(
        self,
        query_embedding: np.ndarray,
        parent_key: str,
        *,
        aggregation: str,
    ) -> list[dict[str, object]]:
        proto_indices = self.parent_to_proto_indices.get(parent_key)
        child_keys = self.parent_to_proto_child_keys.get(parent_key)
        if proto_indices is None or child_keys is None:
            return []

        sims = self.embeddings[proto_indices] @ query_embedding
        grouped: dict[str, list[float]] = defaultdict(list)
        for child_key, sim in zip(child_keys, sims, strict=True):
            grouped[child_key].append(float(sim))

        rows: list[dict[str, object]] = []
        for child_key, values in grouped.items():
            metadata = self.node_metadata.get(child_key, {})
            rows.append(
                {
                    "node_key": child_key,
                    "node_name": str(metadata.get("category_name", child_key)),
                    "depth": int(metadata.get("depth", 0)),
                    "score": aggregate_scores(np.asarray(values), aggregation=aggregation),
                    "n_prototypes": len(values),
                }
            )
        rows.sort(key=lambda item: float(item["score"]), reverse=True)
        return rows


def build_parser() -> argparse.ArgumentParser:
    cfg = PipelineConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Predict Level 1 with the fine-tuned embedder, then run zero-shot "
            "L2-L7 beam search with enriched category prototypes."
        )
    )
    parser.add_argument("--input-path", default=cfg.input_path)
    parser.add_argument("--input-format", choices=["csv", "parquet"], default=cfg.input_format)
    parser.add_argument("--output-path", default=cfg.output_path)
    parser.add_argument("--enrichment-dir", default=cfg.enrichment_dir)
    parser.add_argument(
        "--taxonomy-path",
        default=cfg.taxonomy_path,
        help=(
            "Optional taxonomy.txt path. Used as the source of truth for the "
            "parent-child graph; enriched CSVs are still used for prototypes."
        ),
    )
    parser.add_argument("--l1-model-path", default=cfg.l1_model_path)
    parser.add_argument("--l1-run-dir", default=cfg.l1_run_dir)
    parser.add_argument("--deep-model-name", default=cfg.deep_model_name)
    parser.add_argument("--prototype-cache-dir", default=cfg.prototype_cache_dir)
    parser.add_argument("--hf-cache-dir", default=cfg.hf_cache_dir)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--text-col", default=cfg.text_col)
    parser.add_argument("--id-col", default=cfg.id_col)
    parser.add_argument("--original-l1-col", default=cfg.original_l1_col)
    parser.add_argument("--title-col", default=cfg.title_col)
    parser.add_argument("--description-col", default=cfg.description_col)
    parser.add_argument("--brand-col", default=cfg.brand_col)
    parser.add_argument("--price-col", default=cfg.price_col)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=cfg.random_seed)
    parser.add_argument("--l1-top-k", type=int, default=cfg.l1_top_k)
    parser.add_argument("--beam-width", type=int, default=cfg.beam_width)
    parser.add_argument("--children-top-k", type=int, default=cfg.children_top_k)
    parser.add_argument("--max-depth", type=int, default=cfg.max_depth)
    parser.add_argument("--output-top-paths", type=int, default=cfg.output_top_paths)
    parser.add_argument(
        "--score-aggregation",
        choices=["max", "mean"],
        default=cfg.score_aggregation,
    )
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--device-l1", default=None)
    parser.add_argument("--device-deep", default=None)
    parser.add_argument("--no-normalize-embeddings", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load inputs/prototypes and print counts without loading embedding models.",
    )
    return parser


def parse_args() -> tuple[PipelineConfig, bool]:
    parser = build_parser()
    args = parser.parse_args()
    configure_hf_cache(args.hf_cache_dir)
    device_l1, device_deep = resolve_devices(args.device_l1, args.device_deep)
    print_cuda_summary(device_l1=device_l1, device_deep=device_deep)
    warn_missing_paths(args)
    return (
        PipelineConfig(
            input_path=args.input_path,
            input_format=args.input_format,
            output_path=args.output_path,
            enrichment_dir=args.enrichment_dir,
            taxonomy_path=args.taxonomy_path,
            l1_model_path=args.l1_model_path,
            l1_run_dir=args.l1_run_dir,
            deep_model_name=args.deep_model_name,
            prototype_cache_dir=args.prototype_cache_dir,
            hf_cache_dir=args.hf_cache_dir,
            overwrite_cache=args.overwrite_cache,
            text_col=args.text_col,
            id_col=args.id_col,
            original_l1_col=args.original_l1_col,
            title_col=args.title_col,
            description_col=args.description_col,
            brand_col=args.brand_col,
            price_col=args.price_col,
            sample_size=args.sample_size,
            random_seed=args.random_seed,
            l1_top_k=args.l1_top_k,
            beam_width=args.beam_width,
            children_top_k=args.children_top_k,
            max_depth=args.max_depth,
            output_top_paths=args.output_top_paths,
            score_aggregation=args.score_aggregation,
            batch_size=args.batch_size,
            device_l1=device_l1,
            device_deep=device_deep,
            normalize_embeddings=not args.no_normalize_embeddings,
        ),
        bool(args.dry_run),
    )


def main() -> None:
    config, dry_run = parse_args()
    run_pipeline(config, dry_run=dry_run)


def configure_hf_cache(cache_dir: str) -> None:
    cache_path = Path(cache_dir)
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        cache_path = Path("/tmp/huggingface_cache")
        cache_path.mkdir(parents=True, exist_ok=True)
        print(f"WARNING: cannot write to {cache_dir}; using {cache_path} instead.")
    os.environ.setdefault("HF_HOME", str(cache_path))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_path / "transformers"))
    os.environ.setdefault(
        "SENTENCE_TRANSFORMERS_HOME",
        str(cache_path / "sentence_transformers"),
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def resolve_devices(
    explicit_l1: Optional[str],
    explicit_deep: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    if explicit_l1 or explicit_deep:
        return explicit_l1, explicit_deep
    try:
        import torch
    except ImportError:
        return None, None
    if not torch.cuda.is_available():
        return None, None
    if torch.cuda.device_count() >= 2:
        return "cuda:0", "cuda:1"
    return "cuda:0", "cuda:0"


def print_cuda_summary(*, device_l1: Optional[str], device_deep: Optional[str]) -> None:
    try:
        import torch
    except ImportError:
        print("torch not installed; SentenceTransformer will choose its default device.")
        return
    print("=" * 60)
    print("Kaggle inference device summary")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            print(f"  cuda:{idx}: {torch.cuda.get_device_name(idx)}")
    print(f"L1 model device: {device_l1 or '<auto>'}")
    print(f"L2-L7 model device: {device_deep or '<auto>'}")
    print("=" * 60)


def warn_missing_paths(args: argparse.Namespace) -> None:
    checked_paths = {
        "input_path": args.input_path,
        "enrichment_dir": args.enrichment_dir,
        "l1_model_path": args.l1_model_path,
    }
    if args.taxonomy_path:
        checked_paths["taxonomy_path"] = args.taxonomy_path
    for label, value in checked_paths.items():
        if not Path(value).exists():
            print(
                f"WARNING: {label} does not exist yet: {value}. "
                "Pass the correct Kaggle dataset path with the corresponding argument."
            )


def run_pipeline(config: PipelineConfig, *, dry_run: bool = False) -> pd.DataFrame | None:
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    products = load_products(config)
    taxonomy_metadata = load_taxonomy_metadata(
        Path(config.taxonomy_path),
        max_depth=config.max_depth,
    ) if config.taxonomy_path and Path(config.taxonomy_path).exists() else {}
    enriched_l1_metadata = load_category_metadata(
        Path(config.enrichment_dir), min_depth=1, max_depth=1
    )
    enriched_deep_metadata = load_category_metadata(
        Path(config.enrichment_dir), min_depth=1, max_depth=config.max_depth
    )
    l1_metadata = merge_metadata(
        filter_metadata_by_depth(taxonomy_metadata, min_depth=1, max_depth=1),
        enriched_l1_metadata,
    )
    deep_metadata = merge_metadata(taxonomy_metadata, enriched_deep_metadata)
    if not deep_metadata:
        deep_metadata = enriched_deep_metadata
    parent_source = taxonomy_metadata if taxonomy_metadata else deep_metadata
    parent_to_children = build_parent_to_children(parent_source)

    l1_prototypes = load_reference_texts(
        Path(config.enrichment_dir),
        min_depth=1,
        max_depth=1,
        allowed_node_keys=set(load_l1_candidate_keys(config, l1_metadata)),
    )
    deep_prototypes = load_reference_texts(
        Path(config.enrichment_dir),
        min_depth=2,
        max_depth=config.max_depth,
        allowed_node_keys=set(deep_metadata),
    )

    print(f"Products: {len(products):,}")
    print(f"L1 prototype rows: {len(l1_prototypes):,}")
    print(f"L2-L{config.max_depth} prototype rows: {len(deep_prototypes):,}")
    print(f"Taxonomy nodes loaded: {len(deep_metadata):,}")
    print(
        "Parent-child graph source: "
        f"{'taxonomy.txt' if taxonomy_metadata else 'enriched category CSVs'}"
    )
    if dry_run:
        print("Dry run complete.")
        return None

    l1_model = load_sentence_transformer(config.l1_model_path, device=config.device_l1)
    deep_model = load_sentence_transformer(config.deep_model_name, device=config.device_deep)

    l1_embeddings = encode_texts(
        l1_model,
        l1_prototypes["prototype_text"].tolist(),
        batch_size=config.batch_size,
        device=config.device_l1,
        normalize_embeddings=config.normalize_embeddings,
        show_progress_bar=True,
    )
    l1_index = PrototypeIndex(
        prototypes=l1_prototypes,
        embeddings=l1_embeddings,
        node_metadata=l1_metadata,
        parent_to_children={"__root__": sorted(l1_prototypes["node_key"].unique())},
    )

    deep_embeddings = load_or_build_deep_embeddings(
        config=config,
        model=deep_model,
        prototypes=deep_prototypes,
        metadata=deep_metadata,
        parent_to_children=parent_to_children,
    )
    deep_index = PrototypeIndex(
        prototypes=deep_prototypes,
        embeddings=deep_embeddings,
        node_metadata=deep_metadata,
        parent_to_children=parent_to_children,
    )

    texts = products[config.text_col].fillna("").astype(str).tolist()
    print("Encoding products with L1 model...")
    product_l1_embeddings = encode_texts(
        l1_model,
        texts,
        batch_size=config.batch_size,
        device=config.device_l1,
        normalize_embeddings=config.normalize_embeddings,
        show_progress_bar=True,
    )
    print("Encoding products with deep zero-shot model...")
    product_deep_embeddings = encode_texts(
        deep_model,
        texts,
        batch_size=config.batch_size,
        device=config.device_deep,
        normalize_embeddings=config.normalize_embeddings,
        show_progress_bar=True,
    )

    prediction_rows: list[dict[str, object]] = []
    l1_candidates = sorted(l1_prototypes["node_key"].unique())
    for row, l1_embedding, deep_embedding in zip(
        products.to_dict("records"),
        product_l1_embeddings,
        product_deep_embeddings,
        strict=True,
    ):
        l1_scores = l1_index.score_nodes(
            l1_embedding,
            l1_candidates,
            aggregation=config.score_aggregation,
        )[: config.l1_top_k]
        predicted_l1_key = str(l1_scores[0]["node_key"]) if l1_scores else ""
        predicted_paths = beam_search_deep_path(
            deep_index,
            deep_embedding,
            start_l1_key=predicted_l1_key,
            beam_width=config.beam_width,
            children_top_k=config.children_top_k,
            max_depth=config.max_depth,
            output_top_paths=config.output_top_paths,
            aggregation=config.score_aggregation,
        )
        best_path = predicted_paths[0] if predicted_paths else empty_path(predicted_l1_key)
        prediction_rows.append(
            build_output_row(
                source_row=row,
                config=config,
                l1_scores=l1_scores,
                best_path=best_path,
                top_paths=predicted_paths,
            )
        )

    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(output_path, index=False)
    save_json(asdict(config), output_path.with_suffix(".config.json"))
    print(f"Predictions saved to {output_path}")
    print(predictions.head(10).to_string(index=False))
    return predictions


def load_products(config: PipelineConfig) -> pd.DataFrame:
    input_path = Path(config.input_path)
    if config.input_format == "csv":
        df = pd.read_csv(input_path)
    else:
        df = pd.read_parquet(input_path)
    work = df.copy()
    if config.text_col not in work.columns:
        work[config.text_col] = build_text_series(work, config)
    work[config.text_col] = (
        work[config.text_col]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    work = work[work[config.text_col] != ""].copy()
    if config.sample_size is not None and len(work) > config.sample_size:
        work = work.sample(config.sample_size, random_state=config.random_seed)
    return work.reset_index(drop=True)


def build_text_series(df: pd.DataFrame, config: PipelineConfig) -> pd.Series:
    work = df.copy()
    for col in [config.title_col, config.description_col, config.brand_col]:
        if col not in work.columns:
            work[col] = ""
        work[col] = work[col].fillna("").astype(str).str.strip()
    if config.price_col not in work.columns:
        work[config.price_col] = ""
    text = "Product: " + work[config.title_col]
    has_description = work[config.description_col] != ""
    text = text.where(
        ~has_description, text + ". Description: " + work[config.description_col]
    )
    has_brand = work[config.brand_col] != ""
    text = text.where(~has_brand, text + ". Brand: " + work[config.brand_col])
    return text


def load_category_metadata(
    enrichment_dir: Path,
    *,
    min_depth: int,
    max_depth: int,
) -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {}
    for depth in range(min_depth, max_depth + 1):
        path = enrichment_dir / f"level_{depth}_categories.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for row in df.to_dict("records"):
            node_key = clean_text(row.get("node_key", ""))
            if not node_key:
                continue
            taxonomy_path = clean_text(row.get("taxonomy_path", node_key)) or node_key
            path_names = [part.strip() for part in taxonomy_path.split(" > ") if part.strip()]
            parent_key = clean_text(row.get("parent_key", ""))
            if not parent_key and len(path_names) > 1:
                parent_key = " > ".join(path_names[:-1])
            metadata[node_key] = {
                "node_key": node_key,
                "category_name": clean_text(row.get("category_name", path_names[-1])),
                "depth": int(row.get("depth", depth)),
                "taxonomy_path": taxonomy_path,
                "parent_key": parent_key,
                "parent_name": clean_text(row.get("parent_name", "")),
                "path_names": path_names,
            }
    return metadata


def load_taxonomy_metadata(
    taxonomy_path: Path,
    *,
    max_depth: int,
) -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {}
    for line in taxonomy_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        taxonomy_text = parts[-1].strip()
        if not taxonomy_text:
            continue
        path_parts = [part.strip() for part in taxonomy_text.split(" > ") if part.strip()]
        for depth in range(1, min(len(path_parts), max_depth) + 1):
            path_names = path_parts[:depth]
            node_key = " > ".join(path_names)
            parent_key = " > ".join(path_names[:-1]) if depth > 1 else ""
            metadata.setdefault(
                node_key,
                {
                    "node_key": node_key,
                    "category_name": path_names[-1],
                    "depth": depth,
                    "taxonomy_path": node_key,
                    "parent_key": parent_key,
                    "parent_name": path_names[-2] if depth > 1 else "",
                    "path_names": path_names,
                },
            )
    return metadata


def filter_metadata_by_depth(
    metadata: dict[str, dict[str, object]],
    *,
    min_depth: int,
    max_depth: int,
) -> dict[str, dict[str, object]]:
    return {
        key: value
        for key, value in metadata.items()
        if min_depth <= int(value.get("depth", 0)) <= max_depth
    }


def merge_metadata(
    base: dict[str, dict[str, object]],
    override: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    merged = {key: dict(value) for key, value in base.items()}
    for key, value in override.items():
        if key in merged:
            merged[key].update(value)
        else:
            merged[key] = dict(value)
    return merged


def build_parent_to_children(
    metadata: dict[str, dict[str, object]]
) -> dict[str, list[str]]:
    parent_to_children: dict[str, list[str]] = defaultdict(list)
    for node_key, row in metadata.items():
        depth = int(row["depth"])
        if depth == 1:
            parent_to_children["__root__"].append(node_key)
            continue
        parent_key = str(row.get("parent_key", "")).strip()
        if parent_key:
            parent_to_children[parent_key].append(node_key)
    return {key: sorted(set(values)) for key, values in parent_to_children.items()}


def load_reference_texts(
    enrichment_dir: Path,
    *,
    min_depth: int,
    max_depth: int,
    allowed_node_keys: set[str],
) -> pd.DataFrame:
    frames = []
    for depth in range(min_depth, max_depth + 1):
        path = enrichment_dir / f"level_{depth}_reference_texts.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(
            f"No reference text CSV found in {enrichment_dir} for levels {min_depth}-{max_depth}."
        )

    df = pd.concat(frames, ignore_index=True)
    required = {"node_key", "depth", "category_name", "prototype_type", "prototype_text"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Reference text files are missing columns: {sorted(missing)}")
    df["node_key"] = df["node_key"].fillna("").astype(str).str.strip()
    df["prototype_text"] = (
        df["prototype_text"]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df[(df["node_key"].isin(allowed_node_keys)) & (df["prototype_text"] != "")]
    return df.drop_duplicates(
        subset=["node_key", "prototype_type", "prototype_text"]
    ).reset_index(drop=True)


def load_l1_candidate_keys(
    config: PipelineConfig,
    l1_metadata: dict[str, dict[str, object]],
) -> list[str]:
    predictions_path = Path(config.l1_run_dir) / "retrieval_predictions.csv"
    if predictions_path.exists():
        df = pd.read_csv(predictions_path, usecols=["label_key"])
        labels = sorted({clean_text(value) for value in df["label_key"].tolist()})
        labels = [label for label in labels if label in l1_metadata]
        if labels:
            return labels
    return sorted(l1_metadata)


def load_sentence_transformer(model_name: str, *, device: Optional[str]) -> SentenceTransformer:
    return SentenceTransformer(model_name, device=device, trust_remote_code=True)


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    *,
    batch_size: int,
    device: Optional[str],
    normalize_embeddings: bool,
    show_progress_bar: bool,
) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=normalize_embeddings,
        convert_to_numpy=True,
        device=device,
    ).astype(np.float32)


def load_or_build_deep_embeddings(
    *,
    config: PipelineConfig,
    model: SentenceTransformer,
    prototypes: pd.DataFrame,
    metadata: dict[str, dict[str, object]],
    parent_to_children: dict[str, list[str]],
) -> np.ndarray:
    if not config.prototype_cache_dir:
        print("Encoding deep category prototypes...")
        return encode_texts(
            model,
            prototypes["prototype_text"].tolist(),
            batch_size=config.batch_size,
            device=config.device_deep,
            normalize_embeddings=config.normalize_embeddings,
            show_progress_bar=True,
        )

    cache_dir = Path(config.prototype_cache_dir) / cache_key(config, prototypes)
    embeddings_path = cache_dir / "prototype_embeddings.npy"
    prototypes_path = cache_dir / "prototypes.parquet"
    metadata_path = cache_dir / "metadata.json"
    if (
        not config.overwrite_cache
        and embeddings_path.exists()
        and prototypes_path.exists()
        and metadata_path.exists()
    ):
        cached_prototypes = pd.read_parquet(prototypes_path)
        if same_prototype_frame(cached_prototypes, prototypes):
            print(f"Loading cached deep prototype embeddings from {cache_dir}")
            return np.load(embeddings_path).astype(np.float32)

    print("Encoding deep category prototypes...")
    embeddings = encode_texts(
        model,
        prototypes["prototype_text"].tolist(),
        batch_size=config.batch_size,
        device=config.device_deep,
        normalize_embeddings=config.normalize_embeddings,
        show_progress_bar=True,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    prototypes.to_parquet(prototypes_path, index=False)
    np.save(embeddings_path, embeddings)
    save_json(
        {
            "deep_model_name": config.deep_model_name,
            "max_depth": config.max_depth,
            "score_aggregation": config.score_aggregation,
            "n_prototypes": int(len(prototypes)),
            "n_nodes": int(len(metadata)),
            "n_parent_entries": int(len(parent_to_children)),
        },
        metadata_path,
    )
    return embeddings


def cache_key(config: PipelineConfig, prototypes: pd.DataFrame) -> str:
    digest = hashlib.sha1()
    digest.update(config.deep_model_name.encode("utf-8"))
    digest.update(str(config.max_depth).encode("utf-8"))
    digest.update(str(len(prototypes)).encode("utf-8"))
    digest.update(
        pd.util.hash_pandas_object(
            prototypes[["node_key", "prototype_type", "prototype_text"]],
            index=False,
        )
        .values
        .tobytes()
    )
    return f"{slugify(config.deep_model_name)}__{digest.hexdigest()[:12]}"


def same_prototype_frame(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    cols = ["node_key", "prototype_type", "prototype_text"]
    if len(left) != len(right):
        return False
    return left[cols].reset_index(drop=True).equals(right[cols].reset_index(drop=True))


def beam_search_deep_path(
    index: PrototypeIndex,
    query_embedding: np.ndarray,
    *,
    start_l1_key: str,
    beam_width: int,
    children_top_k: int,
    max_depth: int,
    output_top_paths: int,
    aggregation: str,
) -> list[dict[str, object]]:
    if not start_l1_key:
        return []
    start_metadata = index.node_metadata.get(start_l1_key, {})
    frontier = [
        {
            "path_keys": [start_l1_key],
            "path_names": [str(start_metadata.get("category_name", start_l1_key))],
            "score_trace": [],
            "prototype_trace": [],
            "cumulative_score": 0.0,
            "resolved_depth": 1,
        }
    ]
    completed: list[dict[str, object]] = []

    while frontier:
        expanded: list[dict[str, object]] = []
        for state in frontier:
            current_key = state["path_keys"][-1]
            current_depth = int(state["resolved_depth"])
            if current_depth >= max_depth:
                completed.append(state)
                continue
            child_scores = index.score_children(
                query_embedding,
                current_key,
                aggregation=aggregation,
            )[:children_top_k]
            if not child_scores:
                completed.append(state)
                continue
            for child in child_scores:
                score_trace = [*state["score_trace"], float(child["score"])]
                expanded.append(
                    {
                        "path_keys": [*state["path_keys"], str(child["node_key"])],
                        "path_names": [*state["path_names"], str(child["node_name"])],
                        "score_trace": score_trace,
                        "prototype_trace": [
                            *state["prototype_trace"],
                            {
                                "depth": int(child["depth"]),
                                "node_key": str(child["node_key"]),
                                "node_name": str(child["node_name"]),
                                "score": float(child["score"]),
                                "n_prototypes": int(child["n_prototypes"]),
                            },
                        ],
                        "cumulative_score": float(np.mean(score_trace)),
                        "resolved_depth": int(child["depth"]),
                    }
                )
        if not expanded:
            break
        expanded.sort(key=lambda row: float(row["cumulative_score"]), reverse=True)
        frontier = expanded[:beam_width]

    candidates = completed + frontier
    unique_candidates = []
    seen = set()
    for row in sorted(candidates, key=lambda item: float(item["cumulative_score"]), reverse=True):
        key = tuple(row["path_keys"])
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(row)
    return unique_candidates[:output_top_paths]


def empty_path(l1_key: str) -> dict[str, object]:
    return {
        "path_keys": [l1_key] if l1_key else [],
        "path_names": [l1_key] if l1_key else [],
        "score_trace": [],
        "prototype_trace": [],
        "cumulative_score": 0.0,
        "resolved_depth": 1 if l1_key else 0,
    }


def build_output_row(
    *,
    source_row: dict[str, object],
    config: PipelineConfig,
    l1_scores: list[dict[str, object]],
    best_path: dict[str, object],
    top_paths: list[dict[str, object]],
) -> dict[str, object]:
    path_names = list(best_path.get("path_names", []))
    path_keys = list(best_path.get("path_keys", []))
    row: dict[str, object] = {}
    if config.id_col in source_row:
        row[config.id_col] = source_row[config.id_col]
    if config.original_l1_col in source_row:
        row[f"input_{config.original_l1_col}"] = source_row[config.original_l1_col]
    row[config.text_col] = source_row.get(config.text_col, "")
    row["predicted_taxonomy_path"] = " > ".join(path_names)
    row["predicted_taxonomy_key_path"] = " || ".join(path_keys)
    row["predicted_path_score"] = float(best_path.get("cumulative_score", 0.0))
    row["resolved_depth"] = int(best_path.get("resolved_depth", 0))
    row["l1_candidates_json"] = json.dumps(l1_scores, ensure_ascii=False)
    row["beam_paths_json"] = json.dumps(top_paths, ensure_ascii=False)
    for depth in range(1, config.max_depth + 1):
        row[f"predicted_level_{depth}_name"] = (
            path_names[depth - 1] if len(path_names) >= depth else ""
        )
        row[f"predicted_level_{depth}_key"] = (
            path_keys[depth - 1] if len(path_keys) >= depth else ""
        )
        score_trace = list(best_path.get("score_trace", []))
        row[f"level_{depth}_local_score"] = (
            float(score_trace[depth - 2]) if depth >= 2 and len(score_trace) >= depth - 1 else ""
        )
    return row


def aggregate_scores(values: np.ndarray, *, aggregation: str) -> float:
    if len(values) == 0:
        return float("-inf")
    if aggregation == "max":
        return float(np.max(values))
    if aggregation == "mean":
        return float(np.mean(values))
    raise ValueError("aggregation must be 'max' or 'mean'.")


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return " ".join(str(value).split()).strip()


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "model"


def save_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
