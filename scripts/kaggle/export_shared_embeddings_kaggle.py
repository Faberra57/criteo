"""Export shared embeddings for all unsupervised L2-L7 pipelines on Kaggle.

This script does no prediction. It only encodes:
- product catalog texts;
- enriched category prototype texts for L1-L7;
- global taxonomy path texts for L2-L7.

All downstream scripts consume the exported files and avoid re-encoding.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_INPUT_PATH = "/kaggle/input/criteo-finetuning/preprocessed_lv2.parquet"
DEFAULT_ENRICHMENT_DIR = (
    "/kaggle/input/criteo-finetuning/category_enrichment/category_enrichment"
)
DEFAULT_TAXONOMY_PATH = "/kaggle/input/criteo-finetuning/taxonomy.txt"
DEFAULT_OUTPUT_DIR = "/kaggle/working/shared_l1_l7_embeddings"
DEFAULT_HF_CACHE = "/kaggle/working/huggingface_cache"
model_name: str = "infgrad/Jasper-Token-Compression-600M"


def main() -> None:
    args = parse_args()
    configure_cache(args.hf_cache_dir)
    reset_cuda_peak_memory()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()

    inputs_started = time.perf_counter()
    products = load_products(args)
    nodes = load_category_nodes_from_taxonomy(
        Path(args.taxonomy_path), max_depth=args.max_depth
    )
    prototypes = load_category_prototypes(
        Path(args.enrichment_dir),
        nodes=set(nodes["node_key"]),
        max_depth=args.max_depth,
    )
    global_paths = build_global_paths(nodes, args.global_min_depth, args.max_depth)
    load_inputs_seconds = time.perf_counter() - inputs_started

    print(f"Products: {len(products):,}")
    print(f"Category nodes: {len(nodes):,}")
    print(f"Category prototype rows: {len(prototypes):,}")
    print(f"Global path rows: {len(global_paths):,}")

    write_metadata_started = time.perf_counter()
    products.to_parquet(output_dir / "products.parquet", index=False)
    nodes.to_parquet(output_dir / "category_nodes.parquet", index=False)
    prototypes.to_parquet(output_dir / "category_prototypes.parquet", index=False)
    global_paths.to_parquet(output_dir / "global_paths.parquet", index=False)
    write_metadata_seconds = time.perf_counter() - write_metadata_started

    if args.dry_run:
        print("Dry-run complete. No model loaded and no embeddings exported.")
        return

    device = args.device or resolve_device()
    print(f"Loading embedding model on {device or '<auto>'}: {model_name}")
    model_load_started = time.perf_counter()
    model = SentenceTransformer(
        model_name,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",  # We support flash_attention_2; sdpa; eager
            "trust_remote_code": True,
        },
        trust_remote_code=True,
        tokenizer_kwargs={"padding_side": "left"},
        device=device,
    )
    model_load_seconds = time.perf_counter() - model_load_started

    product_texts = products[args.text_col].tolist()
    prototype_texts = prototypes["prototype_text"].tolist()
    global_path_texts = global_paths["candidate_text"].tolist()
    token_metrics: dict[str, Any] = {}
    token_count_seconds = 0.0
    if not args.skip_token_count:
        token_started = time.perf_counter()
        token_metrics = {
            "products": count_tokens(model, product_texts, batch_size=args.batch_size),
            "category_prototypes": count_tokens(
                model, prototype_texts, batch_size=args.batch_size
            ),
            "global_paths": count_tokens(
                model, global_path_texts, batch_size=args.batch_size
            ),
        }
        token_count_seconds = time.perf_counter() - token_started

    encode_product_started = time.perf_counter()
    product_embeddings = encode_texts(
        model,
        product_texts,
        batch_size=args.batch_size,
        device=device,
    )
    encode_products_seconds = time.perf_counter() - encode_product_started

    encode_prototype_started = time.perf_counter()
    prototype_embeddings = encode_texts(
        model,
        prototype_texts,
        batch_size=args.batch_size,
        device=device,
    )
    encode_category_prototypes_seconds = time.perf_counter() - encode_prototype_started

    encode_global_started = time.perf_counter()
    global_path_embeddings = encode_texts(
        model,
        global_path_texts,
        batch_size=args.batch_size,
        device=device,
    )
    encode_global_paths_seconds = time.perf_counter() - encode_global_started

    save_embeddings_started = time.perf_counter()
    np.save(output_dir / "product_embeddings.npy", product_embeddings)
    np.save(output_dir / "category_prototype_embeddings.npy", prototype_embeddings)
    np.save(output_dir / "global_path_embeddings.npy", global_path_embeddings)
    save_embeddings_seconds = time.perf_counter() - save_embeddings_started

    embedding_dim = int(product_embeddings.shape[1]) if len(product_embeddings) else 0
    cost_metrics = build_embedding_cost_metrics(
        product_embeddings=product_embeddings,
        prototype_embeddings=prototype_embeddings,
        global_path_embeddings=global_path_embeddings,
        token_metrics=token_metrics,
        embedding_dim=embedding_dim,
        batch_size=args.batch_size,
        elapsed_seconds=time.perf_counter() - started_at,
        phase_seconds={
            "load_inputs_seconds": load_inputs_seconds,
            "write_metadata_seconds": write_metadata_seconds,
            "model_load_seconds": model_load_seconds,
            "token_count_seconds": token_count_seconds,
            "encode_products_seconds": encode_products_seconds,
            "encode_category_prototypes_seconds": encode_category_prototypes_seconds,
            "encode_global_paths_seconds": encode_global_paths_seconds,
            "save_embeddings_seconds": save_embeddings_seconds,
        },
    )

    manifest = {
        "created_at_unix": time.time(),
        "elapsed_seconds": time.perf_counter() - started_at,
        "model_path": model_name,
        "input_path": args.input_path,
        "taxonomy_path": args.taxonomy_path,
        "enrichment_dir": args.enrichment_dir,
        "n_products": int(len(products)),
        "n_category_nodes": int(len(nodes)),
        "n_category_prototypes": int(len(prototypes)),
        "n_global_paths": int(len(global_paths)),
        "embedding_dim": embedding_dim,
        "normalize_embeddings": True,
        "metrics": cost_metrics,
        "files": {
            "products": "products.parquet",
            "product_embeddings": "product_embeddings.npy",
            "category_nodes": "category_nodes.parquet",
            "category_prototypes": "category_prototypes.parquet",
            "category_prototype_embeddings": "category_prototype_embeddings.npy",
            "global_paths": "global_paths.parquet",
            "global_path_embeddings": "global_path_embeddings.npy",
        },
    }
    write_json(manifest, output_dir / "manifest.json")
    write_json(
        {
            "args": vars(args),
            "model_path": model_name,
            "metrics": cost_metrics,
        },
        output_dir / "embedding_generation.metrics.json",
    )
    print(f"Shared embeddings exported to {output_dir}")
    print(json.dumps(cost_metrics, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--input-format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--enrichment-dir", default=DEFAULT_ENRICHMENT_DIR)
    parser.add_argument("--taxonomy-path", default=DEFAULT_TAXONOMY_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hf-cache-dir", default=DEFAULT_HF_CACHE)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--id-col", default="hashed_external_id")
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--global-min-depth", type=int, default=2)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--skip-token-count",
        action="store_true",
        help=(
            "Skip explicit token counting. Faster, but token-level cost metrics "
            "will be omitted from embedding_generation.metrics.json."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def configure_cache(cache_dir: str) -> None:
    cache = Path(cache_dir)
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        cache = Path("/tmp/huggingface_cache")
        cache.mkdir(parents=True, exist_ok=True)
        print(f"WARNING: cannot write to {cache_dir}; using {cache}.")
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache / "transformers"))
    os.environ.setdefault(
        "SENTENCE_TRANSFORMERS_HOME", str(cache / "sentence_transformers")
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def resolve_device() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    return "cuda" if torch.cuda.is_available() else None


def load_products(args: argparse.Namespace) -> pd.DataFrame:
    path = Path(args.input_path)
    df = pd.read_parquet(path) if args.input_format == "parquet" else pd.read_csv(path)
    if args.text_col not in df.columns:
        raise ValueError(f"Missing text column {args.text_col!r} in {path}.")
    keep = [args.id_col, args.text_col]
    products = df[keep].copy()
    products[args.text_col] = (
        products[args.text_col].fillna("").astype(str).map(clean_text)
    )
    products = products[products[args.text_col] != ""].reset_index(drop=True)
    if args.sample_size is not None and len(products) > args.sample_size:
        products = products.sample(
            args.sample_size, random_state=args.random_seed
        ).reset_index(drop=True)
    return products


def load_category_nodes_from_taxonomy(
    taxonomy_path: Path, *, max_depth: int
) -> pd.DataFrame:
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"taxonomy.txt not found: {taxonomy_path}")
    rows: list[dict[str, object]] = []
    with taxonomy_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if "\t" in line:
                _, taxonomy_text = line.split("\t", 1)
            else:
                taxonomy_text = line
            taxonomy_text = clean_taxonomy_path(taxonomy_text)
            if not taxonomy_text:
                continue
            parts = [
                part.strip() for part in taxonomy_text.split(" > ") if part.strip()
            ]
            depth = len(parts)
            if depth == 0 or depth > max_depth:
                continue
            parent_path = " > ".join(parts[:-1]) if depth > 1 else ""
            rows.append(
                {
                    "node_key": taxonomy_text,
                    "depth": depth,
                    "category_name": parts[-1],
                    "taxonomy_path": taxonomy_text,
                    "parent_key": parent_path,
                    "parent_name": parts[-2] if depth > 1 else "",
                    "level_1_name": parts[0],
                }
            )
    if not rows:
        raise ValueError(f"No taxonomy nodes parsed from {taxonomy_path}")
    df = pd.DataFrame(rows)
    cols = [
        "node_key",
        "depth",
        "category_name",
        "taxonomy_path",
        "parent_key",
        "parent_name",
        "level_1_name",
    ]
    return (
        df[cols]
        .drop_duplicates("node_key")
        .sort_values(["depth", "taxonomy_path"])
        .reset_index(drop=True)
    )


def load_category_prototypes(
    enrichment_dir: Path, *, nodes: set[str], max_depth: int
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for depth in range(1, max_depth + 1):
        path = enrichment_dir / f"level_{depth}_reference_texts.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(
            f"No level_*_reference_texts.csv files found in {enrichment_dir}"
        )
    df = pd.concat(frames, ignore_index=True)
    required = {
        "node_key",
        "depth",
        "category_name",
        "prototype_type",
        "prototype_text",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Reference text CSVs are missing columns: {sorted(missing)}")
    for col in ["node_key", "category_name", "prototype_type", "prototype_text"]:
        df[col] = df[col].fillna("").astype(str).map(clean_text)
    df["depth"] = pd.to_numeric(df["depth"], errors="coerce").fillna(0).astype(int)
    df = df[(df["node_key"].isin(nodes)) & (df["prototype_text"] != "")]
    return df.drop_duplicates(
        ["node_key", "prototype_type", "prototype_text"]
    ).reset_index(drop=True)


def build_global_paths(
    nodes: pd.DataFrame, min_depth: int, max_depth: int
) -> pd.DataFrame:
    work = nodes[(nodes["depth"] >= min_depth) & (nodes["depth"] <= max_depth)].copy()
    work["candidate_text"] = work.apply(
        lambda row: (
            f"Taxonomy path: {row['taxonomy_path']}. "
            f"Category name: {row['category_name']}. "
            f"Level: {int(row['depth'])}."
        ),
        axis=1,
    )
    return work[
        [
            "node_key",
            "depth",
            "category_name",
            "taxonomy_path",
            "parent_key",
            "level_1_name",
            "candidate_text",
        ]
    ].reset_index(drop=True)


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    *,
    batch_size: int,
    device: str | None,
) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=device,
    ).astype(np.float32)


def count_tokens(
    model: SentenceTransformer,
    texts: list[str],
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Count tokens after SentenceTransformer tokenization.

    This is a measurement pass used only for cost reporting. It may add runtime,
    so it can be disabled with --skip-token-count.
    """
    if not texts:
        return {
            "n_texts": 0,
            "total_tokens": 0,
            "mean_tokens_per_text": 0.0,
            "median_tokens_per_text": 0.0,
            "max_tokens_per_text": 0,
        }
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        features = model.tokenize(batch)
        attention_mask = features.get("attention_mask")
        if attention_mask is None:
            input_ids = features.get("input_ids")
            if input_ids is None:
                continue
            values = np.asarray(input_ids)
            lengths.extend([int(values.shape[1])] * int(values.shape[0]))
            continue
        mask_values = (
            attention_mask.detach().cpu().numpy()
            if hasattr(attention_mask, "detach")
            else np.asarray(attention_mask)
        )
        lengths.extend(mask_values.sum(axis=1).astype(int).tolist())
    if not lengths:
        return {
            "n_texts": int(len(texts)),
            "total_tokens": 0,
            "mean_tokens_per_text": float("nan"),
            "median_tokens_per_text": float("nan"),
            "max_tokens_per_text": 0,
        }
    values = np.asarray(lengths, dtype=np.int64)
    return {
        "n_texts": int(len(texts)),
        "total_tokens": int(values.sum()),
        "mean_tokens_per_text": float(values.mean()),
        "median_tokens_per_text": float(np.median(values)),
        "max_tokens_per_text": int(values.max()),
    }


def build_embedding_cost_metrics(
    *,
    product_embeddings: np.ndarray,
    prototype_embeddings: np.ndarray,
    global_path_embeddings: np.ndarray,
    token_metrics: dict[str, Any],
    embedding_dim: int,
    batch_size: int,
    elapsed_seconds: float,
    phase_seconds: dict[str, float],
) -> dict[str, Any]:
    n_products = int(len(product_embeddings))
    n_prototypes = int(len(prototype_embeddings))
    n_global_paths = int(len(global_path_embeddings))
    n_total_texts = n_products + n_prototypes + n_global_paths
    encode_seconds = (
        float(phase_seconds.get("encode_products_seconds", 0.0))
        + float(phase_seconds.get("encode_category_prototypes_seconds", 0.0))
        + float(phase_seconds.get("encode_global_paths_seconds", 0.0))
    )
    total_tokens = (
        int(sum(int(part.get("total_tokens", 0)) for part in token_metrics.values()))
        if token_metrics
        else 0
    )
    token_dim_units = int(total_tokens * embedding_dim)
    metrics: dict[str, Any] = {
        "mode": "shared_embedding_generation",
        "embedding_generation_cost_unit": "encoded texts, encoded tokens, and M token-dim units",
        "elapsed_seconds": float(elapsed_seconds),
        "seconds_per_encoded_text_total": float(
            elapsed_seconds / max(1, n_total_texts)
        ),
        "encode_seconds": encode_seconds,
        "seconds_per_encoded_text_encode_only": float(
            encode_seconds / max(1, n_total_texts)
        ),
        "throughput_texts_per_second_total": float(n_total_texts / elapsed_seconds)
        if elapsed_seconds > 0
        else float("nan"),
        "throughput_texts_per_second_encode_only": float(n_total_texts / encode_seconds)
        if encode_seconds > 0
        else float("nan"),
        **{key: float(value) for key, value in phase_seconds.items()},
        "batch_size": int(batch_size),
        "embedding_dim": int(embedding_dim),
        "n_product_texts_encoded": n_products,
        "n_category_prototype_texts_encoded": n_prototypes,
        "n_global_path_texts_encoded": n_global_paths,
        "n_total_texts_encoded": n_total_texts,
        "n_forward_batches_estimated": int(np.ceil(n_total_texts / max(1, batch_size))),
        "product_embeddings_mb": float(product_embeddings.nbytes / (1024**2)),
        "category_prototype_embeddings_mb": float(
            prototype_embeddings.nbytes / (1024**2)
        ),
        "global_path_embeddings_mb": float(global_path_embeddings.nbytes / (1024**2)),
        "total_embedding_output_mb": float(
            (
                product_embeddings.nbytes
                + prototype_embeddings.nbytes
                + global_path_embeddings.nbytes
            )
            / (1024**2)
        ),
        **cuda_memory_metrics(),
    }
    if token_metrics:
        metrics.update(
            {
                "token_count_available": True,
                "total_tokens_encoded": total_tokens,
                "tokens_per_encoded_text": float(total_tokens / max(1, n_total_texts)),
                "product_tokens_encoded": int(
                    token_metrics["products"]["total_tokens"]
                ),
                "category_prototype_tokens_encoded": int(
                    token_metrics["category_prototypes"]["total_tokens"]
                ),
                "global_path_tokens_encoded": int(
                    token_metrics["global_paths"]["total_tokens"]
                ),
                "mean_product_tokens": float(
                    token_metrics["products"]["mean_tokens_per_text"]
                ),
                "mean_category_prototype_tokens": float(
                    token_metrics["category_prototypes"]["mean_tokens_per_text"]
                ),
                "mean_global_path_tokens": float(
                    token_metrics["global_paths"]["mean_tokens_per_text"]
                ),
                "max_product_tokens": int(
                    token_metrics["products"]["max_tokens_per_text"]
                ),
                "max_category_prototype_tokens": int(
                    token_metrics["category_prototypes"]["max_tokens_per_text"]
                ),
                "max_global_path_tokens": int(
                    token_metrics["global_paths"]["max_tokens_per_text"]
                ),
                "approx_token_dim_units": token_dim_units,
                "approx_million_token_dim_units": float(token_dim_units / 1_000_000),
                "approx_million_token_dim_units_per_encoded_text": float(
                    token_dim_units / max(1, n_total_texts) / 1_000_000
                ),
            }
        )
    else:
        metrics.update(
            {
                "token_count_available": False,
                "total_tokens_encoded": None,
                "tokens_per_encoded_text": None,
                "approx_token_dim_units": None,
                "approx_million_token_dim_units": None,
                "approx_million_token_dim_units_per_encoded_text": None,
            }
        )
    return metrics


def reset_cuda_peak_memory() -> None:
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    for idx in range(torch.cuda.device_count()):
        try:
            with torch.cuda.device(idx):
                torch.cuda.reset_peak_memory_stats()
        except Exception as exc:
            print(f"WARNING: cannot reset CUDA peak memory for cuda:{idx}: {exc}")


def cuda_memory_metrics() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {
            "cuda_available": False,
            "cuda_device_count": 0,
            "peak_gpu_memory_allocated_mb": np.nan,
            "peak_gpu_memory_reserved_mb": np.nan,
        }
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "cuda_device_count": 0,
            "peak_gpu_memory_allocated_mb": np.nan,
            "peak_gpu_memory_reserved_mb": np.nan,
        }
    allocated = []
    reserved = []
    by_device = {}
    for idx in range(torch.cuda.device_count()):
        try:
            with torch.cuda.device(idx):
                alloc_mb = torch.cuda.max_memory_allocated() / (1024**2)
                reserv_mb = torch.cuda.max_memory_reserved() / (1024**2)
        except Exception as exc:
            by_device[f"cuda:{idx}"] = {"error": str(exc)}
            continue
        allocated.append(alloc_mb)
        reserved.append(reserv_mb)
        by_device[f"cuda:{idx}"] = {
            "peak_allocated_mb": float(alloc_mb),
            "peak_reserved_mb": float(reserv_mb),
        }
    return {
        "cuda_available": True,
        "cuda_device_count": int(torch.cuda.device_count()),
        "peak_gpu_memory_allocated_mb": float(max(allocated)) if allocated else np.nan,
        "peak_gpu_memory_reserved_mb": float(max(reserved)) if reserved else np.nan,
        "gpu_memory_by_device": by_device,
    }


def cuda_memory_metrics() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {
            "cuda_available": False,
            "peak_gpu_memory_allocated_mb": float("nan"),
            "peak_gpu_memory_reserved_mb": float("nan"),
        }
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "peak_gpu_memory_allocated_mb": float("nan"),
            "peak_gpu_memory_reserved_mb": float("nan"),
        }
    allocated = []
    reserved = []
    by_device = {}
    for idx in range(torch.cuda.device_count()):
        alloc_mb = torch.cuda.max_memory_allocated(idx) / (1024**2)
        reserv_mb = torch.cuda.max_memory_reserved(idx) / (1024**2)
        allocated.append(alloc_mb)
        reserved.append(reserv_mb)
        by_device[f"cuda:{idx}"] = {
            "peak_allocated_mb": float(alloc_mb),
            "peak_reserved_mb": float(reserv_mb),
        }
    return {
        "cuda_available": True,
        "peak_gpu_memory_allocated_mb": float(max(allocated) if allocated else 0.0),
        "peak_gpu_memory_reserved_mb": float(max(reserved) if reserved else 0.0),
        "gpu_memory_by_device": by_device,
    }


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def clean_taxonomy_path(value: object) -> str:
    text = clean_text(value)
    text = re.sub(r"\s*>\s*", " > ", text)
    return text.strip()


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
