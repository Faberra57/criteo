from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .embedding_model_registry import resolve_model_name
from .hierarchical_beam_search import BeamSearchConfig, predict_product_path
from .synthetic_generation import (
    load_l2_descendant_map,
    load_level2_keywords,
)
from .synthetic_validation import build_modeling_text


@dataclass(slots=True)
class PrototypeBuildConfig:
    taxonomy_path: str = "taxonomy.txt"
    category_keywords_path: str = "categories_level_2.csv"
    synthetic_dataset_path: Optional[str] = "data/synthetic_l2_llm_targeted_50.csv"
    model_name: str = "level2"
    output_dir: str = "artifacts/hierarchical_prototype_index"
    device: Optional[str] = None
    text_col: str = "text"
    level1_col: str = "level_1_name"
    level2_col: str = "level_2_name"
    max_descendants_per_node: int = 8
    max_synthetic_examples_per_node: int = 6
    batch_size: int = 64
    normalize_embeddings: bool = True
    random_seed: int = 42

class PrototypeIndex:
    def __init__(
        self,
        *,
        model_name: str,
        prototypes_df: pd.DataFrame,
        embeddings: np.ndarray,
        parent_to_children: dict[str, list[str]],
        node_metadata: dict[str, dict[str, object]],
    ) -> None:
        self.model_name = model_name
        self.prototypes_df = prototypes_df.reset_index(drop=True).copy()
        self.embeddings = embeddings.astype(np.float32)
        self.parent_to_children = parent_to_children
        self.node_metadata = node_metadata
        self._prepare_lookup()

    def _prepare_lookup(self) -> None:
        self.prototypes_df["prototype_idx"] = np.arange(len(self.prototypes_df))
        self.node_to_indices = (
            self.prototypes_df.groupby("node_key")["prototype_idx"].apply(list).to_dict()
        )
        self.parent_to_proto_indices: dict[str, np.ndarray] = {}
        self.parent_to_proto_child_keys: dict[str, list[str]] = {}

        for parent_key, children in self.parent_to_children.items():
            indices: list[int] = []
            child_keys: list[str] = []
            for child_key in children:
                child_indices = self.node_to_indices.get(child_key, [])
                indices.extend(child_indices)
                child_keys.extend([child_key] * len(child_indices))
            if indices:
                self.parent_to_proto_indices[parent_key] = np.asarray(indices, dtype=np.int32)
                self.parent_to_proto_child_keys[parent_key] = child_keys

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.prototypes_df.to_parquet(output_dir / "prototypes.parquet", index=False)
        np.save(output_dir / "prototype_embeddings.npy", self.embeddings)
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "model_name": self.model_name,
                    "parent_to_children": self.parent_to_children,
                    "node_metadata": self.node_metadata,
                },
                handle,
                indent=2,
            )

    @classmethod
    def load(cls, input_dir: str | Path) -> "PrototypeIndex":
        input_dir = Path(input_dir)
        prototypes_df = pd.read_parquet(input_dir / "prototypes.parquet")
        embeddings = np.load(input_dir / "prototype_embeddings.npy")
        with (input_dir / "metadata.json").open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return cls(
            model_name=metadata["model_name"],
            prototypes_df=prototypes_df,
            embeddings=embeddings,
            parent_to_children=metadata["parent_to_children"],
            node_metadata=metadata["node_metadata"],
        )

    def encode_texts(
        self,
        texts: list[str],
        *,
        batch_size: int = 64,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        model = SentenceTransformer(self.model_name, device=device)
        return model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
            device=device,
        ).astype(np.float32)

    def score_children(
        self,
        query_embedding: np.ndarray,
        parent_key: str,
        *,
        aggregation: str = "mean_top_k",
        top_k: int = 3,
    ) -> list[dict[str, object]]:
        proto_indices = self.parent_to_proto_indices.get(parent_key)
        child_keys = self.parent_to_proto_child_keys.get(parent_key)
        if proto_indices is None or child_keys is None:
            return []

        sims = self.embeddings[proto_indices] @ query_embedding
        grouped_scores: dict[str, list[float]] = defaultdict(list)
        for child_key, sim in zip(child_keys, sims, strict=True):
            grouped_scores[child_key].append(float(sim))

        results = []
        for child_key, score_values in grouped_scores.items():
            if aggregation == "max":
                score = max(score_values)
            elif aggregation == "mean_top_k":
                selected = sorted(score_values, reverse=True)[: max(1, top_k)]
                score = float(np.mean(selected))
            else:
                raise ValueError("aggregation must be 'max' or 'mean_top_k'.")
            results.append(
                {
                    "node_key": child_key,
                    "node_name": self.node_metadata[child_key]["name"],
                    "depth": int(self.node_metadata[child_key]["depth"]),
                    "score": score,
                    "n_prototypes": len(score_values),
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results


def build_prototype_records(config: PrototypeBuildConfig) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, dict[str, object]]]:
    taxonomy_tree = load_taxonomy_tree(config.taxonomy_path)
    level2_keywords = load_level2_keywords(config.category_keywords_path)
    descendant_map = load_l2_descendant_map(config.taxonomy_path)
    synthetic_df = load_synthetic_dataset(config.synthetic_dataset_path) if config.synthetic_dataset_path else None

    rows: list[dict[str, object]] = []
    for node_key, metadata in taxonomy_tree["nodes"].items():
        if metadata["depth"] < 2:
            continue
        parent_key = metadata["parent_key"]
        if parent_key is None:
            continue

        level2_name = metadata["path_names"][1] if metadata["depth"] >= 2 else metadata["name"]
        keyword_text = level2_keywords.get(level2_name, "")
        descendants = descendant_map.get(
            (metadata["path_names"][0], metadata["path_names"][1]),
            [],
        ) if metadata["depth"] >= 2 else []
        child_names = taxonomy_tree["children_names"].get(node_key, [])
        synthetic_examples = select_synthetic_examples_for_node(
            synthetic_df=synthetic_df,
            metadata=metadata,
            config=config,
        )
        prototype_texts = create_node_prototype_texts(
            metadata=metadata,
            keyword_text=keyword_text,
            descendants=descendants,
            child_names=child_names,
            synthetic_examples=synthetic_examples,
            config=config,
        )
        for prototype_type, prototype_text in prototype_texts:
            rows.append(
                {
                    "node_key": node_key,
                    "parent_key": parent_key,
                    "node_name": metadata["name"],
                    "node_path": metadata["path_text"],
                    "depth": metadata["depth"],
                    "prototype_type": prototype_type,
                    "prototype_text": prototype_text,
                }
            )

    prototypes_df = pd.DataFrame(rows).drop_duplicates(
        subset=["node_key", "prototype_type", "prototype_text"]
    )
    return prototypes_df, taxonomy_tree["parent_to_children"], taxonomy_tree["nodes"]


def build_and_save_prototype_index(config: PrototypeBuildConfig) -> PrototypeIndex:
    config.model_name = resolve_model_name(config.model_name, task="level2")
    prototypes_df, parent_to_children, node_metadata = build_prototype_records(config)
    model = SentenceTransformer(config.model_name, device=config.device)
    embeddings = model.encode(
        prototypes_df["prototype_text"].tolist(),
        batch_size=config.batch_size,
        show_progress_bar=True,
        normalize_embeddings=config.normalize_embeddings,
        convert_to_numpy=True,
        device=config.device,
    ).astype(np.float32)
    index = PrototypeIndex(
        model_name=config.model_name,
        prototypes_df=prototypes_df,
        embeddings=embeddings,
        parent_to_children=parent_to_children,
        node_metadata=node_metadata,
    )
    index.save(config.output_dir)
    return index


def predict_dataframe(
    df: pd.DataFrame,
    *,
    index: PrototypeIndex,
    level1_col: str,
    text_col: str = "text",
    title_col: str = "title",
    description_col: str = "description",
    brand_col: str = "brand",
    price_col: str = "sale_price",
    batch_size: int = 64,
    device: Optional[str] = None,
    search_config: Optional[BeamSearchConfig] = None,
) -> pd.DataFrame:
    work = df.copy()
    work[text_col] = build_modeling_text(
        work,
        text_col=text_col,
        title_col=title_col,
        description_col=description_col,
        brand_col=brand_col,
        price_col=price_col,
    )
    embeddings = index.encode_texts(
        work[text_col].fillna("").astype(str).tolist(),
        batch_size=batch_size,
        device=device,
    )

    predictions = []
    for row, embedding in zip(work.to_dict("records"), embeddings, strict=True):
        level1_name = str(row[level1_col]).strip()
        prediction = predict_product_path(
            index=index,
            query_embedding=embedding,
            start_level1_name=level1_name,
            search_config=search_config,
        )
        predictions.append(
            {
                **row,
                "predicted_taxonomy_path": " > ".join(prediction["predicted_path"]),
                "predicted_level_2_name": prediction["predicted_path"][1]
                if len(prediction["predicted_path"]) > 1
                else "",
                "predicted_level_3_name": prediction["predicted_path"][2]
                if len(prediction["predicted_path"]) > 2
                else "",
                "predicted_level_4_name": prediction["predicted_path"][3]
                if len(prediction["predicted_path"]) > 3
                else "",
                "resolved_depth": prediction["resolved_depth"],
                "prediction_reason": prediction["reason"],
                "score_trace": json.dumps(prediction.get("score_trace", [])),
                "beam_candidates": json.dumps(prediction.get("beam_candidates", [])),
            }
        )
    return pd.DataFrame(predictions)


def load_taxonomy_tree(taxonomy_path: str | Path) -> dict[str, object]:
    nodes: dict[str, dict[str, object]] = {}
    parent_to_children: dict[str, list[str]] = defaultdict(list)
    children_names: dict[str, list[str]] = defaultdict(list)

    for line in Path(taxonomy_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        category_parts = [part.strip() for part in parts[1].split(" > ")]
        for depth, name in enumerate(category_parts, start=1):
            path_names = category_parts[:depth]
            node_key = " > ".join(path_names)
            parent_key = " > ".join(path_names[:-1]) if depth > 1 else None
            nodes.setdefault(
                node_key,
                {
                    "key": node_key,
                    "name": name,
                    "depth": depth,
                    "parent_key": parent_key,
                    "path_names": path_names,
                    "path_text": node_key,
                },
            )
            if parent_key:
                if node_key not in parent_to_children[parent_key]:
                    parent_to_children[parent_key].append(node_key)
                if name not in children_names[parent_key]:
                    children_names[parent_key].append(name)

    root_categories = sorted(
        [key for key, metadata in nodes.items() if metadata["depth"] == 1]
    )
    parent_to_children["__root__"] = root_categories
    return {
        "nodes": nodes,
        "parent_to_children": dict(parent_to_children),
        "children_names": dict(children_names),
    }


def load_synthetic_dataset(path: str | Path) -> Optional[pd.DataFrame]:
    path = Path(path)
    if not path.exists():
        return None
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("Unsupported synthetic dataset format. Use .csv or .parquet.")


def select_synthetic_examples_for_node(
    *,
    synthetic_df: Optional[pd.DataFrame],
    metadata: dict[str, object],
    config: PrototypeBuildConfig,
) -> list[str]:
    if synthetic_df is None:
        return []
    if metadata["depth"] != 2:
        return []

    mask = (
        synthetic_df[config.level1_col].fillna("").astype(str).str.strip()
        == metadata["path_names"][0]
    ) & (
        synthetic_df[config.level2_col].fillna("").astype(str).str.strip()
        == metadata["path_names"][1]
    )
    subset = synthetic_df[mask].copy()
    if subset.empty:
        return []
    examples = build_modeling_text(
        subset,
        text_col=config.text_col,
        title_col="title",
        description_col="description",
        brand_col="brand",
        price_col="sale_price",
    ).tolist()
    return examples[: config.max_synthetic_examples_per_node]


def create_node_prototype_texts(
    *,
    metadata: dict[str, object],
    keyword_text: str,
    descendants: list[str],
    child_names: list[str],
    synthetic_examples: list[str],
    config: PrototypeBuildConfig,
) -> list[tuple[str, str]]:
    node_name = metadata["name"]
    path_text = metadata["path_text"]
    parent_name = metadata["path_names"][-2] if metadata["depth"] >= 2 else ""

    rows: list[tuple[str, str]] = [
        ("category_name", node_name),
        ("path_text", path_text),
        (
            "parent_context",
            f"{node_name} under {parent_name}" if parent_name else node_name,
        ),
    ]

    if keyword_text:
        rows.append(
            (
                "enriched_description",
                f"{node_name}. {keyword_text}",
            )
        )

    if descendants:
        selected_descendants = descendants[: config.max_descendants_per_node]
        rows.append(
            (
                "descendant_summary",
                f"{node_name} includes {', '.join(selected_descendants)}.",
            )
        )
        for descendant in selected_descendants[:3]:
            rows.append(
                (
                    "descendant_focus",
                    f"{node_name} related to {descendant}",
                )
            )

    if child_names:
        rows.append(
            (
                "children_summary",
                f"{node_name} contains subcategories such as {', '.join(child_names[: config.max_descendants_per_node])}.",
            )
        )

    lexical_variants = create_lexical_expansions(node_name, parent_name, descendants, keyword_text)
    for variant in lexical_variants:
        rows.append(("lexical_expansion", variant))

    for example in synthetic_examples:
        rows.append(("synthetic_example", example))

    unique_rows = []
    seen = set()
    for prototype_type, prototype_text in rows:
        normalized = " ".join(str(prototype_text).split()).strip()
        if not normalized:
            continue
        key = (prototype_type, normalized)
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append((prototype_type, normalized))
    return unique_rows


def create_lexical_expansions(
    node_name: str,
    parent_name: str,
    descendants: list[str],
    keyword_text: str,
) -> list[str]:
    rows = [
        f"products for {node_name}",
        f"items related to {node_name}",
    ]
    if parent_name:
        rows.append(f"{node_name} in {parent_name}")
    if descendants:
        rows.append(f"{node_name} including {', '.join(descendants[:3])}")
    if keyword_text:
        rows.append(f"{node_name}: {keyword_text}")
    return rows
