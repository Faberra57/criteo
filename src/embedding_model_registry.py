from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlparse

import pandas as pd


DEFAULT_EMBEDDING_CONFIG_PATH = Path("data/embedding_models.json")
DEFAULT_EMBEDDING_CATALOG_PATH = Path("data/huggingface_models.csv")


@dataclass(slots=True)
class EmbeddingModelConfig:
    level1_model: str = "prdev/mini-gte"
    level2_model: str = "prdev/mini-gte"
    catalog_path: str = str(DEFAULT_EMBEDDING_CATALOG_PATH)


def load_embedding_model_config(
    path: str | Path = DEFAULT_EMBEDDING_CONFIG_PATH,
) -> EmbeddingModelConfig:
    config_path = Path(path)
    if not config_path.exists():
        config = EmbeddingModelConfig()
        save_embedding_model_config(config, config_path)
        return config

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return EmbeddingModelConfig(
        level1_model=str(payload.get("level1_model", "prdev/mini-gte")).strip(),
        level2_model=str(payload.get("level2_model", "prdev/mini-gte")).strip(),
        catalog_path=str(
            payload.get("catalog_path", str(DEFAULT_EMBEDDING_CATALOG_PATH))
        ).strip(),
    )


def save_embedding_model_config(
    config: EmbeddingModelConfig,
    path: str | Path = DEFAULT_EMBEDDING_CONFIG_PATH,
) -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )


def resolve_model_name(
    model_name: Optional[str],
    *,
    task: Literal["level1", "level2"],
    config_path: str | Path = DEFAULT_EMBEDDING_CONFIG_PATH,
) -> str:
    normalized = str(model_name or "").strip()
    if normalized and normalized not in {"default", "level1", "level2"}:
        return normalized

    config = load_embedding_model_config(config_path)
    if normalized == "level2":
        return config.level2_model
    if normalized == "level1":
        return config.level1_model
    return config.level1_model if task == "level1" else config.level2_model


def load_embedding_model_catalog(
    path: str | Path = DEFAULT_EMBEDDING_CATALOG_PATH,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    catalog = df.copy()
    catalog["model_label"] = catalog["Model"].map(_extract_model_label)
    catalog["model_url"] = catalog["Model"].map(_extract_model_url)
    catalog["model_id"] = catalog.apply(_resolve_model_id, axis=1)
    return catalog


def find_catalog_row(
    model_name: str,
    *,
    catalog_path: str | Path = DEFAULT_EMBEDDING_CATALOG_PATH,
) -> Optional[dict[str, object]]:
    catalog = load_embedding_model_catalog(catalog_path)
    normalized = model_name.strip()
    matches = catalog[
        (catalog["model_id"].astype(str) == normalized)
        | (catalog["model_label"].astype(str) == normalized)
    ]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def _extract_model_label(raw_value: object) -> str:
    text = str(raw_value).strip()
    match = re.match(r"\[(.*?)\]\((.*?)\)", text)
    if match:
        return match.group(1).strip()
    return text


def _extract_model_url(raw_value: object) -> str:
    text = str(raw_value).strip()
    match = re.match(r"\[(.*?)\]\((.*?)\)", text)
    if match:
        return match.group(2).strip()
    return ""


def _resolve_model_id(row: pd.Series) -> str:
    url = str(row.get("model_url", "")).strip()
    label = str(row.get("model_label", "")).strip()
    if not url:
        return label

    parsed = urlparse(url)
    if "huggingface.co" not in parsed.netloc:
        return label

    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) >= 2:
        return f"{path_parts[0]}/{path_parts[1]}"
    return label
