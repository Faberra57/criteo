from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.preprocessing import LabelEncoder


@dataclass(slots=True)
class PreprocessingConfig:
    title_col: str = "title"
    desc_col: str = "description"
    brand_col: str = "brand"
    id_col: str = "hashed_external_id"
    price_col: str = "sale_price"
    category_col: str = "level_1_name"
    max_description_chars: int = 800
    missing_brand_value: str = "Unknown"
    currency: str = "USD"


@dataclass(slots=True)
class LanguageDetectionConfig:
    text_col: str = "text"
    output_col: str = "language"
    random_seed: int = 0


def load_catalog_and_ground_truth(
    catalog_path: str | Path,
    ground_truth_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog_df = pd.read_parquet(catalog_path, engine="pyarrow")
    ground_truth_df = pd.read_parquet(ground_truth_path, engine="pyarrow")
    return catalog_df, ground_truth_df


def merge_catalog_with_ground_truth(
    catalog_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    *,
    id_col: str = "hashed_external_id",
    category_col: str = "level_1_name",
) -> pd.DataFrame:
    columns = [id_col, category_col]
    return catalog_df.merge(ground_truth_df[columns], on=id_col, how="left")


def preprocess_for_nlp(
    df: pd.DataFrame,
    config: Optional[PreprocessingConfig] = None,
) -> tuple[pd.DataFrame, Optional[LabelEncoder]]:
    config = config or PreprocessingConfig()
    data = df.copy()

    data[config.title_col] = data[config.title_col].fillna("")
    data[config.desc_col] = data[config.desc_col].fillna("")
    data[config.brand_col] = data[config.brand_col].fillna(config.missing_brand_value)

    if config.price_col in data.columns:
        data[config.price_col] = pd.to_numeric(data[config.price_col], errors="coerce")
        median_price = data[config.price_col].median()
        if pd.isna(median_price):
            median_price = 0.0
        data[config.price_col] = data[config.price_col].fillna(median_price)
    else:
        data[config.price_col] = 0.0

    data["text"] = data.apply(
        lambda row: _create_rich_text(row=row, config=config),
        axis=1,
    )
    data["text"] = data["text"].map(_clean_text)

    label_encoder: Optional[LabelEncoder] = None
    encoded_brand_col = f"{config.brand_col}_encoded"
    if config.brand_col in data.columns:
        label_encoder = LabelEncoder()
        data[encoded_brand_col] = label_encoder.fit_transform(
            data[config.brand_col].astype(str)
        )

    columns_to_keep = [config.id_col, "text", config.price_col]
    if config.category_col in data.columns:
        columns_to_keep.append(config.category_col)
    if encoded_brand_col in data.columns:
        columns_to_keep.append(encoded_brand_col)

    return data[columns_to_keep], label_encoder


def save_preprocessed_frame(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    elif output_path.suffix == ".csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError("Unsupported output format. Use .parquet or .csv.")


def _create_rich_text(row: pd.Series, config: PreprocessingConfig) -> str:
    text = f"Product: {row[config.title_col]}. "

    description = str(row[config.desc_col]).strip()
    if description:
        text += f"Description: {description[: config.max_description_chars]}. "

    if row[config.brand_col] != config.missing_brand_value:
        text += f"Brand: {row[config.brand_col]}. "

    if row[config.price_col] > 0:
        text += f"Price: {row[config.price_col]:.2f} {config.currency}."

    return text.strip()


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()
