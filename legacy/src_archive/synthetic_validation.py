from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


@dataclass(slots=True)
class SyntheticValidationConfig:
    text_col: str = "text"
    level1_col: str = "level_1_name"
    level2_col: str = "level_2_name"
    title_col: str = "title"
    description_col: str = "description"
    brand_col: str = "brand"
    price_col: str = "sale_price"
    hash_col: str = "hashed_external_id"
    parent_value: Optional[str] = None
    min_examples_per_class: int = 5
    max_features: int = 20_000
    ngram_max: int = 2
    cv_folds: int = 5
    random_seed: int = 42


def build_modeling_text(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    title_col: str = "title",
    description_col: str = "description",
    brand_col: str = "brand",
    price_col: str = "sale_price",
) -> pd.Series:
    if text_col in df.columns:
        return df[text_col].fillna("").astype(str).str.strip()

    work = df.copy()
    for col in (title_col, description_col, brand_col):
        if col not in work.columns:
            work[col] = ""
        work[col] = work[col].fillna("").astype(str).str.strip()

    if price_col not in work.columns:
        work[price_col] = ""
    work[price_col] = pd.to_numeric(work[price_col], errors="coerce")

    text = "Product: " + work[title_col]
    non_empty_desc = work[description_col] != ""
    text = text.where(~non_empty_desc, text + ". Description: " + work[description_col])
    non_empty_brand = work[brand_col] != ""
    text = text.where(~non_empty_brand, text + ". Brand: " + work[brand_col])
    valid_price = work[price_col].notna() & (work[price_col] > 0)
    text = text.where(
        ~valid_price, text + ". Price: " + work[price_col].map(lambda x: f"{x:.2f}")
    )
    return text.str.replace(r"\s+", " ", regex=True).str.strip()


def evaluate_real_vs_synthetic(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    config: Optional[SyntheticValidationConfig] = None,
) -> dict[str, object]:
    config = config or SyntheticValidationConfig()
    real_filtered, synthetic_filtered = _align_real_and_synthetic(
        real_df, synthetic_df, config
    )

    combined = pd.concat(
        [
            _prepare_text_frame(real_filtered, config).assign(source_label=0),
            _prepare_text_frame(synthetic_filtered, config).assign(source_label=1),
        ],
        ignore_index=True,
    )
    combined = combined[combined[config.text_col] != ""].copy()

    _ensure_binary_balance(combined["source_label"], "real_vs_synthetic")
    classifier = _build_text_classifier(config)
    cv = _build_cv(combined["source_label"], config)
    probabilities = cross_val_predict(
        classifier,
        combined[config.text_col],
        combined["source_label"],
        cv=cv,
        method="predict_proba",
        n_jobs=None,
    )[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "task": "real_vs_synthetic",
        "n_real": int((combined["source_label"] == 0).sum()),
        "n_synthetic": int((combined["source_label"] == 1).sum()),
        "auc": float(roc_auc_score(combined["source_label"], probabilities)),
        "accuracy": float(accuracy_score(combined["source_label"], predictions)),
        "macro_f1": float(
            f1_score(combined["source_label"], predictions, average="macro")
        ),
    }


def evaluate_sibling_classification(
    synthetic_df: pd.DataFrame,
    config: Optional[SyntheticValidationConfig] = None,
) -> dict[str, object]:
    config = config or SyntheticValidationConfig()
    prepared = _prepare_labeled_frame(synthetic_df, config)
    if prepared[config.level2_col].nunique() < 2:
        raise ValueError("Sibling classification needs at least two level-2 classes.")

    classifier = _build_text_classifier(config)
    cv = _build_cv(prepared[config.level2_col], config)
    predictions = cross_val_predict(
        classifier,
        prepared[config.text_col],
        prepared[config.level2_col],
        cv=cv,
        method="predict",
        n_jobs=None,
    )

    labels = sorted(prepared[config.level2_col].unique())
    report = classification_report(
        prepared[config.level2_col],
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "task": "sibling_classification",
        "n_samples": int(len(prepared)),
        "n_classes": int(prepared[config.level2_col].nunique()),
        "accuracy": float(accuracy_score(prepared[config.level2_col], predictions)),
        "macro_f1": float(
            f1_score(prepared[config.level2_col], predictions, average="macro")
        ),
        "classification_report": report,
    }


def evaluate_synthetic_to_real_transfer(
    synthetic_train_df: pd.DataFrame,
    real_test_df: pd.DataFrame,
    config: Optional[SyntheticValidationConfig] = None,
) -> dict[str, object]:
    config = config or SyntheticValidationConfig()
    synthetic_prepared = _prepare_labeled_frame(synthetic_train_df, config)
    real_prepared = _prepare_labeled_frame(real_test_df, config)
    synthetic_prepared, real_prepared = _keep_shared_classes(
        synthetic_prepared,
        real_prepared,
        label_col=config.level2_col,
    )

    if synthetic_prepared.empty or real_prepared.empty:
        raise ValueError(
            "No shared level-2 classes between synthetic train and real test."
        )

    classifier = _build_text_classifier(config)
    classifier.fit(
        synthetic_prepared[config.text_col], synthetic_prepared[config.level2_col]
    )
    predictions = classifier.predict(real_prepared[config.text_col])
    labels = sorted(real_prepared[config.level2_col].unique())
    report = classification_report(
        real_prepared[config.level2_col],
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    return {
        "task": "synthetic_to_real_transfer",
        "n_train_synthetic": int(len(synthetic_prepared)),
        "n_test_real": int(len(real_prepared)),
        "n_classes": int(len(labels)),
        "accuracy": float(
            accuracy_score(real_prepared[config.level2_col], predictions)
        ),
        "macro_f1": float(
            f1_score(real_prepared[config.level2_col], predictions, average="macro")
        ),
        "classification_report": report,
    }


def save_validation_report(
    results: dict[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def _align_real_and_synthetic(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    config: SyntheticValidationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    real_filtered = _filter_parent(real_df, config)
    synthetic_filtered = _filter_parent(synthetic_df, config)

    if (
        config.level2_col in real_filtered.columns
        and config.level2_col in synthetic_filtered.columns
    ):
        shared_classes = set(
            real_filtered[config.level2_col].dropna().astype(str)
        ) & set(synthetic_filtered[config.level2_col].dropna().astype(str))
        if shared_classes:
            real_filtered = real_filtered[
                real_filtered[config.level2_col].astype(str).isin(shared_classes)
            ].copy()
            synthetic_filtered = synthetic_filtered[
                synthetic_filtered[config.level2_col].astype(str).isin(shared_classes)
            ].copy()
    return real_filtered, synthetic_filtered


def _prepare_text_frame(
    df: pd.DataFrame,
    config: SyntheticValidationConfig,
) -> pd.DataFrame:
    prepared = df.copy()
    prepared[config.text_col] = build_modeling_text(
        prepared,
        text_col=config.text_col,
        title_col=config.title_col,
        description_col=config.description_col,
        brand_col=config.brand_col,
        price_col=config.price_col,
    )
    return prepared


def _prepare_labeled_frame(
    df: pd.DataFrame,
    config: SyntheticValidationConfig,
) -> pd.DataFrame:
    if config.level2_col not in df.columns:
        raise ValueError(f"Missing required label column '{config.level2_col}'.")

    prepared = _prepare_text_frame(_filter_parent(df, config), config)
    prepared = prepared.dropna(subset=[config.level2_col]).copy()
    prepared[config.level2_col] = prepared[config.level2_col].astype(str).str.strip()
    prepared = prepared[prepared[config.text_col] != ""].copy()

    counts = prepared[config.level2_col].value_counts()
    valid_labels = counts[counts >= config.min_examples_per_class].index
    prepared = prepared[prepared[config.level2_col].isin(valid_labels)].copy()
    if prepared.empty:
        raise ValueError(
            "No labeled samples left after minimum class-frequency filtering."
        )
    return prepared


def _keep_shared_classes(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    *,
    label_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shared = set(left_df[label_col]) & set(right_df[label_col])
    left_filtered = left_df[left_df[label_col].isin(shared)].copy()
    right_filtered = right_df[right_df[label_col].isin(shared)].copy()
    return left_filtered, right_filtered


def _filter_parent(
    df: pd.DataFrame,
    config: SyntheticValidationConfig,
) -> pd.DataFrame:
    if config.parent_value is None:
        return df.copy()
    if config.level1_col not in df.columns:
        raise ValueError(
            f"Parent filtering requested but column '{config.level1_col}' is missing."
        )
    mask = (
        df[config.level1_col].fillna("").astype(str).str.strip() == config.parent_value
    )
    filtered = df[mask].copy()
    if filtered.empty:
        raise ValueError(
            f"No rows found for parent '{config.parent_value}' in column '{config.level1_col}'."
        )
    return filtered


def _build_text_classifier(config: SyntheticValidationConfig) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    max_features=config.max_features,
                    ngram_range=(1, config.ngram_max),
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    solver="saga",
                    class_weight="balanced",
                    random_state=config.random_seed,
                ),
            ),
        ]
    )


def _build_cv(labels: pd.Series, config: SyntheticValidationConfig) -> StratifiedKFold:
    min_class_count = int(labels.value_counts().min())
    if min_class_count < 2:
        raise ValueError("Each class needs at least 2 samples for cross-validation.")
    n_splits = min(config.cv_folds, min_class_count)
    if n_splits < 2:
        raise ValueError("Unable to build a valid stratified CV split.")
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=config.random_seed,
    )


def _ensure_binary_balance(labels: pd.Series, task_name: str) -> None:
    counts = labels.value_counts()
    if len(counts) != 2:
        raise ValueError(f"{task_name} requires exactly two source classes.")
    if (counts < 2).any():
        raise ValueError(
            f"{task_name} requires at least 2 samples in each source class."
        )


def format_validation_summary(results: dict[str, object]) -> str:
    lines = []
    for key, value in results.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.4f}")
        elif isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    continue
                if isinstance(sub_value, float):
                    lines.append(f"  {sub_key}: {sub_value:.4f}")
                else:
                    lines.append(f"  {sub_key}: {sub_value}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)
