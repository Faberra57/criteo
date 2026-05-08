from __future__ import annotations

import shutil
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(slots=True)
class AnnotationConfig:
    id_col: str = "hashed_external_id"
    level1_col: str = "level_1_name"
    title_col: str = "title"
    description_col: str = "description"
    brand_col: str = "brand"
    price_col: str = "sale_price"
    candidates_col: str = "candidate_level_2_values"
    label_col: str = "annotated_level_2_name"
    notes_col: str = "annotation_notes"
    status_col: str = "annotation_status"
    updated_at_col: str = "annotation_updated_at"
    wrap_width: int = 110


def load_annotation_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("Unsupported input format. Use .csv or .parquet.")


def save_annotation_frame(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError("Unsupported output format. Use .csv or .parquet.")


def ensure_backup(input_path: str | Path, output_path: str | Path) -> Optional[Path]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path != output_path or not output_path.exists():
        return None
    backup_path = output_path.with_suffix(output_path.suffix + ".bak")
    if not backup_path.exists():
        shutil.copy2(output_path, backup_path)
    return backup_path


def ensure_annotation_columns(
    df: pd.DataFrame,
    config: Optional[AnnotationConfig] = None,
) -> pd.DataFrame:
    config = config or AnnotationConfig()
    work = df.copy()
    defaults = {
        config.label_col: "",
        config.notes_col: "",
        config.status_col: "todo",
        config.updated_at_col: "",
    }
    for column, default_value in defaults.items():
        if column not in work.columns:
            work[column] = default_value
        work[column] = work[column].fillna(default_value).astype("object")
    return work


def filter_annotation_indices(
    df: pd.DataFrame,
    *,
    config: Optional[AnnotationConfig] = None,
    level1_value: Optional[str] = None,
    statuses: Optional[list[str]] = None,
) -> list[int]:
    config = config or AnnotationConfig()
    work = ensure_annotation_columns(df, config)
    mask = pd.Series(True, index=work.index)

    if level1_value is not None:
        mask &= work[config.level1_col].fillna("").astype(str).str.strip() == level1_value

    if statuses:
        valid_statuses = {status.strip() for status in statuses if status.strip()}
        mask &= work[config.status_col].fillna("todo").astype(str).isin(valid_statuses)

    return work.index[mask].tolist()


def parse_candidates(raw_value: object) -> list[str]:
    if pd.isna(raw_value):
        return []
    text = str(raw_value).strip()
    if not text:
        return []
    return [item.strip() for item in text.split("|") if item.strip()]


def mark_row(
    df: pd.DataFrame,
    row_index: int,
    *,
    config: Optional[AnnotationConfig] = None,
    label: Optional[str] = None,
    notes: Optional[str] = None,
    status: Optional[str] = None,
) -> None:
    config = config or AnnotationConfig()
    if label is not None:
        df.at[row_index, config.label_col] = label
    if notes is not None:
        df.at[row_index, config.notes_col] = notes
    if status is not None:
        df.at[row_index, config.status_col] = status
    df.at[row_index, config.updated_at_col] = datetime.now().isoformat(timespec="seconds")


def clear_row_annotation(
    df: pd.DataFrame,
    row_index: int,
    *,
    config: Optional[AnnotationConfig] = None,
) -> None:
    config = config or AnnotationConfig()
    df.at[row_index, config.label_col] = ""
    df.at[row_index, config.notes_col] = ""
    df.at[row_index, config.status_col] = "todo"
    df.at[row_index, config.updated_at_col] = datetime.now().isoformat(timespec="seconds")


def format_row_for_terminal(
    row: pd.Series,
    *,
    position: int,
    total: int,
    config: Optional[AnnotationConfig] = None,
) -> str:
    config = config or AnnotationConfig()
    candidates = parse_candidates(row.get(config.candidates_col, ""))
    title = _wrap_text(str(row.get(config.title_col, "")), config.wrap_width)
    description = _wrap_text(str(row.get(config.description_col, "")), config.wrap_width)
    current_label = _safe_string(row.get(config.label_col, ""))
    current_notes = _safe_string(row.get(config.notes_col, ""))
    current_status = _safe_string(row.get(config.status_col, "todo"))
    brand = _safe_string(row.get(config.brand_col, ""))
    price = _safe_string(row.get(config.price_col, ""))

    lines = [
        "=" * config.wrap_width,
        f"Row {position}/{total}",
        f"Hash: {_safe_string(row.get(config.id_col, ''))}",
        f"Level 1: {_safe_string(row.get(config.level1_col, ''))}",
        f"Status: {current_status}",
        f"Current L2: {current_label or '<empty>'}",
        f"Current notes: {current_notes or '<empty>'}",
        f"Brand: {brand or '<empty>'}",
        f"Price: {price or '<empty>'}",
        "",
        "Title:",
        title or "<empty>",
        "",
        "Description:",
        description or "<empty>",
        "",
        "Candidate level 2 values:",
    ]
    for idx, candidate in enumerate(candidates, start=1):
        lines.append(f"  {idx}. {candidate}")
    lines.extend(
        [
            "",
            "Commands:",
            "  number = annotate with candidate",
            "  s = skip",
            "  r = mark for review",
            "  n = edit note",
            "  c = clear current annotation",
            "  p = previous row",
            "  q = save and quit",
        ]
    )
    return "\n".join(lines)


def _safe_string(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _wrap_text(text: str, width: int) -> str:
    clean_text = " ".join(str(text).split())
    if not clean_text:
        return ""
    return "\n".join(textwrap.wrap(clean_text, width=width))
