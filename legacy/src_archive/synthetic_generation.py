from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "including",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "sold",
    "specifically",
    "that",
    "the",
    "their",
    "this",
    "to",
    "used",
    "with",
}

TITLE_ADJECTIVES = [
    "compact",
    "portable",
    "premium",
    "durable",
    "adjustable",
    "lightweight",
    "professional",
    "everyday",
    "deluxe",
    "essential",
]

USE_CASE_SUFFIXES = [
    "for everyday use",
    "for home setups",
    "for professional use",
    "for retail catalogs",
    "for hobbyists",
    "for regular maintenance",
    "for travel and storage",
    "for organized setups",
]

TITLE_SUFFIXES = [
    "kit",
    "set",
    "bundle",
    "starter pack",
    "accessory",
    "gear",
    "solution",
    "equipment",
    "essentials",
]

BRAND_FALLBACKS = [
    "NorthPeak",
    "BrightLane",
    "UrbanField",
    "EverCraft",
    "BlueHarbor",
    "StoneBridge",
    "SummitCore",
    "AsterCo",
]


@dataclass(slots=True)
class SyntheticGenerationConfig:
    taxonomy_path: str = "taxonomy.txt"
    category_keywords_path: str = "categories_level_2.csv"
    catalog_path: str = "data/ensae_export_without_l1.parquet"
    ground_truth_path: str = "data/ground_truth_level_1.parquet"
    output_synthetic_path: str = "data/synthetic_l2_seed.csv"
    output_annotation_path: str = "data/real_l2_annotation_sample.csv"
    output_summary_path: str = "data/synthetic_l2_generation_summary.json"
    examples_per_l2: int = 8
    annotation_samples_per_l1: int = 15
    random_seed: int = 42
    id_col: str = "hashed_external_id"
    level1_col: str = "level_1_name"
    level2_col: str = "level_2_name"
    title_col: str = "title"
    description_col: str = "description"
    brand_col: str = "brand"
    price_col: str = "sale_price"


def load_taxonomy_l2_map(taxonomy_path: str | Path) -> dict[str, list[str]]:
    taxonomy = defaultdict(list)
    for line in Path(taxonomy_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        categories = [part.strip() for part in parts[1].split(" > ")]
        if len(categories) < 2:
            continue
        level1, level2 = categories[0], categories[1]
        if level2 not in taxonomy[level1]:
            taxonomy[level1].append(level2)
    return dict(sorted(taxonomy.items()))


def load_level2_keywords(category_keywords_path: str | Path) -> dict[str, str]:
    df = pd.read_csv(category_keywords_path, sep=";")
    mapping = dict(
        zip(
            df["Level 2"].astype(str).str.strip(),
            df["English Keywords"].astype(str).str.strip(),
        )
    )
    if "gps tracking devices" not in mapping:
        mapping["gps tracking devices"] = (
            "Portable GPS hardware and locator units used to monitor vehicles, assets, "
            "people, or pets with position tracking and alert features."
        )
    return mapping


def load_l2_descendant_map(
    taxonomy_path: str | Path,
) -> dict[tuple[str, str], list[str]]:
    descendants = defaultdict(list)
    for line in Path(taxonomy_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        categories = [part.strip() for part in parts[1].split(" > ")]
        if len(categories) < 3:
            continue
        key = (categories[0], categories[1])
        for child in categories[2:]:
            if child not in descendants[key]:
                descendants[key].append(child)
    return dict(descendants)


def generate_synthetic_l2_dataset(
    real_df: pd.DataFrame,
    taxonomy_l2_map: dict[str, list[str]],
    level2_keywords: dict[str, str],
    descendant_map: Optional[dict[tuple[str, str], list[str]]] = None,
    config: Optional[SyntheticGenerationConfig] = None,
) -> pd.DataFrame:
    config = config or SyntheticGenerationConfig()
    rng = random.Random(config.random_seed)
    descendant_map = descendant_map or {}

    rows: list[dict[str, object]] = []
    for level1_name, level2_values in taxonomy_l2_map.items():
        style_df = real_df[real_df[config.level1_col] == level1_name].copy()
        if style_df.empty:
            continue
        style_profile = _build_style_profile(style_df, config, rng)
        for level2_name in level2_values:
            keyword_text = level2_keywords.get(level2_name, _fallback_keywords(level2_name))
            descendants = descendant_map.get((level1_name, level2_name), [])
            for example_idx in range(config.examples_per_l2):
                title, description = _generate_title_and_description(
                    level1_name=level1_name,
                    level2_name=level2_name,
                    keyword_text=keyword_text,
                    descendants=descendants,
                    style_profile=style_profile,
                    rng=rng,
                    variant_index=example_idx,
                )
                brand = _sample_brand(style_profile, level2_name, rng)
                price = _sample_price(style_profile, rng)
                rows.append(
                    {
                        config.level1_col: level1_name,
                        config.level2_col: level2_name,
                        config.title_col: title,
                        config.description_col: description,
                        config.brand_col: brand,
                        config.price_col: price,
                        "text": _build_text(title, description, brand, price),
                        "source": "synthetic_llm_targeted",
                        "generation_method": "targeted_l2_context_generation_v2",
                        "seed_keywords": keyword_text,
                        "seed_descendants": " | ".join(descendants[:8]),
                    }
                )
    return pd.DataFrame(rows)


def sample_real_dataset_for_annotation(
    real_df: pd.DataFrame,
    taxonomy_l2_map: dict[str, list[str]],
    config: Optional[SyntheticGenerationConfig] = None,
) -> pd.DataFrame:
    config = config or SyntheticGenerationConfig()
    rng = random.Random(config.random_seed)
    samples: list[pd.DataFrame] = []

    for level1_name, level2_values in taxonomy_l2_map.items():
        subset = real_df[real_df[config.level1_col] == level1_name].copy()
        if subset.empty:
            continue
        n_samples = min(config.annotation_samples_per_l1, len(subset))
        random_state = rng.randint(0, 1_000_000)
        sample = subset.sample(n=n_samples, random_state=random_state).copy()
        sample["candidate_level_2_values"] = " | ".join(level2_values)
        sample["annotated_level_2_name"] = ""
        sample["annotation_notes"] = ""
        sample["annotation_status"] = "todo"
        samples.append(sample)

    if not samples:
        return pd.DataFrame()

    annotation_df = pd.concat(samples, ignore_index=True)
    columns = [
        config.id_col,
        config.level1_col,
        config.title_col,
        config.description_col,
        config.brand_col,
        config.price_col,
        "candidate_level_2_values",
        "annotated_level_2_name",
        "annotation_notes",
        "annotation_status",
    ]
    return annotation_df[columns]


def save_dataframe(df: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet.")


def save_generation_summary(
    synthetic_df: pd.DataFrame,
    annotation_df: pd.DataFrame,
    taxonomy_l2_map: dict[str, list[str]],
    config: Optional[SyntheticGenerationConfig] = None,
) -> dict[str, object]:
    config = config or SyntheticGenerationConfig()
    summary = {
        "config": asdict(config),
        "n_level1_nodes": len(taxonomy_l2_map),
        "n_level2_nodes": int(sum(len(values) for values in taxonomy_l2_map.values())),
        "n_synthetic_rows": int(len(synthetic_df)),
        "n_annotation_rows": int(len(annotation_df)),
        "synthetic_counts_by_level1": synthetic_df[config.level1_col].value_counts()
        .sort_index()
        .to_dict(),
        "synthetic_counts_by_level2": synthetic_df[config.level2_col].value_counts()
        .sort_index()
        .to_dict(),
        "annotation_counts_by_level1": annotation_df[config.level1_col].value_counts()
        .sort_index()
        .to_dict(),
    }
    path = Path(config.output_summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _build_style_profile(
    style_df: pd.DataFrame,
    config: SyntheticGenerationConfig,
    rng: random.Random,
) -> dict[str, object]:
    work = style_df.copy()
    work[config.title_col] = work[config.title_col].fillna("").astype(str)
    work[config.description_col] = work[config.description_col].fillna("").astype(str)
    work[config.brand_col] = work[config.brand_col].replace({"None": np.nan})
    work[config.price_col] = pd.to_numeric(work[config.price_col], errors="coerce")

    brands = (
        work[config.brand_col]
        .dropna()
        .astype(str)
        .str.strip()
    )
    brands = brands[brands != ""]
    prices = work[config.price_col].dropna()
    prices = prices[prices > 0]
    if not prices.empty:
        lower = float(prices.quantile(0.10))
        upper = float(prices.quantile(0.90))
        trimmed_prices = prices[(prices >= lower) & (prices <= upper)]
        if not trimmed_prices.empty:
            prices = trimmed_prices

    title_lengths = work[config.title_col].str.split().map(len)
    desc_lengths = work[config.description_col].str.split().map(len)

    return {
        "brands": brands.tolist(),
        "brand_presence_rate": float(len(brands) / max(len(work), 1)),
        "prices": prices.tolist(),
        "title_length_mean": float(title_lengths.mean()) if not title_lengths.empty else 8.0,
        "desc_length_mean": float(desc_lengths.mean()) if not desc_lengths.empty else 35.0,
        "level1_name": str(work[config.level1_col].iloc[0]),
        "rng_fallback_brand": rng.choice(BRAND_FALLBACKS),
    }


def _generate_title_and_description(
    *,
    level1_name: str,
    level2_name: str,
    keyword_text: str,
    descendants: list[str],
    style_profile: dict[str, object],
    rng: random.Random,
    variant_index: int,
) -> tuple[str, str]:
    terms = _extract_terms(level2_name, keyword_text)
    phrases = _extract_phrases(keyword_text)
    descendant_phrases = _prepare_descendant_phrases(descendants)
    focus_phrases = descendant_phrases + phrases + terms + [level2_name]
    focus_phrases = _deduplicate_preserve_order([item for item in focus_phrases if item])
    lead_term = focus_phrases[0] if focus_phrases else level2_name
    second_term = focus_phrases[1] if len(focus_phrases) > 1 else lead_term
    extra_term = focus_phrases[2] if len(focus_phrases) > 2 else second_term
    adjective = rng.choice(TITLE_ADJECTIVES)
    suffix = rng.choice(TITLE_SUFFIXES)
    use_case = rng.choice(USE_CASE_SUFFIXES)
    commerce_label = rng.choice(
        [
            "collection",
            "series",
            "bundle",
            "set",
            "kit",
            "edition",
            "range",
            "essentials",
        ]
    )
    title_templates = [
        f"{adjective.title()} {lead_term} {suffix}",
        f"{_title_case_phrase(lead_term)} with {second_term}",
        f"{_title_case_phrase(level2_name)} {commerce_label}: {lead_term}",
        f"{_title_case_phrase(second_term)} for {level2_name}",
        f"{adjective.title()} {lead_term} and {second_term} set",
        f"{_title_case_phrase(lead_term)} starter pack for {second_term}",
        f"{_title_case_phrase(level2_name)} gear with {extra_term}",
        f"{adjective.title()} {level2_name} {suffix} for {lead_term}",
        f"{_title_case_phrase(lead_term)} merchandise with {extra_term}",
        f"{_title_case_phrase(level2_name)} listing for {lead_term}",
    ]
    title = title_templates[variant_index % len(title_templates)]
    title = _clean_phrase(title, max_words=max(6, int(style_profile["title_length_mean"])))

    category_cue = _build_category_cue(level2_name, descendant_phrases, phrases, terms)
    description_templates = [
        (
            f"{keyword_text.rstrip('.')}. This catalog-style example emphasizes {category_cue} "
            f"and keeps the merchandising tone typical of {level1_name} listings."
        ),
        (
            f"Built around {level2_name}, this sample focuses on {lead_term}, {second_term}, "
            f"and {extra_term}. The wording is shaped for retail-style descriptions {use_case}."
        ),
        (
            f"{keyword_text.rstrip('.')}. Typical examples in this category mention {lead_term}, "
            f"{second_term}, and {extra_term}, with concrete cues intended {use_case}."
        ),
        (
            f"{_indefinite_article(level2_name)} {level2_name} listing created for {level1_name} shoppers, combining {lead_term} "
            f"positioning with {second_term} support and {extra_term} details."
        ),
        (
            f"This synthetic example stays inside the {level2_name} scope by centering on "
            f"{lead_term} and nearby cues such as {second_term} and {extra_term}. "
            f"It is phrased like a marketplace product detail page."
        ),
        (
            f"For {level2_name}, the title and description highlight {category_cue}. "
            f"The wording avoids broad taxonomy language and instead stays close to sellable item wording."
        ),
    ]
    description = description_templates[variant_index % len(description_templates)]
    description = _clean_phrase(
        description,
        max_words=max(18, int(style_profile["desc_length_mean"])),
    )
    return title, description


def _sample_brand(
    style_profile: dict[str, object],
    level2_name: str,
    rng: random.Random,
) -> Optional[str]:
    service_like_tokens = {
        "live animals",
        "event tickets",
        "finance",
        "insurance",
        "real estate",
        "adult",
        "tobacco products",
        "law enforcement",
    }
    if any(token in level2_name for token in service_like_tokens):
        return None
    brands: list[str] = style_profile["brands"]
    if brands and rng.random() < style_profile["brand_presence_rate"]:
        return rng.choice(brands)
    if rng.random() < 0.35:
        return style_profile["rng_fallback_brand"]
    return None


def _sample_price(style_profile: dict[str, object], rng: random.Random) -> float:
    prices: list[float] = style_profile["prices"]
    if prices:
        price = float(rng.choice(prices))
    else:
        price = float(rng.uniform(10.0, 120.0))
    return round(max(1.99, min(price, 999.99)), 2)


def _extract_terms(category_name: str, keyword_text: str) -> list[str]:
    raw_tokens = re.findall(r"[a-z0-9]+", f"{category_name} {keyword_text}".lower())
    terms = []
    for token in raw_tokens:
        if token in STOPWORDS or len(token) <= 2:
            continue
        if token not in terms:
            terms.append(token)
    return terms[:8]


def _extract_phrases(keyword_text: str) -> list[str]:
    normalized = keyword_text.lower().strip().rstrip(".")
    chunks = re.split(r",| and ", normalized)
    phrases: list[str] = []
    for chunk in chunks:
        phrase = chunk.strip()
        phrase = re.sub(
            r"^.*?\b(including|used for|designed specifically for|designed for|ranging from|"
            r"dedicated to|tailored for|formulated for|found in|sold for|manufactured for|"
            r"created for|made for|used to monitor)\b",
            "",
            phrase,
        ).strip()
        phrase = re.sub(
            r"\b(sold for|used for|intended for|with alert features|for companionship.*|"
            r"for agricultural purposes.*)$",
            "",
            phrase,
        ).strip()
        phrase = re.sub(r"[^a-z0-9\s&-]", " ", phrase)
        phrase = re.sub(r"\s+", " ", phrase).strip(" -")
        if not phrase:
            continue
        if phrase not in phrases:
            phrases.append(phrase)
    return phrases[:6]


def _prepare_descendant_phrases(descendants: list[str]) -> list[str]:
    prepared = []
    for child in descendants:
        normalized = re.sub(r"[^a-z0-9\s&-]", " ", child.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip(" -")
        if normalized and normalized not in prepared:
            prepared.append(normalized)
    return prepared[:20]


def _fallback_keywords(level2_name: str) -> str:
    readable = level2_name.replace("&", "and")
    return (
        f"Products and accessories related to {readable}, typically sold through "
        "consumer or specialty retail catalogs."
    )


def _build_text(
    title: str,
    description: str,
    brand: Optional[str],
    price: float,
) -> str:
    parts = [f"Product: {title}."]
    if description:
        parts.append(f"Description: {description}.")
    if brand:
        parts.append(f"Brand: {brand}.")
    if price > 0:
        parts.append(f"Price: {price:.2f} USD.")
    return " ".join(parts)


def _clean_phrase(text: str, *, max_words: int) -> str:
    words = text.replace("  ", " ").split()
    truncated = words[:max_words]
    cleaned = " ".join(truncated)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")
    return cleaned


def _title_case_phrase(text: str) -> str:
    parts = text.split()
    return " ".join(part.capitalize() if part not in {"and", "or", "for"} else part for part in parts)


def _deduplicate_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _build_category_cue(
    level2_name: str,
    descendant_phrases: list[str],
    phrases: list[str],
    terms: list[str],
) -> str:
    candidates = _deduplicate_preserve_order(
        descendant_phrases[:3] + phrases[:2] + terms[:2] + [level2_name]
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 2:
        return f"{candidates[0]} and {candidates[1]}"
    return f"{candidates[0]}, {candidates[1]}, and {candidates[2]}"


def _indefinite_article(text: str) -> str:
    first_char = text.strip().lower()[:1]
    return "An" if first_char in {"a", "e", "i", "o", "u"} else "A"


def generate_bootstrap_artifacts(
    config: Optional[SyntheticGenerationConfig] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    config = config or SyntheticGenerationConfig()
    taxonomy_l2_map = load_taxonomy_l2_map(config.taxonomy_path)
    descendant_map = load_l2_descendant_map(config.taxonomy_path)
    level2_keywords = load_level2_keywords(config.category_keywords_path)

    catalog_df = pd.read_parquet(config.catalog_path)
    ground_truth_df = pd.read_parquet(config.ground_truth_path)
    real_df = catalog_df.merge(ground_truth_df, on=config.id_col, how="left")
    real_df = real_df[real_df[config.level1_col].isin(taxonomy_l2_map)].copy()

    synthetic_df = generate_synthetic_l2_dataset(
        real_df=real_df,
        taxonomy_l2_map=taxonomy_l2_map,
        level2_keywords=level2_keywords,
        descendant_map=descendant_map,
        config=config,
    )
    annotation_df = sample_real_dataset_for_annotation(
        real_df=real_df,
        taxonomy_l2_map=taxonomy_l2_map,
        config=config,
    )

    save_dataframe(synthetic_df, config.output_synthetic_path)
    save_dataframe(annotation_df, config.output_annotation_path)
    summary = save_generation_summary(
        synthetic_df=synthetic_df,
        annotation_df=annotation_df,
        taxonomy_l2_map=taxonomy_l2_map,
        config=config,
    )
    return synthetic_df, annotation_df, summary
