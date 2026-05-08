from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from huggingface_hub import snapshot_download

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


logger = logging.getLogger(__name__)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_MLX_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DEFAULT_MLX_LOCAL_MODEL_DIR = "models/mlx-community/Qwen2.5-7B-Instruct-4bit"
DEFAULT_GITHUB_MODELS_BASE_URL = "https://models.inference.ai.azure.com"
DEFAULT_GITHUB_MODEL = "gpt-4o-mini"
DEFAULT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


@dataclass(slots=True)
class CategoryEnrichmentConfig:
    taxonomy_path: str = "taxonomy.txt"
    output_dir: str = "artifacts/category_enrichment"
    provider: str = "local"
    model: Optional[str] = None
    api_base_url: str = DEFAULT_GITHUB_MODELS_BASE_URL
    api_key_env: str = "GITHUB_TOKEN"
    local_model_dir: Optional[str] = DEFAULT_MLX_LOCAL_MODEL_DIR
    model_revision: Optional[str] = None
    mlx_backend: str = "auto"
    trust_remote_code: bool = False
    temperature: float = 0.2
    levels: Optional[list[int]] = None
    max_children: int = 12
    max_descendants: int = 12
    min_lexical_variants: int = 10
    priority_category_names: Optional[list[str]] = None
    priority_node_keys: Optional[list[str]] = None
    priority_min_lexical_variants: int = 24
    request_max_tokens: int = 420
    request_batch_size: int = 1
    max_retries: int = 6
    retry_backoff_seconds: float = 65.0
    max_retry_delay_seconds: float = 120.0
    network_max_retries: int = 3
    network_retry_backoff_seconds: float = 5.0
    save_every_n: int = 50
    limit_per_level: Optional[int] = None
    overwrite: bool = False
    sleep_seconds: float = 0.0


def load_taxonomy_nodes(taxonomy_path: str | Path) -> tuple[
    dict[str, dict[str, object]],
    dict[str, list[str]],
]:
    nodes: dict[str, dict[str, object]] = {}
    parent_to_children: dict[str, list[str]] = defaultdict(list)

    for line in Path(taxonomy_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        category_parts = [part.strip() for part in parts[1].split(" > ") if part.strip()]
        for depth, name in enumerate(category_parts, start=1):
            path_names = category_parts[:depth]
            node_key = " > ".join(path_names)
            parent_key = " > ".join(path_names[:-1]) if depth > 1 else None
            if node_key not in nodes:
                nodes[node_key] = {
                    "node_key": node_key,
                    "category_name": name,
                    "depth": depth,
                    "parent_key": parent_key,
                    "parent_name": path_names[-2] if depth > 1 else "",
                    "taxonomy_path": node_key,
                    "path_names": path_names,
                }
            if parent_key and node_key not in parent_to_children[parent_key]:
                parent_to_children[parent_key].append(node_key)

    for node_key, metadata in nodes.items():
        child_keys = parent_to_children.get(node_key, [])
        metadata["child_keys"] = child_keys
        metadata["child_names"] = [nodes[child_key]["category_name"] for child_key in child_keys]
        descendant_names = _collect_descendant_names(
            node_key=node_key,
            nodes=nodes,
            parent_to_children=parent_to_children,
            include_direct_children=False,
        )
        metadata["descendant_names"] = descendant_names

    return nodes, dict(parent_to_children)


def generate_category_enrichment(
    config: Optional[CategoryEnrichmentConfig] = None,
) -> dict[int, dict[str, Path]]:
    config = config or CategoryEnrichmentConfig()
    load_dotenv_file(DEFAULT_ENV_PATH)
    nodes, parent_to_children = load_taxonomy_nodes(config.taxonomy_path)
    levels = sorted(config.levels or {int(metadata["depth"]) for metadata in nodes.values()})

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    client = _build_enrichment_client(config)

    outputs: dict[int, dict[str, Path]] = {}
    for level in levels:
        level_nodes = [
            metadata
            for metadata in nodes.values()
            if int(metadata["depth"]) == level
        ]
        level_nodes.sort(key=lambda item: str(item["taxonomy_path"]))
        if config.limit_per_level is not None:
            level_nodes = level_nodes[: config.limit_per_level]

        category_path = output_dir / f"level_{level}_categories.csv"
        reference_path = output_dir / f"level_{level}_reference_texts.csv"
        outputs[level] = {
            "categories_csv": category_path,
            "reference_texts_csv": reference_path,
        }

        category_rows: list[dict[str, object]] = []
        reference_rows: list[dict[str, object]] = []
        existing_category_keys: set[str] = set()
        existing_reference_keys: set[str] = set()
        expected_reference_counts: dict[str, int] = {}
        actual_reference_counts: dict[str, int] = {}

        if category_path.exists() and not config.overwrite:
            existing_categories = pd.read_csv(category_path)
            category_rows = existing_categories.to_dict("records")
            existing_category_keys = {
                str(value).strip()
                for value in existing_categories.get("node_key", pd.Series(dtype="object"))
            }
            expected_reference_counts = _extract_expected_reference_counts(category_rows)
        if reference_path.exists() and not config.overwrite:
            existing_references = pd.read_csv(reference_path)
            reference_rows = existing_references.to_dict("records")
            existing_reference_keys = {
                str(value).strip()
                for value in existing_references.get("node_key", pd.Series(dtype="object"))
            }
            actual_reference_counts = _count_reference_rows_by_node_key(reference_rows)

        pending_nodes = [
            metadata
            for metadata in level_nodes
            if config.overwrite
            or str(metadata["node_key"]) not in existing_category_keys
            or str(metadata["node_key"]) not in existing_reference_keys
            or actual_reference_counts.get(str(metadata["node_key"]), 0)
            < expected_reference_counts.get(str(metadata["node_key"]), 0)
        ]
        processed_nodes = 0
        progress_bar = _create_progress_bar(
            total=len(pending_nodes),
            description=f"Level {level}",
        )
        try:
            for metadata_batch in _batch_items(
                pending_nodes,
                batch_size=max(1, config.request_batch_size),
            ):
                enrichments = client.enrich_many(metadata_batch, parent_to_children)
                for metadata, enrichment in zip(metadata_batch, enrichments, strict=True):
                    node_key = str(metadata["node_key"])
                    normalized = normalize_enrichment_payload(
                        metadata,
                        enrichment,
                        config,
                    )
                    category_rows = _replace_rows_for_node_key(
                        category_rows,
                        node_key=node_key,
                        new_rows=[build_category_row(metadata, normalized, config)],
                    )
                    reference_rows = _replace_rows_for_node_key(
                        reference_rows,
                        node_key=node_key,
                        new_rows=build_reference_rows(metadata, normalized),
                    )
                    existing_category_keys.add(node_key)
                    existing_reference_keys.add(node_key)
                    actual_reference_counts[node_key] = sum(
                        1 for row in reference_rows if str(row.get("node_key", "")).strip() == node_key
                    )
                    processed_nodes += 1

                    if processed_nodes % max(1, config.save_every_n) == 0:
                        _save_rows(category_rows, category_path)
                        _save_rows(reference_rows, reference_path)

                    if config.sleep_seconds > 0:
                        time.sleep(config.sleep_seconds)

                _advance_progress_bar(progress_bar, len(metadata_batch))
        finally:
            _close_progress_bar(progress_bar)

        if processed_nodes > 0:
            _save_rows(category_rows, category_path)
            _save_rows(reference_rows, reference_path)
        if processed_nodes == 0 and not category_path.exists():
            _save_rows(category_rows, category_path)
        if processed_nodes == 0 and not reference_path.exists():
            _save_rows(reference_rows, reference_path)

    summary_path = output_dir / "generation_summary.json"
    summary_payload = {
        "config": asdict(config),
        "levels": {
            str(level): {
                "categories_csv": str(paths["categories_csv"]),
                "reference_texts_csv": str(paths["reference_texts_csv"]),
            }
            for level, paths in outputs.items()
        },
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return outputs


def build_category_row(
    metadata: dict[str, object],
    enrichment: dict[str, object],
    config: CategoryEnrichmentConfig,
) -> dict[str, object]:
    reference_texts = build_reference_texts(metadata, enrichment, config)
    return {
        "node_key": metadata["node_key"],
        "depth": metadata["depth"],
        "category_name": metadata["category_name"],
        "taxonomy_path": metadata["taxonomy_path"],
        "parent_key": metadata["parent_key"] or "",
        "parent_name": metadata["parent_name"],
        "child_count": len(metadata["child_names"]),
        "children_names": " | ".join(metadata["child_names"]),
        "descendant_count": len(metadata["descendant_names"]),
        "descendant_names": " | ".join(metadata["descendant_names"][: config.max_descendants]),
        "enriched_description": enrichment["enriched_description"],
        "children_summary": enrichment["children_summary"],
        "descendants_summary": enrichment["descendants_summary"],
        "lexical_variants_json": json.dumps(enrichment["lexical_variants"], ensure_ascii=False),
        "reference_texts_json": json.dumps(reference_texts, ensure_ascii=False),
        "is_priority_category": _is_priority_category(metadata, config),
        "lexical_variant_count": len(enrichment["lexical_variants"]),
        "generation_provider": config.provider,
        "generation_model": config.model or "",
    }


def build_reference_rows(
    metadata: dict[str, object],
    enrichment: dict[str, object],
    config: Optional[CategoryEnrichmentConfig] = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for prototype_type, texts in build_reference_texts(metadata, enrichment, config).items():
        if isinstance(texts, str):
            items = [texts]
        else:
            items = list(texts)
        for text in items:
            normalized = " ".join(str(text).split()).strip()
            if not normalized:
                continue
            rows.append(
                {
                    "node_key": metadata["node_key"],
                    "depth": metadata["depth"],
                    "category_name": metadata["category_name"],
                    "prototype_type": prototype_type,
                    "prototype_text": normalized,
                }
            )
    return rows


def build_reference_texts(
    metadata: dict[str, object],
    enrichment: dict[str, object],
    config: Optional[CategoryEnrichmentConfig] = None,
) -> dict[str, object]:
    config = config or CategoryEnrichmentConfig()
    descendant_names = [
        str(item).strip()
        for item in list(metadata["descendant_names"])[: config.max_descendants]
        if str(item).strip()
    ]
    child_names = [
        str(item).strip()
        for item in list(metadata["child_names"])[: config.max_children]
        if str(item).strip()
    ]
    reference_texts: dict[str, object] = {
        "category_name": str(metadata["category_name"]),
        "taxonomy_path": str(metadata["taxonomy_path"]),
        "parent_context": (
            f"{str(metadata['category_name']).strip()} under {str(metadata['parent_name']).strip()}"
            if str(metadata["parent_name"]).strip()
            else str(metadata["category_name"]).strip()
        ),
        "enriched_description": str(enrichment.get("enriched_description", "")).strip(),
        "children_summary": str(enrichment.get("children_summary", "")).strip(),
        "descendants_summary": str(enrichment.get("descendants_summary", "")).strip(),
        "children_names_text": " | ".join(child_names),
        "descendant_names_text": " | ".join(descendant_names),
        "lexical_expansion": list(enrichment["lexical_variants"]),
    }
    return reference_texts


def normalize_enrichment_payload(
    metadata: dict[str, object],
    payload: dict[str, object],
    config: Optional[CategoryEnrichmentConfig] = None,
) -> dict[str, object]:
    config = config or CategoryEnrichmentConfig()
    lexical_variants = payload.get("lexical_variants", [])
    if isinstance(lexical_variants, str):
        lexical_variants = [item.strip() for item in lexical_variants.split("|") if item.strip()]
    elif not isinstance(lexical_variants, list):
        lexical_variants = []

    lexical_variants = _deduplicate_preserve_order(
        [str(item).strip() for item in lexical_variants if str(item).strip()]
    )
    target_count = _target_lexical_variant_count(metadata, config)
    lexical_variants = _ensure_min_lexical_variants(
        metadata,
        lexical_variants,
        min_count=target_count,
    )

    return {
        "enriched_description": str(payload.get("enriched_description") or "").strip()
        or _local_description(metadata),
        "children_summary": str(payload.get("children_summary") or "").strip()
        or _local_children_summary(metadata),
        "descendants_summary": str(payload.get("descendants_summary") or "").strip()
        or _local_descendants_summary(metadata),
        "lexical_variants": lexical_variants,
    }


class _BaseEnrichmentClient:
    def __init__(self, config: CategoryEnrichmentConfig) -> None:
        self.config = config

    def enrich(
        self,
        metadata: dict[str, object],
        parent_to_children: dict[str, list[str]],
    ) -> dict[str, object]:
        raise NotImplementedError

    def enrich_many(
        self,
        metadata_batch: list[dict[str, object]],
        parent_to_children: dict[str, list[str]],
    ) -> list[dict[str, object]]:
        return [
            self.enrich(metadata, parent_to_children)
            for metadata in metadata_batch
        ]


class _TemplateEnrichmentClient(_BaseEnrichmentClient):
    def enrich(
        self,
        metadata: dict[str, object],
        parent_to_children: dict[str, list[str]],
    ) -> dict[str, object]:
        return {
            "enriched_description": _template_description(metadata),
            "children_summary": _template_children_summary(metadata),
            "descendants_summary": _template_descendants_summary(metadata),
            "lexical_variants": _template_lexical_variants(metadata),
        }


class _LocalEnrichmentClient(_BaseEnrichmentClient):
    def __init__(self, config: CategoryEnrichmentConfig) -> None:
        super().__init__(config)
        if not config.model:
            raise ValueError("--model is required when provider is 'local'.")
        self.model_ref = prepare_local_model_reference(config)
        self.mlx_backend = _resolve_mlx_backend(self.model_ref, config.mlx_backend)
        config.mlx_backend = self.mlx_backend
        config.local_model_dir = str(self.model_ref) if isinstance(self.model_ref, Path) else str(
            config.local_model_dir or ""
        )

        if self.mlx_backend == "lm":
            try:
                from mlx_lm import batch_generate as mlx_batch_generate
                from mlx_lm import generate as mlx_generate
                from mlx_lm import load as mlx_load
                from mlx_lm.sample_utils import make_sampler
            except Exception as exc:
                raise ImportError(
                    "Unable to import 'mlx_lm'. Run this script outside the sandbox on Apple "
                    "Silicon with mlx-lm installed."
                ) from exc

            self._mlx_batch_generate = mlx_batch_generate
            self._mlx_generate = mlx_generate
            self._mlx_make_sampler = make_sampler
            self.model, self.tokenizer = mlx_load(
                str(self.model_ref),
                tokenizer_config={"trust_remote_code": self.config.trust_remote_code},
                revision=self.config.model_revision,
            )
            logger.info(
                "Loaded MLX-LM model '%s'.",
                self.model_ref,
            )
            return

        if self.mlx_backend == "vlm":
            try:
                from mlx_vlm import generate as mlx_vlm_generate
                from mlx_vlm import load as mlx_vlm_load
                from mlx_vlm.prompt_utils import apply_chat_template as mlx_vlm_apply_chat_template
            except Exception as exc:
                raise ImportError(
                    "The selected model appears to be an MLX VLM model. Install 'mlx-vlm' "
                    "to use it, for example with `pip install -U mlx-vlm`."
                ) from exc

            self._mlx_generate = mlx_vlm_generate
            self._mlx_vlm_apply_chat_template = mlx_vlm_apply_chat_template
            self.model, self.processor = mlx_vlm_load(str(self.model_ref))
            logger.info(
                "Loaded MLX-VLM model '%s' for text-only prompting.",
                self.model_ref,
            )
            return

        raise ValueError("mlx_backend must resolve to 'lm' or 'vlm'.")

    def enrich(
        self,
        metadata: dict[str, object],
        parent_to_children: dict[str, list[str]],
    ) -> dict[str, object]:
        prompt = build_local_lexical_prompt(metadata, self.config)
        parsed = self._request_local_payload(
            prompt=prompt,
            metadata=metadata,
            max_new_tokens=self.config.request_max_tokens,
            context_label=str(metadata.get("node_key", "<unknown>")),
        )
        if not isinstance(parsed, dict):
            raise ValueError("Local LLM response did not contain a JSON object.")
        return parsed

    def enrich_many(
        self,
        metadata_batch: list[dict[str, object]],
        parent_to_children: dict[str, list[str]],
    ) -> list[dict[str, object]]:
        if self.mlx_backend != "lm" or len(metadata_batch) <= 1:
            return super().enrich_many(metadata_batch, parent_to_children)

        prompts = [
            self._tokenize_chat_prompt(build_local_lexical_prompt(metadata, self.config))
            for metadata in metadata_batch
        ]
        sampler = self._mlx_make_sampler(
            temp=max(0.0, float(self.config.temperature)),
        )
        response = self._mlx_batch_generate(
            self.model,
            self.tokenizer,
            prompts=prompts,
            max_tokens=self.config.request_max_tokens,
            verbose=False,
            sampler=sampler,
        )

        payloads: list[dict[str, object]] = []
        try:
            for metadata, text in zip(metadata_batch, response.texts, strict=True):
                parsed = _parse_local_lexical_response(
                    str(text),
                    metadata,
                    self.config,
                )
                if not isinstance(parsed, dict):
                    raise ValueError(
                        f"Local MLX batch response did not contain a JSON object for '{metadata['node_key']}'."
                    )
                payloads.append(parsed)
            return payloads
        except Exception as exc:
            logger.warning(
                "Local MLX batch parsing failed for %s nodes. Falling back to single-node generation. Error: %s",
                len(metadata_batch),
                exc,
            )
            return super().enrich_many(metadata_batch, parent_to_children)

    def _request_local_payload(
        self,
        *,
        prompt: str,
        metadata: dict[str, object],
        max_new_tokens: int,
        context_label: str,
    ) -> object:
        last_error: Optional[Exception] = None
        attempts = max(1, min(3, self.config.network_max_retries + 1))
        for attempt_idx in range(1, attempts + 1):
            try:
                generated_text = self._generate_text(prompt, max_new_tokens=max_new_tokens)
                return _parse_local_lexical_response(
                    generated_text,
                    metadata,
                    self.config,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Local MLX lexical parsing failed for '%s' on attempt %s/%s: %s",
                    context_label,
                    attempt_idx,
                    attempts,
                    exc,
                )

        raise RuntimeError(
            f"Local LLM request failed for '{context_label}': {last_error}"
        ) from last_error

    def _generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
    ) -> str:
        if self.mlx_backend == "lm":
            prompt_tokens = self._tokenize_chat_prompt(prompt)
            generation_kwargs = {
                "max_tokens": max_new_tokens,
                "verbose": False,
            }
            generation_kwargs["sampler"] = self._mlx_make_sampler(
                temp=max(0.0, float(self.config.temperature)),
            )
            return str(
                self._mlx_generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt_tokens,
                    **generation_kwargs,
                )
            ).strip()

        formatted_prompt = self._build_vlm_chat_text(prompt)
        generation_kwargs = {
            "max_tokens": max_new_tokens,
            "verbose": False,
        }
        if self.config.temperature > 0:
            generation_kwargs["temperature"] = self.config.temperature
        return str(
            self._mlx_generate(
                self.model,
                self.processor,
                formatted_prompt,
                **generation_kwargs,
            )
        ).strip()

    def _build_chat_text(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"System: {SYSTEM_PROMPT}\nUser: {prompt}\nAssistant:"

    def _tokenize_chat_prompt(self, prompt: str) -> list[int]:
        chat_text = self._build_chat_text(prompt)
        return list(self.tokenizer.encode(chat_text, add_special_tokens=False))

    def _build_vlm_chat_text(self, prompt: str) -> str:
        merged_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
        return self._mlx_vlm_apply_chat_template(
            self.processor,
            self.model.config,
            merged_prompt,
            num_images=0,
        )


class _GitHubModelsEnrichmentClient(_BaseEnrichmentClient):
    def __init__(self, config: CategoryEnrichmentConfig) -> None:
        super().__init__(config)
        if OpenAI is None:
            raise ImportError(
                "The 'openai' package is required for provider 'github-models'."
            )
        if not config.model:
            raise ValueError("--model is required when provider is 'github-models'.")
        env_name = str(config.api_key_env or "GITHUB_TOKEN").strip() or "GITHUB_TOKEN"
        api_key = os.getenv(env_name, "").strip()
        if not api_key:
            raise ValueError(
                f"Environment variable '{env_name}' is required for provider "
                "'github-models'."
            )
        self.client = OpenAI(base_url=config.api_base_url, api_key=api_key)

    def enrich(
        self,
        metadata: dict[str, object],
        parent_to_children: dict[str, list[str]],
    ) -> dict[str, object]:
        prompt = build_enrichment_prompt(metadata, self.config)
        parsed = self._request_json(
            prompt=prompt,
            max_tokens=self.config.request_max_tokens,
            context_label=str(metadata.get("node_key", "<unknown>")),
        )
        if not isinstance(parsed, dict):
            raise ValueError("LLM response did not contain a JSON object.")
        return parsed

    def enrich_many(
        self,
        metadata_batch: list[dict[str, object]],
        parent_to_children: dict[str, list[str]],
    ) -> list[dict[str, object]]:
        if len(metadata_batch) <= 1 or self.config.request_batch_size <= 1:
            return super().enrich_many(metadata_batch, parent_to_children)

        prompt = build_batch_enrichment_prompt(metadata_batch, self.config)
        parsed = self._request_json(
            prompt=prompt,
            max_tokens=self.config.request_max_tokens * len(metadata_batch),
            context_label=(
                f"{metadata_batch[0].get('node_key', '<unknown>')} -> "
                f"{metadata_batch[-1].get('node_key', '<unknown>')}"
            ),
        )
        payloads = _extract_batch_enrichment_payloads(parsed)
        if payloads is None:
            logger.warning(
                "Batch enrichment response malformed for %s nodes. Falling back to single-node requests.",
                len(metadata_batch),
            )
            return super().enrich_many(metadata_batch, parent_to_children)

        payloads_by_node_key = {
            str(item.get("node_key", "")).strip(): item
            for item in payloads
            if isinstance(item, dict)
        }
        if any(str(metadata["node_key"]) not in payloads_by_node_key for metadata in metadata_batch):
            logger.warning(
                "Batch enrichment response incomplete for %s nodes. Falling back to single-node requests.",
                len(metadata_batch),
            )
            return super().enrich_many(metadata_batch, parent_to_children)

        return [
            payloads_by_node_key[str(metadata["node_key"])]
            for metadata in metadata_batch
        ]

    def _request_json(
        self,
        *,
        prompt: str,
        max_tokens: int,
        context_label: str,
    ) -> object:
        rate_limit_attempt = 0
        network_attempt = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                break
            except Exception as exc:
                if _is_daily_rate_limit_error(exc):
                    raise RuntimeError(
                        "LLM daily quota exhausted for this model/account. "
                        "GitHub Models returned a UserByModelByDay limit. "
                        "Retry tomorrow or switch model/provider."
                    ) from exc

                if _is_rate_limit_error(exc):
                    rate_limit_attempt += 1
                    if rate_limit_attempt > self.config.max_retries:
                        logger.exception("Category enrichment LLM call failed: %s", exc)
                        raise RuntimeError(f"LLM request failed: {exc}") from exc
                    wait_seconds = _extract_retry_delay_seconds(
                        exc,
                        default_seconds=self.config.retry_backoff_seconds,
                        max_seconds=self.config.max_retry_delay_seconds,
                    )
                    logger.warning(
                        "Rate limit reached for '%s'. Retry %s/%s in %.1fs.",
                        context_label,
                        rate_limit_attempt,
                        self.config.max_retries,
                        wait_seconds,
                    )
                    time.sleep(wait_seconds)
                    continue

                if _is_connection_error(exc):
                    network_attempt += 1
                    if network_attempt > self.config.network_max_retries:
                        logger.exception("Category enrichment LLM call failed: %s", exc)
                        raise RuntimeError(
                            "LLM connection failed repeatedly. Check DNS/network access to "
                            f"{self.config.api_base_url}."
                        ) from exc
                    wait_seconds = self.config.network_retry_backoff_seconds * network_attempt
                    logger.warning(
                        "Connection error for '%s'. Retry %s/%s in %.1fs.",
                        context_label,
                        network_attempt,
                        self.config.network_max_retries,
                        wait_seconds,
                    )
                    time.sleep(wait_seconds)
                    continue

                logger.exception("Category enrichment LLM call failed: %s", exc)
                raise RuntimeError(f"LLM request failed: {exc}") from exc

        if not response.choices:
            raise ValueError("LLM response does not contain any choices.")
        content = response.choices[0].message.content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                item_type = getattr(item, "type", None)
                item_text = getattr(item, "text", None)
                if item_type == "text" and item_text:
                    text_parts.append(str(item_text))
            content_text = "\n".join(text_parts).strip()
        else:
            content_text = str(content or "").strip()
        return _parse_json_response(content_text)


def build_enrichment_prompt(
    metadata: dict[str, object],
    config: CategoryEnrichmentConfig,
) -> str:
    payload = _build_prompt_payload(metadata, config)
    lexical_count = _target_lexical_variant_count(metadata, config)
    priority_hint = _priority_prompt_hint(metadata, config)
    return (
        "Generate concise English lexical variants for an e-commerce taxonomy node.\n"
        "The output must be valid JSON with exactly this key:\n"
        f"- lexical_variants: array of exactly {lexical_count} short strings\n\n"
        "Rules:\n"
        "- Stay faithful to the taxonomy wording.\n"
        "- Do not invent brand names, prices, or product claims.\n"
        "- Keep the output retrieval-oriented and compact.\n"
        f"- Return exactly {lexical_count} distinct short phrasings.\n"
        "- Use synonyms, near-synonyms, or search expressions when useful.\n"
        "- Use the node label and descendants context when relevant.\n"
        f"{priority_hint}\n"
        f"Node context:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def build_local_lexical_prompt(
    metadata: dict[str, object],
    config: CategoryEnrichmentConfig,
) -> str:
    payload = _build_prompt_payload(metadata, config)
    lexical_count = _target_lexical_variant_count(metadata, config)
    priority_hint = _priority_prompt_hint(metadata, config)
    return (
        "Generate short English lexical variants for this e-commerce taxonomy node.\n"
        f"Return exactly {lexical_count} lines.\n"
        "Return plain text only.\n"
        "One variant per line.\n"
        "No numbering.\n"
        "No bullets.\n"
        "No JSON.\n"
        "No commentary.\n"
        "Stay faithful to the taxonomy wording.\n"
        "Use short retrieval-oriented phrasings, synonyms, or search expressions.\n\n"
        f"{priority_hint}\n"
        f"Node context:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def build_batch_enrichment_prompt(
    metadata_batch: list[dict[str, object]],
    config: CategoryEnrichmentConfig,
) -> str:
    lexical_counts = {
        str(metadata["node_key"]): _target_lexical_variant_count(metadata, config)
        for metadata in metadata_batch
    }
    max_lexical_count = max(lexical_counts.values()) if lexical_counts else config.min_lexical_variants
    payload = {
        "items": [_build_prompt_payload(metadata, config) for metadata in metadata_batch],
        "requested_lexical_variant_counts_by_node_key": lexical_counts,
    }
    return (
        "Generate concise English lexical variants for each e-commerce taxonomy node in the list.\n"
        "The output must be valid JSON with exactly this shape:\n"
        "{\n"
        '  "items": [\n'
        "    {\n"
        '      "node_key": string,\n'
        '      "lexical_variants": array of short strings\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Return exactly one output item for each input node_key.\n"
        "- Do not omit any node_key.\n"
        "- Do not add any extra node_key.\n"
        "- Stay faithful to the taxonomy wording.\n"
        "- Do not invent brand names, prices, or product claims.\n"
        "- For each node_key, use the requested count in requested_lexical_variant_counts_by_node_key.\n"
        f"- The largest requested array in this batch has {max_lexical_count} variants.\n"
        "- Use synonyms, near-synonyms, or search expressions when useful.\n"
        "- Use the node label and descendants context when relevant.\n\n"
        f"Nodes context:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _build_prompt_payload(
    metadata: dict[str, object],
    config: CategoryEnrichmentConfig,
) -> dict[str, object]:
    children = list(metadata["child_names"])[: config.max_children]
    descendants = list(metadata["descendant_names"])[: config.max_descendants]
    return {
        "node_key": metadata["node_key"],
        "category_name": metadata["category_name"],
        "taxonomy_path": metadata["taxonomy_path"],
        "depth": metadata["depth"],
        "parent_name": metadata["parent_name"],
        "children_names": children,
        "descendant_names": descendants,
    }


def _build_enrichment_client(config: CategoryEnrichmentConfig) -> _BaseEnrichmentClient:
    provider = config.provider.strip().lower()
    if provider == "template":
        return _TemplateEnrichmentClient(config)
    if provider in {"local", "huggingface", "huggingface-local"}:
        if not config.model:
            config.model = DEFAULT_MLX_MODEL
        return _LocalEnrichmentClient(config)
    if provider in {"github-models", "openai-compatible"}:
        if not config.model:
            config.model = DEFAULT_GITHUB_MODEL
        return _GitHubModelsEnrichmentClient(config)
    raise ValueError(
        "provider must be 'template', 'local', 'huggingface-local', or 'github-models'."
    )


def _collect_descendant_names(
    *,
    node_key: str,
    nodes: dict[str, dict[str, object]],
    parent_to_children: dict[str, list[str]],
    include_direct_children: bool,
) -> list[str]:
    descendants: list[str] = []
    queue = deque((child_key, 1) for child_key in parent_to_children.get(node_key, []))
    while queue:
        child_key, hop = queue.popleft()
        if include_direct_children or hop > 1:
            descendants.append(str(nodes[child_key]["category_name"]))
        queue.extend((grandchild_key, hop + 1) for grandchild_key in parent_to_children.get(child_key, []))
    return descendants


def load_dotenv_file(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key:
            continue
        os.environ.setdefault(key, value)


def _create_progress_bar(
    *,
    total: int,
    description: str,
):
    if tqdm is None:
        return None
    return tqdm(total=total, desc=description, unit="category")


def _advance_progress_bar(progress_bar, amount: int) -> None:
    if progress_bar is None:
        return
    progress_bar.update(amount)


def _close_progress_bar(progress_bar) -> None:
    if progress_bar is None:
        return
    progress_bar.close()


def _batch_items(
    items: list[dict[str, object]],
    *,
    batch_size: int,
) -> list[list[dict[str, object]]]:
    return [
        items[start_idx : start_idx + batch_size]
        for start_idx in range(0, len(items), batch_size)
    ]


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc)
    lowered = text.lower()
    return "ratelimit" in lowered or "rate limit" in lowered or "429" in lowered


def _is_daily_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "userbymodelbyday" in text or "per 86400s" in text or "86400s exceeded" in text


def _is_connection_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "connection error" in text
        or "connecterror" in text
        or "nodename nor servname provided" in text
        or "name or service not known" in text
        or "temporary failure in name resolution" in text
    )


def _extract_retry_delay_seconds(
    exc: Exception,
    default_seconds: float,
    max_seconds: float,
) -> float:
    text = str(exc)
    second_matches = [
        float(value)
        for value in re.findall(
            r"(\d+(?:\.\d+)?)\s*(?:seconds|second|secs|sec)\b",
            text,
            flags=re.IGNORECASE,
        )
    ]
    millisecond_matches = [
        float(value) / 1000.0
        for value in re.findall(
            r"(\d+(?:\.\d+)?)\s*(?:milliseconds|millisecond|ms)\b",
            text,
            flags=re.IGNORECASE,
        )
    ]

    candidates = [value for value in second_matches + millisecond_matches if value > 0]
    if not candidates:
        return min(default_seconds, max_seconds)

    return min(min(candidates) + 1.0, max_seconds)


def _save_rows(rows: list[dict[str, object]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _replace_rows_for_node_key(
    rows: list[dict[str, object]],
    *,
    node_key: str,
    new_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    filtered_rows = [
        row
        for row in rows
        if str(row.get("node_key", "")).strip() != node_key
    ]
    filtered_rows.extend(new_rows)
    return filtered_rows


def _count_reference_rows_by_node_key(
    rows: list[dict[str, object]],
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        node_key = str(row.get("node_key", "")).strip()
        if not node_key:
            continue
        counts[node_key] += 1
    return dict(counts)


def _extract_expected_reference_counts(
    category_rows: list[dict[str, object]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in category_rows:
        node_key = str(row.get("node_key", "")).strip()
        if not node_key:
            continue
        counts[node_key] = _count_reference_items_from_json_payload(
            row.get("reference_texts_json", ""),
        )
    return counts


def _count_reference_items_from_json_payload(payload: object) -> int:
    normalized = str(payload or "").strip()
    if not normalized or normalized.lower() == "nan":
        return 0
    try:
        parsed = json.loads(normalized)
    except json.JSONDecodeError:
        return 0
    if not isinstance(parsed, dict):
        return 0

    count = 0
    for value in parsed.values():
        if isinstance(value, str):
            if " ".join(value.split()).strip():
                count += 1
            continue
        if isinstance(value, list):
            count += sum(1 for item in value if " ".join(str(item).split()).strip())
    return count


def _extract_batch_enrichment_payloads(payload: object) -> Optional[list[dict[str, object]]]:
    if not isinstance(payload, dict):
        return None
    items = payload.get("items")
    if not isinstance(items, list):
        return None
    if not all(isinstance(item, dict) for item in items):
        return None
    return [dict(item) for item in items]


def _template_description(metadata: dict[str, object]) -> str:
    category_name = str(metadata["category_name"])
    parent_name = str(metadata["parent_name"]).strip()
    child_names = list(metadata["child_names"])
    descendant_names = list(metadata["descendant_names"])
    if parent_name:
        base = (
            f"{category_name} is a taxonomy node under {parent_name}. "
            f"It groups products, accessories, media, services, or subcategories that belong "
            f"to this specific branch of the catalog."
        )
    else:
        base = (
            f"{category_name} is a top-level e-commerce taxonomy branch. "
            f"It defines a broad commercial domain used to organize related product families "
            f"and search intents in the catalog."
        )
    if child_names:
        examples = _human_join(child_names[:4])
        return (
            f"{base} Representative direct subcategories include {examples}. "
            f"This node should match products whose terminology and usage clearly align with "
            f"that branch."
        )
    if descendant_names:
        examples = _human_join(descendant_names[:4])
        return (
            f"{base} Representative examples deeper in the taxonomy include {examples}. "
            f"This node acts as a semantic grouping for related catalog items."
        )
    return (
        f"{base} It represents a specific catalog grouping whose meaning is primarily "
        f"defined by the taxonomy label itself."
    )


def _template_children_summary(metadata: dict[str, object]) -> str:
    child_names = list(metadata["child_names"])
    if not child_names:
        return ""
    if len(child_names) <= 6:
        return (
            f"Direct child categories include {_human_join(child_names)}. "
            f"These children describe the main immediate subdivisions of this node."
        )
    return (
        f"Direct child categories include {_human_join(child_names[:6])}, among "
        f"{len(child_names)} children in total. These children capture the main immediate "
        f"subdivisions of this branch."
    )


def _template_descendants_summary(metadata: dict[str, object]) -> str:
    descendant_names = list(metadata["descendant_names"])
    if not descendant_names:
        return ""
    if len(descendant_names) <= 6:
        return (
            f"Representative deeper descendants include {_human_join(descendant_names)}. "
            f"They illustrate how this branch becomes more specific further down the taxonomy."
        )
    return (
        f"Representative deeper descendants include {_human_join(descendant_names[:6])}, "
        f"with {len(descendant_names)} deeper nodes overall. They illustrate the more "
        f"specific product intents nested under this branch."
    )


def _template_lexical_variants(metadata: dict[str, object]) -> list[str]:
    category_name = str(metadata["category_name"])
    parent_name = str(metadata["parent_name"]).strip()
    return _build_fallback_lexical_variants(metadata, min_count=10)


def _local_description(metadata: dict[str, object]) -> str:
    category_name = str(metadata["category_name"])
    parent_name = str(metadata["parent_name"]).strip()
    child_names = list(metadata["child_names"])
    descendant_names = list(metadata["descendant_names"])
    role_hint = _infer_role_hint(category_name)

    if parent_name:
        opening = f"{category_name} is a category in the {parent_name} branch"
    else:
        opening = f"{category_name} is a top-level branch in the taxonomy"

    scope_hint = f" covering {role_hint}" if role_hint else ""

    if child_names:
        child_examples = _human_join(child_names[:4])
        return (
            f"{opening}{scope_hint}. It is used for products and subcategories such as "
            f"{child_examples}. In product matching, this node should capture items whose "
            f"terminology and intended use clearly belong to this branch rather than a sibling category."
        )

    if descendant_names:
        descendant_examples = _human_join(descendant_names[:4])
        return (
            f"{opening}{scope_hint}. Representative examples deeper in the tree include "
            f"{descendant_examples}. This gives additional semantic context for the kinds "
            f"of products organized under the node."
        )

    if parent_name:
        return (
            f"{opening}{scope_hint}. It represents a specific sellable grouping inside "
            f"{parent_name}. It should be used for products that align closely with the "
            f"meaning of the taxonomy label."
        )
    return f"{opening}{scope_hint}."


def _local_children_summary(metadata: dict[str, object]) -> str:
    child_names = list(metadata["child_names"])
    if not child_names:
        return ""
    if len(child_names) <= 6:
        return f"Direct child categories: {_human_join(child_names)}."
    return (
        f"Direct child categories include {_human_join(child_names[:6])}, among "
        f"{len(child_names)} children in total."
    )


def _local_descendants_summary(metadata: dict[str, object]) -> str:
    descendant_names = list(metadata["descendant_names"])
    if not descendant_names:
        return ""
    if len(descendant_names) <= 6:
        return f"Representative deeper descendants: {_human_join(descendant_names)}."
    return (
        f"Representative deeper descendants include {_human_join(descendant_names[:6])}, "
        f"with {len(descendant_names)} deeper nodes overall."
    )


def _local_lexical_variants(metadata: dict[str, object]) -> list[str]:
    return _build_fallback_lexical_variants(metadata, min_count=10)


def _ensure_min_lexical_variants(
    metadata: dict[str, object],
    lexical_variants: list[str],
    *,
    min_count: int,
) -> list[str]:
    combined = _deduplicate_preserve_order(
        lexical_variants + _build_fallback_lexical_variants(metadata, min_count=min_count)
    )
    return combined[:min_count]


def _target_lexical_variant_count(
    metadata: dict[str, object],
    config: CategoryEnrichmentConfig,
) -> int:
    base_count = max(1, int(config.min_lexical_variants))
    if not _is_priority_category(metadata, config):
        return base_count
    return max(base_count, int(config.priority_min_lexical_variants))


def _is_priority_category(
    metadata: dict[str, object],
    config: CategoryEnrichmentConfig,
) -> bool:
    node_key = _normalize_category_key(metadata.get("node_key", ""))
    category_name = _normalize_category_key(metadata.get("category_name", ""))
    priority_node_keys = {
        _normalize_category_key(item)
        for item in (config.priority_node_keys or [])
        if str(item).strip()
    }
    priority_category_names = {
        _normalize_category_key(item)
        for item in (config.priority_category_names or [])
        if str(item).strip()
    }
    return node_key in priority_node_keys or category_name in priority_category_names


def _normalize_category_key(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _priority_prompt_hint(
    metadata: dict[str, object],
    config: CategoryEnrichmentConfig,
) -> str:
    if not _is_priority_category(metadata, config):
        return ""
    return (
        "- This node is under-represented or previously confused: generate broader but faithful "
        "retrieval phrasings covering common product wording, use cases, and sibling distinctions.\n"
    )


def _build_fallback_lexical_variants(
    metadata: dict[str, object],
    *,
    min_count: int,
) -> list[str]:
    category_name = str(metadata["category_name"]).strip()
    taxonomy_path = str(metadata["taxonomy_path"]).strip()
    parent_name = str(metadata["parent_name"]).strip()
    child_names = list(metadata["child_names"])
    descendant_names = list(metadata["descendant_names"])
    category_name_lower = category_name.lower()
    singular_hint = _singularize_hint(category_name_lower)
    path_slash = taxonomy_path.replace(" > ", " / ")

    variants = [
        category_name,
        category_name_lower,
        taxonomy_path,
        path_slash,
        f"products for {category_name_lower}",
        f"items related to {category_name_lower}",
        f"{category_name_lower} products",
        f"{category_name_lower} items",
        f"catalog category for {category_name_lower}",
        f"{category_name_lower} category",
        f"{category_name_lower} product category",
        f"{category_name_lower} catalog",
        f"search for {category_name_lower}",
    ]
    variants.extend(_domain_specific_lexical_variants(category_name_lower))

    if singular_hint and singular_hint != category_name_lower:
        variants.extend(
            [
                singular_hint,
                f"{singular_hint} products",
                f"{singular_hint} items",
            ]
        )

    if parent_name:
        parent_lower = parent_name.lower()
        variants.extend(
            [
                f"{category_name_lower} in {parent_lower}",
                f"{category_name_lower} under {parent_lower}",
                f"{parent_lower} > {category_name_lower}",
                f"{category_name_lower} within {parent_lower}",
            ]
        )

    if child_names:
        child_preview = _human_join([str(item).lower() for item in child_names[:2]])
        variants.extend(
            [
                f"{category_name_lower} including {child_preview}",
                f"{category_name_lower} such as {child_preview}",
            ]
        )

    if descendant_names:
        descendant_preview = _human_join([str(item).lower() for item in descendant_names[:2]])
        variants.extend(
            [
                f"{category_name_lower} including {descendant_preview}",
                f"{category_name_lower} examples {descendant_preview}",
            ]
        )

    role_hint = _infer_role_hint(category_name)
    if role_hint and role_hint != "a coherent set of related products":
        variants.append(f"{category_name_lower} {role_hint}")

    cleaned = _deduplicate_preserve_order(
        [" ".join(str(item).split()).strip() for item in variants if str(item).strip()]
    )
    return cleaned[:min_count]


def _domain_specific_lexical_variants(category_name_lower: str) -> list[str]:
    variants_by_category = {
        "hardware": [
            "tools and hardware",
            "home improvement hardware",
            "building hardware",
            "fasteners and fittings",
            "screws nails and anchors",
            "plumbing hardware",
            "electrical hardware",
            "cabinet hardware",
            "door hardware",
            "workshop supplies",
            "repair hardware",
            "maintenance tools",
        ],
        "software": [
            "computer software",
            "digital software",
            "software download",
            "software license",
            "productivity software",
            "security software",
            "operating system software",
            "application software",
            "video game software",
            "educational software",
            "business software",
            "software program",
        ],
        "office supplies": [
            "stationery supplies",
            "school and office supplies",
            "paper notebooks and pads",
            "pens pencils and markers",
            "desk organization supplies",
            "filing and folders",
            "printer paper and labels",
            "office stationery",
            "writing supplies",
            "planner and calendar supplies",
            "mailing and shipping supplies",
            "workplace supplies",
        ],
        "vehicles & parts": [
            "auto parts",
            "vehicle replacement parts",
            "car accessories",
            "automotive components",
            "motorcycle parts",
            "truck parts",
            "vehicle maintenance parts",
            "engine parts",
            "brake parts",
            "tires and wheels",
            "vehicle exterior accessories",
            "vehicle interior accessories",
        ],
        "arts & entertainment": [
            "entertainment products",
            "music movies and games",
            "collectibles and memorabilia",
            "party and entertainment items",
            "hobby and craft entertainment",
            "musical instruments",
            "stage and performance items",
            "movie merchandise",
            "fan merchandise",
            "creative arts products",
            "game room entertainment",
            "media and entertainment goods",
        ],
        "business & industrial": [
            "industrial supplies",
            "business equipment",
            "commercial equipment",
            "workplace machinery",
            "industrial tools",
            "professional supplies",
            "manufacturing supplies",
            "warehouse supplies",
            "safety and facility supplies",
            "commercial maintenance equipment",
            "industrial parts",
            "business operations supplies",
        ],
        "religious & ceremonial": [
            "religious items",
            "ceremonial supplies",
            "worship supplies",
            "ritual items",
            "spiritual gifts",
            "religious decor",
            "ceremony accessories",
            "faith based products",
            "memorial ceremony items",
            "religious books and symbols",
        ],
        "mature": [
            "adult products",
            "mature products",
            "adult novelty items",
            "intimate products",
            "adult wellness items",
            "mature audience products",
            "adult personal items",
            "restricted adult items",
        ],
    }
    return variants_by_category.get(category_name_lower, [])


def _singularize_hint(text: str) -> str:
    if text.endswith("ies") and len(text) > 4:
        return f"{text[:-3]}y"
    if text.endswith("sses") or text.endswith("shes") or text.endswith("ches"):
        return text[:-2]
    if text.endswith("s") and not text.endswith("ss") and len(text) > 3:
        return text[:-1]
    return text


def _infer_role_hint(category_name: str) -> str:
    text = category_name.lower()
    if any(token in text for token in {"accessories", "accessory"}):
        return "accessories and complementary items"
    if any(token in text for token in {"supplies", "supply"}):
        return "supplies, consumables, and routine-use items"
    if any(token in text for token in {"apparel", "clothing", "shoes"}):
        return "wearable products and apparel items"
    if any(token in text for token in {"food", "treats", "beverages"}):
        return "food, edible products, or drink-related items"
    if any(token in text for token in {"software", "games", "media", "books", "music"}):
        return "digital media, software, or entertainment-related products"
    if any(token in text for token in {"tools", "equipment", "devices", "instruments", "machinery"}):
        return "equipment, devices, or operational tools"
    if any(token in text for token in {"care", "grooming", "beauty", "health"}):
        return "care, hygiene, beauty, or health-related products"
    if any(token in text for token in {"parts", "components", "hardware"}):
        return "parts, components, or hardware elements"
    if any(token in text for token in {"furniture", "beds", "stands", "storage"}):
        return "furniture, support items, or storage-related products"
    if any(token in text for token in {"tickets", "rental", "services", "service"}):
        return "service-like, booking, or access-related offerings"
    return "a coherent set of related products"


def _human_join(items: list[str]) -> str:
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def _deduplicate_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def prepare_local_model_reference(
    config: CategoryEnrichmentConfig,
) -> str | Path:
    if not config.model:
        raise ValueError("A local MLX model reference is required.")

    model_ref = str(config.model).strip()
    candidate_path = Path(model_ref).expanduser()
    if candidate_path.exists():
        return candidate_path

    local_model_dir = str(config.local_model_dir or "").strip()
    if not local_model_dir:
        return model_ref

    local_path = Path(local_model_dir).expanduser()
    config.local_model_dir = str(local_path)
    if local_path.exists() and (local_path / "config.json").exists():
        logger.info("Using cached local MLX model at '%s'.", local_path)
        return local_path

    logger.info(
        "Downloading MLX model '%s' into '%s'.",
        model_ref,
        local_path,
    )
    snapshot_download(
        repo_id=model_ref,
        revision=config.model_revision,
        local_dir=local_path,
        local_files_only=False,
    )
    return local_path


def _resolve_mlx_backend(
    model_ref: str | Path,
    requested_backend: str,
) -> str:
    normalized = str(requested_backend or "auto").strip().lower()
    if normalized in {"lm", "vlm"}:
        return normalized
    if normalized != "auto":
        raise ValueError("mlx_backend must be one of: auto, lm, vlm.")

    config_path = _resolve_model_config_path(model_ref)
    if config_path is not None:
        try:
            parsed = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict) and "vision_config" in parsed:
            return "vlm"

    ref_text = str(model_ref).lower()
    if any(token in ref_text for token in {"vlm", "vision"}):
        return "vlm"
    return "lm"


def _resolve_model_config_path(model_ref: str | Path) -> Optional[Path]:
    candidate = Path(model_ref).expanduser()
    if candidate.exists():
        config_path = candidate / "config.json"
        if config_path.exists():
            return config_path
    return None


def _extract_json_object_text(text: str) -> Optional[str]:
    normalized = str(text or "").strip()
    if not normalized:
        return None
    start_idx = normalized.find("{")
    if start_idx == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start_idx, len(normalized)):
        char = normalized[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return normalized[start_idx : idx + 1]

    return None


def _parse_local_lexical_response(
    text: str,
    metadata: Optional[dict[str, object]],
    config: CategoryEnrichmentConfig,
) -> dict[str, object]:
    normalized = str(text or "").strip()
    lines = []
    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*[-*•]\s*", "", line)
        line = re.sub(r"^\s*\d+[\).\-\:]\s*", "", line)
        line = line.strip().strip('"').strip("'")
        if not line:
            continue
        lines.append(line)

    fallback_metadata = metadata or {
        "category_name": "",
        "taxonomy_path": "",
        "parent_name": "",
        "child_names": [],
        "descendant_names": [],
    }
    lexical_variants = _deduplicate_preserve_order(lines)
    lexical_variants = _ensure_min_lexical_variants(
        fallback_metadata,
        lexical_variants,
        min_count=_target_lexical_variant_count(fallback_metadata, config),
    )
    return {
        "lexical_variants": lexical_variants,
    }


def _parse_json_response(text: str) -> object:
    normalized = text.strip()
    if normalized.startswith("```"):
        normalized = re.sub(r"^```(?:json)?\s*", "", normalized)
        normalized = re.sub(r"\s*```$", "", normalized)
    try:
        return json.loads(normalized)
    except json.JSONDecodeError:
        repaired = _extract_json_object_text(normalized)
        if repaired is None:
            raise
        return json.loads(repaired)


SYSTEM_PROMPT = (
    "You generate structured taxonomy enrichments for product categorization pipelines. "
    "Return semantically rich, accurate English text grounded in the node name and taxonomy "
    "context, with strong retrieval value and no marketing fluff."
)
