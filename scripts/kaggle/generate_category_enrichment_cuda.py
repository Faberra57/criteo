from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lib import category_enrichment_core as ce  # noqa: E402


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


class CudaTransformersEnrichmentClient(ce._BaseEnrichmentClient):
    """Local Hugging Face generation client for Kaggle CUDA GPUs.

    The rest of the enrichment pipeline is kept in scripts.lib.category_enrichment_core:
    taxonomy loading, resume logic, normalization, CSV export and reference texts.
    """

    def __init__(
        self,
        config: ce.CategoryEnrichmentConfig,
        *,
        dtype: str,
        device_map: str,
        load_in_4bit: bool,
        load_in_8bit: bool,
        max_memory: Optional[str],
    ) -> None:
        super().__init__(config)
        if not config.model:
            raise ValueError("--model is required for CUDA generation.")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Kaggle CUDA generation requires transformers and torch. "
                "Install them or enable an image that includes them."
            ) from exc

        self.torch = torch
        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.AutoTokenizer = AutoTokenizer
        self.dtype = resolve_torch_dtype(torch, dtype)
        self.device_map = device_map
        self.max_memory = parse_max_memory(max_memory)

        tokenizer_kwargs = {
            "revision": config.model_revision,
            "trust_remote_code": config.trust_remote_code,
        }
        tokenizer_kwargs = {k: v for k, v in tokenizer_kwargs.items() if v is not None}
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, **tokenizer_kwargs)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "revision": config.model_revision,
            "trust_remote_code": config.trust_remote_code,
            "device_map": self.device_map,
            "torch_dtype": self.dtype,
        }
        if self.max_memory:
            model_kwargs["max_memory"] = self.max_memory
        quantization_config = build_quantization_config(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            torch_dtype=self.dtype,
        )
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        self.model = AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)
        self.model.eval()

    def enrich(
        self,
        metadata: dict[str, object],
        parent_to_children: dict[str, list[str]],
    ) -> dict[str, object]:
        prompt = ce.build_local_lexical_prompt(metadata, self.config)
        generated_text = self._generate_many([prompt], max_new_tokens=self.config.request_max_tokens)[0]
        return ce._parse_local_lexical_response(generated_text, metadata, self.config)

    def enrich_many(
        self,
        metadata_batch: list[dict[str, object]],
        parent_to_children: dict[str, list[str]],
    ) -> list[dict[str, object]]:
        if len(metadata_batch) <= 1:
            return super().enrich_many(metadata_batch, parent_to_children)

        prompts = [
            ce.build_local_lexical_prompt(metadata, self.config)
            for metadata in metadata_batch
        ]
        generated_texts = self._generate_many(
            prompts,
            max_new_tokens=self.config.request_max_tokens,
        )
        payloads = []
        for metadata, generated_text in zip(metadata_batch, generated_texts, strict=True):
            payloads.append(
                ce._parse_local_lexical_response(generated_text, metadata, self.config)
            )
        return payloads

    def _generate_many(self, prompts: list[str], *, max_new_tokens: int) -> list[str]:
        chat_texts = [self._build_chat_text(prompt) for prompt in prompts]
        encoded = self.tokenizer(
            chat_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        prompt_length = int(encoded["input_ids"].shape[1])

        do_sample = float(self.config.temperature) > 0
        generation_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = max(1e-5, float(self.config.temperature))

        with self.torch.inference_mode():
            output_ids = self.model.generate(**encoded, **generation_kwargs)
        generated_ids = output_ids[:, prompt_length:]
        return [
            self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in generated_ids
        ]

    def _build_chat_text(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": ce.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"System: {ce.SYSTEM_PROMPT}\nUser: {prompt}\nAssistant:"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate enriched category CSVs on Kaggle CUDA GPUs. "
            "This is the CUDA/Transformers equivalent of scripts/generate_category_enrichment.py."
        )
    )
    parser.add_argument(
        "--taxonomy-path",
        default="/kaggle/input/criteo-finetuning/taxonomy.txt",
    )
    parser.add_argument(
        "--output-dir",
        default="/kaggle/working/category_enrichment_cuda",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Hugging Face causal LM used for lexical variant generation.",
    )
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=None,
        help="Levels to generate, for example: --levels 1 2 3 4 5 6 7.",
    )
    parser.add_argument("--max-children", type=int, default=12)
    parser.add_argument("--max-descendants", type=int, default=24)
    parser.add_argument("--min-lexical-variants", type=int, default=8)
    parser.add_argument(
        "--priority-category-names",
        nargs="*",
        default=DEFAULT_PRIORITY_CATEGORY_NAMES,
        help="Category names that receive more lexical variants.",
    )
    parser.add_argument("--priority-node-keys", nargs="*", default=None)
    parser.add_argument("--priority-min-lexical-variants", type=int, default=32)
    parser.add_argument("--request-max-tokens", type=int, default=320)
    parser.add_argument(
        "--request-batch-size",
        type=int,
        default=8,
        help=(
            "Number of taxonomy nodes generated per forward pass. "
            "Increase on A100/L4, decrease on T4 if CUDA OOM."
        ),
    )
    parser.add_argument("--save-every-n", type=int, default=25)
    parser.add_argument("--limit-per-level", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Use 'auto' to shard the model over the 2 Kaggle GPUs.",
    )
    parser.add_argument(
        "--max-memory",
        default=None,
        help=(
            "Optional per-GPU max memory, e.g. '0:14GiB,1:14GiB'. "
            "Leave empty to let accelerate infer it."
        ),
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load model with bitsandbytes 4-bit quantization. Recommended on Kaggle T4.",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Alternative to --load-in-4bit. Do not use both.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load taxonomy and print counts without loading the model.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Use either --load-in-4bit or --load-in-8bit, not both.")

    config = ce.CategoryEnrichmentConfig(
        taxonomy_path=args.taxonomy_path,
        output_dir=args.output_dir,
        provider="cuda-transformers",
        model=args.model,
        model_revision=args.model_revision,
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
        save_every_n=args.save_every_n,
        limit_per_level=args.limit_per_level,
        overwrite=args.overwrite,
        sleep_seconds=args.sleep_seconds,
    )

    nodes, _ = ce.load_taxonomy_nodes(config.taxonomy_path)
    levels = sorted(config.levels or {int(metadata["depth"]) for metadata in nodes.values()})
    print("Kaggle CUDA category enrichment")
    print(f"Taxonomy nodes: {len(nodes):,}")
    print(f"Levels: {levels}")
    print(f"Model: {config.model}")
    print(f"Output dir: {config.output_dir}")
    if args.dry_run:
        return

    original_builder = ce._build_enrichment_client

    def build_cuda_client(config_to_build: ce.CategoryEnrichmentConfig):
        provider = config_to_build.provider.strip().lower()
        if provider == "cuda-transformers":
            return CudaTransformersEnrichmentClient(
                config_to_build,
                dtype=args.dtype,
                device_map=args.device_map,
                load_in_4bit=bool(args.load_in_4bit),
                load_in_8bit=bool(args.load_in_8bit),
                max_memory=args.max_memory,
            )
        return original_builder(config_to_build)

    ce._build_enrichment_client = build_cuda_client
    try:
        outputs = ce.generate_category_enrichment(config)
    finally:
        ce._build_enrichment_client = original_builder

    print("Category enrichment generation complete")
    for level, paths in sorted(outputs.items()):
        print(f"Level {level}:")
        print(f"  categories: {paths['categories_csv']}")
        print(f"  reference_texts: {paths['reference_texts_csv']}")


def build_quantization_config(
    *,
    load_in_4bit: bool,
    load_in_8bit: bool,
    torch_dtype,
):
    if not load_in_4bit and not load_in_8bit:
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(
            "bitsandbytes quantization requires a recent transformers install."
        ) from exc
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def resolve_torch_dtype(torch_module, dtype_name: str):
    if dtype_name == "float16":
        return torch_module.float16
    if dtype_name == "bfloat16":
        return torch_module.bfloat16
    if dtype_name == "float32":
        return torch_module.float32
    return torch_module.float16


def parse_max_memory(value: Optional[str]) -> Optional[dict[int | str, str]]:
    if not value:
        return None
    result: dict[int | str, str] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        key, memory = item.split(":", maxsplit=1)
        key = key.strip()
        parsed_key: int | str = int(key) if key.isdigit() else key
        result[parsed_key] = memory.strip()
    return result


if __name__ == "__main__":
    main()
