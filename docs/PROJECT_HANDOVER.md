# Project Handover

This document is the operational starting point for anyone continuing the project without going back through the full experimentation log.

## 1. Objective

Classify a product catalog into the Google Product Taxonomy:
- `Level 1` with supervision;
- `Levels 2–7` without reliable labels, through zero-shot pipelines.

## 2. Current State

The final pipeline relies on:
- category enrichment through multi-prototype texts;
- a fine-tuned `Level 1` model based on `BAAI/bge-small-en-v1.5`;
- `Qwen3-Embedding-0.6B` embeddings for `L2–L7`;
- several unsupervised pipelines compared through cost, internal confidence, inter-pipeline agreement, and NLP audits.

## 3. Operational Pipeline

| Step | Main script | Role |
|---|---|---|
| Preprocessing | `scripts/preprocess_catalog.py` | builds the final dataset |
| Enrichment | `scripts/generate_category_enrichment.py` | generates enriched categories locally |
| Kaggle enrichment | `scripts/kaggle/generate_category_enrichment_cuda.py` | CUDA variant |
| L1 fine-tuning | `scripts/kaggle/train_finetune_with_enriched_categories.py` | trains the `Level 1` model |
| L1 inference | `scripts/predict_l1_finetuned.py` | predicts the top-level category |
| Shared L2–L7 embeddings | `scripts/kaggle/export_shared_embeddings_kaggle.py` | encodes products, prototypes, and global paths |
| Pipeline 1 | `scripts/kaggle/predict_pipeline1_hierarchical_kaggle.py` | `greedy`, `beam`, `beam-rerank` |
| Pipeline 2 | `scripts/kaggle/predict_pipeline2_hierarchical_clustering_kaggle.py` | hierarchical clustering |
| Pipeline 3 | `scripts/kaggle/predict_pipeline3_global_path_kaggle.py` | global path retrieval |
| Family 3 metrics | `scripts/export_l2_l7_family3_metrics.py` | agreements, disagreements, `Cramér's V` |
| NLP audit | `scripts/kaggle/audit_l2_l7_nlp_kaggle.py` | bi-encoder / cross-encoder audit |
| Final benchmark | `scripts/benchmark_final_l1_l7_pipelines.py` | end-to-end comparison of the two final pipelines |

## 4. Retained Configurations

### Level 1

- model: `BAAI/bge-small-en-v1.5`
- max sequence length: `256`
- retained sampler: `tempered`
- `alpha = 0.5`

### Levels 2–7

- retained embedder: `Qwen3-Embedding-0.6B`
- lightweight pipeline: `L1 + greedy`
- robust pipeline: `L1 + beam + global path + selective rerank`

## 5. Important Outputs

| Directory / file | Content |
|---|---|
| `dataset/preprocessed_lv2.parquet` | dataset after preprocessing |
| `dataset/category_enrichment/` | enriched categories |
| `models/embedding_runs/` | `Level 1` fine-tuning runs |
| `dataset/shared_l1_l7_embeddings_qwen/` | shared Qwen embeddings |
| `data/qwen/` | L2–L7 outputs with Qwen |
| `data/jasper/` | L2–L7 outputs with Jasper |
| `data/final_l1_l7_pipeline_benchmark/` | final benchmark outputs |
| `latex/rapport_final/figures/` | figures used in the final report |

## 6. Shared Logic

The reusable logic still needed by the final pipeline now lives in:
- `scripts/lib/preprocessing_core.py`
- `scripts/lib/category_enrichment_core.py`

The original `src/` package is no longer required to run the active pipeline; it has been archived under `legacy/src_archive/`.

## 7. Archived Material

Historical and exploratory material has been moved to:
- `legacy/scripts/`
- `legacy/kaggle/`
- `legacy/src_archive/`

These files are kept only as historical reference.

## 8. Recommended Restart Order

Suggested order for continuing the project:

1. read `README.md`
2. verify local data paths
3. identify the retained `Level 1` run in `models/embedding_runs/`
4. verify shared embeddings in `dataset/shared_l1_l7_embeddings_qwen/`
5. rerun `scripts/kaggle/predict_pipeline*.py` if needed
6. recompute metrics and audits
7. rerun the final benchmark

## 9. Attention Points

- default paths assume the current repository structure;
- Kaggle scripts use `/kaggle/input/...` and `/kaggle/working/...`;
- `bitsandbytes` is not pinned in `pyproject.toml` because it depends on the target CUDA environment;
- `data/`, `dataset/`, `models/`, and `artifacts/` are intentionally excluded from Git.
