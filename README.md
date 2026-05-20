# Criteo Taxonomy Classification

Hierarchical product classification into the Google Product Taxonomy, developed as part of an ENSAE x Criteo project.

The objective was to build a pipeline able to categorize a large-scale e-commerce catalog under strict inference-cost, memory, and compute constraints.

## Overview

This project tackles a real industrial problem: assigning each product to a taxonomy path in a deep hierarchy with more than 5,000 categories, while reliable labels are only available for the first level.

I designed the pipeline around three main components:
- semantic category enrichment through multi-prototype texts;
- supervised `Level 1` fine-tuning with a compact embedding model;
- zero-shot `Levels 2–7` prediction with several unsupervised pipelines compared through a dedicated evaluation framework.

The work was not limited to training a model. It involved building a complete methodology: enriched category generation, model selection under VRAM constraints, metric design without deep ground truth, inference-cost benchmarking, and comparison of competing pipelines.

## Problem

The product catalog contains short, noisy, and heterogeneous texts: incomplete titles, uneven descriptions, business-specific vocabulary, and lexical ambiguity.

The main difficulty comes from the structure of the task:
- classification is hierarchical, not flat;
- an error at `Level 1` sends the product into the wrong subtree;
- `L2–L7` do not have reliable labels, so standard accuracy cannot be used as the primary evaluation metric.

This required:
- strong supervised performance at `Level 1`;
- a credible zero-shot strategy for deeper levels;
- a low-cost solution compatible with large-scale industrial inference.

## What I Built

### 1. Category Enrichment with Multi-Prototype Texts

I introduced a category enrichment step based on a simple idea:
a category is not well represented by its canonical name alone.

Each taxonomy node is enriched with several textual views:
- category name;
- taxonomy path;
- parent / child context;
- semantically close lexical variants;
- LLM-generated enriched descriptions.

This multi-prototype representation improves semantic coverage in the embedding space and makes product-category matching more robust.

### 2. Supervised Level-1 Classifier

For `Level 1`, I selected `BAAI/bge-small-en-v1.5`, a compact model with roughly 33M parameters and 384-dimensional embeddings.

The model was fine-tuned with `Batch Hard Triplet Loss` to learn a metric space aligned with the taxonomy.

I evaluated several sampling strategies to handle the strong class imbalance:
- standard sampling;
- balanced sampling;
- tempered sampling with different `alpha` values.

The final trade-off retained was tempered sampling with `alpha = 0.5`, which improved rare-category behavior without significantly hurting global performance.

### 3. Zero-Shot L2–L7 Research

For deeper levels, I explored several unsupervised pipelines based on embeddings and enriched categories.

#### Pipeline 1 — Local Hierarchical Decoding

Level-by-level decoding inside the selected taxonomy branch:
- `greedy`;
- `beam search`;
- `beam + reranker`.

#### Pipeline 2 — Hierarchical Clustering

Recursive clustering of products within a parent branch, followed by cluster-to-child-category assignment through similarity to enriched prototypes.

#### Pipeline 3 — Global Path Retrieval

Direct retrieval of complete taxonomy paths `L2–L7`, with optional reranking of candidate paths.

## Methodological Contribution

One of the most interesting aspects of the project was not only the modeling, but the evaluation without deep labels.

I defined an `unsupervised assessment` framework based on three metric families:

### Inference Cost

- total runtime;
- time per product;
- throughput;
- memory usage;
- number of similarity operations;
- reranking cost.

### Internal Confidence

- predicted path score;
- top-1 / top-2 margin;
- resolved depth;
- path diversity;
- prediction entropy.

### Agreement Between Pipelines

- exact path agreement;
- level-by-level agreement;
- divergence as depth increases;
- categorical association measured with `Cramér's V`.

I complemented this with an external NLP audit using:
- a `bi-encoder`;
- a `cross-encoder` reranker.

This made it possible to compare pipelines even without annotated `L2–L7` accuracy.

## Main Results

### Level 1

The retained `Level 1` model reaches approximately:
- `90.6%` top-1 accuracy;
- `95.5%` top-3 accuracy;
- `97.1%` top-5 accuracy;
- `90.9%` weighted F1.

These results make `Level 1` robust enough to serve as the entry point to the deeper pipeline.

### Levels 2–7

For deep-level embeddings, `Qwen3-Embedding-0.6B` was retained as the best practical compromise.

Why this choice:
- good qualitative performance;
- lower vector dimension than Jasper in the experiments;
- lower vector-memory and similarity-computation cost;
- better fit for large-scale inference pipelines.

### Operational Recommendation

Two final variants were retained:

- `L1 + greedy`: simple, fast, and low-cost;
- `L1 + beam + global path + selective rerank`: more robust, reserved for ambiguous cases.

The final operational idea is not to pay the reranking cost on the whole catalog, but only on products where pipelines disagree or remain uncertain.

## Engineering Work Delivered

Beyond the methodological research, I structured the project into a reusable engineering workflow:
- preprocessing scripts;
- enriched category generation;
- Kaggle GPU fine-tuning;
- shared embedding export;
- separate inference pipelines;
- end-to-end benchmarking;
- analysis and visualization scripts;
- full LaTeX report;
- defense slides.

I also reorganized the repository to clearly separate:
- active scripts;
- shared logic;
- historical archives;
- handover documentation.

## Tech Stack

- `Python`
- `PyTorch`
- `sentence-transformers`
- `transformers`
- `scikit-learn`
- `pandas`
- `NumPy`
- `Matplotlib`
- `UMAP`
- `uv`
- `LaTeX`
- `Kaggle GPU`
- `MLX` for local generation on Apple Silicon

## Repository Guide

### Main Scripts

- `scripts/preprocess_catalog.py`
- `scripts/generate_category_enrichment.py`
- `scripts/predict_l1_finetuned.py`
- `scripts/benchmark_final_l1_l7_pipelines.py`
- `scripts/kaggle/train_finetune_with_enriched_categories.py`
- `scripts/kaggle/export_shared_embeddings_kaggle.py`
- `scripts/kaggle/predict_pipeline1_hierarchical_kaggle.py`
- `scripts/kaggle/predict_pipeline2_hierarchical_clustering_kaggle.py`
- `scripts/kaggle/predict_pipeline3_global_path_kaggle.py`
- `scripts/kaggle/audit_l2_l7_nlp_kaggle.py`

### Documentation

- project handover: [docs/PROJECT_HANDOVER.md](docs/PROJECT_HANDOVER.md)
- repository layout: [docs/REPOSITORY_LAYOUT.md](docs/REPOSITORY_LAYOUT.md)

## Installation

Base environment:

```bash
uv sync
```

With local MLX support:

```bash
uv sync --extra local-mlx
```

With development extras:

```bash
uv sync --extra dev
```

To generate a lockfile on a machine with PyPI access:

```bash
UV_CACHE_DIR=.uv-cache uv lock
```

Note: the Kaggle CUDA enrichment script can use `bitsandbytes` for 4-bit quantization. It is not pinned in this repository because it strongly depends on the target CUDA image.

## Why This Project Matters

This project highlights several skills that are valuable in applied ML and ML engineering:
- framing an industrial problem under real constraints;
- designing evaluation without full supervision;
- balancing modeling quality, inference cost, and deployment feasibility;
- delivering a complete applied research workflow, from experimentation to tooling to reporting.

It also reflects a pragmatic mindset: instead of searching for a single “magic model”, I built a modular, measurable, and defensible pipeline adapted to a constrained production context.
