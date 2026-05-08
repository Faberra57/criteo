#!/usr/bin/env python3
"""Prepare compact Parquet inputs for kaggle/audit_l2_l7_nlp_kaggle.py."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dataset/kaggle_l2_l7_nlp_audit_inputs"

QWEN = {
    "pipeline1_hierarchical_predictions_greedy": ROOT / "data/qwen/pipeline1_hierarchical_predictions_greedy.csv",
    "pipeline1_hierarchical_predictions_beam": ROOT / "data/qwen/pipeline1_hierarchical_predictions_beam.csv",
    "pipeline1_hierarchical_predictions_beam-rerank": ROOT / "data/qwen/pipeline1_hierarchical_predictions_beam-rerank.csv",
    "pipeline2_hierarchical_clustering_predictions_l1routed_hungarian": ROOT / "data/qwen/pipeline2_hierarchical_clustering_predictions_l1routed_hungarian.csv",
    "pipeline3_global_path_predictions": ROOT / "data/qwen/pipeline3_global_path_predictions.csv",
}

JASPER = {
    "pipeline1_hierarchical_predictions_jasper_greedy": ROOT / "data/jasper/pipeline1_hierarchical_predictions_jasper_greedy.csv",
    "pipeline1_hierarchical_predictions_jasper_beam": ROOT / "data/jasper/pipeline1_hierarchical_predictions_jasper_beam.csv",
    "pipeline1_hierarchical_predictions_beam-rerank_jasper": ROOT / "data/jasper/pipeline1_hierarchical_predictions_beam-rerank_jasper.csv",
    "pipeline2_hierarchical_clustering_predictions_jasper_l1routed_hungarian": ROOT / "data/jasper/pipeline2_hierarchical_clustering_predictions_jasper_l1routed_hungarian.csv",
    "pipeline3_global_path_predictions_jasper": ROOT / "data/jasper/pipeline3_global_path_predictions_jasper.csv",
}

PROTOTYPES = {
    "qwen": ROOT / "dataset/shared_l1_l7_embeddings_qwen/category_prototypes.parquet",
    "jasper": ROOT / "dataset/shared_l1_l7_embeddings_Jasper-Token-Compression-600M/category_prototypes.parquet",
}

KEEP_COLUMNS = {
    "hashed_external_id",
    "text",
    "pipeline_mode",
    "predicted_taxonomy_path",
    "predicted_taxonomy_key_path",
    "resolved_depth",
    "predicted_level_1_name",
    "predicted_level_1_key",
}
for depth in range(2, 8):
    KEEP_COLUMNS.add(f"predicted_level_{depth}_name")
    KEEP_COLUMNS.add(f"predicted_level_{depth}_key")


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "description": "Compact Parquet inputs for kaggle/audit_l2_l7_nlp_kaggle.py",
        "keep_columns": sorted(KEEP_COLUMNS),
        "groups": {},
    }
    for group, files in [("qwen", QWEN), ("jasper", JASPER)]:
        group_dir = OUT / group
        group_dir.mkdir(parents=True, exist_ok=True)
        manifest["groups"][group] = {"predictions": {}, "category_prototypes": None}
        for name, src in files.items():
            if not src.exists():
                print(f"WARNING missing {src}")
                continue
            dst = group_dir / f"{name}.parquet"
            stats = convert_prediction_csv(src, dst)
            manifest["groups"][group]["predictions"][dst.name] = stats
        proto_src = PROTOTYPES[group]
        proto_dst = group_dir / "category_prototypes.parquet"
        if proto_src.exists():
            shutil.copy2(proto_src, proto_dst)
            manifest["groups"][group]["category_prototypes"] = {
                "path": str(proto_dst.relative_to(OUT)),
                "bytes": proto_dst.stat().st_size,
            }
        else:
            print(f"WARNING missing prototypes {proto_src}")

    readme = OUT / "README.md"
    readme.write_text(build_readme(), encoding="utf-8")
    manifest["elapsed_seconds"] = time.perf_counter() - started
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Prepared Kaggle NLP audit inputs in {OUT}")
    print_size_summary(OUT)


def convert_prediction_csv(src: Path, dst: Path) -> dict[str, object]:
    started = time.perf_counter()
    header = pd.read_csv(src, nrows=0)
    columns = [col for col in sorted(KEEP_COLUMNS) if col in header.columns]
    missing = sorted(KEEP_COLUMNS.difference(header.columns))
    print(f"Converting {src.name} -> {dst.name} ({len(columns)} columns)")
    df = pd.read_csv(src, usecols=columns, low_memory=False)
    if "hashed_external_id" in df.columns:
        df["hashed_external_id"] = df["hashed_external_id"].astype(str)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("").astype(str)
    df.to_parquet(dst, index=False, compression="zstd")
    return {
        "source": str(src),
        "output": str(dst),
        "source_bytes": src.stat().st_size,
        "output_bytes": dst.stat().st_size,
        "rows": int(len(df)),
        "columns": columns,
        "missing_columns": missing,
        "elapsed_seconds": time.perf_counter() - started,
    }


def build_readme() -> str:
    return """# Kaggle L2-L7 NLP Audit Inputs

Compact Parquet files for `kaggle/audit_l2_l7_nlp_kaggle.py`.

Usage on Kaggle for Qwen:

```bash
python /kaggle/input/<code-dataset>/audit_l2_l7_nlp_kaggle.py \
  --preset qwen \
  --predictions-dir /kaggle/input/<this-dataset>/qwen \
  --category-prototypes-path /kaggle/input/<this-dataset>/qwen/category_prototypes.parquet \
  --scorer both \
  --device cuda \
  --output-dir /kaggle/working/l2_l7_nlp_audit_qwen
```

Usage on Kaggle for Jasper:

```bash
python /kaggle/input/<code-dataset>/audit_l2_l7_nlp_kaggle.py \
  --preset jasper \
  --predictions-dir /kaggle/input/<this-dataset>/jasper \
  --category-prototypes-path /kaggle/input/<this-dataset>/jasper/category_prototypes.parquet \
  --scorer both \
  --device cuda \
  --output-dir /kaggle/working/l2_l7_nlp_audit_jasper
```

The files keep only the columns needed to build `(product text, predicted enriched path)` pairs and run the audit.
"""


def print_size_summary(path: Path) -> None:
    for item in [path, path / "qwen", path / "jasper"]:
        total = sum(p.stat().st_size for p in item.rglob("*") if p.is_file())
        print(f"{item}: {total / (1024 ** 2):.1f} MB")


if __name__ == "__main__":
    main()
