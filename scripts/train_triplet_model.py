from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.triplet_finetuning import (  # noqa: E402
    TripletTrainingConfig,
    evaluate_embeddings_with_knn,
    train_triplet_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune sentence embeddings with triplet loss.")
    parser.add_argument("--input-path", default="data/preprocessed_lv2.parquet")
    parser.add_argument("--input-format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="level_1_name")
    parser.add_argument(
        "--model-name",
        default="level1",
        help="Explicit model id or one of: level1, level2, default.",
    )
    parser.add_argument("--output-dir", default="./artifacts/triplet_model")
    parser.add_argument("--checkpoint-dir", default="./artifacts/triplet_checkpoint")
    parser.add_argument("--log-file", default="./artifacts/experiments_log.json")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--evaluation-steps", type=int, default=500)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--num-validation-triplets", type=int, default=300)
    parser.add_argument("--distance-metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--knn-metric", choices=["cosine", "euclidean", "minkowski"], default="cosine")
    parser.add_argument("--knn-neighbors", type=int, default=5)
    parser.add_argument("--eval-sample-size", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-progress-bar", action="store_true")
    return parser


def load_input_frame(input_path: str, input_format: str) -> pd.DataFrame:
    if input_format == "csv":
        return pd.read_csv(input_path)
    return pd.read_parquet(input_path)


def main() -> None:
    args = build_parser().parse_args()
    df = load_input_frame(args.input_path, args.input_format)

    config = TripletTrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_file=args.log_file,
        text_col=args.text_col,
        label_col=args.label_col,
        test_size=args.test_size,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        evaluation_steps=args.evaluation_steps,
        max_seq_length=args.max_seq_length,
        num_validation_triplets=args.num_validation_triplets,
        distance_metric=args.distance_metric,
        knn_metric=args.knn_metric,
        knn_neighbors=args.knn_neighbors,
        eval_sample_size=args.eval_sample_size,
        random_seed=args.random_seed,
        show_progress_bar=not args.no_progress_bar,
        device=args.device,
    )

    artifacts = train_triplet_model(df, config=config)
    eval_df = artifacts.test_df.sample(
        min(config.eval_sample_size, len(artifacts.test_df)),
        random_state=config.random_seed,
    )
    knn_results = evaluate_embeddings_with_knn(
        model_path=artifacts.model_path,
        train_df=artifacts.train_df,
        test_df=eval_df,
        text_col=config.text_col,
        encoded_label_col=config.encoded_label_col,
        batch_size=config.batch_size,
        knn_neighbors=config.knn_neighbors,
        metric=config.knn_metric,
        max_seq_length=config.max_seq_length,
        device=config.resolved_device(),
        show_progress_bar=config.show_progress_bar,
        label_encoder=artifacts.label_encoder,
    )

    print(f"Model used: {config.model_name}")
    print(f"Model saved to {artifacts.model_path}")
    print(f"Validation triplet score: {artifacts.final_eval_score}")
    print(f"k-NN accuracy: {knn_results['accuracy']:.4f}")
    if "classification_report" in knn_results:
        print(knn_results["classification_report"])


if __name__ == "__main__":
    main()
