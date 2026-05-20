from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import resource
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
import threading
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.losses import BatchHardTripletLossDistanceFunction
from sentence_transformers.util import batch_to_device
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Sampler

print("=" * 50)
print("🔍 VÉRIFICATION DU MATÉRIEL (GPU)")
print("=" * 50)

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"✅ SUCCÈS : GPU(s) détecté(s) ! Nombre total : {num_gpus}")
    for i in range(num_gpus):
        print(f"   -> GPU {i} : {torch.cuda.get_device_name(i)}")
else:
    print("❌ ALERTE : AUCUN GPU DÉTECTÉ. Le code tourne sur CPU !")

print("=" * 50)


@dataclass(slots=True)
class KaggleFineTuneConfig:
    input_path: str = "/kaggle/input/criteo-finetuning/preprocessed_lv2.parquet"
    input_format: str = "parquet"
    categories_path: str = "/kaggle/input/criteo-finetuning/level_1_categories.csv"
    output_root: str = "/kaggle/working/embedding_runs"
    experiment_name: str = "enriched_triplet_finetune"

    model_name: str = "BAAI/bge-small-en-v1.5"
    max_seq_length: int = 256
    normalize_embeddings: bool = True

    text_col: str = "text"
    title_col: str = "title"
    description_col: str = "description"
    brand_col: str = "brand"
    price_col: str = "sale_price"
    path_cols: list[str] = field(default_factory=lambda: ["level_1_name"])

    test_size: float = 0.1
    sample_size: Optional[int] = None
    min_examples_per_label: int = 2

    batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    dataloader_drop_last: bool = True
    train_sampler: str = "tempered"
    balanced_labels_per_batch: int = 8
    balanced_examples_per_label: int = 4
    balanced_batches_per_epoch: Optional[int] = None
    label_sampling_alpha: float = 0.5

    distance_metric: str = "cosine"
    triplet_margin: float = 0.5
    num_validation_triplets: int = 2000

    train_category_prototype_types: list[str] = field(
        default_factory=lambda: [
            "category_name",
            "path_text",
            "parent_context",
            "enriched_description",
            "children_summary",
            "descendants_summary",
            "children_names_text",
            "descendant_names_text",
            "lexical_expansion",
        ]
    )
    eval_category_prototype_types: list[str] = field(
        default_factory=lambda: [
            "category_name",
            "path_text",
            "parent_context",
            "enriched_description",
            "children_summary",
            "descendants_summary",
            "children_names_text",
            "descendant_names_text",
            "lexical_expansion",
        ]
    )

    retrieval_aggregation: str = "max"
    retrieval_mean_top_k: int = 3
    retrieval_top_k: int = 5

    eval_batch_size: int = 64
    log_every_steps: int = 20
    save_every_epoch: bool = True
    random_seed: int = 42
    device: Optional[str] = None
    performance_sample_interval_sec: float = 5.0
    training_cost_per_hour_usd: float = 0.0

    def resolved_device(self) -> str:
        if self.device:
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"


def parse_args() -> KaggleFineTuneConfig:
    # 1. Crée une instance avec les valeurs par défaut
    default_cfg = KaggleFineTuneConfig()

    parser = argparse.ArgumentParser(
        description="Kaggle-ready fine-tuning with enriched category prototypes."
    )

    # 2. Utilise les attributs de l'instance, PAS ceux de la classe
    parser.add_argument("--input-path", default=default_cfg.input_path)
    parser.add_argument("--input-format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--categories-path", default=default_cfg.categories_path)
    parser.add_argument("--output-root", default=default_cfg.output_root)
    parser.add_argument("--experiment-name", default=default_cfg.experiment_name)

    parser.add_argument("--model-name", default=default_cfg.model_name)
    parser.add_argument(
        "--max-seq-length", type=int, default=default_cfg.max_seq_length
    )
    parser.add_argument("--no-normalize-embeddings", action="store_true")

    parser.add_argument("--text-col", default=default_cfg.text_col)
    parser.add_argument("--title-col", default=default_cfg.title_col)
    parser.add_argument("--description-col", default=default_cfg.description_col)
    parser.add_argument("--brand-col", default=default_cfg.brand_col)
    parser.add_argument("--price-col", default=default_cfg.price_col)

    # Voici la correction clé :
    parser.add_argument("--path-cols", nargs="+", default=default_cfg.path_cols)

    parser.add_argument("--test-size", type=float, default=default_cfg.test_size)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument(
        "--min-examples-per-label", type=int, default=default_cfg.min_examples_per_label
    )

    parser.add_argument("--batch-size", type=int, default=default_cfg.batch_size)
    parser.add_argument("--epochs", type=int, default=default_cfg.epochs)
    parser.add_argument(
        "--learning-rate", type=float, default=default_cfg.learning_rate
    )
    parser.add_argument("--weight-decay", type=float, default=default_cfg.weight_decay)
    parser.add_argument("--warmup-ratio", type=float, default=default_cfg.warmup_ratio)
    parser.add_argument("--warmup-steps", type=int, default=default_cfg.warmup_steps)
    parser.add_argument(
        "--max-grad-norm", type=float, default=default_cfg.max_grad_norm
    )
    parser.add_argument("--no-drop-last", action="store_true")
    parser.add_argument(
        "--train-sampler",
        choices=["tempered", "balanced", "shuffle"],
        default=default_cfg.train_sampler,
        help=(
            "'tempered' samples labels with probability proportional to n_c^alpha; "
            "'balanced' samples labels uniformly; "
            "'shuffle' keeps the previous random DataLoader behavior."
        ),
    )
    parser.add_argument(
        "--balanced-labels-per-batch",
        type=int,
        default=default_cfg.balanced_labels_per_batch,
        help="Number of distinct labels P sampled in each balanced batch.",
    )
    parser.add_argument(
        "--balanced-examples-per-label",
        type=int,
        default=default_cfg.balanced_examples_per_label,
        help="Number of examples K sampled per label in each balanced batch.",
    )
    parser.add_argument(
        "--balanced-batches-per-epoch",
        type=int,
        default=None,
        help=(
            "Number of balanced batches per epoch. Defaults to roughly "
            "len(train_examples) / (P*K), preserving the previous epoch cost."
        ),
    )
    parser.add_argument(
        "--label-sampling-alpha",
        type=float,
        default=default_cfg.label_sampling_alpha,
        help=(
            "Only used with --train-sampler tempered. alpha=0 is uniform per label, "
            "alpha=1 approximates sampling labels by their frequency."
        ),
    )

    parser.add_argument(
        "--distance-metric",
        choices=["cosine", "euclidean"],
        default=default_cfg.distance_metric,
    )
    parser.add_argument(
        "--triplet-margin", type=float, default=default_cfg.triplet_margin
    )
    parser.add_argument(
        "--num-validation-triplets",
        type=int,
        default=default_cfg.num_validation_triplets,
    )

    # Autre correction clé :
    parser.add_argument(
        "--train-category-prototype-types",
        nargs="+",
        default=default_cfg.train_category_prototype_types,
    )
    parser.add_argument(
        "--eval-category-prototype-types",
        nargs="+",
        default=default_cfg.eval_category_prototype_types,
    )

    parser.add_argument(
        "--retrieval-aggregation",
        choices=["max", "mean_top_k"],
        default=default_cfg.retrieval_aggregation,
    )
    parser.add_argument(
        "--retrieval-mean-top-k", type=int, default=default_cfg.retrieval_mean_top_k
    )
    parser.add_argument(
        "--retrieval-top-k", type=int, default=default_cfg.retrieval_top_k
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=default_cfg.eval_batch_size
    )
    parser.add_argument(
        "--log-every-steps", type=int, default=default_cfg.log_every_steps
    )
    parser.add_argument("--random-seed", type=int, default=default_cfg.random_seed)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--performance-sample-interval-sec",
        type=float,
        default=default_cfg.performance_sample_interval_sec,
        help="Interval used to sample RAM/GPU metrics during the run. Set <=0 to disable periodic sampling.",
    )
    parser.add_argument(
        "--training-cost-per-hour-usd",
        type=float,
        default=default_cfg.training_cost_per_hour_usd,
        help=(
            "Optional hardware cost assumption in USD/hour. Kaggle free usage can "
            "be left at 0; the script still reports runtime and resource usage."
        ),
    )

    args = parser.parse_args()
    return KaggleFineTuneConfig(
        input_path=args.input_path,
        input_format=args.input_format,
        categories_path=args.categories_path,
        output_root=args.output_root,
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        normalize_embeddings=not args.no_normalize_embeddings,
        text_col=args.text_col,
        title_col=args.title_col,
        description_col=args.description_col,
        brand_col=args.brand_col,
        price_col=args.price_col,
        path_cols=list(args.path_cols),
        test_size=args.test_size,
        sample_size=args.sample_size,
        min_examples_per_label=args.min_examples_per_label,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        dataloader_drop_last=not args.no_drop_last,
        train_sampler=args.train_sampler,
        balanced_labels_per_batch=args.balanced_labels_per_batch,
        balanced_examples_per_label=args.balanced_examples_per_label,
        balanced_batches_per_epoch=args.balanced_batches_per_epoch,
        label_sampling_alpha=args.label_sampling_alpha,
        distance_metric=args.distance_metric,
        triplet_margin=args.triplet_margin,
        num_validation_triplets=args.num_validation_triplets,
        train_category_prototype_types=list(args.train_category_prototype_types),
        eval_category_prototype_types=list(args.eval_category_prototype_types),
        retrieval_aggregation=args.retrieval_aggregation,
        retrieval_mean_top_k=args.retrieval_mean_top_k,
        retrieval_top_k=args.retrieval_top_k,
        eval_batch_size=args.eval_batch_size,
        log_every_steps=args.log_every_steps,
        random_seed=args.random_seed,
        device=args.device,
        performance_sample_interval_sec=args.performance_sample_interval_sec,
        training_cost_per_hour_usd=args.training_cost_per_hour_usd,
    )


def main() -> None:
    config = parse_args()
    configure_runtime(config)
    target_device = torch.device(config.resolved_device())

    run_dir = build_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = run_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    final_model_dir = run_dir / "final_model"
    best_model_dir = run_dir / "best_model"

    save_json(asdict(config), run_dir / "config.json")
    performance_monitor = PerformanceMonitor(
        sample_interval_sec=config.performance_sample_interval_sec,
        device=str(target_device),
    )
    run_started_perf = time.perf_counter()
    performance_monitor.start()

    df = load_input_frame(config.input_path, config.input_format)
    df = prepare_product_frame(df, config)

    train_df, test_df = split_frame(df, config)
    categories_df = load_categories_frame(
        config.categories_path, label_depth=len(config.path_cols)
    )

    train_label_keys = set(train_df["label_key"].unique())
    train_category_rows = build_category_training_rows(
        categories_df=categories_df,
        allowed_node_keys=train_label_keys,
        prototype_types=config.train_category_prototype_types,
    )
    augmented_train_df = pd.concat(
        [train_df, pd.DataFrame(train_category_rows)],
        ignore_index=True,
    )
    performance_monitor.sample("data_prepared")

    label_to_id = build_label_to_id(augmented_train_df["label_key"].tolist())
    train_examples = build_input_examples(augmented_train_df, label_to_id)
    if not train_examples:
        raise ValueError("No training examples available after augmentation.")
    save_training_label_diagnostics(
        augmented_train_df, run_dir / "train_label_counts.csv"
    )

    model = SentenceTransformer(
        config.model_name,
        device=config.resolved_device(),
        trust_remote_code=True,
        model_kwargs={
            "dtype": torch.bfloat16,
        },  # Recommended for GPUs
    )
    model.max_seq_length = config.max_seq_length
    performance_monitor.sample("model_loaded")

    train_loader = build_train_loader(
        train_examples=train_examples,
        model=model,
        config=config,
    )

    distance_function = resolve_triplet_distance(config.distance_metric)
    train_loss = losses.BatchHardTripletLoss(
        model=model,
        distance_metric=distance_function,
    )

    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    total_steps = max(1, len(train_loader) * config.epochs)
    warmup_steps = (
        config.warmup_steps
        if config.warmup_steps > 0
        else max(0, int(total_steps * config.warmup_ratio))
    )
    scheduler = build_linear_scheduler(
        optimizer, total_steps=total_steps, warmup_steps=warmup_steps
    )

    val_triplets = create_validation_triplets(
        test_df,
        num_triplets=config.num_validation_triplets,
        random_seed=config.random_seed,
    )

    step_history: list[dict[str, object]] = []
    epoch_history: list[dict[str, object]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    global_step = 0
    started_at = time.time()
    training_started_perf = time.perf_counter()
    performance_monitor.sample("training_started")

    for epoch in range(1, config.epochs + 1):
        epoch_started_perf = time.perf_counter()
        reset_cuda_peak_memory_stats_safe()
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        running_loss = 0.0
        batch_count = 0

        for batch_idx, (features, labels) in enumerate(train_loader, start=1):
            features = [
                batch_to_device(feature_dict, target_device)
                for feature_dict in features
            ]
            labels = labels.to(target_device)
            optimizer.zero_grad(set_to_none=True)
            loss_value = train_loss(features, labels)
            loss_value.backward()
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            global_step += 1
            batch_count += 1
            running_loss += float(loss_value.detach().cpu().item())

            step_row = {
                "epoch": epoch,
                "global_step": global_step,
                "batch_idx": batch_idx,
                "split": "train",
                "loss": float(loss_value.detach().cpu().item()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            step_history.append(step_row)

            if batch_idx % max(1, config.log_every_steps) == 0:
                print(
                    f"[epoch {epoch}/{config.epochs}] "
                    f"step {batch_idx}/{len(train_loader)} "
                    f"loss={step_row['loss']:.4f} "
                    f"lr={step_row['learning_rate']:.2e}"
                )

        train_epoch_loss = running_loss / max(1, batch_count)
        train_loop_duration_sec = time.perf_counter() - epoch_started_perf
        validation_started_perf = time.perf_counter()
        val_metrics = evaluate_triplet_validation(
            model=model,
            triplets=val_triplets,
            batch_size=config.eval_batch_size,
            normalize_embeddings=config.normalize_embeddings,
            margin=config.triplet_margin,
            device=config.resolved_device(),
        )
        validation_duration_sec = time.perf_counter() - validation_started_perf
        epoch_duration_sec = time.perf_counter() - epoch_started_perf
        epoch_cuda_memory = get_cuda_memory_summary()

        epoch_row = {
            "epoch": epoch,
            "train_loss": float(train_epoch_loss),
            "val_triplet_loss": float(val_metrics["triplet_loss"]),
            "val_triplet_accuracy": float(val_metrics["triplet_accuracy"]),
            "duration_sec": float(time.time() - started_at),
            "elapsed_sec": float(time.perf_counter() - training_started_perf),
            "epoch_duration_sec": float(epoch_duration_sec),
            "train_loop_duration_sec": float(train_loop_duration_sec),
            "validation_duration_sec": float(validation_duration_sec),
            "steps": int(batch_count),
            "samples_seen": int(batch_count * effective_train_batch_size(train_loader, config)),
            "steps_per_sec": float(batch_count / max(epoch_duration_sec, 1e-9)),
            "samples_per_sec": float(
                (batch_count * effective_train_batch_size(train_loader, config))
                / max(epoch_duration_sec, 1e-9)
            ),
            "gpu_memory_allocated_peak_mb": float(
                epoch_cuda_memory["total_max_allocated_mb"]
            ),
            "gpu_memory_reserved_peak_mb": float(
                epoch_cuda_memory["total_max_reserved_mb"]
            ),
            "process_ram_current_mb": float(get_process_ram_mb()),
            "process_ram_peak_mb": float(get_process_peak_ram_mb()),
        }
        epoch_history.append(epoch_row)
        pd.DataFrame(step_history).to_csv(
            run_dir / "train_step_history.csv", index=False
        )
        pd.DataFrame(epoch_history).to_csv(run_dir / "epoch_metrics.csv", index=False)
        performance_monitor.sample(f"epoch_{epoch}_finished")

        if config.save_every_epoch:
            epoch_dir = checkpoint_root / f"epoch_{epoch:02d}"
            model.save(str(epoch_dir))
            torch.save(
                {
                    "epoch": epoch,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": asdict(config),
                    "metrics": epoch_row,
                },
                epoch_dir / "training_state.pt",
            )

        if val_metrics["triplet_loss"] < best_val_loss:
            best_val_loss = float(val_metrics["triplet_loss"])
            best_epoch = epoch
            if best_model_dir.exists():
                shutil.rmtree(best_model_dir)
            model.save(str(best_model_dir))

        print(
            f"[epoch {epoch}/{config.epochs}] "
            f"train_loss={train_epoch_loss:.4f} "
            f"val_loss={val_metrics['triplet_loss']:.4f} "
            f"val_acc={val_metrics['triplet_accuracy']:.4f}"
        )

    model.save(str(final_model_dir))
    training_duration_sec = time.perf_counter() - training_started_perf
    performance_monitor.sample("training_finished")

    best_model_path = best_model_dir if best_model_dir.exists() else final_model_dir
    best_model_load_started_perf = time.perf_counter()
    best_model = SentenceTransformer(
        str(best_model_path),
        device=config.resolved_device(),
        trust_remote_code=True,
        model_kwargs={
            "dtype": torch.bfloat16,
        },  # Recommended for GPUs
    )
    best_model.max_seq_length = config.max_seq_length
    best_model_load_duration_sec = time.perf_counter() - best_model_load_started_perf
    performance_monitor.sample("best_model_loaded")

    eval_prototypes = build_category_prototype_rows(
        categories_df=categories_df,
        allowed_node_keys=set(test_df["label_key"].unique()),
        prototype_types=config.eval_category_prototype_types,
    )
    eval_name_only = build_category_prototype_rows(
        categories_df=categories_df,
        allowed_node_keys=set(test_df["label_key"].unique()),
        prototype_types=["category_name"],
    )

    retrieval_eval_started_perf = time.perf_counter()
    retrieval_metrics = evaluate_retrieval(
        model=best_model,
        test_df=test_df,
        prototype_rows=eval_prototypes,
        config=config,
        output_predictions_path=run_dir / "retrieval_predictions.csv",
    )
    category_name_only_metrics = evaluate_retrieval(
        model=best_model,
        test_df=test_df,
        prototype_rows=eval_name_only,
        config=config,
        output_predictions_path=run_dir
        / "retrieval_predictions_category_name_only.csv",
    )
    retrieval_eval_duration_sec = time.perf_counter() - retrieval_eval_started_perf
    total_runtime_sec = time.perf_counter() - run_started_perf
    performance_monitor.sample("retrieval_evaluation_finished")
    performance_monitor.stop()

    performance_metrics = build_performance_report(
        config=config,
        run_dir=run_dir,
        monitor=performance_monitor,
        total_runtime_sec=total_runtime_sec,
        training_duration_sec=training_duration_sec,
        best_model_load_duration_sec=best_model_load_duration_sec,
        retrieval_eval_duration_sec=retrieval_eval_duration_sec,
        n_train_products=len(train_df),
        n_train_augmented=len(augmented_train_df),
        n_test_products=len(test_df),
        n_category_training_rows=len(train_category_rows),
        n_epochs=config.epochs,
        n_steps=global_step,
        train_batches_per_epoch=len(train_loader),
        effective_batch_size=effective_train_batch_size(train_loader, config),
        epoch_history=epoch_history,
    )
    save_json(performance_metrics, run_dir / "training_performance_metrics.json")

    summary = {
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_val_triplet_loss": best_val_loss,
        "final_model_dir": str(final_model_dir),
        "best_model_dir": str(best_model_path),
        "n_train_products": int(len(train_df)),
        "n_train_augmented": int(len(augmented_train_df)),
        "n_test_products": int(len(test_df)),
        "n_category_training_rows": int(len(train_category_rows)),
        "retrieval_metrics": retrieval_metrics,
        "category_name_only_metrics": category_name_only_metrics,
        "performance_metrics_path": str(run_dir / "training_performance_metrics.json"),
        "performance_metrics": performance_metrics,
    }
    save_json(summary, run_dir / "summary.json")

    print("Training complete")
    print(f"Run dir: {run_dir}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Retrieval top-1 accuracy: {retrieval_metrics['top1_accuracy']:.4f}")
    print(
        f"Category-name-only top-1 accuracy: {category_name_only_metrics['top1_accuracy']:.4f}"
    )


class PerformanceMonitor:
    """Sample process, CUDA and NVML metrics during the run."""

    def __init__(self, *, sample_interval_sec: float, device: str) -> None:
        self.sample_interval_sec = float(sample_interval_sec)
        self.device = device
        self.samples: list[dict[str, object]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started_perf: Optional[float] = None

    def start(self) -> None:
        self._started_perf = time.perf_counter()
        reset_cuda_peak_memory_stats_safe()
        self.sample("monitor_started")
        if self.sample_interval_sec > 0:
            self._thread = threading.Thread(
                target=self._run,
                name="performance-monitor",
                daemon=True,
            )
            self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.wait(self.sample_interval_sec):
            self.sample("periodic")

    def sample(self, label: str) -> None:
        started_perf = self._started_perf or time.perf_counter()
        snapshot = collect_resource_snapshot(label=label, started_perf=started_perf)
        with self._lock:
            self.samples.append(snapshot)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sample_interval_sec + 1.0))
        self.sample("monitor_stopped")

    def summary(self) -> dict[str, object]:
        with self._lock:
            samples = list(self.samples)
        return summarize_resource_samples(samples)


def collect_resource_snapshot(*, label: str, started_perf: float) -> dict[str, object]:
    process_times = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "label": label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": float(time.perf_counter() - started_perf),
        "process_ram_current_mb": float(get_process_ram_mb()),
        "process_ram_peak_mb": float(get_process_peak_ram_mb()),
        "process_cpu_user_sec": float(process_times.ru_utime),
        "process_cpu_system_sec": float(process_times.ru_stime),
        "cuda_memory": get_cuda_memory_summary(),
        "nvml": get_nvml_snapshot(),
    }


def summarize_resource_samples(samples: list[dict[str, object]]) -> dict[str, object]:
    if not samples:
        return {"sample_count": 0}

    ram_current = [
        float(sample["process_ram_current_mb"])
        for sample in samples
        if sample.get("process_ram_current_mb") is not None
    ]
    ram_peak = [
        float(sample["process_ram_peak_mb"])
        for sample in samples
        if sample.get("process_ram_peak_mb") is not None
    ]
    cuda_allocated = []
    cuda_reserved = []
    cuda_max_allocated = []
    cuda_max_reserved = []
    total_power_w = []
    avg_gpu_util = []
    max_gpu_memory_used_mb = []

    for sample in samples:
        cuda_memory = sample.get("cuda_memory")
        if isinstance(cuda_memory, dict):
            cuda_allocated.append(float(cuda_memory.get("total_allocated_mb", 0.0)))
            cuda_reserved.append(float(cuda_memory.get("total_reserved_mb", 0.0)))
            cuda_max_allocated.append(
                float(cuda_memory.get("total_max_allocated_mb", 0.0))
            )
            cuda_max_reserved.append(
                float(cuda_memory.get("total_max_reserved_mb", 0.0))
            )

        nvml = sample.get("nvml")
        if isinstance(nvml, dict) and nvml.get("available"):
            devices = nvml.get("devices", [])
            if isinstance(devices, list) and devices:
                power_values = [
                    float(device["power_draw_w"])
                    for device in devices
                    if isinstance(device, dict)
                    and device.get("power_draw_w") is not None
                ]
                util_values = [
                    float(device["gpu_utilization_pct"])
                    for device in devices
                    if isinstance(device, dict)
                    and device.get("gpu_utilization_pct") is not None
                ]
                memory_values = [
                    float(device["memory_used_mb"])
                    for device in devices
                    if isinstance(device, dict)
                    and device.get("memory_used_mb") is not None
                ]
                if power_values:
                    total_power_w.append(sum(power_values))
                if util_values:
                    avg_gpu_util.append(float(np.mean(util_values)))
                if memory_values:
                    max_gpu_memory_used_mb.append(max(memory_values))

    elapsed_values = [float(sample.get("elapsed_sec", 0.0)) for sample in samples]
    observed_duration_sec = max(elapsed_values) - min(elapsed_values)
    avg_total_power_w = safe_mean(total_power_w)

    return {
        "sample_count": int(len(samples)),
        "observed_duration_sec": float(observed_duration_sec),
        "process_ram_current_max_mb": safe_max(ram_current),
        "process_ram_current_mean_mb": safe_mean(ram_current),
        "process_ram_peak_max_mb": safe_max(ram_peak),
        "cuda_allocated_max_mb": safe_max(cuda_allocated),
        "cuda_reserved_max_mb": safe_max(cuda_reserved),
        "cuda_peak_allocated_max_mb": safe_max(cuda_max_allocated),
        "cuda_peak_reserved_max_mb": safe_max(cuda_max_reserved),
        "nvml_total_gpu_power_draw_mean_w": avg_total_power_w,
        "nvml_total_gpu_power_draw_max_w": safe_max(total_power_w),
        "nvml_gpu_utilization_mean_pct": safe_mean(avg_gpu_util),
        "nvml_gpu_memory_used_max_mb": safe_max(max_gpu_memory_used_mb),
        "estimated_gpu_energy_wh": (
            None
            if avg_total_power_w is None
            else float(avg_total_power_w * observed_duration_sec / 3600.0)
        ),
        "first_sample": samples[0],
        "last_sample": samples[-1],
    }


def build_performance_report(
    *,
    config: KaggleFineTuneConfig,
    run_dir: Path,
    monitor: PerformanceMonitor,
    total_runtime_sec: float,
    training_duration_sec: float,
    best_model_load_duration_sec: float,
    retrieval_eval_duration_sec: float,
    n_train_products: int,
    n_train_augmented: int,
    n_test_products: int,
    n_category_training_rows: int,
    n_epochs: int,
    n_steps: int,
    train_batches_per_epoch: int,
    effective_batch_size: int,
    epoch_history: list[dict[str, object]],
) -> dict[str, object]:
    total_runtime_hours = total_runtime_sec / 3600.0
    training_hours = training_duration_sec / 3600.0
    cost_per_hour = max(0.0, float(config.training_cost_per_hour_usd))
    samples_seen = int(n_steps * effective_batch_size)
    epoch_durations = [
        float(row["epoch_duration_sec"])
        for row in epoch_history
        if "epoch_duration_sec" in row
    ]

    return {
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "hardware": get_hardware_profile(),
        "config_subset": {
            "model_name": config.model_name,
            "device": config.resolved_device(),
            "max_seq_length": int(config.max_seq_length),
            "epochs": int(config.epochs),
            "batch_size": int(config.batch_size),
            "train_sampler": config.train_sampler,
            "balanced_labels_per_batch": int(config.balanced_labels_per_batch),
            "balanced_examples_per_label": int(config.balanced_examples_per_label),
            "label_sampling_alpha": float(config.label_sampling_alpha),
            "learning_rate": float(config.learning_rate),
            "eval_batch_size": int(config.eval_batch_size),
        },
        "dataset": {
            "n_train_products": int(n_train_products),
            "n_train_augmented": int(n_train_augmented),
            "n_test_products": int(n_test_products),
            "n_category_training_rows": int(n_category_training_rows),
        },
        "runtime": {
            "total_runtime_sec": float(total_runtime_sec),
            "training_duration_sec": float(training_duration_sec),
            "best_model_load_duration_sec": float(best_model_load_duration_sec),
            "retrieval_eval_duration_sec": float(retrieval_eval_duration_sec),
            "mean_epoch_duration_sec": safe_mean(epoch_durations),
            "min_epoch_duration_sec": safe_min(epoch_durations),
            "max_epoch_duration_sec": safe_max(epoch_durations),
            "train_batches_per_epoch": int(train_batches_per_epoch),
            "total_train_steps": int(n_steps),
            "effective_batch_size": int(effective_batch_size),
            "estimated_samples_seen": int(samples_seen),
            "train_steps_per_sec": float(n_steps / max(training_duration_sec, 1e-9)),
            "train_samples_per_sec": float(
                samples_seen / max(training_duration_sec, 1e-9)
            ),
            "train_products_per_sec_equivalent": float(
                n_train_products * n_epochs / max(training_duration_sec, 1e-9)
            ),
        },
        "cost": {
            "cost_per_hour_usd": float(cost_per_hour),
            "estimated_total_cost_usd": float(total_runtime_hours * cost_per_hour),
            "estimated_training_cost_usd": float(training_hours * cost_per_hour),
            "cost_model_note": (
                "Kaggle free GPU usage has no direct billed cost. Set "
                "--training-cost-per-hour-usd to report a cloud-equivalent estimate."
            ),
        },
        "resources": monitor.summary(),
        "epoch_history": epoch_history,
    }


def get_hardware_profile() -> dict[str, object]:
    profile: dict[str, object] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": int(torch.cuda.device_count())
        if torch.cuda.is_available()
        else 0,
        "cpu_count": os.cpu_count(),
    }
    if torch.cuda.is_available():
        devices = []
        for idx in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(idx)
            except RuntimeError:
                continue
            devices.append(
                {
                    "index": int(idx),
                    "name": str(props.name),
                    "total_memory_mb": float(props.total_memory / (1024**2)),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": int(props.multi_processor_count),
                }
            )
        profile["cuda_devices"] = devices
    return profile


def get_process_ram_mb() -> float:
    proc_status = Path("/proc/self/status")
    if proc_status.exists():
        for line in proc_status.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1]) / 1024.0

    try:
        import psutil  # type: ignore

        return float(psutil.Process(os.getpid()).memory_info().rss / (1024**2))
    except Exception:
        return get_process_peak_ram_mb()


def get_process_peak_ram_mb() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system().lower() == "darwin":
        return peak / (1024**2)
    return peak / 1024.0


def reset_cuda_peak_memory_stats_safe() -> None:
    if not torch.cuda.is_available():
        return
    for idx in range(torch.cuda.device_count()):
        try:
            torch.cuda.reset_peak_memory_stats(idx)
        except RuntimeError:
            continue


def get_cuda_memory_summary() -> dict[str, object]:
    if not torch.cuda.is_available():
        return {
            "available": False,
            "devices": [],
            "total_allocated_mb": 0.0,
            "total_reserved_mb": 0.0,
            "total_max_allocated_mb": 0.0,
            "total_max_reserved_mb": 0.0,
        }

    devices = []
    for idx in range(torch.cuda.device_count()):
        try:
            allocated = torch.cuda.memory_allocated(idx) / (1024**2)
            reserved = torch.cuda.memory_reserved(idx) / (1024**2)
            max_allocated = torch.cuda.max_memory_allocated(idx) / (1024**2)
            max_reserved = torch.cuda.max_memory_reserved(idx) / (1024**2)
        except RuntimeError:
            continue
        devices.append(
            {
                "index": int(idx),
                "allocated_mb": float(allocated),
                "reserved_mb": float(reserved),
                "max_allocated_mb": float(max_allocated),
                "max_reserved_mb": float(max_reserved),
            }
        )

    return {
        "available": True,
        "devices": devices,
        "total_allocated_mb": float(sum(d["allocated_mb"] for d in devices)),
        "total_reserved_mb": float(sum(d["reserved_mb"] for d in devices)),
        "total_max_allocated_mb": float(sum(d["max_allocated_mb"] for d in devices)),
        "total_max_reserved_mb": float(sum(d["max_reserved_mb"] for d in devices)),
    }


def get_nvml_snapshot() -> dict[str, object]:
    try:
        import pynvml  # type: ignore
    except Exception:
        return get_nvidia_smi_snapshot()

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        devices = []
        for idx in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization_pct = float(utilization.gpu)
                memory_utilization_pct = float(utilization.memory)
            except Exception:
                gpu_utilization_pct = None
                memory_utilization_pct = None
            try:
                power_draw_w = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
            except Exception:
                power_draw_w = None
            raw_name = pynvml.nvmlDeviceGetName(handle)
            devices.append(
                {
                    "index": int(idx),
                    "name": raw_name.decode("utf-8")
                    if isinstance(raw_name, bytes)
                    else str(raw_name),
                    "memory_used_mb": float(memory_info.used / (1024**2)),
                    "memory_total_mb": float(memory_info.total / (1024**2)),
                    "gpu_utilization_pct": gpu_utilization_pct,
                    "memory_utilization_pct": memory_utilization_pct,
                    "power_draw_w": power_draw_w,
                }
            )
        return {"available": True, "devices": devices}
    except Exception as exc:
        fallback = get_nvidia_smi_snapshot()
        if fallback.get("available"):
            return fallback
        return {
            "available": False,
            "reason": f"pynvml_failed: {exc}; {fallback.get('reason')}",
        }
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def get_nvidia_smi_snapshot() -> dict[str, object]:
    query_fields = [
        "index",
        "name",
        "memory.used",
        "memory.total",
        "utilization.gpu",
        "utilization.memory",
        "power.draw",
    ]
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(query_fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except Exception as exc:
        return {"available": False, "reason": f"nvml_and_nvidia_smi_unavailable: {exc}"}

    devices = []
    for raw_line in output.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != len(query_fields):
            continue
        devices.append(
            {
                "index": parse_optional_int(parts[0]),
                "name": parts[1],
                "memory_used_mb": parse_optional_float(parts[2]),
                "memory_total_mb": parse_optional_float(parts[3]),
                "gpu_utilization_pct": parse_optional_float(parts[4]),
                "memory_utilization_pct": parse_optional_float(parts[5]),
                "power_draw_w": parse_optional_float(parts[6]),
            }
        )
    return {"available": bool(devices), "source": "nvidia-smi", "devices": devices}


def parse_optional_float(value: object) -> Optional[float]:
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "NA", "[N/A]"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_optional_int(value: object) -> Optional[int]:
    parsed = parse_optional_float(value)
    return None if parsed is None else int(parsed)


def effective_train_batch_size(
    train_loader: DataLoader, config: KaggleFineTuneConfig
) -> int:
    batch_sampler = getattr(train_loader, "batch_sampler", None)
    sampler_batch_size = getattr(batch_sampler, "batch_size", None)
    if sampler_batch_size is not None:
        return int(sampler_batch_size)
    loader_batch_size = getattr(train_loader, "batch_size", None)
    if loader_batch_size is not None:
        return int(loader_batch_size)
    return int(config.batch_size)


def safe_mean(values: list[float]) -> Optional[float]:
    return None if not values else float(np.mean(values))


def safe_min(values: list[float]) -> Optional[float]:
    return None if not values else float(np.min(values))


def safe_max(values: list[float]) -> Optional[float]:
    return None if not values else float(np.max(values))


def configure_runtime(config: KaggleFineTuneConfig) -> None:
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os_env = {
        "TOKENIZERS_PARALLELISM": "false",
    }
    for key, value in os_env.items():
        if key not in os.environ:
            os.environ[key] = value


def build_run_dir(config: KaggleFineTuneConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = slugify(config.model_name)
    label_slug = slugify("_".join(config.path_cols))
    run_name = (
        f"{timestamp}"
        f"__{slugify(config.experiment_name)}"
        f"__model-{model_slug}"
        f"__label-{label_slug}"
        f"__bs-{config.batch_size}"
        f"__sampler-{slugify(config.train_sampler)}"
        f"__ep-{config.epochs}"
        f"__lr-{format_float(config.learning_rate)}"
        f"__seq-{config.max_seq_length}"
        f"__seed-{config.random_seed}"
    )
    return Path(config.output_root) / run_name


def load_input_frame(path: str, input_format: str) -> pd.DataFrame:
    file_path = Path(path)
    if input_format == "csv":
        return pd.read_csv(file_path)
    return pd.read_parquet(file_path)


def prepare_product_frame(
    df: pd.DataFrame, config: KaggleFineTuneConfig
) -> pd.DataFrame:
    work = df.copy()
    work["text"] = build_text_series(
        work,
        text_col=config.text_col,
        title_col=config.title_col,
        description_col=config.description_col,
        brand_col=config.brand_col,
        price_col=config.price_col,
    )
    work["label_key"] = build_label_keys(work, config.path_cols)
    work = work[(work["text"] != "") & (work["label_key"] != "")].copy()

    counts = work["label_key"].value_counts()
    valid_labels = counts[counts >= config.min_examples_per_label].index
    work = work[work["label_key"].isin(valid_labels)].copy()

    if config.sample_size is not None and len(work) > config.sample_size:
        work = work.sample(config.sample_size, random_state=config.random_seed)

    return work.reset_index(drop=True)


def build_text_series(
    df: pd.DataFrame,
    *,
    text_col: str,
    title_col: str,
    description_col: str,
    brand_col: str,
    price_col: str,
) -> pd.Series:
    if text_col in df.columns:
        return (
            df[text_col]
            .fillna("")
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    work = df.copy()
    for col in (title_col, description_col, brand_col):
        if col not in work.columns:
            work[col] = ""
        work[col] = work[col].fillna("").astype(str).str.strip()

    if price_col not in work.columns:
        work[price_col] = ""
    work[price_col] = pd.to_numeric(work[price_col], errors="coerce")

    text = "Product: " + work[title_col]
    has_desc = work[description_col] != ""
    text = text.where(~has_desc, text + ". Description: " + work[description_col])
    has_brand = work[brand_col] != ""
    text = text.where(~has_brand, text + ". Brand: " + work[brand_col])
    valid_price = work[price_col].notna() & (work[price_col] > 0)
    text = text.where(
        ~valid_price, text + ". Price: " + work[price_col].map(lambda x: f"{x:.2f}")
    )
    return text.str.replace(r"\s+", " ", regex=True).str.strip()


def build_label_keys(df: pd.DataFrame, path_cols: list[str]) -> pd.Series:
    parts = []
    for col in path_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the dataset.")
        parts.append(df[col].fillna("").astype(str).str.strip())

    rows = pd.concat(parts, axis=1)
    return rows.apply(
        lambda row: " > ".join([value for value in row.tolist() if value]), axis=1
    )


def split_frame(
    df: pd.DataFrame, config: KaggleFineTuneConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=df["label_key"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_categories_frame(path: str, *, label_depth: int) -> pd.DataFrame:
    categories = pd.read_csv(path)
    required = {"node_key", "depth", "category_name", "taxonomy_path", "parent_name"}
    missing = sorted(required.difference(categories.columns))
    if missing:
        raise ValueError(f"Missing columns in categories file: {missing}")
    categories = categories[categories["depth"].astype(int) == label_depth].copy()
    categories["node_key"] = categories["node_key"].fillna("").astype(str).str.strip()
    categories["category_name"] = (
        categories["category_name"].fillna("").astype(str).str.strip()
    )
    categories["taxonomy_path"] = (
        categories["taxonomy_path"].fillna("").astype(str).str.strip()
    )
    categories["parent_name"] = (
        categories["parent_name"].fillna("").astype(str).str.strip()
    )
    return categories


def build_category_training_rows(
    *,
    categories_df: pd.DataFrame,
    allowed_node_keys: set[str],
    prototype_types: list[str],
) -> list[dict[str, object]]:
    rows = build_category_prototype_rows(
        categories_df=categories_df,
        allowed_node_keys=allowed_node_keys,
        prototype_types=prototype_types,
    )
    return [
        {
            "text": row["prototype_text"],
            "label_key": row["node_key"],
            "source": f"category_{row['prototype_type']}",
        }
        for row in rows
    ]


def build_category_prototype_rows(
    *,
    categories_df: pd.DataFrame,
    allowed_node_keys: set[str],
    prototype_types: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    requested = set(prototype_types)

    for _, row in categories_df.iterrows():
        node_key = str(row["node_key"]).strip()
        if node_key not in allowed_node_keys:
            continue

        payload = extract_reference_text_payload(row)
        candidates = {
            "category_name": str(row.get("category_name", "")).strip(),
            "path_text": str(row.get("taxonomy_path", "")).strip(),
            "parent_context": (
                f"{str(row.get('category_name', '')).strip()} under {str(row.get('parent_name', '')).strip()}"
                if str(row.get("parent_name", "")).strip()
                else str(row.get("category_name", "")).strip()
            ),
            "enriched_description": str(row.get("enriched_description", "")).strip(),
            "children_summary": str(row.get("children_summary", "")).strip(),
            "descendants_summary": str(row.get("descendants_summary", "")).strip(),
            "children_names_text": str(payload.get("children_names_text", "")).strip(),
            "descendant_names_text": str(
                payload.get("descendant_names_text", "")
            ).strip(),
        }

        lexical_variants = []
        lexical_json = row.get("lexical_variants_json", "")
        if isinstance(lexical_json, str) and lexical_json.strip():
            try:
                parsed_lexical = json.loads(lexical_json)
                if isinstance(parsed_lexical, list):
                    lexical_variants.extend(
                        [
                            str(item).strip()
                            for item in parsed_lexical
                            if str(item).strip()
                        ]
                    )
            except json.JSONDecodeError:
                pass
        lexical_variants.extend(
            [
                str(item).strip()
                for item in payload.get("lexical_expansion", [])
                if str(item).strip()
            ]
        )
        lexical_variants = deduplicate(lexical_variants)

        for prototype_type in prototype_types:
            if prototype_type == "lexical_expansion":
                for variant in lexical_variants:
                    rows.append(
                        {
                            "node_key": node_key,
                            "category_name": str(row.get("category_name", "")).strip(),
                            "prototype_type": prototype_type,
                            "prototype_text": variant,
                        }
                    )
                continue

            text_value = candidates.get(prototype_type, "")
            if not text_value or prototype_type not in requested:
                continue
            rows.append(
                {
                    "node_key": node_key,
                    "category_name": str(row.get("category_name", "")).strip(),
                    "prototype_type": prototype_type,
                    "prototype_text": text_value,
                }
            )

    unique_rows = []
    seen = set()
    for row in rows:
        key = (row["node_key"], row["prototype_type"], row["prototype_text"])
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows


def extract_reference_text_payload(row: pd.Series) -> dict[str, object]:
    raw_payload = row.get("reference_texts_json", "")
    if not isinstance(raw_payload, str) or not raw_payload.strip():
        return {}
    try:
        parsed = json.loads(raw_payload)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_label_to_id(labels: list[str]) -> dict[str, int]:
    unique = sorted({str(label).strip() for label in labels if str(label).strip()})
    return {label: idx for idx, label in enumerate(unique)}


def build_input_examples(
    df: pd.DataFrame, label_to_id: dict[str, int]
) -> list[InputExample]:
    examples: list[InputExample] = []
    for row in df.to_dict("records"):
        text = str(row.get("text", "")).strip()
        label_key = str(row.get("label_key", "")).strip()
        if not text or not label_key or label_key not in label_to_id:
            continue
        examples.append(InputExample(texts=[text], label=int(label_to_id[label_key])))
    return examples


def save_training_label_diagnostics(df: pd.DataFrame, output_path: Path) -> None:
    group_cols = ["label_key"]
    if "source" in df.columns:
        work = df.copy()
        work["source"] = work["source"].fillna("product").astype(str)
        counts = (
            work.groupby(group_cols + ["source"])
            .size()
            .reset_index(name="count")
            .sort_values(["label_key", "source"])
        )
        pivot = counts.pivot_table(
            index="label_key",
            columns="source",
            values="count",
            fill_value=0,
            aggfunc="sum",
        ).reset_index()
        numeric_cols = [col for col in pivot.columns if col != "label_key"]
        pivot["total"] = pivot[numeric_cols].sum(axis=1)
        pivot.sort_values(["total", "label_key"], inplace=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pivot.to_csv(output_path, index=False)
        return

    counts = (
        df.groupby("label_key")
        .size()
        .reset_index(name="total")
        .sort_values(["total", "label_key"])
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(output_path, index=False)


class BalancedLabelBatchSampler(Sampler[list[int]]):
    """Sample P labels and K examples per label for metric-learning batches.

    label_sampling_alpha controls how aggressively rare labels are oversampled:
    alpha=0.0 is uniform over labels, alpha=1.0 follows label frequency.
    """

    def __init__(
        self,
        labels: list[int],
        *,
        labels_per_batch: int,
        examples_per_label: int,
        batches_per_epoch: int,
        random_seed: int,
        label_sampling_alpha: float = 0.0,
    ) -> None:
        if labels_per_batch < 2:
            raise ValueError("balanced_labels_per_batch must be >= 2.")
        if examples_per_label < 2:
            raise ValueError(
                "balanced_examples_per_label must be >= 2 for triplet loss."
            )
        if batches_per_epoch < 1:
            raise ValueError("balanced_batches_per_epoch must be >= 1.")
        if label_sampling_alpha < 0:
            raise ValueError("label_sampling_alpha must be >= 0.")

        self.labels_per_batch = labels_per_batch
        self.examples_per_label = examples_per_label
        self.batches_per_epoch = batches_per_epoch
        self.random_seed = random_seed
        self.label_sampling_alpha = label_sampling_alpha
        self._iteration = 0

        self.indices_by_label: dict[int, list[int]] = {}
        for index, label in enumerate(labels):
            self.indices_by_label.setdefault(int(label), []).append(index)
        self.unique_labels = sorted(self.indices_by_label)
        if len(self.unique_labels) < 2:
            raise ValueError("BalancedLabelBatchSampler needs at least two labels.")
        self.label_weights = {
            label: float(len(self.indices_by_label[label]) ** self.label_sampling_alpha)
            for label in self.unique_labels
        }

    def __iter__(self):
        rng = random.Random(self.random_seed + self._iteration)
        self._iteration += 1
        label_pool = list(self.unique_labels)
        labels_per_batch = min(self.labels_per_batch, len(label_pool))

        for _ in range(self.batches_per_epoch):
            selected_labels = weighted_sample_without_replacement(
                rng=rng,
                items=label_pool,
                weights=[self.label_weights[label] for label in label_pool],
                k=labels_per_batch,
            )
            batch_indices: list[int] = []
            for label in selected_labels:
                label_indices = self.indices_by_label[label]
                if len(label_indices) >= self.examples_per_label:
                    batch_indices.extend(
                        rng.sample(label_indices, self.examples_per_label)
                    )
                else:
                    # Keep rare labels trainable: duplicate with replacement if needed.
                    batch_indices.extend(
                        rng.choice(label_indices)
                        for _ in range(self.examples_per_label)
                    )
            rng.shuffle(batch_indices)
            yield batch_indices

    def __len__(self) -> int:
        return self.batches_per_epoch

    @property
    def batch_size(self) -> int:
        return (
            min(self.labels_per_batch, len(self.unique_labels))
            * self.examples_per_label
        )


def weighted_sample_without_replacement(
    *,
    rng: random.Random,
    items: list[int],
    weights: list[float],
    k: int,
) -> list[int]:
    if k >= len(items):
        return list(items)

    remaining_items = list(items)
    remaining_weights = [max(0.0, float(weight)) for weight in weights]
    selected: list[int] = []
    for _ in range(k):
        total_weight = sum(remaining_weights)
        if total_weight <= 0:
            selected_index = rng.randrange(len(remaining_items))
        else:
            threshold = rng.random() * total_weight
            cumulative = 0.0
            selected_index = len(remaining_items) - 1
            for index, weight in enumerate(remaining_weights):
                cumulative += weight
                if cumulative >= threshold:
                    selected_index = index
                    break
        selected.append(remaining_items.pop(selected_index))
        remaining_weights.pop(selected_index)
    return selected


def build_train_loader(
    *,
    train_examples: list[InputExample],
    model: SentenceTransformer,
    config: KaggleFineTuneConfig,
) -> DataLoader:
    if config.train_sampler == "shuffle":
        return DataLoader(
            train_examples,
            shuffle=True,
            batch_size=config.batch_size,
            drop_last=config.dataloader_drop_last,
            collate_fn=model.smart_batching_collate,
        )

    labels = [int(example.label) for example in train_examples]
    balanced_batch_size = int(config.balanced_labels_per_batch) * int(
        config.balanced_examples_per_label
    )
    if balanced_batch_size != int(config.batch_size):
        print(
            "Balanced sampler ignores --batch-size directly: "
            f"P*K={balanced_batch_size} while batch_size={config.batch_size}."
        )
    default_batches_per_epoch = max(
        1, len(train_examples) // max(1, balanced_batch_size)
    )
    batches_per_epoch = (
        int(config.balanced_batches_per_epoch)
        if config.balanced_batches_per_epoch is not None
        else default_batches_per_epoch
    )
    batch_sampler = BalancedLabelBatchSampler(
        labels,
        labels_per_batch=int(config.balanced_labels_per_batch),
        examples_per_label=int(config.balanced_examples_per_label),
        batches_per_epoch=batches_per_epoch,
        random_seed=int(config.random_seed),
        label_sampling_alpha=(
            float(config.label_sampling_alpha)
            if config.train_sampler == "tempered"
            else 0.0
        ),
    )
    print(
        f"Using {config.train_sampler} batch sampler: "
        f"P={config.balanced_labels_per_batch}, "
        f"K={config.balanced_examples_per_label}, "
        f"effective_batch_size={batch_sampler.batch_size}, "
        f"batches_per_epoch={len(batch_sampler)}, "
        f"label_sampling_alpha={batch_sampler.label_sampling_alpha:.3f}"
    )
    return DataLoader(
        train_examples,
        batch_sampler=batch_sampler,
        collate_fn=model.smart_batching_collate,
    )


def resolve_triplet_distance(metric_name: str):
    normalized = metric_name.lower()
    if normalized == "cosine":
        return BatchHardTripletLossDistanceFunction.cosine_distance
    if normalized in {"euclidean", "eucledian"}:
        return BatchHardTripletLossDistanceFunction.eucledian_distance
    raise ValueError("distance_metric must be 'cosine' or 'euclidean'.")


def build_linear_scheduler(
    optimizer: AdamW, *, total_steps: int, warmup_steps: int
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        remaining_steps = total_steps - current_step
        decay_steps = max(1, total_steps - warmup_steps)
        return max(0.0, float(remaining_steps) / float(decay_steps))

    return LambdaLR(optimizer, lr_lambda)


def create_validation_triplets(
    df: pd.DataFrame,
    *,
    num_triplets: int,
    random_seed: int,
) -> list[tuple[str, str, str]]:
    rng = random.Random(random_seed)
    grouped = df.groupby("label_key")["text"].apply(list).to_dict()
    valid_labels = [label for label, texts in grouped.items() if len(texts) >= 2]
    if len(valid_labels) < 2:
        raise ValueError(
            "Need at least two labels with two examples each for validation triplets."
        )

    triplets: list[tuple[str, str, str]] = []
    for _ in range(num_triplets):
        positive_label = rng.choice(valid_labels)
        negative_label = rng.choice(
            [label for label in valid_labels if label != positive_label]
        )
        anchor_text, positive_text = rng.sample(grouped[positive_label], 2)
        negative_text = rng.choice(grouped[negative_label])
        triplets.append((str(anchor_text), str(positive_text), str(negative_text)))
    return triplets


def evaluate_triplet_validation(
    *,
    model: SentenceTransformer,
    triplets: list[tuple[str, str, str]],
    batch_size: int,
    normalize_embeddings: bool,
    margin: float,
    device: str,
) -> dict[str, float]:
    anchors = [triplet[0] for triplet in triplets]
    positives = [triplet[1] for triplet in triplets]
    negatives = [triplet[2] for triplet in triplets]

    anchor_embeddings = model.encode(
        anchors,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
        device=device,
        task="retrieval",
    )
    positive_embeddings = model.encode(
        positives,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
        device=device,
        task="retrieval",
    )
    negative_embeddings = model.encode(
        negatives,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
        device=device,
        task="retrieval",
    )

    sim_ap = np.sum(anchor_embeddings * positive_embeddings, axis=1)
    sim_an = np.sum(anchor_embeddings * negative_embeddings, axis=1)
    distance_ap = 1.0 - sim_ap
    distance_an = 1.0 - sim_an
    losses_arr = np.maximum(0.0, distance_ap - distance_an + margin)

    return {
        "triplet_loss": float(losses_arr.mean()),
        "triplet_accuracy": float(np.mean(sim_ap > sim_an)),
    }


def evaluate_retrieval(
    *,
    model: SentenceTransformer,
    test_df: pd.DataFrame,
    prototype_rows: list[dict[str, object]],
    config: KaggleFineTuneConfig,
    output_predictions_path: Path,
) -> dict[str, object]:
    if not prototype_rows:
        raise ValueError("No category prototypes available for retrieval evaluation.")

    prototype_df = pd.DataFrame(prototype_rows)
    prototype_texts = prototype_df["prototype_text"].tolist()
    test_texts = test_df["text"].tolist()

    prototype_embeddings = model.encode(
        prototype_texts,
        batch_size=config.eval_batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=config.normalize_embeddings,
        device=config.resolved_device(),
        task="retrieval",
    ).astype(np.float32)
    query_embeddings = model.encode(
        test_texts,
        batch_size=config.eval_batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=config.normalize_embeddings,
        device=config.resolved_device(),
        task="retrieval",
    ).astype(np.float32)

    node_to_indices = (
        prototype_df.reset_index().groupby("node_key")["index"].apply(list).to_dict()
    )

    predictions = []
    top1 = 0
    top3 = 0
    top5 = 0

    for row, embedding in zip(
        test_df.to_dict("records"), query_embeddings, strict=True
    ):
        node_scores = []
        for node_key, indices in node_to_indices.items():
            sims = prototype_embeddings[np.asarray(indices)] @ embedding
            if config.retrieval_aggregation == "max":
                score = float(np.max(sims))
            else:
                selected = np.sort(sims)[::-1][: max(1, config.retrieval_mean_top_k)]
                score = float(np.mean(selected))
            node_scores.append((node_key, score))
        node_scores.sort(key=lambda item: item[1], reverse=True)

        ranked_keys = [item[0] for item in node_scores[: config.retrieval_top_k]]
        true_key = str(row["label_key"]).strip()
        top1 += int(len(ranked_keys) >= 1 and ranked_keys[0] == true_key)
        top3 += int(true_key in ranked_keys[:3])
        top5 += int(true_key in ranked_keys[:5])

        predictions.append(
            {
                "text": row["text"],
                "label_key": true_key,
                "predicted_node_key": ranked_keys[0] if ranked_keys else "",
                "top_k_predictions_json": json.dumps(
                    [
                        {"node_key": node_key, "score": score}
                        for node_key, score in node_scores[: config.retrieval_top_k]
                    ],
                    ensure_ascii=False,
                ),
            }
        )

    pd.DataFrame(predictions).to_csv(output_predictions_path, index=False)
    n = max(1, len(predictions))
    return {
        "n_samples": int(len(predictions)),
        "n_prototypes": int(len(prototype_rows)),
        "n_categories": int(len(node_to_indices)),
        "top1_accuracy": float(top1 / n),
        "top3_accuracy": float(top3 / n),
        "top5_accuracy": float(top5 / n),
    }


def save_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def deduplicate(items: list[str]) -> list[str]:
    seen = set()
    results = []
    for item in items:
        normalized = " ".join(str(item).split()).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        results.append(normalized)
    return results


def slugify(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", str(text).strip().lower())
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    return normalized or "value"


def format_float(value: float) -> str:
    return f"{value:.0e}" if value < 1e-3 else f"{value:.4f}".rstrip("0").rstrip(".")


if __name__ == "__main__":
    main()
