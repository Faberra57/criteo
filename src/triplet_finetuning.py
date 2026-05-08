from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import BatchHardTripletLossDistanceFunction
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from .embedding_model_registry import resolve_model_name


@dataclass(slots=True)
class TripletTrainingConfig:
    model_name: str = "level1"
    output_dir: str = "./artifacts/triplet_model"
    checkpoint_dir: str = "./artifacts/triplet_checkpoint"
    log_file: str = "./artifacts/experiments_log.json"
    text_col: str = "text"
    label_col: str = "level_1_name"
    encoded_label_col: str = "label_encoded"
    test_size: float = 0.1
    sample_size: Optional[int] = 20_000
    batch_size: int = 8
    epochs: int = 1
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    evaluation_steps: int = 500
    max_seq_length: int = 128
    num_validation_triplets: int = 300
    distance_metric: str = "cosine"
    knn_metric: str = "cosine"
    knn_neighbors: int = 5
    eval_sample_size: int = 2_000
    dataloader_drop_last: bool = True
    random_seed: int = 42
    save_best_model: bool = True
    show_progress_bar: bool = True
    device: Optional[str] = None

    def resolved_device(self) -> str:
        if self.device:
            return self.device
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"


@dataclass(slots=True)
class TripletTrainingArtifacts:
    model_path: str
    checkpoint_dir: str
    label_encoder: LabelEncoder
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    final_eval_score: Optional[float]
    training_duration_sec: float


def prepare_training_frame(
    df: pd.DataFrame,
    *,
    label_col: str = "level_1_name",
    encoded_label_col: str = "label_encoded",
) -> tuple[pd.DataFrame, LabelEncoder]:
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' is missing from the training frame.")

    data = df.dropna(subset=[label_col]).copy()
    label_encoder = LabelEncoder()
    data[encoded_label_col] = label_encoder.fit_transform(data[label_col])
    return data, label_encoder


def train_triplet_model(
    df: pd.DataFrame,
    config: Optional[TripletTrainingConfig] = None,
) -> TripletTrainingArtifacts:
    config = config or TripletTrainingConfig()
    _configure_runtime(config)
    config.model_name = resolve_model_name(config.model_name, task="level1")

    prepared_df, label_encoder = prepare_training_frame(
        df,
        label_col=config.label_col,
        encoded_label_col=config.encoded_label_col,
    )

    train_df, test_df = train_test_split(
        prepared_df,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=prepared_df[config.encoded_label_col],
    )

    if config.sample_size is not None and len(train_df) > config.sample_size:
        train_df = train_df.sample(config.sample_size, random_state=config.random_seed)

    train_examples = [
        InputExample(
            texts=[str(row[config.text_col]).strip()],
            label=int(row[config.encoded_label_col]),
        )
        for _, row in train_df.iterrows()
        if str(row[config.text_col]).strip()
    ]
    if not train_examples:
        raise ValueError(
            "No valid training examples were created from the input dataframe."
        )

    train_loader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=config.batch_size,
        drop_last=config.dataloader_drop_last,
    )

    model = SentenceTransformer(config.model_name, device=config.resolved_device())
    model.max_seq_length = config.max_seq_length

    distance_function = _resolve_triplet_distance(config.distance_metric)
    train_loss = losses.BatchHardTripletLoss(
        model=model, distance_metric=distance_function
    )

    anchors, positives, negatives = create_validation_triplets(
        test_df,
        text_col=config.text_col,
        label_col=config.label_col,
        num_triplets=config.num_validation_triplets,
        random_seed=config.random_seed,
    )
    evaluator = TripletEvaluator(anchors, positives, negatives, name="validation")

    checkpoint_dir = Path(config.checkpoint_dir)
    output_dir = Path(config.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=config.epochs,
        warmup_steps=config.warmup_steps,
        evaluator=evaluator,
        evaluation_steps=config.evaluation_steps,
        output_path=str(checkpoint_dir),
        save_best_model=config.save_best_model,
        show_progress_bar=config.show_progress_bar,
        optimizer_params={"lr": config.learning_rate},
    )

    final_eval_score = evaluator(model, output_path=str(checkpoint_dir))
    duration = time.time() - start_time
    model.save(str(output_dir))

    _append_experiment_log(
        config=config,
        final_eval_score=final_eval_score,
        training_duration_sec=duration,
    )

    return TripletTrainingArtifacts(
        model_path=str(output_dir),
        checkpoint_dir=str(checkpoint_dir),
        label_encoder=label_encoder,
        train_df=train_df,
        test_df=test_df,
        final_eval_score=final_eval_score,
        training_duration_sec=duration,
    )


def create_validation_triplets(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    label_col: str = "level_1_name",
    num_triplets: int = 300,
    random_seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    rng = random.Random(random_seed)
    grouped = df.groupby(label_col)[text_col].apply(list).to_dict()
    valid_labels = [label for label, texts in grouped.items() if len(texts) >= 2]
    if len(valid_labels) < 2:
        raise ValueError(
            "At least two classes with two examples each are required for triplet validation."
        )

    anchors: list[str] = []
    positives: list[str] = []
    negatives: list[str] = []

    for _ in range(num_triplets):
        positive_label = rng.choice(valid_labels)
        negative_label = rng.choice(
            [label for label in valid_labels if label != positive_label]
        )
        anchor_text, positive_text = rng.sample(grouped[positive_label], 2)
        negative_text = rng.choice(grouped[negative_label])
        anchors.append(str(anchor_text))
        positives.append(str(positive_text))
        negatives.append(str(negative_text))

    return anchors, positives, negatives


def evaluate_embeddings_with_knn(
    model_path: str | Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    text_col: str = "text",
    encoded_label_col: str = "label_encoded",
    batch_size: int = 64,
    knn_neighbors: int = 5,
    metric: str = "cosine",
    max_seq_length: int = 128,
    device: Optional[str] = None,
    show_progress_bar: bool = True,
    label_encoder: Optional[LabelEncoder] = None,
) -> dict[str, object]:
    resolved_device = device or TripletTrainingConfig().resolved_device()
    model = SentenceTransformer(str(model_path), device=resolved_device)
    model.max_seq_length = max_seq_length

    x_train = model.encode(
        train_df[text_col].tolist(),
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        device=resolved_device,
    )
    x_test = model.encode(
        test_df[text_col].tolist(),
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        device=resolved_device,
    )

    y_train = train_df[encoded_label_col].to_numpy()
    y_test = test_df[encoded_label_col].to_numpy()

    knn = KNeighborsClassifier(
        n_neighbors=knn_neighbors, metric=metric, weights="distance"
    )
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)

    results: dict[str, object] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "y_true": y_test,
        "y_pred": y_pred,
    }
    if label_encoder is not None:
        unique_classes = np.unique(np.concatenate((y_test, y_pred)))
        results["classification_report"] = classification_report(
            y_test,
            y_pred,
            labels=unique_classes,
            target_names=label_encoder.inverse_transform(unique_classes),
        )
    return results


def _configure_runtime(config: TripletTrainingConfig) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if config.resolved_device() == "mps":
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        torch.mps.empty_cache()
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)


def _resolve_triplet_distance(metric_name: str):
    normalized = metric_name.lower()
    if normalized == "cosine":
        return BatchHardTripletLossDistanceFunction.cosine_distance
    if normalized in {"euclidean", "eucledian"}:
        return BatchHardTripletLossDistanceFunction.eucledian
    raise ValueError("distance_metric must be 'cosine' or 'euclidean'.")


def _append_experiment_log(
    *,
    config: TripletTrainingConfig,
    final_eval_score: Optional[float],
    training_duration_sec: float,
) -> None:
    log_path = Path(config.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hyperparameters": asdict(config),
        "results": {
            "final_eval_score": final_eval_score,
            "training_duration_sec": training_duration_sec,
        },
    }

    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as handle:
            logs = json.load(handle)
    else:
        logs = []

    logs.append(payload)
    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(logs, handle, indent=2)
