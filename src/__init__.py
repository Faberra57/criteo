"""Utilities for LV2 data preprocessing and embedding fine-tuning."""

from .annotation_tool import (
    AnnotationConfig,
    clear_row_annotation,
    ensure_annotation_columns,
    filter_annotation_indices,
    format_row_for_terminal,
    load_annotation_frame,
    mark_row,
    parse_candidates,
    save_annotation_frame,
)
from .category_enrichment import (
    CategoryEnrichmentConfig,
    generate_category_enrichment,
    load_taxonomy_nodes,
)
from .embedding_model_registry import (
    EmbeddingModelConfig,
    find_catalog_row,
    load_embedding_model_catalog,
    load_embedding_model_config,
    resolve_model_name,
    save_embedding_model_config,
)
from .hierarchical_beam_search import BeamSearchConfig, predict_product_path
from .hierarchical_prototype_search import (
    PrototypeBuildConfig,
    PrototypeIndex,
    build_and_save_prototype_index,
    build_prototype_records,
    load_taxonomy_tree,
    predict_dataframe,
)
from .preprocessing import (
    PreprocessingConfig,
    load_catalog_and_ground_truth,
    merge_catalog_with_ground_truth,
    preprocess_for_nlp,
    save_preprocessed_frame,
)
from .synthetic_generation import (
    SyntheticGenerationConfig,
    generate_bootstrap_artifacts,
    generate_synthetic_l2_dataset,
    load_level2_keywords,
    load_taxonomy_l2_map,
    sample_real_dataset_for_annotation,
)
from .synthetic_validation import (
    SyntheticValidationConfig,
    build_modeling_text,
    evaluate_real_vs_synthetic,
    evaluate_sibling_classification,
    evaluate_synthetic_to_real_transfer,
)
from .triplet_finetuning import (
    TripletTrainingArtifacts,
    TripletTrainingConfig,
    evaluate_embeddings_with_knn,
    prepare_training_frame,
    train_triplet_model,
)

__all__ = [
    "AnnotationConfig",
    "BeamSearchConfig",
    "CategoryEnrichmentConfig",
    "EmbeddingModelConfig",
    "PreprocessingConfig",
    "PrototypeBuildConfig",
    "PrototypeIndex",
    "SyntheticGenerationConfig",
    "SyntheticValidationConfig",
    "build_and_save_prototype_index",
    "build_modeling_text",
    "build_prototype_records",
    "clear_row_annotation",
    "ensure_annotation_columns",
    "filter_annotation_indices",
    "find_catalog_row",
    "format_row_for_terminal",
    "generate_category_enrichment",
    "generate_bootstrap_artifacts",
    "generate_synthetic_l2_dataset",
    "TripletTrainingArtifacts",
    "TripletTrainingConfig",
    "evaluate_embeddings_with_knn",
    "evaluate_real_vs_synthetic",
    "evaluate_sibling_classification",
    "evaluate_synthetic_to_real_transfer",
    "load_annotation_frame",
    "load_embedding_model_catalog",
    "load_embedding_model_config",
    "load_level2_keywords",
    "load_catalog_and_ground_truth",
    "load_taxonomy_nodes",
    "load_taxonomy_l2_map",
    "mark_row",
    "merge_catalog_with_ground_truth",
    "parse_candidates",
    "predict_dataframe",
    "predict_product_path",
    "prepare_training_frame",
    "preprocess_for_nlp",
    "resolve_model_name",
    "sample_real_dataset_for_annotation",
    "save_annotation_frame",
    "save_embedding_model_config",
    "save_preprocessed_frame",
    "train_triplet_model",
    "load_taxonomy_tree",
]
