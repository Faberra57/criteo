from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .hierarchical_prototype_search import PrototypeIndex


@dataclass(slots=True)
class BeamSearchConfig:
    max_depth: int = 4
    beam_width: int = 2
    score_aggregation: str = "mean_top_k"
    top_k_prototypes: int = 3
    ambiguity_margin: float = 0.03
    resolution_margin: float = 0.05
    continue_if_no_ambiguity: bool = True


@dataclass(slots=True)
class SearchState:
    node_key: str
    node_name: str
    depth: int
    cumulative_score: float
    local_score: float
    path_keys: list[str]
    path_names: list[str]
    score_trace: list[float]


def predict_product_path(
    index: "PrototypeIndex",
    query_embedding: np.ndarray,
    *,
    start_level1_name: str,
    search_config: Optional[BeamSearchConfig] = None,
) -> dict[str, object]:
    search_config = search_config or BeamSearchConfig()
    start_parent_key = start_level1_name.strip()
    l2_scores = index.score_children(
        query_embedding,
        start_parent_key,
        aggregation=search_config.score_aggregation,
        top_k=search_config.top_k_prototypes,
    )
    if not l2_scores:
        return {
            "start_level_1": start_level1_name,
            "predicted_path": [start_level1_name],
            "resolved_depth": 1,
            "reason": "no_children_for_level1",
            "level2_candidates": [],
        }

    result = {
        "start_level_1": start_level1_name,
        "level2_candidates": l2_scores[: search_config.beam_width],
        "predicted_path": [start_level1_name, l2_scores[0]["node_name"]],
        "resolved_depth": 2,
        "reason": "top_level2",
    }

    if len(l2_scores) == 1:
        if search_config.continue_if_no_ambiguity:
            greedy = _greedy_descend(
                index=index,
                query_embedding=query_embedding,
                state=_initialize_state(l2_scores[0]),
                search_config=search_config,
            )
            result["predicted_path"] = [start_level1_name] + greedy.path_names
            result["resolved_depth"] = greedy.depth
            result["reason"] = "single_level2_child"
            result["score_trace"] = greedy.score_trace
        return result

    top1, top2 = l2_scores[0], l2_scores[1]
    ambiguous = (top1["score"] - top2["score"]) <= search_config.ambiguity_margin
    if not ambiguous:
        if search_config.continue_if_no_ambiguity:
            greedy = _greedy_descend(
                index=index,
                query_embedding=query_embedding,
                state=_initialize_state(top1),
                search_config=search_config,
            )
            result["predicted_path"] = [start_level1_name] + greedy.path_names
            result["resolved_depth"] = greedy.depth
            result["reason"] = "confident_level2_greedy_descent"
            result["score_trace"] = greedy.score_trace
        return result

    frontier = [_initialize_state(top1), _initialize_state(top2)]
    best_state = frontier[0]
    max_depth = max(2, search_config.max_depth)
    has_expanded = False

    while frontier:
        frontier = sorted(frontier, key=lambda state: state.cumulative_score, reverse=True)
        best_state = frontier[0]
        if len(frontier) == 1:
            break
        margin = frontier[0].cumulative_score - frontier[1].cumulative_score
        if has_expanded and margin >= search_config.resolution_margin:
            break
        if best_state.depth >= max_depth:
            break

        expanded: list[SearchState] = []
        expanded_any = False
        for state in frontier[: search_config.beam_width]:
            child_scores = index.score_children(
                query_embedding,
                state.node_key,
                aggregation=search_config.score_aggregation,
                top_k=search_config.top_k_prototypes,
            )
            if not child_scores:
                expanded.append(state)
                continue
            expanded_any = True
            for child in child_scores[: search_config.beam_width]:
                expanded.append(_extend_state(state, child))
        if not expanded_any or expanded == frontier:
            break
        has_expanded = True
        frontier = expanded[: search_config.beam_width * 2]

    return {
        "start_level_1": start_level1_name,
        "predicted_path": [start_level1_name] + best_state.path_names,
        "resolved_depth": best_state.depth,
        "reason": "beam_search_resolution",
        "score_trace": best_state.score_trace,
        "level2_candidates": l2_scores[: search_config.beam_width],
        "beam_candidates": [
            {
                "path": [start_level1_name] + state.path_names,
                "score_trace": state.score_trace,
                "cumulative_score": state.cumulative_score,
            }
            for state in sorted(frontier, key=lambda state: state.cumulative_score, reverse=True)
        ],
    }


def _initialize_state(score_row: dict[str, object]) -> SearchState:
    score = float(score_row["score"])
    return SearchState(
        node_key=str(score_row["node_key"]),
        node_name=str(score_row["node_name"]),
        depth=int(score_row["depth"]),
        cumulative_score=score,
        local_score=score,
        path_keys=[str(score_row["node_key"])],
        path_names=[str(score_row["node_name"])],
        score_trace=[score],
    )


def _extend_state(state: SearchState, child_row: dict[str, object]) -> SearchState:
    child_score = float(child_row["score"])
    new_trace = state.score_trace + [child_score]
    cumulative = float(np.mean(new_trace))
    return SearchState(
        node_key=str(child_row["node_key"]),
        node_name=str(child_row["node_name"]),
        depth=int(child_row["depth"]),
        cumulative_score=cumulative,
        local_score=child_score,
        path_keys=state.path_keys + [str(child_row["node_key"])],
        path_names=state.path_names + [str(child_row["node_name"])],
        score_trace=new_trace,
    )


def _greedy_descend(
    *,
    index: "PrototypeIndex",
    query_embedding: np.ndarray,
    state: SearchState,
    search_config: BeamSearchConfig,
) -> SearchState:
    current = state
    while current.depth < search_config.max_depth:
        child_scores = index.score_children(
            query_embedding,
            current.node_key,
            aggregation=search_config.score_aggregation,
            top_k=search_config.top_k_prototypes,
        )
        if not child_scores:
            break
        current = _extend_state(current, child_scores[0])
    return current
