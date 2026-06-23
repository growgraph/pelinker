"""Fusion helpers and leaderboard logic for model-selection runs."""

from __future__ import annotations

import pathlib
import sys
from typing import Literal

import pandas as pd
from numpy.random import RandomState

from pelinker.config import ClusteringOptimizationConfig, NegativeScreenerConfig
from pelinker.model_selection_checkpoint import combination_key_from_members
from pelinker.reporting import (
    ClusteringSearchSummaryRow,
    ModelSelectionReport,
)
from pelinker.selection import evaluate_selection_from_paths
from pelinker.transform import TransformConfig


def path_by_model_layer(
    valid_files: list[tuple[pathlib.Path, str, str]],
) -> dict[tuple[str, str], pathlib.Path]:
    return {(m, layer): fp for fp, m, layer in valid_files}


def clustering_optimization_config_for_run(
    *,
    min_class_size: int,
    max_scale: int,
    min_scale: int | None,
    clustering_grid_step: int,
    seed: int,
    clustering_sample_rows: int | None,
    batch_size: int,
    negative_label: str,
    screener_kind: str,
    drop_rare_entities: bool,
    min_mentions_per_entity: int,
    max_mentions_per_entity: int | None,
    max_mentions_negative: int | None,
    mention_cap_seed: int,
) -> ClusteringOptimizationConfig:
    kind: Literal["lda", "svm"] = "svm" if screener_kind == "svm" else "lda"
    ns = NegativeScreenerConfig(kind=kind, negative_label=negative_label)
    return ClusteringOptimizationConfig(
        min_class_size=min_class_size,
        max_scale=max_scale,
        min_scale=min_scale,
        clustering_grid_step=clustering_grid_step,
        rns=RandomState(seed=seed),
        base_seed=seed,
        clustering_sample_rows=clustering_sample_rows,
        batch_size=batch_size,
        drop_rare_entities=drop_rare_entities,
        min_mentions_per_entity=min_mentions_per_entity,
        max_mentions_per_entity=max_mentions_per_entity,
        max_mentions_negative=max_mentions_negative,
        mention_cap_seed=mention_cap_seed,
        optimization_method="mean",
        ambient_screener=ns,
    )


def parse_fusion_members(layer_label: str) -> list[tuple[str, str]]:
    members: list[tuple[str, str]] = []
    for part in layer_label.split("+"):
        p = part.strip()
        model, _, layer = p.partition("/")
        members.append((model, layer))
    return sorted(members, key=lambda t: (t[0], t[1]))


def ordered_paths_for_fusion(
    path_by_ml: dict[tuple[str, str], pathlib.Path],
    members: list[tuple[str, str]],
) -> list[pathlib.Path]:
    return [path_by_ml[t] for t in members]


def combo_key_for_results_row(series: pd.Series) -> str:
    m_str = str(series["model"])
    ly_str = str(series["layer"])
    if m_str.startswith("fusion"):
        return combination_key_from_members(parse_fusion_members(ly_str))
    return combination_key_from_members([(m_str, ly_str)])


def update_leaderboard_fixed(
    summary_row: ClusteringSearchSummaryRow,
    *,
    best_overall_score: float | None,
    best_overall_model: str | None,
    best_overall_layer: str | None,
    best_per_model: dict[str, float],
) -> tuple[float | None, str | None, str | None, dict[str, float]]:
    mean_dbcv = summary_row.dbcv.mean
    model, layer = summary_row.model, summary_row.layer
    if not model.startswith("fusion"):
        if best_overall_score is None or mean_dbcv > best_overall_score:
            best_overall_score = mean_dbcv
            best_overall_model = model
            best_overall_layer = layer
        if model not in best_per_model or mean_dbcv > best_per_model[model]:
            best_per_model[model] = mean_dbcv
    return best_overall_score, best_overall_model, best_overall_layer, best_per_model


def recompute_leaderboard_from_results(
    results: list[ClusteringSearchSummaryRow],
) -> tuple[float | None, str | None, str | None, dict[str, float]]:
    best_overall_score = None
    best_overall_model = None
    best_overall_layer = None
    best_per_model: dict[str, float] = {}
    for r in results:
        best_overall_score, best_overall_model, best_overall_layer, best_per_model = (
            update_leaderboard_fixed(
                r,
                best_overall_score=best_overall_score,
                best_overall_model=best_overall_model,
                best_overall_layer=best_overall_layer,
                best_per_model=best_per_model,
            )
        )
    return best_overall_score, best_overall_model, best_overall_layer, best_per_model


def materialize_best_report(
    top: ClusteringSearchSummaryRow,
    *,
    valid_files: list[tuple[pathlib.Path, str, str]],
    path_by_ml: dict[tuple[str, str], pathlib.Path],
    transform_config: TransformConfig,
    optimization_config: ClusteringOptimizationConfig,
    selected_labels: set[str] | None,
) -> ModelSelectionReport | None:
    if top.model.startswith("fusion"):
        members = parse_fusion_members(top.layer)
        try:
            ordered_paths = ordered_paths_for_fusion(path_by_ml, members)
        except KeyError:
            return None
        return evaluate_selection_from_paths(
            transform_config=transform_config,
            optimization_config=optimization_config,
            file_paths=ordered_paths,
            selected_labels=selected_labels,
            all_metrics_dfs=None,
            show_embedding_read_progress=sys.stdout.isatty(),
        )
    key = (top.model, top.layer)
    path = path_by_ml.get(key)
    if path is None:
        return None
    return evaluate_selection_from_paths(
        transform_config=transform_config,
        optimization_config=optimization_config,
        file_path=path,
        selected_labels=selected_labels,
        all_metrics_dfs=None,
        show_embedding_read_progress=sys.stdout.isatty(),
    )
