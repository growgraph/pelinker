"""
Model-selection evaluation: load embeddings, draw stratified samples, evaluate screeners + clustering.

Contrast with :meth:`~pelinker.model.Linker.fit`, which trains on the full mention corpus and
persists a single production artifact (no ``frac`` / ``eval_max_rows`` subsampling).
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable, Sequence
from typing import Literal

import hdbscan
import numpy as np
import pandas as pd

from pelinker.analysis import (
    _mention_quality_frame,
    compute_clustering_fit_metrics,
    drop_entities_with_few_mentions,
    evaluate_all_screeners_cv,
    split_by_negative_label,
)
from pelinker.clustering_grid import (
    aggregate_grid_metrics,
    evaluate_cluster_size_grid,
    solve_optimal_min_cluster_size_from_aggregated,
)
from pelinker.config import ClusteringOptimizationConfig, TransformConfig
from pelinker.embedding_fusion import concat_mention_level_embedding_sources
from pelinker.reporting import (
    AllScreenerCvResult,
    ClusteringHyperparameters,
    ModelSelectionReport,
    PerDatapointScores,
    entity_negative_label_mask_01,
)
from pelinker.sampling import draw_selection_sample
from pelinker.transform import EmbeddingTransformer, score_transform_artifacts


def load_selection_frame(
    *,
    file_path: pathlib.Path | None = None,
    file_paths: Sequence[pathlib.Path] | None = None,
    dfr: pd.DataFrame | None = None,
    config: ClusteringOptimizationConfig,
    selected_labels: set[str] | None = None,
    embedding_read_status: Callable[[str], None] | None = None,
    show_embedding_read_progress: bool = False,
) -> pd.DataFrame | None:
    """
    Load and filter mention-level embeddings for model selection (no subsampling).

    Provide exactly one of ``file_path``, ``file_paths``, or ``dfr``.
    """
    sources = [file_path is not None, file_paths is not None, dfr is not None]
    if sum(bool(x) for x in sources) != 1:
        raise ValueError(
            "Provide exactly one of file_path=, file_paths=, or dfr= to load_selection_frame"
        )

    frame: pd.DataFrame | None
    if dfr is not None:
        frame = dfr.copy()
    else:
        if file_paths is not None:
            paths = list(file_paths)
        else:
            assert file_path is not None
            paths = [file_path]
        if len(paths) == 0:
            return None
        frame = concat_mention_level_embedding_sources(
            paths,
            batch_size=config.batch_size,
            n_embedding_batches=config.n_embedding_batches,
            read_status=embedding_read_status,
            show_read_progress=show_embedding_read_progress,
        )
        if frame is None or len(frame) == 0:
            return None

    if selected_labels is not None:
        frame = frame.loc[frame["entity"].isin(selected_labels)].copy()
        if len(frame) == 0:
            return None

    neg_label = config.ambient_screener.negative_label
    frame = drop_entities_with_few_mentions(
        frame,
        config.min_class_size,
        negative_label=neg_label,
    )
    if len(frame) == 0:
        return None
    return frame


def evaluate_selection_sample(
    frame: pd.DataFrame,
    transform_config: TransformConfig,
    optimization_config: ClusteringOptimizationConfig | None = None,
    *,
    selected_labels: set[str] | None = None,
    all_metrics_dfs: list[pd.DataFrame] | None = None,
    aggregation_level: Literal["mention", "entity"] = "mention",
    entities_pre_filtered: bool = True,
) -> ModelSelectionReport | None:
    """
    Evaluate one selection draw: screener CV, transform, clustering grid, and HDBSCAN fit.

    When ``all_metrics_dfs`` is provided, appends this sample's grid ``metrics_df``. Run
    :func:`~pelinker.analysis.pooled_min_cluster_size_from_metrics_dfs` after all bootstraps
    for a pooled ``min_cluster_size``.

    ``entities_pre_filtered=True`` skips min-mention-per-entity trimming (default when the
    frame came from :func:`load_selection_frame`).
    """
    config = optimization_config or ClusteringOptimizationConfig()
    dfr = frame

    if "embed" not in dfr.columns or "entity" not in dfr.columns:
        return None

    if selected_labels is not None:
        dfr = dfr.loc[dfr["entity"].isin(selected_labels)].copy()
        if len(dfr) == 0:
            return None

    screener_cfg = config.ambient_screener
    neg_label = screener_cfg.negative_label

    if aggregation_level == "mention" and not entities_pre_filtered:
        dfr = drop_entities_with_few_mentions(
            dfr,
            config.min_class_size,
            negative_label=neg_label,
        )
        if len(dfr) == 0:
            return None

    neg_mask, dfr_manifold = split_by_negative_label(dfr, neg_label)
    if len(dfr_manifold) == 0:
        return None

    number_properties = int(dfr_manifold["entity"].nunique())

    embeddings_manifold = np.stack(dfr_manifold["embed"].values).astype(
        np.float32, copy=False
    )
    transformer = EmbeddingTransformer(transform_config).fit(embeddings_manifold)
    manifold_artifacts = score_transform_artifacts(
        dfr_manifold,
        transformer,
        include_umap=True,
    )

    X_embed_full = np.stack(dfr["embed"].values).astype(np.float64, copy=False)
    y_full = (dfr["entity"].astype(str).values == neg_label).astype(np.int64).ravel()
    entity_full = dfr["entity"].astype(str).values
    orig_idx_full = np.arange(len(dfr), dtype=np.int64)
    projection_cfg = config.projection_screener
    full_quality_artifacts = score_transform_artifacts(
        dfr,
        transformer,
        include_umap=False,
    )
    X_manifold_full = np.column_stack(
        [
            np.asarray(full_quality_artifacts.pca_residuals, dtype=np.float64),
            np.asarray(full_quality_artifacts.pca_mahalanobis, dtype=np.float64),
            np.asarray(full_quality_artifacts.pca_spectral_entropy, dtype=np.float64),
        ]
    )
    X_m_cv: np.ndarray | None = X_manifold_full if projection_cfg.enabled else None

    unified = evaluate_all_screeners_cv(
        X_embed=X_embed_full,
        X_manifold=X_m_cv,
        y=y_full,
        entity=entity_full,
        orig_idx=orig_idx_full,
        screener_cfg=screener_cfg,
        oov_cfg=projection_cfg,
    )
    all_screener_cv: AllScreenerCvResult | None = None
    screener_oos_dp: PerDatapointScores | None = None
    if unified is not None:
        all_screener_cv, screener_oos_dp = unified

    artifacts = manifold_artifacts
    umap_clustering_df = artifacts.umap_clustering_df().assign(
        entity=dfr_manifold["entity"].values
    )

    sizes = list(
        np.arange(
            config.resolved_min_scale(),
            config.max_scale,
            config.clustering_grid_step,
        )
    )
    metrics_df = evaluate_cluster_size_grid(
        umap_clustering_df,
        [c for c in umap_clustering_df.columns if c != "entity"],
        sizes,
    )
    if len(metrics_df) == 0:
        return None

    if all_metrics_dfs is not None:
        all_metrics_dfs.append(metrics_df)

    single_sample_aggregated = aggregate_grid_metrics([metrics_df])
    solved_single_sample = solve_optimal_min_cluster_size_from_aggregated(
        single_sample_aggregated,
        objective=config.grid_objective,
        method=config.optimization_method,
        smooth_window=config.grid_smooth_window,
        plateau_fraction=config.grid_plateau_fraction,
        derivative_rel_tol=config.grid_derivative_rel_tol,
    )
    best_size = solved_single_sample.chosen_min_cluster_size
    best_score = solved_single_sample.score_mean_at_chosen

    clusterer = hdbscan.HDBSCAN(min_cluster_size=best_size, gen_min_span_tree=True)
    labels = clusterer.fit_predict(artifacts.umap_clustering)
    fit_metrics = compute_clustering_fit_metrics(
        clusterer,
        dfr_manifold,
        min_cluster_size=best_size,
        cluster_labels=labels,
    )
    assignments = dfr_manifold[["entity"]].copy()
    for optional_col in ["pmid", "mention"]:
        if optional_col in dfr_manifold.columns:
            assignments[optional_col] = dfr_manifold[optional_col]
    assignments["cluster"] = labels.astype(int, copy=False)

    res_a = artifacts.pca_residuals
    mah_a = artifacts.pca_mahalanobis
    ent_a = artifacts.pca_spectral_entropy
    y_neg = entity_negative_label_mask_01(dfr_manifold["entity"], neg_label)
    mention_quality = _mention_quality_frame(
        dfr,
        neg_mask=neg_mask,
        cluster_kb=labels,
        pca_residuals=full_quality_artifacts.pca_residuals,
        pca_mahalanobis=full_quality_artifacts.pca_mahalanobis,
        pca_spectral_entropy=full_quality_artifacts.pca_spectral_entropy,
        negative_label=neg_label,
    )

    return ModelSelectionReport(
        hyperparameters=ClusteringHyperparameters(min_cluster_size=best_size),
        best_score=best_score,
        number_properties=number_properties,
        n_clusters_emergent=fit_metrics.n_clusters_emergent,
        metrics_df=metrics_df,
        assignments=assignments,
        pca_residuals=res_a,
        pca_mahalanobis=mah_a,
        pca_spectral_entropy=ent_a,
        oov_label=y_neg,
        umap_clustering=artifacts.umap_clustering,
        umap_visualization=artifacts.umap_visualization,
        pca_reduced=artifacts.pca_reduced,
        all_screener_cv=all_screener_cv,
        screener_oos_datapoints=screener_oos_dp,
        ari=fit_metrics.ari,
        mention_quality=mention_quality,
    )


def evaluate_selection_from_paths(
    transform_config: TransformConfig,
    optimization_config: ClusteringOptimizationConfig | None = None,
    *,
    file_path: pathlib.Path | None = None,
    file_paths: Sequence[pathlib.Path] | None = None,
    dfr: pd.DataFrame | None = None,
    selected_labels: set[str] | None = None,
    all_metrics_dfs: list[pd.DataFrame] | None = None,
    sample_index: int = 0,
    embedding_read_status: Callable[[str], None] | None = None,
    show_embedding_read_progress: bool = False,
) -> ModelSelectionReport | None:
    """
    Load, draw one stratified sample, and evaluate (convenience for single-shot callers).
    """
    config = optimization_config or ClusteringOptimizationConfig()
    base = load_selection_frame(
        file_path=file_path,
        file_paths=file_paths,
        dfr=dfr,
        config=config,
        selected_labels=selected_labels,
        embedding_read_status=embedding_read_status,
        show_embedding_read_progress=show_embedding_read_progress,
    )
    if base is None:
        return None
    sample_frame = draw_selection_sample(base, config, sample_index=sample_index)
    return evaluate_selection_sample(
        sample_frame,
        transform_config,
        optimization_config=config,
        all_metrics_dfs=all_metrics_dfs,
    )
