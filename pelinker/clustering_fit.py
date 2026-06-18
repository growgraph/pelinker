"""Shared PCA/UMAP + HDBSCAN manifold clustering for model selection and Linker.fit."""

from __future__ import annotations

from dataclasses import dataclass

import hdbscan
import numpy as np
import pandas as pd

from pelinker.analysis import compute_clustering_fit_metrics
from pelinker.config import TransformConfig
from pelinker.reporting import ClusteringFitMetrics
from pelinker.transform import EmbeddingTransformer, score_transform_artifacts


@dataclass(frozen=True, slots=True)
class ManifoldTransformerFitResult:
    """PCA + UMAP transformer fit on a KB manifold mention frame."""

    transformer: EmbeddingTransformer
    umap_clustering: np.ndarray
    cluster_viz: np.ndarray
    pca_residuals: np.ndarray
    pca_mahalanobis: np.ndarray
    pca_spectral_entropy: np.ndarray
    pca_reduced: np.ndarray


@dataclass(frozen=True, slots=True)
class ManifoldHdbscanFitResult:
    """HDBSCAN fit on UMAP clustering coordinates."""

    clusterer: hdbscan.HDBSCAN
    cluster_labels: np.ndarray
    fit_metrics: ClusteringFitMetrics


@dataclass(frozen=True, slots=True)
class ManifoldClusteringFitResult:
    """Full manifold clustering: transformer + HDBSCAN on the same mention rows."""

    transformer: EmbeddingTransformer
    clusterer: hdbscan.HDBSCAN
    cluster_labels: np.ndarray
    umap_clustering: np.ndarray
    cluster_viz: np.ndarray
    pca_residuals: np.ndarray
    pca_mahalanobis: np.ndarray
    pca_spectral_entropy: np.ndarray
    pca_reduced: np.ndarray
    fit_metrics: ClusteringFitMetrics


def fit_transformer_on_manifold(
    manifold_df: pd.DataFrame,
    transform_config: TransformConfig,
) -> ManifoldTransformerFitResult:
    """Fit PCA + UMAP on KB-only mention rows (no HDBSCAN)."""
    embeddings = np.stack(manifold_df["embed"].values).astype(np.float32, copy=False)
    transformer = EmbeddingTransformer(transform_config)
    umap_c, cluster_v, pca_r, pca_m, pca_e = transformer.fit_transform(embeddings)
    embeddings_normed = transformer._l2_normalize_rows(embeddings)
    pca_reduced = transformer.pca.transform(embeddings_normed)
    return ManifoldTransformerFitResult(
        transformer=transformer,
        umap_clustering=np.asarray(umap_c, dtype=np.float32),
        cluster_viz=np.asarray(cluster_v, dtype=np.float32),
        pca_residuals=np.asarray(pca_r, dtype=np.float32),
        pca_mahalanobis=np.asarray(pca_m, dtype=np.float32),
        pca_spectral_entropy=np.asarray(pca_e, dtype=np.float32),
        pca_reduced=np.asarray(pca_reduced, dtype=np.float32),
    )


def fit_transformer_artifacts_on_manifold(
    manifold_df: pd.DataFrame,
    transform_config: TransformConfig,
):
    """Fit transformer and return :class:`~pelinker.transform.TransformArtifacts` with UMAP."""
    tx = fit_transformer_on_manifold(manifold_df, transform_config)
    return score_transform_artifacts(
        manifold_df,
        tx.transformer,
        include_umap=True,
    )


def fit_hdbscan_on_umap(
    umap_clustering: np.ndarray,
    manifold_df: pd.DataFrame,
    min_cluster_size: int,
    *,
    prediction_data: bool = False,
) -> ManifoldHdbscanFitResult:
    """Run HDBSCAN on precomputed UMAP clustering coordinates."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        gen_min_span_tree=True,
        prediction_data=prediction_data,
    )
    cluster_labels_arr = clusterer.fit_predict(umap_clustering)
    cluster_labels = cluster_labels_arr.astype(int, copy=False)
    fit_metrics = compute_clustering_fit_metrics(
        clusterer,
        manifold_df,
        min_cluster_size=min_cluster_size,
        cluster_labels=cluster_labels,
    )
    return ManifoldHdbscanFitResult(
        clusterer=clusterer,
        cluster_labels=cluster_labels,
        fit_metrics=fit_metrics,
    )


def fit_manifold_clustering(
    manifold_df: pd.DataFrame,
    *,
    transform_config: TransformConfig,
    min_cluster_size: int,
    prediction_data: bool = False,
) -> ManifoldClusteringFitResult:
    """Fit PCA/UMAP + HDBSCAN on the same KB manifold mention frame."""
    tx = fit_transformer_on_manifold(manifold_df, transform_config)
    cl = fit_hdbscan_on_umap(
        tx.umap_clustering,
        manifold_df,
        min_cluster_size,
        prediction_data=prediction_data,
    )
    return ManifoldClusteringFitResult(
        transformer=tx.transformer,
        clusterer=cl.clusterer,
        cluster_labels=cl.cluster_labels,
        umap_clustering=tx.umap_clustering,
        cluster_viz=tx.cluster_viz,
        pca_residuals=tx.pca_residuals,
        pca_mahalanobis=tx.pca_mahalanobis,
        pca_spectral_entropy=tx.pca_spectral_entropy,
        pca_reduced=tx.pca_reduced,
        fit_metrics=cl.fit_metrics,
    )
