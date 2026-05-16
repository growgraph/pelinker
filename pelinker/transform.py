"""
Configurable transformation pipeline for embedding reduction.

This module provides a flexible transformation pipeline that reduces high-dimensional
embeddings through PCA and UMAP before clustering with HDBSCAN.

Pipeline: LLM embeddings -> PCA -> UMAP -> HDBSCAN
"""

import numpy as np
import pandas as pd
import umap
from dataclasses import dataclass
from sklearn.decomposition import PCA

from pelinker.config import TransformConfig


@dataclass(frozen=True)
class TransformArtifacts:
    """Typed outputs from PCA+UMAP transformation."""

    index: pd.Index
    pca_reduced: np.ndarray
    umap_clustering: np.ndarray
    umap_visualization: np.ndarray
    pca_residuals: np.ndarray
    pca_mahalanobis: np.ndarray
    pca_spectral_entropy: np.ndarray

    def umap_clustering_df(self) -> pd.DataFrame:
        n_umap = int(self.umap_clustering.shape[1])
        return pd.DataFrame(
            self.umap_clustering,
            index=self.index,
            columns=[f"u_{j:02d}" for j in range(n_umap)],
        )

    def umap_visualization_df(self) -> pd.DataFrame:
        n_umap_viz = int(self.umap_visualization.shape[1])
        return pd.DataFrame(
            self.umap_visualization,
            index=self.index,
            columns=[f"uviz_{j:02d}" for j in range(n_umap_viz)],
        )

    def pca_df(self) -> pd.DataFrame:
        n_pca = int(self.pca_reduced.shape[1])
        return pd.DataFrame(
            self.pca_reduced,
            index=self.index,
            columns=[f"p_{j:02d}" for j in range(n_pca)],
        )

    def anomaly_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "pca_residual": self.pca_residuals,
                "pca_mahalanobis": self.pca_mahalanobis,
                "pca_spectral_entropy": self.pca_spectral_entropy,
            },
            index=self.index,
        )


class EmbeddingTransformer:
    """
    Transform embeddings through PCA and UMAP reduction.

    Pipeline:
        1. PCA: Reduce embeddings to pca_components dimensions
        2. UMAP: Further reduce PCA output to umap_components dimensions (for clustering)
        3. UMAP (viz): Reduce PCA output to umap_viz_components dimensions (for visualization)
    """

    def __init__(self, config: TransformConfig | None = None):
        """
        Initialize the transformer with configuration.

        Args:
            config: TransformConfig instance. If None, uses default configuration.
        """
        self.config = config or TransformConfig()
        self.pca: PCA | None = None
        self.umap: umap.UMAP | None = None
        self.umap_viz: umap.UMAP | None = None
        self._mahalanobis_eps = 1e-12
        self._entropy_row_sum_eps = 1e-12
        self._entropy_log_eps = 1e-10

    @staticmethod
    def _l2_normalize_rows(embeddings: np.ndarray) -> np.ndarray:
        """L2-normalize each embedding row; keep zero rows unchanged."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        safe_norms = np.where(norms > 0.0, norms, 1.0)
        return embeddings / safe_norms

    def _compute_pca_metrics(
        self, embeddings_normed: np.ndarray, pca_reduced: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.pca is None:
            raise ValueError("PCA is not initialized")
        pca_reconstructed = self.pca.inverse_transform(pca_reduced)
        residual_norms = np.linalg.norm(embeddings_normed - pca_reconstructed, axis=1)
        explained_variance = np.asarray(self.pca.explained_variance_, dtype=np.float64)
        safe_var = np.maximum(explained_variance, self._mahalanobis_eps)
        mahalanobis = np.sqrt(
            np.sum((pca_reduced / np.sqrt(safe_var)) ** 2, axis=1),
        )
        pr = np.asarray(pca_reduced, dtype=np.float64)
        squared = pr * pr
        row_sums = np.maximum(
            squared.sum(axis=1, keepdims=True),
            self._entropy_row_sum_eps,
        )
        p = squared / row_sums
        spectral_entropy = -np.sum(p * np.log(p + self._entropy_log_eps), axis=1)
        return residual_norms, mahalanobis, spectral_entropy

    def fit(self, embeddings: np.ndarray) -> "EmbeddingTransformer":
        """
        Fit the transformation pipeline on training embeddings.

        Args:
            embeddings: Array of shape (n_samples, n_features) containing embeddings

        Returns:
            self for method chaining
        """
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")

        n_samples, n_features = embeddings.shape

        # PCA allows at most min(n_samples, n_features) components (svd_solver='full').
        pca_n = min(self.config.pca_components, n_samples, n_features)
        self.pca = PCA(n_components=pca_n)
        embeddings_normed = self._l2_normalize_rows(embeddings)
        pca_reduced = self.pca.fit_transform(embeddings_normed)

        # UMAP requires n_neighbors < n_samples; cap the default (15) for tiny frames.
        n_neighbors = min(15, max(2, n_samples - 1))
        if n_neighbors >= n_samples:
            n_neighbors = max(1, n_samples - 1)

        # Fit UMAP for clustering
        self.umap = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=self.config.umap_components,
            metric=self.config.umap_metric,
        )
        self.umap.fit(pca_reduced)

        # Fit UMAP for visualization
        self.umap_viz = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=self.config.umap_viz_components,
            metric=self.config.umap_viz_metric,
        )
        self.umap_viz.fit(pca_reduced)

        return self

    def transform(
        self, embeddings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform embeddings through the pipeline.

        Args:
            embeddings: Array of shape (n_samples, n_features) containing embeddings

        Returns:
            Tuple of (umap_clustering, umap_visualization, pca_residuals, pca_mahalanobis,
            pca_spectral_entropy) arrays
            - umap_clustering: Shape (n_samples, umap_components) for clustering
            - umap_visualization: Shape (n_samples, umap_viz_components) for visualization
            - pca_residuals: Shape (n_samples,) PCA reconstruction residual norm per sample
            - pca_mahalanobis: Shape (n_samples,) Mahalanobis distance in PCA subspace
            - pca_spectral_entropy: Shape (n_samples,) Shannon entropy of normalized squared PCA coords
        """
        if self.pca is None or self.umap is None or self.umap_viz is None:
            raise ValueError(
                "Transformer must be fitted before transform. Call fit() first."
            )

        embeddings_normed = self._l2_normalize_rows(embeddings)
        pca_reduced = self.pca.transform(embeddings_normed)
        pca_residuals, pca_mahalanobis, pca_spectral_entropy = (
            self._compute_pca_metrics(embeddings_normed, pca_reduced)
        )

        # Apply UMAP for clustering
        umap_clustering = self.umap.transform(pca_reduced)

        # Apply UMAP for visualization
        umap_visualization = self.umap_viz.transform(pca_reduced)

        return (
            umap_clustering,
            umap_visualization,
            pca_residuals,
            pca_mahalanobis,
            pca_spectral_entropy,
        )

    def fit_transform(
        self, embeddings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit the pipeline and transform embeddings in one step.

        Args:
            embeddings: Array of shape (n_samples, n_features) containing embeddings

        Returns:
            Tuple of (umap_clustering, umap_visualization, pca_residuals, pca_mahalanobis,
            pca_spectral_entropy) arrays
        """
        return self.fit(embeddings).transform(embeddings)


def score_transform_artifacts(
    df: pd.DataFrame,
    transformer: EmbeddingTransformer,
    *,
    embed_column: str = "embed",
    include_umap: bool = False,
) -> TransformArtifacts:
    """
    Score embeddings with a fitted :class:`EmbeddingTransformer` (no refit).

    When ``include_umap`` is False, UMAP arrays are empty ``(n_rows, 0)`` — use for
    PCA quality diagnostics on rows outside the manifold fit set.
    """
    if embed_column not in df.columns:
        raise ValueError(f"Column '{embed_column}' not found in DataFrame")
    if transformer.pca is None:
        raise ValueError(
            "Transformer must be fitted before score_transform_artifacts. Call fit() first."
        )

    embedding_vectors = np.stack(df[embed_column].values).astype(np.float32, copy=False)
    embedding_vectors_normed = transformer._l2_normalize_rows(embedding_vectors)
    pca_reduced = transformer.pca.transform(embedding_vectors_normed)
    pca_residuals, pca_mahalanobis, pca_spectral_entropy = (
        transformer._compute_pca_metrics(embedding_vectors_normed, pca_reduced)
    )

    n_rows = len(df)
    if include_umap:
        if transformer.umap is None or transformer.umap_viz is None:
            raise ValueError(
                "Transformer UMAP heads must be fitted when include_umap=True."
            )
        umap_clustering = transformer.umap.transform(pca_reduced)
        umap_visualization = transformer.umap_viz.transform(pca_reduced)
    else:
        umap_clustering = np.empty((n_rows, 0), dtype=np.float64)
        umap_visualization = np.empty((n_rows, 0), dtype=np.float64)

    return TransformArtifacts(
        index=df.index.copy(),
        pca_reduced=pca_reduced,
        umap_clustering=umap_clustering,
        umap_visualization=umap_visualization,
        pca_residuals=pca_residuals,
        pca_mahalanobis=pca_mahalanobis,
        pca_spectral_entropy=pca_spectral_entropy,
    )


def compute_transform_artifacts(
    df: pd.DataFrame,
    config: TransformConfig | None = None,
    embed_column: str = "embed",
) -> TransformArtifacts:
    """
    Transform embeddings in a DataFrame using PCA -> UMAP pipeline.

    Args:
        df: DataFrame with embeddings in the specified column
        config: TransformConfig instance. If None, uses default configuration.
        embed_column: Name of column containing embeddings (default: "embed")

    Returns:
        Typed transformation artifacts
    """
    if embed_column not in df.columns:
        raise ValueError(f"Column '{embed_column}' not found in DataFrame")

    config = config or TransformConfig()

    embedding_vectors = np.stack(df[embed_column].values).astype(np.float32, copy=False)
    transformer = EmbeddingTransformer(config)
    transformer.fit(embedding_vectors)
    return score_transform_artifacts(
        df, transformer, embed_column=embed_column, include_umap=True
    )
