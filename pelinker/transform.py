"""
Configurable transformation pipeline for embedding reduction.

This module provides a flexible transformation pipeline that reduces high-dimensional
embeddings through PCA and UMAP before clustering with HDBSCAN.

Pipeline: LLM embeddings -> PCA -> UMAP -> HDBSCAN
"""

from typing import Optional

import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA

from pelinker.config import TransformConfig


class EmbeddingTransformer:
    """
    Transform embeddings through PCA and UMAP reduction.

    Pipeline:
        1. PCA: Reduce embeddings to pca_components dimensions
        2. UMAP: Further reduce PCA output to umap_components dimensions (for clustering)
        3. UMAP (viz): Reduce PCA output to umap_viz_components dimensions (for visualization)
    """

    def __init__(self, config: Optional[TransformConfig] = None):
        """
        Initialize the transformer with configuration.

        Args:
            config: TransformConfig instance. If None, uses default configuration.
        """
        self.config = config or TransformConfig()
        self.pca: Optional[PCA] = None
        self.umap: Optional[umap.UMAP] = None
        self.umap_viz: Optional[umap.UMAP] = None

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
        pca_reduced = self.pca.fit_transform(embeddings)

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

    def transform(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform embeddings through the pipeline.

        Args:
            embeddings: Array of shape (n_samples, n_features) containing embeddings

        Returns:
            Tuple of (umap_clustering, umap_visualization) arrays
            - umap_clustering: Shape (n_samples, umap_components) for clustering
            - umap_visualization: Shape (n_samples, umap_viz_components) for visualization
        """
        if self.pca is None or self.umap is None or self.umap_viz is None:
            raise ValueError(
                "Transformer must be fitted before transform. Call fit() first."
            )

        # Apply PCA
        pca_reduced = self.pca.transform(embeddings)

        # Apply UMAP for clustering
        umap_clustering = self.umap.transform(pca_reduced)

        # Apply UMAP for visualization
        umap_visualization = self.umap_viz.transform(pca_reduced)

        return umap_clustering, umap_visualization

    def fit_transform(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit the pipeline and transform embeddings in one step.

        Args:
            embeddings: Array of shape (n_samples, n_features) containing embeddings

        Returns:
            Tuple of (umap_clustering, umap_visualization) arrays
        """
        return self.fit(embeddings).transform(embeddings)


def transform_embeddings(
    df: pd.DataFrame,
    config: Optional[TransformConfig] = None,
    embed_column: str = "embed",
) -> pd.DataFrame:
    """
    Transform embeddings in a DataFrame using PCA -> UMAP pipeline.

    This function adds columns to the DataFrame:
    - PCA columns: p_00, p_01, ... (one per fitted component; may be fewer than
      ``pca_components`` when the sample count is small)
    - UMAP clustering columns: u_00, u_01, ..., u_{umap_components-1:02d}
    - UMAP visualization columns: uviz_00, uviz_01, ..., uviz_{umap_viz_components-1:02d}

    Args:
        df: DataFrame with embeddings in the specified column
        config: TransformConfig instance. If None, uses default configuration.
        embed_column: Name of column containing embeddings (default: "embed")

    Returns:
        DataFrame with added transformation columns
    """
    if embed_column not in df.columns:
        raise ValueError(f"Column '{embed_column}' not found in DataFrame")

    config = config or TransformConfig()

    # OPTIMIZATION: convert once to float32 to keep memory usage lower for
    # large embedding matrices (e.g., tens of thousands of rows).
    embedding_vectors = np.stack(df[embed_column].values).astype(np.float32, copy=False)

    # Apply transformation
    transformer = EmbeddingTransformer(config)
    transformer.fit(embedding_vectors)

    # OPTIMIZATION: compute PCA transform once and reuse it for both UMAP outputs
    # and exported PCA columns (avoids duplicate PCA transform pass).
    pca_reduced = transformer.pca.transform(embedding_vectors)
    umap_clustering = transformer.umap.transform(pca_reduced)
    umap_visualization = transformer.umap_viz.transform(pca_reduced)

    # Create DataFrames for each transformation (widths follow fitted dims when PCA is capped).
    n_pca = int(pca_reduced.shape[1])
    df_pca = pd.DataFrame(
        pca_reduced,
        index=df.index,
        columns=[f"p_{j:02d}" for j in range(n_pca)],
    )

    df_umap = pd.DataFrame(
        umap_clustering,
        index=df.index,
        columns=[f"u_{j:02d}" for j in range(config.umap_components)],
    )

    df_umap_viz = pd.DataFrame(
        umap_visualization,
        index=df.index,
        columns=[f"uviz_{j:02d}" for j in range(config.umap_viz_components)],
    )

    # Concatenate all transformations
    df_result = pd.concat([df, df_pca, df_umap, df_umap_viz], axis=1)

    return df_result


def get_umap_columns(config: TransformConfig) -> list[str]:
    """
    Get the list of UMAP column names for clustering from a TransformConfig.

    Args:
        config: TransformConfig instance

    Returns:
        List of column names like ["u_00", "u_01", ..., "u_{n-1:02d}"]
    """
    return [f"u_{j:02d}" for j in range(config.umap_components)]
