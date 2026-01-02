"""
Configurable transformation pipeline for embedding reduction.

This module provides a flexible transformation pipeline that reduces high-dimensional
embeddings through PCA and UMAP before clustering with HDBSCAN.

Pipeline: LLM embeddings -> PCA -> UMAP -> HDBSCAN
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA


@dataclass
class TransformConfig:
    """Configuration for the embedding transformation pipeline."""

    # PCA configuration
    pca_components: int = 50
    """Number of principal components to keep after PCA reduction."""

    # UMAP configuration
    umap_components: int = 4
    """Number of UMAP dimensions for clustering (typically 3-5)."""
    umap_metric: str = "cosine"
    """Distance metric for UMAP (default: 'cosine')."""

    # Visualization UMAP configuration
    umap_viz_components: int = 3
    """Number of UMAP dimensions for visualization (default: 3)."""
    umap_viz_metric: str = "cosine"
    """Distance metric for visualization UMAP (default: 'cosine')."""

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.pca_components < 1:
            raise ValueError("pca_components must be >= 1")
        if self.umap_components < 2:
            raise ValueError("umap_components must be >= 2")
        if self.umap_viz_components < 2:
            raise ValueError("umap_viz_components must be >= 2")


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

        # Fit PCA
        self.pca = PCA(n_components=self.config.pca_components)
        pca_reduced = self.pca.fit_transform(embeddings)

        # Fit UMAP for clustering
        self.umap = umap.UMAP(
            n_components=self.config.umap_components,
            metric=self.config.umap_metric,
        )
        self.umap.fit(pca_reduced)

        # Fit UMAP for visualization
        self.umap_viz = umap.UMAP(
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
    - PCA columns: p_00, p_01, ..., p_{pca_components-1:02d}
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

    # Extract embeddings
    embedding_vectors = np.stack(df[embed_column].values)

    # Apply transformation
    transformer = EmbeddingTransformer(config)
    umap_clustering, umap_visualization = transformer.fit_transform(embedding_vectors)

    # Create DataFrames for each transformation
    df_pca = pd.DataFrame(
        transformer.pca.transform(embedding_vectors),
        index=df.index,
        columns=[f"p_{j:02d}" for j in range(config.pca_components)],
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
