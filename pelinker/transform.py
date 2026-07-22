"""
Configurable transformation pipeline for embedding reduction.

Pipeline: LLM embeddings -> PCA -> UMAP/ParametricUMAP (clustering) -> HDBSCAN
Visualization: umap_clustering -> PCA or UMAP -> plot coords
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA

from pelinker.config import TransformConfig

logger = logging.getLogger(__name__)

# Union type for clustering manifold (standard UMAP or ParametricUMAP).
ClusteringManifold = umap.UMAP


def is_parametric_umap(obj: object) -> bool:
    return type(obj).__name__ == "ParametricUMAP"


def parametric_umap_sidecar_dir(file_spec: str | pathlib.Path) -> pathlib.Path:
    """Directory beside a linker artifact for ParametricUMAP ``save`` / ``load``."""
    p = pathlib.Path(file_spec).expanduser()
    if p.suffix == ".gz":
        stem = p.name[: -len(".gz")]
        return p.with_name(stem + ".parametric_umap")
    return p.with_name(p.name + ".parametric_umap")


def save_clustering_manifold(
    manifold: ClusteringManifold | None, path: str | pathlib.Path
) -> None:
    """Persist a ParametricUMAP for predict (encoder + picklable state).

    umap-learn's ``ParametricUMAP.save`` tries to serialize ``parametric_model`` as
    Keras 3, which fails on ``UMAPModel``. Predict only needs the encoder, so we
    save ``encoder.keras`` and pickle the UMAP object with ``parametric_model`` cleared.
    """
    import pickle

    if manifold is None:
        return
    if not is_parametric_umap(manifold):
        raise TypeError(
            f"save_clustering_manifold only supports ParametricUMAP; got {type(manifold)!r}"
        )
    out = pathlib.Path(path)
    out.mkdir(parents=True, exist_ok=True)

    encoder = manifold.encoder
    if encoder is not None:
        encoder_path = out / "encoder.keras"
        encoder.save(str(encoder_path))
        logger.info("Wrote ParametricUMAP encoder to %s", encoder_path)

    parametric_model = manifold.parametric_model
    raw_data = getattr(manifold, "_raw_data", None)
    manifold.parametric_model = None
    manifold.encoder = None
    if hasattr(manifold, "_raw_data"):
        delattr(manifold, "_raw_data")
    try:
        model_pkl = out / "model.pkl"
        with model_pkl.open("wb") as fh:
            pickle.dump(manifold, fh, pickle.HIGHEST_PROTOCOL)
        logger.info("Wrote ParametricUMAP pickle to %s", model_pkl)
    finally:
        manifold.parametric_model = parametric_model
        manifold.encoder = encoder
        if raw_data is not None:
            manifold._raw_data = raw_data


def load_clustering_manifold(path: str | pathlib.Path) -> ClusteringManifold:
    """Load a ParametricUMAP written by :func:`save_clustering_manifold`."""
    from umap.parametric_umap import load_ParametricUMAP

    return load_ParametricUMAP(str(pathlib.Path(path)), verbose=False)


def _build_clustering_manifold(
    config: TransformConfig, *, n_neighbors: int
) -> ClusteringManifold:
    if config.manifold_kind == "parametric":
        from umap.parametric_umap import ParametricUMAP

        kwargs: dict[str, object] = {
            "n_neighbors": n_neighbors,
            "n_components": config.umap_components,
            "metric": config.umap_metric,
            "random_state": config.umap_seed,
        }
        if config.parametric_umap_batch_size is not None:
            kwargs["batch_size"] = config.parametric_umap_batch_size
        pumap = ParametricUMAP(**kwargs)
        # Not a constructor kwarg in umap-learn 0.5.x — set after init.
        pumap.n_training_epochs = int(config.parametric_umap_n_training_epochs)
        return pumap
    return umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=config.umap_components,
        metric=config.umap_metric,
        random_state=config.umap_seed,
    )


@dataclass(frozen=True)
class TransformArtifacts:
    """Typed outputs from PCA+UMAP transformation."""

    index: pd.Index
    pca_reduced: np.ndarray
    umap_clustering: np.ndarray
    cluster_viz: np.ndarray
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

    def cluster_viz_df(self) -> pd.DataFrame:
        n_viz = int(self.cluster_viz.shape[1])
        return pd.DataFrame(
            self.cluster_viz,
            index=self.index,
            columns=[f"cviz_{j:02d}" for j in range(n_viz)],
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
    Transform embeddings through PCA and UMAP / ParametricUMAP reduction.

    Pipeline:
        1. PCA: Reduce embeddings to pca_components dimensions
        2. UMAP or ParametricUMAP: Further reduce PCA output to umap_components
        3. Cluster viz: Reduce umap_clustering to cluster_viz_components (PCA or UMAP)
    """

    def __init__(self, config: TransformConfig | None = None):
        """
        Initialize the transformer with configuration.

        Args:
            config: TransformConfig instance. If None, uses default configuration.
        """
        self.config = config or TransformConfig()
        self.pca: PCA | None = None
        self.umap: ClusteringManifold | None = None
        self.cluster_viz_pca: PCA | None = None
        self.cluster_viz_umap: umap.UMAP | None = None
        self._mahalanobis_eps = 1e-12
        self._entropy_row_sum_eps = 1e-12
        self._entropy_log_eps = 1e-10

    @staticmethod
    def _l2_normalize_rows(embeddings: np.ndarray) -> np.ndarray:
        """L2-normalize each embedding row; keep zero rows unchanged."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        safe_norms = np.where(norms > 0.0, norms, 1.0)
        return embeddings / safe_norms

    def _cluster_viz_fitted(self) -> bool:
        if self.config.cluster_viz_method == "pca":
            return self.cluster_viz_pca is not None
        return self.cluster_viz_umap is not None

    def _transform_cluster_viz(self, umap_clustering: np.ndarray) -> np.ndarray:
        if self.config.cluster_viz_method == "pca":
            if self.cluster_viz_pca is None:
                raise ValueError("cluster_viz PCA is not fitted")
            return self.cluster_viz_pca.transform(umap_clustering)
        if self.cluster_viz_umap is None:
            raise ValueError("cluster_viz UMAP is not fitted")
        return self.cluster_viz_umap.transform(umap_clustering)

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

    def fit(self, embeddings: np.ndarray) -> EmbeddingTransformer:
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
        self.pca = PCA(n_components=pca_n, random_state=self.config.pca_seed)
        embeddings_normed = self._l2_normalize_rows(embeddings)
        pca_reduced = self.pca.fit_transform(embeddings_normed)

        # UMAP requires n_neighbors < n_samples; cap the default (15) for tiny frames.
        n_neighbors = min(15, max(2, n_samples - 1))
        if n_neighbors >= n_samples:
            n_neighbors = max(1, n_samples - 1)

        self.umap = _build_clustering_manifold(self.config, n_neighbors=n_neighbors)
        logger.info(
            "Fitting clustering manifold kind=%s n_neighbors=%s n_components=%s",
            self.config.manifold_kind,
            n_neighbors,
            self.config.umap_components,
        )
        self.umap.fit(pca_reduced)

        umap_clustering_train = self.umap.transform(pca_reduced)
        n_viz = min(
            self.config.cluster_viz_components,
            self.config.umap_components,
            n_samples,
        )

        self.cluster_viz_pca = None
        self.cluster_viz_umap = None
        viz_umap_seed = (
            None if self.config.umap_seed is None else self.config.umap_seed + 1
        )
        if self.config.cluster_viz_method == "pca":
            self.cluster_viz_pca = PCA(
                n_components=n_viz, random_state=self.config.pca_seed
            )
            self.cluster_viz_pca.fit(umap_clustering_train)
        else:
            self.cluster_viz_umap = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=n_viz,
                metric=self.config.cluster_viz_umap_metric,
                random_state=viz_umap_seed,
            )
            self.cluster_viz_umap.fit(umap_clustering_train)

        return self

    def transform(
        self, embeddings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform embeddings through the pipeline.

        Args:
            embeddings: Array of shape (n_samples, n_features) containing embeddings

        Returns:
            Tuple of (umap_clustering, cluster_viz, pca_residuals, pca_mahalanobis,
            pca_spectral_entropy) arrays
            - umap_clustering: Shape (n_samples, umap_components) for clustering
            - cluster_viz: Shape (n_samples, cluster_viz_components) for visualization
            - pca_residuals: Shape (n_samples,) PCA reconstruction residual norm per sample
            - pca_mahalanobis: Shape (n_samples,) Mahalanobis distance in PCA subspace
            - pca_spectral_entropy: Shape (n_samples,) Shannon entropy of normalized squared PCA coords
        """
        if self.pca is None or self.umap is None or not self._cluster_viz_fitted():
            raise ValueError(
                "Transformer must be fitted before transform. Call fit() first."
            )

        embeddings_normed = self._l2_normalize_rows(embeddings)
        pca_reduced = self.pca.transform(embeddings_normed)
        pca_residuals, pca_mahalanobis, pca_spectral_entropy = (
            self._compute_pca_metrics(embeddings_normed, pca_reduced)
        )

        umap_clustering = self.umap.transform(pca_reduced)
        cluster_viz = self._transform_cluster_viz(umap_clustering)

        return (
            umap_clustering,
            cluster_viz,
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
            Tuple of (umap_clustering, cluster_viz, pca_residuals, pca_mahalanobis,
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
        if transformer.umap is None or not transformer._cluster_viz_fitted():
            raise ValueError(
                "Transformer UMAP and cluster viz must be fitted when include_umap=True."
            )
        umap_clustering = transformer.umap.transform(pca_reduced)
        cluster_viz = transformer._transform_cluster_viz(umap_clustering)
    else:
        umap_clustering = np.empty((n_rows, 0), dtype=np.float64)
        cluster_viz = np.empty((n_rows, 0), dtype=np.float64)

    return TransformArtifacts(
        index=df.index.copy(),
        pca_reduced=pca_reduced,
        umap_clustering=umap_clustering,
        cluster_viz=cluster_viz,
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
