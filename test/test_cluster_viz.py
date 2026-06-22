"""Tests for cluster-space visualization (PCA/UMAP on clustering UMAP coords)."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from pelinker.transform import EmbeddingTransformer, TransformConfig


def test_cluster_viz_pca_matches_manual_projection() -> None:
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(20, 10)).astype(np.float32)
    cfg = TransformConfig(
        pca_components=6,
        umap_components=4,
        cluster_viz_components=3,
        cluster_viz_method="pca",
        pca_seed=13,
        umap_seed=13,
    )
    transformer = EmbeddingTransformer(cfg)
    umap_c, cluster_v, *_ = transformer.fit_transform(embeddings)

    assert umap_c.shape == (20, 4)
    assert cluster_v.shape == (20, 3)
    assert transformer.cluster_viz_umap is None
    assert transformer.cluster_viz_pca is not None

    manual = transformer.cluster_viz_pca.transform(umap_c)
    np.testing.assert_allclose(cluster_v, manual, rtol=1e-5, atol=1e-5)


def test_cluster_viz_umap_produces_expected_shape() -> None:
    rng = np.random.default_rng(1)
    embeddings = rng.normal(size=(20, 10)).astype(np.float32)
    cfg = TransformConfig(
        pca_components=6,
        umap_components=4,
        cluster_viz_components=3,
        cluster_viz_method="umap",
        pca_seed=13,
        umap_seed=13,
    )
    transformer = EmbeddingTransformer(cfg)
    umap_c, cluster_v, *_ = transformer.fit_transform(embeddings)

    assert umap_c.shape == (20, 4)
    assert cluster_v.shape == (20, 3)
    assert transformer.cluster_viz_pca is None
    assert transformer.cluster_viz_umap is not None

    manual = transformer.cluster_viz_umap.transform(umap_c)
    np.testing.assert_allclose(cluster_v, manual, rtol=1e-5, atol=1e-5)


def test_cluster_viz_fits_on_clustering_coords_not_pca() -> None:
    rng = np.random.default_rng(2)
    embeddings = rng.normal(size=(16, 8)).astype(np.float32)
    cfg = TransformConfig(
        pca_components=5,
        umap_components=4,
        cluster_viz_components=2,
        cluster_viz_method="pca",
        pca_seed=7,
        umap_seed=7,
    )
    transformer = EmbeddingTransformer(cfg).fit(embeddings)

    assert transformer.cluster_viz_pca is not None
    assert transformer.cluster_viz_pca.n_features_in_ == cfg.umap_components

    embeddings_normed = transformer._l2_normalize_rows(embeddings)
    pca_reduced = transformer.pca.transform(embeddings_normed)
    umap_clustering = transformer.umap.transform(pca_reduced)
    got = transformer._transform_cluster_viz(umap_clustering)

    wrong_space = (
        PCA(n_components=2, random_state=cfg.pca_seed)
        .fit(pca_reduced)
        .transform(pca_reduced)
    )
    assert not np.allclose(got, wrong_space)


def test_transform_config_rejects_viz_dims_above_clustering_dims() -> None:
    try:
        TransformConfig(pca_components=10, umap_components=3, cluster_viz_components=4)
    except ValueError as exc:
        assert "cluster_viz_components" in str(exc)
    else:
        raise AssertionError("expected ValueError")
