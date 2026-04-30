"""Tests for typed clustering reports and multi-sample summaries."""

import numpy as np
import pandas as pd
import pytest

from pelinker.reporting import (
    ClusteringHyperparameters,
    ClusteringReport,
    ClusteringSearchSummaryRow,
    summarize_clustering_reports_for_search,
)


def _minimal_report(
    min_cluster_size: int,
    best_score: float,
    *,
    n_clusters_emergent: int = 3,
    ari: float | None = None,
) -> ClusteringReport:
    metrics_df = pd.DataFrame(
        {"min_cluster_size": [min_cluster_size], "dbcv": [best_score]}
    )
    assignments = pd.DataFrame({"entity": ["p1"], "cluster": [0]})
    return ClusteringReport(
        hyperparameters=ClusteringHyperparameters(min_cluster_size=min_cluster_size),
        best_score=best_score,
        number_properties=5,
        n_clusters_emergent=n_clusters_emergent,
        metrics_df=metrics_df,
        assignments=assignments,
        pca_residuals=np.array([0.1], dtype=np.float64),
        pca_mahalanobis=np.array([0.2], dtype=np.float64),
        umap_clustering=np.array([[0.0, 0.1]], dtype=np.float64),
        umap_visualization=np.array([[0.0, 0.1]], dtype=np.float64),
        pca_reduced=np.array([[0.0, 0.1]], dtype=np.float64),
        ari=ari,
    )


def test_clustering_report_hyperparameter_access() -> None:
    r = _minimal_report(42, 0.88)
    assert r.hyperparameters.min_cluster_size == 42


def test_summarize_clustering_reports_for_search_means_and_flat_dict() -> None:
    r1 = _minimal_report(10, 0.5, n_clusters_emergent=4, ari=0.8)
    r2 = _minimal_report(20, 0.7, n_clusters_emergent=8, ari=0.9)
    row = summarize_clustering_reports_for_search(
        [r1, r2],
        model="m1",
        layer="L0",
    )
    assert isinstance(row, ClusteringSearchSummaryRow)
    assert row.hyperparameters.min_cluster_size.mean == pytest.approx(15.0)
    assert row.hyperparameters.min_cluster_size.std > 0
    assert row.dbcv.mean == pytest.approx(0.6)
    flat = row.to_flat_dict()
    assert flat["model"] == "m1"
    assert flat["layer"] == "L0"
    assert flat["best_size"] == pytest.approx(15.0)
    assert flat["best_score"] == pytest.approx(0.6)
    assert flat["n_clusters_emergent"] == pytest.approx(6.0)
    assert flat["n_clusters_emergent_std"] > 0
    assert flat["ari"] == pytest.approx(0.85)


def test_summarize_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        summarize_clustering_reports_for_search([], model="m", layer="l")


def test_summarize_ari_none_when_all_missing() -> None:
    r = _minimal_report(5, 0.3, ari=None)
    row = summarize_clustering_reports_for_search([r], model="m", layer="l")
    assert row.ari is None
    flat = row.to_flat_dict()
    assert flat["ari"] is None
    assert flat["ari_std"] == 0.0


def test_summarize_with_pooled_min_cluster_size_uses_consensus_and_per_sample_dbcv() -> (
    None
):
    """Pooled summary reports one ``best_size`` and DBCV mean/std at that grid point."""
    r1 = ClusteringReport(
        hyperparameters=ClusteringHyperparameters(min_cluster_size=10),
        best_score=0.5,
        number_properties=5,
        n_clusters_emergent=3,
        metrics_df=pd.DataFrame(
            {
                "min_cluster_size": [10, 20],
                "dbcv": [0.4, 0.55],
            }
        ),
        assignments=pd.DataFrame({"entity": ["p1"], "cluster": [0]}),
        pca_residuals=np.array([0.1], dtype=np.float64),
        pca_mahalanobis=np.array([0.2], dtype=np.float64),
        umap_clustering=np.array([[0.0, 0.1]], dtype=np.float64),
        umap_visualization=np.array([[0.0, 0.1]], dtype=np.float64),
        pca_reduced=np.array([[0.0, 0.1]], dtype=np.float64),
        ari=0.8,
    )
    r2 = ClusteringReport(
        hyperparameters=ClusteringHyperparameters(min_cluster_size=15),
        best_score=0.6,
        number_properties=5,
        n_clusters_emergent=4,
        metrics_df=pd.DataFrame(
            {
                "min_cluster_size": [10, 20],
                "dbcv": [0.45, 0.50],
            }
        ),
        assignments=pd.DataFrame({"entity": ["p2"], "cluster": [1]}),
        pca_residuals=np.array([0.1], dtype=np.float64),
        pca_mahalanobis=np.array([0.2], dtype=np.float64),
        umap_clustering=np.array([[0.0, 0.1]], dtype=np.float64),
        umap_visualization=np.array([[0.0, 0.1]], dtype=np.float64),
        pca_reduced=np.array([[0.0, 0.1]], dtype=np.float64),
        ari=0.85,
    )
    row = summarize_clustering_reports_for_search(
        [r1, r2],
        model="m",
        layer="L0",
        pooled_min_cluster_size=20,
    )
    assert row.hyperparameters.min_cluster_size.mean == 20.0
    assert row.hyperparameters.min_cluster_size.std == 0.0
    assert row.dbcv.mean == pytest.approx((0.55 + 0.50) / 2)
    assert row.dbcv.std > 0
