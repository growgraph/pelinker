"""Tests for typed clustering reports and multi-sample summaries."""

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
    hungarian: float | None = None,
) -> ClusteringReport:
    metrics_df = pd.DataFrame(
        {"min_cluster_size": [min_cluster_size], "dbcv": [best_score]}
    )
    df = pd.DataFrame({"x": [1.0]})
    return ClusteringReport(
        hyperparameters=ClusteringHyperparameters(min_cluster_size=min_cluster_size),
        best_score=best_score,
        number_properties=5,
        n_clusters_emergent=n_clusters_emergent,
        metrics_df=metrics_df,
        df=df,
        hungarian_accuracy=hungarian,
    )


def test_clustering_report_best_size_alias() -> None:
    r = _minimal_report(42, 0.88)
    assert r.best_size == 42
    assert r.hyperparameters.min_cluster_size == 42


def test_summarize_clustering_reports_for_search_means_and_flat_dict() -> None:
    r1 = _minimal_report(10, 0.5, n_clusters_emergent=4, hungarian=0.8)
    r2 = _minimal_report(20, 0.7, n_clusters_emergent=8, hungarian=0.9)
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
    assert flat["hungarian_accuracy"] == pytest.approx(0.85)


def test_summarize_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        summarize_clustering_reports_for_search([], model="m", layer="l")


def test_summarize_hungarian_none_when_all_missing() -> None:
    r = _minimal_report(5, 0.3, hungarian=None)
    row = summarize_clustering_reports_for_search([r], model="m", layer="l")
    assert row.hungarian_accuracy is None
    flat = row.to_flat_dict()
    assert flat["hungarian_accuracy"] is None
    assert flat["hungarian_accuracy_std"] == 0.0
