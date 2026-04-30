"""Tests for fine clustering metadata extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pelinker.reporting import ClusteringHyperparameters, ClusteringReport
from run.analysis.clustering_quality import _fine_clustering_metadata_df


def _report_with_assignments(assignments: pd.DataFrame) -> ClusteringReport:
    return ClusteringReport(
        hyperparameters=ClusteringHyperparameters(min_cluster_size=10),
        best_score=0.5,
        number_properties=2,
        n_clusters_emergent=2,
        metrics_df=pd.DataFrame({"min_cluster_size": [10], "dbcv": [0.5]}),
        assignments=assignments,
        pca_residuals=np.array([0.1], dtype=np.float64),
        pca_mahalanobis=np.array([0.2], dtype=np.float64),
        umap_clustering=np.array([[0.0, 0.1]], dtype=np.float64),
        umap_visualization=np.array([[0.0, 0.1]], dtype=np.float64),
        pca_reduced=np.array([[0.0, 0.1]], dtype=np.float64),
        ari=0.1,
    )


def test_fine_metadata_includes_core_and_optional_columns() -> None:
    report = _report_with_assignments(
        pd.DataFrame(
            {
                "pmid": [1],
                "mention": ["abc"],
                "entity": ["term_a"],
                "cluster": [3],
            }
        )
    )
    got = _fine_clustering_metadata_df(
        report,
        model="m",
        layer="L1",
        sample_idx=0,
    )
    assert list(got.columns) == [
        "model",
        "layer",
        "sample_idx",
        "entity",
        "cluster",
        "pmid",
        "mention",
    ]
    assert got.iloc[0]["model"] == "m"
    assert got.iloc[0]["layer"] == "L1"
    assert got.iloc[0]["sample_idx"] == 0
    assert got.iloc[0]["entity"] == "term_a"
    assert got.iloc[0]["cluster"] == 3


def test_fine_metadata_returns_empty_if_required_columns_missing() -> None:
    report = _report_with_assignments(pd.DataFrame({"entity": ["term_a"]}))
    got = _fine_clustering_metadata_df(
        report,
        model="m",
        layer="L1",
        sample_idx=1,
    )
    assert got.empty
