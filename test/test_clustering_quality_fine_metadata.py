"""Tests for fine clustering metadata extraction."""

from __future__ import annotations

import pandas as pd

from pelinker.reporting import ClusteringHyperparameters, ClusteringReport
from run.analysis.clustering_quality import _fine_clustering_metadata_df


def _report_with_df(df: pd.DataFrame) -> ClusteringReport:
    return ClusteringReport(
        hyperparameters=ClusteringHyperparameters(min_cluster_size=10),
        best_score=0.5,
        number_properties=2,
        n_clusters_emergent=2,
        metrics_df=pd.DataFrame({"min_cluster_size": [10], "dbcv": [0.5]}),
        df=df,
        ari=0.1,
    )


def test_fine_metadata_includes_core_and_optional_columns() -> None:
    report = _report_with_df(
        pd.DataFrame(
            {
                "pmid": [1],
                "mention": ["abc"],
                "property": ["term_a"],
                "class": [3],
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
        "property",
        "class",
        "pmid",
        "mention",
    ]
    assert got.iloc[0]["model"] == "m"
    assert got.iloc[0]["layer"] == "L1"
    assert got.iloc[0]["sample_idx"] == 0
    assert got.iloc[0]["property"] == "term_a"
    assert got.iloc[0]["class"] == 3


def test_fine_metadata_returns_empty_if_required_columns_missing() -> None:
    report = _report_with_df(pd.DataFrame({"property": ["term_a"]}))
    got = _fine_clustering_metadata_df(
        report,
        model="m",
        layer="L1",
        sample_idx=1,
    )
    assert got.empty
