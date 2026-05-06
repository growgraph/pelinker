"""Tests for fine clustering metadata extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pelinker.onto import NEGATIVE_LABEL
from pelinker.reporting import (
    ClusteringHyperparameters,
    ModelSelectionReport,
    entity_negative_label_mask_01,
)
from run.analysis.model_selection import _fine_clustering_metadata_df


def _report_with_assignments(assignments: pd.DataFrame) -> ModelSelectionReport:
    res = np.array([0.1], dtype=np.float64)
    mah = np.array([0.2], dtype=np.float64)
    ent = np.array([0.15], dtype=np.float64)
    y_neg = entity_negative_label_mask_01(assignments["entity"], NEGATIVE_LABEL)
    return ModelSelectionReport(
        hyperparameters=ClusteringHyperparameters(min_cluster_size=10),
        best_score=0.5,
        number_properties=2,
        n_clusters_emergent=2,
        metrics_df=pd.DataFrame({"min_cluster_size": [10], "dbcv": [0.5]}),
        assignments=assignments,
        pca_residuals=res,
        pca_mahalanobis=mah,
        pca_spectral_entropy=ent,
        pca_residual_label_01=y_neg,
        pca_mahalanobis_label_01=y_neg,
        pca_spectral_entropy_label_01=y_neg,
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
        "pca_residual",
        "pca_mahalanobis",
        "pca_spectral_entropy",
        "pca_residual_label_01",
        "pca_mahalanobis_label_01",
        "pca_spectral_entropy_label_01",
    ]
    assert got.iloc[0]["model"] == "m"
    assert got.iloc[0]["layer"] == "L1"
    assert got.iloc[0]["sample_idx"] == 0
    assert got.iloc[0]["entity"] == "term_a"
    assert got.iloc[0]["cluster"] == 3
    assert float(got.iloc[0]["pca_residual"]) == 0.1
    assert float(got.iloc[0]["pca_mahalanobis"]) == 0.2
    assert float(got.iloc[0]["pca_spectral_entropy"]) == 0.15
    assert int(got.iloc[0]["pca_residual_label_01"]) == 0
    assert int(got.iloc[0]["pca_mahalanobis_label_01"]) == 0
    assert int(got.iloc[0]["pca_spectral_entropy_label_01"]) == 0


def test_fine_metadata_skips_pca_when_length_mismatch() -> None:
    report = _report_with_assignments(
        pd.DataFrame(
            {
                "pmid": [1, 2],
                "mention": ["a", "b"],
                "entity": ["term_a", "term_b"],
                "cluster": [0, 1],
            }
        )
    )
    res = np.array([0.1], dtype=np.float64)
    mah = np.array([0.2], dtype=np.float64)
    ent = np.array([0.15], dtype=np.float64)
    y_short = np.array([0], dtype=np.int64)
    report = ModelSelectionReport(
        hyperparameters=report.hyperparameters,
        best_score=report.best_score,
        number_properties=report.number_properties,
        n_clusters_emergent=report.n_clusters_emergent,
        metrics_df=report.metrics_df,
        assignments=report.assignments,
        pca_residuals=res,
        pca_mahalanobis=mah,
        pca_spectral_entropy=ent,
        pca_residual_label_01=y_short,
        pca_mahalanobis_label_01=y_short,
        pca_spectral_entropy_label_01=y_short,
        umap_clustering=report.umap_clustering,
        umap_visualization=report.umap_visualization,
        pca_reduced=report.pca_reduced,
        ari=report.ari,
    )
    got = _fine_clustering_metadata_df(
        report,
        model="m",
        layer="L1",
        sample_idx=0,
    )
    assert "pca_residual" not in got.columns


def test_fine_metadata_returns_empty_if_required_columns_missing() -> None:
    report = _report_with_assignments(pd.DataFrame({"entity": ["term_a"]}))
    got = _fine_clustering_metadata_df(
        report,
        model="m",
        layer="L1",
        sample_idx=1,
    )
    assert got.empty
