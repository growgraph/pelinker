"""Tests for fine clustering metadata extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pelinker.onto import NEGATIVE_LABEL
from pelinker.reporting import (
    ClusteringHyperparameters,
    ModelSelectionReport,
    entity_negative_label_mask_01,
)
from run.analysis.model_selection import (
    _fine_clustering_metadata_df,
    _validated_oov_label_series,
)


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
        oov_label=y_neg,
        umap_clustering=np.array([[0.0, 0.1]], dtype=np.float64),
        cluster_viz=np.array([[0.0, 0.1]], dtype=np.float64),
        cluster_viz_method="pca",
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
        "oov_label",
    ]
    assert got.iloc[0]["model"] == "m"
    assert got.iloc[0]["layer"] == "L1"
    assert got.iloc[0]["sample_idx"] == 0
    assert got.iloc[0]["entity"] == "term_a"
    assert got.iloc[0]["cluster"] == 3
    assert float(got.iloc[0]["pca_residual"]) == 0.1
    assert float(got.iloc[0]["pca_mahalanobis"]) == 0.2
    assert float(got.iloc[0]["pca_spectral_entropy"]) == 0.15
    assert int(got.iloc[0]["oov_label"]) == 0


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
    y_full = np.array([0, 0], dtype=np.int64)
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
        oov_label=y_full,
        umap_clustering=report.umap_clustering,
        cluster_viz=report.cluster_viz,
        cluster_viz_method=report.cluster_viz_method,
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


def test_fine_metadata_raises_when_oov_label_length_mismatch() -> None:
    report = _report_with_assignments(
        pd.DataFrame({"entity": ["term_a", "term_b"], "cluster": [0, 1]})
    )
    bad = ModelSelectionReport(
        hyperparameters=report.hyperparameters,
        best_score=report.best_score,
        number_properties=report.number_properties,
        n_clusters_emergent=report.n_clusters_emergent,
        metrics_df=report.metrics_df,
        assignments=report.assignments,
        pca_residuals=report.pca_residuals,
        pca_mahalanobis=report.pca_mahalanobis,
        pca_spectral_entropy=report.pca_spectral_entropy,
        oov_label=np.array([0], dtype=np.int64),
        umap_clustering=report.umap_clustering,
        cluster_viz=report.cluster_viz,
        cluster_viz_method=report.cluster_viz_method,
        pca_reduced=report.pca_reduced,
        ari=report.ari,
    )
    with pytest.raises(ValueError, match="len\\(oov_label\\)"):
        _fine_clustering_metadata_df(bad, model="m", layer="L1", sample_idx=0)


def test_fine_metadata_returns_empty_if_required_columns_missing() -> None:
    report = _report_with_assignments(pd.DataFrame({"entity": ["term_a"]}))
    got = _fine_clustering_metadata_df(
        report,
        model="m",
        layer="L1",
        sample_idx=1,
    )
    assert got.empty


def test_validated_oov_label_series_raises_for_all_zeroes() -> None:
    df = pd.DataFrame({"oov_label": [0, 0, 0]})
    with pytest.raises(ValueError, match="trivial oov_label mask"):
        _validated_oov_label_series(df, source_name="unit")


def test_validated_oov_label_series_raises_for_all_ones() -> None:
    df = pd.DataFrame({"oov_label": [1, 1, 1]})
    with pytest.raises(ValueError, match="trivial oov_label mask"):
        _validated_oov_label_series(df, source_name="unit")


def test_validated_oov_label_series_raises_for_non_binary() -> None:
    df = pd.DataFrame({"oov_label": [0, 2, 1]})
    with pytest.raises(ValueError, match="must be binary"):
        _validated_oov_label_series(df, source_name="unit")


def test_validated_oov_label_series_accepts_both_classes() -> None:
    df = pd.DataFrame({"oov_label": [0, 1, 0, 1]})
    got = _validated_oov_label_series(df, source_name="unit")
    assert got.tolist() == [0, 1, 0, 1]


def test_fine_metadata_prefers_mention_quality_with_mixed_oov_label() -> None:
    mq = pd.DataFrame(
        {
            "entity": ["term_a", NEGATIVE_LABEL],
            "pmid": [1, 2],
            "mention": ["a", "b"],
            "cluster": [0, -1],
            "oov_label": [0, 1],
            "pca_residual": [0.1, 0.2],
            "pca_mahalanobis": [0.3, 0.4],
            "pca_spectral_entropy": [0.5, 0.6],
        }
    )
    report = ModelSelectionReport(
        hyperparameters=ClusteringHyperparameters(min_cluster_size=10),
        best_score=0.5,
        number_properties=1,
        n_clusters_emergent=1,
        metrics_df=pd.DataFrame({"min_cluster_size": [10], "dbcv": [0.5]}),
        assignments=pd.DataFrame({"entity": ["term_a"], "cluster": [0]}),
        pca_residuals=np.array([0.1], dtype=np.float64),
        pca_mahalanobis=np.array([0.2], dtype=np.float64),
        pca_spectral_entropy=np.array([0.15], dtype=np.float64),
        oov_label=np.array([0], dtype=np.int64),
        umap_clustering=np.array([[0.0]], dtype=np.float64),
        cluster_viz=np.array([[0.0]], dtype=np.float64),
        cluster_viz_method="pca",
        pca_reduced=np.array([[0.0]], dtype=np.float64),
        ari=0.1,
        mention_quality=mq,
    )
    got = _fine_clustering_metadata_df(
        report,
        model="m",
        layer="L1",
        sample_idx=0,
    )
    assert len(got) == 2
    assert set(got["oov_label"].tolist()) == {0, 1}
    _validated_oov_label_series(got, source_name="unit")
