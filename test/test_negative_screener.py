"""Negative-class LDA/SVM screener and manifold split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pelinker.analysis import estimate_clustering_from_frame
from pelinker.config import ClusteringOptimizationConfig, NegativeScreenerConfig
from pelinker.negative_screener import NegativeClassScreener
from pelinker.onto import NEGATIVE_LABEL
from pelinker.reporting import (
    ClusteringSearchSummaryRow,
    HyperparameterSearchStats,
    MeanWithUncertainty,
    MetricMeanStd,
    NegativeScreenerCvSummary,
    ScreenerModelCvBlock,
    clustering_search_summary_row_from_flat_dict,
)
from pelinker.transform import TransformConfig


def _tiny_frame(*, dim: int = 8, n_pos: int = 40, n_neg: int = 15) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    pos = rng.standard_normal((n_pos, dim)).astype(np.float32)
    neg = rng.standard_normal((n_neg, dim)).astype(np.float32) + 1.5
    rows: list[dict[str, object]] = []
    for i in range(n_pos):
        rows.append(
            {
                "pmid": i,
                "entity": "kb_a" if i % 2 == 0 else "kb_b",
                "mention": f"p{i}",
                "embed": pos[i],
            }
        )
    for j in range(n_neg):
        rows.append(
            {
                "pmid": 1000 + j,
                "entity": NEGATIVE_LABEL,
                "mention": f"n{j}",
                "embed": neg[j],
            }
        )
    return pd.DataFrame(rows)


def test_estimate_clustering_negative_screener_cv_and_manifold_only_positives() -> None:
    dfr = _tiny_frame()
    opt = ClusteringOptimizationConfig(
        min_class_size=5,
        max_scale=30,
        min_scale=5,
        clustering_grid_step=10,
        negative_screener=NegativeScreenerConfig(
            cv_n_splits=5,
            cv_test_size=0.25,
            cv_random_state=0,
        ),
    )
    tc = TransformConfig(pca_components=6, umap_components=3, umap_viz_components=3)
    report = estimate_clustering_from_frame(
        dfr,
        tc,
        optimization_config=opt,
        aggregation_level="mention",
    )
    assert report is not None
    assert report.negative_screener_cv is not None
    assert report.negative_screener_cv.lda.f1.mean >= 0.0
    assert len(report.assignments) == len(dfr[dfr["entity"] != NEGATIVE_LABEL])
    assert report.number_properties == 2
    assert NEGATIVE_LABEL not in set(report.assignments["entity"].astype(str))


def test_negative_class_screener_trivial_single_class_all_non_negative() -> None:
    """No synthetic-negative rows → trivial screener; nothing predicted as negative."""
    dfr = _tiny_frame(dim=4, n_pos=20, n_neg=0)
    cfg = NegativeScreenerConfig(kind="lda", negative_label=NEGATIVE_LABEL)
    scr = NegativeClassScreener.fit_from_frame(dfr, cfg)
    X = np.stack(dfr["embed"].values)
    assert not scr.predict_is_negative(X).any()


def test_negative_class_screener_fit_predict() -> None:
    dfr = _tiny_frame(dim=4, n_pos=30, n_neg=20)
    cfg = NegativeScreenerConfig(kind="lda", negative_label=NEGATIVE_LABEL)
    scr = NegativeClassScreener.fit_from_frame(dfr, cfg)
    assert scr is not None
    X = np.stack(dfr["embed"].values)
    mask = scr.predict_is_negative(X)
    assert mask.sum() >= 1
    margin = scr.decision_function(X)
    assert margin.shape == (len(dfr),)


def test_clustering_search_summary_screener_flat_round_trip() -> None:
    ns = NegativeScreenerCvSummary(
        lda=ScreenerModelCvBlock(
            precision=MetricMeanStd(0.7, 0.1),
            recall=MetricMeanStd(0.6, 0.05),
            f1=MetricMeanStd(0.65, 0.08),
        ),
        svm=ScreenerModelCvBlock(
            precision=MetricMeanStd(0.72, 0.09),
            recall=MetricMeanStd(0.61, 0.04),
            f1=MetricMeanStd(0.66, 0.07),
        ),
    )
    row = ClusteringSearchSummaryRow(
        model="m",
        layer="l",
        hyperparameters=HyperparameterSearchStats(
            min_cluster_size=MeanWithUncertainty(mean=10.0, std=0.0),
        ),
        number_properties=MeanWithUncertainty(mean=5.0, std=0.0),
        n_clusters_emergent=MeanWithUncertainty(mean=3.0, std=0.0),
        dbcv=MeanWithUncertainty(mean=0.4, std=0.0),
        ari=MeanWithUncertainty(mean=0.3, std=0.0),
        negative_screener_cv=ns,
    )
    flat = row.to_flat_dict()
    back = clustering_search_summary_row_from_flat_dict(flat)
    assert back.negative_screener_cv is not None
    assert back.negative_screener_cv.lda.f1.mean == pytest.approx(0.65)
