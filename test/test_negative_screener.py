"""Negative-class LDA/SVM screener and manifold split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pelinker.analysis import (
    fit_ambient_screener_with_metrics,
    split_by_negative_label,
)
from pelinker.selection import evaluate_selection_sample
from pelinker.config import ClusteringOptimizationConfig, NegativeScreenerConfig
from pelinker.screener.ambient_screener import NegativeClassScreener
from pelinker.onto import NEGATIVE_LABEL
from pelinker.reporting import (
    AllScreenerCvResult,
    BinaryClassifierMetrics,
    ClusteringSearchSummaryRow,
    HyperparameterSearchStats,
    MeanWithUncertainty,
    MetricMeanStd,
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


def _bmc(
    pm: float,
    ps: float,
    rm: float,
    rs: float,
    fm: float,
    fs: float,
    am: float,
    ais: float,
) -> BinaryClassifierMetrics:
    return BinaryClassifierMetrics(
        precision=MetricMeanStd(pm, ps),
        recall=MetricMeanStd(rm, rs),
        f1=MetricMeanStd(fm, fs),
        auc=MetricMeanStd(am, ais),
    )


def test_evaluate_selection_sample_ambient_screener_cv_and_manifold_only_positives() -> (
    None
):
    dfr = _tiny_frame(n_pos=80)
    opt = ClusteringOptimizationConfig(
        drop_rare_entities=True,
        min_mentions_per_entity=5,
        max_scale=30,
        min_scale=5,
        clustering_grid_step=5,
        ambient_screener=NegativeScreenerConfig(
            cv_n_splits=5,
            cv_test_size=0.25,
            cv_random_state=0,
        ),
    )
    tc = TransformConfig(
        pca_components=6,
        umap_components=3,
        cluster_viz_components=3,
        pca_seed=0,
        umap_seed=0,
    )
    report = evaluate_selection_sample(
        dfr,
        tc,
        optimization_config=opt,
        entities_pre_filtered=False,
    )
    assert report is not None
    assert report.all_screener_cv is not None
    assert report.all_screener_cv.screener_lda.f1.mean >= 0.0
    assert len(report.assignments) == len(dfr[dfr["entity"] != NEGATIVE_LABEL])
    assert report.number_properties == 2
    assert NEGATIVE_LABEL not in set(report.assignments["entity"].astype(str))
    assert report.mention_quality is not None
    assert len(report.mention_quality) == len(dfr)
    assert set(np.unique(report.mention_quality["oov_label"]).tolist()) == {0, 1}
    assert (
        report.mention_quality.loc[
            report.mention_quality["entity"] == NEGATIVE_LABEL, "cluster"
        ]
        == -1
    ).all()
    assert NEGATIVE_LABEL in set(report.mention_quality["entity"].astype(str))


def test_split_by_negative_label_excludes_synthetic_rows() -> None:
    dfr = _tiny_frame()
    neg_mask, manifold = split_by_negative_label(dfr, NEGATIVE_LABEL)
    assert neg_mask.sum() == sum(dfr["entity"] == NEGATIVE_LABEL)
    assert len(manifold) == len(dfr) - neg_mask.sum()
    assert NEGATIVE_LABEL not in set(manifold["entity"].astype(str).unique())


def test_fit_ambient_screener_with_metrics_none_when_single_class() -> None:
    dfr = _tiny_frame(dim=4, n_pos=20, n_neg=0)
    cfg = NegativeScreenerConfig(kind="lda", negative_label=NEGATIVE_LABEL)
    scr, metrics = fit_ambient_screener_with_metrics(dfr, cfg)
    assert metrics is None
    assert scr._estimator is None


def test_fit_ambient_screener_with_metrics_matches_direct_fit() -> None:
    dfr = _tiny_frame(dim=4, n_pos=30, n_neg=20)
    cfg = NegativeScreenerConfig(kind="lda", negative_label=NEGATIVE_LABEL)
    scr_direct = NegativeClassScreener.fit_from_frame(dfr, cfg)
    scr_wrapped, metrics = fit_ambient_screener_with_metrics(dfr, cfg)
    assert metrics is not None
    X = np.stack(dfr["embed"].values).astype(np.float32, copy=False)
    assert np.array_equal(
        scr_direct.predict_is_negative(X),
        scr_wrapped.predict_is_negative(X),
    )


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
    lda = _bmc(0.7, 0.1, 0.6, 0.05, 0.65, 0.08, 0.74, 0.02)
    svm = _bmc(0.72, 0.09, 0.61, 0.04, 0.66, 0.07, 0.76, 0.02)
    sb = _bmc(0.71, 0.08, 0.62, 0.03, 0.66, 0.06, 0.77, 0.02)
    oov = _bmc(0.69, 0.02, 0.68, 0.02, 0.67, 0.02, 0.75, 0.02)
    cb = _bmc(0.72, 0.02, 0.7, 0.02, 0.7, 0.02, 0.78, 0.02)
    ac = AllScreenerCvResult(
        screener_lda=lda,
        screener_svm=svm,
        screener_best_kind="svm",
        screener_best=sb,
        oov_winner_kind="lda",
        oov=oov,
        combined=cb,
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
        all_screener_cv=ac,
    )
    flat = row.to_flat_dict()
    back = clustering_search_summary_row_from_flat_dict(flat)
    assert back.all_screener_cv is not None
    assert back.all_screener_cv.screener_lda.f1.mean == pytest.approx(0.65)
    assert back.all_screener_cv.screener_best_kind == "svm"
