"""Tests for shared outer DBCV+ARI ranking."""

from __future__ import annotations

import pandas as pd
import pytest

from pelinker.clustering.ranking import (
    OUTER_SCORE_COL,
    attach_outer_scores,
    outer_score_from_summary_row,
    pick_best_row,
    raw_outer_dbcv_ari_score,
)
from pelinker.reports.schema import (
    ClusteringSearchSummaryRow,
    HyperparameterSearchStats,
    MeanWithUncertainty,
)


def test_raw_outer_is_the_clipped_geomean() -> None:
    score, std = raw_outer_dbcv_ari_score(0.8, 0.4, dbcv_std=0.0, ari_std=0.0)
    assert score == pytest.approx((0.8 * 0.4) ** 0.5)
    assert std == 0.0


def test_raw_outer_clips_negative_dbcv_to_zero() -> None:
    score, _ = raw_outer_dbcv_ari_score(-0.2, 0.5)
    assert score == 0.0


def test_raw_outer_falls_back_to_dbcv() -> None:
    score, _ = raw_outer_dbcv_ari_score(0.7, None)
    assert score == 0.7


def test_attach_outer_breaks_the_tie_minmax_used_to_manufacture() -> None:
    """Per-curve min-max mapped both rows to (1,0) and (0,1), scoring both 0.5."""
    df = pd.DataFrame(
        {
            "best_score": [0.9, 0.5],
            "best_score_std": [0.0, 0.0],
            "ari": [0.1, 0.9],
            "ari_std": [0.0, 0.0],
            "model": ["a", "b"],
            "layer": ["1", "1"],
        }
    )
    scored = attach_outer_scores(df)
    assert scored[OUTER_SCORE_COL].tolist() == pytest.approx(
        [(0.9 * 0.1) ** 0.5, (0.5 * 0.9) ** 0.5]
    )
    # b is better on both metrics jointly; the geometric mean says so.
    assert scored[OUTER_SCORE_COL][1] > scored[OUTER_SCORE_COL][0]


def test_outer_scores_do_not_depend_on_other_candidates() -> None:
    """Min-max across the leaderboard let an irrelevant row reorder the rest."""
    base = pd.DataFrame(
        {
            "best_score": [0.6, 0.4],
            "best_score_std": [0.0, 0.0],
            "ari": [0.3, 0.5],
            "ari_std": [0.0, 0.0],
        }
    )
    extra = pd.concat(
        [
            base,
            pd.DataFrame(
                {
                    "best_score": [0.01],
                    "best_score_std": [0.0],
                    "ari": [0.99],
                    "ari_std": [0.0],
                }
            ),
        ],
        ignore_index=True,
    )
    assert attach_outer_scores(base)[OUTER_SCORE_COL].tolist() == pytest.approx(
        attach_outer_scores(extra)[OUTER_SCORE_COL].tolist()[:2]
    )


def test_pick_best_row_breaks_ties_by_model() -> None:
    df = pd.DataFrame(
        {
            "best_score": [0.5, 0.5],
            "best_score_std": [0.0, 0.0],
            "ari": [0.5, 0.5],
            "ari_std": [0.0, 0.0],
            "model": ["b", "a"],
            "layer": ["1", "1"],
        }
    )
    winner = pick_best_row(df, tie_break_cols=("model", "layer"))
    assert winner["model"] == "a"


def test_outer_score_from_summary_row() -> None:
    row = ClusteringSearchSummaryRow(
        model="m",
        layer="l",
        hyperparameters=HyperparameterSearchStats(
            min_cluster_size=MeanWithUncertainty(mean=10.0, std=0.0),
        ),
        number_properties=MeanWithUncertainty(mean=5.0, std=0.0),
        n_clusters_emergent=MeanWithUncertainty(mean=3.0, std=0.0),
        dbcv=MeanWithUncertainty(mean=0.8, std=0.0),
        ari=MeanWithUncertainty(mean=0.4, std=0.0),
    )
    assert outer_score_from_summary_row(row) == pytest.approx((0.8 * 0.4) ** 0.5)
