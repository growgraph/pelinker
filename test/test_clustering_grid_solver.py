"""Tests for typed grid aggregation and smoothed min_cluster_size selection."""

import numpy as np
import pandas as pd
import pytest

from pelinker.clustering_grid import (
    AggregatedGridPoint,
    AggregatedGridReport,
    ScalarMetricAggregate,
    aggregate_grid_metrics,
    aggregated_grid_report_to_dataframe,
    solve_optimal_min_cluster_size_from_aggregated,
)


def _report_from_arrays(
    sizes: list[int],
    means: list[float],
    stds: list[float],
    counts: list[int],
) -> AggregatedGridReport:
    points = [
        AggregatedGridPoint(
            min_cluster_size=s,
            dbcv=ScalarMetricAggregate(mean=m, std=sd, count=c),
            icm_mean=float("nan"),
            n_clusters_mean=float("nan"),
        )
        for s, m, sd, c in zip(sizes, means, stds, counts, strict=True)
    ]
    return AggregatedGridReport(points=tuple(points))


def test_aggregate_grid_metrics_empty() -> None:
    r = aggregate_grid_metrics([])
    assert r.points == ()


def test_aggregate_grid_metrics_preserves_std_and_count() -> None:
    a = pd.DataFrame(
        {
            "min_cluster_size": [10, 20],
            "icm": [0.1, 0.2],
            "n_clusters": [3, 4],
            "dbcv": [0.5, 0.6],
        }
    )
    b = pd.DataFrame(
        {
            "min_cluster_size": [10, 20],
            "icm": [0.15, 0.25],
            "n_clusters": [3, 5],
            "dbcv": [0.7, 0.4],
        }
    )
    r = aggregate_grid_metrics([a, b])
    assert len(r.points) == 2
    p10 = next(p for p in r.points if p.min_cluster_size == 10)
    assert p10.dbcv.mean == pytest.approx(0.6)
    assert p10.dbcv.count == 2
    assert p10.dbcv.std == pytest.approx(np.std([0.5, 0.7], ddof=1))


def test_aggregated_grid_report_to_dataframe_columns() -> None:
    r = _report_from_arrays([5, 10], [1.0, 2.0], [0.1, 0.2], [3, 4])
    df = aggregated_grid_report_to_dataframe(r)
    assert list(df.columns) == [
        "min_cluster_size",
        "dbcv_mean",
        "dbcv_std",
        "dbcv_count",
        "icm_mean",
        "n_clusters_mean",
    ]
    assert df.iloc[0]["dbcv_count"] == 3


def test_aggregated_grid_report_rejects_unsorted_points() -> None:
    p1 = AggregatedGridPoint(
        10,
        ScalarMetricAggregate(1.0, 0.0, 1),
        0.0,
        0.0,
    )
    p2 = AggregatedGridPoint(
        5,
        ScalarMetricAggregate(1.0, 0.0, 1),
        0.0,
        0.0,
    )
    with pytest.raises(ValueError, match="sorted"):
        AggregatedGridReport(points=(p1, p2))


def test_solve_empty_report_raises() -> None:
    with pytest.raises(ValueError, match="No aggregated"):
        solve_optimal_min_cluster_size_from_aggregated(AggregatedGridReport(points=()))


def test_solve_plateau_prefers_leftmost_high_flat_region() -> None:
    """Sharp rise then flat top: derivative small on the right; pick early plateau x."""
    sizes = [10, 15, 20, 25, 30, 35, 40]
    means = [0.1, 0.2, 0.5, 1.5, 2.0, 2.01, 2.0]
    stds = [0.05] * len(sizes)
    counts = [10] * len(sizes)
    r = _report_from_arrays(sizes, means, stds, counts)
    out = solve_optimal_min_cluster_size_from_aggregated(
        r,
        method="mean",
        smooth_window=3,
        plateau_fraction=0.9,
        derivative_rel_tol=0.2,
        precision_weighted_smooth=False,
    )
    assert out.selection == "plateau_derivative"
    assert out.chosen_min_cluster_size in {25, 30, 35}
    assert out.score_mean_at_chosen == pytest.approx(
        means[sizes.index(out.chosen_min_cluster_size)]
    )


def test_solve_smoothed_argmax_when_no_plateau() -> None:
    """Strictly increasing objective: derivative stays positive → fall back to smoothed peak."""
    sizes = [10, 15, 20, 25, 30]
    means = [0.1, 0.4, 0.7, 1.0, 1.3]
    r = _report_from_arrays(sizes, means, [0.01] * 5, [5] * 5)
    out = solve_optimal_min_cluster_size_from_aggregated(
        r,
        method="mean",
        smooth_window=3,
        plateau_fraction=0.999,
        derivative_rel_tol=1e-9,
        precision_weighted_smooth=False,
    )
    assert out.selection == "smoothed_argmax"
    assert out.chosen_min_cluster_size == 30


def test_solve_lower_bound_objective() -> None:
    sizes = [10, 20]
    means = [0.2, 1.0]
    stds = [0.05, 0.01]
    r = _report_from_arrays(sizes, means, stds, [5, 5])
    out = solve_optimal_min_cluster_size_from_aggregated(
        r,
        method="lower_bound",
        uncertainty_penalty=1.0,
        smooth_window=1,
        plateau_fraction=0.9,
        derivative_rel_tol=1.0,
        precision_weighted_smooth=False,
    )
    assert out.chosen_min_cluster_size == 20


def test_solve_unknown_method_raises() -> None:
    r = _report_from_arrays([10], [1.0], [0.0], [1])
    with pytest.raises(ValueError, match="Unknown optimization method"):
        solve_optimal_min_cluster_size_from_aggregated(r, method="nope")


def test_user_style_noisy_sequence_reasonable_choice() -> None:
    """Noisy scores that rise then wiggle near a ceiling (similar to user example)."""
    sizes = list(range(10, 10 + 8 * 5, 5))
    means = [0.9, 0.8, 1.6, 2.2, 2.4, 2.3, 2.5, 2.6]
    stds = [0.15] * len(sizes)
    counts = [8] * len(sizes)
    r = _report_from_arrays(sizes, means, stds, counts)
    out = solve_optimal_min_cluster_size_from_aggregated(
        r,
        method="mean",
        smooth_window=3,
        plateau_fraction=0.88,
        derivative_rel_tol=0.25,
        precision_weighted_smooth=True,
    )
    assert out.chosen_min_cluster_size in set(sizes)
    assert out.score_mean_at_chosen in means
    assert len(out.dy_dx) == len(out.x)


def test_finite_mask_drops_non_finite_objective() -> None:
    sizes = [10, 15, 20]
    means = [float("nan"), 1.0, 2.0]
    r = _report_from_arrays(sizes, means, [0.0, 0.0, 0.0], [1, 1, 1])
    out = solve_optimal_min_cluster_size_from_aggregated(
        r,
        smooth_window=1,
        plateau_fraction=0.5,
        derivative_rel_tol=1.0,
        precision_weighted_smooth=False,
    )
    assert out.x == (15.0, 20.0)
