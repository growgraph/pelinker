"""Tests for typed grid aggregation and min_cluster_size selection."""

import numpy as np
import pandas as pd
import pytest

from pelinker.clustering.grid import (
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
    *,
    n_clusters: list[float] | None = None,
) -> AggregatedGridReport:
    if n_clusters is None:
        n_clusters = [float("nan")] * len(sizes)
    points = [
        AggregatedGridPoint(
            min_cluster_size=s,
            dbcv=ScalarMetricAggregate(mean=m, std=sd, count=c),
            icm_mean=float("nan"),
            n_clusters_mean=float(nc),
            ari=ScalarMetricAggregate(mean=float("nan"), std=0.0, count=0),
        )
        for s, m, sd, c, nc in zip(sizes, means, stds, counts, n_clusters, strict=True)
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
            "ari": [0.3, 0.4],
        }
    )
    b = pd.DataFrame(
        {
            "min_cluster_size": [10, 20],
            "icm": [0.15, 0.25],
            "n_clusters": [3, 5],
            "dbcv": [0.7, 0.4],
            "ari": [0.5, 0.35],
        }
    )
    r = aggregate_grid_metrics([a, b])
    assert len(r.points) == 2
    p10 = next(p for p in r.points if p.min_cluster_size == 10)
    assert p10.dbcv.mean == pytest.approx(0.6)
    assert p10.dbcv.count == 2
    assert p10.dbcv.std == pytest.approx(np.std([0.5, 0.7], ddof=1))
    assert p10.ari.mean == pytest.approx(0.4)
    assert p10.ari.count == 2
    assert p10.ari.std == pytest.approx(np.std([0.3, 0.5], ddof=1))


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
        "ari_mean",
        "ari_std",
        "ari_count",
    ]
    assert df.iloc[0]["dbcv_count"] == 3


def test_aggregated_grid_report_rejects_unsorted_points() -> None:
    p1 = AggregatedGridPoint(
        10,
        ScalarMetricAggregate(1.0, 0.0, 1),
        0.0,
        0.0,
        ScalarMetricAggregate(float("nan"), 0.0, 0),
    )
    p2 = AggregatedGridPoint(
        5,
        ScalarMetricAggregate(1.0, 0.0, 1),
        0.0,
        0.0,
        ScalarMetricAggregate(float("nan"), 0.0, 0),
    )
    with pytest.raises(ValueError, match="sorted"):
        AggregatedGridReport(points=(p1, p2))


def test_solve_empty_report_raises() -> None:
    with pytest.raises(ValueError, match="No aggregated"):
        solve_optimal_min_cluster_size_from_aggregated(AggregatedGridReport(points=()))


def test_solve_takes_argmax_when_nothing_ties() -> None:
    """Well-separated peak with tight errors: the one-SE rule must not wander off it."""
    sizes = [10, 15, 20, 25, 30]
    means = [0.1, 0.4, 0.7, 1.0, 1.3]
    r = _report_from_arrays(sizes, means, [0.01] * 5, [5] * 5)
    out = solve_optimal_min_cluster_size_from_aggregated(
        r, objective="dbcv", smooth_window=1
    )
    assert out.selection == "argmax"
    assert out.chosen_min_cluster_size == 30
    assert out.argmax_min_cluster_size == 30


def test_solve_slides_right_across_a_statistical_tie() -> None:
    """Flat top within noise: prefer the largest tying min_cluster_size (parsimony)."""
    sizes = [10, 15, 20, 25, 30, 35, 40]
    means = [0.1, 0.2, 0.5, 1.5, 2.00, 2.01, 2.00]
    r = _report_from_arrays(sizes, means, [0.30] * len(sizes), [10] * len(sizes))
    out = solve_optimal_min_cluster_size_from_aggregated(
        r, objective="dbcv", smooth_window=1, one_se_k=1.0
    )
    assert out.argmax_min_cluster_size == 35
    assert out.chosen_min_cluster_size == 40
    assert out.selection == "one_se_paired"


def test_one_se_never_promotes_a_worse_mean() -> None:
    """The old ``mean - k*std`` discount let a low-variance, low-mean point win. It must not."""
    sizes = [10, 20]
    means = [0.2, 1.0]
    stds = [0.05, 0.40]  # the *worse* point is far more precise
    r = _report_from_arrays(sizes, means, stds, [5, 5])
    out = solve_optimal_min_cluster_size_from_aggregated(
        r, objective="dbcv", smooth_window=1
    )
    assert out.chosen_min_cluster_size == 20


def test_one_se_k_zero_is_pure_argmax() -> None:
    sizes = [10, 20, 30]
    r = _report_from_arrays(sizes, [1.0, 2.0, 1.99], [0.5] * 3, [8] * 3)
    tied = solve_optimal_min_cluster_size_from_aggregated(
        r, objective="dbcv", smooth_window=1, one_se_k=1.0
    )
    strict = solve_optimal_min_cluster_size_from_aggregated(
        r, objective="dbcv", smooth_window=1, one_se_k=0.0
    )
    assert tied.chosen_min_cluster_size == 30
    assert strict.chosen_min_cluster_size == 20
    assert strict.selection == "argmax"


def test_contiguity_blocks_an_isolated_far_point() -> None:
    """A wide-error point past a clearly worse one is not part of the plateau."""
    sizes = [10, 20, 30, 40]
    means = [1.0, 2.0, 0.1, 1.95]
    stds = [0.05, 0.05, 0.05, 0.05]
    r = _report_from_arrays(sizes, means, stds, [50] * 4)
    contig = solve_optimal_min_cluster_size_from_aggregated(
        r, objective="dbcv", smooth_window=1, one_se_k=5.0, one_se_contiguous=True
    )
    loose = solve_optimal_min_cluster_size_from_aggregated(
        r, objective="dbcv", smooth_window=1, one_se_k=5.0, one_se_contiguous=False
    )
    assert contig.chosen_min_cluster_size == 20
    assert loose.chosen_min_cluster_size == 40


def test_one_se_rule_is_skipped_below_min_samples() -> None:
    """An SE estimated from 3 numbers is mostly noise; acting on it hurts reproducibility."""
    sizes = [10, 20, 30]
    means = [1.0, 2.0, 1.99]
    r = _report_from_arrays(sizes, means, [0.5] * 3, [3] * 3)
    out = solve_optimal_min_cluster_size_from_aggregated(
        r, objective="dbcv", smooth_window=1, one_se_k=1.0, one_se_min_samples=6
    )
    assert out.chosen_min_cluster_size == 20
    assert out.selection == "argmax"
    assert out.one_se_k == 0.0  # records the k actually applied

    enough = _report_from_arrays(sizes, means, [0.5] * 3, [8] * 3)
    slid = solve_optimal_min_cluster_size_from_aggregated(
        enough, objective="dbcv", smooth_window=1, one_se_k=1.0, one_se_min_samples=6
    )
    assert slid.chosen_min_cluster_size == 30


def test_solve_unknown_objective_raises() -> None:
    r = _report_from_arrays([10], [1.0], [0.0], [1])
    with pytest.raises(ValueError, match="Unknown grid objective"):
        solve_optimal_min_cluster_size_from_aggregated(
            r,
            objective="not_an_objective",  # type: ignore[arg-type]
        )


def test_cluster_count_reward_prefers_more_clusters_on_flat_dbcv() -> None:
    """When DBCV is flat, the log cluster reward should favour smaller min_cluster_size."""
    sizes = [20, 40, 60, 80]
    means = [0.70, 0.71, 0.69, 0.70]
    n_clusters = [120.0, 80.0, 60.0, 50.0]
    r = _report_from_arrays(sizes, means, [0.02] * 4, [5] * 4, n_clusters=n_clusters)
    without = solve_optimal_min_cluster_size_from_aggregated(
        r, objective="dbcv", smooth_window=1, cluster_count_reward=0.0
    )
    with_reward = solve_optimal_min_cluster_size_from_aggregated(
        r, objective="dbcv", smooth_window=1, cluster_count_reward=0.15
    )
    assert without.chosen_min_cluster_size >= with_reward.chosen_min_cluster_size
    assert with_reward.argmax_min_cluster_size == 20
    assert with_reward.y_cluster_term[0] == pytest.approx(0.0)
    assert all(t <= 0.0 for t in with_reward.y_cluster_term)


def test_pooled_grid_solve_from_metrics_dfs_returns_y_objective() -> None:
    from pelinker.search.grid_solver import pooled_grid_solve_from_metrics_dfs

    sizes = [20, 40, 60]
    df = pd.DataFrame(
        {
            "min_cluster_size": sizes,
            "icm": [0.1, 0.1, 0.1],
            "n_clusters": [120.0, 80.0, 50.0],
            "dbcv": [0.70, 0.71, 0.69],
            "ari": [0.9, 0.91, 0.92],
        }
    )
    solved = pooled_grid_solve_from_metrics_dfs([df], optimization_config=None)
    assert len(solved.y_objective) == len(solved.x)
    assert len(solved.y_se) == len(solved.x)
    assert len(solved.y_eligible) == len(solved.x)
    assert solved.chosen_min_cluster_size in sizes


def test_cluster_count_reward_negative_raises() -> None:
    r = _report_from_arrays([10], [1.0], [0.0], [1], n_clusters=[5.0])
    with pytest.raises(ValueError, match="cluster_count_reward"):
        solve_optimal_min_cluster_size_from_aggregated(r, cluster_count_reward=-0.1)


def test_finite_mask_drops_non_finite_objective() -> None:
    sizes = [10, 15, 20]
    means = [float("nan"), 1.0, 2.0]
    r = _report_from_arrays(sizes, means, [0.0, 0.0, 0.0], [1, 1, 1])
    out = solve_optimal_min_cluster_size_from_aggregated(
        r, objective="dbcv", smooth_window=1
    )
    assert out.x == (15.0, 20.0)
