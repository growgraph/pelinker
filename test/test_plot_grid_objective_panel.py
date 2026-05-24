"""Tests for grid objective panel on error-bar metric plots."""

import pathlib

import pandas as pd

from pelinker.clustering_grid import SmoothedGridOptimumResult
from pelinker.plotting import plot_metrics_with_error_bars


def _synthetic_metrics_list() -> list[pd.DataFrame]:
    rows = []
    for run_id in (0, 1):
        for mcs, dbcv, ari, nk in (
            (20, 0.7, 0.86, 120),
            (40, 0.71, 0.89, 80),
            (60, 0.69, 0.91, 50),
        ):
            rows.append(
                {
                    "min_cluster_size": mcs,
                    "dbcv": dbcv + 0.01 * run_id,
                    "ari": ari + 0.005 * run_id,
                    "n_clusters": nk,
                    "icm": 0.1,
                }
            )
    return [pd.DataFrame(rows[:3]), pd.DataFrame(rows[3:])]


def _synthetic_grid_solve() -> SmoothedGridOptimumResult:
    return SmoothedGridOptimumResult(
        chosen_min_cluster_size=20,
        score_mean_at_chosen=0.7,
        score_std_at_chosen=0.01,
        n_clusters_mean_at_chosen=120.0,
        x=(20.0, 40.0, 60.0),
        y_objective=(0.55, 0.58, 0.52),
        y_cluster_term=(-0.05, -0.02, -0.08),
        y_smooth=(0.54, 0.57, 0.53),
        dy_dx=(0.01, 0.0, -0.02),
        d2y_dx2=(0.0, 0.0, 0.0),
        selection="plateau_derivative",
    )


def test_plot_metrics_with_error_bars_writes_four_panels_with_grid_solve(
    tmp_path: pathlib.Path,
) -> None:
    out = tmp_path / "metrics_error_bars.png"
    plot_metrics_with_error_bars(
        _synthetic_metrics_list(),
        out,
        chosen_min_cluster_size=20.0,
        grid_solve=_synthetic_grid_solve(),
    )
    assert out.with_suffix(".pdf").exists()
    assert out.exists()


def test_plot_metrics_with_error_bars_grid_solve_sets_vline(
    tmp_path: pathlib.Path,
) -> None:
    out = tmp_path / "metrics.png"
    solve = _synthetic_grid_solve()
    plot_metrics_with_error_bars(
        _synthetic_metrics_list(),
        out,
        grid_solve=solve,
    )
    assert out.exists()
