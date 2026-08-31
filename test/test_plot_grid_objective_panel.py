"""Tests for grid objective panel on error-bar metric plots."""

import pathlib

import pandas as pd

from matplotlib import pyplot as plt

from pelinker.clustering.grid import SmoothedGridOptimumResult
from pelinker.plotting import plot_metrics, plot_metrics_with_error_bars


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
        selection="one_se_paired",
        argmax_min_cluster_size=30,
        y_se=(0.02, 0.0, 0.03),
        y_eligible=(False, True, True),
        one_se_k=1.0,
        n_samples=5,
    )


def _objective_axis(monkeypatch) -> list[plt.Axes]:
    """Capture the figures a plot call closes, so panels can be inspected after the fact."""
    captured: list[plt.Figure] = []
    real_close = plt.close

    def _spy(fig):
        if hasattr(fig, "axes"):
            captured.append(fig)
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _spy)
    return captured


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


def test_objective_panel_is_populated_on_the_live_run_path(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    """Regression: runners pass ``chosen_min_cluster_size`` and no ``grid_solve``.

    That combination used to short-circuit the solve and leave the fourth axis empty with
    an "(unavailable)" title in every real run — only the replot path rendered it.
    """
    figs = _objective_axis(monkeypatch)
    plot_metrics_with_error_bars(
        _synthetic_metrics_list(),
        tmp_path / "live.png",
        chosen_min_cluster_size=20.0,
    )
    assert figs, "no figure captured"
    ax_obj = figs[-1].axes[3]
    assert "unavailable" not in ax_obj.get_title().lower()
    assert ax_obj.lines, "objective panel has no curves"


def test_single_sample_plot_metrics_also_gets_an_objective_panel(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    figs = _objective_axis(monkeypatch)
    plot_metrics(_synthetic_metrics_list()[0], tmp_path / "single.png")
    assert figs
    fig = figs[-1]
    assert len(fig.axes) == 4
    ax_obj = fig.axes[3]
    assert "unavailable" not in ax_obj.get_title().lower()
    assert ax_obj.lines


def test_plot_metrics_with_error_bars_grid_solve_sets_vline(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    figs = _objective_axis(monkeypatch)
    out = tmp_path / "metrics.png"
    solve = _synthetic_grid_solve()
    plot_metrics_with_error_bars(_synthetic_metrics_list(), out, grid_solve=solve)
    assert out.exists()
    # The vline marks the solver's choice on every panel.
    xs = {
        round(line.get_xdata()[0], 6)
        for line in figs[-1].axes[0].lines
        if len(set(line.get_xdata())) == 1
    }
    assert float(solve.chosen_min_cluster_size) in xs
