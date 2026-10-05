"""Statistical properties of the clipped-geomean objective and the paired one-SE rule.

These pin the two problems the previous selector had: an objective whose ranking depended
on the arbitrary scales of DBCV and ARI, and an uncertainty discount that could promote a
grid point with a genuinely worse mean.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from pelinker.clustering.grid import (
    build_paired_grid_scores,
    combine_clipped_geomean,
    geomean_std,
    solve_optimal_min_cluster_size_from_metrics_dfs as solve,
)

GRID = list(range(10, 60, 5))


def _sample(
    dbcv: list[float] | np.ndarray,
    ari: list[float] | np.ndarray,
    *,
    grid: list[int] | None = None,
    n_clusters: list[float] | None = None,
) -> pd.DataFrame:
    g = grid if grid is not None else GRID
    return pd.DataFrame(
        {
            "min_cluster_size": g,
            "icm": [0.1] * len(g),
            "n_clusters": n_clusters if n_clusters is not None else [50.0] * len(g),
            "dbcv": list(dbcv),
            "ari": list(ari),
        }
    )


def _curves(seed: int, *, shared_offset_sd: float = 0.0) -> pd.DataFrame:
    """DBCV saturating with size, ARI peaking mid-grid, plus a whole-curve sample offset."""
    r = np.random.default_rng(seed)
    x = np.array(GRID, dtype=float)
    off = r.normal(0.0, shared_offset_sd)
    dbcv = 0.20 + 0.30 * (1 - np.exp(-(x - 10) / 12)) + off + r.normal(0, 0.01, x.size)
    ari = (
        0.30 + 0.10 * np.exp(-(((x - 30) / 14) ** 2)) + off + r.normal(0, 0.01, x.size)
    )
    return _sample(dbcv, ari)


# --------------------------------------------------------------------------- combiner


def test_geomean_is_conjunctive() -> None:
    """A high DBCV must not buy off a near-zero ARI, as an arithmetic mean would."""
    lopsided = combine_clipped_geomean(np.array([0.9]), np.array([0.01]))[0]
    balanced = combine_clipped_geomean(np.array([0.4]), np.array([0.35]))[0]
    assert lopsided < balanced
    # The arithmetic mean gets this backwards, which is what the old objective used.
    assert 0.5 * (0.9 + 0.01) > 0.5 * (0.4 + 0.35)


def test_geomean_clips_negative_metrics_to_zero() -> None:
    assert combine_clipped_geomean(np.array([-0.3]), np.array([0.5]))[0] == 0.0
    assert np.isnan(combine_clipped_geomean(np.array([np.nan]), np.array([0.5]))[0])


def test_geomean_std_uses_relative_errors() -> None:
    d, a, sd, sa = 0.36, 0.64, 0.036, 0.128
    got = geomean_std(np.array([d]), np.array([sd]), np.array([a]), np.array([sa]))[0]
    expected = 0.5 * np.sqrt(d * a) * np.sqrt((sd / d) ** 2 + (sa / a) ** 2)
    assert got == pytest.approx(expected)
    # A pinned-at-zero score carries no uncertainty.
    assert (
        geomean_std(np.array([0.0]), np.array([0.1]), np.array([0.5]), np.array([0.1]))[
            0
        ]
        == 0.0
    )


# ------------------------------------------------------------------- scale invariance


def test_chosen_mcs_is_invariant_to_rescaling_either_metric() -> None:
    """The whole point of the geometric mean: no normalization step is needed."""
    dfs = [_curves(s) for s in range(6)]
    base = solve(dfs).chosen_min_cluster_size
    for fd, fa in [(10.0, 0.1), (0.25, 4.0), (100.0, 100.0)]:
        scaled = [d.assign(dbcv=d.dbcv * fd, ari=d.ari * fa) for d in dfs]
        assert solve(scaled).chosen_min_cluster_size == base, (fd, fa)


def test_rescaling_reorders_the_arithmetic_mean_it_replaced() -> None:
    """Guards the premise: an unweighted average really is scale-dependent."""
    d = np.array([0.60, 0.10])
    a = np.array([0.10, 0.50])
    assert np.argmax(0.5 * (d + a)) == 0
    assert np.argmax(0.5 * (d * 0.1 + a)) == 1  # rescale DBCV -> different winner
    # The geometric mean's ranking survives the same rescaling.
    assert np.argmax(combine_clipped_geomean(d, a)) == np.argmax(
        combine_clipped_geomean(d * 0.1, a)
    )


# ------------------------------------------------------------------------ one-SE rule


def test_pairing_cancels_the_shared_per_sample_offset() -> None:
    """Paired SEs must ignore variance common to every grid point."""
    quiet = [_curves(s, shared_offset_sd=0.0) for s in range(8)]
    noisy = [_curves(s, shared_offset_sd=0.15) for s in range(8)]

    se_quiet = np.max(solve(quiet).y_se)
    se_noisy = np.max(solve(noisy).y_se)
    # A 0.15 whole-curve offset would dominate an unpaired SE (~0.15/sqrt(8) = 0.053);
    # paired, it barely registers.
    assert se_noisy < 4 * se_quiet
    assert se_noisy < 0.02
    assert solve(noisy).chosen_min_cluster_size == solve(quiet).chosen_min_cluster_size


def test_one_se_is_stable_under_leave_one_sample_out() -> None:
    dfs = [_curves(s, shared_offset_sd=0.12) for s in range(8)]
    chosen = {
        solve([d for j, d in enumerate(dfs) if j != i]).chosen_min_cluster_size
        for i in range(len(dfs))
    }
    assert len(chosen) == 1, f"choice moved across leave-one-out folds: {chosen}"


def test_slide_is_disabled_below_min_samples() -> None:
    """Measured on this repo's grid exports: below ~6 samples the slide adds variance.

    Leave-one-out over 3-sample exports gave sd 2.68 with the slide vs 1.59 without, because
    a standard error estimated from two folds is itself mostly noise.
    """
    few = [_curves(s, shared_offset_sd=0.10) for s in range(4)]
    guarded = solve(few, one_se_k=1.0, one_se_min_samples=6)
    argmax = solve(few, one_se_k=0.0)
    assert guarded.chosen_min_cluster_size == argmax.chosen_min_cluster_size
    assert guarded.selection == "argmax"
    assert guarded.one_se_k == 0.0

    many = [_curves(s, shared_offset_sd=0.10) for s in range(8)]
    assert solve(many, one_se_k=1.0, one_se_min_samples=6).one_se_k == 1.0


def test_single_sample_degenerates_to_argmax() -> None:
    """The per-sample fit path (search/selection.py) solves on exactly one sample."""
    out = solve([_curves(0)])
    assert out.selection == "argmax"
    assert out.n_samples == 1
    assert max(out.y_se) == 0.0
    assert out.chosen_min_cluster_size == out.argmax_min_cluster_size


# --------------------------------------------------------------------- ragged / NaN


def test_ragged_grids_reindex_onto_the_union() -> None:
    """Samples drop grid points that produced no clusters; the grid must not shrink."""
    short = _sample([0.5] * 6, [0.4] * 6, grid=GRID[:6])
    full = _curves(1)
    out = solve([short, full])
    assert [int(v) for v in out.x] == GRID


def test_nan_metrics_do_not_poison_the_pooled_curve() -> None:
    a = _curves(2)
    b = _curves(3)
    b.loc[b.min_cluster_size < 25, "dbcv"] = np.nan
    out = solve([a, b])
    assert np.all(np.isfinite(out.y_smooth))
    assert len(out.x) == len(GRID)


# ------------------------------------------------------------------------ degenerate


def test_all_zero_ari_falls_back_to_dbcv_with_a_warning(caplog) -> None:
    dfs = [_curves(s).assign(ari=0.0) for s in range(4)]
    with caplog.at_level(logging.WARNING, logger="pelinker.clustering.grid"):
        out = solve(dfs)
    assert out.selection == "degenerate_fallback"
    assert "flat" in caplog.text.lower()


def test_missing_ari_falls_back_to_dbcv_only() -> None:
    dfs = [_curves(s).assign(ari=np.nan) for s in range(4)]
    paired = build_paired_grid_scores(dfs)
    assert paired.objective == "dbcv"
    assert not paired.degenerate  # expected, not an alarm
    assert np.all(np.isfinite(solve(dfs).y_smooth))


def test_negative_dbcv_is_clipped_and_counted() -> None:
    dfs = [_curves(s) for s in range(3)]
    dfs[0].loc[dfs[0].min_cluster_size < 20, "dbcv"] = -0.4
    paired = build_paired_grid_scores(dfs)
    assert paired.n_clipped == 2
    assert np.all(paired.scores[0, :2] == 0.0)
