"""Unit tests for the empirical min_cluster_size scaling curve."""

from __future__ import annotations

import math

import pytest

from pelinker.core.scaling import (
    MIN_RUNGS_FOR_FIT,
    ScaleCurve,
    ScaleRung,
    extrapolate_min_cluster_size,
    fit_scale_curve,
)


def _rung(n: int, mcs: int, **kw) -> ScaleRung:
    return ScaleRung(
        n_rows_realized=n,
        chosen_min_cluster_size=mcs,
        score_at_chosen=kw.pop("score", 0.5),
        n_clusters_mean=kw.pop("n_clusters_mean", 40.0),
        n_sample=kw.pop("n_sample", 3),
        **kw,
    )


# MCS is an integer, so synthetic rungs are quantized. Keeping the generated values
# well above ~30 keeps the rounding error small relative to the signal; a smaller
# intercept would make rounding, not the exponent, dominate the fit.
_INTERCEPT = -1.0
_SLOPE = 0.5


def _true_mcs(n: int, *, intercept: float = _INTERCEPT, slope: float = _SLOPE) -> float:
    return math.exp(intercept + slope * math.log(n))


def _rungs_from_power_law(
    ns: list[int], *, intercept: float = _INTERCEPT, slope: float = _SLOPE
) -> list[ScaleRung]:
    """Rungs lying on ``MCS = exp(intercept) * N**slope``, rounded to integers."""
    return [
        _rung(n, max(2, int(round(_true_mcs(n, intercept=intercept, slope=slope)))))
        for n in ns
    ]


def test_fit_recovers_a_known_exponent() -> None:
    ns = [10_000, 25_000, 50_000, 100_000, 250_000]
    rungs = _rungs_from_power_law(ns)

    curve = fit_scale_curve(rungs)

    assert curve.n_rungs == len(ns)
    assert curve.log_slope == pytest.approx(_SLOPE, abs=0.01)
    assert curve.log_intercept == pytest.approx(_INTERCEPT, abs=0.05)
    assert curve.r_squared > 0.999


def test_fit_recovers_zero_exponent_when_mcs_is_scale_invariant() -> None:
    """A constant MCS across N is the 'today's behaviour was right' outcome."""
    rungs = [_rung(n, 20) for n in (10_000, 50_000, 250_000)]

    curve = fit_scale_curve(rungs)

    assert curve.log_slope == pytest.approx(0.0, abs=1e-9)
    # ss_tot == 0 here; the fit must report 1.0 rather than divide by zero.
    assert curve.r_squared == pytest.approx(1.0)
    assert curve.predict(2_000_000) == pytest.approx(20.0, rel=1e-6)


def test_fit_requires_at_least_three_distinct_sample_sizes() -> None:
    assert MIN_RUNGS_FOR_FIT == 3

    with pytest.raises(ValueError, match="at least 3 rungs"):
        fit_scale_curve([_rung(10_000, 12), _rung(50_000, 25)])


def test_repeated_sample_size_does_not_count_as_leverage() -> None:
    """Three rungs at two distinct N carry no more evidence than two points."""
    with pytest.raises(ValueError, match="at least 3 rungs"):
        fit_scale_curve([_rung(10_000, 12), _rung(10_000, 13), _rung(50_000, 25)])


def test_rungs_are_ordered_by_sample_size_regardless_of_input_order() -> None:
    curve = fit_scale_curve([_rung(50_000, 25), _rung(10_000, 12), _rung(100_000, 34)])

    assert [r.n_rows_realized for r in curve.rungs] == [10_000, 50_000, 100_000]


def test_extrapolation_records_ratio_slope_and_fit_quality() -> None:
    curve = fit_scale_curve(_rungs_from_power_law([10_000, 50_000, 100_000]))

    out = extrapolate_min_cluster_size(curve, 2_000_000)

    assert out.min_cluster_size == pytest.approx(_true_mcs(2_000_000), rel=0.02)
    assert out.n_rows_target == 2_000_000
    assert out.extrapolation_ratio == pytest.approx(20.0)
    assert out.log_slope == pytest.approx(_SLOPE, abs=0.01)
    assert out.r_squared > 0.999
    assert not out.clamped


def test_extrapolation_ratio_below_one_means_interpolation() -> None:
    curve = fit_scale_curve(_rungs_from_power_law([10_000, 50_000, 100_000]))

    assert extrapolate_min_cluster_size(curve, 25_000).extrapolation_ratio < 1.0


def test_extrapolation_clamps_to_the_hdbscan_floor_and_flags_it() -> None:
    """A steep negative slope can predict a value HDBSCAN would reject outright."""
    curve = fit_scale_curve([_rung(1_000, 50), _rung(10_000, 10), _rung(100_000, 2)])

    out = extrapolate_min_cluster_size(curve, 10_000_000)

    assert out.min_cluster_size == 2
    assert out.clamped
    assert out.raw < 2


def test_extrapolation_respects_an_explicit_ceiling() -> None:
    curve = fit_scale_curve(_rungs_from_power_law([10_000, 50_000, 100_000]))

    out = extrapolate_min_cluster_size(curve, 2_000_000, ceiling=60)

    assert out.min_cluster_size == 60
    assert out.clamped


def test_extrapolation_rejects_a_floor_hdbscan_would_reject() -> None:
    curve = fit_scale_curve(_rungs_from_power_law([10_000, 50_000, 100_000]))

    with pytest.raises(ValueError, match="floor must be >= 2"):
        extrapolate_min_cluster_size(curve, 100_000, floor=1)


def test_pinned_rungs_are_flagged_at_both_grid_bounds() -> None:
    """A chosen value stuck on a grid bound means the grid, not N, drove the answer."""
    at_floor = _rung(10_000, 10, grid_min_scale=10, grid_max_scale=60)
    interior = _rung(50_000, 30, grid_min_scale=10, grid_max_scale=60)
    at_ceiling = _rung(250_000, 55, grid_min_scale=10, grid_max_scale=60)

    curve = fit_scale_curve([at_floor, interior, at_ceiling])

    assert curve.pinned_rungs == (at_floor,)
    assert at_floor.is_pinned
    assert not interior.is_pinned
    # 55 is the last reachable point of arange(10, 60, 5) but is below max_scale.
    assert not at_ceiling.is_pinned
    assert _rung(250_000, 60, grid_min_scale=10, grid_max_scale=60).is_pinned


def test_rung_rejects_degenerate_inputs() -> None:
    with pytest.raises(ValueError, match="n_rows_realized must be >= 1"):
        _rung(0, 10)
    with pytest.raises(ValueError, match="chosen_min_cluster_size must be >= 1"):
        _rung(10_000, 0)
    with pytest.raises(ValueError, match="n_sample must be >= 1"):
        _rung(10_000, 10, n_sample=0)


def test_curve_round_trips_through_json() -> None:
    curve = fit_scale_curve(
        [
            _rung(10_000, 12, grid_min_scale=10, grid_max_scale=60),
            _rung(50_000, 25, grid_min_scale=10, grid_max_scale=60),
            _rung(100_000, 34, grid_min_scale=10, grid_max_scale=60),
        ]
    )

    restored = ScaleCurve.from_jsonable(curve.to_jsonable())

    assert restored.rungs == curve.rungs
    assert restored.log_slope == pytest.approx(curve.log_slope)
    assert restored.log_intercept == pytest.approx(curve.log_intercept)
    assert restored.r_squared == pytest.approx(curve.r_squared)
    assert restored.predict(500_000) == pytest.approx(curve.predict(500_000))


def test_noisy_non_monotone_rungs_yield_a_low_r_squared() -> None:
    """The documented 'widen the grid before trusting this' signal."""
    curve = fit_scale_curve([_rung(10_000, 30), _rung(50_000, 12), _rung(100_000, 28)])

    assert curve.r_squared < 0.5
