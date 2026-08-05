"""Empirical scaling of ``min_cluster_size`` with mention-frame size.

Why this exists
---------------
``min_cluster_size`` is an **absolute row count**, and so (by HDBSCAN's default
``min_samples = min_cluster_size``) is the density threshold that follows from it.
A value of 25 means "0.05 % of the data" at 50k rows and "0.0005 %" at 5M. Model
selection runs on a stratified subsample of ``clustering_sample_rows``; the fit then
runs on the whole corpus. Transferring the integer verbatim silently changes what it
means.

The grid solver compounds this. :func:`
~pelinker.clustering_grid.solve_optimal_min_cluster_size_from_aggregated` min-max
normalizes each metric *within* the curve and takes the leftmost point of a plateau
defined relative to that curve's own range — so the choice is driven by the *shape* of
f(MCS) over a fixed absolute window. Change N, the curve shifts, the plateau moves, and
the window does not.

Rather than assume a correction, this module **measures** one: evaluate the same search
at several sample sizes ("rungs"), fit ``log(MCS*) ~ a + b·log(N)``, and extrapolate.
The exponent ``b`` is an output, not an input.

Reading the result
------------------
``b ≈ 1`` would mean MCS is really a constant *fraction* of N; ``b ≈ 0`` means the
absolute value transfers as-is and today's behaviour was right all along; ``0 < b < 1``
(the expected regime) means it grows sublinearly. A **flat or non-monotone** curve with
low :attr:`ScaleCurve.r_squared` is a real finding, not a failure: it usually means the
plateau selector is pinned against the fixed ``[min_scale, max_scale)`` grid, and the
grid must be widened before any extrapolation from it means anything. Check
:attr:`ScaleCurve.pinned_rungs` before trusting a fit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

MIN_RUNGS_FOR_FIT = 3
"""Two points always fit a line exactly; three is the minimum that can disagree."""


@dataclass(frozen=True)
class ScaleRung:
    """One sample-size rung: the search outcome at a single realized N."""

    n_rows_realized: int
    """Mention rows actually clustered — never the ``clustering_sample_rows`` cap."""
    chosen_min_cluster_size: int
    score_at_chosen: float
    n_clusters_mean: float
    n_sample: int
    """Bootstrap draws pooled into this rung's chosen value."""
    grid_min_scale: int | None = None
    grid_max_scale: int | None = None
    """Inclusive/exclusive grid bounds, so :func:`fit_scale_curve` can flag pinning."""

    def __post_init__(self) -> None:
        if self.n_rows_realized < 1:
            raise ValueError("n_rows_realized must be >= 1")
        if self.chosen_min_cluster_size < 1:
            raise ValueError("chosen_min_cluster_size must be >= 1")
        if self.n_sample < 1:
            raise ValueError("n_sample must be >= 1")

    @property
    def is_pinned(self) -> bool:
        """True when the chosen value sits on a grid boundary (the fit is then suspect)."""
        lo, hi = self.grid_min_scale, self.grid_max_scale
        if lo is not None and self.chosen_min_cluster_size <= lo:
            return True
        # ``max_scale`` is an exclusive np.arange bound, so the last reachable point is
        # strictly below it; anything at or above that is at the ceiling.
        return hi is not None and self.chosen_min_cluster_size >= hi

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "n_rows_realized": int(self.n_rows_realized),
            "chosen_min_cluster_size": int(self.chosen_min_cluster_size),
            "score_at_chosen": float(self.score_at_chosen),
            "n_clusters_mean": float(self.n_clusters_mean),
            "n_sample": int(self.n_sample),
            "grid_min_scale": (
                None if self.grid_min_scale is None else int(self.grid_min_scale)
            ),
            "grid_max_scale": (
                None if self.grid_max_scale is None else int(self.grid_max_scale)
            ),
            "is_pinned": bool(self.is_pinned),
        }

    @staticmethod
    def from_jsonable(data: dict[str, Any]) -> ScaleRung:
        return ScaleRung(
            n_rows_realized=int(data["n_rows_realized"]),
            chosen_min_cluster_size=int(data["chosen_min_cluster_size"]),
            score_at_chosen=float(data["score_at_chosen"]),
            n_clusters_mean=float(data["n_clusters_mean"]),
            n_sample=int(data["n_sample"]),
            grid_min_scale=(
                None
                if data.get("grid_min_scale") is None
                else int(data["grid_min_scale"])
            ),
            grid_max_scale=(
                None
                if data.get("grid_max_scale") is None
                else int(data["grid_max_scale"])
            ),
        )


@dataclass(frozen=True)
class ScaleCurve:
    """A fitted power law ``MCS*(N) = exp(log_intercept) · N ** log_slope``."""

    rungs: tuple[ScaleRung, ...]
    log_intercept: float
    log_slope: float
    """The measured exponent ``b``. 0 = absolute transfer, 1 = constant fraction of N."""
    r_squared: float
    """Coefficient of determination in log-log space; low means do not extrapolate."""

    @property
    def n_rungs(self) -> int:
        return len(self.rungs)

    @property
    def max_n_rows_realized(self) -> int:
        return max(r.n_rows_realized for r in self.rungs)

    @property
    def pinned_rungs(self) -> tuple[ScaleRung, ...]:
        """Rungs whose chosen MCS sat on a grid boundary — widen the grid and re-run."""
        return tuple(r for r in self.rungs if r.is_pinned)

    def predict(self, n_rows: int) -> float:
        """Un-rounded ``MCS*`` at ``n_rows`` (may be below 1 for tiny N)."""
        if n_rows < 1:
            raise ValueError("n_rows must be >= 1")
        return float(math.exp(self.log_intercept + self.log_slope * math.log(n_rows)))

    def extrapolation_ratio(self, n_rows: int) -> float:
        """``n_rows`` relative to the largest rung actually measured.

        Anything much above 1 is extrapolation rather than interpolation; callers should
        surface it rather than silently trusting a long reach.
        """
        return float(n_rows) / float(self.max_n_rows_realized)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "rungs": [r.to_jsonable() for r in self.rungs],
            "log_intercept": float(self.log_intercept),
            "log_slope": float(self.log_slope),
            "r_squared": float(self.r_squared),
            "n_rungs": int(self.n_rungs),
            "max_n_rows_realized": int(self.max_n_rows_realized),
            "n_pinned_rungs": len(self.pinned_rungs),
        }

    @staticmethod
    def from_jsonable(data: dict[str, Any]) -> ScaleCurve:
        return ScaleCurve(
            rungs=tuple(ScaleRung.from_jsonable(r) for r in data["rungs"]),
            log_intercept=float(data["log_intercept"]),
            log_slope=float(data["log_slope"]),
            r_squared=float(data["r_squared"]),
        )


def fit_scale_curve(rungs: Sequence[ScaleRung]) -> ScaleCurve:
    """Least-squares fit of ``log(MCS*) ~ a + b·log(N)`` over the measured rungs.

    Raises:
        ValueError: with fewer than :data:`MIN_RUNGS_FOR_FIT` rungs (two points fit any
            line exactly, so the fit would carry no evidence), or when every rung shares
            the same ``n_rows_realized`` (no leverage on the exponent).
    """
    unique_by_n: dict[int, ScaleRung] = {}
    for rung in rungs:
        # A repeated N carries no extra leverage; keep the last one deterministically.
        unique_by_n[rung.n_rows_realized] = rung
    ordered = tuple(unique_by_n[n] for n in sorted(unique_by_n))

    if len(ordered) < MIN_RUNGS_FOR_FIT:
        raise ValueError(
            f"need at least {MIN_RUNGS_FOR_FIT} rungs at distinct n_rows_realized to fit "
            f"a scale curve, got {len(ordered)}"
        )

    x = np.log(np.array([r.n_rows_realized for r in ordered], dtype=np.float64))
    y = np.log(np.array([r.chosen_min_cluster_size for r in ordered], dtype=np.float64))
    if float(np.ptp(x)) <= 0.0:
        raise ValueError("all rungs share the same n_rows_realized; cannot fit a slope")

    slope, intercept = np.polyfit(x, y, deg=1)
    residuals = y - (intercept + slope * x)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    # A perfectly flat MCS across rungs has ss_tot == 0: the model explains nothing, but
    # it also mispredicts nothing. Report 1.0 rather than dividing by zero.
    r_squared = 1.0 if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot

    return ScaleCurve(
        rungs=ordered,
        log_intercept=float(intercept),
        log_slope=float(slope),
        r_squared=float(r_squared),
    )


@dataclass(frozen=True)
class ExtrapolatedMinClusterSize:
    """The extrapolated hyperparameter plus everything needed to distrust it."""

    min_cluster_size: int
    raw: float
    """Before rounding and clamping — keeps the rounding visible in reports."""
    n_rows_target: int
    extrapolation_ratio: float
    log_slope: float
    r_squared: float
    clamped: bool
    """True when the raw prediction fell outside ``[min_cluster_size_floor, ceiling]``."""

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "min_cluster_size": int(self.min_cluster_size),
            "raw": float(self.raw),
            "n_rows_target": int(self.n_rows_target),
            "extrapolation_ratio": float(self.extrapolation_ratio),
            "log_slope": float(self.log_slope),
            "r_squared": float(self.r_squared),
            "clamped": bool(self.clamped),
        }


DEFAULT_MIN_CLUSTER_SIZE = 20
"""Fallback when neither an explicit value nor a scale curve is supplied."""

MinClusterSizeSource = Literal["explicit", "scale_curve", "default"]


@dataclass(frozen=True)
class MinClusterSizeProvenance:
    """Where the ``min_cluster_size`` actually used by a fit came from.

    Recorded on the fit report so a model never carries an unexplained hyperparameter.
    """

    min_cluster_size: int
    source: MinClusterSizeSource
    n_rows_realized: int
    extrapolation: ExtrapolatedMinClusterSize | None = None
    """Present when ``source == "scale_curve"``, or when a curve was available but an
    explicit value overrode it — in which case it records what the curve *would* have
    chosen, so the disagreement is visible."""

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "min_cluster_size": int(self.min_cluster_size),
            "source": str(self.source),
            "n_rows_realized": int(self.n_rows_realized),
            "extrapolation": (
                None if self.extrapolation is None else self.extrapolation.to_jsonable()
            ),
        }


def resolve_min_cluster_size(
    *,
    explicit: int | None,
    n_rows_realized: int,
    scale_curve: ScaleCurve | None,
) -> tuple[int, MinClusterSizeProvenance]:
    """Pick ``min_cluster_size`` for a fit, preferring an explicit value over the curve.

    Precedence is explicit > scale curve > :data:`DEFAULT_MIN_CLUSTER_SIZE`. When both an
    explicit value and a curve are present the explicit one wins, but the curve's
    prediction is still recorded so the two can be compared after the fact.
    """
    extrapolation: ExtrapolatedMinClusterSize | None = None
    if scale_curve is not None:
        extrapolation = extrapolate_min_cluster_size(scale_curve, n_rows_realized)

    if explicit is not None:
        return int(explicit), MinClusterSizeProvenance(
            min_cluster_size=int(explicit),
            source="explicit",
            n_rows_realized=int(n_rows_realized),
            extrapolation=extrapolation,
        )
    if extrapolation is not None:
        return extrapolation.min_cluster_size, MinClusterSizeProvenance(
            min_cluster_size=extrapolation.min_cluster_size,
            source="scale_curve",
            n_rows_realized=int(n_rows_realized),
            extrapolation=extrapolation,
        )
    return DEFAULT_MIN_CLUSTER_SIZE, MinClusterSizeProvenance(
        min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE,
        source="default",
        n_rows_realized=int(n_rows_realized),
    )


def extrapolate_min_cluster_size(
    curve: ScaleCurve,
    n_rows_target: int,
    *,
    floor: int = 2,
    ceiling: int | None = None,
) -> ExtrapolatedMinClusterSize:
    """Predict ``min_cluster_size`` at ``n_rows_target`` from a fitted curve.

    ``floor`` defaults to 2 because HDBSCAN rejects ``min_cluster_size < 2``. The
    returned record carries :attr:`~ExtrapolatedMinClusterSize.extrapolation_ratio` and
    :attr:`~ExtrapolatedMinClusterSize.r_squared` so a long reach or a bad fit shows up
    in the fit report instead of vanishing into a single integer.
    """
    if floor < 2:
        raise ValueError("floor must be >= 2 (HDBSCAN rejects min_cluster_size < 2)")
    if ceiling is not None and ceiling < floor:
        raise ValueError(f"ceiling ({ceiling}) must be >= floor ({floor})")

    raw = curve.predict(n_rows_target)
    rounded = int(round(raw))
    clamped_value = max(floor, rounded)
    if ceiling is not None:
        clamped_value = min(ceiling, clamped_value)

    return ExtrapolatedMinClusterSize(
        min_cluster_size=clamped_value,
        raw=raw,
        n_rows_target=int(n_rows_target),
        extrapolation_ratio=curve.extrapolation_ratio(n_rows_target),
        log_slope=curve.log_slope,
        r_squared=curve.r_squared,
        clamped=clamped_value != rounded,
    )
