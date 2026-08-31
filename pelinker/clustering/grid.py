"""HDBSCAN min_cluster_size grid evaluation, cross-sample aggregation, and smooth optimum selection."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pelinker.core.config import GridObjectiveSpec

import hdbscan
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score
from torch.nn import functional as F

logger = logging.getLogger(__name__)

_FLAT_CURVE_EPS = 1e-12
"""Below this range, a pooled objective curve is treated as carrying no signal."""


@dataclass(frozen=True)
class ScalarMetricAggregate:
    """Mean, dispersion, and sample count for one metric at a single grid point."""

    mean: float
    std: float
    count: int


@dataclass(frozen=True)
class AggregatedGridPoint:
    """One grid value of ``min_cluster_size`` with aggregated metrics across samples."""

    min_cluster_size: int
    dbcv: ScalarMetricAggregate
    icm_mean: float
    n_clusters_mean: float
    ari: ScalarMetricAggregate


@dataclass(frozen=True)
class AggregatedGridReport:
    """Typed aggregation of per-sample grid metrics; points are sorted by ``min_cluster_size``."""

    points: tuple[AggregatedGridPoint, ...]

    def __post_init__(self) -> None:
        sizes = [p.min_cluster_size for p in self.points]
        if sizes != sorted(sizes):
            raise ValueError(
                "AggregatedGridReport.points must be sorted by min_cluster_size"
            )
        if len(set(sizes)) != len(sizes):
            raise ValueError(
                "Duplicate min_cluster_size in AggregatedGridReport.points"
            )


SelectionKind = Literal[
    # Legacy values, retained so previously written grid_chosen_hyperparameters.json stays readable.
    "plateau_derivative",
    "smoothed_argmax",
    # Current selector.
    "argmax",
    "one_se_paired",
    "degenerate_fallback",
]


@dataclass(frozen=True)
class PairedGridScores:
    """Per-sample objective values on a shared ``min_cluster_size`` grid.

    ``scores`` is ``(n_samples, n_grid)``; entry ``[b, i]`` is the combined objective for
    bootstrap sample ``b`` at grid point ``i``, or NaN where that sample had no usable fit.
    Keeping the samples un-pooled is what makes *paired* comparisons across grid points
    possible: every column was evaluated on the same samples, so differencing two columns
    within a sample cancels the sample-level variation they share.
    """

    min_cluster_size: tuple[int, ...]
    scores: np.ndarray
    n_clusters_mean: tuple[float, ...]
    objective: GridObjectiveSpec
    n_clipped: int = 0
    """Count of ``(sample, grid)`` cells where a negative DBCV or ARI was clipped to 0."""
    degenerate: bool = False
    """True when the requested combined objective carried no signal and DBCV was used instead."""

    def __post_init__(self) -> None:
        if self.scores.ndim != 2:
            raise ValueError("PairedGridScores.scores must be 2-D (n_samples, n_grid)")
        if self.scores.shape[1] != len(self.min_cluster_size):
            raise ValueError("scores columns must match min_cluster_size length")
        if len(self.n_clusters_mean) != len(self.min_cluster_size):
            raise ValueError("n_clusters_mean must match min_cluster_size length")

    @property
    def n_samples(self) -> int:
        return int(self.scores.shape[0])


@dataclass(frozen=True)
class SmoothedGridOptimumResult:
    """Diagnostics for the ``min_cluster_size`` solvers.

    ``score_mean_at_chosen`` / ``score_std_at_chosen`` refer to the raw objective (before
    smoothing and before any cluster-count term) at the chosen grid point.

    ``y_se`` holds the standard error of ``y_smooth[i] - y_smooth[argmax]`` — a *paired*
    quantity, so it is 0 at the argmax by construction and is undefined without a reference
    point. ``y_eligible[i]`` is True when grid point ``i`` is within ``one_se_k`` of those
    standard errors of the best point.
    """

    chosen_min_cluster_size: int
    score_mean_at_chosen: float
    score_std_at_chosen: float
    n_clusters_mean_at_chosen: float
    x: tuple[float, ...]
    y_objective: tuple[float, ...]
    y_cluster_term: tuple[float, ...]
    y_smooth: tuple[float, ...]
    selection: SelectionKind
    y_se: tuple[float, ...] = ()
    y_eligible: tuple[bool, ...] = ()
    argmax_min_cluster_size: int | None = None
    one_se_k: float = 0.0
    n_clipped: int = 0
    n_samples: int = 0


def _ensure_odd_window(window: int) -> int:
    if window < 1:
        raise ValueError("smooth window must be >= 1")
    return window if window % 2 == 1 else window + 1


def _uniform_centered_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average along the last axis, ignoring NaNs within each window.

    Applied per bootstrap sample rather than to the pooled curve. A moving average is
    linear, so smoothing each sample and then averaging gives the same pooled curve as
    smoothing the pooled curve — but it keeps the paired differences between grid points
    consistent with the smoothed means they are compared against.
    """
    w = _ensure_odd_window(window)
    half = w // 2
    arr = np.atleast_2d(np.asarray(values, dtype=np.float64))
    n = arr.shape[-1]
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = arr[:, lo:hi]
        with warnings.catch_warnings():
            # All-NaN windows are expected on ragged grids; they stay NaN.
            warnings.simplefilter("ignore", RuntimeWarning)
            out[:, i] = np.nanmean(chunk, axis=1)
    return out.reshape(np.shape(values))


def _metric_vectors(
    report: AggregatedGridReport,
    which: Literal["dbcv", "ari"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if which == "dbcv":
        means = np.array([p.dbcv.mean for p in report.points], dtype=np.float64)
        stds = np.array([p.dbcv.std for p in report.points], dtype=np.float64)
        counts = np.array([p.dbcv.count for p in report.points], dtype=np.float64)
    else:
        means = np.array([p.ari.mean for p in report.points], dtype=np.float64)
        stds = np.array([p.ari.std for p in report.points], dtype=np.float64)
        counts = np.array([p.ari.count for p in report.points], dtype=np.float64)
    return means, stds, counts


def combine_clipped_geomean(dbcv: np.ndarray, ari: np.ndarray) -> np.ndarray:
    """Combine DBCV and ARI into one "both must be good" score on a fixed [0, 1] scale.

    Both metrics already share a run-independent meaning: 0 is "no density structure" /
    "chance agreement", 1 is perfect. Clip to that floor and take the geometric mean::

        g = sqrt(max(dbcv, 0) * max(ari, 0))

    Two properties matter, and neither holds for the arithmetic mean:

    * **Scale invariance.** ``g(c1*dbcv, c2*ari) = sqrt(c1*c2) * g(dbcv, ari)`` — one
      positive factor across the whole grid, so the *ranking* of grid points does not
      change. The metrics never have to be put on a common scale, which is what made the
      old per-curve min–max normalization necessary (and made it amplify the noise of
      whichever metric happened to be flat).
    * **Conjunctivity.** A high DBCV cannot compensate for a near-zero ARI. "Maximize both"
      is a conjunction; an arithmetic mean lets one metric buy off the other.

    NaN in either input propagates, marking that grid point unusable for that sample.
    """
    d = np.clip(np.asarray(dbcv, dtype=np.float64), 0.0, None)
    a = np.clip(np.asarray(ari, dtype=np.float64), 0.0, None)
    return np.sqrt(d * a)


def geomean_std(
    dbcv: np.ndarray,
    dbcv_std: np.ndarray,
    ari: np.ndarray,
    ari_std: np.ndarray,
) -> np.ndarray:
    """Propagate uncertainty through :func:`combine_clipped_geomean` (delta method).

    For ``g = sqrt(d*a)`` the *relative* errors add in quadrature::

        se_g / g = 0.5 * sqrt((se_d/d)**2 + (se_a/a)**2)

    A clipped-to-zero metric pins ``g`` at 0, so its uncertainty is reported as 0 too.
    """
    d = np.clip(np.asarray(dbcv, dtype=np.float64), 0.0, None)
    a = np.clip(np.asarray(ari, dtype=np.float64), 0.0, None)
    g = np.sqrt(d * a)
    sd = np.nan_to_num(np.asarray(dbcv_std, dtype=np.float64), nan=0.0)
    sa = np.nan_to_num(np.asarray(ari_std, dtype=np.float64), nan=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.sqrt(np.square(sd / d) + np.square(sa / a))
        out = 0.5 * g * rel
    return np.where(np.isfinite(out), out, 0.0)


def _combine_for_objective(
    dbcv: np.ndarray, ari: np.ndarray, objective: GridObjectiveSpec
) -> np.ndarray:
    if objective == "dbcv":
        return np.asarray(dbcv, dtype=np.float64)
    if objective == "ari":
        return np.asarray(ari, dtype=np.float64)
    if objective == "dbcv_ari_geomean":
        return combine_clipped_geomean(dbcv, ari)
    raise ValueError(f"Unknown grid objective: {objective!r}")


def build_paired_grid_scores(
    metrics_dfs: Sequence[pd.DataFrame],
    *,
    objective: GridObjectiveSpec = "dbcv_ari_geomean",
) -> PairedGridScores:
    """Stack per-sample grid metrics into an ``(n_samples, n_grid)`` objective matrix.

    Samples may cover different ``min_cluster_size`` values — ``evaluate_cluster_size_grid``
    drops points that produced no clusters — so every sample is reindexed onto the *union*
    of grid values with NaN. Intersecting instead would silently shrink the grid to whatever
    the worst sample happened to support.

    Falls back to a DBCV-only objective when no sample has any ARI (no ``entity`` column).
    """
    if not metrics_dfs:
        raise ValueError("metrics_dfs must be non-empty")

    grid = sorted(
        {
            int(v)
            for df in metrics_dfs
            for v in pd.to_numeric(df["min_cluster_size"], errors="coerce").dropna()
        }
    )
    if not grid:
        raise ValueError("No min_cluster_size values in metrics_dfs")

    n_grid = len(grid)
    dbcv = np.full((len(metrics_dfs), n_grid), np.nan, dtype=np.float64)
    ari = np.full((len(metrics_dfs), n_grid), np.nan, dtype=np.float64)
    n_clusters = np.full((len(metrics_dfs), n_grid), np.nan, dtype=np.float64)

    for b, df in enumerate(metrics_dfs):
        # Last row wins on duplicate grid values, matching the per-sample CSV dedupe.
        indexed = df.set_index(df["min_cluster_size"].astype(int))
        indexed = indexed[~indexed.index.duplicated(keep="last")].reindex(grid)
        dbcv[b] = pd.to_numeric(indexed.get("dbcv"), errors="coerce").to_numpy(
            dtype=np.float64
        )
        if "ari" in indexed.columns:
            ari[b] = pd.to_numeric(indexed["ari"], errors="coerce").to_numpy(
                dtype=np.float64
            )
        if "n_clusters" in indexed.columns:
            n_clusters[b] = pd.to_numeric(
                indexed["n_clusters"], errors="coerce"
            ).to_numpy(dtype=np.float64)

    resolved = objective
    if objective == "dbcv_ari_geomean" and not np.any(np.isfinite(ari)):
        logger.info(
            "No ARI values on the min_cluster_size grid (no entity labels); "
            "falling back to a DBCV-only grid objective."
        )
        resolved = "dbcv"

    n_clipped = 0
    if resolved == "dbcv_ari_geomean":
        n_clipped = int(np.sum((dbcv < 0) | (ari < 0)))
        if n_clipped:
            logger.info(
                "Clipped %d negative DBCV/ARI cell(s) to 0 before the geometric mean; "
                "heavy clipping means the grid is largely outside the useful region.",
                n_clipped,
            )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scores = _combine_for_objective(dbcv, ari, resolved)
        n_clusters_mean = np.nanmean(n_clusters, axis=0)

        degenerate = False
        if resolved == "dbcv_ari_geomean":
            pooled = np.nanmean(scores, axis=0)
            finite = pooled[np.isfinite(pooled)]
            if (
                finite.size
                and float(np.max(finite) - np.min(finite)) <= _FLAT_CURVE_EPS
            ):
                # e.g. ARI identically 0, or DBCV <= 0 everywhere: the geometric mean is
                # flat and carries no information about min_cluster_size. Say so loudly
                # rather than picking an arbitrary point off a constant curve.
                logger.warning(
                    "Combined DBCV+ARI objective is flat across the min_cluster_size grid "
                    "(range <= %g); falling back to DBCV alone. Check whether ARI is "
                    "identically zero or DBCV is non-positive over the whole grid.",
                    _FLAT_CURVE_EPS,
                )
                resolved = "dbcv"
                degenerate = True
                scores = _combine_for_objective(dbcv, ari, resolved)

    return PairedGridScores(
        min_cluster_size=tuple(grid),
        scores=scores,
        n_clusters_mean=tuple(float(v) for v in n_clusters_mean),
        objective=resolved,
        n_clipped=n_clipped,
        degenerate=degenerate,
    )


def _cluster_count_reward_term(
    n_clusters: np.ndarray,
    *,
    cluster_count_reward: float,
    n_entities: int | None,
) -> np.ndarray:
    """Additive log penalty: 0 at ``n_ref`` clusters, negative when fewer clusters."""
    if cluster_count_reward == 0.0:
        return np.zeros_like(n_clusters, dtype=np.float64)
    n_clust = np.maximum(n_clusters.astype(np.float64), 1.0)
    n_ref = float(n_entities) if n_entities is not None else float(np.max(n_clust))
    if n_ref <= 0:
        return np.zeros_like(n_clust, dtype=np.float64)
    return cluster_count_reward * np.log(n_clust / n_ref)


def _validate_solver_args(
    *, cluster_count_reward: float, n_entities: int | None, one_se_k: float
) -> None:
    if cluster_count_reward < 0:
        raise ValueError("cluster_count_reward must be >= 0")
    if n_entities is not None and n_entities < 1:
        raise ValueError("n_entities must be >= 1 when provided")
    if one_se_k < 0:
        raise ValueError("one_se_k must be >= 0")


def _paired_standard_errors(y: np.ndarray, i_star: int) -> np.ndarray:
    """Standard error of ``y[:, i] - y[:, i_star]`` for every column ``i``.

    Every grid point was evaluated on the *same* bootstrap samples, so comparisons across
    ``min_cluster_size`` are paired. Differencing within each sample before taking the
    spread removes the sample-level variation that all grid points share, which is usually
    the dominant term. The result is much tighter — and far more stable across runs — than
    the unpaired ``sqrt(se_i**2 + se_istar**2)``.

    Zero at ``i_star`` by construction, and zero wherever fewer than two samples overlap.
    """
    n_grid = y.shape[1]
    se = np.zeros(n_grid, dtype=np.float64)
    ref = y[:, i_star]
    for i in range(n_grid):
        if i == i_star:
            continue
        delta = y[:, i] - ref
        valid = np.isfinite(delta)
        n_pair = int(np.count_nonzero(valid))
        if n_pair < 2:
            continue
        se[i] = float(np.std(delta[valid], ddof=1) / np.sqrt(n_pair))
    return se


def _one_se_pick(
    mu: np.ndarray,
    se: np.ndarray,
    *,
    one_se_k: float,
    contiguous: bool,
) -> tuple[int, int, np.ndarray]:
    """Apply the one-standard-error rule. Returns ``(chosen_idx, argmax_idx, eligible)``.

    Uncertainty may only move the choice to a grid point that is *statistically
    indistinguishable* from the best one — it can never promote a point with a genuinely
    worse mean, which is what a ``mean - k*std`` discount does.
    """
    i_star = int(np.nanargmax(mu))
    with np.errstate(invalid="ignore"):
        eligible = np.isfinite(mu) & (mu >= mu[i_star] - one_se_k * se)
    eligible[i_star] = True

    chosen = i_star
    if contiguous:
        # Walk right through consecutive eligible points. An isolated far-right point that
        # only qualifies because its error bar is wide is not a plateau.
        j = i_star + 1
        while j < len(mu) and eligible[j]:
            chosen = j
            j += 1
    else:
        candidates = np.flatnonzero(eligible)
        chosen = int(candidates[-1]) if candidates.size else i_star
    return chosen, i_star, eligible


def solve_optimal_min_cluster_size(
    paired: PairedGridScores,
    *,
    smooth_window: int = 3,
    one_se_k: float = 1.0,
    one_se_contiguous: bool = True,
    one_se_min_samples: int = 6,
    cluster_count_reward: float = 0.0,
    n_entities: int | None = None,
) -> SmoothedGridOptimumResult:
    """
    Choose ``min_cluster_size`` from per-sample grid scores with a paired one-SE rule.

    1. Smooth each **sample's** curve with a centered moving average (linear, so the pooled
       curve is unchanged, but the paired differences stay consistent with it).
    2. Optionally add ``cluster_count_reward * log(n_clusters / n_ref)`` — deterministic per
       grid point, so it shifts whole columns and leaves the paired standard errors alone.
    3. Take ``i* = argmax`` of the pooled mean, then move to the **largest**
       ``min_cluster_size`` reachable through consecutive grid points whose mean is within
       ``one_se_k`` paired standard errors of ``i*``. Larger is the parsimonious direction:
       fewer, denser, more reproducible clusters.

    Step 3 is skipped below ``one_se_min_samples`` bootstrap samples, falling back to the
    plain argmax. A standard error estimated from three numbers is itself mostly noise, and
    acting on it makes the choice *less* reproducible, not more — measured on this repo's
    grid exports, the slide starts paying for itself only around 6–8 samples. With a single
    sample the paired standard errors are identically zero, so the fallback is automatic.
    """
    _validate_solver_args(
        cluster_count_reward=cluster_count_reward,
        n_entities=n_entities,
        one_se_k=one_se_k,
    )

    x_all = np.array(paired.min_cluster_size, dtype=np.float64)
    n_clusters_all = np.array(paired.n_clusters_mean, dtype=np.float64)
    scores = np.asarray(paired.scores, dtype=np.float64)

    cluster_term_all = _cluster_count_reward_term(
        n_clusters_all,
        cluster_count_reward=cluster_count_reward,
        n_entities=n_entities,
    )

    smoothed = _uniform_centered_moving_average(scores, smooth_window)
    y_all = smoothed + cluster_term_all[np.newaxis, :]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu_all = np.nanmean(y_all, axis=0)

    keep = np.isfinite(x_all) & np.isfinite(mu_all)
    if not np.any(keep):
        raise ValueError("No finite objective values on the min_cluster_size grid")

    order = np.argsort(x_all[keep])
    idx = np.flatnonzero(keep)[order]
    x = x_all[idx]
    mu = mu_all[idx]
    y = y_all[:, idx]
    raw = scores[:, idx]
    n_clusters = n_clusters_all[idx]
    cluster_term = cluster_term_all[idx]

    se = _paired_standard_errors(y, int(np.nanargmax(mu)))
    effective_k = one_se_k
    if paired.n_samples < one_se_min_samples:
        if one_se_k > 0 and paired.n_samples > 1:
            logger.info(
                "Only %d bootstrap sample(s) on the grid (< %d): standard errors are not "
                "estimable, using the plain argmax instead of the one-SE rule.",
                paired.n_samples,
                one_se_min_samples,
            )
        effective_k = 0.0
    chosen_idx, argmax_idx, eligible = _one_se_pick(
        mu, se, one_se_k=effective_k, contiguous=one_se_contiguous
    )

    selection: SelectionKind
    if paired.degenerate:
        selection = "degenerate_fallback"
    elif chosen_idx == argmax_idx:
        selection = "argmax"
    else:
        selection = "one_se_paired"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw_at = raw[:, chosen_idx]
        raw_valid = raw_at[np.isfinite(raw_at)]
        score_mean_at = float(np.mean(raw_valid)) if raw_valid.size else float("nan")
        score_std_at = float(np.std(raw_valid, ddof=1)) if raw_valid.size > 1 else 0.0
        y_objective = np.nanmean(raw, axis=0) + cluster_term

    return SmoothedGridOptimumResult(
        chosen_min_cluster_size=int(x[chosen_idx]),
        score_mean_at_chosen=score_mean_at,
        score_std_at_chosen=score_std_at,
        n_clusters_mean_at_chosen=float(n_clusters[chosen_idx]),
        x=tuple(float(v) for v in x),
        y_objective=tuple(float(v) for v in y_objective),
        y_cluster_term=tuple(float(v) for v in cluster_term),
        y_smooth=tuple(float(v) for v in mu),
        selection=selection,
        y_se=tuple(float(v) for v in se),
        y_eligible=tuple(bool(v) for v in eligible),
        argmax_min_cluster_size=int(x[argmax_idx]),
        one_se_k=float(effective_k),
        n_clipped=paired.n_clipped,
        n_samples=paired.n_samples,
    )


def solve_optimal_min_cluster_size_from_metrics_dfs(
    metrics_dfs: Sequence[pd.DataFrame],
    *,
    objective: GridObjectiveSpec = "dbcv_ari_geomean",
    smooth_window: int = 3,
    one_se_k: float = 1.0,
    one_se_contiguous: bool = True,
    one_se_min_samples: int = 6,
    cluster_count_reward: float = 0.0,
    n_entities: int | None = None,
) -> SmoothedGridOptimumResult:
    """Build per-sample scores then solve. This is the preferred entry point."""
    paired = build_paired_grid_scores(metrics_dfs, objective=objective)
    return solve_optimal_min_cluster_size(
        paired,
        smooth_window=smooth_window,
        one_se_k=one_se_k,
        one_se_contiguous=one_se_contiguous,
        one_se_min_samples=one_se_min_samples,
        cluster_count_reward=cluster_count_reward,
        n_entities=n_entities,
    )


def solve_optimal_min_cluster_size_from_aggregated(
    report: AggregatedGridReport,
    *,
    objective: GridObjectiveSpec = "dbcv_ari_geomean",
    smooth_window: int = 3,
    one_se_k: float = 1.0,
    one_se_contiguous: bool = True,
    one_se_min_samples: int = 6,
    cluster_count_reward: float = 0.0,
    n_entities: int | None = None,
) -> SmoothedGridOptimumResult:
    """
    Solve from already-pooled per-metric aggregates, for callers that no longer hold the
    per-sample tables.

    The pairing across grid points is unrecoverable here, so the standard errors fall back
    to the unpaired ``sqrt(se_i**2 + se_istar**2)`` — wider than the paired version and
    therefore more willing to slide toward larger ``min_cluster_size``. Prefer
    :func:`solve_optimal_min_cluster_size_from_metrics_dfs` whenever the per-sample frames
    are available.
    """
    if len(report.points) == 0:
        raise ValueError("No aggregated grid points provided")
    _validate_solver_args(
        cluster_count_reward=cluster_count_reward,
        n_entities=n_entities,
        one_se_k=one_se_k,
    )

    md, sd, cd = _metric_vectors(report, "dbcv")
    ma, sa, ca = _metric_vectors(report, "ari")
    if objective == "dbcv_ari_geomean" and not np.any(np.isfinite(ma)):
        logger.info(
            "No ARI on the aggregated grid; falling back to a DBCV-only objective."
        )
        objective = "dbcv"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = _combine_for_objective(md, ma, objective)
    # Relative-error propagation through the geometric mean (delta method):
    #   se_g / g = 0.5 * sqrt((se_d/d)**2 + (se_a/a)**2)
    if objective == "dbcv_ari_geomean":
        stds = geomean_std(md, sd, ma, sa)
        counts = np.minimum(cd, ca)
    elif objective == "ari":
        stds, counts = sa, ca
    else:
        stds, counts = sd, cd

    x_all = np.array([p.min_cluster_size for p in report.points], dtype=np.float64)
    n_clusters_all = np.array(
        [p.n_clusters_mean for p in report.points], dtype=np.float64
    )
    cluster_term_all = _cluster_count_reward_term(
        n_clusters_all, cluster_count_reward=cluster_count_reward, n_entities=n_entities
    )
    smoothed = _uniform_centered_moving_average(means, smooth_window)
    mu_all = smoothed + cluster_term_all

    keep = np.isfinite(x_all) & np.isfinite(mu_all)
    if not np.any(keep):
        raise ValueError("No finite objective values in aggregated grid report")
    order = np.argsort(x_all[keep])
    idx = np.flatnonzero(keep)[order]
    x, mu = x_all[idx], mu_all[idx]
    base_means, base_stds = means[idx], stds[idx]
    n_obs = np.maximum(counts[idx], 1.0)
    n_clusters, cluster_term = n_clusters_all[idx], cluster_term_all[idx]

    point_se = np.nan_to_num(base_stds, nan=0.0) / np.sqrt(n_obs)
    i_star = int(np.nanargmax(mu))
    se = np.sqrt(point_se**2 + point_se[i_star] ** 2)
    se[i_star] = 0.0

    n_samples = int(np.nanmax(counts)) if counts.size else 0
    effective_k = 0.0 if n_samples < one_se_min_samples else one_se_k
    chosen_idx, argmax_idx, eligible = _one_se_pick(
        mu, se, one_se_k=effective_k, contiguous=one_se_contiguous
    )

    return SmoothedGridOptimumResult(
        chosen_min_cluster_size=int(x[chosen_idx]),
        score_mean_at_chosen=float(base_means[chosen_idx]),
        score_std_at_chosen=float(base_stds[chosen_idx]),
        n_clusters_mean_at_chosen=float(n_clusters[chosen_idx]),
        x=tuple(float(v) for v in x),
        y_objective=tuple(float(v) for v in (base_means + cluster_term)),
        y_cluster_term=tuple(float(v) for v in cluster_term),
        y_smooth=tuple(float(v) for v in mu),
        selection="argmax" if chosen_idx == argmax_idx else "one_se_paired",
        y_se=tuple(float(v) for v in se),
        y_eligible=tuple(bool(v) for v in eligible),
        argmax_min_cluster_size=int(x[argmax_idx]),
        one_se_k=float(effective_k),
        n_samples=n_samples,
    )


def aggregated_grid_report_to_dataframe(report: AggregatedGridReport) -> pd.DataFrame:
    """Lossless round-trip style export for notebooks (typed report → table)."""
    rows: list[dict[str, float | int]] = []
    for p in report.points:
        rows.append(
            {
                "min_cluster_size": p.min_cluster_size,
                "dbcv_mean": p.dbcv.mean,
                "dbcv_std": p.dbcv.std,
                "dbcv_count": p.dbcv.count,
                "icm_mean": p.icm_mean,
                "n_clusters_mean": p.n_clusters_mean,
                "ari_mean": p.ari.mean,
                "ari_std": p.ari.std,
                "ari_count": p.ari.count,
            }
        )
    return pd.DataFrame(rows)


def cosine_similarity_std(
    tensor: torch.Tensor, max_pairs: int = 200_000, random_seed: int = 13
) -> torch.Tensor:
    """
    Calculate the standard deviation of pairwise cosine similarities
    for a tensor of shape (n_b, dim_emb).
    """
    normalized = F.normalize(tensor.float(), p=2, dim=1)

    n_points = normalized.size(0)
    if n_points < 2:
        return torch.tensor(float("nan"), dtype=normalized.dtype)

    total_pairs = n_points * (n_points - 1) // 2

    if total_pairs <= max_pairs:
        cos_sim_matrix = torch.mm(normalized, normalized.t())
        triu_indices = torch.triu_indices(
            cos_sim_matrix.size(0), cos_sim_matrix.size(1), offset=1
        )
        cos_similarities = cos_sim_matrix[triu_indices[0], triu_indices[1]]
        return torch.std(cos_similarities)

    sample_size = min(max_pairs, total_pairs)
    generator = torch.Generator(device=normalized.device)
    generator.manual_seed(random_seed)

    idx_i = torch.randint(
        0, n_points, (sample_size,), generator=generator, device=normalized.device
    )
    idx_j = torch.randint(
        0, n_points - 1, (sample_size,), generator=generator, device=normalized.device
    )
    idx_j = idx_j + (idx_j >= idx_i).long()  # ensure i != j
    cos_similarities = (normalized[idx_i] * normalized[idx_j]).sum(dim=1)
    return torch.std(cos_similarities)


def _adjusted_rand_index_vs_entity_codes(
    entity_codes: np.ndarray,
    cluster_labels: np.ndarray,
) -> float:
    """ARI between KB entity codes and HDBSCAN labels; noise (-1) excluded (matches analysis)."""
    valid_mask = cluster_labels != -1
    if not valid_mask.any():
        return 0.0
    y_true = entity_codes[valid_mask]
    y_pred = cluster_labels[valid_mask]
    if len(y_pred) == 0:
        return 0.0
    return float(adjusted_rand_score(y_true, y_pred))


def evaluate_cluster_size_grid(
    dfr2: pd.DataFrame,
    umap_columns: list[str],
    sizes: list[int],
    max_pairs_per_cluster: int = 200_000,
) -> pd.DataFrame:
    """
    Evaluate clustering metrics on a grid of min_cluster_size values.

    Uses DBCV (Density-Based Clustering Validation) and, when ``entity`` is present,
    adjusted Rand index vs. entity codes (noise label -1 excluded).

    Returns:
        DataFrame with columns: min_cluster_size, icm, n_clusters, dbcv, ari
    """
    umap_values = dfr2[umap_columns].to_numpy(dtype=np.float32, copy=False)
    entity_codes: np.ndarray | None = None
    if "entity" in dfr2.columns:
        entity_codes = (
            dfr2["entity"]
            .astype("category")
            .cat.codes.to_numpy(dtype=np.int64, copy=False)
        )

    metrics = []
    for size in sizes:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=size, gen_min_span_tree=True)
        labels = clusterer.fit_predict(umap_values)

        ic = []
        for ix in np.unique(labels):
            if ix == -1:
                continue
            cluster_values = umap_values[labels == ix]
            if len(cluster_values) < 2:
                continue
            tgroup = torch.from_numpy(cluster_values)
            st = cosine_similarity_std(
                tgroup, max_pairs=max_pairs_per_cluster, random_seed=13
            )
            ic += [float(st)]

        icm = np.mean(ic) if ic else np.nan

        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        if n_clusters >= 2:
            rv = getattr(clusterer, "relative_validity_", None)
            dbcv = float(rv) if rv is not None else float("nan")
        else:
            dbcv = np.nan

        if entity_codes is not None:
            ari = _adjusted_rand_index_vs_entity_codes(entity_codes, labels)
        else:
            ari = float("nan")

        if n_clusters >= 1:
            metrics += [(size, icm, n_clusters, dbcv, ari)]

    return pd.DataFrame(
        metrics, columns=["min_cluster_size", "icm", "n_clusters", "dbcv", "ari"]
    )


def aggregate_grid_metrics(all_metrics_dfs: list[pd.DataFrame]) -> AggregatedGridReport:
    """
    Aggregate grid evaluation metrics across multiple samples into a typed report.

    Per ``min_cluster_size`` we keep DBCV mean, std, and count (so uncertainty is not
    discarded). ICM and cluster count are aggregated as means for diagnostics.
    """
    if not all_metrics_dfs:
        return AggregatedGridReport(points=())

    combined = pd.concat(all_metrics_dfs, ignore_index=True)
    if "ari" not in combined.columns:
        combined["ari"] = np.nan

    aggregated = (
        combined.groupby("min_cluster_size")
        .agg(
            {
                "dbcv": ["mean", "std", "count"],
                "icm": "mean",
                "n_clusters": "mean",
                "ari": ["mean", "std", "count"],
            }
        )
        .reset_index()
    )

    aggregated.columns = [
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

    aggregated["dbcv_std"] = aggregated["dbcv_std"].fillna(0.0)
    aggregated["dbcv_count"] = aggregated["dbcv_count"].astype(int)
    aggregated["ari_std"] = aggregated["ari_std"].fillna(0.0)
    aggregated["ari_count"] = aggregated["ari_count"].astype(int)

    points: list[AggregatedGridPoint] = []
    for _, row in aggregated.sort_values("min_cluster_size").iterrows():
        mcs = int(row["min_cluster_size"])
        points.append(
            AggregatedGridPoint(
                min_cluster_size=mcs,
                dbcv=ScalarMetricAggregate(
                    mean=float(row["dbcv_mean"]),
                    std=float(row["dbcv_std"]),
                    count=int(row["dbcv_count"]),
                ),
                icm_mean=float(row["icm_mean"]),
                n_clusters_mean=float(row["n_clusters_mean"]),
                ari=ScalarMetricAggregate(
                    mean=float(row["ari_mean"]),
                    std=float(row["ari_std"]),
                    count=int(row["ari_count"]),
                ),
            )
        )

    return AggregatedGridReport(points=tuple(points))
