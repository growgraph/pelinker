"""HDBSCAN min_cluster_size grid evaluation, cross-sample aggregation, and smooth optimum selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import hdbscan
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score
from torch.nn import functional as F


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


@dataclass(frozen=True)
class SmoothedGridOptimumResult:
    """Diagnostics for ``solve_optimal_min_cluster_size_from_aggregated``."""

    chosen_min_cluster_size: int
    score_mean_at_chosen: float
    score_std_at_chosen: float
    x: tuple[float, ...]
    y_objective: tuple[float, ...]
    y_smooth: tuple[float, ...]
    dy_dx: tuple[float, ...]
    d2y_dx2: tuple[float, ...]
    selection: Literal["plateau_derivative", "smoothed_argmax"]


def _ensure_odd_window(window: int) -> int:
    if window < 1:
        raise ValueError("smooth window must be >= 1")
    return window if window % 2 == 1 else window + 1


def _precision_weights(
    std: np.ndarray, count: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """Inverse-variance style weights; uses sample count as an extra multiplier."""
    var = std * std + eps
    return count.astype(float) / var


def _weighted_centered_moving_average(
    values: np.ndarray,
    point_weights: np.ndarray,
    window: int,
) -> np.ndarray:
    """Centered moving average; within each window, averages are weighted by ``point_weights``."""
    w = _ensure_odd_window(window)
    half = w // 2
    n = len(values)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        pw = point_weights[lo:hi]
        vv = values[lo:hi]
        denom = float(np.sum(pw))
        out[i] = float(np.sum(pw * vv) / denom) if denom > 0 else float("nan")
    return out


def _uniform_centered_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    w = _ensure_odd_window(window)
    half = w // 2
    n = len(values)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = float(np.mean(values[lo:hi]))
    return out


def _build_objective_y(
    report: AggregatedGridReport,
    method: str,
    uncertainty_penalty: float,
) -> np.ndarray:
    means = np.array([p.dbcv.mean for p in report.points], dtype=np.float64)
    stds = np.array([p.dbcv.std for p in report.points], dtype=np.float64)
    if method == "mean":
        return means
    if method == "lower_bound":
        return means - uncertainty_penalty * stds
    if method == "weighted":
        return means
    raise ValueError(f"Unknown optimization method: {method!r}")


def solve_optimal_min_cluster_size_from_aggregated(
    report: AggregatedGridReport,
    *,
    method: str = "mean",
    uncertainty_penalty: float = 1.0,
    smooth_window: int = 3,
    plateau_fraction: float = 0.92,
    derivative_rel_tol: float = 0.12,
    precision_weighted_smooth: bool | None = None,
) -> SmoothedGridOptimumResult:
    """
    Choose ``min_cluster_size`` from aggregated noisy grid scores.

    Builds an objective f(x) from per-point means (optionally mean - penalty·std), smooths f
    with a centered moving average (uniform or precision-weighted by count / variance), then
    prefers the **leftmost** x where the smoothed curve is near its plateau
    (f ≥ ``plateau_fraction`` · max f and |df/dx| small). If none qualify, uses the smoothed
    maximum.

    ``precision_weighted_smooth`` defaults to True for ``lower_bound`` and ``weighted``,
    and False for ``mean``.
    """
    if len(report.points) == 0:
        raise ValueError("No aggregated grid points provided")

    x = np.array([p.min_cluster_size for p in report.points], dtype=np.float64)
    y = _build_objective_y(report, method, uncertainty_penalty)
    stds = np.array([p.dbcv.std for p in report.points], dtype=np.float64)
    counts = np.array([p.dbcv.count for p in report.points], dtype=np.float64)
    dbcv_means = np.array([p.dbcv.mean for p in report.points], dtype=np.float64)

    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        raise ValueError("No finite objective values in aggregated grid report")

    x = x[finite]
    y = y[finite]
    stds = stds[finite]
    counts = counts[finite]
    dbcv_means = dbcv_means[finite]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    stds = stds[order]
    counts = counts[order]
    dbcv_means = dbcv_means[order]

    if precision_weighted_smooth is None:
        precision_weighted_smooth = method in ("lower_bound", "weighted")

    if precision_weighted_smooth:
        pweights = _precision_weights(stds, counts)
        y_s = _weighted_centered_moving_average(y, pweights, smooth_window)
    else:
        y_s = _uniform_centered_moving_average(y, smooth_window)

    dydx = np.gradient(y_s, x)
    d2ydx2 = np.gradient(dydx, x)

    y_max = float(np.nanmax(y_s))
    if not np.isfinite(y_max):
        raise ValueError("Smoothed objective is non-finite")

    abs_dydx = np.abs(dydx)
    scale = float(np.nanmax(abs_dydx))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    thresh = derivative_rel_tol * scale
    level = plateau_fraction * y_max

    chosen_idx: int | None = None
    selection: Literal["plateau_derivative", "smoothed_argmax"] = "smoothed_argmax"
    for i in range(len(x)):
        if not np.isfinite(y_s[i]) or not np.isfinite(dydx[i]):
            continue
        if y_s[i] >= level and abs_dydx[i] <= thresh:
            chosen_idx = i
            selection = "plateau_derivative"
            break

    if chosen_idx is None:
        chosen_idx = int(np.nanargmax(y_s))

    chosen_x = int(x[chosen_idx])
    score_mean_at = float(dbcv_means[chosen_idx])
    score_std_at = float(stds[chosen_idx])

    return SmoothedGridOptimumResult(
        chosen_min_cluster_size=chosen_x,
        score_mean_at_chosen=score_mean_at,
        score_std_at_chosen=score_std_at,
        x=tuple(float(v) for v in x),
        y_objective=tuple(float(v) for v in y),
        y_smooth=tuple(float(v) for v in y_s),
        dy_dx=tuple(float(v) for v in dydx),
        d2y_dx2=tuple(float(v) for v in d2ydx2),
        selection=selection,
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


def _adjusted_rand_index_vs_property_codes(
    property_codes: np.ndarray,
    cluster_labels: np.ndarray,
) -> float:
    """ARI between KB property codes and HDBSCAN labels; noise (-1) excluded (matches analysis)."""
    valid_mask = cluster_labels != -1
    if not valid_mask.any():
        return 0.0
    y_true = property_codes[valid_mask]
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

    Uses DBCV (Density-Based Clustering Validation) and, when ``property`` is present,
    adjusted Rand index vs. property codes (noise label -1 excluded).

    Returns:
        DataFrame with columns: min_cluster_size, icm, n_clusters, dbcv, ari
    """
    umap_values = dfr2[umap_columns].to_numpy(dtype=np.float32, copy=False)
    property_codes: np.ndarray | None = None
    if "property" in dfr2.columns:
        property_codes = (
            dfr2["property"]
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

        if property_codes is not None:
            ari = _adjusted_rand_index_vs_property_codes(property_codes, labels)
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
