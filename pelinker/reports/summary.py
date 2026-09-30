"""Flat-row conversion and cross-report aggregation for search summaries."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


from pelinker.reports.schema import (
    AllScreenerCvResult,
    BinaryClassifierMetrics,
    ClusteringSearchSummaryRow,
    HyperparameterSearchStats,
    MeanWithUncertainty,
    MetricMeanStd,
    ModelSelectionReport,
    _pool_all_screener_cv_results,
)


def _binary_metrics_from_flat_row(
    row: dict[str, str | float | None],
    prefix: str,
) -> BinaryClassifierMetrics:
    """Inverse of :func:`_binary_metrics_into_row` / JSON branch of :func:`_binary_metrics_to_jsonable`."""

    def _m(field: str) -> MetricMeanStd:
        mk = f"{prefix}_{field}_mean"
        sk = f"{prefix}_{field}_std"
        return MetricMeanStd(
            mean=float(row[mk]),
            std=float(row.get(sk) or 0.0),
        )

    return BinaryClassifierMetrics(
        precision=_m("precision"),
        recall=_m("recall"),
        f1=_m("f1"),
        auc=_m("auc"),
    )


def _all_screener_cv_from_flat_row(
    row: dict[str, str | float | None],
) -> AllScreenerCvResult | None:
    """Reconstruct unified CV summary from flat CSV/checkpoint rows."""
    if "screener_lda_precision_mean" not in row or "screener_precision_mean" not in row:
        return None
    sb = row.get("screener_best_kind")
    ow = row.get("oov_winner_kind")
    if (
        not isinstance(sb, str)
        or not sb
        or not isinstance(ow, str)
        or not ow
        or "combined_precision_mean" not in row
    ):
        return None
    return AllScreenerCvResult(
        screener_lda=_binary_metrics_from_flat_row(row, "screener_lda"),
        screener_svm=_binary_metrics_from_flat_row(row, "screener_svm"),
        screener_best_kind=sb,
        screener_best=_binary_metrics_from_flat_row(row, "screener"),
        oov_winner_kind=ow,
        oov=_binary_metrics_from_flat_row(row, "oov"),
        combined=_binary_metrics_from_flat_row(row, "combined"),
    )


def clustering_search_summary_row_from_flat_dict(
    row: dict[str, str | float | None],
) -> ClusteringSearchSummaryRow:
    """Reconstruct :class:`ClusteringSearchSummaryRow` from :meth:`to_flat_dict` output."""
    ari_raw = row.get("ari")
    ari_block: MeanWithUncertainty | None
    if ari_raw is None or (isinstance(ari_raw, float) and math.isnan(ari_raw)):
        ari_block = None
    else:
        ari_block = MeanWithUncertainty(
            mean=float(ari_raw),
            std=float(row.get("ari_std") or 0.0),
        )
    # Absent in checkpoints written before n_rows_realized existed; stays None there.
    nrr_raw = row.get("n_rows_realized")
    nrr_block: MeanWithUncertainty | None
    if nrr_raw is None or (isinstance(nrr_raw, float) and math.isnan(nrr_raw)):
        nrr_block = None
    else:
        nrr_block = MeanWithUncertainty(
            mean=float(nrr_raw),
            std=float(row.get("n_rows_realized_std") or 0.0),
        )
    return ClusteringSearchSummaryRow(
        model=str(row["model"]),
        layer=str(row["layer"]),
        hyperparameters=HyperparameterSearchStats(
            min_cluster_size=MeanWithUncertainty(
                mean=float(row["best_size"]),
                std=float(row["best_size_std"] or 0.0),
            ),
        ),
        number_properties=MeanWithUncertainty(
            mean=float(row["number_properties"]),
            std=float(row["number_properties_std"] or 0.0),
        ),
        n_clusters_emergent=MeanWithUncertainty(
            mean=float(row["n_clusters_emergent"]),
            std=float(row["n_clusters_emergent_std"] or 0.0),
        ),
        dbcv=MeanWithUncertainty(
            mean=float(row["best_score"]),
            std=float(row["best_score_std"] or 0.0),
        ),
        ari=ari_block,
        all_screener_cv=_all_screener_cv_from_flat_row(row),
        n_rows_realized=nrr_block,
    )


def summarize_clustering_reports_for_search(
    reports: Sequence[ModelSelectionReport],
    *,
    model: str,
    layer: str,
    pooled_min_cluster_size: int | None = None,
) -> ClusteringSearchSummaryRow:
    """
    Aggregate repeated :class:`ClusteringReport` runs into one search summary row.

    When ``pooled_min_cluster_size`` is set (after aggregating grid curves across samples),
    ``best_size`` / ``best_size_std`` report that single consensus hyperparameter (std is 0)
    and ``dbcv`` is the mean (and std) of each sample's DBCV **at that grid point**.
    ``n_clusters_emergent`` remains the mean of per-sample fits at each sample's grid-optimal
    size; use :func:`n_clusters_at_min_cluster_size` on pooled grid CSV rows for a fixed MCS.

    Otherwise (independent runs or legacy callers) ``best_size`` is the mean of per-report
    chosen sizes and ``dbcv`` is the mean of each report's ``best_score``.

    Raises:
        ValueError: if ``reports`` is empty.
    """
    if not reports:
        raise ValueError("reports must be non-empty")

    sizes = np.array(
        [r.hyperparameters.min_cluster_size for r in reports], dtype=np.float64
    )
    scores = np.array([r.best_score for r in reports], dtype=np.float64)
    nprops = np.array([r.number_properties for r in reports], dtype=np.float64)
    n_clusters = np.array([r.n_clusters_emergent for r in reports], dtype=np.float64)
    ari_vals = [float(r.ari) for r in reports if r.ari is not None]

    n = len(reports)
    std_nprops = float(np.std(nprops)) if n > 1 else 0.0
    std_n_clusters = float(np.std(n_clusters)) if n > 1 else 0.0

    def _metric_at_pooled(column: str) -> list[float]:
        """Per-sample value of ``column`` at the pooled MCS (skipping samples without it)."""
        out: list[float] = []
        for r in reports:
            m = r.metrics_df
            if column not in m.columns:
                continue
            hit = m.loc[m["min_cluster_size"] == pooled_min_cluster_size, column]
            if len(hit) > 0 and np.isfinite(float(hit.iloc[0])):
                out.append(float(hit.iloc[0]))
        return out

    if pooled_min_cluster_size is not None:
        sizes_mean = float(pooled_min_cluster_size)
        std_sizes = 0.0
        dbcv_at = _metric_at_pooled("dbcv")
        if dbcv_at:
            arr_dbcv = np.array(dbcv_at, dtype=np.float64)
            dbcv_mean = float(np.mean(arr_dbcv))
            dbcv_std = float(np.std(arr_dbcv)) if len(arr_dbcv) > 1 else 0.0
        else:
            dbcv_mean = float(np.mean(scores))
            dbcv_std = float(np.std(scores)) if n > 1 else 0.0
        # DBCV and ARI must be read at the *same* min_cluster_size: they are combined into
        # one outer score, and r.ari is measured at each sample's own best size, not here.
        ari_at_pooled = _metric_at_pooled("ari")
        if ari_at_pooled:
            ari_vals = ari_at_pooled
    else:
        sizes_mean = float(np.mean(sizes))
        std_sizes = float(np.std(sizes)) if n > 1 else 0.0
        dbcv_mean = float(np.mean(scores))
        dbcv_std = float(np.std(scores)) if n > 1 else 0.0

    ari_block: MeanWithUncertainty | None
    if ari_vals:
        arr = np.array(ari_vals, dtype=np.float64)
        ari_block = MeanWithUncertainty(
            mean=float(np.mean(arr)),
            std=float(np.std(arr)) if len(arr) > 1 else 0.0,
        )
    else:
        ari_block = None

    acv_reports = [r.all_screener_cv for r in reports if r.all_screener_cv is not None]
    pooled_acv = _pool_all_screener_cv_results(acv_reports) if acv_reports else None

    rows_realized = [
        float(r.n_rows_realized) for r in reports if r.n_rows_realized is not None
    ]
    n_rows_block: MeanWithUncertainty | None = None
    if rows_realized:
        arr_rows = np.array(rows_realized, dtype=np.float64)
        n_rows_block = MeanWithUncertainty(
            mean=float(np.mean(arr_rows)),
            std=float(np.std(arr_rows)) if len(arr_rows) > 1 else 0.0,
        )

    return ClusteringSearchSummaryRow(
        model=model,
        layer=layer,
        hyperparameters=HyperparameterSearchStats(
            min_cluster_size=MeanWithUncertainty(
                mean=sizes_mean,
                std=std_sizes,
            ),
        ),
        number_properties=MeanWithUncertainty(
            mean=float(np.mean(nprops)),
            std=std_nprops,
        ),
        n_clusters_emergent=MeanWithUncertainty(
            mean=float(np.mean(n_clusters)),
            std=std_n_clusters,
        ),
        dbcv=MeanWithUncertainty(
            mean=dbcv_mean,
            std=dbcv_std,
        ),
        ari=ari_block,
        all_screener_cv=pooled_acv,
        n_rows_realized=n_rows_block,
    )
