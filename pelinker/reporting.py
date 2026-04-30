from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

_JSON_CLUSTERING_REPORT_SCHEMA = "pelinker.clustering_report.v1"


@dataclass(frozen=True)
class MeanWithUncertainty:
    """Sample mean and standard deviation (ddof=1) over repeated runs; ``std=0`` for a single run."""

    mean: float
    std: float


@dataclass(frozen=True)
class ClusteringHyperparameters:
    """
    HDBSCAN (and related) choices selected by the grid search / smoother.

    Add fields here as more knobs participate in optimization; call sites then stay typed.
    """

    min_cluster_size: int


@dataclass(frozen=True)
class HyperparameterSearchStats:
    """Distribution of chosen hyperparameters across repeated clustering samples."""

    min_cluster_size: MeanWithUncertainty


@dataclass
class ClusteringReport:
    """Report containing clustering analysis results for one sample."""

    hyperparameters: ClusteringHyperparameters
    best_score: float
    """DBCV (``relative_validity_``) at the chosen ``min_cluster_size`` (mean when from aggregate)."""

    number_properties: int
    """Count of distinct KB properties in the (filtered) frame used for clustering."""

    n_clusters_emergent: int
    """Number of HDBSCAN clusters at the chosen ``min_cluster_size`` (noise label -1 excluded)."""

    metrics_df: pd.DataFrame
    assignments: pd.DataFrame
    pca_residuals: np.ndarray
    pca_mahalanobis: np.ndarray
    umap_clustering: np.ndarray
    umap_visualization: np.ndarray
    pca_reduced: np.ndarray
    ari: float | None = None


def _json_normalize(obj: object) -> object:
    """Map values to types accepted by :func:`json.dumps` (no NaN/Inf; no numpy scalars)."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _json_normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_normalize(x) for x in obj]
    return str(obj)


def _dataframe_to_jsonable_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(k): _json_normalize(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _ndarray_to_jsonable_nested(arr: np.ndarray) -> Any:
    return _json_normalize(np.asarray(arr).tolist())


def clustering_report_to_jsonable_dict(report: ClusteringReport) -> dict[str, Any]:
    """
    Flatten a :class:`ClusteringReport` into JSON-serializable built-ins (no DataFrames/ndarrays).

    Intended for ``json.dumps`` or for pickling a stable, language-adjacent blob. Schema version
    is stored under ``\"schema\"`` for forward compatibility.
    """
    ari_out: float | None
    if report.ari is None:
        ari_out = None
    else:
        ari_f = float(report.ari)
        ari_out = None if math.isnan(ari_f) or math.isinf(ari_f) else ari_f

    return {
        "schema": _JSON_CLUSTERING_REPORT_SCHEMA,
        "hyperparameters": {
            "min_cluster_size": int(report.hyperparameters.min_cluster_size),
        },
        "best_score": _json_normalize(float(report.best_score)),
        "number_properties": int(report.number_properties),
        "n_clusters_emergent": int(report.n_clusters_emergent),
        "metrics_df": _dataframe_to_jsonable_records(report.metrics_df),
        "assignments": _dataframe_to_jsonable_records(report.assignments),
        "pca_residuals": _ndarray_to_jsonable_nested(report.pca_residuals),
        "pca_mahalanobis": _ndarray_to_jsonable_nested(report.pca_mahalanobis),
        "umap_clustering": _ndarray_to_jsonable_nested(report.umap_clustering),
        "umap_visualization": _ndarray_to_jsonable_nested(report.umap_visualization),
        "pca_reduced": _ndarray_to_jsonable_nested(report.pca_reduced),
        "ari": ari_out,
    }


@dataclass(frozen=True)
class ClusteringSearchSummaryRow:
    """
    One row of the model×layer clustering search table (singleton or fusion label).

    Use :meth:`to_flat_dict` for CSV / pandas / heatmaps (legacy column names).
    """

    model: str
    layer: str
    hyperparameters: HyperparameterSearchStats
    number_properties: MeanWithUncertainty
    n_clusters_emergent: MeanWithUncertainty
    dbcv: MeanWithUncertainty
    ari: MeanWithUncertainty | None

    def to_flat_dict(self) -> dict[str, str | float | None]:
        """Keys aligned with historical ``results.csv`` and ``plot_heatmap`` expectations."""
        h = self.hyperparameters.min_cluster_size
        p = self.number_properties
        k = self.n_clusters_emergent
        d = self.dbcv
        row: dict[str, str | float | None] = {
            "model": self.model,
            "layer": self.layer,
            "best_size": h.mean,
            "best_size_std": h.std,
            "number_properties": p.mean,
            "number_properties_std": p.std,
            "n_clusters_emergent": k.mean,
            "n_clusters_emergent_std": k.std,
            "best_score": d.mean,
            "best_score_std": d.std,
        }
        if self.ari is None:
            row["ari"] = None
            row["ari_std"] = 0.0
        else:
            ari = self.ari
            row["ari"] = ari.mean
            row["ari_std"] = ari.std
        return row


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
    )


def summarize_clustering_reports_for_search(
    reports: Sequence[ClusteringReport],
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

    if pooled_min_cluster_size is not None:
        sizes_mean = float(pooled_min_cluster_size)
        std_sizes = 0.0
        dbcv_at: list[float] = []
        for r in reports:
            m = r.metrics_df
            hit = m.loc[m["min_cluster_size"] == pooled_min_cluster_size, "dbcv"]
            if len(hit) > 0:
                dbcv_at.append(float(hit.iloc[0]))
        if dbcv_at:
            arr_dbcv = np.array(dbcv_at, dtype=np.float64)
            dbcv_mean = float(np.mean(arr_dbcv))
            dbcv_std = float(np.std(arr_dbcv)) if len(arr_dbcv) > 1 else 0.0
        else:
            dbcv_mean = float(np.mean(scores))
            dbcv_std = float(np.std(scores)) if n > 1 else 0.0
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
    )
