from __future__ import annotations

import gzip
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
import math
import pathlib
from typing import Any
import numpy as np
import pandas as pd

from pelinker.config import ScreenerKind

_JSON_CLUSTERING_REPORT_SCHEMA = "pelinker.clustering_report.v2"


@dataclass(frozen=True)
class MeanWithUncertainty:
    """Sample mean and standard deviation (ddof=1) over repeated runs; ``std=0`` for a single run."""

    mean: float
    std: float


@dataclass(frozen=True)
class MetricMeanStd:
    """Mean and spread (sample std over CV folds) for one scalar metric."""

    mean: float
    std: float


@dataclass(frozen=True)
class ScreenerModelCvBlock:
    """Precision / recall / F1 for detecting the negative class (label 1) on held-out folds."""

    precision: MetricMeanStd
    recall: MetricMeanStd
    f1: MetricMeanStd


@dataclass(frozen=True)
class NegativeScreenerCvSummary:
    """Cross-validated LDA vs linear SVM on the same binary negative-detection task."""

    lda: ScreenerModelCvBlock
    svm: ScreenerModelCvBlock


def negative_screener_cv_summary_from_eval_dict(
    raw: dict[str, dict[str, dict[str, float]]],
) -> NegativeScreenerCvSummary:
    """Build a typed summary from :func:`pelinker.negative_screener.evaluate_negative_screener_models` output."""

    def _block(name: str) -> ScreenerModelCvBlock:
        b = raw[name]
        return ScreenerModelCvBlock(
            precision=MetricMeanStd(
                mean=float(b["precision"]["mean"]),
                std=float(b["precision"]["std"]),
            ),
            recall=MetricMeanStd(
                mean=float(b["recall"]["mean"]),
                std=float(b["recall"]["std"]),
            ),
            f1=MetricMeanStd(
                mean=float(b["f1"]["mean"]),
                std=float(b["f1"]["std"]),
            ),
        )

    return NegativeScreenerCvSummary(lda=_block("lda"), svm=_block("svm"))


@dataclass(frozen=True)
class NegativeScreenerInSampleMetrics:
    """Train-set precision / recall / F1 for detecting ``negative_label`` (binary label 1)."""

    precision: float
    recall: float
    f1: float
    n_kb_mentions: int
    """Rows whose ``entity`` is not the synthetic negative label (class 0)."""
    n_negative_label_mentions: int
    """Rows whose ``entity`` equals the synthetic negative label (class 1)."""
    kind: ScreenerKind


@dataclass(frozen=True)
class ClusteringFitMetrics:
    """Fit-time clustering diagnostics at a fixed ``min_cluster_size``."""

    min_cluster_size: int
    dbcv: float | None
    """HDBSCAN ``relative_validity_`` when available."""
    ari: float | None
    n_clusters_emergent: int
    noise_fraction: float
    n_samples: int


def _screener_cv_block_to_jsonable(
    block: ScreenerModelCvBlock,
) -> dict[str, dict[str, float]]:
    return {
        "precision": {"mean": block.precision.mean, "std": block.precision.std},
        "recall": {"mean": block.recall.mean, "std": block.recall.std},
        "f1": {"mean": block.f1.mean, "std": block.f1.std},
    }


def _negative_screener_cv_summary_to_jsonable(
    summary: NegativeScreenerCvSummary,
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        "lda": _screener_cv_block_to_jsonable(summary.lda),
        "svm": _screener_cv_block_to_jsonable(summary.svm),
    }


def _pool_negative_screener_cv_summaries(
    summaries: Sequence[NegativeScreenerCvSummary],
) -> NegativeScreenerCvSummary:
    """Mean/std across repeated clustering reports of the per-report CV ``mean`` fields."""

    def _collect(
        getter: Callable[[NegativeScreenerCvSummary], MetricMeanStd],
    ) -> MetricMeanStd:
        means = np.array([getter(s).mean for s in summaries], dtype=np.float64)
        return MetricMeanStd(
            mean=float(np.mean(means)),
            std=float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
        )

    lda = ScreenerModelCvBlock(
        precision=_collect(lambda s: s.lda.precision),
        recall=_collect(lambda s: s.lda.recall),
        f1=_collect(lambda s: s.lda.f1),
    )
    svm = ScreenerModelCvBlock(
        precision=_collect(lambda s: s.svm.precision),
        recall=_collect(lambda s: s.svm.recall),
        f1=_collect(lambda s: s.svm.f1),
    )
    return NegativeScreenerCvSummary(lda=lda, svm=svm)


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
    """Count of distinct KB ``entity`` labels in the frame used for PCA→UMAP (excludes ``pelinker.onto.NEGATIVE_LABEL`` when screening)."""

    n_clusters_emergent: int
    """Number of HDBSCAN clusters at the chosen ``min_cluster_size`` (noise label -1 excluded)."""

    metrics_df: pd.DataFrame
    assignments: pd.DataFrame
    pca_residuals: np.ndarray
    pca_mahalanobis: np.ndarray
    umap_clustering: np.ndarray
    umap_visualization: np.ndarray
    pca_reduced: np.ndarray
    negative_screener_cv: NegativeScreenerCvSummary | None = None
    """Stratified CV metrics for LDA and linear SVM (negative vs KB); ``None`` when screening is off or infeasible."""
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


# Basenames for artifacts under one report directory (``pelinker-fit`` / clustering search).
LINKER_FIT_CLUSTERING_REPORT_BASENAME = "linker_fit.clustering_report.json.gz"
CLUSTERING_SEARCH_RESULTS_CSV_BASENAME = "results.csv"
CLUSTERING_SEARCH_GRID_PER_SAMPLE_CSV_BASENAME = "results_grid_per_sample.csv"
CLUSTERING_SEARCH_FINE_METADATA_BASENAME = "fine_clustering_metadata.pkl.gz"
CLUSTERING_QUALITY_CHECKPOINT_BASENAME = "clustering_quality.state.json.gz"


def linker_fit_clustering_report_path(report_dir: str | pathlib.Path) -> pathlib.Path:
    """Filesystem path for the fit-time :class:`ClusteringReport` JSON under ``report_dir``."""
    return pathlib.Path(report_dir).expanduser() / LINKER_FIT_CLUSTERING_REPORT_BASENAME


def clustering_report_to_jsonable_dict(report: ClusteringReport) -> dict[str, Any]:
    """
    Flatten a :class:`ClusteringReport` into JSON-serializable built-ins (no DataFrames/ndarrays).

    Intended for ``json.dumps`` or for pickling a stable, language-adjacent blob. Schema version
    is stored under ``"schema"`` for forward compatibility.
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
        "negative_screener_cv": (
            None
            if report.negative_screener_cv is None
            else _json_normalize(
                _negative_screener_cv_summary_to_jsonable(report.negative_screener_cv)
            )
        ),
    }


def write_clustering_report_json(
    path: str | pathlib.Path, report: ClusteringReport, *, indent: int = 2
) -> None:
    """
    Serialize ``report`` with :func:`clustering_report_to_jsonable_dict` to UTF-8 JSON.

    Parent directories are created when missing.
    """
    p = pathlib.Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = clustering_report_to_jsonable_dict(report)

    with gzip.open(p, mode="wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(payload, f, indent=indent)


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
    negative_screener_cv: NegativeScreenerCvSummary | None = None

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
        ns = self.negative_screener_cv
        if ns is not None:

            def _flat(prefix: str, block: ScreenerModelCvBlock) -> None:
                row[f"{prefix}_precision_mean"] = block.precision.mean
                row[f"{prefix}_precision_std"] = block.precision.std
                row[f"{prefix}_recall_mean"] = block.recall.mean
                row[f"{prefix}_recall_std"] = block.recall.std
                row[f"{prefix}_f1_mean"] = block.f1.mean
                row[f"{prefix}_f1_std"] = block.f1.std

            _flat("screener_lda", ns.lda)
            _flat("screener_svm", ns.svm)
        return row


def _screener_cv_block_from_flat(
    row: dict[str, str | float | None], prefix: str
) -> ScreenerModelCvBlock:
    mean_key = f"{prefix}_precision_mean"
    return ScreenerModelCvBlock(
        precision=MetricMeanStd(
            mean=float(row[mean_key]),
            std=float(row.get(f"{prefix}_precision_std") or 0.0),
        ),
        recall=MetricMeanStd(
            mean=float(row[f"{prefix}_recall_mean"]),
            std=float(row.get(f"{prefix}_recall_std") or 0.0),
        ),
        f1=MetricMeanStd(
            mean=float(row[f"{prefix}_f1_mean"]),
            std=float(row.get(f"{prefix}_f1_std") or 0.0),
        ),
    )


def _negative_screener_cv_summary_from_flat_row(
    row: dict[str, str | float | None],
) -> NegativeScreenerCvSummary | None:
    if "screener_lda_precision_mean" not in row:
        return None
    return NegativeScreenerCvSummary(
        lda=_screener_cv_block_from_flat(row, "screener_lda"),
        svm=_screener_cv_block_from_flat(row, "screener_svm"),
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
        negative_screener_cv=_negative_screener_cv_summary_from_flat_row(row),
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

    ns_reports = [
        r.negative_screener_cv for r in reports if r.negative_screener_cv is not None
    ]
    ns_pooled: NegativeScreenerCvSummary | None = None
    if ns_reports:
        ns_pooled = _pool_negative_screener_cv_summaries(ns_reports)

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
        negative_screener_cv=ns_pooled,
    )
