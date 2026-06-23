from __future__ import annotations

import gzip
import json
import logging
import math
import pathlib
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from pelinker.config import ScreenerKind

logger = logging.getLogger(__name__)

_JSON_CLUSTERING_REPORT_SCHEMA = "pelinker.clustering_report.v10"
MODEL_SELECTION_RUN_REPORT_SCHEMA = "pelinker.model_selection.run_report.v2"


def entity_negative_label_mask_01(
    entities: pd.Series | np.ndarray,
    negative_label: str,
) -> np.ndarray:
    """
    Per-row binary labels aligned with ``entities``: ``1`` if the row's ``entity`` equals
    ``negative_label`` (same convention as the negative screener positive class), else ``0``.
    """
    if isinstance(entities, pd.Series):
        s = entities.astype(str).to_numpy()
    else:
        s = np.asarray(entities).astype(str)
    if s.size == 0:
        return np.zeros(0, dtype=np.int64)
    return (s == negative_label).astype(np.int64, copy=False)


def mention_quality_frame(
    dfr: pd.DataFrame,
    *,
    neg_mask: np.ndarray,
    cluster_kb: np.ndarray,
    pca_residuals: np.ndarray,
    pca_mahalanobis: np.ndarray,
    pca_spectral_entropy: np.ndarray,
    negative_label: str,
) -> pd.DataFrame:
    """Per-mention PCA quality and labels for all rows (KB clustered; negatives cluster=-1)."""
    optional = ["pmid", "mention"]
    optional_cols = [c for c in optional if c in dfr.columns]
    out = dfr[["entity", *optional_cols]].copy()
    cluster_full = np.full(len(dfr), -1, dtype=np.int64)
    cluster_full[~neg_mask] = np.asarray(cluster_kb, dtype=np.int64).ravel()
    out["cluster"] = cluster_full
    out["oov_label"] = entity_negative_label_mask_01(dfr["entity"], negative_label)
    out["pca_residual"] = np.asarray(pca_residuals, dtype=np.float64).ravel()
    out["pca_mahalanobis"] = np.asarray(pca_mahalanobis, dtype=np.float64).ravel()
    out["pca_spectral_entropy"] = np.asarray(
        pca_spectral_entropy, dtype=np.float64
    ).ravel()
    ordered = [
        "entity",
        *optional_cols,
        "cluster",
        "oov_label",
        "pca_residual",
        "pca_mahalanobis",
        "pca_spectral_entropy",
    ]
    return out[ordered]


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
class BinaryClassifierMetrics:
    """Precision / recall / F1 / AUC vs negative class (label 1); spread is fold-wise."""

    precision: MetricMeanStd
    recall: MetricMeanStd
    f1: MetricMeanStd
    auc: MetricMeanStd


@dataclass(frozen=True)
class AllScreenerCvResult:
    """Unified stratified CV: embedding screener (LDA vs SVM), manifold OOV, and stacked score."""

    screener_lda: BinaryClassifierMetrics
    screener_svm: BinaryClassifierMetrics
    screener_best_kind: str
    screener_best: BinaryClassifierMetrics
    oov_winner_kind: str
    oov: BinaryClassifierMetrics
    combined: BinaryClassifierMetrics


@dataclass(frozen=True)
class PerDatapointScores:
    """Out-of-sample scores per stratified-fold test datapoint."""

    orig_idx: list[int]
    entity: list[str]
    y_true: list[int]
    screener_lda_score: list[float]
    screener_svm_score: list[float]
    screener_best_score: list[float]
    oov_score: list[float]
    combined_score: list[float]


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


# --- flat row ↔ BinaryClassifierMetrics (symmetric pair) ---


def _binary_metrics_to_jsonable(
    m: BinaryClassifierMetrics,
) -> dict[str, dict[str, float]]:
    return {
        "precision": {"mean": m.precision.mean, "std": m.precision.std},
        "recall": {"mean": m.recall.mean, "std": m.recall.std},
        "f1": {"mean": m.f1.mean, "std": m.f1.std},
        "auc": {"mean": m.auc.mean, "std": m.auc.std},
    }


def _all_screener_cv_to_jsonable(result: AllScreenerCvResult) -> dict[str, object]:
    return {
        "screener_lda": _binary_metrics_to_jsonable(result.screener_lda),
        "screener_svm": _binary_metrics_to_jsonable(result.screener_svm),
        "screener_best_kind": result.screener_best_kind,
        "screener_best": _binary_metrics_to_jsonable(result.screener_best),
        "oov_winner_kind": result.oov_winner_kind,
        "oov": _binary_metrics_to_jsonable(result.oov),
        "combined": _binary_metrics_to_jsonable(result.combined),
    }


def _pool_binary_classifier_metrics_branch(
    summaries: Sequence[AllScreenerCvResult],
    branch: Callable[[AllScreenerCvResult], BinaryClassifierMetrics],
) -> BinaryClassifierMetrics:
    def _collect(field: str) -> MetricMeanStd:
        vals = np.array(
            [getattr(branch(s), field).mean for s in summaries],
            dtype=np.float64,
        )
        return MetricMeanStd(
            mean=float(np.mean(vals)),
            std=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        )

    return BinaryClassifierMetrics(
        precision=_collect("precision"),
        recall=_collect("recall"),
        f1=_collect("f1"),
        auc=_collect("auc"),
    )


def _pool_all_screener_cv_results(
    summaries: Sequence[AllScreenerCvResult],
) -> AllScreenerCvResult:
    """Pool mean±std across bootstrap samples from each report's CV ``mean`` fields."""
    if not summaries:
        raise ValueError("summaries must be non-empty")

    def _mode_str(values: list[str]) -> str:
        c = Counter(values)
        return str(c.most_common(1)[0][0])

    kinds_s = [s.screener_best_kind for s in summaries]
    kinds_o = [s.oov_winner_kind for s in summaries]
    return AllScreenerCvResult(
        screener_lda=_pool_binary_classifier_metrics_branch(
            summaries, lambda r: r.screener_lda
        ),
        screener_svm=_pool_binary_classifier_metrics_branch(
            summaries, lambda r: r.screener_svm
        ),
        screener_best_kind=_mode_str(kinds_s),
        screener_best=_pool_binary_classifier_metrics_branch(
            summaries, lambda r: r.screener_best
        ),
        oov_winner_kind=_mode_str(kinds_o),
        oov=_pool_binary_classifier_metrics_branch(summaries, lambda r: r.oov),
        combined=_pool_binary_classifier_metrics_branch(
            summaries, lambda r: r.combined
        ),
    )


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


@dataclass(frozen=True)
class LinkerFitDiagnostics:
    """Per-row training diagnostics for plotting (often stratified-subsampled)."""

    pca_residual: np.ndarray
    pca_mahalanobis: np.ndarray
    pca_spectral_entropy: np.ndarray
    oov_label: np.ndarray
    """``1`` iff ``entity == negative_label`` (same convention as :class:`ModelSelectionReport`)."""
    screener_decision: np.ndarray
    projection_score: np.ndarray
    n_total: int
    """Original mention count before subsampling (same as ``len(prepared)`` at fit time)."""
    sample_random_state: int
    """RNG seed used for stratified subsampling (or configured seed when no subsample)."""


def _copy_linker_fit_diagnostics(
    full: LinkerFitDiagnostics, random_state: int
) -> LinkerFitDiagnostics:
    return LinkerFitDiagnostics(
        pca_residual=np.asarray(full.pca_residual, dtype=np.float64).copy(),
        pca_mahalanobis=np.asarray(full.pca_mahalanobis, dtype=np.float64).copy(),
        pca_spectral_entropy=np.asarray(
            full.pca_spectral_entropy, dtype=np.float64
        ).copy(),
        oov_label=np.asarray(full.oov_label, dtype=np.int64).copy(),
        screener_decision=np.asarray(full.screener_decision, dtype=np.float64).copy(),
        projection_score=np.asarray(full.projection_score, dtype=np.float64).copy(),
        n_total=full.n_total,
        sample_random_state=random_state,
    )


def _slice_linker_fit_diagnostics(
    full: LinkerFitDiagnostics,
    indices: np.ndarray,
    random_state: int,
) -> LinkerFitDiagnostics:
    return LinkerFitDiagnostics(
        pca_residual=np.asarray(full.pca_residual, dtype=np.float64)[indices],
        pca_mahalanobis=np.asarray(full.pca_mahalanobis, dtype=np.float64)[indices],
        pca_spectral_entropy=np.asarray(full.pca_spectral_entropy, dtype=np.float64)[
            indices
        ],
        oov_label=np.asarray(full.oov_label, dtype=np.int64)[indices],
        screener_decision=np.asarray(full.screener_decision, dtype=np.float64)[indices],
        projection_score=np.asarray(full.projection_score, dtype=np.float64)[indices],
        n_total=full.n_total,
        sample_random_state=random_state,
    )


def subsample_diagnostics_stratified(
    full: LinkerFitDiagnostics,
    *,
    max_rows: int,
    random_state: int,
) -> LinkerFitDiagnostics:
    """
    Stratified subsample by ``oov_label`` (preserve class proportions, at least one row per
    non-empty class when two classes exist). Mirrors logic used in pairgrid plotting.
    """
    if max_rows < 1:
        raise ValueError("max_rows must be >= 1")
    n_total = full.n_total
    n = int(len(full.pca_residual))
    if n != n_total:
        raise ValueError(
            f"LinkerFitDiagnostics length mismatch: len(arrays)={n} vs n_total={n_total}"
        )

    def _copy_same() -> LinkerFitDiagnostics:
        return _copy_linker_fit_diagnostics(full, random_state)

    if n <= max_rows:
        return _copy_same()

    rng = np.random.default_rng(random_state)
    y = np.asarray(full.oov_label, dtype=np.int64).ravel()
    classes = np.unique(y)

    if len(classes) == 1:
        idx_all = np.flatnonzero(y == int(classes[0]))
        k = min(max_rows, len(idx_all))
        chosen = rng.choice(idx_all, size=k, replace=False)
    else:
        parts: list[np.ndarray] = []
        for cls in classes:
            idx = np.flatnonzero(y == int(cls))
            k_i = max(1, int(round(max_rows * len(idx) / n)))
            k_i = min(k_i, len(idx))
            parts.append(rng.choice(idx, size=k_i, replace=False))
        chosen = np.concatenate(parts)
        if len(chosen) > max_rows:
            chosen = rng.choice(chosen, size=max_rows, replace=False)

    chosen = np.sort(chosen.astype(np.int64, copy=False))
    return _slice_linker_fit_diagnostics(full, chosen, random_state)


@dataclass
class ModelSelectionReport:
    """Report containing clustering analysis results for one sample."""

    hyperparameters: ClusteringHyperparameters
    best_score: float
    """DBCV (``relative_validity_``) at the chosen ``min_cluster_size`` (mean when from aggregate)."""

    number_properties: int
    """Count of distinct KB ``entity`` labels in the frame used for PCA→UMAP (excludes ``pelinker.onto.NEGATIVE_LABEL`` when screening)."""

    n_clusters_emergent: int
    """HDBSCAN emergent cluster count at :attr:`hyperparameters` ``min_cluster_size``.

    In model-selection this is the grid-optimal ``best_size``, not an arbitrary MCS.
    To compare with ``Linker.fit`` at a fixed MCS, use
    :func:`n_clusters_at_min_cluster_size` on :attr:`metrics_df` instead.
    """

    metrics_df: pd.DataFrame
    assignments: pd.DataFrame
    pca_residuals: np.ndarray
    pca_mahalanobis: np.ndarray
    pca_spectral_entropy: np.ndarray
    oov_label: np.ndarray
    """Per-row OOV mask: ``1`` iff ``entity == negative_label`` (same length as ``pca_residuals``)."""
    umap_clustering: np.ndarray
    cluster_viz: np.ndarray
    cluster_viz_method: str
    pca_reduced: np.ndarray
    all_screener_cv: AllScreenerCvResult | None = None
    """Unified stratified CV for embedding screener, manifold OOV, and stacked score."""
    screener_oos_datapoints: PerDatapointScores | None = None
    """Per-datapoint OOS scores (not serialized in JSON clustering report)."""
    ari: float | None = None
    training_diagnostics: LinkerFitDiagnostics | None = None
    """Stratified-subsampled PCA quality + screener / manifold OOV scores (linker fit only)."""
    mention_quality: pd.DataFrame | None = None
    """All mentions (pos+neg) with PCA quality scores and oov_label; cluster=-1 for negatives."""


def n_clusters_at_min_cluster_size(
    metrics_df: pd.DataFrame,
    min_cluster_size: int,
) -> int | None:
    """``n_clusters`` from a grid ``metrics_df`` row at ``min_cluster_size`` (for fit parity)."""
    if (
        "min_cluster_size" not in metrics_df.columns
        or "n_clusters" not in metrics_df.columns
    ):
        return None
    hit = metrics_df.loc[
        metrics_df["min_cluster_size"] == min_cluster_size, "n_clusters"
    ]
    if hit.empty:
        return None
    return int(hit.iloc[0])


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
    logger.warning(
        "json_normalize: coercing unexpected type %s to str", type(obj).__name__
    )
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
LINKER_FIT_CLUSTER_COMPOSITION_BASENAME = "linker_fit.cluster_composition.json.gz"
LINKER_FIT_EMERGENT_CLUSTERS_BASENAME = "linker_fit.emergent_clusters.json"
LINKER_FIT_CLUSTER_KB_BASENAME = "linker_fit.cluster_kb.json"
_FIT_CLUSTER_COMPOSITION_SCHEMA = "pelinker.fit_cluster_composition.v2"
_EMERGENT_CLUSTERS_SCHEMA = "pelinker.emergent_clusters.v1"
MODEL_SELECTION_RUN_REPORT_BASENAME = "model_selection.run_report.json.gz"
MODEL_SELECTION_SUMMARY_JSON_SCHEMA = "pelinker.model_selection.summary.v1"
MODEL_SELECTION_SUMMARY_JSON_BASENAME = "model_selection.summary.json"
CLUSTERING_SEARCH_GRID_PER_SAMPLE_CSV_BASENAME = "results_grid_per_sample.csv"
CLUSTERING_SEARCH_GRID_CHOSEN_JSON_BASENAME = "grid_chosen_hyperparameters.json"
CLUSTERING_SEARCH_FINE_METADATA_BASENAME = "fine_clustering_metadata.jsonl.gz"
FINE_SCREENER_EVAL_BASENAME = "fine_screener_eval.jsonl.gz"
MODEL_SELECTION_CHECKPOINT_BASENAME = "model_selection.state.json.gz"


def linker_fit_clustering_report_path(report_dir: str | pathlib.Path) -> pathlib.Path:
    """Filesystem path for the fit-time :class:`ClusteringReport` JSON under ``report_dir``."""
    return pathlib.Path(report_dir).expanduser() / LINKER_FIT_CLUSTERING_REPORT_BASENAME


def linker_fit_cluster_kb_path(report_dir: str | pathlib.Path) -> pathlib.Path:
    """Filesystem path for the cluster-derived KB labels-map JSON under ``report_dir``."""
    return pathlib.Path(report_dir).expanduser() / LINKER_FIT_CLUSTER_KB_BASENAME


def linker_fit_cluster_composition_path(
    report_dir: str | pathlib.Path,
) -> pathlib.Path:
    """Filesystem path for the entity-weighted cluster composition artifact."""
    return (
        pathlib.Path(report_dir).expanduser() / LINKER_FIT_CLUSTER_COMPOSITION_BASENAME
    )


def linker_fit_emergent_clusters_path(report_dir: str | pathlib.Path) -> pathlib.Path:
    """Filesystem path for the emergent-cluster catalog JSON under ``report_dir``."""
    return pathlib.Path(report_dir).expanduser() / LINKER_FIT_EMERGENT_CLUSTERS_BASENAME


def write_cluster_composition_json(
    path: str | pathlib.Path,
    composition_df: pd.DataFrame,
    *,
    top_n: int = 3,
    weighting: str = "inv_sqrt_mention_count",
    summary: dict[str, int | float] | None = None,
    max_clusters_in_rows: int | None = None,
    indent: int = 2,
) -> None:
    """Serialize a processed cluster-composition table written at fit time."""
    from pelinker.cluster_composition_viz import ENTITY_WEIGHTING_INV_SQRT

    p = pathlib.Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": _FIT_CLUSTER_COMPOSITION_SCHEMA,
        "top_n": int(top_n),
        "weighting": weighting or ENTITY_WEIGHTING_INV_SQRT,
        "exclude_noise": True,
        "rows": _dataframe_to_jsonable_records(composition_df),
    }
    if summary is not None:
        payload["summary"] = _json_normalize(summary)
    if max_clusters_in_rows is not None:
        payload["max_clusters_in_rows"] = int(max_clusters_in_rows)
    with gzip.open(p, mode="wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(payload, f, indent=indent)


def read_cluster_composition_json(
    path: str | pathlib.Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a composition table and metadata written by :func:`write_cluster_composition_json`."""
    p = pathlib.Path(path).expanduser()
    with gzip.open(p, mode="rt", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)
    schema = str(raw.get("schema", ""))
    if schema not in (
        "pelinker.fit_cluster_composition.v1",
        _FIT_CLUSTER_COMPOSITION_SCHEMA,
    ):
        raise ValueError(f"Unsupported cluster composition schema: {schema!r}")
    meta = {k: v for k, v in raw.items() if k != "rows"}
    return pd.DataFrame(raw["rows"]), meta


def write_emergent_clusters_json(
    path: str | pathlib.Path,
    payload: dict[str, Any],
    *,
    indent: int = 2,
) -> None:
    """Write :func:`~pelinker.cluster_composition_viz.build_emergent_clusters_catalog` output."""
    p = pathlib.Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    if str(payload.get("schema", "")) != _EMERGENT_CLUSTERS_SCHEMA:
        raise ValueError(
            f"Expected schema {_EMERGENT_CLUSTERS_SCHEMA!r}, got {payload.get('schema')!r}"
        )
    with p.open("w", encoding="utf-8") as f:
        json.dump(_json_normalize(payload), f, indent=indent, ensure_ascii=False)


def read_emergent_clusters_json(path: str | pathlib.Path) -> dict[str, Any]:
    """Load emergent-cluster catalog JSON."""
    p = pathlib.Path(path).expanduser()
    with p.open(encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)
    if str(raw.get("schema", "")) != _EMERGENT_CLUSTERS_SCHEMA:
        raise ValueError(f"Unsupported emergent clusters schema: {raw.get('schema')!r}")
    return raw


def write_cluster_derived_labels_map_json(
    path: str | pathlib.Path,
    labels_map: dict[str, str],
    *,
    indent: int = 2,
) -> None:
    """
    Write a cluster-derived labels map (``entity_id`` → ``cluster_name``) to a plain JSON file.

    The file is human-readable and can be passed directly to a subsequent fit as a new KB
    ``labels_map``.  Parent directories are created when missing.
    """
    p = pathlib.Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(labels_map, f, indent=indent, ensure_ascii=False)


def model_selection_run_report_path(report_dir: str | pathlib.Path) -> pathlib.Path:
    """Filesystem path for the standardized model-selection aggregate report."""
    return pathlib.Path(report_dir).expanduser() / MODEL_SELECTION_RUN_REPORT_BASENAME


def model_selection_summary_json_path(report_dir: str | pathlib.Path) -> pathlib.Path:
    """Top-level replot summary (rankings, best combos) as plain JSON."""
    return pathlib.Path(report_dir).expanduser() / MODEL_SELECTION_SUMMARY_JSON_BASENAME


def write_model_selection_summary_json(
    path: str | pathlib.Path,
    payload: dict[str, Any],
    *,
    indent: int = 2,
) -> None:
    p = pathlib.Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    body = cast(dict[str, Any], _json_normalize(payload))
    with p.open("w", encoding="utf-8") as f:
        json.dump(body, f, indent=indent, ensure_ascii=False)


def _linker_fit_diagnostics_to_jsonable(d: LinkerFitDiagnostics) -> dict[str, Any]:
    return {
        "pca_residual": _ndarray_to_jsonable_nested(d.pca_residual),
        "pca_mahalanobis": _ndarray_to_jsonable_nested(d.pca_mahalanobis),
        "pca_spectral_entropy": _ndarray_to_jsonable_nested(d.pca_spectral_entropy),
        "oov_label": _ndarray_to_jsonable_nested(d.oov_label),
        "screener_decision": _ndarray_to_jsonable_nested(d.screener_decision),
        "projection_score": _ndarray_to_jsonable_nested(d.projection_score),
        "n_total": int(d.n_total),
        "sample_random_state": int(d.sample_random_state),
    }


def _linker_fit_diagnostics_from_jsonable(obj: object) -> LinkerFitDiagnostics | None:
    if obj is None or not isinstance(obj, dict):
        return None
    d = cast(dict[str, Any], obj)
    needed = (
        "pca_residual",
        "pca_mahalanobis",
        "pca_spectral_entropy",
        "oov_label",
        "screener_decision",
        "projection_score",
        "n_total",
        "sample_random_state",
    )
    if not all(k in d for k in needed):
        return None
    pr = np.asarray(d["pca_residual"], dtype=np.float64)
    pm = np.asarray(d["pca_mahalanobis"], dtype=np.float64)
    pe = np.asarray(d["pca_spectral_entropy"], dtype=np.float64)
    ol = np.asarray(d["oov_label"], dtype=np.int64).ravel()
    sd = np.asarray(d["screener_decision"], dtype=np.float64).ravel()
    mo = np.asarray(d["projection_score"], dtype=np.float64).ravel()
    n_tot = int(d["n_total"])
    srs = int(d["sample_random_state"])
    m = len(pr)
    if not (
        len(pm) == m and len(pe) == m and len(ol) == m and len(sd) == m and len(mo) == m
    ):
        return None
    return LinkerFitDiagnostics(
        pca_residual=pr,
        pca_mahalanobis=pm,
        pca_spectral_entropy=pe,
        oov_label=ol,
        screener_decision=sd,
        projection_score=mo,
        n_total=n_tot,
        sample_random_state=srs,
    )


def clustering_report_to_jsonable_dict(report: ModelSelectionReport) -> dict[str, Any]:
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
        "pca_spectral_entropy": _ndarray_to_jsonable_nested(
            report.pca_spectral_entropy
        ),
        "oov_label": _ndarray_to_jsonable_nested(report.oov_label),
        "umap_clustering": _ndarray_to_jsonable_nested(report.umap_clustering),
        "cluster_viz": _ndarray_to_jsonable_nested(report.cluster_viz),
        "cluster_viz_method": str(report.cluster_viz_method),
        "pca_reduced": _ndarray_to_jsonable_nested(report.pca_reduced),
        "ari": ari_out,
        "all_screener_cv": (
            None
            if report.all_screener_cv is None
            else _json_normalize(_all_screener_cv_to_jsonable(report.all_screener_cv))
        ),
        "training_diagnostics": (
            None
            if report.training_diagnostics is None
            else _json_normalize(
                _linker_fit_diagnostics_to_jsonable(report.training_diagnostics)
            )
        ),
    }


def write_clustering_report_json(
    path: str | pathlib.Path, report: ModelSelectionReport, *, indent: int = 2
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


def read_clustering_report_json(path: str | pathlib.Path) -> ModelSelectionReport:
    """
    Load a :class:`ModelSelectionReport` written by :func:`write_clustering_report_json`.

    Supports schema ``pelinker.clustering_report.v10`` (cluster-space viz coords).
    """
    p = pathlib.Path(path).expanduser()
    with gzip.open(p, mode="rt", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    schema = str(raw.get("schema", ""))
    if schema != "pelinker.clustering_report.v10":
        raise ValueError(f"Unsupported clustering report schema: {schema!r}")

    hp = raw["hyperparameters"]
    h = ClusteringHyperparameters(min_cluster_size=int(hp["min_cluster_size"]))
    metrics_df = pd.DataFrame(raw["metrics_df"])
    assignments = pd.DataFrame(raw["assignments"])

    def _farr(key: str) -> np.ndarray:
        return np.asarray(raw[key], dtype=np.float64)

    def _iarr(key: str) -> np.ndarray:
        return np.asarray(raw[key], dtype=np.int64)

    ari_raw = raw.get("ari")
    ari: float | None
    if ari_raw is None:
        ari = None
    else:
        ari = float(ari_raw)

    # Nested CV summaries are not round-tripped here (linker fit reports use ``None``).
    all_cv: AllScreenerCvResult | None = None

    td_raw = raw.get("training_diagnostics")
    training_diagnostics = _linker_fit_diagnostics_from_jsonable(td_raw)

    return ModelSelectionReport(
        hyperparameters=h,
        best_score=float(raw["best_score"]),
        number_properties=int(raw["number_properties"]),
        n_clusters_emergent=int(raw["n_clusters_emergent"]),
        metrics_df=metrics_df,
        assignments=assignments,
        pca_residuals=_farr("pca_residuals"),
        pca_mahalanobis=_farr("pca_mahalanobis"),
        pca_spectral_entropy=_farr("pca_spectral_entropy"),
        oov_label=_iarr("oov_label"),
        umap_clustering=np.asarray(raw["umap_clustering"], dtype=np.float64),
        cluster_viz=np.asarray(raw["cluster_viz"], dtype=np.float64),
        cluster_viz_method=str(raw["cluster_viz_method"]),
        pca_reduced=np.asarray(raw["pca_reduced"], dtype=np.float64),
        all_screener_cv=all_cv,
        screener_oos_datapoints=None,
        ari=ari,
        training_diagnostics=training_diagnostics,
    )


def _binary_metrics_into_row(
    row: dict[str, str | float | None],
    metrics: BinaryClassifierMetrics,
    prefix: str,
) -> None:
    row[f"{prefix}_precision_mean"] = metrics.precision.mean
    row[f"{prefix}_precision_std"] = metrics.precision.std
    row[f"{prefix}_recall_mean"] = metrics.recall.mean
    row[f"{prefix}_recall_std"] = metrics.recall.std
    row[f"{prefix}_f1_mean"] = metrics.f1.mean
    row[f"{prefix}_f1_std"] = metrics.f1.std
    row[f"{prefix}_auc_mean"] = metrics.auc.mean
    row[f"{prefix}_auc_std"] = metrics.auc.std


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
    all_screener_cv: AllScreenerCvResult | None = None

    def to_flat_dict(self) -> dict[str, str | float | None]:
        """Keys aligned with grid CSV / checkpoint / ``plot_heatmap`` expectations."""
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
        acv = self.all_screener_cv
        if acv is not None:
            row["screener_best_kind"] = acv.screener_best_kind
            _binary_metrics_into_row(row, acv.screener_best, "screener")
            row["oov_winner_kind"] = acv.oov_winner_kind
            _binary_metrics_into_row(row, acv.oov, "oov")
            _binary_metrics_into_row(row, acv.combined, "combined")
            _binary_metrics_into_row(row, acv.screener_lda, "screener_lda")
            _binary_metrics_into_row(row, acv.screener_svm, "screener_svm")
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
        all_screener_cv=_all_screener_cv_from_flat_row(row),
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

    acv_reports = [r.all_screener_cv for r in reports if r.all_screener_cv is not None]
    pooled_acv = _pool_all_screener_cv_results(acv_reports) if acv_reports else None

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
    )


@dataclass(frozen=True)
class ModelSelectionRunReport:
    """Standardized aggregate report for one model-selection run."""

    schema: str
    generated_at: str
    run_fingerprint: str
    run_config: dict[str, Any]
    checkpoint: dict[str, Any]
    combinations: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    best_overall: dict[str, Any] | None
    best_per_model: dict[str, float]


def model_selection_run_report_to_jsonable_dict(
    report: ModelSelectionRunReport,
) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        _json_normalize(
            {
                "schema": report.schema,
                "generated_at": report.generated_at,
                "run_fingerprint": report.run_fingerprint,
                "run_config": report.run_config,
                "checkpoint": report.checkpoint,
                "combinations": report.combinations,
                "failures": report.failures,
                "best_overall": report.best_overall,
                "best_per_model": report.best_per_model,
            }
        ),
    )


def write_model_selection_run_report_json(
    path: str | pathlib.Path,
    report: ModelSelectionRunReport,
    *,
    indent: int = 2,
) -> None:
    p = pathlib.Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = model_selection_run_report_to_jsonable_dict(report)
    with gzip.open(p, mode="wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(payload, f, indent=indent)
