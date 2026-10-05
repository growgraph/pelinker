from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pelinker.core.config import ScreenerKind
from pelinker.linker.distillation import DistillationFidelityMetrics
from pelinker.core.scaling import MinClusterSizeProvenance

"""Report dataclasses: the in-memory shape of every persisted pelinker report."""


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
    """Count of distinct KB ``entity`` labels in the frame used for PCA→UMAP (excludes ``pelinker.core.onto.NEGATIVE_LABEL`` when screening)."""

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
    distillation_fidelity: DistillationFidelityMetrics | None = None
    """Held-out agreement between the compact entity head and its HDBSCAN teacher.

    ``None`` for legacy fits (no student) and when the holdout was disabled or skipped.
    """
    min_cluster_size_provenance: MinClusterSizeProvenance | None = None
    """Where :attr:`hyperparameters` ``min_cluster_size`` came from (linker fit only).

    Explicit, extrapolated from a scale curve, or the bare default — recorded so a fitted
    model never carries an unexplained hyperparameter.
    """
    n_rows_realized: int | None = None
    """Mention rows this draw actually clustered on.

    The realized N, **not** the ``clustering_sample_rows`` cap: when the frame is smaller
    than the cap, or the cap is unset, the two differ. Sample-size-dependent
    hyperparameters (``min_cluster_size``, and through it HDBSCAN ``min_samples``) are
    only interpretable against this number — see :mod:`pelinker.core.scaling`.
    """


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
    n_rows_realized: MeanWithUncertainty | None = None
    """Mean (and std) realized mention-row count across this combination's draws.

    ``best_size`` is only interpretable against this scale; carrying it here is what lets
    :mod:`pelinker.core.scaling` fit ``min_cluster_size*(N)`` across runs.
    """

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
        nrr = self.n_rows_realized
        row["n_rows_realized"] = None if nrr is None else nrr.mean
        row["n_rows_realized_std"] = 0.0 if nrr is None else nrr.std
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
