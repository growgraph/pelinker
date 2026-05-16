from __future__ import annotations

import math
import pathlib
from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from pelinker.clustering_grid import (
    aggregate_grid_metrics,
    solve_optimal_min_cluster_size_from_aggregated,
)
from pelinker.plotting import (
    GRID_COL_CHOSEN_MIN_CLUSTER_SIZE,
    GRID_COL_SAMPLE_ARI,
    GRID_COL_SAMPLE_BEST_DBCV,
)
from pelinker.embedding_fusion import concat_mention_level_embedding_sources
from pelinker.reporting import (
    AllScreenerCvResult,
    BinaryClassifierMetrics,
    ClusteringFitMetrics,
    MetricMeanStd,
    ModelSelectionReport,
    NegativeScreenerInSampleMetrics,
    PerDatapointScores,
    entity_negative_label_mask_01,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from pelinker.config import (
    ClusteringOptimizationConfig,
    ManifoldOovScreenerConfig,
    NegativeScreenerConfig,
)
from pelinker.screener.projection_screener import (
    ManifoldOovKind,
    make_manifold_linear_svc,
    make_manifold_rbf_svc,
    oov_estimator_scores,
    pick_projection_winner_by_mean_f1,
)
from pelinker.screener.ambient_screener import (
    NegativeClassScreener,
    _linear_svc_for_embeddings,
)


def get_word_frequencies_from_library(
    language: str = "en",
    wordlist: str = "best",
) -> object | None:
    """
    Get word frequency lookup object from wordfreq library.

    Args:
        language: Language code (default: "en" for English)
        wordlist: Wordlist size - "best", "large", or "small" (default: "best")

    Returns:
        WordFrequencyLookup object with .get() method, or None if library not available
    """
    try:
        from wordfreq import word_frequency  # type: ignore

        # Return a callable that looks up frequencies
        # We'll use a lazy lookup approach
        class WordFrequencyLookup:
            def __init__(self, lang: str, wlist: str):
                self.lang = lang
                self.wlist = wlist
                self._cache: dict[str, float] = {}

            def get(self, word: str, default: float = 0.0) -> float:
                word_lower = word.lower()
                if word_lower not in self._cache:
                    try:
                        self._cache[word_lower] = word_frequency(
                            word_lower, self.lang, wordlist=self.wlist
                        )
                    except (KeyError, ValueError):
                        self._cache[word_lower] = default
                return self._cache[word_lower]

            def __getitem__(self, word: str) -> float:
                return self.get(word)

        return WordFrequencyLookup(language, wordlist)  # type: ignore
    except ImportError:
        return None


def _measure_label_simplicity(
    label: str,
    word_frequencies: Mapping[str, float],
    stopwords: Iterable[str] = (
        "is",
        "of",
        "the",
        "a",
        "an",
        "to",
        "for",
        "or",
        "in",
        "has",
    ),
    zero_freq_penalty: float = 1e-8,
    multiword_penalty: float = 0.2,
    stopword_penalty: float = 0.3,
) -> dict[str, int | float]:
    """..."""

    text = label.strip().lower()

    # Handle empty labels
    if not text:
        return {"char_count": 0, "word_count": 0, "simplicity_score": 0.0}

    words = text.split()
    word_count = len(words)

    stopword_set = set(stopwords)
    content_words = [w for w in words if w not in stopword_set]
    stopword_count = word_count - len(content_words)

    # Handle labels with only stopwords
    if not content_words:
        return {
            "char_count": len(text),
            "word_count": word_count,
            "simplicity_score": zero_freq_penalty,
        }

    # Get frequencies for content words
    content_freqs = [word_frequencies.get(w, zero_freq_penalty) for w in content_words]

    # Harmonic mean
    harmonic_mean_freq = len(content_freqs) / sum(
        1.0 / max(f, zero_freq_penalty) for f in content_freqs
    )

    # Apply penalties multiplicatively (but ensure they don't go negative)
    penalty_factor = 1.0

    if word_count > 1:
        penalty_factor *= max(0.0, 1.0 - multiword_penalty * (word_count - 1))

    if stopword_count > 0 and word_count > 1:
        penalty_factor *= max(0.0, 1.0 - stopword_penalty * stopword_count)

    simplicity_score = harmonic_mean_freq * penalty_factor

    return {
        "char_count": len(text),
        "word_count": word_count,
        "simplicity_score": simplicity_score,
    }


def compute_adjusted_rand_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute clustering quality via adjusted Rand index (ARI).

    Args:
        y_true: True labels (e.g., property names)
        y_pred: Predicted cluster labels

    Returns:
        ARI score.
    """
    # Filter out noise points (label -1) for accuracy computation
    valid_mask = y_pred != -1
    if not valid_mask.any():
        return 0.0

    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    if len(y_true_valid) == 0:
        return 0.0

    ari = adjusted_rand_score(y_true_valid, y_pred_valid)
    return float(ari)


def pooled_min_cluster_size_from_metrics_dfs(
    metrics_dfs: Sequence[pd.DataFrame],
    optimization_config: ClusteringOptimizationConfig | None = None,
) -> tuple[int, float]:
    """
    After all bootstrap samples have run a min_cluster_size grid, aggregate their metrics
    once and return the smoothed ``(chosen_min_cluster_size, raw objective mean at that grid point)``.
    The objective is set by ``ClusteringOptimizationConfig.grid_objective`` (default: pooled
    min–max normalized DBCV and ARI).
    """
    if not metrics_dfs:
        raise ValueError("metrics_dfs must be non-empty")
    config = optimization_config or ClusteringOptimizationConfig()
    aggregated = aggregate_grid_metrics(list(metrics_dfs))
    solved = solve_optimal_min_cluster_size_from_aggregated(
        aggregated,
        objective=config.grid_objective,
        method=config.optimization_method,
        smooth_window=config.grid_smooth_window,
        plateau_fraction=config.grid_plateau_fraction,
        derivative_rel_tol=config.grid_derivative_rel_tol,
    )
    return solved.chosen_min_cluster_size, solved.score_mean_at_chosen


def metrics_df_with_grid_sample_columns(
    report: ModelSelectionReport,
    *,
    model: str,
    layer: str,
    sample_idx: int,
    chosen_min_cluster_size: int | None = None,
) -> pd.DataFrame:
    """
    Per-sample grid rows for ``results_grid_per_sample.csv``.

    ``chosen_min_cluster_size`` defaults to the value used to fit this sample's clusters
    (per-sample grid argmax). Pass the pooled choice from
    :func:`pooled_min_cluster_size_from_metrics_dfs` so every row shares one consensus marker.
    """
    ari = report.ari
    h = (
        chosen_min_cluster_size
        if chosen_min_cluster_size is not None
        else report.hyperparameters.min_cluster_size
    )
    return report.metrics_df.assign(
        model=model,
        layer=layer,
        sample_idx=sample_idx,
        **{
            GRID_COL_CHOSEN_MIN_CLUSTER_SIZE: int(h),
            GRID_COL_SAMPLE_BEST_DBCV: float(report.best_score),
            GRID_COL_SAMPLE_ARI: float("nan") if ari is None else float(ari),
        },
    )


def split_by_negative_label(
    dfr: pd.DataFrame,
    negative_label: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Split a mention frame into a boolean mask of synthetic-negative rows and the
    manifold frame (KB / non-negative rows only).
    """
    neg_mask = dfr["entity"].astype(str).values == negative_label
    manifold_df = dfr.loc[~neg_mask].copy()
    return neg_mask, manifold_df


def _mention_quality_frame(
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
    optional = [c for c in ("pmid", "mention") if c in dfr.columns]
    out = dfr[["entity", *optional]].copy()
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
        *optional,
        "cluster",
        "oov_label",
        "pca_residual",
        "pca_mahalanobis",
        "pca_spectral_entropy",
    ]
    return out[ordered]


def _unified_cv_fold_count(y: np.ndarray, n_splits_requested: int) -> int | None:
    n0 = int(np.sum(y == 0))
    n1 = int(np.sum(y == 1))
    if n0 < 2 or n1 < 2:
        return None
    max_splits = min(n0, n1)
    n_eff = min(int(n_splits_requested), max_splits)
    return n_eff if n_eff >= 2 else None


def _minmax01_fold(x: np.ndarray) -> np.ndarray:
    xf = np.asarray(x, dtype=np.float64).ravel()
    lo = float(np.min(xf))
    hi = float(np.max(xf))
    if hi <= lo:
        return np.full(xf.shape[0], 0.5, dtype=np.float64)
    return (xf - lo) / (hi - lo)


def _fold_prfa(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[float, float, float, float]:
    y_i = np.asarray(y_true, dtype=np.int64).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    y_pred = (s > 0.0).astype(np.int64)
    prec = float(precision_score(y_i, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_i, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_i, y_pred, pos_label=1, zero_division=0))
    try:
        auc_v = float(roc_auc_score(y_i, s))
    except ValueError:
        auc_v = 0.5
    if math.isnan(auc_v) or math.isinf(auc_v):
        auc_v = 0.5
    return prec, rec, f1, auc_v


def _metrics_from_fold_lists(
    precs: list[float],
    recalls: list[float],
    f1s: list[float],
    aucs: list[float],
) -> BinaryClassifierMetrics:
    def _one(vals: list[float]) -> MetricMeanStd:
        arr = np.asarray(vals, dtype=np.float64)
        return MetricMeanStd(
            mean=float(np.mean(arr)),
            std=float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        )

    return BinaryClassifierMetrics(
        precision=_one(precs),
        recall=_one(recalls),
        f1=_one(f1s),
        auc=_one(aucs),
    )


def _zero_binary_metrics() -> BinaryClassifierMetrics:
    z = MetricMeanStd(0.0, 0.0)
    return BinaryClassifierMetrics(
        precision=z, recall=z, f1=z, auc=MetricMeanStd(0.5, 0.0)
    )


def evaluate_all_screeners_cv(
    X_embed: np.ndarray,
    X_manifold: np.ndarray | None,
    y: np.ndarray,
    entity: np.ndarray,
    orig_idx: np.ndarray,
    screener_cfg: NegativeScreenerConfig,
    oov_cfg: ManifoldOovScreenerConfig,
) -> tuple[AllScreenerCvResult, PerDatapointScores] | None:
    """
    Shared-stratified-fold CV for LDA/SVM negative screener, manifold OOV model, and stacked score.

    ``screener_best`` scores use the ROC winner (LDA vs SVM) on pooled OOS predictions.

    When ``oov_cfg.enabled`` is False or ``X_manifold`` is None, OOV branch is skipped:
    ``combined`` metrics match ``screener_best`` and ``oov_winner_kind`` is ``"disabled"``.
    """
    y_i = np.asarray(y, dtype=np.int64).ravel()
    n_splits = _unified_cv_fold_count(y_i, screener_cfg.cv_n_splits)
    if n_splits is None:
        return None

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=screener_cfg.cv_random_state,
    )
    Xe = np.asarray(X_embed, dtype=np.float64)
    fold_pairs = list(splitter.split(Xe, y_i))
    rs_emb = screener_cfg.cv_random_state

    oov_run = bool(oov_cfg.enabled) and X_manifold is not None
    Xm: np.ndarray | None = (
        np.asarray(X_manifold, dtype=np.float64) if oov_run else None
    )
    winner_kind_oov: ManifoldOovKind | str = "lda"

    if oov_run and Xm is not None:
        svm_oov_fold: list[float] = []
        lda_oov_fold: list[float] = []
        rbf_oov_fold: list[float] = []

        for train_idx, test_idx in fold_pairs:
            Xtr_m, Xtst_m = Xm[train_idx], Xm[test_idx]
            ytr_m, ytst_m = y_i[train_idx], y_i[test_idx]
            if len(np.unique(ytst_m)) < 2:
                continue
            xm64 = Xtr_m.astype(np.float64, copy=False)
            xt64 = Xtst_m.astype(np.float64, copy=False)

            lin_o = make_manifold_linear_svc(xm64, oov_cfg)
            lin_o.fit(xm64, ytr_m)
            svm_oov_fold.append(
                float(
                    f1_score(
                        ytst_m,
                        lin_o.predict(xt64),
                        pos_label=1,
                        zero_division=0,
                    )
                )
            )

            rbf_o = make_manifold_rbf_svc(oov_cfg)
            rbf_o.fit(xm64, ytr_m)
            rbf_oov_fold.append(
                float(
                    f1_score(
                        ytst_m,
                        rbf_o.predict(xt64),
                        pos_label=1,
                        zero_division=0,
                    )
                )
            )

            lda_est_o = LinearDiscriminantAnalysis(solver="svd")
            lda_est_o.fit(xm64, ytr_m)
            lda_oov_fold.append(
                float(
                    f1_score(
                        ytst_m,
                        lda_est_o.predict(xt64),
                        pos_label=1,
                        zero_division=0,
                    )
                )
            )

        if not lda_oov_fold:
            oov_run = False
            Xm = None
        else:
            svm_m_o = float(np.mean(svm_oov_fold))
            lda_m_o = float(np.mean(lda_oov_fold))
            rbf_m_o = float(np.mean(rbf_oov_fold))
            winner_kind_oov = pick_projection_winner_by_mean_f1(
                lda_m_o, svm_m_o, rbf_m_o
            )

    embed_fold_rows: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    pool_y_pooled: list[int] = []
    pool_lda_pooled: list[float] = []
    pool_svm_pooled: list[float] = []
    for train_idx, test_idx in fold_pairs:
        Xtr_e, Xtst_e = Xe[train_idx], Xe[test_idx]
        ytr_e, ytst_e = y_i[train_idx], y_i[test_idx]
        if len(np.unique(ytst_e)) < 2:
            continue

        lda_emb = LinearDiscriminantAnalysis(solver="svd")
        lda_emb.fit(Xtr_e, ytr_e)
        sc_lda = np.asarray(lda_emb.transform(Xtst_e), dtype=np.float64).ravel()
        svm_emb = _linear_svc_for_embeddings(Xtr_e, random_state=rs_emb)
        svm_emb.fit(Xtr_e, ytr_e)
        sc_svm = np.asarray(
            svm_emb.decision_function(Xtst_e),
            dtype=np.float64,
        ).ravel()
        embed_fold_rows.append((train_idx, test_idx, ytst_e, sc_lda, sc_svm))
        for j in range(int(ytst_e.shape[0])):
            pool_y_pooled.append(int(ytst_e[j]))
            pool_lda_pooled.append(float(sc_lda[j]))
            pool_svm_pooled.append(float(sc_svm[j]))

    if not pool_y_pooled:
        return None

    y_p = np.asarray(pool_y_pooled, dtype=np.int64)
    s_ld_p = np.asarray(pool_lda_pooled, dtype=np.float64)
    s_sv_p = np.asarray(pool_svm_pooled, dtype=np.float64)
    auc_lda_g = float(roc_auc_score(y_p, s_ld_p))
    auc_svm_g = float(roc_auc_score(y_p, s_sv_p))
    screener_best_kind: str = "lda" if auc_lda_g >= auc_svm_g else "svm"

    pool_orig: list[int] = []
    pool_ent: list[str] = []
    pool_y: list[int] = []
    pool_lda_s: list[float] = []
    pool_svm_s: list[float] = []
    pool_sb: list[float] = []
    pool_oov: list[float] = []
    pool_comb: list[float] = []

    lda_p, lda_r, lda_f1, lda_auc_l = [], [], [], []
    svm_p, svm_r, svm_f1, svm_auc_l = [], [], [], []
    sb_p, sb_r, sb_f1, sb_a = [], [], [], []
    oov_p, oov_r, oov_f1, oov_a = [], [], [], []
    cb_p, cb_r, cb_f1, cb_a = [], [], [], []

    oov_disabled = not oov_run or Xm is None
    oov_kind_disp: str = "disabled" if oov_disabled else str(winner_kind_oov)

    for train_idx, test_idx, ytst_e, sc_lda, sc_svm in embed_fold_rows:
        sb_scores = sc_lda if screener_best_kind == "lda" else sc_svm

        if not oov_disabled and Xm is not None:
            Xtr_m, Xtst_m = Xm[train_idx], Xm[test_idx]
            ytr_e = y_i[train_idx]
            xm_tr = Xtr_m.astype(np.float64, copy=False)
            xm_ts = Xtst_m.astype(np.float64, copy=False)
            if winner_kind_oov == "svm":
                svm_w = make_manifold_linear_svc(xm_tr, oov_cfg)
                svm_w.fit(xm_tr, ytr_e)
                o_scores = oov_estimator_scores(svm_w, xm_ts)
            elif winner_kind_oov == "rbf":
                rbf_w = make_manifold_rbf_svc(oov_cfg)
                rbf_w.fit(xm_tr, ytr_e)
                o_scores = oov_estimator_scores(rbf_w, xm_ts)
            else:
                lda_w = LinearDiscriminantAnalysis(solver="svd")
                lda_w.fit(xm_tr, ytr_e)
                o_scores = oov_estimator_scores(lda_w, xm_ts)
            mn_s = _minmax01_fold(sb_scores)
            mn_o = _minmax01_fold(o_scores)
            comb_scores = 0.5 * mn_s + 0.5 * mn_o
        else:
            o_scores = np.zeros_like(sb_scores, dtype=np.float64)
            comb_scores = sb_scores.astype(np.float64, copy=False)

        oi_fold = np.asarray(orig_idx[test_idx], dtype=np.int64).ravel()
        ent_fold = entity[test_idx]
        yt = ytst_e.astype(np.int64, copy=False)

        for j in range(int(yt.shape[0])):
            pool_orig.append(int(oi_fold[j]))
            pool_ent.append(str(ent_fold[j]))
            pool_y.append(int(yt[j]))
            pool_lda_s.append(float(sc_lda[j]))
            pool_svm_s.append(float(sc_svm[j]))
            pool_sb.append(float(sb_scores[j]))
            pool_oov.append(float(o_scores[j]))
            pool_comb.append(float(comb_scores[j]))

        p0, r0, f00, a0 = _fold_prfa(ytst_e, sc_lda)
        lda_p.append(p0)
        lda_r.append(r0)
        lda_f1.append(f00)
        lda_auc_l.append(a0)
        p1, r1, f01, a1 = _fold_prfa(ytst_e, sc_svm)
        svm_p.append(p1)
        svm_r.append(r1)
        svm_f1.append(f01)
        svm_auc_l.append(a1)
        p2, r2, f02, a2 = _fold_prfa(ytst_e, sb_scores)
        sb_p.append(p2)
        sb_r.append(r2)
        sb_f1.append(f02)
        sb_a.append(a2)
        p3, r3, f03, a3 = _fold_prfa(ytst_e, o_scores)
        oov_p.append(p3)
        oov_r.append(r3)
        oov_f1.append(f03)
        oov_a.append(a3)
        p4, r4, f04, a4 = _fold_prfa(ytst_e, comb_scores)
        cb_p.append(p4)
        cb_r.append(r4)
        cb_f1.append(f04)
        cb_a.append(a4)

    if not lda_p:
        return None

    lda_mets = _metrics_from_fold_lists(lda_p, lda_r, lda_f1, lda_auc_l)
    svm_mets = _metrics_from_fold_lists(svm_p, svm_r, svm_f1, svm_auc_l)
    sb_mets = _metrics_from_fold_lists(sb_p, sb_r, sb_f1, sb_a)

    if oov_disabled:
        oov_mets = _zero_binary_metrics()
        comb_mets = sb_mets
    else:
        oov_mets = _metrics_from_fold_lists(oov_p, oov_r, oov_f1, oov_a)
        comb_mets = _metrics_from_fold_lists(cb_p, cb_r, cb_f1, cb_a)

    result = AllScreenerCvResult(
        screener_lda=lda_mets,
        screener_svm=svm_mets,
        screener_best_kind=screener_best_kind,
        screener_best=sb_mets,
        oov_winner_kind=oov_kind_disp,
        oov=oov_mets,
        combined=comb_mets,
    )

    datapoints = PerDatapointScores(
        orig_idx=list(pool_orig),
        entity=list(pool_ent),
        y_true=list(pool_y),
        screener_lda_score=list(pool_lda_s),
        screener_svm_score=list(pool_svm_s),
        screener_best_score=list(pool_sb),
        oov_score=list(pool_oov),
        combined_score=list(pool_comb),
    )
    return result, datapoints


def fit_ambient_screener_with_metrics(
    dfr: pd.DataFrame,
    config: NegativeScreenerConfig,
) -> tuple[NegativeClassScreener, NegativeScreenerInSampleMetrics | None]:
    """
    Fit the persisted screener on ``dfr`` and report in-sample PR/F1 for detecting
    ``negative_label`` when both classes are present.
    """
    screener = NegativeClassScreener.fit_from_frame(dfr, config)
    y_true = (dfr["entity"].astype(str).values == config.negative_label).astype(
        np.int64
    )
    n_kb = int(np.sum(y_true == 0))
    n_neg = int(np.sum(y_true == 1))
    if n_kb == 0 or n_neg == 0:
        return screener, None
    X = np.stack(dfr["embed"].values).astype(np.float32, copy=False)
    y_pred = screener.predict_is_negative(X).astype(np.int64)
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    return screener, NegativeScreenerInSampleMetrics(
        precision=prec,
        recall=rec,
        f1=f1,
        n_kb_mentions=n_kb,
        n_negative_label_mentions=n_neg,
        kind=config.kind,
    )


def compute_clustering_fit_metrics(
    clusterer: object,
    manifold_df: pd.DataFrame,
    *,
    min_cluster_size: int,
    cluster_labels: np.ndarray,
) -> ClusteringFitMetrics:
    """DBCV, ARI vs ``entity``, and cluster counts for a fitted HDBSCAN model."""
    labels = np.asarray(cluster_labels, dtype=np.int64).ravel()
    n = int(labels.shape[0])
    label_set = set(labels.tolist())
    n_clusters_emergent = len(label_set) - (1 if -1 in label_set else 0)
    noise_count = int(np.sum(labels == -1))
    noise_fraction = float(noise_count) / float(n) if n > 0 else 0.0

    rv = getattr(clusterer, "relative_validity_", None)
    dbcv: float | None
    if rv is None:
        dbcv = None
    else:
        rv_f = float(rv)
        if math.isnan(rv_f) or math.isinf(rv_f):
            dbcv = None
        else:
            dbcv = rv_f

    ari_score: float | None
    if "entity" in manifold_df.columns and len(manifold_df) == n:
        property_labels = manifold_df["entity"].astype("category").cat.codes.values
        ari_score = compute_adjusted_rand_index(property_labels, labels)
    else:
        ari_score = None

    return ClusteringFitMetrics(
        min_cluster_size=min_cluster_size,
        dbcv=dbcv,
        ari=ari_score,
        n_clusters_emergent=n_clusters_emergent,
        noise_fraction=noise_fraction,
        n_samples=n,
    )


def mention_frame_from_embedding_paths(
    paths: Sequence[pathlib.Path],
    *,
    optimization_config: ClusteringOptimizationConfig | None = None,
    read_status: Callable[[str], None] | None = None,
    show_read_progress: bool = False,
) -> pd.DataFrame | None:
    """
    Load mention-level rows from parquet file(s) like :func:`~pelinker.selection.load_selection_frame`
    (batched read, optional multi-source inner join on keys), without subsampling.
    """
    cfg = optimization_config or ClusteringOptimizationConfig()
    return concat_mention_level_embedding_sources(
        paths,
        batch_size=cfg.batch_size,
        n_embedding_batches=cfg.n_embedding_batches,
        read_status=read_status,
        show_read_progress=show_read_progress,
    )


def drop_entities_with_few_mentions(
    frame: pd.DataFrame,
    min_mentions_per_entity: int,
    *,
    negative_label: str | None = None,
) -> pd.DataFrame:
    """
    Drop entities with fewer than ``min_mentions_per_entity`` rows (same rule as
    :func:`~pelinker.selection.load_selection_frame` / mention-level selection eval).

    When ``negative_label`` is set, that label is never dropped for low mention count
    (so thin negative tails remain for screener training).
    """
    if "entity" not in frame.columns:
        raise ValueError("frame must contain an 'entity' column")
    mention_count = frame["entity"].value_counts()
    low_count = mention_count[
        ~(mention_count >= min_mentions_per_entity)
    ].index.to_list()
    if negative_label is not None:
        low_count = [e for e in low_count if e != negative_label]
    return frame.loc[~frame["entity"].isin(low_count)].copy()


def embeddings_dict_to_dataframe(
    embeddings_dict: dict[str, tuple[str, torch.Tensor | np.ndarray]],
) -> pd.DataFrame:
    """
    Convert embeddings dictionary to DataFrame format expected by transform artifacts.

    Args:
        embeddings_dict: Dictionary mapping id -> (label, embedding)

    Returns:
        DataFrame with columns: id, label, embed
    """
    embeddings_list = []
    id_list = []
    label_list = []

    for id_val, (label, emb) in embeddings_dict.items():
        if isinstance(emb, torch.Tensor):
            emb_np = emb.detach().cpu().numpy()
        else:
            emb_np = np.array(emb)
        embeddings_list.append(emb_np)
        id_list.append(id_val)
        label_list.append(label)

    return pd.DataFrame({"id": id_list, "label": label_list, "embed": embeddings_list})


def compute_kb_generality_scores(
    embeddings: np.ndarray,
    labels: list[str],
    k_neighbors: int = 10,
    metric: str = "cosine",
    word_frequencies: Mapping[str, float] | None = None,
    density_weight: float = 0.5,
) -> np.ndarray:
    """
    Compute generality scores for entities based on KB statistics.

    Combines embedding-space density with label simplicity to identify generic vs specific terms.
    Generic terms tend to have:
    - Many similar neighbors (high density)
    - High average similarity to neighbors
    - Shorter, simpler labels (fewer words, common words)
    - Central position in semantic space

    Args:
        embeddings: Array of shape (n_points, n_features) containing embeddings
        labels: List of labels corresponding to embeddings
        k_neighbors: Number of nearest neighbors to consider
        metric: Distance metric ('cosine' or 'euclidean')
        word_frequencies: Optional word frequency mapping for simplicity scoring
        density_weight: Weight for embedding density vs label simplicity (0.0 = pure simplicity, 1.0 = pure density)

    Returns:
        Array of generality scores (higher = more generic), shape (n_points,)
    """
    from sklearn.neighbors import NearestNeighbors

    n_points = embeddings.shape[0]
    k_neighbors = min(k_neighbors, n_points - 1)

    if k_neighbors < 1:
        return np.ones(n_points)

    # Normalize embeddings for cosine distance
    if metric == "cosine":
        embeddings_norm = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        )
    else:
        embeddings_norm = embeddings

    # Find k nearest neighbors for each point
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric=metric)
    nn.fit(embeddings_norm)
    distances, indices = nn.kneighbors(embeddings_norm)

    # Compute embedding-space density scores
    density_scores = np.zeros(n_points)

    for i in range(n_points):
        # Get neighbors (excluding self)
        neighbor_distances = distances[i, 1:]

        # Convert distances to similarities (for cosine: similarity = 1 - distance)
        if metric == "cosine":
            similarities = 1.0 - neighbor_distances
        else:
            # For euclidean, use inverse distance (with smoothing)
            similarities = 1.0 / (1.0 + neighbor_distances)

        # Density = average similarity to neighbors
        # Higher similarity means the term is in a dense region (more generic)
        density_scores[i] = similarities.mean()

    density_scores = np.log(density_scores)
    # Normalize density scores to [0, 1] range
    if density_scores.max() > density_scores.min():
        density_scores_norm = (density_scores - density_scores.min()) / (
            density_scores.max() - density_scores.min()
        )
    else:
        density_scores_norm = np.ones_like(density_scores)

    # Compute label simplicity scores
    if word_frequencies is None:
        word_frequencies = {}

    simplicity_scores = np.zeros(n_points)
    for i, label in enumerate(labels):
        simplicity_metrics = _measure_label_simplicity(
            str(label), word_frequencies=word_frequencies
        )
        simplicity_scores[i] = simplicity_metrics["simplicity_score"]

    simplicity_scores = np.log(simplicity_scores)

    # Normalize simplicity scores to [0, 1] range
    if simplicity_scores.max() > simplicity_scores.min():
        simplicity_scores_norm = (simplicity_scores - simplicity_scores.min()) / (
            simplicity_scores.max() - simplicity_scores.min()
        )
    else:
        simplicity_scores_norm = np.ones_like(simplicity_scores)

    # Combine density and simplicity scores
    # Shorter, simpler terms should be preferred even if density is similar
    generality_scores = (
        density_weight * density_scores_norm
        + (1 - density_weight) * simplicity_scores_norm
    )

    return generality_scores
