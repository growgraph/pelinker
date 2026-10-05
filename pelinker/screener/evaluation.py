from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pelinker.reports.schema import (
    AllScreenerCvResult,
    BinaryClassifierMetrics,
    MetricMeanStd,
    NegativeScreenerInSampleMetrics,
    PerDatapointScores,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from pelinker.core.config import (
    ManifoldOovScreenerConfig,
    NegativeScreenerConfig,
)
from pelinker.screener.projection import (
    ManifoldOovKind,
    make_manifold_linear_svc,
    make_manifold_rbf_svc,
    oov_estimator_scores,
    pick_projection_winner_by_mean_f1,
)
from pelinker.screener.ambient import (
    NegativeClassScreener,
    _linear_svc_for_embeddings,
)

"""Cross-validated evaluation of the negative and manifold-OOV screeners."""


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


def _pick_oov_winner_by_cv_folds(
    fold_pairs: list[tuple[np.ndarray, np.ndarray]],
    Xm: np.ndarray,
    y_i: np.ndarray,
    oov_cfg: ManifoldOovScreenerConfig,
) -> tuple[bool, ManifoldOovKind | str]:
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
        return False, "lda"
    svm_m_o = float(np.mean(svm_oov_fold))
    lda_m_o = float(np.mean(lda_oov_fold))
    rbf_m_o = float(np.mean(rbf_oov_fold))
    return True, pick_projection_winner_by_mean_f1(lda_m_o, svm_m_o, rbf_m_o)


def _collect_embedding_cv_folds(
    fold_pairs: list[tuple[np.ndarray, np.ndarray]],
    Xe: np.ndarray,
    y_i: np.ndarray,
    rs_emb: int,
) -> (
    tuple[
        list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        str,
    ]
    | None
):
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
    screener_best_kind = "lda" if auc_lda_g >= auc_svm_g else "svm"
    return embed_fold_rows, screener_best_kind


def _oov_scores_for_fold(
    winner_kind_oov: ManifoldOovKind | str,
    Xm: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y_i: np.ndarray,
    oov_cfg: ManifoldOovScreenerConfig,
) -> np.ndarray:
    Xtr_m, Xtst_m = Xm[train_idx], Xm[test_idx]
    ytr_e = y_i[train_idx]
    xm_tr = Xtr_m.astype(np.float64, copy=False)
    xm_ts = Xtst_m.astype(np.float64, copy=False)
    if winner_kind_oov == "svm":
        svm_w = make_manifold_linear_svc(xm_tr, oov_cfg)
        svm_w.fit(xm_tr, ytr_e)
        return oov_estimator_scores(svm_w, xm_ts)
    if winner_kind_oov == "rbf":
        rbf_w = make_manifold_rbf_svc(oov_cfg)
        rbf_w.fit(xm_tr, ytr_e)
        return oov_estimator_scores(rbf_w, xm_ts)
    lda_w = LinearDiscriminantAnalysis(solver="svd")
    lda_w.fit(xm_tr, ytr_e)
    return oov_estimator_scores(lda_w, xm_ts)


def _score_folds_with_oov_and_pool(
    embed_fold_rows: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
    *,
    screener_best_kind: str,
    oov_disabled: bool,
    winner_kind_oov: ManifoldOovKind | str,
    Xm: np.ndarray | None,
    y_i: np.ndarray,
    entity: np.ndarray,
    orig_idx: np.ndarray,
    oov_cfg: ManifoldOovScreenerConfig,
) -> (
    tuple[
        list[int],
        list[str],
        list[int],
        list[float],
        list[float],
        list[float],
        list[float],
        list[float],
        list[list[float]],
    ]
    | None
):
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

    for train_idx, test_idx, ytst_e, sc_lda, sc_svm in embed_fold_rows:
        sb_scores = sc_lda if screener_best_kind == "lda" else sc_svm

        if not oov_disabled and Xm is not None:
            o_scores = _oov_scores_for_fold(
                winner_kind_oov, Xm, train_idx, test_idx, y_i, oov_cfg
            )
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

    fold_lists = [
        lda_p,
        lda_r,
        lda_f1,
        lda_auc_l,
        svm_p,
        svm_r,
        svm_f1,
        svm_auc_l,
        sb_p,
        sb_r,
        sb_f1,
        sb_a,
        oov_p,
        oov_r,
        oov_f1,
        oov_a,
        cb_p,
        cb_r,
        cb_f1,
        cb_a,
    ]
    return (
        pool_orig,
        pool_ent,
        pool_y,
        pool_lda_s,
        pool_svm_s,
        pool_sb,
        pool_oov,
        pool_comb,
        fold_lists,
    )


def _build_all_screener_cv_result(
    fold_lists: list[list[float]],
    *,
    screener_best_kind: str,
    oov_kind_disp: str,
    oov_disabled: bool,
) -> AllScreenerCvResult:
    (
        lda_p,
        lda_r,
        lda_f1,
        lda_auc_l,
        svm_p,
        svm_r,
        svm_f1,
        svm_auc_l,
        sb_p,
        sb_r,
        sb_f1,
        sb_a,
        oov_p,
        oov_r,
        oov_f1,
        oov_a,
        cb_p,
        cb_r,
        cb_f1,
        cb_a,
    ) = fold_lists
    lda_mets = _metrics_from_fold_lists(lda_p, lda_r, lda_f1, lda_auc_l)
    svm_mets = _metrics_from_fold_lists(svm_p, svm_r, svm_f1, svm_auc_l)
    sb_mets = _metrics_from_fold_lists(sb_p, sb_r, sb_f1, sb_a)
    if oov_disabled:
        oov_mets = _zero_binary_metrics()
        comb_mets = sb_mets
    else:
        oov_mets = _metrics_from_fold_lists(oov_p, oov_r, oov_f1, oov_a)
        comb_mets = _metrics_from_fold_lists(cb_p, cb_r, cb_f1, cb_a)
    return AllScreenerCvResult(
        screener_lda=lda_mets,
        screener_svm=svm_mets,
        screener_best_kind=screener_best_kind,
        screener_best=sb_mets,
        oov_winner_kind=oov_kind_disp,
        oov=oov_mets,
        combined=comb_mets,
    )


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
        oov_run, winner_kind_oov = _pick_oov_winner_by_cv_folds(
            fold_pairs, Xm, y_i, oov_cfg
        )
        if not oov_run:
            Xm = None

    collected = _collect_embedding_cv_folds(fold_pairs, Xe, y_i, rs_emb)
    if collected is None:
        return None
    embed_fold_rows, screener_best_kind = collected

    oov_disabled = not oov_run or Xm is None
    oov_kind_disp = "disabled" if oov_disabled else str(winner_kind_oov)

    scored = _score_folds_with_oov_and_pool(
        embed_fold_rows,
        screener_best_kind=screener_best_kind,
        oov_disabled=oov_disabled,
        winner_kind_oov=winner_kind_oov,
        Xm=Xm,
        y_i=y_i,
        entity=entity,
        orig_idx=orig_idx,
        oov_cfg=oov_cfg,
    )
    if scored is None:
        return None
    (
        pool_orig,
        pool_ent,
        pool_y,
        pool_lda_s,
        pool_svm_s,
        pool_sb,
        pool_oov,
        pool_comb,
        fold_lists,
    ) = scored

    result = _build_all_screener_cv_result(
        fold_lists,
        screener_best_kind=screener_best_kind,
        oov_kind_disp=oov_kind_disp,
        oov_disabled=oov_disabled,
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
