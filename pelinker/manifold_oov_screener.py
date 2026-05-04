"""3D manifold OOV score model: residual, Mahalanobis, spectral entropy vs synthetic negatives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from pelinker.config import ManifoldOovScreenerConfig

ManifoldOovKind = Literal["dt", "svm", "lda"]


def _oob_scores(estimator: object, X: np.ndarray) -> np.ndarray:
    """Higher score => more like class 1 (OOV / synthetic negative)."""
    if isinstance(estimator, DecisionTreeClassifier):
        proba = estimator.predict_proba(X.astype(np.float64, copy=False))
        p1 = np.asarray(proba[:, 1], dtype=np.float64).ravel()
        return (p1 - 0.5) * 2.0
    if isinstance(estimator, (LinearDiscriminantAnalysis, SVC)):
        df = estimator.decision_function(X.astype(np.float64, copy=False))
        return np.asarray(df, dtype=np.float64).ravel()
    raise TypeError(f"Unsupported estimator: {type(estimator)!r}")


@dataclass
class ManifoldOovScoreModel:
    """Fitted winner; :meth:`score` is used at predict time (higher => more OOV-like)."""

    kind: ManifoldOovKind
    threshold: float
    _estimator: DecisionTreeClassifier | LinearDiscriminantAnalysis | SVC
    dt_max_depth: int | None = None
    dt_min_samples_leaf: int | None = None

    def score(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError(f"X must be (n, 3), got {X.shape}")
        return _oob_scores(self._estimator, X)

    def is_oov(self, X: np.ndarray) -> np.ndarray:
        return self.score(X) > self.threshold


def _stack_metrics(
    residuals: np.ndarray,
    mahalanobis: np.ndarray,
    spectral_entropy: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            np.asarray(residuals, dtype=np.float64),
            np.asarray(mahalanobis, dtype=np.float64),
            np.asarray(spectral_entropy, dtype=np.float64),
        ]
    )


def build_manifold_oov_training_arrays(
    prepared: pd.DataFrame,
    manifold_df: pd.DataFrame,
    transformer: object,
    *,
    negative_label: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    KB rows (class 0) aligned with ``manifold_df`` order; synthetic negatives (class 1)
    from ``prepared`` projected through ``transformer.transform``.
    """
    neg_mask = prepared["entity"].astype(str).values == negative_label
    n_neg = int(np.sum(neg_mask))
    if n_neg == 0:
        return None
    if len(manifold_df) == 0:
        return None

    emb_kb = np.stack(manifold_df["embed"].values).astype(np.float32, copy=False)
    t = transformer
    _u0, _u1, res_kb, mah_kb, ent_kb = t.transform(emb_kb)  # type: ignore[union-attr]
    X0 = _stack_metrics(res_kb, mah_kb, ent_kb)
    y0 = np.zeros(len(manifold_df), dtype=np.int64)

    neg_df = prepared.loc[neg_mask]
    emb_neg = np.stack(neg_df["embed"].values).astype(np.float32, copy=False)
    _v0, _v1, res_n, mah_n, ent_n = t.transform(emb_neg)  # type: ignore[union-attr]
    X1 = _stack_metrics(res_n, mah_n, ent_n)
    y1 = np.ones(len(neg_df), dtype=np.int64)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])
    return X, y


def _cv_feasible(y: np.ndarray, n_splits: int) -> int | None:
    n0 = int(np.sum(y == 0))
    n1 = int(np.sum(y == 1))
    if n0 < 2 or n1 < 2:
        return None
    max_splits = min(n0, n1)
    n_eff = min(int(n_splits), max_splits)
    return n_eff if n_eff >= 2 else None


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return m, s


def evaluate_manifold_oov_cv(
    X: np.ndarray,
    y: np.ndarray,
    cfg: ManifoldOovScreenerConfig,
) -> tuple[dict[str, object], ManifoldOovKind, tuple[int | None, int] | None] | None:
    """
    Stratified CV: DT grid vs SVM vs LDA by mean test F1 (pos_label=1).

    Returns ``(cv_payload, winner_kind, winner_dt_params)`` or ``None`` if CV infeasible.
    ``cv_payload`` is JSON-serializable summary; ``winner_dt_params`` is
    ``(max_depth, min_samples_leaf)`` when DT wins else ``None``.
    """
    y_i = np.asarray(y, dtype=np.int64).ravel()
    n_eff = _cv_feasible(y_i, cfg.cv_n_splits)
    if n_eff is None:
        return None

    splitter = StratifiedKFold(
        n_splits=n_eff,
        shuffle=True,
        random_state=cfg.cv_random_state,
    )

    depth_list = list(cfg.dt_max_depth_candidates)
    leaf_list = list(cfg.dt_min_samples_leaf_candidates)
    dt_cell_f1s: dict[tuple[int | None, int], list[float]] = {}
    for d in depth_list:
        for leaf in leaf_list:
            dt_cell_f1s[(d, int(leaf))] = []

    svm_f1s: list[float] = []
    lda_f1s: list[float] = []

    for train_idx, test_idx in splitter.split(X, y_i):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_i[train_idx], y_i[test_idx]

        for d in depth_list:
            for leaf in leaf_list:
                dt = DecisionTreeClassifier(
                    max_depth=d,
                    min_samples_leaf=int(leaf),
                    random_state=cfg.cv_random_state,
                )
                dt.fit(X_train, y_train)
                y_pred = dt.predict(X_test)
                dt_cell_f1s[(d, int(leaf))].append(
                    float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
                )

        svm = SVC(kernel="linear", random_state=cfg.cv_random_state)
        svm.fit(X_train, y_train)
        y_pred_s = svm.predict(X_test)
        svm_f1s.append(float(f1_score(y_test, y_pred_s, pos_label=1, zero_division=0)))

        lda = LinearDiscriminantAnalysis(solver="svd")
        lda.fit(X_train, y_train)
        y_pred_l = lda.predict(X_test)
        lda_f1s.append(float(f1_score(y_test, y_pred_l, pos_label=1, zero_division=0)))

    dt_summary: list[dict[str, float | int | None]] = []
    best_dt_mean = -1.0
    best_dt_cell: tuple[int | None, int] | None = None
    for (d, leaf), f1_list in dt_cell_f1s.items():
        m, s = _mean_std(f1_list)
        dt_summary.append(
            {
                "max_depth": d,
                "min_samples_leaf": leaf,
                "f1_mean": m,
                "f1_std": s,
            }
        )
        if m > best_dt_mean:
            best_dt_mean = m
            best_dt_cell = (d, leaf)

    svm_m, svm_s = _mean_std(svm_f1s)
    lda_m, lda_s = _mean_std(lda_f1s)

    winner: ManifoldOovKind
    winner_dt: tuple[int | None, int] | None = None
    if best_dt_mean >= svm_m and best_dt_mean >= lda_m:
        winner = "dt"
        winner_dt = best_dt_cell
    elif svm_m >= lda_m:
        winner = "svm"
    else:
        winner = "lda"

    cv_payload: dict[str, object] = {
        "dt_cells": dt_summary,
        "svm": {"f1_mean": svm_m, "f1_std": svm_s},
        "lda": {"f1_mean": lda_m, "f1_std": lda_s},
        "winner_kind": winner,
        "winner_f1_mean": float(
            best_dt_mean if winner == "dt" else (svm_m if winner == "svm" else lda_m)
        ),
        "winner_dt_max_depth": winner_dt[0] if winner_dt is not None else None,
        "winner_dt_min_samples_leaf": winner_dt[1] if winner_dt is not None else None,
    }
    return cv_payload, winner, winner_dt


def fit_manifold_oov_lda_no_cv(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[ManifoldOovScoreModel, dict[str, object]]:
    """
    Fit LDA on all data when stratified CV is infeasible (e.g. only one minority-class sample).

    ``cv_payload`` records that CV was skipped.
    """
    Xf = X.astype(np.float64, copy=False)
    y_i = np.asarray(y, dtype=np.int64).ravel()
    est = LinearDiscriminantAnalysis(solver="svd")
    est.fit(Xf, y_i)
    payload: dict[str, object] = {
        "winner_kind": "lda",
        "winner_f1_mean": float("nan"),
        "cv_skipped": True,
        "reason": "insufficient_samples_per_class_for_stratified_cv",
        "n_kb": int(np.sum(y_i == 0)),
        "n_neg": int(np.sum(y_i == 1)),
    }
    return ManifoldOovScoreModel(kind="lda", threshold=0.0, _estimator=est), payload


def fit_manifold_oov_score_model(
    X: np.ndarray,
    y: np.ndarray,
    cfg: ManifoldOovScreenerConfig,
    *,
    cv_payload_and_winner: tuple[
        dict[str, object], ManifoldOovKind, tuple[int | None, int] | None
    ]
    | None,
) -> tuple[ManifoldOovScoreModel | None, dict[str, object] | None]:
    """
    If ``cv_payload_and_winner`` is ``None`` (CV infeasible), return ``(None, None)``.
    Otherwise refit the winner on all data and return ``(model, cv_payload)``.
    """
    if cv_payload_and_winner is None:
        return None, None
    cv_payload, winner, winner_dt = cv_payload_and_winner
    Xf = X.astype(np.float64, copy=False)
    y_i = np.asarray(y, dtype=np.int64).ravel()

    if winner == "dt":
        assert winner_dt is not None
        d, leaf = winner_dt
        est: DecisionTreeClassifier | LinearDiscriminantAnalysis | SVC = (
            DecisionTreeClassifier(
                max_depth=d,
                min_samples_leaf=int(leaf),
                random_state=cfg.cv_random_state,
            )
        )
        est.fit(Xf, y_i)
        return (
            ManifoldOovScoreModel(
                kind="dt",
                threshold=0.0,
                _estimator=est,
                dt_max_depth=d,
                dt_min_samples_leaf=int(leaf),
            ),
            cv_payload,
        )
    if winner == "svm":
        est = SVC(kernel="linear", random_state=cfg.cv_random_state)
        est.fit(Xf, y_i)
        return (
            ManifoldOovScoreModel(kind="svm", threshold=0.0, _estimator=est),
            cv_payload,
        )
    est = LinearDiscriminantAnalysis(solver="svd")
    est.fit(Xf, y_i)
    return (
        ManifoldOovScoreModel(kind="lda", threshold=0.0, _estimator=est),
        cv_payload,
    )
