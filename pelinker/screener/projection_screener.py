"""3D manifold OOV score model: residual, Mahalanobis, spectral entropy vs synthetic negatives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC, SVC

from pelinker.config import ManifoldOovScreenerConfig

ManifoldOovKind = Literal["lda", "svm", "rbf"]


def make_manifold_linear_svc(
    X_train: np.ndarray, cfg: ManifoldOovScreenerConfig
) -> LinearSVC:
    """Primal ``LinearSVC`` when ``n_samples > n_features`` (typical for 3D features)."""
    n, p = X_train.shape
    dual = n < p
    return LinearSVC(
        C=1.0,
        dual=dual,
        max_iter=10_000,
        random_state=cfg.cv_random_state,
    )


def make_manifold_rbf_svc(cfg: ManifoldOovScreenerConfig) -> SVC:
    g: float | Literal["scale", "auto"] = cfg.oov_rbf_gamma
    return SVC(
        kernel="rbf",
        C=float(cfg.oov_rbf_C),
        gamma=g,
        random_state=cfg.cv_random_state,
    )


def pick_projection_winner_by_mean_f1(
    lda_mean: float, svm_mean: float, rbf_mean: float
) -> ManifoldOovKind:
    """Argmax mean test F1; ties prefer simpler models (lda, then svm, then rbf)."""
    if lda_mean >= svm_mean and lda_mean >= rbf_mean:
        return "lda"
    if svm_mean >= rbf_mean:
        return "svm"
    return "rbf"


def _oob_scores(estimator: object, X: np.ndarray) -> np.ndarray:
    """Higher score => more like class 1 (OOV / synthetic negative)."""
    xf = X.astype(np.float64, copy=False)
    if isinstance(estimator, LinearDiscriminantAnalysis):
        df = estimator.decision_function(xf)
        return np.asarray(df, dtype=np.float64).ravel()
    if isinstance(estimator, LinearSVC):
        df = estimator.decision_function(xf)
        return np.asarray(df, dtype=np.float64).ravel()
    if isinstance(estimator, SVC):
        df = estimator.decision_function(xf)
        return np.asarray(df, dtype=np.float64).ravel()
    raise TypeError(f"Unsupported estimator: {type(estimator)!r}")


def oov_estimator_scores(estimator: object, X: np.ndarray) -> np.ndarray:
    """Public wrapper for OOV decision scores (higher => more like class 1)."""
    return _oob_scores(estimator, X)


@dataclass
class ManifoldOovScoreModel:
    """Fitted winner; :meth:`score` is used at predict time (higher => more OOV-like)."""

    kind: ManifoldOovKind
    threshold: float
    _estimator: LinearDiscriminantAnalysis | LinearSVC | SVC

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


def build_projection_training_arrays(
    prepared: pd.DataFrame,
    manifold_df: pd.DataFrame,
    transformer: object,
    *,
    negative_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    KB rows (class 0) aligned with ``manifold_df`` order; synthetic negatives (class 1)
    from ``prepared`` projected through ``transformer.transform``.

    Returns ``(X, y, prepared_row_pos)`` where ``prepared_row_pos[i]`` is the integer row
    index in ``prepared`` (iloc order) for stacked row ``X[i]`` (KB block then negatives).
    """
    neg_mask = prepared["entity"].astype(str).values == negative_label
    n_neg = int(np.sum(neg_mask))
    if n_neg == 0:
        return None
    if len(manifold_df) == 0:
        return None

    pos_kb = np.flatnonzero(~neg_mask)
    pos_neg = np.flatnonzero(neg_mask)
    prepared_row_pos = np.concatenate([pos_kb, pos_neg])

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
    return X, y, prepared_row_pos


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


def evaluate_projection_cv(
    X: np.ndarray,
    y: np.ndarray,
    cfg: ManifoldOovScreenerConfig,
) -> tuple[dict[str, object], ManifoldOovKind] | None:
    """
    Stratified CV: LDA vs linear ``LinearSVC`` vs RBF ``SVC`` by mean test F1 (pos_label=1).

    Returns ``(cv_payload, winner_kind)`` or ``None`` if CV infeasible.
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

    svm_f1s: list[float] = []
    lda_f1s: list[float] = []
    rbf_f1s: list[float] = []

    for train_idx, test_idx in splitter.split(X, y_i):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_i[train_idx], y_i[test_idx]

        lin = make_manifold_linear_svc(X_train, cfg)
        lin.fit(X_train.astype(np.float64, copy=False), y_train)
        svm_f1s.append(
            float(
                f1_score(
                    y_test,
                    lin.predict(X_test.astype(np.float64, copy=False)),
                    pos_label=1,
                    zero_division=0,
                )
            )
        )

        rbf = make_manifold_rbf_svc(cfg)
        rbf.fit(X_train.astype(np.float64, copy=False), y_train)
        rbf_f1s.append(
            float(
                f1_score(
                    y_test,
                    rbf.predict(X_test.astype(np.float64, copy=False)),
                    pos_label=1,
                    zero_division=0,
                )
            )
        )

        lda = LinearDiscriminantAnalysis(solver="svd")
        lda.fit(X_train.astype(np.float64, copy=False), y_train)
        lda_f1s.append(
            float(
                f1_score(
                    y_test,
                    lda.predict(X_test.astype(np.float64, copy=False)),
                    pos_label=1,
                    zero_division=0,
                )
            )
        )

    svm_m, svm_s = _mean_std(svm_f1s)
    lda_m, lda_s = _mean_std(lda_f1s)
    rbf_m, rbf_s = _mean_std(rbf_f1s)

    winner = pick_projection_winner_by_mean_f1(lda_m, svm_m, rbf_m)
    winner_f1 = lda_m if winner == "lda" else (svm_m if winner == "svm" else rbf_m)

    cv_payload: dict[str, object] = {
        "lda": {"f1_mean": lda_m, "f1_std": lda_s},
        "svm": {"f1_mean": svm_m, "f1_std": svm_s},
        "rbf": {"f1_mean": rbf_m, "f1_std": rbf_s},
        "winner_kind": winner,
        "winner_f1_mean": float(winner_f1),
    }
    return cv_payload, winner


def fit_projection_lda_no_cv(
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


def fit_projection_score_model(
    X: np.ndarray,
    y: np.ndarray,
    cfg: ManifoldOovScreenerConfig,
    *,
    cv_payload_and_winner: tuple[dict[str, object], ManifoldOovKind] | None,
) -> tuple[ManifoldOovScoreModel | None, dict[str, object] | None]:
    """
    If ``cv_payload_and_winner`` is ``None`` (CV infeasible), return ``(None, None)``.
    Otherwise refit the winner on all data and return ``(model, cv_payload)``.
    """
    if cv_payload_and_winner is None:
        return None, None
    cv_payload, winner = cv_payload_and_winner
    Xf = X.astype(np.float64, copy=False)
    y_i = np.asarray(y, dtype=np.int64).ravel()

    if winner == "svm":
        est = make_manifold_linear_svc(Xf, cfg)
        est.fit(Xf, y_i)
        return (
            ManifoldOovScoreModel(kind="svm", threshold=0.0, _estimator=est),
            cv_payload,
        )
    if winner == "rbf":
        est = make_manifold_rbf_svc(cfg)
        est.fit(Xf, y_i)
        return (
            ManifoldOovScoreModel(kind="rbf", threshold=0.0, _estimator=est),
            cv_payload,
        )
    est = LinearDiscriminantAnalysis(solver="svd")
    est.fit(Xf, y_i)
    return (
        ManifoldOovScoreModel(kind="lda", threshold=0.0, _estimator=est),
        cv_payload,
    )
