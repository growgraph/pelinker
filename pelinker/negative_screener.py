"""LDA/SVM binary screen for synthetic negative mentions before PCA→UMAP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC

from pelinker.config import NegativeScreenerConfig, ScreenerKind


def _linear_svc_for_embeddings(X: np.ndarray, *, random_state: int | None) -> LinearSVC:
    """``dual=False`` when ``n_samples > n_features`` (typical for embedding rows)."""
    n, p = X.shape
    dual = n < p
    return LinearSVC(
        C=1.0,
        dual=dual,
        max_iter=10_000,
        random_state=random_state,
    )


@dataclass(frozen=True)
class _ModelCvLists:
    precision: list[float]
    recall: list[float]
    f1: list[float]


def evaluate_negative_screener_models(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 20,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, dict[str, dict[str, float]]] | None:
    """
    Stratified CV for LDA and linear SVM; ``y`` is 0/1 with 1 = negative class.

    Returns nested ``{model: {metric: {mean, std}}}`` or ``None`` if stratified CV is not feasible.
    """
    y_i = np.asarray(y, dtype=np.int64).ravel()
    if y_i.ndim != 1 or len(y_i) != len(X):
        raise ValueError("y must be 1D with same length as X")
    n0 = int(np.sum(y_i == 0))
    n1 = int(np.sum(y_i == 1))
    if n0 < 2 or n1 < 2:
        return None

    max_splits = min(n0, n1)
    n_splits_eff = min(int(n_splits), max_splits)
    if n_splits_eff < 2:
        return None

    splitter = StratifiedShuffleSplit(
        n_splits=n_splits_eff,
        test_size=test_size,
        random_state=random_state,
    )

    lda_lists = _ModelCvLists([], [], [])
    svm_lists = _ModelCvLists([], [], [])

    for train_idx, test_idx in splitter.split(X, y_i):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_i[train_idx], y_i[test_idx]

        lda = LinearDiscriminantAnalysis(solver="svd")
        lda.fit(X_train, y_train)
        y_pred_lda = lda.predict(X_test)
        lda_lists.precision.append(
            float(precision_score(y_test, y_pred_lda, pos_label=1, zero_division=0))
        )
        lda_lists.recall.append(
            float(recall_score(y_test, y_pred_lda, pos_label=1, zero_division=0))
        )
        lda_lists.f1.append(
            float(f1_score(y_test, y_pred_lda, pos_label=1, zero_division=0))
        )

        svm = _linear_svc_for_embeddings(X_train, random_state=random_state)
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)
        svm_lists.precision.append(
            float(precision_score(y_test, y_pred_svm, pos_label=1, zero_division=0))
        )
        svm_lists.recall.append(
            float(recall_score(y_test, y_pred_svm, pos_label=1, zero_division=0))
        )
        svm_lists.f1.append(
            float(f1_score(y_test, y_pred_svm, pos_label=1, zero_division=0))
        )

    def _summarize(lists: _ModelCvLists) -> dict[str, dict[str, float]]:
        return {
            "precision": {
                "mean": float(np.mean(lists.precision)),
                "std": float(np.std(lists.precision, ddof=1))
                if len(lists.precision) > 1
                else 0.0,
            },
            "recall": {
                "mean": float(np.mean(lists.recall)),
                "std": float(np.std(lists.recall, ddof=1))
                if len(lists.recall) > 1
                else 0.0,
            },
            "f1": {
                "mean": float(np.mean(lists.f1)),
                "std": float(np.std(lists.f1, ddof=1)) if len(lists.f1) > 1 else 0.0,
            },
        }

    return {
        "lda": _summarize(lda_lists),
        "svm": _summarize(svm_lists),
    }


@dataclass
class NegativeClassScreener:
    """Fitted binary classifier: predict whether a row is ``negative_label``.

    When training data has only one class (no ``negative_label`` rows or no KB rows),
    :attr:`_estimator` is ``None`` and :meth:`predict_is_negative` is always false
    (PCA→UMAP still runs on all rows at predict time).
    """

    kind: ScreenerKind
    negative_label: str
    _estimator: LinearDiscriminantAnalysis | LinearSVC | None

    @classmethod
    def fit_from_frame(
        cls,
        dfr: pd.DataFrame,
        config: NegativeScreenerConfig,
    ) -> NegativeClassScreener:
        """
        Fit on all rows of ``dfr`` (must include ``entity`` and ``embed``).

        Returns a trivial screener (no sklearn model) when labels are single-class so
        :class:`~pelinker.model.Linker` always holds a non-``None`` screener after fit.
        """
        if "embed" not in dfr.columns or "entity" not in dfr.columns:
            raise ValueError("dfr must contain 'embed' and 'entity' columns")
        if len(dfr) == 0:
            raise ValueError("Cannot fit NegativeClassScreener on an empty frame")
        X = np.stack(dfr["embed"].values).astype(np.float32, copy=False)
        y = (dfr["entity"].astype(str).values == config.negative_label).astype(np.int64)
        n0 = int(np.sum(y == 0))
        n1 = int(np.sum(y == 1))
        if n0 == 0 or n1 == 0:
            return cls(
                kind=config.kind, negative_label=config.negative_label, _estimator=None
            )

        X64 = X.astype(np.float64, copy=False)
        if config.kind == "lda":
            est: LinearDiscriminantAnalysis | LinearSVC = LinearDiscriminantAnalysis(
                solver="svd"
            )
        else:
            est = _linear_svc_for_embeddings(X64, random_state=config.cv_random_state)
        est.fit(X64, y)
        return cls(
            kind=config.kind, negative_label=config.negative_label, _estimator=est
        )

    def predict_is_negative(self, X: np.ndarray) -> np.ndarray:
        """Boolean mask, shape ``(n_samples,)`` — True iff predicted negative class."""
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if self._estimator is None:
            return np.zeros(X.shape[0], dtype=bool)
        pred = self._estimator.predict(X.astype(np.float64, copy=False))
        pred_i = np.asarray(pred, dtype=np.int64).ravel()
        return pred_i == 1

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Signed margin when available (``LinearSVC``, LDA); shape ``(n_samples,)``."""
        if self._estimator is None:
            return np.zeros(X.shape[0], dtype=np.float64)
        xf = X.astype(np.float64, copy=False)
        if isinstance(self._estimator, LinearSVC):
            return np.asarray(
                self._estimator.decision_function(xf), dtype=np.float64
            ).ravel()
        if isinstance(self._estimator, LinearDiscriminantAnalysis):
            t = self._estimator.transform(xf)
            return np.asarray(t, dtype=np.float64).ravel()
        raise TypeError(f"Unsupported estimator type: {type(self._estimator)}")
