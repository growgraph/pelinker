"""Entity-assignment heads trained on HDBSCAN cluster labels (compact predict path)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

DEFAULT_ENTITY_HEAD_HIDDEN_LAYERS: tuple[int, ...] = (256, 128, 128)


@dataclass
class EntityHead:
    """Multi-class head mapping UMAP coords → emergent cluster id + confidence score."""

    kind: str
    _estimator: MLPClassifier | LinearSVC
    classes_: np.ndarray

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict cluster labels and per-row scores.

        Returns:
            ``(cluster_ids, scores)`` each shape ``(n_samples,)``. Scores are in ``[0, 1]``
            (``max(predict_proba)`` for MLP; logistic of max decision margin for LinearSVC).
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        xf = np.asarray(X, dtype=np.float64)
        if isinstance(self._estimator, MLPClassifier):
            proba = self._estimator.predict_proba(xf)
            idx = np.argmax(proba, axis=1)
            scores = proba[np.arange(len(idx)), idx].astype(np.float64, copy=False)
            labels = self._estimator.classes_[idx].astype(np.int64, copy=False)
            return labels, scores

        # LinearSVC: one-vs-rest decision margins → soft scores via logistic of max margin.
        pred = self._estimator.predict(xf)
        labels = np.asarray(pred, dtype=np.int64).ravel()
        decision = np.asarray(self._estimator.decision_function(xf), dtype=np.float64)
        if decision.ndim == 1:
            margin = decision
        else:
            margin = decision.max(axis=1)
        scores = 1.0 / (1.0 + np.exp(-margin))
        return labels, scores.astype(np.float64, copy=False)


def fit_mlp_entity_head(
    X_umap: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    hidden_layer_sizes: tuple[int, ...] = DEFAULT_ENTITY_HEAD_HIDDEN_LAYERS,
    random_state: int = 13,
) -> EntityHead:
    """Fit an MLP on non-noise HDBSCAN labels (``cluster_labels != -1``)."""
    X, y = _non_noise_xy(X_umap, cluster_labels)
    est = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        early_stopping=True,
        validation_fraction=0.1,
        max_iter=500,
        random_state=random_state,
    )
    est.fit(X, y)
    return EntityHead(
        kind="mlp",
        _estimator=est,
        classes_=np.asarray(est.classes_, dtype=np.int64),
    )


def fit_linear_svc_entity_head(
    X_umap: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    random_state: int = 13,
) -> EntityHead:
    """Fit a linear multi-class SVM (study underfitting control; not the shipped default)."""
    X, y = _non_noise_xy(X_umap, cluster_labels)
    est = LinearSVC(max_iter=5000, random_state=random_state)
    est.fit(X, y)
    return EntityHead(
        kind="linear_svc",
        _estimator=est,
        classes_=np.asarray(est.classes_, dtype=np.int64),
    )


def _non_noise_xy(
    X_umap: np.ndarray, cluster_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if X_umap.ndim != 2:
        raise ValueError(f"X_umap must be 2D, got shape {X_umap.shape}")
    y = np.asarray(cluster_labels, dtype=np.int64).ravel()
    if len(y) != X_umap.shape[0]:
        raise ValueError(
            f"X_umap rows ({X_umap.shape[0]}) != cluster_labels length ({len(y)})"
        )
    mask = y != -1
    if not np.any(mask):
        raise ValueError("No non-noise cluster labels available to train entity head")
    X = np.asarray(X_umap[mask], dtype=np.float64)
    y_fit = y[mask]
    if len(np.unique(y_fit)) < 2:
        raise ValueError("Entity head requires at least two emergent clusters")
    return X, y_fit
