from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pelinker.reports.schema import ClusteringFitMetrics
from sklearn.metrics import (
    adjusted_rand_score,
)


"""Clustering quality metrics computed against known entity labels."""


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
