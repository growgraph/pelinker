from dataclasses import dataclass

import pandas as pd


@dataclass
class ClusteringReport:
    """Report containing clustering analysis results."""

    best_size: int
    best_score: float
    number_properties: int
    metrics_df: (
        pd.DataFrame
    )  # DataFrame with columns: min_cluster_size, icm, n_clusters, dbcv
    df: pd.DataFrame  # DataFrame with UMAP embeddings and cluster labels
    hungarian_accuracy: float | None = (
        None  # Hungarian matching accuracy (None if not computed)
    )
