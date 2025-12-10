import logging
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
    )  # DataFrame with columns: min_cluster_size, icm, n_clusters, silhouette
    df: pd.DataFrame  # DataFrame with UMAP embeddings and cluster labels


def log_clustering_scores(results: list[dict], logger: logging.Logger) -> None:
    """
    Log clustering scores for different target cluster counts.

    Args:
        results: List of result dictionaries with target_count, actual_count, scores
        logger: Logger instance for output
    """
    logger.info("\n" + "=" * 80)
    logger.info("CLUSTERING SCORES")
    logger.info("=" * 80)
    logger.info(
        "%-15s %-15s %-20s %-20s %-15s",
        "Target",
        "Actual",
        "Score (before)",
        "Score (after)",
        "Valid Clusters",
    )
    logger.info("-" * 85)

    for r in results:
        logger.info(
            "%-15d %-15d %-20.4f %-20.4f %-15d",
            r["target_count"],
            r["actual_count"],
            r["score_before_filtering"],
            r["score_after_filtering"],
            r["n_valid_clusters"],
        )

    logger.info("=" * 80)


def log_clustering_results(cluster_results: list[dict], logger: logging.Logger) -> None:
    """
    Log clustering results in a formatted table.

    Args:
        cluster_results: List of cluster result dictionaries
        logger: Logger instance for output
    """
    logger.info("\n" + "=" * 80)
    logger.info("CLUSTER DETAILS")
    logger.info("=" * 80)
    logger.info("Total clusters: %d (excluding noise)", len(cluster_results))
    logger.info("\nCluster details:")
    logger.info(
        "%-10s %-30s %-50s %-10s",
        "Cluster ID",
        "Center ID",
        "Center Label",
        "Size",
    )
    logger.info("-" * 100)

    for cr in cluster_results:
        logger.info(
            "%-10d %-30s %-50s %-10d",
            cr["cluster_id"],
            str(cr["center_id"])[:30],
            str(cr["center_label"])[:50],
            cr["cluster_size"],
        )

    logger.info("=" * 80)
