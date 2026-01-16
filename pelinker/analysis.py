import pathlib

import numpy as np
import pandas as pd
import torch
import hdbscan
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F
from collections.abc import Mapping
from typing import Iterable
from numpy.random import RandomState

from pelinker.transform import TransformConfig, get_umap_columns, transform_embeddings


def evaluate_cluster_size_grid(
    dfr2: pd.DataFrame,
    umap_columns: list[str],
    sizes: list[int],
) -> pd.DataFrame:
    """
    Evaluate clustering metrics on a grid of min_cluster_size values.

    Uses DBCV (Density-Based Clustering Validation) metric.

    Args:
        dfr2: DataFrame with UMAP-reduced embeddings
        umap_columns: List of UMAP column names
        sizes: List of min_cluster_size values to evaluate

    Returns:
        DataFrame with columns: min_cluster_size, icm, n_clusters, dbcv
    """
    metrics = []
    for size in sizes:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=size, gen_min_span_tree=True)
        labels = clusterer.fit_predict(dfr2[umap_columns])
        dfr2_temp = dfr2.copy()
        dfr2_temp["class"] = pd.DataFrame(
            labels, columns=["class"], index=dfr2_temp.index
        )

        ic = []
        for ix, group in dfr2_temp.groupby("class"):
            if ix == -1:  # Skip noise points
                continue
            tgroup = torch.from_numpy(group[umap_columns].values)
            st = cosine_similarity_std(tgroup)
            ic += [st]

        icm = np.mean(ic) if ic else np.nan

        # Compute DBCV score
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        # DBCV is available as relative_validity_ attribute
        if n_clusters >= 2 and hasattr(clusterer, "relative_validity_"):
            dbcv = float(clusterer.relative_validity_)
        else:
            dbcv = np.nan

        # Only record if we have at least one valid cluster
        if n_clusters >= 1:
            metrics += [(size, icm, n_clusters, dbcv)]

    return pd.DataFrame(
        metrics, columns=["min_cluster_size", "icm", "n_clusters", "dbcv"]
    )


def aggregate_grid_metrics(
    all_metrics_dfs: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Aggregate grid evaluation metrics across multiple samples.

    Args:
        all_metrics_dfs: List of metrics DataFrames from multiple samples

    Returns:
        DataFrame with columns:
        - min_cluster_size
        - dbcv_mean, dbcv_std, dbcv_count
        - icm_mean, n_clusters_mean
    """
    if not all_metrics_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_metrics_dfs, ignore_index=True)

    aggregated = (
        combined.groupby("min_cluster_size")
        .agg(
            {
                "dbcv": ["mean", "std", "count"],
                "icm": "mean",
                "n_clusters": "mean",
            }
        )
        .reset_index()
    )

    # Flatten column names
    aggregated.columns = [
        "min_cluster_size",
        "dbcv_mean",
        "dbcv_std",
        "dbcv_count",
        "icm_mean",
        "n_clusters_mean",
    ]

    # Fill NaN std with 0 (single sample case)
    aggregated["dbcv_std"] = aggregated["dbcv_std"].fillna(0.0)

    return aggregated


def find_optimal_from_grid(
    aggregated_metrics: pd.DataFrame,
    method: str = "mean",
    uncertainty_penalty: float = 1.0,
) -> tuple[int, float, float]:
    """
    Find optimal min_cluster_size from aggregated grid metrics.

    Uses DBCV (Density-Based Clustering Validation) as the metric (maximize).

    Args:
        aggregated_metrics: DataFrame from aggregate_grid_metrics()
        method: How to select optimum:
            - "mean": Use mean DBCV score (default)
            - "lower_bound": Use mean - uncertainty_penalty * std (conservative)
            - "weighted": Weight by inverse variance (more samples = more weight)
        uncertainty_penalty: Multiplier for std when using "lower_bound" method

    Returns:
        (best_size, best_score_mean, best_score_std) where score is DBCV
    """
    if len(aggregated_metrics) == 0:
        raise ValueError("No aggregated metrics provided")

    sizes = aggregated_metrics["min_cluster_size"].values
    means = aggregated_metrics["dbcv_mean"].values
    stds = aggregated_metrics["dbcv_std"].values

    if method == "mean":
        scores = means
    elif method == "lower_bound":
        # Conservative: prefer points with lower uncertainty
        scores = means - uncertainty_penalty * stds
    elif method == "weighted":
        # Weight by inverse variance (more reliable = higher weight)
        weights = 1.0 / (stds + 1e-8)  # Add small epsilon to avoid division by zero
        # Normalize weights
        weights = weights / weights.sum()
        # Weighted average (but we still need to pick a discrete point)
        # So we'll use weighted mean as score
        scores = means * weights
    else:
        raise ValueError(f"Unknown method: {method}")

    best_idx = np.argmax(scores)
    best_size = int(sizes[best_idx])
    best_score_mean = float(means[best_idx])
    best_score_std = float(stds[best_idx])

    return best_size, best_score_mean, best_score_std


def cosine_similarity_std(tensor):
    """
    Calculate the standard deviation of pairwise cosine similarities
    for a tensor of shape (n_b, dim_emb).

    Args:
        tensor: torch.Tensor of shape (n_b, dim_emb)

    Returns:
        torch.Tensor: scalar tensor containing the standard deviation
    """

    # Normalize the embeddings to unit vectors
    normalized = F.normalize(tensor, p=2, dim=1)

    # Compute pairwise cosine similarities
    cos_sim_matrix = torch.mm(normalized, normalized.t())

    # Get upper triangular part (excluding diagonal) to avoid duplicates and self-similarity
    triu_indices = torch.triu_indices(
        cos_sim_matrix.size(0), cos_sim_matrix.size(1), offset=1
    )
    cos_similarities = cos_sim_matrix[triu_indices[0], triu_indices[1]]

    # Calculate standard deviation
    std_dev = torch.std(cos_similarities)

    return std_dev


def adjust_cluster_count(
    df_umap: pd.DataFrame,
    umap_columns: list[str],
    current_n_clusters: int,
    target_n_clusters: int,
    base_min_cluster_size: int,
) -> tuple[pd.DataFrame, int, hdbscan.HDBSCAN]:
    """
    Adjust HDBSCAN clustering to get closer to target number of clusters.

    Args:
        df_umap: DataFrame with UMAP-reduced embeddings
        umap_columns: List of column names for UMAP dimensions
        current_n_clusters: Current number of clusters
        target_n_clusters: Desired number of clusters
        base_min_cluster_size: Base min_cluster_size to adjust from

    Returns:
        tuple: (clustered_dataframe, final_n_clusters, clusterer)
    """
    # Initial clusterer
    initial_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=base_min_cluster_size, gen_min_span_tree=True
    )
    initial_labels = initial_clusterer.fit_predict(df_umap[umap_columns])

    if current_n_clusters == target_n_clusters:
        initial_df = df_umap.copy()
        initial_df["class"] = initial_labels
        return initial_df, current_n_clusters, initial_clusterer

    best_n_clusters = current_n_clusters
    best_df = df_umap.copy()
    best_df["class"] = initial_labels  # Ensure best_df always has "class" column
    best_clusterer = initial_clusterer

    # Try increasing min_cluster_size to reduce number of clusters
    if current_n_clusters > target_n_clusters:
        for size_mult in [1.2, 1.5, 2.0, 2.5]:
            test_size = int(base_min_cluster_size * size_mult)
            if test_size >= len(df_umap):
                continue
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=test_size, gen_min_span_tree=True
            )
            labels = clusterer.fit_predict(df_umap[umap_columns])
            test_n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if abs(test_n_clusters - target_n_clusters) < abs(
                best_n_clusters - target_n_clusters
            ):
                best_n_clusters = test_n_clusters
                df_test = df_umap.copy()
                df_test["class"] = labels
                best_df = df_test
                best_clusterer = clusterer
                if best_n_clusters == target_n_clusters:
                    break

    # Try decreasing min_cluster_size to increase number of clusters
    elif current_n_clusters < target_n_clusters:
        for size_mult in [0.8, 0.6, 0.5, 0.4]:
            test_size = max(2, int(base_min_cluster_size * size_mult))
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=test_size, gen_min_span_tree=True
            )
            labels = clusterer.fit_predict(df_umap[umap_columns])
            test_n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if abs(test_n_clusters - target_n_clusters) < abs(
                best_n_clusters - target_n_clusters
            ):
                best_n_clusters = test_n_clusters
                df_test = df_umap.copy()
                df_test["class"] = labels
                best_df = df_test
                best_clusterer = clusterer
                if best_n_clusters == target_n_clusters:
                    break

    return best_df, best_n_clusters, best_clusterer


def cluster_with_target_count(
    df_umap: pd.DataFrame,
    umap_columns: list[str],
    target_n_clusters: int,
    base_min_cluster_size: int | None = None,
) -> tuple[pd.DataFrame, int, float]:
    """
    Cluster data to get approximately target number of clusters and compute DBCV score.

    Args:
        df_umap: DataFrame with UMAP-reduced embeddings
        umap_columns: List of column names for UMAP dimensions
        target_n_clusters: Desired number of clusters
        base_min_cluster_size: Starting min_cluster_size (if None, estimates from data size)

    Returns:
        tuple: (clustered_dataframe, actual_n_clusters, dbcv_score)
    """
    if base_min_cluster_size is None:
        # Estimate starting point: aim for clusters of roughly equal size
        base_min_cluster_size = max(2, len(df_umap) // (target_n_clusters * 3))

    # Start with estimated size
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=base_min_cluster_size, gen_min_span_tree=True
    )
    labels = clusterer.fit_predict(df_umap[umap_columns])
    current_n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Adjust to get closer to target
    df_clustered, final_n_clusters, final_clusterer = adjust_cluster_count(
        df_umap,
        umap_columns,
        current_n_clusters,
        target_n_clusters,
        base_min_cluster_size,
    )

    # Compute DBCV score from the final clusterer
    if hasattr(final_clusterer, "relative_validity_"):
        dbcv = float(final_clusterer.relative_validity_)
    else:
        dbcv = 0.0  # Invalid clustering

    return df_clustered, final_n_clusters, dbcv


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


def find_cluster_centers(
    df_clustered: pd.DataFrame,
    id_column: str = "id",
    label_column: str = "label",
    min_cluster_size: int = 5,
    max_complexity_chars: int | None = None,
    max_complexity_words: int | None = None,
    max_word_length: int | None = None,
    word_frequencies: dict[str, float] | None = None,
) -> list[dict]:
    """
    Find a representative item for each cluster, filtering by size and complexity.

    Args:
        df_clustered: DataFrame with cluster assignments in 'class' column
        id_column: Name of column containing item IDs
        label_column: Name of column containing item labels
        min_cluster_size: Minimum number of members required (default: 5)
        max_complexity_chars: Maximum character count for candidates (None = no limit)
        max_complexity_words: Maximum word count for candidates (None = no limit)
        max_word_length: Maximum length for any word in the label (None = no limit)
        word_frequencies: Optional dictionary mapping words to frequencies for simplicity calculation.

    Returns:
        List of dictionaries with cluster_id, cluster_size, center_id, center_label,
        sorted by cluster_size (descending)
    """
    cluster_results = []

    # Filter out noise points and group by cluster
    df_valid = df_clustered[df_clustered["class"] != -1].copy()

    for cluster_id, cluster_data in df_valid.groupby("class"):
        cluster_size = len(cluster_data)

        # Skip small clusters
        if cluster_size < min_cluster_size:
            continue

        # Compute simplicity scores and filter candidates in one pass
        valid_candidates = []

        for idx, row in cluster_data.iterrows():
            label = str(row[label_column])

            # Apply quick filters first (before expensive simplicity calculation)
            if max_word_length is not None:
                if any(len(word) > max_word_length for word in label.split()):
                    continue

            # Compute simplicity metrics
            complexity = _measure_label_simplicity(
                label, word_frequencies=word_frequencies
            )

            # Apply complexity filters
            if (
                max_complexity_chars is not None
                and complexity["char_count"] > max_complexity_chars
            ):
                continue
            if (
                max_complexity_words is not None
                and complexity["word_count"] > max_complexity_words
            ):
                continue

            # This candidate passes all filters
            valid_candidates.append((idx, complexity["simplicity_score"]))

        # Skip cluster if no valid candidates
        if not valid_candidates:
            continue

        # Select the candidate with the highest simplicity score
        best_idx, _ = max(valid_candidates, key=lambda x: x[1])
        selected_row = cluster_data.loc[best_idx]

        cluster_results.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": cluster_size,
                "center_id": selected_row[id_column],
                "center_label": selected_row[label_column],
            }
        )

    # Sort by cluster size (largest first)
    cluster_results.sort(key=lambda x: x["cluster_size"], reverse=True)
    return cluster_results


def compute_dbcv_after_filtering(
    df_clustered: pd.DataFrame,
    valid_cluster_ids: set[int],
) -> float:
    """
    Compute DBCV score after filtering out perplex clusters.

    Args:
        df_clustered: DataFrame with cluster assignments
        valid_cluster_ids: Set of cluster IDs to keep

    Returns:
        DBCV score for filtered clusters
    """

    # Filter to only valid clusters (exclude noise and filtered-out clusters)
    df_filtered = df_clustered[df_clustered["class"].isin(valid_cluster_ids)].copy()

    if len(df_filtered) < 2:
        return 0.0

    # Detect UMAP columns (columns starting with "u_")
    umap_columns = [col for col in df_filtered.columns if col.startswith("u_")]
    if not umap_columns:
        return 0.0

    # Re-fit HDBSCAN on filtered data to get DBCV
    # We need to estimate min_cluster_size - use a reasonable default
    min_size = max(
        2, len(df_filtered) // 20
    )  # At least 2, but aim for ~20 points per cluster
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, gen_min_span_tree=True)
    clusterer.fit(df_filtered[umap_columns])

    if hasattr(clusterer, "relative_validity_"):
        return float(clusterer.relative_validity_)
    else:
        return 0.0


def compute_hungarian_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute clustering accuracy using Hungarian algorithm to optimally match
    predicted cluster labels to true labels.

    Args:
        y_true: True labels (e.g., property names)
        y_pred: Predicted cluster labels

    Returns:
        Accuracy score between 0 and 1
    """
    # Filter out noise points (label -1) for accuracy computation
    valid_mask = y_pred != -1
    if not valid_mask.any():
        return 0.0

    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    if len(y_true_valid) == 0:
        return 0.0

    # Compute confusion matrix
    cm = confusion_matrix(y_true_valid, y_pred_valid)

    # Use Hungarian algorithm to find optimal assignment
    # We negate the matrix because linear_sum_assignment minimizes cost
    row_ind, col_ind = linear_sum_assignment(-cm)

    # Compute accuracy based on optimal assignment
    accuracy = cm[row_ind, col_ind].sum() / cm.sum()

    return float(accuracy)


def estimate_model_clustering(
    file_path: pathlib.Path,
    rns: RandomState,
    transform_config: TransformConfig,
    *,
    min_class_size: int = 20,
    max_scale: int = 120,
    tol: float = 0.05,
    frac: float = 0.1,
    head: int | None = None,
    batch_size: int = 1000,
    selected_labels: set[str] | None = None,
    all_metrics_dfs: list[pd.DataFrame] | None = None,
    optimization_method: str = "mean",
):
    """
    Estimate optimal cluster size for a single model/layer file.

    Args:
        file_path: Path to parquet file
        rns: RandomState object for reproducible sampling
        transform_config: TransformConfig instance specifying transformation parameters
        min_class_size: Minimum class size for filtering
        max_scale: Maximum value for grid evaluation of min_cluster_size (default: 120)
        tol: Tolerance for optimization (deprecated, kept for compatibility)
        frac: Fraction of dataset to sample
        head: Number of batches to take (None for all)
        batch_size: Batch size for reading
        selected_labels: Optional set of labels from selected labels KB to filter by
        all_metrics_dfs: Optional list of metrics DataFrames from previous samples.
                         If provided, will aggregate and find optimum from grid.
                         If None, evaluates grid for this sample only.
        optimization_method: Method for finding optimum ("mean", "lower_bound", "weighted")

    Returns:
        ClusteringReport or None: Report containing clustering results, or None if processing failed
    """
    from pelinker.io import read_batches
    from pelinker.reporting import ClusteringReport

    # Simple check: if file doesn't exist (handles broken symlinks), skip it
    if not file_path.exists():
        return None
    agg = []

    try:
        for i, batch in enumerate(
            read_batches(file_path.as_posix(), batch_size=batch_size)
        ):
            sample = batch.sample(frac=frac, random_state=rns)
            agg += [sample]
            if head is not None and i >= head - 1:
                break
    except Exception:
        return None

    if not agg:
        return None

    dfr = pd.concat(agg)

    umap_columns = get_umap_columns(transform_config)

    # Filter by selected labels if provided
    if selected_labels is not None:
        dfr = dfr.loc[dfr["property"].isin(selected_labels)].copy()
        if len(dfr) == 0:
            return None

    # trim rare mentions
    mention_count = dfr["property"].value_counts()
    low_count_properties = mention_count[
        ~(mention_count >= min_class_size)
    ].index.to_list()

    dfr = dfr.loc[~dfr["property"].isin(low_count_properties)].copy()

    if len(dfr) == 0:
        return None

    # Get number of unique properties before transformation
    number_properties = dfr["property"].nunique()

    dfr2 = transform_embeddings(dfr, config=transform_config, embed_column="embed")

    # Grid evaluation
    sizes = list(np.arange(int(0.5 * min_class_size), max_scale, 5))

    metrics_df = evaluate_cluster_size_grid(dfr2, umap_columns, sizes)

    # Find optimal cluster size
    if all_metrics_dfs is not None:
        # Aggregate with previous samples and find optimum from grid
        all_metrics_dfs.append(metrics_df)
        aggregated = aggregate_grid_metrics(all_metrics_dfs)
        best_size, best_score, best_score_std = find_optimal_from_grid(
            aggregated, method=optimization_method
        )
    else:
        # Single sample: just use max DBCV from this sample
        if len(metrics_df) == 0:
            return None
        best_idx = metrics_df["dbcv"].idxmax()
        best_size = int(metrics_df.loc[best_idx, "min_cluster_size"])
        best_score = float(metrics_df.loc[best_idx, "dbcv"])

    # Apply final clustering with best size
    clusterer = hdbscan.HDBSCAN(min_cluster_size=best_size, gen_min_span_tree=True)
    labels = clusterer.fit_predict(dfr2[umap_columns])
    dfr2_final = dfr2.copy()
    dfr2_final["class"] = pd.DataFrame(
        labels, columns=["class"], index=dfr2_final.index
    )

    # Compute Hungarian matching accuracy
    # Use property column as true labels and class column as predicted clusters
    hungarian_acc = None
    if "property" in dfr2_final.columns and "class" in dfr2_final.columns:
        # Convert property labels to numeric for confusion matrix
        property_labels = dfr2_final["property"].astype("category").cat.codes.values
        cluster_labels = dfr2_final["class"].values
        hungarian_acc = compute_hungarian_accuracy(property_labels, cluster_labels)

    return ClusteringReport(
        best_size=best_size,
        best_score=best_score,
        number_properties=number_properties,
        metrics_df=metrics_df,
        df=dfr2_final,
        hungarian_accuracy=hungarian_acc,
    )


def embeddings_dict_to_dataframe(
    embeddings_dict: dict[str, tuple[str, torch.Tensor | np.ndarray]],
) -> pd.DataFrame:
    """
    Convert embeddings dictionary to DataFrame format expected by transform_embeddings.

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


def farthest_point_sampling(
    embeddings: np.ndarray,
    n_samples: int,
    metric: str = "cosine",
    random_state: int | None = None,
) -> np.ndarray:
    """
    Select n_samples points using farthest point sampling (FPS) for maximum diversity.

    FPS is a deterministic greedy algorithm that selects points that are maximally
    distant from already selected points. This ensures semantic diversity.

    Args:
        embeddings: Array of shape (n_points, n_features) containing embeddings
        n_samples: Number of points to select
        metric: Distance metric ('cosine' or 'euclidean')
        random_state: Random seed for selecting the first point (None = use first point)

    Returns:
        Array of indices of selected points, shape (n_samples,)
    """
    from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

    n_points = embeddings.shape[0]
    if n_samples >= n_points:
        return np.arange(n_points)

    if n_samples <= 0:
        return np.array([], dtype=int)

    # Normalize embeddings for cosine distance
    if metric == "cosine":
        embeddings_norm = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        )
        dist_fn = cosine_distances
    else:
        embeddings_norm = embeddings
        dist_fn = euclidean_distances

    # Initialize: select first point (or random if random_state provided)
    selected_indices = []
    if random_state is not None:
        rng = np.random.RandomState(random_state)
        selected_indices.append(rng.randint(0, n_points))
    else:
        selected_indices.append(0)

    # Compute distances from all points to selected points
    selected_embeddings = embeddings_norm[selected_indices]
    distances = dist_fn(embeddings_norm, selected_embeddings)
    min_distances = distances.min(axis=1)  # Minimum distance to any selected point

    # Iteratively select the point farthest from all selected points
    for _ in range(n_samples - 1):
        # Find point with maximum minimum distance
        next_idx = np.argmax(min_distances)
        selected_indices.append(next_idx)

        # Update minimum distances
        new_distances = dist_fn(embeddings_norm, embeddings_norm[[next_idx]])
        min_distances = np.minimum(min_distances, new_distances.ravel())

    return np.array(selected_indices)


def farthest_point_sampling_weighted(
    embeddings: np.ndarray,
    weights: np.ndarray,
    n_samples: int,
    metric: str = "cosine",
    diversity_weight: float = 0.7,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Select n_samples points using weighted farthest point sampling.

    Combines semantic diversity (FPS) with preference weights (e.g., simplicity).
    Higher weights are preferred when distances are similar.

    Args:
        embeddings: Array of shape (n_points, n_features) containing embeddings
        weights: Array of shape (n_points,) with preference weights (higher = better)
        n_samples: Number of points to select
        metric: Distance metric ('cosine' or 'euclidean')
        diversity_weight: Weight for diversity vs preference (0.0 = pure preference, 1.0 = pure FPS)
        random_state: Random seed for selecting the first point

    Returns:
        Array of indices of selected points, shape (n_samples,)
    """
    from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

    n_points = embeddings.shape[0]
    if n_samples >= n_points:
        return np.arange(n_points)

    if n_samples <= 0:
        return np.array([], dtype=int)

    # Normalize weights to [0, 1] range
    if weights.max() > weights.min():
        weights_norm = (weights - weights.min()) / (weights.max() - weights.min())
    else:
        weights_norm = np.ones_like(weights)

    # Normalize embeddings for cosine distance
    if metric == "cosine":
        embeddings_norm = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        )
        dist_fn = cosine_distances
    else:
        embeddings_norm = embeddings
        dist_fn = euclidean_distances

    # Initialize: select first point (or random if random_state provided)
    selected_indices = []
    if random_state is not None:
        rng = np.random.RandomState(random_state)
        selected_indices.append(rng.randint(0, n_points))
    else:
        # Start with highest weighted point
        selected_indices.append(np.argmax(weights_norm))

    # Compute distances from all points to selected points
    selected_embeddings = embeddings_norm[selected_indices]
    distances = dist_fn(embeddings_norm, selected_embeddings)
    min_distances = distances.min(axis=1)  # Minimum distance to any selected point

    # Normalize distances to [0, 1] for combination with weights
    if min_distances.max() > min_distances.min():
        dist_norm = (min_distances - min_distances.min()) / (
            min_distances.max() - min_distances.min()
        )
    else:
        dist_norm = np.ones_like(min_distances)

    # Iteratively select points using weighted combination
    for _ in range(n_samples - 1):
        # Combine diversity (distance) with preference (weight)
        # Higher score = better candidate
        combined_scores = (
            diversity_weight * dist_norm + (1 - diversity_weight) * weights_norm
        )

        # Don't select already selected points
        combined_scores[selected_indices] = -np.inf

        # Find point with maximum combined score
        next_idx = np.argmax(combined_scores)
        selected_indices.append(next_idx)

        # Update minimum distances
        new_distances = dist_fn(embeddings_norm, embeddings_norm[[next_idx]])
        min_distances = np.minimum(min_distances, new_distances.ravel())

        # Re-normalize distances for next iteration
        if min_distances.max() > min_distances.min():
            dist_norm = (min_distances - min_distances.min()) / (
                min_distances.max() - min_distances.min()
            )
        else:
            dist_norm = np.ones_like(min_distances)

    return np.array(selected_indices)


def is_specific_term(
    label: str,
    word_frequencies: Mapping[str, float] | None = None,
    max_word_length: int = 18,
    min_word_freq: float = 1e-6,
) -> bool:
    """
    Check if a label represents a very specific/technical term.

    Args:
        label: Label text to check
        word_frequencies: Optional word frequency mapping
        max_word_length: Maximum length for a word before considered specific
        min_word_freq: Minimum word frequency to be considered generic

    Returns:
        True if the label appears to be very specific/technical
    """
    words = label.strip().lower().split()

    # Check for very long words (likely technical compounds)
    if any(len(word) > max_word_length for word in words):
        return True

    # Check word frequencies if available
    if word_frequencies is not None:
        content_words = [w for w in words if len(w) > 2]  # Ignore very short words
        if content_words:
            # If most content words are rare, it's likely specific
            rare_count = sum(
                1 for w in content_words if word_frequencies.get(w, 0) < min_word_freq
            )
            if rare_count > len(content_words) * 0.5:  # More than 50% rare words
                return True

    # Check for common technical patterns
    technical_patterns = [
        "transferase",
        "phosphatase",
        "kinase",
        "synthase",
        "reductase",
        "oxidase",
        "hydrolase",
        "activity",
        "specific",
    ]
    label_lower = label.lower()
    if any(pattern in label_lower for pattern in technical_patterns):
        # But allow common terms like "has activity" or "specific to"
        if not any(
            phrase in label_lower
            for phrase in ["has activity", "specific to", "specific for"]
        ):
            return True

    return False


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


def is_specific_term_kb(
    label: str,
    generality_score: float,
    word_frequencies: Mapping[str, float] | None = None,
    min_generality: float = 0.3,
    max_word_length: int = 18,
    min_word_freq: float = 1e-6,
    strict_rare_word_check: bool = True,
) -> bool:
    """
    Check if a label represents a very specific/technical term using KB-based criteria.

    Combines KB generality score with word-level patterns and domain-specific vocabulary.

    Args:
        label: Label text to check
        generality_score: KB-based generality score (0-1, higher = more generic)
        word_frequencies: Optional word frequency mapping
        min_generality: Minimum generality score to be considered generic
        max_word_length: Maximum length for a word before considered specific
        min_word_freq: Minimum word frequency to be considered generic
        strict_rare_word_check: If True, filter out if ANY content word is very rare

    Returns:
        True if the label appears to be very specific/technical
    """
    label_lower = label.strip().lower()
    words = label_lower.split()

    # Low generality score = specific term
    if generality_score < min_generality:
        return True

    # Check for very long words (likely technical compounds)
    if any(len(word) > max_word_length for word in words):
        return True

    # Domain-specific vocabulary patterns (anatomy, neuroscience, etc.)
    domain_specific_patterns = [
        # Anatomy/neuroscience
        "soma",
        "dendrite",
        "axon",
        "synapse",
        "neuron",
        "muscle antagonist",
        "muscle insertion",
        "muscle origin",
        "fasciculate",
        "innervate",
        "synapsed",
        "synapsed to",
        "synapsed by",
        "synapsed in",
        # Domain-specific processes
        "myristoyl",
        "ubiquitin",
        "phosphoryl",
        "methylat",
        # Very specific biological terms
        "endoparasite",
        "ectoparasite",
        "mesoparasite",
        "hyperparasite",
        "kleptoparasite",
        "epiphyte",
        "roost",
        "co-roost",
        # Ecological terms
        "commensual",
        "mutualistic",
        "symbiotic",
        "trophic",
        # Very specific location terms
        "soma location",
        "dendrite location",
        "2d boundary",
        # Specific anatomical structures
        "trachea",
        "tracheate",
        "bounding layer",
        # Very specific temporal/spatial relations
        "during which",
        "existence starts",
        "existence ends",
        "existence overlaps",
    ]

    # Check for domain-specific patterns
    # These are always too specific, regardless of generality score
    for pattern in domain_specific_patterns:
        if pattern in label_lower:
            # Domain-specific terms are always filtered out
            # They may have high embedding similarity but are still too niche
            return True

    # Check word frequencies if available
    if word_frequencies is not None:
        content_words = [w for w in words if len(w) > 2]  # Ignore very short words
        if content_words:
            # Strict check: if ANY key content word is very rare, filter it out
            if strict_rare_word_check:
                # Check for very rare words (much stricter threshold)
                very_rare_threshold = min_word_freq * 0.1  # 10x stricter
                for word in content_words:
                    freq = word_frequencies.get(word, 0)
                    if (
                        freq < very_rare_threshold and len(word) > 4
                    ):  # Only check longer words
                        # This is a very rare word, likely domain-specific
                        return True

            # Also check if most content words are rare
            rare_count = sum(
                1 for w in content_words if word_frequencies.get(w, 0) < min_word_freq
            )
            if rare_count > len(content_words) * 0.5:  # More than 50% rare words
                return True

    # Check for common technical patterns (but allow if generality is high)
    technical_patterns = [
        "transferase",
        "phosphatase",
        "kinase",
        "synthase",
        "reductase",
        "oxidase",
        "hydrolase",
    ]
    if any(pattern in label_lower for pattern in technical_patterns):
        # Only consider specific if generality is also low
        if generality_score < min_generality * 1.5:  # Slightly more lenient threshold
            return True

    return False
