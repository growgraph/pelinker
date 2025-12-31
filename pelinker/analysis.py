import pathlib

import numpy as np
import pandas as pd
import torch
import umap
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F
from collections.abc import Mapping
from typing import Iterable
from numpy.random import RandomState


def evaluate_cluster_size_grid(
    dfr2: pd.DataFrame,
    umap_columns: list[str],
    sizes: list[int],
) -> pd.DataFrame:
    """
    Evaluate clustering metrics on a grid of min_cluster_size values.

    Args:
        dfr2: DataFrame with UMAP-reduced embeddings
        umap_columns: List of UMAP column names
        sizes: List of min_cluster_size values to evaluate

    Returns:
        DataFrame with columns: min_cluster_size, icm, n_clusters, silhouette
    """
    metrics = []
    for size in sizes:
        clusterer = HDBSCAN(min_cluster_size=size, metric="cosine")
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

        # Only compute silhouette score if we have valid clusters (at least 2 clusters, excluding noise)
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        if n_clusters > 2:
            score = silhouette_score(dfr2[umap_columns], labels)
            metrics += [(size, icm, n_clusters, score)]

    return pd.DataFrame(
        metrics, columns=["min_cluster_size", "icm", "n_clusters", "silhouette"]
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
        - silhouette_mean, silhouette_std, silhouette_count
        - icm_mean, n_clusters_mean
    """
    if not all_metrics_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_metrics_dfs, ignore_index=True)

    aggregated = (
        combined.groupby("min_cluster_size")
        .agg(
            {
                "silhouette": ["mean", "std", "count"],
                "icm": "mean",
                "n_clusters": "mean",
            }
        )
        .reset_index()
    )

    # Flatten column names
    aggregated.columns = [
        "min_cluster_size",
        "silhouette_mean",
        "silhouette_std",
        "silhouette_count",
        "icm_mean",
        "n_clusters_mean",
    ]

    # Fill NaN std with 0 (single sample case)
    aggregated["silhouette_std"] = aggregated["silhouette_std"].fillna(0.0)

    return aggregated


def find_optimal_from_grid(
    aggregated_metrics: pd.DataFrame,
    method: str = "mean",
    uncertainty_penalty: float = 1.0,
) -> tuple[int, float, float]:
    """
    Find optimal min_cluster_size from aggregated grid metrics.

    Args:
        aggregated_metrics: DataFrame from aggregate_grid_metrics()
        method: How to select optimum:
            - "mean": Use mean silhouette score (default)
            - "lower_bound": Use mean - uncertainty_penalty * std (conservative)
            - "weighted": Weight by inverse variance (more samples = more weight)
        uncertainty_penalty: Multiplier for std when using "lower_bound" method

    Returns:
        (best_size, best_score_mean, best_score_std)
    """
    if len(aggregated_metrics) == 0:
        raise ValueError("No aggregated metrics provided")

    sizes = aggregated_metrics["min_cluster_size"].values
    means = aggregated_metrics["silhouette_mean"].values
    stds = aggregated_metrics["silhouette_std"].values

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


def umap_it(df, umap_dim=15):
    embedding_vectors = np.stack(df["embed"].values)
    reduced = umap.UMAP(n_components=umap_dim, metric="cosine").fit_transform(
        embedding_vectors
    )
    df_reduced = pd.DataFrame(
        reduced, index=df.index, columns=[f"u_{j:02d}" for j in range(umap_dim)]
    )
    reduced_viz = umap.UMAP(n_components=3, metric="cosine").fit_transform(reduced)
    df_reduced_viz = pd.DataFrame(
        reduced_viz, index=df.index, columns=[f"uviz_{j:02d}" for j in range(3)]
    )
    df = pd.concat([df, df_reduced, df_reduced_viz], axis=1)
    return df


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
) -> tuple[pd.DataFrame, int]:
    """
    Adjust HDBSCAN clustering to get closer to target number of clusters.

    Args:
        df_umap: DataFrame with UMAP-reduced embeddings
        umap_columns: List of column names for UMAP dimensions
        current_n_clusters: Current number of clusters
        target_n_clusters: Desired number of clusters
        base_min_cluster_size: Base min_cluster_size to adjust from

    Returns:
        tuple: (clustered_dataframe, final_n_clusters)
    """
    if current_n_clusters == target_n_clusters:
        return df_umap, current_n_clusters

    best_n_clusters = current_n_clusters
    best_df = df_umap.copy()

    # Try increasing min_cluster_size to reduce number of clusters
    if current_n_clusters > target_n_clusters:
        for size_mult in [1.2, 1.5, 2.0, 2.5]:
            test_size = int(base_min_cluster_size * size_mult)
            if test_size >= len(df_umap):
                continue
            clusterer = HDBSCAN(min_cluster_size=test_size, metric="cosine")
            labels = clusterer.fit_predict(df_umap[umap_columns])
            test_n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if abs(test_n_clusters - target_n_clusters) < abs(
                best_n_clusters - target_n_clusters
            ):
                best_n_clusters = test_n_clusters
                df_test = df_umap.copy()
                df_test["class"] = labels
                best_df = df_test
                if best_n_clusters == target_n_clusters:
                    break

    # Try decreasing min_cluster_size to increase number of clusters
    elif current_n_clusters < target_n_clusters:
        for size_mult in [0.8, 0.6, 0.5, 0.4]:
            test_size = max(2, int(base_min_cluster_size * size_mult))
            clusterer = HDBSCAN(min_cluster_size=test_size, metric="cosine")
            labels = clusterer.fit_predict(df_umap[umap_columns])
            test_n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if abs(test_n_clusters - target_n_clusters) < abs(
                best_n_clusters - target_n_clusters
            ):
                best_n_clusters = test_n_clusters
                df_test = df_umap.copy()
                df_test["class"] = labels
                best_df = df_test
                if best_n_clusters == target_n_clusters:
                    break

    return best_df, best_n_clusters


def cluster_with_target_count(
    df_umap: pd.DataFrame,
    umap_columns: list[str],
    target_n_clusters: int,
    base_min_cluster_size: int | None = None,
) -> tuple[pd.DataFrame, int, float]:
    """
    Cluster data to get approximately target number of clusters and compute silhouette score.

    Args:
        df_umap: DataFrame with UMAP-reduced embeddings
        umap_columns: List of column names for UMAP dimensions
        target_n_clusters: Desired number of clusters
        base_min_cluster_size: Starting min_cluster_size (if None, estimates from data size)

    Returns:
        tuple: (clustered_dataframe, actual_n_clusters, silhouette_score)
    """
    if base_min_cluster_size is None:
        # Estimate starting point: aim for clusters of roughly equal size
        base_min_cluster_size = max(2, len(df_umap) // (target_n_clusters * 3))

    # Start with estimated size
    clusterer = HDBSCAN(min_cluster_size=base_min_cluster_size, metric="cosine")
    labels = clusterer.fit_predict(df_umap[umap_columns])
    current_n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Adjust to get closer to target
    df_clustered, final_n_clusters = adjust_cluster_count(
        df_umap,
        umap_columns,
        current_n_clusters,
        target_n_clusters,
        base_min_cluster_size,
    )

    # Compute silhouette score
    labels = df_clustered["class"].values
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    if n_clusters < 2:
        score = 0.0  # Invalid clustering
    else:
        score = silhouette_score(df_umap[umap_columns], labels)

    return df_clustered, final_n_clusters, score


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
    stopwords: Iterable[str] = ("is", "of", "the", "a", "an", "to", "for", "or", "in"),
    zero_freq_penalty: float = 1e-8,
    multiword_penalty: float = 0.15,
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
    min_simplicity_score: float | None = None,
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
        min_simplicity_score: Minimum simplicity score (based on word frequency harmonic mean)
                             Higher = simpler. None = no limit.
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
            if (
                min_simplicity_score is not None
                and complexity["simplicity_score"] < min_simplicity_score
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


def compute_silhouette_after_filtering(
    df_clustered: pd.DataFrame,
    umap_columns: list[str],
    valid_cluster_ids: set[int],
) -> float:
    """
    Compute silhouette score after filtering out perplex clusters.

    Args:
        df_clustered: DataFrame with cluster assignments
        umap_columns: List of UMAP column names
        valid_cluster_ids: Set of cluster IDs to keep

    Returns:
        Silhouette score for filtered clusters
    """

    # Filter to only valid clusters (exclude noise and filtered-out clusters)
    df_filtered = df_clustered[df_clustered["class"].isin(valid_cluster_ids)].copy()

    if len(df_filtered) < 2:
        return 0.0

    labels = df_filtered["class"].values
    return silhouette_score(df_filtered[umap_columns], labels)


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
    umap_dim: int = 15,
    min_class_size: int = 20,
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
        umap_dim: UMAP dimensionality
        min_class_size: Minimum class size for filtering
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

    umap_columns = [f"u_{j:02d}" for j in range(umap_dim)]

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

    # Get number of unique properties before UMAP
    number_properties = dfr["property"].nunique()

    dfr2 = umap_it(dfr, umap_dim=umap_dim)

    # Grid evaluation
    sizes = list(np.arange(int(0.5 * min_class_size), int(4 * min_class_size), 5))

    if int(2 * number_properties) > sizes[-1]:
        sizes += [int(2 * number_properties)]

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
        # Single sample: just use max from this sample
        if len(metrics_df) == 0:
            return None
        best_idx = metrics_df["silhouette"].idxmax()
        best_size = int(metrics_df.loc[best_idx, "min_cluster_size"])
        best_score = float(metrics_df.loc[best_idx, "silhouette"])

    # Apply final clustering with best size
    clusterer = HDBSCAN(min_cluster_size=best_size, metric="cosine")
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
    Convert embeddings dictionary to DataFrame format expected by umap_it.

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
