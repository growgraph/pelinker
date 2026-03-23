import pathlib

import numpy as np
import pandas as pd
import torch
import hdbscan
from pelinker.io import read_batches
from pelinker.reporting import ClusteringReport
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
    max_pairs_per_cluster: int = 200_000,
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
    # OPTIMIZATION: materialize UMAP features once as float32 to reduce repeated
    # DataFrame slicing and keep HDBSCAN input memory footprint lower.
    umap_values = dfr2[umap_columns].to_numpy(dtype=np.float32, copy=False)

    metrics = []
    for size in sizes:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=size, gen_min_span_tree=True)
        labels = clusterer.fit_predict(umap_values)

        ic = []
        for ix in np.unique(labels):
            if ix == -1:  # Skip noise points
                continue
            cluster_values = umap_values[labels == ix]
            if len(cluster_values) < 2:
                continue
            tgroup = torch.from_numpy(cluster_values)
            st = cosine_similarity_std(
                tgroup, max_pairs=max_pairs_per_cluster, random_seed=13
            )
            ic += [float(st)]

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


def cosine_similarity_std(
    tensor: torch.Tensor, max_pairs: int = 200_000, random_seed: int = 13
):
    """
    Calculate the standard deviation of pairwise cosine similarities
    for a tensor of shape (n_b, dim_emb).

    Args:
        tensor: torch.Tensor of shape (n_b, dim_emb)

    Returns:
        torch.Tensor: scalar tensor containing the standard deviation
    """

    # OPTIMIZATION: use float32 to reduce memory pressure from intermediate tensors.
    normalized = F.normalize(tensor.float(), p=2, dim=1)

    n_points = normalized.size(0)
    if n_points < 2:
        return torch.tensor(float("nan"), dtype=normalized.dtype)

    total_pairs = n_points * (n_points - 1) // 2

    # For small clusters, exact computation is cheap and keeps original behavior.
    if total_pairs <= max_pairs:
        cos_sim_matrix = torch.mm(normalized, normalized.t())
        triu_indices = torch.triu_indices(
            cos_sim_matrix.size(0), cos_sim_matrix.size(1), offset=1
        )
        cos_similarities = cos_sim_matrix[triu_indices[0], triu_indices[1]]
        return torch.std(cos_similarities)

    # OPTIMIZATION: avoid O(n^2) similarity matrix for large clusters by sampling
    # random pairs and estimating the std from sampled cosine similarities.
    sample_size = min(max_pairs, total_pairs)
    generator = torch.Generator(device=normalized.device)
    generator.manual_seed(random_seed)

    idx_i = torch.randint(
        0, n_points, (sample_size,), generator=generator, device=normalized.device
    )
    idx_j = torch.randint(
        0, n_points - 1, (sample_size,), generator=generator, device=normalized.device
    )
    idx_j = idx_j + (idx_j >= idx_i).long()  # ensure i != j
    cos_similarities = (normalized[idx_i] * normalized[idx_j]).sum(dim=1)
    return torch.std(cos_similarities)


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
