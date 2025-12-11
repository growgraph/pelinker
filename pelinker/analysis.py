import numpy as np
import pandas as pd
import torch
import umap
from matplotlib import pyplot as plt
from plotly import express as px, graph_objects as go
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score
from torch.nn import functional as F
from collections.abc import Mapping
from typing import Iterable

from skopt import gp_minimize


def compute_optimal_min_cluster_size(dfr2, umap_columns, tol=0.05, bounds=None):
    if bounds is None:
        bounds = [(5, 200)]  # min_cluster_size range

    # Cache to avoid recomputing the same cluster sizes
    cache = {}

    def objective(size):
        size = int(size[0])

        # Check cache first
        if size in cache:
            score = cache[size]
            return -score

        clusterer = HDBSCAN(min_cluster_size=size, metric="cosine")
        labels = clusterer.fit_predict(dfr2[umap_columns])

        if len(set(labels)) <= 2:
            score = 1e-2  # invalid clustering → low score
        else:
            score = silhouette_score(dfr2[umap_columns], labels)

        cache[size] = score
        return -score

    result = gp_minimize(objective, bounds, n_calls=15, random_state=42)
    best_size = int(result.x[0])
    best_score = -result.fun  # Convert back from negative

    clusterer = HDBSCAN(min_cluster_size=best_size, metric="cosine")
    labels = clusterer.fit_predict(dfr2[umap_columns])
    dfr2["class"] = pd.DataFrame(labels, columns=["class"], index=dfr2.index)
    return best_size, best_score, dfr2


def plot_metrics(df: pd.DataFrame, fname):
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = "tab:blue"
    ax1.set_xlabel("min_cluster_size")
    ax1.set_ylabel("Silhouette score", color=color1)
    ax1.plot(
        df["min_cluster_size"],
        df["silhouette"],
        marker="o",
        color=color1,
        label="Silhouette",
    )
    ax1.tick_params(axis="y", labelcolor=color1)

    # Add a second y-axis for icm
    ax2 = ax1.twinx()
    color2 = "tab:orange"
    # ax2.set_yscale("log")
    ax2.set_ylabel("ICM", color=color2)
    ax2.plot(
        df["min_cluster_size"],
        df["icm"],
        marker="s",
        linestyle="--",
        color=color2,
        label="ICM",
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    # Add a second y-axis for icm
    ax3 = ax1.twinx()
    color2 = "tab:green"
    # ax3.set_yscale("log")
    ax3.set_ylabel("n_clusters", color=color2)
    ax3.plot(
        df["min_cluster_size"],
        df["n_clusters"],
        marker="s",
        linestyle="--",
        color=color2,
        label="ICM",
    )

    ax3.tick_params(axis="y", labelcolor=color2)

    # Titles and layout
    plt.title("Clustering metrics vs. min_cluster_size (HDBSCAN)")
    fig.tight_layout()

    plt.savefig(fname, bbox_inches="tight", dpi=300)


def plot_umap_viz(df, output_path="umap.html"):
    df["show_label"] = df["property"]
    show_rate = max(len(df) // 20, 1)
    df.loc[df.index % show_rate != 0, "show_label"] = ""

    # Ensure class is treated as categorical
    df["class"] = df["class"].astype(str)

    # Base scatter plot
    fig = px.scatter_3d(
        df,
        x="uviz_00",
        y="uviz_01",
        z="uviz_02",
        color="class",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        hover_name="property",
        labels={"uviz_00": "Dim 1", "uviz_01": "Dim 2", "uviz_02": "Dim 3"},
    )

    # Add text labels as a separate trace
    df_labels = df[df["show_label"] != ""]
    text_trace = go.Scatter3d(
        x=df_labels["uviz_00"],
        y=df_labels["uviz_01"],
        z=df_labels["uviz_02"],
        mode="text",
        text=df_labels["show_label"],
        textposition="top center",
        showlegend=False,
        hoverinfo="skip",
        textfont=dict(size=10, color="black"),
    )
    fig.add_trace(text_trace)

    # Update layout
    fig.update_layout(
        title="3D Scatter Plot of Embeddings",
        scene=dict(
            xaxis_title="uviz_00",
            yaxis_title="uviz_01",
            zaxis_title="uviz_02",
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=30),
    )

    fig.write_html(output_path)


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
    from sklearn.metrics import silhouette_score

    # Filter to only valid clusters (exclude noise and filtered-out clusters)
    df_filtered = df_clustered[df_clustered["class"].isin(valid_cluster_ids)].copy()

    if len(df_filtered) == 0:
        return 0.0

    labels = df_filtered["class"].values
    unique_labels = set(labels)
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        return 0.0

    return silhouette_score(df_filtered[umap_columns], labels)


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
