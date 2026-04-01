from __future__ import annotations

import pathlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd
import torch
import hdbscan
from pelinker.clustering_grid import (
    aggregate_grid_metrics,
    evaluate_cluster_size_grid,
    solve_optimal_min_cluster_size_from_aggregated,
)
from pelinker.embedding_fusion import mention_level_concat_frames
from pelinker.io import read_batches
from pelinker.reporting import ClusteringHyperparameters, ClusteringReport
from sklearn.metrics import adjusted_rand_score
from numpy.random import RandomState

from pelinker.config import ClusteringOptimizationConfig, TransformConfig
from pelinker.transform import get_umap_columns, transform_embeddings


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


def _read_batches_concat(
    path: pathlib.Path,
    config: ClusteringOptimizationConfig,
) -> pd.DataFrame | None:
    if not path.exists():
        return None
    agg: list[pd.DataFrame] = []
    try:
        for i, batch in enumerate(
            read_batches(path.as_posix(), batch_size=config.batch_size)
        ):
            agg.append(batch)
            if config.head is not None and i >= config.head - 1:
                break
    except Exception:
        return None
    if not agg:
        return None
    return pd.concat(agg, ignore_index=True)


def estimate_clustering_from_frame(
    dfr: pd.DataFrame,
    transform_config: TransformConfig,
    optimization_config: ClusteringOptimizationConfig | None = None,
    *,
    selected_labels: set[str] | None = None,
    all_metrics_dfs: list[pd.DataFrame] | None = None,
    aggregation_level: Literal["mention", "property"] = "mention",
) -> ClusteringReport | None:
    """
    Run clustering grid search and optional aggregation on a pre-built frame with an ``embed`` column.

    ``aggregation_level="property"`` expects one row per distinct ``property`` (e.g. fused
    KB vectors) and skips min-mention-per-property trimming used for mention-level corpora.
    """

    config = optimization_config or ClusteringOptimizationConfig()

    if "embed" not in dfr.columns or "property" not in dfr.columns:
        return None

    if selected_labels is not None:
        dfr = dfr.loc[dfr["property"].isin(selected_labels)].copy()
        if len(dfr) == 0:
            return None

    if aggregation_level == "mention":
        mention_count = dfr["property"].value_counts()
        low_count_properties = mention_count[
            ~(mention_count >= config.min_class_size)
        ].index.to_list()
        dfr = dfr.loc[~dfr["property"].isin(low_count_properties)].copy()
        if len(dfr) == 0:
            return None

    number_properties = int(dfr["property"].nunique())

    umap_columns = get_umap_columns(transform_config)
    dfr2 = transform_embeddings(dfr, config=transform_config, embed_column="embed")

    sizes = list(np.arange(int(0.5 * config.min_class_size), config.max_scale, 5))
    metrics_df = evaluate_cluster_size_grid(dfr2, umap_columns, sizes)

    if all_metrics_dfs is not None:
        all_metrics_dfs.append(metrics_df)
        aggregated = aggregate_grid_metrics(all_metrics_dfs)
        solved = solve_optimal_min_cluster_size_from_aggregated(
            aggregated,
            method=config.optimization_method,
            smooth_window=config.grid_smooth_window,
            plateau_fraction=config.grid_plateau_fraction,
            derivative_rel_tol=config.grid_derivative_rel_tol,
        )
        best_size = solved.chosen_min_cluster_size
        best_score = solved.score_mean_at_chosen
    else:
        if len(metrics_df) == 0:
            return None
        best_idx = metrics_df["dbcv"].idxmax()
        best_size = int(metrics_df.loc[best_idx, "min_cluster_size"])
        best_score = float(metrics_df.loc[best_idx, "dbcv"])

    clusterer = hdbscan.HDBSCAN(min_cluster_size=best_size, gen_min_span_tree=True)
    labels = clusterer.fit_predict(dfr2[umap_columns])
    label_set = set(labels.tolist())
    n_clusters_emergent = len(label_set) - (1 if -1 in label_set else 0)
    dfr2_final = dfr2.copy()
    dfr2_final["class"] = pd.DataFrame(
        labels, columns=["class"], index=dfr2_final.index
    )

    ari_score = None
    if "property" in dfr2_final.columns and "class" in dfr2_final.columns:
        property_labels = dfr2_final["property"].astype("category").cat.codes.values
        cluster_labels = dfr2_final["class"].values
        ari_score = compute_adjusted_rand_index(property_labels, cluster_labels)

    return ClusteringReport(
        hyperparameters=ClusteringHyperparameters(min_cluster_size=best_size),
        best_score=best_score,
        number_properties=number_properties,
        n_clusters_emergent=n_clusters_emergent,
        metrics_df=metrics_df,
        df=dfr2_final,
        ari=ari_score,
    )


def estimate_model_clustering(
    transform_config: TransformConfig,
    optimization_config: ClusteringOptimizationConfig | None = None,
    *,
    file_path: pathlib.Path | None = None,
    file_paths: Sequence[pathlib.Path] | None = None,
    dfr: pd.DataFrame | None = None,
    selected_labels: set[str] | None = None,
    all_metrics_dfs: list[pd.DataFrame] | None = None,
):
    """
    Estimate optimal cluster size from parquet file(s) or a preloaded DataFrame.

    Provide exactly one of ``file_path``, ``file_paths``, or ``dfr``. For multiple parquets,
    rows are inner-joined on (pmid, property, mention) and ``embed`` vectors are concatenated
    in path order (must match ``EmbeddingModelMetadata.sources``). Sampling ``frac`` / ``head``
    are applied while loading each file (batches), then ``frac`` is applied once on the merged
    mention-level frame.

    Args:
        transform_config: TransformConfig instance specifying transformation parameters
        optimization_config: Clustering optimization settings. If None, defaults
            to ClusteringOptimizationConfig().
        file_path: Single parquet path (backward-compatible entry point).
        file_paths: Multiple parquets to fuse at mention level before clustering.
        dfr: Optional pre-built frame (e.g. fused) with ``property`` and ``embed`` columns.
        selected_labels: Optional set of labels from selected labels KB to filter by
        all_metrics_dfs: Optional list of metrics DataFrames from previous samples.

    Returns:
        ClusteringReport or None if processing failed
    """
    config = optimization_config or ClusteringOptimizationConfig()
    rns: RandomState = config.rns

    sources = [file_path is not None, file_paths is not None, dfr is not None]
    if sum(bool(x) for x in sources) != 1:
        raise ValueError(
            "Provide exactly one of file_path=, file_paths=, or dfr= to estimate_model_clustering"
        )

    frame: pd.DataFrame | None
    if dfr is not None:
        frame = dfr.copy()
    elif file_paths is not None:
        paths = list(file_paths)
        if len(paths) == 0:
            return None
        parts: list[pd.DataFrame] = []
        for p in paths:
            part = _read_batches_concat(p, config)
            if part is None or len(part) == 0:
                return None
            parts.append(part)
        if len(parts) == 1:
            frame = parts[0]
        else:
            try:
                frame = mention_level_concat_frames(parts)
            except Exception:
                return None
        if frame is None or len(frame) == 0:
            return None
    else:
        assert file_path is not None
        frame = _read_batches_concat(file_path, config)
        if frame is None or len(frame) == 0:
            return None

    frame = frame.sample(frac=config.frac, random_state=rns, replace=False)
    if len(frame) == 0:
        return None

    return estimate_clustering_from_frame(
        frame,
        transform_config,
        optimization_config=config,
        selected_labels=selected_labels,
        all_metrics_dfs=all_metrics_dfs,
        aggregation_level="mention",
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
