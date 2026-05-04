from __future__ import annotations

import math
import pathlib
from collections.abc import Callable, Iterable, Mapping, Sequence
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
from pelinker.plotting import (
    GRID_COL_CHOSEN_MIN_CLUSTER_SIZE,
    GRID_COL_SAMPLE_ARI,
    GRID_COL_SAMPLE_BEST_DBCV,
)
from pelinker.embedding_fusion import concat_mention_level_embedding_sources
from pelinker.reporting import (
    ClusteringFitMetrics,
    ClusteringHyperparameters,
    ClusteringReport,
    NegativeScreenerCvSummary,
    NegativeScreenerInSampleMetrics,
    entity_negative_label_mask_01,
    negative_screener_cv_summary_from_eval_dict,
)
from sklearn.metrics import adjusted_rand_score, f1_score, precision_score, recall_score
from numpy.random import RandomState

from pelinker.config import (
    ClusteringOptimizationConfig,
    NegativeScreenerConfig,
    TransformConfig,
)
from pelinker.negative_screener import (
    NegativeClassScreener,
    evaluate_negative_screener_models,
)
from pelinker.transform import compute_transform_artifacts


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


def pooled_min_cluster_size_from_metrics_dfs(
    metrics_dfs: Sequence[pd.DataFrame],
    optimization_config: ClusteringOptimizationConfig | None = None,
) -> tuple[int, float]:
    """
    After all bootstrap samples have run a min_cluster_size grid, aggregate their metrics
    once and return the smoothed ``(chosen_min_cluster_size, raw objective mean at that grid point)``.
    The objective is set by ``ClusteringOptimizationConfig.grid_objective`` (default: pooled
    min–max normalized DBCV and ARI).
    """
    if not metrics_dfs:
        raise ValueError("metrics_dfs must be non-empty")
    config = optimization_config or ClusteringOptimizationConfig()
    aggregated = aggregate_grid_metrics(list(metrics_dfs))
    solved = solve_optimal_min_cluster_size_from_aggregated(
        aggregated,
        objective=config.grid_objective,
        method=config.optimization_method,
        smooth_window=config.grid_smooth_window,
        plateau_fraction=config.grid_plateau_fraction,
        derivative_rel_tol=config.grid_derivative_rel_tol,
    )
    return solved.chosen_min_cluster_size, solved.score_mean_at_chosen


def metrics_df_with_grid_sample_columns(
    report: ClusteringReport,
    *,
    model: str,
    layer: str,
    sample_idx: int,
    chosen_min_cluster_size: int | None = None,
) -> pd.DataFrame:
    """
    Per-sample grid rows for ``results_grid_per_sample.csv``.

    ``chosen_min_cluster_size`` defaults to the value used to fit this sample's clusters
    (per-sample grid argmax). Pass the pooled choice from
    :func:`pooled_min_cluster_size_from_metrics_dfs` so every row shares one consensus marker.
    """
    ari = report.ari
    h = (
        chosen_min_cluster_size
        if chosen_min_cluster_size is not None
        else report.hyperparameters.min_cluster_size
    )
    return report.metrics_df.assign(
        model=model,
        layer=layer,
        sample_idx=sample_idx,
        **{
            GRID_COL_CHOSEN_MIN_CLUSTER_SIZE: int(h),
            GRID_COL_SAMPLE_BEST_DBCV: float(report.best_score),
            GRID_COL_SAMPLE_ARI: float("nan") if ari is None else float(ari),
        },
    )


def split_by_negative_label(
    dfr: pd.DataFrame,
    negative_label: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Split a mention frame into a boolean mask of synthetic-negative rows and the
    manifold frame (KB / non-negative rows only).
    """
    neg_mask = dfr["entity"].astype(str).values == negative_label
    manifold_df = dfr.loc[~neg_mask].copy()
    return neg_mask, manifold_df


def evaluate_negative_screener_cv_summary(
    dfr: pd.DataFrame,
    cfg: NegativeScreenerConfig,
) -> NegativeScreenerCvSummary | None:
    """Stratified CV for LDA vs linear SVM on negative vs KB (same task as grid analysis)."""
    neg_mask = dfr["entity"].astype(str).values == cfg.negative_label
    if not neg_mask.any():
        return None
    X_all = np.stack(dfr["embed"].values).astype(np.float32, copy=False)
    y_bin = neg_mask.astype(np.int64)
    raw_cv = evaluate_negative_screener_models(
        X_all,
        y_bin,
        n_splits=cfg.cv_n_splits,
        test_size=cfg.cv_test_size,
        random_state=cfg.cv_random_state,
    )
    if raw_cv is None:
        return None
    return negative_screener_cv_summary_from_eval_dict(raw_cv)


def fit_negative_screener_with_metrics(
    dfr: pd.DataFrame,
    config: NegativeScreenerConfig,
) -> tuple[NegativeClassScreener, NegativeScreenerInSampleMetrics | None]:
    """
    Fit the persisted screener on ``dfr`` and report in-sample PR/F1 for detecting
    ``negative_label`` when both classes are present.
    """
    screener = NegativeClassScreener.fit_from_frame(dfr, config)
    y_true = (dfr["entity"].astype(str).values == config.negative_label).astype(
        np.int64
    )
    n_kb = int(np.sum(y_true == 0))
    n_neg = int(np.sum(y_true == 1))
    if n_kb == 0 or n_neg == 0:
        return screener, None
    X = np.stack(dfr["embed"].values).astype(np.float32, copy=False)
    y_pred = screener.predict_is_negative(X).astype(np.int64)
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    return screener, NegativeScreenerInSampleMetrics(
        precision=prec,
        recall=rec,
        f1=f1,
        n_kb_mentions=n_kb,
        n_negative_label_mentions=n_neg,
        kind=config.kind,
    )


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


def estimate_clustering_from_frame(
    dfr: pd.DataFrame,
    transform_config: TransformConfig,
    optimization_config: ClusteringOptimizationConfig | None = None,
    *,
    selected_labels: set[str] | None = None,
    all_metrics_dfs: list[pd.DataFrame] | None = None,
    aggregation_level: Literal["mention", "entity"] = "mention",
) -> ClusteringReport | None:
    """
    Run clustering grid search and optional **accumulation** of per-sample grid tables.

    When ``all_metrics_dfs`` is provided, each call appends this sample's ``metrics_df`` to
    that list. The optimal ``min_cluster_size`` for the final HDBSCAN fit **on this sample**
    is always the per-sample DBCV argmax on its own grid; run
    :func:`pooled_min_cluster_size_from_metrics_dfs` after all samples to obtain one consensus
    choice across bootstraps (for summaries, plots, and optional grid CSV markers).

    ``aggregation_level="entity"`` expects one row per distinct ``entity`` (e.g. fused
    KB vectors) and skips min-mention-per-entity trimming used for mention-level corpora.
    """

    config = optimization_config or ClusteringOptimizationConfig()

    if "embed" not in dfr.columns or "entity" not in dfr.columns:
        return None

    if selected_labels is not None:
        dfr = dfr.loc[dfr["entity"].isin(selected_labels)].copy()
        if len(dfr) == 0:
            return None

    screener_cfg = config.negative_screener
    neg_label = screener_cfg.negative_label

    if aggregation_level == "mention":
        mention_count = dfr["entity"].value_counts()
        low_count_entities = mention_count[
            ~(mention_count >= config.min_class_size)
        ].index.to_list()
        low_count_entities = [e for e in low_count_entities if e != neg_label]
        dfr = dfr.loc[~dfr["entity"].isin(low_count_entities)].copy()
        if len(dfr) == 0:
            return None

    negative_screener_cv = evaluate_negative_screener_cv_summary(dfr, screener_cfg)
    _, dfr_manifold = split_by_negative_label(dfr, neg_label)
    if len(dfr_manifold) == 0:
        return None

    number_properties = int(dfr_manifold["entity"].nunique())

    artifacts = compute_transform_artifacts(
        dfr_manifold,
        config=transform_config,
        embed_column="embed",
    )
    umap_clustering_df = artifacts.umap_clustering_df()

    sizes = list(
        np.arange(
            config.resolved_min_scale(),
            config.max_scale,
            config.clustering_grid_step,
        )
    )
    metrics_df = evaluate_cluster_size_grid(
        umap_clustering_df,
        list(umap_clustering_df.columns),
        sizes,
    )
    if len(metrics_df) == 0:
        return None

    if all_metrics_dfs is not None:
        all_metrics_dfs.append(metrics_df)

    best_idx = metrics_df["dbcv"].idxmax()
    best_size = int(metrics_df.loc[best_idx, "min_cluster_size"])
    best_score = float(metrics_df.loc[best_idx, "dbcv"])

    clusterer = hdbscan.HDBSCAN(min_cluster_size=best_size, gen_min_span_tree=True)
    labels = clusterer.fit_predict(artifacts.umap_clustering)
    fit_metrics = compute_clustering_fit_metrics(
        clusterer,
        dfr_manifold,
        min_cluster_size=best_size,
        cluster_labels=labels,
    )
    assignments = dfr_manifold[["entity"]].copy()
    for optional_col in ["pmid", "mention"]:
        if optional_col in dfr_manifold.columns:
            assignments[optional_col] = dfr_manifold[optional_col]
    assignments["cluster"] = labels.astype(int, copy=False)

    res_a = artifacts.pca_residuals
    mah_a = artifacts.pca_mahalanobis
    ent_a = artifacts.pca_spectral_entropy
    y_neg = entity_negative_label_mask_01(dfr_manifold["entity"], neg_label)
    return ClusteringReport(
        hyperparameters=ClusteringHyperparameters(min_cluster_size=best_size),
        best_score=best_score,
        number_properties=number_properties,
        n_clusters_emergent=fit_metrics.n_clusters_emergent,
        metrics_df=metrics_df,
        assignments=assignments,
        pca_residuals=res_a,
        pca_mahalanobis=mah_a,
        pca_spectral_entropy=ent_a,
        pca_residual_label_01=y_neg,
        pca_mahalanobis_label_01=y_neg,
        pca_spectral_entropy_label_01=y_neg,
        umap_clustering=artifacts.umap_clustering,
        umap_visualization=artifacts.umap_visualization,
        pca_reduced=artifacts.pca_reduced,
        negative_screener_cv=negative_screener_cv,
        manifold_oov_cv=None,
        ari=fit_metrics.ari,
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
    embedding_read_status: Callable[[str], None] | None = None,
    show_embedding_read_progress: bool = False,
):
    """
    Estimate optimal cluster size from parquet file(s) or a preloaded DataFrame.

    Provide exactly one of ``file_path``, ``file_paths``, or ``dfr``. For multiple parquets,
    rows are inner-joined on (pmid, entity, mention) and ``embed`` vectors are concatenated
    in path order (must match ``EmbeddingModelMetadata.sources``). Sampling ``frac`` /
    ``n_embedding_batches`` are applied while loading each file (batches), then ``frac`` is applied
    once on the merged
    mention-level frame.

    Args:
        transform_config: TransformConfig instance specifying transformation parameters
        optimization_config: Clustering optimization settings. If None, defaults
            to ClusteringOptimizationConfig().
        file_path: Single parquet path (backward-compatible entry point).
        file_paths: Multiple parquets to fuse at mention level before clustering.
        dfr: Optional pre-built frame (e.g. fused) with ``entity`` and ``embed`` columns.
        selected_labels: Optional set of labels from selected labels KB to filter by
        all_metrics_dfs: Optional mutable list that receives each sample's grid ``DataFrame``
            (for a pooled choice after the batch via :func:`pooled_min_cluster_size_from_metrics_dfs`).
        embedding_read_status: Callback for embedding parquet batch progress lines
            (e.g. append to an existing Rich ``Progress`` task description).
        show_embedding_read_progress: When True and ``embedding_read_status`` is omitted,
            show a transient Rich progress display while loading parquet batches.

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
    else:
        if file_paths is not None:
            paths = list(file_paths)
        else:
            assert file_path is not None
            paths = [file_path]
        if len(paths) == 0:
            return None
        frame = concat_mention_level_embedding_sources(
            paths,
            batch_size=config.batch_size,
            n_embedding_batches=config.n_embedding_batches,
            read_status=embedding_read_status,
            show_read_progress=show_embedding_read_progress,
        )
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


def mention_frame_from_embedding_paths(
    paths: Sequence[pathlib.Path],
    *,
    optimization_config: ClusteringOptimizationConfig | None = None,
    read_status: Callable[[str], None] | None = None,
    show_read_progress: bool = False,
) -> pd.DataFrame | None:
    """
    Load mention-level rows from parquet file(s) like ``estimate_model_clustering``
    (batched read, optional multi-source inner join on keys), without ``frac`` subsampling.
    """
    cfg = optimization_config or ClusteringOptimizationConfig()
    return concat_mention_level_embedding_sources(
        paths,
        batch_size=cfg.batch_size,
        n_embedding_batches=cfg.n_embedding_batches,
        read_status=read_status,
        show_read_progress=show_read_progress,
    )


def drop_entities_with_few_mentions(
    frame: pd.DataFrame,
    min_mentions_per_entity: int,
    *,
    negative_label: str | None = None,
) -> pd.DataFrame:
    """
    Drop entities with fewer than ``min_mentions_per_entity`` rows (same rule as
    ``estimate_clustering_from_frame`` with ``aggregation_level='mention'``).

    When ``negative_label`` is set, that label is never dropped for low mention count
    (so thin negative tails remain for screener training).
    """
    if "entity" not in frame.columns:
        raise ValueError("frame must contain an 'entity' column")
    mention_count = frame["entity"].value_counts()
    low_count = mention_count[
        ~(mention_count >= min_mentions_per_entity)
    ].index.to_list()
    if negative_label is not None:
        low_count = [e for e in low_count if e != negative_label]
    return frame.loc[~frame["entity"].isin(low_count)].copy()


def embeddings_dict_to_dataframe(
    embeddings_dict: dict[str, tuple[str, torch.Tensor | np.ndarray]],
) -> pd.DataFrame:
    """
    Convert embeddings dictionary to DataFrame format expected by transform artifacts.

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
