from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np


"""Lexical scoring: corpus word frequencies, label simplicity, KB generality."""


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
