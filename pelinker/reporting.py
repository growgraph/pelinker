from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MeanWithUncertainty:
    """Sample mean and standard deviation (ddof=1) over repeated runs; ``std=0`` for a single run."""

    mean: float
    std: float


@dataclass(frozen=True)
class ClusteringHyperparameters:
    """
    HDBSCAN (and related) choices selected by the grid search / smoother.

    Add fields here as more knobs participate in optimization; call sites then stay typed.
    """

    min_cluster_size: int


@dataclass(frozen=True)
class HyperparameterSearchStats:
    """Distribution of chosen hyperparameters across repeated clustering samples."""

    min_cluster_size: MeanWithUncertainty


@dataclass
class ClusteringReport:
    """Report containing clustering analysis results for one sample."""

    hyperparameters: ClusteringHyperparameters
    best_score: float
    """DBCV (``relative_validity_``) at the chosen ``min_cluster_size`` (mean when from aggregate)."""

    number_properties: int
    metrics_df: pd.DataFrame
    df: pd.DataFrame
    hungarian_accuracy: float | None = None

    @property
    def best_size(self) -> int:
        """Backward-compatible alias for ``hyperparameters.min_cluster_size``."""
        return self.hyperparameters.min_cluster_size


@dataclass(frozen=True)
class ClusteringSearchSummaryRow:
    """
    One row of the model×layer clustering search table (singleton or fusion label).

    Use :meth:`to_flat_dict` for CSV / pandas / heatmaps (legacy column names).
    """

    model: str
    layer: str
    hyperparameters: HyperparameterSearchStats
    number_properties: MeanWithUncertainty
    dbcv: MeanWithUncertainty
    hungarian_accuracy: MeanWithUncertainty | None

    def to_flat_dict(self) -> dict[str, str | float | None]:
        """Keys aligned with historical ``results.csv`` and ``plot_heatmap`` expectations."""
        h = self.hyperparameters.min_cluster_size
        p = self.number_properties
        d = self.dbcv
        row: dict[str, str | float | None] = {
            "model": self.model,
            "layer": self.layer,
            "best_size": h.mean,
            "best_size_std": h.std,
            "number_properties": p.mean,
            "number_properties_std": p.std,
            "best_score": d.mean,
            "best_score_std": d.std,
        }
        if self.hungarian_accuracy is None:
            row["hungarian_accuracy"] = None
            row["hungarian_accuracy_std"] = 0.0
        else:
            ha = self.hungarian_accuracy
            row["hungarian_accuracy"] = ha.mean
            row["hungarian_accuracy_std"] = ha.std
        return row


def summarize_clustering_reports_for_search(
    reports: Sequence[ClusteringReport],
    *,
    model: str,
    layer: str,
) -> ClusteringSearchSummaryRow:
    """
    Aggregate repeated :class:`ClusteringReport` runs into one search summary row.

    Raises:
        ValueError: if ``reports`` is empty.
    """
    if not reports:
        raise ValueError("reports must be non-empty")

    sizes = np.array(
        [r.hyperparameters.min_cluster_size for r in reports], dtype=np.float64
    )
    scores = np.array([r.best_score for r in reports], dtype=np.float64)
    nprops = np.array([r.number_properties for r in reports], dtype=np.float64)
    hungarian_vals = [
        float(r.hungarian_accuracy) for r in reports if r.hungarian_accuracy is not None
    ]

    n = len(reports)
    std_sizes = float(np.std(sizes)) if n > 1 else 0.0
    std_scores = float(np.std(scores)) if n > 1 else 0.0
    std_nprops = float(np.std(nprops)) if n > 1 else 0.0

    hungarian_block: MeanWithUncertainty | None
    if hungarian_vals:
        arr = np.array(hungarian_vals, dtype=np.float64)
        hungarian_block = MeanWithUncertainty(
            mean=float(np.mean(arr)),
            std=float(np.std(arr)) if len(arr) > 1 else 0.0,
        )
    else:
        hungarian_block = None

    return ClusteringSearchSummaryRow(
        model=model,
        layer=layer,
        hyperparameters=HyperparameterSearchStats(
            min_cluster_size=MeanWithUncertainty(
                mean=float(np.mean(sizes)),
                std=std_sizes,
            ),
        ),
        number_properties=MeanWithUncertainty(
            mean=float(np.mean(nprops)),
            std=std_nprops,
        ),
        dbcv=MeanWithUncertainty(
            mean=float(np.mean(scores)),
            std=std_scores,
        ),
        hungarian_accuracy=hungarian_block,
    )
