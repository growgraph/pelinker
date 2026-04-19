from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

from numpy.random import RandomState

GridObjectiveSpec = Literal[
    "dbcv",
    "ari",
    "dbcv_ari_mean_minmax",
    "dbcv_ari_mean_raw",
]

_GRID_OBJECTIVES: frozenset[str] = frozenset(
    ("dbcv", "ari", "dbcv_ari_mean_minmax", "dbcv_ari_mean_raw")
)


def _validate_semver(version: str) -> None:
    """Require semver 2.0.0 core MAJOR.MINOR.PATCH; allow optional -prerelease+build."""
    s = version.strip()
    if not s:
        raise ValueError("version must be a non-empty string")
    if "+" in s:
        s = s.split("+", 1)[0]
    if "-" in s:
        s = s.split("-", 1)[0]
    parts = s.split(".")
    if len(parts) != 3:
        raise ValueError(
            f"version core must be semver MAJOR.MINOR.PATCH, got {version!r}"
        )
    for p in parts:
        if not p.isdigit():
            raise ValueError(f"invalid semver numeric segment {p!r} in {version!r}")
        n = int(p)
        if p != "0" and p != str(n):
            raise ValueError(
                f"semver numeric segments must not have leading zeros: {version!r}"
            )


@dataclass(frozen=True)
class ClusterCompositionSnapshot:
    """
    Mention-weighted mixture of KB ``property`` labels per HDBSCAN cluster after ``Linker.fit``.

    * :attr:`global_property_mass` — total mention count per property in the fitted corpus
      (denominator for “fraction of that property’s mass” views).
    * :attr:`cluster_within_fraction` — within each cluster, each property’s share of that
      cluster’s mention mass (sums to 1.0 per cluster).
    * :attr:`cluster_fraction_of_property_mass` — for each cluster and property,
      ``mentions(cluster ∩ property) / global_property_mass[property]`` (how much of that
      property’s corpus sits in this cluster; sums to ≤ 1.0 across disjoint cluster rows
      for a fixed property, excluding double-counting issues from overlapping keys).
    """

    global_property_mass: dict[str, int]
    cluster_within_fraction: dict[int, dict[str, float]]
    cluster_fraction_of_property_mass: dict[int, dict[str, float]]


@dataclass(frozen=True)
class KBConfig:
    """Metadata for the knowledge base packaged with a fitted Linker."""

    name: str
    version: str
    created_at: date
    description: str = ""
    entity_count: int | None = None
    """Set after fit from vocabulary size when None at construction time."""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be a non-empty string")
        _validate_semver(self.version)
        if self.entity_count is not None and self.entity_count < 0:
            raise ValueError("entity_count must be >= 0 when provided")


@dataclass(frozen=True)
class EmbeddingSourceSpec:
    """One backbone + layer selection (e.g. for a single encoder or one branch of a fused model)."""

    model_type: str
    layers_spec: str

    def __post_init__(self) -> None:
        if not self.model_type:
            raise ValueError("model_type must be a non-empty string")
        if not self.layers_spec:
            raise ValueError("layers_spec must be a non-empty string")


@dataclass(frozen=True)
class EmbeddingModelMetadata:
    """Describes which embedding backbones/layers produced the model (saved with the Linker)."""

    sources: tuple[EmbeddingSourceSpec, ...]

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("sources must contain at least one EmbeddingSourceSpec")

    @classmethod
    def from_single(cls, model_type: str, layers_spec: str) -> EmbeddingModelMetadata:
        return cls(
            sources=(
                EmbeddingSourceSpec(model_type=model_type, layers_spec=layers_spec),
            )
        )


@dataclass
class EmbeddingTrainingConfig:
    """Inputs and runtime settings used only while embedding the corpus (not part of model identity)."""

    input_text_table_path: Path
    kb_csv_path: Path
    use_gpu: bool = False
    input_buffer_rows: int = 1000
    """Rows read per ``pandas.read_csv(..., chunksize=...)`` pass over the text table (I/O buffer only)."""
    encoder_batch_size: int = 200
    """How many table rows are encoded per transformer forward pass; lower if GPU memory is tight."""
    nlp_model: str = "en_core_web_trf"
    max_input_buffers: int | None = None
    """If set, stop after this many text-table read passes (each up to ``input_buffer_rows`` rows)."""

    def __post_init__(self) -> None:
        if self.input_buffer_rows < 1:
            raise ValueError("input_buffer_rows must be >= 1")
        if self.encoder_batch_size < 1:
            raise ValueError("encoder_batch_size must be >= 1")
        if self.max_input_buffers is not None and self.max_input_buffers < 1:
            raise ValueError("max_input_buffers must be >= 1 when provided")
        self.input_text_table_path = Path(
            os.path.expandvars(os.fspath(self.input_text_table_path))
        ).expanduser()
        self.kb_csv_path = Path(
            os.path.expandvars(os.fspath(self.kb_csv_path))
        ).expanduser()


@dataclass
class ClusteringOptimizationConfig:
    """Configuration for clustering optimization grid search."""

    min_class_size: int = 20
    # Exclusive end of ``np.arange(resolved_min_scale(), max_scale, clustering_grid_step)``.
    max_scale: int = 100
    min_scale: int | None = None
    """Lower bound (inclusive) for the ``min_cluster_size`` grid.

    When ``None``, defaults to ``max(1, min_class_size // 2)`` (legacy behavior: half of
    :attr:`min_class_size`). Set explicitly to decouple grid start from mention-level
    filtering (:attr:`min_class_size`).
    """
    clustering_grid_step: int = 5
    """Step between consecutive ``min_cluster_size`` values on the grid (``numpy.arange`` step)."""
    rns: RandomState = field(default_factory=lambda: RandomState(seed=13))
    frac: float = 1.0
    n_embedding_batches: int | None = None
    """Cap parquet reads at this many batches (`batch_size` rows each); None = read all."""
    batch_size: int = 1000
    """Rows per batch when **reading mention-level embedding parquet** (not encoder batch size)."""
    optimization_method: str = "mean"
    """How to build the objective f(min_cluster_size) before smoothing (mean / lower_bound / weighted)."""
    grid_objective: GridObjectiveSpec = "dbcv_ari_mean_minmax"
    """Which scalar to optimize on the grid (single metric or pooled DBCV+ARI; see ``clustering_grid``)."""
    grid_smooth_window: int = 3
    """Odd-length centered moving-average window for smoothing f(x). Even values are bumped up by one."""
    grid_plateau_fraction: float = 0.92
    """Plateau threshold on the **smoothed** curve: ``y_min + this * (y_max - y_min)`` (finite values only)."""
    grid_derivative_rel_tol: float = 0.12
    """|df/dx| below this times max|df/dx| counts as “derivative near zero” on the smoothed curve."""

    def resolved_min_scale(self) -> int:
        """Inclusive start of the ``min_cluster_size`` grid (HDBSCAN hyperparameter)."""
        if self.min_scale is not None:
            return self.min_scale
        return max(1, self.min_class_size // 2)

    def __post_init__(self) -> None:
        if self.min_class_size < 1:
            raise ValueError("min_class_size must be >= 1")
        if self.min_scale is not None and self.min_scale < 1:
            raise ValueError("min_scale must be >= 1 when provided")
        lo = self.resolved_min_scale()
        if self.max_scale < lo:
            raise ValueError(
                f"max_scale must be >= resolved min_scale ({lo}); got max_scale={self.max_scale}"
            )
        if self.clustering_grid_step < 1:
            raise ValueError("clustering_grid_step must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not 0 < self.frac <= 1:
            raise ValueError("frac must be in range (0, 1]")
        if self.n_embedding_batches is not None and self.n_embedding_batches < 1:
            raise ValueError("n_embedding_batches must be >= 1 when provided")
        if not self.optimization_method:
            raise ValueError("optimization_method must be a non-empty string")
        if self.grid_objective not in _GRID_OBJECTIVES:
            raise ValueError(
                f"grid_objective must be one of {sorted(_GRID_OBJECTIVES)}"
            )
        if self.grid_smooth_window < 1:
            raise ValueError("grid_smooth_window must be >= 1")
        if not 0 < self.grid_plateau_fraction <= 1:
            raise ValueError("grid_plateau_fraction must be in (0, 1]")
        if self.grid_derivative_rel_tol <= 0:
            raise ValueError("grid_derivative_rel_tol must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TransformConfig:
    """Configuration for the embedding transformation pipeline."""

    # PCA configuration
    pca_components: int = 50
    """Number of principal components to keep after PCA reduction."""

    # UMAP configuration
    umap_components: int = 4
    """Number of UMAP dimensions for clustering (typically 3-5)."""
    umap_metric: str = "cosine"
    """Distance metric for UMAP (default: 'cosine')."""

    # Visualization UMAP configuration
    umap_viz_components: int = 3
    """Number of UMAP dimensions for visualization (default: 3)."""
    umap_viz_metric: str = "cosine"
    """Distance metric for visualization UMAP (default: 'cosine')."""

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.pca_components < 1:
            raise ValueError("pca_components must be >= 1")
        if self.umap_components < 2:
            raise ValueError("umap_components must be >= 2")
        if self.umap_viz_components < 2:
            raise ValueError("umap_viz_components must be >= 2")
