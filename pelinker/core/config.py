from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, ClassVar, Literal

from numpy.random import RandomState

from pelinker.core.onto import NEGATIVE_LABEL
from pelinker.core.scaling import ScaleCurve

GridObjectiveSpec = Literal[
    "dbcv",
    "ari",
    "dbcv_ari_geomean",
]

ScreenerKind = Literal["lda", "svm"]
ManifoldKind = Literal["parametric", "umap"]
PredictMode = Literal["compact", "legacy"]

_GRID_OBJECTIVES: frozenset[str] = frozenset(("dbcv", "ari", "dbcv_ari_geomean"))


def _validate_batch_size(batch_size: int) -> None:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")


def _validate_clustering_sample_rows(clustering_sample_rows: int | None) -> None:
    if clustering_sample_rows is not None and clustering_sample_rows < 1:
        raise ValueError("clustering_sample_rows must be >= 1 when provided")


def _validate_min_mentions_per_entity(min_mentions_per_entity: int) -> None:
    if min_mentions_per_entity < 1:
        raise ValueError("min_mentions_per_entity must be >= 1")


def _validate_max_mentions_per_entity(max_mentions_per_entity: int | None) -> None:
    if max_mentions_per_entity is not None and max_mentions_per_entity < 1:
        raise ValueError("max_mentions_per_entity must be >= 1 when provided")


def _validate_max_mentions_negative(max_mentions_negative: int | None) -> None:
    if max_mentions_negative is not None and max_mentions_negative < 1:
        raise ValueError("max_mentions_negative must be >= 1 when provided")


def _validate_mention_frame_load_fields(
    *,
    batch_size: int,
    min_mentions_per_entity: int,
    max_mentions_per_entity: int | None,
    max_mentions_negative: int | None,
) -> None:
    _validate_batch_size(batch_size)
    _validate_min_mentions_per_entity(min_mentions_per_entity)
    _validate_max_mentions_per_entity(max_mentions_per_entity)
    _validate_max_mentions_negative(max_mentions_negative)


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
    nlp_model: str = "en_core_web_lg"
    max_input_buffers: int | None = None
    """If set, stop after this many text-table read passes (each up to ``input_buffer_rows`` rows)."""
    negatives_per_positive: float = 0.0
    """Number of random negative mentions to sample per positive mention."""
    negative_label: str = NEGATIVE_LABEL
    """Entity label to use for synthetic negative rows."""
    negative_seed: int | None = 13
    """Optional random seed for deterministic negative sampling."""

    def __post_init__(self) -> None:
        if self.input_buffer_rows < 1:
            raise ValueError("input_buffer_rows must be >= 1")
        if self.encoder_batch_size < 1:
            raise ValueError("encoder_batch_size must be >= 1")
        if self.max_input_buffers is not None and self.max_input_buffers < 1:
            raise ValueError("max_input_buffers must be >= 1 when provided")
        if self.negatives_per_positive < 0:
            raise ValueError("negatives_per_positive must be >= 0")
        if not self.negative_label:
            raise ValueError("negative_label must be a non-empty string")
        self.input_text_table_path = Path(
            os.path.expandvars(os.fspath(self.input_text_table_path))
        ).expanduser()
        self.kb_csv_path = Path(
            os.path.expandvars(os.fspath(self.kb_csv_path))
        ).expanduser()


@dataclass(frozen=True)
class NegativeScreenerConfig:
    """Binary LDA/SVM screen for ``negative_label`` vs KB mentions before PCA→UMAP."""

    kind: ScreenerKind = "lda"
    """Estimator persisted on :class:`~pelinker.model.Linker` (``Linker.screener``)."""
    negative_label: str = NEGATIVE_LABEL
    cv_n_splits: int = 5
    cv_test_size: float = 0.2
    cv_random_state: int = 42

    def __post_init__(self) -> None:
        if not self.negative_label.strip():
            raise ValueError("negative_label must be non-empty")
        if self.cv_n_splits < 2:
            raise ValueError("cv_n_splits must be >= 2")
        if not 0.0 < self.cv_test_size < 1.0:
            raise ValueError("cv_test_size must be in (0, 1)")


@dataclass(frozen=True)
class ManifoldOovScreenerConfig:
    """3D (residual, Mahalanobis, spectral entropy) OOV score model; predict-time gate only."""

    enabled: bool = True
    cv_n_splits: int = 5
    cv_test_size: float = 0.2
    cv_random_state: int = 42
    oov_rbf_C: float = 1.0
    oov_rbf_gamma: float | Literal["scale", "auto"] = "scale"

    def __post_init__(self) -> None:
        if self.cv_n_splits < 2:
            raise ValueError("cv_n_splits must be >= 2")
        if not 0.0 < self.cv_test_size < 1.0:
            raise ValueError("cv_test_size must be in (0, 1)")
        if self.oov_rbf_C <= 0.0:
            raise ValueError("oov_rbf_C must be > 0")
        if isinstance(self.oov_rbf_gamma, (int, float)):
            if float(self.oov_rbf_gamma) <= 0.0:
                raise ValueError(
                    "oov_rbf_gamma must be > 0 when a numeric value is used"
                )
        elif self.oov_rbf_gamma not in ("scale", "auto"):
            raise ValueError(
                'oov_rbf_gamma must be "scale", "auto", or a positive float'
            )


@dataclass(frozen=True)
class DistillationGateConfig:
    """Quality bounds on the compact entity head, checked against a held-out slice.

    Defaults match the bounds `run/analysis/compact_predict_study.py` applied by hand
    before compact became the shipped default; they live here so a fit can state whether
    it met them instead of nobody knowing.
    """

    enabled: bool = True
    min_entity_agreement: float = 0.95
    """Floor on held-out student/teacher entity-id agreement."""
    max_emit_rate_rel_delta: float = 0.10
    """Cap on |student - teacher| / teacher emit rate at :attr:`emit_rate_threshold`."""
    emit_rate_threshold: float = 0.3
    """Reference ``thr_score`` for the emit-rate comparison."""
    on_failure: Literal["warn", "raise"] = "warn"
    """``warn`` keeps the fitted model (and records the failure); ``raise`` aborts.

    Warning is the default because a fit consumes an expensive corpus embedding, and
    discarding it mid-run is worse than shipping a model whose report says it failed.
    """

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_entity_agreement <= 1.0:
            raise ValueError("min_entity_agreement must be in [0, 1]")
        if self.max_emit_rate_rel_delta < 0.0:
            raise ValueError("max_emit_rate_rel_delta must be >= 0")
        if not 0.0 <= self.emit_rate_threshold <= 1.0:
            raise ValueError("emit_rate_threshold must be in [0, 1]")
        if self.on_failure not in ("warn", "raise"):
            raise ValueError(
                f"on_failure must be 'warn' or 'raise', got {self.on_failure!r}"
            )


@dataclass
class MentionFrameLoadConfig:
    """Shared mention-level parquet load and pre-subsample filters."""

    batch_size: int = 1000
    drop_rare_entities: bool = False
    """When true, drop KB entities with fewer than :attr:`min_mentions_per_entity` rows."""
    min_mentions_per_entity: int = 20
    max_mentions_per_entity: int | None = None
    """Cap mention rows per KB entity (seeded); ``None`` = no cap."""
    max_mentions_negative: int | None = None
    """Cap for :attr:`~NegativeScreenerConfig.negative_label`; ``None`` = exempt."""
    mention_cap_seed: int = 13

    def __post_init__(self) -> None:
        _validate_mention_frame_load_fields(
            batch_size=self.batch_size,
            min_mentions_per_entity=self.min_mentions_per_entity,
            max_mentions_per_entity=self.max_mentions_per_entity,
            max_mentions_negative=self.max_mentions_negative,
        )


@dataclass
class LinkerFitConfig:
    """Parquet read + mention filters + screener settings for :meth:`~pelinker.model.Linker.fit`."""

    batch_size: int = 1000
    drop_rare_entities: bool = False
    min_mentions_per_entity: int = 20
    max_mentions_per_entity: int | None = None
    max_mentions_negative: int | None = None
    mention_cap_seed: int = 13
    ambient_screener: NegativeScreenerConfig = field(
        default_factory=NegativeScreenerConfig
    )
    projection_screener: ManifoldOovScreenerConfig = field(
        default_factory=ManifoldOovScreenerConfig
    )
    screener_max_rows: int | None = 100_000
    """Max rows for ambient + projection screener training when using the full frame (stratified). None = no cap."""
    screener_seed: int = 13
    """Random seed for the stratified screener training draw when using the full frame."""
    clustering_sample_rows: int | None = None
    """Max mention rows per clustering bootstrap draw (stratified). None = use all loaded rows."""
    base_seed: int = 13
    """Seed for stratified clustering draws; draw seed is ``base_seed + clustering_sample_index``."""
    clustering_sample_index: int = 0
    """Bootstrap index for the clustering subsample (same contract as model selection ``sample_idx``)."""
    diagnostics_sample_size: int = 20_000
    """Max rows of :class:`~pelinker.reports.schema.LinkerFitDiagnostics` stored on the fit report."""
    diagnostics_random_state: int = 0
    """Stratified subsample seed for training diagnostics."""
    scale_curve: ScaleCurve | None = None
    """Fitted ``min_cluster_size``-vs-N law from ``pelinker-scale-curve``.

    When set (and no explicit ``min_cluster_size`` was given), the hyperparameter is
    extrapolated to the realized manifold row count instead of being transferred verbatim
    from whatever sample size selection happened to run at. See :mod:`pelinker.core.scaling`.
    """
    predict_mode: PredictMode = "compact"
    """``compact``: ParametricUMAP + MLP entity head (no shipped HDBSCAN). ``legacy``: UMAP + HDBSCAN ``approximate_predict``."""
    entity_head_hidden_layers: tuple[int, ...] = (256, 128, 128)
    """Hidden layer sizes for the compact MLP entity head (ignored in ``legacy`` mode)."""
    entity_head_holdout_fraction: float = 0.15
    """Rows withheld from entity-head training to measure distillation fidelity.

    ``0.0`` trains on every teacher-labelled row (the pre-fidelity behaviour) and skips
    the measurement — use it only to reproduce an existing artifact.
    """
    entity_head_holdout_group_col: str = "pmid"
    """Column kept whole across the holdout split; mentions from one document are
    correlated, so a row-level split inflates the measured agreement."""
    entity_head_holdout_seed: int = 13
    distillation_gates: DistillationGateConfig = field(
        default_factory=DistillationGateConfig
    )

    def to_clustering_sample_config(self) -> ClusteringOptimizationConfig:
        """Build a :class:`ClusteringOptimizationConfig` for load + subsample helpers."""
        return ClusteringOptimizationConfig(
            base_seed=self.base_seed,
            clustering_sample_rows=self.clustering_sample_rows,
            batch_size=self.batch_size,
            drop_rare_entities=self.drop_rare_entities,
            min_mentions_per_entity=self.min_mentions_per_entity,
            max_mentions_per_entity=self.max_mentions_per_entity,
            max_mentions_negative=self.max_mentions_negative,
            mention_cap_seed=self.mention_cap_seed,
            ambient_screener=self.ambient_screener,
            projection_screener=self.projection_screener,
        )

    def __post_init__(self) -> None:
        _validate_mention_frame_load_fields(
            batch_size=self.batch_size,
            min_mentions_per_entity=self.min_mentions_per_entity,
            max_mentions_per_entity=self.max_mentions_per_entity,
            max_mentions_negative=self.max_mentions_negative,
        )
        if self.screener_max_rows is not None and self.screener_max_rows < 1:
            raise ValueError("screener_max_rows must be >= 1 when provided")
        _validate_clustering_sample_rows(self.clustering_sample_rows)
        if self.clustering_sample_index < 0:
            raise ValueError("clustering_sample_index must be >= 0")
        if self.diagnostics_sample_size < 1:
            raise ValueError("diagnostics_sample_size must be >= 1")
        if self.predict_mode not in ("compact", "legacy"):
            raise ValueError(
                f"predict_mode must be 'compact' or 'legacy', got {self.predict_mode!r}"
            )
        if not self.entity_head_hidden_layers:
            raise ValueError("entity_head_hidden_layers must be a non-empty tuple")
        if any(int(h) < 1 for h in self.entity_head_hidden_layers):
            raise ValueError("entity_head_hidden_layers values must be >= 1")
        if not 0.0 <= self.entity_head_holdout_fraction < 1.0:
            raise ValueError("entity_head_holdout_fraction must be in [0, 1)")
        if not self.entity_head_holdout_group_col:
            raise ValueError("entity_head_holdout_group_col must be a non-empty string")


@dataclass
class ClusteringOptimizationConfig:
    """Configuration for clustering optimization grid search."""

    min_class_size: int = 20
    # Exclusive end of ``np.arange(resolved_min_scale(), max_scale, clustering_grid_step)``.
    max_scale: int = 100
    min_scale: int | None = None
    """Lower bound (inclusive) for the ``min_cluster_size`` grid.

    When ``None``, defaults to ``max(1, min_class_size // 2)``.
    """
    clustering_grid_step: int = 5
    """Step between consecutive ``min_cluster_size`` values on the grid (``numpy.arange`` step)."""
    rns: RandomState = field(default_factory=lambda: RandomState(seed=13))
    base_seed: int = 13
    """Seed for stratified selection draws; per-bootstrap seed is ``base_seed + sample_index``."""
    clustering_sample_rows: int | None = None
    """Max mention rows per clustering bootstrap draw (stratified). None = use all loaded rows."""
    batch_size: int = 1000
    """Rows per batch when **reading mention-level embedding parquet** (not encoder batch size)."""
    drop_rare_entities: bool = False
    min_mentions_per_entity: int = 20
    max_mentions_per_entity: int | None = None
    max_mentions_negative: int | None = None
    mention_cap_seed: int = 13
    grid_objective: GridObjectiveSpec = "dbcv_ari_geomean"
    """Which scalar to optimize on the grid.

    ``dbcv_ari_geomean`` is ``sqrt(max(dbcv, 0) * max(ari, 0))``: a conjunctive score whose
    ranking is invariant to the (arbitrary, and very different) scales of DBCV and ARI.
    """
    grid_smooth_window: int = 3
    """Odd-length centered moving-average window applied to each sample's curve. Even values are bumped up by one."""
    grid_one_se_k: float = 1.0
    """How many paired standard errors from the best grid point still count as a tie.

    The chosen ``min_cluster_size`` is the largest one reachable through consecutive grid
    points within ``k`` standard errors of the argmax. 0 disables the rule (pure argmax).
    """
    grid_one_se_contiguous: bool = True
    """Require the tying grid points to form an unbroken run from the argmax."""
    grid_one_se_min_samples: int = 6
    """Minimum bootstrap samples before the one-SE rule is applied at all.

    Below this, fall back to the plain argmax. A standard error estimated from three
    numbers is mostly noise, and acting on it makes the choice *less* reproducible: on this
    repo's grid exports the slide only starts paying for itself around 6-8 samples.
    """
    grid_cluster_count_reward: float = 0.0
    """Weight on ``log(n_clusters / n_ref)`` added to the grid objective (0 = disabled)."""
    grid_n_entities: int | None = None
    """Reference entity count for the cluster-count term; when ``None``, uses max mean cluster count on the grid."""
    ambient_screener: NegativeScreenerConfig = field(
        default_factory=NegativeScreenerConfig
    )
    """Negative-class screening before PCA→UMAP (see :class:`NegativeScreenerConfig`)."""
    projection_screener: ManifoldOovScreenerConfig = field(
        default_factory=ManifoldOovScreenerConfig
    )
    """Validation config for manifold OOV model selection (analysis reporting only)."""

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
        _validate_mention_frame_load_fields(
            batch_size=self.batch_size,
            min_mentions_per_entity=self.min_mentions_per_entity,
            max_mentions_per_entity=self.max_mentions_per_entity,
            max_mentions_negative=self.max_mentions_negative,
        )
        _validate_clustering_sample_rows(self.clustering_sample_rows)
        if self.grid_objective not in _GRID_OBJECTIVES:
            raise ValueError(
                f"grid_objective must be one of {sorted(_GRID_OBJECTIVES)}"
            )
        if self.grid_smooth_window < 1:
            raise ValueError("grid_smooth_window must be >= 1")
        if self.grid_one_se_k < 0:
            raise ValueError("grid_one_se_k must be >= 0")
        if self.grid_one_se_min_samples < 1:
            raise ValueError("grid_one_se_min_samples must be >= 1")
        if self.grid_cluster_count_reward < 0:
            raise ValueError("grid_cluster_count_reward must be >= 0")
        if self.grid_n_entities is not None and self.grid_n_entities < 1:
            raise ValueError("grid_n_entities must be >= 1 when provided")

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
    umap_n_neighbors: int | None = None
    """UMAP ``n_neighbors``. ``None`` keeps the library default (15) for any workable frame.

    This is a **scale-dependent** knob: 15 neighbours describe a very different
    neighbourhood at 10k rows than at 2M. Set it explicitly (or sweep it with
    ``pelinker-scale-curve``) when the fit N differs markedly from the selection N.
    See :meth:`resolve_n_neighbors`.
    """
    manifold_kind: ManifoldKind = "umap"
    """Clustering manifold: ``parametric`` (ParametricUMAP) or ``umap`` (standard UMAP). Compact fit forces ``parametric``."""
    parametric_umap_n_training_epochs: int = 10
    """ParametricUMAP training epochs over the UMAP graph (ignored for ``manifold_kind='umap'``)."""
    parametric_umap_batch_size: int | None = None
    """ParametricUMAP edge batch size; ``None`` uses library default."""

    # Cluster-space visualization (reduces umap_clustering coords for plotting)
    cluster_viz_components: int = 3
    """Number of dimensions for cluster-space visualization (default: 3)."""
    cluster_viz_method: Literal["pca", "umap"] = "pca"
    """Reducer applied to clustering UMAP coords: ``pca`` (linear) or ``umap``."""
    cluster_viz_umap_metric: str = "euclidean"
    """Distance metric for cluster-space UMAP viz (only when ``cluster_viz_method='umap'``)."""

    pca_seed: int = 13
    """Random seed for PCA and cluster-viz PCA."""
    umap_seed: int | None = None
    """UMAP random seed; ``None`` enables parallel UMAP (non-reproducible). Cluster-viz UMAP uses ``umap_seed + 1`` when set."""

    DEFAULT_UMAP_N_NEIGHBORS: ClassVar[int] = 15
    """umap-learn's own default; used when :attr:`umap_n_neighbors` is ``None``."""

    def resolve_n_neighbors(self, n_samples: int) -> int:
        """``n_neighbors`` for a frame of ``n_samples`` rows, clamped to what UMAP accepts.

        UMAP requires ``n_neighbors < n_samples``, so tiny frames are capped; this is the
        single place that rule lives.
        """
        requested = (
            self.DEFAULT_UMAP_N_NEIGHBORS
            if self.umap_n_neighbors is None
            else int(self.umap_n_neighbors)
        )
        return max(1, min(requested, n_samples - 1))

    def __post_init__(self):
        if self.pca_components < 1:
            raise ValueError("pca_components must be >= 1")
        if self.umap_components < 2:
            raise ValueError("umap_components must be >= 2")
        if self.umap_n_neighbors is not None and self.umap_n_neighbors < 2:
            raise ValueError("umap_n_neighbors must be >= 2 when provided")
        if self.cluster_viz_components < 2:
            raise ValueError("cluster_viz_components must be >= 2")
        if self.cluster_viz_components > self.umap_components:
            raise ValueError(
                "cluster_viz_components must be <= umap_components "
                f"(got {self.cluster_viz_components} > {self.umap_components})"
            )
        if self.cluster_viz_method not in ("pca", "umap"):
            raise ValueError(
                "cluster_viz_method must be 'pca' or 'umap', "
                f"got {self.cluster_viz_method!r}"
            )
        if self.manifold_kind not in ("parametric", "umap"):
            raise ValueError(
                "manifold_kind must be 'parametric' or 'umap', "
                f"got {self.manifold_kind!r}"
            )
        if self.parametric_umap_n_training_epochs < 1:
            raise ValueError("parametric_umap_n_training_epochs must be >= 1")
        if (
            self.parametric_umap_batch_size is not None
            and self.parametric_umap_batch_size < 1
        ):
            raise ValueError("parametric_umap_batch_size must be >= 1 when provided")
