from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from numpy.random import RandomState


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
    chunk_size: int = 1000
    batch_size: int = 200
    nlp_model: str = "en_core_web_trf"
    head: int | None = None

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.head is not None and self.head < 1:
            raise ValueError("head must be >= 1 when provided")
        self.input_text_table_path = Path(self.input_text_table_path).expanduser()
        self.kb_csv_path = Path(self.kb_csv_path).expanduser()


@dataclass
class ClusteringOptimizationConfig:
    """Configuration for clustering optimization grid search."""

    min_class_size: int = 20
    max_scale: int = 120
    rns: RandomState = field(default_factory=lambda: RandomState(seed=13))
    frac: float = 1.0
    head: int | None = None
    batch_size: int = 1000
    optimization_method: str = "mean"

    def __post_init__(self) -> None:
        if self.min_class_size < 1:
            raise ValueError("min_class_size must be >= 1")
        if self.max_scale < self.min_class_size:
            raise ValueError("max_scale must be >= min_class_size")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not 0 < self.frac <= 1:
            raise ValueError("frac must be in range (0, 1]")
        if self.head is not None and self.head < 1:
            raise ValueError("head must be >= 1 when provided")
        if not self.optimization_method:
            raise ValueError("optimization_method must be a non-empty string")

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
