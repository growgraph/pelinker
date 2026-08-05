import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, cast

import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from pelinker.config import (
    EmbeddingModelMetadata,
    EmbeddingSourceSpec,
    EmbeddingTrainingConfig,
    KBConfig,
    LinkerFitConfig,
    DistillationGateConfig,
    ManifoldOovScreenerConfig,
    NegativeScreenerConfig,
    TransformConfig,
)
from pelinker.embedder import embed_kb_corpus
from pelinker.model import Linker
from pelinker.cluster_composition_viz import (
    DEFAULT_MAX_CLUSTERS_FOR_PLOTS,
    build_cluster_composition_df,
    cluster_entity_mass_summary,
    cluster_score_percentile_summary,
    with_noise_cluster_label,
)
from pelinker.kb_out import KbOutNamingConfig, cluster_labels_from_catalog
from pelinker.reporting import (
    linker_fit_cluster_composition_path,
    linker_fit_clustering_report_path,
    linker_fit_kb_out_path,
    write_cluster_composition_json,
    write_clustering_report_json,
    write_kb_out_json,
)
from pelinker.onto import NEGATIVE_LABEL
from pelinker.util import expand_config_path

logger = logging.getLogger(__name__)

FitPipeline = Literal["auto", "embed_only", "fit_only", "both"]
_PIPELINE_VALUES: frozenset[str] = frozenset(("auto", "embed_only", "fit_only", "both"))

# Longest first so e.g. ``biobert`` does not steal a match from ``biobert-stsb``.
_KNOWN_EMBEDDING_MODEL_TYPES: tuple[str, ...] = (
    "biobert-stsb",
    "pubmedbert",
    "bluebert",
    "scibert",
    "biobert",
    "bert",
)


@dataclass
class FitCliConfig:
    """Hydra config for ``python -m pelinker.cli.fit``."""

    model_type: str = "pubmedbert"
    layers_spec: str = "1"
    kb_path: str = MISSING
    selection_report: str | None = None
    """``selected_hyperparameters.json`` (or the report dir containing it) from
    ``pelinker-model-selection`` / ``pelinker-dim-selection``.

    Fills in ``pca_components``, ``umap_dim``, ``umap_n_neighbors`` and
    ``min_cluster_size`` when those are not set explicitly here. Explicit overrides always
    win, and any disagreement is logged rather than silently resolved."""
    pca_components: int | None = None
    """PCA components; omit to take the selection report's value, else 100."""
    umap_dim: int | None = None
    """UMAP output dimension; omit to take the selection report's value, else 8."""
    umap_n_neighbors: int | None = None
    """UMAP ``n_neighbors``; omit for the library default (15). Scale-dependent — see
    ``pelinker-scale-curve`` when the fit N differs markedly from the selection N."""
    cluster_viz_method: str = "pca"
    drop_rare_entities: bool = False
    min_mentions_per_entity: int = 20
    max_mentions_per_entity: int | None = None
    max_mentions_negative: int | None = None
    mention_cap_seed: int | None = None
    """Seed for per-entity mention cap; defaults to ``seed`` when omitted."""
    seed: int = 13
    """Bootstrap seed for clustering subsample draws (``base_seed``); also default for mention-cap and screener draws."""
    pca_seed: int = 13
    """Random seed for PCA and cluster-viz PCA."""
    umap_seed: int | None = None
    """UMAP random seed; omit (None) for parallel UMAP. Set for reproducible production fits."""
    clustering_sample_rows: int | None = None
    """Max mention rows per clustering bootstrap draw (stratified). None = use all loaded rows."""
    clustering_sample_index: int = 0
    """Bootstrap index for clustering subsample (match model-selection ``sample_idx``)."""
    # Stage-B HDBSCAN ``min_cluster_size`` (choose upstream, e.g. ``pelinker.model_selection``).
    min_cluster_size: int | None = None
    """Explicit HDBSCAN ``min_cluster_size``. Omit to resolve from ``scale_curve_path``,
    or fall back to 20 when neither is given. An explicit value always wins."""
    scale_curve_path: str | None = None
    """``scale_curve.json`` from ``pelinker-scale-curve``. When set (and
    ``min_cluster_size`` is not), ``min_cluster_size`` is extrapolated to this fit's
    realized manifold row count instead of transferred verbatim from the selection run."""
    # Filesystem base path for ``Linker.dump`` (``.gz`` added by the linker).
    model_path: str | None = None
    # Directory for fit-time reports (``linker_fit.clustering_report.json``).
    report_path: str | None = None
    embeddings_parquet: Any = MISSING
    input_text_table_path: str | None = None
    use_gpu: bool = False
    nlp_model: str = "en_core_web_lg"
    # Stage (A): text table I/O buffer rows, encoder batch size (GPU), optional cap on read passes.
    input_buffer_rows: int = 1000
    encoder_batch_size: int = 200
    max_input_buffers: int | None = None
    negatives_per_positive: float = 0.0
    negative_label: str = NEGATIVE_LABEL
    negative_seed: int | None = None
    screener_kind: str = "lda"
    """``lda`` or ``svm``; persisted as :attr:`~pelinker.model.Linker.screener`."""
    projection_enabled: bool = True
    """When false, skip 3D manifold OOV score model (no predict-time gate from that path)."""
    # Stage (B): parquet batching (``batch_size`` rows per read batch).
    batch_size: int = 1000
    kb_name: str | None = None
    kb_version: str = "0.1.0"
    kb_created_at: str | None = None
    kb_description: str = ""
    kb_entity_count: int | None = None
    kb_out_name_min_fraction: float = 0.05
    kb_out_name_top_n: int = 3
    kb_out_ambiguity_min_capture: float = 0.10
    # Discriminator: auto = fit from parquet only if no text table; else embed then fit (legacy).
    # str (not Literal): OmegaConf structured configs reject Literal annotations on fields.
    pipeline: str = "embed_only"
    # Per-parquet backbone/layer (length 1 broadcast, or same length as ``embeddings_parquet``).
    # When omitted, ``model_type`` / ``layers_spec`` scalars apply unless the parquet stem matches
    # ``..._<model>_<layers>`` (see ``_parse_embedding_parquet_stem``).
    model_types: list[str] | None = None
    layers_specs: list[str] | None = None
    # Compact = ParametricUMAP + MLP entity head (default). Legacy = UMAP + HDBSCAN predict.
    predict_mode: str = "compact"
    entity_head_hidden_layers: list[int] | None = None
    """MLP hidden sizes for compact mode; default ``[256, 128, 128]`` when omitted."""
    entity_head_holdout_fraction: float = 0.15
    """Rows withheld from entity-head training to measure distillation fidelity.
    Set ``0.0`` to train on every row and skip the measurement (pre-fidelity behaviour)."""
    entity_head_holdout_group_col: str = "pmid"
    """Column kept whole across the holdout split, so correlated mentions from one
    document cannot straddle it and inflate the measured agreement."""
    entity_head_holdout_seed: int = 13
    distillation_gates_enabled: bool = True
    distillation_min_entity_agreement: float = 0.95
    distillation_max_emit_rate_rel_delta: float = 0.10
    distillation_emit_rate_threshold: float = 0.3
    distillation_on_failure: str = "warn"
    """``warn`` keeps the fitted model and records the failure; ``raise`` aborts the fit."""
    parametric_umap_n_training_epochs: int = 10
    parametric_umap_batch_size: int | None = None

    def __post_init__(self) -> None:
        if self.pipeline not in _PIPELINE_VALUES:
            raise ValueError(
                "pipeline must be one of "
                f"{sorted(_PIPELINE_VALUES)}, got {self.pipeline!r}"
            )
        if self.screener_kind not in ("lda", "svm"):
            raise ValueError(
                f"screener_kind must be 'lda' or 'svm', got {self.screener_kind!r}"
            )
        if self.cluster_viz_method not in ("pca", "umap"):
            raise ValueError(
                f"cluster_viz_method must be 'pca' or 'umap', got {self.cluster_viz_method!r}"
            )
        if self.predict_mode not in ("compact", "legacy"):
            raise ValueError(
                f"predict_mode must be 'compact' or 'legacy', got {self.predict_mode!r}"
            )
        if self.parametric_umap_n_training_epochs < 1:
            raise ValueError("parametric_umap_n_training_epochs must be >= 1")
        if (
            self.parametric_umap_batch_size is not None
            and self.parametric_umap_batch_size < 1
        ):
            raise ValueError("parametric_umap_batch_size must be >= 1 when provided")
        if self.entity_head_hidden_layers is not None:
            if not self.entity_head_hidden_layers or any(
                int(h) < 1 for h in self.entity_head_hidden_layers
            ):
                raise ValueError(
                    "entity_head_hidden_layers must be a non-empty list of ints >= 1"
                )
        if self.min_cluster_size is not None and self.min_cluster_size < 2:
            raise ValueError("min_cluster_size must be >= 2")
        if self.umap_n_neighbors is not None and self.umap_n_neighbors < 2:
            raise ValueError("umap_n_neighbors must be >= 2 when provided")
        if not 0.0 <= self.entity_head_holdout_fraction < 1.0:
            raise ValueError("entity_head_holdout_fraction must be in [0, 1)")
        if self.distillation_on_failure not in ("warn", "raise"):
            raise ValueError(
                "distillation_on_failure must be 'warn' or 'raise', "
                f"got {self.distillation_on_failure!r}"
            )
        if self.clustering_sample_rows is not None and self.clustering_sample_rows < 1:
            raise ValueError("clustering_sample_rows must be >= 1 when provided")
        if self.min_mentions_per_entity < 1:
            raise ValueError("min_mentions_per_entity must be >= 1")
        if (
            self.max_mentions_per_entity is not None
            and self.max_mentions_per_entity < 1
        ):
            raise ValueError("max_mentions_per_entity must be >= 1 when provided")
        if self.max_mentions_negative is not None and self.max_mentions_negative < 1:
            raise ValueError("max_mentions_negative must be >= 1 when provided")
        if self.clustering_sample_index < 0:
            raise ValueError("clustering_sample_index must be >= 0")
        if not 0.0 <= self.kb_out_name_min_fraction <= 1.0:
            raise ValueError("kb_out_name_min_fraction must be in [0, 1]")
        if self.kb_out_name_top_n < 1:
            raise ValueError("kb_out_name_top_n must be >= 1")
        if not 0.0 <= self.kb_out_ambiguity_min_capture <= 1.0:
            raise ValueError("kb_out_ambiguity_min_capture must be in [0, 1]")


def _coerce_str_list(val: object) -> list[str]:
    if val is None or val is MISSING:
        return []
    if isinstance(val, str):
        return [val]
    resolved = OmegaConf.to_container(val, resolve=True)
    if isinstance(resolved, list):
        return [str(x) for x in resolved]
    return [str(resolved)]


def _coerce_optional_str_list(val: object) -> list[str] | None:
    if val is None or val is MISSING:
        return None
    if isinstance(val, str):
        return [val]
    resolved = OmegaConf.to_container(val, resolve=True)
    if isinstance(resolved, list):
        out = [str(x) for x in resolved]
        return out if out else None
    return [str(resolved)]


def _parquet_stem_for_embedding_meta(path: Path) -> str:
    """Filename stem used to infer backbone/layers (supports ``.parquet`` and ``.parquet.gz``)."""
    name = path.name
    if name.endswith(".parquet.gz"):
        return name[: -len(".parquet.gz")]
    if name.endswith(".parquet"):
        return name[: -len(".parquet")]
    return path.stem


def _normalize_layers_filename_part(layer_part: str) -> str | None:
    """
    Map a filename layer segment to ``layers_spec``.

    Accepts a compact digit string (e.g. ``12`` → layers 1 and 2) or underscore-separated
    indices as produced by ``run/loop.fit.sh`` (``1_2_3`` → ``1,2,3``).
    """
    s = layer_part.strip()
    if not s:
        return None
    if "_" in s:
        parts = s.split("_")
        if not parts or not all(p.isdigit() for p in parts):
            return None
        return ",".join(parts)
    if s.isdigit():
        return s
    return None


def _parse_embedding_parquet_stem(stem: str) -> tuple[str, str] | None:
    """
    Parse ``<prefix>_<model>_<layers>`` or ``<model>_<layers>`` (``layers`` as in
    ``res_pubmedbert_1.parquet`` / ``res_pubmedbert_1_2_3.parquet``).
    """
    for mt in _KNOWN_EMBEDDING_MODEL_TYPES:
        needle = f"_{mt}_"
        idx = stem.rfind(needle)
        if idx >= 0:
            layer_part = stem[idx + len(needle) :]
            ls = _normalize_layers_filename_part(layer_part)
            if ls is not None:
                return mt, ls
        prefix = f"{mt}_"
        if stem.startswith(prefix):
            layer_part = stem[len(prefix) :]
            ls = _normalize_layers_filename_part(layer_part)
            if ls is not None:
                return mt, ls
    return None


def _parse_embedding_parquet_path(path: Path) -> tuple[str, str] | None:
    return _parse_embedding_parquet_stem(_parquet_stem_for_embedding_meta(path))


def _broadcast_model_types(
    model_type: str, model_types: list[str] | None, n: int
) -> tuple[str, ...]:
    if model_types is None:
        return (model_type,) * n
    if len(model_types) == 1 and n > 1:
        return (model_types[0],) * n
    if len(model_types) == n:
        return tuple(model_types)
    raise ValueError(
        f"model_types must have length 1 or {n} (one per parquet), got {len(model_types)}"
    )


def _broadcast_layers_specs(
    layers_spec: str, layers_specs: list[str] | None, n: int
) -> tuple[str, ...]:
    if layers_specs is None:
        return (layers_spec,) * n
    if len(layers_specs) == 1 and n > 1:
        return (layers_specs[0],) * n
    if len(layers_specs) == n:
        return tuple(layers_specs)
    raise ValueError(
        f"layers_specs must have length 1 or {n} (one per parquet), got {len(layers_specs)}"
    )


def _embedding_metadata(
    embed_paths: list[Path],
    model_type: str,
    layers_spec: str,
    model_types: list[str] | None,
    layers_specs: list[str] | None,
) -> EmbeddingModelMetadata:
    n = len(embed_paths)
    if n < 1:
        raise ValueError("At least one embeddings parquet path is required")

    parsed_per_path = [_parse_embedding_parquet_path(p) for p in embed_paths]

    if model_types is not None:
        mts = _broadcast_model_types(model_type, model_types, n)
    else:
        mts = tuple(pc[0] if pc is not None else model_type for pc in parsed_per_path)

    if layers_specs is not None:
        lss = _broadcast_layers_specs(layers_spec, layers_specs, n)
    else:
        lss = tuple(pc[1] if pc is not None else layers_spec for pc in parsed_per_path)

    for p, pc in zip(embed_paths, parsed_per_path):
        if pc is None:
            continue
        inferred_bits: list[str] = []
        if model_types is None:
            inferred_bits.append(f"model_type={pc[0]!r}")
        if layers_specs is None:
            inferred_bits.append(f"layers_spec={pc[1]!r}")
        if inferred_bits:
            logger.info(
                "Inferred %s from parquet filename %s",
                ", ".join(inferred_bits),
                p.name,
            )

    return EmbeddingModelMetadata(
        sources=tuple(EmbeddingSourceSpec(m, ls) for m, ls in zip(mts, lss))
    )


def _abort_if_outputs_exist(paths: list[Path], *, context: str) -> None:
    existing = [p for p in paths if p.is_file()]
    if not existing:
        return
    logger.warning(
        "%s: refusing to write — %s already exist(s): %s",
        context,
        "file" if len(existing) == 1 else "files",
        existing,
    )
    raise SystemExit(1)


def _load_kb_labels_map(cfg: FitCliConfig) -> tuple[Path, dict[str, str], set[str]]:
    kb_path = expand_config_path(cfg.kb_path)
    if kb_path is None:
        raise ValueError("kb_path must be provided")
    logger.info("Using KB: %s", kb_path)

    df0 = pd.read_csv(kb_path)
    logger.info("Loaded %s properties from KB", len(df0))

    if "entity_id" not in df0.columns:
        raise ValueError(
            "KB CSV must contain an 'entity_id' column "
            "(see run/embed_kb_corpus --kb-csv-path)."
        )
    if "label" not in df0.columns:
        raise ValueError("KB CSV must contain a 'label' column.")

    labels_map: dict[str, str] = {
        str(eid): str(lbl)
        for eid, lbl in zip(df0["entity_id"], df0["label"])
        if pd.notna(lbl)
    }
    kb_labels = set(df0["label"].dropna().unique())
    logger.info("Extracted %s unique entity labels from KB", len(kb_labels))
    return kb_path, labels_map, kb_labels


def _resolve_fit_pipeline(
    cfg: FitCliConfig,
    *,
    input_text_table_path: Path | None,
    embed_paths: list[Path],
    model_path: Path | None,
    report_path_resolved: Path | None,
) -> FitPipeline:
    pipeline = cfg.pipeline
    if pipeline == "auto":
        effective: FitPipeline = "fit_only" if input_text_table_path is None else "both"
    else:
        effective = cast(FitPipeline, pipeline)

    if effective == "fit_only" and input_text_table_path is not None:
        raise ValueError(
            "pipeline=fit_only (or auto with no text table): omit input_text_table_path."
        )
    if effective in ("both", "embed_only") and input_text_table_path is None:
        raise ValueError(
            f"pipeline={effective} requires input_text_table_path for stage (A)."
        )

    if effective in ("fit_only", "both"):
        if model_path is None:
            raise ValueError(
                "model_path is required for pipeline fit_only, both, or auto when fitting"
            )
        if report_path_resolved is None:
            raise ValueError(
                "report_path is required for pipeline fit_only, both, or auto when fitting"
            )

    if effective == "fit_only":
        missing = [p for p in embed_paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Embedding parquet(s) not found for fit_only: {missing}"
            )

    if effective == "both":
        _abort_if_outputs_exist(
            embed_paths,
            context="pipeline=both: embeddings_parquet target(s)",
        )
    elif effective == "embed_only":
        _abort_if_outputs_exist(
            embed_paths,
            context="pipeline=embed_only",
        )
    return effective


def _run_embed_stage(
    cfg: FitCliConfig,
    *,
    effective: FitPipeline,
    input_text_table_path: Path,
    kb_path: Path,
    embed_paths: list[Path],
    embedding_metadata: EmbeddingModelMetadata,
) -> None:
    if effective not in ("both", "embed_only"):
        return
    logger.info(
        "Stage (A): embed_kb_corpus → %s",
        embed_paths if len(embed_paths) > 1 else embed_paths[0],
    )
    training = EmbeddingTrainingConfig(
        input_text_table_path=input_text_table_path,
        kb_csv_path=kb_path,
        use_gpu=cfg.use_gpu,
        input_buffer_rows=cfg.input_buffer_rows,
        encoder_batch_size=cfg.encoder_batch_size,
        nlp_model=cfg.nlp_model,
        max_input_buffers=cfg.max_input_buffers,
        negatives_per_positive=cfg.negatives_per_positive,
        negative_label=cfg.negative_label,
        negative_seed=cfg.negative_seed,
    )
    if len(embed_paths) == 1:
        embed_kb_corpus(
            metadata=embedding_metadata,
            training=training,
            output_parquet_path=embed_paths[0],
        )
    else:
        embed_kb_corpus(
            metadata=embedding_metadata,
            training=training,
            output_parquet_paths=tuple(embed_paths),
        )


DEFAULT_PCA_COMPONENTS = 100
DEFAULT_UMAP_DIM = 8


@dataclass(frozen=True)
class _ResolvedSelection:
    """Transform dims after merging explicit overrides with a selection report."""

    pca_components: int
    umap_dim: int
    umap_n_neighbors: int | None
    min_cluster_size: int | None


def _resolve_selection_hyperparameters(cfg: FitCliConfig) -> _ResolvedSelection:
    """Merge ``selection_report`` into the transform dims; explicit values win.

    Every substitution and every disagreement is logged, so a fit never silently runs on
    dims that differ from the ones the search actually chose.
    """
    selected = None
    if cfg.selection_report:
        from pelinker.selected_hyperparameters import load_selected_hyperparameters

        path = expand_config_path(cfg.selection_report)
        assert path is not None
        selected = load_selected_hyperparameters(path)
        logger.info(
            "Loaded selection report from %s (%s: %s/%s, pca=%d, umap=%d, mcs=%d, "
            "manifold=%s, N=%s)",
            path,
            selected.source,
            selected.model,
            selected.layer,
            selected.pca_components,
            selected.umap_dim,
            selected.min_cluster_size,
            selected.manifold_kind,
            selected.n_rows_realized,
        )
        implied = "parametric" if cfg.predict_mode == "compact" else "umap"
        if selected.manifold_kind != implied:
            logger.warning(
                "Selection ran on manifold_kind=%r but predict_mode=%r implies %r. "
                "min_cluster_size=%d was chosen on different coordinates than this fit "
                "will use; re-run the search with --manifold-kind %s to align them.",
                selected.manifold_kind,
                cfg.predict_mode,
                implied,
                selected.min_cluster_size,
                implied,
            )

    def _pick(name: str, explicit, from_report, fallback):
        if explicit is not None:
            if from_report is not None and from_report != explicit:
                logger.info(
                    "%s=%s set explicitly, overriding selection report value %s",
                    name,
                    explicit,
                    from_report,
                )
            return explicit
        if from_report is not None:
            return from_report
        return fallback

    return _ResolvedSelection(
        pca_components=_pick(
            "pca_components",
            cfg.pca_components,
            None if selected is None else selected.pca_components,
            DEFAULT_PCA_COMPONENTS,
        ),
        umap_dim=_pick(
            "umap_dim",
            cfg.umap_dim,
            None if selected is None else selected.umap_dim,
            DEFAULT_UMAP_DIM,
        ),
        umap_n_neighbors=_pick(
            "umap_n_neighbors",
            cfg.umap_n_neighbors,
            None if selected is None else selected.umap_n_neighbors,
            None,
        ),
        min_cluster_size=_pick(
            "min_cluster_size",
            cfg.min_cluster_size,
            None if selected is None else selected.min_cluster_size,
            None,
        ),
    )


def _build_linker_fit_config(cfg: FitCliConfig) -> LinkerFitConfig:
    cap_seed = cfg.seed if cfg.mention_cap_seed is None else cfg.mention_cap_seed
    scale_curve = None
    if cfg.scale_curve_path:
        # Imported lazily: only fits that opt into the curve pay for the import.
        from pelinker.scale_curve import load_scale_curve

        curve_path = expand_config_path(cfg.scale_curve_path)
        assert curve_path is not None
        scale_curve = load_scale_curve(curve_path)
        logger.info(
            "Loaded scale curve from %s (%d rungs, slope=%.3f, R2=%.3f)",
            curve_path,
            scale_curve.n_rungs,
            scale_curve.log_slope,
            scale_curve.r_squared,
        )
    hidden = (
        tuple(int(h) for h in cfg.entity_head_hidden_layers)
        if cfg.entity_head_hidden_layers is not None
        else (256, 128, 128)
    )
    return LinkerFitConfig(
        batch_size=cfg.batch_size,
        drop_rare_entities=cfg.drop_rare_entities,
        min_mentions_per_entity=cfg.min_mentions_per_entity,
        max_mentions_per_entity=cfg.max_mentions_per_entity,
        max_mentions_negative=cfg.max_mentions_negative,
        mention_cap_seed=cap_seed,
        clustering_sample_rows=cfg.clustering_sample_rows,
        base_seed=cfg.seed,
        clustering_sample_index=cfg.clustering_sample_index,
        screener_seed=cfg.seed,
        ambient_screener=NegativeScreenerConfig(
            kind=cfg.screener_kind,
            negative_label=cfg.negative_label,
        ),
        projection_screener=ManifoldOovScreenerConfig(
            enabled=cfg.projection_enabled,
        ),
        predict_mode=cfg.predict_mode,  # type: ignore[arg-type]
        entity_head_hidden_layers=hidden,
        entity_head_holdout_fraction=cfg.entity_head_holdout_fraction,
        entity_head_holdout_group_col=cfg.entity_head_holdout_group_col,
        entity_head_holdout_seed=cfg.entity_head_holdout_seed,
        distillation_gates=DistillationGateConfig(
            enabled=cfg.distillation_gates_enabled,
            min_entity_agreement=cfg.distillation_min_entity_agreement,
            max_emit_rate_rel_delta=cfg.distillation_max_emit_rate_rel_delta,
            emit_rate_threshold=cfg.distillation_emit_rate_threshold,
            on_failure=cfg.distillation_on_failure,  # type: ignore[arg-type]
        ),
        scale_curve=scale_curve,
    )


def _build_kb_config(cfg: FitCliConfig, kb_path: Path) -> KBConfig:
    kb_created = (
        date.fromisoformat(cfg.kb_created_at) if cfg.kb_created_at else date.today()
    )
    kb_display_name = (cfg.kb_name or "").strip() or kb_path.stem
    return KBConfig(
        name=kb_display_name,
        version=cfg.kb_version,
        created_at=kb_created,
        description=cfg.kb_description,
        entity_count=cfg.kb_entity_count,
    )


def _write_fit_outputs(
    linker: Linker,
    cfg: FitCliConfig,
    *,
    model_path: Path,
    report_path_resolved: Path,
) -> None:
    report_path_resolved.mkdir(parents=True, exist_ok=True)
    fit_report = linker.take_fit_clustering_report()
    if fit_report is None:
        raise RuntimeError("Linker.fit produced no clustering report to serialize")

    mass_summary: dict[str, Any] = dict(
        cluster_entity_mass_summary(fit_report.assignments)
    )
    score_pcts = cluster_score_percentile_summary(fit_report.assignments)
    mass_summary["cluster_score_percentiles"] = score_pcts
    logger.info(
        "HDBSCAN emergent clusters on clustering subsample: %s "
        "(comparable to model-selection report n_clusters_emergent at the same MCS); "
        "distinct cluster labels on full KB assignments after approximate_predict: %s "
        "(noise mentions=%s, noise fraction=%.3f)",
        fit_report.n_clusters_emergent,
        mass_summary["n_emergent_clusters"],
        mass_summary["n_noise_mentions"],
        mass_summary["noise_fraction"],
    )
    emerg_p50 = score_pcts["emergent"].get("p50")
    noise_p50 = score_pcts["noise"].get("p50")
    logger.info(
        "cluster_score percentiles (p50 emergent=%.3f, p50 noise=%.3f; full table in composition summary)",
        emerg_p50 if emerg_p50 == emerg_p50 else float("nan"),
        noise_p50 if noise_p50 == noise_p50 else float("nan"),
    )

    report_json = linker_fit_clustering_report_path(report_path_resolved)
    write_clustering_report_json(report_json, fit_report)
    logger.info("Wrote clustering report to %s", report_json)

    composition_df = build_cluster_composition_df(
        fit_report.assignments,
        top_n=3,
        weight_by_entity=True,
        exclude_noise=False,
        max_clusters=DEFAULT_MAX_CLUSTERS_FOR_PLOTS,
    )
    catalog_raw = linker.kb_out_catalog
    if catalog_raw is None:
        raise RuntimeError("Linker.fit produced no KB-out catalog")
    catalog = cast(dict[str, Any], catalog_raw)
    cluster_labels = with_noise_cluster_label(
        cluster_labels_from_catalog(catalog, label_kind="display")
    )
    if not composition_df.empty:
        composition_df = composition_df.copy()
        composition_df["cluster_label"] = (
            composition_df["cluster"]
            .astype(int)
            .map(lambda cid: cluster_labels.get(int(cid), str(cid)))
        )
    composition_json = linker_fit_cluster_composition_path(report_path_resolved)
    write_cluster_composition_json(
        composition_json,
        composition_df,
        top_n=3,
        exclude_noise=False,
        summary=mass_summary,
        max_clusters_in_rows=DEFAULT_MAX_CLUSTERS_FOR_PLOTS,
    )
    logger.info("Wrote cluster composition artifact to %s", composition_json)

    kb_out_path = linker_fit_kb_out_path(report_path_resolved)
    write_kb_out_json(kb_out_path, catalog)
    logger.info("Wrote KB-out catalog to %s", kb_out_path)

    logger.info("Saving model to %s", model_path)
    linker.dump(model_path)
    logger.info("Model saved successfully!")


def fit(cfg: FitCliConfig) -> None:
    """
    Run embedding (optional), fit a ``Linker`` from parquet(s) (optional), and write outputs.

    Paths (no implicit fallbacks — missing required paths raise):

    - ``embeddings_parquet``: output path(s) for ``embed_only`` / ``both`` stage (A), or input
      parquet(s) for ``fit_only`` / ``both`` stage (B).
    - ``report_path``: directory; fit stages write ``linker_fit.clustering_report.json.gz``,
      ``linker_fit.cluster_composition.json.gz``, and ``linker_fit.kb_out.json`` there.
    - ``model_path``: filesystem path passed to ``Linker.dump`` for fit stages.

    Pipelines:

    - ``pipeline=auto``: embed then fit if ``input_text_table_path`` is set; else fit from parquet.
    - ``pipeline=embed_only``: write parquet(s) only (``model_path`` / ``report_path`` not used).
    - ``pipeline=fit_only``: fit from existing parquet(s); requires ``model_path`` and ``report_path``.
    - ``pipeline=both``: text table + embed then fit; requires ``model_path`` and ``report_path``.

    Multiple ``embeddings_parquet`` values fuse in list order (inner join on pmid/entity/mention).
    Set ``model_types`` / ``layers_specs`` (or scalars) so ``embedding_metadata.sources`` matches;
    or infer ``model_type`` / ``layers_spec`` from each filename stem when lists are omitted.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    kb_path, labels_map, _kb_labels = _load_kb_labels_map(cfg)

    resolved = _resolve_selection_hyperparameters(cfg)

    transform_config = TransformConfig(
        pca_components=resolved.pca_components,
        umap_components=resolved.umap_dim,
        umap_n_neighbors=resolved.umap_n_neighbors,
        cluster_viz_method=cfg.cluster_viz_method,
        pca_seed=cfg.pca_seed,
        umap_seed=cfg.umap_seed,
        parametric_umap_n_training_epochs=cfg.parametric_umap_n_training_epochs,
        parametric_umap_batch_size=cfg.parametric_umap_batch_size,
    )

    input_text_table_path = expand_config_path(cfg.input_text_table_path)
    model_path = expand_config_path(cfg.model_path)
    report_path_resolved = expand_config_path(cfg.report_path)

    path_strs = _coerce_str_list(cfg.embeddings_parquet)
    if not path_strs:
        raise ValueError("embeddings_parquet must be one or more paths")

    embed_paths: list[Path] = []
    for s in path_strs:
        p = expand_config_path(s)
        if p is None:
            raise ValueError(f"Invalid embeddings path: {s!r}")
        embed_paths.append(p)

    mts = _coerce_optional_str_list(cfg.model_types)
    lss = _coerce_optional_str_list(cfg.layers_specs)
    embedding_metadata = _embedding_metadata(
        embed_paths, cfg.model_type, cfg.layers_spec, mts, lss
    )

    effective = _resolve_fit_pipeline(
        cfg,
        input_text_table_path=input_text_table_path,
        embed_paths=embed_paths,
        model_path=model_path,
        report_path_resolved=report_path_resolved,
    )

    if effective in ("both", "embed_only"):
        assert input_text_table_path is not None
        _run_embed_stage(
            cfg,
            effective=effective,
            input_text_table_path=input_text_table_path,
            kb_path=kb_path,
            embed_paths=embed_paths,
            embedding_metadata=embedding_metadata,
        )

    if effective == "embed_only":
        logger.info("Embed-only pipeline finished; not fitting or saving a linker.")
        return

    linker_fit_cfg = _build_linker_fit_config(cfg)
    kb_config = _build_kb_config(cfg, kb_path)
    kb_out_naming = KbOutNamingConfig(
        min_fraction=cfg.kb_out_name_min_fraction,
        top_n=cfg.kb_out_name_top_n,
        ambiguity_min_capture=cfg.kb_out_ambiguity_min_capture,
    )

    linker = Linker(
        labels_map=labels_map,
        transform_config=transform_config,
        embedding_metadata=embedding_metadata,
    )

    logger.info("Stage (B): Linker.fit from %s", embed_paths)

    linker.fit(
        embeddings=embed_paths if len(embed_paths) > 1 else embed_paths[0],
        transform_config=transform_config,
        min_cluster_size=resolved.min_cluster_size,
        fit_config=linker_fit_cfg,
        embedding_training=None,
        kb_config=kb_config,
        kb_out_naming=kb_out_naming,
        kb_in_labels_map_path=str(kb_path),
    )

    logger.info("Fitted Linker model with %s KB-out entities", len(linker.vocabulary))
    logger.info(
        "KB-out emergent clusters: %s distinct ids",
        len(linker.cluster_id_to_entity_id),
    )

    if model_path is None or report_path_resolved is None:
        raise ValueError("model_path and report_path must be set when fitting")

    _write_fit_outputs(
        linker,
        cfg,
        model_path=model_path,
        report_path_resolved=report_path_resolved,
    )


CONFIG_STORE = ConfigStore.instance()
CONFIG_STORE.store(name="fit_config", node=FitCliConfig)


@hydra.main(version_base=None, config_path="pkg://pelinker.conf", config_name="fit")
def run(cfg: FitCliConfig) -> None:
    logger.info("Running fit with config:\n%s", OmegaConf.to_yaml(cfg))
    fit(cfg)


if __name__ == "__main__":
    run()
