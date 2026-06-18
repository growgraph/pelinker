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
    ManifoldOovScreenerConfig,
    NegativeScreenerConfig,
    TransformConfig,
)
from pelinker.embedder import embed_kb_corpus
from pelinker.model import Linker
from pelinker.cluster_composition_viz import (
    DEFAULT_MAX_CLUSTERS_FOR_PLOTS,
    build_cluster_composition_df,
    build_emergent_clusters_catalog,
    cluster_entity_mass_summary,
)
from pelinker.reporting import (
    linker_fit_cluster_composition_path,
    linker_fit_cluster_kb_path,
    linker_fit_clustering_report_path,
    linker_fit_emergent_clusters_path,
    write_cluster_composition_json,
    write_cluster_derived_labels_map_json,
    write_clustering_report_json,
    write_emergent_clusters_json,
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
    pca_components: int = 100
    umap_dim: int = 8
    cluster_viz_method: str = "pca"
    min_class_size: int = 20
    seed: int = 13
    """PCA/UMAP seed and ``LinkerFitConfig.base_seed`` for clustering subsample draws."""
    frac: float = 1.0
    """Stratified mention fraction for clustering; match model-selection ``--frac``."""
    eval_max_rows: int | None = None
    """Cap rows per clustering draw after ``frac``; None or 0 = frac only (no row cap)."""
    clustering_sample_index: int = 0
    """Bootstrap index for clustering subsample (match model-selection ``sample_idx``)."""
    # Stage-B HDBSCAN ``min_cluster_size`` (choose upstream, e.g. ``run/analysis/model_selection.py``).
    min_cluster_size: int = 20
    # Filesystem base path for ``Linker.dump`` (``.gz`` added by the linker).
    model_path: str | None = None
    # Directory for fit-time reports (``linker_fit.clustering_report.json``).
    report_path: str | None = None
    embeddings_parquet: Any = MISSING
    input_text_table_path: str | None = None
    use_gpu: bool = False
    nlp_model: str = "en_core_web_trf"
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
    n_embedding_batches: int | None = None  # max read batches per parquet; None = all
    batch_size: int = 1000
    kb_name: str | None = None
    kb_version: str = "0.1.0"
    kb_created_at: str | None = None
    kb_description: str = ""
    kb_entity_count: int | None = None
    # Discriminator: auto = fit from parquet only if no text table; else embed then fit (legacy).
    # str (not Literal): OmegaConf structured configs reject Literal annotations on fields.
    pipeline: str = "embed_only"
    # Per-parquet backbone/layer (length 1 broadcast, or same length as ``embeddings_parquet``).
    # When omitted, ``model_type`` / ``layers_spec`` scalars apply unless the parquet stem matches
    # ``..._<model>_<layers>`` (see ``_parse_embedding_parquet_stem``).
    model_types: list[str] | None = None
    layers_specs: list[str] | None = None

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
        if self.min_cluster_size < 2:
            raise ValueError("min_cluster_size must be >= 2")
        if not 0 < self.frac <= 1:
            raise ValueError("frac must be in range (0, 1]")
        if self.clustering_sample_index < 0:
            raise ValueError("clustering_sample_index must be >= 0")


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


def fit(cfg: FitCliConfig) -> None:
    """
    Run embedding (optional), fit a ``Linker`` from parquet(s) (optional), and write outputs.

    Paths (no implicit fallbacks — missing required paths raise):

    - ``embeddings_parquet``: output path(s) for ``embed_only`` / ``both`` stage (A), or input
      parquet(s) for ``fit_only`` / ``both`` stage (B).
    - ``report_path``: directory; fit stages write ``linker_fit.clustering_report.json.gz`` and
      ``linker_fit.cluster_composition.json.gz`` there (see
      :func:`pelinker.reporting.linker_fit_clustering_report_path` and
      :func:`pelinker.reporting.linker_fit_cluster_composition_path`).
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

    transform_config = TransformConfig(
        pca_components=cfg.pca_components,
        umap_components=cfg.umap_dim,
        cluster_viz_method=cfg.cluster_viz_method,
        seed=cfg.seed,
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

    if effective in ("both", "embed_only"):
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

    if effective == "embed_only":
        logger.info("Embed-only pipeline finished; not fitting or saving a linker.")
        return

    linker_fit_cfg = LinkerFitConfig(
        min_class_size=cfg.min_class_size,
        batch_size=cfg.batch_size,
        n_embedding_batches=cfg.n_embedding_batches,
        frac=cfg.frac,
        eval_max_rows=(
            None
            if cfg.eval_max_rows is None or cfg.eval_max_rows <= 0
            else int(cfg.eval_max_rows)
        ),
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
    )

    kb_created = (
        date.fromisoformat(cfg.kb_created_at) if cfg.kb_created_at else date.today()
    )
    kb_display_name = (cfg.kb_name or "").strip() or kb_path.stem
    kb_config = KBConfig(
        name=kb_display_name,
        version=cfg.kb_version,
        created_at=kb_created,
        description=cfg.kb_description,
        entity_count=cfg.kb_entity_count,
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
        min_cluster_size=cfg.min_cluster_size,
        fit_config=linker_fit_cfg,
        embedding_training=None,
        kb_config=kb_config,
    )

    logger.info("Fitted Linker model with %s entities", len(linker.vocabulary))
    logger.info(
        "Entity-level provisional clusters: %s distinct ids",
        len(set(linker.cluster_assignments.values())),
    )

    if model_path is None or report_path_resolved is None:
        raise ValueError("model_path and report_path must be set when fitting")

    report_path_resolved.mkdir(parents=True, exist_ok=True)
    fit_report = linker.take_fit_clustering_report()
    if fit_report is None:
        raise RuntimeError("Linker.fit produced no clustering report to serialize")

    mass_summary = cluster_entity_mass_summary(fit_report.assignments)
    logger.info(
        "Emergent HDBSCAN clusters: %s (report n_clusters_emergent=%s, "
        "noise mentions=%s, noise fraction=%.3f)",
        mass_summary["n_emergent_clusters"],
        fit_report.n_clusters_emergent,
        mass_summary["n_noise_mentions"],
        mass_summary["noise_fraction"],
    )

    report_json = linker_fit_clustering_report_path(report_path_resolved)
    write_clustering_report_json(report_json, fit_report)
    logger.info("Wrote clustering report to %s", report_json)

    composition_df = build_cluster_composition_df(
        fit_report.assignments,
        top_n=3,
        weight_by_entity=True,
        exclude_noise=True,
        max_clusters=DEFAULT_MAX_CLUSTERS_FOR_PLOTS,
    )
    composition_json = linker_fit_cluster_composition_path(report_path_resolved)
    write_cluster_composition_json(
        composition_json,
        composition_df,
        top_n=3,
        summary=mass_summary,
        max_clusters_in_rows=DEFAULT_MAX_CLUSTERS_FOR_PLOTS,
    )
    logger.info("Wrote cluster composition artifact to %s", composition_json)

    emergent_catalog = build_emergent_clusters_catalog(
        linker.cluster_composition,
        linker.cluster_consensus_names,
        fit_report.assignments,
        min_cluster_size=cfg.min_cluster_size,
    )
    emergent_path = linker_fit_emergent_clusters_path(report_path_resolved)
    write_emergent_clusters_json(emergent_path, emergent_catalog)
    logger.info("Wrote emergent cluster catalog to %s", emergent_path)

    cluster_kb_json = linker_fit_cluster_kb_path(report_path_resolved)
    write_cluster_derived_labels_map_json(
        cluster_kb_json, linker.cluster_derived_labels_map
    )
    logger.info("Wrote cluster-derived KB labels map to %s", cluster_kb_json)

    logger.info("Saving model to %s", model_path)
    linker.dump(model_path)

    logger.info("Model saved successfully!")


CONFIG_STORE = ConfigStore.instance()
CONFIG_STORE.store(name="fit_config", node=FitCliConfig)


@hydra.main(version_base=None, config_path="pkg://pelinker.conf", config_name="fit")
def run(cfg: FitCliConfig) -> None:
    logger.info("Running fit with config:\n%s", OmegaConf.to_yaml(cfg))
    fit(cfg)


if __name__ == "__main__":
    run()
