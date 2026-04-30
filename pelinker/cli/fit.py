import logging
from dataclasses import dataclass
from datetime import date
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, cast

import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
from numpy.random import RandomState
from omegaconf import MISSING, OmegaConf

from pelinker.clustering_quality_checkpoint import combination_key_from_members
from pelinker.config import (
    ClusteringOptimizationConfig,
    EmbeddingModelMetadata,
    EmbeddingSourceSpec,
    EmbeddingTrainingConfig,
    KBConfig,
    TransformConfig,
)
from pelinker.embedder import embed_kb_corpus
from pelinker.model import Linker
from pelinker.util import expand_config_path, layers2str, str2layers

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
    min_class_size: int = 20
    max_scale: int = 120
    # None → max(1, min_class_size // 2) in ClusteringOptimizationConfig.resolved_min_scale
    min_scale: int | None = None
    clustering_grid_step: int = 5
    # Stage-B HDBSCAN hyperparameter chosen upstream (e.g. run/analysis/clustering_quality.py).
    # When set, fit uses this exact value unless optimize_clustering=True.
    min_cluster_size: int | None = None
    # If true, rerun grid search during fit and ignore ``min_cluster_size``.
    optimize_clustering: bool = False
    # Filesystem path for the saved model; None → bundled store filename under ``pelinker.store``.
    output_path: str | None = None
    # If set, clustering reports are written exactly here. If unset, defaults to
    # ``{base}/reports/{YYYY-MM-DD}_{model-abbrev}/`` with ``base`` = parent of
    # ``output_path`` when that is set, else the process working directory.
    clustering_report_dir: str | None = None
    # When True and reports dir is resolved, also gzip-pickle a JSON-serializable report dict.
    dump_clustering_report: bool = True
    embeddings_parquet: Any = MISSING
    input_text_table_path: str | None = None
    use_gpu: bool = False
    nlp_model: str = "en_core_web_trf"
    # Stage (A): text table I/O buffer rows, encoder batch size (GPU), optional cap on read passes.
    input_buffer_rows: int = 1000
    encoder_batch_size: int = 200
    max_input_buffers: int | None = None
    negatives_per_positive: float = 0.0
    negative_label: str = "__NEGATIVE__"
    negative_seed: int | None = None
    # Stage (B): parquet batching (``batch_size`` rows per read batch).
    # ``frac`` subsamples rows only for ``min_cluster_size`` grid search when ``optimize_clustering`` is on; final fit uses all prepared rows.
    frac: float = 1.0
    n_embedding_batches: int | None = None  # max read batches per parquet; None = all
    batch_size: int = 1000
    # RNG seed for grid-search row subsampling (``ClusteringOptimizationConfig.rns``); final fit is not subsampled.
    clustering_seed: int = 13
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


def _default_model_store_name(meta: EmbeddingModelMetadata) -> str:
    if len(meta.sources) == 1:
        s0 = meta.sources[0]
        return (
            f"pelinker.model.{s0.model_type}.{layers2str(str2layers(s0.layers_spec))}"
        )
    members = [(s.model_type, s.layers_spec) for s in meta.sources]
    key = combination_key_from_members(members)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
    return f"pelinker.model.{safe}"


def _embedding_model_report_abbrev(meta: EmbeddingModelMetadata) -> str:
    """Short, path-safe tag for default clustering report subdirs (aligned with store naming)."""
    if len(meta.sources) == 1:
        s0 = meta.sources[0]
        raw = f"{s0.model_type}.{layers2str(str2layers(s0.layers_spec))}"
    else:
        members = [(s.model_type, s.layers_spec) for s in meta.sources]
        raw = combination_key_from_members(members)
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in raw)


def _clustering_report_path_for_fit(
    *,
    explicit_dir: object,
    output_path: Path | None,
    embedding_metadata: EmbeddingModelMetadata,
    today: date | None = None,
) -> Path:
    """
    Where to write clustering optimization reports.

    - Non-empty ``explicit_dir`` (config ``clustering_report_dir``): that directory
      exactly (after expanduser / env vars).
    - Otherwise: ``{base}/reports/{iso-date}_{model-abbrev}/`` where ``base`` is
      ``output_path.parent`` if the model is saved to a path, else ``Path.cwd()``.
    """
    if explicit_dir is not None and explicit_dir is not MISSING:
        s = str(explicit_dir).strip()
        if s:
            cr = expand_config_path(s)
            if cr is None:
                raise ValueError(f"Invalid clustering_report_dir: {explicit_dir!r}")
            return Path(cr)
    day = today or date.today()
    run_tag = f"{day.isoformat()}_{_embedding_model_report_abbrev(embedding_metadata)}"
    base = Path(output_path).parent if output_path is not None else Path.cwd()
    return base / "reports" / run_tag


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
    Run embedding (optional), fit a ``Linker`` from parquet(s) (optional), and save the model (optional).

    - ``pipeline=auto``: if ``input_text_table_path`` is set, embed then fit (unless outputs exist);
      if unset, fit only from existing parquet(s).
    - ``pipeline=embed_only``: write parquet(s) only.
    - ``pipeline=fit_only``: load mention-level parquet(s), train like ``estimate_model_clustering`` / ``Linker.fit``, save.
    - ``pipeline=both``: require a text table; write parquet(s) then fit and save.

    Multiple values for ``embeddings_parquet`` fuse mention-level files in list order (inner join on
    pmid/entity/mention, same as ``estimate_model_clustering``). Set ``model_types`` /
    ``layers_specs`` (or scalars ``model_type`` / ``layers_spec``) so ``embedding_metadata.sources``
    matches that order. If a list field is omitted, each path may supply ``model_type`` and/or
    ``layers_spec`` via a matching filename stem (``..._<model>_<layers>.parquet``, as in
    ``run/loop.fit.sh``); otherwise the scalar defaults apply for that component.
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
    )

    input_text_table_path = expand_config_path(cfg.input_text_table_path)
    output_path = expand_config_path(cfg.output_path)

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

    clustering_config = ClusteringOptimizationConfig(
        min_class_size=cfg.min_class_size,
        max_scale=cfg.max_scale,
        min_scale=cfg.min_scale,
        clustering_grid_step=cfg.clustering_grid_step,
        rns=RandomState(cfg.clustering_seed),
        frac=cfg.frac,
        n_embedding_batches=cfg.n_embedding_batches,
        batch_size=cfg.batch_size,
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

    resolved_clustering_reports = _clustering_report_path_for_fit(
        explicit_dir=cfg.clustering_report_dir,
        output_path=output_path,
        embedding_metadata=embedding_metadata,
    )
    logger.info("Clustering reports directory: %s", resolved_clustering_reports)

    logger.info("Stage (B): Linker.fit from %s", embed_paths)
    if cfg.min_cluster_size is not None and cfg.min_cluster_size < 2:
        raise ValueError("min_cluster_size must be >= 2 when provided")

    if cfg.optimize_clustering and cfg.min_cluster_size is not None:
        logger.warning(
            "optimize_clustering=True: ignoring fixed min_cluster_size=%s",
            cfg.min_cluster_size,
        )

    effective_min_cluster_size = (
        None if cfg.optimize_clustering else cfg.min_cluster_size
    )

    linker.fit(
        embeddings=embed_paths if len(embed_paths) > 1 else embed_paths[0],
        transform_config=transform_config,
        min_cluster_size=effective_min_cluster_size,
        kb_labels=kb_labels,
        optimize_clustering=cfg.optimize_clustering,
        clustering_optimization_config=clustering_config,
        embedding_training=None,
        kb_config=kb_config,
        clustering_report_dir=resolved_clustering_reports,
        dump_clustering_report=cfg.dump_clustering_report,
    )

    logger.info("Fitted Linker model with %s entities", len(linker.vocabulary))
    logger.info(
        "Number of clusters: %s",
        len(set(linker.cluster_assignments.values())),
    )

    if output_path is None:
        file_spec = files("pelinker.store").joinpath(
            _default_model_store_name(embedding_metadata)
        )
    else:
        file_spec = output_path

    logger.info("Saving model to %s", file_spec)
    linker.dump(file_spec)

    logger.info("Model saved successfully!")


CONFIG_STORE = ConfigStore.instance()
CONFIG_STORE.store(name="fit_config", node=FitCliConfig)


@hydra.main(version_base=None, config_path="pkg://pelinker.conf", config_name="fit")
def run(cfg: FitCliConfig) -> None:
    logger.info("Running fit with config:\n%s", OmegaConf.to_yaml(cfg))
    fit(cfg)


if __name__ == "__main__":
    run()
