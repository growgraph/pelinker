import logging
from dataclasses import dataclass
from datetime import date
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, cast

import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
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
    output_path: str | None = None
    # Mention-level parquet path(s). Order matches ``model_types`` / ``layers_specs`` (or scalars broadcast).
    embeddings_parquet: Any = MISSING
    input_text_table_path: str | None = None
    use_gpu: bool = False
    nlp_model: str = "en_core_web_trf"
    # Stage (A): text table I/O buffer rows, encoder batch size (GPU), optional cap on read passes.
    input_buffer_rows: int = 1000
    encoder_batch_size: int = 200
    max_input_buffers: int | None = None
    # Stage (B): sampling / parquet read batching (``batch_size`` = rows per embedding file batch).
    frac: float = 1.0
    head: int | None = None
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


def _broadcast_specs(
    model_type: str,
    layers_spec: str,
    model_types: list[str] | None,
    layers_specs: list[str] | None,
    n: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if n < 1:
        raise ValueError("At least one embeddings parquet path is required")
    if model_types is None:
        mts: tuple[str, ...] = (model_type,) * n
    elif len(model_types) == 1 and n > 1:
        mts = (model_types[0],) * n
    elif len(model_types) == n:
        mts = tuple(model_types)
    else:
        raise ValueError(
            f"model_types must have length 1 or {n} (one per parquet), got {len(model_types)}"
        )
    if layers_specs is None:
        lss: tuple[str, ...] = (layers_spec,) * n
    elif len(layers_specs) == 1 and n > 1:
        lss = (layers_specs[0],) * n
    elif len(layers_specs) == n:
        lss = tuple(layers_specs)
    else:
        raise ValueError(
            f"layers_specs must have length 1 or {n} (one per parquet), got {len(layers_specs)}"
        )
    return mts, lss


def _embedding_metadata(
    model_type: str,
    layers_spec: str,
    model_types: list[str] | None,
    layers_specs: list[str] | None,
    n: int,
) -> EmbeddingModelMetadata:
    mts, lss = _broadcast_specs(model_type, layers_spec, model_types, layers_specs, n)
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
    - ``pipeline=fit_only``: load parquet(s), fuse like ``clustering_quality`` / ``Linker.fit``, train, save.
    - ``pipeline=both``: require a text table; write parquet(s) then fit and save.

    Multiple values for ``embeddings_parquet`` fuse mention-level files in list order (see
    ``Linker.fit`` / ``fused_property_vectors_from_paths``); use ``model_types`` and
    ``layers_specs`` (or scalar ``model_type`` / ``layers_spec`` broadcast) so metadata matches.
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

    logger.info("Extracted %s unique property labels from KB", len(kb_labels))

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
        cfg.model_type, cfg.layers_spec, mts, lss, len(embed_paths)
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
        frac=cfg.frac,
        head=cfg.head,
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

    logger.info("Stage (B): Linker.fit from %s", embed_paths)
    linker.fit(
        embeddings=embed_paths if len(embed_paths) > 1 else embed_paths[0],
        transform_config=transform_config,
        min_cluster_size=cfg.min_class_size,
        kb_labels=kb_labels,
        optimize_clustering=True,
        clustering_optimization_config=clustering_config,
        embedding_training=None,
        kb_config=kb_config,
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
