import logging
from dataclasses import dataclass
from datetime import date
from importlib.resources import files
from pathlib import Path

import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from pelinker.config import (
    ClusteringOptimizationConfig,
    EmbeddingModelMetadata,
    EmbeddingTrainingConfig,
    KBConfig,
    TransformConfig,
)
from pelinker.model import Linker
from pelinker.util import str2layers, layers2str

logger = logging.getLogger(__name__)


@dataclass
class FitCliConfig:
    model_type: str = "pubmedbert"
    layers_spec: str = "1"
    kb_path: str = MISSING
    pca_components: int = 100
    umap_dim: int = 8
    min_class_size: int = 20
    max_scale: int = 120
    output_path: str | None = None
    input_text_table_path: str | None = None
    use_gpu: bool = False
    chunk_size: int = 1000
    batch_size: int = 200
    nlp_model: str = "en_core_web_trf"
    head: int | None = None
    embeddings_path: str | None = None
    kb_name: str | None = None
    kb_version: str = "0.1.0"
    kb_created_at: str | None = None
    kb_description: str = ""
    kb_entity_count: int | None = None


def _to_path(path: str | None) -> Path | None:
    if path is None:
        return None
    return Path(path).expanduser()


def fit(cfg: FitCliConfig) -> None:
    """
    Fit a Linker model by embedding corpus and selecting KB entities.

    This follows the logic of:
    1. embed_kb_corpus.py - for embedding the whole corpus (step a)
    2. Filtering to only KB entities and aggregating embeddings per property
    3. clustering_quality.py - for creating clustering model (step b)

    If --embeddings-path is provided, step (a) is skipped and we go directly to step (b).
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    kb_path = _to_path(cfg.kb_path)
    if kb_path is None:
        raise ValueError("kb_path must be provided")
    logger.info(f"Using KB: {kb_path}")

    df0 = pd.read_csv(kb_path)

    logger.info(f"Loaded {len(df0)} properties from KB")

    # Extract property labels from KB for corpus embedding
    kb_labels = set(df0["label"].dropna().unique())

    logger.info(f"Extracted {len(kb_labels)} unique property labels from KB")

    # Create transform config
    transform_config = TransformConfig(
        pca_components=cfg.pca_components,
        umap_components=cfg.umap_dim,
    )

    input_text_table_path = _to_path(cfg.input_text_table_path)
    embeddings_path = _to_path(cfg.embeddings_path)
    output_path = _to_path(cfg.output_path)

    if embeddings_path is None and input_text_table_path is None:
        raise ValueError(
            "Either input_text_table_path or embeddings_path must be provided"
        )

    # Create Linker model
    layers = str2layers(cfg.layers_spec)
    layers_str = layers2str(layers)

    embedding_metadata = EmbeddingModelMetadata.from_single(
        cfg.model_type, cfg.layers_spec
    )

    # Initialize with empty vocabulary (will be set during fit)
    linker = Linker(
        layers=layers,
        transform_config=transform_config,
        embedding_metadata=embedding_metadata,
    )

    embedding_training: EmbeddingTrainingConfig | None = None
    if input_text_table_path is not None:
        embedding_training = EmbeddingTrainingConfig(
            input_text_table_path=input_text_table_path,
            kb_csv_path=kb_path,
            use_gpu=cfg.use_gpu,
            chunk_size=cfg.chunk_size,
            batch_size=cfg.batch_size,
            nlp_model=cfg.nlp_model,
            head=cfg.head,
        )
    clustering_config = ClusteringOptimizationConfig(
        min_class_size=cfg.min_class_size,
        max_scale=cfg.max_scale,
        batch_size=cfg.chunk_size,
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

    # Fit the linker - this handles embedding (if needed), loading, filtering, aggregation, and clustering
    linker.fit(
        embeddings=embeddings_path,
        transform_config=transform_config,
        min_cluster_size=cfg.min_class_size,
        kb_labels=kb_labels,
        optimize_clustering=True,
        clustering_optimization_config=clustering_config,
        embedding_training=embedding_training,
        kb_config=kb_config,
    )

    logger.info(f"Fitted Linker model with {len(linker.vocabulary)} entities")
    logger.info(f"Number of clusters: {len(set(linker.cluster_assignments.values()))}")

    # Save model
    if output_path is None:
        file_spec = files("pelinker.store").joinpath(
            f"pelinker.model.{cfg.model_type}.{layers_str}"
        )
    else:
        file_spec = output_path

    logger.info(f"Saving model to {file_spec}")
    linker.dump(file_spec)

    logger.info("Model saved successfully!")


CONFIG_STORE = ConfigStore.instance()
CONFIG_STORE.store(name="fit_config", node=FitCliConfig)


@hydra.main(version_base=None, config_path=None, config_name="fit_config")
def run(cfg: FitCliConfig) -> None:
    logger.info("Running fit with config:\n%s", OmegaConf.to_yaml(cfg))
    fit(cfg)


if __name__ == "__main__":
    run()
