# pylint: disable=E1120

import click
import pathlib
import logging
import pandas as pd

from pelinker.config import (
    ClusteringOptimizationConfig,
    EmbeddingModelMetadata,
    EmbeddingTrainingConfig,
    TransformConfig,
)
from pelinker.model import Linker
from importlib.resources import files
from pelinker.util import str2layers, layers2str

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model-type",
    type=click.STRING,
    default="pubmedbert",
    help="run over BERT flavours",
)
@click.option(
    "--layers-spec",
    type=click.STRING,
    default="1",
    help="`sent` or a string of layers, `1,2,3` would correspond to layers [-1, -2, -3]",
)
@click.option(
    "--kb-path",
    type=click.Path(path_type=pathlib.Path),
    help="Path to KB CSV file.",
)
@click.option(
    "--pca-components",
    type=click.INT,
    default=100,
    help="Number of PCA components for dimensionality reduction",
)
@click.option(
    "--umap-dim",
    type=click.INT,
    default=8,
    help="UMAP dimensionality for clustering (range: 3-5 recommended)",
)
@click.option(
    "--min-class-size",
    type=click.INT,
    default=20,
    help="Minimum class size for filtering",
)
@click.option(
    "--max-scale",
    type=click.INT,
    default=120,
    help="Maximum value for grid evaluation of min_cluster_size",
)
@click.option(
    "--output-path",
    type=click.Path(path_type=pathlib.Path),
    help="Output path for saved model. If not provided, uses default in pelinker.store",
)
@click.option(
    "--input-text-table-path",
    type=click.Path(path_type=pathlib.Path),
    help="Input file (TSV/CSV, optionally gzipped) with pmid and text columns. If not provided, will be required.",
)
@click.option(
    "--use-gpu",
    is_flag=True,
    default=False,
    help="Enable GPU acceleration if CUDA is available",
)
@click.option(
    "--chunk-size",
    type=click.INT,
    default=1000,
    help="Chunk size for streaming input. Each chunk is split into batches and serialized.",
)
@click.option(
    "--batch-size",
    type=click.INT,
    default=200,
    help="Embedding batch size inside a chunk.",
)
@click.option(
    "--nlp-model",
    type=click.STRING,
    default="en_core_web_trf",
    help="spaCy model to use for tokenization/lemmas.",
)
@click.option(
    "--head",
    type=click.INT,
    default=None,
    help="Number of chunks to process (skip it for all chunks).",
)
@click.option(
    "--embeddings-path",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help="Optional path to pre-computed embeddings parquet file. If provided, skips embedding step (a) and goes directly to clustering (b).",
)
def fit(
    model_type: str,
    layers_spec: str,
    kb_path: pathlib.Path | None,
    pca_components: int,
    umap_dim: int,
    min_class_size: int,
    max_scale: int,
    output_path: pathlib.Path | None,
    input_text_table_path: pathlib.Path | None,
    use_gpu: bool,
    chunk_size: int,
    batch_size: int,
    nlp_model: str,
    head: int | None,
    embeddings_path: pathlib.Path | None,
):
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

    kb_path = kb_path.expanduser()
    logger.info(f"Using KB: {kb_path}")

    df0 = pd.read_csv(kb_path)

    logger.info(f"Loaded {len(df0)} properties from KB")

    # Extract property labels from KB for corpus embedding
    kb_labels = set(df0["label"].dropna().unique())

    logger.info(f"Extracted {len(kb_labels)} unique property labels from KB")

    # Create transform config
    transform_config = TransformConfig(
        pca_components=pca_components,
        umap_components=umap_dim,
    )

    if embeddings_path is None and input_text_table_path is None:
        raise ValueError(
            "Either --input-text-table-path or --embeddings-path must be provided"
        )

    # Create Linker model
    layers = str2layers(layers_spec)
    layers_str = layers2str(layers)

    embedding_metadata = EmbeddingModelMetadata.from_single(model_type, layers_spec)

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
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            batch_size=batch_size,
            nlp_model=nlp_model,
            head=head,
        )
    clustering_config = ClusteringOptimizationConfig(
        min_class_size=min_class_size,
        max_scale=max_scale,
        batch_size=chunk_size,
    )

    # Fit the linker - this handles embedding (if needed), loading, filtering, aggregation, and clustering
    linker.fit(
        embeddings=embeddings_path,
        transform_config=transform_config,
        min_cluster_size=min_class_size,
        kb_labels=kb_labels,
        optimize_clustering=True,
        clustering_optimization_config=clustering_config,
        embedding_training=embedding_training,
    )

    logger.info(f"Fitted Linker model with {len(linker.vocabulary)} entities")
    logger.info(f"Number of clusters: {len(set(linker.cluster_assignments.values()))}")

    # Save model
    if output_path is None:
        file_spec = files("pelinker.store").joinpath(
            f"pelinker.model.{model_type}.{layers_str}"
        )
    else:
        file_spec = output_path.expanduser()

    logger.info(f"Saving model to {file_spec}")
    linker.dump(file_spec)

    logger.info("Model saved successfully!")


if __name__ == "__main__":
    fit()
