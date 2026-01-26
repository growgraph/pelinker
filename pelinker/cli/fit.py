# pylint: disable=E1120

import click
import pathlib
import numpy as np
import pandas as pd
import logging
import tempfile
import pyarrow.parquet as pq

from pelinker.model import Linker
from pelinker.util import fetch_latest_kb
from pelinker.embedder import embed_kb_corpus
from pelinker.transform import TransformConfig
from pelinker.analysis import (
    evaluate_cluster_size_grid,
    get_umap_columns,
)
from importlib.resources import files
from pathlib import Path

logger = logging.getLogger(__name__)


def fit_clustering_model(
    embeddings: np.ndarray,
    transform_config: TransformConfig,
    min_class_size: int = 20,
    max_scale: int = 120,
) -> tuple[int, float]:
    """
    Fit clustering model and find optimal min_cluster_size.

    This follows the logic from clustering_quality.py.

    Args:
        embeddings: Array of shape (n_samples, n_features)
        transform_config: TransformConfig instance
        min_class_size: Minimum class size for filtering
        max_scale: Maximum value for grid evaluation

    Returns:
        Tuple of (best_min_cluster_size, best_score)
    """
    from pelinker.transform import EmbeddingTransformer

    # Transform embeddings
    transformer = EmbeddingTransformer(transform_config)
    umap_clustering, _ = transformer.fit_transform(embeddings)

    # Get UMAP columns for grid evaluation
    umap_columns = get_umap_columns(transform_config)

    # Create DataFrame for grid evaluation
    df_umap = pd.DataFrame(
        umap_clustering,
        columns=[f"u_{j:02d}" for j in range(transform_config.umap_components)],
    )

    # Grid evaluation
    sizes = list(np.arange(int(0.5 * min_class_size), max_scale, 5))
    metrics_df = evaluate_cluster_size_grid(df_umap, umap_columns, sizes)

    if len(metrics_df) == 0:
        logger.warning(
            "No valid clusters found in grid evaluation, using default min_cluster_size"
        )
        return min_class_size, 0.0

    # Find optimal cluster size
    best_idx = metrics_df["dbcv"].idxmax()
    best_size = int(metrics_df.loc[best_idx, "min_cluster_size"])
    best_score = float(metrics_df.loc[best_idx, "dbcv"])

    return best_size, best_score


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
    default="sent",
    help="`sent` or a string of layers, `1,2,3` would correspond to layers [-1, -2, -3]",
)
@click.option(
    "--kb-path",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help="Path to KB CSV file. If not provided, uses latest from data/derived/",
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
    default=None,
    help="Output path for saved model. If not provided, uses default in pelinker.store",
)
@click.option(
    "--input-text-table-path",
    type=click.Path(path_type=pathlib.Path),
    default=None,
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
def run(
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
):
    """
    Fit a Linker model by embedding corpus and selecting KB entities.

    This follows the logic of:
    1. embed_kb_corpus.py - for embedding the whole corpus
    2. Filtering to only KB entities and aggregating embeddings per property
    3. clustering_quality.py - for creating clustering model
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Validate input_text_table_path
    if input_text_table_path is None:
        logger.error("--input-text-table-path is required")
        raise ValueError("--input-text-table-path is required")
    input_text_table_path = input_text_table_path.expanduser()

    # Load KB
    if kb_path is None:
        path_derived = Path("./data/derived/")
        fname, version = fetch_latest_kb(path_derived)
        if fname is None:
            logger.error(f"KB not found at {path_derived}")
            raise FileNotFoundError(f"KB not found at {path_derived}")
        kb_path = path_derived / fname
        logger.info(f"Using KB: {kb_path} (version {version})")
    else:
        kb_path = kb_path.expanduser()
        logger.info(f"Using KB: {kb_path}")

    try:
        df0 = pd.read_csv(kb_path)
    except Exception as e:
        logger.error(f"Failed to load KB from {kb_path}: {e}")
        raise

    logger.info(f"Loaded {len(df0)} properties from KB")

    # Extract property labels from KB for corpus embedding
    kb_labels = set(df0["label"].dropna().unique())
    property_label_map = dict(df0[["entity_id", "label"]].values)
    label_to_entity_id = {
        label: entity_id for entity_id, label in property_label_map.items()
    }

    logger.info(f"Extracted {len(kb_labels)} unique property labels from KB")

    # Create temporary properties.txt file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        properties_txt_path = Path(f.name)
        for label in sorted(kb_labels):
            f.write(f"{label}\n")
    logger.info(f"Created temporary properties file: {properties_txt_path}")

    # Create temporary output parquet path
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        output_parquet_path = Path(f.name)

    try:
        # Embed corpus using embed_kb_corpus
        logger.info("Embedding corpus...")
        embed_kb_corpus(
            model_type=model_type,
            layers_spec=layers_spec,
            input_text_table_path=input_text_table_path,
            properties_txt_path=properties_txt_path,
            output_parquet_path=output_parquet_path,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            batch_size=batch_size,
            nlp_model=nlp_model,
            head=head,
        )

        # Read parquet output and filter to KB entities
        logger.info("Reading embedded corpus and filtering to KB entities...")
        parquet_table = pq.read_table(output_parquet_path)
        df_parquet = parquet_table.to_pandas()

        # Filter to only KB properties
        df_kb_filtered = df_parquet[df_parquet["property"].isin(kb_labels)].copy()
        logger.info(
            f"Filtered to {len(df_kb_filtered)} mentions from {len(df_parquet)} total mentions"
        )

        if len(df_kb_filtered) == 0:
            logger.error("No mentions found for KB properties in corpus")
            raise ValueError("No mentions found for KB properties in corpus")

        # Aggregate embeddings per property (average)
        logger.info("Aggregating embeddings per property...")
        # Convert embed lists to numpy arrays for aggregation
        df_kb_filtered["embed_array"] = df_kb_filtered["embed"].apply(np.array)

        # Group by property and average embeddings
        property_embeddings = {}
        for prop_label, group in df_kb_filtered.groupby("property"):
            embeddings_list = group["embed_array"].tolist()
            # Stack and average
            embeddings_array = np.stack(embeddings_list)
            avg_embedding = np.mean(embeddings_array, axis=0)
            property_embeddings[prop_label] = avg_embedding

        # Map property labels to entity_ids and create embeddings array
        entity_ids = []
        embeddings_list = []
        for prop_label in sorted(property_embeddings.keys()):
            if prop_label in label_to_entity_id:
                entity_ids.append(label_to_entity_id[prop_label])
                embeddings_list.append(property_embeddings[prop_label])

        if len(embeddings_list) == 0:
            logger.error("No valid embeddings after mapping to entity_ids")
            raise ValueError("No valid embeddings after mapping to entity_ids")

        embeddings = np.stack(embeddings_list)

        logger.info(
            f"Embedded {len(embeddings)} KB properties into {embeddings.shape[1]}-dimensional vectors"
        )

    finally:
        # Clean up temporary files
        try:
            properties_txt_path.unlink()
            logger.debug(f"Removed temporary properties file: {properties_txt_path}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary properties file: {e}")

        try:
            output_parquet_path.unlink()
            logger.debug(f"Removed temporary parquet file: {output_parquet_path}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary parquet file: {e}")

    # Create transform config
    transform_config = TransformConfig(
        pca_components=pca_components,
        umap_components=umap_dim,
    )

    # Fit clustering model
    logger.info("Fitting clustering model...")
    best_min_cluster_size, best_score = fit_clustering_model(
        embeddings,
        transform_config,
        min_class_size=min_class_size,
        max_scale=max_scale,
    )

    logger.info(
        f"Optimal min_cluster_size: {best_min_cluster_size} (score: {best_score:.3f})"
    )

    # Create Linker model
    layers = Linker.str2layers(layers_spec)
    layers_str = Linker.layers2str(layers)

    linker = Linker(
        vocabulary=entity_ids,
        layers=layers,
        labels_map=property_label_map,
        transform_config=transform_config,
    )

    # Fit the linker
    linker.fit(
        embeddings,
        transform_config=transform_config,
        min_cluster_size=best_min_cluster_size,
    )

    logger.info(f"Fitted Linker model with {len(entity_ids)} entities")
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
    run()
