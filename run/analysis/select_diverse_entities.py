import logging
import pathlib

import click
import pandas as pd
import torch
import spacy
import numpy as np
from sklearn.decomposition import PCA

from pelinker.model import Linker
from pelinker.ops import load_dataframe
from pelinker.util import load_models, embed_texts
from sklearn.cluster import KMeans

from pelinker.analysis import (
    embeddings_dict_to_dataframe,
    get_word_frequencies_from_library,
    compute_kb_generality_scores,
)

logger = logging.getLogger(__name__)


def select_diverse_entities(
    embeddings_dict: dict[str, tuple[str, torch.Tensor]],
    id_column: str,
    label_column: str,
    n_select: int = 10,
    *,
    pca_components: int = 20,
    random_state=12,
    metric: str = "cosine",
) -> pd.DataFrame:
    """
    Select semantically diverse entities using farthest point sampling.

    Uses weighted FPS to balance semantic diversity with preference for generic/simple terms.
    Can filter out very specific/technical terms before selection.

    Args:
        embeddings_dict: Dictionary mapping id -> (label, embedding)
        id_column: Name of column containing item IDs
        label_column: Name of column containing item labels
        n_select: Number of entities to select (default: 15)
        pca_components: Number of PCA components if use_pca=True
        metric: Distance metric ('cosine' or 'euclidean')
        random_state: Random seed for selecting first point (None = deterministic, uses first point)
        prefer_simple: If True, use weighted FPS that prefers simpler/generic labels

    Returns:
        DataFrame with selected entities (columns: id_column, label_column, and metadata)
    """
    logger.info(
        "Selecting %d diverse entities from %d candidates",
        n_select,
        len(embeddings_dict),
    )

    # Convert to dataframe
    df = embeddings_dict_to_dataframe(embeddings_dict)

    # Rename columns to match the actual column names
    df = df.rename(columns={"id": id_column, "label": label_column})

    # Get word frequencies for filtering and scoring
    word_frequencies = get_word_frequencies_from_library(language="en", wordlist="best")

    # Extract embeddings for KB-based analysis
    embeddings = np.stack(df["embed"].values)

    # Compute KB-based generality scores if requested
    logger.info("Computing KB-based generality scores (density + label simplicity)...")
    # Use raw embeddings for generality computation (before PCA)
    labels_list = df[label_column].astype(str).tolist()
    kb_generality_scores = compute_kb_generality_scores(
        embeddings,
        labels_list,
        k_neighbors=min(10, len(df) - 1),
        metric=metric,
        word_frequencies=word_frequencies,
        density_weight=0.4,
    )

    pca = PCA(n_components=min(pca_components, len(df) - 1))
    embeddings = pca.fit_transform(embeddings)
    logger.info(
        "PCA explained variance ratio: %.3f", pca.explained_variance_ratio_.sum()
    )

    preference_scores = kb_generality_scores

    kmeans = KMeans(n_clusters=n_select, random_state=random_state, n_init="auto").fit(
        embeddings
    )

    labels = kmeans.labels_

    dfw = df[[id_column, label_column]].copy()
    dfw["score"] = preference_scores
    dfw["icluster"] = labels
    df_selected = dfw.groupby("icluster").apply(lambda x: x.loc[x["score"].idxmax()])
    return df_selected


@click.command()
@click.option(
    "--input-table-path",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    help="Path to the dataframe to load (CSV/TSV, optionally gzipped).",
)
@click.option(
    "--label-column",
    type=click.STRING,
    default="label",
    show_default=True,
    help="Column containing the labels/phrases to embed.",
)
@click.option(
    "--id-column",
    type=click.STRING,
    default="entity_id",
    show_default=True,
    help="Column containing the IDs to use as dictionary keys.",
)
@click.option(
    "--n-select",
    type=click.INT,
    default=10,
    help="Number of diverse entities to select (10-20 recommended).",
)
@click.option(
    "--model-type",
    type=click.STRING,
    default="pubmedbert",
    help="Backbone model identifier passed to pelinker.util.load_models.",
)
@click.option(
    "--layers-spec",
    type=click.STRING,
    default="4",
    help="Layer spec string (digits for token layers).",
)
@click.option(
    "--use-gpu",
    is_flag=True,
    help="Move the encoder model to CUDA if available.",
)
@click.option(
    "--pca-components",
    type=click.INT,
    default=20,
    help="Number of PCA components for dimensionality reduction.",
)
@click.option(
    "--metric",
    type=click.Choice(["cosine", "euclidean"]),
    default="cosine",
    help="Distance metric for FPS.",
)
@click.option(
    "--random-state",
    type=click.INT,
    default=13,
    help="Random seed for selecting first point (None = deterministic, uses first point).",
)
@click.option(
    "--output-path",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    help="Path for saving selected entities CSV file.",
)
def run(
    input_table_path,
    label_column,
    id_column,
    n_select,
    model_type,
    layers_spec,
    use_gpu,
    pca_components,
    metric,
    random_state,
    output_path,
):
    """
    Select semantically diverse entities using farthest point sampling.

    This approach is deterministic and stable, unlike clustering-based methods.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    layers = Linker.str2layers(layers_spec)

    logger.info("Loading dataframe from %s", input_table_path)
    df = load_dataframe(input_table_path)

    if label_column not in df.columns:
        raise click.BadParameter(
            f"Column '{label_column}' not found in dataframe columns {list(df.columns)}",
            param_hint="--label-column",
        )

    if id_column not in df.columns:
        raise click.BadParameter(
            f"Column '{id_column}' not found in dataframe columns {list(df.columns)}",
            param_hint="--id-column",
        )

    tokenizer, model = load_models(model_type)

    if use_gpu:
        if torch.cuda.is_available():
            logger.info("Moving model to CUDA")
            model = model.to("cuda")
        else:
            logger.warning("CUDA not available, falling back to CPU")

    # Load spacy model for texts_to_vrep
    logger.info("Loading spaCy model")
    nlp = spacy.load("en_core_web_trf")

    # Filter rows where both id and label are not null
    df_filtered = df[[id_column, label_column]].dropna()
    if df_filtered.empty:
        logger.warning(
            "No rows with both '%s' and '%s' columns non-null", id_column, label_column
        )
        return

    # Filter out empty labels
    valid_mask = df_filtered[label_column].apply(
        lambda x: pd.notna(x) and str(x).strip() != ""
    )
    df_valid = df_filtered[valid_mask]

    if df_valid.empty:
        logger.warning("No rows with non-empty labels after filtering")
        return

    ids = df_valid[id_column].tolist()
    labels = df_valid[label_column].tolist()

    logger.info("Embedding %d labels...", len(labels))
    # Convert embeddings to list format
    text_embeddings = embed_texts(
        labels,
        tokenizer=tokenizer,
        model=model,
        layers=layers,
        nlp=nlp,
    )

    # Create dictionary mapping id -> (label, embedding)
    result = {}
    for id_val, label, emb in zip(ids, labels, text_embeddings):
        result[str(id_val)] = (str(label), emb)

    logger.info(
        "Embedded %d items from columns '%s' (labels) and '%s' (ids)",
        len(result),
        label_column,
        id_column,
    )

    # Select diverse entities
    selected_df = select_diverse_entities(
        result,
        id_column=id_column,
        label_column=label_column,
        n_select=n_select,
        pca_components=pca_components,
        metric=metric,
        random_state=random_state,
    )

    # Save results
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving selected entities to %s", output_path)
    selected_df.to_csv(output_path, index=False)
    logger.info("Saved %d diverse entities", len(selected_df))


if __name__ == "__main__":
    run()
