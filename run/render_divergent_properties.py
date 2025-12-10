import json
import logging
import pathlib

import click
import pandas as pd
import torch
import spacy

from pelinker.model import LinkerModel
from pelinker.ops import load_dataframe
from pelinker.util import load_models, embed_texts
from pelinker.analysis import (
    umap_it,
    cluster_with_target_count,
    find_cluster_centers,
    embeddings_dict_to_dataframe,
    get_word_frequencies_from_library,
    compute_silhouette_after_filtering,
)
from pelinker.reporting import log_clustering_scores, log_clustering_results

logger = logging.getLogger(__name__)


def _perform_clustering_analysis(
    embeddings_dict: dict[str, tuple[str, torch.Tensor]],
    id_column: str,
    label_column: str,
    target_cluster_counts: list[int] | None = None,
    umap_dim: int = 15,
    min_cluster_size: int = 5,
    max_chars: int | None = 30,
    max_words: int | None = 4,
    min_simplicity_score: float | None = 5e-8,
    results_dir: pathlib.Path | None = None,
    sem_div_kb_path: pathlib.Path | None = None,
) -> dict:
    """
    Perform clustering analysis on embeddings and output results for multiple cluster counts.

    Args:
        embeddings_dict: Dictionary mapping id -> (label, embedding)
        logger: Logger instance for output
        target_cluster_counts: List of target cluster counts to evaluate (default: [5, 10, 15, 20])
        umap_dim: UMAP dimensionality (default: 15)
        min_cluster_size: Minimum cluster size for filtering
        max_chars: Maximum characters for simplest example
        max_words: Maximum words for simplest example
        min_simplicity_score: Minimum simplicity score threshold
        results_dir: Output directory for saving JSON report (None = don't save)
        sem_div_kb_path: Path for saving reduced KB dataframe (None = don't save)

    Returns:
        Dictionary with report data
    """
    if target_cluster_counts is None:
        target_cluster_counts = [5, 10, 15, 20]

    logger.info("Performing clustering analysis...")

    # Convert embeddings to dataframe format
    df_emb = embeddings_dict_to_dataframe(embeddings_dict)

    # Apply UMAP reduction
    logger.info("Applying UMAP reduction to %d dimensions...", umap_dim)
    df_umap = umap_it(df_emb, umap_dim=umap_dim)
    umap_columns = [f"u_{j:02d}" for j in range(umap_dim)]

    # Get word frequencies from external library for better simplicity scoring
    logger.info("Loading word frequencies from wordfreq library...")
    word_frequencies = get_word_frequencies_from_library(language="en", wordlist="best")
    if word_frequencies is None:
        logger.warning(
            "wordfreq library not available. Install with: pip install wordfreq\n"
            "Falling back to word length-based simplicity scoring."
        )
    else:
        logger.info("Using wordfreq library for frequency lookups")

    logger.info(
        "Filtering clusters: min_size=%d, max_chars=%s, max_words=%s, min_simplicity=%.2e",
        min_cluster_size,
        max_chars if max_chars else "unlimited",
        max_words if max_words else "unlimited",
        min_simplicity_score if min_simplicity_score else 0.0,
    )

    # Compute clustering scores for each target count
    logger.info(
        "Computing clustering scores for target counts: %s", target_cluster_counts
    )
    results = []

    for target_count in target_cluster_counts:
        logger.info("Clustering for target count: %d...", target_count)
        df_clustered, actual_count, score = cluster_with_target_count(
            df_umap, umap_columns, target_count
        )

        # Find valid clusters after filtering
        cluster_results = find_cluster_centers(
            df_clustered,
            umap_columns,
            method="simplest",
            min_cluster_size=min_cluster_size,
            max_complexity_chars=max_chars,
            max_complexity_words=max_words,
            min_simplicity_score=min_simplicity_score,
            word_frequencies=word_frequencies,
        )
        valid_cluster_ids = {cr["cluster_id"] for cr in cluster_results}

        # Compute silhouette score after filtering
        filtered_score = compute_silhouette_after_filtering(
            df_clustered, umap_columns, valid_cluster_ids
        )

        results.append(
            {
                "target_count": target_count,
                "actual_count": actual_count,
                "score_before_filtering": score,
                "score_after_filtering": filtered_score,
                "n_valid_clusters": len(valid_cluster_ids),
                "df": df_clustered,
                "cluster_results": cluster_results,
            }
        )
        logger.info(
            "  Target: %d, Actual: %d, Before filtering: %.4f, After filtering: %.4f, Valid clusters: %d",
            target_count,
            actual_count,
            score,
            filtered_score,
            len(valid_cluster_ids),
        )

    # Output summary table
    log_clustering_scores(results, logger)

    # Output detailed results for the best scoring clustering (or 15 clusters if available)
    best_result = max(results, key=lambda x: x["score_after_filtering"])
    default_result = next((r for r in results if r["target_count"] == 15), best_result)

    logger.info(
        "\nShowing cluster details for %d clusters (score after filtering: %.4f)...",
        default_result["target_count"],
        default_result["score_after_filtering"],
    )

    logger.info("\n--- Cluster representatives (simplest examples) ---")
    log_clustering_results(default_result["cluster_results"], logger)

    # Build report
    report = {
        "clustering_results": [
            {
                "target_count": r["target_count"],
                "actual_count": r["actual_count"],
                "score_before_filtering": r["score_before_filtering"],
                "score_after_filtering": r["score_after_filtering"],
                "n_valid_clusters": r["n_valid_clusters"],
            }
            for r in results
        ],
        "best_result": {
            "target_count": default_result["target_count"],
            "actual_count": default_result["actual_count"],
            "score_before_filtering": default_result["score_before_filtering"],
            "score_after_filtering": default_result["score_after_filtering"],
            "n_valid_clusters": default_result["n_valid_clusters"],
            "clusters": default_result["cluster_results"],
        },
        "filtering_parameters": {
            "min_cluster_size": min_cluster_size,
            "max_chars": max_chars,
            "max_words": max_words,
            "min_simplicity_score": min_simplicity_score,
        },
    }

    # Build dataframe with selected labels and IDs
    selected_data = []
    for cluster_result in default_result["cluster_results"]:
        selected_data.append(
            {
                id_column: cluster_result["center_id"],
                label_column: cluster_result["center_label"],
                "cluster_id": cluster_result["cluster_id"],
                "cluster_size": cluster_result["cluster_size"],
            }
        )

    df_selected = pd.DataFrame(selected_data)

    # Save JSON report
    if results_dir:
        results_dir = results_dir.expanduser()
        results_dir.mkdir(parents=True, exist_ok=True)
        json_path = results_dir / "semantic_divergent_clustering_report.json"
        logger.info("Saving JSON report to %s", json_path)
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
    else:
        logger.info("No results directory specified, skipping JSON report output")

    # Save reduced KB dataframe
    if sem_div_kb_path:
        sem_div_kb_path = sem_div_kb_path.expanduser()
        sem_div_kb_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving reduced KB to %s", sem_div_kb_path)

        # Detect format from extension
        if sem_div_kb_path.suffix == ".csv":
            df_selected.to_csv(sem_div_kb_path, index=False)
        else:
            logger.warning(
                "Unknown file extension, defaulting to CSV format. "
                "Supported: .csv, .tsv, .parquet"
            )
            csv_path = sem_div_kb_path.with_suffix(".csv")
            df_selected.to_csv(csv_path, index=False)
            logger.info("Saved to %s instead", csv_path)
    else:
        logger.info("No reduced KB path specified, skipping reduced KB output")

    return {
        "report": report,
        "selected_dataframe": df_selected,
    }


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
    required=True,
    help="Column containing the labels/phrases to embed.",
)
@click.option(
    "--id-column",
    type=click.STRING,
    required=True,
    help="Column containing the IDs to use as dictionary keys.",
)
@click.option(
    "--model-type",
    type=click.STRING,
    default="biobert",
    show_default=True,
    help="Backbone model identifier passed to pelinker.util.load_models.",
)
@click.option(
    "--layers-spec",
    type=click.STRING,
    default="1",
    show_default=True,
    help="Layer spec string (digits for token layers).",
)
@click.option(
    "--use-gpu",
    is_flag=True,
    default=False,
    help="Move the encoder model to CUDA if available.",
)
@click.option(
    "--min-cluster-size",
    type=click.INT,
    default=5,
    show_default=True,
    help="Minimum number of members required in a cluster.",
)
@click.option(
    "--max-chars",
    type=click.INT,
    default=30,
    show_default=True,
    help="Maximum character count for simplest example in cluster.",
)
@click.option(
    "--max-words",
    type=click.INT,
    default=4,
    show_default=True,
    help="Maximum word count for simplest example in cluster.",
)
@click.option(
    "--min-simplicity-score",
    type=click.FLOAT,
    default=5e-8,
    show_default=True,
    help="Minimum simplicity score (based on word frequency harmonic mean).",
)
@click.option(
    "--results-dir",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help="Output directory for saving JSON clustering report.",
)
@click.option(
    "--sem-div-kb-path",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help="Path for saving semantically divergent KB dataframe (selected labels). Supports .csv",
)
def run(
    input_table_path,
    label_column,
    id_column,
    model_type,
    layers_spec,
    use_gpu,
    min_cluster_size,
    max_chars,
    max_words,
    min_simplicity_score,
    results_dir,
    sem_div_kb_path,
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    layers = LinkerModel.str2layers(layers_spec)

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
        return {}

    # Filter out empty labels and keep track of valid indices
    valid_mask = df_filtered[label_column].apply(
        lambda x: pd.notna(x) and str(x).strip() != ""
    )
    df_valid = df_filtered[valid_mask]

    if df_valid.empty:
        logger.warning("No rows with non-empty labels after filtering")
        return {}

    ids = df_valid[id_column].tolist()
    labels = df_valid[label_column].tolist()

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
    for idx, (id_val, (label, embedding)) in enumerate(result.items()):
        logger.info(
            "Sample #%d id='%s' label='%s' -> dim=%d",
            idx + 1,
            str(id_val)[:30],
            str(label)[:60],
            len(embedding),
        )
        if idx >= 2:
            break

    # Perform clustering
    _ = _perform_clustering_analysis(
        result,
        label_column=label_column,
        id_column=id_column,
        min_cluster_size=min_cluster_size,
        max_chars=max_chars,
        max_words=max_words,
        min_simplicity_score=min_simplicity_score,
        results_dir=results_dir,
        sem_div_kb_path=sem_div_kb_path,
    )


if __name__ == "__main__":
    run()
