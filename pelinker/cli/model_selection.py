"""CLI for model-selection grid search over embedding combinations."""

from __future__ import annotations

import pathlib

import click

from pelinker.model_selection import run_model_selection
from pelinker.model_selection_checkpoint import DEFAULT_CHECKPOINT_NAME, RunMode
from pelinker.onto import NEGATIVE_LABEL
from pelinker.reporting import MODEL_SELECTION_RUN_REPORT_BASENAME


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--input-dir",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    help="Directory containing parquet files",
)
@click.option(
    "--report-path",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    help=(
        "Directory for all run outputs. Canonical artifact: "
        f"{MODEL_SELECTION_RUN_REPORT_BASENAME}."
    ),
)
@click.option(
    "--umap-dim",
    type=click.INT,
    default=8,
    help="UMAP dimensionality for clustering (range: 3-5)",
)
@click.option(
    "--pca-components",
    type=click.INT,
    default=100,
    help="Number of PCA components for dimensionality reduction",
)
@click.option(
    "--cluster-viz-method",
    type=click.Choice(["pca", "umap"], case_sensitive=False),
    default="pca",
    show_default=True,
    help="Reducer for cluster-space visualization (PCA or UMAP on clustering coords).",
)
@click.option(
    "--min-class-size",
    type=click.INT,
    default=20,
    help="Anchor for min_cluster_size grid (lower bound defaults to half this value).",
)
@click.option(
    "--seed",
    type=click.INT,
    default=13,
    help="Bootstrap seed for clustering subsample draws and mention-cap defaults.",
)
@click.option(
    "--pca-seed",
    type=click.INT,
    default=13,
    show_default=True,
    help="Random seed for PCA and cluster-viz PCA.",
)
@click.option(
    "--umap-seed",
    type=click.INT,
    default=None,
    help="UMAP random seed; omit for parallel UMAP (default). Set for reproducible runs.",
)
@click.option(
    "--clustering-sample-rows",
    type=click.INT,
    default=None,
    help="Max mention rows per clustering bootstrap draw (stratified). Omit to use all loaded rows.",
)
@click.option(
    "--drop-rare-entities/--no-drop-rare-entities",
    default=False,
    show_default=True,
    help="Drop KB entities with fewer than --min-mentions-per-entity rows.",
)
@click.option(
    "--min-mentions-per-entity",
    type=click.INT,
    default=20,
    show_default=True,
    help="Minimum mention rows per KB entity when --drop-rare-entities is set.",
)
@click.option(
    "--max-mentions-per-entity",
    type=click.INT,
    default=None,
    help="Cap mention rows per KB entity (seeded); omit for no cap.",
)
@click.option(
    "--max-mentions-negative",
    type=click.INT,
    default=None,
    help="Cap synthetic negative rows; omit to leave negatives uncapped.",
)
@click.option(
    "--mention-cap-seed",
    type=click.INT,
    default=None,
    help="Seed for per-entity mention cap draws (default: --seed).",
)
@click.option(
    "--batch-size",
    type=click.INT,
    default=1000,
    help="Rows per batch when reading mention-level embedding parquet files",
)
@click.option(
    "--prefix",
    type=click.STRING,
    default="res",
    help="Optional prefix for input embedding files to differentiate between models",
)
@click.option(
    "--n-sample",
    type=click.INT,
    default=1,
    help="Number of samples/runs per (model, layer) combination",
)
@click.option(
    "--selected-labels-kb-path",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help="Optional path to selected labels KB CSV file. If provided, clustering will only use labels from this KB.",
)
@click.option(
    "--max-scale",
    type=click.INT,
    default=60,
    show_default=True,
    help="Exclusive upper bound for grid evaluation of min_cluster_size (numpy.arange end).",
)
@click.option(
    "--min-scale",
    type=click.INT,
    default=None,
    help=(
        "Inclusive lower bound for min_cluster_size on the grid. "
        "Default: max(1, min_class_size // 2) (legacy: half of --min-class-size)."
    ),
)
@click.option(
    "--clustering-grid-step",
    type=click.INT,
    default=5,
    show_default=True,
    help="Step between consecutive min_cluster_size values on the optimization grid.",
)
@click.option(
    "--fusion-pairs",
    type=click.INT,
    default=5,
    show_default=True,
    help=(
        "After scoring single embeddings (DBCV), evaluate fused pairs: "
        "pick this many distinct pairs with highest sum of singleton DBCV. 0 disables."
    ),
)
@click.option(
    "--fusion-triples",
    type=click.INT,
    default=0,
    show_default=True,
    help=("Same as --fusion-pairs but for three-way fusions (costly). 0 disables."),
)
@click.option(
    "--resume/--no-resume",
    default=True,
    show_default=True,
    help=(
        "If the checkpoint file exists and matches the run fingerprint, skip completed work. "
        "If the file is missing, start fresh and create it. Use --no-resume to ignore an "
        "existing checkpoint and reinitialize (overwrites on save)."
    ),
)
@click.option(
    "--checkpoint-path",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help=f"Checkpoint JSON path (default: <report-path>/{DEFAULT_CHECKPOINT_NAME})",
)
@click.option(
    "--negative-label",
    type=str,
    default=NEGATIVE_LABEL,
    show_default=True,
    help="Entity label for synthetic negatives (must match embedding parquet).",
)
@click.option(
    "--screener-kind",
    type=click.Choice(["lda", "svm"]),
    default="lda",
    show_default=True,
    help="Estimator saved on Linker when fitting from this pipeline (analysis always logs both).",
)
@click.option(
    "--mode",
    type=click.Choice(["single", "fusion2", "fusion3", "all"]),
    default="all",
    show_default=True,
    help=(
        "single: only single-embedding combinations; fusion2/fusion3: only that fusion order "
        "(requires prior singleton scores in checkpoint); all: singletons then enabled fusions."
    ),
)
def main(
    input_dir: pathlib.Path,
    report_path: pathlib.Path,
    umap_dim: int,
    pca_components: int,
    cluster_viz_method: str,
    min_class_size: int,
    seed: int,
    pca_seed: int,
    umap_seed: int | None,
    clustering_sample_rows: int | None,
    drop_rare_entities: bool,
    min_mentions_per_entity: int,
    max_mentions_per_entity: int | None,
    max_mentions_negative: int | None,
    mention_cap_seed: int | None,
    batch_size: int,
    n_sample: int,
    prefix: str,
    selected_labels_kb_path: pathlib.Path | None,
    max_scale: int,
    min_scale: int | None,
    clustering_grid_step: int,
    fusion_pairs: int,
    fusion_triples: int,
    resume: bool,
    checkpoint_path: pathlib.Path | None,
    mode: RunMode,
    negative_label: str,
    screener_kind: str,
) -> None:
    run_model_selection(
        input_dir=input_dir,
        report_path=report_path,
        umap_dim=umap_dim,
        pca_components=pca_components,
        cluster_viz_method=cluster_viz_method,
        min_class_size=min_class_size,
        seed=seed,
        pca_seed=pca_seed,
        umap_seed=umap_seed,
        clustering_sample_rows=clustering_sample_rows,
        drop_rare_entities=drop_rare_entities,
        min_mentions_per_entity=min_mentions_per_entity,
        max_mentions_per_entity=max_mentions_per_entity,
        max_mentions_negative=max_mentions_negative,
        mention_cap_seed=seed if mention_cap_seed is None else mention_cap_seed,
        batch_size=batch_size,
        n_sample=n_sample,
        prefix=prefix,
        selected_labels_kb_path=selected_labels_kb_path,
        max_scale=max_scale,
        min_scale=min_scale,
        clustering_grid_step=clustering_grid_step,
        fusion_pairs=fusion_pairs,
        fusion_triples=fusion_triples,
        resume=resume,
        checkpoint_path=checkpoint_path,
        mode=mode,
        negative_label=negative_label,
        screener_kind=screener_kind,
    )
