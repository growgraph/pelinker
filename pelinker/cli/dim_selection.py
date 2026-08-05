"""CLI for PCA / UMAP dimension selection on one embedding combination."""

from __future__ import annotations

import pathlib

import click

from pelinker.dim_selection import run_dim_selection
from pelinker.dim_selection.checkpoint import DEFAULT_CHECKPOINT_NAME
from pelinker.dim_selection.grids import DEFAULT_PCA_GRID, DEFAULT_UMAP_GRID
from pelinker.onto import NEGATIVE_LABEL

_EPILOG = """
MCS = min_cluster_size (HDBSCAN hyperparameter on the inner grid).

Metrics (same two-level stack as model selection):

  Inner (choose MCS): grid_objective=dbcv_ari_mean_minmax —
    min-max normalize mean DBCV and mean ARI on the MCS curve, average,
    smooth, and pick the left plateau.

  Outer (rank pca × umap cells): at each cell's pooled MCS, combine mean
    DBCV and mean ARI with the same DBCV+ARI pooling (minmax across cells).
    best_score stays mean DBCV for heatmaps; outer_score chooses the winner.
"""


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    epilog=_EPILOG,
)
@click.option(
    "--input-parquet",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    help="Single mention-level embedding parquet (one model/layer).",
)
@click.option(
    "--report-path",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    help="Directory for dim-selection outputs and checkpoint.",
)
@click.option(
    "--pca-grid",
    type=click.STRING,
    default=",".join(str(v) for v in DEFAULT_PCA_GRID),
    show_default=True,
    help="Comma-separated coarse PCA component values.",
)
@click.option(
    "--umap-grid",
    type=click.STRING,
    default=",".join(str(v) for v in DEFAULT_UMAP_GRID),
    show_default=True,
    help="Comma-separated coarse UMAP dimension values.",
)
@click.option(
    "--refine/--no-refine",
    default=True,
    show_default=True,
    help="After coarse search, evaluate a local neighborhood around the winner.",
)
@click.option(
    "--cluster-viz-method",
    type=click.Choice(["pca", "umap"], case_sensitive=False),
    default="pca",
    show_default=True,
    help="Reducer for cluster-space visualization (PCA or UMAP on clustering coords).",
)
@click.option(
    "--manifold-kind",
    type=click.Choice(["umap", "parametric"], case_sensitive=False),
    default="umap",
    show_default=True,
    help=(
        "Clustering manifold to search on. Must match the manifold the fit will use "
        "(pelinker-fit predict_mode=compact implies 'parametric'), or the chosen "
        "min_cluster_size is transferred across a coordinate-system change."
    ),
)
@click.option(
    "--umap-n-neighbors",
    type=click.INT,
    default=None,
    help="UMAP n_neighbors; omit for the library default (15). Scale-dependent.",
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
    help="Filename prefix used to parse model/layer from --input-parquet.",
)
@click.option(
    "--model",
    type=click.STRING,
    default=None,
    help="Override model name (default: parse from parquet filename).",
)
@click.option(
    "--layer",
    type=click.STRING,
    default=None,
    help="Override layer label (default: parse from parquet filename).",
)
@click.option(
    "--n-sample",
    type=click.INT,
    default=3,
    show_default=True,
    help="Number of bootstrap samples per (pca, umap) cell.",
)
@click.option(
    "--selected-labels-kb-path",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help="Optional path to selected labels KB CSV. If provided, clustering uses only those labels.",
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
        "Default: max(1, min_class_size // 2)."
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
    "--resume/--no-resume",
    default=True,
    show_default=True,
    help=(
        "If the checkpoint file exists and matches the run fingerprint, skip completed cells. "
        "Use --no-resume to ignore an existing checkpoint and reinitialize."
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
def main(
    input_parquet: pathlib.Path,
    report_path: pathlib.Path,
    pca_grid: str,
    umap_grid: str,
    refine: bool,
    cluster_viz_method: str,
    manifold_kind: str,
    umap_n_neighbors: int | None,
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
    model: str | None,
    layer: str | None,
    selected_labels_kb_path: pathlib.Path | None,
    max_scale: int,
    min_scale: int | None,
    clustering_grid_step: int,
    resume: bool,
    checkpoint_path: pathlib.Path | None,
    negative_label: str,
    screener_kind: str,
) -> None:
    run_dim_selection(
        input_parquet=input_parquet,
        report_path=report_path,
        pca_grid=pca_grid,
        umap_grid=umap_grid,
        refine=refine,
        cluster_viz_method=cluster_viz_method,
        manifold_kind=manifold_kind.lower(),
        umap_n_neighbors=umap_n_neighbors,
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
        model=model,
        layer=layer,
        selected_labels_kb_path=selected_labels_kb_path,
        max_scale=max_scale,
        min_scale=min_scale,
        clustering_grid_step=clustering_grid_step,
        resume=resume,
        checkpoint_path=checkpoint_path,
        negative_label=negative_label,
        screener_kind=screener_kind,
    )


if __name__ == "__main__":
    main()
