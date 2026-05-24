"""Regenerate model-selection summary figures from existing on-disk artifacts."""

from __future__ import annotations

import pathlib
import sys

import click

import model_selection

from pelinker.model_selection_checkpoint import DEFAULT_CHECKPOINT_NAME


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Regenerate all model-selection figures under a report directory: "
        "per-combination DBCV/ARI/n_clusters metric plots (PNG+PDF), "
        "DBCV vs ARI scatter, perf/ARI heatmaps, screener LDA & SVM & best AUC, "
        "OOV and combined AUC, grouped AUC bar chart, top-combo and best-case ROC "
        "curves, model_selection.summary.json (top screener/combined rankings), "
        "and per-combination PCA quality pair grids (one sample per combo by default). "
        "Requires model_selection.state.json.gz (checkpoint); optional "
        "results_grid_per_sample.csv, fine_screener_eval.jsonl.gz, and "
        "fine_metadata.jsonl.gz."
    ),
)
@click.argument(
    "report_dir",
    type=click.Path(
        path_type=pathlib.Path, exists=True, file_okay=False, dir_okay=True
    ),
)
@click.option(
    "--checkpoint",
    "-c",
    type=click.Path(path_type=pathlib.Path, dir_okay=False),
    default=None,
    help=f"Checkpoint path (default: <report-dir>/{DEFAULT_CHECKPOINT_NAME})",
)
@click.option(
    "--grid-cluster-count-reward",
    type=float,
    default=None,
    help="Weight on log(n_clusters/n_ref) when re-solving chosen min_cluster_size from grid CSV.",
)
@click.option(
    "--grid-n-entities",
    type=int,
    default=None,
    help="Reference entity count for the cluster-count term (default: max clusters on grid).",
)
@click.option(
    "--all-pca-pairgrid-samples",
    is_flag=True,
    default=False,
    help=(
        "Emit a PCA quality pair grid for every sample index per (model, layer). "
        "Default: only the lowest sample index per combination."
    ),
)
def main(
    report_dir: pathlib.Path,
    checkpoint: pathlib.Path | None,
    grid_cluster_count_reward: float | None,
    grid_n_entities: int | None,
    all_pca_pairgrid_samples: bool,
) -> None:
    from pelinker.config import ClusteringOptimizationConfig

    report_dir = report_dir.expanduser().resolve()
    opt_config: ClusteringOptimizationConfig | None = None
    if grid_cluster_count_reward is not None or grid_n_entities is not None:
        opt_config = ClusteringOptimizationConfig(
            grid_cluster_count_reward=grid_cluster_count_reward or 0.0,
            grid_n_entities=grid_n_entities,
        )
    res = model_selection.render_model_selection_summary_figures(
        report_dir,
        checkpoint_path=checkpoint,
        optimization_config=opt_config,
        all_pca_pairgrid_samples=all_pca_pairgrid_samples,
    )
    if res.chosen_by_combo:
        click.echo("Resolved chosen_min_cluster_size per (model, layer):")
        for model, layer, mcs in res.chosen_by_combo:
            click.echo(f"  {model} / {layer}: {mcs}")
    if res.chosen_hyperparameters_path is not None:
        click.echo(f"Grid hyperparameters: {res.chosen_hyperparameters_path}")
        click.echo(
            "Updated results_grid_per_sample.csv (chosen_min_cluster_size column)."
        )
    if res.summary_json_path is not None:
        click.echo(f"Summary report: {res.summary_json_path}")
    for path in res.written_paths:
        click.echo(f"Wrote {path}")
    for msg in res.skipped_messages:
        click.echo(f"Skipped: {msg}", err=True)
    if not res.written_paths:
        click.echo(
            "No figures were written; check checkpoint and report directory artifacts.",
            err=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
