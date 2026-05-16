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
        "OOV and combined AUC, grouped AUC bar chart, top-combo ROC curves, "
        "and PCA quality pair grid. "
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
def main(
    report_dir: pathlib.Path,
    checkpoint: pathlib.Path | None,
) -> None:
    report_dir = report_dir.expanduser().resolve()
    res = model_selection.render_model_selection_summary_figures(
        report_dir,
        checkpoint_path=checkpoint,
    )
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
