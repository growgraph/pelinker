"""Replot DBCV vs Hungarian scatter from an existing ``results_grid_per_sample.csv``."""

from __future__ import annotations

import pathlib
import sys

import click
import pandas as pd

from pelinker.plotting import plot_dbcv_vs_hungarian_from_grid


@click.command()
@click.argument(
    "grid_csv",
    type=click.Path(path_type=pathlib.Path, exists=True, dir_okay=False),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help="Output PNG path (default: alongside CSV as model.dbcv_vs_hungarian.png)",
)
def main(grid_csv: pathlib.Path, output: pathlib.Path | None) -> None:
    df = pd.read_csv(grid_csv)
    out = (
        output
        if output is not None
        else grid_csv.parent / "model.dbcv_vs_hungarian.png"
    )
    if not plot_dbcv_vs_hungarian_from_grid(df, out):
        click.echo(
            "Could not build figure (need model, layer, sample_idx, "
            "sample_best_dbcv, sample_hungarian_accuracy with at least one "
            "non-null Hungarian row).",
            err=True,
        )
        sys.exit(1)
    click.echo(f"Wrote {out}")


if __name__ == "__main__":
    main()
