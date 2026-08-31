"""CLI for the ``min_cluster_size`` sample-size sweep."""

from __future__ import annotations

import pathlib

import click

from pelinker.core.onto import NEGATIVE_LABEL
from pelinker.search.scale_curve import DEFAULT_RUNGS, run_scale_curve

_EPILOG = """
Why: min_cluster_size is an absolute row count (and, by HDBSCAN's default,
so is min_samples). Model selection runs on a subsample; the fit runs on the
whole corpus. This command measures how the chosen value moves with N instead
of assuming a correction, then writes scale_curve.json for pelinker-fit to
consume via scale_curve_path=.

Reading the fitted exponent b in log(MCS) = a + b*log(N):

  b ~ 0    min_cluster_size is scale-invariant; transferring the absolute
           value between sample sizes is fine.
  0 < b < 1  it grows sublinearly (the expected regime).
  b ~ 1    it is really a constant fraction of N.

A low R-squared, or rungs flagged as "pinned", means the fixed
[--min-scale, --max-scale) grid decided the answer rather than the sample
size. Widen the grid and re-run before trusting the slope.
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
    help="Directory for scale_curve.json, the log-log figure, and grid rows.",
)
@click.option(
    "--rungs",
    type=click.STRING,
    default=",".join(str(v) for v in DEFAULT_RUNGS),
    show_default=True,
    help="Comma-separated clustering_sample_rows values (>=3 distinct).",
)
@click.option(
    "--pca-components",
    type=click.INT,
    default=100,
    show_default=True,
    help="PCA components; hold fixed across rungs so only N varies.",
)
@click.option(
    "--umap-dim",
    type=click.INT,
    default=8,
    show_default=True,
    help="UMAP output dimension; hold fixed across rungs.",
)
@click.option(
    "--umap-n-neighbors",
    type=click.INT,
    default=None,
    help="UMAP n_neighbors; omit for the library default (15).",
)
@click.option(
    "--cluster-viz-method",
    type=click.Choice(["pca", "umap"], case_sensitive=False),
    default="pca",
    show_default=True,
)
@click.option("--min-class-size", type=click.INT, default=20, show_default=True)
@click.option(
    "--max-scale",
    type=click.INT,
    default=60,
    show_default=True,
    help="Exclusive upper bound of the min_cluster_size grid.",
)
@click.option(
    "--min-scale",
    type=click.INT,
    default=None,
    help="Inclusive lower bound of the grid; defaults to min_class_size // 2.",
)
@click.option("--clustering-grid-step", type=click.INT, default=5, show_default=True)
@click.option("--seed", type=click.INT, default=13, show_default=True)
@click.option("--pca-seed", type=click.INT, default=13, show_default=True)
@click.option(
    "--umap-seed",
    type=click.INT,
    default=None,
    help="Set for reproducible rungs; omit for parallel UMAP.",
)
@click.option("--batch-size", type=click.INT, default=1000, show_default=True)
@click.option(
    "--n-sample",
    type=click.INT,
    default=3,
    show_default=True,
    help="Bootstrap draws pooled per rung.",
)
@click.option("--prefix", type=click.STRING, default="res", show_default=True)
@click.option("--model", type=click.STRING, default=None)
@click.option("--layer", type=click.STRING, default=None)
@click.option("--negative-label", type=click.STRING, default=NEGATIVE_LABEL)
@click.option(
    "--screener-kind",
    type=click.Choice(["lda", "svm"], case_sensitive=False),
    default="lda",
    show_default=True,
)
@click.option("--drop-rare-entities/--no-drop-rare-entities", default=False)
@click.option(
    "--min-mentions-per-entity", type=click.INT, default=20, show_default=True
)
@click.option("--max-mentions-per-entity", type=click.INT, default=None)
@click.option("--max-mentions-negative", type=click.INT, default=None)
@click.option(
    "--mention-cap-seed",
    type=click.INT,
    default=None,
    help="Defaults to --seed when omitted.",
)
def main(
    input_parquet: pathlib.Path,
    report_path: pathlib.Path,
    rungs: str,
    pca_components: int,
    umap_dim: int,
    umap_n_neighbors: int | None,
    cluster_viz_method: str,
    min_class_size: int,
    max_scale: int,
    min_scale: int | None,
    clustering_grid_step: int,
    seed: int,
    pca_seed: int,
    umap_seed: int | None,
    batch_size: int,
    n_sample: int,
    prefix: str,
    model: str | None,
    layer: str | None,
    negative_label: str,
    screener_kind: str,
    drop_rare_entities: bool,
    min_mentions_per_entity: int,
    max_mentions_per_entity: int | None,
    max_mentions_negative: int | None,
    mention_cap_seed: int | None,
) -> None:
    """Measure how the chosen min_cluster_size scales with the mention-frame size."""
    curve = run_scale_curve(
        input_parquet=input_parquet,
        report_path=report_path,
        rungs=rungs,
        pca_components=pca_components,
        umap_dim=umap_dim,
        umap_n_neighbors=umap_n_neighbors,
        cluster_viz_method=cluster_viz_method,
        min_class_size=min_class_size,
        seed=seed,
        pca_seed=pca_seed,
        umap_seed=umap_seed,
        batch_size=batch_size,
        n_sample=n_sample,
        prefix=prefix,
        model=model,
        layer=layer,
        max_scale=max_scale,
        min_scale=min_scale,
        clustering_grid_step=clustering_grid_step,
        negative_label=negative_label,
        screener_kind=screener_kind,
        drop_rare_entities=drop_rare_entities,
        min_mentions_per_entity=min_mentions_per_entity,
        max_mentions_per_entity=max_mentions_per_entity,
        max_mentions_negative=max_mentions_negative,
        mention_cap_seed=seed if mention_cap_seed is None else mention_cap_seed,
    )
    if curve is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
