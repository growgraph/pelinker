"""Visualise cluster composition from a linker fit clustering report.

Produces figures per report directory (emergent clusters only; HDBSCAN noise ``-1`` excluded):
  - ``fit_cluster_composition_bars.{png,pdf}``  – horizontal bar chart (top clusters by mass)
  - ``fit_cluster_composition_pies.{png,pdf}``  – pie-chart grid (all plotted clusters)
  - ``fit_cluster_composition_pies_sample.{png,pdf}``  – compact top-cluster sample
  - ``fit_cluster_viz.html``  – cluster-space viz (Plotly, emergent mentions only)
  - ``fit_cluster_entity_sankey.{png,pdf}``  – capped entity→cluster Sankey

Reads ``linker_fit.clustering_report.json.gz``, ``linker_fit.cluster_composition.json.gz``,
and ``linker_fit.emergent_clusters.json`` when present.

With ``--pmid-text-table``, cluster viz hover text includes a five-word context window around
each mention (resolved via ``pmid``, ``a_abs``, and ``b_abs`` provenance).
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Callable

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pelinker.cluster_composition_viz import (
    DEFAULT_MAX_CLUSTERS_FOR_PLOTS,
    DEFAULT_MAX_ENTITIES_FOR_FLOW_PLOTS,
    build_cluster_composition_df,
    cluster_entity_mass_summary,
    top_cluster_ids_by_mass,
)

from pelinker.plotting import (
    build_fit_cluster_viz_plot_df,
    enrich_fit_cluster_viz_plot_df_with_context,
    plot_cluster_entity_sankey,
    plot_cluster_viz,
)
from pelinker.reporting import (
    linker_fit_cluster_composition_path,
    linker_fit_clustering_report_path,
    linker_fit_emergent_clusters_path,
    read_cluster_composition_json,
    read_clustering_report_json,
    read_emergent_clusters_json,
)

_PIE_SAMPLE_MAX_CLUSTERS = 6
_FIGURE_EXTS = ("png", "pdf")


def plot_seaborn_bars(
    processed_df: pd.DataFrame,
    *,
    save_dir: pathlib.Path | None = None,
    show: bool = False,
) -> list[pathlib.Path]:
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=processed_df,
        y="entity",
        x="count",
        hue="entity",
        col="cluster",
        col_wrap=3,
        kind="bar",
        sharey=False,
        sharex=False,
        palette="muted",
        legend=False,
        height=3.2,
        aspect=1.4,
    )
    g.set_titles("Cluster {col_name}", weight="bold", size=11)
    g.set_axis_labels("Weighted mass", "")
    plt.tight_layout()

    written: list[pathlib.Path] = []
    if save_dir is not None:
        for ext in _FIGURE_EXTS:
            p = save_dir / f"fit_cluster_composition_bars.{ext}"
            g.savefig(p)
            written.append(p)
    if show:
        plt.show()
    plt.close("all")
    return written


def _pie_grid_layout(n_clusters: int) -> tuple[int, float, float]:
    """(columns, width per column, row height) for a readable figure size."""
    if n_clusters <= 1:
        return 1, 4.5, 3.6
    if n_clusters <= 9:
        return 3, 4.5, 3.2
    if n_clusters <= 24:
        return 4, 3.6, 2.6
    return 6, 3.0, 2.2


def _pie_grid_layout_sample(n_clusters: int) -> tuple[int, float, float]:
    if n_clusters <= 1:
        return 1, 3.2, 2.8
    if n_clusters <= 4:
        return 2, 3.0, 2.6
    return 3, 2.6, 2.2


def plot_pie_grid(
    processed_df: pd.DataFrame,
    *,
    save_dir: pathlib.Path | None = None,
    show: bool = False,
    max_clusters: int | None = None,
    filename_stem: str = "fit_cluster_composition_pies",
    layout_fn: Callable[[int], tuple[int, float, float]] = _pie_grid_layout,
    label_fontsize: float = 8,
    title_fontsize: float = 10,
    autopct_min_pct: float = 3.0,
) -> list[pathlib.Path]:
    if processed_df.empty:
        return []

    plot_df = processed_df
    if max_clusters is not None:
        keep = top_cluster_ids_by_mass(plot_df, max_clusters=max_clusters)
        plot_df = plot_df.loc[plot_df["cluster"].isin(keep)]

    clusters = top_cluster_ids_by_mass(plot_df, max_clusters=None)
    if not clusters:
        return []

    n_clusters = len(clusters)
    cols, width_per_col, row_height = layout_fn(n_clusters)
    rows = (n_clusters + cols - 1) // cols
    fig_w = max(width_per_col * cols, width_per_col)
    fig_h = max(row_height * rows, row_height)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(fig_w, fig_h),
        subplot_kw={"aspect": "equal"},
    )
    if n_clusters == 1:
        axes_flat = [axes]
    else:
        axes_flat = list(np.asarray(axes).ravel())

    for i, cluster in enumerate(clusters):
        cluster_data = plot_df.loc[plot_df["cluster"] == cluster]
        n_slices = len(cluster_data)
        colors = sns.color_palette("Pastel1", max(4, n_slices))[:n_slices]
        axes_flat[i].pie(
            cluster_data["count"],
            labels=cluster_data["entity"],
            autopct=lambda p, t=autopct_min_pct: f"{p:.1f}%" if p > t else "",
            startangle=140,
            colors=colors,
            textprops={"fontsize": label_fontsize},
        )
        axes_flat[i].set_title(
            f"Cluster {cluster} (mass={cluster_data['count'].sum():.2f})",
            fontweight="bold",
            fontsize=title_fontsize,
        )

    last_i = len(clusters) - 1
    for j in range(last_i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout()

    written: list[pathlib.Path] = []
    if save_dir is not None:
        for ext in _FIGURE_EXTS:
            p = save_dir / f"{filename_stem}.{ext}"
            fig.savefig(p, bbox_inches="tight")
            written.append(p)
    if show:
        plt.show()
    plt.close("all")
    return written


def _load_composition_df(
    report_dir: pathlib.Path,
    report: object,
    *,
    top_n: int,
    max_clusters: int | None,
) -> pd.DataFrame:
    composition_path = linker_fit_cluster_composition_path(report_dir)
    if composition_path.exists():
        df, meta = read_cluster_composition_json(composition_path)
        stored_max = meta.get("max_clusters_in_rows")
        if max_clusters is not None and stored_max is not None:
            if int(stored_max) < int(max_clusters):
                from pelinker.reporting import ModelSelectionReport

                if isinstance(report, ModelSelectionReport):
                    return build_cluster_composition_df(
                        report.assignments,
                        top_n=top_n,
                        weight_by_entity=True,
                        exclude_noise=True,
                        max_clusters=max_clusters,
                    )
        return df
    from pelinker.reporting import ModelSelectionReport

    if not isinstance(report, ModelSelectionReport):
        raise TypeError("report must be a ModelSelectionReport")
    return build_cluster_composition_df(
        report.assignments,
        top_n=top_n,
        weight_by_entity=True,
        exclude_noise=True,
        max_clusters=max_clusters,
    )


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Regenerate cluster-composition figures from a linker fit report directory. "
        "Uses emergent HDBSCAN clusters only (excludes noise label -1)."
    ),
)
@click.argument(
    "report_dir",
    type=click.Path(
        path_type=pathlib.Path, exists=True, file_okay=False, dir_okay=True
    ),
)
@click.option(
    "--top-n",
    type=int,
    default=3,
    show_default=True,
    help="Number of top entities per cluster to show individually (rest → Other).",
)
@click.option(
    "--max-clusters",
    type=int,
    default=DEFAULT_MAX_CLUSTERS_FOR_PLOTS,
    show_default=True,
    help="Largest emergent clusters by weighted mass to include in all figures.",
)
@click.option(
    "--max-entities",
    type=int,
    default=DEFAULT_MAX_ENTITIES_FOR_FLOW_PLOTS,
    show_default=True,
    help="Top entities by mass for Sankey and bump charts.",
)
@click.option(
    "--show",
    is_flag=True,
    default=False,
    help="Also display figures interactively (plt.show).",
)
@click.option(
    "--pmid-text-table",
    type=click.Path(path_type=pathlib.Path, dir_okay=False),
    default=None,
    help=(
        "TSV/CSV (optional gzip) with PMID and full text columns. "
        "Used to add a 5-word context window around each mention in the cluster viz hover."
    ),
)
def main(
    report_dir: pathlib.Path,
    top_n: int,
    max_clusters: int,
    max_entities: int,
    show: bool,
    pmid_text_table: pathlib.Path | None,
) -> None:
    report_dir = report_dir.expanduser().resolve()

    report_path = linker_fit_clustering_report_path(report_dir)
    report = read_clustering_report_json(report_path)
    summary = cluster_entity_mass_summary(report.assignments)
    n_emergent = int(summary["n_emergent_clusters"])

    emergent_path = linker_fit_emergent_clusters_path(report_dir)
    if emergent_path.exists():
        catalog = read_emergent_clusters_json(emergent_path)
        n_emergent = int(catalog.get("n_emergent_clusters", n_emergent))

    click.echo(
        f"Emergent clusters: {n_emergent} "
        f"(plotting top {max_clusters}; noise fraction "
        f"{float(summary['noise_fraction']):.3f})"
    )
    if n_emergent > max_clusters:
        click.echo(
            f"Note: {n_emergent - max_clusters} smaller emergent clusters omitted from "
            "figures; see linker_fit.emergent_clusters.json for the full catalog.",
            err=True,
        )

    processed_df = _load_composition_df(
        report_dir,
        report,
        top_n=top_n,
        max_clusters=max_clusters,
    )
    written: list[pathlib.Path] = []
    written += plot_seaborn_bars(processed_df, save_dir=report_dir, show=show)
    written += plot_pie_grid(processed_df, save_dir=report_dir, show=show)
    written += plot_pie_grid(
        processed_df,
        save_dir=report_dir,
        show=show,
        max_clusters=_PIE_SAMPLE_MAX_CLUSTERS,
        filename_stem="fit_cluster_composition_pies_sample",
        layout_fn=_pie_grid_layout_sample,
        label_fontsize=6,
        title_fontsize=8,
        autopct_min_pct=5.0,
    )

    plot_df, viz_method = build_fit_cluster_viz_plot_df(report, exclude_noise=True)
    if plot_df is not None and pmid_text_table is not None:
        plot_df = enrich_fit_cluster_viz_plot_df_with_context(
            plot_df,
            pmid_text_table,
        )
    if plot_df is not None and "cviz_00" in plot_df.columns:
        viz_path = report_dir / "fit_cluster_viz.html"
        plot_cluster_viz(plot_df, output_path=str(viz_path), viz_method=viz_method)
        written.append(viz_path)

    written += plot_cluster_entity_sankey(
        processed_df,
        save_dir=report_dir,
        max_clusters=max_clusters,
        max_entities=max_entities,
    )

    if written:
        for p in written:
            click.echo(f"Wrote {p}")
    else:
        click.echo("No figures written.", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
