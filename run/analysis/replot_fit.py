"""Visualise cluster composition from a linker fit clustering report.

Produces figures per report directory (includes HDBSCAN noise ``-1`` as diagnostic):
  - ``fit_cluster_composition_bars.{png,pdf}``  – horizontal bar chart (top clusters by mass)
  - ``fit_cluster_composition_pies.{png,pdf}``  – pie-chart grid (all plotted clusters)
  - ``fit_cluster_composition_pies_sample.{png,pdf}``  – compact top-cluster sample
  - ``fit_cluster_viz.html``  – cluster-space viz (Plotly; HDBSCAN-fit + screener/OOV-pass rows)
  - ``fit_cluster_entity_sankey.{png,pdf}``  – capped KB-in entity→cluster Sankey

Reads ``linker_fit.clustering_report.json.gz``, ``linker_fit.cluster_composition.json.gz``,
and ``linker_fit.kb_out.json`` when present.

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

from pelinker.clustering.composition import (
    DEFAULT_MAX_CLUSTERS_FOR_PLOTS,
    DEFAULT_MAX_ENTITIES_FOR_FLOW_PLOTS,
    HDBSCAN_NOISE_CLUSTER_ID,
    aggregate_cluster_entity_mass,
    build_cluster_composition_df,
    cluster_entity_mass_summary,
    top_cluster_ids_by_mass,
    with_noise_cluster_label,
)

from pelinker.kb.kb_out import cluster_labels_from_catalog
from pelinker.plotting import (
    build_fit_cluster_viz_plot_df,
    enrich_fit_cluster_viz_plot_df_with_context,
    plot_cluster_entity_sankey,
    plot_cluster_viz,
)
from pelinker.reports.io import (
    read_cluster_composition_json,
    read_clustering_report_json,
    read_kb_out_json,
)
from pelinker.reports.paths import (
    linker_fit_cluster_composition_path,
    linker_fit_clustering_report_path,
    linker_fit_kb_out_path,
)

_PIE_SAMPLE_MAX_CLUSTERS = 6
_FIGURE_EXTS = ("png", "pdf")


def _always_include_noise(df: pd.DataFrame) -> list[int] | None:
    if "cluster" not in df.columns:
        return None
    if HDBSCAN_NOISE_CLUSTER_ID in set(df["cluster"].astype(int)):
        return [HDBSCAN_NOISE_CLUSTER_ID]
    return None


def _load_cluster_catalog(report_dir: pathlib.Path) -> dict | None:
    kb_out_path = linker_fit_kb_out_path(report_dir)
    if kb_out_path.exists():
        return read_kb_out_json(kb_out_path)
    return None


def _apply_cluster_labels_to_composition(
    df: pd.DataFrame,
    cluster_labels: dict[int, str],
) -> pd.DataFrame:
    if df.empty:
        return df
    labels = with_noise_cluster_label(cluster_labels)
    out = df.copy()
    out["cluster"] = out["cluster"].astype(int)
    out["cluster_label"] = out["cluster"].map(
        lambda cid: labels.get(int(cid), str(cid))
    )
    return out


def _cluster_title(cluster_id: int, cluster_data: pd.DataFrame) -> str:
    if "cluster_label" in cluster_data.columns and len(cluster_data):
        label = str(cluster_data["cluster_label"].iloc[0])
        return f"{label} (mass={cluster_data['count'].sum():.2f})"
    return f"Cluster {cluster_id} (mass={cluster_data['count'].sum():.2f})"


def plot_seaborn_bars(
    processed_df: pd.DataFrame,
    *,
    save_dir: pathlib.Path | None = None,
    show: bool = False,
) -> list[pathlib.Path]:
    if processed_df.empty:
        return []
    plot_df = processed_df.copy()
    facet_col = "cluster_label" if "cluster_label" in plot_df.columns else "cluster"
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=plot_df,
        y="entity",
        x="count",
        hue="entity",
        col=facet_col,
        col_wrap=3,
        kind="bar",
        sharey=False,
        sharex=False,
        palette="muted",
        legend=False,
        height=3.2,
        aspect=1.4,
    )
    g.set_titles("{col_name}", weight="bold", size=11)
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
    always = _always_include_noise(plot_df)
    if max_clusters is not None:
        keep = top_cluster_ids_by_mass(
            plot_df, max_clusters=max_clusters, always_include=always
        )
        plot_df = plot_df.loc[plot_df["cluster"].isin(keep)]

    clusters = top_cluster_ids_by_mass(
        plot_df, max_clusters=None, always_include=always
    )
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
            _cluster_title(int(cluster), cluster_data),
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
        stored_exclude = meta.get("exclude_noise")
        needs_rebuild = False
        if max_clusters is not None and stored_max is not None:
            if int(stored_max) < int(max_clusters):
                needs_rebuild = True
        if stored_exclude is True:
            needs_rebuild = True
        if needs_rebuild:
            from pelinker.reports.schema import ModelSelectionReport

            if isinstance(report, ModelSelectionReport):
                return build_cluster_composition_df(
                    report.assignments,
                    top_n=top_n,
                    weight_by_entity=True,
                    exclude_noise=False,
                    max_clusters=max_clusters,
                )
        return df
    from pelinker.reports.schema import ModelSelectionReport

    if not isinstance(report, ModelSelectionReport):
        raise TypeError("report must be a ModelSelectionReport")
    return build_cluster_composition_df(
        report.assignments,
        top_n=top_n,
        weight_by_entity=True,
        exclude_noise=False,
        max_clusters=max_clusters,
    )


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Regenerate cluster-composition figures from a linker fit report directory. "
        "Includes HDBSCAN noise label -1 as a diagnostic cluster."
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
    help="Top KB-in entities by mass for the Sankey chart.",
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
@click.option(
    "--viz-all-kb",
    is_flag=True,
    default=False,
    help=(
        "Cluster viz: plot all KB assignments (legacy). Default restricts to rows on which "
        "HDBSCAN was fit and that pass ambient screener and manifold OOV gates."
    ),
)
@click.option(
    "--sankey-min-frac",
    type=float,
    default=0.0,
    show_default=True,
    help=(
        "Drop entity→cluster edges below this within-cluster mass fraction in the Sankey."
    ),
)
@click.option(
    "--cluster-label",
    type=click.Choice(["display", "short"], case_sensitive=False),
    default="display",
    show_default=True,
    help="Use KB-out display or short label for cluster facets and Sankey.",
)
def main(
    report_dir: pathlib.Path,
    top_n: int,
    max_clusters: int,
    max_entities: int,
    show: bool,
    pmid_text_table: pathlib.Path | None,
    viz_all_kb: bool,
    sankey_min_frac: float,
    cluster_label: str,
) -> None:
    report_dir = report_dir.expanduser().resolve()

    report_path = linker_fit_clustering_report_path(report_dir)
    report = read_clustering_report_json(report_path)
    summary = cluster_entity_mass_summary(report.assignments)
    n_emergent = int(summary["n_emergent_clusters"])

    catalog = _load_cluster_catalog(report_dir)
    if catalog is not None:
        n_emergent = int(catalog.get("n_emergent_clusters", n_emergent))

    cluster_labels: dict[int, str] = {}
    if catalog is not None:
        label_kind = "short" if cluster_label == "short" else "display"
        cluster_labels = cluster_labels_from_catalog(catalog, label_kind=label_kind)
    cluster_labels = with_noise_cluster_label(cluster_labels)

    click.echo(
        f"Emergent clusters: {n_emergent} "
        f"(plotting top {max_clusters} + noise; noise fraction "
        f"{float(summary['noise_fraction']):.3f})"
    )
    if n_emergent > max_clusters:
        click.echo(
            f"Note: {n_emergent - max_clusters} smaller emergent clusters omitted from "
            "figures; see linker_fit.kb_out.json for the full catalog.",
            err=True,
        )

    processed_df = _load_composition_df(
        report_dir,
        report,
        top_n=top_n,
        max_clusters=max_clusters,
    )
    processed_df = _apply_cluster_labels_to_composition(processed_df, cluster_labels)
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

    plot_df, viz_method = build_fit_cluster_viz_plot_df(
        report,
        exclude_noise=False,
        hdbscan_fit_scope=not viz_all_kb,
        cluster_labels=cluster_labels or None,
    )
    if (
        not viz_all_kb
        and plot_df is not None
        and not any(c in report.assignments.columns for c in ("clustering_in_sample",))
    ):
        click.echo(
            "Note: report lacks clustering membership columns; cluster viz shows all KB rows. "
            "Re-run fit to persist in-sample / screener / OOV flags.",
            err=True,
        )
    if plot_df is not None and pmid_text_table is not None:
        plot_df = enrich_fit_cluster_viz_plot_df_with_context(
            plot_df,
            pmid_text_table,
        )
    if plot_df is not None and "cviz_00" in plot_df.columns:
        viz_path = report_dir / "fit_cluster_viz.html"
        plot_cluster_viz(plot_df, output_path=str(viz_path), viz_method=viz_method)
        written.append(viz_path)

    entity_flow = aggregate_cluster_entity_mass(
        report.assignments,
        weight_by_entity=True,
        exclude_noise=False,
    )
    written += plot_cluster_entity_sankey(
        entity_flow,
        save_dir=report_dir,
        max_clusters=max_clusters,
        max_entities=max_entities,
        min_within_cluster_fraction=sankey_min_frac,
        cluster_labels=cluster_labels or None,
    )

    if written:
        for p in written:
            click.echo(f"Wrote {p}")
    else:
        click.echo("No figures written.", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
