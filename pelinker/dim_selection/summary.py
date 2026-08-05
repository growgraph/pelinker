"""Summary artifacts for PCA/UMAP dimension selection."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # registers 3d projection

from pelinker.clustering_search_ranking import (
    METRICS_TWO_LEVEL_DOC,
    OUTER_SCORE_COL,
    attach_outer_scores,
)
from pelinker.dim_selection.grids import pick_winner_row
from pelinker.grid_export import select_grid_points_at_chosen_min_cluster_size
from pelinker.plotting import _save_figure_multi_format, plot_dbcv_vs_ari_from_grid

DIM_SELECTION_RESULTS_CSV_BASENAME = "dim_selection.results.csv"
DIM_SELECTION_SUMMARY_JSON_BASENAME = "dim_selection.summary.json"
DIM_SELECTION_SUMMARY_JSON_SCHEMA = "pelinker.dim_selection.summary.v1"

# Stem basenames (PNG + PDF siblings via ``_save_figure_multi_format``).
DIM_SELECTION_DBCV_HEATMAP_STEM = "dim.dbcv.heatmap"
DIM_SELECTION_ARI_HEATMAP_STEM = "dim.ari.heatmap"
DIM_SELECTION_OUTER_HEATMAP_STEM = "dim.outer.heatmap"
DIM_SELECTION_OUTER_SURFACE_STEM = "dim.outer.surface"
DIM_SELECTION_METRICS_VIOLIN_STEM = "dim.metrics.violin"
DIM_SELECTION_DBCV_VS_ARI_STEM = "dim.dbcv_vs_ari"

# Legacy PNG-only basename aliases (tests / callers that expect ``.png``).
DIM_SELECTION_DBCV_HEATMAP_BASENAME = f"{DIM_SELECTION_DBCV_HEATMAP_STEM}.png"
DIM_SELECTION_ARI_HEATMAP_BASENAME = f"{DIM_SELECTION_ARI_HEATMAP_STEM}.png"
DIM_SELECTION_OUTER_HEATMAP_BASENAME = f"{DIM_SELECTION_OUTER_HEATMAP_STEM}.png"

METRICS_DOC = dict(METRICS_TWO_LEVEL_DOC)


def results_dataframe_from_summaries(
    summaries: list[dict[str, str | float | int | None]],
) -> pd.DataFrame:
    if not summaries:
        return pd.DataFrame(
            columns=[
                "model",
                "layer",
                "pca_components",
                "umap_dim",
                "best_score",
                "best_score_std",
                "ari",
                "ari_std",
                "best_size",
                "best_size_std",
            ]
        )
    return pd.DataFrame(summaries)


def write_results_csv(df: pd.DataFrame, report_path: pathlib.Path) -> pathlib.Path:
    out = report_path / DIM_SELECTION_RESULTS_CSV_BASENAME
    scored = attach_outer_scores(df, use_minmax=True) if not df.empty else df
    tmp = out.with_suffix(out.suffix + ".tmp")
    scored.to_csv(tmp, index=False)
    tmp.replace(out)
    return out


def _figure_basenames(written: tuple[pathlib.Path, ...]) -> list[str]:
    return [p.name for p in written]


def _plot_dim_heatmap(
    df: pd.DataFrame,
    *,
    metric: str,
    metric_label: str,
    output_path: pathlib.Path,
) -> list[str]:
    if df.empty or metric not in df.columns:
        return []
    if df[metric].isna().all():
        return []
    pivot = df.pivot_table(
        index="umap_dim",
        columns="pca_components",
        values=metric,
        aggfunc="mean",
    )
    if pivot.empty:
        return []
    fig, ax = plt.subplots(
        figsize=(
            max(6, len(pivot.columns) * 1.1),
            max(4, len(pivot.index) * 0.8),
        )
    )
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        cbar_kws={"label": metric_label, "shrink": 0.8},
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_xlabel("pca_components")
    ax.set_ylabel("umap_dim")
    ax.set_title(metric_label)
    fig.tight_layout()
    try:
        paths = _save_figure_multi_format(fig, output_path)
    finally:
        plt.close(fig)
    return _figure_basenames(paths)


def write_dim_heatmaps(df: pd.DataFrame, report_path: pathlib.Path) -> list[str]:
    written: list[str] = []
    scored = attach_outer_scores(df, use_minmax=True) if not df.empty else df
    written.extend(
        _plot_dim_heatmap(
            scored,
            metric=OUTER_SCORE_COL,
            metric_label="Outer DBCV+ARI (minmax)",
            output_path=report_path / DIM_SELECTION_OUTER_HEATMAP_STEM,
        )
    )
    written.extend(
        _plot_dim_heatmap(
            scored,
            metric="best_score",
            metric_label="Mean DBCV (at pooled MCS)",
            output_path=report_path / DIM_SELECTION_DBCV_HEATMAP_STEM,
        )
    )
    if "ari" in scored.columns:
        written.extend(
            _plot_dim_heatmap(
                scored,
                metric="ari",
                metric_label="Mean ARI (at pooled MCS)",
                output_path=report_path / DIM_SELECTION_ARI_HEATMAP_STEM,
            )
        )
    return written


def write_dim_outer_surface(df: pd.DataFrame, report_path: pathlib.Path) -> list[str]:
    """Pseudo-3D surface of ``outer_score`` over PCA × UMAP (PNG + PDF)."""
    if df.empty or "pca_components" not in df.columns or "umap_dim" not in df.columns:
        return []
    scored = attach_outer_scores(df, use_minmax=True)
    if OUTER_SCORE_COL not in scored.columns or scored[OUTER_SCORE_COL].isna().all():
        return []

    agg = (
        scored.groupby(["pca_components", "umap_dim"], as_index=False)[OUTER_SCORE_COL]
        .mean()
        .dropna(subset=[OUTER_SCORE_COL])
    )
    if agg.empty:
        return []

    winner = pick_winner_row(scored)
    win_pca = float(winner["pca_components"])
    win_umap = float(winner["umap_dim"])
    win_z = float(winner.get(OUTER_SCORE_COL) or winner["best_score"])

    pca_vals = np.sort(agg["pca_components"].unique().astype(np.float64))
    umap_vals = np.sort(agg["umap_dim"].unique().astype(np.float64))
    pivot = agg.pivot_table(
        index="umap_dim",
        columns="pca_components",
        values=OUTER_SCORE_COL,
        aggfunc="mean",
    ).reindex(index=umap_vals, columns=pca_vals)
    rectangular = (
        bool(pivot.notna().all().all()) and pivot.shape[0] >= 2 and pivot.shape[1] >= 2
    )

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    if rectangular:
        xx, yy = np.meshgrid(pca_vals, umap_vals)
        zz = pivot.to_numpy(dtype=np.float64)
        surf = ax.plot_surface(
            xx,
            yy,
            zz,
            cmap="RdBu_r",
            edgecolor="white",
            linewidth=0.3,
            alpha=0.92,
            antialiased=True,
        )
        fig.colorbar(surf, ax=ax, shrink=0.65, label="Outer DBCV+ARI (minmax)")
    else:
        xs = agg["pca_components"].to_numpy(dtype=np.float64)
        ys = agg["umap_dim"].to_numpy(dtype=np.float64)
        zs = agg[OUTER_SCORE_COL].to_numpy(dtype=np.float64)
        if len(xs) < 3:
            ax.scatter(xs, ys, zs, c=zs, cmap="RdBu_r", s=60)
        else:
            surf = ax.plot_trisurf(
                xs,
                ys,
                zs,
                cmap="RdBu_r",
                edgecolor="white",
                linewidth=0.2,
                alpha=0.92,
            )
            fig.colorbar(surf, ax=ax, shrink=0.65, label="Outer DBCV+ARI (minmax)")

    ax.scatter(
        [win_pca],
        [win_umap],
        [win_z],
        color="black",
        s=80,
        depthshade=False,
        label="winner",
        zorder=10,
    )
    ax.set_xlabel("pca_components")
    ax.set_ylabel("umap_dim")
    ax.set_zlabel("outer_score")
    ax.set_title("Outer DBCV+ARI surface (minmax)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    try:
        paths = _save_figure_multi_format(
            fig, report_path / DIM_SELECTION_OUTER_SURFACE_STEM
        )
    finally:
        plt.close(fig)
    return _figure_basenames(paths)


def _cell_label(pca: object, umap: object) -> str:
    return f"{int(pca)}×{int(umap)}"


def _grid_points_with_dims(grid_df: pd.DataFrame) -> pd.DataFrame:
    """Per-sample DBCV/ARI at pooled MCS, with ``pca_components`` / ``umap_dim``."""
    if grid_df.empty:
        return grid_df.iloc[0:0].copy()
    points = select_grid_points_at_chosen_min_cluster_size(grid_df)
    if points.empty:
        return points
    if "pca_components" in points.columns and "umap_dim" in points.columns:
        return points
    if "pca_components" in grid_df.columns and "umap_dim" in grid_df.columns:
        dim_map = (
            grid_df[["model", "layer", "pca_components", "umap_dim"]]
            .drop_duplicates(subset=["model", "layer"])
            .set_index(["model", "layer"])
        )
        joined = points.join(dim_map, on=["model", "layer"], how="left")
        return joined
    # Parse from export layer ``…:pcaK:umapD``.
    parsed_pca: list[int | None] = []
    parsed_umap: list[int | None] = []
    for layer in points["layer"].astype(str):
        pca_v: int | None = None
        umap_v: int | None = None
        for part in layer.split(":"):
            if part.startswith("pca") and part[3:].isdigit():
                pca_v = int(part[3:])
            elif part.startswith("umap") and part[4:].isdigit():
                umap_v = int(part[4:])
        parsed_pca.append(pca_v)
        parsed_umap.append(umap_v)
    out = points.copy()
    out["pca_components"] = parsed_pca
    out["umap_dim"] = parsed_umap
    return out.dropna(subset=["pca_components", "umap_dim"])


def write_dim_metrics_violin(
    grid_df: pd.DataFrame, report_path: pathlib.Path
) -> list[str]:
    """Two-panel violin of per-bootstrap DBCV / ARI across PCA×UMAP cells."""
    points = _grid_points_with_dims(grid_df)
    if points.empty:
        return []
    if points["sample_idx"].nunique() < 2:
        return []
    if "dbcv" not in points.columns or "ari" not in points.columns:
        return []

    plot_df = points.dropna(subset=["dbcv", "ari", "pca_components", "umap_dim"]).copy()
    if plot_df.empty:
        return []
    plot_df["cell"] = [
        _cell_label(p, u)
        for p, u in zip(plot_df["pca_components"], plot_df["umap_dim"], strict=True)
    ]
    cell_order = (
        plot_df.groupby("cell", as_index=False)[["pca_components", "umap_dim"]]
        .first()
        .sort_values(["pca_components", "umap_dim"])["cell"]
        .tolist()
    )

    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(cell_order) * 0.7), 5))
    for ax, metric, title in (
        (axes[0], "dbcv", "DBCV at pooled MCS"),
        (axes[1], "ari", "ARI at pooled MCS"),
    ):
        sns.violinplot(
            data=plot_df,
            x="cell",
            y=metric,
            order=cell_order,
            inner="quartile",
            cut=0,
            ax=ax,
            color="#6BAED6",
        )
        if plot_df["sample_idx"].nunique() <= 5:
            sns.stripplot(
                data=plot_df,
                x="cell",
                y=metric,
                order=cell_order,
                ax=ax,
                color="black",
                size=3,
                alpha=0.65,
            )
        ax.set_xlabel("pca × umap")
        ax.set_ylabel(metric.upper() if metric == "ari" else "DBCV")
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=45)
        for label in ax.get_xticklabels():
            label.set_ha("right")

    fig.tight_layout()
    try:
        paths = _save_figure_multi_format(
            fig, report_path / DIM_SELECTION_METRICS_VIOLIN_STEM
        )
    finally:
        plt.close(fig)
    return _figure_basenames(paths)


def write_dim_dbcv_vs_ari(
    grid_df: pd.DataFrame, report_path: pathlib.Path
) -> list[str]:
    """DBCV vs ARI scatter (one point / ellipse per PCA×UMAP cell)."""
    if grid_df.empty:
        return []
    out = report_path / DIM_SELECTION_DBCV_VS_ARI_STEM
    if not plot_dbcv_vs_ari_from_grid(grid_df, out):
        return []
    written: list[str] = []
    for fmt in ("png", "pdf"):
        path = report_path / f"{DIM_SELECTION_DBCV_VS_ARI_STEM}.{fmt}"
        if path.is_file():
            written.append(path.name)
    return written


def build_dim_selection_summary_payload(
    df: pd.DataFrame,
    *,
    model: str,
    layer: str,
    n_sample: int,
    refine: bool,
) -> dict[str, Any]:
    chosen: dict[str, Any] | None = None
    if not df.empty and "best_score" in df.columns:
        winner = pick_winner_row(df)
        ari_raw = winner.get("ari")
        ari_out: float | None
        if ari_raw is None or (isinstance(ari_raw, float) and np.isnan(ari_raw)):
            ari_out = None
        else:
            ari_out = float(ari_raw)
        n_rows_raw = winner.get("n_rows_realized")
        chosen = {
            "pca_components": int(winner["pca_components"]),
            "umap_dim": int(winner["umap_dim"]),
            "outer_score": float(winner.get(OUTER_SCORE_COL) or winner["best_score"]),
            "best_score": float(winner["best_score"]),
            "best_score_std": float(winner.get("best_score_std") or 0.0),
            "ari": ari_out,
            "best_size": float(winner.get("best_size") or 0.0),
            # The scale the chosen min_cluster_size is only meaningful against.
            "n_rows_realized": (
                None
                if n_rows_raw is None
                or (isinstance(n_rows_raw, float) and np.isnan(n_rows_raw))
                else int(n_rows_raw)
            ),
        }
    return {
        "schema": DIM_SELECTION_SUMMARY_JSON_SCHEMA,
        "model": model,
        "layer": layer,
        "n_sample": int(n_sample),
        "refine": bool(refine),
        "metrics": dict(METRICS_DOC),
        "chosen": chosen,
        "n_cells": int(len(df)),
    }


def write_dim_selection_summary_json(
    payload: dict[str, Any], report_path: pathlib.Path
) -> pathlib.Path:
    out = report_path / DIM_SELECTION_SUMMARY_JSON_BASENAME
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(out)
    return out


def _load_grid_csv(grid_csv_path: pathlib.Path | None) -> pd.DataFrame:
    if grid_csv_path is None or not grid_csv_path.is_file():
        return pd.DataFrame()
    return pd.read_csv(grid_csv_path)


def render_dim_selection_summary(
    df: pd.DataFrame,
    report_path: pathlib.Path,
    *,
    model: str,
    layer: str,
    n_sample: int,
    refine: bool,
    grid_csv_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    report_path.mkdir(parents=True, exist_ok=True)
    write_results_csv(df, report_path)
    figures: list[str] = []
    heatmaps = write_dim_heatmaps(df, report_path)
    figures.extend(heatmaps)
    figures.extend(write_dim_outer_surface(df, report_path))

    grid_df = _load_grid_csv(grid_csv_path)
    figures.extend(write_dim_metrics_violin(grid_df, report_path))
    figures.extend(write_dim_dbcv_vs_ari(grid_df, report_path))

    payload = build_dim_selection_summary_payload(
        df, model=model, layer=layer, n_sample=n_sample, refine=refine
    )
    payload["heatmaps"] = heatmaps
    payload["figures"] = figures
    write_dim_selection_summary_json(payload, report_path)
    return payload
