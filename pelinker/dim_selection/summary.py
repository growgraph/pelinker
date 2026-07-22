"""Summary artifacts for PCA/UMAP dimension selection."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pelinker.clustering_search_ranking import (
    METRICS_TWO_LEVEL_DOC,
    OUTER_SCORE_COL,
    attach_outer_scores,
)
from pelinker.dim_selection.grids import pick_winner_row

DIM_SELECTION_RESULTS_CSV_BASENAME = "dim_selection.results.csv"
DIM_SELECTION_SUMMARY_JSON_BASENAME = "dim_selection.summary.json"
DIM_SELECTION_SUMMARY_JSON_SCHEMA = "pelinker.dim_selection.summary.v1"
DIM_SELECTION_DBCV_HEATMAP_BASENAME = "dim.dbcv.heatmap.png"
DIM_SELECTION_ARI_HEATMAP_BASENAME = "dim.ari.heatmap.png"
DIM_SELECTION_OUTER_HEATMAP_BASENAME = "dim.outer.heatmap.png"

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


def _plot_dim_heatmap(
    df: pd.DataFrame,
    *,
    metric: str,
    metric_label: str,
    output_path: pathlib.Path,
) -> bool:
    if df.empty or metric not in df.columns:
        return False
    if df[metric].isna().all():
        return False
    pivot = df.pivot_table(
        index="umap_dim",
        columns="pca_components",
        values=metric,
        aggfunc="mean",
    )
    if pivot.empty:
        return False
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def write_dim_heatmaps(df: pd.DataFrame, report_path: pathlib.Path) -> list[str]:
    written: list[str] = []
    scored = attach_outer_scores(df, use_minmax=True) if not df.empty else df
    outer_path = report_path / DIM_SELECTION_OUTER_HEATMAP_BASENAME
    if _plot_dim_heatmap(
        scored,
        metric=OUTER_SCORE_COL,
        metric_label="Outer DBCV+ARI (minmax)",
        output_path=outer_path,
    ):
        written.append(outer_path.name)
    dbcv_path = report_path / DIM_SELECTION_DBCV_HEATMAP_BASENAME
    if _plot_dim_heatmap(
        scored,
        metric="best_score",
        metric_label="Mean DBCV (at pooled MCS)",
        output_path=dbcv_path,
    ):
        written.append(dbcv_path.name)
    ari_path = report_path / DIM_SELECTION_ARI_HEATMAP_BASENAME
    if "ari" in scored.columns and _plot_dim_heatmap(
        scored,
        metric="ari",
        metric_label="Mean ARI (at pooled MCS)",
        output_path=ari_path,
    ):
        written.append(ari_path.name)
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
        chosen = {
            "pca_components": int(winner["pca_components"]),
            "umap_dim": int(winner["umap_dim"]),
            "outer_score": float(winner.get(OUTER_SCORE_COL) or winner["best_score"]),
            "best_score": float(winner["best_score"]),
            "best_score_std": float(winner.get("best_score_std") or 0.0),
            "ari": ari_out,
            "best_size": float(winner.get("best_size") or 0.0),
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


def render_dim_selection_summary(
    df: pd.DataFrame,
    report_path: pathlib.Path,
    *,
    model: str,
    layer: str,
    n_sample: int,
    refine: bool,
) -> dict[str, Any]:
    write_results_csv(df, report_path)
    heatmaps = write_dim_heatmaps(df, report_path)
    payload = build_dim_selection_summary_payload(
        df, model=model, layer=layer, n_sample=n_sample, refine=refine
    )
    payload["heatmaps"] = heatmaps
    write_dim_selection_summary_json(payload, report_path)
    return payload
