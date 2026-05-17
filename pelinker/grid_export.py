"""Per-sample clustering grid CSV schema and selection at the consensus hyperparameter."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pelinker.reporting import ModelSelectionReport

GRID_COL_CHOSEN_MIN_CLUSTER_SIZE = "chosen_min_cluster_size"

# Primary key for one row per grid evaluation point in ``results_grid_per_sample.csv``.
GRID_EXPORT_ID_COLUMNS: tuple[str, ...] = (
    "model",
    "layer",
    "sample_idx",
    "min_cluster_size",
)

_GRID_POINT_COLUMNS: frozenset[str] = frozenset(
    {
        "model",
        "layer",
        "sample_idx",
        "min_cluster_size",
        "dbcv",
        "ari",
        GRID_COL_CHOSEN_MIN_CLUSTER_SIZE,
    }
)


def grid_export_column_order() -> list[str]:
    """Canonical column order for ``results_grid_per_sample.csv``."""
    return [
        "model",
        "layer",
        "sample_idx",
        GRID_COL_CHOSEN_MIN_CLUSTER_SIZE,
        "min_cluster_size",
        "icm",
        "n_clusters",
        "dbcv",
        "ari",
    ]


def grid_export_rows_from_report(
    report: ModelSelectionReport,
    *,
    model: str,
    layer: str,
    sample_idx: int,
    chosen_min_cluster_size: int,
) -> pd.DataFrame:
    """
    Expand one sample's grid ``metrics_df`` into rows for ``results_grid_per_sample.csv``.

    ``chosen_min_cluster_size`` is the pooled consensus hyperparameter for the
    (model, layer) combination; it is duplicated on every grid row for that sample.
    """
    return report.metrics_df.assign(
        model=model,
        layer=layer,
        sample_idx=sample_idx,
        **{GRID_COL_CHOSEN_MIN_CLUSTER_SIZE: int(chosen_min_cluster_size)},
    )


def _resolved_chosen_min_cluster_size(series: pd.Series) -> float | None:
    vals = series.dropna()
    if vals.empty:
        return None
    unique = np.unique(vals.astype(np.float64))
    if len(unique) == 1:
        return float(unique[0])
    counts = np.array([(vals.astype(np.float64) == u).sum() for u in unique])
    return float(unique[int(counts.argmax())])


def select_grid_points_at_chosen_min_cluster_size(
    df_grid: pd.DataFrame,
    *,
    require_n_clusters_gt_one: bool = True,
) -> pd.DataFrame:
    """
      One row per (model, layer, sample_idx): ``(dbcv, ari)`` at ``chosen_min_cluster_size``.

      Rows are taken from the grid sweep where ``min_cluster_size`` equals the consensus
    ``chosen_min_cluster_size`` for that (model, layer). This matches the vertical marker
      on per-combination error-bar plots.
    """
    if not _GRID_POINT_COLUMNS.issubset(df_grid.columns):
        missing = sorted(_GRID_POINT_COLUMNS - set(df_grid.columns))
        raise ValueError(f"grid frame missing required columns: {missing}")

    parts: list[pd.DataFrame] = []
    for (model, layer), g in df_grid.groupby(["model", "layer"], sort=False):
        chosen = _resolved_chosen_min_cluster_size(g[GRID_COL_CHOSEN_MIN_CLUSTER_SIZE])
        if chosen is None:
            continue
        mcs = g["min_cluster_size"].astype(np.float64)
        at = g.loc[mcs == chosen].copy()
        if require_n_clusters_gt_one and "n_clusters" in at.columns:
            at = at.loc[at["n_clusters"] > 1]
        if at.empty:
            continue
        at = at.drop_duplicates(
            subset=["model", "layer", "sample_idx"],
            keep="last",
        )
        at = at.loc[at["ari"].notna()]
        if at.empty:
            continue
        parts.append(at)

    if not parts:
        return df_grid.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def has_grid_points_for_dbcv_ari_scatter(df_grid: pd.DataFrame) -> bool:
    """True if ``df_grid`` has the columns needed for the DBCV vs ARI scatter."""
    return _GRID_POINT_COLUMNS.issubset(df_grid.columns)
