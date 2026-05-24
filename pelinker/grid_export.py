"""Per-sample clustering grid CSV schema and selection at the consensus hyperparameter."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd

from pelinker.clustering_grid import SmoothedGridOptimumResult
from pelinker.config import ClusteringOptimizationConfig
from pelinker.reporting import ModelSelectionReport, _json_normalize

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


def per_combo_metrics_from_grid(
    df_grid: pd.DataFrame,
) -> dict[tuple[str, str], list[pd.DataFrame]]:
    """
    Per (model, layer), list of per-sample grid metric tables for re-solving ``min_cluster_size``.

    Each DataFrame has columns among ``min_cluster_size``, ``dbcv``, ``ari``, ``n_clusters``, ``icm``.
    """
    metric_cols = [
        c
        for c in ("min_cluster_size", "dbcv", "ari", "n_clusters", "icm")
        if c in df_grid.columns
    ]
    if "min_cluster_size" not in metric_cols or "dbcv" not in metric_cols:
        return {}

    out: dict[tuple[str, str], list[pd.DataFrame]] = {}
    for (model, layer), combo_df in df_grid.groupby(["model", "layer"], sort=False):
        metrics_list: list[pd.DataFrame] = []
        if "sample_idx" in combo_df.columns:
            groups = combo_df.groupby("sample_idx", sort=True)
        else:
            groups = [(0, combo_df)]
        for _sidx, sample_df in groups:
            metrics_list.append(sample_df[metric_cols].reset_index(drop=True))
        if metrics_list:
            out[(str(model), str(layer))] = metrics_list
    return out


def apply_chosen_min_cluster_size_to_grid(
    df_grid: pd.DataFrame,
    chosen_by_combo: dict[tuple[str, str], int],
) -> pd.DataFrame:
    """Overwrite ``chosen_min_cluster_size`` per (model, layer) (e.g. after re-solving the grid)."""
    if GRID_COL_CHOSEN_MIN_CLUSTER_SIZE not in df_grid.columns:
        raise ValueError(f"grid frame missing {GRID_COL_CHOSEN_MIN_CLUSTER_SIZE!r}")
    out = df_grid.copy()
    for (model, layer), mcs in chosen_by_combo.items():
        mask = (out["model"].astype(str) == model) & (out["layer"].astype(str) == layer)
        out.loc[mask, GRID_COL_CHOSEN_MIN_CLUSTER_SIZE] = int(mcs)
    return out


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


def _solver_config_snapshot(
    config: ClusteringOptimizationConfig,
) -> dict[str, object]:
    return {
        "grid_objective": config.grid_objective,
        "optimization_method": config.optimization_method,
        "grid_cluster_count_reward": config.grid_cluster_count_reward,
        "grid_n_entities": config.grid_n_entities,
        "grid_smooth_window": config.grid_smooth_window,
        "grid_plateau_fraction": config.grid_plateau_fraction,
        "grid_derivative_rel_tol": config.grid_derivative_rel_tol,
    }


def _solved_combo_to_jsonable(
    model: str,
    layer: str,
    solved: SmoothedGridOptimumResult,
    config: ClusteringOptimizationConfig,
) -> dict[str, object]:
    return {
        "model": model,
        "layer": layer,
        "chosen_min_cluster_size": solved.chosen_min_cluster_size,
        "selection": solved.selection,
        "score_mean_at_chosen": solved.score_mean_at_chosen,
        "score_std_at_chosen": solved.score_std_at_chosen,
        "n_clusters_mean_at_chosen": solved.n_clusters_mean_at_chosen,
        "solver_config": _solver_config_snapshot(config),
        "grid_curve": {
            "min_cluster_size": list(solved.x),
            "y_objective": list(solved.y_objective),
            "y_cluster_term": list(solved.y_cluster_term),
            "y_smooth": list(solved.y_smooth),
        },
    }


def grid_chosen_hyperparameters_to_jsonable(
    solved_by_combo: dict[tuple[str, str], SmoothedGridOptimumResult],
    optimization_config: ClusteringOptimizationConfig,
) -> dict[str, Any]:
    """Build a JSON-serializable document of pooled grid solver results per (model, layer)."""
    config = optimization_config
    combos = [
        _solved_combo_to_jsonable(model, layer, solved, config)
        for (model, layer), solved in sorted(solved_by_combo.items())
    ]
    return {
        "schema": "pelinker.grid_chosen_hyperparameters.v1",
        "solver_config": _solver_config_snapshot(config),
        "combinations": combos,
    }


def write_grid_chosen_hyperparameters(
    path: pathlib.Path,
    solved_by_combo: dict[tuple[str, str], SmoothedGridOptimumResult],
    optimization_config: ClusteringOptimizationConfig,
) -> None:
    """Write ``grid_chosen_hyperparameters.json`` for downstream final fit."""
    payload = grid_chosen_hyperparameters_to_jsonable(
        solved_by_combo, optimization_config
    )
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _json_normalize(payload)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
