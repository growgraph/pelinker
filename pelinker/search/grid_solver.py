from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import pandas as pd
from pelinker.clustering.grid import (
    SmoothedGridOptimumResult,
    solve_optimal_min_cluster_size_from_metrics_dfs,
)
from pelinker.search.grid_export import per_combo_metrics_from_grid

from pelinker.core.config import (
    ClusteringOptimizationConfig,
    GridObjectiveSpec,
)

"""Resolve a chosen min_cluster_size from exported per-sample grid metrics.

Search-layer policy on top of :mod:`pelinker.clustering.grid`: pooling across
samples, applying config overrides, and deciding when a grid solve is warranted.
"""


def pooled_grid_solve_from_metrics_dfs(
    metrics_dfs: Sequence[pd.DataFrame],
    optimization_config: ClusteringOptimizationConfig | None = None,
) -> SmoothedGridOptimumResult:
    """
    After all bootstrap samples have run a min_cluster_size grid, solve for the consensus
    ``min_cluster_size`` and return full grid diagnostics (including the ``y_objective`` curve).

    The per-sample frames are passed through un-pooled: every grid point was evaluated on
    the same samples, so the solver can use *paired* standard errors, which are far tighter
    and far more stable than pooling first would allow.
    """
    if not metrics_dfs:
        raise ValueError("metrics_dfs must be non-empty")
    config = optimization_config or ClusteringOptimizationConfig()
    return solve_optimal_min_cluster_size_from_metrics_dfs(
        list(metrics_dfs),
        objective=config.grid_objective,
        smooth_window=config.grid_smooth_window,
        one_se_k=config.grid_one_se_k,
        one_se_contiguous=config.grid_one_se_contiguous,
        one_se_min_samples=config.grid_one_se_min_samples,
        cluster_count_reward=config.grid_cluster_count_reward,
        n_entities=config.grid_n_entities,
    )


def pooled_min_cluster_size_from_metrics_dfs(
    metrics_dfs: Sequence[pd.DataFrame],
    optimization_config: ClusteringOptimizationConfig | None = None,
) -> tuple[int, float]:
    """
    Convenience wrapper returning only ``(chosen_min_cluster_size, raw objective mean there)``.
    The objective is set by ``ClusteringOptimizationConfig.grid_objective`` (default: the
    clipped geometric mean of DBCV and ARI).

    Prefer :func:`pooled_grid_solve_from_metrics_dfs` when the caller also plots the grid —
    it returns the full curve, and dropping it is what leaves the objective panel blank.
    """
    solved = pooled_grid_solve_from_metrics_dfs(metrics_dfs, optimization_config)
    return solved.chosen_min_cluster_size, solved.score_mean_at_chosen


def grid_solver_config(
    optimization_config: ClusteringOptimizationConfig | None,
    *,
    grid_cluster_count_reward: float | None = None,
    grid_n_entities: int | None = None,
    grid_objective: GridObjectiveSpec | None = None,
    grid_one_se_k: float | None = None,
) -> ClusteringOptimizationConfig:
    """Merge optional solver overrides into ``optimization_config`` (or defaults)."""
    base = optimization_config or ClusteringOptimizationConfig()
    overrides: dict[str, object] = {}
    if grid_cluster_count_reward is not None:
        overrides["grid_cluster_count_reward"] = grid_cluster_count_reward
    if grid_n_entities is not None:
        overrides["grid_n_entities"] = grid_n_entities
    if grid_objective is not None:
        overrides["grid_objective"] = grid_objective
    if grid_one_se_k is not None:
        overrides["grid_one_se_k"] = grid_one_se_k
    return replace(base, **overrides) if overrides else base


def grid_solver_overrides_active(
    *,
    optimization_config: ClusteringOptimizationConfig | None,
    grid_cluster_count_reward: float | None,
    grid_n_entities: int | None,
    grid_objective: GridObjectiveSpec | None,
    grid_one_se_k: float | None,
) -> bool:
    """True when any solver knob was supplied, i.e. a re-solve was explicitly requested."""
    return any(
        v is not None
        for v in (
            optimization_config,
            grid_cluster_count_reward,
            grid_n_entities,
            grid_objective,
            grid_one_se_k,
        )
    )


def should_resolve_chosen_min_cluster_size(
    *,
    chosen_min_cluster_size: float | None,
    optimization_config: ClusteringOptimizationConfig | None,
    grid_cluster_count_reward: float | None,
    grid_n_entities: int | None,
    grid_objective: GridObjectiveSpec | None,
    grid_one_se_k: float | None,
) -> bool:
    """Re-solve only when no chosen value is on hand *and* solver knobs were supplied."""
    if chosen_min_cluster_size is not None:
        return False
    return grid_solver_overrides_active(
        optimization_config=optimization_config,
        grid_cluster_count_reward=grid_cluster_count_reward,
        grid_n_entities=grid_n_entities,
        grid_objective=grid_objective,
        grid_one_se_k=grid_one_se_k,
    )


def solve_pooled_grid_from_metrics_list(
    metrics_list: list[pd.DataFrame],
    optimization_config: ClusteringOptimizationConfig | None = None,
    *,
    grid_cluster_count_reward: float | None = None,
    grid_n_entities: int | None = None,
    grid_objective: GridObjectiveSpec | None = None,
    grid_one_se_k: float | None = None,
) -> SmoothedGridOptimumResult:
    """Pooled grid solve on per-sample metric tables; returns full diagnostics."""
    cfg = grid_solver_config(
        optimization_config,
        grid_cluster_count_reward=grid_cluster_count_reward,
        grid_n_entities=grid_n_entities,
        grid_objective=grid_objective,
        grid_one_se_k=grid_one_se_k,
    )
    return pooled_grid_solve_from_metrics_dfs(metrics_list, cfg)


def solve_pooled_grid_by_combo_from_grid(
    df_grid: pd.DataFrame,
    optimization_config: ClusteringOptimizationConfig | None = None,
    *,
    grid_cluster_count_reward: float | None = None,
    grid_n_entities: int | None = None,
    grid_objective: GridObjectiveSpec | None = None,
    grid_one_se_k: float | None = None,
) -> dict[tuple[str, str], SmoothedGridOptimumResult]:
    """Pooled grid solve per (model, layer) from a grid export CSV frame."""
    cfg = grid_solver_config(
        optimization_config,
        grid_cluster_count_reward=grid_cluster_count_reward,
        grid_n_entities=grid_n_entities,
        grid_objective=grid_objective,
        grid_one_se_k=grid_one_se_k,
    )
    return {
        combo: pooled_grid_solve_from_metrics_dfs(metrics_list, cfg)
        for combo, metrics_list in per_combo_metrics_from_grid(df_grid).items()
    }


def resolve_chosen_min_cluster_size_by_combo_from_grid(
    df_grid: pd.DataFrame,
    optimization_config: ClusteringOptimizationConfig | None = None,
    *,
    grid_cluster_count_reward: float | None = None,
    grid_n_entities: int | None = None,
    grid_objective: GridObjectiveSpec | None = None,
    grid_one_se_k: float | None = None,
) -> dict[tuple[str, str], int]:
    """Re-solve ``chosen_min_cluster_size`` per (model, layer) from a grid export CSV frame."""
    solved = solve_pooled_grid_by_combo_from_grid(
        df_grid,
        optimization_config,
        grid_cluster_count_reward=grid_cluster_count_reward,
        grid_n_entities=grid_n_entities,
        grid_objective=grid_objective,
        grid_one_se_k=grid_one_se_k,
    )
    return {combo: result.chosen_min_cluster_size for combo, result in solved.items()}
