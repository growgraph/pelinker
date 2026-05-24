"""Tests for per-sample grid CSV export and selection at chosen hyperparameter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import pathlib

import json

from pelinker.clustering_grid import SmoothedGridOptimumResult
from pelinker.config import ClusteringOptimizationConfig
from pelinker.grid_export import (
    GRID_COL_CHOSEN_MIN_CLUSTER_SIZE,
    apply_chosen_min_cluster_size_to_grid,
    grid_chosen_hyperparameters_to_jsonable,
    grid_export_rows_from_report,
    select_grid_points_at_chosen_min_cluster_size,
    write_grid_chosen_hyperparameters,
)
from pelinker.reporting import (
    ClusteringHyperparameters,
    ModelSelectionReport,
    entity_negative_label_mask_01,
)


def _minimal_report(
    metrics_df: pd.DataFrame,
    *,
    min_cluster_size: int = 10,
) -> ModelSelectionReport:
    ent_df = pd.DataFrame({"entity": ["p1"], "cluster": [0]})
    y = entity_negative_label_mask_01(ent_df["entity"], "__negative__")
    z = np.zeros((1, 2), dtype=np.float64)

    return ModelSelectionReport(
        hyperparameters=ClusteringHyperparameters(min_cluster_size=min_cluster_size),
        best_score=0.5,
        number_properties=1,
        n_clusters_emergent=2,
        metrics_df=metrics_df,
        assignments=ent_df,
        pca_residuals=np.array([0.1], dtype=np.float64),
        pca_mahalanobis=np.array([0.2], dtype=np.float64),
        pca_spectral_entropy=np.array([0.3], dtype=np.float64),
        oov_label=y,
        umap_clustering=z,
        umap_visualization=z,
        pca_reduced=z,
        ari=0.8,
    )


def test_grid_export_rows_from_report_sets_chosen_mcs() -> None:
    metrics = pd.DataFrame(
        {
            "min_cluster_size": [10, 20],
            "icm": [0.1, 0.1],
            "n_clusters": [3, 3],
            "dbcv": [0.4, 0.55],
            "ari": [0.7, 0.75],
        }
    )
    report = _minimal_report(metrics)
    out = grid_export_rows_from_report(
        report,
        model="m",
        layer="L",
        sample_idx=0,
        chosen_min_cluster_size=20,
    )
    assert (out[GRID_COL_CHOSEN_MIN_CLUSTER_SIZE] == 20).all()
    assert "sample_best_dbcv" not in out.columns


def test_write_grid_chosen_hyperparameters(tmp_path: pathlib.Path) -> None:
    solved = SmoothedGridOptimumResult(
        chosen_min_cluster_size=30,
        score_mean_at_chosen=0.7,
        score_std_at_chosen=0.01,
        n_clusters_mean_at_chosen=84.0,
        x=(20.0, 30.0, 40.0),
        y_objective=(0.5, 0.6, 0.55),
        y_cluster_term=(-0.1, 0.0, -0.05),
        y_smooth=(0.52, 0.58, 0.56),
        dy_dx=(0.01, 0.0, -0.01),
        d2y_dx2=(0.0, 0.0, 0.0),
        selection="plateau_derivative",
    )
    cfg = ClusteringOptimizationConfig(grid_cluster_count_reward=0.05)
    path = tmp_path / "grid_chosen_hyperparameters.json"
    write_grid_chosen_hyperparameters(path, {("m1", "2"): solved}, cfg)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "pelinker.grid_chosen_hyperparameters.v1"
    assert payload["combinations"][0]["chosen_min_cluster_size"] == 30
    assert payload["combinations"][0]["grid_curve"]["y_objective"] == [0.5, 0.6, 0.55]


def test_grid_chosen_hyperparameters_to_jsonable_sorted() -> None:
    solved_a = SmoothedGridOptimumResult(
        chosen_min_cluster_size=20,
        score_mean_at_chosen=0.5,
        score_std_at_chosen=0.0,
        n_clusters_mean_at_chosen=10.0,
        x=(20.0,),
        y_objective=(0.5,),
        y_cluster_term=(0.0,),
        y_smooth=(0.5,),
        dy_dx=(0.0,),
        d2y_dx2=(0.0,),
        selection="smoothed_argmax",
    )
    solved_b = SmoothedGridOptimumResult(
        chosen_min_cluster_size=25,
        score_mean_at_chosen=0.6,
        score_std_at_chosen=0.0,
        n_clusters_mean_at_chosen=12.0,
        x=(25.0,),
        y_objective=(0.6,),
        y_cluster_term=(0.0,),
        y_smooth=(0.6,),
        dy_dx=(0.0,),
        d2y_dx2=(0.0,),
        selection="smoothed_argmax",
    )
    doc = grid_chosen_hyperparameters_to_jsonable(
        {("z", "1"): solved_a, ("a", "2"): solved_b},
        ClusteringOptimizationConfig(),
    )
    models = [c["model"] for c in doc["combinations"]]
    assert models == ["a", "z"]


def test_apply_chosen_min_cluster_size_to_grid() -> None:
    df = pd.DataFrame(
        {
            "model": ["m1", "m1", "m2"],
            "layer": ["a", "a", "b"],
            GRID_COL_CHOSEN_MIN_CLUSTER_SIZE: [10, 10, 30],
            "min_cluster_size": [10, 20, 30],
        }
    )
    out = apply_chosen_min_cluster_size_to_grid(df, {("m1", "a"): 25, ("m2", "b"): 35})
    assert out.loc[out["model"] == "m1", GRID_COL_CHOSEN_MIN_CLUSTER_SIZE].tolist() == [
        25,
        25,
    ]
    assert (
        int(out.loc[out["model"] == "m2", GRID_COL_CHOSEN_MIN_CLUSTER_SIZE].iloc[0])
        == 35
    )


def test_select_grid_points_at_chosen_min_cluster_size() -> None:
    rows = []
    for sample_idx in (0, 1):
        for mcs in (10, 20):
            rows.append(
                {
                    "model": "m1",
                    "layer": "2",
                    "sample_idx": sample_idx,
                    GRID_COL_CHOSEN_MIN_CLUSTER_SIZE: 20,
                    "min_cluster_size": mcs,
                    "n_clusters": 3,
                    "dbcv": 0.1 + mcs * 0.01 + sample_idx * 0.05,
                    "ari": 0.6 + mcs * 0.01 + sample_idx * 0.02,
                }
            )
    df = pd.DataFrame(rows)
    pts = select_grid_points_at_chosen_min_cluster_size(df)
    assert len(pts) == 2
    assert pts["min_cluster_size"].eq(20).all()
    assert pts.loc[pts["sample_idx"] == 0, "dbcv"].iloc[0] == pytest.approx(0.3)
    assert pts.loc[pts["sample_idx"] == 1, "ari"].iloc[0] == pytest.approx(0.82)


def test_select_skips_n_clusters_le_one() -> None:
    df = pd.DataFrame(
        {
            "model": ["m"],
            "layer": ["L"],
            "sample_idx": [0],
            GRID_COL_CHOSEN_MIN_CLUSTER_SIZE: [15],
            "min_cluster_size": [15],
            "n_clusters": [1],
            "dbcv": [0.5],
            "ari": [0.9],
        }
    )
    assert select_grid_points_at_chosen_min_cluster_size(df).empty


def test_select_missing_chosen_mcs_yields_empty() -> None:
    df = pd.DataFrame(
        {
            "model": ["m"],
            "layer": ["L"],
            "sample_idx": [0],
            GRID_COL_CHOSEN_MIN_CLUSTER_SIZE: [float("nan")],
            "min_cluster_size": [10],
            "n_clusters": [3],
            "dbcv": [0.5],
            "ari": [0.9],
        }
    )
    assert select_grid_points_at_chosen_min_cluster_size(df).empty
