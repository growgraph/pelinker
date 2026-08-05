"""Smoke test for dim-selection runner with mocked evaluate_selection_sample."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from pelinker.dim_selection.grids import cell_key
from pelinker.dim_selection.runner import run_dim_selection
from pelinker.dim_selection.summary import (
    DIM_SELECTION_RESULTS_CSV_BASENAME,
    DIM_SELECTION_SUMMARY_JSON_BASENAME,
)
from pelinker.reporting import (
    ClusteringHyperparameters,
    ModelSelectionReport,
    entity_negative_label_mask_01,
)
from pelinker.transform import TransformConfig


def _metrics_for_score(dbcv: float, ari: float = 0.7) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "min_cluster_size": [10, 20, 30],
            "icm": [0.1, 0.1, 0.1],
            "n_clusters": [4, 3, 2],
            "dbcv": [dbcv - 0.05, dbcv, dbcv - 0.02],
            "ari": [ari - 0.05, ari, ari - 0.01],
        }
    )


def _fake_report(transform_config: TransformConfig) -> ModelSelectionReport:
    # Prefer pca=80, umap=6 for outer ranking.
    pca = int(transform_config.pca_components)
    umap = int(transform_config.umap_components)
    dbcv = 0.3 + 0.001 * pca - 0.01 * abs(umap - 6)
    if pca == 80 and umap == 6:
        dbcv = 0.9
    ent_df = pd.DataFrame({"entity": ["p1"], "cluster": [0]})
    y = entity_negative_label_mask_01(ent_df["entity"], "__negative__")
    z = np.zeros((1, max(2, umap)), dtype=np.float64)
    return ModelSelectionReport(
        hyperparameters=ClusteringHyperparameters(min_cluster_size=20),
        best_score=dbcv,
        number_properties=1,
        n_clusters_emergent=3,
        metrics_df=_metrics_for_score(dbcv),
        assignments=ent_df,
        pca_residuals=np.array([0.1], dtype=np.float64),
        pca_mahalanobis=np.array([0.2], dtype=np.float64),
        pca_spectral_entropy=np.array([0.3], dtype=np.float64),
        oov_label=y,
        umap_clustering=z[:, :umap] if umap <= z.shape[1] else z,
        cluster_viz=z[:, : min(3, umap)],
        cluster_viz_method="pca",
        pca_reduced=np.zeros((1, pca), dtype=np.float64),
        ari=0.75,
    )


def _fake_evaluate(frame, transform_config, **kwargs):
    report = _fake_report(transform_config)
    all_metrics_dfs = kwargs.get("all_metrics_dfs")
    if all_metrics_dfs is not None:
        all_metrics_dfs.append(report.metrics_df)
    return report


def test_run_dim_selection_coarse_then_refine(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parquet = tmp_path / "res_toy_1.parquet"
    parquet.write_bytes(b"")
    report_path = tmp_path / "reports"
    base = pd.DataFrame({"entity": ["a"], "embed": [np.zeros(4)]})

    monkeypatch.setattr(
        "pelinker.dim_selection.runner.load_selection_frame",
        lambda **_kwargs: base,
    )
    monkeypatch.setattr(
        "pelinker.dim_selection.runner.draw_selection_sample",
        lambda frame, _cfg, sample_index: frame,
    )
    monkeypatch.setattr(
        "pelinker.dim_selection.runner.plot_metrics", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "pelinker.dim_selection.runner.plot_metrics_with_error_bars",
        lambda *_a, **_k: None,
    )
    # Avoid matplotlib write noise for summary figures in CI.
    monkeypatch.setattr(
        "pelinker.dim_selection.summary.write_dim_heatmaps",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "pelinker.dim_selection.summary.write_dim_outer_surface",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "pelinker.dim_selection.summary.write_dim_metrics_violin",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "pelinker.dim_selection.summary.write_dim_dbcv_vs_ari",
        lambda *_a, **_k: [],
    )

    evaluated: list[tuple[int, int]] = []

    def tracking_evaluate(frame, transform_config, **kwargs):
        evaluated.append(
            (
                int(transform_config.pca_components),
                int(transform_config.umap_components),
            )
        )
        return _fake_evaluate(frame, transform_config, **kwargs)

    monkeypatch.setattr(
        "pelinker.dim_selection.runner.evaluate_selection_sample",
        tracking_evaluate,
    )

    run_dim_selection(
        input_parquet=parquet,
        report_path=report_path,
        pca_grid=(40, 80),
        umap_grid=(4, 6),
        refine=True,
        n_sample=1,
        resume=False,
        model="toy",
        layer="1",
    )

    # Coarse: 4 cells; refine around (80,6) adds neighbors not already done.
    assert (40, 4) in evaluated
    assert (80, 6) in evaluated
    assert len(evaluated) > 4

    results = pd.read_csv(report_path / DIM_SELECTION_RESULTS_CSV_BASENAME)
    assert "pca_components" in results.columns
    assert "umap_dim" in results.columns
    assert "best_score" in results.columns

    import json

    summary = json.loads(
        (report_path / DIM_SELECTION_SUMMARY_JSON_BASENAME).read_text(encoding="utf-8")
    )
    assert summary["chosen"]["pca_components"] == 80
    assert summary["chosen"]["umap_dim"] == 6
    assert "outer_score" in summary["chosen"]
    assert "mcs" in summary["metrics"]
    assert "inner_min_cluster_size" in summary["metrics"]
    assert "outer_candidate_ranking" in summary["metrics"]

    # Resume skips completed cells.
    evaluated.clear()
    run_dim_selection(
        input_parquet=parquet,
        report_path=report_path,
        pca_grid=(40, 80),
        umap_grid=(4, 6),
        refine=True,
        n_sample=1,
        resume=True,
        model="toy",
        layer="1",
    )
    assert evaluated == []


def test_fingerprint_mismatch_aborts(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parquet = tmp_path / "res_toy_1.parquet"
    parquet.write_bytes(b"")
    report_path = tmp_path / "reports2"
    base = pd.DataFrame({"entity": ["a"], "embed": [np.zeros(4)]})

    monkeypatch.setattr(
        "pelinker.dim_selection.runner.load_selection_frame",
        lambda **_kwargs: base,
    )
    monkeypatch.setattr(
        "pelinker.dim_selection.runner.draw_selection_sample",
        lambda frame, _cfg, sample_index: frame,
    )
    monkeypatch.setattr(
        "pelinker.dim_selection.runner.evaluate_selection_sample",
        _fake_evaluate,
    )
    monkeypatch.setattr(
        "pelinker.dim_selection.runner.plot_metrics", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "pelinker.dim_selection.summary.write_dim_heatmaps",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "pelinker.dim_selection.summary.write_dim_outer_surface",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "pelinker.dim_selection.summary.write_dim_metrics_violin",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "pelinker.dim_selection.summary.write_dim_dbcv_vs_ari",
        lambda *_a, **_k: [],
    )

    run_dim_selection(
        input_parquet=parquet,
        report_path=report_path,
        pca_grid=(40,),
        umap_grid=(4,),
        refine=False,
        n_sample=1,
        resume=False,
        model="toy",
        layer="1",
    )
    assert cell_key(40, 4)  # sanity

    # Changing grid invalidates fingerprint → early return, results unchanged.
    before = (report_path / DIM_SELECTION_RESULTS_CSV_BASENAME).read_text(
        encoding="utf-8"
    )
    run_dim_selection(
        input_parquet=parquet,
        report_path=report_path,
        pca_grid=(40, 80),
        umap_grid=(4,),
        refine=False,
        n_sample=1,
        resume=True,
        model="toy",
        layer="1",
    )
    after = (report_path / DIM_SELECTION_RESULTS_CSV_BASENAME).read_text(
        encoding="utf-8"
    )
    assert before == after
