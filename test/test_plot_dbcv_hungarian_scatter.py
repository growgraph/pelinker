"""Smoke tests for DBCV vs Hungarian grid scatter."""

from __future__ import annotations

import pathlib

import pandas as pd

from pelinker.plotting import (
    GRID_COL_SAMPLE_BEST_DBCV,
    GRID_COL_SAMPLE_HUNGARIAN,
    _layer_spec_code,
    plot_dbcv_vs_hungarian_from_grid,
)


def test_layer_spec_code() -> None:
    assert _layer_spec_code("m1", "2") == "2"
    assert _layer_spec_code("fusion2", "bluebert/2+pubmedbert/3") == "2+3"
    assert _layer_spec_code("fusion3", "a/2+b/2+c/3") == "2+2+3"


def test_plot_dbcv_vs_hungarian_from_grid_writes_file(tmp_path: pathlib.Path) -> None:
    rows = []
    for sample_idx in (0, 1):
        for mcs in (10, 20):
            rows.append(
                {
                    "model": "m1",
                    "layer": "2",
                    "sample_idx": sample_idx,
                    "min_cluster_size": mcs,
                    "icm": 0.1,
                    "n_clusters": 3,
                    "dbcv": 0.2 + mcs * 0.01 + sample_idx * 0.05,
                    GRID_COL_SAMPLE_BEST_DBCV: 0.5 + sample_idx * 0.02,
                    GRID_COL_SAMPLE_HUNGARIAN: 0.7 + sample_idx * 0.03,
                }
            )
    df = pd.DataFrame(rows)
    out = tmp_path / "s.png"
    assert plot_dbcv_vs_hungarian_from_grid(df, out) is True
    assert out.is_file()


def test_plot_dbcv_vs_hungarian_fusion_multicolor(tmp_path: pathlib.Path) -> None:
    """Pair / triple fusion rows exercise custom square and wedge markers."""
    rows: list[dict[str, object]] = []
    for sample_idx in (0, 1):
        rows.append(
            {
                "model": "fusion2",
                "layer": "bluebert/2+pubmedbert/3",
                "sample_idx": sample_idx,
                "min_cluster_size": 20,
                "dbcv": 0.4,
                GRID_COL_SAMPLE_BEST_DBCV: 0.5 + sample_idx * 0.01,
                GRID_COL_SAMPLE_HUNGARIAN: 0.75 + sample_idx * 0.02,
            }
        )
        rows.append(
            {
                "model": "fusion3",
                "layer": "bluebert/2+pubmedbert/2+pubmedbert/3",
                "sample_idx": sample_idx,
                "min_cluster_size": 20,
                "dbcv": 0.4,
                GRID_COL_SAMPLE_BEST_DBCV: 0.55 + sample_idx * 0.01,
                GRID_COL_SAMPLE_HUNGARIAN: 0.72 + sample_idx * 0.02,
            }
        )
    df = pd.DataFrame(rows)
    out = tmp_path / "fusion.png"
    assert plot_dbcv_vs_hungarian_from_grid(df, out) is True
    assert out.is_file()


def test_plot_dbcv_vs_hungarian_missing_columns_skips(tmp_path: pathlib.Path) -> None:
    df = pd.DataFrame(
        {
            "model": ["a"],
            "layer": ["1"],
            "sample_idx": [0],
            "dbcv": [0.1],
        }
    )
    out = tmp_path / "x.png"
    assert plot_dbcv_vs_hungarian_from_grid(df, out) is False
    assert not out.exists()
