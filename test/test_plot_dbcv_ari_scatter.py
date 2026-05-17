"""Smoke tests for DBCV vs ARI grid scatter."""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from pelinker.grid_export import GRID_COL_CHOSEN_MIN_CLUSTER_SIZE
from pelinker.plotting import _layer_spec_code, plot_dbcv_vs_ari_from_grid


def test_layer_spec_code() -> None:
    assert _layer_spec_code("m1", "2") == "2"
    assert _layer_spec_code("fusion2", "bluebert/2+pubmedbert/3") == "2,3"
    assert _layer_spec_code("fusion3", "a/2+b/2+c/3") == "2,2,3"


def _grid_df_for_combo(
    *,
    chosen_mcs: int = 20,
    sample_indices: tuple[int, ...] = (0, 1),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_idx in sample_indices:
        for mcs in (10, chosen_mcs):
            rows.append(
                {
                    "model": "m1",
                    "layer": "2",
                    "sample_idx": sample_idx,
                    GRID_COL_CHOSEN_MIN_CLUSTER_SIZE: chosen_mcs,
                    "min_cluster_size": mcs,
                    "icm": 0.1,
                    "n_clusters": 3,
                    "dbcv": 0.2 + mcs * 0.01 + sample_idx * 0.05,
                    "ari": 0.7 + mcs * 0.01 + sample_idx * 0.03,
                }
            )
    return pd.DataFrame(rows)


def test_plot_dbcv_vs_ari_from_grid_writes_file(tmp_path: pathlib.Path) -> None:
    df = _grid_df_for_combo()
    out = tmp_path / "s.png"
    assert plot_dbcv_vs_ari_from_grid(df, out) is True
    assert out.is_file()


def test_plot_dbcv_vs_ari_uses_chosen_mcs_not_other_grid_points(
    tmp_path: pathlib.Path,
) -> None:
    """Scatter must use (dbcv, ari) at chosen_min_cluster_size, not other grid rows."""
    df = _grid_df_for_combo(chosen_mcs=20)
    # If the plot used mcs=10 or arbitrary dedupe, means would differ from chosen-mcs values.
    from pelinker.grid_export import select_grid_points_at_chosen_min_cluster_size

    pts = select_grid_points_at_chosen_min_cluster_size(df)
    expected_x = float(pts["dbcv"].mean())
    expected_y = float(pts["ari"].mean())
    assert expected_x == pytest.approx(0.2 + 20 * 0.01 + 0.05 / 2)
    assert expected_y == pytest.approx(0.7 + 20 * 0.01 + 0.03 / 2)

    out = tmp_path / "s.png"
    assert plot_dbcv_vs_ari_from_grid(df, out) is True


def test_plot_dbcv_vs_ari_fusion_multicolor(tmp_path: pathlib.Path) -> None:
    """Pair / triple fusion rows exercise custom square and wedge markers."""
    rows: list[dict[str, object]] = []
    for sample_idx in (0, 1):
        rows.append(
            {
                "model": "fusion2",
                "layer": "bluebert/2+pubmedbert/3",
                "sample_idx": sample_idx,
                GRID_COL_CHOSEN_MIN_CLUSTER_SIZE: 20,
                "min_cluster_size": 20,
                "n_clusters": 3,
                "dbcv": 0.5 + sample_idx * 0.01,
                "ari": 0.75 + sample_idx * 0.02,
            }
        )
        rows.append(
            {
                "model": "fusion3",
                "layer": "bluebert/2+pubmedbert/2+pubmedbert/3",
                "sample_idx": sample_idx,
                GRID_COL_CHOSEN_MIN_CLUSTER_SIZE: 20,
                "min_cluster_size": 20,
                "n_clusters": 3,
                "dbcv": 0.55 + sample_idx * 0.01,
                "ari": 0.72 + sample_idx * 0.02,
            }
        )
    df = pd.DataFrame(rows)
    out = tmp_path / "fusion.png"
    assert plot_dbcv_vs_ari_from_grid(df, out) is True
    assert out.is_file()


def test_plot_dbcv_vs_ari_missing_columns_skips(tmp_path: pathlib.Path) -> None:
    df = pd.DataFrame(
        {
            "model": ["a"],
            "layer": ["1"],
            "sample_idx": [0],
            "dbcv": [0.1],
        }
    )
    out = tmp_path / "x.png"
    assert plot_dbcv_vs_ari_from_grid(df, out) is False
    assert not out.exists()
