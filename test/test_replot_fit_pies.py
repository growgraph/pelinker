"""Tests for replot_fit pie grid helpers."""

from __future__ import annotations

import pathlib

import pandas as pd

from run.analysis.replot_fit import (
    _PIE_SAMPLE_MAX_CLUSTERS,
    _pie_grid_layout,
    _pie_grid_layout_sample,
    plot_pie_grid,
)


def _composition_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cluster in range(8):
        rows.append(
            {
                "cluster": cluster,
                "entity": f"e{cluster}a",
                "count": 10.0 - cluster,
            }
        )
        rows.append(
            {
                "cluster": cluster,
                "entity": f"e{cluster}b",
                "count": 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_pie_grid_layout_scales_with_cluster_count() -> None:
    assert _pie_grid_layout(6)[0] == 3
    assert _pie_grid_layout(30)[0] == 6
    assert _pie_grid_layout_sample(6)[0] == 3


def test_plot_pie_grid_writes_full_and_sample(tmp_path: pathlib.Path) -> None:
    df = _composition_df()
    full = plot_pie_grid(df, save_dir=tmp_path)
    sample = plot_pie_grid(
        df,
        save_dir=tmp_path,
        max_clusters=_PIE_SAMPLE_MAX_CLUSTERS,
        filename_stem="fit_cluster_composition_pies_sample",
        layout_fn=_pie_grid_layout_sample,
    )
    assert len(full) == 2
    assert len(sample) == 2
    assert (tmp_path / "fit_cluster_composition_pies.png").is_file()
    assert (tmp_path / "fit_cluster_composition_pies_sample.png").is_file()
    assert full[0].stat().st_size > sample[0].stat().st_size
