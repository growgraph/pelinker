"""Tests for dim-selection grid helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from pelinker.dim_selection.grids import (
    cell_key,
    cluster_viz_components_for_umap,
    coarse_cells,
    parse_cell_key,
    parse_int_grid,
    pick_winner_row,
    refine_cells,
)


def test_parse_int_grid_csv_and_dedupe() -> None:
    assert parse_int_grid("40, 80, 40, 120", name="pca") == (40, 80, 120)
    assert parse_int_grid([4, 6, 8], name="umap") == (4, 6, 8)


def test_parse_int_grid_rejects_empty_and_nonpositive() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        parse_int_grid("", name="pca")
    with pytest.raises(ValueError, match=">= 1"):
        parse_int_grid("0,8", name="umap")


def test_cell_key_roundtrip() -> None:
    key = cell_key(100, 8)
    assert key == "pca100/umap8"
    assert parse_cell_key(key) == (100, 8)


def test_coarse_cells_cartesian() -> None:
    assert coarse_cells([40, 80], [4, 6]) == [
        (40, 4),
        (40, 6),
        (80, 4),
        (80, 6),
    ]


def test_refine_cells_clips_and_skips_already() -> None:
    cells = refine_cells(80, 6, already={(80, 6), (60, 6)})
    assert (80, 6) not in cells
    assert (60, 6) not in cells
    assert (100, 6) in cells
    assert (80, 4) in cells
    assert (80, 8) in cells
    # UMAP cannot exceed PCA.
    assert all(u <= p for p, u in cells)
    # Clipped floor.
    near_floor = refine_cells(2, 2, already=set())
    assert all(p >= 2 and u >= 2 for p, u in near_floor)


def test_cluster_viz_components_clamped() -> None:
    assert cluster_viz_components_for_umap(8) == 3
    assert cluster_viz_components_for_umap(2) == 2


def test_pick_winner_tie_break() -> None:
    df = pd.DataFrame(
        [
            {
                "pca_components": 120,
                "umap_dim": 8,
                "best_score": 0.5,
                "best_score_std": 0.02,
                "ari": 0.5,
                "ari_std": 0.02,
            },
            {
                "pca_components": 80,
                "umap_dim": 6,
                "best_score": 0.5,
                "best_score_std": 0.01,
                "ari": 0.5,
                "ari_std": 0.01,
            },
            {
                "pca_components": 100,
                "umap_dim": 8,
                "best_score": 0.5,
                "best_score_std": 0.01,
                "ari": 0.5,
                "ari_std": 0.01,
            },
        ]
    )
    winner = pick_winner_row(df)
    # Equal outer after minmax → lower std → then smaller pca → then smaller umap.
    assert int(winner["pca_components"]) == 80
    assert int(winner["umap_dim"]) == 6


def test_pick_winner_prefers_higher_ari_when_dbcv_tied() -> None:
    df = pd.DataFrame(
        [
            {
                "pca_components": 100,
                "umap_dim": 8,
                "best_score": 0.6,
                "best_score_std": 0.0,
                "ari": 0.2,
                "ari_std": 0.0,
            },
            {
                "pca_components": 80,
                "umap_dim": 6,
                "best_score": 0.6,
                "best_score_std": 0.0,
                "ari": 0.9,
                "ari_std": 0.0,
            },
        ]
    )
    winner = pick_winner_row(df)
    assert int(winner["pca_components"]) == 80
    assert int(winner["umap_dim"]) == 6
