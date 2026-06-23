"""Regression tests for cluster-entity flow plots."""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from pelinker.plotting import plot_cluster_entity_sankey


def _composition_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cluster in range(4):
        for entity in ("expresses", "binds to", "regulates"):
            rows.append(
                {
                    "cluster": cluster,
                    "entity": entity,
                    "count": float(10 - cluster + hash(entity) % 3),
                }
            )
    return pd.DataFrame(rows)


def test_plot_cluster_entity_sankey_writes_non_blank_figure(
    tmp_path: pathlib.Path,
) -> None:
    written = plot_cluster_entity_sankey(
        _composition_df(),
        save_dir=tmp_path,
        max_clusters=4,
        max_entities=3,
    )
    assert len(written) == 2
    png = tmp_path / "fit_cluster_entity_sankey.png"
    pdf = tmp_path / "fit_cluster_entity_sankey.pdf"
    assert png in written
    assert pdf in written
    # Empty pre-created figures are ~35 KiB at 300 dpi; real Sankey content is much larger.
    assert png.stat().st_size > 80_000


def test_plot_cluster_entity_sankey_scales_small_counts(tmp_path: pathlib.Path) -> None:
    """Tiny masses should still produce a readable figure via min-band scaling."""
    df = pd.DataFrame(
        [
            {"cluster": 0, "entity": "expresses", "count": 0.001},
            {"cluster": 0, "entity": "binds to", "count": 0.002},
            {"cluster": 1, "entity": "expresses", "count": 0.001},
            {"cluster": 1, "entity": "regulates", "count": 0.003},
        ]
    )
    written = plot_cluster_entity_sankey(
        df, save_dir=tmp_path, max_clusters=2, max_entities=3
    )
    png = tmp_path / "fit_cluster_entity_sankey.png"
    assert png in written
    assert png.stat().st_size > 80_000
