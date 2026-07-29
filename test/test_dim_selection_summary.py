"""Tests for dim-selection summary figures (PNG+PDF, surface, violin, scatter)."""

from __future__ import annotations

import json
import pathlib

import pandas as pd

from pelinker.dim_selection.summary import (
    DIM_SELECTION_ARI_HEATMAP_STEM,
    DIM_SELECTION_DBCV_HEATMAP_STEM,
    DIM_SELECTION_DBCV_VS_ARI_STEM,
    DIM_SELECTION_METRICS_VIOLIN_STEM,
    DIM_SELECTION_OUTER_HEATMAP_STEM,
    DIM_SELECTION_OUTER_SURFACE_STEM,
    DIM_SELECTION_SUMMARY_JSON_BASENAME,
    render_dim_selection_summary,
)


def _results_df() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for pca in (40, 80):
        for umap in (4, 6):
            dbcv = 0.5 + 0.001 * pca - 0.01 * abs(umap - 6)
            if pca == 80 and umap == 6:
                dbcv = 0.9
            rows.append(
                {
                    "model": "toy",
                    "layer": "1",
                    "pca_components": pca,
                    "umap_dim": umap,
                    "best_score": dbcv,
                    "best_score_std": 0.02,
                    "ari": 0.6 + 0.05 * (1 if umap == 6 else 0),
                    "ari_std": 0.01,
                    "best_size": 20.0,
                    "best_size_std": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _grid_df(*, n_sample: int = 3) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for pca in (40, 80):
        for umap in (4, 6):
            layer = f"1:pca{pca}:umap{umap}"
            chosen = 20
            for sample_idx in range(n_sample):
                for mcs in (10, 20, 30):
                    dbcv = 0.4 + 0.001 * pca + 0.01 * sample_idx
                    if mcs == chosen:
                        dbcv += 0.1
                    if pca == 80 and umap == 6:
                        dbcv += 0.3
                    rows.append(
                        {
                            "model": "toy",
                            "layer": layer,
                            "sample_idx": sample_idx,
                            "chosen_min_cluster_size": chosen,
                            "min_cluster_size": mcs,
                            "icm": 0.1,
                            "n_clusters": 3,
                            "dbcv": dbcv,
                            "ari": 0.55 + 0.02 * sample_idx + (0.1 if umap == 6 else 0),
                            "pca_components": pca,
                            "umap_dim": umap,
                        }
                    )
    return pd.DataFrame(rows)


def test_render_dim_selection_summary_writes_png_pdf_and_extra_figures(
    tmp_path: pathlib.Path,
) -> None:
    report_path = tmp_path / "report"
    grid_path = report_path / "results_grid_per_sample.csv"
    report_path.mkdir(parents=True)
    grid_df = _grid_df(n_sample=3)
    grid_df.to_csv(grid_path, index=False)

    payload = render_dim_selection_summary(
        _results_df(),
        report_path,
        model="toy",
        layer="1",
        n_sample=3,
        refine=True,
        grid_csv_path=grid_path,
    )

    assert payload["chosen"]["pca_components"] == 80
    assert payload["chosen"]["umap_dim"] == 6
    assert "figures" in payload
    assert "heatmaps" in payload

    for stem in (
        DIM_SELECTION_OUTER_HEATMAP_STEM,
        DIM_SELECTION_DBCV_HEATMAP_STEM,
        DIM_SELECTION_ARI_HEATMAP_STEM,
        DIM_SELECTION_OUTER_SURFACE_STEM,
        DIM_SELECTION_METRICS_VIOLIN_STEM,
        DIM_SELECTION_DBCV_VS_ARI_STEM,
    ):
        assert (report_path / f"{stem}.png").is_file(), stem
        assert (report_path / f"{stem}.pdf").is_file(), stem
        assert f"{stem}.png" in payload["figures"]
        assert f"{stem}.pdf" in payload["figures"]

    for stem in (
        DIM_SELECTION_OUTER_HEATMAP_STEM,
        DIM_SELECTION_DBCV_HEATMAP_STEM,
        DIM_SELECTION_ARI_HEATMAP_STEM,
    ):
        assert f"{stem}.png" in payload["heatmaps"]
        assert f"{stem}.pdf" in payload["heatmaps"]

    summary = json.loads(
        (report_path / DIM_SELECTION_SUMMARY_JSON_BASENAME).read_text(encoding="utf-8")
    )
    assert summary["figures"] == payload["figures"]


def test_render_dim_selection_summary_skips_violin_when_n_sample_lt_2(
    tmp_path: pathlib.Path,
) -> None:
    report_path = tmp_path / "report_one"
    grid_path = report_path / "results_grid_per_sample.csv"
    report_path.mkdir(parents=True)
    _grid_df(n_sample=1).to_csv(grid_path, index=False)

    payload = render_dim_selection_summary(
        _results_df(),
        report_path,
        model="toy",
        layer="1",
        n_sample=1,
        refine=False,
        grid_csv_path=grid_path,
    )

    assert not (report_path / f"{DIM_SELECTION_METRICS_VIOLIN_STEM}.png").exists()
    assert f"{DIM_SELECTION_METRICS_VIOLIN_STEM}.png" not in payload["figures"]
    # Heatmaps + surface still written.
    assert (report_path / f"{DIM_SELECTION_OUTER_HEATMAP_STEM}.png").is_file()
    assert (report_path / f"{DIM_SELECTION_OUTER_HEATMAP_STEM}.pdf").is_file()
    assert (report_path / f"{DIM_SELECTION_OUTER_SURFACE_STEM}.png").is_file()
