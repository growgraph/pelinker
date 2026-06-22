"""Tests for per-combination PCA quality pair grid export."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from pelinker.plotting import (
    _balanced_subsample_by_class,
    plot_pca_quality_pairgrid,
)
from pelinker.model_selection.fine_metadata import (
    fine_metadata_one_sample_per_combo,
    pca_pairgrid_output_path,
    safe_combo_plot_stem,
    write_pca_quality_pairgrids_from_fine_metadata,
)


def _fine_metadata_frame(
    *,
    model: str = "m1",
    layer: str = "L2",
    sample_idx: int = 0,
    n_per_class: int = 4,
) -> pd.DataFrame:
    n = n_per_class * 2
    oov = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.int64)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "model": [model] * n,
            "layer": [layer] * n,
            "sample_idx": [sample_idx] * n,
            "entity": [f"e{i}" for i in range(n)],
            "cluster": list(range(n)),
            "oov_label": oov,
            "pca_residual": rng.uniform(0.0, 1.0, n),
            "pca_mahalanobis": rng.uniform(0.0, 1.0, n),
            "pca_spectral_entropy": rng.uniform(0.0, 1.0, n),
        }
    )


def test_balanced_subsample_equal_per_class() -> None:
    n_pos, n_neg = 10_000, 50
    df = pd.DataFrame(
        {
            "class_label": ["positive"] * n_pos + ["negative"] * n_neg,
            "x": range(n_pos + n_neg),
        }
    )
    got = _balanced_subsample_by_class(df, 400, "class_label")
    counts = got["class_label"].value_counts()
    assert int(counts["positive"]) == 200
    assert int(counts["negative"]) == 50


def test_safe_combo_plot_stem_escapes_layer() -> None:
    assert safe_combo_plot_stem("fusion2", "a/2+b/3") == "fusion2_a_2__b_3"


def test_pca_pairgrid_output_path() -> None:
    path = pca_pairgrid_output_path(
        pathlib.Path("/tmp/out"),
        model="m1",
        layer="L2",
        sample_idx=1,
    )
    assert path.name == "m1_L2_sample1_pca_quality_pairgrid.png"


def test_fine_metadata_one_sample_per_combo_keeps_lowest_index() -> None:
    fm = pd.concat(
        [
            _fine_metadata_frame(model="m1", layer="L1", sample_idx=0),
            _fine_metadata_frame(model="m1", layer="L1", sample_idx=1),
            _fine_metadata_frame(model="m2", layer="L2", sample_idx=0),
        ],
        ignore_index=True,
    )
    slim = fine_metadata_one_sample_per_combo(fm)
    assert set(slim["sample_idx"].unique()) == {0}
    assert len(slim) == 16
    assert len(slim) < len(fm)


def test_write_pairgrids_default_one_sample_per_combo(tmp_path: pathlib.Path) -> None:
    fm = pd.concat(
        [
            _fine_metadata_frame(model="m1", layer="L1", sample_idx=0),
            _fine_metadata_frame(model="m1", layer="L1", sample_idx=1),
            _fine_metadata_frame(model="m2", layer="L2", sample_idx=0),
        ],
        ignore_index=True,
    )
    written, skipped = write_pca_quality_pairgrids_from_fine_metadata(
        fm, tmp_path, source_name="test"
    )
    assert skipped == []
    assert len(written) == 2
    assert {p.name for p in written} == {
        "m1_L1_sample0_pca_quality_pairgrid.png",
        "m2_L2_sample0_pca_quality_pairgrid.png",
    }


def test_write_pairgrids_all_samples_flag(tmp_path: pathlib.Path) -> None:
    fm = pd.concat(
        [
            _fine_metadata_frame(model="m1", layer="L1", sample_idx=0),
            _fine_metadata_frame(model="m1", layer="L1", sample_idx=1),
            _fine_metadata_frame(model="m2", layer="L2", sample_idx=0),
        ],
        ignore_index=True,
    )
    written, skipped = write_pca_quality_pairgrids_from_fine_metadata(
        fm, tmp_path, source_name="test", all_samples=True
    )
    assert skipped == []
    assert len(written) == 3
    assert all(p.is_file() for p in written)
    assert {p.name for p in written} == {
        "m1_L1_sample0_pca_quality_pairgrid.png",
        "m1_L1_sample1_pca_quality_pairgrid.png",
        "m2_L2_sample0_pca_quality_pairgrid.png",
    }


def test_write_pairgrids_skips_trivial_oov_mask(tmp_path: pathlib.Path) -> None:
    fm = _fine_metadata_frame()
    fm["oov_label"] = 0
    written, skipped = write_pca_quality_pairgrids_from_fine_metadata(
        fm, tmp_path, source_name="test"
    )
    assert written == []
    assert len(skipped) == 1
    assert "trivial oov_label" in skipped[0]


def test_write_pairgrids_missing_group_columns(tmp_path: pathlib.Path) -> None:
    fm = _fine_metadata_frame().drop(columns=["sample_idx"])
    written, skipped = write_pca_quality_pairgrids_from_fine_metadata(
        fm, tmp_path, source_name="test"
    )
    assert written == []
    assert "sample_idx" in skipped[0]


def test_plot_pca_quality_pairgrid_with_subtitle(tmp_path: pathlib.Path) -> None:
    df = _fine_metadata_frame()
    plot_df = df[
        ["pca_residual", "pca_mahalanobis", "pca_spectral_entropy", "oov_label"]
    ].copy()
    plot_df["class_label"] = np.where(plot_df["oov_label"] == 1, "negative", "positive")
    out = tmp_path / "one.png"
    assert (
        plot_pca_quality_pairgrid(
            plot_df, out, class_col="class_label", subtitle="m1/L1 sample 0"
        )
        is True
    )
    assert out.is_file()
