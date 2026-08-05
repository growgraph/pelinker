"""Tests for dim-selection checkpoint fingerprint and resume helpers."""

from __future__ import annotations

import json
import pathlib

import pytest

from pelinker.dim_selection.checkpoint import (
    CHECKPOINT_VERSION,
    DimSelectionCheckpoint,
    compute_run_fingerprint,
    fingerprint_config_from_cli,
    load_checkpoint,
    mark_cell_done,
    new_checkpoint,
    save_checkpoint_atomic,
)
from pelinker.dim_selection.grids import cell_key


def test_checkpoint_save_load_roundtrip(tmp_path: pathlib.Path) -> None:
    fp = compute_run_fingerprint({"a": 1, "b": 2})
    ckpt = new_checkpoint(fp)
    key = cell_key(80, 6)
    ckpt.completed_cells.append(key)
    ckpt.summaries_by_key[key] = {
        "model": "m",
        "layer": "1",
        "pca_components": 80,
        "umap_dim": 6,
        "best_score": 0.42,
    }
    path = tmp_path / "dim_selection.state.json.gz"
    save_checkpoint_atomic(path, ckpt)
    loaded = load_checkpoint(path)
    assert loaded.run_fingerprint == fp
    assert loaded.completed_cells == [key]
    assert loaded.summaries_by_key[key]["best_score"] == 0.42


def test_load_wrong_version_raises(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"version": 999, "run_fingerprint": "x"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Unsupported checkpoint version"):
        load_checkpoint(path)


def test_fingerprint_differs_for_grid_and_parquet(tmp_path: pathlib.Path) -> None:
    p1 = tmp_path / "a.parquet"
    p2 = tmp_path / "b.parquet"
    p1.write_bytes(b"")
    p2.write_bytes(b"")
    common = dict(
        model="m",
        layer="1",
        pca_grid=(40, 80),
        umap_grid=(4, 6),
        refine=True,
        cluster_viz_method="pca",
        min_class_size=20,
        seed=13,
        pca_seed=13,
        umap_seed=None,
        clustering_sample_rows=None,
        batch_size=1000,
        n_sample=3,
        selected_labels_kb_path=None,
        max_scale=60,
    )
    fa = compute_run_fingerprint(
        fingerprint_config_from_cli(input_parquet=p1, **common)
    )
    fb = compute_run_fingerprint(
        fingerprint_config_from_cli(input_parquet=p2, **common)
    )
    assert fa != fb
    fc = compute_run_fingerprint(
        fingerprint_config_from_cli(
            input_parquet=p1, **{**common, "pca_grid": (40, 80, 120)}
        )
    )
    assert fa != fc


def test_mark_cell_done_persists(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "st.json"
    ckpt = new_checkpoint(compute_run_fingerprint({"k": 1}))
    assert ckpt.version == CHECKPOINT_VERSION
    key = cell_key(40, 4)
    mark_cell_done(
        ckpt,
        path,
        cell_key=key,
        summary_flat={"model": "m", "layer": "1", "best_score": 0.1},
    )
    loaded = load_checkpoint(path)
    assert key in loaded.completed_cells
    assert loaded.summaries_by_key[key]["best_score"] == 0.1


def test_checkpoint_to_json_dict_sorted(tmp_path: pathlib.Path) -> None:
    ckpt = DimSelectionCheckpoint(
        version=CHECKPOINT_VERSION,
        run_fingerprint="ab",
        summaries_by_key={"z": {"model": "z"}, "a": {"model": "a"}},
    )
    d = ckpt.to_json_dict()
    assert list(d["summaries_by_key"].keys()) == ["a", "z"]
