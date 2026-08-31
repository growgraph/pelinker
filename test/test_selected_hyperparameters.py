"""Typed search→fit handoff: schema, round-trip, and override precedence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pelinker.cli.fit import (
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_UMAP_DIM,
    FitCliConfig,
    _resolve_selection_hyperparameters,
)
from pelinker.search.selected_hyperparameters import (
    SELECTED_HYPERPARAMETERS_BASENAME,
    SELECTED_HYPERPARAMETERS_SCHEMA,
    SelectedHyperparameters,
    load_selected_hyperparameters,
    write_selected_hyperparameters,
)


def _selected(**over) -> SelectedHyperparameters:
    base = dict(
        source="dim_selection",
        model="pubmedbert",
        layer="2",
        pca_components=22,
        umap_dim=3,
        min_cluster_size=20,
        manifold_kind="umap",
        n_rows_realized=48_000,
        umap_n_neighbors=None,
        run_fingerprint="abc123",
        outer_score=0.69,
    )
    base.update(over)
    return SelectedHyperparameters(**base)


def _cfg(**over) -> FitCliConfig:
    base = dict(kb_path="kb.csv", embeddings_parquet="x.parquet")
    base.update(over)
    return FitCliConfig(**base)


def test_handoff_round_trips_through_disk(tmp_path: Path) -> None:
    out = write_selected_hyperparameters(_selected(), tmp_path)

    assert out.name == SELECTED_HYPERPARAMETERS_BASENAME
    assert load_selected_hyperparameters(out) == _selected()
    # A directory resolves to the file inside it.
    assert load_selected_hyperparameters(tmp_path) == _selected()


def test_handoff_rejects_an_unknown_schema(tmp_path: Path) -> None:
    p = tmp_path / SELECTED_HYPERPARAMETERS_BASENAME
    p.write_text(json.dumps({"schema": "pelinker.search.selected_hyperparameters.v99"}))

    with pytest.raises(ValueError, match="expected schema"):
        load_selected_hyperparameters(p)


def test_handoff_validates_its_fields() -> None:
    with pytest.raises(ValueError, match="min_cluster_size must be >= 2"):
        _selected(min_cluster_size=1)
    with pytest.raises(ValueError, match="umap_dim must be >= 2"):
        _selected(umap_dim=1)
    with pytest.raises(ValueError, match="manifold_kind"):
        _selected(manifold_kind="nonsense")
    with pytest.raises(ValueError, match="model must be"):
        _selected(model="")


def test_written_payload_carries_the_schema(tmp_path: Path) -> None:
    write_selected_hyperparameters(_selected(), tmp_path)

    payload = json.loads((tmp_path / SELECTED_HYPERPARAMETERS_BASENAME).read_text())
    assert payload["schema"] == SELECTED_HYPERPARAMETERS_SCHEMA
    assert payload["n_rows_realized"] == 48_000


# ------------------------------------------------------------------- fit precedence


def test_fit_falls_back_to_documented_defaults_without_a_report() -> None:
    r = _resolve_selection_hyperparameters(_cfg())

    assert r.pca_components == DEFAULT_PCA_COMPONENTS == 100
    assert r.umap_dim == DEFAULT_UMAP_DIM == 8
    assert r.min_cluster_size is None  # left for the scale curve / Linker default
    assert r.umap_n_neighbors is None


def test_fit_takes_dims_from_the_selection_report(tmp_path: Path) -> None:
    """The exact case that motivated this: pca=22/umap=3 chosen, 100/8 used."""
    write_selected_hyperparameters(_selected(), tmp_path)

    r = _resolve_selection_hyperparameters(_cfg(selection_report=str(tmp_path)))

    assert r.pca_components == 22
    assert r.umap_dim == 3
    assert r.min_cluster_size == 20


def test_explicit_values_override_the_report(tmp_path: Path) -> None:
    write_selected_hyperparameters(_selected(), tmp_path)

    r = _resolve_selection_hyperparameters(
        _cfg(selection_report=str(tmp_path), pca_components=64, min_cluster_size=35)
    )

    assert r.pca_components == 64
    assert r.min_cluster_size == 35
    # Unset fields still come from the report.
    assert r.umap_dim == 3


def test_manifold_mismatch_between_search_and_fit_is_warned(
    tmp_path: Path, caplog
) -> None:
    """Search on standard UMAP + compact fit means MCS crosses coordinate systems."""
    import logging

    write_selected_hyperparameters(_selected(manifold_kind="umap"), tmp_path)

    with caplog.at_level(logging.WARNING, logger="pelinker.cli.fit"):
        _resolve_selection_hyperparameters(
            _cfg(selection_report=str(tmp_path), predict_mode="compact")
        )

    assert "different coordinates" in caplog.text


def test_matching_manifold_produces_no_warning(tmp_path: Path, caplog) -> None:
    import logging

    write_selected_hyperparameters(_selected(manifold_kind="umap"), tmp_path)

    with caplog.at_level(logging.WARNING, logger="pelinker.cli.fit"):
        _resolve_selection_hyperparameters(
            _cfg(selection_report=str(tmp_path), predict_mode="legacy")
        )

    assert "different coordinates" not in caplog.text
