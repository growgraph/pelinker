"""Loading persisted bootstrap labels into stability draws."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "run" / "analysis" / "cluster_stability.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("cluster_stability_cli", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cli = _load_module()


def _labels(**overrides) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "pmid": ["p1", "p1", "p2", "p1", "p1", "p2"],
            "itext": [0, 0, 0, 0, 0, 0],
            "a_abs": [10, 40, 10, 10, 40, 10],
            "b_abs": [15, 45, 15, 15, 45, 15],
            "entity": ["x", "y", "x", "x", "y", "x"],
            "cluster": [0, 1, 0, 0, 1, 0],
            "sample_idx": [0, 0, 0, 1, 1, 1],
            "pca_components": [50, 50, 50, 50, 50, 50],
            "umap_dim": [4, 4, 4, 4, 4, 4],
        }
    )
    for key, value in overrides.items():
        base[key] = value
    return base


def test_row_keys_separate_repeated_mentions_in_one_document() -> None:
    frame = _labels()

    keys = cli.row_keys(frame)

    # Two mentions in p1 differ by offset, so they must not collapse to one key.
    assert keys.nunique() == 3
    assert keys.iloc[0] != keys.iloc[1]


def test_draws_are_split_by_sample_index() -> None:
    draws = cli.draws_from_labels(_labels(), pca_components=None, umap_dim=None)

    assert len(draws) == 2
    assert len(draws[0]) == 3
    assert set(draws[0]) == set(draws[1])  # same rows, comparable across draws


def test_mixed_cells_are_rejected_rather_than_silently_matched() -> None:
    """Clusters from different (pca, umap) cells are not the same objects."""
    frame = pd.concat([_labels(), _labels(umap_dim=8)], ignore_index=True)

    with pytest.raises(Exception, match="umap_dim"):
        cli.draws_from_labels(frame, pca_components=None, umap_dim=None)


def test_a_cell_can_be_selected_explicitly() -> None:
    frame = pd.concat([_labels(), _labels(umap_dim=8)], ignore_index=True)

    draws = cli.draws_from_labels(frame, pca_components=None, umap_dim=8)

    assert len(draws) == 2


def test_missing_row_identity_columns_is_an_explicit_error() -> None:
    frame = _labels().drop(columns=["pmid", "itext", "a_abs", "b_abs"])

    with pytest.raises(Exception, match="identify"):
        cli.draws_from_labels(frame, pca_components=None, umap_dim=None)


def test_end_to_end_stability_on_persisted_labels() -> None:
    from pelinker.clustering.stability import analyze_stability

    draws = cli.draws_from_labels(_labels(), pca_components=None, umap_dim=None)

    summary = analyze_stability(draws).to_jsonable()

    assert summary["mean_pair_jaccard"] == 1.0
    assert summary["persistent_fraction"] == 1.0
