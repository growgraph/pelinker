"""Converse-pair collision diagnostic (run/analysis/direction_diagnostic.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "run" / "analysis" / "direction_diagnostic.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("direction_diagnostic", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diag = _load_module()


@pytest.fixture
def kb() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": ["RO.1", "RO.2", "RO.3", "RO.4", "RO.5"],
            "label": ["activates", "activated by", "part of", "has part", None],
            "inverse_entity_id": ["RO.2", "RO.1", "RO.4", "RO.3", "RO.1"],
        }
    )


def test_inverse_pairs_are_unordered_and_deduplicated(kb) -> None:
    pairs = diag.load_inverse_pairs(kb)

    # Each pair once, smaller id first; the unlabelled RO.5 row contributes nothing.
    assert pairs == [("RO.1", "RO.2"), ("RO.3", "RO.4")]


def test_missing_inverse_column_is_an_explicit_error() -> None:
    with pytest.raises(Exception, match="inverse_entity_id"):
        diag.load_inverse_pairs(pd.DataFrame({"entity_id": ["x"], "label": ["y"]}))


def test_colliding_pair_shares_its_top_cluster(kb) -> None:
    assignments = pd.DataFrame(
        {
            "entity": ["activates"] * 4 + ["activated by"] * 4,
            "cluster": [1, 1, 1, 2, 1, 1, 1, 2],
        }
    )

    rows, summary = diag.diagnose(
        assignments,
        [("RO.1", "RO.2")],
        {"RO.1": "activates", "RO.2": "activated by"},
        collision_threshold=0.5,
    )

    assert rows[0]["same_top_cluster"] is True
    assert rows[0]["collides"] is True
    assert rows[0]["mass_overlap"] == pytest.approx(1.0)
    assert summary["collision_rate"] == 1.0
    assert summary["disjoint_rate"] == 0.0


def test_separated_pair_is_disjoint(kb) -> None:
    assignments = pd.DataFrame(
        {
            "entity": ["part of"] * 3 + ["has part"] * 3,
            "cluster": [5, 5, 5, 7, 7, 7],
        }
    )

    rows, summary = diag.diagnose(
        assignments,
        [("RO.3", "RO.4")],
        {"RO.3": "part of", "RO.4": "has part"},
        collision_threshold=0.5,
    )

    assert rows[0]["disjoint"] is True
    assert rows[0]["collides"] is False
    assert summary["collision_rate"] == 0.0
    assert summary["disjoint_rate"] == 1.0


def test_pairs_absent_from_the_corpus_are_not_counted(kb) -> None:
    assignments = pd.DataFrame({"entity": ["activates"] * 2, "cluster": [1, 1]})

    rows, summary = diag.diagnose(
        assignments,
        [("RO.1", "RO.2")],
        {"RO.1": "activates", "RO.2": "activated by"},
        collision_threshold=0.5,
    )

    # One member never appears, so the pair is unobservable rather than separated.
    assert rows == []
    assert summary["n_pairs_observed"] == 0
    assert summary["collision_rate"] is None


def test_noise_rows_are_excluded_before_diagnosis() -> None:
    from pelinker.clustering.composition import filter_emergent_assignments

    assignments = pd.DataFrame(
        {"entity": ["activates", "activates"], "cluster": [-1, 3]}
    )

    kept = filter_emergent_assignments(assignments)

    assert list(kept["cluster"]) == [3]
