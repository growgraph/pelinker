"""Weak-label acceptance check: each violation is detected, a clean frame passes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "run" / "analysis" / "weak_label_check.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("weak_label_check", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


wl = _load_module()

KB = pd.DataFrame(
    {
        "label": ["activates", "activated by", "interacts with"],
        "canonical_entity_id": ["PEL.0", "PEL.0", "RO.1"],
        "is_symmetric": [False, False, True],
    }
)


def _rows(*records: tuple) -> pd.DataFrame:
    return pd.DataFrame(
        records,
        columns=["pmid", "entity", "a_abs", "b_abs", "direction", "surface_rule"],
    )


def test_a_clean_frame_has_no_violations() -> None:
    rows = _rows(
        ("p1", "activates", 0, 9, "forward", "active"),
        ("p1", "activated by", 20, 29, "forward", "passive_agent"),
        ("p2", "interacts with", 3, 12, "symmetric", "active"),
    )

    assert set(wl.check(rows, KB)["violations"].values()) == {0}


def test_a_relation_and_its_converse_on_one_span_are_caught() -> None:
    rows = _rows(
        ("p1", "activates", 0, 9, "forward", "lexical"),
        ("p1", "activated by", 0, 9, "forward", "passive_agent"),
    )

    violations = wl.check(rows, KB)["violations"]

    assert violations["multi_label_spans"] == 1
    assert violations["converse_collisions"] == 1


def test_nested_sites_and_flipped_symmetric_are_caught() -> None:
    rows = _rows(
        ("p1", "activates", 0, 9, "forward", "active"),
        ("p1", "activated by", 0, 12, "forward", "passive_agent"),
        ("p2", "interacts with", 3, 12, "inverse", "active"),
    )

    violations = wl.check(rows, KB)["violations"]

    assert violations["nested_spans"] == 1
    assert violations["symmetric_with_inverse_direction"] == 1


def test_a_label_outside_the_kb_is_a_violation() -> None:
    rows = _rows(("p1", "binds", 0, 5, "forward", "active"))

    report = wl.check(rows, KB)

    assert report["violations"]["labels_not_in_kb"] == 1
    assert report["labels_not_in_kb"] == ["binds"]


def test_views_count_classes_consistently_across_converse_coverage() -> None:
    """A passive of a relation with a converse entry and a passive of one without it
    are both an inverse class in reldir; raw keeps them as the labels they carry."""
    kb = pd.DataFrame(
        {
            "entity_id": ["PEL.0", "PEL.1", "PEL.2"],
            "label": ["activates", "activated by", "inhibits"],
            "is_canonical": [True, False, True],
            "canonical_entity_id": ["PEL.0", "PEL.0", "PEL.2"],
            "is_symmetric": [False, False, False],
        }
    )
    rows = _rows(
        ("p1", "activates", 0, 9, "forward", "active"),
        ("p1", "activated by", 20, 29, "forward", "passive_agent"),
        ("p2", "inhibits", 3, 12, "inverse", "passive"),
    )

    views = wl.check(rows, kb)["views"]

    assert views["classes_raw"] == 3
    assert views["classes_rel"] == 2
    assert views["classes_reldir"] == 3
    assert views["relation_direction_mass"] == {"inverse": 2, "forward": 1}


def test_views_are_skipped_without_a_pairs_kb() -> None:
    rows = _rows(("p1", "activates", 0, 9, "forward", "active"))
    assert wl.check(rows, KB)["views"] is None
