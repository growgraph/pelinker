"""Label-based annotation decoding: canonical vocabulary, label -> id, rejections."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "run" / "eval" / "annotate_llm.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("annotate_llm", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: the module defines a dataclass, and dataclasses resolve
    # field types through sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ann = _load_module()

TEXT = "TAMs secrete IL-10, and IL-6 is regulated by TAMs."


@pytest.fixture
def kb() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": ["PEL.1", "RO.2", "PEL.3"],
            "label": ["regulates", "regulated by", "secretes"],
            "description": ["regulation relation", "converse of regulates", ""],
            "inverse_entity_id": ["RO.2", "PEL.1", None],
            "inverse_label": ["regulated by", "regulates", None],
            "is_canonical": [True, False, True],
            "canonical_entity_id": ["PEL.1", "PEL.1", "PEL.3"],
        }
    )


def test_a_kb_without_pair_derivation_is_refused(kb) -> None:
    """Silently offering both pair members would change what the gold means."""
    legacy = kb.drop(columns=["is_canonical"])

    with pytest.raises(Exception, match="is_canonical"):
        ann.canonical_kb(legacy)


def test_only_canonical_labels_are_offered(kb) -> None:
    offered = ann.canonical_kb(kb)

    assert set(offered["label"]) == {"regulates", "secretes"}
    # The converse member is deliberately absent: one valid encoding per mention.
    assert "regulated by" not in set(offered["label"])


def test_kb_table_carries_no_ids_but_shows_the_converse(kb) -> None:
    table = ann.kb_table_text(ann.canonical_kb(kb))
    lines = table.split("\n")

    assert "PEL.1" not in table and "RO.2" not in table
    assert lines[0] == (
        '1. "regulates" | def: regulation relation | converse wording: "regulated by"'
    )
    # A relation with no converse gets no dangling field.
    assert lines[1] == '2. "secretes"'
    # One line per offered relation, whatever the descriptions look like.
    assert len(lines) == len(ann.canonical_kb(kb))


def test_labels_decode_to_ids(kb) -> None:
    index = ann.build_label_index(ann.canonical_kb(kb))

    hits, rejected = ann.resolve_annotations(
        TEXT,
        [
            {
                "surface": "secrete",
                "occurrence": 1,
                "label": "secretes",
                "direction": "forward",
            }
        ],
        label_index=index,
        annotator="llm-a",
    )

    assert rejected == []
    assert hits[0]["entity_id"] == "PEL.3"
    assert hits[0]["label"] == "secretes"
    assert TEXT[hits[0]["a"] : hits[0]["b"]] == "secrete"


def test_label_matching_tolerates_case_and_padding(kb) -> None:
    index = ann.build_label_index(ann.canonical_kb(kb))

    hits, rejected = ann.resolve_annotations(
        TEXT,
        [{"surface": "secrete", "occurrence": 1, "label": "  Secretes "}],
        label_index=index,
        annotator="llm-a",
    )

    assert rejected == []
    assert hits[0]["entity_id"] == "PEL.3"


def test_a_passive_mention_uses_the_canonical_label_with_inverse_direction(kb) -> None:
    index = ann.build_label_index(ann.canonical_kb(kb))

    hits, _ = ann.resolve_annotations(
        TEXT,
        [
            {
                "surface": "is regulated by",
                "occurrence": 1,
                "label": "regulates",
                "direction": "inverse",
            }
        ],
        label_index=index,
        annotator="llm-a",
    )

    assert hits[0]["entity_id"] == "PEL.1"
    assert hits[0]["direction"] == "inverse"


def test_an_unlisted_label_is_rejected_not_guessed(kb) -> None:
    index = ann.build_label_index(ann.canonical_kb(kb))

    hits, rejected = ann.resolve_annotations(
        TEXT,
        [{"surface": "secrete", "occurrence": 1, "label": "emits"}],
        label_index=index,
        annotator="llm-a",
    )

    assert hits == []
    assert rejected[0]["reasons"] == ["unknown_label"]


def test_the_suppressed_converse_label_is_not_silently_accepted(kb) -> None:
    """'regulated by' is a real KB label but not offered; using it is an error."""
    index = ann.build_label_index(ann.canonical_kb(kb))

    hits, rejected = ann.resolve_annotations(
        TEXT,
        [{"surface": "is regulated by", "occurrence": 1, "label": "regulated by"}],
        label_index=index,
        annotator="llm-a",
    )

    assert hits == []
    assert "unknown_label" in rejected[0]["reasons"]


# --------------------------------------------------- occurrence + span resolution


def test_word_boundaries_are_respected() -> None:
    """A short answer must not be located inside a longer word."""
    text = "The drug reduces risk."

    assert ann.iter_occurrences(text, "reduce") == []
    assert ann.iter_occurrences(text, "reduces") == [(9, 16)]


def test_multiword_surface_respects_boundaries() -> None:
    text = "was correlated to outcome"

    assert ann.iter_occurrences(text, "related to") == []
    assert ann.iter_occurrences(text, "correlated to") == [(4, 17)]


def test_requested_occurrence_is_used_when_it_exists() -> None:
    text = "binds here and binds there"

    span, fallback = ann.find_occurrence(text, "binds", 2)

    assert span == (15, 20) and fallback is False


def test_overcounted_occurrence_falls_back_to_an_unclaimed_one() -> None:
    """Models miscount repeats; the surface is still required verbatim."""
    text = "the gene is expressed."

    span, fallback = ann.find_occurrence(text, "expressed", 3)

    assert span == (12, 21) and fallback is True


def test_a_claimed_span_is_never_annotated_twice() -> None:
    text = "the gene is expressed."
    claimed = {(12, 21)}

    span, fallback = ann.find_occurrence(text, "expressed", 1, claimed=claimed)

    assert span is None and fallback is False


def test_nested_spans_collapse_to_the_longest() -> None:
    hits = [
        {"a": 10, "b": 19, "label": "expresses"},
        {"a": 10, "b": 22, "label": "expressed in"},
        {"a": 40, "b": 45, "label": "binds"},
    ]

    kept, dropped = ann.drop_nested_spans(hits)

    assert [(h["a"], h["b"]) for h in kept] == [(10, 22), (40, 45)]
    assert dropped[0]["reasons"] == ["nested_in_longer_span"]


# ------------------------------------------------------- converse-label recovery


def test_a_converse_label_maps_to_canonical_with_flipped_direction(kb) -> None:
    """Dropping these would remove inverse-direction mentions specifically."""
    index = ann.build_label_index(kb)

    hits, rejected = ann.resolve_annotations(
        "IL-6 is regulated by TAMs.",
        [
            {
                "surface": "regulated by",
                "occurrence": 1,
                "label": "regulated by",
                "direction": "forward",
            }
        ],
        label_index=index,
        annotator="llm-a",
    )

    assert rejected == []
    assert hits[0]["entity_id"] == "PEL.1"  # canonical member
    assert hits[0]["direction"] == "inverse"  # orientation flipped
    assert hits[0]["label_canonicalized"] is True


def test_flip_direction_leaves_symmetric_and_na_alone() -> None:
    assert ann.flip_direction("forward") == "inverse"
    assert ann.flip_direction("inverse") == "forward"
    assert ann.flip_direction("symmetric") == "symmetric"
    assert ann.flip_direction(None) is None
