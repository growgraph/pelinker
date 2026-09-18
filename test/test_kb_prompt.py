"""Rendering the KB catalog into a prompt: cleaning and the one-line-per-relation contract."""

from __future__ import annotations

import pandas as pd
import pytest

from pelinker.eval.kb_prompt import clean_description, render_kb_catalog


def _kb(**cols) -> pd.DataFrame:
    base = {"entity_id": ["E1"], "label": ["regulates"], "description": [""]}
    base.update(cols)
    return pd.DataFrame(base)


# ------------------------------------------------------------------- cleaning


def test_newlines_are_collapsed() -> None:
    """The defect that mattered: a multi-line description became several relations."""
    raw = "X outer_layer_of Y iff:\n. X part_of Y\n. exists Z :surface\n"

    cleaned = clean_description(raw)

    assert "\n" not in cleaned
    assert cleaned.startswith("X outer_layer_of Y iff: . X part_of Y")


def test_formal_restatement_is_dropped() -> None:
    raw = (
        "x precedes y if and only if the time point at which x ends is before y. "
        "Formally: x precedes y iff ω(x) <= α(y), where α maps a process to a start point."
    )

    cleaned = clean_description(raw)

    assert cleaned.endswith("before y.")
    for symbol in ("ω", "α", "Formally"):
        assert symbol not in cleaned


def test_pure_cross_references_are_dropped_entirely() -> None:
    """They carry no definition and collide with the converse field."""
    assert clean_description("inverse of develops from") == ""
    assert clean_description("Inverse of characteristic_of") == ""


def test_a_provenance_bracket_is_stripped_but_real_content_survives() -> None:
    raw = "[copied from inverse property 'occurs in'] b occurs_in c =def b is a process"

    cleaned = clean_description(raw)

    assert cleaned == "b occurs_in c =def b is a process"


def test_urls_and_padding_are_removed() -> None:
    raw = "  Relation between a neuron and a structure. http://purl.obolibrary.org/obo/RO_1 \n "

    assert clean_description(raw) == "Relation between a neuron and a structure."


def test_missing_descriptions_are_empty_not_nan() -> None:
    assert clean_description(None) == ""
    assert clean_description(float("nan")) == ""
    assert clean_description("") == ""


def test_long_descriptions_truncate_on_a_word_boundary() -> None:
    raw = "word " * 200

    cleaned = clean_description(raw, max_chars=50)

    assert len(cleaned) <= 51  # plus the ellipsis
    assert cleaned.endswith("…")
    assert "wor…" not in cleaned  # never mid-word


# ------------------------------------------------------------------ rendering


def test_one_line_per_relation_even_with_dirty_descriptions() -> None:
    kb = pd.DataFrame(
        {
            "entity_id": ["E1", "E2", "E3"],
            "label": ["regulates", "bounding layer of", "secretes"],
            "description": ["p regulates q", "a:\n. b\n. c\n", None],
        }
    )

    lines = render_kb_catalog(kb).split("\n")

    assert len(lines) == 3, lines


def test_entries_are_numbered_and_labels_quoted() -> None:
    kb = _kb(label=["has part"])

    line = render_kb_catalog(kb)

    assert line.startswith('1. "has part"')


def test_converse_wording_is_shown_when_present() -> None:
    kb = _kb(inverse_label=["regulated by"], description=["p regulates q"])

    line = render_kb_catalog(kb)

    assert 'converse wording: "regulated by"' in line
    assert "def: p regulates q" in line


def test_absent_converse_adds_no_dangling_field() -> None:
    kb = _kb(inverse_label=[None])

    line = render_kb_catalog(kb)

    assert "converse wording" not in line


def test_converse_can_be_suppressed() -> None:
    kb = _kb(inverse_label=["regulated by"])

    line = render_kb_catalog(kb, include_converse=False)

    assert "converse wording" not in line


def test_numbering_follows_row_order() -> None:
    kb = pd.DataFrame(
        {
            "entity_id": ["E1", "E2"],
            "label": ["alpha", "beta"],
            "description": ["", ""],
        }
    )

    lines = render_kb_catalog(kb).split("\n")

    assert lines[0].startswith('1. "alpha"')
    assert lines[1].startswith('2. "beta"')


@pytest.mark.parametrize("label", ["has  part", "has\npart"])
def test_labels_with_stray_whitespace_are_normalized(label: str) -> None:
    line = render_kb_catalog(_kb(label=[label]))

    assert line == '1. "has part"'


def test_the_shipped_kb_renders_one_line_per_relation() -> None:
    """Guards the real artifact, not just synthetic rows."""
    kb = pd.read_csv("data/derived/properties.synthesis.2.pairs.csv")
    kb = kb.loc[kb["is_canonical"].astype(bool)]

    lines = render_kb_catalog(kb).split("\n")

    assert len(lines) == len(kb)
