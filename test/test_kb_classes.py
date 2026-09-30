"""Class views of the pairs KB: raw label, canonical relation, relation + direction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pelinker.core.onto import NEGATIVE_LABEL
from pelinker.kb.classes import (
    ENTITY_CLASS_COLUMN,
    RELATION_COLUMN,
    RELATION_DIRECTION_COLUMN,
    KbClasses,
    add_view_columns,
    check_class_view,
    reference_label_column,
    relation_direction,
)


@pytest.fixture
def kb() -> pd.DataFrame:
    """``regulates`` has a converse entry; ``inhibits`` has none; ``associates`` is symmetric."""
    return pd.DataFrame(
        {
            "entity_id": ["RO.1", "RO.2", "PEL.3", "PEL.4", "RO.5", "RO.6"],
            "label": [
                "regulates",
                "regulated by",
                "inhibits",
                "associates",
                "has part",
                "part of",
            ],
            "is_canonical": [True, False, True, True, True, False],
            "canonical_entity_id": ["RO.1", "RO.1", "PEL.3", "PEL.4", "RO.5", "RO.5"],
            "is_symmetric": [False, False, False, True, False, False],
        }
    )


@pytest.fixture
def classes(kb) -> KbClasses:
    return KbClasses.from_kb(kb)


def _mentions(rows: list[tuple[str, str | None]]) -> pd.DataFrame:
    return pd.DataFrame(
        {"entity": [label for label, _ in rows], "direction": [d for _, d in rows]}
    )


def test_passives_share_one_class_whether_or_not_the_kb_has_a_converse_entry(
    classes,
) -> None:
    """The matcher stores a passive under the converse entry when one exists, else under
    the active label with ``direction=inverse``. ``reldir`` must not tell those apart."""
    frame = _mentions(
        [
            ("regulated by", "forward"),  # passive of regulates, via its converse entry
            ("inhibits", "inverse"),  # passive of inhibits, which has no converse entry
            ("regulates", "forward"),
        ]
    )

    out = add_view_columns(frame, classes, "reldir")

    assert out[ENTITY_CLASS_COLUMN].tolist() == [
        "regulates|inverse",
        "inhibits|inverse",
        "regulates|forward",
    ]
    assert out[RELATION_COLUMN].tolist() == ["regulates", "inhibits", "regulates"]
    assert out[RELATION_DIRECTION_COLUMN].tolist() == ["inverse", "inverse", "forward"]


def test_raw_view_keeps_the_matched_label(classes) -> None:
    frame = _mentions([("regulated by", "forward"), ("inhibits", "inverse")])

    out = add_view_columns(frame, classes, "raw")

    assert out[ENTITY_CLASS_COLUMN].tolist() == ["regulated by", "inhibits"]
    # The entity column itself is never rewritten by any view.
    assert out["entity"].tolist() == ["regulated by", "inhibits"]


def test_rel_view_drops_direction(classes) -> None:
    frame = _mentions([("regulated by", "forward"), ("regulates", "forward")])

    out = add_view_columns(frame, classes, "rel")

    assert out[ENTITY_CLASS_COLUMN].tolist() == ["regulates", "regulates"]


def test_a_symmetric_relation_is_never_flipped(classes) -> None:
    assert relation_direction("associates", "inverse", classes) == "symmetric"
    out = add_view_columns(_mentions([("associates", "inverse")]), classes, "reldir")
    assert out[ENTITY_CLASS_COLUMN].tolist() == ["associates"]


def test_a_lexical_match_reads_its_own_label_forward(classes) -> None:
    """A non-verb match carries no direction; it uses the label's own wording, so a
    converse label ("part of") is the canonical relation read the other way."""
    out = add_view_columns(
        _mentions([("part of", None), ("has part", None)]), classes, "reldir"
    )

    assert out[ENTITY_CLASS_COLUMN].tolist() == ["has part|inverse", "has part|forward"]


def test_the_negative_label_passes_through_and_unknown_labels_are_refused(
    classes,
) -> None:
    frame = _mentions([(NEGATIVE_LABEL, None), ("regulates", "forward")])
    out = add_view_columns(
        frame, classes, "reldir", passthrough_labels=frozenset({NEGATIVE_LABEL})
    )
    assert out[ENTITY_CLASS_COLUMN].tolist() == [NEGATIVE_LABEL, "regulates|forward"]

    with pytest.raises(ValueError, match="not in the class KB"):
        add_view_columns(_mentions([("binds", "forward")]), classes, "reldir")


def test_a_kb_without_the_pair_derivation_is_refused() -> None:
    plain = pd.DataFrame({"entity_id": ["RO.1"], "label": ["regulates"]})
    with pytest.raises(ValueError, match="predates the converse-pair derivation"):
        KbClasses.from_kb(plain)


def test_duplicate_labels_are_refused(kb) -> None:
    dup = pd.concat([kb, kb.iloc[[0]].assign(entity_id="RO.9")], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        KbClasses.from_kb(dup)


def test_from_csv_reads_the_same_classes(kb, tmp_path: Path) -> None:
    path = tmp_path / "kb.pairs.csv"
    kb.to_csv(path, index=False)

    assert KbClasses.from_csv(path) == KbClasses.from_kb(kb)


def test_check_class_view() -> None:
    assert check_class_view("raw", None) == "raw"
    assert check_class_view("reldir", "kb.csv") == "reldir"
    with pytest.raises(ValueError, match="needs --class-kb-path"):
        check_class_view("reldir", None)
    with pytest.raises(ValueError, match="must be one of"):
        check_class_view("voice", "kb.csv")


def test_reference_label_column_prefers_the_view(classes) -> None:
    frame = _mentions([("regulates", "forward")])
    assert reference_label_column(frame) == "entity"
    viewed = add_view_columns(frame, classes, "reldir")
    assert reference_label_column(viewed) == ENTITY_CLASS_COLUMN
