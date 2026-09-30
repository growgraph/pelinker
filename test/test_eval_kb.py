"""The canonical KB view shared by annotation and scoring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pelinker.eval.kb import (
    canonical_id_map,
    canonical_kb,
    pending_review_pairs,
    require_canonical_kb,
    symmetric_ids,
)


@pytest.fixture
def kb() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": ["PEL.1", "RO.2", "PEL.3"],
            "label": ["regulates", "regulated by", "secretes"],
            "is_canonical": [True, False, True],
            "canonical_entity_id": ["PEL.1", "PEL.1", "PEL.3"],
            "needs_review": [True, True, False],
        }
    )


def test_only_canonical_members_are_offered(kb) -> None:
    assert list(canonical_kb(kb)["entity_id"]) == ["PEL.1", "PEL.3"]


def test_the_map_folds_a_converse_member_onto_its_canonical_partner(kb) -> None:
    assert canonical_id_map(kb) == {
        "PEL.1": "PEL.1",
        "RO.2": "PEL.1",
        "PEL.3": "PEL.3",
    }


def test_a_kb_predating_the_derivation_is_refused(kb) -> None:
    legacy = kb.drop(columns=["is_canonical", "canonical_entity_id"])

    with pytest.raises(ValueError, match="is_canonical"):
        require_canonical_kb(legacy)


def test_the_refusal_names_the_pairs_file_when_one_exists(tmp_path: Path, kb) -> None:
    """The reported failure: an *.inverse.csv was passed, with its pairs file right there."""
    inverse = tmp_path / "properties.synthesis.2.inverse.csv"
    inverse.write_text("entity_id,label\n", encoding="utf-8")
    pairs = tmp_path / "properties.synthesis.2.pairs.csv"
    pairs.write_text("entity_id,label\n", encoding="utf-8")
    legacy = kb.drop(columns=["is_canonical", "canonical_entity_id"])

    with pytest.raises(ValueError, match=str(pairs)):
        require_canonical_kb(legacy, kb_csv_path=inverse)


def test_without_a_sibling_the_refusal_names_the_derivation_script(
    tmp_path: Path, kb
) -> None:
    lonely = tmp_path / "properties.synthesis.2.inverse.csv"
    lonely.write_text("entity_id,label\n", encoding="utf-8")
    legacy = kb.drop(columns=["is_canonical", "canonical_entity_id"])

    with pytest.raises(ValueError, match="derive_inverse_pairs.py"):
        require_canonical_kb(legacy, kb_csv_path=lonely)


def test_unreviewed_pairs_are_reported_for_a_warning(kb) -> None:
    assert list(pending_review_pairs(kb)["entity_id"]) == ["PEL.1", "RO.2"]


def test_a_kb_without_the_review_column_has_nothing_pending(kb) -> None:
    assert len(pending_review_pairs(kb.drop(columns=["needs_review"]))) == 0


def test_symmetric_ids_reads_the_flag(kb) -> None:
    flagged = kb.assign(is_symmetric=[False, False, True])

    assert symmetric_ids(flagged) == frozenset({"PEL.3"})


def test_a_kb_without_the_symmetry_column_marks_nothing(kb) -> None:
    assert symmetric_ids(kb) == frozenset()
