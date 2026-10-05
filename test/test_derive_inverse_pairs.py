"""Curator review of derived converse pairs: accept, reject, and what stays pending."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "run"
    / "preprocessing"
    / "derive_inverse_pairs.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("derive_inverse_pairs", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dip = _load_module()


@pytest.fixture
def paired() -> pd.DataFrame:
    """Two derived pairs plus one unpaired entry, as ``derive_pairs`` would emit them."""
    return pd.DataFrame(
        {
            "entity_id": ["PEL.1", "RO.2", "PEL.3", "PEL.4", "PEL.9"],
            "label": [
                "regulates",
                "regulated by",
                "expresses",
                "expressed in",
                "secretes",
            ],
            "inverse_entity_id": ["RO.2", "PEL.1", "PEL.4", "PEL.3", None],
            "inverse_label": [
                "regulated by",
                "regulates",
                "expressed in",
                "expresses",
                None,
            ],
            "inverse_source": [
                "passive_by",
                "passive_by",
                "passive_prep",
                "passive_prep",
                None,
            ],
            "needs_review": [True, True, True, True, False],
            "canonical_entity_id": ["PEL.1", "PEL.1", "PEL.3", "PEL.3", "PEL.9"],
            "is_canonical": [True, False, True, False, True],
            "description": ["", "", "", "", ""],
        }
    )


def _verdicts(**pairs: str) -> dict[frozenset[str], str]:
    return {frozenset(k.split("__")): v for k, v in pairs.items()}


def test_accepting_a_pair_only_clears_the_review_flag(paired) -> None:
    reviewed = dip.apply_review(paired, _verdicts(**{"PEL.1__RO.2": "accept"}))

    row = reviewed.loc[reviewed["entity_id"] == "RO.2"].iloc[0]
    assert not bool(row["needs_review"])
    # The pair itself is untouched: the curator confirmed it, so the converse member still
    # folds onto the canonical id.
    assert row["inverse_entity_id"] == "PEL.1"
    assert row["canonical_entity_id"] == "PEL.1"


def test_rejecting_a_pair_unlinks_both_members(paired) -> None:
    """A rejected pair must stop collapsing two relations onto one canonical id."""
    reviewed = dip.apply_review(paired, _verdicts(**{"PEL.3__PEL.4": "reject"}))

    for entity_id in ("PEL.3", "PEL.4"):
        row = reviewed.loc[reviewed["entity_id"] == entity_id].iloc[0]
        assert pd.isna(row["inverse_entity_id"]) or row["inverse_entity_id"] is None
        assert not bool(row["needs_review"])
        assert row["canonical_entity_id"] == entity_id
        assert bool(row["is_canonical"])


def test_an_untouched_pair_is_unaffected_by_another_verdict(paired) -> None:
    reviewed = dip.apply_review(paired, _verdicts(**{"PEL.3__PEL.4": "reject"}))

    row = reviewed.loc[reviewed["entity_id"] == "RO.2"].iloc[0]
    assert row["inverse_entity_id"] == "PEL.1"
    assert bool(row["needs_review"])


def test_pending_sheet_has_one_row_per_pair_not_per_member(paired) -> None:
    pending = dip.pending_review_sheet(paired)

    assert len(pending) == 2
    assert set(pending["verdict"]) == {""}
    assert {
        frozenset(p) for p in zip(pending["entity_id"], pending["inverse_entity_id"])
    } == {
        frozenset({"PEL.1", "RO.2"}),
        frozenset({"PEL.3", "PEL.4"}),
    }


def test_reviewed_pairs_leave_the_pending_sheet(paired) -> None:
    reviewed = dip.apply_review(
        paired, _verdicts(**{"PEL.1__RO.2": "accept", "PEL.3__PEL.4": "reject"})
    )

    assert len(dip.pending_review_sheet(reviewed)) == 0


def test_load_review_rejects_an_unknown_verdict(tmp_path: Path) -> None:
    sheet = tmp_path / "review.csv"
    sheet.write_text(
        "entity_id,inverse_entity_id,verdict,note\nPEL.1,RO.2,maybe,\n",
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="verdict"):
        dip.load_review(sheet)


def test_load_review_skips_rows_the_curator_has_not_decided(tmp_path: Path) -> None:
    """An unfilled row is 'not yet reviewed', not an error — sheets arrive half-done."""
    sheet = tmp_path / "review.csv"
    sheet.write_text(
        "entity_id,inverse_entity_id,verdict,note\nPEL.1,RO.2,accept,\nPEL.3,PEL.4,,\n",
        encoding="utf-8",
    )

    verdicts = dip.load_review(sheet)

    assert verdicts == {frozenset({"PEL.1", "RO.2"}): "accept"}


def test_load_review_requires_the_expected_columns(tmp_path: Path) -> None:
    sheet = tmp_path / "review.csv"
    sheet.write_text("entity_id,verdict\nPEL.1,accept\n", encoding="utf-8")

    with pytest.raises(Exception, match="inverse_entity_id"):
        dip.load_review(sheet)


# ------------------------------------------------------------------ symmetry


def test_symmetry_is_joined_from_the_ro_extraction() -> None:
    """The curated KB carries no RO axioms, so the flag comes from the RO table by id."""
    kb = pd.DataFrame({"entity_id": ["RO.1", "PEL.2"], "label": ["overlaps", "binds"]})
    ro = pd.DataFrame({"entity_id": ["RO.1", "RO.9"], "is_symmetric": [True, True]})

    out = dip.attach_symmetry(kb, ro)

    assert out.set_index("entity_id")["is_symmetric"].to_dict() == {
        "RO.1": True,
        "PEL.2": False,
    }


def test_a_kb_that_already_flags_symmetry_is_not_overridden() -> None:
    kb = pd.DataFrame({"entity_id": ["RO.1"], "label": ["x"], "is_symmetric": [False]})
    ro = pd.DataFrame({"entity_id": ["RO.1"], "is_symmetric": [True]})

    out = dip.attach_symmetry(kb, ro)

    assert not bool(out["is_symmetric"].iloc[0])


def test_without_an_ro_table_nothing_is_symmetric() -> None:
    kb = pd.DataFrame({"entity_id": ["RO.1"], "label": ["overlaps"]})

    assert not dip.attach_symmetry(kb, None)["is_symmetric"].any()


def test_a_symmetric_relation_is_never_paired_by_a_surface_rule(nlp) -> None:
    """ "connected to" looks passive, but it has no converse: pairing it with "connects"
    would fold two distinct relations onto one canonical id."""
    kb = pd.DataFrame(
        {
            "entity_id": ["RO.70", "RO.76", "PEL.1", "RO.2"],
            "label": ["connected to", "connects", "regulates", "regulated by"],
            "is_symmetric": [True, False, False, False],
        }
    )

    paired = dip.derive_pairs(kb, nlp).set_index("entity_id")

    assert pd.isna(paired.loc["RO.70", "inverse_entity_id"])
    assert pd.isna(paired.loc["RO.76", "inverse_entity_id"])
    assert bool(paired.loc["RO.70", "is_canonical"])
    # The non-symmetric pair next to it is still derived.
    assert paired.loc["RO.2", "canonical_entity_id"] == "PEL.1"


def test_irregular_participles_count_as_passive() -> None:
    """ "bound by" ↔ "binds to": the participle ends in neither -ed nor -en."""
    assert dip.is_passive_form("bound by")
    assert dip.is_passive_form("regulated by")
    assert not dip.is_passive_form("binds to")
    assert not dip.is_passive_form("bound")


def test_curated_ids_are_symmetric_on_top_of_ro() -> None:
    """RO does not cover GeneWays verbs; a curator can still mark "binds to" symmetric."""
    kb = pd.DataFrame(
        {
            "entity_id": ["RO.1", "PEL.3", "PEL.4"],
            "label": ["overlaps", "binds to", "x"],
        }
    )
    ro = pd.DataFrame({"entity_id": ["RO.1"], "is_symmetric": [True]})

    out = dip.attach_symmetry(kb, ro, frozenset({"PEL.3"}))

    assert out.set_index("entity_id")["is_symmetric"].to_dict() == {
        "RO.1": True,
        "PEL.3": True,
        "PEL.4": False,
    }


def test_load_symmetric_reads_entity_ids(tmp_path: Path) -> None:
    sheet = tmp_path / "symmetric.csv"
    sheet.write_text("entity_id,label,note\nPEL.3,binds to,\n,,\n", encoding="utf-8")

    assert dip.load_symmetric(sheet) == frozenset({"PEL.3"})
