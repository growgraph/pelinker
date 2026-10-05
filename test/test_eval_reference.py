"""The reference inventory's tier 0+1 fold and its equivalences file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pelinker.eval.reference import (
    fold_map,
    folded_symmetric_ids,
    load_equivalences,
)

_REPO = Path(__file__).resolve().parents[1]


def _kb() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": ["R.1", "R.1i", "S.1", "S.1c", "X.1"],
            "label": ["regulates", "regulated by", "binds to", "bound by", "causes"],
            "is_symmetric": [False, False, True, True, False],
            "is_canonical": [True, False, True, True, True],
            "canonical_entity_id": ["R.1", "R.1", "S.1", "S.1c", "X.1"],
        }
    )


def _equivalences(tmp_path: Path, rows: list[tuple[str, str, str]]) -> Path:
    path = tmp_path / "equivalences.csv"
    pd.DataFrame(
        {
            "entity_id": [a for a, _, _ in rows],
            "equivalent_to": [b for _, b, _ in rows],
            "tier": [tier for _, _, tier in rows],
            "note": ["" for _ in rows],
        }
    ).to_csv(path, index=False)
    return path


def test_tier0_folds_a_converse_member_onto_its_canonical_member() -> None:
    fold = fold_map(_kb())

    assert fold["R.1i"] == "R.1"
    # Without tier 1, a minted converse stays its own class.
    assert fold["S.1c"] == "S.1c"


def test_tier1_folds_kb_implied_identities(tmp_path: Path) -> None:
    kb = _kb()
    eq = load_equivalences(
        _equivalences(tmp_path, [("S.1c", "S.1", "kb_implied")]), kb=kb
    )

    fold = fold_map(kb, eq)

    assert fold["S.1c"] == "S.1"
    assert folded_symmetric_ids(kb, fold) == frozenset({"S.1"})


def test_equivalence_chains_resolve_to_their_end(tmp_path: Path) -> None:
    kb = _kb()
    eq = load_equivalences(
        _equivalences(
            tmp_path, [("S.1c", "S.1", "kb_implied"), ("S.1", "X.1", "kb_implied")]
        ),
        kb=kb,
    )

    assert fold_map(kb, eq)["S.1c"] == "X.1"


def test_an_equivalence_cycle_is_refused(tmp_path: Path) -> None:
    kb = _kb()
    eq = load_equivalences(
        _equivalences(
            tmp_path, [("S.1c", "S.1", "kb_implied"), ("S.1", "S.1c", "kb_implied")]
        ),
        kb=kb,
    )

    with pytest.raises(ValueError, match="cycle"):
        fold_map(kb, eq)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([("S.1c", "S.1", "adjudicated")], "unknown tier"),
        ([("S.1", "S.1", "kb_implied")], "onto themselves"),
        (
            [("S.1c", "S.1", "kb_implied"), ("S.1c", "X.1", "kb_implied")],
            "more than one",
        ),
        ([("NOPE.1", "S.1", "kb_implied")], "not in the KB"),
    ],
)
def test_malformed_equivalences_are_refused(
    tmp_path: Path, rows: list[tuple[str, str, str]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_equivalences(_equivalences(tmp_path, rows), kb=_kb())


def test_the_shipped_equivalences_load_against_the_pairs_kb() -> None:
    kb = pd.read_csv(_REPO / "data/derived/properties.synthesis.2.pairs.csv")
    eq = load_equivalences(_REPO / "data/curated/equivalences.csv", kb=kb)

    fold = fold_map(kb, eq)

    assert fold["PEL.000040"] == "PEL.000002"
    assert fold["PEL.000041"] == "PEL.000003"
