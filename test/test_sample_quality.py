"""Eligibility gate for the gold sample: each rule, rule order, overlap and duplicates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pelinker.eval import sample_quality as sq

ABSTRACT = (
    "Tumour-associated macrophages secrete IL-10 in the tumour microenvironment. "
    "We show that this secretion is regulated by STAT3 and that it suppresses the "
    "activity of cytotoxic T cells. Inhibition of STAT3 restored T-cell function in "
    "two mouse models, and the effect was confirmed in samples from patients with "
    "ovarian cancer. These results identify STAT3 as a target for combination therapy "
    "and suggest that the balance of cytokines in the tumour shapes the response."
)
RULES = sq.Eligibility()


def test_a_research_abstract_is_eligible() -> None:
    assert sq.drop_reason(ABSTRACT, 2015, RULES) is None


def test_each_rule_names_its_reason() -> None:
    assert sq.drop_reason(ABSTRACT, 1985, RULES) == sq.REASON_YEAR
    assert sq.drop_reason("J Pediatr. (2001). 15, 211.", 2015, RULES) == sq.REASON_SHORT
    assert sq.drop_reason(ABSTRACT * 20, 2015, RULES) == sq.REASON_LONG
    one_sentence = ABSTRACT.replace(". ", "; ")
    assert sq.drop_reason(one_sentence, 2015, RULES) == sq.REASON_SENTENCES
    french = (
        "Il peut paraitre indecent de dire que la maladie a virus Ebola peut representer "
        "une opportunite. Les systemes de sante sont fragiles. Une reforme est urgente. "
    ) * 3
    assert sq.drop_reason(french, 2015, RULES) == sq.REASON_LANGUAGE


def test_an_unparsable_year_is_not_eligible() -> None:
    assert sq.drop_reason(ABSTRACT, "n/a", RULES) == sq.REASON_YEAR


def test_training_overlap_and_duplicates_are_table_level() -> None:
    other = ABSTRACT.replace("IL-10", "IL-6")
    df = pd.DataFrame(
        {
            "summary": [ABSTRACT, ABSTRACT, other, other],
            "publication_year": [2015, 2015, 2015, 2015],
            "mag": ["1", "2", "3", "4"],
        }
    )
    training = sq.TrainingKeys(mag=frozenset({"3"}), text_sha1=frozenset())

    reasons = sq.apply_gate(df, RULES, training=training)

    assert reasons.tolist()[:3] == [None, sq.REASON_DUPLICATE, sq.REASON_TRAIN_OVERLAP]
    # The text of row 3 also appears in row 4; the overlap is by mag, so row 4 stays.
    assert reasons.iloc[3] is None


def test_training_text_hash_catches_a_reissued_id() -> None:
    df = pd.DataFrame({"summary": [ABSTRACT], "publication_year": [2015], "mag": ["9"]})
    training = sq.TrainingKeys(
        mag=frozenset(), text_sha1=frozenset({sq.text_sha1(ABSTRACT)})
    )

    assert (
        sq.apply_gate(df, RULES, training=training).iloc[0] == sq.REASON_TRAIN_OVERLAP
    )


def test_load_training_keys_reads_id_and_text_columns(tmp_path: Path) -> None:
    path = tmp_path / "fit.tsv"
    path.write_text(f"0\t1\n42\t{ABSTRACT}\n", encoding="utf-8")

    keys = sq.load_training_keys(path)

    assert keys.mag == frozenset({"42"})
    assert keys.text_sha1 == frozenset({sq.text_sha1(ABSTRACT)})


def test_reason_counts_includes_eligible() -> None:
    assert sq.reason_counts([None, sq.REASON_SHORT, None]) == {
        "eligible": 2,
        sq.REASON_SHORT: 1,
    }
