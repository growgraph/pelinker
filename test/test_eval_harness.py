"""Gold-evaluation harness: document loading, both scoring regimes, lexical baseline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pelinker.eval.baselines import LexicalLemmaBaseline, LexicalSpanProposer
from pelinker.eval.harness import (
    GoldDoc,
    evaluate_end_to_end,
    evaluate_linking_only,
    load_gold_docs,
)

TEXT_A = "TAMs secrete IL-10 in the tumor microenvironment."
TEXT_B = "IL-6 is activated by TAMs."


@pytest.fixture
def gold_file(tmp_path: Path) -> Path:
    p = tmp_path / "gold.json"
    p.write_text(
        json.dumps(
            [
                {
                    "doc_id": 7,
                    "text": TEXT_A,
                    "ground_truth": [
                        {
                            "a": 5,
                            "b": 12,
                            "entity_id": "PEL.1",
                            "direction": "forward",
                        }
                    ],
                },
                {
                    "doc_id": 9,
                    "text": TEXT_B,
                    "ground_truth": [
                        {
                            "a": 5,
                            "b": 20,
                            "entity_id": "PEL.2",
                            "direction": "inverse",
                        }
                    ],
                },
            ]
        )
    )
    return p


def test_load_gold_docs_groups_spans_by_document(gold_file: Path) -> None:
    docs = load_gold_docs(gold_file)

    assert [d.doc_id for d in docs] == [7, 9]
    assert [d.text for d in docs] == [TEXT_A, TEXT_B]
    # itext follows position in the file, matching what predictions are indexed by.
    assert docs[0].spans[0].itext == 0
    assert docs[1].spans[0].itext == 1
    assert docs[1].spans[0].direction == "inverse"


def test_end_to_end_scores_a_perfect_predictor(gold_file: Path) -> None:
    docs = load_gold_docs(gold_file)

    def predict(texts: list[str]) -> list[dict]:
        assert texts == [TEXT_A, TEXT_B]
        return [
            {"itext": 0, "a": 5, "b": 12, "entity_id_predicted": "PEL.1"},
            {"itext": 1, "a": 5, "b": 20, "entity_id_predicted": "PEL.2"},
        ]

    run = evaluate_end_to_end(predict, docs, system="oracle")

    assert run.regime == "end_to_end"
    assert run.n_docs == 2
    assert run.wall_seconds >= 0.0
    assert run.score["f1"] == 1.0
    assert run.score["entity_accuracy"] == 1.0


def test_end_to_end_applies_the_kb_out_id_bridge(gold_file: Path) -> None:
    docs = load_gold_docs(gold_file)

    def predict(texts: list[str]) -> list[dict]:
        return [{"itext": 0, "a": 5, "b": 12, "entity_id_predicted": "kb::C0003"}]

    run = evaluate_end_to_end(
        predict,
        docs,
        system="linker",
        predicted_id_to_kb_in={"kb::C0003": "PEL.1"},
    )

    assert run.score["n_id_comparable"] == 1
    assert run.score["entity_accuracy"] == 1.0


def test_linking_only_counts_abstentions_as_wrong_overall(gold_file: Path) -> None:
    docs = load_gold_docs(gold_file)

    def link(text: str, a: int, b: int) -> str | None:
        return "PEL.1" if text.startswith("TAMs") else None

    run = evaluate_linking_only(link, docs, system="half-abstaining")

    assert run.regime == "linking_only"
    assert run.score["n_spans"] == 2
    assert run.score["n_predicted"] == 1
    assert run.score["coverage"] == 0.5
    # Abstention is free on the predicted-only view and costly on the overall one.
    assert run.score["accuracy_on_predicted"] == 1.0
    assert run.score["accuracy_overall"] == 0.5


def test_linking_only_is_undefined_without_gold_ids() -> None:
    docs = [GoldDoc(doc_id=0, text="x y", spans=())]

    run = evaluate_linking_only(lambda t, a, b: "PEL.1", docs, system="none")

    assert run.score["accuracy_overall"] is None


def test_eval_run_serializes_flat(gold_file: Path) -> None:
    docs = load_gold_docs(gold_file)
    run = evaluate_linking_only(lambda t, a, b: None, docs, system="abstainer")

    payload = run.to_jsonable()

    json.dumps(payload)
    assert payload["system"] == "abstainer"
    assert payload["regime"] == "linking_only"
    assert payload["coverage"] == 0.0


# --------------------------------------------------------------- lexical baseline


@pytest.fixture
def kb_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": ["PEL.1", "PEL.2", "PEL.9"],
            "label": ["secretes", "is activated by", "secretes"],
            "description": ["secretion relation", "converse of activates", "dup"],
        }
    )


def test_lexical_baseline_proposes_and_links_spans(kb_frame, nlp) -> None:
    baseline = LexicalLemmaBaseline(kb_frame, nlp)

    rows = baseline.predict([TEXT_A, TEXT_B])

    by_doc = {r["itext"]: r for r in rows}
    assert by_doc[0]["entity_id_predicted"] == "PEL.1"
    assert TEXT_A[by_doc[0]["a"] : by_doc[0]["b"]] == "secrete"
    assert by_doc[1]["entity_id_predicted"] == "PEL.2"
    assert TEXT_B[by_doc[1]["a"] : by_doc[1]["b"]] == "is activated by"


def test_duplicate_labels_resolve_to_the_smallest_id(kb_frame, nlp) -> None:
    """PEL.1 and PEL.9 share the label 'secretes'; the choice must be deterministic."""
    baseline = LexicalLemmaBaseline(kb_frame, nlp)

    assert baseline.link(TEXT_A, 5, 12) == "PEL.1"


def test_longest_match_wins_over_its_own_sub_span(kb_frame, nlp) -> None:
    proposer = LexicalSpanProposer(kb_frame, nlp)

    spans = proposer.propose(TEXT_B)

    # "is activated by" (3 tokens) must not be split into a shorter competing match.
    assert [TEXT_B[a:b] for a, b, _ in spans] == ["is activated by"]


def test_unmatched_span_links_to_nothing(kb_frame, nlp) -> None:
    baseline = LexicalLemmaBaseline(kb_frame, nlp)

    assert baseline.link("the tumor microenvironment", 4, 9) is None


def test_wider_gold_span_backs_off_to_the_matching_sub_window(nlp) -> None:
    """Gold annotates the whole phrase; the KB often carries only the bare form."""
    kb = pd.DataFrame(
        {
            "entity_id": ["PEL.5"],
            "label": ["activated"],
            "description": ["activation relation"],
        }
    )
    baseline = LexicalLemmaBaseline(kb, nlp)

    # Exact-only matching would abstain here and make the baseline a strawman.
    assert baseline.link(TEXT_B, 5, 20) == "PEL.5"
