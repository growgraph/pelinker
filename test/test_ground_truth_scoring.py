"""Task-level scoring: char-offset ground truth and the KB-lemma consistency rate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pelinker.kb.ground_truth import (
    GtSpan,
    load_ground_truth_spans,
    score_predictions_against_ground_truth,
)
from pelinker.linker.kb_lemma import aggregate_kb_lemma_validation


def _pred(a: int, b: int, eid: str | None = None, itext: int = 0) -> dict:
    return {"itext": itext, "a": a, "b": b, "entity_id_predicted": eid}


def _gold(a: int, b: int, eid: str | None = None, itext: int = 0) -> GtSpan:
    return GtSpan(itext=itext, a=a, b=b, entity_id=eid)


# ------------------------------------------------------------------------ loading


def test_loads_the_repo_ground_truth_fixture() -> None:
    spans = load_ground_truth_spans(Path("data/ground_truth/sample.0.gt.json"))

    assert len(spans) == 41
    assert all(s.b > s.a for s in spans)
    assert spans[0].entity_id == "PEL.000032"


def test_loads_a_list_of_documents_and_defaults_itext(tmp_path: Path) -> None:
    p = tmp_path / "gt.json"
    p.write_text(
        json.dumps(
            [
                {"text": "x", "ground_truth": [{"a": 0, "b": 3, "entity_id": "E1"}]},
                {"text": "y", "ground_truth": [{"a": 5, "b": 9, "entity_id": "E2"}]},
            ]
        )
    )

    spans = load_ground_truth_spans(p)

    assert [s.itext for s in spans] == [0, 1]


def test_zero_length_span_is_rejected() -> None:
    with pytest.raises(ValueError, match="span end must exceed start"):
        GtSpan(itext=0, a=5, b=5)


def test_loads_direction_and_provenance_fields(tmp_path: Path) -> None:
    p = tmp_path / "gt.json"
    p.write_text(
        json.dumps(
            {
                "text": "IL-6 is activated by TAMs",
                "ground_truth": [
                    {
                        "a": 5,
                        "b": 20,
                        "entity_id": "PEL.000032",
                        "direction": "inverse",
                        "subject_span": [21, 25],
                        "object_span": [0, 4],
                        "surface": "is activated by",
                        "annotator": "llm-a",
                        "source": "llm",
                        "confidence": 0.9,
                    },
                    # A legacy hit in the same file stays loadable.
                    {"a": 0, "b": 4, "entity_id": "PEL.000001"},
                ],
            }
        )
    )

    spans = load_ground_truth_spans(p)

    assert spans[0].direction == "inverse"
    assert spans[0].subject_span == (21, 25)
    assert spans[0].object_span == (0, 4)
    assert spans[0].source == "llm"
    assert spans[0].confidence == 0.9
    assert spans[1].direction is None
    assert spans[1].subject_span is None


def test_unknown_direction_is_rejected() -> None:
    with pytest.raises(ValueError, match="direction must be one of"):
        GtSpan(itext=0, a=0, b=3, direction="backwards")


def test_malformed_subject_span_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "gt.json"
    p.write_text(
        json.dumps(
            {
                "text": "x",
                "ground_truth": [{"a": 0, "b": 3, "subject_span": [1]}],
            }
        )
    )

    with pytest.raises(ValueError, match="subject_span must be"):
        load_ground_truth_spans(p)


# ----------------------------------------------------------------------- detection


def test_perfect_detection() -> None:
    preds = [_pred(0, 5), _pred(10, 15)]
    gold = [_gold(0, 5), _gold(10, 15)]

    s = score_predictions_against_ground_truth(preds, gold)

    assert s.n_matched == 2
    assert s.precision == 1.0
    assert s.recall == 1.0
    assert s.f1 == 1.0


def test_partial_detection_splits_precision_and_recall() -> None:
    preds = [_pred(0, 5), _pred(100, 105)]  # second is a false positive
    gold = [_gold(0, 5), _gold(10, 15), _gold(20, 25)]

    s = score_predictions_against_ground_truth(preds, gold)

    assert s.n_matched == 1
    assert s.precision == pytest.approx(0.5)
    assert s.recall == pytest.approx(1 / 3)


def test_overlap_mode_tolerates_boundary_drift_but_exact_does_not() -> None:
    preds = [_pred(2, 7)]
    gold = [_gold(0, 5)]

    assert score_predictions_against_ground_truth(preds, gold).n_matched == 1
    assert (
        score_predictions_against_ground_truth(
            preds, gold, match_mode="exact"
        ).n_matched
        == 0
    )


def test_spans_in_different_documents_never_match() -> None:
    preds = [_pred(0, 5, itext=0)]
    gold = [_gold(0, 5, itext=1)]

    assert score_predictions_against_ground_truth(preds, gold).n_matched == 0


def test_a_gold_span_is_consumed_once_so_duplicates_are_false_positives() -> None:
    preds = [_pred(0, 5), _pred(1, 4)]  # both overlap the same annotation
    gold = [_gold(0, 5)]

    s = score_predictions_against_ground_truth(preds, gold)

    assert s.n_matched == 1
    assert s.recall == 1.0
    assert s.precision == pytest.approx(0.5)


def test_empty_inputs_give_undefined_rather_than_zero() -> None:
    s = score_predictions_against_ground_truth([], [])

    assert s.precision is None
    assert s.recall is None
    assert s.f1 is None
    assert s.entity_accuracy is None


def test_unknown_match_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="match_mode must be"):
        score_predictions_against_ground_truth([], [], match_mode="fuzzy")


# ------------------------------------------------------------------ entity accuracy


def test_minted_kb_out_ids_are_reported_as_incomparable_not_wrong() -> None:
    """The trap this metric has to avoid: KB-out ids never equal KB-in gold ids."""
    preds = [_pred(0, 5, "kb::C0007")]
    gold = [_gold(0, 5, "PEL.000032")]

    s = score_predictions_against_ground_truth(preds, gold)

    assert s.n_matched == 1
    # Comparable, and simply wrong, when no map is supplied — the id really did differ.
    assert s.n_id_comparable == 1
    assert s.n_id_correct == 0


def test_an_id_map_makes_kb_out_predictions_comparable() -> None:
    preds = [_pred(0, 5, "kb::C0007"), _pred(10, 15, "kb::C0008")]
    gold = [_gold(0, 5, "PEL.000032"), _gold(10, 15, "RO.0002206")]

    s = score_predictions_against_ground_truth(
        preds,
        gold,
        predicted_id_to_kb_in={"kb::C0007": "PEL.000032", "kb::C0008": "RO.0002206"},
    )

    assert s.n_id_comparable == 2
    assert s.entity_accuracy == 1.0


def test_unmappable_predictions_are_excluded_from_the_denominator() -> None:
    preds = [_pred(0, 5, "kb::C0007"), _pred(10, 15, "kb::CUNKNOWN")]
    gold = [_gold(0, 5, "PEL.000032"), _gold(10, 15, "RO.0002206")]

    s = score_predictions_against_ground_truth(
        preds, gold, predicted_id_to_kb_in={"kb::C0007": "PEL.000032"}
    )

    assert s.n_matched == 2
    assert s.n_id_comparable == 1  # the unmappable one neither helps nor hurts
    assert s.entity_accuracy == 1.0


def test_entity_accuracy_is_undefined_when_gold_carries_no_ids() -> None:
    s = score_predictions_against_ground_truth([_pred(0, 5, "x")], [_gold(0, 5, None)])

    assert s.n_matched == 1
    assert s.n_id_comparable == 0
    assert s.entity_accuracy is None


def test_score_serializes_to_json() -> None:
    s = score_predictions_against_ground_truth([_pred(0, 5, "a")], [_gold(0, 5, "a")])

    json.dumps(s.to_jsonable())
    assert s.to_jsonable()["entity_accuracy"] == 1.0


# --------------------------------------------------------- kb-lemma consistency rate


def _row(from_lemma, matches, predicted="kb::C1"):
    return {
        "entity_id_predicted": predicted,
        "kb_training_entity_from_lemma": from_lemma,
        "lemma_kb_matches_predicted_entity": matches,
    }


def test_lemma_validation_aggregates_the_previously_unused_flag() -> None:
    rows = [_row("p1", True), _row("p2", False), _row("p3", True)]

    m = aggregate_kb_lemma_validation(rows)

    assert m.n_rows == 3
    assert m.n_resolvable == 3
    assert m.n_matches == 2
    assert m.match_rate == pytest.approx(2 / 3)
    assert m.resolvable_rate == 1.0


def test_rows_whose_lemma_resolves_to_nothing_leave_the_denominator() -> None:
    rows = [_row("p1", True), _row(None, False)]

    m = aggregate_kb_lemma_validation(rows)

    assert m.n_resolvable == 1
    assert m.match_rate == 1.0
    assert m.resolvable_rate == pytest.approx(0.5)


def test_lemma_validation_rates_are_undefined_on_empty_input() -> None:
    m = aggregate_kb_lemma_validation([])

    assert m.match_rate is None
    assert m.resolvable_rate is None
    assert m.to_jsonable()["n_rows"] == 0


def test_unenriched_rows_lower_the_resolvable_rate_rather_than_the_match_rate() -> None:
    rows = [_row("p1", True), {"entity_id_predicted": "kb::C2"}]

    m = aggregate_kb_lemma_validation(rows)

    assert m.n_rows == 2
    assert m.n_predicted == 2
    assert m.n_resolvable == 1
    assert m.match_rate == 1.0
