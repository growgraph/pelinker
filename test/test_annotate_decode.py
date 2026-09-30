"""Label-based annotation decoding: canonical vocabulary, label -> id, rejections."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from pelinker.core.onto import SimplifiedToken

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


def test_an_oriented_answer_on_a_symmetric_relation_is_coerced_and_flagged() -> None:
    kb = pd.DataFrame(
        {
            "entity_id": ["RO.34"],
            "label": ["interacts with"],
            "is_symmetric": [True],
            "is_canonical": [True],
            "canonical_entity_id": ["RO.34"],
        }
    )

    hits, rejected = ann.resolve_annotations(
        "p53 interacts with MDM2.",
        [
            {
                "surface": "interacts with",
                "occurrence": 1,
                "label": "interacts with",
                "direction": "forward",
            }
        ],
        label_index=ann.build_label_index(kb),
        annotator="llm-a",
        symmetric=ann.symmetric_ids(kb),
    )

    assert rejected == []
    assert hits[0]["direction"] == "symmetric"
    assert hits[0]["direction_coerced"] is True


def test_a_symmetric_answer_needs_no_coercion() -> None:
    kb = pd.DataFrame(
        {
            "entity_id": ["RO.34"],
            "label": ["interacts with"],
            "is_symmetric": [True],
            "is_canonical": [True],
            "canonical_entity_id": ["RO.34"],
        }
    )

    hits, _ = ann.resolve_annotations(
        "p53 interacts with MDM2.",
        [
            {
                "surface": "interacts with",
                "occurrence": 1,
                "label": "interacts with",
                "direction": "symmetric",
            }
        ],
        label_index=ann.build_label_index(kb),
        annotator="llm-a",
        symmetric=ann.symmetric_ids(kb),
    )

    assert hits[0]["direction"] == "symmetric"
    assert "direction_coerced" not in hits[0]


def test_a_nominal_surface_on_a_verb_label_is_rejected(kb) -> None:
    """Training sees verb mentions only; a noun in gold scores a skill never taught."""
    text = "The association of IL-6 with TAMs regulates growth."
    item = {"surface": "association", "occurrence": 1, "label": "regulates"}

    hits, rejected = ann.resolve_annotations(
        text,
        [item],
        label_index=ann.build_label_index(kb),
        annotator="llm-a",
        verb_labels=frozenset({"regulates"}),
        has_verb=lambda a, b: False,
    )

    assert hits == []
    assert rejected[0]["reasons"] == ["non_verbal"]


def test_a_non_verb_label_keeps_its_noun_surface(kb) -> None:
    hits, rejected = ann.resolve_annotations(
        "TAMs secrete IL-10.",
        [{"surface": "secrete", "occurrence": 1, "label": "secretes"}],
        label_index=ann.build_label_index(kb),
        annotator="llm-a",
        verb_labels=frozenset({"regulates"}),  # "secretes" is not in the verb set here
        has_verb=lambda a, b: False,
    )

    assert rejected == [] and len(hits) == 1


def _tok(ix: int, text: str, lemma: str) -> SimplifiedToken:
    return SimplifiedToken(ix=ix, ix_end=ix + len(text), text=text, lemma=lemma, tag="")


def test_anchor_check_tells_label_wording_from_paraphrase() -> None:
    # "Smoking leads to cancer and causes harm."
    tokens = [
        _tok(0, "Smoking", "smoking"),
        _tok(8, "leads", "lead"),
        _tok(14, "to", "to"),
        _tok(17, "cancer", "cancer"),
        _tok(24, "and", "and"),
        _tok(28, "causes", "cause"),
        _tok(35, "harm", "harm"),
    ]
    anchors = {"PEL.4": [frozenset({"cause"})]}
    anchored = ann.anchor_check(tokens, anchors)

    assert anchored(28, 34, "PEL.4") is True  # "causes"
    assert anchored(8, 16, "PEL.4") is False  # "leads to": a paraphrase
    assert anchored(28, 34, "PEL.unknown") is False


def test_resolved_hits_carry_the_anchor_flag(kb) -> None:
    hits, _ = ann.resolve_annotations(
        "TAMs secrete IL-10.",
        [{"surface": "secrete", "occurrence": 1, "label": "secretes"}],
        label_index=ann.build_label_index(kb),
        annotator="llm-a",
        is_anchored=lambda a, b, entity_id: entity_id == "PEL.3",
    )

    assert hits[0]["surface_anchored"] is True


# ------------------------------------------------------------- response handling


def test_an_object_wrapping_one_array_is_unwrapped() -> None:
    """Object-only JSON modes wrap the requested array; the records are what counts."""
    raw = '{"annotations": [{"surface": "binds", "occurrence": 1}]}'

    assert ann.parse_llm_response(raw) == [{"surface": "binds", "occurrence": 1}]


def test_an_object_that_is_not_a_wrapper_is_still_refused() -> None:
    with pytest.raises(ValueError, match="JSON array"):
        ann.parse_llm_response('{"surface": "binds", "occurrence": 1}')


def test_a_reemitted_mention_is_reported_as_a_duplicate(kb) -> None:
    """Quoting text that is absent and repeating a mention are different failures."""
    item = {"surface": "secrete", "occurrence": 1, "label": "secretes"}

    hits, rejected = ann.resolve_annotations(
        TEXT,
        [item, dict(item), {**item, "surface": "secreted"}],
        label_index=ann.build_label_index(kb),
        annotator="llm-a",
    )

    assert len(hits) == 1
    assert [r["reasons"] for r in rejected] == [
        ["duplicate_occurrence"],
        ["surface_not_found"],
    ]


def test_a_truncated_document_is_reported_and_left_out(
    kb, nlp, tmp_path: Path, monkeypatch
) -> None:
    """An unanswered document must not read as one where the annotator found nothing."""
    import json

    from click.testing import CliRunner

    from pelinker.eval.llm import LLMIncompleteError

    kb_path = tmp_path / "kb.csv"
    kb.assign(needs_review=False, is_symmetric=False).to_csv(kb_path, index=False)
    sample = tmp_path / "gold"
    sample.mkdir()
    pd.DataFrame({"doc_id": [1, 2], "role": ["primary", "primary"]}).to_csv(
        sample / "sample_manifest.csv", index=False
    )
    (sample / "sample_texts.jsonl").write_text(
        "\n".join(json.dumps({"doc_id": i, "text": TEXT}) for i in (1, 2)),
        encoding="utf-8",
    )

    calls: list[str] = []

    def fake_complete(*, user, **kw):
        calls.append(user)
        if len(calls) > 1:
            raise LLMIncompleteError("stopped at max_output_tokens")
        return '[{"surface": "secrete", "occurrence": 1, "label": "secretes"}]'

    monkeypatch.setattr(ann, "complete", fake_complete)
    monkeypatch.setattr(ann.spacy, "load", lambda name: nlp)
    out = tmp_path / "gold.llm-b.json"

    result = CliRunner().invoke(
        ann.main,
        [
            "--sample-dir",
            str(sample),
            "--kb-csv-path",
            str(kb_path),
            "--output-path",
            str(out),
            "--roles",
            "primary",
        ],
    )

    assert result.exit_code == 0, result.output
    docs = json.loads(out.read_text(encoding="utf-8"))
    assert [d["doc_id"] for d in docs] == [1]
    report = json.loads(out.with_suffix(".report.json").read_text(encoding="utf-8"))
    assert report["n_truncated"] == 1
    assert [
        r["doc_id"] for r in report["rejected"] if r["reasons"] == ["truncated"]
    ] == [2]
