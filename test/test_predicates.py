"""Voice-aware predicate detection: one label per verb mention, chosen by voice."""

from __future__ import annotations

import pytest

from pelinker.core.onto import SimplifiedToken
from pelinker.text.embed import LEXICAL_RULE, one_label_per_span
from pelinker.text.predicates import (
    PredicateMatcher,
    is_passive_form,
    predicate_spec,
    verb_voice,
)
from pelinker.text.tokenize import text_to_tokens

LABELS = [
    "activates",
    "activated by",
    "regulates",
    "regulated by",
    "negatively regulates",
    "negatively regulated by",
    "controls",
    "binds to",
    "bound by",
    "transmitted by",
    "interacts with",
    "connected to",
    "connects",
    "expresses",
    "expressed in",
]
SYMMETRIC = {"interacts with", "connected to"}


@pytest.fixture(scope="module")
def matcher(nlp) -> PredicateMatcher:
    specs = [
        predicate_spec(label, nlp, symmetric=label in SYMMETRIC) for label in LABELS
    ]
    assert all(s is not None for s in specs), [
        label for label, s in zip(LABELS, specs) if s is None
    ]
    return PredicateMatcher([s for s in specs if s is not None])


def _matches(nlp, matcher, text: str) -> list[tuple[str, str, str, str]]:
    tokens = text_to_tokens(nlp, text)
    by_i = {t.i: t for t in tokens}
    return [
        (
            text[by_i[m.start].ix : by_i[m.head].ix_end],
            m.label,
            m.direction,
            m.rule,
        )
        for m in matcher.match(tokens)
    ]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("TNF activates NF-kB.", [("activates", "activates", "forward", "active")]),
        (
            "NF-kB is activated by TNF.",
            [("activated", "activated by", "forward", "passive_agent")],
        ),
        # The agent is not adjacent: a lemma window would have labelled this `activates`.
        (
            "NF-kB was activated in vivo by TNF.",
            [("activated", "activated by", "forward", "passive_agent")],
        ),
        (
            "NF-kB was activated.",
            [("activated", "activated by", "forward", "passive_agentless")],
        ),
        (
            "IL-6 is negatively regulated by TAMs.",
            [
                (
                    "negatively regulated",
                    "negatively regulated by",
                    "forward",
                    "passive_agent",
                )
            ],
        ),
        (
            "MDM2 is bound by p53.",
            [("bound", "bound by", "forward", "passive_agent")],
        ),
        # No active entry for "transmit": the converse label, read the other way.
        (
            "Mosquitoes transmit the virus.",
            [("transmit", "transmitted by", "inverse", "active")],
        ),
        (
            "p53 interacts with MDM2.",
            [("interacts", "interacts with", "symmetric", "active")],
        ),
        (
            "A is connected to B.",
            [("connected", "connected to", "symmetric", "passive_agentless")],
        ),
        (
            "The protein activated by TNF was degraded.",
            [("activated", "activated by", "forward", "passive_agent")],
        ),
    ],
)
def test_each_verb_mention_gets_one_label_by_voice(
    nlp, matcher, text: str, expected
) -> None:
    assert _matches(nlp, matcher, text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Patients were compared with healthy controls.",  # noun, not the verb
        "The activated T cells expanded.",  # prenominal participle: dropped
    ],
)
def test_non_verbal_and_adjectival_uses_are_not_labelled(nlp, matcher, text) -> None:
    assert _matches(nlp, matcher, text) == []


@pytest.mark.parametrize(
    "label",
    ["is tautomer of", "coincident with", "has part", "part of", "is marker for"],
)
def test_labels_that_are_not_verb_predicates_have_no_spec(nlp, label: str) -> None:
    assert predicate_spec(label, nlp) is None


def test_a_passive_label_is_recognised_as_one(nlp) -> None:
    spec = predicate_spec("negatively regulated by", nlp)

    assert spec is not None
    assert (spec.head_lemma, spec.modifiers, spec.prep, spec.passive) == (
        "regulate",
        ("negatively",),
        "by",
        True,
    )


def test_is_passive_form_needs_a_participle_and_a_preposition() -> None:
    assert is_passive_form("bound by")
    assert is_passive_form("expressed in")
    assert not is_passive_form("binds to")
    assert not is_passive_form("regulated")


def _tok(
    i: int, text: str, *, head: int, dep: str, tag: str = "VBN"
) -> SimplifiedToken:
    return SimplifiedToken(
        ix=i * 10,
        ix_end=i * 10 + len(text),
        text=text,
        lemma=text,
        tag=tag,
        pos="VERB",
        i=i,
        head_i=head,
        dep=dep,
    )


def test_a_bare_past_participle_has_no_voice() -> None:
    verb = _tok(0, "activated", head=0, dep="ROOT")

    assert verb_voice(verb, []) is None


def test_a_perfect_participle_is_active() -> None:
    verb = _tok(1, "activated", head=1, dep="ROOT")
    aux = _tok(0, "have", head=1, dep="aux", tag="VBZ")

    assert verb_voice(verb, [aux]) == ("active", "active")


def test_without_a_parse_nothing_is_matched() -> None:
    token = SimplifiedToken(
        ix=0, ix_end=9, text="activates", lemma="activate", tag="VBZ"
    )

    assert PredicateMatcher([]).match([token]) == []


def _row(a: int, b: int, entity: str, rule: str) -> dict:
    return {"a_abs": a, "b_abs": b, "entity": entity, "surface_rule": rule}


def test_a_span_nested_in_a_longer_match_is_dropped() -> None:
    rows = [
        _row(0, 9, "regulates", "active"),
        _row(0, 22, "regulates transport of", LEXICAL_RULE),
    ]

    assert [r["entity"] for r in one_label_per_span(rows)] == ["regulates transport of"]


def test_on_an_identical_span_the_parse_label_wins() -> None:
    rows = [
        _row(0, 9, "activates", LEXICAL_RULE),
        _row(0, 9, "activated by", "passive_agent"),
    ]

    assert [r["entity"] for r in one_label_per_span(rows)] == ["activated by"]


def test_disjoint_mentions_are_all_kept() -> None:
    rows = [_row(0, 5, "causes", "active"), _row(10, 16, "caused by", "passive_agent")]

    assert len(one_label_per_span(rows)) == 2


@pytest.mark.parametrize(
    "text",
    [
        "Five patients developed complications.",  # not `develops from`
        "Most deaths occur during labor.",  # not `occurs in`
    ],
)
def test_a_label_preposition_must_attach_to_the_verb(nlp, text: str) -> None:
    labels = ["develops from", "occurs in"]
    specs = [predicate_spec(label, nlp) for label in labels]
    matcher = PredicateMatcher([s for s in specs if s is not None])

    assert matcher.match(text_to_tokens(nlp, text)) == []


def test_the_preposition_is_matched_when_present(nlp) -> None:
    spec = predicate_spec("develops from", nlp)
    assert spec is not None
    matcher = PredicateMatcher([spec])

    found = matcher.match(text_to_tokens(nlp, "The tumour develops from stem cells."))

    assert [m.label for m in found] == ["develops from"]


def test_a_prenominal_modifier_without_an_object_is_not_a_mention(nlp, matcher) -> None:
    assert _matches(nlp, matcher, "We found enrichment of binding sites.") == []
