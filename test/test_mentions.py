"""The shared mention detector: verb labels by voice, other labels lexically, one per site."""

from __future__ import annotations

import pytest

from pelinker.text.mentions import LEXICAL_RULE, MentionDetector, keep_one_per_site
from pelinker.text.tokenize import text_to_tokens

LABELS = [
    "activates",
    "activated by",
    "regulates",
    "regulates transport of",
    "part of",
    "interacts with",
    "controls",
]


@pytest.fixture(scope="module")
def detector(nlp) -> MentionDetector:
    return MentionDetector(LABELS, nlp, frozenset({"interacts with"}))


def _detect(nlp, detector, text: str) -> list[tuple[str, str, str | None, str]]:
    return [
        (text[m.a : m.b], m.label, m.direction, m.rule)
        for m in detector.detect(text_to_tokens(nlp, text))
    ]


def test_verb_and_lexical_labels_are_both_found(nlp, detector) -> None:
    text = "NF-kB is activated by TNF, and Bcl-2 is part of the complex."

    assert _detect(nlp, detector, text) == [
        ("activated", "activated by", "forward", "passive_agent"),
        ("part of", "part of", None, LEXICAL_RULE),
    ]


def test_a_noun_does_not_count_as_a_verb_mention(nlp, detector) -> None:
    assert _detect(nlp, detector, "Patients were compared with healthy controls.") == []


def test_a_longer_lexical_match_absorbs_the_verb_inside_it(nlp, detector) -> None:
    text = "Insulin regulates transport of glucose."

    assert [m[1] for m in _detect(nlp, detector, text)] == ["regulates transport of"]


def test_verb_labels_are_split_from_lexical_ones(detector) -> None:
    assert "part of" not in detector.verb_labels
    assert {"activates", "activated by", "controls"} <= detector.verb_labels


def test_keep_one_per_site_prefers_the_smaller_preference_on_a_tie() -> None:
    items = [(0, 5, "b"), (0, 5, "a"), (10, 12, "c")]

    kept = keep_one_per_site(
        items, span=lambda x: (x[0], x[1]), preference=lambda x: (x[2],)
    )

    assert kept == [(0, 5, "a"), (10, 12, "c")]
