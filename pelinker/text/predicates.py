"""Voice-aware detection of KB predicate mentions, for weak supervision.

Matching a KB label against every contiguous lemma window of the text cannot tell voice:
"X is activated by Y" matches both ``activates`` (the 1-gram ``activated``) and
``activated by`` (the 2-gram), so one mention carries both a relation and its converse.
A window cannot see an agent separated from its verb ("activated in vivo by") or a
passive with no agent at all ("NF-kB was activated"), and it cannot tell the verb
"controls" from the noun "controls".

This module reads the dependency parse instead. A KB label that is a verb predicate is
reduced to a :class:`PredicateSpec` — head verb lemma, ``-ly`` modifiers, trailing
preposition, and whether the label is itself a passive form ("regulated by"). A verb token
in the text gets **at most one** label:

- its voice is decided from the parse (passive auxiliary, passive subject, ``by`` agent,
  reduced relative; a prenominal participle — "activated T cells" — is dropped);
- the label whose own reading has the same voice is preferred ("activated by" for a
  passive mention); failing that, the opposite-voice label is used with
  ``direction="inverse"``;
- a label's preposition is part of its meaning, so it must attach to the verb in the
  text ("developed complications" is not ``develops from``; "occur during" is not
  ``occurs across``). The one exception is the agent of a passive ``by`` label, which
  may be absent ("NF-kB was activated");
- among candidates, the most specific wins: every modifier of the label must modify the
  verb in the text;
- a symmetric relation is never flipped: its direction is ``symmetric``.

The embedded span is the verb token, extended to the label's modifier when that modifier
immediately precedes it ("negatively regulated"). This matches what prediction proposes —
contiguous windows embedded by the encoder — so the parse is used to *label* training
mentions and never as a feature.

Labels that are not verb predicates ("part of", "has part", "capable of") are left to the
caller's lexical matching.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Callable

from pelinker.core.onto import SimplifiedToken
from pelinker.text.tokenize import text_to_tokens

PREPOSITIONS = ("by", "in", "of", "from", "to", "for", "with", "at", "on")
# A verb label may end in any of these; only ``PREPOSITIONS`` mark a passive form.
_LABEL_PREPOSITIONS = PREPOSITIONS + (
    "into",
    "across",
    "after",
    "before",
    "during",
    "through",
    "upon",
    "onto",
    "within",
    "against",
    "via",
)

# Past participles that end in neither "-ed" nor "-en" ("bound by" ↔ "binds to").
IRREGULAR_PARTICIPLES = frozenset(
    {
        "bound",
        "bred",
        "brought",
        "built",
        "caught",
        "cut",
        "fed",
        "found",
        "held",
        "kept",
        "led",
        "left",
        "lost",
        "made",
        "met",
        "put",
        "sent",
        "set",
        "shut",
        "sought",
        "split",
        "spread",
        "struck",
        "stuck",
        "taught",
        "told",
        "wound",
    }
)

_COPULAS = frozenset({"is", "are", "be", "was", "were"})

# Dependency labels that mark a passive clause on the verb's children.
_PASSIVE_CHILD_DEPS = frozenset({"auxpass", "nsubjpass", "agent"})
_PREP_CHILD_DEPS = frozenset({"prep", "agent", "dative", "prt"})

DIRECTION_FORWARD = "forward"
DIRECTION_INVERSE = "inverse"
DIRECTION_SYMMETRIC = "symmetric"


def is_passive_form(label: str) -> bool:
    """Whether a label reads as a participle plus preposition ("regulated by")."""
    parts = label.split()
    if len(parts) < 2 or parts[-1].lower() not in PREPOSITIONS:
        return False
    head = parts[-2].lower()
    return head.endswith("ed") or head.endswith("en") or head in IRREGULAR_PARTICIPLES


@dataclasses.dataclass(frozen=True)
class PredicateSpec:
    """A KB label reduced to what a verb mention has to agree with."""

    label: str
    head_lemma: str
    modifiers: tuple[str, ...]
    prep: str | None
    passive: bool
    symmetric: bool = False


@dataclasses.dataclass(frozen=True)
class PredicateMatch:
    """One labelled verb mention; token indices are positions in the parsed chunk."""

    label: str
    head: int
    start: int
    direction: str
    rule: str


def predicate_spec(
    label: str,
    nlp,
    *,
    symmetric: bool = False,
    tokenize: Callable[..., list[SimplifiedToken]] = text_to_tokens,
) -> PredicateSpec | None:
    """Reduce ``label`` to a verb predicate, or ``None`` when it is not one.

    The head is lemmatized inside a template sentence rather than alone: a bare label
    ("regulates", "controls") is as likely to be tagged a noun as a verb, and its lemma
    then depends on that guess. ``tokenize`` is the ``(nlp, text)`` tokenizer to use.
    """
    words = label.split()
    while len(words) > 1 and words[0].lower() in _COPULAS:
        words = words[1:]
    prep = (
        words[-1].lower()
        if len(words) > 1 and words[-1].lower() in _LABEL_PREPOSITIONS
        else None
    )
    core = words[:-1] if prep else words
    if not core:
        return None
    *mods, head = core
    if any(not m.lower().endswith("ly") for m in mods):
        return None
    passive = prep is not None and is_passive_form(f"{head} {prep}")
    # KB labels write active verbs in the third person ("regulates", "binds to"). The
    # template below forces a verb reading on whatever fills the slot, so the surface
    # form is what keeps "is tautomer of" or "coincident with" out.
    if not passive and not head.lower().endswith("s"):
        return None
    if passive:
        template = f"They were {head} {prep} it ."
    elif prep:
        template = f"It {head} {prep} them ."
    else:
        template = f"It {head} them ."
    token = next((t for t in tokenize(nlp, template) if t.text == head), None)
    if token is None or token.pos != "VERB":
        return None
    return PredicateSpec(
        label=label,
        head_lemma=token.lemma.lower(),
        modifiers=tuple(m.lower() for m in mods),
        prep=prep,
        passive=passive,
        symmetric=symmetric,
    )


def verb_voice(
    token: SimplifiedToken, children: list[SimplifiedToken]
) -> tuple[str, str] | None:
    """``(voice, rule)`` for a verb token, or ``None`` when voice cannot be told.

    A prenominal modifier ("activated T cells", "TNF-induced apoptosis", "binding sites")
    is dropped: it names a state or a kind, not a relation between two arguments, and its
    orientation is not recoverable from the parse. A present participle that takes a
    direct object is kept, as an active mention — spaCy sometimes attaches a verb phrase
    ("the kinase phosphorylating Akt") as a modifier. A bare ``VBN`` with neither passive
    marking nor a perfect ``have`` is dropped for the same reason.
    """
    deps = {c.dep for c in children}
    if token.dep == "amod" and (token.tag == "VBN" or "dobj" not in deps):
        return None
    if "agent" in deps:
        return "passive", "passive_agent"
    if deps & _PASSIVE_CHILD_DEPS:
        return "passive", "passive_agentless"
    if token.tag == "VBN":
        if token.dep == "acl":
            return "passive", "reduced_relative"
        if any(c.dep == "aux" and c.lemma.lower() == "have" for c in children):
            return "active", "active"
        return None
    return "active", "active"


class PredicateMatcher:
    """One label per verb mention, from the parse; see the module docstring."""

    def __init__(self, specs: list[PredicateSpec]) -> None:
        self._by_lemma: dict[str, list[PredicateSpec]] = defaultdict(list)
        for spec in specs:
            self._by_lemma[spec.head_lemma].append(spec)

    @property
    def labels(self) -> frozenset[str]:
        return frozenset(s.label for specs in self._by_lemma.values() for s in specs)

    def match(self, tokens: list[SimplifiedToken]) -> list[PredicateMatch]:
        """Labelled verb mentions in one parsed chunk; ``[]`` if the parse is missing."""
        parsed = [
            (t.i, t.head_i, t)
            for t in tokens
            if t.i is not None and t.head_i is not None
        ]
        if not parsed or len(parsed) != len(tokens):
            return []
        by_i = {i: t for i, _, t in parsed}
        children: dict[int, list[SimplifiedToken]] = defaultdict(list)
        for i, head_i, t in parsed:
            if head_i != i:
                children[head_i].append(t)

        matches: list[PredicateMatch] = []
        for i, _, token in parsed:
            if token.pos != "VERB":
                continue
            specs = self._by_lemma.get(token.lemma.lower())
            if not specs:
                continue
            kids = children.get(i, [])
            voiced = verb_voice(token, kids)
            if voiced is None:
                continue
            voice, rule = voiced
            chosen = _choose(specs, kids, passive=voice == "passive")
            if chosen is None:
                continue
            spec, same_voice = chosen
            if spec.symmetric:
                direction = DIRECTION_SYMMETRIC
            elif same_voice:
                direction = DIRECTION_FORWARD
            else:
                direction = DIRECTION_INVERSE
            matches.append(
                PredicateMatch(
                    label=spec.label,
                    head=i,
                    start=_span_start(i, spec, by_i),
                    direction=direction,
                    rule=rule,
                )
            )
        return matches


def _choose(
    specs: list[PredicateSpec], children: list[SimplifiedToken], *, passive: bool
) -> tuple[PredicateSpec, bool] | None:
    """Best label for one verb: same voice first, then the converse reading."""
    adverbs = {c.lemma.lower() for c in children if c.dep == "advmod"}
    preps = {c.lemma.lower() for c in children if c.dep in _PREP_CHILD_DEPS}

    def prep_satisfied(s: PredicateSpec) -> bool:
        return (
            s.prep is None
            or s.prep in preps
            or (s.passive and s.prep == "by")  # agentless passive
        )

    def best(pool: list[PredicateSpec]) -> PredicateSpec | None:
        fitting = [s for s in pool if set(s.modifiers) <= adverbs and prep_satisfied(s)]
        if not fitting:
            return None

        def rank(s: PredicateSpec) -> tuple[int, str]:
            score = 2 * len(s.modifiers) + (1 if s.prep and s.prep in preps else 0)
            return (-score, s.label)

        return min(fitting, key=rank)

    same = best([s for s in specs if s.passive == passive])
    if same is not None:
        return same, True
    other = best([s for s in specs if s.passive != passive])
    if other is not None:
        return other, False
    return None


def _span_start(i: int, spec: PredicateSpec, by_i: dict[int, SimplifiedToken]) -> int:
    """Extend the span to the label's modifier when it immediately precedes the verb."""
    prev = by_i.get(i - 1)
    if (
        prev is not None
        and spec.modifiers
        and prev.head_i == i
        and prev.lemma.lower() in spec.modifiers
    ):
        return i - 1
    return i
