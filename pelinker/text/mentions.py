"""KB mention detection without an encoder: one definition of "a mention" for every caller.

Stage (A) of the fit labels mentions to train on; the gold sampler counts mentions to
stratify abstracts. Both must agree on what a mention is, or the sample is stratified on
something the model is never trained to find. This module is that shared definition:

- **verb-predicate labels** are matched from the dependency parse, one label per verb,
  chosen by voice (:class:`pelinker.text.predicates.PredicateMatcher`);
- **other labels** ("part of", "has part") are matched on lemma windows;
- **one label per mention site** (:func:`keep_one_per_site`): a span nested in a longer
  match is dropped, and on an identical span a parse-based label beats a lexical one.

No embeddings are computed here; :mod:`pelinker.text.embed` adds them for the fit.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from typing import TypeVar

from pelinker.core.onto import SimplifiedToken
from pelinker.text.predicates import PredicateMatcher, PredicateSpec, predicate_spec
from pelinker.text.tokenize import text_to_tokens

LEXICAL_RULE = "lexical"
# Bump when detection rules change: caches of detected mentions (the gold sampler's pool)
# key on it, so a stale cache is recomputed rather than silently reused.
DETECTOR_VERSION = "2"
MAX_LEXICAL_WINDOW = 4

T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class Mention:
    """A labelled KB mention; ``a``/``b`` are character offsets into the parsed text."""

    a: int
    b: int
    label: str
    direction: str | None
    rule: str

    @property
    def is_verbal(self) -> bool:
        return self.rule != LEXICAL_RULE


def split_predicate_labels(
    labels: Iterable[str],
    nlp,
    symmetric_labels: frozenset[str] = frozenset(),
    *,
    tokenize: Callable[..., list[SimplifiedToken]] = text_to_tokens,
) -> tuple[list[PredicateSpec], list[str]]:
    """Verb-predicate labels (matched from the parse) and the rest (matched lexically)."""
    specs: list[PredicateSpec] = []
    lexical: list[str] = []
    for label in labels:
        spec = predicate_spec(
            label, nlp, symmetric=label in symmetric_labels, tokenize=tokenize
        )
        if spec is None:
            lexical.append(label)
        else:
            specs.append(spec)
    return specs, lexical


def keep_one_per_site(
    items: list[T],
    *,
    span: Callable[[T], tuple[int, int]],
    preference: Callable[[T], tuple],
) -> list[T]:
    """Keep one item per mention site.

    An item whose span lies strictly inside another item's span is dropped — the longer,
    more specific label wins ("regulates transport of" over "regulates"). On an identical
    span the smallest ``preference`` wins, so the result does not depend on input order.
    """
    if len(items) < 2:
        return items
    by_span: dict[tuple[int, int], T] = {}
    for item in items:
        key = span(item)
        if key not in by_span or preference(item) < preference(by_span[key]):
            by_span[key] = item
    spans = list(by_span)
    kept = [
        item
        for key, item in by_span.items()
        if not any(o != key and o[0] <= key[0] and key[1] <= o[1] for o in spans)
    ]
    kept.sort(key=span)
    return kept


class MentionDetector:
    """KB mentions in parsed text: parse-based for verb labels, lexical for the rest."""

    def __init__(
        self,
        labels: Iterable[str],
        nlp,
        symmetric_labels: frozenset[str] = frozenset(),
    ) -> None:
        specs, lexical = split_predicate_labels(labels, nlp, symmetric_labels)
        self.matcher = PredicateMatcher(specs)
        self.lexical_index: dict[tuple[str, ...], str] = {}
        for label in lexical:
            lemmas = tuple(t.lemma.lower() for t in text_to_tokens(nlp, label))
            if 0 < len(lemmas) <= MAX_LEXICAL_WINDOW:
                self.lexical_index.setdefault(lemmas, label)
        self._widths = sorted({len(k) for k in self.lexical_index})

    @property
    def verb_labels(self) -> frozenset[str]:
        return self.matcher.labels

    def detect(self, tokens: list[SimplifiedToken]) -> list[Mention]:
        """Mentions in one parsed text, one per site, sorted by position."""
        found: list[Mention] = []
        by_i = {t.i: t for t in tokens if t.i is not None}
        for m in self.matcher.match(tokens):
            found.append(
                Mention(
                    a=by_i[m.start].ix,
                    b=by_i[m.head].ix_end,
                    label=m.label,
                    direction=m.direction,
                    rule=m.rule,
                )
            )
        lemmas = [t.lemma.lower() for t in tokens]
        for width in self._widths:
            for k in range(len(tokens) - width + 1):
                label = self.lexical_index.get(tuple(lemmas[k : k + width]))
                if label is not None:
                    found.append(
                        Mention(
                            a=tokens[k].ix,
                            b=tokens[k + width - 1].ix_end,
                            label=label,
                            direction=None,
                            rule=LEXICAL_RULE,
                        )
                    )
        return keep_one_per_site(
            found,
            span=lambda m: (m.a, m.b),
            preference=lambda m: (0 if m.is_verbal else 1, m.label),
        )
