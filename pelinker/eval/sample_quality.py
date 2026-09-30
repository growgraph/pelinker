"""Eligibility gate for the gold sample: which held-out abstracts may be drawn at all.

A gold abstract has to be a real, English research abstract that the linker could in
principle be scored on, and that the linker never saw in training. Without this gate a
random draw lands on citation strings, news items, letters, truncated full texts and
non-English records — documents that carry no predicate mentions and dilute every rate.

Every check is a pure function of the text and metadata, and every dropped row carries the
**first** reason it failed, so a sampling report can say how many documents each rule
removed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

# Frequent English function words. Their share of word tokens separates English prose from
# other languages without a language-identification dependency.
ENGLISH_FUNCTION_WORDS = frozenset(
    {
        "the",
        "of",
        "and",
        "in",
        "to",
        "a",
        "is",
        "was",
        "were",
        "are",
        "with",
        "for",
        "that",
        "by",
        "we",
        "this",
        "on",
        "as",
        "from",
        "these",
        "which",
        "be",
        "an",
        "or",
    }
)

_WORD_RE = re.compile(r"[A-Za-z]+")
_SENTENCE_END_RE = re.compile(r"[.!?](?:\s+|$)")

REASON_YEAR = "year_before_min"
REASON_SHORT = "too_short"
REASON_LONG = "too_long"
REASON_SENTENCES = "too_few_sentences"
REASON_LANGUAGE = "not_english"
REASON_DUPLICATE = "duplicate_text"
REASON_TRAIN_OVERLAP = "in_training_corpus"


@dataclasses.dataclass(frozen=True)
class Eligibility:
    """Thresholds of the gate; defaults describe a typical research abstract."""

    min_year: int = 1990
    min_chars: int = 400
    max_chars: int = 3000
    min_sentences: int = 3
    min_english_share: float = 0.2


def text_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def english_share(text: str) -> float:
    """Share of word tokens that are common English function words."""
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if not words:
        return 0.0
    return sum(w in ENGLISH_FUNCTION_WORDS for w in words) / len(words)


def count_sentences(text: str) -> int:
    """Sentence-final punctuation followed by whitespace or the end of the text."""
    return len(_SENTENCE_END_RE.findall(text.strip()))


def drop_reason(text: str, year: object, rules: Eligibility) -> str | None:
    """The first rule ``text`` fails, or ``None`` when it is eligible.

    Duplicates and training overlap need the whole table; see :func:`apply_gate`.
    """
    try:
        year_value = int(float(str(year)))
    except ValueError:
        return REASON_YEAR
    if year_value < rules.min_year:
        return REASON_YEAR
    n = len(text)
    if n < rules.min_chars:
        return REASON_SHORT
    if n > rules.max_chars:
        return REASON_LONG
    if count_sentences(text) < rules.min_sentences:
        return REASON_SENTENCES
    if english_share(text) < rules.min_english_share:
        return REASON_LANGUAGE
    return None


@dataclasses.dataclass(frozen=True)
class TrainingKeys:
    """Identifiers of the fit corpus, for keeping gold documents out of training."""

    mag: frozenset[str]
    text_sha1: frozenset[str]


def load_training_keys(
    path: str | Path, *, id_col: int = 0, text_col: int = 1, chunksize: int = 100_000
) -> TrainingKeys:
    """Read the fit corpus (a TSV with an id column and a text column) in chunks."""
    mags: set[str] = set()
    hashes: set[str] = set()
    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=chunksize):
        ids = chunk.iloc[:, id_col].dropna().astype(str).str.strip()
        mags.update(ids)
        texts = chunk.iloc[:, text_col].fillna("").astype(str)
        hashes.update(text_sha1(t) for t in texts)
    return TrainingKeys(mag=frozenset(mags), text_sha1=frozenset(hashes))


def apply_gate(
    df: pd.DataFrame,
    rules: Eligibility,
    *,
    text_col: str = "summary",
    year_col: str = "publication_year",
    mag_col: str = "mag",
    training: TrainingKeys | None = None,
) -> pd.Series:
    """Per-row drop reason (``None`` = eligible), in rule order.

    Row-level rules first, then training overlap, then exact duplicates among the rows
    that are still eligible (the first occurrence is kept).
    """
    reasons = pd.Series(
        [drop_reason(t, y, rules) for t, y in zip(df[text_col], df[year_col])],
        index=df.index,
        dtype=object,
    )
    hashes = df[text_col].map(text_sha1)
    if training is not None:
        mags = df[mag_col].fillna("").astype(str).str.strip()
        overlap = mags.isin(training.mag) | hashes.isin(training.text_sha1)
        reasons[reasons.isna() & overlap] = REASON_TRAIN_OVERLAP
    alive = reasons.isna()
    dup = hashes[alive].duplicated(keep="first")
    reasons[dup[dup].index] = REASON_DUPLICATE
    return reasons


def reason_counts(reasons: Iterable[str | None]) -> dict[str, int]:
    """``{reason: count}`` plus ``eligible``, for the sampling report."""
    counts: dict[str, int] = {}
    for r in reasons:
        key = "eligible" if r is None or (isinstance(r, float) and pd.isna(r)) else r
        counts[key] = counts.get(key, 0) + 1
    return counts
