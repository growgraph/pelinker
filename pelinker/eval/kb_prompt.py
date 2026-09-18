"""Rendering the property KB into a prompt catalog.

The KB descriptions come from OBO sources and are not prose written for a reader. Left
raw they carry four things that hurt an annotation prompt, in rough order of severity:

1. **Embedded newlines.** A description laid out over several lines silently turns one
   relation into several entries, so the model is shown fragments like ``. X part_of Y``
   as if they were relations to choose from.
2. **Pure cross-references.** ``inverse of develops from``,
   ``[copied from inverse property 'occurs in']`` — no definitional content, and they
   introduce a second, unrelated notion of "inverse" next to the converse field.
3. **Formal restatements.** ``Formally: x precedes y iff ω(x) <= α(y) ∧ …`` — Greek
   symbols and quantifier notation that say nothing about how the relation is *worded* in
   a sentence, which is the only thing the annotator has to judge.
4. **Boilerplate**: source URLs, indentation, trailing whitespace.

Entries are numbered and labels quoted so each is a discrete, unambiguously delimited
item — most labels contain spaces ("has part", "is concretized as"), and an unquoted
label in a dash-joined line has no visible boundary.
"""

from __future__ import annotations

import re

import pandas as pd

_WHITESPACE_RE = re.compile(r"\s+")
_LEADING_BRACKET_RE = re.compile(r"^\s*\[[^\]]*\]\s*")
_URL_RE = re.compile(r"https?://\S+")
_FORMAL_RE = re.compile(r"\s*\bFormally\b\s*[:,].*$", re.IGNORECASE | re.DOTALL)
_XREF_ONLY_RE = re.compile(r"^inverse\s+(of|property)\b", re.IGNORECASE)

MAX_DESCRIPTION_CHARS = 240


def clean_description(raw: object, *, max_chars: int = MAX_DESCRIPTION_CHARS) -> str:
    """Normalize one KB description into a single-line definition, or ``""``.

    Returns an empty string when nothing definitional survives — a cross-reference to
    another relation tells the annotator nothing the converse field does not already say.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    text = str(raw)
    text = _LEADING_BRACKET_RE.sub("", text)
    text = _URL_RE.sub("", text)
    text = _FORMAL_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if not text or _XREF_ONLY_RE.match(text):
        return ""
    if len(text) > max_chars:
        cut = text[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:.")
        text = f"{cut}…"
    return text


def render_kb_catalog(
    kb: pd.DataFrame,
    *,
    include_converse: bool = True,
    max_chars: int = MAX_DESCRIPTION_CHARS,
) -> str:
    """Numbered catalog, exactly one line per relation.

    One line per entry is a contract, not a formatting preference: the count of lines has
    to equal the count of relations, or the model is choosing from a list that does not
    match the vocabulary the answers are validated against.
    """
    lines: list[str] = []
    for position, (_, row) in enumerate(kb.iterrows(), start=1):
        label = _WHITESPACE_RE.sub(" ", str(row["label"])).strip()
        parts = [f'{position}. "{label}"']
        description = clean_description(row.get("description"), max_chars=max_chars)
        if description:
            parts.append(f"def: {description}")
        if include_converse:
            converse = row.get("inverse_label")
            if isinstance(converse, str) and converse.strip():
                clean_converse = _WHITESPACE_RE.sub(" ", converse).strip()
                parts.append(f'converse wording: "{clean_converse}"')
        line = " | ".join(parts)
        if "\n" in line:  # pragma: no cover - defended against, not expected
            raise ValueError(f"catalog line {position} contains a newline: {line!r}")
        lines.append(line)
    return "\n".join(lines)
