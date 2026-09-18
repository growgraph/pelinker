"""Derive direct/inverse label pairs for the whole KB, not just RO's declared ones.

`owl:inverseOf` covers a minority of the vocabulary, but the KB carries both members of
many more converse pairs as separate entries ("regulates" / "regulated by", "has part" /
"part of"). Directionality work needs the *whole* mapping: stage (A) has to send a passive
mention to the converse entry, and the evaluation has to know which pairs exist before it
can report accuracy by direction.

Pairs come from four rules, each recorded in ``inverse_source`` so nothing derived is
mistaken for something asserted:

- ``ro`` — declared ``owl:inverseOf``. Authoritative; never overridden.
- ``has_of`` — the OBO surface pattern ``has X`` ↔ ``X of`` ("has part" / "part of").
- ``passive_by`` — ``X-ed by`` ↔ the active entry with the same verb lemma
  ("regulated by" / "regulates").
- ``passive_prep`` — ``X-ed <prep>`` ↔ the same active lemma, for the remaining
  prepositions ("expressed in" / "expresses").

Only the first is a fact about the ontology. The other three are **candidates for human
review**: they are proposals a curator accepts or rejects, and the output marks them
``needs_review`` so an unreviewed guess can never silently become ground truth.

Each pair also gets a **canonical member**. This matters for annotation: while the KB
holds both members, a passive mention has two equally valid encodings — ("regulated by",
forward) or ("regulates", inverse) — and letting an annotator choose either is exactly the
ambiguity that put contradictory labels on the same span in the weak supervision. Gold
therefore offers only canonical labels and carries the orientation in ``direction``.
Canonical is the non-converse surface form (not ``X-ed by``/``X of``); ties break on the
smaller entity id so the choice is stable.

Output: ``data/derived/properties.synthesis.2.pairs.csv`` — the KB plus
``inverse_entity_id`` / ``inverse_label`` / ``inverse_source`` / ``needs_review`` /
``is_canonical`` / ``canonical_entity_id``.

Usage:

    uv run python run/preprocessing/derive_inverse_pairs.py \
        --kb-csv-path data/derived/properties.synthesis.2.inverse.csv \
        --output-path data/derived/properties.synthesis.2.pairs.csv
"""

from __future__ import annotations

import logging

import click
import pandas as pd
import spacy
from pelinker.core.paths import ExpandedPath

logger = logging.getLogger(__name__)

_PREPOSITIONS = ("by", "in", "of", "from", "to", "for", "with", "at", "on")


def verb_lemma_key(label: str, nlp) -> str | None:
    """Lemma of a label's leading verb phrase, ignoring a trailing preposition.

    ``regulated by`` and ``regulates`` both key on ``regulate``, which is what makes the
    two entries recognisable as one relation seen from two ends.
    """
    doc = nlp(label)
    tokens = [t for t in doc if not (t.is_punct or t.is_space)]
    if not tokens:
        return None
    if tokens[-1].text.lower() in _PREPOSITIONS:
        tokens = tokens[:-1]
    if not tokens:
        return None
    # Drop a leading copula ("is expressed in" → "expressed in").
    if tokens[0].lemma_.lower() in {"be", "is"} and len(tokens) > 1:
        tokens = tokens[1:]
    return " ".join(t.lemma_.lower() for t in tokens)


def is_passive_form(label: str) -> bool:
    parts = label.split()
    if len(parts) < 2 or parts[-1].lower() not in _PREPOSITIONS:
        return False
    head = parts[-2].lower()
    return head.endswith("ed") or head.endswith("en")


def derive_pairs(kb: pd.DataFrame, nlp) -> pd.DataFrame:
    """Attach an inverse to every entry we can, recording how it was obtained."""
    work = kb.copy()
    work["label"] = work["label"].astype(str)
    work["entity_id"] = work["entity_id"].astype(str)

    by_label = dict(zip(work["label"], work["entity_id"]))
    label_of = {v: k for k, v in by_label.items()}
    known_ids = set(work["entity_id"])

    inverse_id: dict[str, str] = {}
    source: dict[str, str] = {}

    def link(a_id: str, b_id: str, rule: str) -> None:
        """Record a symmetric pair, never overwriting a stronger rule."""
        for x, y in ((a_id, b_id), (b_id, a_id)):
            if source.get(x) == "ro":
                continue
            if x in inverse_id and source.get(x) != rule:
                continue
            inverse_id[x] = y
            source[x] = rule

    # 1. Declared inverses win outright.
    if "inverse_entity_id" in work.columns:
        for _, row in work.iterrows():
            other = row["inverse_entity_id"]
            if isinstance(other, str) and other in known_ids:
                inverse_id[row["entity_id"]] = other
                source[row["entity_id"]] = "ro"

    # 2. "has X" <-> "X of".
    for label, eid in by_label.items():
        low = label.lower()
        if low.startswith("has "):
            stem = low[4:].strip()
            partner = by_label.get(f"{stem} of")
            if partner is not None:
                link(eid, partner, "has_of")

    # 3/4. Passive surface forms against the active entry with the same verb lemma.
    lemma_to_active: dict[str, list[str]] = {}
    for label, eid in by_label.items():
        if is_passive_form(label):
            continue
        key = verb_lemma_key(label, nlp)
        if key:
            lemma_to_active.setdefault(key, []).append(eid)

    for label, eid in by_label.items():
        if not is_passive_form(label):
            continue
        key = verb_lemma_key(label, nlp)
        if not key:
            continue
        candidates = [c for c in lemma_to_active.get(key, []) if c != eid]
        if len(candidates) != 1:
            # Ambiguous or absent: leave it for a curator rather than guessing.
            continue
        rule = "passive_by" if label.lower().endswith(" by") else "passive_prep"
        link(eid, candidates[0], rule)

    work["inverse_entity_id"] = work["entity_id"].map(inverse_id)
    work["inverse_label"] = work["inverse_entity_id"].map(label_of)
    work["inverse_source"] = work["entity_id"].map(source)
    work["needs_review"] = work["inverse_source"].isin(
        ["has_of", "passive_by", "passive_prep"]
    )

    def converse_surface(label: str) -> bool:
        low = label.lower()
        return is_passive_form(label) or low.endswith(" of")

    canonical: dict[str, str] = {}
    for _, row in work.iterrows():
        eid, label = row["entity_id"], row["label"]
        other = inverse_id.get(eid)
        if other is None:
            canonical[eid] = eid  # unpaired entries are their own canonical form
            continue
        other_label = label_of[other]
        self_converse, other_converse = (
            converse_surface(label),
            converse_surface(other_label),
        )
        if self_converse != other_converse:
            canonical[eid] = other if self_converse else eid
        else:
            canonical[eid] = min(eid, other)

    work["canonical_entity_id"] = work["entity_id"].map(canonical)
    work["is_canonical"] = work["canonical_entity_id"] == work["entity_id"]
    return work


@click.command()
@click.option(
    "--kb-csv-path",
    default="data/derived/properties.synthesis.2.inverse.csv",
    show_default=True,
    type=ExpandedPath(exists=True),
)
@click.option(
    "--output-path",
    default="data/derived/properties.synthesis.2.pairs.csv",
    show_default=True,
    type=ExpandedPath(),
)
@click.option("--nlp-model", default="en_core_web_lg", show_default=True)
def main(kb_csv_path: str, output_path: str, nlp_model: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    kb = pd.read_csv(kb_csv_path)
    nlp = spacy.load(nlp_model, exclude=["parser", "ner"])

    paired = derive_pairs(kb, nlp)

    counts = paired["inverse_source"].value_counts(dropna=False).to_dict()
    logger.info("Entries: %d", len(paired))
    logger.info("With an inverse: %d", int(paired["inverse_entity_id"].notna().sum()))
    logger.info("By rule: %s", counts)
    logger.info("Needing review: %d", int(paired["needs_review"].sum()))
    logger.info(
        "Canonical labels offered to annotation: %d of %d",
        int(paired["is_canonical"].sum()),
        len(paired),
    )

    cols = [
        "entity_id",
        "label",
        "description",
        "inverse_entity_id",
        "inverse_label",
        "inverse_source",
        "needs_review",
        "is_canonical",
        "canonical_entity_id",
        "example",
    ]
    paired[[c for c in cols if c in paired.columns]].to_csv(output_path, index=False)
    logger.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
