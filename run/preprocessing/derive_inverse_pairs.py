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

**Symmetric relations have no converse.** "interacts with", "overlaps", "connected to"
read the same from either end, so a reversed or passive mention is still the same relation
with ``direction=symmetric``. They are flagged ``is_symmetric`` — from RO (see
``extract_properties_ro.py``) plus a curated list for entries RO does not cover
(``--symmetric-csv``, e.g. "binds to", "associated with") — and the surface rules never
pair them: a symmetric label that
merely looks passive ("connected to") would otherwise be folded onto an unrelated
look-alike ("connects").

Each pair also gets a **canonical member**. This matters for annotation: while the KB
holds both members, a passive mention has two equally valid encodings — ("regulated by",
forward) or ("regulates", inverse) — and letting an annotator choose either is exactly the
ambiguity that put contradictory labels on the same span in the weak supervision. Gold
therefore offers only canonical labels and carries the orientation in ``direction``.
Canonical is the non-converse surface form (not ``X-ed by``/``X of``); ties break on the
smaller entity id so the choice is stable.

Output: ``data/derived/properties.synthesis.2.pairs.csv`` — the KB plus
``inverse_entity_id`` / ``inverse_label`` / ``inverse_source`` / ``needs_review`` /
``is_symmetric`` / ``is_canonical`` / ``canonical_entity_id``. This is the KB the gold
pipeline consumes; the ``*.inverse.csv`` read here is its *input*, and has none of those
columns.

**The review loop.** Every unreviewed derived pair is written to
``<output>.pending_review.csv``. A curator fills in ``verdict`` (``accept`` or ``reject``)
and passes the file back as ``--review-csv``: an accepted pair loses ``needs_review``, and
a rejected one is unlinked outright, so each member goes back to being its own canonical
form. Rejection matters more than it looks — an unreviewed wrong pair collapses two
distinct relations onto one canonical id, and prompts, gold and scores then agree with the
mistake instead of exposing it.

Usage:

    uv run python run/preprocessing/derive_inverse_pairs.py \
        --kb-csv-path data/derived/properties.synthesis.2.inverse.csv \
        --output-path data/derived/properties.synthesis.2.pairs.csv \
        --review-csv data/curated/inverse_pairs.review.csv \
        --ro-csv-path data/derived/properties.ro.csv \
        --symmetric-csv data/curated/symmetric.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import pandas as pd
import spacy
from pelinker.core.paths import ExpandedPath
from pelinker.text.predicates import PREPOSITIONS as _PREPOSITIONS
from pelinker.text.predicates import is_passive_form

logger = logging.getLogger(__name__)


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


def attach_symmetry(
    kb: pd.DataFrame,
    ro: pd.DataFrame | None,
    curated: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """Ensure ``is_symmetric``: the KB's own flag, else RO's, plus the ``curated`` ids.

    The curated KB is hand-selected from RO and does not carry RO's axioms, so symmetry is
    joined back on ``entity_id``. Entries RO does not know (GeneWays, PEL) are symmetric
    only when a curator lists them.
    """
    work = kb.copy()
    ids = work["entity_id"].astype(str)
    if "is_symmetric" in work.columns:
        flag = work["is_symmetric"].fillna(False).astype(bool)
    elif ro is not None and "is_symmetric" in ro.columns:
        from_ro = set(ro.loc[ro["is_symmetric"].astype(bool), "entity_id"].astype(str))
        flag = ids.isin(from_ro)
    else:
        flag = pd.Series(False, index=work.index)
    work["is_symmetric"] = flag | ids.isin(curated)
    return work


def load_symmetric(path: str | Path) -> frozenset[str]:
    """Curated symmetric entity ids (``entity_id`` column; ``label``/``note`` are for people)."""
    sheet = pd.read_csv(path, dtype=str).fillna("")
    if "entity_id" not in sheet.columns:
        raise click.ClickException(f"{path}: symmetric sheet needs an entity_id column")
    return frozenset(e.strip() for e in sheet["entity_id"] if e.strip())


def derive_pairs(kb: pd.DataFrame, nlp) -> pd.DataFrame:
    """Attach an inverse to every entry we can, recording how it was obtained."""
    work = kb.copy()
    work["label"] = work["label"].astype(str)
    work["entity_id"] = work["entity_id"].astype(str)
    if "is_symmetric" not in work.columns:
        work["is_symmetric"] = False
    symmetric = set(work.loc[work["is_symmetric"].astype(bool), "entity_id"])

    by_label = dict(zip(work["label"], work["entity_id"]))
    label_of = {v: k for k, v in by_label.items()}
    known_ids = set(work["entity_id"])

    inverse_id: dict[str, str] = {}
    source: dict[str, str] = {}

    def link(a_id: str, b_id: str, rule: str) -> None:
        """Record a pair both ways, never overwriting a stronger rule.

        A symmetric relation is its own converse, so a surface rule that pairs it with
        another entry is always wrong.
        """
        if a_id in symmetric or b_id in symmetric:
            return
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


_VERDICTS = ("accept", "reject")

REVIEW_COLUMNS = ("entity_id", "inverse_entity_id", "verdict", "note")


def load_review(path: str | Path) -> dict[frozenset[str], str]:
    """Curator verdicts, keyed on the unordered pair so one row decides both members."""
    review = pd.read_csv(path, dtype=str).fillna("")
    missing = [c for c in REVIEW_COLUMNS if c not in review.columns]
    if missing:
        raise click.ClickException(
            f"{path}: review sheet is missing column(s) {', '.join(missing)}"
        )
    verdicts: dict[frozenset[str], str] = {}
    for _, row in review.iterrows():
        verdict = row["verdict"].strip().lower()
        if not verdict:
            continue
        if verdict not in _VERDICTS:
            raise click.ClickException(
                f"{path}: verdict for {row['entity_id']}/{row['inverse_entity_id']} must "
                f"be one of {list(_VERDICTS)}, got {row['verdict']!r}"
            )
        verdicts[
            frozenset({row["entity_id"].strip(), row["inverse_entity_id"].strip()})
        ] = verdict
    return verdicts


def apply_review(
    paired: pd.DataFrame, verdicts: dict[frozenset[str], str]
) -> pd.DataFrame:
    """Accepted pairs stop needing review; rejected pairs are unlinked entirely.

    A rejected pair is not merely "unreviewed": the curator has said these two entries are
    *not* converses, so leaving the link in place would keep collapsing them onto one
    canonical id. Each member becomes unpaired and its own canonical form.
    """
    if not verdicts:
        return paired
    work = paired.copy()
    rejected: set[str] = set()
    for _, row in work.iterrows():
        other = row["inverse_entity_id"]
        if not isinstance(other, str) or not other:
            continue
        verdict = verdicts.get(frozenset({str(row["entity_id"]), other}))
        if verdict == "reject":
            rejected.add(str(row["entity_id"]))
        elif verdict == "accept":
            work.loc[work["entity_id"] == row["entity_id"], "needs_review"] = False

    if rejected:
        mask = work["entity_id"].isin(rejected)
        work.loc[mask, ["inverse_entity_id", "inverse_label", "inverse_source"]] = None
        work.loc[mask, "needs_review"] = False
        work.loc[mask, "canonical_entity_id"] = work.loc[mask, "entity_id"]
        work["is_canonical"] = work["canonical_entity_id"] == work["entity_id"]
    return work


def pending_review_sheet(paired: pd.DataFrame) -> pd.DataFrame:
    """One row per unreviewed heuristic pair, with the wording a curator has to judge."""
    pending = paired.loc[paired["needs_review"].astype(bool)]
    rows = []
    seen: set[frozenset[str]] = set()
    for _, row in pending.iterrows():
        key = frozenset({str(row["entity_id"]), str(row["inverse_entity_id"])})
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "entity_id": row["entity_id"],
                "label": row["label"],
                "inverse_entity_id": row["inverse_entity_id"],
                "inverse_label": row["inverse_label"],
                "inverse_source": row["inverse_source"],
                "canonical_entity_id": row["canonical_entity_id"],
                "description": row.get("description", ""),
                "verdict": "",
                "note": "",
            }
        )
    return pd.DataFrame(rows, columns=list(rows[0]) if rows else list(REVIEW_COLUMNS))


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
@click.option(
    "--review-csv",
    default="data/curated/inverse_pairs.review.csv",
    show_default=True,
    help=(
        "Curator verdicts (entity_id, inverse_entity_id, verdict=accept|reject, note). "
        "Missing file is fine on a first run."
    ),
)
@click.option(
    "--ro-csv-path",
    default="data/derived/properties.ro.csv",
    show_default=True,
    help=(
        "RO extraction carrying `is_symmetric`; used when the KB lacks the column. "
        "Missing file means no entry is treated as symmetric."
    ),
)
@click.option(
    "--symmetric-csv",
    default="data/curated/symmetric.csv",
    show_default=True,
    help="Curated symmetric entity ids, for entries RO does not declare. Optional.",
)
@click.option("--nlp-model", default="en_core_web_lg", show_default=True)
def main(
    kb_csv_path: str,
    output_path: str,
    review_csv: str,
    ro_csv_path: str,
    symmetric_csv: str,
    nlp_model: str,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    kb = pd.read_csv(kb_csv_path)
    ro_path = Path(ro_csv_path).expanduser()
    ro = pd.read_csv(ro_path) if ro_path.exists() else None
    if ro is None:
        logger.warning("No RO extraction at %s — RO symmetry is not applied", ro_path)
    sym_path = Path(symmetric_csv).expanduser()
    curated = load_symmetric(sym_path) if sym_path.exists() else frozenset()
    unknown = curated - set(kb["entity_id"].astype(str))
    if unknown:
        raise click.ClickException(
            f"{sym_path}: ids not in the KB: {', '.join(sorted(unknown))}"
        )
    kb = attach_symmetry(kb, ro, curated)
    nlp = spacy.load(nlp_model, exclude=["parser", "ner"])

    paired = derive_pairs(kb, nlp)

    review_path = Path(review_csv).expanduser()
    if review_path.exists():
        verdicts = load_review(review_path)
        paired = apply_review(paired, verdicts)
        logger.info("Applied %d curator verdicts from %s", len(verdicts), review_path)
    else:
        logger.info(
            "No review sheet at %s — every derived pair stays unreviewed", review_path
        )

    pending = pending_review_sheet(paired)
    pending_path = Path(output_path).with_suffix(".pending_review.csv")
    if len(pending):
        pending.to_csv(pending_path, index=False)
        logger.warning(
            "%d derived pairs await a curator verdict → %s. Fill in `verdict` "
            "(accept|reject) and pass the file as --review-csv; until then a wrong pair "
            "silently merges two relations onto one canonical id.",
            len(pending),
            pending_path,
        )
    elif pending_path.exists():
        pending_path.unlink()

    counts = paired["inverse_source"].value_counts(dropna=False).to_dict()
    logger.info("Entries: %d", len(paired))
    logger.info("With an inverse: %d", int(paired["inverse_entity_id"].notna().sum()))
    logger.info("By rule: %s", counts)
    logger.info("Needing review: %d", int(paired["needs_review"].sum()))
    logger.info("Symmetric: %d", int(paired["is_symmetric"].sum()))
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
        "is_symmetric",
        "is_canonical",
        "canonical_entity_id",
        "example",
    ]
    paired[[c for c in cols if c in paired.columns]].to_csv(output_path, index=False)
    logger.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
