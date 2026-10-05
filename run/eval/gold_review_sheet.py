"""Human-verification round-trip for LLM-pre-annotated gold, plus agreement stats.

``export`` flattens one gold ``*.json`` (the ``annotate_llm.py`` output) into a review
TSV — one row per candidate span with a marked context window and empty ``verdict`` /
``fix_entity_id`` / ``fix_direction`` columns — for spreadsheet review. With ``--other``
(a second annotator's file), each row also shows the other annotator's call
(``other_entity_id``, ``other_label``, ``other_direction``), spans only the second
annotator proposed are appended as ``origin=other`` rows, and the command prints Cohen's
κ.

Agreement is read in the reference inventory's tiers 0 and 1
(:mod:`pelinker.eval.reference`): ids are compared after folding converse members onto
their canonical member and, with ``--equivalences``, along the KB-implied identities.
Two annotators who call one mention by two names of the same relation therefore agree.
``agrees_direction`` is filled only where the relation agrees, and a symmetric relation
agrees in any direction. κ is reported raw and folded.

The second annotator may cover fewer documents than the first (a slice, or documents it
could not answer), so two things follow: rows on documents it never saw read
``not-covered`` rather than an empty disagreement, and κ is computed over the shared
documents alone.

``import`` merges the filled sheet back: ``verdict`` ∈ ``accept`` / ``reject`` / ``fix``
(with the ``fix_*`` columns), everything else is an error so nothing is silently kept.
Accepted/fixed hits become ``source="human"``. Accepting an ``origin=other`` row needs
``--other`` as well, since the hit itself (and its argument spans) lives in that file —
without it, verified gold could never exceed the first annotator's recall.

``agreement`` prints κ between two gold files without exporting a sheet.

Usage:

    uv run python run/eval/gold_review_sheet.py export \
        --gold gold.llm-a.json --other gold.llm-b.json --kb-csv-path <kb> \
        --equivalences data/curated/equivalences.csv --output review.tsv
    uv run python run/eval/gold_review_sheet.py import \
        --gold gold.llm-a.json --sheet review.tsv --other gold.llm-b.json \
        --verified-by <name> --output gold.verified.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from pelinker.core.paths import ExpandedPath
from pelinker.eval.reference import fold_map, folded_symmetric_ids, load_equivalences

logger = logging.getLogger(__name__)

_CONTEXT_CHARS = 60
_VERDICTS = {"accept", "reject", "fix"}


def load_gold(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, list) else [data]


def _context(text: str, a: int, b: int) -> str:
    lo = max(0, a - _CONTEXT_CHARS)
    hi = min(len(text), b + _CONTEXT_CHARS)
    return (
        (text[lo:a] + "【" + text[a:b] + "】" + text[b:hi])
        .replace("\t", " ")
        .replace("\n", " ")
    )


def _spans_overlap(a1: int, b1: int, a2: int, b2: int) -> bool:
    return a1 < b2 and a2 < b1


def shared_doc_ids(docs_a: list[dict], docs_b: list[dict]) -> set:
    """Doc ids both annotators actually saw.

    The second annotator normally covers only the ``double`` slice. Scoring agreement over
    every document of the first would count each of its hits on an unannotated document as
    a disagreement, driving κ toward 0 for a reason that has nothing to do with agreement.
    """
    return {d["doc_id"] for d in docs_a} & {d["doc_id"] for d in docs_b}


def align_hits(
    docs_a: list[dict], docs_b: list[dict], *, restrict_to_shared: bool = False
) -> list[tuple[dict | None, dict | None]]:
    """Greedy span alignment of two annotators' hits per document (by doc_id).

    Pairs are ``(hit_a, hit_b)`` with either side ``None`` when only one annotator marked
    that span. B-only spans are emitted last, per document, so a caller that only wants
    A's rows can ignore them by skipping ``hit_a is None``.
    """
    shared = shared_doc_ids(docs_a, docs_b) if restrict_to_shared else None
    b_by_doc = {d["doc_id"]: list(d.get("ground_truth") or []) for d in docs_b}
    pairs: list[tuple[dict | None, dict | None]] = []
    for doc in docs_a:
        if shared is not None and doc["doc_id"] not in shared:
            continue
        others = b_by_doc.get(doc["doc_id"], [])
        used: set[int] = set()
        for hit in doc.get("ground_truth") or []:
            match = None
            for j, other in enumerate(others):
                if j in used:
                    continue
                if _spans_overlap(hit["a"], hit["b"], other["a"], other["b"]):
                    match = other
                    used.add(j)
                    break
            pairs.append((hit, match))
        for j, other in enumerate(others):
            if j not in used:
                pairs.append((None, other))
    return pairs


def load_fold(
    kb: pd.DataFrame, equivalences_path: str | None
) -> tuple[dict[str, str], frozenset[str]]:
    """The T̂ tier 0+1 fold of ``kb`` and the folded ids of its symmetric relations."""
    equivalences = (
        None
        if equivalences_path is None
        else load_equivalences(equivalences_path, kb=kb)
    )
    fold = fold_map(kb, equivalences)
    return fold, folded_symmetric_ids(kb, fold)


def ids_agree(id_a: object, id_b: object, fold: dict[str, str]) -> bool:
    a, b = str(id_a), str(id_b)
    return fold.get(a, a) == fold.get(b, b)


def directions_agree(
    hit: dict, other: dict, fold: dict[str, str], symmetric: frozenset[str]
) -> bool | None:
    """Direction agreement on a span both annotators linked to the same relation.

    ``None`` where it is not defined: the relations differ (a direction comparison
    between two relations says nothing), or a side gave no direction. A symmetric
    relation has no orientation, so any two directions agree.
    """
    if not ids_agree(hit["entity_id"], other["entity_id"], fold):
        return None
    relation = fold.get(str(hit["entity_id"]), str(hit["entity_id"]))
    if relation in symmetric:
        return True
    if not hit.get("direction") or not other.get("direction"):
        return None
    return str(hit["direction"]) == str(other["direction"])


def kappa_report(
    docs_a: list[dict], docs_b: list[dict], *, fold: dict[str, str] | None = None
) -> dict:
    """Cohen's κ on entity id and direction over span-aligned pairs.

    Computed over the documents **both** annotators covered; ``n_docs_shared`` reports how
    many that was, and a κ over zero shared documents is absent rather than 0.

    An unmatched span enters the entity-id κ as an explicit ``∅`` category on the side that
    missed it — in both directions, so an annotator that hallucinates spans and one that
    misses them are penalized alike. Direction κ is computed over pairs where both sides
    matched and gave a direction.

    With ``fold`` (the T̂ tier 0+1 map), ``kappa_entity_id_folded`` is added: the same κ
    after both sides' ids are folded, so naming one relation two ways is not a
    disagreement.
    """
    pairs = align_hits(docs_a, docs_b, restrict_to_shared=True)
    ids_a = [("∅" if h is None else str(h["entity_id"])) for h, _ in pairs]
    ids_b = [("∅" if o is None else str(o["entity_id"])) for _, o in pairs]
    report: dict = {
        "n_docs_shared": len(shared_doc_ids(docs_a, docs_b)),
        "n_pairs": len(pairs),
        "n_span_matched": sum(1 for h, o in pairs if h is not None and o is not None),
        "n_only_a": sum(1 for h, o in pairs if h is not None and o is None),
        "n_only_b": sum(1 for h, o in pairs if h is None and o is not None),
    }
    if len(set(ids_a) | set(ids_b)) > 1 and pairs:
        report["kappa_entity_id"] = float(cohen_kappa_score(ids_a, ids_b))
    if fold is not None and pairs:
        folded_a = [fold.get(i, i) for i in ids_a]
        folded_b = [fold.get(i, i) for i in ids_b]
        if len(set(folded_a) | set(folded_b)) > 1:
            report["kappa_entity_id_folded"] = float(
                cohen_kappa_score(folded_a, folded_b)
            )
    directed = [
        (str(h.get("direction")), str(o.get("direction")))
        for h, o in pairs
        if h is not None and o is not None and h.get("direction") and o.get("direction")
    ]
    report["n_direction_pairs"] = len(directed)
    if directed and len({d for pair in directed for d in pair}) > 1:
        da, db = zip(*directed)
        report["kappa_direction"] = float(cohen_kappa_score(list(da), list(db)))
    return report


@click.group()
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


@main.command("export")
@click.option("--gold", required=True, type=ExpandedPath(exists=True))
@click.option(
    "--other",
    default=None,
    type=ExpandedPath(exists=True),
    help="Second annotator's gold over the same docs (adds agreement columns + κ).",
)
@click.option("--kb-csv-path", required=True, type=ExpandedPath(exists=True))
@click.option(
    "--equivalences",
    default=None,
    type=ExpandedPath(exists=True),
    help="KB-implied identities (T̂ tier 1) folded before agreement is read.",
)
@click.option("--output", required=True, type=ExpandedPath())
def export_cmd(
    gold: str,
    other: str | None,
    kb_csv_path: str,
    equivalences: str | None,
    output: str,
) -> None:
    if Path(output).exists():
        # A sheet on disk may hold hours of verdicts; a re-export must go to a new file.
        raise click.ClickException(
            f"{output} exists; export to a new path rather than overwrite a review sheet"
        )
    docs = load_gold(gold)
    kb = pd.read_csv(kb_csv_path)
    label_of = dict(zip(kb["entity_id"].astype(str), kb["label"].astype(str)))
    fold, symmetric = load_fold(kb, equivalences)

    text_of = {doc["doc_id"]: doc["text"] for doc in docs}

    def _row(doc_id, hit: dict, *, origin: str) -> dict:
        text = text_of[doc_id]
        return {
            "doc_id": doc_id,
            "origin": origin,
            "itext": hit.get("itext"),
            "a": hit["a"],
            "b": hit["b"],
            "context": _context(text, hit["a"], hit["b"]),
            "surface": hit.get("surface", text[hit["a"] : hit["b"]]),
            "entity_id": hit["entity_id"],
            "label": label_of.get(str(hit["entity_id"]), ""),
            "direction": hit.get("direction") or "",
            # "False" marks a paraphrase: the surface does not use the label's wording.
            "surface_anchored": hit.get("surface_anchored", ""),
            "confidence": hit.get("confidence", ""),
            "annotator": hit.get("annotator", ""),
        }

    rows: list[dict] = []
    if other is None:
        for doc in docs:
            for hit in doc.get("ground_truth") or []:
                rows.append(_row(doc["doc_id"], hit, origin="gold"))
    else:
        docs_b = load_gold(other)
        shared = shared_doc_ids(docs, docs_b)
        b_doc_of: dict[int, int] = {}
        for d in docs_b:
            for h in d.get("ground_truth") or []:
                b_doc_of[id(h)] = d["doc_id"]
        for doc in docs:
            doc_pairs = align_hits([doc], docs_b)
            for hit, match in doc_pairs:
                if hit is None:
                    # A span only the second annotator marked. Without these rows the
                    # verified gold can never exceed the first annotator's recall, and the
                    # adjudicator is never shown what they missed.
                    assert match is not None
                    row = _row(b_doc_of[id(match)], match, origin="other")
                    row["other_entity_id"] = ""
                    row["other_label"] = ""
                    row["other_direction"] = ""
                    row["agrees"] = ""
                    row["agrees_direction"] = ""
                else:
                    row = _row(doc["doc_id"], hit, origin="gold")
                    if doc["doc_id"] not in shared:
                        # The second annotator never saw this document; an empty
                        # comparison here means "not covered", not "disagreed".
                        row["other_entity_id"] = "not-covered"
                        row["other_label"] = "not-covered"
                        row["other_direction"] = "not-covered"
                        row["agrees"] = "not-covered"
                        row["agrees_direction"] = "not-covered"
                    elif match is None:
                        row["other_entity_id"] = ""
                        row["other_label"] = ""
                        row["other_direction"] = ""
                        row["agrees"] = ""
                        row["agrees_direction"] = ""
                    else:
                        row["other_entity_id"] = match["entity_id"]
                        row["other_label"] = label_of.get(str(match["entity_id"]), "")
                        row["other_direction"] = match.get("direction") or ""
                        row["agrees"] = str(
                            ids_agree(hit["entity_id"], match["entity_id"], fold)
                        )
                        same_direction = directions_agree(hit, match, fold, symmetric)
                        row["agrees_direction"] = (
                            "" if same_direction is None else str(same_direction)
                        )
                rows.append(row)

    for row in rows:
        row["verdict"] = ""
        row["fix_entity_id"] = ""
        row["fix_direction"] = ""

    pd.DataFrame(rows).to_csv(output, sep="\t", index=False)
    logger.info(
        "Wrote %d review rows to %s (%d proposed by the second annotator only)",
        len(rows),
        output,
        sum(1 for r in rows if r["origin"] == "other"),
    )
    if other is not None:
        report = kappa_report(docs, load_gold(other), fold=fold)
        logger.info("Agreement: %s", json.dumps(report, indent=1))


@main.command("import")
@click.option("--gold", required=True, type=ExpandedPath(exists=True))
@click.option("--sheet", required=True, type=ExpandedPath(exists=True))
@click.option(
    "--other",
    default=None,
    type=ExpandedPath(exists=True),
    help=(
        "The second annotator's gold, required when the sheet carries origin=other rows; "
        "accepted ones are taken from here with their argument spans intact."
    ),
)
@click.option("--verified-by", required=True)
@click.option("--output", required=True, type=ExpandedPath())
def import_cmd(
    gold: str, sheet: str, other: str | None, verified_by: str, output: str
) -> None:
    docs = load_gold(gold)
    sheet_df = pd.read_csv(sheet, sep="\t", dtype=str).fillna("")

    verdicts: dict[tuple[int, int, int], tuple[str, str, str]] = {}
    accepted_other: list[tuple[int, int, int]] = []
    for _, row in sheet_df.iterrows():
        verdict = row["verdict"].strip().lower()
        if verdict not in _VERDICTS:
            raise click.ClickException(
                f"doc_id={row['doc_id']} span=({row['a']},{row['b']}): verdict must be "
                f"one of {sorted(_VERDICTS)}, got {row['verdict']!r} — every row needs "
                "an explicit decision"
            )
        key = (int(row["doc_id"]), int(row["a"]), int(row["b"]))
        verdicts[key] = (
            verdict,
            row["fix_entity_id"].strip(),
            row["fix_direction"].strip(),
        )
        # Legacy sheets predate the column; everything in them came from --gold.
        origin = row["origin"].strip() if "origin" in sheet_df.columns else "gold"
        if origin == "other" and verdict != "reject":
            accepted_other.append(key)

    if accepted_other and other is None:
        raise click.ClickException(
            f"{len(accepted_other)} accepted rows were proposed by the second annotator "
            "(origin=other); pass --other with that annotator's gold file so their spans "
            "can be carried over"
        )

    other_hits: dict[tuple[int, int, int], dict] = {}
    if other is not None:
        for doc in load_gold(other):
            for hit in doc.get("ground_truth") or []:
                other_hits[(int(doc["doc_id"]), int(hit["a"]), int(hit["b"]))] = hit

    n_kept = n_fixed = n_rejected = n_adopted = 0
    # Verdict counts split by surface type: the correction rate is reported per slice.
    by_slice: dict[str, dict[str, int]] = {}

    def _count(hit: dict, verdict: str) -> None:
        anchored = hit.get("surface_anchored")
        name = (
            "unknown"
            if anchored is None
            else ("anchored" if anchored else "paraphrase")
        )
        slot = by_slice.setdefault(name, {"accept": 0, "fix": 0, "reject": 0})
        slot[verdict] += 1

    def _apply(hit: dict, verdict: str, fix_id: str, fix_dir: str) -> dict:
        if verdict == "fix":
            if fix_id:
                hit["entity_id"] = fix_id
            if fix_dir:
                hit["direction"] = fix_dir
        hit["source"] = "human"
        hit["annotator"] = verified_by
        hit["verdict"] = verdict
        return hit

    for doc in docs:
        doc_id = int(doc["doc_id"])
        kept = []
        for hit in doc.get("ground_truth") or []:
            key = (doc_id, int(hit["a"]), int(hit["b"]))
            if key not in verdicts:
                raise click.ClickException(f"sheet is missing a row for {key}")
            verdict, fix_id, fix_dir = verdicts[key]
            _count(hit, verdict)
            if verdict == "reject":
                n_rejected += 1
                continue
            n_fixed += verdict == "fix"
            n_kept += verdict == "accept"
            kept.append(_apply(hit, verdict, fix_id, fix_dir))

        for key in accepted_other:
            if key[0] != doc_id:
                continue
            hit = other_hits.get(key)
            if hit is None:
                raise click.ClickException(
                    f"--other has no hit for the accepted origin=other row {key}"
                )
            verdict, fix_id, fix_dir = verdicts[key]
            n_adopted += 1
            kept.append(_apply(dict(hit), verdict, fix_id, fix_dir))

        kept.sort(key=lambda h: (int(h["a"]), int(h["b"])))
        doc["ground_truth"] = kept

    Path(output).write_text(
        json.dumps(docs, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    reviewed = n_kept + n_fixed + n_rejected
    summary = {
        "verified_by": verified_by,
        "reviewed_candidates": reviewed,
        "accepted": n_kept,
        "fixed": n_fixed,
        "rejected": n_rejected,
        "adopted_from_second_annotator": n_adopted,
        # The share of first-annotator candidates a curator changed or removed.
        "correction_rate": round((n_fixed + n_rejected) / reviewed, 4)
        if reviewed
        else None,
        "by_slice": by_slice,
    }
    summary_path = Path(output).with_suffix(".verification.json")
    summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    logger.info(
        "Verified gold: %d accepted, %d fixed, %d rejected, %d adopted from the second "
        "annotator → %s",
        n_kept,
        n_fixed,
        n_rejected,
        n_adopted,
        output,
    )


@main.command("agreement")
@click.option("--gold", required=True, type=ExpandedPath(exists=True))
@click.option("--other", required=True, type=ExpandedPath(exists=True))
@click.option(
    "--kb-csv-path",
    default=None,
    type=ExpandedPath(exists=True),
    help="Pairs KB; adds the κ folded in T̂ tier 0 (and tier 1 with --equivalences).",
)
@click.option("--equivalences", default=None, type=ExpandedPath(exists=True))
def agreement_cmd(
    gold: str, other: str, kb_csv_path: str | None, equivalences: str | None
) -> None:
    if equivalences is not None and kb_csv_path is None:
        raise click.ClickException("--equivalences needs --kb-csv-path")
    fold = None
    if kb_csv_path is not None:
        fold, _ = load_fold(pd.read_csv(kb_csv_path), equivalences)
    report = kappa_report(load_gold(gold), load_gold(other), fold=fold)
    click.echo(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
