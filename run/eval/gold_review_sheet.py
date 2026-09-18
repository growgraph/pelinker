"""Human-verification round-trip for LLM-pre-annotated gold, plus agreement stats.

``export`` flattens one gold ``*.json`` (the ``annotate_llm.py`` output) into a review
TSV — one row per candidate span with a marked context window and empty ``verdict`` /
``fix_entity_id`` / ``fix_direction`` columns — for spreadsheet review. With
``--other`` (a second annotator's file over the same docs), each row also shows the
other annotator's call and the command prints Cohen's κ on entity id and direction over
span-aligned pairs.

``import`` merges the filled sheet back: ``verdict`` ∈ ``accept`` / ``reject`` /
``fix`` (with the ``fix_*`` columns), everything else is an error so nothing is
silently kept. Accepted/fixed hits become ``source="human"``.

``agreement`` prints κ between two gold files without exporting a sheet.

Usage:

    uv run python run/eval/gold_review_sheet.py export \
        --gold gold.llm-a.json --other gold.llm-b.json --kb-csv-path <kb> \
        --output review.tsv
    uv run python run/eval/gold_review_sheet.py import \
        --gold gold.llm-a.json --sheet review.tsv --verified-by <name> \
        --output gold.verified.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from pelinker.core.paths import ExpandedPath

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


def align_hits(
    docs_a: list[dict], docs_b: list[dict]
) -> list[tuple[dict, dict | None]]:
    """Greedy span alignment of two annotators' hits per document (by doc_id)."""
    b_by_doc = {d["doc_id"]: list(d.get("ground_truth") or []) for d in docs_b}
    pairs: list[tuple[dict, dict | None]] = []
    for doc in docs_a:
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
    return pairs


def kappa_report(docs_a: list[dict], docs_b: list[dict]) -> dict:
    """Cohen's κ on entity id and direction over span-aligned pairs.

    Unmatched spans enter the entity-id κ as an explicit ``∅`` category, so an
    annotator that hallucinates or misses spans is penalized rather than ignored.
    Direction κ is computed over pairs where both sides matched and gave a direction.
    """
    pairs = align_hits(docs_a, docs_b)
    ids_a = [str(h["entity_id"]) for h, _ in pairs]
    ids_b = [("∅" if o is None else str(o["entity_id"])) for _, o in pairs]
    report: dict = {
        "n_pairs": len(pairs),
        "n_span_matched": sum(1 for _, o in pairs if o is not None),
    }
    if len(set(ids_a) | set(ids_b)) > 1 and pairs:
        report["kappa_entity_id"] = float(cohen_kappa_score(ids_a, ids_b))
    directed = [
        (str(h.get("direction")), str(o.get("direction")))
        for h, o in pairs
        if o is not None and h.get("direction") and o.get("direction")
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
@click.option("--output", required=True, type=ExpandedPath())
def export_cmd(gold: str, other: str | None, kb_csv_path: str, output: str) -> None:
    docs = load_gold(gold)
    kb = pd.read_csv(kb_csv_path)
    label_of = dict(zip(kb["entity_id"].astype(str), kb["label"].astype(str)))

    other_by_key: dict[tuple, dict | None] = {}
    if other is not None:
        docs_b = load_gold(other)
        for hit, match in align_hits(docs, docs_b):
            other_by_key[(id(hit))] = match

    rows = []
    for doc in docs:
        text = doc["text"]
        for hit in doc.get("ground_truth") or []:
            row = {
                "doc_id": doc["doc_id"],
                "itext": hit.get("itext"),
                "a": hit["a"],
                "b": hit["b"],
                "context": _context(text, hit["a"], hit["b"]),
                "surface": hit.get("surface", text[hit["a"] : hit["b"]]),
                "entity_id": hit["entity_id"],
                "label": label_of.get(str(hit["entity_id"]), ""),
                "direction": hit.get("direction") or "",
                "confidence": hit.get("confidence", ""),
                "annotator": hit.get("annotator", ""),
            }
            if other is not None:
                match = other_by_key.get(id(hit))
                row["other_entity_id"] = "" if match is None else match["entity_id"]
                row["other_direction"] = (
                    "" if match is None else (match.get("direction") or "")
                )
                row["agrees"] = (
                    "" if match is None else str(match["entity_id"] == hit["entity_id"])
                )
            row["verdict"] = ""
            row["fix_entity_id"] = ""
            row["fix_direction"] = ""
            rows.append(row)

    pd.DataFrame(rows).to_csv(output, sep="\t", index=False)
    logger.info("Wrote %d review rows to %s", len(rows), output)
    if other is not None:
        report = kappa_report(docs, load_gold(other))
        logger.info("Agreement: %s", json.dumps(report, indent=1))


@main.command("import")
@click.option("--gold", required=True, type=ExpandedPath(exists=True))
@click.option("--sheet", required=True, type=ExpandedPath(exists=True))
@click.option("--verified-by", required=True)
@click.option("--output", required=True, type=ExpandedPath())
def import_cmd(gold: str, sheet: str, verified_by: str, output: str) -> None:
    docs = load_gold(gold)
    sheet_df = pd.read_csv(sheet, sep="\t", dtype=str).fillna("")

    verdicts: dict[tuple[int, int, int], tuple[str, str, str]] = {}
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

    n_kept = n_fixed = n_rejected = 0
    for doc in docs:
        kept = []
        for hit in doc.get("ground_truth") or []:
            key = (int(doc["doc_id"]), int(hit["a"]), int(hit["b"]))
            if key not in verdicts:
                raise click.ClickException(f"sheet is missing a row for {key}")
            verdict, fix_id, fix_dir = verdicts[key]
            if verdict == "reject":
                n_rejected += 1
                continue
            if verdict == "fix":
                if fix_id:
                    hit["entity_id"] = fix_id
                if fix_dir:
                    hit["direction"] = fix_dir
                n_fixed += 1
            else:
                n_kept += 1
            hit["source"] = "human"
            hit["annotator"] = verified_by
            kept.append(hit)
        doc["ground_truth"] = kept

    Path(output).write_text(
        json.dumps(docs, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    logger.info(
        "Verified gold: %d accepted, %d fixed, %d rejected → %s",
        n_kept,
        n_fixed,
        n_rejected,
        output,
    )


@main.command("agreement")
@click.option("--gold", required=True, type=ExpandedPath(exists=True))
@click.option("--other", required=True, type=ExpandedPath(exists=True))
def agreement_cmd(gold: str, other: str) -> None:
    report = kappa_report(load_gold(gold), load_gold(other))
    click.echo(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
