"""Audit a gold sample and report KB coverage — run after every batch.

Two questions, answered from the artifacts in a gold directory:

**Is the sample sound?**

- **Eligibility.** Re-applies the gate (:mod:`pelinker.eval.sample_quality`) to every
  drawn text and counts failures by reason. A sample drawn by the current sampler shows
  zero failures; an older draw shows what it let through.
- **Density vs length.** The correlation between the stratification density and abstract
  length. If it is high, the strata are length strata in disguise.
- **Empty documents.** Documents with no gold hit, by stratum. Negative controls — strata
  ending in ``-none`` (or ``-zero`` in older samples) — *should* be empty; any hit there is
  a mention the detector missed.
- **Concentration.** How much of the gold mass sits on the ten most frequent labels, and
  how many labels occur only once.
- **Paraphrases and corrections.** The share of hits whose surface does not use the
  label's wording (``surface_anchored`` false), and — for a verified gold file — the
  curator's accept / fix / reject counts from ``<gold file>.verification.json``.

**How far has coverage come?**

Per canonical label: abstracts in the eligible pool where the detector finds it, gold hits
so far, and whether it has reached ``--min-mentions`` (the smallest class a partition
metric can use). ``reachable`` counts the labels the pool supports at that level. The
cumulative curve over batches is the stopping rule: grow the sample until newly covered
labels per batch level off.

Usage:

    uv run python run/eval/audit_sample.py \\
        --gold-dir <workdir>/gold \\
        --kb-csv-path data/derived/properties.synthesis.2.pairs.csv

Writes ``<gold-dir>/sample_audit.json`` (override with ``--output-path``). Numbers are
measurement output; cite them from the measurement writeup, not from this repository.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click
import pandas as pd

from pelinker.core.paths import ExpandedPath
from pelinker.eval import sample_quality as sq

logger = logging.getLogger(__name__)

NEGATIVE_SUFFIXES = ("-none", "-zero")


def load_gold_hits(path: Path) -> dict[int, list[dict]]:
    """``doc_id -> ground_truth`` from a ``*.gt.json``-shaped gold file."""
    return {
        int(d["doc_id"]): list(d.get("ground_truth", []))
        for d in json.loads(path.read_text(encoding="utf-8"))
    }


def canonical_label_map(kb: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    """``(entity_id -> canonical label, label -> canonical label)``."""
    label_of = dict(zip(kb["entity_id"].astype(str), kb["label"].astype(str)))
    canon = (
        kb["canonical_entity_id"].astype(str)
        if "canonical_entity_id" in kb.columns
        else kb["entity_id"].astype(str)
    )
    by_id = {
        e: label_of.get(c, label_of[e])
        for e, c in zip(kb["entity_id"].astype(str), canon)
    }
    by_label = {label_of[e]: by_id[e] for e in by_id}
    return by_id, by_label


def density_column(manifest: pd.DataFrame) -> str | None:
    for col in ("n_verb_mentions", "n_mentions_proxy"):
        if col in manifest.columns:
            return col
    return None


def quality_section(manifest: pd.DataFrame, texts: dict[int, str]) -> dict[str, Any]:
    rules = sq.Eligibility()
    reasons = [
        sq.drop_reason(texts.get(int(d), ""), y, rules)
        for d, y in zip(manifest["doc_id"], manifest["publication_year"])
    ]
    lengths = manifest["doc_id"].map(lambda d: len(texts.get(int(d), "")))
    out: dict[str, Any] = {"eligibility": sq.reason_counts(reasons)}
    col = density_column(manifest)
    if col is not None and len(manifest) > 2:
        out["density_column"] = col
        out["corr_density_length"] = round(float(manifest[col].corr(lengths)), 3)
    return out


def gold_section(
    manifest: pd.DataFrame,
    gold: dict[int, list[dict]],
    by_id: dict[str, str],
    min_mentions: int,
) -> tuple[dict[str, Any], pd.Series]:
    n_hits = manifest["doc_id"].map(lambda d: len(gold.get(int(d), [])))
    stratum = manifest["stratum"].astype(str)
    negative = stratum.str.endswith(NEGATIVE_SUFFIXES)
    labels = pd.Series(
        [
            by_id.get(str(h.get("entity_id")), str(h.get("entity_id")))
            for d in manifest["doc_id"]
            for h in gold.get(int(d), [])
        ],
        dtype=object,
    )
    counts = labels.value_counts()
    section = {
        "documents": int(len(manifest)),
        "hits": int(n_hits.sum()),
        "documents_without_hits": int((n_hits == 0).sum()),
        "documents_without_hits_excluding_negatives": int(
            ((n_hits == 0) & ~negative).sum()
        ),
        "negative_controls": int(negative.sum()),
        "negative_controls_with_hits": int(((n_hits > 0) & negative).sum()),
        "without_hits_by_stratum": {
            k: f"{int(v.sum())}/{len(v)}" for k, v in (n_hits == 0).groupby(stratum)
        },
        "distinct_labels": int(counts.size),
        "labels_at_min_mentions": int((counts >= min_mentions).sum()),
        "top10_share": round(float(counts.head(10).sum() / max(1, counts.sum())), 3),
        "singleton_labels": int((counts == 1).sum()),
    }
    flags = [
        h["surface_anchored"]
        for d in manifest["doc_id"]
        for h in gold.get(int(d), [])
        if "surface_anchored" in h
    ]
    if flags:
        # Paraphrases ("leads to" for `causes`) are valid gold; the linker's training is
        # lemma-anchored, so they are scored as their own slice.
        section["paraphrase_hits"] = int(sum(not f for f in flags))
        section["paraphrase_share"] = round(sum(not f for f in flags) / len(flags), 3)
    return section, counts


def coverage_section(
    manifest: pd.DataFrame,
    gold: dict[int, list[dict]],
    counts: pd.Series,
    pool_labels: pd.Series | None,
    by_id: dict[str, str],
    by_label: dict[str, str],
    min_mentions: int,
) -> dict[str, Any]:
    section: dict[str, Any] = {"min_mentions": min_mentions}
    pool_freq: pd.Series | None = None
    if pool_labels is not None:
        canon = [
            by_label.get(label, label)
            for value in pool_labels
            if isinstance(value, str) and value
            for label in set(value.split("|"))
        ]
        pool_freq = pd.Series(canon, dtype=object).value_counts()
        reachable = set(pool_freq[pool_freq >= min_mentions].index)
        covered = {k for k, v in counts.items() if v >= min_mentions}
        section["reachable_labels"] = len(reachable)
        section["covered_reachable"] = len(reachable & covered)
    if "batch" in manifest.columns:
        curve = []
        seen: dict[str, int] = {}
        for batch, rows in manifest.sort_values("batch").groupby("batch"):
            for d in rows["doc_id"]:
                for h in gold.get(int(d), []):
                    key = by_id.get(str(h.get("entity_id")), str(h.get("entity_id")))
                    seen[key] = seen.get(key, 0) + 1
            curve.append(
                {
                    "batch": int(str(batch)),
                    "documents": int(len(rows)),
                    "labels_at_min_mentions": sum(
                        v >= min_mentions for v in seen.values()
                    ),
                    "distinct_labels": len(seen),
                }
            )
        section["curve"] = curve
    table = pd.DataFrame({"gold_hits": counts})
    if pool_freq is not None:
        table = table.join(pool_freq.rename("pool_abstracts"), how="outer")
    table = (
        table.fillna(0)
        .astype(int)
        .sort_values(
            [c for c in ("pool_abstracts", "gold_hits") if c in table.columns],
            ascending=False,
        )
    )
    section["per_label"] = table.reset_index(names="label").to_dict(orient="records")
    return section


@click.command()
@click.option("--gold-dir", required=True, type=ExpandedPath(exists=True))
@click.option("--kb-csv-path", required=True, type=ExpandedPath(exists=True))
@click.option(
    "--gold-file",
    default="gold.llm-a.json",
    show_default=True,
    help="Gold file inside --gold-dir; a verified file supersedes the LLM draft.",
)
@click.option("--min-mentions", default=5, show_default=True)
@click.option("--output-path", default=None, type=ExpandedPath())
def main(
    gold_dir: str,
    kb_csv_path: str,
    gold_file: str,
    min_mentions: int,
    output_path: str | None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = Path(gold_dir)
    manifest = pd.read_csv(root / "sample_manifest.csv")
    texts: dict[int, str] = {}
    with (root / "sample_texts.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            texts[int(rec["doc_id"])] = rec["text"]
    kb = pd.read_csv(kb_csv_path)
    by_id, by_label = canonical_label_map(kb)

    audit: dict[str, Any] = {
        "gold_dir": str(root),
        "quality": quality_section(manifest, texts),
    }
    gold_path = root / gold_file
    if gold_path.exists():
        gold = load_gold_hits(gold_path)
        audit["gold_file"] = gold_file
        # Only documents the gold file covers: reserve documents are never annotated, and
        # counting them as "without hits" would read as annotation failures.
        annotated = manifest.loc[manifest["doc_id"].astype(int).isin(gold)]
        audit["documents_not_in_gold_file"] = int(len(manifest) - len(annotated))
        audit["gold"], counts = gold_section(annotated, gold, by_id, min_mentions)
        verification = gold_path.with_suffix(".verification.json")
        if verification.exists():
            # Written by `gold_review_sheet.py import`: accept / fix / reject counts.
            audit["verification"] = json.loads(verification.read_text(encoding="utf-8"))
        pool_path = root / "pool_mentions.csv.gz"
        pool_labels = (
            pd.read_csv(pool_path)["labels_detected"] if pool_path.exists() else None
        )
        audit["coverage"] = coverage_section(
            annotated, gold, counts, pool_labels, by_id, by_label, min_mentions
        )
    else:
        logger.warning(
            "No %s in %s — gold and coverage sections skipped", gold_file, root
        )

    out = Path(output_path) if output_path else root / "sample_audit.json"
    out.write_text(json.dumps(audit, indent=1), encoding="utf-8")
    summary = {
        k: v for k, v in audit.get("gold", {}).items() if k != "without_hits_by_stratum"
    }
    logger.info("Quality: %s", audit["quality"])
    logger.info("Gold: %s", summary)
    if "coverage" in audit:
        cov = {
            k: v
            for k, v in audit["coverage"].items()
            if k not in ("per_label", "curve")
        }
        logger.info("Coverage: %s", cov)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
