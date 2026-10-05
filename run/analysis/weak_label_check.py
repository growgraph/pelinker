"""Acceptance check for the weak labels in a stage-(A) mention parquet.

Run it on a small embedding run before launching a full grid. The weak labels are the
target the hyperparameter searches optimise, so a defect here — one span carrying a
relation *and* its converse, a symmetric relation given an inverse direction —
silently becomes the objective.

Checks (positive rows only; synthetic negatives are skipped):

- **one label per span**: no ``(pmid, a_abs, b_abs)`` carries two labels;
- **no nested sites**: no labelled span lies inside another labelled span of the same
  document;
- **no relation/converse collision**: no span is labelled with two members of one
  converse pair (same ``canonical_entity_id`` in the pairs KB);
- **symmetric stays symmetric**: no row whose label is symmetric has
  ``direction=inverse``;
- **every label is in the KB**: a label the KB does not hold cannot be given a class,
  so the class views (:mod:`pelinker.kb.classes`) refuse the parquet.

It also prints how labels were assigned (``surface_rule``, ``direction``), the most
frequent labels, and — for a pairs KB — how many classes each view yields and how the
mention mass splits by direction relative to the canonical relation. The ``reldir``
count is the number of classes the selection objective scores against. Any violation
makes the exit status non-zero.

Usage:

    uv run python run/analysis/weak_label_check.py \\
        --parquet <dir>/res_pubmedbert_1.parquet \\
        --kb-csv-path data/derived/properties.synthesis.2.pairs.csv
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import click
import pandas as pd

from pelinker.core.onto import NEGATIVE_LABEL
from pelinker.core.paths import ExpandedPath
from pelinker.kb.classes import (
    CANONICAL_COLUMNS,
    CLASS_VIEWS,
    ENTITY_CLASS_COLUMN,
    RELATION_DIRECTION_COLUMN,
    KbClasses,
    add_view_columns,
)

logger = logging.getLogger(__name__)

_COLUMNS = ["pmid", "entity", "a_abs", "b_abs", "direction", "surface_rule"]


def load_mentions(path: str) -> pd.DataFrame:
    """Positive mention rows, without the embedding column."""
    frame = pd.read_parquet(path)
    missing = [c for c in _COLUMNS if c not in frame.columns]
    if missing:
        raise click.ClickException(
            f"{path}: no {', '.join(missing)} column(s) — the parquet predates "
            "parse-based weak labels; re-embed it"
        )
    frame = frame[_COLUMNS]
    return frame.loc[frame["entity"] != NEGATIVE_LABEL].reset_index(drop=True)


def multi_label_spans(rows: pd.DataFrame) -> pd.DataFrame:
    """Spans carrying more than one distinct label."""
    per_span = rows.groupby(["pmid", "a_abs", "b_abs"])["entity"].nunique()
    return per_span[per_span > 1].reset_index()


def nested_spans(rows: pd.DataFrame) -> int:
    """Labelled spans lying strictly inside another labelled span of the same document."""
    n = 0
    for _, doc in rows.drop_duplicates(["pmid", "a_abs", "b_abs"]).groupby("pmid"):
        spans = list(zip(doc["a_abs"], doc["b_abs"]))
        for a, b in spans:
            if any((a2, b2) != (a, b) and a2 <= a and b <= b2 for a2, b2 in spans):
                n += 1
    return n


def converse_collisions(rows: pd.DataFrame, kb: pd.DataFrame) -> int:
    """Spans labelled with two different members of one converse pair."""
    if "canonical_entity_id" not in kb.columns:
        return 0
    canon = dict(zip(kb["label"].astype(str), kb["canonical_entity_id"].astype(str)))
    work = rows.assign(canonical=rows["entity"].map(canon))
    grouped = work.groupby(["pmid", "a_abs", "b_abs", "canonical"])["entity"].nunique()
    return int((grouped > 1).sum())


def flipped_symmetric(rows: pd.DataFrame, kb: pd.DataFrame) -> int:
    if "is_symmetric" not in kb.columns:
        return 0
    symmetric = set(kb.loc[kb["is_symmetric"].fillna(False).astype(bool), "label"])
    return int(
        (rows["entity"].isin(symmetric) & (rows["direction"] == "inverse")).sum()
    )


def labels_not_in_kb(rows: pd.DataFrame, kb: pd.DataFrame) -> list[str]:
    """Mention labels the KB does not hold, sorted."""
    return sorted(set(rows["entity"].astype(str)) - set(kb["label"].astype(str)))


def view_summary(rows: pd.DataFrame, kb: pd.DataFrame) -> dict[str, Any] | None:
    """Classes per view, and mass by direction relative to the canonical relation.

    ``None`` for a KB without the converse-pair derivation: views need the pairs KB.
    Labels outside the KB are left out here; :func:`labels_not_in_kb` reports them.
    """
    if "entity_id" not in kb.columns or any(
        c not in kb.columns for c in CANONICAL_COLUMNS
    ):
        return None
    classes = KbClasses.from_kb(kb)
    known = rows.loc[rows["entity"].astype(str).isin(set(classes.canonical_label))]
    summary: dict[str, Any] = {
        f"classes_{view}": int(
            add_view_columns(known, classes, view)[ENTITY_CLASS_COLUMN].nunique()
        )
        for view in CLASS_VIEWS
    }
    reldir = add_view_columns(known, classes, "reldir")
    summary["relation_direction_mass"] = (
        reldir[RELATION_DIRECTION_COLUMN].value_counts().to_dict()
    )
    return summary


def check(rows: pd.DataFrame, kb: pd.DataFrame, top: int = 15) -> dict[str, Any]:
    multi = multi_label_spans(rows)
    unknown = labels_not_in_kb(rows, kb)
    return {
        "positive_rows": int(len(rows)),
        "by_surface_rule": rows["surface_rule"].fillna("none").value_counts().to_dict(),
        "by_direction": rows["direction"].fillna("none").value_counts().to_dict(),
        "top_labels": rows["entity"].value_counts().head(top).to_dict(),
        "views": view_summary(rows, kb),
        "labels_not_in_kb": unknown[:top],
        "violations": {
            "multi_label_spans": int(len(multi)),
            "nested_spans": nested_spans(rows),
            "converse_collisions": converse_collisions(rows, kb),
            "symmetric_with_inverse_direction": flipped_symmetric(rows, kb),
            "labels_not_in_kb": len(unknown),
        },
    }


@click.command()
@click.option("--parquet", required=True, type=ExpandedPath(exists=True))
@click.option("--kb-csv-path", required=True, type=ExpandedPath(exists=True))
@click.option("--output-path", default=None, type=ExpandedPath())
def main(parquet: str, kb_csv_path: str, output_path: str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = check(load_mentions(parquet), pd.read_csv(kb_csv_path))
    text = json.dumps(report, indent=1)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(text)
    click.echo(text)
    bad = {k: v for k, v in report["violations"].items() if v}
    if bad:
        logger.error("Weak-label violations: %s", bad)
        sys.exit(1)
    logger.info("Weak labels pass: one label per span, no converse collisions")


if __name__ == "__main__":
    main()
