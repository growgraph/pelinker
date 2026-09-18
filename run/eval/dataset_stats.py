"""Corpus, KB and gold-set statistics for the evaluation tables.

The submission reported no corpus size, no mention counts, and no dataset table at all —
a reviewer could not tell how much data any number rested on. This produces those figures
from the artifacts themselves rather than from memory.

Any subset of the three inputs may be given:

- ``--input-text-table-path`` — abstract table (``ids``/``publication_year``/``summary``):
  document count, year range, character-length distribution, pmid coverage.
- ``--kb-csv-path`` — property KB: entity count, labels with descriptions, label token
  lengths, declared inverse pairs.
- ``--fit-report`` — a fit's clustering report: mention rows, emergent clusters, noise
  fraction, mentions per KB label, and the share of label mass that is passive-voice.
- ``--gold`` — verified gold JSON: documents, spans, spans per document, direction
  distribution, label coverage.

Numbers are measurement output: write them outside this repository and cite them from the
measurement writeup.

Usage:

    uv run python run/eval/dataset_stats.py \
        --kb-csv-path data/derived/properties.synthesis.2.inverse.csv \
        --fit-report reports/fit-b-full-2/linker_fit.clustering_report.json.gz \
        --report-dir <workdir>/eval-runs/dataset-stats
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click
import pandas as pd
from pelinker.core.paths import ExpandedPath

logger = logging.getLogger(__name__)


def corpus_stats(path: str) -> dict[str, Any]:
    frame = pd.read_csv(path, sep="\t")
    text = frame["summary"].fillna("").astype(str)
    lengths = text.str.len()
    return {
        "n_documents": int(len(frame)),
        "n_with_pmid": int(
            frame["ids"].astype(str).str.contains("pubmed.ncbi.nlm.nih.gov").sum()
        )
        if "ids" in frame.columns
        else None,
        "publication_year_min": int(frame["publication_year"].min())
        if "publication_year" in frame.columns
        else None,
        "publication_year_max": int(frame["publication_year"].max())
        if "publication_year" in frame.columns
        else None,
        "chars_mean": float(lengths.mean()),
        "chars_median": float(lengths.median()),
        "chars_p90": float(lengths.quantile(0.9)),
        "n_empty": int((lengths == 0).sum()),
    }


def kb_stats(path: str) -> dict[str, Any]:
    kb = pd.read_csv(path)
    labelled = kb[kb["label"].notna()]
    token_counts = labelled["label"].astype(str).str.split().map(len)
    out: dict[str, Any] = {
        "n_entities": int(len(kb)),
        "n_labelled": int(len(labelled)),
        "n_with_description": int(labelled["description"].notna().sum())
        if "description" in labelled.columns
        else None,
        "label_tokens_mean": float(token_counts.mean()),
        "label_tokens_max": int(token_counts.max()),
        "n_multiword_labels": int((token_counts > 1).sum()),
    }
    if "inverse_entity_id" in kb.columns:
        known = set(kb["entity_id"].astype(str))
        resolvable = kb["inverse_entity_id"].dropna().astype(str).isin(known)
        out["n_declared_inverses"] = int(kb["inverse_entity_id"].notna().sum())
        out["n_resolvable_inverses"] = int(resolvable.sum())
    return out


def fit_stats(path: str) -> dict[str, Any]:
    from pelinker.reports.io import read_clustering_report_json

    report = read_clustering_report_json(path)
    assignments = report.assignments
    emergent = assignments[assignments["cluster"].astype(int) >= 0]
    per_label = assignments["entity"].astype(str).value_counts()
    passive = assignments["entity"].astype(str).str.endswith(" by")
    return {
        "n_mention_rows": int(len(assignments)),
        "n_emergent_rows": int(len(emergent)),
        "noise_fraction": float(1.0 - len(emergent) / len(assignments))
        if len(assignments)
        else None,
        "n_emergent_clusters": int(emergent["cluster"].nunique()),
        "n_labels_present": int(assignments["entity"].nunique()),
        "mentions_per_label_median": float(per_label.median()),
        "mentions_per_label_p90": float(per_label.quantile(0.9)),
        "mentions_per_label_max": int(per_label.max()),
        "passive_label_mention_share": float(passive.mean()),
        "min_cluster_size": int(report.hyperparameters.min_cluster_size),
    }


def gold_stats(path: str) -> dict[str, Any]:
    from pelinker.eval.harness import load_gold_docs

    docs = load_gold_docs(path)
    spans = [s for d in docs for s in d.spans]
    per_doc = [len(d.spans) for d in docs]
    directions = pd.Series(
        [s.direction for s in spans if s.direction is not None], dtype=object
    )
    sources = pd.Series([s.source for s in spans if s.source is not None], dtype=object)
    return {
        "n_documents": len(docs),
        "n_spans": len(spans),
        "spans_per_doc_mean": float(pd.Series(per_doc).mean()) if per_doc else None,
        "n_documents_without_spans": int(sum(1 for n in per_doc if n == 0)),
        "n_distinct_labels": len({s.entity_id for s in spans if s.entity_id}),
        "direction_counts": directions.value_counts().to_dict(),
        "source_counts": sources.value_counts().to_dict(),
    }


@click.command()
@click.option("--input-text-table-path", default=None, type=ExpandedPath(exists=True))
@click.option("--kb-csv-path", default=None, type=ExpandedPath(exists=True))
@click.option("--fit-report", default=None, type=ExpandedPath(exists=True))
@click.option("--gold", default=None, type=ExpandedPath(exists=True))
@click.option("--report-dir", required=True, type=ExpandedPath())
def main(
    input_text_table_path: str | None,
    kb_csv_path: str | None,
    fit_report: str | None,
    gold: str | None,
    report_dir: str,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not any([input_text_table_path, kb_csv_path, fit_report, gold]):
        raise click.ClickException("give at least one input to describe")

    stats: dict[str, Any] = {}
    if input_text_table_path:
        stats["corpus"] = corpus_stats(input_text_table_path)
        stats["corpus"]["path"] = input_text_table_path
    if kb_csv_path:
        stats["kb"] = kb_stats(kb_csv_path)
        stats["kb"]["path"] = kb_csv_path
    if fit_report:
        stats["fit"] = fit_stats(fit_report)
        stats["fit"]["path"] = fit_report
    if gold:
        stats["gold"] = gold_stats(gold)
        stats["gold"]["path"] = gold

    logger.info("%s", json.dumps(stats, indent=1, default=str))
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_stats.json").write_text(json.dumps(stats, indent=1, default=str))
    logger.info("Wrote %s", out / "dataset_stats.json")


if __name__ == "__main__":
    main()
