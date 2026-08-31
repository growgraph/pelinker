"""Score the linker and the reference baselines on a verified gold file.

Runs each selected system in both regimes where it applies (end-to-end span+link, and
linking-only over gold spans) through the shared harness, so quality and runtime come
from one pass. Results are written as JSON + CSV to ``--report-dir``; keep that directory
outside this repository — measured numbers belong with the measurement writeup, not in
the package.

Systems:

- ``lexical`` — lemma match to KB labels (both regimes)
- ``encoder`` — sentence-encoder cosine top-1 over KB label+description (linking-only;
  end-to-end reuses the lexical span proposer)
- ``llm`` — LLM constrained choice over the KB (linking-only; needs the ``eval`` extra
  and a credential; responses are cached under ``--llm-cache-dir``)
- ``linker`` — a fitted PELinker artifact (both regimes), with the KB-out → KB-in id
  bridge applied so entity accuracy is comparable

Usage:

    uv run python run/eval/run_baselines.py \
        --gold <workdir>/gold/gold.verified.json \
        --kb-csv-path data/derived/properties.synthesis.2.csv \
        --report-dir <workdir>/eval-runs/<run-name> \
        --systems lexical,encoder \
        --model-path <models>/pelinker.pubmedbert.1
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import pandas as pd
import spacy

from pelinker.eval.baselines import (
    EncoderKnnBaseline,
    LexicalLemmaBaseline,
    LlmLinkerBaseline,
)
from pelinker.eval.harness import (
    evaluate_end_to_end,
    evaluate_linking_only,
    load_gold_docs,
)

logger = logging.getLogger(__name__)

_ALL_SYSTEMS = ("lexical", "encoder", "llm", "linker")


def _linker_predict_fn(linker, thr_score: float):
    def predict(texts: list[str]) -> list[dict]:
        result = linker.predict(texts, thr_score=thr_score)
        return [dict(row) for row in (result.entities or [])]

    return predict


def _linker_link_fn(linker, thr_score: float):
    """Linking-only: predict over the span's own text and take the covering row."""

    def link(text: str, a: int, b: int) -> str | None:
        result = linker.predict([text], thr_score=thr_score)
        best: tuple[int, str] | None = None
        for row in result.entities or []:
            ra, rb = int(row["a"]), int(row["b"])
            if ra < b and a < rb:  # overlaps the gold span
                overlap = min(b, rb) - max(a, ra)
                eid = row.get("entity_id_predicted")
                if eid is not None and (best is None or overlap > best[0]):
                    best = (overlap, str(eid))
        return None if best is None else best[1]

    return link


@click.command()
@click.option("--gold", required=True, type=click.Path(exists=True))
@click.option("--kb-csv-path", required=True, type=click.Path(exists=True))
@click.option("--report-dir", required=True, type=click.Path())
@click.option("--systems", default="lexical,encoder", show_default=True)
@click.option("--nlp-model", default="en_core_web_lg", show_default=True)
@click.option(
    "--encoder-model",
    default="neuml/pubmedbert-base-embeddings",
    show_default=True,
)
@click.option("--llm-model", default="claude-sonnet-5", show_default=True)
@click.option("--llm-cache-dir", default=None)
@click.option(
    "--model-path", default=None, help="Fitted linker artifact (system: linker)."
)
@click.option("--thr-score", default=None, type=float)
@click.option("--match-mode", default="overlap", show_default=True)
def main(
    gold: str,
    kb_csv_path: str,
    report_dir: str,
    systems: str,
    nlp_model: str,
    encoder_model: str,
    llm_model: str,
    llm_cache_dir: str | None,
    model_path: str | None,
    thr_score: float | None,
    match_mode: str,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    wanted = [s.strip() for s in systems.split(",") if s.strip()]
    unknown = set(wanted) - set(_ALL_SYSTEMS)
    if unknown:
        raise click.ClickException(
            f"unknown systems {sorted(unknown)}; choose from {list(_ALL_SYSTEMS)}"
        )

    docs = load_gold_docs(gold)
    n_spans = sum(len(d.spans) for d in docs)
    logger.info("Gold: %d docs, %d spans", len(docs), n_spans)
    kb = pd.read_csv(kb_csv_path)

    runs = []

    if {"lexical", "encoder", "llm"} & set(wanted):
        nlp = spacy.load(nlp_model)
        lexical = LexicalLemmaBaseline(kb, nlp)

    if "lexical" in wanted:
        runs.append(
            evaluate_end_to_end(
                lexical.predict, docs, system="lexical", match_mode=match_mode
            )
        )
        runs.append(evaluate_linking_only(lexical.link, docs, system="lexical"))

    if "encoder" in wanted:
        enc = EncoderKnnBaseline(kb, model_name=encoder_model)
        runs.append(evaluate_linking_only(enc.link, docs, system="encoder_knn"))

        # End-to-end: lexical spans, encoder ids — isolates the id decision.
        def encoder_predict(texts: list[str]) -> list[dict]:
            rows = []
            for itext, text in enumerate(texts):
                for a, b, _ in lexical.proposer.propose(text):
                    rows.append(
                        {
                            "itext": itext,
                            "a": a,
                            "b": b,
                            "entity_id_predicted": enc.link(text, a, b),
                        }
                    )
            return rows

        runs.append(
            evaluate_end_to_end(
                encoder_predict, docs, system="encoder_knn", match_mode=match_mode
            )
        )

    if "llm" in wanted:
        cache = Path(llm_cache_dir or (Path(report_dir) / ".llm_cache"))
        llm = LlmLinkerBaseline(kb, model=llm_model, cache_dir=cache)
        runs.append(evaluate_linking_only(llm.link, docs, system=f"llm:{llm_model}"))

    if "linker" in wanted:
        if model_path is None:
            raise click.ClickException("--model-path is required for system 'linker'")
        from pelinker.kb.kb_out import kb_out_to_kb_in_map
        from pelinker.model import DEFAULT_CLUSTER_MEMBERSHIP_THRESHOLD, Linker

        linker = Linker.load(model_path)
        thr = (
            thr_score if thr_score is not None else DEFAULT_CLUSTER_MEMBERSHIP_THRESHOLD
        )
        bridge = (
            kb_out_to_kb_in_map(linker.kb_out_catalog)
            if linker.kb_out_catalog is not None
            else None
        )
        if not bridge:
            logger.warning(
                "No KB-out → KB-in id bridge on this artifact; entity accuracy will be "
                "undefined (see the catalog's kb_in provenance block)."
            )
        runs.append(
            evaluate_end_to_end(
                _linker_predict_fn(linker, thr),
                docs,
                system="pelinker",
                match_mode=match_mode,
                predicted_id_to_kb_in=bridge,
            )
        )

        link_fn = _linker_link_fn(linker, thr)
        if bridge:

            def bridged(text: str, a: int, b: int) -> str | None:
                minted = link_fn(text, a, b)
                return None if minted is None else bridge.get(minted)

            runs.append(evaluate_linking_only(bridged, docs, system="pelinker"))
        else:
            runs.append(evaluate_linking_only(link_fn, docs, system="pelinker"))

    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "gold_path": str(gold),
        "kb_csv_path": str(kb_csv_path),
        "n_docs": len(docs),
        "n_gold_spans": n_spans,
        "match_mode": match_mode,
        "runs": [r.to_jsonable() for r in runs],
    }
    (out / "baseline_results.json").write_text(json.dumps(payload, indent=1))
    pd.DataFrame([r.to_jsonable() for r in runs]).to_csv(
        out / "baseline_results.csv", index=False
    )
    for r in runs:
        logger.info("%s", json.dumps(r.to_jsonable()))
    logger.info("Wrote %d runs to %s", len(runs), out)


if __name__ == "__main__":
    main()
